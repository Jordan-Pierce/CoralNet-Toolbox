import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import gc
import sys
import requests
import traceback
from functools import lru_cache

import torch
import numpy as np

import rasterio
from rasterio.windows import Window
from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection

from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QMessageBox, QApplication, QPushButton

from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def get_available_device():
    """
    Get available devices

    :return:
    """
    devices = ['cpu',]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f'cuda:{i}')
    if torch.backends.mps.is_available():
        devices.append('mps')
    return devices


@lru_cache(maxsize=32)
def rasterio_open(image_path):
    """
    Open an image with rasterio and handle potential errors.

    Args:
        image_path (str): Path to the image file

    Returns:
        rasterio.DatasetReader: Opened rasterio dataset or None if error
    """
    try:
        # Use a local variable rather than instance attribute to avoid thread issues
        src = rasterio.open(image_path)
        return src
    except Exception as e:
        print(f"Error opening image with rasterio: {image_path}")
        print(f"Exception: {str(e)}")

        # Try to inspect file existence and permissions
        if not os.path.exists(image_path):
            print(f"File does not exist: {image_path}")
        elif not os.path.isfile(image_path):
            print(f"Path is not a file: {image_path}")
        elif not os.access(image_path, os.R_OK):
            print(f"File is not readable: {image_path}")

        # Return None on failure
        return None


def rasterio_to_qimage(rasterio_src, longest_edge=None):
    """
    Load a scaled version of an image with colormap support

    Args:
        rasterio_src (rasterio.DatasetReader): Rasterio dataset reader object
        longest_edge (int, optional): If provided, scales to longest_edge while maintaining aspect ratio.

    Returns:
        QImage: Scaled image
    """
    try:
        # Get the original size of the image
        original_width = rasterio_src.width
        original_height = rasterio_src.height

        # Calculate the scaled size based on input parameters
        if longest_edge is not None:
            # Scale to fit within max_size (for thumbnails)
            scale = min(longest_edge / original_width, longest_edge / original_height)
            scaled_width = int(original_width * scale)
            scaled_height = int(original_height * scale)
        else:
            # Scale to 1/100 of original size (for initial preview)
            scaled_width = original_width // 100
            scaled_height = original_height // 100

        # Read a downsampled version of the image
        window = Window(0, 0, original_width, original_height)

        # Check for single-band image with colormap
        if rasterio_src.count == 1 and rasterio_src.colormap(1):
            # Read the single band
            image = rasterio_src.read(1,
                                      window=window,
                                      out_shape=(scaled_height, scaled_width),
                                      resampling=rasterio.enums.Resampling.bilinear)
            
            # Get the colormap
            colormap = rasterio_src.colormap(1)
            
            # Create a lookup table for the colormap
            max_idx = max(colormap.keys()) + 1
            lut = np.zeros((max_idx, 3), dtype=np.uint8)
            
            # Fill the lookup table with RGB values
            for idx, color in colormap.items():
                if idx < max_idx:  # Safety check
                    lut[idx] = [color[0], color[1], color[2]]  # Ignore alpha
            
            # Clip image indices to valid range for the LUT
            image_indices = np.clip(image, 0, max_idx - 1).astype(np.uint8)
            
            # Use the image as indices into the lookup table
            # This is a vectorized operation that maps each pixel to its RGB value
            rgb_image = lut[image_indices]
            
            # Use RGB format for colormap images
            qimage_format = QImage.Format_RGB888
            image = rgb_image
            num_bands = 3
            
        elif rasterio_src.count < 3:
            # Grayscale image without colormap
            num_bands = 1
            qimage_format = QImage.Format_Grayscale8
            
            # Read a single band
            image = rasterio_src.read(1,
                                      window=window,
                                      out_shape=(scaled_height, scaled_width),
                                      resampling=rasterio.enums.Resampling.bilinear)
            
        else:
            # Read RGB bands
            num_bands = 3
            qimage_format = QImage.Format_RGB888
            
            image = rasterio_src.read([1, 2, 3],
                                      window=window,
                                      out_shape=(3, scaled_height, scaled_width),
                                      resampling=rasterio.enums.Resampling.bilinear)
            
            # Transpose to height, width, channels format
            image = np.transpose(image, (1, 2, 0))

        # Convert to uint8 if not already
        if image.dtype != np.uint8:
            image = image.astype(float) * (255.0 / image.max())
            image = image.astype(np.uint8)

        # Convert the numpy array to QImage
        qimage = QImage(image.data.tobytes(),
                        scaled_width,
                        scaled_height,
                        scaled_width * num_bands,  # bytes per line for Greyscale or RGB
                        qimage_format)

        return qimage

    except Exception as e:
        print(f"Error loading scaled image: {str(e)}")
        traceback.print_exc()
        return QImage()  # Return an empty QImage if there's an error


def rasterio_to_cropped_image(rasterio_src, window):
    """
    Convert a rasterio window to a QImage, supporting colormaps.

    Args:
        rasterio_src (rasterio.DatasetReader): Rasterio dataset reader object
        window (rasterio.windows.Window): Window to read from
    Returns:
        QImage: Cropped image as a QImage
    """
    try:
        # Check for single-band image with colormap
        if rasterio_src.count == 1 and rasterio_src.colormap(1):
            # Read only the first band
            image = rasterio_src.read(1, window=window)
            
            # Get the colormap
            colormap = rasterio_src.colormap(1)
            
            # Create a lookup table for the colormap
            max_idx = max(colormap.keys()) + 1
            lut = np.zeros((max_idx, 3), dtype=np.uint8)
            
            # Fill the lookup table with RGB values
            for idx, color in colormap.items():
                if idx < max_idx:  # Safety check
                    lut[idx] = [color[0], color[1], color[2]]  # Ignore alpha
            
            # Clip image indices to valid range for the LUT
            image_indices = np.clip(image, 0, max_idx - 1).astype(np.uint8)
            
            # Use the image as indices into the lookup table
            # This is a vectorized operation that maps each pixel to its RGB value
            rgb_image = lut[image_indices]

            # Use RGB format for colormap images
            qimage_format = QImage.Format_RGB888
            image = rgb_image
            num_bands = 3

        elif rasterio_src.count < 3:
            # Grayscale image without colormap
            num_bands = 1
            qimage_format = QImage.Format_Grayscale8
            # Read the single band
            image = rasterio_src.read(1, window=window)

        else:
            # Read RGB bands
            num_bands = 3
            qimage_format = QImage.Format_RGB888
            # Read the three bands and transpose to (height, width, channels)
            image = rasterio_src.read([1, 2, 3], window=window)
            image = np.transpose(image, (1, 2, 0))

        # Convert to uint8 if not already
        if image.dtype != np.uint8:
            image = image.astype(float) * (255.0 / image.max())
            image = image.astype(np.uint8)
        
        # Convert the numpy array to QImage
        qimage = QImage(image.data.tobytes(),
                        int(window.width),
                        int(window.height),
                        int(window.width * num_bands),  # bytes per line for Greyscale or RGB
                        qimage_format)

        return qimage

    except Exception as e:
        print(f"Error loading cropped image: {str(e)}")
        traceback.print_exc()
        return QImage()  # Return an empty QImage if there's an error
    

def rasterio_to_numpy(rasterio_src, longest_edge=None):
    """
    Convert a rasterio dataset to a numpy array, with colormap support.

    Args:
        rasterio_src (rasterio.DatasetReader): Rasterio dataset reader object
        longest_edge (int, optional): If provided, scales to longest_edge while maintaining aspect ratio.

    Returns:
        numpy.ndarray: Image as a numpy array in format (h, w, c) for RGB or (h, w) for grayscale
    """
    try:
        # Get the original size of the image
        original_width = rasterio_src.width
        original_height = rasterio_src.height

        # Calculate the scaled size based on input parameters
        if longest_edge is not None:
            # Scale to fit within max_size
            scale = min(longest_edge / original_width, longest_edge / original_height)
            scaled_width = int(original_width * scale)
            scaled_height = int(original_height * scale)
        else:
            # Use original size
            scaled_width = original_width
            scaled_height = original_height

        # Read a downsampled version of the image
        window = Window(0, 0, original_width, original_height)

        # Check for single-band image with colormap
        if rasterio_src.count == 1 and rasterio_src.colormap(1):
            # Read the single band
            image = rasterio_src.read(1,
                                      window=window,
                                      out_shape=(scaled_height, scaled_width),
                                      resampling=rasterio.enums.Resampling.bilinear)
            
            # Get the colormap
            colormap = rasterio_src.colormap(1)
            
            # Create a lookup table for the colormap
            max_idx = max(colormap.keys()) + 1
            lut = np.zeros((max_idx, 3), dtype=np.uint8)
            
            # Fill the lookup table with RGB values
            for idx, color in colormap.items():
                if idx < max_idx:  # Safety check
                    lut[idx] = [color[0], color[1], color[2]]  # Ignore alpha
            
            # Clip image indices to valid range for the LUT
            image_indices = np.clip(image, 0, max_idx - 1).astype(np.uint8)
            
            # Use the image as indices into the lookup table
            # This is a vectorized operation that maps each pixel to its RGB value
            rgb_image = lut[image_indices]
            
            # Use the colorized RGB version of the image
            image = rgb_image
            
        elif rasterio_src.count == 1:
            # Single-band image without colormap (grayscale)
            image = rasterio_src.read(1,
                                      window=window,
                                      out_shape=(scaled_height, scaled_width),
                                      resampling=rasterio.enums.Resampling.bilinear)
            
            # Convert to 3-channel grayscale image
            image = np.stack([image] * 3, axis=-1)  # Stack the single band to create 3 channels
            
        elif rasterio_src.count >= 3:
            # Multi-band image (RGB)
            image = rasterio_src.read([1, 2, 3],
                                      window=window,
                                      out_shape=(3, scaled_height, scaled_width),
                                      resampling=rasterio.enums.Resampling.bilinear)

            # Transpose to height, width, channels format
            image = np.transpose(image, (1, 2, 0))
            
        else:
            raise ValueError(f"Unsupported number of bands: {rasterio_src.count}")

        # Convert to uint8 if not already
        if image.dtype != np.uint8:
            image = image.astype(float) * (255.0 / image.max())
            image = image.astype(np.uint8)

        return image

    except Exception as e:
        print(f"Error converting rasterio image to numpy: {str(e)}")
        traceback.print_exc()
        # Return a small empty array
        return np.zeros((100, 100, 3), dtype=np.uint8)


def work_area_to_numpy(rasterio_src, work_area):
    """
    Extract image data from a work area as a numpy array.
    Properly handles colormaps and different band types.
    
    Args:
        rasterio_src: rasterio DatasetReader object
        work_area: WorkArea object or QRectF
            
    Returns:
        numpy.ndarray: Image data from the work area as numpy array (h, w, 3) for RGB, (h, w) for grayscale
    """
    if not rasterio_src:
        return None
    
    # If we got a WorkArea object, use its rect
    if hasattr(work_area, 'rect'):
        rect = work_area.rect
    else:
        rect = work_area
        
    # Create a rasterio window from the rect
    window = Window(
        col_off=int(rect.x()),
        row_off=int(rect.y()),
        width=int(rect.width()),
        height=int(rect.height())
    )
    
    try:
        if rasterio_src.count == 1 and rasterio_src.colormap(1):
            # Read the single band
            image = rasterio_src.read(1, window=window)
            # Get the colormap
            colormap = rasterio_src.colormap(1)
            
            # Create a lookup table for the colormap
            max_idx = max(colormap.keys()) + 1
            lut = np.zeros((max_idx, 3), dtype=np.uint8)
            
            # Fill the lookup table with RGB values
            for idx, color in colormap.items():
                if idx < max_idx:  # Safety check
                    lut[idx] = [color[0], color[1], color[2]]  # Ignore alpha
            
            # Clip image indices to valid range for the LUT
            image_indices = np.clip(image, 0, max_idx - 1).astype(np.uint8)
            
            # Use the image as indices into the lookup table
            # This is a vectorized operation that maps each pixel to its RGB value
            rgb_image = lut[image_indices]
            
            # Use the colorized RGB version of the image
            image = rgb_image
            
        elif rasterio_src.count < 3:
            # Grayscale image without colormap
            image = rasterio_src.read(1, window=window)
            
            # Convert to 3-channel grayscale image
            image = np.stack([image] * 3, axis=-1)
                
        else:
            # Read RGB bands
            image = rasterio_src.read([1, 2, 3], window=window)
            
            # Transpose to height, width, channels format
            image = np.transpose(image, (1, 2, 0))
        
        # Convert to uint8 if not already
        if image.dtype != np.uint8:
            if image.max() > 0:  # Avoid division by zero
                image = image.astype(float) * (255.0 / image.max())
            image = image.astype(np.uint8)

        return image
        
    except Exception as e:
        traceback.print_exc()
        return None
    

def pixmap_to_numpy(pixmap):
    """
    Convert a QPixmap to a NumPy array.

    :param pixmap: QPixmap to convert
    :return: numpy.ndarray in format (h, w, 3) with RGB values
    """
    try:
        image = pixmap.toImage()
        
        # Get image dimensions
        width = image.width()
        height = image.height()

        # Convert QImage to numpy array
        byte_array = image.bits().asstring(width * height * 4)  # 4 for RGBA
        numpy_array = np.frombuffer(byte_array, dtype=np.uint8).reshape((height, width, 4))

        # If the image format is ARGB32, swap the first and last channels (A and B)
        if format == QImage.Format_ARGB32:
            numpy_array = numpy_array[:, :, [2, 1, 0, 3]]
            
        numpy_array = numpy_array[:, :, :3]  # Remove the alpha channel if present
        
    except Exception as e:
        print(f"Error converting QImage to numpy: {e}")
        # Return a small empty array if conversion fails
        numpy_array = np.zeros((256, 256, 3), dtype=np.uint8)  
        
    return numpy_array


def attempt_download_asset(app, asset_name, asset_url):
    """
    Attempt to download an asset from the given URL.

    :param app:
    :param asset_name:
    :param asset_url:
    :return:
    """
    # Create a progress dialog
    progress_dialog = ProgressBar(app, title=f"Downloading {asset_name}")

    try:
        # Get the asset name
        asset_name = os.path.basename(asset_name)
        asset_path = os.path.join(os.getcwd(), asset_name)

        if os.path.exists(asset_path):
            return

        # Download the asset
        response = requests.get(asset_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        # Initialize the progress bar
        progress_dialog.start_progress(total_size // block_size)
        progress_dialog.show()

        with open(asset_path, 'wb') as f:
            for data in response.iter_content(block_size):
                if progress_dialog.wasCanceled():
                    raise Exception("Download canceled by user")
                f.write(data)
                progress_dialog.update_progress()

    except Exception as e:
        QMessageBox.critical(app, "Error", f"Failed to download {asset_name}.\n{e}")

    # Close the progress dialog
    progress_dialog.set_value(progress_dialog.max_value)
    progress_dialog.close()


def clean_polygon(xy_points, simplify_tolerance=0.1):
    """
    Filter a list of points to keep only the largest polygon and simplify it.
    
    :param xy_points: List of (x, y) coordinates that might form multiple polygons
    :param simplify_tolerance: Tolerance parameter for polygon simplification (higher values = more simplification)
    :return: List of (x, y) coordinates for the largest, simplified polygon
    """
    # Convert input points to a numpy array if not already
    if not isinstance(xy_points, np.ndarray):
        xy_points = np.array(xy_points)
    
    try:
        # Create a polygon from the points
        polygon = Polygon(xy_points)
        
        # If we have an invalid polygon, handle it
        if not polygon.is_valid:
            # Buffer(0) is a common trick to fix invalid polygons
            polygon = polygon.buffer(0)
        
        # Get all polygons (in case we have a MultiPolygon)
        if polygon.geom_type == 'MultiPolygon':
            # Find the polygon with the largest area
            polygons = list(polygon.geoms)
            largest_polygon = max(polygons, key=lambda p: p.area)
        else:
            largest_polygon = polygon
        
        # Simplify the largest polygon to reduce the number of vertices
        # preserve_topology=True ensures the simplified polygon doesn't self-intersect
        simplified_polygon = largest_polygon.simplify(tolerance=simplify_tolerance, preserve_topology=True)
        
        # In rare cases, simplification could create a MultiPolygon
        # If that happens, take only the largest part
        if simplified_polygon.geom_type == 'MultiPolygon':
            simplified_polygon = max(list(simplified_polygon.geoms), key=lambda p: p.area)
            
        # Extract the exterior coordinates from the simplified polygon
        simplified_coords = list(simplified_polygon.exterior.coords)
        
        # Return all points except the last one (Shapely adds a duplicate point at the end)
        return simplified_coords[:-1]
    
    except Exception as e:
        print(f"Error filtering/simplifying polygon: {e}")
        # Return original points if something went wrong
        return xy_points.tolist() if isinstance(xy_points, np.ndarray) else xy_points


# TODO deal with optimized model types
def check_model_architecture(weights_file):
    """
    Determine the model architecture type and task from weights file.

    Args:
        weights_file (str): Path to model weights (.pt or .pth)

    Returns:
        tuple: (architecture_type, task_type) where both are strings.
               Returns ("", "") if architecture cannot be determined.
    """
    try:
        model = torch.load(weights_file)
        if 'model' not in model:
            return "", ""

        if model['model'] is not None:
            decoder = model["model"].model[-1]
        else:
            decoder = model["ema"].model[-1]

        decoder_name = decoder.__class__.__name__

        if decoder_name == "RTDETRDecoder":
            return "rtdetr", "detect"

        if not any(task in decoder_name for task in ["Detect", "Segment", "Classify"]):
            return "", ""

        task_map = {
            "Detect": "detect",
            "Segment": "segment",
            "Classify": "classify"
        }

        for key, task in task_map.items():
            if key in decoder_name:
                return "yolo", task

    except Exception:
        return "", ""


def console_user(error_msg, parent=None):
    """
    Display an error message to the user via both terminal and GUI dialog.

    :param error_msg: The error message to display
    :param parent: Parent widget for the QMessageBox (optional)
    :return: None
    """
    url = "https://github.com/Jordan-Pierce/CoralNet-Toolbox/issues"

    # Show error in terminal
    print("\n\n\nUh oh! It looks like something went wrong!")
    print(f"{'∨' * 60}")
    print(f"\n{error_msg}\n")
    print(f"{'^' * 60}")
    print("Please create a ticket and copy this error so we can get this fixed:")
    print(f"{url}")


def except_hook(cls, exception, traceback_obj, main_window=None):
    """Handle uncaught exceptions including Qt errors"""
    error_msg = f"{cls.__name__}: {exception}\n\n"
    error_msg += ''.join(traceback.format_tb(traceback_obj))

    # Log the error
    print(error_msg)

    # If Qt is initialized, show error in GUI
    if QApplication.instance() is not None:
        msg_box = QMessageBox()
        msg_box.setWindowTitle("CoralNet-Toolbox Error")
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText(
            "An unexpected error occurred! Please copy the error below and create a ticket so we can solve this problem. If possible, save your project before closing the application."
        )
        msg_box.setDetailedText(error_msg)

        # Add Save Project option if main_window exists
        save_button = None
        if main_window is not None and hasattr(main_window, 'open_save_project_dialog'):
            save_button = QPushButton("Save Project")
            msg_box.addButton(save_button, QMessageBox.AcceptRole)

        msg_box.addButton(QMessageBox.Ok)

        # Make the dialog bigger
        msg_box.resize(600, 1000)

        result = msg_box.exec_()

        # Handle save action if requested
        if save_button and msg_box.clickedButton() == save_button:
            try:
                main_window.open_save_project_dialog()
            except Exception as save_error:
                QMessageBox.warning(None,
                                    "Save Error",
                                    f"Could not save project: {save_error}")

    sys.__excepthook__(cls, exception, traceback_obj)
    sys.exit(1)


def convert_to_ultralytics(ultralytics_model, weights, output_path="converted_model.pt"):
    """Convert a PyTorch model to Ultralytics format"""
    src_state_dict = ultralytics_model.model.model.state_dict()
    
    temp_model = torch.load(weights, map_location='cpu')
    dst_state_dict = temp_model['net']

    try:
        for (src_key, src_val), (dst_key, dst_val) in zip(src_state_dict.items(), dst_state_dict.items()):
            if src_val.shape == dst_val.shape:
                src_state_dict[src_key] = dst_val
            else:
                print("Warning: Skipping mismatched layer ", src_key, src_val.shape, dst_key, dst_val.shape)

    except Exception as e:
        print(f"Error converting model: {e}")
        return

    ultralytics_model.model.model.load_state_dict(src_state_dict)
    ultralytics_model.model.model.eval()
    
    ultralytics_model.task = 'classify'
    ultralytics_model.save(output_path)
    print(f"Model saved to {output_path}")

    del dst_state_dict
    gc.collect()
