import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import gc
import sys
import requests
import traceback
from functools import lru_cache

import cv2
import torch
import numpy as np

import rasterio
from rasterio.windows import Window

from shapely.geometry import Polygon

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QMessageBox, QApplication, QPushButton

from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


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

        # Test read a small sample to catch corruption early
        try:
            test_window = rasterio.windows.Window(0, 0, min(10, src.width), min(10, src.height))
            src.read(1, window=test_window)
        except Exception as read_error:
            src.close()
            raise read_error

        return src
    except Exception as e:
        error_msg = f"Error opening image with rasterio: {image_path}\nException: {str(e)}"

        # Try to inspect file existence and permissions for more detailed error info
        if not os.path.exists(image_path):
            error_msg += f"\nFile does not exist: {image_path}"
        elif not os.path.isfile(image_path):
            error_msg += f"\nPath is not a file: {image_path}"
        elif not os.access(image_path, os.R_OK):
            error_msg += f"\nFile is not readable: {image_path}"
        else:
            error_msg += f"\nFile appears to be corrupted or in an unsupported format"

        # Show critical message dialog if Qt application is available
        if QApplication.instance() is not None:
            QMessageBox.critical(
                None,
                "Image Loading Error",
                f"Failed to open image file:\n\n{error_msg}\n\nThis file may be corrupted or in an unsupported format."
            )

        # Print to console for logging
        print(error_msg)

        # Raise a custom exception with detailed information
        raise RuntimeError(f"Failed to open rasterio image: {image_path}. {error_msg}")


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
        # Check if the dataset is closed
        if not rasterio_src or getattr(rasterio_src, 'closed', True):
            # Attempt to reopen the dataset if we can get the path
            if hasattr(rasterio_src, 'name'):
                try:
                    rasterio_src = rasterio.open(rasterio_src.name)
                except Exception as reopen_error:
                    print(f"Error reopening dataset: {str(reopen_error)}")
                    return QImage()
            else:
                print("Cannot read from closed dataset without path information")
                return QImage()
                
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
        has_colormap = False
        if rasterio_src.count == 1:
            try:
                has_colormap = rasterio_src.colormap(1) is not None
            except ValueError:
                has_colormap = False

        if rasterio_src.count == 1 and has_colormap:
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

        # Convert to uint8 if image is not already
        if image.dtype != np.uint8:
            if image.max() > 0:  # Avoid division by zero
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
        # Check if the dataset is closed
        if not rasterio_src or getattr(rasterio_src, 'closed', True):
            # Attempt to reopen the dataset if we can get the path
            if hasattr(rasterio_src, 'name'):
                try:
                    rasterio_src = rasterio.open(rasterio_src.name)
                except Exception as reopen_error:
                    print(f"Error reopening dataset: {str(reopen_error)}")
                    return QImage()
            else:
                print("Cannot read from closed dataset without path information")
                return QImage()

        # Check for single-band image with colormap
        has_colormap = False
        if rasterio_src.count == 1:
            try:
                has_colormap = rasterio_src.colormap(1) is not None
            except ValueError:
                has_colormap = False

        if rasterio_src.count == 1 and has_colormap:
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
            if image.max() > 0:  # Avoid division by zero
                image = image.astype(float) * (255.0 / image.max())
            image = image.astype(np.uint8)

        # Convert the numpy array to QImage
        qimage = QImage(image.data.tobytes(),
                        int(window.width),
                        int(window.height),
                        int(window.width * num_bands),  # bytes per line
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
        # Check if the dataset is closed
        if not rasterio_src or getattr(rasterio_src, 'closed', True):
            # Attempt to reopen the dataset if we can get the path
            if hasattr(rasterio_src, 'name'):
                try:
                    rasterio_src = rasterio.open(rasterio_src.name)
                except Exception as reopen_error:
                    print(f"Error reopening dataset: {str(reopen_error)}")
                    return np.zeros((100, 100, 3), dtype=np.uint8)
            else:
                print("Cannot read from closed dataset without path information")
                return np.zeros((100, 100, 3), dtype=np.uint8)
                
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
        has_colormap = False
        if rasterio_src.count == 1:
            try:
                has_colormap = rasterio_src.colormap(1) is not None
            except ValueError:
                has_colormap = False

        if rasterio_src.count == 1 and has_colormap:
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
            if image.max() > 0:  # Avoid division by zero
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
        
    # Check if the dataset is closed
    if getattr(rasterio_src, 'closed', True):
        # Attempt to reopen the dataset if we can get the path
        if hasattr(rasterio_src, 'name'):
            try:
                rasterio_src = rasterio.open(rasterio_src.name)
            except Exception as reopen_error:
                print(f"Error reopening dataset: {str(reopen_error)}")
                return None
        else:
            print("Cannot read from closed dataset without path information")
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
        # Check for single-band image with colormap
        has_colormap = False
        if rasterio_src.count == 1:
            try:
                has_colormap = rasterio_src.colormap(1) is not None
            except ValueError:
                has_colormap = False

        if rasterio_src.count == 1 and has_colormap:
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


def pixmap_to_pil(pixmap):
    """
    Convert a QPixmap to a PIL Image.
    
    :param pixmap: QPixmap to convert
    :return: PIL Image in RGB format
    """
    from PIL import Image
    
    # Convert pixmap to numpy array first
    image_np = pixmap_to_numpy(pixmap)
    
    # Convert numpy array to PIL Image
    if len(image_np.shape) == 2:  # Grayscale
        pil_image = Image.fromarray(image_np, mode='L').convert('RGB')
    else:  # RGB
        pil_image = Image.fromarray(image_np, mode='RGB')
    
    return pil_image


def scale_pixmap(pixmap, max_size):
    """Scale pixmap and graphic if they exceed max dimension while preserving aspect ratio"""
    width = pixmap.width()
    height = pixmap.height()
    
    # Check if scaling is needed
    if width <= max_size and height <= max_size:
        return pixmap
        
    # Calculate scale factor based on largest dimension
    scale = max_size / max(width, height)
    
    # Scale pixmap
    scaled_pixmap = pixmap.scaled(
        int(width * scale), 
        int(height * scale),
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation
    )
    
    return scaled_pixmap


def convert_scale_units(value, from_unit, to_unit):
    """
    Convert a value from one unit to another. Supports common metric and imperial units.

    Args:
        value (float): The value to convert.
        from_unit (str): The unit to convert from (e.g., 'metre', 'm', 'cm', 'foot', 'us survey foot').
        to_unit (str): The unit to convert to (e.g., 'mm', 'cm', 'm', 'km').

    Returns:
        float: The converted value.
    """
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    # Conversion factors to meters
    to_meters = {
        'metre': 1.0,
        'm': 1.0,
        'mm': 0.001,
        'cm': 0.01,
        'km': 1000.0,
        'in': 0.0254,
        'ft': 0.3048,
        'foot': 0.3048,
        'yd': 0.9144,
        'mi': 1609.344,
        'us survey foot': 1200 / 3937,
    }

    # Conversion factors from meters
    from_meters = {
        'mm': 1000.0,
        'cm': 100.0,
        'metre': 1.0,
        'm': 1.0,
        'km': 0.001,
        'in': 1 / 0.0254,
        'ft': 1 / 0.3048,
        'foot': 1 / 0.3048,
        'yd': 1 / 0.9144,
        'mi': 1 / 1609.344,
        'us survey foot': 3937 / 1200,
    }

    if from_unit not in to_meters:
        # If from_unit is unknown, return original value as a fallback
        return value

    # Convert from_unit to meters
    value_in_meters = value * to_meters[from_unit]

    if to_unit not in from_meters:
        # If to_unit is unknown, return value in meters as a fallback
        return value_in_meters

    # Convert from meters to to_unit
    return value_in_meters * from_meters[to_unit]


def load_z_channel_from_file(z_channel_path, target_width=None, target_height=None):
    """
    Load a depth map / height map / DEM from file using rasterio.
    
    The z_channel data will be either:
    - float32: Actual depth/height values (e.g., meters, feet, etc.)
    - uint8: Relative depth/height values (0-255 range)
    
    Args:
        z_channel_path (str): Path to the depth/height/DEM file
        target_width (int, optional): Target width to match raster dimensions
        target_height (int, optional): Target height to match raster dimensions
        
    Returns:
        tuple: (z_data, z_path, z_nodata) where z_data is a 2D numpy array (float32 or uint8),
               z_path is the file path, and z_nodata is the nodata value from the GeoTIFF
               (or None if no nodata value is defined), or (None, None, None) if loading fails
    """
    try:
        # Check if file exists
        if not os.path.exists(z_channel_path):
            print(f"Z-channel file does not exist: {z_channel_path}")
            return None, None, None
            
        # Open the z-channel file with rasterio
        with rasterio.open(z_channel_path) as src:
            # Validate it's a single band file
            if src.count != 1:
                print(f"Z-channel file must be single band, found {src.count} bands: {z_channel_path}")
                return None, None, None
            
            # Extract the nodata value from the rasterio source
            z_nodata = src.nodata
            if z_nodata is not None:
                print(f"Z-channel has nodata value: {z_nodata}")
            
            # Read the single band
            z_data = src.read(1)
            
            # Check if we need to resize to match target dimensions
            if target_width is not None and target_height is not None:
                if z_data.shape != (target_height, target_width):
                    # Resample to match target dimensions
                    window = Window(0, 0, src.width, src.height)
                    z_data = src.read(1,
                                      window=window,
                                      out_shape=(target_height, target_width),
                                      resampling=rasterio.enums.Resampling.bilinear)
                    print(f"Resampled z-channel from {src.height}x{src.width} to {target_height}x{target_width}")
            
            # Handle data type conversion
            if z_data.dtype == np.uint8:
                # Already uint8, no conversion needed
                pass
            elif z_data.dtype in [np.float32, np.float64]:
                # Keep as float32 for actual depth/height values
                z_data = z_data.astype(np.float32)
            elif z_data.dtype in [np.int8, np.int16, np.int32, np.uint16, np.uint32]:
                # Convert integer types to float32 for better precision
                z_data = z_data.astype(np.float32)
            else:
                print(f"Warning: Unsupported z-channel data type {z_data.dtype}, converting to float32")
                z_data = z_data.astype(np.float32)
                
            # Preserve NaN values in floating-point data (don't convert to 0)
            # NaN values represent missing or NULL data and will be handled by the UI layer
            if np.issubdtype(z_data.dtype, np.floating):
                nan_count = np.sum(np.isnan(z_data))
                if nan_count > 0:
                    print(f"Z-channel contains {nan_count} NaN values (NULL/missing data)")
            
            # Final validation - ensure 2D array
            if z_data.ndim != 2:
                print(f"Z-channel data must be 2D, found {z_data.ndim}D")
                return None, None, None
                
            # Final data type check
            if z_data.dtype not in [np.float32, np.uint8]:
                print(f"Z-channel data type {z_data.dtype} not supported, must be float32 or uint8")
                return None, None, None
                
            print(f"Successfully loaded z-channel: {z_data.shape}, dtype: {z_data.dtype}, "
                  f"range: [{np.min(z_data):.2f}, {np.max(z_data):.2f}]")
                  
            return z_data, z_channel_path, z_nodata
            
    except Exception as e:
        print(f"Error loading z-channel from {z_channel_path}: {str(e)}")
        traceback.print_exc()
        return None, None, None
    

def detect_z_channel_units_from_file(z_channel_path):
    """
    Attempt to detect z-channel units from GeoTIFF metadata (CRS).
    
    Args:
        z_channel_path (str): Path to the z-channel file
        
    Returns:
        tuple: (units_str, confidence) where confidence is 'high', 'medium', or None
               Examples: ('metres', 'high'), ('feet', 'high'), (None, None)
    """
    try:
        if not os.path.exists(z_channel_path):
            return None, None
            
        with rasterio.open(z_channel_path) as src:
            # Try to get CRS information
            if src.crs is not None:
                try:
                    # Get the linear units from the CRS
                    linear_units = src.crs.linear_units
                    if linear_units:
                        # rasterio returns units as lowercase string (e.g., 'metre', 'foot')
                        return linear_units.lower(), 'high'
                except Exception as e:
                    print(f"Warning: Could not extract linear units from CRS: {e}")
                    return None, None
            
            # If no CRS, return None
            return None, None
            
    except Exception as e:
        print(f"Error detecting z-channel units from {z_channel_path}: {e}")
        return None, None


def normalize_z_unit(unit_str):
    """
    Normalize a z-unit string to a standard short-form format.
    This ensures consistency with scale unit conventions.
    
    Args:
        unit_str (str): Unit string to normalize (e.g., 'metre', 'm', 'foot', 'ft')
        
    Returns:
        str: Normalized unit string in short form (e.g., 'm', 'ft', 'cm', 'px')
    """
    if unit_str is None:
        return None
        
    unit_str = unit_str.lower().strip()
    
    # Map various spellings to normalized short forms
    unit_map = {
        # Metric - to short form
        'metre': 'm',
        'meter': 'm',
        'meters': 'm',
        'metres': 'm',
        'mm': 'mm',
        'millimeter': 'mm',
        'millimeters': 'mm',
        'millimetres': 'mm',
        'cm': 'cm',
        'centimeter': 'cm',
        'centimeters': 'cm',
        'centimetres': 'cm',
        'km': 'km',
        'kilometer': 'km',
        'kilometers': 'km',
        'kilometres': 'km',
        
        # Imperial - to short form
        'ft': 'ft',
        'foot': 'ft',
        'feet': 'ft',
        'in': 'in',
        'inch': 'in',
        'inches': 'in',
        'yd': 'yd',
        'yard': 'yd',
        'yards': 'yd',
        'mi': 'mi',
        'mile': 'mi',
        'miles': 'mi',
        'us survey foot': 'ft',
        'us survey feet': 'ft',
        
        # Special - to short form
        'pixel': 'px',
        'pixels': 'px',
        'px': 'px',
        'pix': 'px',
    }
    
    return unit_map.get(unit_str, unit_str)


def get_standard_z_units():
    """
    Get a list of standard z-channel units for UI selection.
    Returns short-form units matching scale unit conventions.
    
    Returns:
        list: List of unit abbreviations ('mm', 'cm', 'm', etc.)
    """
    return ['mm', 'cm', 'm', 'km', 'in', 'ft', 'yd', 'mi', 'px']
    

def simplify_polygon(xy_points, simplify_tolerance=0.1):
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
        print(f"Warning: Error filtering/simplifying polygon: {e}")
        # Return original points if something went wrong
        return xy_points.tolist() if isinstance(xy_points, np.ndarray) else xy_points


def densify_polygon(xy_points):
    """
    Densify a polygon by adding one vertex between each pair of consecutive vertices.
    :param xy_points: List of (x, y) coordinates
    :return: List of (x, y) coordinates with more points
    """
    if not isinstance(xy_points, np.ndarray):
        xy_points = np.array(xy_points)

    try:
        # Remove duplicate last point if present (Shapely convention)
        if np.allclose(xy_points[0], xy_points[-1]):
            xy_points = xy_points[:-1]

        densified = []
        n = len(xy_points)
        for i in range(n):
            p1 = xy_points[i]
            p2 = xy_points[(i + 1) % n]  # wrap around for closed polygon
            densified.append(tuple(p1))
            # Insert midpoint
            midpoint = (p1 + p2) / 2
            densified.append(tuple(midpoint))
        return densified

    except Exception as e:
        print(f"Warning: Error densifying polygon: {e}")
        return xy_points.tolist() if isinstance(xy_points, np.ndarray) else xy_points


def polygonize_mask_with_holes(mask_tensor):
    """
    Converts a boolean mask tensor to an exterior polygon and a list of interior hole polygons.

    Args:
        mask_tensor (torch.Tensor): A 2D boolean tensor from the prediction results.

    Returns:
        A tuple containing:
        - exterior (list of tuples): The (x, y) vertices of the outer boundary.
        - holes (list of lists of tuples): A list where each element is a list of (x, y) vertices for a hole.
    """
    # Convert the tensor to a NumPy array format that OpenCV can use
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)

    # Find all contours and their hierarchy
    # cv2.RETR_CCOMP organizes contours into a two-level hierarchy: external boundaries and holes inside them.
    contours, hierarchy = cv2.findContours(mask_np, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if not contours or hierarchy is None:
        return [], []

    exterior = []
    holes = []

    # Process the hierarchy to separate the exterior from the holes
    for i, contour in enumerate(contours):
        # An external contour's parent in the hierarchy is -1
        if hierarchy[0][i][3] == -1:
            # Squeeze to convert from [[x, y]] to [x, y] format
            exterior = contour.squeeze(axis=1).tolist()
        else:
            # Any other contour is treated as a hole
            holes.append(contour.squeeze(axis=1).tolist())

    return exterior, holes


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
