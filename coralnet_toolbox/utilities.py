import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import gc
import sys
import requests
import traceback

import torch
import numpy as np

from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import Qt, QBuffer, QByteArray
from PyQt5.QtWidgets import QMessageBox, QApplication, QTextEdit, QPushButton

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

        decoder = model["model"].model[-1]
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


def preprocess_image(image):
    """
    Ensure the image has correct dimensions (h, w, 3).

    :param image:
    :return:
    """
    if len(image.shape) == 2:  # Grayscale image
        image = np.stack((image,) * 3, axis=-1)
    elif len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA image
            image = image[..., :3]  # Drop alpha channel
        elif image.shape[2] != 3:  # If channels are not last
            # Check if channels are first (c, h, w)
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            elif image.shape[0] == 4:  # RGBA in channels-first format
                image = np.transpose(image, (1, 2, 0))[..., :3]
            else:
                raise ValueError("Image must have 3 or 4 color channels")
    else:
        raise ValueError("Image must be 2D or 3D array")

    return image


def open_image(image_path):
    """
    Open an image from the given path.

    :param image_path:
    :return:
    """
    return preprocess_image(qimage_to_numpy(QImage(image_path)))


def rasterio_to_numpy(rasterio_src):
    """
    Convert a Rasterio dataset to a NumPy array.

    :param rasterio_src:
    :return:
    """
    return rasterio_src.read().transpose(1, 2, 0)


def pixmap_to_numpy(pixmap):
    """
    Convert a QPixmap to a NumPy array.

    :param pixmap:
    :return:
    """
    # Initialize a blank numpy array
    numpy_array = np.zeros((256, 256, 3), dtype=np.uint8)
    
    try:
        # Convert QPixmap to QImage
        image = pixmap.toImage()
        
    except Exception as e:
        print(f"Error converting QPixmap to QImage: {e}")
        # Handle error by returning a blank numpy array
        return numpy_array

    try:
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

    return numpy_array


def qimage_to_numpy(qimage):
    """
    Convert a QImage to a NumPy array.

    :param qimage:
    :return:
    """
    # Get image dimensions
    width = qimage.width()
    height = qimage.height()
    # Get the number of bytes per line
    bytes_per_line = qimage.bytesPerLine()
    # Convert QImage to numpy array
    byte_array = qimage.bits().asstring(height * bytes_per_line)
    image = np.frombuffer(byte_array, dtype=np.uint8).reshape((height, width, 4))
    return image[:, :, :3]  # Remove the alpha channel if present


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
