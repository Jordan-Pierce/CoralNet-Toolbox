import warnings
import pkg_resources

import torch
import numpy as np

from PyQt5.QtGui import QImage

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def get_icon_path(icon_name):
    """

    :param icon_name:
    :return:
    """
    return pkg_resources.resource_filename('toolbox', f'icons/{icon_name}')


def get_available_device():
    """
    Get available devices

    :param self:
    :return:
    """
    devices = ['cpu',]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f'cuda:{i}')
    if torch.backends.mps.is_available():
        devices.append('mps')
    return devices


def pixmap_to_numpy(pixmap):
    """
    Convert a QPixmap to a NumPy array.

    :param pixmap:
    :return:
    """
    # Convert QPixmap to QImage
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

    return numpy_array[:, :, :3]


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


def console_user(error_msg):
    """

    :param error_msg:
    :return:
    """
    url = "https://github.com/Jordan-Pierce/CoralNet-Toolbox/issues"

    print(f"\n\n\nUh oh! It looks like something went wrong!")
    print(f"{'∨' * 60}")
    print(f"\n{error_msg}\n")
    print(f"{'^' * 60}")
    print(f"Please, create a ticket and copy this error so we can get this fixed:")
    print(f"{url}")