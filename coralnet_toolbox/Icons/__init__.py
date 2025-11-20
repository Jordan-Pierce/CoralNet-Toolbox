import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

from importlib.resources import files
from PyQt5.QtGui import QIcon


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def get_icon_path(icon_name):
    """

    :param icon_name:
    :return:
    """
    icon_dir = files('coralnet_toolbox').joinpath('Icons')
    return str(icon_dir.joinpath(icon_name))


def get_icon(icon_name):
    """

    :param icon_name:
    :return:
    """
    return QIcon(get_icon_path(icon_name))