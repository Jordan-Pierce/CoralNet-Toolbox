import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pkg_resources

from PyQt5.QtGui import QIcon


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def get_icon_path(icon_name):
    """

    :param icon_name:
    :return:
    """
    return pkg_resources.resource_filename('coralnet_toolbox', f'Icons/{icon_name}')


def get_icon(icon_name):
    """

    :param icon_name:
    :return:
    """
    return QIcon(get_icon_path(icon_name))