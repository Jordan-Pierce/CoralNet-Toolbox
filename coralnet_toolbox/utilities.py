import pkg_resources

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_icon_path(icon_name):
    return pkg_resources.resource_filename('coralnet_toolbox', f'icons/{icon_name}')