import os
import glob


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

# Get the directory where the config files are stored
CONFIG_DIR = os.path.dirname(__file__)


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def get_available_configs(task=None):
    """Get available model configuration files.
    
    Args:
        task (str, optional): Filter by task type ('classify', 'detect', 'segment').
            If None, return all configs.
    
    Returns:
        dict: Dictionary with config names as keys and file paths as values
    """
    configs = {}
    
    # Determine which directories to search
    if task:
        task_dirs = [os.path.join(CONFIG_DIR, task)]
    else:
        sub_dirs = os.listdir(CONFIG_DIR)
        task_dirs = [os.path.join(CONFIG_DIR, d) for d in sub_dirs if os.path.isdir(os.path.join(CONFIG_DIR, d))]
    
    # Find all YAML files
    for task_dir in task_dirs:
        if os.path.exists(task_dir):
            for yaml_file in glob.glob(os.path.join(task_dir, "*.yaml")):
                config_name = os.path.basename(yaml_file)
                configs[config_name] = yaml_file
    
    return configs