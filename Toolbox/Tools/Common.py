import os
import sys
import logging
import datetime

# ------------------------------------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------------------------------------

# Get the current script's directory (where init_project.py is located)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory (Project) to the Python path
PROJECT_DIR = os.path.dirname(script_dir)
sys.path.append(PROJECT_DIR)

# Make the Data directory
DATA_DIR = f"{os.path.dirname(PROJECT_DIR)}\\Data"
os.makedirs(DATA_DIR, exist_ok=True)

# Make the Cache directory
CACHE_DIR = f"{DATA_DIR}\\Cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Patch extractor path
PATCH_EXTRACTOR = f'{PROJECT_DIR}\\Tools\\Patch_Extractor\\CNNDataExtractor.exe'

# For all the logging
LOG_PATH = f"{DATA_DIR}\\Cache\\logs.log"

# MIR specific, mapping path
MIR_MAPPING = f'{DATA_DIR}\\Mission_Iconic_Reefs\\MIR_VPI_CoralNet_Mapping.csv'

# Coralnet labelset file for dropdown menu in gooey
CORALNET_LABELSET_FILE = f"{CACHE_DIR}\\CoralNet_Labelset_List.csv"

# Constant for the CoralNet url
CORALNET_URL = "https://coralnet.ucsd.edu"

# CoralNet Source page, lists all sources
CORALNET_SOURCE_URL = CORALNET_URL + "/source/about/"

# CoralNet Labelset page, lists all labelsets
CORALNET_LABELSET_URL = CORALNET_URL + "/label/list/"

# URL of the login page
LOGIN_URL = "https://coralnet.ucsd.edu/accounts/login/"

# CoralNet functional groups
FUNC_GROUPS_LIST = [
    "Other Invertebrates",
    "Hard coral",
    "Soft Substrate",
    "Hard Substrate",
    "Other",
    "Algae",
    "Seagrass"]

# Mapping from group to ID
FUNC_GROUPS_DICT = {
    "Other Invertebrates": "14",
    "Hard coral": "10",
    "Soft Substrate": "15",
    "Hard Substrate": "16",
    "Other": "18",
    "Algae": "19",
    "Seagrass": "20"}

# Image Formats
IMG_FORMATS = ["jpg", "jpeg", "png", "tif", "tiff", "bmp"]


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------


def print_progress(prg, prg_total):
    """
    Formatted print for Gooey to show progress in progress bar
    """
    log("progress: {}/{}".format(prg, prg_total))


def get_now():
    """
    Returns a timestamp; used for file and folder names
    """
    # Get the current datetime
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    return now


def setup_logger(log_file_path):
    """

    """
    # Create a logger instance
    logger = logging.getLogger(__name__)

    # Configure the root logger (optional, if you want to set a default level)
    logger.setLevel(logging.DEBUG)

    # Create a file handler to log to the specified file (INFO level)
    file_handler_info = logging.FileHandler(log_file_path)
    file_handler_info.setLevel(logging.INFO)

    # Create a console handler to log to the console (INFO level)
    console_handler_info = logging.StreamHandler()
    console_handler_info.setLevel(logging.INFO)

    # Set the formatter for all handlers
    formatter = logging.Formatter('%(message)s')
    file_handler_info.setFormatter(formatter)
    console_handler_info.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler_info)
    logger.addHandler(console_handler_info)

    return logger


# Setup logger
LOGGER = setup_logger(LOG_PATH)


# Define a custom logging function that mimics 'print'
def log(*args):
    message = ' '.join(map(str, args))
    LOGGER.info(message)  # Log the message at INFO level
