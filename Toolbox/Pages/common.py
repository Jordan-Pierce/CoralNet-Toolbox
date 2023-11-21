import os
import sys
from tkinter import Tk, filedialog

from Toolbox.Tools.Common import LOG_PATH


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

# Get the Pages directory
PAGES_DIR = os.path.dirname(os.path.abspath(__file__))


SERVER_PORTS = {"main": 7860,
                "download": 7861,
                "test": 7862}


# ----------------------------------------------------------------------------------------------------------------------
# Logger
# ----------------------------------------------------------------------------------------------------------------------

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


def read_logs():
    """

    """
    sys.stdout.flush()
    with open(LOG_PATH, "r") as f:
        return f.read()


def reset_logs():
    """

    """
    # Clear it
    with open(LOG_PATH, 'w') as file:
        pass


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def choose_directory():
    """

    """
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()

    filename = filedialog.askdirectory()
    if filename:
        if os.path.isdir(filename):
            root.destroy()
            return str(filename)
        else:
            root.destroy()
            return str(filename)
    else:
        filename = "Folder not selected"
        root.destroy()
        return str(filename)