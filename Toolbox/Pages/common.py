import os
import re
import sys
import time
import socket
import string
import argparse
import traceback
from tkinter import Tk, filedialog

from Toolbox.Tools.Common import DATA_DIR
from Toolbox.Tools.Common import LOG_PATH
from Toolbox.Tools.Common import PATCH_EXTRACTOR
from Toolbox.Tools.Common import FUNC_GROUPS_LIST


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

# Get the Pages directory
PAGES_DIR = os.path.dirname(os.path.abspath(__file__))

js = """function () {
  gradioURL = window.location.href
  if (!gradioURL.endsWith('?__theme=dark')) {
    window.location.replace(gradioURL + '?__theme=dark');
  }
}"""


# ----------------------------------------------------------------------------------------------------------------------
# Ports
# ----------------------------------------------------------------------------------------------------------------------

def get_port():
    """
    Find the first available port from the given list.
    """

    PORT_RANGE_START = 7860
    PORT_RANGE_END = 7880

    for port in list(range(PORT_RANGE_START, PORT_RANGE_END + 1)):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        if result != 0:  # If the connection attempt fails, the port is available
            return port

    raise RuntimeError("No available ports in the specified list.")


# ----------------------------------------------------------------------------------------------------------------------
# Logger
# ----------------------------------------------------------------------------------------------------------------------
def read_logs():
    sys.stdout.flush()

    with open(LOG_PATH, "r") as f:
        log_content = f.readlines()

    # Filter out lines containing null characters
    log_content = [line for line in log_content if '\x00' not in line]

    # Get the latest 30 lines
    recent_lines = log_content[-30:]

    return ''.join(recent_lines)


class Logger:
    def __init__(self, filename):

        self.filename = filename
        self.clear_console()
        self.terminal = sys.stdout
        self.reset_logs()
        self.log = open(self.filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False

    def clear_console(self):
        if os.name == 'nt':  # Windows
            os.system('cls')
        else:  # Linux, macOS, etc.
            os.system('clear')

    def reset_logs(self):
        with open(self.filename, 'w') as file:
            file.truncate(0)


# ----------------------------------------------------------------------------------------------------------------------
# Browsing
# ----------------------------------------------------------------------------------------------------------------------
def choose_file():
    """

    """
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()

    file_path = filedialog.askopenfilename()
    if file_path:
        if os.path.isfile(file_path):
            root.destroy()
            return str(file_path)
        else:
            root.destroy()
            return "Invalid file selected"
    else:
        file_path = "File not selected"
        root.destroy()
        return file_path


def choose_files():
    """

    """
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()

    file_paths = list(filedialog.askopenfilenames())

    if file_paths:
        valid_file_paths = " ".join([path for path in file_paths if os.path.isfile(path)])
        root.destroy()
        return valid_file_paths
    else:
        root.destroy()
        return []


def choose_directory():
    """

    """
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()

    dir_path = filedialog.askdirectory()
    if dir_path:
        if os.path.isdir(dir_path):
            root.destroy()
            return str(dir_path)
        else:
            root.destroy()
            return "Invalid directory selected"
    else:
        dir_path = "Folder not selected"
        root.destroy()
        return str(dir_path)
