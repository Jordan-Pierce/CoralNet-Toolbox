import os
import re
import sys
import socket
import string
from tkinter import Tk, filedialog

from Toolbox.Tools.Common import LOG_PATH

# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

# Get the Pages directory
PAGES_DIR = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------------------------------------------------
# Ports
# ----------------------------------------------------------------------------------------------------------------------

def find_available_ports(ports):
    """
    Find available ports from the given list.
    """
    available_ports = []
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        if result != 0:  # If the connection attempt fails, the port is available
            available_ports.append(port)

    if not available_ports:
        raise RuntimeError("No available ports in the specified list.")

    return available_ports


PROCESS_LIST = ["main", "download", "api"]
PORT_RANGE_START = 7860
PORT_RANGE_END = 7880

# Create a list of ports for each process
process_ports = find_available_ports(list(range(PORT_RANGE_START, PORT_RANGE_END + 1)))

# Create a dictionary mapping each process to its assigned port
SERVER_PORTS = dict(zip(PROCESS_LIST, process_ports))


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
    sys.stdout.flush()
    with open(LOG_PATH, "r") as f:
        log_content = f.read()
        log_content = filter_logs(log_content)
        log_content = filter_printable(log_content)
        return log_content


def reset_logs():
    with open(LOG_PATH, 'w') as file:
        pass


def filter_printable(text):
    printable_chars = set(string.printable)
    return ''.join(char for char in text if char in printable_chars)


def filter_logs(text):
    return re.sub(r'^.*progress:.*$', '', text, flags=re.MULTILINE)


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