# TODO Later:
#   - CoralNet tools
#   - Merge Datasets
#   - Resample polygon to be smoother, uniformly spaced
#   - Don't access all points when updating polygon
#   - Fix the cuda device string
#   - Show a loading model window when using AutoDistill (like SAM)
#   - Look into SAM resizing images bigger than 1024

import traceback

from PyQt5.QtWidgets import QApplication

from toolbox.QtMainWindow import MainWindow
from toolbox.utilities import console_user


# ----------------------------------------------------------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------------------------------------------------------


def run():
    try:
        app = QApplication([])
        app.setStyle('WindowsXP')
        main_window = MainWindow()
        main_window.show()
        app.exec_()

    except Exception as e:
        console_user(f"{e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    run()