# TODO Later:
#   - CoralNet tools
#   - exporting image names for creating dataset
#   - Sample may cause crash
#   - Clicking outside of SAM working area causes a crash, fix that
#   - Import and export of instance segmentation masks / object detection YOLO format
#   - Train models
#   - Deploy

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