# TODO Later:
#   - update pre-warm inference based on input size of model
#   - CoralNet tools
#   - file explorer for annotations to edit, delete in batches; spotlight?
#   - exporting image names for creating dataset
#   - filter images when creating dataset
#   - return early while loading old high resolution image
#   - include imgsz variable when evaluating model
#   - pass class_mapping to confusion matrix function

import traceback

from PyQt5.QtWidgets import QApplication

from toolbox.QtMain import MainWindow
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