# TODO Later:
#   - update pre-warm inference based on input size of model
#   - CoralNet tools
#   - file explorer for annotations to edit, delete in batches
#   - master transparency for all annotations / classes
#   - exporting image names for creating dataset
#   - filter images when creating dataset

# TODO Now:
#   - add importing of coralnet predictions / output  for Viscore Keep all of the input data, and export with

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