# TODO
#   - update pre-warm inference based on input size of model
#   - CoralNet tools
#   - Clean up QImage, Rasterio Image, PixMap
#   - Write Class Mapping JSON when creating a dataset
#   - Read Class Mapping JSON when training a model, stash
#   - Add labels when importing a model for deployment

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