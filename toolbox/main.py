# TODO
#   - update pre-warm inference based on input size of model
#   - CoralNet tools
#   - Short or Long code for ML?
#   - if import labels are too long, add "..."
#   - don't crash when importing the same image twice
#   - Clean up QImage, Rasterio Image, PixMap
#   - Update labels found with model deploy

import traceback
from PyQt5.QtWidgets import QApplication
from toolbox.QtMain import MainWindow
from toolbox.utilities import console_user


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