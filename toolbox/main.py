# TODO Later:
#   - update pre-warm inference based on input size of model
#   - CoralNet tools
#   - file explorer for annotations to edit, delete in batches
#   - master transparency for all annotations / classes
#   - exporting image names for creating dataset
#   - filter images when creating dataset
#   - filter, prev, next, all images when batch inference

# TODO Now:
#   - add importing of coralnet predictions / output (also for Viscore); Keep all of the input data, and export with
#   - Reef Future short label code json?
#   - Batch inference goes to every image, even without annotations?
#   - progress bar for cropping images when sampling
#   - Reading tiled tifs and show increasing resolution as it loads (if they are tiled), viewed; rasterio image pyramid

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