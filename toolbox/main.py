# TODO Later:
# CoralNet tools
# Resample polygon to be smoother, uniformly spaced
# Move the following to folder structures: batch inference, deploy model, export dataset, train model
# Create grouping, and form layout for all dialogs
# Add to the RangeSlider to allow for one handle as well, add spin boxes to either side of the slider, connect
# Create workarea class, add to eventFilters for deploying models
# Add to Results processes to deal with thresholding area, confidence, iou (add to deploy dialogs)
# Include Segment everything to eventFilters

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