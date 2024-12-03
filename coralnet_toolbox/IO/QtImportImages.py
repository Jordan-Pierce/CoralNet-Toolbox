import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox)

from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ImportImages:
    def __init__(self, main_window):
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

    def import_images(self):
        self.main_window.untoggle_all_tools()

        file_names, _ = QFileDialog.getOpenFileNames(self.image_window,
                                                     "Open Image Files",
                                                     "",
                                                     "Image Files (*.png *.jpg *.jpeg *.tif* *.bmp)")
        if file_names:
            # Set the cursor to waiting (busy) cursor
            QApplication.setOverrideCursor(Qt.WaitCursor)

            progress_bar = ProgressBar(self.image_window, title="Importing Images")
            progress_bar.show()
            progress_bar.start_progress(len(file_names))
            progress_bar.set_value(1)

            for i, file_name in enumerate(file_names):
                if file_name not in set(self.image_window.image_paths):
                    self.image_window.add_image(file_name)
                progress_bar.update_progress()

            progress_bar.stop_progress()
            progress_bar.close()

            # Update filtered images
            self.image_window.filter_images()
            # Show the last image
            self.image_window.load_image_by_path(self.image_window.image_paths[-1])

            # Restore the cursor to the default cursor
            QApplication.restoreOverrideCursor()

            QMessageBox.information(self.image_window,
                                    "Image(s) Imported",
                                    "Image(s) have been successfully imported.")