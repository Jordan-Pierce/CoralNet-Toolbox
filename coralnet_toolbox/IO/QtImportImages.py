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
            self._process_image_files(file_names)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        file_names = [url.toLocalFile() for url in urls if url.isLocalFile()]

        if file_names:
            self._process_image_files(file_names)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        event.accept()
    
    def _process_image_files(self, file_names):
        """Helper method to process a list of image files with progress tracking."""
        # Make the cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        progress_bar = ProgressBar(self.image_window, title="Importing Images")
        progress_bar.show()
        progress_bar.start_progress(len(file_names))

        try:
            # Add images to the image window
            for file_name in file_names:
                if file_name not in set(self.image_window.image_paths):
                    try:
                        self.image_window.add_image(file_name)
                    except Exception as e:
                        print(f"Warning: Could not import image {file_name}. Error: {e}")

                # Update the progress bar
                progress_bar.update_progress()

            # Update filtered images
            self.image_window.filter_images()
            
            # Show the last image
            if self.image_window.image_paths:
                self.image_window.load_image_by_path(self.image_window.image_paths[-1])

            self._show_success_message()
        except Exception as e:
            self._show_error_message(str(e))
        finally:
            # Restore the cursor to the default cursor
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()
    
    def _show_success_message(self):
        """Display a success message after importing images."""
        QMessageBox.information(self.image_window,
                                "Image(s) Imported",
                                "Image(s) have been successfully imported.")
    
    def _show_error_message(self, error_msg):
        """Display an error message when image import fails."""
        QMessageBox.warning(self.image_window,
                            "Error Importing Image(s)",
                            f"An error occurred while importing image(s): {error_msg}")
