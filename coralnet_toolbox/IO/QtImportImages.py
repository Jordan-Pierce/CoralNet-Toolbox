import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox)

from coralnet_toolbox.QtProgressBar import ProgressBar


SUPPORTED_IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff'}


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ImportImages:
    def __init__(self, main_window):
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

    @staticmethod
    def _is_supported_image_file(file_name: str) -> bool:
        return os.path.splitext(file_name)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS

    def import_images(self):
        self.main_window.untoggle_all_tools()

        file_names, _ = QFileDialog.getOpenFileNames(self.image_window,
                                                     "Open Image Files",
                                                     "",
                                                     "Image Files (*.png *.jpg *.jpeg *.tif* *.bmp)")
        if file_names:
            self._process_image_files(file_names)

    def import_orthomosaics(self):
        """Import files as OrthoRaster instances (orthomosaics)."""
        self.main_window.untoggle_all_tools()

        file_names, _ = QFileDialog.getOpenFileNames(self.image_window,
                                                     "Open Orthomosaic Files",
                                                     "",
                                                     "Image Files (*.png *.jpg *.jpeg *.tif* *.bmp)")
        if file_names:
            self._process_image_files(file_names, raster_type='OrthoRaster')

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            file_names = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            if any(self._is_supported_image_file(file_name) for file_name in file_names):
                event.acceptProposedAction()
            else:
                event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        file_names = [url.toLocalFile() for url in urls
                      if url.isLocalFile() and self._is_supported_image_file(url.toLocalFile())]

        if file_names:
            self._process_image_files(file_names, suppress_errors=True)
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            file_names = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            if any(self._is_supported_image_file(file_name) for file_name in file_names):
                event.acceptProposedAction()
            else:
                event.ignore()

    def dragLeaveEvent(self, event):
        event.accept()
    
    def _process_image_files(self, file_names, raster_type: str = None, suppress_errors: bool = False):
        """Helper method to process a list of image files with progress tracking.

        Args:
            file_names (list): list of file paths to import
            raster_type (str, optional): if 'OrthoRaster', create OrthoRaster instances
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)

        progress_bar = ProgressBar(self.image_window, title="Importing Images")
        progress_bar.show()
        progress_bar.start_progress(len(file_names))

        try:
            imported_paths = []
            progress_batch = 0
            PROGRESS_BATCH_SIZE = 50  # Update UI every 50 images instead of every image
            
            # Add images directly to the manager without emitting signals
            for file_name in file_names:
                try:
                    if file_name not in self.image_window.raster_manager.image_paths:
                        # Call the manager directly to add the raster silently,
                        # bypassing ImageWindow.add_image and its signal handlers.
                        if raster_type == 'OrthoRaster':
                            if self.image_window.raster_manager.add_ortho_raster(file_name, emit_signal=False):
                                imported_paths.append(file_name)
                        else:
                            if self.image_window.raster_manager.add_raster(file_name, emit_signal=False):
                                imported_paths.append(file_name)
                    else:
                        imported_paths.append(file_name)
                except Exception:
                    if not suppress_errors:
                        raise

                # Batch progress updates to reduce UI thread load
                progress_batch += 1
                if progress_batch >= PROGRESS_BATCH_SIZE:
                    for _ in range(progress_batch):
                        progress_bar.update_progress()
                    progress_batch = 0
            
            # Flush remaining progress
            for _ in range(progress_batch):
                progress_bar.update_progress()

            # After the silent loop, manually update the UI exactly once.
            self.image_window.update_search_bars()
            self.image_window.filter_images(use_threading=False)

        except Exception as e:
            if not suppress_errors:
                self._show_error_message(str(e))
        finally:
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