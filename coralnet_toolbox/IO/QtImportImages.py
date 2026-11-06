import os


from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox)

from coralnet_toolbox.QtProgressBar import ProgressBar


SUPPORTED_IMAGE_EXTENSIONS = {'.bmp', '.jfif', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}


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
                                                     "Image Files (*.png *.jpg *.jpeg *.jfif *.tif* *.bmp *.webp)")
        file_names = self._check_merged_work_areas(file_names)
        if file_names:
            self._process_image_files(file_names)

    def _check_merged_work_areas(self, file_names):
        """Ask before importing images that were already merged back into the raster they came from.

        Imported by hand they are ordinary images: they cannot be merged back,
        and Active Learning and Export would see the same pixels twice. The way
        back in is to extract the raster again, which reuses the files.

        Returns:
            list: the files to import (all, or without those images), or [] if canceled.
        """
        if not file_names:
            return file_names
        from coralnet_toolbox.Rasters import extracted_images
        try:
            merged = extracted_images.merged_work_area_files(file_names)
        except Exception as e:
            print(f"[ImportImages] Could not check for merged work area images: {e}")
            return file_names
        if not merged:
            return file_names

        flagged = {path for group in merged.values() for path in group['paths']}
        names = ", ".join(os.path.basename(parent or "") for parent in merged)
        dates = ", ".join(sorted({group['merged'][:10] for group in merged.values()}))
        raster_manager = self.image_window.raster_manager
        parents_here = [parent for parent in merged if parent and raster_manager.has_image_path(parent)]

        box = QMessageBox(self.image_window)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Already Merged Back")
        box.setText(f"{len(flagged)} of these images are Work Area Images of {names}, already merged back on "
                    f"{dates}.")
        box.setInformativeText(
            "Imported this way they are ordinary images: they cannot be merged back, and Active Learning and "
            "Export would see the same pixels twice.\n\n"
            "To work on these areas again, extract them again from the Parent Image: the saved files are reused "
            "and its annotations as they are now are copied onto them.")
        extract_button = box.addButton("Extract Again", QMessageBox.AcceptRole) if parents_here else None
        import_button = box.addButton("Import as Ordinary Images", QMessageBox.DestructiveRole)
        cancel_button = box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(cancel_button)
        box.exec_()

        clicked = box.clickedButton()
        if clicked is import_button:
            return file_names
        others = [path for path in file_names if path not in flagged]
        if extract_button is not None and clicked is extract_button:
            # The other files still come in; the flagged ones come back through extraction
            if others:
                self._process_image_files(others)
            self.image_window._open_extract_work_areas_dialog(parents_here)
            return []
        return []

    def import_orthomosaics(self):
        """Import files as OrthoRaster instances (orthomosaics)."""
        self.main_window.untoggle_all_tools()

        file_names, _ = QFileDialog.getOpenFileNames(self.image_window,
                                                     "Open Orthomosaic Files",
                                                     "",
                                                     "Image Files (*.png *.jpg *.jpeg *.jfif *.tif* *.bmp *.webp)")
        file_names = self._check_merged_work_areas(file_names)
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
            file_names = self._check_merged_work_areas(file_names)
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

        # (path, reason) for every file that did not make it in. The add methods
        # return False and print; collecting the reasons here is what stops a
        # part-way failure -- running out of file descriptors, say -- from
        # looking like a clean import that was simply short. Declared outside the
        # try so the report below is reachable however the loop ends.
        failures = []

        try:
            imported_paths = []
            progress_batch = 0
            PROGRESS_BATCH_SIZE = 50  # Update UI every 50 images instead of every image

            # Add images directly to the manager without emitting signals
            for file_name in file_names:
                try:
                    if not self.image_window.raster_manager.has_image_path(file_name):
                        # Call the manager directly to add the raster silently,
                        # bypassing ImageWindow.add_image and its signal handlers.
                        if raster_type == 'OrthoRaster':
                            added = self.image_window.raster_manager.add_ortho_raster(file_name,
                                                                                      emit_signal=False)
                        else:
                            added = self.image_window.raster_manager.add_raster(file_name,
                                                                               emit_signal=False)
                        if added:
                            imported_paths.append(file_name)
                        else:
                            reason = self.image_window.raster_manager.last_error
                            failures.append((file_name, reason or "Unknown error"))
                    else:
                        imported_paths.append(file_name)
                except Exception as e:
                    if not suppress_errors:
                        raise
                    failures.append((file_name, str(e)))

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

        # Reported after the progress bar is down, and reported even when
        # `suppress_errors` is set: that flag exists to keep a drag-and-drop
        # from raising, not to hide files the user asked for and did not get.
        if failures:
            self._show_import_failures(failures)

    def _show_import_failures(self, failures):
        """Report the files that did not import, once, with the reason for each.

        Args:
            failures (list): (file path, reason) pairs.
        """
        # Grouped by reason rather than listed per file. A cause that is really
        # about the process and not the file -- descriptor exhaustion, a missing
        # drive -- produces an identical message for every file after the first,
        # and thousands of copies of it would bury anything else.
        by_reason = {}
        for file_name, reason in failures:
            by_reason.setdefault(reason, []).append(file_name)

        blocks = []
        for reason, names in list(by_reason.items())[:3]:
            if len(names) == 1:
                blocks.append(f"{os.path.basename(names[0])}:\n{reason}")
            else:
                blocks.append(f"{len(names)} files, including {os.path.basename(names[0])}:\n{reason}")

        if len(by_reason) > 3:
            blocks.append(f"...and {len(by_reason) - 3} further distinct error(s); see the console.")

        QMessageBox.warning(self.image_window,
                            "Error Importing Image(s)",
                            f"{len(failures)} file(s) could not be imported.\n\n" + "\n\n".join(blocks))
    
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