import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import rasterio
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSize, QThread, QEventLoop
from PyQt5.QtGui import QImage, QImageReader
from PyQt5.QtWidgets import (QSizePolicy, QMessageBox, QCheckBox, QWidget, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout,
                             QTableWidget, QTableWidgetItem, QFileDialog, QApplication, QMenu)

from toolbox.QtProgressBar import ProgressBar

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class LoadFullResolutionImageWorker(QThread):
    imageLoaded = pyqtSignal(QImage)
    cancelled = pyqtSignal()

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self._is_cancelled = False
        self.full_resolution_image = None

    def run(self):
        # Load the QImage
        self.full_resolution_image = QImage(self.image_path)
        # Emit the signal with the loaded image
        if not self._is_cancelled:
            self.imageLoaded.emit(self.full_resolution_image)
        else:
            self.cancelled.emit()

    def cancel(self):
        self._is_cancelled = True


class ImageWindow(QWidget):
    imageSelected = pyqtSignal(str)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Create a horizontal layout for the checkboxes
        self.checkbox_layout = QHBoxLayout()
        self.layout.addLayout(self.checkbox_layout)

        # Add checkboxes for filtering images based on annotations
        self.has_annotations_checkbox = QCheckBox("Has Annotations", self)
        self.has_annotations_checkbox.stateChanged.connect(self.update_has_annotations_checkbox)
        self.has_annotations_checkbox.stateChanged.connect(self.filter_images)
        self.checkbox_layout.addWidget(self.has_annotations_checkbox)

        self.needs_review_checkbox = QCheckBox("Needs Review", self)
        self.needs_review_checkbox.stateChanged.connect(self.update_needs_review_checkbox)
        self.needs_review_checkbox.stateChanged.connect(self.filter_images)
        self.checkbox_layout.addWidget(self.needs_review_checkbox)

        self.no_annotations_checkbox = QCheckBox("No Annotations", self)
        self.no_annotations_checkbox.stateChanged.connect(self.update_no_annotations_checkbox)
        self.no_annotations_checkbox.stateChanged.connect(self.filter_images)
        self.checkbox_layout.addWidget(self.no_annotations_checkbox)

        # Create a horizontal layout for the search bar
        self.search_layout = QHBoxLayout()
        self.layout.addLayout(self.search_layout)

        # Add a search bar
        self.search_bar = QLineEdit(self)
        self.search_bar.setPlaceholderText("Search images...")
        self.search_bar.textChanged.connect(self.filter_images)
        self.search_layout.addWidget(self.search_bar)

        # Create a horizontal layout for the labels
        self.info_layout = QHBoxLayout()
        self.layout.addLayout(self.info_layout)

        # Add a label to display the index of the currently selected image
        self.current_image_index_label = QLabel("Current Image: None", self)
        self.current_image_index_label.setAlignment(Qt.AlignCenter)
        self.current_image_index_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # Set the desired height (to align with AnnotationWindow)
        self.current_image_index_label.setFixedHeight(24)
        self.info_layout.addWidget(self.current_image_index_label)

        # Add a label to display the total number of images
        self.image_count_label = QLabel("Total Images: 0", self)
        self.image_count_label.setAlignment(Qt.AlignCenter)
        self.image_count_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # Set the desired height (to align with AnnotationWindow)
        self.image_count_label.setFixedHeight(24)
        self.info_layout.addWidget(self.image_count_label)

        self.tableWidget = QTableWidget(self)
        self.tableWidget.setColumnCount(1)
        self.tableWidget.setHorizontalHeaderLabels(["Image Name"])
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.setSelectionBehavior(QTableWidget.SelectRows)
        self.tableWidget.setSelectionMode(QTableWidget.SingleSelection)
        self.tableWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tableWidget.customContextMenuRequested.connect(self.show_context_menu)
        self.tableWidget.cellClicked.connect(self.load_image)
        self.tableWidget.keyPressEvent = self.tableWidget_keyPressEvent
        self.layout.addWidget(self.tableWidget)

        self.image_paths = []  # Store all image paths
        self.image_dict = {}  # Dictionary to store all image information
        self.filtered_image_paths = []  # List to store filtered image paths
        self.selected_image_path = None

        self.images = {}  # Dictionary to store image paths and their QImage representation
        self.rasterio_images = {}  # Dictionary to store image paths and their Rasterio representation
        self.image_cache = {}  # Cache for images

        self.show_confirmation_dialog = True

        self.search_timer = QTimer(self)
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self.filter_images)
        self.search_bar.textChanged.connect(self.debounce_search)

    def show_context_menu(self, position):
        context_menu = QMenu(self)
        delete_annotations_action = context_menu.addAction("Delete Annotations")
        delete_annotations_action.triggered.connect(self.delete_annotations)
        context_menu.exec_(self.tableWidget.viewport().mapToGlobal(position))

    def delete_annotations(self):
        reply = QMessageBox.question(self,
                                     "Confirm Delete",
                                     "Are you sure you want to delete annotations for this image?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            # Proceed with deleting annotations
            self.annotation_window.delete_image_annotations(self.selected_image_path)

    def import_images(self):
        file_names, _ = QFileDialog.getOpenFileNames(self,
                                                     "Open Image Files",
                                                     "",
                                                     "Image Files (*.png *.jpg *.jpeg *.tif* *.bmp)")
        if file_names:
            # Set the cursor to waiting (busy) cursor
            QApplication.setOverrideCursor(Qt.WaitCursor)

            progress_bar = ProgressBar(self, title="Importing Images")
            progress_bar.show()
            progress_bar.start_progress(len(file_names))
            progress_bar.set_value(1)

            for i, file_name in enumerate(file_names):
                if file_name not in set(self.image_paths):
                    self.add_image(file_name)
                progress_bar.update_progress()
                QApplication.processEvents()

            progress_bar.stop_progress()
            progress_bar.close()

            # Update filtered images
            self.filter_images()
            # Show the last image
            self.load_image_by_path(self.image_paths[-1])

            # Restore the cursor to the default cursor
            QApplication.restoreOverrideCursor()

            QMessageBox.information(self,
                                    "Image(s) Imported",
                                    "Image(s) have been successfully imported.")

    def add_image(self, image_path):
        if image_path not in self.image_paths:
            self.image_paths.append(image_path)
            filename = os.path.basename(image_path)
            self.image_dict[image_path] = {
                'filename': filename,
                'has_annotations': False,
                'needs_review': False
            }
            self.update_table_widget()
            self.update_image_count_label()

    def update_image_annotations(self, image_path):
        if image_path in self.image_dict:
            annotations = self.annotation_window.get_image_annotations(image_path)
            review_annotations = self.annotation_window.get_image_review_annotations(image_path)
            self.image_dict[image_path]['has_annotations'] = bool(annotations)
            self.image_dict[image_path]['needs_review'] = bool(review_annotations)

    def load_image(self, row, column):
        # Get the image path associated with the selected row, load
        image_path = self.filtered_image_paths[row]
        self.load_image_by_path(image_path)

    def load_image_by_path(self, image_path, update=False):
        if image_path not in self.image_paths:
            return

        if image_path == self.selected_image_path and update is False:
            return

        # Cancel the previous worker thread if it is still running
        if hasattr(self, 'full_res_worker') and self.full_res_worker.isRunning():
            # Restore the cursor to the default cursor
            QApplication.restoreOverrideCursor()
            self.full_res_worker.cancel()

        # Update the selected image path
        self.selected_image_path = image_path
        self.imageSelected.emit(image_path)
        # Update the table, current image index label
        self.update_table_selection()
        self.update_current_image_index_label()

        # Load a scaled-down version of the image using QImageReader
        def load_scaled_image():
            reader = QImageReader(image_path)
            original_size = reader.size()  # Get the original size of the image
            scaled_width = original_size.width() // 100
            scaled_height = original_size.height() // 100
            scaled_size = original_size.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio)
            reader.setScaledSize(scaled_size)  # Set the desired scaled size
            scaled_image = reader.read()
            return scaled_image

        # Set the cursor to waiting (busy) cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Load the scaled-down image
        scaled_image = load_scaled_image()
        # Display lower resolution image while threading loads full resolution image
        self.annotation_window.display_image_item(scaled_image)

        # Create and start the worker thread to load the full-resolution image
        self.full_res_worker = LoadFullResolutionImageWorker(image_path)
        self.full_res_worker.imageLoaded.connect(self.on_full_resolution_image_loaded)
        self.full_res_worker.start()

        # Use QTimer to periodically check if the full-resolution image is loaded
        self.full_res_timer = QTimer()
        self.full_res_timer.timeout.connect(self.check_full_res_image_loaded)
        self.full_res_timer.start(1)  # Check every 1 milliseconds

        # # Restore the cursor to the default cursor
        # QApplication.restoreOverrideCursor()

    def check_full_res_image_loaded(self):
        if self.full_res_worker.isFinished():
            self.full_res_timer.stop()
            # Connect the signal to the on_full_resolution_image_loaded method
            self.full_res_worker.imageLoaded.connect(self.on_full_resolution_image_loaded)
            # Emit the signal to trigger the on_full_resolution_image_loaded method
            self.full_res_worker.imageLoaded.emit(self.full_res_worker.full_resolution_image)

    def on_full_resolution_image_loaded(self, full_resolution_image):
        # Access the selected_image
        image_path = self.selected_image_path

        # Load the Rasterio
        self.rasterio_images[image_path] = self.rasterio_open(image_path)

        # Update the display with the full-resolution image
        self.images[image_path] = full_resolution_image
        self.annotation_window.set_image(image_path)

        # Restore the cursor to the default cursor
        QApplication.restoreOverrideCursor()

    @lru_cache(maxsize=32)
    def rasterio_open(self, image_path):
        # Open the image with Rasterio
        self.src = rasterio.open(image_path)
        return self.src

    def rasterio_close(self, image_path):
        # Close the image with Rasterio
        self.rasterio_images[image_path] = None

    def delete_image(self, image_path):
        if image_path in self.image_paths:
            self.image_paths.remove(image_path)
            del self.image_dict[image_path]
            if image_path in self.filtered_image_paths:
                self.filtered_image_paths.remove(image_path)

            # Remove the image's annotations
            self.annotation_window.delete_image(image_path)

            # Update the table widget
            self.update_table_widget()

            # Update the image count label
            self.update_image_count_label()

            # Select a new image if available
            if self.filtered_image_paths:
                new_image_path = self.filtered_image_paths[0]
                self.load_image_by_path(new_image_path)
            else:
                self.selected_image_path = None
                self.annotation_window.clear_scene()

            # Update the current image index label
            self.update_current_image_index_label()

    def delete_selected_image(self):
        if self.selected_image_path:
            if self._confirm_delete() == QMessageBox.Yes:
                self.delete_image(self.selected_image_path)

    def _confirm_delete(self):
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setWindowTitle("Confirm Delete")
        msg_box.setText("Are you sure you want to delete this image?\n"
                        "This will delete all associated annotations.")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        checkbox = QCheckBox("Do not show this message again")
        msg_box.setCheckBox(checkbox)

        result = msg_box.exec_()

        if checkbox.isChecked():
            self.show_confirmation_dialog = False

        return result

    def update_image_count_label(self):
        total_images = len(self.filtered_image_paths)
        self.image_count_label.setText(f"Total Images: {total_images}")

    def update_current_image_index_label(self):
        if self.selected_image_path and self.selected_image_path in self.filtered_image_paths:
            index = self.filtered_image_paths.index(self.selected_image_path) + 1
            self.current_image_index_label.setText(f"Current Image: {index}")
        else:
            self.current_image_index_label.setText("Current Image: None")

    def update_table_widget(self):
        self.tableWidget.setRowCount(0)  # Clear the table
        for path in self.filtered_image_paths:
            row_position = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row_position)
            self.tableWidget.setItem(row_position, 0, QTableWidgetItem(self.image_dict[path]['filename']))
        self.update_table_selection()

    def update_table_selection(self):
        if self.selected_image_path in self.filtered_image_paths:
            row = self.filtered_image_paths.index(self.selected_image_path)
            self.tableWidget.selectRow(row)
        else:
            self.tableWidget.clearSelection()

    def tableWidget_keyPressEvent(self, event):
        if event.key() == Qt.Key_Up or event.key() == Qt.Key_Down:
            # Ignore up and down arrow keys
            return
        else:
            # Call the base class method for other keys
            super(QTableWidget, self.tableWidget).keyPressEvent(event)

    def cycle_previous_image(self):
        if not self.filtered_image_paths:
            return

        current_index = self.filtered_image_paths.index(self.selected_image_path)
        new_index = (current_index - 1) % len(self.filtered_image_paths)
        self.load_image_by_path(self.filtered_image_paths[new_index])

    def cycle_next_image(self):
        if not self.filtered_image_paths:
            return

        current_index = self.filtered_image_paths.index(self.selected_image_path)
        new_index = (current_index + 1) % len(self.filtered_image_paths)
        self.load_image_by_path(self.filtered_image_paths[new_index])

    def debounce_search(self):
        self.search_timer.start(500)

    def filter_images(self):
        search_text = self.search_bar.text().lower()
        has_annotations = self.has_annotations_checkbox.isChecked()
        needs_review = self.needs_review_checkbox.isChecked()
        no_annotations = self.no_annotations_checkbox.isChecked()

        # Return early if none of the search bar or checkboxes are being used
        if not search_text and not has_annotations and not needs_review and not no_annotations:
            self.filtered_image_paths = self.image_paths.copy()
            self.update_table_widget()
            self.update_current_image_index_label()
            self.update_image_count_label()
            return

        self.filtered_image_paths = []

        # Use a ThreadPoolExecutor to filter images in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            for path in self.image_paths:
                future = executor.submit(
                    self.filter_image,
                    path,
                    search_text,
                    has_annotations,
                    needs_review,
                    no_annotations
                )
                futures.append(future)

            for future in as_completed(futures):
                if future.result():
                    self.filtered_image_paths.append(future.result())

        # Sort the filtered image paths
        self.filtered_image_paths.sort()

        self.update_table_widget()

        # Load the first filtered image if available, otherwise clear the scene
        if self.filtered_image_paths:
            self.load_image_by_path(self.filtered_image_paths[0])
        else:
            self.selected_image_path = None
            self.annotation_window.clear_scene()

        self.update_current_image_index_label()
        self.update_image_count_label()

    def filter_image(self, path, search_text, has_annotations, needs_review, no_annotations):
        filename = os.path.basename(path).lower()
        annotations = self.annotation_window.get_image_annotations(path)
        review_annotations = self.annotation_window.get_image_review_annotations(path)

        if search_text and search_text not in filename:
            return None

        if has_annotations and not annotations:
            return None
        if needs_review and not review_annotations:
            return None
        if no_annotations and annotations:
            return None

        return path

    def load_first_filtered_image(self):
        if self.filtered_image_paths:
            self.annotation_window.clear_scene()
            self.load_image_by_path(self.filtered_image_paths[0])

    def update_has_annotations_checkbox(self):
        if self.has_annotations_checkbox.isChecked():
            self.has_annotations_checkbox.setChecked(True)
            self.needs_review_checkbox.setChecked(False)
            self.no_annotations_checkbox.setChecked(False)

        if not self.has_annotations_checkbox.isChecked():
            self.has_annotations_checkbox.setChecked(False)

    def update_needs_review_checkbox(self):
        if self.needs_review_checkbox.isChecked():
            self.needs_review_checkbox.setChecked(True)
            self.has_annotations_checkbox.setChecked(False)
            self.no_annotations_checkbox.setChecked(False)

        if not self.needs_review_checkbox.isChecked():
            self.needs_review_checkbox.setChecked(False)

    def update_no_annotations_checkbox(self):
        if self.no_annotations_checkbox.isChecked():
            self.no_annotations_checkbox.setChecked(True)
            self.has_annotations_checkbox.setChecked(False)
            self.needs_review_checkbox.setChecked(False)

        if not self.no_annotations_checkbox.isChecked():
            self.no_annotations_checkbox.setChecked(False)