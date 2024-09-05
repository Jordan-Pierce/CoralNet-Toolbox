import os
from functools import lru_cache

import rasterio

from toolbox.QtProgressBar import ProgressBar

from PyQt5.QtWidgets import (QSizePolicy, QMessageBox, QCheckBox, QWidget, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout,
                             QTableWidget, QTableWidgetItem, QFileDialog, QApplication)

from PyQt5.QtGui import QImage
from PyQt5.QtCore import Qt, pyqtSignal

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


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
        self.tableWidget.cellClicked.connect(self.load_image)
        self.tableWidget.keyPressEvent = self.tableWidget_keyPressEvent

        self.layout.addWidget(self.tableWidget)

        self.image_paths = []  # Store all image paths
        self.filtered_image_paths = []  # Store filtered image paths
        self.selected_image_path = None

        self.images = {}  # Dictionary to store image paths and their QImage representation
        self.rasterio_images = {}  # Dictionary to store image paths and their Rasterio representation
        self.image_cache = {}  # Cache for images

        self.show_confirmation_dialog = True

    def import_images(self):
        file_names, _ = QFileDialog.getOpenFileNames(self, "Open Image Files", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_names:
            progress_bar = ProgressBar(self, title="Importing Images")
            progress_bar.show()
            progress_bar.start_progress(len(file_names))

            for i, file_name in enumerate(file_names):
                if file_name not in set(self.image_paths):
                    self.add_image(file_name)
                    progress_bar.update_progress()
                    QApplication.processEvents()

            progress_bar.stop_progress()
            progress_bar.close()

            # Show the last image
            self.load_image_by_path(self.filtered_image_paths[-1])

            QMessageBox.information(self,
                                    "Image(s) Imported",
                                    "Image(s) have been successfully imported.")

    def add_image(self, image_path):
        if image_path not in self.image_paths:
            self.image_paths.append(image_path)
            self.filter_images()
            self.update_table_widget()
            # Update the image count label
            self.update_image_count_label()

    def load_image(self, row, column):
        # Get the image path associated with the selected row, load
        image_path = self.filtered_image_paths[row]
        self.load_image_by_path(image_path)

    def load_image_by_path(self, image_path, update=False):
        if image_path not in self.image_paths:
            return

        if image_path == self.selected_image_path and update is False:
            return

        # Load the QImage
        image = QImage(image_path)
        self.images[image_path] = image
        # Load the Rasterio
        rasterio_image = self.rasterio_open(image_path)
        self.rasterio_images[image_path] = rasterio_image

        # Update the selected image
        self.selected_image_path = image_path
        self.update_table_selection()

        # Pass to the AnnotationWindow to be displayed / selected
        self.annotation_window.set_image(image_path)
        self.imageSelected.emit(image_path)

        # Update the current image index label
        self.update_current_image_index_label()

    @lru_cache(maxsize=32)
    def rasterio_open(self, image_path):
        # Open the image with Rasterio
        self.src = rasterio.open(image_path)
        return self.src

    def rasterio_close(self, image_path):
        # Close the image with Rasterio
        self.rasterio_images[image_path] = None

    def delete_image(self, image_path):
        if image_path not in self.image_paths:
            return

        # Remove the image from the lists and dictionaries
        self.image_paths.remove(image_path)
        if image_path in self.filtered_image_paths:
            self.filtered_image_paths.remove(image_path)
        if image_path in self.images:
            del self.images[image_path]
        if image_path in self.rasterio_images:
            self.rasterio_close(image_path)
            del self.rasterio_images[image_path]

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
        if self.selected_image_path:
            index = self.filtered_image_paths.index(self.selected_image_path) + 1
            self.current_image_index_label.setText(f"Current Image: {index}")
        else:
            self.current_image_index_label.setText("Current Image: None")

    def update_table_widget(self):
        self.tableWidget.setRowCount(0)  # Clear the table
        for path in self.filtered_image_paths:
            row_position = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row_position)
            self.tableWidget.setItem(row_position, 0, QTableWidgetItem(os.path.basename(path)))
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

    def filter_images(self):
        search_text = self.search_bar.text().lower()
        has_annotations = self.has_annotations_checkbox.isChecked()
        needs_review = self.needs_review_checkbox.isChecked()
        no_annotations = self.no_annotations_checkbox.isChecked()

        self.filtered_image_paths = []

        for path in self.image_paths:
            filename = os.path.basename(path).lower()
            annotations = self.annotation_window.get_image_annotations(path)
            review_annotations = self.annotation_window.get_image_review_annotations(path)

            if search_text and search_text not in filename:
                continue

            if has_annotations and not annotations:
                continue
            if needs_review and not review_annotations:
                continue
            if no_annotations and annotations:
                continue

            self.filtered_image_paths.append(path)

        self.update_table_widget()

        # Load the first filtered image if available, otherwise clear the scene
        if self.filtered_image_paths:
            self.load_image_by_path(self.filtered_image_paths[0])
        else:
            self.selected_image_path = None
            self.annotation_window.clear_scene()

        self.update_current_image_index_label()
        self.update_image_count_label()

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