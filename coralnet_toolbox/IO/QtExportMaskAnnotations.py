import warnings

import os
import ujson as json

import cv2
import numpy as np
import rasterio
from PIL import Image

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
                             QCheckBox, QComboBox, QLineEdit, QPushButton, QFileDialog,
                             QApplication, QMessageBox, QLabel, QTableWidgetItem,
                             QButtonGroup, QWidget, QTableWidget, QHeaderView,
                             QAbstractItemView, QSpinBox)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ExportMaskAnnotations(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Export Masks")
        self.resize(1000, 750)

        self.selected_labels = []
        self.annotation_types = []
        self.class_mapping = {}

        # Main layout for the dialog
        self.main_layout = QVBoxLayout(self)

        # Top section - Add information, output, and mask format sections
        top_section = QVBoxLayout()
        self.setup_info_layout(parent_layout=top_section)
        self.setup_output_layout(parent_layout=top_section)
        self.setup_mask_format_layout(parent_layout=top_section)
        self.main_layout.addLayout(top_section)

        # Middle section - Two columns layout for annotations and labels
        columns_layout = QHBoxLayout()
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()

        # Add bottom-left widgets - annotations and apply to
        self.setup_annotation_layout(parent_layout=left_col)
        self.setup_image_selection_layout(parent_layout=left_col)

        # Add bottom-right widgets - label selection
        self.setup_label_layout(parent_layout=right_col)

        # Add columns to the middle section layout
        columns_layout.addLayout(left_col, 1)
        columns_layout.addLayout(right_col, 2)
        self.main_layout.addLayout(columns_layout)

        # Buttons at the bottom
        self.setup_buttons_layout(parent_layout=self.main_layout)

    def showEvent(self, event):
        """Handle the show event"""
        super().showEvent(event)
        # Update the labels in the label selection list
        self.update_label_selection_list()

    def setup_info_layout(self, parent_layout=None):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with more comprehensive explanatory text
        info_text = (
            "<b>Export Annotations to Masks</b><br><br>"
            "<b>Semantic Segmentation:</b> Create masks where each class has a different value "
            "(0 is typically reserved for background). These masks are used for training "
            "segmentation models or analyzing area coverage.<br><br>"
            "<b>Structure from Motion (SfM):</b> Create binary masks where 0 is background "
            "(areas to be masked out during reconstruction) and 255 is for objects of interest "
            "(areas to retain in reconstruction). These masks can be used with SfM software "
            "like Metashape or COLMAP."
        )
        
        info_label = QLabel(info_text)
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        if parent_layout is not None:
            parent_layout.addWidget(group_box)
        else:
            self.layout.addWidget(group_box)

    def setup_output_layout(self, parent_layout=None):
        """Setup the output directory and file format layout."""
        groupbox = QGroupBox("Output Directory and File Format")
        layout = QFormLayout()

        # Output directory selection
        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_button = QPushButton("Browse...")
        self.output_dir_button.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_dir_button)
        layout.addRow("Output Directory:", output_dir_layout)

        # Output folder name
        self.output_name_edit = QLineEdit("")
        layout.addRow("Folder Name:", self.output_name_edit)

        groupbox.setLayout(layout)
        if parent_layout is not None:
            parent_layout.addWidget(groupbox)
        else:
            self.layout.addWidget(groupbox)

    def setup_image_selection_layout(self, parent_layout=None):
        """Setup the image selection layout."""
        group_box = QGroupBox("Apply To")
        layout = QVBoxLayout()

        self.apply_filtered_checkbox = QCheckBox("▼ Apply to filtered images")
        self.apply_prev_checkbox = QCheckBox("↑ Apply to previous images")
        self.apply_next_checkbox = QCheckBox("↓ Apply to next images")
        self.apply_all_checkbox = QCheckBox("↕ Apply to all images")

        layout.addWidget(self.apply_filtered_checkbox)
        layout.addWidget(self.apply_prev_checkbox)
        layout.addWidget(self.apply_next_checkbox)
        layout.addWidget(self.apply_all_checkbox)

        self.apply_group = QButtonGroup(self)
        self.apply_group.addButton(self.apply_filtered_checkbox)
        self.apply_group.addButton(self.apply_prev_checkbox)
        self.apply_group.addButton(self.apply_next_checkbox)
        self.apply_group.addButton(self.apply_all_checkbox)
        self.apply_group.setExclusive(True)

        group_box.setLayout(layout)
        if parent_layout is not None:
            parent_layout.addWidget(group_box)
        else:
            self.layout.addWidget(group_box)

    def setup_annotation_layout(self, parent_layout=None):
        """Setup the annotation types layout."""
        groupbox = QGroupBox("Annotations to Include")
        layout = QVBoxLayout()

        # Annotation types checkboxes
        self.patch_checkbox = QCheckBox("Patch Annotations")
        self.patch_checkbox.setChecked(True)
        self.rectangle_checkbox = QCheckBox("Rectangle Annotations")
        self.rectangle_checkbox.setChecked(True)
        self.polygon_checkbox = QCheckBox("Polygon Annotations")
        self.polygon_checkbox.setChecked(True)

        # Include negative samples
        self.include_negative_samples_checkbox = QCheckBox("Include negative samples")
        self.include_negative_samples_checkbox.setChecked(True)

        layout.addWidget(self.patch_checkbox)
        layout.addWidget(self.rectangle_checkbox)
        layout.addWidget(self.polygon_checkbox)
        layout.addWidget(self.include_negative_samples_checkbox)

        groupbox.setLayout(layout)
        if parent_layout is not None:
            parent_layout.addWidget(groupbox)
        else:
            self.layout.addWidget(groupbox)

    def setup_label_layout(self, parent_layout=None):
        """Setup the label selection and reordering layout."""
        groupbox = QGroupBox("Labels to Include / Rasterization Order")
        layout = QVBoxLayout()
        
        # Create a standard QTableWidget
        self.label_table = QTableWidget()
        self.label_table.setColumnCount(3)
        self.label_table.setHorizontalHeaderLabels(["Include", "Label Name", "Mask Value"])
        self.label_table.setMinimumHeight(200)
        self.label_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.label_table.setSelectionMode(QAbstractItemView.SingleSelection)
        
        # Configure table properties for a better user experience
        header = self.label_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        
        # Add the table to the layout
        layout.addWidget(self.label_table)
        
        # Create horizontal layout for buttons below the table (instead of on the right)
        button_layout = QHBoxLayout()
        
        # Add stretch before buttons to push them toward the center
        button_layout.addStretch(1)

        self.move_up_button = QPushButton("▲ Move Up")
        self.move_down_button = QPushButton("▼ Move Down")

        self.move_up_button.clicked.connect(self.move_row_up)
        self.move_down_button.clicked.connect(self.move_row_down)

        button_layout.addWidget(self.move_up_button)
        button_layout.addWidget(self.move_down_button)

        # Add another stretch after buttons with equal weight to center them
        button_layout.addStretch(1)
        
        # Add the button layout to the main layout
        layout.addLayout(button_layout)
        
        # Add a note about the rasterization order with more detailed explanation
        order_note = QLabel(
            "<b>Layer Order is Important:</b><br>"
            "Use the up/down buttons to change the rasterization order. "
            "Labels lower in the list will be drawn on top of labels higher in the list.<br>"
            "• For overlapping annotations, only the topmost class will appear in that area<br>"
            "• Example: If coral growing on rock is drawn after the rock layer, the coral will be visible<br>"
            "• For semantic segmentation training, proper ordering ensures accurate class boundaries<br>"
            "• For SfM masks, important objects should be placed lower in the list to ensure they're included"
        )
        order_note.setStyleSheet("color: #666;")
        order_note.setWordWrap(True)
        layout.addWidget(order_note)
        
        groupbox.setLayout(layout)
        if parent_layout is not None:
            parent_layout.addWidget(groupbox)
        else:
            self.layout.addWidget(groupbox)

    def move_row_up(self):
        """Move the selected row up in the table."""
        current_row = self.label_table.currentRow()
        if current_row <= 0:  # Can't move up if it's the first row
            return
            
        # Remember selection
        self.swap_rows(current_row, current_row - 1)
        self.label_table.selectRow(current_row - 1)

    def move_row_down(self):
        """Move the selected row down in the table."""
        current_row = self.label_table.currentRow()
        if current_row >= self.label_table.rowCount() - 1 or current_row < 0:  # Can't move down if it's the last row
            return
            
        # Remember selection
        self.swap_rows(current_row, current_row + 1)
        self.label_table.selectRow(current_row + 1)

    def swap_rows(self, row1, row2):
        """Swap two rows in the table."""
        # Store data from row1
        row1_checkbox = self.label_table.cellWidget(row1, 0).findChild(QCheckBox)
        row1_checked = row1_checkbox.isChecked() if row1_checkbox else True
        
        row1_item = self.label_table.item(row1, 1)
        row1_text = row1_item.text()
        row1_data = row1_item.data(Qt.UserRole)
        
        row1_spinbox = self.label_table.cellWidget(row1, 2)
        row1_value = row1_spinbox.value() if row1_spinbox else 0
        
        # Store data from row2
        row2_checkbox = self.label_table.cellWidget(row2, 0).findChild(QCheckBox)
        row2_checked = row2_checkbox.isChecked() if row2_checkbox else True
        
        row2_item = self.label_table.item(row2, 1)
        row2_text = row2_item.text()
        row2_data = row2_item.data(Qt.UserRole)
        
        row2_spinbox = self.label_table.cellWidget(row2, 2)
        row2_value = row2_spinbox.value() if row2_spinbox else 0
        
        # Update row1 with row2 data
        row1_checkbox.setChecked(row2_checked)
        row1_item.setText(row2_text)
        row1_item.setData(Qt.UserRole, row2_data)
        row1_spinbox.setValue(row2_value)
        
        # Update row2 with row1 data
        row2_checkbox.setChecked(row1_checked)
        row2_item.setText(row1_text)
        row2_item.setData(Qt.UserRole, row1_data)
        row2_spinbox.setValue(row1_value)

    def setup_mask_format_layout(self, parent_layout=None):
        """Setup the mask format layout."""
        groupbox = QGroupBox("Mask Format")
        layout = QFormLayout()

        # File format combo box
        self.file_format_combo = QComboBox()
        self.file_format_combo.addItems([".png", ".bmp", ".tif"])
        self.file_format_combo.setEditable(True)
        layout.addRow("File Format:", self.file_format_combo)

        # Add option to preserve georeferencing
        self.preserve_georef_checkbox = QCheckBox("Preserve georeferencing (if available)")
        self.preserve_georef_checkbox.setChecked(True)
        layout.addRow("", self.preserve_georef_checkbox)

        # Connect the checkbox to update based on file format
        self.file_format_combo.currentTextChanged.connect(self.update_georef_availability)

        # Add a note about georeferencing
        self.georef_note = QLabel("Note: Georeferencing can only be preserved with TIF format")
        self.georef_note.setStyleSheet("color: #666; font-style: italic;")
        layout.addRow("", self.georef_note)

        groupbox.setLayout(layout)
        if parent_layout is not None:
            parent_layout.addWidget(groupbox)
        else:
            self.layout.addWidget(groupbox)

        # Initial update based on default format
        self.update_georef_availability()

    def setup_buttons_layout(self, parent_layout=None):
        """Setup the buttons layout at the bottom-right."""
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)  # Push buttons to the right

        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.export_masks)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.cancel_button)

        if parent_layout is not None:
            parent_layout.addLayout(button_layout)
        else:
            self.layout.addLayout(button_layout)

    def browse_output_dir(self):
        """Open a file dialog to select the output directory."""
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", "", options=options
        )
        if directory:
            self.output_dir_edit.setText(directory)

    def update_label_selection_list(self):
        """Update the label selection table with labels from the label window."""
        # Block signals to prevent them from firing during the update, which is more efficient
        self.label_table.blockSignals(True)
        self.label_table.setRowCount(0)  # Clear the table completely
        
        # Add background label first (row 0)
        self.label_table.insertRow(0)
        
        # Create checkbox for background
        checkbox = QCheckBox()
        checkbox.setChecked(True)
        cell_widget = QWidget()
        layout = QHBoxLayout(cell_widget)
        layout.addWidget(checkbox)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label_table.setCellWidget(0, 0, cell_widget)
        
        # Create label item for background
        label_item = QTableWidgetItem("Background")
        label_item.setFlags(label_item.flags() & ~Qt.ItemIsEditable)
        label_item.setData(Qt.UserRole, "Background")
        self.label_table.setItem(0, 1, label_item)
        
        # Modified spinbox for background to allow editing and different values
        spinbox = QSpinBox()
        spinbox.setMinimum(0)
        spinbox.setMaximum(255)  # Allow any value up to 255
        spinbox.setValue(0)      # Default is still 0
        spinbox.setEnabled(True) # Make it editable
        self.label_table.setCellWidget(0, 2, spinbox)

        # Add actual labels starting from row 1
        for i, label in enumerate(self.label_window.labels):
            row = i + 1  # +1 to account for background row
            self.label_table.insertRow(row)

            # --- Column 0: Include CheckBox ---
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            # We use a container widget to center the checkbox in the cell
            cell_widget = QWidget()
            layout = QHBoxLayout(cell_widget)
            layout.addWidget(checkbox)
            layout.setAlignment(Qt.AlignCenter)
            layout.setContentsMargins(0, 0, 0, 0)
            self.label_table.setCellWidget(row, 0, cell_widget)

            # --- Column 1: Label Name ---
            label_item = QTableWidgetItem(label.short_label_code)
            # Make the label name read-only
            label_item.setFlags(label_item.flags() & ~Qt.ItemIsEditable)
            # Store only the label's short code as a string instead of the full object
            label_item.setData(Qt.UserRole, label.short_label_code)
            self.label_table.setItem(row, 1, label_item)

            # --- Column 2: Mask Value SpinBox ---
            spinbox = QSpinBox()
            spinbox.setMinimum(0)    # Start at 1 since 0 is background
            spinbox.setMaximum(255)  # The max value for an 8-bit mask (np.uint8)
            spinbox.setValue(i + 1)  # Value equals index + 1
            self.label_table.setCellWidget(row, 2, spinbox)

        # Select the first row
        if self.label_table.rowCount() > 0:
            self.label_table.selectRow(0)
            
        # Re-enable signals after the table is populated
        self.label_table.blockSignals(False)

    def update_georef_availability(self):
        """Update georeferencing checkbox availability based on file format"""
        current_format = self.file_format_combo.currentText().lower()
        is_tif = '.tif' in current_format

        self.preserve_georef_checkbox.setEnabled(is_tif)
        if not is_tif:
            self.preserve_georef_checkbox.setChecked(False)
            self.georef_note.setStyleSheet("color: red; font-style: italic;")
        else:
            self.georef_note.setStyleSheet("color: #666; font-style: italic;")

    def get_selected_image_paths(self):
        """
        Get the selected image paths based on the options.

        :return: List of selected image paths
        """
        # Current image path showing
        current_image_path = self.annotation_window.current_image_path
        if not current_image_path:
            return []

        # Determine which images to export annotations for
        if self.apply_filtered_checkbox.isChecked():
            return self.image_window.table_model.filtered_paths
        elif self.apply_prev_checkbox.isChecked():
            if current_image_path in self.image_window.table_model.filtered_paths:
                current_index = self.image_window.table_model.get_row_for_path(current_image_path)
                return self.image_window.table_model.filtered_paths[:current_index + 1]
            else:
                return [current_image_path]
        elif self.apply_next_checkbox.isChecked():
            if current_image_path in self.image_window.table_model.filtered_paths:
                current_index = self.image_window.table_model.get_row_for_path(current_image_path)
                return self.image_window.table_model.filtered_paths[current_index:]
            else:
                return [current_image_path]
        elif self.apply_all_checkbox.isChecked():
            return self.image_window.raster_manager.image_paths
        else:
            # Only apply to the current image
            return [current_image_path]

    def export_class_mapping(self, output_path):
        """Export the class mapping to a JSON file."""
        mapping_file = os.path.join(output_path, "class_mapping.json")

        with open(mapping_file, 'w') as f:
            json.dump(self.class_mapping, f, indent=4)

        if not os.path.exists(mapping_file):
            print(f"Warning: Failed to save class mapping to {mapping_file}")

    def export_masks(self):
        """Export masks based on the configuration in the UI."""
        if not self.output_dir_edit.text():
            QMessageBox.warning(self, "Missing Output Directory", "Please select an output directory.")
            return

        if not any([self.patch_checkbox.isChecked(), 
                    self.rectangle_checkbox.isChecked(), 
                    self.polygon_checkbox.isChecked()]):
            QMessageBox.warning(self, 
                                "No Annotation Type Selected", 
                                "Please select at least one annotation type.")
            return

        self.labels_to_render = []
        self.background_value = 0  # Default background value is 0
        used_mask_values = {}

        # Parse the table to separate the background value from drawable labels
        for i in range(self.label_table.rowCount()):
            checkbox = self.label_table.cellWidget(i, 0).findChild(QCheckBox)
            if not (checkbox and checkbox.isChecked()):
                continue

            label_item = self.label_table.item(i, 1)
            label_code = label_item.data(Qt.UserRole)
            spinbox = self.label_table.cellWidget(i, 2)
            mask_value = spinbox.value()

            # Track for duplicate value warnings later
            if mask_value not in used_mask_values:
                used_mask_values[mask_value] = []
            used_mask_values[mask_value].append(label_code)

            # Separate the special 'Background' case from real labels
            if label_code == "Background":
                self.background_value = mask_value
                continue  # Go to the next row

            # For all other rows, find the corresponding label object
            label = next((l for l in self.label_window.labels if l.short_label_code == label_code), None)

            if not label:
                # This check now correctly ignores the "Background" code
                QMessageBox.warning(self, "Label Not Found", f"Could not find a real label with code '{label_code}'.")
                return

            self.labels_to_render.append((label, mask_value))

        # Check if at least one actual label is selected for export
        if not self.labels_to_render:
            QMessageBox.warning(self, "No Labels Selected", "Please select at least one drawable label to include.")
            return

        # --- (Duplicate value warning logic remains the same) ---
        duplicate_values = {value: labels for value, labels in used_mask_values.items() if len(labels) > 1}
        if duplicate_values:
            warning_message = "The following mask values are used by multiple labels:\n\n"
            for value, labels in duplicate_values.items():
                warning_message += f"Value {value}: {', '.join(labels)}\n"
            warning_message += "\nThis may be intentional. Do you want to continue?"
            reply = QMessageBox.warning(self, 
                                        "Duplicate Mask Values", 
                                        warning_message, QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
        
        # Create the class mapping, correctly handling the background
        self.class_mapping = {}
        # Manually add the background entry if it was checked
        background_checkbox = self.label_table.cellWidget(0, 0).findChild(QCheckBox)
        if background_checkbox and background_checkbox.isChecked():
            self.class_mapping["Background"] = {"label": "Background", "index": self.background_value}
        
        # Add the rest of the labels from the render list
        for label, mask_value in self.labels_to_render:
            self.class_mapping[label.short_label_code] = {
                "label": label.to_dict(),
                "index": mask_value
            }
            
        output_dir = self.output_dir_edit.text()
        folder_name = self.output_name_edit.text().strip()
        file_format = self.file_format_combo.currentText()

        if not file_format.startswith('.'):
            file_format = '.' + file_format

        output_path = os.path.join(output_dir, folder_name)
        try:
            os.makedirs(output_path, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Error Creating Directory", f"Failed to create output directory: {str(e)}")
            return

        images = self.get_selected_image_paths()
        if not images:
            QMessageBox.warning(self, "No Images", "No images found for processing.")
            return

        self.annotation_types = []
        if self.patch_checkbox.isChecked():
            self.annotation_types.append(PatchAnnotation)
        if self.rectangle_checkbox.isChecked():
            self.annotation_types.append(RectangleAnnotation)
        if self.polygon_checkbox.isChecked():
            self.annotation_types.append(PolygonAnnotation)
            self.annotation_types.append(MultiPolygonAnnotation)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, "Exporting Masks")
        progress_bar.show()
        progress_bar.start_progress(len(images))

        try:
            for image_path in images:
                self.create_mask_for_image(image_path, output_path, file_format)
                progress_bar.update_progress()

            self.export_class_mapping(output_path)
            QMessageBox.information(self, "Export Complete", "Masks have been successfully exported.")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error Exporting Masks", f"An error occurred: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.finish_progress()
            progress_bar.stop_progress()
            progress_bar.close()

    def get_image_metadata(self, image_path, file_format):
        """Get the dimensions of the image, and check for georeferencing."""
        # Check if image has georeferencing that needs to be preserved
        transform = None
        crs = None
        has_georef = False
        height = None
        width = None

        # Get the raster from the raster manager
        raster = self.image_window.raster_manager.get_raster(image_path)

        # Only check for georeferencing if using TIF format and checkbox is checked
        can_preserve_georef = self.preserve_georef_checkbox.isChecked() and file_format.lower() == '.tif'

        if raster and raster.rasterio_src:
            # Get dimensions from the raster
            width = raster.width
            height = raster.height

            # Check for georeferencing if needed
            if can_preserve_georef and hasattr(raster.rasterio_src, 'transform'):
                transform = raster.rasterio_src.transform
                if transform and not transform.is_identity:
                    crs = raster.rasterio_src.crs
                    has_georef = True
        else:
            # Fallback to direct file access if raster is not available
            try:
                if can_preserve_georef:
                    with rasterio.open(image_path) as src:
                        if src.transform and not src.transform.is_identity:
                            transform = src.transform
                            crs = src.crs
                            has_georef = True
                        width, height = src.width, src.height
                else:
                    # Use PIL for non-georeferenced images
                    image = Image.open(image_path)
                    width, height = image.size
            except Exception as e:
                print(f"Error loading image {image_path}: {str(e)}")

        return height, width, has_georef, transform, crs

    def create_mask_for_image(self, image_path, output_path, file_format):
        """Create a mask for a single image, respecting render order."""
        height, width, has_georef, transform, crs = self.get_image_metadata(image_path, file_format)

        if not height or not width:
            print(f"Could not get dimensions for image: {image_path}")
            return

        # CHANGED: Initialize the mask with the user-defined background value
        mask = np.full((height, width), self.background_value, dtype=np.uint8)
        has_annotations_on_image = False

        # Iterate through the ordered list created in export_masks
        for label, mask_value in self.labels_to_render:

            # Get annotations for this specific label
            annotations = self.get_annotations_for_image(image_path, label)
            
            if annotations:
                has_annotations_on_image = True
                # Draw these annotations onto the mask with the specified value
                mask = self.draw_annotations_on_mask(mask, annotations, mask_value)

        # Skip saving if the image has no relevant annotations and we're not including negatives
        if not has_annotations_on_image and not self.include_negative_samples_checkbox.isChecked():
            return

        # Save the mask
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]
        mask_filename = f"{name_without_ext}{file_format}"
        mask_path = os.path.join(output_path, mask_filename)

        if has_georef and file_format.lower() == '.tif':
            with rasterio.open(
                mask_path, 'w', driver='GTiff', height=height, width=width,
                count=1, dtype=mask.dtype, crs=crs, transform=transform, compress='lzw',
            ) as dst:
                dst.write(mask, 1)
        else:
            cv2.imwrite(mask_path, mask)

        if not os.path.exists(mask_path):
            print(f"Warning: Failed to save mask to {mask_path}")

    def get_annotations_for_image(self, image_path, label):
        """Get annotations for a specific image AND a specific label."""
        annotations = []
        label_code_to_match = label.short_label_code

        for annotation in self.annotation_window.get_image_annotations(image_path):
            # Check that the annotation's label matches the one we're looking for
            if annotation.label.short_label_code != label_code_to_match:
                continue
            
            # Check that the annotation is of a type the user wants to include
            if not isinstance(annotation, tuple(self.annotation_types)):
                continue

            if isinstance(annotation, MultiPolygonAnnotation):
                for polygon in annotation.polygons:
                    annotations.append(polygon)
            else:
                annotations.append(annotation)
                
        return annotations

    def draw_annotations_on_mask(self, mask, annotations, mask_value):
        """Draw a list of annotations on the mask with a specific integer value."""
        for annotation in annotations:
            if isinstance(annotation, PatchAnnotation):
                cv2.rectangle(mask,
                              (int(annotation.center_xy.x() - annotation.annotation_size / 2),
                               int(annotation.center_xy.y() - annotation.annotation_size / 2)),
                              (int(annotation.center_xy.x() + annotation.annotation_size / 2),
                               int(annotation.center_xy.y() + annotation.annotation_size / 2)),
                              mask_value, -1)
            elif isinstance(annotation, RectangleAnnotation):
                cv2.rectangle(mask,
                              (int(annotation.top_left.x()), int(annotation.top_left.y())),
                              (int(annotation.bottom_right.x()), int(annotation.bottom_right.y())),
                              mask_value, -1)
            elif isinstance(annotation, PolygonAnnotation):
                points = np.array([[p.x(), p.y()] for p in annotation.points]).astype(np.int32)
                cv2.fillPoly(mask, [points], mask_value)

        return mask

    def closeEvent(self, event):
        """Handle the close event."""
        # Clean up any resources if needed
        super().closeEvent(event)