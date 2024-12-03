import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import json
import random

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QCheckBox,
                             QVBoxLayout, QLabel, QLineEdit, QDialog, QHBoxLayout,
                             QPushButton, QFormLayout, QDialogButtonBox, QDoubleSpinBox,
                             QGroupBox, QTableWidget, QTableWidgetItem)

from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Base(QDialog):
    def __init__(self, main_window, parent=None):
        """
        Initialize the ExportDatasetDialog class.

        Args:
            main_window: The main window object.
            parent: The parent widget.
        """
        super().__init__(parent)
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window
        self.image_window = main_window.image_window

        self.resize(1000, 600)
        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Export Dataset")

        self.selected_labels = []
        self.selected_annotations = []
        self.updating_summary_statistics = False
        
        self.output_dir = None
        self.dataset_name = None
        self.train_ratio = 0.7
        self.val_ratio = 0.2
        self.test_ratio = 0.1

        self.layout = QVBoxLayout(self)

        # Setup the layout
        self.setup_info_layout()
        self.setup_output_layout()
        self.setup_ratio_layout()
        self.setup_annotation_layout()
        self.setup_table_layout()
        self.setup_status_layout()
        self.setup_button_layout()

    def showEvent(self, event):
        """
        Handle the show event to update annotation type checkboxes, populate class filter list,
        and update summary statistics.

        Args:
            event: The show event.
        """
        super().showEvent(event)
        self.update_annotation_type_checkboxes()
        self.populate_class_filter_list()
        self.update_summary_statistics()
        
    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()
        
        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Export Patches, Rectangles, and Polygons to create a YOLO-formatted \
                             Classification, Detection or Segmentation dataset.")
        
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_layout(self):
        """
        Set up the layout for the ExportDatasetDialog.
        """
        # Ready Status
        self.ready_label = QLabel()
        self.layout.addWidget(self.ready_label)

        # Shuffle Button
        self.shuffle_button = QPushButton("Shuffle")
        self.shuffle_button.clicked.connect(self.update_summary_statistics)
        self.layout.addWidget(self.shuffle_button)

    def setup_output_layout(self):
        """Setup output directory layout."""
        group_box = QGroupBox("Output Parameters")
        layout = QFormLayout()

        # Dataset Name and Output Directory
        self.dataset_name_edit = QLineEdit()
        self.output_dir_edit = QLineEdit()
        self.output_dir_button = QPushButton("Browse...")
        self.output_dir_button.clicked.connect(self.browse_output_dir)

        layout.addRow("Dataset Name:", self.dataset_name_edit)
        layout.addRow("Output Directory:", self.output_dir_edit)
        layout.addRow(self.output_dir_button)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_ratio_layout(self):
        """Setup the train, validation, and test ratio layout."""
        group_box = QGroupBox("Split Ratios")
        layout = QHBoxLayout()

        # Split Ratios
        self.train_ratio_spinbox = QDoubleSpinBox()
        self.train_ratio_spinbox.setRange(0.0, 1.0)
        self.train_ratio_spinbox.setSingleStep(0.1)
        self.train_ratio_spinbox.setValue(0.7)

        self.val_ratio_spinbox = QDoubleSpinBox()
        self.val_ratio_spinbox.setRange(0.0, 1.0)
        self.val_ratio_spinbox.setSingleStep(0.1)
        self.val_ratio_spinbox.setValue(0.2)

        self.test_ratio_spinbox = QDoubleSpinBox()
        self.test_ratio_spinbox.setRange(0.0, 1.0)
        self.test_ratio_spinbox.setSingleStep(0.1)
        self.test_ratio_spinbox.setValue(0.1)

        layout.addWidget(QLabel("Train Ratio:"))
        layout.addWidget(self.train_ratio_spinbox)
        layout.addWidget(QLabel("Validation Ratio:"))
        layout.addWidget(self.val_ratio_spinbox)
        layout.addWidget(QLabel("Test Ratio:"))
        layout.addWidget(self.test_ratio_spinbox)

        self.train_ratio_spinbox.valueChanged.connect(self.update_summary_statistics)
        self.val_ratio_spinbox.valueChanged.connect(self.update_summary_statistics)
        self.test_ratio_spinbox.valueChanged.connect(self.update_summary_statistics)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_annotation_layout(self):
        """Setup the annotation type checkboxes layout."""
        group_box = QGroupBox("Annotation Types")
        layout = QHBoxLayout()

        self.include_patches_checkbox = QCheckBox("Include Patch Annotations")
        self.include_rectangles_checkbox = QCheckBox("Include Rectangle Annotations")
        self.include_polygons_checkbox = QCheckBox("Include Polygon Annotations")

        # Connect checkbox signals
        self.include_patches_checkbox.stateChanged.connect(self.update_summary_statistics)
        self.include_rectangles_checkbox.stateChanged.connect(self.update_summary_statistics)
        self.include_polygons_checkbox.stateChanged.connect(self.update_summary_statistics)

        layout.addWidget(self.include_patches_checkbox)
        layout.addWidget(self.include_rectangles_checkbox)
        layout.addWidget(self.include_polygons_checkbox)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_table_layout(self):
        """Setup the label counts table layout."""
        group_box = QGroupBox("Annotation Table")
        layout = QVBoxLayout()

        # Label Counts Table
        self.label_counts_table = QTableWidget(0, 7)
        self.label_counts_table.setHorizontalHeaderLabels(["Include",
                                                           "Label",
                                                           "Annotations",
                                                           "Train",
                                                           "Val",
                                                           "Test",
                                                           "Images"])
        # Connect
        self.label_counts_table.cellChanged.connect(self.update_summary_statistics)
        layout.addWidget(self.label_counts_table)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_status_layout(self):
        """Setup the ready status layout."""
        group_box = QGroupBox("Status")
        layout = QHBoxLayout()

        self.ready_label = QLabel()
        layout.addWidget(self.ready_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_button_layout(self):
        """Setup the button layout."""
        button_layout = QHBoxLayout()

        # Add Shuffle button
        self.shuffle_button = QPushButton("Shuffle")
        self.shuffle_button.clicked.connect(self.update_summary_statistics)
        button_layout.addWidget(self.shuffle_button)

        # Add spacer to push OK/Cancel to right
        button_layout.addStretch()

        # Add OK and Cancel buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        button_layout.addWidget(self.buttons)

        self.layout.addLayout(button_layout)

    def update_annotation_type_checkboxes(self):
        raise NotImplementedError("Method must be implemented in the subclass.")

    def on_include_checkbox_state_changed(self, state):
        """
        Handle the state change event of the include checkboxes.

        Args:
            state: The new state of the checkbox.
        """
        if state == Qt.Checked:
            self.update_summary_statistics()
        elif state == Qt.Unchecked:
            self.update_summary_statistics()

    def set_cell_color(self, row, column, color):
        """
        Set the background color of a cell in the label counts table.

        Args:
            row: The row index of the cell.
            column: The column index of the cell.
            color: The color to set as the background.
        """
        item = self.label_counts_table.item(row, column)
        if item is not None:
            item.setBackground(QBrush(color))

    def browse_output_dir(self):
        """
        Browse and select an output directory.
        """
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory")
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def get_class_mapping(self):
        """
        Get the class mapping for the selected labels.

        Returns:
            dict: Dictionary containing class mappings.
        """
        # Get the label objects for the selected labels
        class_mapping = {}
        
        for label in self.main_window.label_window.labels:
            if label.short_label_code in self.selected_labels:
                class_mapping[label.short_label_code] = label.to_dict()

        return class_mapping

    @staticmethod
    def save_class_mapping_json(class_mapping, output_dir_path):
        """
        Save the class mapping dictionary as a JSON file.

        Args:
            class_mapping (dict): Dictionary containing class mappings.
            output_dir_path (str): Path to the output directory.
        """
        # Save the class_mapping dictionary as a JSON file
        class_mapping_path = os.path.join(output_dir_path, "class_mapping.json")
        with open(class_mapping_path, 'w') as json_file:
            json.dump(class_mapping, json_file, indent=4)

    @staticmethod
    def merge_class_mappings(existing_mapping, new_mapping):
        """
        Merge the new class mappings with the existing ones without duplicates.

        Args:
            existing_mapping (dict): Existing class mappings.
            new_mapping (dict): New class mappings.

        Returns:
            dict: Merged class mappings.
        """
        # Merge the new class mappings with the existing ones without duplicates
        merged_mapping = existing_mapping.copy()
        for key, value in new_mapping.items():
            if key not in merged_mapping:
                merged_mapping[key] = value

        return merged_mapping

    def filter_annotations(self):
        """
        Filter annotations based on the selected annotation types and current tab.

        Returns:
            list: List of filtered annotations.
        """
        annotations = list(self.annotation_window.annotations_dict.values())
        filtered_annotations = []

        if self.include_patches_checkbox.isChecked():
            filtered_annotations += [a for a in annotations if isinstance(a, PatchAnnotation)]
        if self.include_rectangles_checkbox.isChecked():
            filtered_annotations += [a for a in annotations if isinstance(a, RectangleAnnotation)]
        if self.include_polygons_checkbox.isChecked():
            filtered_annotations += [a for a in annotations if isinstance(a, PolygonAnnotation)]

        # Filter annotations based on the selected labels
        annotations = [a for a in filtered_annotations if a.label.short_label_code in self.selected_labels]
        
        return annotations

    def populate_class_filter_list(self):
        """
        Populate the class filter list with labels and their counts.
        """
        try:
            # Temporarily disconnect the cellChanged signal
            self.label_counts_table.cellChanged.disconnect()
        except TypeError:
            # Ignore the error if the signal was not connected
            pass

        # Set the row count to 0
        self.label_counts_table.setRowCount(0)

        label_counts = {}
        label_image_counts = {}
        # Count the occurrences of each label and unique images per label
        for annotation in self.annotation_window.annotations_dict.values():
            label = annotation.label.short_label_code
            image_path = annotation.image_path
            if label != 'Review':
                if label in label_counts:
                    label_counts[label] += 1
                    if image_path not in label_image_counts[label]:
                        label_image_counts[label].add(image_path)
                else:
                    label_counts[label] = 1
                    label_image_counts[label] = {image_path}

        # Sort the labels by their counts in descending order
        sorted_label_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)

        # Populate the label counts table with labels and their counts
        self.label_counts_table.setColumnCount(7)
        self.label_counts_table.setHorizontalHeaderLabels(["Include",
                                                           "Label",
                                                           "Annotations",
                                                           "Train",
                                                           "Val",
                                                           "Test",
                                                           "Images"])

        # Populate the label counts table with labels and their counts
        row = 0
        for label, count in sorted_label_counts:
            include_checkbox = QCheckBox()
            include_checkbox.setChecked(True)
            include_checkbox.stateChanged.connect(self.on_include_checkbox_state_changed)
            label_item = QTableWidgetItem(label)
            anno_count = QTableWidgetItem(str(count))
            train_item = QTableWidgetItem("0")
            val_item = QTableWidgetItem("0")
            test_item = QTableWidgetItem("0")
            images_item = QTableWidgetItem(str(len(label_image_counts[label])))

            self.label_counts_table.insertRow(row)
            self.label_counts_table.setCellWidget(row, 0, include_checkbox)
            self.label_counts_table.setItem(row, 1, label_item)
            self.label_counts_table.setItem(row, 2, anno_count)
            self.label_counts_table.setItem(row, 3, train_item)
            self.label_counts_table.setItem(row, 4, val_item)
            self.label_counts_table.setItem(row, 5, test_item)
            self.label_counts_table.setItem(row, 6, images_item)

            row += 1

        # Reconnect the cellChanged signal
        self.label_counts_table.cellChanged.connect(
            self.update_summary_statistics)

    def split_data(self):
        """
        Split the data by images based on the specified ratios.
        """
        self.train_ratio = self.train_ratio_spinbox.value()
        self.val_ratio = self.val_ratio_spinbox.value()
        self.test_ratio = self.test_ratio_spinbox.value()

        images = self.image_window.image_paths
        random.shuffle(images)

        train_split = int(len(images) * self.train_ratio)
        val_split = int(len(images) * (self.train_ratio + self.val_ratio))

        # Initialize splits
        self.train_images = []
        self.val_images = []
        self.test_images = []

        # Assign images to splits based on ratios
        if self.train_ratio > 0:
            self.train_images = images[:train_split]
        if self.val_ratio > 0:
            self.val_images = images[train_split:val_split]
        if self.test_ratio > 0:
            self.test_images = images[val_split:]

    def determine_splits(self):
        """
        Determine the splits for train, validation, and test annotations.
        """
        self.train_annotations = [a for a in self.selected_annotations if a.image_path in self.train_images]
        self.val_annotations = [a for a in self.selected_annotations if a.image_path in self.val_images]
        self.test_annotations = [a for a in self.selected_annotations if a.image_path in self.test_images]

    def check_label_distribution(self):
        """
        Check the label distribution in the splits to ensure all labels are present.

        Returns:
            bool: True if all labels are present in all splits, False otherwise.
        """
        # Get the ratios from the spinboxes
        train_ratio = self.train_ratio_spinbox.value()
        val_ratio = self.val_ratio_spinbox.value()
        test_ratio = self.test_ratio_spinbox.value()

        # Initialize dictionaries to store label counts for each split
        train_label_counts = {}
        val_label_counts = {}
        test_label_counts = {}

        # Count annotations for each label in each split
        for annotation in self.train_annotations:
            label = annotation.label.short_label_code
            train_label_counts[label] = train_label_counts.get(label, 0) + 1

        for annotation in self.val_annotations:
            label = annotation.label.short_label_code
            val_label_counts[label] = val_label_counts.get(label, 0) + 1

        for annotation in self.test_annotations:
            label = annotation.label.short_label_code
            test_label_counts[label] = test_label_counts.get(label, 0) + 1

        # Check the conditions for each split
        for label in self.selected_labels:
            if train_ratio > 0 and (label not in train_label_counts or train_label_counts[label] == 0):
                return False
            if val_ratio > 0 and (label not in val_label_counts or val_label_counts[label] == 0):
                return False
            if test_ratio > 0 and (label not in test_label_counts or test_label_counts[label] == 0):
                return False

        # Additional checks to ensure no empty splits
        if train_ratio > 0 and len(self.train_annotations) == 0:
            return False
        if val_ratio > 0 and len(self.val_annotations) == 0:
            return False
        if test_ratio > 0 and len(self.test_annotations) == 0:
            return False

        # Allow creation of dataset if
        if train_ratio >= 0 and val_ratio >= 0 and test_ratio >= 0:
            return True
        if train_ratio >= 0 and val_ratio >= 0 and test_ratio == 0:
            return True
        if train_ratio == 0 and val_ratio == 0 and test_ratio == 1:
            return True

        return True

    def update_summary_statistics(self):
        """
        Update the summary statistics for the dataset creation.
        """
        if self.updating_summary_statistics:
            return

        self.updating_summary_statistics = True

        # Split the data by images
        self.split_data()

        # Selected labels based on user's selection
        self.selected_labels = []
        for row in range(self.label_counts_table.rowCount()):
            include_checkbox = self.label_counts_table.cellWidget(row, 0)
            if include_checkbox.isChecked():
                label = self.label_counts_table.item(row, 1).text()
                self.selected_labels.append(label)

        # Filter annotations based on the selected annotation types and current tab
        self.selected_annotations = self.filter_annotations()

        # Split the data by annotations
        self.determine_splits()

        # Update the label counts table
        for row in range(self.label_counts_table.rowCount()):
            include_checkbox = self.label_counts_table.cellWidget(row, 0)
            label = self.label_counts_table.item(row, 1).text()
            anno_count = sum(1 for a in self.selected_annotations if a.label.short_label_code == label)
            if include_checkbox.isChecked():
                train_count = sum(1 for a in self.train_annotations if a.label.short_label_code == label)
                val_count = sum(1 for a in self.val_annotations if a.label.short_label_code == label)
                test_count = sum(1 for a in self.test_annotations if a.label.short_label_code == label)
            else:
                train_count = 0
                val_count = 0
                test_count = 0

            self.label_counts_table.item(row, 2).setText(str(anno_count))
            self.label_counts_table.item(row, 3).setText(str(train_count))
            self.label_counts_table.item(row, 4).setText(str(val_count))
            self.label_counts_table.item(row, 5).setText(str(test_count))

            # Set cell colors based on the counts and ratios
            red = QColor(255, 0, 0)
            green = QColor(0, 255, 0)

            if include_checkbox.isChecked():
                self.set_cell_color(row, 3, red if train_count == 0 and self.train_ratio > 0 else green)
                self.set_cell_color(row, 4, red if val_count == 0 and self.val_ratio > 0 else green)
                self.set_cell_color(row, 5, red if test_count == 0 and self.test_ratio > 0 else green)
            else:
                self.set_cell_color(row, 3, green)
                self.set_cell_color(row, 4, green)
                self.set_cell_color(row, 5, green)

        self.ready_status = self.check_label_distribution()
        self.split_status = abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-9
        self.ready_label.setText("✅ Ready" if (self.ready_status and self.split_status) else "❌ Not Ready")

        self.updating_summary_statistics = False
        
    def is_ready(self):
        """Check if the dataset is ready to be created."""
        # Extract the input values, store them in the class variables
        self.dataset_name = self.dataset_name_edit.text()
        self.output_dir = self.output_dir_edit.text()
        self.train_ratio = self.train_ratio_spinbox.value()
        self.val_ratio = self.val_ratio_spinbox.value()
        self.test_ratio = self.test_ratio_spinbox.value()

        if not self.ready_status:
            QMessageBox.warning(self,
                                "Dataset Not Ready",
                                "Not all labels are present in all sets.\n"
                                "Please adjust your selections or sample more data.")
            return False

        if not self.dataset_name or not self.output_dir:
            QMessageBox.warning(self,
                                "Input Error",
                                "All fields must be filled.")
            return False

        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-9:
            QMessageBox.warning(self,
                                "Input Error",
                                "Train, Validation, and Test ratios must sum to 1.0")
            return False
        
        return True
    
    def accept(self):
        """
        Handle the OK button click event to create the dataset.
        """
        if not self.is_ready():
            return

        # Create the output folder
        output_dir_path = os.path.join(self.output_dir, self.dataset_name)
        
        # Check if the output directory exists
        if os.path.exists(output_dir_path):
            reply = QMessageBox.question(self,
                                         "Directory Exists",
                                         "The output directory already exists. Do you want to merge the datasets?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return

            # Read the existing class_mapping.json file if it exists
            class_mapping_path = os.path.join(output_dir_path, "class_mapping.json")
            if os.path.exists(class_mapping_path):
                with open(class_mapping_path, 'r') as json_file:
                    existing_class_mapping = json.load(json_file)
            else:
                existing_class_mapping = {}

            # Merge the new class mappings with the existing ones
            new_class_mapping = self.get_class_mapping()
            merged_class_mapping = self.merge_class_mappings(existing_class_mapping, new_class_mapping)
            self.save_class_mapping_json(merged_class_mapping, output_dir_path)
        else:
            # Save the class mapping JSON file
            os.makedirs(output_dir_path, exist_ok=True)
            class_mapping = self.get_class_mapping()
            self.save_class_mapping_json(class_mapping, output_dir_path)

        # Set the cursor to waiting (busy) cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.create_dataset(output_dir_path)
        # Restore the cursor to the default cursor
        QApplication.restoreOverrideCursor()

        QMessageBox.information(self,
                                "Dataset Created",
                                "Dataset has been successfully created.")
        super().accept()
        
    def create_dataset(self):
        raise NotImplementedError("Method must be implemented in the subclass.")
    
    def process_annotations(self):
        raise NotImplementedError("Method must be implemented in the subclass.")