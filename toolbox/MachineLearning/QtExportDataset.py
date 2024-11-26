import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import yaml
import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby
from operator import attrgetter

import pandas as pd

from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QCheckBox, QVBoxLayout, QLabel, QLineEdit, QDialog,
                             QHBoxLayout, QPushButton, QFormLayout, QDialogButtonBox, QDoubleSpinBox, QGroupBox,
                             QTableWidget, QTableWidgetItem, QRadioButton)

from toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

from toolbox.QtProgressBar import ProgressBar
from toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ExportDatasetDialog(QDialog):
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

        self.selected_labels = []
        self.selected_annotations = []

        # Flag to prevent recursive calls
        self.updating_summary_statistics = False

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Create Dataset")
        self.layout = QVBoxLayout(self)

        # Create horizontal radio box
        self.dataset_type_group = QGroupBox("Dataset Type")
        self.dataset_type_layout = QHBoxLayout()

        self.radio_classification = QRadioButton("Image Classification")
        self.radio_detection = QRadioButton("Object Detection")
        self.radio_segmentation = QRadioButton("Instance Segmentation")

        self.radio_classification.setChecked(True)

        self.dataset_type_layout.addWidget(self.radio_classification)
        self.dataset_type_layout.addWidget(self.radio_detection)
        self.dataset_type_layout.addWidget(self.radio_segmentation)

        self.dataset_type_group.setLayout(self.dataset_type_layout)
        self.layout.addWidget(self.dataset_type_group)

        self.setup_layout()

        # Add OK and Cancel buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

        # Connect signals to update summary statistics
        self.train_ratio_spinbox.valueChanged.connect(self.update_summary_statistics)
        self.val_ratio_spinbox.valueChanged.connect(self.update_summary_statistics)
        self.test_ratio_spinbox.valueChanged.connect(self.update_summary_statistics)
        self.label_counts_table.cellChanged.connect(self.update_summary_statistics)
        self.radio_classification.toggled.connect(self.update_annotation_type_checkboxes)
        self.radio_detection.toggled.connect(self.update_annotation_type_checkboxes)
        self.radio_segmentation.toggled.connect(self.update_annotation_type_checkboxes)

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

    def browse_output_dir(self):
        """
        Browse and select an output directory.
        """
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def get_class_mapping(self):
        """
        Get the class mapping for the selected labels.

        Returns:
            dict: Dictionary containing class mappings.
        """
        # Get the label objects for the selected labels
        labels = [l for l in self.main_window.label_window.labels if l.short_label_code in self.selected_labels]

        class_mapping = {}
        for label in labels:
            # Assuming each label has attributes short_label_code, long_label_code, and label_id
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

    def setup_layout(self):
        """
        Set up the layout for the ExportDatasetDialog.
        """
        # Dataset Name and Output Directory
        self.dataset_name_edit = QLineEdit()
        self.output_dir_edit = QLineEdit()
        self.output_dir_button = QPushButton("Browse...")
        self.output_dir_button.clicked.connect(self.browse_output_dir)

        form_layout = QFormLayout()
        form_layout.addRow("Dataset Name:", self.dataset_name_edit)
        form_layout.addRow("Output Directory:", self.output_dir_edit)
        form_layout.addRow(self.output_dir_button)

        self.layout.addLayout(form_layout)

        # Split Ratios
        split_layout = QHBoxLayout()
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

        split_layout.addWidget(QLabel("Train Ratio:"))
        split_layout.addWidget(self.train_ratio_spinbox)
        split_layout.addWidget(QLabel("Validation Ratio:"))
        split_layout.addWidget(self.val_ratio_spinbox)
        split_layout.addWidget(QLabel("Test Ratio:"))
        split_layout.addWidget(self.test_ratio_spinbox)

        self.layout.addLayout(split_layout)

        # Annotation Type Selection
        self.annotation_type_group = QGroupBox("Annotation Types")
        self.annotation_type_layout = QVBoxLayout()

        self.include_patches_checkbox = QCheckBox("Include Patch Annotations")
        self.include_rectangles_checkbox = QCheckBox("Include Rectangle Annotations")
        self.include_polygons_checkbox = QCheckBox("Include Polygon Annotations")

        # Connect checkbox signals
        self.include_patches_checkbox.stateChanged.connect(self.update_summary_statistics)
        self.include_rectangles_checkbox.stateChanged.connect(self.update_summary_statistics)
        self.include_polygons_checkbox.stateChanged.connect(self.update_summary_statistics)

        self.annotation_type_layout.addWidget(self.include_patches_checkbox)
        self.annotation_type_layout.addWidget(self.include_rectangles_checkbox)
        self.annotation_type_layout.addWidget(self.include_polygons_checkbox)
        self.annotation_type_group.setLayout(self.annotation_type_layout)

        self.layout.addWidget(self.annotation_type_group)

        # Class Filtering
        self.class_filter_group = QGroupBox("Class Filtering")
        self.class_filter_layout = QVBoxLayout()

        # Label Counts Table
        self.label_counts_table = QTableWidget(0, 7)
        self.label_counts_table.setHorizontalHeaderLabels([
            "Include", "Label", "Annotations", "Train", "Val", "Test", "Images"
        ])

        self.class_filter_layout.addWidget(self.label_counts_table)
        self.class_filter_group.setLayout(self.class_filter_layout)
        self.layout.addWidget(self.class_filter_group)

        # Ready Status
        self.ready_label = QLabel()
        self.layout.addWidget(self.ready_label)

        # Shuffle Button
        self.shuffle_button = QPushButton("Shuffle")
        self.shuffle_button.clicked.connect(self.update_summary_statistics)
        self.layout.addWidget(self.shuffle_button)

    def update_annotation_type_checkboxes(self):
        """
        Update the state of annotation type checkboxes based on the selected dataset type.
        """
        if self.radio_classification.isChecked():  # Classification
            self.include_patches_checkbox.setChecked(True)
            self.include_patches_checkbox.setEnabled(False)
            self.include_rectangles_checkbox.setChecked(True)
            self.include_rectangles_checkbox.setEnabled(True)
            self.include_polygons_checkbox.setChecked(True)
            self.include_polygons_checkbox.setEnabled(True)
        elif self.radio_detection.isChecked():  # Detection
            self.include_patches_checkbox.setChecked(False)
            self.include_patches_checkbox.setEnabled(False)
            self.include_rectangles_checkbox.setChecked(True)
            self.include_rectangles_checkbox.setEnabled(False)
            self.include_polygons_checkbox.setChecked(True)
            self.include_polygons_checkbox.setEnabled(True)
        elif self.radio_segmentation.isChecked():  # Segmentation
            self.include_patches_checkbox.setChecked(False)
            self.include_patches_checkbox.setEnabled(False)
            self.include_rectangles_checkbox.setChecked(False)
            self.include_rectangles_checkbox.setEnabled(False)
            self.include_polygons_checkbox.setChecked(True)
            self.include_polygons_checkbox.setEnabled(False)

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
            filtered_annotations += [a for a in filtered_annotations if isinstance(a, PolygonAnnotation)]

        return [a for a in filtered_annotations if a.label.short_label_code in self.selected_labels]

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
        self.label_counts_table.cellChanged.connect(self.update_summary_statistics)

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

    def accept(self):
        """
        Handle the OK button click event to create the dataset.
        """
        dataset_name = self.dataset_name_edit.text()
        output_dir = self.output_dir_edit.text()
        train_ratio = self.train_ratio_spinbox.value()
        val_ratio = self.val_ratio_spinbox.value()
        test_ratio = self.test_ratio_spinbox.value()

        if not self.ready_status:
            QMessageBox.warning(self,
                                "Dataset Not Ready",
                                "Not all labels are present in all sets.\n"
                                "Please adjust your selections or sample more data.")
            return

        if not dataset_name or not output_dir:
            QMessageBox.warning(self,
                                "Input Error",
                                "All fields must be filled.")
            return

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
            QMessageBox.warning(self,
                                "Input Error",
                                "Train, Validation, and Test ratios must sum to 1.0")
            return

        output_dir_path = os.path.join(output_dir, dataset_name)
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

        if self.radio_classification.isChecked():  # Image Classification
            self.create_classification_dataset(output_dir_path)
        elif self.radio_detection.isChecked():  # Object Detection
            self.create_detection_dataset(output_dir_path)
        elif self.radio_segmentation.isChecked():  # Instance Segmentation
            self.create_segmentation_dataset(output_dir_path)

        # Restore the cursor to the default cursor
        QApplication.restoreOverrideCursor()

        QMessageBox.information(self,
                                "Dataset Created",
                                "Dataset has been successfully created.")
        super().accept()

    def create_classification_dataset(self, output_dir_path):
        """
        Create an image classification dataset.

        Args:
            output_dir_path (str): Path to the output directory.
        """
        # Create the train, val, and test directories
        train_dir = os.path.join(output_dir_path, 'train')
        val_dir = os.path.join(output_dir_path, 'val')
        test_dir = os.path.join(output_dir_path, 'test')

        # Create a blank sample in train folder it's a test-only dataset
        # Ultralytics bug... it doesn't like empty directories (hacky)
        for label in self.selected_labels:
            label_folder = os.path.join(train_dir, label)
            os.makedirs(f"{train_dir}/{label}/", exist_ok=True)
            with open(os.path.join(label_folder, 'NULL.jpg'), 'w') as f:
                f.write("")

        self.process_classification_annotations(self.train_annotations, train_dir, "Training")
        self.process_classification_annotations(self.val_annotations, val_dir, "Validation")
        self.process_classification_annotations(self.test_annotations, test_dir, "Testing")

        # Output the annotations as CoralNet CSV file
        df = []

        for annotation in self.selected_annotations:
            df.append(annotation.to_coralnet())

        pd.DataFrame(df).to_csv(f"{output_dir_path}/dataset.csv", index=False)

    def process_classification_annotations(self, annotations, split_dir, split):
        """
        Process and save classification annotations.

        Args:
            annotations (list): List of annotations.
            split_dir (str): Path to the split directory.
            split (str): Split name (e.g., "Training", "Validation", "Testing").
        """
        # Get unique image paths
        image_paths = list(set(a.image_path for a in annotations))
        if not image_paths:
            return

        progress_bar = ProgressBar(self, title=f"Creating {split} Dataset")
        progress_bar.show()
        progress_bar.start_progress(len(image_paths))

        def process_image_annotations(image_path, image_annotations):
            # Crop the image based on the annotations
            image_annotations = self.annotation_window.crop_these_image_annotations(image_path, image_annotations)
            cropped_images = []
            for annotation in image_annotations:
                label_code = annotation.label.short_label_code
                output_path = os.path.join(split_dir, label_code)
                # Create a split / label directory if it does not exist
                os.makedirs(output_path, exist_ok=True)
                output_filename = f"{label_code}_{annotation.id}.jpg"
                full_output_path = os.path.join(output_path, output_filename)
                # Add the cropped image and its output path to the list
                cropped_images.append((annotation.cropped_image, full_output_path))
            return cropped_images

        def save_annotations(cropped_images):
            for pixmap, path in cropped_images:
                try:
                    pixmap.save(path, "JPG", quality=100)
                except Exception as e:
                    print(f"ERROR: Issue saving image {path}: {e}")
                    # Optionally, save as PNG if JPG fails
                    png_path = path.replace(".jpg", ".png")
                    pixmap.save(png_path, "PNG")

        # Group annotations by image path
        grouped_annotations = groupby(sorted(annotations, key=attrgetter('image_path')), key=attrgetter('image_path'))

        with ThreadPoolExecutor() as executor:
            future_to_image = {}
            for image_path, group in grouped_annotations:
                future = executor.submit(process_image_annotations, image_path, list(group))
                future_to_image[future] = image_path

            for future in as_completed(future_to_image):
                image_path = future_to_image[future]
                try:
                    cropped_images = future.result()
                    save_annotations(cropped_images)
                except Exception as exc:
                    print(f'{image_path} generated an exception: {exc}')
                finally:
                    progress_bar.update_progress()

        progress_bar.stop_progress()
        progress_bar.close()

    def create_detection_dataset(self, output_dir_path):
        """
        Create an object detection dataset.

        Args:
            output_dir_path (str): Path to the output directory.
        """
        # Create the yaml file
        yaml_path = os.path.join(output_dir_path, 'data.yaml')

        # Create the train, val, and test directories
        train_dir = os.path.join(output_dir_path, 'train')
        val_dir = os.path.join(output_dir_path, 'valid')
        test_dir = os.path.join(output_dir_path, 'test')
        names = self.selected_labels
        num_classes = len(self.selected_labels)

        # Define the data as a dictionary
        data = {
            'train': '../train/images',
            'val': '../valid/images',
            'test': '../test/images',
            'nc': num_classes,  # Replace `num_classes` with the actual number of classes
            'names': names  # Replace `names` with the actual list of class names
        }

        # Write the data to the YAML file
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        # Create the train, val, and test directories
        os.makedirs(f"{train_dir}/images", exist_ok=True)
        os.makedirs(f"{train_dir}/labels", exist_ok=True)
        os.makedirs(f"{val_dir}/images", exist_ok=True)
        os.makedirs(f"{val_dir}/labels", exist_ok=True)
        os.makedirs(f"{test_dir}/images", exist_ok=True)
        os.makedirs(f"{test_dir}/labels", exist_ok=True)

        self.process_detection_annotations(self.train_annotations, train_dir, "Training")
        self.process_detection_annotations(self.val_annotations, val_dir, "Validation")
        self.process_detection_annotations(self.test_annotations, test_dir, "Testing")

    def process_detection_annotations(self, annotations, split_dir, split):
        """
        Process and save detection annotations.

        Args:
            annotations (list): List of annotations.
            split_dir (str): Path to the split directory.
            split (str): Split name (e.g., "Training", "Validation", "Testing").
        """
        # Get unique image paths
        image_paths = list(set(a.image_path for a in annotations))
        if not image_paths:
            return

        progress_bar = ProgressBar(self, title=f"Creating {split} Dataset")
        progress_bar.show()
        progress_bar.start_progress(len(image_paths))

        for image_path in image_paths:
            yolo_annotations = []
            image_height, image_width = self.image_window.rasterio_open(image_path).shape
            image_annotations = [a for a in annotations if a.image_path == image_path]

            for image_annotation in image_annotations:
                class_label, annotation = image_annotation.to_yolo_detection(image_width, image_height)
                class_number = self.selected_labels.index(class_label)
                yolo_annotations.append(f"{class_number} {annotation}")

            # Save the annotations to a text file
            file_ext = image_path.split(".")[-1]
            text_file = os.path.basename(image_path).replace(f".{file_ext}", ".txt")
            text_path = os.path.join(f"{split_dir}/labels", text_file)

            # Write the annotations to the text file
            with open(text_path, 'w') as f:
                for annotation in yolo_annotations:
                    f.write(annotation + '\n')

            # Copy the image to the split directory
            shutil.copy(image_path, f"{split_dir}/images/{os.path.basename(image_path)}")

            progress_bar.update_progress()

        progress_bar.stop_progress()
        progress_bar.close()

    def create_segmentation_dataset(self, output_dir_path):
        """
        Create an instance segmentation dataset.

        Args:
            output_dir_path (str): Path to the output directory.
        """
        # Create the yaml file
        yaml_path = os.path.join(output_dir_path, 'data.yaml')

        # Create the train, val, and test directories
        train_dir = os.path.join(output_dir_path, 'train')
        val_dir = os.path.join(output_dir_path, 'valid')
        test_dir = os.path.join(output_dir_path, 'test')
        names = self.selected_labels
        num_classes = len(self.selected_labels)

        # Create the data.yaml file
        with open(yaml_path, 'w') as f:
            f.write(f"train: ../train/images\n")
            f.write(f"val: ../valid/images\n")
            f.write(f"test: ../test/images\n\n")
            f.write(f"nc: {num_classes}\n")
            f.write(f"names: {names}\n")

        # Create the train, val, and test directories
        os.makedirs(f"{train_dir}/images", exist_ok=True)
        os.makedirs(f"{train_dir}/labels", exist_ok=True)
        os.makedirs(f"{val_dir}/images", exist_ok=True)
        os.makedirs(f"{val_dir}/labels", exist_ok=True)
        os.makedirs(f"{test_dir}/images", exist_ok=True)
        os.makedirs(f"{test_dir}/labels", exist_ok=True)

        self.process_segmentation_annotations(self.train_annotations, train_dir, "Training")
        self.process_segmentation_annotations(self.val_annotations, val_dir, "Validation")
        self.process_segmentation_annotations(self.test_annotations, test_dir, "Testing")

    def process_segmentation_annotations(self, annotations, split_dir, split):
        """
        Process and save segmentation annotations.

        Args:
            annotations (list): List of annotations.
            split_dir (str): Path to the split directory.
            split (str): Split name (e.g., "Training", "Validation", "Testing").
        """
        # Get unique image paths
        image_paths = list(set(a.image_path for a in annotations))
        if not image_paths:
            return

        progress_bar = ProgressBar(self, title=f"Creating {split} Dataset")
        progress_bar.show()
        progress_bar.start_progress(len(image_paths))

        for image_path in image_paths:
            yolo_annotations = []
            image_height, image_width = self.image_window.rasterio_open(image_path).shape
            image_annotations = [a for a in annotations if a.image_path == image_path]

            for image_annotation in image_annotations:
                class_label, annotation = image_annotation.to_yolo_segmentation(image_width, image_height)
                class_number = self.selected_labels.index(class_label)
                yolo_annotations.append(f"{class_number} {annotation}")

            # Save the annotations to a text file
            file_ext = image_path.split(".")[-1]
            text_file = os.path.basename(image_path).replace(f".{file_ext}", ".txt")
            text_path = os.path.join(f"{split_dir}/labels", text_file)

            # Write the annotations to the text file
            with open(text_path, 'w') as f:
                for annotation in yolo_annotations:
                    f.write(annotation + '\n')

            # Copy the image to the split directory
            shutil.copy(image_path, f"{split_dir}/images/{os.path.basename(image_path)}")

            progress_bar.update_progress()

        progress_bar.stop_progress()
        progress_bar.close()
