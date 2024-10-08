import datetime
import gc
import json
import os
import random
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby
from operator import attrgetter
from pathlib import Path

import numpy as np
import ultralytics.engine.validator as validator
import ultralytics.models.yolo.classify.train as train_build

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QBrush, QColor, QShowEvent
from PyQt5.QtWidgets import (QFileDialog, QApplication, QScrollArea, QMessageBox, QCheckBox, QWidget, QVBoxLayout,
                             QLabel, QLineEdit, QDialog, QHBoxLayout, QTextEdit, QPushButton, QComboBox, QSpinBox,
                             QFormLayout, QTabWidget, QDialogButtonBox, QDoubleSpinBox, QGroupBox, QTableWidget,
                             QTableWidgetItem, QSlider, QButtonGroup)

from torch.cuda import empty_cache
from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset

from toolbox.QtProgressBar import ProgressBar
from toolbox.utilities import pixmap_to_numpy

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class CreateDatasetDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window
        self.image_window = main_window.image_window

        self.resize(1000, 600)

        self.selected_labels = []
        self.selected_annotations = []

        # Flag to prevent recursive calls
        self.updating_summary_statistics = False

        self.setWindowTitle("Create Dataset")
        self.layout = QVBoxLayout(self)

        # Create tabs
        self.tabs = QTabWidget()
        self.tab_classification = QWidget()
        self.tab_segmentation = QWidget()  # Future work

        self.tabs.addTab(self.tab_classification, "Image Classification")
        self.tabs.addTab(self.tab_segmentation, "Instance Segmentation")

        # Setup classification tab
        self.setup_classification_tab()
        # Setup segmentation tab
        self.setup_segmentation_tab()
        # Add the tabs to the layout
        self.layout.addWidget(self.tabs)

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

    def setup_classification_tab(self):
        layout = QVBoxLayout()

        # Dataset Name and Output Directory
        self.dataset_name_edit = QLineEdit()
        self.output_dir_edit = QLineEdit()
        self.output_dir_button = QPushButton("Browse...")
        self.output_dir_button.clicked.connect(self.browse_output_dir)

        form_layout = QFormLayout()
        form_layout.addRow("Dataset Name:", self.dataset_name_edit)
        form_layout.addRow("Output Directory:", self.output_dir_edit)
        form_layout.addRow(self.output_dir_button)

        layout.addLayout(form_layout)

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

        layout.addLayout(split_layout)

        # Class Filtering
        self.class_filter_group = QGroupBox("Class Filtering")
        self.class_filter_layout = QVBoxLayout()

        # Label Counts Table
        self.label_counts_table = QTableWidget(0, 6)
        self.label_counts_table.setHorizontalHeaderLabels(["Include", "Label", "Annotations", "Train", "Val", "Test"])
        self.class_filter_layout.addWidget(self.label_counts_table)
        self.class_filter_group.setLayout(self.class_filter_layout)
        layout.addWidget(self.class_filter_group)

        # Ready Status
        self.ready_label = QLabel()
        layout.addWidget(self.ready_label)

        # Shuffle Button
        self.shuffle_button = QPushButton("Shuffle")
        self.shuffle_button.clicked.connect(self.update_summary_statistics)
        layout.addWidget(self.shuffle_button)

        # Add the layout
        self.tab_classification.setLayout(layout)

        # Populate class filter list
        self.populate_class_filter_list()
        # Initial update of summary statistics
        self.update_summary_statistics()

    def setup_segmentation_tab(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Instance Segmentation"))
        self.tab_segmentation.setLayout(layout)

    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def set_cell_color(self, row, column, color):
        item = self.label_counts_table.item(row, column)
        if item is not None:
            item.setBackground(QBrush(color))

    def populate_class_filter_list(self):
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

    def on_include_checkbox_state_changed(self, state):
        if state == Qt.Checked:
            self.update_summary_statistics()
        elif state == Qt.Unchecked:
            self.update_summary_statistics()

    def split_data(self):
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
        self.train_annotations = [a for a in self.selected_annotations if a.image_path in self.train_images]
        self.val_annotations = [a for a in self.selected_annotations if a.image_path in self.val_images]
        self.test_annotations = [a for a in self.selected_annotations if a.image_path in self.test_images]

    def check_label_distribution(self):
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

        # All annotations in project
        annotations = list(self.annotation_window.annotations_dict.values())
        self.selected_annotations = [a for a in annotations if a.label.short_label_code in self.selected_labels]

        # Split the data by annotations
        self.determine_splits()

        # Update the label counts table
        for row in range(self.label_counts_table.rowCount()):
            include_checkbox = self.label_counts_table.cellWidget(row, 0)
            label = self.label_counts_table.item(row, 1).text()
            anno_count = sum(1 for a in annotations if a.label.short_label_code == label)
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

    def get_class_mapping(self):
        # Get the label objects for the selected labels
        labels = [l for l in self.main_window.label_window.labels if l.short_label_code in self.selected_labels]

        class_mapping = {}
        for label in labels:
            # Assuming each label has attributes short_label_code, long_label_code, and label_id
            class_mapping[label.short_label_code] = label.to_dict()

        return class_mapping

    def save_class_mapping_json(self, class_mapping, output_dir_path):
        # Save the class_mapping dictionary as a JSON file
        class_mapping_path = os.path.join(output_dir_path, "class_mapping.json")
        with open(class_mapping_path, 'w') as json_file:
            json.dump(class_mapping, json_file, indent=4)

    def merge_class_mappings(self, existing_mapping, new_mapping):
        # Merge the new class mappings with the existing ones without duplicates
        merged_mapping = existing_mapping.copy()
        for key, value in new_mapping.items():
            if key not in merged_mapping:
                merged_mapping[key] = value

        return merged_mapping

    def accept(self):
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
            QMessageBox.warning(self, "Input Error", "All fields must be filled.")
            return

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
            QMessageBox.warning(self, "Input Error", "Train, Validation, and Test ratios must sum to 1.0")
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

        # Create the train, val, and test directories
        train_dir = os.path.join(output_dir_path, 'train')
        val_dir = os.path.join(output_dir_path, 'val')
        test_dir = os.path.join(output_dir_path, 'test')

        # Set the cursor to waiting (busy) cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Create a blank sample in train folder it's a test-only dataset
        # Ultralytics bug... it doesn't like empty directories (hacky)
        if not sum([train_ratio, val_ratio]):
            for label in self.selected_labels:
                label_folder = os.path.join(train_dir, label)
                os.makedirs(f"{train_dir}/{label}/", exist_ok=True)
                with open(os.path.join(label_folder, 'NULL.jpg'), 'w') as f:
                    f.write("")

        self.process_annotations(self.train_annotations, train_dir, "Training")
        self.process_annotations(self.val_annotations, val_dir, "Validation")
        self.process_annotations(self.test_annotations, test_dir, "Testing")

        # Restore the cursor to the default cursor
        QApplication.restoreOverrideCursor()

        QMessageBox.information(self,
                                "Dataset Created",
                                "Dataset has been successfully created.")
        super().accept()

    def process_annotations(self, annotations, split_dir, split):
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

        def save_images(cropped_images):
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
                    save_images(cropped_images)
                except Exception as exc:
                    print(f'{image_path} generated an exception: {exc}')
                finally:
                    progress_bar.update_progress()

        progress_bar.stop_progress()
        progress_bar.close()

    def showEvent(self, event):
        super().showEvent(event)
        self.populate_class_filter_list()
        self.update_summary_statistics()


class MergeDatasetsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Merge Datasets")
        self.resize(400, 200)

        self.layout = QVBoxLayout(self)

        # Dataset Name
        self.dataset_name_edit = QLineEdit()
        self.layout.addWidget(QLabel("Dataset Name:"))
        self.layout.addWidget(self.dataset_name_edit)

        # Output Directory Chooser
        self.output_dir_edit = QLineEdit()
        self.output_dir_button = QPushButton("Browse...")
        self.output_dir_button.clicked.connect(self.browse_output_directory)
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_dir_button)
        self.layout.addWidget(QLabel("Output Directory:"))
        self.layout.addLayout(output_dir_layout)

        # Create tabs
        self.tabs = QTabWidget()
        self.tab_classification = QWidget()
        self.tab_segmentation = QWidget()

        self.tabs.addTab(self.tab_classification, "Image Classification")
        self.tabs.addTab(self.tab_segmentation, "Instance Segmentation")

        self.layout.addWidget(self.tabs)

        # Setup tabs
        self.setup_tab(self.tab_classification)

        # OK and Cancel Buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

        # Track valid directories and their class mappings
        self.valid_directories = []

    def setup_tab(self, tab):
        layout = QVBoxLayout()

        # Existing Dataset Directories
        self.existing_dirs_layout = QVBoxLayout()
        layout.addWidget(QLabel("Existing Dataset Directories:"))
        layout.addLayout(self.existing_dirs_layout)

        # Add two default directory choosers
        self.add_directory_chooser()
        self.add_directory_chooser()

        # Add Directory Button
        self.add_dir_button = QPushButton("Add Dataset")
        self.add_dir_button.clicked.connect(self.add_directory_chooser)
        layout.addWidget(self.add_dir_button)

        tab.setLayout(layout)

    def browse_output_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def add_directory_chooser(self):
        dir_chooser = QWidget()
        dir_layout = QHBoxLayout(dir_chooser)

        status_label = QLabel()
        dir_layout.addWidget(status_label)

        dir_edit = QLineEdit()
        dir_layout.addWidget(dir_edit)

        dir_button = QPushButton("Browse...")
        dir_button.clicked.connect(lambda: self.browse_existing_directory(dir_edit, status_label))
        dir_layout.addWidget(dir_button)

        self.existing_dirs_layout.addWidget(dir_chooser)

    def browse_existing_directory(self, dir_edit, status_label):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Existing Dataset Directory")
        if dir_path:
            dir_edit.setText(dir_path)
            self.validate_directory(dir_path, dir_edit, status_label)

    def validate_directory(self, dir_path, dir_edit, status_label):
        class_mapping_path = os.path.join(dir_path, "class_mapping.json")
        if os.path.exists(class_mapping_path):
            status_label.setText("✅")
            self.valid_directories.append((dir_path, class_mapping_path))
        else:
            status_label.setText("❌")
            self.valid_directories = [(d, c) for d, c in self.valid_directories if d != dir_path]

    def merge_datasets(self):
        output_dir = self.output_dir_edit.text()
        if not output_dir:
            QMessageBox.warning(self, "Input Error", "Output directory must be specified.")
            return

        dataset_name = self.dataset_name_edit.text()
        if not dataset_name:
            QMessageBox.warning(self, "Input Error", "Dataset name must be specified.")
            return

        output_dir_path = os.path.join(output_dir, dataset_name)
        os.makedirs(output_dir_path, exist_ok=True)

        merged_class_mapping = {}

        # Create a progress dialog
        progress_bar = ProgressBar(self, title=f"Merging Datasets")
        progress_bar.show()
        progress_bar.start_progress(len(self.valid_directories))

        def copy_directory(src, dest):
            shutil.copytree(src, dest, dirs_exist_ok=True)

        with ThreadPoolExecutor() as executor:
            futures = []
            for dir_path, class_mapping_path in self.valid_directories:
                with open(class_mapping_path, 'r') as json_file:
                    class_mapping = json.load(json_file)
                    merged_class_mapping.update(class_mapping)

                for split in ['train', 'val', 'test']:
                    src_split_dir = os.path.join(dir_path, split)
                    dest_split_dir = os.path.join(output_dir_path, split)
                    if os.path.exists(src_split_dir):
                        future = executor.submit(copy_directory, src_split_dir, dest_split_dir)
                        futures.append(future)

            # Wait for all copying tasks to complete
            for i, future in enumerate(as_completed(futures)):
                future.result()
                progress_bar.update_progress()

        progress_bar.stop_progress()
        progress_bar.close()

        merged_class_mapping_path = os.path.join(output_dir_path, "class_mapping.json")
        with open(merged_class_mapping_path, 'w') as json_file:
            json.dump(merged_class_mapping, json_file, indent=4)

        QMessageBox.information(self, "Success", "Datasets merged successfully!")

    def accept(self):
        self.merge_datasets()
        super().accept()


class WeightedClassificationDataset(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        """
        Initialize the WeightedClassificationDataset.

        Args:
            class_weights (list or numpy array): A list or array of weights corresponding to each class.
        """
        super(WeightedClassificationDataset, self).__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix

        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts

        # Aggregation function
        self.agg_func = np.mean

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

    def count_instances(self):
        """
        Count the number of instances per class

        Returns:
            dict: A dict containing the counts for each class.
        """
        self.counts = [0 for i in range(len(self.base.classes))]
        for _, class_idx, _, _ in self.samples:
            self.counts[class_idx] += 1

        self.counts = np.array(self.counts)
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        """
        Calculate the aggregated weight for each label based on class weights.

        Returns:
            list: A list of aggregated weights corresponding to each label.
        """
        weights = []
        for _, class_idx, _, _ in self.samples:
            weight = self.agg_func(self.class_weights[class_idx])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        """
        Calculate and store the sampling probabilities based on the weights.

        Returns:
            list: A list of sampling probabilities corresponding to each label.
        """
        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]
        return probabilities

    def __getitem__(self, index):
        """
        Return transformed label information based on the sampled index.
        """
        if self.train_mode:
            index = np.random.choice(len(self.samples), p=self.probabilities)

        return super(WeightedClassificationDataset, self).__getitem__(index)


class TrainModelWorker(QThread):
    training_started = pyqtSignal()
    training_completed = pyqtSignal()
    training_error = pyqtSignal(str)

    def __init__(self, params, device, class_mapping):
        super().__init__()
        self.params = params
        self.device = device
        self.class_mapping = class_mapping
        self.model = None

    def run(self):
        try:
            # Emit signal to indicate training has started
            self.training_started.emit()

            # Extract parameters
            model_path = self.params.pop('model', None)
            weighted = self.params.pop('weighted', False)

            if weighted:
                # Use the custom dataset class for weighted sampling
                train_build.ClassificationDataset = WeightedClassificationDataset

            # Load the model, train, and save the best weights
            self.model = YOLO(model_path)
            self.model.train(**self.params, device=self.device)

            # Evaluate the model after training
            self._evaluate_model()
            # Emit signal to indicate training has completed
            self.training_completed.emit()

        except Exception as e:
            self.training_error.emit(str(e))
        finally:
            self._cleanup()

    def _evaluate_model(self):
        try:
            # Create an instance of EvaluateModelWorker and start it
            eval_params = {
                'data': self.params['data'],
                'imgsz': self.params['imgsz'],
                'split': 'test',  # Evaluate on the test set only
                'save_dir': Path(self.params['project']) / self.params['name'] / 'test'
            }
            # Update the class mapping with target model names
            class_mapping = {name: self.class_mapping[name] for name in self.model.names}

            # Create and start the worker thread
            eval_worker = EvaluateModelWorker(model=self.model,
                                              params=eval_params,
                                              class_mapping=class_mapping)

            eval_worker.evaluation_error.connect(self.on_evaluation_error)
            eval_worker.run()  # Run the evaluation synchronously (same thread)
        except Exception as e:
            self.training_error.emit(str(e))

    def on_evaluation_started(self):
        pass

    def on_evaluation_completed(self):
        pass

    def on_evaluation_error(self, error_message):
        # Handle any errors that occur during evaluation
        self.training_error.emit(error_message)

    def _cleanup(self):
        del self.model
        gc.collect()
        empty_cache()


class TrainModelDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        # For holding parameters
        self.params = {}
        self.custom_params = []
        # Best model weights
        self.model_path = None
        # Class mapping
        self.class_mapping = {}

        self.setWindowTitle("Train Model")

        # Set window settings
        self.setWindowFlags(Qt.Window |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint |
                            Qt.WindowTitleHint)

        self.resize(600, 800)

        # Main layout
        self.main_layout = QVBoxLayout()

        # Create and set up the tabs, parameters form, and console output
        self.setup_ui()

        # Wrap the main layout in a QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_widget.setLayout(self.main_layout)
        scroll_area.setWidget(scroll_widget)

        # Set the scroll area as the main layout of the dialog
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scroll_area)

    def setup_ui(self):
        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Details on different hyperparameters can be found "
                            "<a href='https://docs.ultralytics.com/modes/train/#train-settings'>here</a>.")
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        self.main_layout.addWidget(info_label)

        # Create tabs
        self.tabs = QTabWidget()
        self.tab_classification = QWidget()
        self.tab_segmentation = QWidget()

        self.tabs.addTab(self.tab_classification, "Image Classification")
        self.tabs.addTab(self.tab_segmentation, "Instance Segmentation")

        # Setup tabs
        self.setup_classification_tab()
        self.setup_segmentation_tab()
        self.main_layout.addWidget(self.tabs)

        # Parameters Form
        self.form_layout = QFormLayout()

        # Project
        self.project_edit = QLineEdit()
        self.project_button = QPushButton("Browse...")
        self.project_button.clicked.connect(self.browse_project_dir)
        project_layout = QHBoxLayout()
        project_layout.addWidget(self.project_edit)
        project_layout.addWidget(self.project_button)
        self.form_layout.addRow("Project:", project_layout)

        # Name
        self.name_edit = QLineEdit()
        self.form_layout.addRow("Name:", self.name_edit)

        # Existing Model
        self.model_edit = QLineEdit()
        self.model_button = QPushButton("Browse...")
        self.model_button.clicked.connect(self.browse_model_file)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(self.model_button)
        self.form_layout.addRow("Existing Model:", model_layout)

        # Epochs
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setMinimum(1)
        self.epochs_spinbox.setMaximum(1000)
        self.epochs_spinbox.setValue(100)
        self.form_layout.addRow("Epochs:", self.epochs_spinbox)

        # Patience
        self.patience_spinbox = QSpinBox()
        self.patience_spinbox.setMinimum(1)
        self.patience_spinbox.setMaximum(1000)
        self.patience_spinbox.setValue(30)
        self.form_layout.addRow("Patience:", self.patience_spinbox)

        # Imgsz
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setMinimum(16)
        self.imgsz_spinbox.setMaximum(4096)
        self.imgsz_spinbox.setValue(256)
        self.form_layout.addRow("Image Size:", self.imgsz_spinbox)

        # Batch
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setMinimum(1)
        self.batch_spinbox.setMaximum(1024)
        self.batch_spinbox.setValue(512)
        self.form_layout.addRow("Batch Size:", self.batch_spinbox)

        # Workers
        self.workers_spinbox = QSpinBox()
        self.workers_spinbox.setMinimum(1)
        self.workers_spinbox.setMaximum(64)
        self.workers_spinbox.setValue(8)
        self.form_layout.addRow("Workers:", self.workers_spinbox)

        # Save
        self.save_checkbox = QCheckBox()
        self.save_checkbox.setChecked(True)
        self.form_layout.addRow("Save:", self.save_checkbox)

        # Save Period
        self.save_period_spinbox = QSpinBox()
        self.save_period_spinbox.setMinimum(-1)
        self.save_period_spinbox.setMaximum(1000)
        self.save_period_spinbox.setValue(-1)
        self.form_layout.addRow("Save Period:", self.save_period_spinbox)

        # Pretrained
        self.pretrained_checkbox = QCheckBox()
        self.pretrained_checkbox.setChecked(True)
        self.form_layout.addRow("Pretrained:", self.pretrained_checkbox)

        # Freeze
        self.freeze_edit = QLineEdit()
        self.form_layout.addRow("Freeze Layers:", self.freeze_edit)

        # Weighted Dataset
        self.weighted_checkbox = QCheckBox()
        self.weighted_checkbox.setChecked(False)
        self.form_layout.addRow("Weighted:", self.weighted_checkbox)

        # Dropout
        self.dropout_spinbox = QDoubleSpinBox()
        self.dropout_spinbox.setMinimum(0.0)
        self.dropout_spinbox.setMaximum(1.0)
        self.dropout_spinbox.setValue(0.0)
        self.form_layout.addRow("Dropout:", self.dropout_spinbox)

        # Optimizer
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"])
        self.optimizer_combo.setCurrentText("auto")
        self.form_layout.addRow("Optimizer:", self.optimizer_combo)

        # Lr0
        self.lr0_spinbox = QDoubleSpinBox()
        self.lr0_spinbox.setMinimum(0.0001)
        self.lr0_spinbox.setMaximum(1.0)
        self.lr0_spinbox.setValue(0.01)
        self.form_layout.addRow("Learning Rate (lr0):", self.lr0_spinbox)

        # Val
        self.val_checkbox = QCheckBox()
        self.val_checkbox.setChecked(True)
        self.form_layout.addRow("Validation:", self.val_checkbox)

        # Fraction
        self.fraction_spinbox = QDoubleSpinBox()
        self.fraction_spinbox.setMinimum(0.1)
        self.fraction_spinbox.setMaximum(1.0)
        self.fraction_spinbox.setValue(1.0)
        self.form_layout.addRow("Fraction:", self.fraction_spinbox)

        # Verbose
        self.verbose_checkbox = QCheckBox()
        self.verbose_checkbox.setChecked(True)
        self.form_layout.addRow("Verbose:", self.verbose_checkbox)

        # Add custom parameters section
        self.custom_params_layout = QVBoxLayout()
        self.form_layout.addRow("Additional Parameters:", self.custom_params_layout)

        # Add button for new parameter pairs
        self.add_param_button = QPushButton("Add Parameter")
        self.add_param_button.clicked.connect(self.add_parameter_pair)
        self.form_layout.addRow("", self.add_param_button)

        self.main_layout.addLayout(self.form_layout)

        # Add OK and Cancel buttons
        self.buttons = QPushButton("OK")
        self.buttons.clicked.connect(self.accept)
        self.main_layout.addWidget(self.buttons)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.main_layout.addWidget(self.cancel_button)

    def add_parameter_pair(self):
        param_layout = QHBoxLayout()
        param_name = QLineEdit()
        param_value = QLineEdit()
        param_layout.addWidget(param_name)
        param_layout.addWidget(param_value)

        self.custom_params.append((param_name, param_value))
        self.custom_params_layout.addLayout(param_layout)

    def browse_dataset_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if dir_path:
            self.dataset_dir_edit.setText(dir_path)
            class_mapping_path = f"{dir_path}/class_mapping.json"
            if os.path.exists(class_mapping_path):
                self.class_mapping_edit.setText(class_mapping_path)
                self.class_mapping = json.load(open(class_mapping_path, 'r'))

    def browse_class_mapping_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Select Class Mapping File",
                                                   "",
                                                   "JSON Files (*.json)")
        if file_path:
            self.class_mapping_edit.setText(file_path)
            self.class_mapping = json.load(open(file_path, 'r'))

    def browse_dataset_yaml(self):
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Select Dataset YAML File",
                                                   "",
                                                   "YAML Files (*.yaml *.yml)")
        if file_path:
            self.dataset_yaml_edit.setText(file_path)

    def browse_project_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if dir_path:
            self.project_edit.setText(dir_path)

    def browse_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File")
        if file_path:
            self.model_edit.setText(file_path)

    def setup_classification_tab(self):
        layout = QVBoxLayout()

        # Dataset Directory
        self.dataset_dir_edit = QLineEdit()
        self.dataset_dir_button = QPushButton("Browse...")
        self.dataset_dir_button.clicked.connect(self.browse_dataset_dir)
        dataset_dir_layout = QHBoxLayout()
        dataset_dir_layout.addWidget(QLabel("Dataset Directory:"))
        dataset_dir_layout.addWidget(self.dataset_dir_edit)
        dataset_dir_layout.addWidget(self.dataset_dir_button)
        layout.addLayout(dataset_dir_layout)

        # Class Mapping
        self.class_mapping_edit = QLineEdit()
        self.class_mapping_button = QPushButton("Browse...")
        self.class_mapping_button.clicked.connect(self.browse_class_mapping_file)
        class_mapping_layout = QHBoxLayout()
        class_mapping_layout.addWidget(QLabel("Class Mapping:"))
        class_mapping_layout.addWidget(self.class_mapping_edit)
        class_mapping_layout.addWidget(self.class_mapping_button)
        layout.addLayout(class_mapping_layout)

        # Classification Model Dropdown
        self.classification_model_combo = QComboBox()
        self.classification_model_combo.addItems(["yolov8n-cls.pt",
                                                  "yolov8s-cls.pt",
                                                  "yolov8m-cls.pt",
                                                  "yolov8l-cls.pt",
                                                  "yolov8x-cls.pt"])

        self.classification_model_combo.setEditable(True)
        layout.addWidget(QLabel("Select or Enter Classification Model:"))
        layout.addWidget(self.classification_model_combo)

        self.tab_classification.setLayout(layout)

    def setup_segmentation_tab(self):
        layout = QVBoxLayout()

        self.dataset_yaml_edit = QLineEdit()
        self.dataset_yaml_button = QPushButton("Browse...")
        self.dataset_yaml_button.clicked.connect(self.browse_dataset_yaml)
        dataset_yaml_layout = QHBoxLayout()
        dataset_yaml_layout.addWidget(QLabel("Dataset YAML:"))
        dataset_yaml_layout.addWidget(self.dataset_yaml_edit)
        dataset_yaml_layout.addWidget(self.dataset_yaml_button)
        layout.addLayout(dataset_yaml_layout)

        # Segmentation Model Dropdown
        self.segmentation_model_combo = QComboBox()
        self.segmentation_model_combo.addItems(["yolov8n-seg.pt",
                                                "yolov8s-seg.pt",
                                                "yolov8m-seg.pt",
                                                "yolov8l-seg.pt",
                                                "yolov8x-seg.pt"])

        self.segmentation_model_combo.setEditable(True)
        layout.addWidget(QLabel("Select or Enter Segmentation Model:"))
        layout.addWidget(self.segmentation_model_combo)

        self.tab_segmentation.setLayout(layout)

    def accept(self):
        self.train_classification_model()
        super().accept()

    def get_parameters(self):

        # Extract values from dialog widgets
        params = {
            'project': self.project_edit.text(),
            'name': self.name_edit.text(),
            'model': self.model_edit.text(),
            'data': self.dataset_dir_edit.text(),
            'epochs': self.epochs_spinbox.value(),
            'patience': self.patience_spinbox.value(),
            'batch': self.batch_spinbox.value(),
            'imgsz': self.imgsz_spinbox.value(),
            'save': self.save_checkbox.isChecked(),
            'save_period': self.save_period_spinbox.value(),
            'workers': self.workers_spinbox.value(),
            'pretrained': self.pretrained_checkbox.isChecked(),
            'optimizer': self.optimizer_combo.currentText(),
            'verbose': self.verbose_checkbox.isChecked(),
            'fraction': self.fraction_spinbox.value(),
            'freeze': self.freeze_edit.text(),
            'lr0': self.lr0_spinbox.value(),
            'weighted': self.weighted_checkbox.isChecked(),
            'dropout': self.dropout_spinbox.value(),
            'val': self.val_checkbox.isChecked(),
            'exist_ok': True,
            'plots': True,
        }
        # Default project folder
        project = 'Data/Training'
        params['project'] = params['project'] if params['project'] else project
        # Default project name
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d_%H-%M-%S")
        params['name'] = params['name'] if params['name'] else now
        # Provided model path, else use default model
        params['model'] = params['model'] if params['model'] else self.classification_model_combo.currentText()

        # Add custom parameters
        for param_name, param_value in self.custom_params:
            name = param_name.text().strip()
            value = param_value.text().strip().lower()
            if name:
                if value == 'true':
                    params[name] = True
                elif value == 'false':
                    params[name] = False
                else:
                    try:
                        params[name] = int(value)
                    except ValueError:
                        try:
                            params[name] = float(value)
                        except ValueError:
                            params[name] = value

        # Return the dictionary of parameters
        return params

    def train_classification_model(self):

        # Get training parameters
        self.params = self.get_parameters()
        # Create and start the worker thread
        self.worker = TrainModelWorker(self.params, self.main_window.device, self.class_mapping)
        self.worker.training_started.connect(self.on_training_started)
        self.worker.training_completed.connect(self.on_training_completed)
        self.worker.training_error.connect(self.on_training_error)
        self.worker.start()

    def on_training_started(self):
        # Save the class mapping JSON file
        output_dir_path = os.path.join(self.params['project'], self.params['name'])
        os.makedirs(output_dir_path, exist_ok=True)
        if os.path.exists(self.class_mapping_edit.text()):
            shutil.copyfile(self.class_mapping_edit.text(), f"{output_dir_path}\\class_mapping.json")

        message = "Model training has commenced.\nMonitor the console for real-time progress."
        QMessageBox.information(self, "Model Training Status", message)

    def on_training_completed(self):
        message = "Model training has successfully been completed."
        QMessageBox.information(self, "Model Training Status", message)

    def on_training_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        print(error_message)


class ConfusionMatrixMetrics:
    """
    A class for calculating TP, FP, TN, FN, precision, recall, accuracy,
    and per-class accuracy from a confusion matrix.

    Attributes:
        matrix (np.ndarray): The confusion matrix.
        num_classes (int): The number of classes.
    """

    def __init__(self, matrix, class_mapping):
        """
        Initialize the ConfusionMatrixMetrics with a given confusion matrix.

        Args:
            matrix (np.ndarray): The confusion matrix.
        """
        self.matrix = matrix
        self.num_classes = matrix.shape[0]
        self.class_mapping = class_mapping

    def calculate_tp(self):
        """
        Calculate true positives for each class.

        Returns:
            np.ndarray: An array of true positives for each class.
        """
        return np.diagonal(self.matrix)

    def calculate_fp(self):
        """
        Calculate false positives for each class.

        Returns:
            np.ndarray: An array of false positives for each class.
        """
        return self.matrix.sum(axis=0) - np.diagonal(self.matrix)

    def calculate_fn(self):
        """
        Calculate false negatives for each class.

        Returns:
            np.ndarray: An array of false negatives for each class.
        """
        return self.matrix.sum(axis=1) - np.diagonal(self.matrix)

    def calculate_tn(self):
        """
        Calculate true negatives for each class.

        Returns:
            np.ndarray: An array of true negatives for each class.
        """
        total = self.matrix.sum()
        tp = self.calculate_tp()
        fp = self.calculate_fp()
        fn = self.calculate_fn()
        return total - (tp + fp + fn)

    def calculate_precision(self):
        """
        Calculate precision for each class.

        Returns:
            np.ndarray: An array of precision values for each class.
        """
        tp = self.calculate_tp()
        fp = self.calculate_fp()
        return tp / (tp + fp + 1e-16)  # avoid division by zero

    def calculate_recall(self):
        """
        Calculate recall for each class.

        Returns:
            np.ndarray: An array of recall values for each class.
        """
        tp = self.calculate_tp()
        fn = self.calculate_fn()
        return tp / (tp + fn + 1e-16)  # avoid division by zero

    def calculate_accuracy(self):
        """
        Calculate accuracy for all classes combined.

        Returns:
            float: The accuracy value.
        """
        tp = self.calculate_tp().sum()
        total = self.matrix.sum()
        return tp / total

    def calculate_per_class_accuracy(self):
        """
        Calculate per-class accuracy.

        Returns:
            np.ndarray: An array of accuracy values for each class.
        """
        tp = self.calculate_tp()
        total_per_class = self.matrix.sum(axis=1)
        return tp / (total_per_class + 1e-16)  # avoid division by zero

    def get_metrics_all(self):
        """
        Get all metrics (TP, FP, TN, FN, precision, recall, accuracy) for all classes combined.

        Returns:
            dict: A dictionary containing all calculated metrics for all classes combined.
        """
        tp = self.calculate_tp().sum()
        fp = self.calculate_fp().sum()
        tn = self.calculate_tn().sum()
        fn = self.calculate_fn().sum()
        precision = tp / (tp + fp + 1e-16)  # avoid division by zero
        recall = tp / (tp + fn + 1e-16)  # avoid division by zero
        accuracy = self.calculate_accuracy()

        return {
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'Accuracy': accuracy
        }

    def get_metrics_per_class(self):
        """
        Get all metrics (TP, FP, TN, FN, precision, recall, accuracy)
        per class in a dictionary.

        Returns:
            dict: A dictionary containing all calculated metrics per class.
        """
        tp = self.calculate_tp()
        fp = self.calculate_fp()
        tn = self.calculate_tn()
        fn = self.calculate_fn()
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        accuracy = self.calculate_per_class_accuracy()

        metrics_per_class = {}
        for i in range(self.num_classes):
            class_name = list(self.class_mapping.keys())[i]
            metrics_per_class[f'{class_name}'] = {
                'TP': tp[i],
                'FP': fp[i],
                'TN': tn[i],
                'FN': fn[i],
                'Precision': precision[i],
                'Recall': recall[i],
                'Accuracy': accuracy[i]
            }

        return metrics_per_class

    def save_metrics_to_json(self, directory, filename="metrics.json"):
        """
        Save the metrics to a JSON file.

        Args:
            directory (str): The directory where the JSON file will be saved.
            filename (str): The name of the JSON file. Default is "metrics.json".
        """
        os.makedirs(directory, exist_ok=True)

        metrics_all = self.get_metrics_all()
        metrics_per_class = self.get_metrics_per_class()

        results = {
            'All Classes': metrics_all,
            'Per Class': metrics_per_class
        }

        file_path = os.path.join(directory, filename)
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)


class EvaluateModelWorker(QThread):
    evaluation_started = pyqtSignal()
    evaluation_completed = pyqtSignal()
    evaluation_error = pyqtSignal(str)

    def __init__(self, model, params, class_mapping):
        super().__init__()
        self.model = model
        self.params = params
        self.class_mapping = class_mapping

    def run(self):
        try:
            # Emit signal to indicate evaluation has started
            self.evaluation_started.emit()

            # Modify the save directory
            save_dir = self.params['save_dir']
            validator.get_save_dir = lambda x: save_dir

            # Evaluate the model
            results = self.model.val(
                data=self.params['data'],
                imgsz=self.params['imgsz'],
                split=self.params['split'],
                save_json=True,
                plots=True
            )

            # Update the class mapping with target model names (ordered)
            class_mapping = {name: self.class_mapping[name] for name in self.model.names.values()}

            # Output confusion matrix metrics as json
            metrics = ConfusionMatrixMetrics(results.confusion_matrix.matrix, class_mapping)
            metrics.save_metrics_to_json(save_dir)

            # Emit signal to indicate evaluation has completed
            self.evaluation_completed.emit()

        except Exception as e:
            self.evaluation_error.emit(str(e))


class EvaluateModelDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        # For holding parameters
        self.params = {}
        self.class_mapping = {}

        self.setWindowTitle("Evaluate Model")

        # Set window settings
        self.setWindowFlags(Qt.Window |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint |
                            Qt.WindowTitleHint)

        self.resize(400, 200)

        # Main layout
        self.main_layout = QVBoxLayout()

        # Create and set up the tabs, parameters form, and console output
        self.setup_ui()

        # Set the main layout as the layout of the dialog
        self.setLayout(self.main_layout)

    def setup_ui(self):
        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Details on different evaluation settings can be found "
                            "<a href='https://docs.ultralytics.com/modes/val/#arguments-for-yolo-model-validation"
                            "'>here</a>.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        self.main_layout.addWidget(info_label)

        # Parameters Form
        self.form_layout = QFormLayout()

        # Existing Model
        self.model_edit = QLineEdit()
        self.model_button = QPushButton("Browse...")
        self.model_button.clicked.connect(self.browse_model_file)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(self.model_button)
        self.form_layout.addRow("Existing Model:", model_layout)

        # Class Mapping
        self.class_mapping_edit = QLineEdit()
        self.class_mapping_button = QPushButton("Browse...")
        self.class_mapping_button.clicked.connect(self.browse_class_mapping_file)
        class_mapping_layout = QHBoxLayout()
        class_mapping_layout.addWidget(self.class_mapping_edit)
        class_mapping_layout.addWidget(self.class_mapping_button)
        self.form_layout.addRow("Class Mapping:", class_mapping_layout)

        # Dataset Directory
        self.dataset_dir_edit = QLineEdit()
        self.dataset_dir_button = QPushButton("Browse...")
        self.dataset_dir_button.clicked.connect(self.browse_dataset_dir)
        dataset_dir_layout = QHBoxLayout()
        dataset_dir_layout.addWidget(self.dataset_dir_edit)
        dataset_dir_layout.addWidget(self.dataset_dir_button)
        self.form_layout.addRow("Dataset Directory:", dataset_dir_layout)

        # Split
        self.split_combo = QComboBox()
        self.split_combo.addItems(["train", "val", "test"])
        self.split_combo.setCurrentText("test")
        self.form_layout.addRow("Split:", self.split_combo)

        # Save Directory
        self.save_dir_edit = QLineEdit()
        self.save_dir_button = QPushButton("Browse...")
        self.save_dir_button.clicked.connect(self.browse_save_dir)
        save_dir_layout = QHBoxLayout()
        save_dir_layout.addWidget(self.save_dir_edit)
        save_dir_layout.addWidget(self.save_dir_button)
        self.form_layout.addRow("Save Directory:", save_dir_layout)

        # Name
        self.name_edit = QLineEdit()
        self.form_layout.addRow("Name:", self.name_edit)

        # Imgsz
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setMinimum(16)
        self.imgsz_spinbox.setMaximum(4096)
        self.imgsz_spinbox.setValue(256)
        self.form_layout.addRow("Image Size:", self.imgsz_spinbox)

        self.main_layout.addLayout(self.form_layout)

        # Add OK and Cancel buttons
        self.buttons = QPushButton("OK")
        self.buttons.clicked.connect(self.accept)
        self.main_layout.addWidget(self.buttons)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.main_layout.addWidget(self.cancel_button)

    def browse_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File")
        if file_path:
            self.model_edit.setText(file_path)
            # Get the directory two above file path
            dir_path = os.path.dirname(os.path.dirname(file_path))
            class_mapping_path = f"{dir_path}/class_mapping.json"
            if os.path.exists(class_mapping_path):
                self.class_mapping_edit.setText(class_mapping_path)
                self.class_mapping = json.load(open(class_mapping_path, 'r'))

    def browse_class_mapping_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Select Class Mapping File",
                                                   "",
                                                   "JSON Files (*.json)")
        if file_path:
            self.class_mapping_edit.setText(file_path)
            self.class_mapping = json.load(open(file_path, 'r'))

    def browse_dataset_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if dir_path:
            self.dataset_dir_edit.setText(dir_path)

    def browse_save_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if dir_path:
            self.save_dir_edit.setText(dir_path)

    def accept(self):
        if not self.model_edit.text():
            QMessageBox.critical(self, "Error", "Existing Model field cannot be empty.")
            return
        self.evaluate_model()
        super().accept()

    def get_evaluation_parameters(self):
        # Extract values from dialog widgets
        params = {
            'name': self.name_edit.text(),
            'model': self.model_edit.text(),
            'data': self.dataset_dir_edit.text(),
            'split': self.split_combo.currentText(),
            'imgsz': int(self.imgsz_spinbox.value()),
            'verbose': True,
            'exist_ok': True,
            'plots': True,
        }

        # Default project name
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d_%H-%M-%S")
        params['name'] = params['name'] if params['name'] else now

        save_dir = self.save_dir_edit.text()
        save_dir = Path(save_dir) / params['name']
        params['save_dir'] = save_dir

        # Return the dictionary of parameters
        return params

    def evaluate_model(self):
        # Get evaluation parameters
        self.params = self.get_evaluation_parameters()

        try:
            # Initialize the model, evaluate, and save the results
            self.model = YOLO(self.params['model'])

            # Create and start the worker thread
            self.worker = EvaluateModelWorker(self.model, self.params, self.class_mapping)
            self.worker.evaluation_started.connect(self.on_evaluation_started)
            self.worker.evaluation_completed.connect(self.on_evaluation_completed)
            self.worker.evaluation_error.connect(self.on_evaluation_error)
            self.worker.start()

            # Empty cache
            del self.model
            gc.collect()
            empty_cache()
        except Exception as e:
            error_message = f"An error occurred when evaluating model: {e}"
            QMessageBox.critical(self, "Error", error_message)
            print(error_message)

    def on_evaluation_started(self):
        message = "Model evaluation has commenced.\nMonitor the console for real-time progress."
        QMessageBox.information(self, "Model Evaluation Status", message)

    def on_evaluation_completed(self):
        message = "Model evaluation has successfully been completed."
        QMessageBox.information(self, "Model Evaluation Status", message)

    def on_evaluation_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        print(error_message)


class OptimizeModelDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        self.custom_params = []

        self.setWindowTitle("Optimize Model")
        self.resize(300, 200)

        self.layout = QVBoxLayout(self)

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Details on different production formats can be found "
                            "<a href='https://docs.ultralytics.com/modes/export/#export-formats'>here</a>.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        self.layout.addWidget(info_label)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_file)
        self.layout.addWidget(browse_button)

        self.model_text_area = QTextEdit("No model file selected")
        self.model_text_area.setReadOnly(True)
        self.layout.addWidget(self.model_text_area)

        # Export Format Dropdown
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["torchscript",
                                           "onnx",
                                           "openvino",
                                           "engine"])

        self.export_format_combo.setEditable(True)
        self.layout.addWidget(QLabel("Select or Enter Export Format:"))
        self.layout.addWidget(self.export_format_combo)

        # Parameters Form
        self.form_layout = QFormLayout()

        # Add custom parameters section
        self.custom_params_layout = QVBoxLayout()
        self.form_layout.addRow("Parameters:", self.custom_params_layout)

        # Add button for new parameter pairs
        self.add_param_button = QPushButton("Add Parameter")
        self.add_param_button.clicked.connect(self.add_parameter_pair)
        self.form_layout.addRow("", self.add_param_button)

        self.layout.addLayout(self.form_layout)

        accept_button = QPushButton("Accept")
        accept_button.clicked.connect(self.optimize_model)
        self.layout.addWidget(accept_button)

        self.setLayout(self.layout)

    def add_parameter_pair(self):
        param_layout = QHBoxLayout()
        param_name = QLineEdit()
        param_value = QLineEdit()
        param_layout.addWidget(param_name)
        param_layout.addWidget(param_value)

        self.custom_params.append((param_name, param_value))
        self.custom_params_layout.addLayout(param_layout)

    def browse_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Open Model File", "",
                                                   "Model Files (*.pt)", options=options)
        if file_path:
            self.model_path = file_path
            self.model_text_area.setText("Model file selected")

    def accept(self):
        self.optimize_model()
        super().accept()

    def get_optimization_parameters(self):
        # Extract values from dialog widgets
        params = {'format': self.export_format_combo.currentText()}

        for param_name, param_value in self.custom_params:
            name = param_name.text().strip()
            value = param_value.text().strip().lower()
            if name:
                if value == 'true':
                    params[name] = True
                elif value == 'false':
                    params[name] = False
                else:
                    try:
                        params[name] = int(value)
                    except ValueError:
                        try:
                            params[name] = float(value)
                        except ValueError:
                            params[name] = value

        # Return the dictionary of parameters
        return params

    def optimize_model(self):

        # Get training parameters
        params = self.get_optimization_parameters()

        # Set the cursor to waiting (busy) cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # Initialize the model, export given params
            YOLO(self.model_path).export(**params)

            message = "Model export successful."
            QMessageBox.information(self, "Model Export Status", message)

        except Exception as e:
            # Display an error message box to the user
            error_message = f"An error occurred when converting model: {e}"
            QMessageBox.critical(self, "Error", error_message)
            print(error_message)

        # Restore the cursor to the default cursor
        QApplication.restoreOverrideCursor()


class DeployModelDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

        self.setWindowTitle("Deploy Model")
        self.resize(300, 200)

        self.layout = QVBoxLayout(self)

        self.model_path = None
        self.loaded_model = None
        self.class_mapping = {}

        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)

        self.classification_tab = QWidget()
        self.segmentation_tab = QWidget()

        self.tab_widget.addTab(self.classification_tab, "Image Classification")
        self.tab_widget.addTab(self.segmentation_tab, "Instance Segmentation")

        self.init_classification_tab()
        self.init_segmentation_tab()

        # Status bar label
        self.status_bar = QLabel("No model loaded")
        self.layout.addWidget(self.status_bar)

        self.setLayout(self.layout)

    def showEvent(self, event: QShowEvent):
        super().showEvent(event)
        self.check_and_display_class_names()

    def init_classification_tab(self):
        layout = QVBoxLayout()

        self.classification_text_area = QTextEdit()
        self.classification_text_area.setReadOnly(True)
        layout.addWidget(self.classification_text_area)

        browse_model_button = QPushButton("Browse Model")
        browse_model_button.clicked.connect(self.browse_file)
        layout.addWidget(browse_model_button)

        browse_class_mapping_button = QPushButton("Browse Class Mapping")
        browse_class_mapping_button.clicked.connect(self.browse_class_mapping_file)
        layout.addWidget(browse_class_mapping_button)

        load_button = QPushButton("Load Model")
        load_button.clicked.connect(self.load_model)
        layout.addWidget(load_button)

        deactivate_button = QPushButton("Deactivate Model")
        deactivate_button.clicked.connect(self.deactivate_model)
        layout.addWidget(deactivate_button)

        self.classification_tab.setLayout(layout)

    def init_segmentation_tab(self):
        layout = QVBoxLayout()

        self.segmentation_text_area = QTextEdit()
        self.segmentation_text_area.setReadOnly(True)
        layout.addWidget(self.segmentation_text_area)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_file)
        layout.addWidget(browse_button)

        load_button = QPushButton("Load Model")
        load_button.clicked.connect(self.load_model)
        layout.addWidget(load_button)

        deactivate_button = QPushButton("Deactivate Model")
        deactivate_button.clicked.connect(self.deactivate_model)
        layout.addWidget(deactivate_button)

        self.segmentation_tab.setLayout(layout)

    def browse_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Open Model File", "",
                                                   "Model Files (*.pt *.onnx *.torchscript *.engine *.bin)",
                                                   options=options)
        if file_path:
            if ".bin" in file_path:
                # OpenVINO is a directory
                file_path = os.path.dirname(file_path)

            self.model_path = file_path
            if self.tab_widget.currentIndex() == 0:
                self.classification_text_area.setText("Model file selected")
            else:
                self.segmentation_file_path.setText("Model file selected")

            # Try to load the class mapping file if it exists in the directory above
            parent_dir = os.path.dirname(os.path.dirname(file_path))
            class_mapping_path = os.path.join(parent_dir, "class_mapping.json")
            if os.path.exists(class_mapping_path):
                self.load_class_mapping(class_mapping_path)

    def browse_class_mapping_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Open Class Mapping File", "",
                                                   "JSON Files (*.json)",
                                                   options=options)
        if file_path:
            self.load_class_mapping(file_path)

    def load_class_mapping(self, file_path):
        try:
            with open(file_path, 'r') as f:
                self.class_mapping = json.load(f)
            self.classification_text_area.append("Class mapping file selected")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load class mapping file: {str(e)}")

    def load_model(self):
        if self.model_path:
            try:
                # Set the cursor to waiting (busy) cursor
                QApplication.setOverrideCursor(Qt.WaitCursor)
                # Load the model, and run a dummy prediction to build the graph
                self.loaded_model = YOLO(self.model_path, task='classify')
                self.loaded_model(np.zeros((224, 224, 3), dtype=np.uint8))

                try:
                    # Add the labels to the LabelWindow if they don't already exist
                    self.add_labels_to_label_window()
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to add labels: {str(e)}")

                QMessageBox.information(self, "Model Loaded", "Model weights loaded successfully.")

                # Check and display class names
                self.check_and_display_class_names()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            finally:
                # Restore the cursor to the default cursor
                QApplication.restoreOverrideCursor()
        else:
            QMessageBox.warning(self, "Warning", "No model file selected")

    def add_labels_to_label_window(self):
        for label in self.class_mapping.values():
            self.label_window.add_label_if_not_exists(label['short_label_code'],
                                                      label['long_label_code'],
                                                      QColor(*label['color']))

    def check_and_display_class_names(self):
        if self.loaded_model:
            # Get the class names the model can predict
            class_names = list(self.loaded_model.names.values())
            class_names_str = "Class Names: \n"
            missing_labels = []

            for class_name in class_names:
                label = self.label_window.get_label_by_short_code(class_name)
                if label:
                    class_names_str += f"✅ {label.short_label_code}: {label.long_label_code} \n"
                else:
                    class_names_str += f"❌ {class_name} \n"
                    missing_labels.append(class_name)

            self.classification_text_area.setText(class_names_str)
            self.status_bar.setText(f"Model loaded: {os.path.basename(self.model_path)}")

            if missing_labels:
                missing_labels_str = "\n".join(missing_labels)
                QMessageBox.warning(self,
                                    "Warning",
                                    f"The following short labels are missing and "
                                    f"cannot be predicted until added manually:"
                                    f"\n{missing_labels_str}")

    def deactivate_model(self):
        self.loaded_model = None
        self.model_path = None
        self.class_mapping = {}
        gc.collect()
        empty_cache()
        self.status_bar.setText("No model loaded")
        if self.tab_widget.currentIndex() == 0:
            self.classification_text_area.setText("No model file selected")
        else:
            self.segmentation_file_path.setText("No model file selected")

    def predict(self, annotations=None):
        if self.loaded_model is None:
            return

        # Set the cursor to waiting (busy) cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)
        # Get the selected annotation
        selected_annotation = self.annotation_window.selected_annotation
        if selected_annotation and not annotations:
            # Make predictions on a single, specific annotation
            self.predict_annotation(selected_annotation)
            self.main_window.annotation_window.unselect_annotation()
            self.main_window.annotation_window.select_annotation(selected_annotation)
        else:
            if not annotations:
                # Get the annotations for the image with Review label
                annotations = self.annotation_window.get_image_review_annotations()

            # Make predictions on the annotations
            self.preprocess_annotations(annotations)

        # Restore the cursor to the default cursor
        QApplication.restoreOverrideCursor()

        # Clear cache
        gc.collect()
        empty_cache()

    def preprocess_annotations(self, annotations):

        # Convert QImages to numpy arrays
        images_np = []
        for annotation in annotations:
            images_np.append(pixmap_to_numpy(annotation.cropped_image))

        # Make predictions on annotations
        progress_bar = ProgressBar(self, title=f"Making Predictions")
        progress_bar.show()
        progress_bar.start_progress(len(annotations))

        # Perform batch prediction
        results = self.loaded_model(images_np, stream=True, device=self.main_window.device)

        for annotation, result in zip(annotations, results):
            # Process the results
            self.process_prediction_result(annotation, result)
            progress_bar.update_progress()

        # Show the last annotation in the confidence window (aesthetic)
        self.main_window.confidence_window.display_cropped_image(annotation)

        # Group annotations by image path
        image_paths = list(set([annotation.image_path for annotation in annotations]))
        for image_path in image_paths:
            # Update the image window's image dict (fast search filtering)
            self.main_window.image_window.update_image_annotations(image_path)

        progress_bar.stop_progress()
        progress_bar.close()

    def predict_annotation(self, annotation):
        # Get the cropped image
        image = annotation.cropped_image
        # Convert QImage to np
        image_np = pixmap_to_numpy(image)
        # Perform prediction
        result = self.loaded_model(image_np, device=self.main_window.device)[0]
        # Process the results
        self.process_prediction_result(annotation, result)

    def process_prediction_result(self, annotation, result):
        # Extract the results
        class_names = result.names
        top5 = result.probs.top5
        top5conf = result.probs.top5conf
        top1conf = top5conf[0].item()

        # Initialize an empty dictionary to store the results
        predictions = {}

        # Iterate over the top 5 predictions
        for idx, conf in zip(top5, top5conf):
            class_name = class_names[idx]
            label = self.label_window.get_label_by_short_code(class_name)

            if label:
                predictions[label] = float(conf)
            else:
                # User does not have label loaded; skip.
                pass

        if predictions:
            # Update the machine confidence
            annotation.update_machine_confidence(predictions)
            # If the top prediction is below the threshold, label as Review
            if top1conf < self.main_window.get_uncertainty_thresh():
                label = self.label_window.get_label_by_id('-1')
                annotation.update_label(label)


class BatchInferenceDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window
        self.deploy_model_dialog = main_window.deploy_model_dialog

        self.loaded_model = self.deploy_model_dialog.loaded_model

        self.annotations = []
        self.processed_annotations = []
        self.image_paths = []

        self.setWindowTitle("Batch Inference")
        self.resize(400, 100)

        self.layout = QVBoxLayout(self)

        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)

        self.classification_tab = QWidget()
        self.segmentation_tab = QWidget()

        self.tab_widget.addTab(self.classification_tab, "Image Classification")
        self.tab_widget.addTab(self.segmentation_tab, "Instance Segmentation")

        # Make the segmentation tab unclickable
        self.segmentation_tab.setEnabled(False)

        # Initialize the tabs
        self.init_classification_tab()
        self.init_segmentation_tab()

        # Set the threshold slider for uncertainty
        self.uncertainty_threshold_slider = QSlider(Qt.Horizontal)
        self.uncertainty_threshold_slider.setRange(0, 100)
        self.uncertainty_threshold_slider.setValue(int(self.main_window.get_uncertainty_thresh() * 100))
        self.uncertainty_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.uncertainty_threshold_slider.setTickInterval(10)
        self.uncertainty_threshold_slider.valueChanged.connect(self.update_uncertainty_label)

        self.uncertainty_threshold_label = QLabel(f"{self.main_window.get_uncertainty_thresh():.2f}")
        self.layout.addWidget(QLabel("Uncertainty Threshold"))
        self.layout.addWidget(self.uncertainty_threshold_slider)
        self.layout.addWidget(self.uncertainty_threshold_label)

        # Add the "Okay" and "Cancel" buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.on_ok_clicked)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

        self.setLayout(self.layout)

        # Connect to the shared data signal
        self.main_window.uncertaintyChanged.connect(self.on_uncertainty_changed)

    def update_uncertainty_label(self):
        # Convert the slider value to a ratio (0-1)
        value = self.uncertainty_threshold_slider.value() / 100.0
        self.main_window.update_uncertainty_thresh(value)

    def on_uncertainty_changed(self, value):
        # Update the slider and label when the shared data changes
        self.uncertainty_threshold_slider.setValue(int(value * 100))
        self.uncertainty_threshold_label.setText(f"{value:.2f}")

    def init_segmentation_tab(self):
        pass

    def init_classification_tab(self):
        layout = QVBoxLayout()

        # Create a group box for annotation options
        annotation_group_box = QGroupBox("Annotation Options")
        annotation_layout = QVBoxLayout()

        # Create a button group for the annotation checkboxes
        self.annotation_options_group = QButtonGroup(self)

        self.classification_review_checkbox = QCheckBox("Predict Review Annotation")
        self.classification_all_checkbox = QCheckBox("Predict All Annotations")

        # Add the checkboxes to the button group
        self.annotation_options_group.addButton(self.classification_review_checkbox)
        self.annotation_options_group.addButton(self.classification_all_checkbox)

        # Ensure only one checkbox can be checked at a time
        self.annotation_options_group.setExclusive(True)

        # Set the default checkbox
        self.classification_review_checkbox.setChecked(True)

        annotation_layout.addWidget(self.classification_review_checkbox)
        annotation_layout.addWidget(self.classification_all_checkbox)
        annotation_group_box.setLayout(annotation_layout)

        layout.addWidget(annotation_group_box)

        # Create a group box for image options
        image_group_box = QGroupBox("Image Options")
        image_layout = QVBoxLayout()

        # Create a button group for the image checkboxes
        self.image_options_group = QButtonGroup(self)

        self.apply_filtered_checkbox = QCheckBox("Apply to filtered images")
        self.apply_prev_checkbox = QCheckBox("Apply to previous images")
        self.apply_next_checkbox = QCheckBox("Apply to next images")
        self.apply_all_checkbox = QCheckBox("Apply to all images")

        # Add the checkboxes to the button group
        self.image_options_group.addButton(self.apply_filtered_checkbox)
        self.image_options_group.addButton(self.apply_prev_checkbox)
        self.image_options_group.addButton(self.apply_next_checkbox)
        self.image_options_group.addButton(self.apply_all_checkbox)

        # Ensure only one checkbox can be checked at a time
        self.image_options_group.setExclusive(True)

        # Set the default checkbox
        self.apply_all_checkbox.setChecked(True)

        image_layout.addWidget(self.apply_filtered_checkbox)
        image_layout.addWidget(self.apply_prev_checkbox)
        image_layout.addWidget(self.apply_next_checkbox)
        image_layout.addWidget(self.apply_all_checkbox)
        image_group_box.setLayout(image_layout)

        layout.addWidget(image_group_box)

        self.classification_tab.setLayout(layout)

    def on_ok_clicked(self):
        if self.classification_all_checkbox.isChecked():
            reply = QMessageBox.warning(self,
                                        "Warning",
                                        "This will overwrite the existing labels. Are you sure?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return  # Do not accept the dialog if the user clicks "No"

        self.apply()
        self.accept()  # Close the dialog after applying the changes

    def apply(self):
        # Pause the cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # Get the Review Annotations
            if self.classification_review_checkbox.isChecked():
                for image_path in self.get_selected_image_paths():
                    self.annotations.extend(self.annotation_window.get_image_review_annotations(image_path))
            else:
                # Get all the annotations
                for image_path in self.get_selected_image_paths():
                    self.annotations.extend(self.annotation_window.get_image_annotations(image_path))

            # Crop them, if not already cropped
            self.preprocess_annotations()
            self.batch_inference()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to make predictions: {str(e)}")
        finally:
            self.annotations = []
            self.processed_annotations = []
            self.image_paths = []

        # Resume the cursor
        QApplication.restoreOverrideCursor()

    def get_selected_image_paths(self):
        if self.apply_filtered_checkbox.isChecked():
            return self.image_window.filtered_image_paths
        elif self.apply_prev_checkbox.isChecked():
            current_image_index = self.image_window.image_paths.index(self.annotation_window.current_image_path)
            return self.image_window.image_paths[:current_image_index + 1]
        elif self.apply_next_checkbox.isChecked():
            current_image_index = self.image_window.image_paths.index(self.annotation_window.current_image_path)
            return self.image_window.image_paths[current_image_index:]
        else:
            return self.image_window.image_paths

    def preprocess_annotations(self):
        # Get unique image paths
        self.image_paths = list(set(a.image_path for a in self.annotations))
        if not self.image_paths:
            return

        progress_bar = ProgressBar(self, title=f"Cropping Annotations")
        progress_bar.show()
        progress_bar.start_progress(len(self.image_paths))

        def crop(image_path, image_annotations):
            # Crop the image based on the annotations
            return self.annotation_window.crop_these_image_annotations(image_path, image_annotations)

        # Group annotations by image path
        groups = groupby(sorted(self.annotations, key=attrgetter('image_path')), key=attrgetter('image_path'))

        with ThreadPoolExecutor() as executor:
            future_to_image = {}
            for path, group in groups:
                future = executor.submit(crop, path, list(group))
                future_to_image[future] = path

            for future in as_completed(future_to_image):
                image_path = future_to_image[future]
                try:
                    self.processed_annotations.extend(future.result())
                except Exception as exc:
                    print(f'{image_path} generated an exception: {exc}')
                finally:
                    progress_bar.update_progress()

        progress_bar.stop_progress()
        progress_bar.close()

    def batch_inference(self):
        # Make predictions on each image's annotations
        progress_bar = ProgressBar(self, title=f"Batch Inference")
        progress_bar.show()
        progress_bar.start_progress(len(self.image_paths))

        # Group annotations by image path
        groups = groupby(sorted(self.processed_annotations, key=attrgetter('image_path')), key=attrgetter('image_path'))
        # Make predictions on each image's annotations
        for path, group in groups:
            self.deploy_model_dialog.predict(annotations=list(group))
            progress_bar.update_progress()

        progress_bar.stop_progress()
        progress_bar.close()