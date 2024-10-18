import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import yaml
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QWidget, QVBoxLayout,
                             QLabel, QLineEdit, QDialog, QHBoxLayout, QPushButton, QTabWidget, QDialogButtonBox)

from toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MergeDatasetsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Merge Datasets")
        self.resize(400, 200)

        self.layout = QVBoxLayout(self)

        # Create tabs
        self.tabs = QTabWidget()
        self.tab_classification = QWidget()
        self.tab_detection = QWidget()
        self.tab_segmentation = QWidget()

        self.tabs.addTab(self.tab_classification, "Image Classification")
        self.tabs.addTab(self.tab_detection, "Object Detection")
        self.tabs.addTab(self.tab_segmentation, "Instance Segmentation")

        self.layout.addWidget(self.tabs)

        # Setup tabs
        self.setup_tab(self.tab_classification, "classification")
        self.setup_tab(self.tab_detection, "detection")
        self.setup_tab(self.tab_segmentation, "segmentation")

        # OK and Cancel Buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

        # Track valid directories and their class mappings
        self.valid_directories = []

    def setup_tab(self, tab, tab_type):
        layout = QVBoxLayout()

        # Dataset Name
        dataset_name_edit = QLineEdit()
        dataset_name_edit.setObjectName("dataset_name_edit")  # Set object name
        layout.addWidget(QLabel("Dataset Name:"))
        layout.addWidget(dataset_name_edit)

        # Output Directory Chooser
        output_dir_edit = QLineEdit()
        output_dir_edit.setObjectName("output_dir_edit")  # Set object name
        output_dir_button = QPushButton("Browse...")
        output_dir_button.clicked.connect(lambda: self.browse_output_directory(output_dir_edit))
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(output_dir_edit)
        output_dir_layout.addWidget(output_dir_button)
        layout.addWidget(QLabel("Output Directory:"))
        layout.addLayout(output_dir_layout)

        # Existing Dataset Directories
        existing_dirs_layout = QVBoxLayout()
        layout.addWidget(QLabel("Existing Dataset Directories:"))
        layout.addLayout(existing_dirs_layout)

        # Add two default choosers
        self.add_chooser(existing_dirs_layout, tab, tab_type)
        self.add_chooser(existing_dirs_layout, tab, tab_type)

        # Add Directory Button
        add_dir_button = QPushButton("Add Dataset")
        add_dir_button.clicked.connect(lambda: self.add_chooser(existing_dirs_layout, tab, tab_type))
        layout.addWidget(add_dir_button)

        tab.setLayout(layout)

    def add_chooser(self, existing_dirs_layout, tab, tab_type):
        chooser = QWidget()
        chooser_layout = QHBoxLayout(chooser)

        status_label = QLabel()
        chooser_layout.addWidget(status_label)

        if tab_type == "classification":
            dir_edit = QLineEdit()
            dir_button = QPushButton("Browse Directory...")
            dir_button.clicked.connect(lambda: self.browse_existing_directory(dir_edit, status_label, tab))
            chooser_layout.addWidget(dir_edit)
            chooser_layout.addWidget(dir_button)

            class_mapping_edit = QLineEdit()
            class_mapping_edit.setObjectName("class_mapping_edit")  # Set object name
            class_mapping_button = QPushButton("Select Class Mapping")
            class_mapping_button.clicked.connect(
                lambda: self.browse_class_mapping(class_mapping_edit, status_label, tab))
            chooser_layout.addWidget(class_mapping_edit)
            chooser_layout.addWidget(class_mapping_button)
        else:
            yaml_edit = QLineEdit()
            yaml_button = QPushButton("Select YAML")
            yaml_button.clicked.connect(lambda: self.browse_data_yaml(yaml_edit, status_label, tab))
            chooser_layout.addWidget(yaml_edit)
            chooser_layout.addWidget(yaml_button)

            class_mapping_edit = QLineEdit()
            class_mapping_edit.setObjectName("class_mapping_edit")  # Set object name
            class_mapping_button = QPushButton("Select Class Mapping")
            class_mapping_button.clicked.connect(
                lambda: self.browse_class_mapping(class_mapping_edit, status_label, tab))
            chooser_layout.addWidget(class_mapping_edit)
            chooser_layout.addWidget(class_mapping_button)

        existing_dirs_layout.addWidget(chooser)

    def browse_output_directory(self, output_dir_edit):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            output_dir_edit.setText(dir_path)

    def browse_existing_directory(self, dir_edit, status_label, tab):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Existing Dataset Directory")
        if dir_path:
            dir_edit.setText(dir_path)
            self.validate_directory(dir_path, status_label, tab)

            # Auto-fill class_mapping.json if it exists in the same directory
            class_mapping_path = os.path.join(dir_path, "class_mapping.json")
            if os.path.exists(class_mapping_path):
                class_mapping_edit = dir_edit.parent().findChild(QLineEdit, "class_mapping_edit")
                if class_mapping_edit:
                    class_mapping_edit.setText(class_mapping_path)
                    self.validate_class_mapping(class_mapping_path, status_label, tab)

    def browse_data_yaml(self, yaml_edit, status_label, tab):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select data.yaml", "", "YAML Files (*.yaml);;All Files (*)", options=options
        )
        if file_path:
            yaml_edit.setText(file_path)
            self.validate_yaml(file_path, status_label, tab)

            # Auto-fill class_mapping.json if it exists in the same directory
            yaml_dir = os.path.dirname(file_path)
            class_mapping_path = os.path.join(yaml_dir, "class_mapping.json")
            if os.path.exists(class_mapping_path):
                class_mapping_edit = yaml_edit.parent().findChild(QLineEdit, "class_mapping_edit")
                if class_mapping_edit:
                    class_mapping_edit.setText(class_mapping_path)
                    self.validate_class_mapping(class_mapping_path, status_label, tab)

    def browse_class_mapping(self, class_mapping_edit, status_label, tab):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select class_mapping.json", "", "JSON Files (*.json);;All Files (*)", options=options
        )
        if file_path:
            class_mapping_edit.setText(file_path)
            self.validate_class_mapping(file_path, status_label, tab)

    def validate_directory(self, dir_path, status_label, tab):
        if os.path.exists(dir_path):
            status_label.setText("✅")
            self.valid_directories.append((dir_path, None, tab))
        else:
            status_label.setText("❌")
            self.valid_directories = [(d, c, t) for d, c, t in self.valid_directories if d != dir_path]

    def validate_yaml(self, yaml_path, status_label, tab):
        try:
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)
                if 'names' in data and 'nc' in data:
                    status_label.setText("✅")
                    self.valid_directories.append((yaml_path, None, tab))
                else:
                    status_label.setText("❌")
                    self.valid_directories = [(d, c, t) for d, c, t in self.valid_directories if d != yaml_path]
        except Exception as e:
            status_label.setText("❌")
            self.valid_directories = [(d, c, t) for d, c, t in self.valid_directories if d != yaml_path]

    def validate_class_mapping(self, class_mapping_path, status_label, tab):
        try:
            with open(class_mapping_path, 'r') as file:
                data = json.load(file)
                if isinstance(data, dict):
                    status_label.setText("✅")
                    for d, c, t in self.valid_directories:
                        if t == tab and c is None:
                            self.valid_directories.remove((d, c, t))
                            self.valid_directories.append((d, class_mapping_path, t))
                            break
                else:
                    status_label.setText("❌")
        except Exception as e:
            status_label.setText("❌")

    def merge_datasets(self):
        current_tab = self.tabs.currentWidget()
        if current_tab not in [self.tab_classification]:
            QMessageBox.warning(self, "Warning", "Only Image Classification merging has been implemented.")
            return

        output_dir = current_tab.findChild(QLineEdit, "output_dir_edit").text()
        if not output_dir:
            QMessageBox.warning(self, "Input Error", "Output directory must be specified.")
            return

        dataset_name = current_tab.findChild(QLineEdit, "dataset_name_edit").text()
        if not dataset_name:
            QMessageBox.warning(self, "Input Error", "Dataset name must be specified.")
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        output_dir_path = os.path.join(output_dir, dataset_name)
        os.makedirs(output_dir_path, exist_ok=True)

        merged_class_mapping = {}

        def copy_directory(src, dest):
            shutil.copytree(src, dest, dirs_exist_ok=True)

        with ThreadPoolExecutor() as executor:
            futures = []
            for dir_path, class_mapping_path, tab in self.valid_directories:

                if tab == self.tab_classification:
                    if class_mapping_path:
                        try:
                            with open(class_mapping_path, 'r') as json_file:
                                class_mapping = json.load(json_file)
                                merged_class_mapping.update(class_mapping)
                        except Exception as e:
                            print(f"Error reading class mapping: {e}")

                    for split in ['train', 'val', 'test']:
                        src_split_dir = os.path.join(dir_path, split)
                        dest_split_dir = os.path.join(output_dir_path, split)
                        if os.path.exists(src_split_dir):
                            future = executor.submit(copy_directory, src_split_dir, dest_split_dir)
                            futures.append(future)

            # Create a progress dialog
            progress_bar = ProgressBar(self, title=f"Merging Datasets")
            progress_bar.show()
            progress_bar.start_progress(len(futures))

            # Wait for all copying tasks to complete
            for i, future in enumerate(as_completed(futures)):
                future.result()
                progress_bar.update_progress()

            progress_bar.stop_progress()
            progress_bar.close()

        # Check if the merged class mapping is empty
        if merged_class_mapping:
            merged_class_mapping_path = os.path.join(output_dir_path, "class_mapping.json")
            with open(merged_class_mapping_path, 'w') as json_file:
                json.dump(merged_class_mapping, json_file, indent=4)

        QMessageBox.information(self, "Success", "Datasets merged successfully!")

        # Restore cursor
        QApplication.restoreOverrideCursor()

    def accept(self):
        self.merge_datasets()
        super().accept()
