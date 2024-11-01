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
    """
    Dialog for merging multiple datasets into a single dataset for machine learning tasks such as
    image classification, object detection, and instance segmentation.
    """

    def __init__(self, parent=None):
        """
        Initializes the MergeDatasetsDialog.

        :param parent: Parent widget, default is None.
        """
        super().__init__(parent)
        self.setWindowTitle("Merge Datasets")
        self.resize(400, 200)

        self.layout = QVBoxLayout(self)

        # Create tabs for different dataset types
        self.tabs = QTabWidget()
        self.tab_classification = QWidget()
        self.tab_detection = QWidget()
        self.tab_segmentation = QWidget()

        self.tabs.addTab(self.tab_classification, "Image Classification")
        # Uncomment the following lines to add tabs for detection and segmentation
        # self.tabs.addTab(self.tab_detection, "Object Detection")
        # self.tabs.addTab(self.tab_segmentation, "Instance Segmentation")

        self.layout.addWidget(self.tabs)

        # Setup each tab with its respective layout and widgets
        self.setup_tab(self.tab_classification, "classification")
        # Uncomment the following lines to setup detection and segmentation tabs
        # self.setup_tab(self.tab_detection, "detection")
        # self.setup_tab(self.tab_segmentation, "segmentation")

        # OK and Cancel Buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

        # List to track valid directories and their class mappings
        self.valid_directories = []

    def setup_tab(self, tab, tab_type):
        """
        Sets up the layout and widgets for a given tab.

        :param tab: The QWidget representing the tab.
        :param tab_type: The type of task (e.g., "classification").
        """
        layout = QVBoxLayout()

        # Dataset Name Input
        dataset_name_edit = QLineEdit()
        dataset_name_edit.setObjectName("dataset_name_edit")  # Set object name for access
        layout.addWidget(QLabel("Dataset Name:"))
        layout.addWidget(dataset_name_edit)

        # Output Directory Chooser
        output_dir_edit = QLineEdit()
        output_dir_edit.setObjectName("output_dir_edit")  # Set object name for access
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

        # Add two default directory choosers
        self.add_chooser(existing_dirs_layout, tab, tab_type)
        self.add_chooser(existing_dirs_layout, tab, tab_type)

        # Button to add additional dataset directories
        add_dir_button = QPushButton("Add Dataset")
        add_dir_button.clicked.connect(lambda: self.add_chooser(existing_dirs_layout, tab, tab_type))
        layout.addWidget(add_dir_button)

        tab.setLayout(layout)

    def add_chooser(self, existing_dirs_layout, tab, tab_type):
        """
        Adds a directory chooser widget to the existing directories layout.

        :param existing_dirs_layout: The QVBoxLayout to add the chooser to.
        :param tab: The current tab QWidget.
        :param tab_type: The type of task (e.g., "classification").
        """
        chooser = QWidget()
        chooser_layout = QHBoxLayout(chooser)

        # Status label to indicate validation status
        status_label = QLabel()
        chooser_layout.addWidget(status_label)

        if tab_type == "classification":
            # Directory chooser for classification tasks
            dir_edit = QLineEdit()
            dir_button = QPushButton("Browse Directory...")
            dir_button.clicked.connect(lambda: self.browse_existing_directory(dir_edit, status_label, tab))
            chooser_layout.addWidget(dir_edit)
            chooser_layout.addWidget(dir_button)

            # Class mapping chooser
            class_mapping_edit = QLineEdit()
            class_mapping_edit.setObjectName("class_mapping_edit")  # Set object name for access
            class_mapping_button = QPushButton("Select Class Mapping")
            class_mapping_button.clicked.connect(
                lambda: self.browse_class_mapping(class_mapping_edit, status_label, tab))
            chooser_layout.addWidget(class_mapping_edit)
            chooser_layout.addWidget(class_mapping_button)
        else:
            # Directory chooser for other task types (e.g., detection, segmentation)
            yaml_edit = QLineEdit()
            yaml_button = QPushButton("Select YAML")
            yaml_button.clicked.connect(lambda: self.browse_data_yaml(yaml_edit, status_label, tab))
            chooser_layout.addWidget(yaml_edit)
            chooser_layout.addWidget(yaml_button)

            # Class mapping chooser
            class_mapping_edit = QLineEdit()
            class_mapping_edit.setObjectName("class_mapping_edit")  # Set object name for access
            class_mapping_button = QPushButton("Select Class Mapping")
            class_mapping_button.clicked.connect(
                lambda: self.browse_class_mapping(class_mapping_edit, status_label, tab))
            chooser_layout.addWidget(class_mapping_edit)
            chooser_layout.addWidget(class_mapping_button)

        existing_dirs_layout.addWidget(chooser)

    def browse_output_directory(self, output_dir_edit):
        """
        Opens a dialog to select the output directory and sets the selected path.

        :param output_dir_edit: The QLineEdit widget to display the selected directory.
        """
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            output_dir_edit.setText(dir_path)

    def browse_existing_directory(self, dir_edit, status_label, tab):
        """
        Opens a dialog to select an existing dataset directory, sets the path, and validates it.

        :param dir_edit: The QLineEdit widget to display the selected directory.
        :param status_label: The QLabel to display validation status.
        :param tab: The current tab QWidget.
        """
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
        """
        Opens a dialog to select a YAML file, sets the path, and validates it.

        :param yaml_edit: The QLineEdit widget to display the selected YAML file.
        :param status_label: The QLabel to display validation status.
        :param tab: The current tab QWidget.
        """
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
        """
        Opens a dialog to select a class mapping JSON file, sets the path, and validates it.

        :param class_mapping_edit: The QLineEdit widget to display the selected JSON file.
        :param status_label: The QLabel to display validation status.
        :param tab: The current tab QWidget.
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select class_mapping.json", "", "JSON Files (*.json);;All Files (*)", options=options
        )
        if file_path:
            class_mapping_edit.setText(file_path)
            self.validate_class_mapping(file_path, status_label, tab)

    def validate_directory(self, dir_path, status_label, tab):
        """
        Validates if the selected directory exists and updates the status.

        :param dir_path: Path to the directory to validate.
        :param status_label: The QLabel to display validation status.
        :param tab: The current tab QWidget.
        """
        if os.path.exists(dir_path):
            status_label.setText("✅")
            self.valid_directories.append((dir_path, None, tab))
        else:
            status_label.setText("❌")
            # Remove invalid directory from the list
            self.valid_directories = [(d, c, t) for d, c, t in self.valid_directories if d != dir_path]

    def validate_yaml(self, yaml_path, status_label, tab):
        """
        Validates the selected YAML file for required fields.

        :param yaml_path: Path to the YAML file to validate.
        :param status_label: The QLabel to display validation status.
        :param tab: The current tab QWidget.
        """
        try:
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)
                if 'names' in data and 'nc' in data:
                    status_label.setText("✅")
                    self.valid_directories.append((yaml_path, None, tab))
                else:
                    status_label.setText("❌")
                    # Remove invalid YAML from the list
                    self.valid_directories = [(d, c, t) for d, c, t in self.valid_directories if d != yaml_path]
        except Exception as e:
            status_label.setText("❌")
            # Remove invalid YAML from the list
            self.valid_directories = [(d, c, t) for d, c, t in self.valid_directories if d != yaml_path]

    def validate_class_mapping(self, class_mapping_path, status_label, tab):
        """
        Validates the selected class mapping JSON file.

        :param class_mapping_path: Path to the class mapping JSON file.
        :param status_label: The QLabel to display validation status.
        :param tab: The current tab QWidget.
        """
        try:
            with open(class_mapping_path, 'r') as file:
                data = json.load(file)
                if isinstance(data, dict):
                    status_label.setText("✅")
                    # Update class mapping in the valid directories list
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
        """
        Merges the selected datasets into a single output directory.
        """
        current_tab = self.tabs.currentWidget()
        if current_tab not in [self.tab_classification]:
            QMessageBox.warning(self, "Warning", "Only Image Classification merging has been implemented.")
            return

        # Retrieve output directory from the current tab
        output_dir = current_tab.findChild(QLineEdit, "output_dir_edit").text()
        if not output_dir:
            QMessageBox.warning(self, "Input Error", "Output directory must be specified.")
            return

        # Retrieve dataset name from the current tab
        dataset_name = current_tab.findChild(QLineEdit, "dataset_name_edit").text()
        if not dataset_name:
            QMessageBox.warning(self, "Input Error", "Dataset name must be specified.")
            return

        # Make cursor busy to indicate processing
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Create the output directory
        output_dir_path = os.path.join(output_dir, dataset_name)
        os.makedirs(output_dir_path, exist_ok=True)

        merged_class_mapping = {}

        def copy_directory(src, dest):
            """
            Copies contents from the source directory to the destination directory.

            :param src: Source directory path.
            :param dest: Destination directory path.
            """
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

                    # Copy train, val, and test splits
                    for split in ['train', 'val', 'test']:
                        src_split_dir = os.path.join(dir_path, split)
                        dest_split_dir = os.path.join(output_dir_path, split)
                        if os.path.exists(src_split_dir):
                            future = executor.submit(copy_directory, src_split_dir, dest_split_dir)
                            futures.append(future)

            # Create and display a progress bar
            progress_bar = ProgressBar(self, title=f"Merging Datasets")
            progress_bar.show()
            progress_bar.start_progress(len(futures))

            # Wait for all copy operations to complete
            for i, future in enumerate(as_completed(futures)):
                future.result()
                progress_bar.update_progress()

            # Close the progress bar
            progress_bar.stop_progress()
            progress_bar.close()

        # Save the merged class mapping if available
        if merged_class_mapping:
            merged_class_mapping_path = os.path.join(output_dir_path, "class_mapping.json")
            with open(merged_class_mapping_path, 'w') as json_file:
                json.dump(merged_class_mapping, json_file, indent=4)

        QMessageBox.information(self, "Success", "Datasets merged successfully!")

        # Restore the cursor to default
        QApplication.restoreOverrideCursor()

    def accept(self):
        """
        Overrides the accept method to perform dataset merging before closing the dialog.
        """
        self.merge_datasets()
        super().accept()