import warnings

import os
import shutil
import ujson as json
from concurrent.futures import ThreadPoolExecutor, as_completed

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QWidget, QVBoxLayout,
                             QLabel, QLineEdit, QDialog, QHBoxLayout, QPushButton,
                             QDialogButtonBox, QFormLayout, QGroupBox, QScrollArea)

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Base(QDialog):
    """
    Dialog for merging multiple datasets into a single dataset for machine learning tasks such as
    image classification.
    """

    def __init__(self, parent=None):
        """
        Initializes the MergeDatasetsDialog.

        :param parent: Parent widget, default is None.
        """
        super().__init__(parent)

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Merge Datasets")
        self.resize(500, 500)

        self.task = None
        self.valid_directories = []
        self.dataset_count = 0

        self.layout = QVBoxLayout(self)

        # Setup the info layout
        self.setup_info_layout()
        # Setup the outputs_layout
        self.setup_outputs_layout()
        # Setup the datasets layout
        self.setup_datasets_layout()
        # Setup the buttons layout
        self.setup_buttons_layout()

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Select multiple Classification datasets to merge.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_outputs_layout(self):
        """Setup the outputs layout."""
        group_box = QGroupBox("Output Dataset")
        layout = QFormLayout()

        # Dataset Name Input
        self.dataset_name_edit = QLineEdit()
        layout.addRow("Dataset Name:", self.dataset_name_edit)

        # Output Directory Chooser
        self.output_dir_edit = QLineEdit()
        output_dir_button = QPushButton("Browse...")
        output_dir_button.clicked.connect(lambda: self.browse_output_directory(self.output_dir_edit))
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.output_dir_edit)
        dir_layout.addWidget(output_dir_button)
        layout.addRow("Output Directory:", dir_layout)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_datasets_layout(self):
        """Setup the datasets layout."""
        group_box = QGroupBox("Datasets")
        layout = QVBoxLayout()

        # Create a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Create a widget to hold the dataset entries
        self.datasets_widget = QWidget()
        self.datasets_layout = QVBoxLayout(self.datasets_widget)

        # Add the widget to the scroll area
        scroll.setWidget(self.datasets_widget)

        # Add button to add new dataset
        add_dataset_button = QPushButton("Add Dataset")
        add_dataset_button.clicked.connect(self.add_dataset_entry)

        layout.addWidget(scroll)
        layout.addWidget(add_dataset_button)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_buttons_layout(self):
        """Setup the buttons layout."""
        # OK and Cancel Buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

    def browse_output_directory(self, output_dir_edit):
        """
        Opens a dialog to select the output directory and sets the selected path."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            output_dir_edit.setText(dir_path)

    def add_dataset_entry(self):
        """Add a new dataset entry row."""
        self.dataset_count += 1

        group_box = QGroupBox(f"Dataset {self.dataset_count}")
        form_layout = QFormLayout()

        # Status label in corner
        status_label = QLabel()
        group_box.setLayout(form_layout)

        # Directory input row
        dir_edit = QLineEdit()
        dir_button = QPushButton("Browse Directory...")
        dir_button.clicked.connect(lambda: self.browse_existing_directory(dir_edit, status_label))
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(dir_edit)
        dir_layout.addWidget(dir_button)
        form_layout.addRow("Directory:", dir_layout)

        # Class mapping input row
        class_mapping_edit = QLineEdit()
        class_mapping_edit.setObjectName("class_mapping_edit")
        class_mapping_button = QPushButton("Select Class Mapping")
        class_mapping_button.clicked.connect(lambda: self.browse_class_mapping(class_mapping_edit, status_label))
        mapping_layout = QHBoxLayout()
        mapping_layout.addWidget(class_mapping_edit)
        mapping_layout.addWidget(class_mapping_button)
        form_layout.addRow("Class Mapping:", mapping_layout)

        # Remove button row
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(lambda: self.remove_dataset_entry(group_box))
        button_layout = QHBoxLayout()
        button_layout.addWidget(status_label)
        button_layout.addWidget(remove_button)
        button_layout.setAlignment(Qt.AlignRight)
        form_layout.addRow("", button_layout)

        self.datasets_layout.addWidget(group_box)

    def remove_dataset_entry(self, entry_widget):
        """Remove a dataset entry."""
        entry_widget.deleteLater()
        # Remove from valid directories if present
        dir_edit = entry_widget.findChild(QLineEdit)
        if dir_edit:
            dir_path = dir_edit.text()
            self.valid_directories = [(d, c) for d, c in self.valid_directories if d != dir_path]

    def browse_existing_directory(self, dir_edit, status_label):
        """Opens a dialog to select an existing dataset directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Existing Dataset Directory")
        if dir_path:
            dir_edit.setText(dir_path)
            self.validate_directory(dir_path, status_label)

            # Auto-fill class_mapping.json if it exists
            class_mapping_path = os.path.join(dir_path, "class_mapping.json")
            if os.path.exists(class_mapping_path):
                class_mapping_edit = dir_edit.parent().findChild(QLineEdit, "class_mapping_edit")
                if class_mapping_edit:
                    class_mapping_edit.setText(class_mapping_path)
                    self.validate_class_mapping(class_mapping_path, status_label)

    def browse_class_mapping(self, class_mapping_edit, status_label):
        """Opens a dialog to select a class mapping JSON file, sets the path, and validates it."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Select class_mapping.json",
                                                   "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_path:
            class_mapping_edit.setText(file_path)
            self.validate_class_mapping(file_path, status_label)

    def validate_directory(self, dir_path, status_label):
        """Validates if the selected directory exists."""
        if os.path.exists(dir_path):
            status_label.setText("✅")
            self.valid_directories.append((dir_path, None))
        else:
            status_label.setText("❌")
            self.valid_directories = [(d, c) for d, c in self.valid_directories if d != dir_path]

    def validate_class_mapping(self, class_mapping_path, status_label):
        """Validates the selected class mapping JSON file."""
        try:
            with open(class_mapping_path, 'r') as file:
                data = json.load(file)
                if isinstance(data, dict):
                    status_label.setText("✅")
                    # Update class mapping in valid directories
                    for i, (d, c) in enumerate(self.valid_directories):
                        if c is None:
                            self.valid_directories[i] = (d, class_mapping_path)
                            break
                else:
                    status_label.setText("❌")
        except Exception:
            status_label.setText("❌")

    def merge_datasets(self):
        """
        Merges the selected datasets into a single output directory.
        """
        self.output_dir = self.output_dir_edit.text()
        self.dataset_name = self.dataset_name_edit.text()

        if self.task != 'classify':
            QMessageBox.warning(self, "Warning", "Only Image Classification merging has been implemented.")
            return

        # Use class-level output directory and dataset name
        if not self.output_dir:
            QMessageBox.warning(self, "Input Error", "Output directory must be specified.")
            return

        if not self.dataset_name:
            QMessageBox.warning(self, "Input Error", "Dataset name must be specified.")
            return

        # Make cursor busy to indicate processing
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Create the output directory
        output_dir_path = os.path.join(self.output_dir, self.dataset_name)
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
            for dir_path, class_mapping_path in self.valid_directories:
                if self.task == 'classify':
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
                else:
                    QMessageBox.warning(self, "Warning", "Only Classification datasets merging has been implemented.")
                    return

            # Create and display a progress bar
            progress_bar = ProgressBar(self, title="Merging Datasets")
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
