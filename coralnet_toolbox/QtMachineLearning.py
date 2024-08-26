import os
import random
import datetime

import numpy as np
from ultralytics import YOLO

from coralnet_toolbox.QtProgressBar import ProgressBar

from PyQt5.QtWidgets import (QFileDialog, QApplication, QScrollArea, QMessageBox, QCheckBox, QWidget, QVBoxLayout,
                             QLabel, QLineEdit, QDialog, QHBoxLayout, QTextEdit, QPushButton, QComboBox, QSpinBox,
                             QFormLayout, QTabWidget, QDialogButtonBox, QDoubleSpinBox, QGroupBox, QTableWidget,
                             QTableWidgetItem)

from PyQt5.QtGui import QImage, QBrush, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class CreateDatasetDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.annotation_window = main_window.annotation_window
        self.image_window = main_window.image_window

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
        self.label_counts_table.setHorizontalHeaderLabels(["Include", "Label", "Total", "Train", "Val", "Test"])
        self.class_filter_layout.addWidget(self.label_counts_table)
        self.class_filter_group.setLayout(self.class_filter_layout)
        layout.addWidget(self.class_filter_group)

        # Ready Status
        self.ready_label = QLabel()
        layout.addWidget(self.ready_label)

        # Add the layout
        self.tab_classification.setLayout(layout)

        # Populate class filter list
        self.populate_class_filter_list()
        # Initial update of summary statistics
        self.update_summary_statistics()

    def setup_segmentation_tab(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Instance Segmentation tab (Future Work)"))
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

        self.label_counts_table.setRowCount(0)

        # Create a dictionary to count occurrences of each label
        label_counts = {}
        for annotation in self.annotation_window.annotations_dict.values():
            label = annotation.label.short_label_code
            if label != 'Review':
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1

        # Populate the label counts table with labels and their counts
        row = 0
        for label, count in label_counts.items():
            include_checkbox = QCheckBox()
            include_checkbox.setChecked(True)
            include_checkbox.stateChanged.connect(self.on_include_checkbox_state_changed)
            label_item = QTableWidgetItem(label)
            total_item = QTableWidgetItem(str(count))
            train_item = QTableWidgetItem("0")
            val_item = QTableWidgetItem("0")
            test_item = QTableWidgetItem("0")

            self.label_counts_table.insertRow(row)
            self.label_counts_table.setCellWidget(row, 0, include_checkbox)
            self.label_counts_table.setItem(row, 1, label_item)
            self.label_counts_table.setItem(row, 2, total_item)
            self.label_counts_table.setItem(row, 3, train_item)
            self.label_counts_table.setItem(row, 4, val_item)
            self.label_counts_table.setItem(row, 5, test_item)

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

        images = list(self.annotation_window.loaded_image_paths)
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
        # Check if the train ratio is greater than 0
        train_ratio = self.train_ratio_spinbox.value()
        val_ratio = self.val_ratio_spinbox.value()
        test_ratio = self.test_ratio_spinbox.value()

        # Ensure there is at least one train set
        if train_ratio == 0:
            return False

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

        # Check if there are any labels in splits with a ratio of 0
        if val_ratio == 0 and len(val_label_counts) > 0:
            return False
        if test_ratio == 0 and len(test_label_counts) > 0:
            return False

        # Additional checks to ensure no empty splits
        if train_ratio > 0 and len(self.train_annotations) == 0:
            return False
        if val_ratio > 0 and len(self.val_annotations) == 0:
            return False
        if test_ratio > 0 and len(self.test_annotations) == 0:
            return False

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
            total_count = sum(1 for a in annotations if a.label.short_label_code == label)
            if include_checkbox.isChecked():
                train_count = sum(1 for a in self.train_annotations if a.label.short_label_code == label)
                val_count = sum(1 for a in self.val_annotations if a.label.short_label_code == label)
                test_count = sum(1 for a in self.test_annotations if a.label.short_label_code == label)
            else:
                train_count = 0
                val_count = 0
                test_count = 0

            self.label_counts_table.item(row, 2).setText(str(total_count))
            self.label_counts_table.item(row, 3).setText(str(train_count))
            self.label_counts_table.item(row, 4).setText(str(val_count))
            self.label_counts_table.item(row, 5).setText(str(test_count))

            # Set cell colors based on the counts
            if include_checkbox.isChecked():
                self.set_cell_color(row, 3, QColor(255, 0, 0) if train_count == 0 else QColor(255, 255, 255))
                self.set_cell_color(row, 4, QColor(255, 0, 0) if val_count == 0 else QColor(255, 255, 255))
                self.set_cell_color(row, 5, QColor(255, 0, 0) if test_count == 0 else QColor(255, 255, 255))
            else:
                self.set_cell_color(row, 3, QColor(255, 255, 255))
                self.set_cell_color(row, 4, QColor(255, 255, 255))
                self.set_cell_color(row, 5, QColor(255, 255, 255))

        self.ready_status = self.check_label_distribution()
        self.split_status = abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-9
        self.ready_label.setText("✅ Ready" if (self.ready_status and self.split_status) else "❌ Not Ready")

        self.updating_summary_statistics = False

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
        if os.path.exists(output_dir_path):
            reply = QMessageBox.question(self,
                                         "Directory Exists",
                                         "The output directory already exists. Do you want to merge the datasets?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return

        os.makedirs(output_dir_path, exist_ok=True)
        train_dir = os.path.join(output_dir_path, 'train')
        val_dir = os.path.join(output_dir_path, 'val')
        test_dir = os.path.join(output_dir_path, 'test')

        # Set the cursor to waiting (busy) cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

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
        image_paths = list(set(a.image_path for a in annotations))
        if not image_paths:
            return

        progress_bar = ProgressBar(self, title=f"Creating {split} Dataset")
        progress_bar.show()
        progress_bar.start_progress(len(image_paths))

        def process_image(image_path):
            if progress_bar.wasCanceled():
                return

            # Crop all image annotations
            image_annotations = self.annotation_window.crop_image_annotations(image_path, return_annotations=True)

            for image_annotation in image_annotations:
                # Save the crop in the correct folder
                cropped_image = image_annotation.cropped_image
                label_code = image_annotation.label.short_label_code
                output_path = os.path.join(split_dir, label_code)
                os.makedirs(output_path, exist_ok=True)
                output_filename = f"{label_code}_{image_annotation.id}.jpg"
                cropped_image.save(os.path.join(output_path, output_filename))

        for image_path in image_paths:
            process_image(image_path)
            progress_bar.update_progress()
            QApplication.processEvents()

        progress_bar.stop_progress()
        progress_bar.close()

    def showEvent(self, event):
        super().showEvent(event)
        self.populate_class_filter_list()
        self.update_summary_statistics()


class TrainModelWorker(QThread):
    training_started = pyqtSignal()
    training_completed = pyqtSignal()
    training_error = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self.training_started.emit()

            # Initialize the model
            model_path = self.params.pop('model', None)
            target_model = YOLO(model_path)

            # Train the model
            results = target_model.train(**self.params)

            self.training_completed.emit()

        except Exception as e:
            self.training_error.emit(str(e))


class TrainModelDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        # For holding parameters
        self.custom_params = []
        # Best model weights
        self.model_path = None

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
        self.patience_spinbox.setValue(100)
        self.form_layout.addRow("Patience:", self.patience_spinbox)

        # Batch
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setMinimum(-1)
        self.batch_spinbox.setMaximum(1024)
        self.batch_spinbox.setValue(-1)
        self.form_layout.addRow("Batch Size:", self.batch_spinbox)

        # Imgsz
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setMinimum(16)
        self.imgsz_spinbox.setMaximum(4096)
        self.imgsz_spinbox.setValue(224)
        self.form_layout.addRow("Image Size:", self.imgsz_spinbox)

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

        # Workers
        self.workers_spinbox = QSpinBox()
        self.workers_spinbox.setMinimum(1)
        self.workers_spinbox.setMaximum(64)
        self.workers_spinbox.setValue(8)
        self.form_layout.addRow("Workers:", self.workers_spinbox)

        # Pretrained
        self.pretrained_checkbox = QCheckBox()
        self.pretrained_checkbox.setChecked(True)
        self.form_layout.addRow("Pretrained:", self.pretrained_checkbox)

        # Optimizer
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"])
        self.optimizer_combo.setCurrentText("auto")
        self.form_layout.addRow("Optimizer:", self.optimizer_combo)

        # Freeze
        self.freeze_edit = QLineEdit()
        self.form_layout.addRow("Freeze Layers:", self.freeze_edit)

        # Lr0
        self.lr0_spinbox = QDoubleSpinBox()
        self.lr0_spinbox.setMinimum(0.0001)
        self.lr0_spinbox.setMaximum(1.0)
        self.lr0_spinbox.setValue(0.01)
        self.form_layout.addRow("Learning Rate (lr0):", self.lr0_spinbox)

        # Dropout
        self.dropout_spinbox = QDoubleSpinBox()
        self.dropout_spinbox.setMinimum(0.0)
        self.dropout_spinbox.setMaximum(1.0)
        self.dropout_spinbox.setValue(0.0)
        self.form_layout.addRow("Dropout:", self.dropout_spinbox)

        # Fraction
        self.fraction_spinbox = QDoubleSpinBox()
        self.fraction_spinbox.setMinimum(0.1)
        self.fraction_spinbox.setMaximum(1.0)
        self.fraction_spinbox.setValue(1.0)
        self.form_layout.addRow("Fraction:", self.fraction_spinbox)

        # Val
        self.val_checkbox = QCheckBox()
        self.val_checkbox.setChecked(True)
        self.form_layout.addRow("Validation:", self.val_checkbox)

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

    def get_training_parameters(self):
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
            'dropout': self.dropout_spinbox.value(),
            'val': self.val_checkbox.isChecked(),
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
        message = "Model training has commenced.\nMonitor the console for real-time progress."
        QMessageBox.information(self, "Model Training Status", message)

        # Get training parameters
        params = self.get_training_parameters()

        # Create and start the worker thread
        self.worker = TrainModelWorker(params)
        self.worker.training_started.connect(self.on_training_started)
        self.worker.training_completed.connect(self.on_training_completed)
        self.worker.training_error.connect(self.on_training_error)
        self.worker.start()

    def on_training_started(self):
        pass

    def on_training_completed(self):
        message = "Model training has successfully been completed."
        QMessageBox.information(self, "Model Training Status", message)

    def on_training_error(self, error_message):
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

        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)

        self.classification_tab = QWidget()
        self.segmentation_tab = QWidget()

        self.tab_widget.addTab(self.classification_tab, "Image Classification")
        self.tab_widget.addTab(self.segmentation_tab, "Instance Segmentation")

        self.init_classification_tab()
        self.init_segmentation_tab()

        self.status_bar = QLabel("No model loaded")
        self.layout.addWidget(self.status_bar)

        self.setLayout(self.layout)

    def init_classification_tab(self):
        layout = QVBoxLayout()

        self.classification_text_area = QTextEdit()
        self.classification_text_area.setReadOnly(True)
        layout.addWidget(self.classification_text_area)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_file)
        layout.addWidget(browse_button)

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

    def load_model(self):
        if self.model_path:
            try:
                # Set the cursor to waiting (busy) cursor
                QApplication.setOverrideCursor(Qt.WaitCursor)

                self.loaded_model = YOLO(self.model_path, task='classify')
                self.loaded_model(np.zeros((224, 224, 3), dtype=np.uint8))

                # Get the class names the model can predict
                class_names = list(self.loaded_model.names.values())
                class_names_str = "Class Names: \n"
                missing_labels = []

                for class_name in class_names:
                    if self.label_window.get_label_by_long_code(class_name):
                        class_names_str += f"✅ {class_name} \n"
                    else:
                        class_names_str += f"❌ {class_name} \n"
                        missing_labels.append(class_name)

                self.classification_text_area.setText(class_names_str)
                self.status_bar.setText(f"Model loaded: {os.path.basename(self.model_path)}")

                if missing_labels:
                    missing_labels_str = "\n".join(missing_labels)
                    QMessageBox.warning(self,
                                        "Warning",
                                        f"The following classes are missing and cannot be predicted:"
                                        f"\n{missing_labels_str}")

                QMessageBox.information(self, "Model Loaded", "Model weights loaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            finally:
                # Restore the cursor to the default cursor
                QApplication.restoreOverrideCursor()
        else:
            QMessageBox.warning(self, "Warning", "No model file selected")

    def deactivate_model(self):
        self.loaded_model = None
        self.model_path = None
        self.status_bar.setText("No model loaded")
        if self.tab_widget.currentIndex() == 0:
            self.classification_text_area.setText("No model file selected")
        else:
            self.segmentation_file_path.setText("No model file selected")

    def pixmap_to_numpy(self, pixmap):
        # Convert QPixmap to QImage
        image = pixmap.toImage()
        # Get image dimensions
        width = image.width()
        height = image.height()

        # Convert QImage to numpy array
        byte_array = image.bits().asstring(width * height * 4)  # 4 for RGBA
        numpy_array = np.frombuffer(byte_array, dtype=np.uint8).reshape((height, width, 4))

        # If the image format is ARGB32, swap the first and last channels (A and B)
        if format == QImage.Format_ARGB32:
            numpy_array = numpy_array[:, :, [2, 1, 0, 3]]

        return numpy_array

    def predict(self, annotations=None):
        # If model isn't loaded
        if self.loaded_model is None:
            return

        # Set the cursor to waiting (busy) cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Get the selected annotation
        selected_annotation = self.annotation_window.selected_annotation

        if selected_annotation:
            # Make predictions on a single, specific annotation
            self.predict_annotation(selected_annotation)
            # Update everything (essentially)
            self.main_window.annotation_window.unselect_annotation()
            self.main_window.annotation_window.select_annotation(selected_annotation)
        else:
            # Run predictions on multiple annotations
            if not annotations:
                # If not supplied with annotations, get all of those for current image
                annotations = self.annotation_window.get_image_annotations()

            # Convert QImages to numpy arrays
            images_np = [self.pixmap_to_numpy(annotation.cropped_image) for annotation in annotations]

            # Perform batch prediction
            results = self.loaded_model(images_np)

            for annotation, result in zip(annotations, results):
                # Process the results
                self.process_prediction_result(annotation, result)

                # Show last in the confidence window
                self.main_window.confidence_window.display_cropped_image(annotation)

        # Restore the cursor to the default cursor
        QApplication.restoreOverrideCursor()

    def predict_annotation(self, annotation):
        # Get the cropped image
        image = annotation.cropped_image
        # Convert QImage to np
        image_np = self.pixmap_to_numpy(image)
        # Perform prediction
        results = self.loaded_model(image_np)[0]

        # Extract the results
        class_names = results.names
        top5 = results.probs.top5
        top5conf = results.probs.top5conf

        # Initialize an empty dictionary to store the results
        predictions = {}

        # Iterate over the top 5 predictions
        for idx, conf in zip(top5, top5conf):
            class_name = class_names[idx]
            label = self.label_window.get_label_by_long_code(class_name)

            if label:
                predictions[label] = float(conf)
            else:
                # Users does not have label loaded; skip.
                pass

        if predictions:
            # Update the machine confidence
            annotation.update_machine_confidence(predictions)

    def process_prediction_result(self, annotation, result):
        # Extract the results
        class_names = result.names
        top5 = result.probs.top5
        top5conf = result.probs.top5conf

        # Initialize an empty dictionary to store the results
        predictions = {}

        # Iterate over the top 5 predictions
        for idx, conf in zip(top5, top5conf):
            class_name = class_names[idx]
            label = self.label_window.get_label_by_long_code(class_name)

            if label:
                predictions[label] = float(conf)
            else:
                # User does not have label loaded; skip.
                pass

        if predictions:
            # Update the machine confidence
            annotation.update_machine_confidence(predictions)