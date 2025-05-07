import warnings

import os
import gc
import random
import ujson as json

from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QFileDialog, QMessageBox, QVBoxLayout, QLabel, QDialog,
                             QTextEdit, QPushButton, QGroupBox, QHBoxLayout)

from torch.cuda import empty_cache

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Base(QDialog):
    """
    Base class for deploying machine learning models.

    :param main_window: MainWindow object
    :param parent: Parent widget
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.label_window = main_window.label_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window
        self.sam_dialog = None

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Deploy Model")
        self.resize(400, 325)

        # Initialize variables
        self.imgsz = 1024
        self.iou_thresh = 0.20
        self.uncertainty_thresh = 0.30
        self.area_thresh_min = 0.00
        self.area_thresh_max = 0.40

        self.task = None
        self.max_detect = 300
        self.model_path = None
        self.loaded_model = None
        self.class_names = []
        self.class_mapping = {}

        self.layout = QVBoxLayout(self)

        # Setup the info layout
        self.setup_info_layout()
        # Setup the labels layout
        self.setup_labels_layout()
        # Setup parameters layout
        self.setup_parameters_layout()
        # Setup SAM layout
        self.setup_sam_layout()
        # Setup the button layout
        self.setup_buttons_layout()
        # Setup the status layout
        self.setup_status_layout()

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Deploy an Ultralytics model to use.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_labels_layout(self):
        """

        """
        group_box = QGroupBox("Labels")
        layout = QVBoxLayout()

        # Text area for displaying model info
        self.label_area = QTextEdit()
        self.label_area.setReadOnly(True)
        layout.addWidget(self.label_area)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_parameters_layout(self):
        raise NotImplementedError("Subclasses must implement this method")

    def setup_sam_layout(self):
        raise NotImplementedError("Subclasses must implement this method")

    def setup_buttons_layout(self):
        """
        Set up the buttons layout in a 2x2 grid
        """
        # Model controls group
        group_box = QGroupBox("Actions")
        layout = QVBoxLayout()  # Main vertical layout

        # Create two horizontal layouts for each row
        top_row = QHBoxLayout()
        bottom_row = QHBoxLayout()

        # Model control buttons
        self.browse_button = QPushButton("Browse Model")
        self.browse_button.clicked.connect(self.browse_file)

        self.mapping_button = QPushButton("Browse Class Mapping")
        self.mapping_button.clicked.connect(self.browse_class_mapping_file)

        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_model)

        self.deactivate_button = QPushButton("Deactivate Model")
        self.deactivate_button.clicked.connect(self.deactivate_model)

        # Add buttons to rows
        top_row.addWidget(self.browse_button)
        top_row.addWidget(self.mapping_button)
        bottom_row.addWidget(self.load_button)
        bottom_row.addWidget(self.deactivate_button)

        # Add rows to main layout
        layout.addLayout(top_row)
        layout.addLayout(bottom_row)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_status_layout(self):
        """

        """
        # Create a group box for the status bar
        group_box = QGroupBox("Status")
        layout = QVBoxLayout()

        # Status bar for model status
        self.status_bar = QLabel("No model loaded")
        layout.addWidget(self.status_bar)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def initialize_uncertainty_threshold(self):
        """Initialize the uncertainty threshold slider with the current value"""
        current_value = self.main_window.get_uncertainty_thresh()
        self.uncertainty_threshold_slider.setValue(int(current_value * 100))
        self.uncertainty_thresh = current_value

    def initialize_iou_threshold(self):
        """Initialize the IOU threshold slider with the current value"""
        current_value = self.main_window.get_iou_thresh()
        self.iou_threshold_slider.setValue(int(current_value * 100))
        self.iou_thresh = current_value

    def initialize_area_threshold(self):
        """Initialize the area threshold range slider"""
        current_min, current_max = self.main_window.get_area_thresh()
        self.area_threshold_min_slider.setValue(int(current_min * 100))
        self.area_threshold_max_slider.setValue(int(current_max * 100))
        self.area_thresh_min = current_min
        self.area_thresh_max = current_max

    def update_uncertainty_label(self, value):
        """Update uncertainty threshold and label"""
        value = value / 100.0
        self.uncertainty_thresh = value
        self.main_window.update_uncertainty_thresh(value)
        self.uncertainty_threshold_label.setText(f"{value:.2f}")

    def update_iou_label(self, value):
        """Update IoU threshold and label"""
        value = value / 100.0
        self.iou_thresh = value
        self.main_window.update_iou_thresh(value)
        self.iou_threshold_label.setText(f"{value:.2f}")

    def update_area_label(self):
        """Handle changes to area threshold range slider"""
        min_val = self.area_threshold_min_slider.value()
        max_val = self.area_threshold_max_slider.value()
        if min_val > max_val:
            min_val = max_val
            self.area_threshold_min_slider.setValue(min_val)
        self.area_thresh_min = min_val / 100.0
        self.area_thresh_max = max_val / 100.0
        self.main_window.update_area_thresh(self.area_thresh_min, self.area_thresh_max)
        self.area_threshold_label.setText(f"{self.area_thresh_min:.2f} - {self.area_thresh_max:.2f}")

    def is_sam_model_deployed(self):
        """
        Check if the SAM model is deployed and update the checkbox state accordingly.

        :return: Boolean indicating whether the SAM model is deployed
        """
        if not hasattr(self.main_window, 'sam_deploy_predictor_dialog'):
            return False

        self.sam_dialog = self.main_window.sam_deploy_predictor_dialog

        if not self.sam_dialog.loaded_model:
            self.use_sam_dropdown.setCurrentText("False")
            QMessageBox.critical(self, "Error", "Please deploy the SAM model first.")
            return False

        return True

    def browse_file(self):
        """Browse and select a model file"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Model File", "",
            "Model Files (*.pt *.onnx *.torchscript *.engine *.bin)",
            options=options
        )

        if file_path:
            # Clear the class mapping
            self.class_mapping = {}

            if ".bin" in file_path:
                # OpenVINO is a directory
                file_path = os.path.dirname(file_path)

            self.model_path = file_path
            self.label_area.setText("Model file selected")

            # Try to load the class mapping file if it exists
            parent_dir = os.path.dirname(os.path.dirname(file_path))
            class_mapping_path = os.path.join(parent_dir, "class_mapping.json")
            if os.path.exists(class_mapping_path):
                self.load_class_mapping(class_mapping_path)

    def browse_class_mapping_file(self):
        """Browse and select a class mapping file"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Class Mapping File", "",
            "JSON Files (*.json)",
            options=options
        )
        if file_path:
            self.load_class_mapping(file_path)

    def load_class_mapping(self, file_path):
        """
        Load the class mapping file

        :param file_path: Path to the class mapping file
        """
        try:
            with open(file_path, 'r') as f:
                self.class_mapping = json.load(f)
            self.label_area.append("Class mapping file selected")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load class mapping file: {str(e)}")

    def load_model(self):
        """
        Load the model
        """
        raise NotImplementedError("Subclasses must implement this method")

    def check_and_display_class_names(self):
        """
        Check and display the class names
        """
        if not self.loaded_model:
            return

        class_names_str = ""
        missing_labels = []

        for class_name in self.class_names:
            label = self.label_window.get_label_by_short_code(class_name)
            if label:
                class_names_str += f"✅ {label.short_label_code}: {label.long_label_code}\n"
            else:
                class_names_str += f"❌ {class_name}\n"
                missing_labels.append(class_name)

        self.label_area.setText(class_names_str)

        if missing_labels:
            missing_labels_str = "\n".join(missing_labels)
            QMessageBox.warning(
                self,
                "Warning",
                f"The following short labels are missing and cannot be predicted "
                f"until added manually:\n{missing_labels_str}"
            )

    def add_labels_to_label_window(self):
        """
        Add labels to the label window based on the class mapping.
        """
        if self.class_mapping:
            for label in self.class_mapping.values():
                self.label_window.add_label_if_not_exists(label['short_label_code'],
                                                          label['long_label_code'],
                                                          QColor(*label['color']))

    def handle_missing_class_mapping(self):
        """
        Handle the case when the class mapping file is missing.
        """
        reply = QMessageBox.question(self,
                                     'No Class Mapping Found',
                                     'Do you want to create generic labels automatically?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.create_generic_labels()


    def create_generic_labels(self):
        """
        Create generic labels for the given class names
        """
        for class_name in self.class_names:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            # Create the label in the label window
            self.label_window.add_label_if_not_exists(
                class_name,
                class_name,
                QColor(r, g, b)
            )
            label = self.label_window.get_label_by_short_code(class_name)
            self.class_mapping[class_name] = label.to_dict()

    def predict(self, inputs):
        """
        Predict using deployed model
        """
        raise NotImplementedError("Subclasses must implement predict method")

    def deactivate_model(self):
        """
        Deactivate the current model
        """
        self.loaded_model = None
        self.model_path = None
        self.class_mapping = None
        gc.collect()
        empty_cache()
        self.status_bar.setText("No model loaded")
        self.label_area.setText("No model file selected")
