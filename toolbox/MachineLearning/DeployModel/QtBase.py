import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import gc
import json
import os
import random

import numpy as np

from PyQt5.QtGui import QColor, QShowEvent
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QWidget, QVBoxLayout,
                             QLabel, QDialog, QTextEdit, QPushButton, QGroupBox)

from torch.cuda import empty_cache
from ultralytics import YOLO


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
        self.annotation_window = main_window.annotation_window
        self.sam_dialog = None

        self.model_path = None
        self.loaded_model = None  
        self.class_mapping = None
        self.use_sam = None

        self.setWindowTitle("Deploy Model")
        self.resize(400, 300)

        self.layout = QVBoxLayout(self)
        
        # Set up the common layout components
        self.setup_generic_layout()
        
        # Create a group box for the status bar
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        # Status bar for model status
        self.status_bar = QLabel("No model loaded")
        status_layout.addWidget(self.status_bar)
        
        status_group.setLayout(status_layout)
        self.layout.addWidget(status_group)

        self.setLayout(self.layout)

    def setup_generic_layout(self, title="Deploy Model"):
        """
        Set up the common layout elements for model deployment
        """
        # Text area for displaying model info
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.layout.addWidget(self.text_area)

        # Model controls group
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()

        # Model control buttons
        self.browse_button = QPushButton("Browse Model")
        self.browse_button.clicked.connect(self.browse_file)
        
        self.mapping_button = QPushButton("Browse Class Mapping") 
        self.mapping_button.clicked.connect(self.browse_class_mapping_file)
        
        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_model)
        
        self.deactivate_button = QPushButton("Deactivate Model")
        self.deactivate_button.clicked.connect(self.deactivate_model)

        for button in [self.browse_button, 
                       self.mapping_button, 
                       self.load_button, 
                       self.deactivate_button]:
            model_layout.addWidget(button)

        model_group.setLayout(model_layout)
        
        self.layout.addWidget(model_group)
        
    def is_sam_model_deployed(self):
        """
        Check if the SAM model is deployed and update the checkbox state accordingly.

        :return: Boolean indicating whether the SAM model is deployed
        """
        self.sam_dialog = self.main_window.sam_deploy_model_dialog

        if not self.sam_dialog.loaded_model:
            # Ensure that the checkbox is not checked
            self.sender().setChecked(False)
            QMessageBox.warning(self, "SAM Model", "SAM model not currently deployed")
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
            if ".bin" in file_path:
                # OpenVINO is a directory
                file_path = os.path.dirname(file_path)

            self.model_path = file_path
            self.text_area.setText("Model file selected")

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
            self.text_area.append("Class mapping file selected")
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
            
        class_names = list(self.loaded_model.names.values())
        class_names_str = "Class Names:\n"
        missing_labels = []

        for class_name in class_names:
            label = self.label_window.get_label_by_short_code(class_name)
            if label:
                class_names_str += f"✅ {label.short_label_code}: {label.long_label_code}\n"
            else:
                class_names_str += f"❌ {class_name}\n"
                missing_labels.append(class_name)

        self.text_area.setText(class_names_str)
        
        if missing_labels:
            missing_labels_str = "\n".join(missing_labels)
            QMessageBox.warning(
                self,
                "Warning",
                f"The following short labels are missing and cannot be predicted "
                f"until added manually:\n{missing_labels_str}"
            )
            
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
        else:
            self.check_and_display_class_names()
            QMessageBox.information(self, "Model Loaded", "Model loaded successfully.")

    def add_labels_to_label_window(self):
        """
        Add labels to the label window based on the class mapping.
        """
        if self.class_mapping:
            for label in self.class_mapping.values():
                self.label_window.add_label_if_not_exists(label['short_label_code'],
                                                          label['long_label_code'],
                                                          QColor(*label['color']))
    
    def create_generic_labels(self, class_names):
        """
        Create generic labels for the given class names
        
        :param class_names: List of class names
        """
        for class_name in class_names:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            self.label_window.add_label_if_not_exists(
                class_name,
                class_name,
                QColor(r, g, b)
            )

    def get_confidence_threshold(self):
        """
        Get the confidence threshold for predictions
        """
        threshold = self.main_window.get_uncertainty_thresh()
        return threshold if threshold < 0.10 else 0.10
    
    def get_iou_threshold(self):
        """
        Get the IoU threshold for predictions
        """
        return self.main_window.get_iou_thresh()
    
    def predict(self, inputs):
        """
        Predict using deployed model
        """
        raise NotImplementedError("Subclasses must implement predict()")
    
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
        self.text_area.setText("No model file selected")
