import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import gc
import json
import os
import random

import numpy as np

from qtrangeslider import QRangeSlider
from PyQt5.QtGui import QColor, QShowEvent
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QWidget, QVBoxLayout,
                             QLabel, QDialog, QTextEdit, QPushButton, QGroupBox, QCheckBox,
                             QFormLayout, QComboBox, QSpinBox, QSlider)

from torch.cuda import empty_cache
from ultralytics import YOLO

from toolbox.MachineLearning.DeployModel.QtBase import Base

from toolbox.ResultsProcessor import ResultsProcessor
from toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Segment(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Deploy Segmentation Model")
        
        # Setup parameters layout
        self.setup_parameters_layout()
             
    def showEvent(self, event):
        """
        Handle the show event to update label options and sync uncertainty threshold.

        Args:
            event: The event object.
        """
        super().showEvent(event)
        self.initialize_area_threshold
        self.initialize_uncertainty_threshold()
        self.initialize_iou_threshold()
             
    def setup_parameters_layout(self):
        """
        Setup parameter control section in a group box.
        """
        group_box = QGroupBox("Parameters")
        form_layout = QFormLayout()
        
        # Area threshold controls
        self.area_threshold_slider = QRangeSlider(Qt.Horizontal)
        self.area_threshold_slider.setMinimum(0)
        self.area_threshold_slider.setMaximum(100)
        self.area_threshold_slider.setSingleStep(1)
        self.area_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.area_threshold_slider.setTickInterval(10)
        min_val = self.area_thresh_min
        max_val = self.area_thresh_max
        self.area_threshold_slider.setValue((int(min_val * 100), int(max_val * 100)))
        self.area_threshold_slider.valueChanged.connect(self.update_area_label)
        self.area_threshold_label = QLabel(f"{min_val:.2f} - {max_val:.2f}")
        form_layout.addRow("Area Threshold", self.area_threshold_slider)
        form_layout.addRow("", self.area_threshold_label)

        # Uncertainty threshold controls
        self.uncertainty_threshold_slider = QSlider(Qt.Horizontal)
        self.uncertainty_threshold_slider.setRange(0, 100)
        self.uncertainty_threshold_slider.setValue(int(self.main_window.get_uncertainty_thresh() * 100))
        self.uncertainty_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.uncertainty_threshold_slider.setTickInterval(10)
        self.uncertainty_threshold_slider.valueChanged.connect(self.update_uncertainty_label)
        self.uncertainty_threshold_label = QLabel(f"{self.main_window.get_uncertainty_thresh():.2f}")
        form_layout.addRow("Uncertainty Threshold", self.uncertainty_threshold_slider)
        form_layout.addRow("", self.uncertainty_threshold_label)
        
        # IoU threshold controls
        self.iou_threshold_slider = QSlider(Qt.Horizontal)
        self.iou_threshold_slider.setRange(0, 100)
        self.iou_threshold_slider.setValue(int(self.main_window.get_iou_thresh() * 100))
        self.iou_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.iou_threshold_slider.setTickInterval(10)
        self.iou_threshold_slider.valueChanged.connect(self.update_iou_label)
        self.iou_threshold_label = QLabel(f"{self.main_window.get_iou_thresh():.2f}")
        form_layout.addRow("IoU Threshold", self.iou_threshold_slider)
        form_layout.addRow("", self.iou_threshold_label)
        
        # SAM dropdown
        self.use_sam_dropdown = QComboBox()
        self.use_sam_dropdown.addItems(["False", "True"])
        self.use_sam_dropdown.currentIndexChanged.connect(self.is_sam_model_deployed)
        form_layout.addRow("Use SAM for creating Polygons:", self.use_sam_dropdown)
        
        group_box.setLayout(form_layout)
        self.layout.addWidget(group_box)
        
    def initialize_area_threshold(self):
        """Initialize the area threshold range slider"""
        min_val = int(self.area_thresh_min * 100)
        max_val = int(self.area_thresh_max * 100)
        self.area_threshold_slider.setLow(min_val)
        self.area_threshold_slider.setHigh(max_val)
        self.area_threshold_label.setText(f"Area Threshold: {min_val}% - {max_val}%")
        
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
        
    def update_area_label(self):
        """Handle changes to area threshold range slider"""
        min_val, max_val = self.area_threshold_slider.value()  # Returns tuple of values
        self.area_thresh_min = min_val / 100.0
        self.area_thresh_max = max_val / 100.0
        self.area_threshold_label.setText(f"{self.area_thresh_min:.2f} - {self.area_thresh_max:.2f}")

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
    
    def on_uncertainty_changed(self):
        """Update the slider and label when the shared data changes"""
        value = self.main_window.get_uncertainty_thresh()
        self.uncertainty_threshold_slider.setValue(int(value * 100))
        self.uncertainty_thresh = value
        
    def on_iou_changed(self):
        """Update the slider and label when the shared data changes"""
        value = self.main_window.get_iou_thresh()
        self.iou_threshold_slider.setValue(int(value * 100))
        self.iou_thresh = value 
        
    def load_model(self):
        """
        Load the segmentation model.
        """
        if not self.model_path:
            QMessageBox.warning(self, "Warning", "Please select a model file first")
            return
            
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.loaded_model = YOLO(self.model_path, task='segment')
            self.loaded_model(np.zeros((640, 640, 3), dtype=np.uint8))
            self.class_names = list(self.loaded_model.names.values())

            if not self.class_mapping:
                self.handle_missing_class_mapping()
            else:
                self.add_labels_to_label_window()
                self.check_and_display_class_names()
            
            # Update the status bar
            self.status_bar.setText(f"Model loaded: {os.path.basename(self.model_path)}")
            QMessageBox.information(self, "Model Loaded", "Model loaded successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()
            
    def predict(self, inputs=None):
        """
        Predict the segmentation results for the given image paths.

        :param inputs: List of image paths (optional)
        """
        if self.loaded_model is None:
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        if not inputs:
            # Predict only the current image
            inputs = [self.annotation_window.current_image_path]

        # Predict the segmentation results
        results = self.loaded_model(inputs,
                                    agnostic_nms=True,
                                    conf=self.get_uncertainty_threshold(),
                                    iou=self.get_iou_threshold(),
                                    device=self.main_window.device,
                                    stream=True)

        # Create a result processor
        results_processor = ResultsProcessor(self.main_window,
                                             self.class_mapping,
                                             uncertainty_thresh=self.get_uncertainty_threshold(),
                                             iou_thresh=self.get_iou_threshold(),
                                             min_area_thresh=self.area_thresh_min,
                                             max_area_thresh=self.area_thresh_max)
        # Check if SAM model is deployed
        if self.use_sam_dropdown.currentText() == "True":
            # Apply SAM to the segmentation results
            results = self.sam_dialog.predict_from_results(results, self.class_mapping)

        # Process the segmentation results
        results_processor.process_segmentation_results(results)

        QApplication.restoreOverrideCursor()
        gc.collect()
        empty_cache()