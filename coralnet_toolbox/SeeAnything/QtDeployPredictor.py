import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import gc
import os

import numpy as np
import torch

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QFormLayout,
                             QHBoxLayout, QLabel, QMessageBox, QPushButton,
                             QSlider, QSpinBox, QVBoxLayout, QGroupBox)

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
from ultralytics.models.yolo.yoloe import YOLOEVPDetectPredictor

from torch.cuda import empty_cache
from ultralytics.utils import ops

from coralnet_toolbox.ResultsProcessor import ResultsProcessor

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon

from coralnet_toolbox.utilities import preprocess_image
from coralnet_toolbox.utilities import rasterio_to_numpy


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DeployPredictorDialog(QDialog):
    def __init__(self, main_window, parent=None):
        """Initialize the SeeAnything Deploy Model dialog."""
        super().__init__(parent)
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_icon("eye.png"))
        self.setWindowTitle("See Anything Deploy Model")
        self.resize(400, 325)

        # Initialize instance variables
        self.imgsz = 1024
        self.iou_thresh = 0.20
        self.uncertainty_thresh = 0.30
        self.area_thresh_min = 0.00
        self.area_thresh_max = 0.40
        self.model_path = None
        self.loaded_model = None
        self.image_path = None
        self.task = "segment"

        # Create the layout
        self.layout = QVBoxLayout(self)

        # Setup the info layout
        self.setup_info_layout()
        # Setup the model layout
        self.setup_models_layout()
        # Setup the parameter layout
        self.setup_parameters_layout()
        # Setup the buttons layout
        self.setup_buttons_layout()
        # Setup the status layout
        self.setup_status_layout()

    def showEvent(self, event):
        """
        Handle the show event to update label options and sync uncertainty threshold.

        Args:
            event: The event object.
        """
        super().showEvent(event)
        self.initialize_uncertainty_threshold()
        self.initialize_iou_threshold()
        self.initialize_area_threshold()

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Choose a Predictor to deploy and use interactively with the See Anything tool.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_models_layout(self):
        """
        Setup the models layout.
        """
        group_box = QGroupBox("Models")
        layout = QVBoxLayout()

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)

        # Define available models
        models = [
            'yoloe-v8s-seg.pt',
            'yoloe-v8m-seg.pt',
            'yoloe-v8l-seg.pt',
            'yoloe-11s-seg.pt',
            'yoloe-11m-seg.pt',
            'yoloe-11l-seg.pt',
        ]

        # Add all models to combo box
        self.model_combo.addItems(models)

        layout.addWidget(QLabel("Select Model:"))
        layout.addWidget(self.model_combo)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_parameters_layout(self):
        """
        Setup parameter control section in a group box.
        """
        group_box = QGroupBox("Parameters")
        layout = QFormLayout()
        
        # Task dropdown
        self.task_dropdown = QComboBox()
        self.task_dropdown.addItems(["detect", "segment"])
        layout.addRow("Task:", self.task_dropdown)
        
        # Resize image dropdown
        self.resize_image_dropdown = QComboBox()
        self.resize_image_dropdown.addItems(["True", "False"])
        self.resize_image_dropdown.setCurrentIndex(0)
        layout.addRow("Resize Image:", self.resize_image_dropdown)

        # Image size control
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(512, 65536)
        self.imgsz_spinbox.setSingleStep(256)
        self.imgsz_spinbox.setValue(self.imgsz)
        layout.addRow("Image Size (imgsz):", self.imgsz_spinbox)

        # Uncertainty threshold controls
        self.uncertainty_thresh = self.main_window.get_uncertainty_thresh()
        self.uncertainty_threshold_slider = QSlider(Qt.Horizontal)
        self.uncertainty_threshold_slider.setRange(0, 100)
        self.uncertainty_threshold_slider.setValue(int(self.main_window.get_uncertainty_thresh() * 100))
        self.uncertainty_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.uncertainty_threshold_slider.setTickInterval(10)
        self.uncertainty_threshold_slider.valueChanged.connect(self.update_uncertainty_label)
        self.uncertainty_threshold_label = QLabel(f"{self.uncertainty_thresh:.2f}")
        layout.addRow("Uncertainty Threshold", self.uncertainty_threshold_slider)
        layout.addRow("", self.uncertainty_threshold_label)

        # IoU threshold controls
        self.iou_thresh = self.main_window.get_iou_thresh()
        self.iou_threshold_slider = QSlider(Qt.Horizontal)
        self.iou_threshold_slider.setRange(0, 100)
        self.iou_threshold_slider.setValue(int(self.iou_thresh * 100))
        self.iou_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.iou_threshold_slider.setTickInterval(10)
        self.iou_threshold_slider.valueChanged.connect(self.update_iou_label)
        self.iou_threshold_label = QLabel(f"{self.iou_thresh:.2f}")
        layout.addRow("IoU Threshold", self.iou_threshold_slider)
        layout.addRow("", self.iou_threshold_label)

        # Area threshold controls
        min_val, max_val = self.main_window.get_area_thresh()
        self.area_thresh_min = int(min_val * 100)
        self.area_thresh_max = int(max_val * 100)
        self.area_threshold_min_slider = QSlider(Qt.Horizontal)
        self.area_threshold_min_slider.setRange(0, 100)
        self.area_threshold_min_slider.setValue(self.area_thresh_min)
        self.area_threshold_min_slider.setTickPosition(QSlider.TicksBelow)
        self.area_threshold_min_slider.setTickInterval(10)
        self.area_threshold_min_slider.valueChanged.connect(self.update_area_label)
        self.area_threshold_max_slider = QSlider(Qt.Horizontal)
        self.area_threshold_max_slider.setRange(0, 100)
        self.area_threshold_max_slider.setValue(self.area_thresh_max)
        self.area_threshold_max_slider.setTickPosition(QSlider.TicksBelow)
        self.area_threshold_max_slider.setTickInterval(10)
        self.area_threshold_max_slider.valueChanged.connect(self.update_area_label)
        self.area_threshold_label = QLabel(f"{self.area_thresh_min:.2f} - {self.area_thresh_max:.2f}")
        layout.addRow("Area Threshold Min", self.area_threshold_min_slider)
        layout.addRow("Area Threshold Max", self.area_threshold_max_slider)
        layout.addRow("", self.area_threshold_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_buttons_layout(self):
        """
        Setup action buttons in a group box.
        """
        group_box = QGroupBox("Actions")
        layout = QHBoxLayout()

        load_button = QPushButton("Load Model")
        load_button.clicked.connect(self.load_model)
        layout.addWidget(load_button)

        deactivate_button = QPushButton("Deactivate Model")
        deactivate_button.clicked.connect(self.deactivate_model)
        layout.addWidget(deactivate_button)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_status_layout(self):
        """
        Setup status display in a group box.
        """
        group_box = QGroupBox("Status")
        layout = QVBoxLayout()

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

    def load_model(self):
        """
        Load the selected model.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Loading Model")
        progress_bar.show()

        try:
            # Get selected model path and download weights if needed
            self.model_path = self.model_combo.currentText()

            # Load model using registry
            self.loaded_model = YOLOE(self.model_path).to(self.main_window.device)
            
            progress_bar.finish_progress()
            self.status_bar.setText("Model loaded")
            QMessageBox.information(self.annotation_window, "Model Loaded", "Model loaded successfully")

        except Exception as e:
            QMessageBox.critical(self.annotation_window, "Error Loading Model", f"Error loading model: {e}")

        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()
            # Stop the progress bar
            progress_bar.stop_progress()
            progress_bar.close()
            progress_bar = None
            
        self.accept()
        
    def resize_image(self, image):
        """
        Resize the image to the specified size.
        """
        imgsz = self.imgsz_spinbox.value()
        target_shape = self.get_target_shape(image, imgsz)
        return ops.scale_image(image, target_shape)

    def get_target_shape(self, image, imgsz):
        """
        Determine the target shape based on the long side.
        """
        h, w = image.shape[:2]
        if h > w:
            return imgsz, int(w * (imgsz / h))
        else:
            return int(h * (imgsz / w)), imgsz

    def set_image(self, image, image_path):
        """
        Set the image in the predictor.
        """
        if image is None and image_path is not None:
            # Open the image using rasterio
            image = self.main_window.image_window.rasterio_open(image_path)
            image = rasterio_to_numpy(image)

        # Preprocess the image
        image = preprocess_image(image)

        # Save the original image
        self.original_image = image
        self.image_path = image_path

        # Resize the image if the checkbox is checked
        if self.resize_image_dropdown.currentText() == "True":
            self.resized_image = self.resize_image(image)
        else:
            self.resized_image = image
            
    def predict_from_prompts(self, bboxes):
        """
        Make predictions using the currently loaded model using prompts.

        Args:
            bbox (np.ndarray): The bounding boxes to use as prompts.

        Returns:
            results (Results): Ultralytics Results object
        """
        if not self.loaded_model:
            QMessageBox.critical(self.annotation_window,
                                 "Model Not Loaded",
                                 "Model not loaded, cannot make predictions")
            return None
        
        if not len(bboxes):
            return None
        
        # Update the bbox coordinates to be relative to the resized image
        bboxes = np.array(bboxes)
        bboxes[:, 0] = (bboxes[:, 0] / self.original_image.shape[1]) * self.resized_image.shape[1]
        bboxes[:, 1] = (bboxes[:, 1] / self.original_image.shape[0]) * self.resized_image.shape[0]
        bboxes[:, 2] = (bboxes[:, 2] / self.original_image.shape[1]) * self.resized_image.shape[1]
        bboxes[:, 3] = (bboxes[:, 3] / self.original_image.shape[0]) * self.resized_image.shape[0]
        
        # Create a visual dictionary
        visuals = {
            'bboxes': np.array(bboxes),
            'cls': np.zeros(len(bboxes))  # TODO figure this out
        }
        
        try:
            # Set the predictor
            self.task = self.task_dropdown.currentText()
            predictor = YOLOEVPSegPredictor if self.task == "segment" else YOLOEVPDetectPredictor
            
            results = self.loaded_model.predict(self.resized_image,
                                                visual_prompts=visuals.copy(),
                                                predictor=predictor,
                                                conf=self.main_window.get_uncertainty_thresh(),
                                                iou=self.main_window.get_iou_thresh())

        except Exception as e:
            QMessageBox.critical(self.annotation_window,
                                 "Prediction Error",
                                 f"Error predicting: {e}")
            return None

        return results

    def deactivate_model(self):
        """
        Deactivate the currently loaded model.
        """
        # Clear the model
        self.loaded_model = None
        self.model_path = None
        self.image_path = None
        self.original_image = None
        self.resized_image = None
        # Clear the cache
        gc.collect()
        empty_cache()
        # Untoggle all tools
        self.main_window.untoggle_all_tools()
        # Update the status bar
        self.status_bar.setText("No model loaded")
        QMessageBox.information(self.annotation_window, "Model Deactivated", "Model deactivated")
