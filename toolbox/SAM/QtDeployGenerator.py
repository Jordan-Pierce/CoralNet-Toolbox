import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import gc

import numpy as np

import torch
from ultralytics import FastSAM

from qtrangeslider import QRangeSlider
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                             QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                             QMessageBox, QPushButton, QSlider, QSpinBox,
                             QVBoxLayout, QGroupBox)

from torch.cuda import empty_cache

from toolbox.QtProgressBar import ProgressBar
from toolbox.ResultsProcessor import ResultsProcessor
from toolbox.utilities import rasterio_to_numpy
from toolbox.utilities import attempt_download_asset


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DeployGeneratorDialog(QDialog):
    """
    Dialog for deploying FastSAM.
    Allows users to load, configure, and deactivate models, as well as make predictions on images.
    """

    def __init__(self, main_window, parent=None):
        """
        Initialize the AutoDistillDeployModelDialog.

        Args:
            main_window: The main application window.
            parent: The parent widget, default is None.
        """
        super().__init__(parent)
        self.main_window = main_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window
        
        self.setWindowTitle("FastSAM Generator")
        self.resize(400, 325)

        # Initialize variables
        self.imgsz = 1024
        self.iou_thresh = 0.70
        self.uncertainty_thresh = 0.30
        self.area_thresh_min = 0.01
        self.area_thresh_max = 0.75
        self.loaded_model = None
        self.model_path = None
        self.class_mapping = {0: 'Review'}

        # Create the layout
        self.layout = QVBoxLayout(self)
        
        # Setup the model layout
        self.setup_models_layout()
        # Setup the parameter layout
        self.setup_parameters_layout()
        # Setup the status layout
        self.setup_status_layout()
        # Setup the buttons layout
        self.setup_buttons_layout()
        
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

    def setup_models_layout(self):
        """
        Setup model selection dropdown in a group box.
        """
        group_box = QGroupBox("Model Selection")
        layout = QVBoxLayout()
        
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["FastSAM-s.pt", "FastSAM-x.pt"])
        layout.addWidget(self.model_dropdown)
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_parameters_layout(self):
        """
        Setup parameter control section in a group box.
        """
        group_box = QGroupBox("Parameters")
        form_layout = QFormLayout()
        
        # Resize image dropdown
        self.resize_image_dropdown = QComboBox()
        self.resize_image_dropdown.addItems(["True", "False"])
        self.resize_image_dropdown.setCurrentIndex(0)
        form_layout.addRow("Resize Image:", self.resize_image_dropdown)
        
        # Image size control
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(512, 4096)
        self.imgsz_spinbox.setSingleStep(1024)
        self.imgsz_spinbox.setValue(self.imgsz)
        form_layout.addRow("Image Size (imgsz):", self.imgsz_spinbox)
        
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
        
        group_box.setLayout(form_layout)
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
        Load the selected model with the current configuration.
        """
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        # Show a progress bar
        progress_bar = ProgressBar(self.annotation_window, title="Loading Model")
        progress_bar.show()
        try:
            # Get selected model path
            self.model_path = self.model_dropdown.currentText()

            # Load the model
            self.loaded_model = FastSAM(self.model_path)
            # Run a blank through the model to initialize it
            self.loaded_model(np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8))

            self.status_bar.setText(f"Model loaded: {self.model_path}")
            QMessageBox.information(self, "Model Loaded", "Model loaded successfully")

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Model", str(e))

        # Stop the progress bar
        progress_bar.stop_progress()
        progress_bar.close()
        # Restore cursor
        QApplication.restoreOverrideCursor()
        # Exit the dialog box
        self.accept()
    
    def get_area_thresh(self, image):
        """
        Calculate area thresholds based on image dimensions.

        Args:
            image_path: Path to the image.

        Returns:
            Tuple of (min_area_thresh, max_area_thresh).
        """
        h, w, _ = image.shape
        area_thresh_min = (h * w) * self.area_thresh_min
        area_thresh_max = (h * w) * self.area_thresh_max
        return area_thresh_min, area_thresh_max

    def get_uncertainty_threshold(self):
        """
        Get the uncertainty threshold, limiting it to a maximum of 0.10.

        Returns:
            Adjusted uncertainty threshold value.
        """
        if self.main_window.get_uncertainty_thresh() < 0.10:
            return self.main_window.get_uncertainty_thresh()
        else:
            return 0.10  # Arbitrary value to prevent too many detections
        
    def get_iou_threshold(self):
        """
        Get the IoU threshold for predictions.

        Returns:
            IoU threshold value.
        """
        return self.main_window.get_iou_thresh()

    def predict(self, image_paths=None):
        """
        Make predictions on the given image paths using the loaded model.

        Args:
            image_paths: List of image paths to process. If None, uses the current image.
        """
        if not self.loaded_model:
            QMessageBox.critical(self, "Error", "No model loaded")
            return
   
        if not image_paths:
            image_paths = [self.annotation_window.current_image_path]
            
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
            
        progress_bar = ProgressBar(self.annotation_window, title=f"Making FastSAM Predictions")
        progress_bar.show()
        progress_bar.start_progress(len(image_paths))

        for image_path in image_paths:

            # Predict the image
            results = self.loaded_model(image_path, 
                                        retina_masks=True, 
                                        imgsz=self.imgsz, 
                                        conf=self.get_uncertainty_threshold(), 
                                        iou=self.get_iou_threshold(),
                                        device=self.main_window.device)
            
            # Update the results names
            results[0].names = self.class_mapping
            
            # Create a results processor
            results_processor = ResultsProcessor(self.main_window, 
                                                 self.class_mapping,
                                                 uncertainty_thresh=self.get_uncertainty_threshold(),
                                                 iou_thresh=self.get_iou_threshold(),
                                                 min_area_thresh=self.area_thresh_min,
                                                 max_area_thresh=self.area_thresh_max)
                        
            # Update the progress bar
            progress_bar.update_progress()

            # Process the results
            results_processor.process_segmentation_results(results)
                        
        # Stop the progress bar
        progress_bar.stop_progress()
        progress_bar.close()
                
        # Make cursor normal
        QApplication.restoreOverrideCursor()
        gc.collect()
        empty_cache()

    def deactivate_model(self):
        """
        Deactivate the currently loaded model and clean up resources.
        """
        self.loaded_model = None
        self.model_path = None
        gc.collect()
        torch.cuda.empty_cache()
        self.main_window.untoggle_all_tools()
        self.status_bar.setText("No model loaded")
        QMessageBox.information(self, "Model Deactivated", "Model deactivated")