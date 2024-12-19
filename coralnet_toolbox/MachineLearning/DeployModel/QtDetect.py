import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import gc
import os

import numpy as np

from qtrangeslider import QRangeSlider
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox, QLabel, QGroupBox, QFormLayout, QComboBox, QSlider)

from torch.cuda import empty_cache
from ultralytics import YOLO

from coralnet_toolbox.MachineLearning.DeployModel.QtBase import Base

from coralnet_toolbox.ResultsProcessor import ResultsProcessor


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Detect(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Deploy Detection Model")
        
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
             
    def setup_parameters_layout(self):
        """
        Setup parameter control section in a group box.
        """
        group_box = QGroupBox("Parameters")
        layout = QFormLayout()

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
        self.area_threshold_slider = QRangeSlider(Qt.Horizontal)
        self.area_threshold_slider.setRange(0, 100)
        self.area_threshold_slider.setValue((self.area_thresh_min, self.area_thresh_max))
        self.area_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.area_threshold_slider.setTickInterval(10)
        self.area_threshold_slider.valueChanged.connect(self.update_area_label)
        self.area_threshold_label = QLabel(f"{self.area_thresh_min:.2f} - {self.area_thresh_max:.2f}")
        layout.addRow("Area Threshold", self.area_threshold_slider)
        layout.addRow("", self.area_threshold_label)
        
        # SAM dropdown
        self.use_sam_dropdown = QComboBox()
        self.use_sam_dropdown.addItems(["False", "True"])
        self.use_sam_dropdown.currentIndexChanged.connect(self.is_sam_model_deployed)
        layout.addRow("Use SAM for creating Polygons:", self.use_sam_dropdown)
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
    def load_model(self):
        """
        Load the detection model.
        """
        if not self.model_path:
            QMessageBox.warning(self, "Warning", "Please select a model file first")
            return
            
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.loaded_model = YOLO(self.model_path, task='detect')
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
        Predict the detection results for the given inputs.
        """
        if self.loaded_model is None:
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        if not inputs:
            # Predict only the current image
            inputs = [self.annotation_window.current_image_path]

        # Predict the detection results
        results = self.loaded_model(inputs,
                                    agnostic_nms=True,
                                    conf=self.main_window.get_uncertainty_thresh(),
                                    iou=self.main_window.get_iou_thresh(),
                                    device=self.main_window.device,
                                    stream=True)

        # Create a result processor
        results_processor = ResultsProcessor(self.main_window,
                                             self.class_mapping,
                                             uncertainty_thresh=self.main_window.get_uncertainty_thresh(),
                                             iou_thresh=self.main_window.get_iou_thresh(),
                                             min_area_thresh=self.main_window.get_area_thresh_min(),
                                             max_area_thresh=self.main_window.get_area_thresh_max())
        
        # Check if SAM model is deployed
        if self.use_sam_dropdown.currentText() == "True":
            # Apply SAM to the detection results
            results = self.sam_dialog.predict_from_results(results, self.class_mapping)
            # Process the segmentation results
            results_processor.process_segmentation_results(results)
        else:
            # Process the detection results
            results_processor.process_detection_results(results)

        QApplication.restoreOverrideCursor()
        gc.collect()
        empty_cache()