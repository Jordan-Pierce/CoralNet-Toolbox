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
                             QLabel, QDialog, QTextEdit, QPushButton, QGroupBox, QCheckBox)

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
        
    def setup_generic_layout(self):
        """
        Adopt the layout from the Base class
        """
        super().setup_generic_layout()
        
        # Add a new grouping 
        self.layout.addWidget(QLabel("Parameters"))
        
        # Add the SAM checkbox       
        use_sam_checkbox = QCheckBox("Use SAM for creating Polygons")
        use_sam_checkbox.stateChanged.connect(self.is_sam_model_deployed)
        self.layout.addWidget(use_sam_checkbox)
        self.use_sam = use_sam_checkbox 
        
    def load_model(self):
        """
        Load the segmentation model.
        """
        if not self.model_path:
            QMessageBox.warning(self, "Warning", "Please select a model file first")
            return
            
        try:
            self.loaded_model = YOLO(self.model_path, task='segment')
            self.loaded_model(np.zeros((640, 640, 3), dtype=np.uint8))
            self.status_bar.setText("Model loaded successfully")
            self.check_and_display_class_names()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            
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
                                    conf=self.get_confidence_threshold(),
                                    iou=self.get_iou_thresh(),
                                    device=self.main_window.device,
                                    stream=True)

        # Create a result processor
        results_processor = ResultsProcessor(self.main_window,
                                             self.class_mapping)
        # Check if SAM model is deployed
        if self.use_sam.isChecked():
            # Apply SAM to the segmentation results
            results = self.sam_dialog.predict_from_results(results, self.class_mapping)

        # Process the segmentation results
        results_processor.process_segmentation_results(results)

        QApplication.restoreOverrideCursor()
        gc.collect()
        empty_cache()