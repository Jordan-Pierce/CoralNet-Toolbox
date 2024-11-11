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

from toolbox.MachineLearning.DeployModel.QtBase import Base

from toolbox.ResultsProcessor import ResultsProcessor
from toolbox.QtProgressBar import ProgressBar

from toolbox.utilities import pixmap_to_numpy


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Classify(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Deploy Classification Model")
        
    def setup_generic_layout(self):
        """
        Adopt the layout from the Base class
        """
        super().setup_generic_layout()        
        
    def load_model(self):
        """
        Load the classification model.
        """
        if not self.model_path:
            QMessageBox.warning(self, "Warning", "Please select a model file first")
            return
            
        try:
            self.loaded_model = YOLO(self.model_path, task='classify')
            self.loaded_model(np.zeros((224, 224, 3), dtype=np.uint8))
            self.status_bar.setText("Model loaded successfully")
            self.check_and_display_class_names()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            
    def predict(self, inputs=None):
        """
        Predict the classification results for the given annotations.

        :param inputs: List of annotations (optional)
        """
        if self.loaded_model is None:
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        if not inputs:
            # Predict only the selected annotation
            inputs = self.annotation_window.selected_annotations
        if not inputs:
            # If no annotations are selected, predict all annotations in the image
            inputs = self.annotation_window.get_image_review_annotations()

        images_np = []
        for annotation in inputs:
            images_np.append(pixmap_to_numpy(annotation.cropped_image))

        # Predict the classification results
        results = self.loaded_model(images_np,
                                    device=self.main_window.device,
                                    stream=True)
        # Create a result processor
        results_processor = ResultsProcessor(self.main_window,
                                             self.class_mapping)

        # Process the classification results
        results_processor.process_classification_results(results, inputs)

        # Make cursor normal
        QApplication.restoreOverrideCursor()
        gc.collect()
        empty_cache()