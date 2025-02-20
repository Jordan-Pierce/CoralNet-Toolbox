import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import gc
import os

import numpy as np

from superqt import QRangeSlider
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox, QLabel, QGroupBox, QFormLayout,
                             QComboBox, QSlider)

from torch.cuda import empty_cache
from ultralytics import YOLO, RTDETR

from coralnet_toolbox.MachineLearning.DeployModel.QtBase import Base

from coralnet_toolbox.ResultsProcessor import ResultsProcessor

from coralnet_toolbox.utilities import check_model_architecture


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Segment(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Deploy Segmentation Model")

        self.task = 'segment'

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
        Load the segmentation model.
        """
        if not self.model_path:
            QMessageBox.warning(self, "Warning", "Please select a model file first")
            return

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # Get the model architecture and task
            model_architecture, task = check_model_architecture(self.model_path)

            # Check if the model is supported
            if model_architecture == "yolo":
                self.loaded_model = YOLO(self.model_path)
            elif model_architecture == "rtdetr":
                self.loaded_model = RTDETR(self.model_path)
            else:
                QMessageBox.critical(self, "Error", "Model not currently supported.")
                return

            try:
                imgsz = self.loaded_model.__dict__['overrides']['imgsz']
            except:
                imgsz = 640

            self.loaded_model(np.zeros((imgsz, imgsz, 3), dtype=np.uint8))
            self.class_names = list(self.loaded_model.names.values())

            if not self.class_mapping:
                self.handle_missing_class_mapping()
            else:
                self.add_labels_to_label_window()

            # Display the class names
            self.check_and_display_class_names()

            # Update the status bar
            self.status_bar.setText(f"Model loaded: {os.path.basename(self.model_path)}")
            QMessageBox.information(self, "Model Loaded", "Model loaded successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

    def predict(self, image_paths=None):
        """
        Predict the segmentation results for the given inputs.
        """
        if self.loaded_model is None:
            return  # Early exit if there is no model loaded

        # Create a result processor
        results_processor = ResultsProcessor(
            self.main_window,
            self.class_mapping,
            uncertainty_thresh=self.main_window.get_uncertainty_thresh(),
            iou_thresh=self.main_window.get_iou_thresh(),
            min_area_thresh=self.main_window.get_area_thresh_min(),
            max_area_thresh=self.main_window.get_area_thresh_max()
        )

        # Use current image if image_paths is not provided.
        if not image_paths:
            image_paths = [self.annotation_window.current_image_path]

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # Loop through the image paths
            for image_path in image_paths:
                inputs = self._get_inputs(image_path)
                if inputs is None:
                    continue

                results = self._apply_model(inputs)
                results = self._apply_sam(results, image_path)
                results = self._apply_tile_postprocessing(results)
                self._process_results(results_processor, results)
        except Exception as e:
            print("An error occurred during prediction:", e)
        finally:
            QApplication.restoreOverrideCursor()

        gc.collect()
        empty_cache()

    def _get_inputs(self, image_path):
        """Get the inputs for the model prediction."""
        # Check if tile inference tool is enabled
        if self.main_window.tile_inference_tool_action.isChecked():
            inputs = self.main_window.tile_processor.make_crops(self.loaded_model, image_path)
            if not inputs:
                return None
        else:
            inputs = image_path
        return inputs

    def _apply_model(self, inputs):
        """Apply the model to the inputs."""
        return self.loaded_model(
            inputs,
            agnostic_nms=True,
            conf=self.main_window.get_uncertainty_thresh(),
            iou=self.main_window.get_iou_thresh(),
            device=self.main_window.device,
            stream=True
        )

    def _apply_sam(self, results, image_path):
        """Apply SAM to the results if needed."""
        # Check if SAM model is deployed
        if self.use_sam_dropdown.currentText() == "True":
            results = self.sam_dialog.predict_from_results(results, self.class_mapping, image_path)

        return results

    def _apply_tile_postprocessing(self, results):
        """Apply tile postprocessing if needed."""
        # Check if tile inference tool is enabled
        if self.main_window.tile_inference_tool_action.isChecked():
            results = self.main_window.tile_processor.detect_them(results, self.task == 'segment')
        return results

    def _process_results(self, result_processor, results):
        """Process the results using the result processor."""
        # Process the segmentations
        result_processor.process_segmentation_results(results)
