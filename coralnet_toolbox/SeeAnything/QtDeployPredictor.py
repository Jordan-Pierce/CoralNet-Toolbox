import warnings

import os
import gc

import numpy as np

import torch
from torch.cuda import empty_cache
from ultralytics.utils import ops

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
from ultralytics.models.yolo.yoloe import YOLOEVPDetectPredictor

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QFormLayout,
                             QHBoxLayout, QLabel, QMessageBox, QPushButton,
                             QSlider, QSpinBox, QVBoxLayout, QGroupBox,
                             QWidget, QLineEdit, QFileDialog)

from coralnet_toolbox.Results import ResultsProcessor

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon

from coralnet_toolbox.utilities import rasterio_to_numpy

warnings.filterwarnings("ignore", category=DeprecationWarning)


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

        self.task = "detect"
        self.max_detect = 500
        self.model_path = None
        self.loaded_model = None
        self.image_path = None

        self.class_mapping = {}

        # Create the layout
        self.layout = QVBoxLayout(self)

        # Setup the info layout
        self.setup_info_layout()
        # Setup the model layout
        self.setup_models_layout()
        # Setup the parameter layout
        self.setup_parameters_layout()
        # Setup the SAM layout
        self.setup_sam_layout()
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
        info_label = QLabel(
            "Choose a Predictor to deploy and use interactively with the See Anything tool. "
            "Optionally include a custom visual prompt encoding (VPE) file."
        )

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_models_layout(self):
        """
        Setup the models layout with standard models and file selection.
        """
        group_box = QGroupBox("Model Selection")
        layout = QFormLayout()
    
        # Model dropdown
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
    
        # Define available models
        standard_models = [
            'yoloe-v8s-seg.pt',
            'yoloe-v8m-seg.pt',
            'yoloe-v8l-seg.pt',
            'yoloe-11s-seg.pt',
            'yoloe-11m-seg.pt',
            'yoloe-11l-seg.pt',
        ]
    
        # Add all models to combo box
        self.model_combo.addItems(standard_models)
        
        # Set the default model
        self.model_combo.setCurrentIndex(standard_models.index('yoloe-v8s-seg.pt'))
        # Create a layout for the model selection
        layout.addRow("Models:", self.model_combo)
        
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
        layout.addRow("Task", self.task_dropdown)

        # Max detections spinbox
        self.max_detections_spinbox = QSpinBox()
        self.max_detections_spinbox.setRange(1, 10000)
        self.max_detections_spinbox.setValue(self.max_detect)
        label = QLabel("Max Detections")
        layout.addRow(label, self.max_detections_spinbox)

        # Resize image dropdown
        self.resize_image_dropdown = QComboBox()
        self.resize_image_dropdown.addItems(["True", "False"])
        self.resize_image_dropdown.setCurrentIndex(0)
        self.resize_image_dropdown.setEnabled(False)
        layout.addRow("Resize Image", self.resize_image_dropdown)

        # Image size control
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(512, 65536)
        self.imgsz_spinbox.setSingleStep(256)
        self.imgsz_spinbox.setValue(self.imgsz)
        layout.addRow("Image Size (imgsz)", self.imgsz_spinbox)

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

    def setup_sam_layout(self):
        """Use SAM model for segmentation."""
        group_box = QGroupBox("Use SAM Model for Creating Polygons")
        layout = QFormLayout()

        # SAM dropdown
        self.use_sam_dropdown = QComboBox()
        self.use_sam_dropdown.addItems(["False", "True"])
        self.use_sam_dropdown.currentIndexChanged.connect(self.is_sam_model_deployed)
        layout.addRow("Use SAM Polygons:", self.use_sam_dropdown)

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

    def get_max_detections(self):
        """Get the maximum number of detections to return."""
        self.max_detect = self.max_detections_spinbox.value()
        return self.max_detect

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
    
            # Create a dummy visual dictionary for standard model loading
            visuals = dict(
                bboxes=np.array(
                    [
                        [120, 425, 160, 445],  # Random box
                    ],
                ),
                cls=np.array(
                    np.zeros(1),
                ),
            )
    
            # Run a dummy prediction to load the model
            self.loaded_model.predict(
                np.zeros((640, 640, 3), dtype=np.uint8),
                visual_prompts=visuals.copy(),  # This needs to happen to properly initialize the predictor
                predictor=YOLOEVPSegPredictor,  # This also needs to be SegPredictor, no matter what
                imgsz=640,
                conf=0.99,
            )

            self.status_bar.setText(f"Loaded ({self.model_path}")
            QMessageBox.information(self.annotation_window, "Model Loaded", "Model loaded successfully")

        except Exception as e:
            self.loaded_model = None
            self.status_bar.setText(f"Error loading model: {self.model_path}")
            QMessageBox.critical(self.annotation_window, "Error Loading Model", f"Error loading model: {e}")
    
        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()
            # Stop the progress bar
            progress_bar.stop_progress()
            progress_bar.close()
            progress_bar = None

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
        Ensures the maximum dimension is a multiple of 32.
        """
        h, w = image.shape[:2]

        # Round imgsz to the nearest multiple of 32
        imgsz = round(imgsz / 32) * 32

        if h > w:
            # Height is the longer side
            new_h = imgsz
            new_w = int(w * (new_h / h))
            # Make width a multiple of 32
            new_w = round(new_w / 32) * 32
        else:
            # Width is the longer side
            new_w = imgsz
            new_h = int(h * (new_w / w))
            # Make height a multiple of 32
            new_h = round(new_h / 32) * 32

        # Ensure neither dimension is zero
        new_h = max(32, new_h)
        new_w = max(32, new_w)

        return new_h, new_w

    def set_image(self, image, image_path):
        """
        Set the image in the predictor.
        """
        if image is None and image_path is not None:
            # Open the image using rasterio
            image = rasterio_to_numpy(self.main_window.image_window.rasterio_images[image_path])

        # Save the original image
        self.original_image = image
        self.image_path = image_path

        # Resize the image if the checkbox is checked
        if self.resize_image_dropdown.currentText() == "True":
            self.resized_image = self.resize_image(image)
        else:
            self.resized_image = image
            
    def scale_prompts(self, bboxes, masks=None):
        """
        Scale the bounding boxes and masks to the resized image.
        """
        # Update the bbox coordinates to be relative to the resized image
        bboxes = np.array(bboxes)
        bboxes[:, 0] = (bboxes[:, 0] / self.original_image.shape[1]) * self.resized_image.shape[1]
        bboxes[:, 1] = (bboxes[:, 1] / self.original_image.shape[0]) * self.resized_image.shape[0]
        bboxes[:, 2] = (bboxes[:, 2] / self.original_image.shape[1]) * self.resized_image.shape[1]
        bboxes[:, 3] = (bboxes[:, 3] / self.original_image.shape[0]) * self.resized_image.shape[0]

        # Set the predictor
        self.task = self.task_dropdown.currentText()

        # Create a visual dictionary
        visual_prompts = {
            'bboxes': np.array(bboxes),
            'cls': np.zeros(len(bboxes))
        }
        if self.task == 'segment':
            if masks:
                scaled_masks = []
                for mask in masks:
                    scaled_mask = np.array(mask, dtype=np.float32)
                    scaled_mask[:, 0] = (scaled_mask[:, 0] / self.original_image.shape[1]) * self.resized_image.shape[1]
                    scaled_mask[:, 1] = (scaled_mask[:, 1] / self.original_image.shape[0]) * self.resized_image.shape[0]
                    scaled_masks.append(scaled_mask)
                visual_prompts['masks'] = scaled_masks
            else:  # Fallback to creating masks from bboxes if no masks are provided
                fallback_masks = []
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    fallback_masks.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))
                visual_prompts['masks'] = fallback_masks
        
        return visual_prompts

    def predict_from_prompts(self, bboxes, masks=None):
        """
        Make predictions using the currently loaded model using prompts.

        Args:
            bboxes (np.ndarray): The bounding boxes to use as prompts.
            masks (list, optional): A list of polygons to use as prompts for segmentation.

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

        # Get the scaled visual prompts
        visual_prompts = self.scale_prompts(bboxes, masks)

        try:
            # Make predictions
            results = self.loaded_model.predict(self.resized_image,
                                                visual_prompts=visual_prompts.copy(),  
                                                predictor=YOLOEVPSegPredictor,
                                                imgsz=max(self.resized_image.shape[:2]),
                                                conf=self.main_window.get_uncertainty_thresh(),
                                                iou=self.main_window.get_iou_thresh(),
                                                max_det=self.get_max_detections(),
                                                retina_masks=self.task == "segment")

        except Exception as e:
            QMessageBox.critical(self.annotation_window,
                                 "Prediction Error",
                                 f"Error predicting: {e}")
            results = None

        finally:
            # Clear the cache
            gc.collect()
            empty_cache()

        return results

    def predict_from_annotations(self, refer_image, refer_label, refer_bboxes, refer_masks, target_images):
        """"""
        # Create a class mapping
        class_mapping = {0: refer_label}

        # Create a results processor
        results_processor = ResultsProcessor(
            self.main_window,
            class_mapping,
            uncertainty_thresh=self.main_window.get_uncertainty_thresh(),
            iou_thresh=self.main_window.get_iou_thresh(),
            min_area_thresh=self.main_window.get_area_thresh_min(),
            max_area_thresh=self.main_window.get_area_thresh_max()
        )

        # Get the scaled visual prompts
        visual_prompts = self.scale_prompts(refer_bboxes, refer_masks)

        # If VPEs are being used
        if self.vpe is not None:
            # Generate a new VPE from the current visual prompts
            new_vpe = self.prompts_to_vpes(visual_prompts, self.resized_image)
            
            if new_vpe is not None:
                # If we already have a VPE, average with the existing one
                if self.vpe.shape == new_vpe.shape:
                    self.vpe = (self.vpe + new_vpe) / 2
                    # Re-normalize
                    self.vpe = torch.nn.functional.normalize(self.vpe, p=2, dim=-1)
                else:
                    # Replace with the new VPE if shapes don't match
                    self.vpe = new_vpe
                
                # Set the updated VPE in the model
                self.loaded_model.is_fused = lambda: False
                self.loaded_model.set_classes(["object0"], self.vpe)
            
            # Clear visual prompts since we're using VPE
            visual_prompts = {}  # this is okay with a fused model

        # Create a progress bar
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Making Predictions")
        progress_bar.show()
        progress_bar.start_progress(len(target_images))

        for target_image in target_images:

            try:
                # Make predictions
                results = self.loaded_model.predict(target_image,
                                                    refer_image=refer_image,
                                                    visual_prompts=visual_prompts.copy(),
                                                    predictor=YOLOEVPSegPredictor,
                                                    imgsz=self.imgsz_spinbox.value(),
                                                    conf=self.main_window.get_uncertainty_thresh(),
                                                    iou=self.main_window.get_iou_thresh(),
                                                    max_det=self.get_max_detections(),
                                                    retina_masks=self.task == "segment")

                results[0].names = {0: refer_label.short_label_code}

                # Process the detections
                if self.task == 'segment':
                    results_processor.process_segmentation_results(results)
                else:
                    results_processor.process_detection_results(results)

            except Exception as e:
                print(f"Error predicting: {e}")

            finally:
                progress_bar.update_progress()
                # Clear the cache
                gc.collect()
                empty_cache()

        # Make cursor normal
        QApplication.restoreOverrideCursor()
        progress_bar.finish_progress()
        progress_bar.stop_progress()
        progress_bar.close()

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
