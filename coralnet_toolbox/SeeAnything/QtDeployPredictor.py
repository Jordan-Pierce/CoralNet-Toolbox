import warnings

import os
import gc

import cv2
import numpy as np

import torch
from torch.cuda import empty_cache

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
from ultralytics.models.yolo.yoloe import YOLOEVPDetectPredictor

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QFormLayout,
                             QHBoxLayout, QLabel, QMessageBox, QPushButton,
                             QSpinBox, QVBoxLayout, QGroupBox, QTabWidget,
                             QWidget, QLineEdit, QFileDialog)

from coralnet_toolbox.Results import ResultsProcessor

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Common import ThresholdsWidget

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
        self.task = "detect"
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
        # Setup the thresholds layout
        self.setup_thresholds_layout()
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
        self.thresholds_widget.initialize_thresholds()

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel(
            "Choose a Predictor to deploy and use interactively with the See Anything tool. "
        )

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_models_layout(self):
        """
        Setup the models layout with tabbed interface for model selection.
        """
        group_box = QGroupBox("Model Selection")
        layout = QVBoxLayout()

        # Create tabbed widget
        tab_widget = QTabWidget()

        # Tab 1: Select model from dropdown
        model_select_tab = QWidget()
        model_select_layout = QFormLayout(model_select_tab)

        # Model combo box
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
        model_select_layout.addRow("Model:", self.model_combo)

        tab_widget.addTab(model_select_tab, "Select Model")

        # Tab 2: Use existing model (custom weights)
        model_existing_tab = QWidget()
        model_existing_layout = QFormLayout(model_existing_tab)

        # Existing Model
        self.model_edit = QLineEdit()
        self.model_button = QPushButton("Browse...")
        self.model_button.clicked.connect(self.browse_model_file)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(self.model_button)
        model_existing_layout.addRow("Model File:", model_layout)

        tab_widget.addTab(model_existing_tab, "Use Existing Model")

        layout.addWidget(tab_widget)
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def browse_model_file(self):
        """
        Open a file dialog to browse for a model file.
        """
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Select Model File",
                                                   "",
                                                   "Model Files (*.pt *.pth);;All Files (*)")
        if file_path:
            self.model_edit.setText(file_path)

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

        # Resize image dropdown
        self.resize_image_dropdown = QComboBox()
        self.resize_image_dropdown.addItems(["True", "False"])
        self.resize_image_dropdown.setCurrentIndex(0)
        self.resize_image_dropdown.setEnabled(False)
        layout.addRow("Resize Image", self.resize_image_dropdown)

        # Image size control
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(512, 65536)
        self.imgsz_spinbox.setSingleStep(1024)
        self.imgsz_spinbox.setValue(self.imgsz)
        layout.addRow("Image Size (imgsz)", self.imgsz_spinbox)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_sam_layout(self):
        """Use SAM model for segmentation."""
        group_box = QGroupBox("Use SAM to Create Polygons")
        layout = QFormLayout()

        # SAM dropdown
        self.use_sam_dropdown = QComboBox()
        self.use_sam_dropdown.addItems(["False", "True"])
        self.use_sam_dropdown.currentIndexChanged.connect(self.is_sam_model_deployed)
        layout.addRow("Use SAM Polygons:", self.use_sam_dropdown)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
    def setup_thresholds_layout(self):
        """
        Setup thresholds control section in a group box.
        """
        # Add ThresholdsWidget for all threshold controls
        self.thresholds_widget = ThresholdsWidget(
            self.main_window,
            show_max_detections=True,
            show_uncertainty=True,
            show_iou=True,
            show_area=True
        )

        self.layout.addWidget(self.thresholds_widget)

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

    def is_sam_model_deployed(self):
        """
        Check if the SAM model is deployed and update the checkbox state accordingly.
        If SAM is enabled for polygons, sync and disable the imgsz spinbox.

        :return: Boolean indicating whether the SAM model is deployed
        """
        if not hasattr(self.main_window, 'sam_deploy_predictor_dialog'):
            return False

        self.sam_dialog = self.main_window.sam_deploy_predictor_dialog

        if not self.sam_dialog.loaded_model:
            self.use_sam_dropdown.setCurrentText("False")
            QMessageBox.critical(self, "Error", "Please deploy the SAM model first.")
            return False
        
        # Check if SAM polygons are enabled
        if self.use_sam_dropdown.currentText() == "True":
            # Sync the imgsz spinbox with SAM's value
            self.imgsz_spinbox.setValue(self.sam_dialog.imgsz_spinbox.value())
            # Disable the spinbox
            self.imgsz_spinbox.setEnabled(False)
            
            # Connect SAM's imgsz_spinbox valueChanged signal to update our value
            # First disconnect any existing connection to avoid duplicates
            try:
                self.sam_dialog.imgsz_spinbox.valueChanged.disconnect(self.update_from_sam_imgsz)
            except TypeError:
                # No connection exists yet
                pass
            
            # Connect the signal
            self.sam_dialog.imgsz_spinbox.valueChanged.connect(self.update_from_sam_imgsz)
        else:
            # Re-enable the spinbox when SAM polygons are disabled
            self.imgsz_spinbox.setEnabled(True)
            
            # Disconnect the signal when SAM is disabled
            try:
                self.sam_dialog.imgsz_spinbox.valueChanged.disconnect(self.update_from_sam_imgsz)
            except TypeError:
                # No connection exists
                pass

        return True

    def update_from_sam_imgsz(self, value):
        """
        Update the SeeAnything image size when SAM's image size changes.
        Only takes effect when SAM polygons are enabled.
        
        Args:
            value (int): The new image size value from SAM dialog
        """
        if self.use_sam_dropdown.currentText() == "True":
            self.imgsz_spinbox.setValue(value)

    def load_model(self):
        """
        Load the selected model (from dropdown or file).
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Loading Model")
        progress_bar.show()
    
        try:
            # Update the task
            self.task = self.task_dropdown.currentText()
            
            # Get model path - either from custom file or dropdown
            if self.model_edit.text().strip():
                # Use custom model file
                self.model_path = self.model_edit.text().strip()
            else:
                # Use selected model from dropdown
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
                predictor=YOLOEVPDetectPredictor if self.task == 'detect' else YOLOEVPSegPredictor,
                imgsz=640,
                conf=0.99,
            )
            # Finish the progress bar
            progress_bar.finish_progress()
            # Update the status bar
            self.status_bar.setText(f"Loaded ({os.path.basename(self.model_path)})")
            QMessageBox.information(self.annotation_window, "Model Loaded", "Model loaded successfully")

        except Exception as e:
            self.loaded_model = None
            self.status_bar.setText(f"Error loading model: {os.path.basename(self.model_path)}")
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
        return self.scale_image(image, target_shape)

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
    
    def scale_image(self, masks, im0_shape, ratio_pad=None):
        """
        Rescale masks to original image size.

        Takes resized and padded masks and rescales them back to the original image dimensions, removing any padding
        that was applied during preprocessing.

        Args:
            masks (np.ndarray): Resized and padded masks with shape [H, W, N] or [H, W, 3].
            im0_shape (tuple): Original image shape as HWC or HW (supports both).
            ratio_pad (tuple, optional): Ratio and padding values as ((ratio_h, ratio_w), (pad_h, pad_w)).

        Returns:
            (np.ndarray): Rescaled masks with shape [H, W, N] matching original image dimensions.
        """
        # Rescale coordinates (xyxy) from im1_shape to im0_shape
        im0_h, im0_w = im0_shape[:2]  # supports both HWC or HW shapes
        im1_h, im1_w, _ = masks.shape
        if im1_h == im0_h and im1_w == im0_w:
            return masks

        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_h / im0_h, im1_w / im0_w)  # gain  = old / new
            pad = (im1_w - im0_w * gain) / 2, (im1_h - im0_h * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        pad_w, pad_h = pad
        top = int(round(pad_h - 0.1))
        left = int(round(pad_w - 0.1))
        bottom = im1_h - int(round(pad_h + 0.1))
        right = im1_w - int(round(pad_w + 0.1))

        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(masks, (im0_w, im0_h))
        if len(masks.shape) == 2:
            masks = masks[:, :, None]

        return masks

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
        
        # Set the predictor
        predictor = YOLOEVPDetectPredictor if self.task == 'detect' else YOLOEVPSegPredictor

        try:
            # Make predictions
            results = self.loaded_model.predict(self.resized_image,
                                                visual_prompts=visual_prompts.copy(),
                                                predictor=predictor,
                                                imgsz=max(self.resized_image.shape[:2]),
                                                conf=self.thresholds_widget.get_uncertainty_thresh(),
                                                iou=self.thresholds_widget.get_iou_thresh(),
                                                max_det=self.thresholds_widget.get_max_detections(),
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
            class_mapping
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
        
        # Set the predictor
        predictor = YOLOEVPDetectPredictor if self.task == 'detect' else YOLOEVPSegPredictor

        for target_image in target_images:

            try:
                # Make predictions
                results = self.loaded_model.predict(target_image,
                                                    refer_image=refer_image,
                                                    visual_prompts=visual_prompts.copy(),
                                                    predictor=predictor,
                                                    imgsz=self.imgsz_spinbox.value(),
                                                    conf=self.thresholds_widget.get_uncertainty_thresh(),
                                                    iou=self.thresholds_widget.get_iou_thresh(),
                                                    max_det=self.thresholds_widget.get_max_detections(),
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
