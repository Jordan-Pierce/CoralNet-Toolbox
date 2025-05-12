import warnings

import os
import gc
import ujson as json

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QFormLayout,
                             QHBoxLayout, QLabel, QMessageBox, QPushButton,
                             QSlider, QSpinBox, QVBoxLayout, QGroupBox, QTabWidget,
                             QWidget, QLineEdit, QFileDialog)

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
from ultralytics.models.yolo.yoloe import YOLOEVPDetectPredictor

from torch.cuda import empty_cache
from ultralytics.utils import ops

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
        info_label = QLabel("Choose a Predictor to deploy and use interactively with the See Anything tool.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_models_layout(self):
        """
        Setup the models layout with tabs for standard and custom models.
        """
        group_box = QGroupBox("Model Selection")
        layout = QVBoxLayout()

        # Create tabbed widget
        tab_widget = QTabWidget()

        # Tab 1: Standard models
        standard_tab = QWidget()
        standard_layout = QVBoxLayout(standard_tab)

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

        standard_layout.addWidget(QLabel("Models"))
        standard_layout.addWidget(self.model_combo)

        tab_widget.addTab(standard_tab, "Use Existing Model")

        # Tab 2: Custom model
        custom_tab = QWidget()
        custom_layout = QFormLayout(custom_tab)

        # Custom model file selection
        self.model_path_edit = QLineEdit()
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_model_file)

        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(self.model_path_edit)
        model_path_layout.addWidget(browse_button)
        custom_layout.addRow("Custom Model:", model_path_layout)

        # Class Mapping
        self.mapping_edit = QLineEdit()
        self.mapping_button = QPushButton("Browse...")
        self.mapping_button.clicked.connect(self.browse_class_mapping_file)

        class_mapping_layout = QHBoxLayout()
        class_mapping_layout.addWidget(self.mapping_edit)
        class_mapping_layout.addWidget(self.mapping_button)
        custom_layout.addRow("Class Mapping:", class_mapping_layout)

        tab_widget.addTab(custom_tab, "Custom Model")

        # Add the tab widget to the main layout
        layout.addWidget(tab_widget)

        # Store the tab widget for later reference
        self.model_tab_widget = tab_widget

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
            self.model_path_edit.setText(file_path)

            # Load the class mapping if it exists
            dir_path = os.path.dirname(os.path.dirname(file_path))
            class_mapping_path = f"{dir_path}/class_mapping.json"
            if os.path.exists(class_mapping_path):
                self.class_mapping = json.load(open(class_mapping_path, 'r'))
                self.mapping_edit.setText(class_mapping_path)

    def browse_class_mapping_file(self):
        """
        Browse and select a class mapping file.
        """
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Select Class Mapping File",
                                                   "",
                                                   "JSON Files (*.json)")
        if file_path:
            # Load the class mapping
            self.class_mapping = json.load(open(file_path, 'r'))
            self.mapping_edit.setText(file_path)

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

            # Create a dummy visual dictionary
            visuals = dict(
                bboxes=np.array(
                    [
                        [120, 425, 160, 445],
                    ],
                ),
                cls=np.array(
                    np.zeros(1),
                ),
            )

            # Run a dummy prediction to load the model
            self.loaded_model.predict(
                np.zeros((640, 640, 3), dtype=np.uint8),
                visual_prompts=visuals.copy(),
                predictor=YOLOEVPDetectPredictor,
                imgsz=640,
                conf=0.99,
            )

            # Load the model class names if available
            if self.class_mapping:
                self.add_labels_to_label_window()

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

    def add_labels_to_label_window(self):
        """
        Add labels to the label window based on the class mapping.
        """
        if self.class_mapping:
            for label in self.class_mapping.values():
                self.main_window.label_window.add_label_if_not_exists(label['short_label_code'],
                                                                      label['long_label_code'],
                                                                      QColor(*label['color']))

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

        # Preprocess the image
        # image = preprocess_image(image)

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

        # Set the predictor
        self.task = self.task_dropdown.currentText()
        predictor = YOLOEVPSegPredictor if self.task == "segment" else YOLOEVPDetectPredictor

        try:
            # Make predictions
            results = self.loaded_model.predict(self.resized_image,
                                                visual_prompts=visuals.copy(),
                                                predictor=predictor,
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

    def predict_from_annotations(self, refer_image, refer_label, refer_annotations, target_images):
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

        # Create a visual dictionary
        visuals = {
            'bboxes': np.array(refer_annotations),
            'cls': np.zeros(len(refer_annotations))
        }

        # Set the predictor
        self.task = self.task_dropdown.currentText()
        predictor = YOLOEVPSegPredictor if self.task == "segment" else YOLOEVPDetectPredictor

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
                                                    visual_prompts=visuals.copy(),
                                                    predictor=predictor,
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
