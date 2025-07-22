import warnings

import os
import gc

import numpy as np

import torch
from torch.cuda import empty_cache

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
from ultralytics.models.yolo.yoloe import YOLOEVPDetectPredictor

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox, QCheckBox, QVBoxLayout,
                             QLabel, QDialog, QDialogButtonBox, QGroupBox, QButtonGroup,
                             QFormLayout, QComboBox, QSpinBox, QSlider, QPushButton,
                             QHBoxLayout)

from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

from coralnet_toolbox.Results import ResultsProcessor
from coralnet_toolbox.Results import MapResults

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DeployGeneratorDialog(QDialog):
    """
    Perform See Anything (YOLOE) on multiple images using a reference image and label.

    :param main_window: MainWindow object
    :param parent: Parent widget
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.label_window = main_window.label_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window
        self.sam_dialog = None

        self.setWindowIcon(get_icon("eye.png"))
        self.setWindowTitle("See Anything (YOLOE) Generator (Ctrl + 5)")
        self.resize(600, 100)

        self.deploy_model_dialog = None
        self.loaded_model = None
        
        # Initialize variables
        self.imgsz = 1024
        self.iou_thresh = 0.20
        self.uncertainty_thresh = 0.30
        self.area_thresh_min = 0.00
        self.area_thresh_max = 0.40

        self.task = 'detect'
        self.max_detect = 300
        self.loaded_model = None
        self.model_path = None
        self.class_mapping = None

        # Reference image and label
        self.source_images = []
        self.source_label = None
        # Target images
        self.target_images = []

        self.layout = QVBoxLayout(self)

        # Setup the info layout
        self.setup_info_layout()
        # Setup the source layout
        self.setup_source_layout()
        # Setup the image options layout
        self.setup_options_layout()
        # Setup the buttons layout
        self.setup_buttons_layout()

    def showEvent(self, event):
        """
        Set up the layout when the dialog is shown.

        :param event: Show event
        """
        super().showEvent(event)
        self.initialize_uncertainty_threshold()
        self.initialize_iou_threshold()
        self.initialize_area_threshold()
        self.update_detect_as_combo()
        
        self.deploy_model_dialog = self.main_window.see_anything_deploy_predictor_dialog
        self.loaded_model = self.deploy_model_dialog.loaded_model
        
        # Update the source images (now assuming sources are valid)
        self.update_source_images()

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Choose a Generator to deploy and use.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
    def setup_models_layout(self):
        """
        Setup model selection dropdown in a group box.
        """
        group_box = QGroupBox("Models")
        layout = QVBoxLayout()

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)

        # Define available models
        self.models = {
            "YOLOE-8S": "yoloe-v8s-seg.pt",
            "YOLOE-8M": "yoloe-v8m-seg.pt",
            "YOLOE-8L": "yoloe-v8l-seg.pt",
            "YOLOE-11S": "yoloe-11s-seg.pt",
            "YOLOE-11M": "yoloe-11m-seg.pt",
            "YOLOE-11L": "yoloe-11l-seg.pt",
        }

        # Add all models to combo box
        for model_name in self.models.keys():
            self.model_combo.addItem(model_name)
            
        # Set the default model
        self.model_combo.setCurrentText("YOLOE-8S")

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
        self.use_task_dropdown = QComboBox()
        self.use_task_dropdown.addItems(["detect", "segment"])
        self.use_task_dropdown.currentIndexChanged.connect(self.update_task)
        layout.addRow("Task:", self.use_task_dropdown)

        # Max detections spinbox
        self.max_detections_spinbox = QSpinBox()
        self.max_detections_spinbox.setRange(1, 10000)
        self.max_detections_spinbox.setValue(self.max_detect)
        layout.addRow("Max Detections:", self.max_detections_spinbox)

        # Resize image dropdown
        self.resize_image_dropdown = QComboBox()
        self.resize_image_dropdown.addItems(["True", "False"])
        self.resize_image_dropdown.setCurrentIndex(0)
        self.resize_image_dropdown.setEnabled(False)  # Grey out the dropdown
        layout.addRow("Resize Image:", self.resize_image_dropdown)

        # Image size control
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(512, 65536)
        self.imgsz_spinbox.setSingleStep(1024)
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
        
    def detect_as_layout(self):
        """Detect objects as layout."""
        group_box = QGroupBox("Detect as: ")
        layout = QFormLayout()

        # Sample Label
        self.detect_as_combo = QComboBox()
        for label in self.label_window.labels:
            self.detect_as_combo.addItem(label.short_label_code, label.id)
        self.detect_as_combo.setCurrentIndex(0)
        self.detect_as_combo.currentIndexChanged.connect(self.update_class_mapping)
        layout.addRow("Detect as:", self.detect_as_combo)

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

    def setup_source_layout(self):
        """
        Set up the layout with source image and label selection.
        Contains dropdown combo boxes for selecting the source image and label.
        """
        group_box = QGroupBox("Source Selection")
        layout = QFormLayout()

        # Create the source image combo box
        self.source_image_combo_box = QComboBox()
        self.source_image_combo_box.currentIndexChanged.connect(self.update_source_labels)
        layout.addRow("Source Image:", self.source_image_combo_box)

        # Create the source label combo box
        self.source_label_combo_box = QComboBox()
        layout.addRow("Source Label:", self.source_label_combo_box)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_options_layout(self):
        """
        Set up the layout with image options.
        """
        # Create a group box for image options
        group_box = QGroupBox("Image Options")
        layout = QVBoxLayout()

        # Create a button group for the image checkboxes
        image_options_group = QButtonGroup(self)

        self.apply_filtered_checkbox = QCheckBox("▼ Apply to filtered images")
        self.apply_prev_checkbox = QCheckBox("↑ Apply to previous images")
        self.apply_next_checkbox = QCheckBox("↓ Apply to next images")
        self.apply_all_checkbox = QCheckBox("↕ Apply to all images")

        # Add the checkboxes to the button group
        image_options_group.addButton(self.apply_filtered_checkbox)
        image_options_group.addButton(self.apply_prev_checkbox)
        image_options_group.addButton(self.apply_next_checkbox)
        image_options_group.addButton(self.apply_all_checkbox)

        # Ensure only one checkbox can be checked at a time
        image_options_group.setExclusive(True)

        # Set the default checkbox
        self.apply_all_checkbox.setChecked(True)

        layout.addWidget(self.apply_filtered_checkbox)
        layout.addWidget(self.apply_prev_checkbox)
        layout.addWidget(self.apply_next_checkbox)
        layout.addWidget(self.apply_all_checkbox)

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

    # TODO something with this
    def setup_buttons_layout(self):
        """
        Set up the layout with buttons.
        """
        # Create a button box for the buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.apply)
        button_box.rejected.connect(self.reject)

        self.layout.addWidget(button_box)
        
    def update_detect_as_combo(self):
        """Update the label combo box with the current labels, preserving previous selection."""
        # Store the previously selected index
        previous_index = self.detect_as_combo.currentIndex() if hasattr(self, 'detect_as_combo') else 0

        self.detect_as_combo.clear()
        for label in self.label_window.labels:
            self.detect_as_combo.addItem(label.short_label_code, label.id)

        # Restore the previous selection if possible
        if 0 <= previous_index < self.detect_as_combo.count():
            self.detect_as_combo.setCurrentIndex(previous_index)
        else:
            self.detect_as_combo.setCurrentIndex(0)
            
    def update_class_mapping(self):
        """Update the class mapping based on the selected label."""
        detect_as = self.detect_as_combo.currentText()
        label = self.label_window.get_label_by_short_code(detect_as)
        self.class_mapping = {0: label}

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

    def update_sam_task_state(self):
        """
        Centralized method to check if SAM is loaded and update task and dropdown accordingly.
        """
        sam_active = (
            self.sam_dialog is not None and 
            self.sam_dialog.loaded_model is not None and
            self.use_sam_dropdown.currentText() == "True"
        )
        if sam_active:
            self.task = 'segment'
        else:
            self.task = 'detect'
            self.use_sam_dropdown.setCurrentText("False")
            
    def update_task(self):
        """Update the task based on the dropdown selection and handle UI/model effects."""
        self.task = self.use_task_dropdown.currentText()

        # Update UI elements based on task
        if self.task == "segment":
            # Deactivate model if one is loaded and we're switching to segment task
            if self.loaded_model:
                self.deactivate_model()

    def has_valid_sources(self):
        """
        Check if there are any valid source images with polygon or rectangle annotations.

        :return: True if valid sources exist, False otherwise
        """
        # Check if there are any images
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.information(None,
                                    "No Images",
                                    "No images available for batch inference.")
            return False

        # Check for images with valid annotations
        for image_path in self.image_window.raster_manager.image_paths:
            # Get annotations for this image
            annotations = self.annotation_window.get_image_annotations(image_path)

            # Check if there's at least one valid polygon/rectangle annotation
            for annotation in annotations:
                if isinstance(annotation, PolygonAnnotation) or isinstance(annotation, RectangleAnnotation):
                    return True

        QMessageBox.information(None,
                                "No Valid Annotations",
                                "No images have polygon or rectangle annotations for batch inference.")
        return False

    def check_valid_sources(self):
        """
        Check if there are any valid source images with polygon or rectangle annotations.

        :return: True if valid sources exist, False otherwise
        """
        # Check if there are any images
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.information(self,
                                    "No Images",
                                    "No images available for batch inference.")
            return False

        # Check for images with valid annotations
        for image_path in self.image_window.raster_manager.image_paths:
            # Get annotations for this image
            annotations = self.annotation_window.get_image_annotations(image_path)

            # Check if there's at least one valid polygon/rectangle annotation
            for annotation in annotations:
                if isinstance(annotation, PolygonAnnotation) or isinstance(annotation, RectangleAnnotation):
                    return True

        QMessageBox.information(self,
                                "No Valid Annotations",
                                "No images have polygon or rectangle annotations for batch inference.")
        return False

    def update_source_images(self):
        """
        Updates the source image combo box with images that have at least one label
        with a valid polygon or rectangle annotation.

        :return: True if valid source images were found, False otherwise
        """
        self.source_image_combo_box.clear()
        valid_images_found = False

        # Get all image paths from the raster_manager
        for image_path in self.image_window.raster_manager.image_paths:
            # Get annotations for this image
            annotations = self.annotation_window.get_image_annotations(image_path)

            # Check if there's at least one valid polygon/rectangle annotation
            valid_annotation_found = False
            for annotation in annotations:
                if isinstance(annotation, PolygonAnnotation) or isinstance(annotation, RectangleAnnotation):
                    valid_annotation_found = True
                    break

            if valid_annotation_found:
                # Get the basename (filename)
                basename = os.path.basename(image_path)
                # Add item to combo box with full path as data
                self.source_image_combo_box.addItem(basename, image_path)
                valid_images_found = True

        if not valid_images_found:
            QMessageBox.information(self,
                                    "No Source Images",
                                    "No images available for batch inference.")
            # Close the dialog since batch inference can't proceed
            QApplication.processEvents()  # Process pending events
            self.reject()
            return False

        # Update the combo box to have the selected image first
        if self.annotation_window.current_image_path in self.image_window.raster_manager.image_paths:
            self.source_image_combo_box.setCurrentText(os.path.basename(self.annotation_window.current_image_path))

        # Update the source labels given changes in the source images
        return self.update_source_labels()

    def update_source_labels(self):
        """
        Updates the source label combo box with labels that have at least one
        polygon or rectangle annotation from the current image.

        :return: True if valid source labels were found, False otherwise
        """
        self.source_label_combo_box.clear()

        source_image_path = self.source_image_combo_box.currentData()
        if not source_image_path:
            return False

        # Get annotations for this image
        annotations = self.annotation_window.get_image_annotations(source_image_path)

        # Create a dict of labels with valid annotations
        valid_labels = {}
        for annotation in annotations:
            if isinstance(annotation, PolygonAnnotation) or isinstance(annotation, RectangleAnnotation):
                valid_labels[annotation.label.short_label_code] = annotation.label

        # Add valid labels to combo box
        for label_code, label_obj in valid_labels.items():
            self.source_label_combo_box.addItem(label_code, label_obj)

        if not valid_labels:
            QMessageBox.information(self,
                                    "No Valid Labels",
                                    "No labels with polygon or rectangle annotations available for batch inference.")
            # Close the dialog since batch inference can't proceed
            QApplication.processEvents()  # Process pending events
            self.reject()
            return False

        return True

    def get_source_annotations(self):
        """Return a list of bboxes and masks for the source image
        belonging to the selected label."""
        source_image_path = self.source_image_combo_box.currentData()
        source_label = self.source_label_combo_box.currentData()

        # Get annotations for this image
        annotations = self.annotation_window.get_image_annotations(source_image_path)

        # Filter annotations by label
        source_bboxes = []
        source_masks = []
        for annotation in annotations:
            if annotation.label.short_label_code == source_label.short_label_code:
                if isinstance(annotation, (PolygonAnnotation, RectangleAnnotation)):
                    bbox = annotation.cropped_bbox
                    source_bboxes.append(bbox)
                    if isinstance(annotation, PolygonAnnotation):
                        points = np.array([[p.x(), p.y()] for p in annotation.points])
                        source_masks.append(points)
                    elif isinstance(annotation, RectangleAnnotation):
                        x1, y1, x2, y2 = bbox
                        rect_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                        source_masks.append(rect_points)

        return np.array(source_bboxes), source_masks

    def get_selected_image_paths(self):
        """
        Get the selected image paths based on the options.
        Excludes the source image path if present.
    
        :return: List of selected image paths
        """
        # Get the source image path to exclude
        source_image_path = self.source_image_combo_box.currentData()
        
        # Current image path showing
        current_image_path = self.annotation_window.current_image_path
        if not current_image_path:
            return []
    
        # Determine which images to export annotations for
        if self.apply_filtered_checkbox.isChecked():
            selected_paths = self.image_window.table_model.filtered_paths.copy()
        elif self.apply_prev_checkbox.isChecked():
            if current_image_path in self.image_window.table_model.filtered_paths:
                current_index = self.image_window.table_model.get_row_for_path(current_image_path)
                selected_paths = self.image_window.table_model.filtered_paths[:current_index + 1].copy()
            else:
                selected_paths = [current_image_path]
        elif self.apply_next_checkbox.isChecked():
            if current_image_path in self.image_window.table_model.filtered_paths:
                current_index = self.image_window.table_model.get_row_for_path(current_image_path)
                selected_paths = self.image_window.table_model.filtered_paths[current_index:].copy()
            else:
                selected_paths = [current_image_path]
        elif self.apply_all_checkbox.isChecked():
            selected_paths = self.image_window.raster_manager.image_paths.copy()
        else:
            # Only apply to the current image
            selected_paths = [current_image_path]
    
        # Remove the source image path if it's in the selected paths
        if source_image_path and source_image_path in selected_paths:
            selected_paths.remove(source_image_path)
    
        return selected_paths

    def apply(self):
        """
        Apply the selected batch inference options.
        """
        # Pause the cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # Get the source image path and label
            self.source_image_path = self.source_image_combo_box.currentData()
            self.source_label = self.source_label_combo_box.currentData()
            # Get the source annotations
            self.source_bboxes, self.source_masks = self.get_source_annotations()
            # Get the selected image paths
            self.target_images = self.get_selected_image_paths()
            # Perform batch inference
            self.batch_inference()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to make predictions: {str(e)}")
        finally:
            # Resume the cursor
            QApplication.restoreOverrideCursor()

        self.accept()
        
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
            # Check if SAM is active and update task state
            self.update_sam_task_state()
            
            # Get selected model path
            self.model_path = self.models[self.model_combo.currentText()]
            self.task = self.use_task_dropdown.currentText()

            # Set the parameters
            overrides = dict(model=self.model_path,
                             task=self.task,
                             mode='predict',
                             save=False,
                             retina_masks=self.task == "segment",
                             max_det=self.get_max_detections(),
                             imgsz=self.get_imgsz(),
                             conf=self.main_window.get_uncertainty_thresh(),
                             iou=self.main_window.get_iou_thresh(),
                             device=self.main_window.device)

            # Load the model
            self.loaded_model = FastSAMPredictor(overrides=overrides)
            self.loaded_model.names = {0: self.class_mapping[0].short_label_code}

            with torch.no_grad():
                # Run a blank through the model to initialize it
                self.loaded_model(np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8))

            progress_bar.finish_progress()
            self.status_bar.setText(f"Model loaded: {self.model_path}")
            QMessageBox.information(self, "Model Loaded", "Model loaded successfully")

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Model", str(e))

        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()
            # Stop the progress bar
            progress_bar.stop_progress()
            progress_bar.close()

        # Exit the dialog box
        self.accept()

    def get_imgsz(self):
        """Get the image size for the model."""
        self.imgsz = self.imgsz_spinbox.value()
        return self.imgsz

    def predict(self, image_paths=None):
        """
        Make predictions on the given image paths using the loaded model.

        Args:
            image_paths: List of image paths to process. If None, uses the current image.
        """
        if not self.loaded_model:
            return

        # Create a results processor
        results_processor = ResultsProcessor(
            self.main_window,
            self.class_mapping,
            uncertainty_thresh=self.main_window.get_uncertainty_thresh(),
            iou_thresh=self.main_window.get_iou_thresh(),
            min_area_thresh=self.main_window.get_area_thresh_min(),
            max_area_thresh=self.main_window.get_area_thresh_max()
        )

        if not image_paths:
            # Predict only the current image
            image_paths = [self.annotation_window.current_image_path]

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Start the progress bar
        progress_bar = ProgressBar(self.annotation_window, title="Prediction Workflow")
        progress_bar.show()
        progress_bar.start_progress(len(image_paths))

        try:
            for image_path in image_paths:
                inputs = self._get_inputs(image_path)
                if inputs is None:
                    continue

                results = self._apply_model(inputs)
                results = self._apply_sam(results, image_path)
                self._process_results(results_processor, results, image_path)

                # Update the progress bar
                progress_bar.update_progress()

        except Exception as e:
            print("An error occurred during prediction:", e)
        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.finish_progress()
            progress_bar.stop_progress()
            progress_bar.close()

        gc.collect()
        empty_cache()

    def _get_inputs(self, image_path):
        """Get the inputs for the model prediction."""
        raster = self.image_window.raster_manager.get_raster(image_path)
        if self.annotation_window.get_selected_tool() != "work_area":
            # Use the image path
            work_areas_data = [raster.image_path]
        else:
            # Get the work areas
            work_areas_data = raster.get_work_areas_data()

        return work_areas_data

    def _apply_model(self, inputs):
        """Apply the model to the inputs."""
        # Update the model with user parameters
        self.loaded_model.conf = self.main_window.get_uncertainty_thresh()
        self.loaded_model.iou = self.main_window.get_iou_thresh()
        self.loaded_model.max_det = self.get_max_detections()

        # Start the progress bar
        progress_bar = ProgressBar(self.annotation_window, title="Making Predictions")
        progress_bar.show()
        progress_bar.start_progress(len(inputs))

        results_list = []

        # Process each input separately
        for idx, input_image in enumerate(inputs):
            # Make predictions on single image
            with torch.no_grad():
                results = self.loaded_model(input_image)
                results_list.append(results)
                # Update the progress bar
                progress_bar.update_progress()
                # Clean up GPU memory after each prediction
                gc.collect()
                empty_cache()

        # Close the progress bar
        progress_bar.finish_progress()
        progress_bar.stop_progress()
        progress_bar.close()

        return results_list

    def _apply_sam(self, results_list, image_path):
        """Apply SAM to the results if needed."""
        # Check if SAM model is deployed and loaded
        self.update_sam_task_state()
        if self.task != 'segment':
            return results_list
        
        if not self.sam_dialog or self.use_sam_dropdown.currentText() == "False":
            # If SAM is not deployed or not selected, return the results as is
            return results_list

        if self.sam_dialog.loaded_model is None:
            # If SAM is not loaded, ensure we do not use it accidentally
            self.task = 'detect'
            self.use_sam_dropdown.setCurrentText("False")
            return results_list

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Predicting with SAM")
        progress_bar.show()
        progress_bar.start_progress(len(results_list))

        updated_results = []

        for idx, results in enumerate(results_list):
            # Each Results is a list (within the results_list, [[], ]
            if results:
                # Run it rough the SAM model
                results = self.sam_dialog.predict_from_results(results, image_path)
                updated_results.append(results)

            # Update the progress bar
            progress_bar.update_progress()

        # Make cursor normal
        QApplication.restoreOverrideCursor()
        progress_bar.finish_progress()
        progress_bar.stop_progress()
        progress_bar.close()

        return updated_results

    def _process_results(self, results_processor, results_list, image_path):
        """Process the results using the result processor."""
        # Get the raster object and number of work items
        raster = self.image_window.raster_manager.get_raster(image_path)
        total = raster.count_work_items()

        # Get the work areas (if any)
        work_areas = raster.get_work_areas()

        # Start the progress bar
        progress_bar = ProgressBar(self.annotation_window, title="Processing Results")
        progress_bar.show()
        progress_bar.start_progress(total)

        updated_results = []

        for idx, results in enumerate(results_list):
            # Each Results is a list (within the results_list, [[], ]
            if results:
                # Update path and names
                results[0].path = image_path
                results[0].names = {0: self.class_mapping[0].short_label_code}

                # Check if the work area is valid, or the image path is being used
                if work_areas and self.annotation_window.get_selected_tool() == "work_area":
                    # Map results from work area to the full image
                    results = MapResults().map_results_from_work_area(results[0], 
                                                                      raster, 
                                                                      work_areas[idx],
                                                                      self.task == "segment")
                else:
                    results = results[0]

                # Append the result object (not a list) to the updated results list
                updated_results.append(results)

                # Update the index for the next work area
                idx += 1
                progress_bar.update_progress()

        # Process the Results
        if self.task == 'segment' or self.use_sam_dropdown.currentText() == "True":
            results_processor.process_segmentation_results(updated_results)
        else:
            results_processor.process_detection_results(updated_results)

        # Close the progress bar
        progress_bar.finish_progress()
        progress_bar.stop_progress()
        progress_bar.close()

    def batch_inference(self):
        """
        Perform batch inference on the selected images.

        """
        # Make predictions on each image's annotations
        progress_bar = ProgressBar(self, title="Batch Inference")
        progress_bar.show()
        progress_bar.start_progress(len(self.target_images))

        if self.loaded_model is not None:
            self.deploy_model_dialog.predict_from_annotations(refer_image=self.source_image_path,
                                                              refer_label=self.source_label,
                                                              refer_bboxes=self.source_bboxes,
                                                              refer_masks=self.source_masks,
                                                              target_images=self.target_images)
        progress_bar.stop_progress()
        progress_bar.close()
        
    def deactivate_model(self):
        """
        Deactivate the currently loaded model and clean up resources.
        """
        self.loaded_model = None
        self.model_path = None
        # Clean up resources
        gc.collect()
        torch.cuda.empty_cache()
        # Untoggle all tools
        self.main_window.untoggle_all_tools()
        # Update status bar
        self.status_bar.setText("No model loaded")
        QMessageBox.information(self, "Model Deactivated", "Model deactivated")

