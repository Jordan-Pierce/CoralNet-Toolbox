import warnings

import os
import gc
import json
import copy

import numpy as np

import torch
from torch.cuda import empty_cache

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
from ultralytics.models.yolo.yoloe import YOLOEVPDetectPredictor

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QMessageBox, QCheckBox, QVBoxLayout, QApplication,
                             QLabel, QDialog, QDialogButtonBox, QGroupBox, QLineEdit,
                             QFormLayout, QComboBox, QSpinBox, QSlider, QPushButton,
                             QHBoxLayout, QWidget, QFileDialog)

from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

from coralnet_toolbox.Results import ResultsProcessor
from coralnet_toolbox.Results import MapResults
from coralnet_toolbox.Results import CombineResults

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.QtImageWindow import ImageWindow

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
        self.resize(800, 800)  # Increased size to accommodate the horizontal layout

        self.deploy_model_dialog = None
        self.loaded_model = None
        self.last_selected_label_code = None
        
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
        self.class_mapping = {}

        # Reference image and label
        self.source_images = []
        self.source_label = None
        # Target images
        self.target_images = []

        # Main vertical layout for the dialog
        self.layout = QVBoxLayout(self)

        # Setup the info layout at the top
        self.setup_info_layout()
        
        # Create horizontal layout for the two panels
        self.horizontal_layout = QHBoxLayout()
        self.layout.addLayout(self.horizontal_layout)
        
        # Create left panel
        self.left_panel = QVBoxLayout()
        self.horizontal_layout.addLayout(self.left_panel)
        
        # Create right panel
        self.right_panel = QVBoxLayout()
        self.horizontal_layout.addLayout(self.right_panel)
        
        # Add layouts to the left panel
        self.setup_models_layout()
        self.setup_parameters_layout()
        self.setup_sam_layout()
        self.setup_model_buttons_layout()
        self.setup_status_layout()
        
        # Add layouts to the right panel
        self.setup_source_layout()

        # # Add a full ImageWindow instance for target image selection
        self.image_selection_window = ImageWindow(self.main_window)
        self.right_panel.addWidget(self.image_selection_window)
        
        # Setup the buttons layout at the bottom
        self.setup_buttons_layout()

    def configure_image_window_for_dialog(self):
        """
        Disables parts of the internal ImageWindow UI to guide user selection.
        This forces the image list to only show images with annotations
        matching the selected reference label.
        """
        iw = self.image_selection_window

        # Block signals to prevent setChecked from triggering the ImageWindow's
        # own filtering logic. We want to be in complete control.
        iw.highlighted_checkbox.blockSignals(True)
        iw.has_predictions_checkbox.blockSignals(True)
        iw.no_annotations_checkbox.blockSignals(True)
        iw.has_annotations_checkbox.blockSignals(True)

        # Disable and set filter checkboxes
        iw.highlighted_checkbox.setEnabled(False)
        iw.has_predictions_checkbox.setEnabled(False)
        iw.no_annotations_checkbox.setEnabled(False)
        iw.has_annotations_checkbox.setEnabled(False)
        
        iw.highlighted_checkbox.setChecked(False)
        iw.has_predictions_checkbox.setChecked(False)
        iw.no_annotations_checkbox.setChecked(False)
        iw.has_annotations_checkbox.setChecked(True)  # This will no longer trigger a filter

        # Unblock signals now that we're done.
        iw.highlighted_checkbox.blockSignals(False)
        iw.has_predictions_checkbox.blockSignals(False)
        iw.no_annotations_checkbox.blockSignals(False)
        iw.has_annotations_checkbox.blockSignals(False)

        # Disable search UI elements
        iw.home_button.setEnabled(False)
        iw.image_search_button.setEnabled(False)
        iw.label_search_button.setEnabled(False)
        iw.search_bar_images.setEnabled(False)
        iw.search_bar_labels.setEnabled(False)
        iw.top_k_combo.setEnabled(False)

        # Set Top-K to Top1
        iw.top_k_combo.setCurrentText("Top1")

        # Disconnect the double-click signal to prevent it from loading an image
        # in the main window, as this dialog is for selection only.
        try:
            iw.tableView.doubleClicked.disconnect()
        except TypeError:
            pass
        
        # CRITICAL: Override the load_first_filtered_image method to prevent auto-loading
        # This is the key fix to prevent unwanted load_image_by_path calls
        iw.load_first_filtered_image = lambda: None

    def showEvent(self, event):
        """
        Set up the layout when the dialog is shown.

        :param event: Show event
        """
        super().showEvent(event)
        self.initialize_uncertainty_threshold()
        self.initialize_iou_threshold()
        self.initialize_area_threshold()
        
        # Configure the image window's UI elements for this specific dialog
        self.configure_image_window_for_dialog()
        # Sync with main window's images BEFORE updating labels
        self.sync_image_window()
        # This now populates the dropdown, restores the last selection,
        # and then manually triggers the image filtering.
        self.update_source_labels()

    def sync_image_window(self):
        """
        Syncs by directly adopting the main manager's up-to-date raster objects,
        avoiding redundant and slow re-calculation of annotation info.
        """
        main_manager = self.main_window.image_window.raster_manager
        dialog_manager = self.image_selection_window.raster_manager

        # Since the main_manager's rasters are always up-to-date, we can
        # simply replace the dialog's raster dictionary and path list entirely.
        # This is a shallow copy of the dictionary, which is extremely fast.
        # The Raster objects themselves are not copied, just referenced.
        dialog_manager.rasters = main_manager.rasters.copy()
        
        # Update the path list to match the new dictionary of rasters.
        dialog_manager.image_paths = list(dialog_manager.rasters.keys())

        # The slow 'for' loop that called update_annotation_info is now gone.
        # We are trusting that each raster object from the main_manager
        # already has its .label_set and .annotation_type_set correctly populated.
            
    def filter_images_by_label_and_type(self):
        """
        Filters the image list to show only images that contain at least one
        annotation that has BOTH the selected label AND a valid type (Polygon or Rectangle).
        This uses the fast, pre-computed cache for performance.
        """
        source_label = self.source_label_combo_box.currentData()
        source_label_text = self.source_label_combo_box.currentText()

        # Store the last selected label for a better user experience on re-opening.
        if source_label_text:
            self.last_selected_label_code = source_label_text

        if not source_label:
            # If no label is selected (e.g., during initialization), show an empty list.
            self.image_selection_window.table_model.set_filtered_paths([])
            return

        all_paths = self.image_selection_window.raster_manager.image_paths
        final_filtered_paths = []
        
        valid_types = {"RectangleAnnotation", "PolygonAnnotation"}
        selected_label_code = source_label.short_label_code

        # Loop through paths and check the pre-computed map on each raster
        for path in all_paths:
            raster = self.image_selection_window.raster_manager.get_raster(path)
            if not raster:
                continue
                
            # 1. From the cache, get the set of annotation types specifically for our selected label.
            #    Use .get() to safely return an empty set if the label isn't on this image at all.
            types_for_this_label = raster.label_to_types_map.get(selected_label_code, set())
            
            # 2. Check for any overlap between the types found FOR THIS LABEL and the
            #    valid types we need (Polygon/Rectangle). This is the key check.
            if not valid_types.isdisjoint(types_for_this_label):
                # This image is a valid reference because the selected label exists
                # on a Polygon or Rectangle. Add it to the list.
                final_filtered_paths.append(path)

        # Directly set the filtered list in the table model.
        self.image_selection_window.table_model.set_filtered_paths(final_filtered_paths)
        
    def accept(self):
        """
        Validate selections and store them before closing the dialog.
        """
        if not self.loaded_model:
            QMessageBox.warning(self, 
                                "No Model", 
                                "A model must be loaded before running predictions.")
            super().reject()
            return

        current_label = self.source_label_combo_box.currentData()
        if not current_label:
            QMessageBox.warning(self, 
                                "No Source Label", 
                                "A source label must be selected.")
            super().reject()
            return

        # Get highlighted paths from our internal image window to use as targets
        highlighted_images = self.image_selection_window.table_model.get_highlighted_paths()

        if not highlighted_images:
            QMessageBox.warning(self, 
                                "No Target Images", 
                                "You must highlight at least one image in the list to process.")
            super().reject()
            return

        # Store the selections for the caller to use after the dialog closes.
        self.source_label = current_label
        self.target_images = highlighted_images

        # Do not call self.predict here; just close the dialog and let the caller handle prediction
        super().accept()

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout that spans the top.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Choose a Generator to deploy. "
                            "Select a reference label, then highlight reference images that contain examples. "
                            "Each additional reference image may increase accuracy but also processing time.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)  # Add to main layout so it spans both panels
        
    def setup_models_layout(self):
        """
        Setup the models layout with a simple model selection combo box (no tabs).
        """
        group_box = QGroupBox("Model Selection")
        layout = QVBoxLayout()

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)

        # Define available models (keep the existing dictionary)
        self.models = [
            "yoloe-v8s-seg.pt",
            "yoloe-v8m-seg.pt",
            "yoloe-v8l-seg.pt",
            "yoloe-11s-seg.pt",
            "yoloe-11m-seg.pt",
            "yoloe-11l-seg.pt",
        ]

        # Add all models to combo box
        for model_name in self.models:
            self.model_combo.addItem(model_name)
        
        # Set the default model
        self.model_combo.setCurrentText("yoloe-v8s-seg.pt")

        layout.addWidget(QLabel("Select Model:"))
        layout.addWidget(self.model_combo)

        group_box.setLayout(layout)
        self.left_panel.addWidget(group_box)  # Add to left panel
        
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
        self.area_threshold_label = QLabel(f"{self.area_thresh_min / 100.0:.2f} - {self.area_thresh_max / 100.0:.2f}")
        layout.addRow("Area Threshold Min", self.area_threshold_min_slider)
        layout.addRow("Area Threshold Max", self.area_threshold_max_slider)
        layout.addRow("", self.area_threshold_label)

        group_box.setLayout(layout)
        self.left_panel.addWidget(group_box)  # Add to left panel

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
        self.left_panel.addWidget(group_box)  # Add to left panel

    def setup_model_buttons_layout(self):
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
        self.left_panel.addWidget(group_box)  # Add to left panel

    def setup_status_layout(self):
        """
        Setup status display in a group box.
        """
        group_box = QGroupBox("Status")
        layout = QVBoxLayout()

        self.status_bar = QLabel("No model loaded")
        layout.addWidget(self.status_bar)

        group_box.setLayout(layout)
        self.left_panel.addWidget(group_box)  # Add to left panel

    def setup_source_layout(self):
        """
        Set up the layout with source label selection.
        The source image is implicitly the currently active image.
        """
        group_box = QGroupBox("Reference Label")
        layout = QFormLayout()

        # Create the source label combo box
        self.source_label_combo_box = QComboBox()
        self.source_label_combo_box.currentIndexChanged.connect(self.filter_images_by_label_and_type)
        layout.addRow("Reference Label:", self.source_label_combo_box)

        group_box.setLayout(layout)
        self.right_panel.addWidget(group_box)  # Add to right panel

    def setup_buttons_layout(self):
        """
        Set up the layout with buttons.
        """
        # Create a button box for the buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        self.layout.addWidget(button_box)

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
        Centralized method to check if SAM is loaded and update task accordingly.
        If the user has selected to use SAM, this function ensures the task is set to 'segment'.
        Crucially, it does NOT alter the task if SAM is not selected, respecting the
        user's choice from the 'Task' dropdown.
        """
        # Check if the user wants to use the SAM model
        if self.use_sam_dropdown.currentText() == "True":
            # SAM is requested. Check if it's actually available.
            sam_is_available = (
                hasattr(self, 'sam_dialog') and
                self.sam_dialog is not None and
                self.sam_dialog.loaded_model is not None
            )

            if sam_is_available:
                # If SAM is wanted and available, the task must be segmentation.
                self.task = 'segment'
            else:
                # If SAM is wanted but not available, revert the dropdown and do nothing else.
                # The 'is_sam_model_deployed' function already handles showing an error message.
                self.use_sam_dropdown.setCurrentText("False")

        # If use_sam_dropdown is "False", do nothing. Let self.task be whatever the user set.
            
    def update_task(self):
        """Update the task based on the dropdown selection and handle UI/model effects."""
        self.task = self.use_task_dropdown.currentText()

        # Update UI elements based on task
        if self.task == "segment":
            # Deactivate model if one is loaded and we're switching to segment task
            if self.loaded_model:
                self.deactivate_model()

    def update_source_labels(self):
        """
        Updates the source label combo box with labels that are associated with
        valid reference annotations (Polygons or Rectangles), using the fast cache.
        """
        self.source_label_combo_box.blockSignals(True)
        
        try:
            self.source_label_combo_box.clear()

            dialog_manager = self.image_selection_window.raster_manager
            valid_types = {"RectangleAnnotation", "PolygonAnnotation"}
            valid_labels = set()  # This will store the full Label objects

            # Create a lookup map to get full label objects from their codes
            all_project_labels = {lbl.short_label_code: lbl for lbl in self.main_window.label_window.labels}

            # Use the cached data to find all labels that have valid reference types.
            for raster in dialog_manager.rasters.values():
                # raster.label_to_types_map is like: {'coral': {'Point'}, 'rock': {'Polygon'}}
                for label_code, types_for_label in raster.label_to_types_map.items():
                    # Check if the set of types for this specific label
                    # has any overlap with our valid reference types.
                    if not valid_types.isdisjoint(types_for_label):
                        # This label is a valid reference label.
                        # Add its full Label object to our set of valid labels.
                        if label_code in all_project_labels:
                            valid_labels.add(all_project_labels[label_code])

            # Add the valid labels to the combo box, sorted alphabetically.
            sorted_valid_labels = sorted(list(valid_labels), key=lambda x: x.short_label_code)
            for label_obj in sorted_valid_labels:
                self.source_label_combo_box.addItem(label_obj.short_label_code, label_obj)

            # Restore the last selected label if it's still present in the list.
            if self.last_selected_label_code:
                index = self.source_label_combo_box.findText(self.last_selected_label_code)
                if index != -1:
                    self.source_label_combo_box.setCurrentIndex(index)
        finally:
            self.source_label_combo_box.blockSignals(False)
        
        # Manually trigger the filtering now that the combo box is stable.
        self.filter_images_by_label_and_type()

        return True

    def get_source_annotations(self, reference_label, reference_image_path):
        """
        Return a list of bboxes and masks for a specific image
        belonging to the selected label.

        :param reference_label: The Label object to filter annotations by.
        :param reference_image_path: The path of the image to get annotations from.
        :return: A tuple containing a numpy array of bboxes and a list of masks.
        """
        if not all([reference_label, reference_image_path]):
            return np.array([]), []

        # Get all annotations for the specified image
        annotations = self.annotation_window.get_image_annotations(reference_image_path)

        # Filter annotations by the provided label
        source_bboxes = []
        source_masks = []
        for annotation in annotations:
            if annotation.label.short_label_code == reference_label.short_label_code:
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

            progress_bar.finish_progress()
            self.status_bar.setText("Model loaded")
            QMessageBox.information(self.annotation_window, 
                                    "Model Loaded", 
                                    "Model loaded successfully")

        except Exception as e:
            QMessageBox.critical(self.annotation_window, 
                                 "Error Loading Model", 
                                 f"Error loading model: {e}")

        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()
            # Stop the progress bar
            progress_bar.stop_progress()
            progress_bar.close()
            progress_bar = None

    def predict(self, image_paths=None):
        """
        Make predictions on the given image paths using the loaded model.

        Args:
            image_paths: List of image paths to process. If None, uses the current image.
        """
        if not self.loaded_model or not self.source_label:
            return
        
        # Update class mapping with the selected reference label
        self.class_mapping = {0: self.source_label}

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
        """
        Apply the model to the target inputs, using each highlighted source
        image as an individual reference for a separate prediction run.
        """
        # Update the model with user parameters
        self.loaded_model.conf = self.main_window.get_uncertainty_thresh()
        self.loaded_model.iou = self.main_window.get_iou_thresh()
        self.loaded_model.max_det = self.get_max_detections()
        
        # NOTE: self.target_images contains the reference images highlighted in the dialog
        reference_image_paths = self.target_images

        if not reference_image_paths:
            QMessageBox.warning(self, 
                                "No Reference Images", 
                                "You must highlight at least one reference image.")
            return []

        # Get the selected reference label from the stored variable
        source_label = self.source_label
        
        # Create a dictionary of reference annotations, with image path as the key.
        reference_annotations_dict = {}
        for path in reference_image_paths:
            bboxes, masks = self.get_source_annotations(source_label, path)
            if bboxes.size > 0:
                reference_annotations_dict[path] = {
                    'bboxes': bboxes,
                    'masks': masks,
                    'cls': np.zeros(len(bboxes))
                }

        # Set the task
        self.task = self.use_task_dropdown.currentText()
        predictor = YOLOEVPSegPredictor if self.task == "segment" else YOLOEVPDetectPredictor

        # Create a progress bar for iterating through reference images
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Making Predictions per Reference")
        progress_bar.show()
        progress_bar.start_progress(len(reference_annotations_dict))
        
        results_list = []
        # The 'inputs' list contains work areas from the single target image.
        # We will predict on the first work area/full image.
        input_image = inputs[0] 

        # Iterate through each reference image and its annotations
        for ref_path, ref_annotations in reference_annotations_dict.items():
            # The 'refer_image' parameter is the path to the current reference image
            # The 'visual_prompts' are the annotations from that same reference image
            visuals = {
                'bboxes': ref_annotations['bboxes'],
                'cls': ref_annotations['cls'],
            }
            if self.task == 'segment':
                visuals['masks'] = ref_annotations['masks']

            # Make predictions on the target using the current reference
            results = self.loaded_model.predict(input_image,
                                                refer_image=ref_path,
                                                visual_prompts=visuals,
                                                predictor=predictor,
                                                imgsz=self.imgsz_spinbox.value(),
                                                conf=self.main_window.get_uncertainty_thresh(),
                                                iou=self.main_window.get_iou_thresh(),
                                                max_det=self.get_max_detections(),
                                                retina_masks=self.task == "segment")
            
            if not len(results[0].boxes):
                # If no boxes were detected, skip to the next reference
                progress_bar.update_progress()
                continue
            
            # Update the name of the results and append to the list
            results[0].names = {0: self.class_mapping[0].short_label_code}
            results_list.extend(results[0])
            
            progress_bar.update_progress()
            gc.collect()
            empty_cache()

        # Clean up
        QApplication.restoreOverrideCursor()
        progress_bar.finish_progress()
        progress_bar.stop_progress()
        progress_bar.close()
        
        # Combine results if there are any
        combined_results = CombineResults().combine_results(results_list)
        if combined_results is None:
            return []
        
        return [[combined_results]]

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
                # This needs to be done again, in case SAM was used

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