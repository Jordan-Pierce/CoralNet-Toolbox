import warnings

import os
import gc

import numpy as np
from sklearn.decomposition import PCA

import torch
from torch.cuda import empty_cache

import pyqtgraph as pg

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
from ultralytics.models.yolo.yoloe import YOLOEVPDetectPredictor

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QMessageBox, QVBoxLayout, QApplication, QFileDialog,
                             QLabel, QDialog, QDialogButtonBox, QGroupBox, QLineEdit,
                             QFormLayout, QComboBox, QSpinBox, QSlider, QPushButton,
                             QHBoxLayout)

from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

from coralnet_toolbox.Results import ResultsProcessor
from coralnet_toolbox.Results import MapResults

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
        self.reference_label = None
        self.reference_image_paths = []
        
        # Visual Prompting Encoding (VPE) - legacy single tensor variable
        self.vpe_path = None
        self.vpe = None
        
        # New separate VPE collections
        self.imported_vpes = []  # VPEs loaded from file
        self.reference_vpes = []  # VPEs created from reference images

        self.device = None  # Will be set in showEvent

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
        self.setup_reference_layout()

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
        
        # Hide the "Current" label as it is not applicable in this dialog
        iw.current_image_index_label.hide()

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
        
        # Update the device
        self.device = self.main_window.device
        # Configure the image window's UI elements for this specific dialog
        self.configure_image_window_for_dialog()
        # Sync with main window's images BEFORE updating labels
        self.sync_image_window()
        # This now populates the dropdown, restores the last selection,
        # and then manually triggers the image filtering.
        self.update_reference_labels()

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
        # Persist the user's current highlights from the table model before filtering.
        # This ensures that if the user highlights items and then changes the filter,
        # their selection is not lost.
        current_highlights = self.image_selection_window.table_model.get_highlighted_paths()
        if current_highlights:
            self.reference_image_paths = current_highlights

        reference_label = self.reference_label_combo_box.currentData()
        reference_label_text = self.reference_label_combo_box.currentText()

        # Store the last selected label for a better user experience on re-opening.
        if reference_label_text:
            self.last_selected_label_code = reference_label_text
            # Also store the reference label object itself
            self.reference_label = reference_label

        if not reference_label:
            # If no label is selected (e.g., during initialization), show an empty list.
            self.image_selection_window.table_model.set_filtered_paths([])
            return

        all_paths = self.image_selection_window.raster_manager.image_paths
        final_filtered_paths = []
        
        valid_types = {"RectangleAnnotation", "PolygonAnnotation"}
        selected_label_code = reference_label.short_label_code

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
        
        # Try to preserve any previous selections
        if hasattr(self, 'reference_image_paths') and self.reference_image_paths:
            # Find which of our previously selected paths are still in the filtered list
            valid_selections = [p for p in self.reference_image_paths if p in final_filtered_paths]
            if valid_selections:
                # Highlight previously selected paths that are still valid
                self.image_selection_window.table_model.set_highlighted_paths(valid_selections)

        # After filtering, update all labels with the correct counts.
        dialog_iw = self.image_selection_window
        dialog_iw.update_image_count_label(len(final_filtered_paths)) # Set "Total" to filtered count
        dialog_iw.update_current_image_index_label()
        dialog_iw.update_highlighted_count_label()
                
    def accept(self):
        """
        Validate selections and store them before closing the dialog.
        A prediction is valid if a model and label are selected, and the user
        has provided either reference images or an imported VPE file.
        """
        if not self.loaded_model:
            QMessageBox.warning(self, 
                                "No Model", 
                                "A model must be loaded before running predictions.")
            return

        # Set reference label from combo box
        self.reference_label = self.reference_label_combo_box.currentData()
        if not self.reference_label:
            QMessageBox.warning(self, 
                                "No Reference Label", 
                                "A reference label must be selected.")
            return

        # Stash the current UI selection before validating.
        self.update_stashed_references_from_ui()

        # Check for a valid VPE source using the now-stashed list.
        has_reference_images = bool(self.reference_image_paths)
        has_imported_vpes = bool(self.imported_vpes)

        if not has_reference_images and not has_imported_vpes:
            QMessageBox.warning(self, 
                                "No VPE Source Provided", 
                                "You must highlight at least one reference image or load a VPE file to proceed.")
            return

        # If validation passes, close the dialog.
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
        layout = QFormLayout()

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)

        # Define available models (keep the existing dictionary)
        self.models = [
            'yoloe-v8s-seg.pt',
            'yoloe-v8m-seg.pt',
            'yoloe-v8l-seg.pt',
            'yoloe-11s-seg.pt',
            'yoloe-11m-seg.pt',
            'yoloe-11l-seg.pt',
        ]

        # Add all models to combo box
        for model_name in self.models:
            self.model_combo.addItem(model_name)
        
        # Set the default model
        self.model_combo.setCurrentIndex(self.models.index('yoloe-v8s-seg.pt'))
        # Create a layout for the model selection
        layout.addRow(QLabel("Models:"), self.model_combo)

        # Add custom vpe file selection
        self.vpe_path_edit = QLineEdit()
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_vpe_file)

        vpe_path_layout = QHBoxLayout()
        vpe_path_layout.addWidget(self.vpe_path_edit)
        vpe_path_layout.addWidget(browse_button)
        layout.addRow("Custom VPE:", vpe_path_layout)

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
        self.imgsz_spinbox.setRange(1024, 65536)
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
        main_layout = QVBoxLayout()

        # First row: Load and Deactivate buttons side by side
        button_row = QHBoxLayout()
        load_button = QPushButton("Load Model")
        load_button.clicked.connect(self.load_model)
        button_row.addWidget(load_button)

        deactivate_button = QPushButton("Deactivate Model")
        deactivate_button.clicked.connect(self.deactivate_model)
        button_row.addWidget(deactivate_button)

        main_layout.addLayout(button_row)

        # Second row: VPE action buttons
        vpe_row = QHBoxLayout()
        
        generate_vpe_button = QPushButton("Generate VPEs")
        generate_vpe_button.clicked.connect(self.generate_vpes_from_references)
        vpe_row.addWidget(generate_vpe_button)

        save_vpe_button = QPushButton("Save VPE")
        save_vpe_button.clicked.connect(self.save_vpe)
        vpe_row.addWidget(save_vpe_button)

        show_vpe_button = QPushButton("Show VPE")
        show_vpe_button.clicked.connect(self.show_vpe)
        vpe_row.addWidget(show_vpe_button)

        main_layout.addLayout(vpe_row)

        group_box.setLayout(main_layout)
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

    def setup_reference_layout(self):
        """
        Set up the layout for reference selection, including the output label,
        reference method, and the number of prototype clusters (K).
        """
        group_box = QGroupBox("Reference")
        layout = QFormLayout()

        # Create the reference label combo box
        self.reference_label_combo_box = QComboBox()
        self.reference_label_combo_box.currentIndexChanged.connect(self.filter_images_by_label_and_type)
        layout.addRow("Reference Label:", self.reference_label_combo_box)
        
        # Create a Reference model combobox (VPE, Images)
        self.reference_method_combo_box = QComboBox()
        self.reference_method_combo_box.addItems(["VPE", "Images"])
        layout.addRow("Reference Method:", self.reference_method_combo_box)

        # Add a spinbox for the user to define the number of prototypes (K)
        self.k_prototypes_spinbox = QSpinBox()
        self.k_prototypes_spinbox.setRange(0, 1000)
        self.k_prototypes_spinbox.setValue(0)
        self.k_prototypes_spinbox.setToolTip(
            "Set the number of prototype clusters (K) to generate from references.\n"
            "Set to 0 to treat every unique reference image/VPE as its own prototype (K=N)."
        )
        layout.addRow("Number of Prototypes (K):", self.k_prototypes_spinbox)

        group_box.setLayout(layout)
        self.right_panel.addWidget(group_box)

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
        
    def update_stashed_references_from_ui(self):
        """Updates the internal reference path list from the current UI selection."""
        self.reference_image_paths = self.image_selection_window.table_model.get_highlighted_paths()

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

    def update_reference_labels(self):
        """
        Updates the reference label combo box with ALL available project labels.
        This dropdown now serves as the "Output Label" for all predictions.
        The "Review" label with id "-1" is excluded.
        """
        self.reference_label_combo_box.blockSignals(True)
        
        try:
            self.reference_label_combo_box.clear()

            # Get all labels from the main label window
            all_project_labels = self.main_window.label_window.labels

            # Filter out the special "Review" label and create a list of valid labels
            valid_labels = [
                label_obj for label_obj in all_project_labels
                if not (label_obj.short_label_code == "Review" and str(label_obj.id) == "-1")
            ]

            # Add the valid labels to the combo box, sorted alphabetically.
            sorted_valid_labels = sorted(valid_labels, key=lambda x: x.short_label_code)
            for label_obj in sorted_valid_labels:
                self.reference_label_combo_box.addItem(label_obj.short_label_code, label_obj)

            # Restore the last selected label if it's still present in the list.
            if self.last_selected_label_code:
                index = self.reference_label_combo_box.findText(self.last_selected_label_code)
                if index != -1:
                    self.reference_label_combo_box.setCurrentIndex(index)
        finally:
            self.reference_label_combo_box.blockSignals(False)
        
        # Manually trigger the image filtering now that the combo box is stable.
        # This will still filter the image list to help find references if needed.
        self.filter_images_by_label_and_type()

    def get_reference_annotations(self, reference_label, reference_image_path):
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
        reference_bboxes = []
        reference_masks = []
        for annotation in annotations:
            if annotation.label.short_label_code == reference_label.short_label_code:
                if isinstance(annotation, (PolygonAnnotation, RectangleAnnotation)):
                    bbox = annotation.cropped_bbox
                    reference_bboxes.append(bbox)
                    if isinstance(annotation, PolygonAnnotation):
                        points = np.array([[p.x(), p.y()] for p in annotation.points])
                        reference_masks.append(points)
                    elif isinstance(annotation, RectangleAnnotation):
                        x1, y1, x2, y2 = bbox
                        rect_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                        reference_masks.append(rect_points)

        return np.array(reference_bboxes), reference_masks
    
    def browse_vpe_file(self):
        """
        Open a file dialog to browse for a VPE file and load it.
        Stores imported VPEs separately from reference-generated VPEs.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Visual Prompt Encoding (VPE) File",
            "",
            "VPE Files (*.pt);;All Files (*)"
        )
        
        if not file_path:
            return
            
        self.vpe_path_edit.setText(file_path)
        self.vpe_path = file_path
        
        try:
            # Load the VPE file
            loaded_data = torch.load(file_path)

            # Move tensors to the appropriate device
            device = self.main_window.device
            
            # Check format type and handle appropriately
            if isinstance(loaded_data, list):
                # New format: list of VPE tensors
                self.imported_vpes = [vpe.to(device) for vpe in loaded_data]
                vpe_count = len(self.imported_vpes)
                self.status_bar.setText(f"Loaded {vpe_count} VPE tensors from file")
                
            elif isinstance(loaded_data, torch.Tensor):
                # Legacy format: single tensor - convert to list for consistency
                loaded_vpe = loaded_data.to(device)
                # Store as a single-item list
                self.imported_vpes = [loaded_vpe]
                self.status_bar.setText("Loaded 1 VPE tensor from file (legacy format)")
                
            else:
                # Invalid format
                self.imported_vpes = []
                self.status_bar.setText("Invalid VPE file format")
                QMessageBox.warning(
                    self, 
                    "Invalid VPE", 
                    "The file does not appear to be a valid VPE format."
                )
                # Clear the VPE path edit field
                self.vpe_path_edit.clear()
                    
            # For backward compatibility - set self.vpe to the average of imported VPEs
            # This ensures older code paths still work
            if self.imported_vpes:
                combined_vpe = torch.cat(self.imported_vpes).mean(dim=0, keepdim=True)
                self.vpe = torch.nn.functional.normalize(combined_vpe, p=2, dim=-1)
                
        except Exception as e:
            self.imported_vpes = []
            self.vpe = None
            self.status_bar.setText(f"Error loading VPE: {str(e)}")
            QMessageBox.critical(
                self, 
                "Error Loading VPE", 
                f"Failed to load VPE file: {str(e)}"
            )
            
    def save_vpe(self):
        """
        Saves the combined collection of VPEs (imported and pre-generated from references) to disk.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            # Create a list to hold all VPEs to be saved
            all_vpes = []
            
            # Add imported VPEs if available
            if self.imported_vpes:
                all_vpes.extend(self.imported_vpes)
            
            # Add pre-generated reference VPEs if available
            if self.reference_vpes:
                all_vpes.extend(self.reference_vpes)
            
            # Check if we have any VPEs to save
            if not all_vpes:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(
                    self,
                    "No VPEs Available",
                    "No VPEs available to save. "
                    "Please either load a VPE file or generate VPEs from reference images first."
                )
                return
            
            QApplication.restoreOverrideCursor()
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save VPE Collection",
                "",
                "PyTorch Tensor (*.pt);;All Files (*)"
            )
            
            if not file_path:
                return
            
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            if not file_path.endswith('.pt'):
                file_path += '.pt'
            
            vpe_list_cpu = [vpe.cpu() for vpe in all_vpes]
            
            torch.save(vpe_list_cpu, file_path)
            
            self.status_bar.setText(f"Saved {len(all_vpes)} VPE tensors to {os.path.basename(file_path)}")
            
            QApplication.restoreOverrideCursor()
            QMessageBox.information(
                self,
                "VPE Saved",
                f"Saved {len(all_vpes)} VPE tensors to {file_path}"
            )
            
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(
                self,
                "Error Saving VPE",
                f"Failed to save VPE: {str(e)}"
            )
        finally:
            try:
                QApplication.restoreOverrideCursor()
            except:
                pass

    def load_model(self):
        """
        Load the selected model.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Loading Model")
        progress_bar.show()

        try:
            # Load the model using reload_model method
            self.reload_model()

            # Calculate total number of VPEs from both sources
            total_vpes = len(self.imported_vpes) + len(self.reference_vpes)
            
            if total_vpes > 0:
                if self.imported_vpes and self.reference_vpes:
                    message = f"Model loaded with {len(self.imported_vpes)} imported VPEs "
                    message += f"and {len(self.reference_vpes)} reference VPEs"
                elif self.imported_vpes:
                    message = f"Model loaded with {len(self.imported_vpes)} imported VPEs"
                else:
                    message = f"Model loaded with {len(self.reference_vpes)} reference VPEs"
                    
                self.status_bar.setText(message)
            else:
                message = "Model loaded with default VPE"
                self.status_bar.setText("Model loaded with default VPE")

            # Finish progress bar
            progress_bar.finish_progress()
            QMessageBox.information(self.annotation_window, "Model Loaded", message)

        except Exception as e:
            self.loaded_model = None
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
            
    def reload_model(self):
        """
        Subset of the load_model method. This is needed when additional 
        reference images and annotations (i.e., VPEs) are added (we have 
        to re-load the model each time).
        
        This method also ensures that we stash the currently highlighted reference
        image paths before reloading, so they're available for predictions
        even if the user switches the active image in the main window.
        """        
        self.loaded_model = None
        
        # Get selected model path and download weights if needed
        self.model_path = self.model_combo.currentText()

        # Load model using registry
        self.loaded_model = YOLOE(self.model_path, verbose=False).to(self.device)

        # Create a dummy visual dictionary for standard model loading
        visual_prompts = dict(
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
            visual_prompts=visual_prompts.copy(),  # This needs to happen to properly initialize the predictor
            predictor=YOLOEVPDetectPredictor if self.task == "detect" else YOLOEVPSegPredictor,
            imgsz=640,
            conf=0.99,
        )

        # If a VPE file was loaded, use it with the model after the dummy prediction
        if self.vpe is not None and isinstance(self.vpe, torch.Tensor):
            # Directly set the final tensor as the prompt for the predictor
            self.loaded_model.is_fused = lambda: False
            self.loaded_model.set_classes(["object0"], self.vpe)
            
    def predict(self, image_paths=None):
        """
        Make predictions on the given image paths using the loaded model.

        Args:
            image_paths: List of image paths to process. If None, uses the current image.
        """
        if not self.loaded_model or not self.reference_label:
            return
        
        # Update class mapping with the selected reference label
        self.class_mapping = {0: self.reference_label}

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
    
    def _get_references(self):
        """
        Get the reference annotations using the stashed list of reference images
        that was saved when the user accepted the dialog.
        
        Returns:
            dict: Dictionary mapping image paths to annotation data, or empty dict if no valid references.
        """
        # Use the "stashed" list of paths. Do NOT query the table_model again,
        # as the UI's highlight state may have been cleared by other actions.
        reference_paths = self.reference_image_paths

        if not reference_paths:
            print("No reference image paths were stashed to use for prediction.")
            return {}

        # Get the reference label that was also stashed
        reference_label = self.reference_label
        if not reference_label:
            # This check is a safeguard; the accept() method should prevent this.
            print("No reference label was selected.")
            return {}
        
        # Create a dictionary of reference annotations from the stashed paths
        reference_annotations_dict = {}
        for path in reference_paths:
            bboxes, masks = self.get_reference_annotations(reference_label, path)
            if bboxes.size > 0:
                reference_annotations_dict[path] = {
                    'bboxes': bboxes,
                    'masks': masks,
                    'cls': np.zeros(len(bboxes))
                }

        return reference_annotations_dict

    def _apply_model_using_images(self, inputs, reference_dict):
        """
        Apply the model using reference images. This method now generates VPEs
        from the reference annotations, clusters them into K prototypes, and
        runs a single, efficient multi-class prediction.

        Args:
            inputs (list): List of input images.
            reference_dict (dict): Dictionary containing reference annotations for each image.

        Returns:
            list: List of prediction results.
        """
        # Note: This method may require 'from sklearn.cluster import KMeans'

        # 1. Reload model and generate initial VPEs from the reference images
        self.reload_model()
        initial_vpes = self.references_to_vpe(reference_dict, update_reference_vpes=False)

        if not initial_vpes:
            QMessageBox.warning(self, 
                                "VPE Generation Failed", 
                                "Could not generate VPEs from the selected reference images.")
            return []

        # 2. Generate K prototypes from the N generated VPEs
        k = self.k_prototypes_spinbox.value()
        num_available_vpes = len(initial_vpes)
        prototype_vpes = []

        # If K is 0 (use all) or K >= N, no clustering is needed.
        if k == 0 or k >= num_available_vpes:
            prototype_vpes = initial_vpes
        else:
            # Perform K-Means clustering to find K prototypes
            try:
                # Prepare tensor for scikit-learn: shape (N, E)
                all_vpes_tensor = torch.cat([vpe.squeeze(1) for vpe in initial_vpes], dim=0)
                vpes_np = all_vpes_tensor.cpu().numpy()

                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(vpes_np)
                centroids_np = kmeans.cluster_centers_

                # Convert centroids back to a list of normalized PyTorch tensors
                centroids_tensor = torch.from_numpy(centroids_np).to(self.device)
                for i in range(k):
                    # Reshape centroid to (1, 1, E)
                    centroid = centroids_tensor[i].unsqueeze(0).unsqueeze(0)
                    normalized_centroid = torch.nn.functional.normalize(centroid, p=2, dim=-1)
                    prototype_vpes.append(normalized_centroid)
            except Exception as e:
                QMessageBox.critical(self, "Clustering Error", f"Failed to perform K-Means clustering: {e}")
                return []

        # 3. Configure the model with the K prototypes
        if not prototype_vpes:
            QMessageBox.warning(self, "Prototype Error", "Could not generate any prototypes for prediction.")
            return []

        num_prototypes = len(prototype_vpes)
        proto_class_names = [f"object{i}" for i in range(num_prototypes)]
        stacked_vpes = torch.cat(prototype_vpes, dim=1)  # Shape: (1, K, E)

        self.loaded_model.is_fused = lambda: False
        self.loaded_model.set_classes(proto_class_names, stacked_vpes)

        # 4. Make a single prediction on the target using the K prototypes
        results = self.loaded_model.predict(inputs[0],
                                            visual_prompts=[],
                                            imgsz=self.imgsz_spinbox.value(),
                                            conf=self.main_window.get_uncertainty_thresh(),
                                            iou=self.main_window.get_iou_thresh(),
                                            max_det=self.get_max_detections(),
                                            retina_masks=self.task == "segment")

        return [results]
    
    def generate_vpes_from_references(self):
        """
        Calculates VPEs from the currently highlighted reference images and
        stores them in self.reference_vpes, overwriting any previous ones.
        """
        if not self.loaded_model:
            QMessageBox.warning(self, "No Model Loaded", "A model must be loaded before generating VPEs.")
            return

        # Always sync with the live UI selection before generating.
        self.update_stashed_references_from_ui()
        references_dict = self._get_references()

        if not references_dict:
            QMessageBox.information(
                self,
                "No References Selected",
                "Please highlight one or more reference images in the table to generate VPEs."
            )
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self, title="Generating VPEs")
        progress_bar.show()

        try:
            # Make progress bar busy
            progress_bar.set_busy_mode("Generating VPEs...")
            # Reload the model to ensure a clean state for VPE generation
            self.reload_model()

            # The references_to_vpe method will calculate and update self.reference_vpes
            new_vpes = self.references_to_vpe(references_dict, update_reference_vpes=True)

            if new_vpes:
                num_vpes = len(new_vpes)
                num_images = len(references_dict)
                message = f"Successfully generated {num_vpes} VPEs from {num_images} reference image(s)."
                self.status_bar.setText(message)
                QMessageBox.information(self, "VPEs Generated", message)
            else:
                message = "Could not generate VPEs. Ensure annotations are valid."
                self.status_bar.setText(message)
                QMessageBox.warning(self, "Generation Failed", message)

        except Exception as e:
            QMessageBox.critical(self, "Error Generating VPEs", f"An unexpected error occurred: {str(e)}")
            self.status_bar.setText("Error during VPE generation.")
        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()
    
    def references_to_vpe(self, reference_dict, update_reference_vpes=True):
        """
        Converts the contents of a reference dictionary to VPEs (Visual Prompt Embeddings).
        Reference dictionaries contain information about the visual prompts for each reference image:
        dict[image_path]: {bboxes, masks, cls}

        Args:
            reference_dict (dict): The reference dictionary containing visual prompts for each image.
            update_reference_vpes (bool): Whether to update self.reference_vpes with the results.

        Returns:
            list: List of individual VPE tensors (normalized), or None if empty reference_dict
        """
        # Check if the reference dictionary is empty
        if not reference_dict:
            return None
            
        # Create a list to hold the individual VPE tensors
        vpe_list = []

        for ref_path, ref_annotations in reference_dict.items():
            # Set the prompts to the model predictor
            self.loaded_model.predictor.set_prompts(ref_annotations)

            # Get the VPE from the model
            vpe = self.loaded_model.predictor.get_vpe(ref_path)
            
            # Normalize individual VPE
            vpe_normalized = torch.nn.functional.normalize(vpe, p=2, dim=-1)
            vpe_list.append(vpe_normalized)

        # Check if we have any valid VPEs
        if not vpe_list:
            return None
        
        # Update the reference_vpes list if requested
        if update_reference_vpes:
            self.reference_vpes = vpe_list
            
        return vpe_list

    def _apply_model_using_vpe(self, inputs):
        """
        Apply the model using VPEs. This method now supports clustering N VPEs
        into K prototypes before running a single multi-class prediction.
        
        Args:
            inputs (list): List of input images.
            
        Returns:
            list: List of prediction results.
        """
        # Note: This method may require 'from sklearn.cluster import KMeans'
        
        # First reload the model to clear any cached data
        self.reload_model()
        
        # 1. Gather all available VPEs from imported files and generated references
        combined_vpes = []
        if self.imported_vpes:
            combined_vpes.extend(self.imported_vpes)
        if self.reference_vpes:
            combined_vpes.extend(self.reference_vpes)
        
        if not combined_vpes:
            QMessageBox.warning(
                self,
                "No VPEs Available",
                "No VPEs are available for prediction. "
                "Please either load a VPE file or generate VPEs from reference images."
            )
            return []
        
        # 2. Generate K prototypes from the N available VPEs
        k = self.k_prototypes_spinbox.value()
        num_available_vpes = len(combined_vpes)
        prototype_vpes = []

        # If K is 0 (use all) or K >= N, no clustering is needed. Each VPE is a prototype.
        if k == 0 or k >= num_available_vpes:
            prototype_vpes = combined_vpes
        else:
            # Perform K-Means clustering to find K prototypes
            try:
                # Prepare tensor for scikit-learn: shape (N, E)
                all_vpes_tensor = torch.cat([vpe.squeeze(1) for vpe in combined_vpes], dim=0)
                vpes_np = all_vpes_tensor.cpu().numpy()

                # Lazily import KMeans to avoid making sklearn a hard dependency if not used
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(vpes_np)
                centroids_np = kmeans.cluster_centers_

                # Convert centroids back to a list of normalized PyTorch tensors
                centroids_tensor = torch.from_numpy(centroids_np).to(self.device)
                for i in range(k):
                    # Reshape centroid to (1, 1, E) for model compatibility
                    centroid = centroids_tensor[i].unsqueeze(0).unsqueeze(0)
                    normalized_centroid = torch.nn.functional.normalize(centroid, p=2, dim=-1)
                    prototype_vpes.append(normalized_centroid)
            except Exception as e:
                QMessageBox.critical(self, "Clustering Error", f"Failed to perform K-Means clustering: {e}")
                return []

        # 3. Configure the model with the K prototypes
        if not prototype_vpes:
            QMessageBox.warning(self, "Prototype Error", "Could not generate any prototypes for prediction.")
            return []

        # For backward compatibility, set self.vpe to the average of the final prototypes
        averaged_prototype = torch.cat(prototype_vpes).mean(dim=0, keepdim=True)
        self.vpe = torch.nn.functional.normalize(averaged_prototype, p=2, dim=-1)

        # Set up the model for multi-class detection with K proto-classes
        num_prototypes = len(prototype_vpes)
        proto_class_names = [f"object{i}" for i in range(num_prototypes)]
        
        # Stack prototypes into a single tensor of shape (1, K, E) for set_classes
        stacked_vpes = torch.cat(prototype_vpes, dim=1)

        self.loaded_model.is_fused = lambda: False 
        self.loaded_model.set_classes(proto_class_names, stacked_vpes)
        
        # 4. Make predictions on the target using the K prototypes
        results = self.loaded_model.predict(inputs[0],
                                            visual_prompts=[],
                                            imgsz=self.imgsz_spinbox.value(),
                                            conf=self.main_window.get_uncertainty_thresh(),
                                            iou=self.main_window.get_iou_thresh(),
                                            max_det=self.get_max_detections(),
                                            retina_masks=self.task == "segment")

        return [results]

    def _apply_model(self, inputs):
        """
        Apply the model to the target inputs. This method handles both image-based 
        references and VPE-based references.
        """        
        # Update the model with user parameters
        self.task = self.use_task_dropdown.currentText()
        
        self.loaded_model.conf = self.main_window.get_uncertainty_thresh()
        self.loaded_model.iou = self.main_window.get_iou_thresh()
        self.loaded_model.max_det = self.get_max_detections()
        
        # Get the reference information for the currently selected rows
        references_dict = self._get_references()
        
        # Check if the user is using VPE or Reference Images
        if self.reference_method_combo_box.currentText() == "VPE":
            # The VPE method will use pre-loaded/pre-generated VPEs.
            # The internal checks for whether any VPEs exist are now inside _apply_model_using_vpe.
            results = self._apply_model_using_vpe(inputs)
        else:  
            # Use Reference Images method - requires reference images
            if not references_dict:
                QMessageBox.warning(
                    self,
                    "No References Selected",
                    "No reference images with valid annotations were selected. "
                    "Please select at least one reference image."
                )
                return []
                
            results = self._apply_model_using_images(inputs, references_dict)

        return results

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
        """Process the results, merging K proto-class detections into a single target class."""
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
        target_label_name = self.reference_label.short_label_code

        for idx, results in enumerate(results_list):
            # Each Results is a list (within the results_list, [[], ]
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                # Clone the data tensor and set all classes to 0
                new_data = results[0].boxes.data.clone()
                new_data[:, 5] = 0   # The 6th column (index 5) is the class

                # Create a new Boxes object of the same type
                new_boxes = type(results[0].boxes)(new_data, results[0].boxes.orig_shape)
                results[0].boxes = new_boxes

                # Update the 'names' dictionary to map our single class ID (0)
                #    to the final target label name chosen by the user.
                results[0].names = {0: target_label_name}

                # Update path (original logic)
                results[0].path = image_path

                # Check if the work area is valid, or the image path is being used
                if work_areas and self.annotation_window.get_selected_tool() == "work_area":
                    # Map results from work area to the full image
                    results = MapResults().map_results_from_work_area(results[0], 
                                                                      raster, 
                                                                      work_areas[idx],
                                                                      self.task == "segment")
                else:
                    results = results[0]

                # Append the result object to the updated results list
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
        
    def show_vpe(self):
        """
        Show a visualization of the stored VPEs and their K-prototypes.
        """
        try:
            # 1. Gather all raw VPEs from imports and references
            vpes_with_source = []
            if self.imported_vpes:
                for vpe in self.imported_vpes:
                    vpes_with_source.append((vpe, "Import"))
            if self.reference_vpes:
                for vpe in self.reference_vpes:
                    vpes_with_source.append((vpe, "Reference"))

            if not vpes_with_source:
                QMessageBox.warning(
                    self,
                    "No VPEs Available",
                    "No VPEs available to visualize. Please load or generate VPEs first."
                )
                return

            raw_vpes = [vpe for vpe, source in vpes_with_source]
            num_raw = len(raw_vpes)
            
            # 2. Get K and determine if we need to cluster
            k = self.k_prototypes_spinbox.value()
            prototypes = []  # This will be the centroids if clustering is performed
            final_vpe = None
            clustering_performed = False

            # Case 1: We want to cluster (1 <= k < num_raw)
            if 1 <= k < num_raw:
                try:
                    # Prepare tensor for scikit-learn: shape (N, E)
                    all_vpes_tensor = torch.cat([vpe.squeeze(1) for vpe in raw_vpes], dim=0)
                    vpes_np = all_vpes_tensor.cpu().numpy()

                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(vpes_np)
                    centroids_np = kmeans.cluster_centers_

                    centroids_tensor = torch.from_numpy(centroids_np).to(self.device)
                    for i in range(k):
                        centroid = centroids_tensor[i].unsqueeze(0).unsqueeze(0)
                        normalized_centroid = torch.nn.functional.normalize(centroid, p=2, dim=-1)
                        prototypes.append(normalized_centroid)

                    # The final VPE is the average of the centroids
                    stacked_prototypes = torch.cat(prototypes, dim=1)  # Shape: (1, k, E)
                    averaged_prototype = stacked_prototypes.mean(dim=1, keepdim=True)  # Shape: (1, 1, E)
                    final_vpe = torch.nn.functional.normalize(averaged_prototype, p=2, dim=-1)
                    clustering_performed = True

                except Exception as e:
                    QMessageBox.critical(self, 
                                        "Clustering Error", 
                                        f"Could not perform clustering for visualization: {e}")
                    # If clustering fails, fall back to using all raw VPEs
                    prototypes = []
                    clustering_performed = False

            # Case 2: k==0 -> use all raw VPEs as prototypes (no clustering)
            if k == 0 or not clustering_performed:
                # We are not clustering, so we use all raw VPEs as prototypes
                # For visualization purposes, we'll show the raw VPEs and their average
                stacked_raw = torch.cat(raw_vpes, dim=1)  # Shape: (1, num_raw, E)
                averaged_raw = stacked_raw.mean(dim=1, keepdim=True)  # Shape: (1, 1, E)
                final_vpe = torch.nn.functional.normalize(averaged_raw, p=2, dim=-1)
                # Don't set prototypes here - we'll show raw VPEs separately in the visualization
            
            # Case 3: k >= num_raw -> use all raw VPEs as prototypes (no clustering needed)
            elif k >= num_raw:
                # We have more requested prototypes than available VPEs, so we use all VPEs
                stacked_raw = torch.cat(raw_vpes, dim=1)  # Shape: (1, num_raw, E)
                averaged_raw = stacked_raw.mean(dim=1, keepdim=True)  # Shape: (1, 1, E)
                final_vpe = torch.nn.functional.normalize(averaged_raw, p=2, dim=-1)
                # Don't set prototypes here - we'll show raw VPEs separately in the visualization

            # 3. Create and show the visualization dialog
            QApplication.setOverrideCursor(Qt.WaitCursor)
            dialog = VPEVisualizationDialog(
                vpes_with_source, 
                final_vpe, 
                prototypes=prototypes, 
                clustering_performed=clustering_performed,
                k_value=k,
                parent=self
            )
            QApplication.restoreOverrideCursor()
            dialog.exec_()

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error Visualizing VPE", f"An error occurred: {str(e)}")
        
    def deactivate_model(self):
        """
        Deactivate the currently loaded model and clean up resources.
        """
        self.loaded_model = None
        self.model_path = None
        
        # Clear all VPE-related data
        self.vpe_path_edit.clear()
        self.vpe_path = None
        self.vpe = None
        self.imported_vpes = []
        self.reference_vpes = []
        
        # Clean up references
        gc.collect()
        torch.cuda.empty_cache()
        
        # Untoggle all tools
        self.main_window.untoggle_all_tools()
        
        # Update status bar
        self.status_bar.setText("No model loaded")
        QMessageBox.information(self, "Model Deactivated", "Model deactivated")

        
class VPEVisualizationDialog(QDialog):
    """
    Dialog for visualizing VPE embeddings, now including K-prototypes.
    """
    def __init__(self, vpe_list_with_source, final_vpe=None, prototypes=None, 
                 clustering_performed=False, k_value=0, parent=None):
        """
        Initialize the dialog.
        
        Args:
            vpe_list_with_source (list): List of (VPE tensor, source_str) tuples for raw VPEs.
            final_vpe (torch.Tensor, optional): The final (averaged) VPE.
            prototypes (list, optional): List of K-prototype VPE tensors (cluster centroids).
            clustering_performed (bool): Whether clustering was performed.
            k_value (int): The K value used for clustering.
            parent (QWidget, optional): Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("VPE Visualization")
        self.resize(1000, 1000)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        
        # Store the VPEs and clustering info
        self.vpe_list_with_source = vpe_list_with_source
        self.final_vpe = final_vpe
        self.prototypes = prototypes if prototypes else []
        self.clustering_performed = clustering_performed
        self.k_value = k_value
        
        # Create the layout
        layout = QVBoxLayout(self)
        
        # Create the plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setTitle("PCA Visualization of Visual Prompt Embeddings", color="#000000", size="10pt")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot_widget)
        layout.addSpacing(20)
        
        # Add information label
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)
        
        # Create the button box
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Visualize the VPEs
        self.visualize_vpes()
    
    def visualize_vpes(self):
        """
        Apply PCA to all VPEs (raw, prototypes, final) and visualize them.
        """
        if not self.vpe_list_with_source:
            self.info_label.setText("No VPEs available to visualize.")
            return

        # 1. Collect all numpy arrays for PCA transformation
        raw_vpe_arrays = [vpe.detach().cpu().numpy().squeeze() for vpe, source in self.vpe_list_with_source]
        prototype_arrays = [p.detach().cpu().numpy().squeeze() for p in self.prototypes]
        
        all_arrays_for_pca = raw_vpe_arrays + prototype_arrays
        
        final_vpe_array = None
        if self.final_vpe is not None:
            final_vpe_array = self.final_vpe.detach().cpu().numpy().squeeze()
            all_arrays_for_pca.append(final_vpe_array)

        if len(all_arrays_for_pca) < 2:
            self.info_label.setText("At least 2 VPEs are needed for PCA visualization.")
            return
            
        # 2. Apply PCA
        all_vpes_stacked = np.vstack(all_arrays_for_pca)
        pca = PCA(n_components=2)
        vpes_2d = pca.fit_transform(all_vpes_stacked)
        
        # 3. Plot the results
        self.plot_widget.clear()
        legend = self.plot_widget.addLegend(colCount=3)
        
        # Slicing indices
        num_raw = len(raw_vpe_arrays)
        num_prototypes = len(prototype_arrays)

        # Determine if each raw VPE is effectively a prototype (k==0 or k>=N)
        each_vpe_is_prototype = (self.k_value == 0 or self.k_value >= num_raw)
        
        # Plot individual raw VPEs 
        colors = self.generate_distinct_colors(num_raw)
        for i, (vpe_tuple, vpe_2d) in enumerate(zip(self.vpe_list_with_source, vpes_2d[:num_raw])):
            source_char = 'I' if vpe_tuple[1] == 'Import' else 'R'
            
            # Use diamonds if each VPE is a prototype, circles otherwise
            symbol = 'd' if each_vpe_is_prototype else 'o'
            
            # If it's a prototype, add a black border
            pen = pg.mkPen(color='k', width=1.5) if each_vpe_is_prototype else None
            
            # Create label with prototype indicator if applicable
            name_suffix = " (Prototype)" if each_vpe_is_prototype else ""
            name = f"VPE {i+1} ({source_char}){name_suffix}"
            
            scatter = pg.ScatterPlotItem(
                x=[vpe_2d[0]], 
                y=[vpe_2d[1]], 
                brush=pg.mkColor(colors[i]), 
                pen=pen,
                size=15 if not each_vpe_is_prototype else 18,
                symbol=symbol, 
                name=name
            )
            self.plot_widget.addItem(scatter)
        
        # Plot K-Prototypes (blue diamonds) if we have any and explicit clustering was performed
        if self.prototypes and self.clustering_performed:
            prototype_vpes_2d = vpes_2d[num_raw: num_raw + num_prototypes]
            scatter = pg.ScatterPlotItem(
                x=prototype_vpes_2d[:, 0], 
                y=prototype_vpes_2d[:, 1],
                brush=pg.mkBrush(color=(0, 0, 255, 150)), 
                pen=pg.mkPen(color='k', width=1.5),
                size=18, 
                symbol='d', 
                name=f"K-Prototypes (K={self.k_value})"
            )
            self.plot_widget.addItem(scatter)

        # Plot the final (averaged) VPE (red star)
        if final_vpe_array is not None:
            final_vpe_2d = vpes_2d[-1]
            scatter = pg.ScatterPlotItem(
                x=[final_vpe_2d[0]], 
                y=[final_vpe_2d[1]],
                brush=pg.mkBrush(color='r'), 
                size=20, 
                symbol='star', 
                name="Final VPE (Avg)"
            )
            self.plot_widget.addItem(scatter)
        
        # 4. Update the information label
        orig_dim = self.vpe_list_with_source[0][0].shape[-1]
        explained_variance = sum(pca.explained_variance_ratio_)
        
        info_text = (f"Original dimension: {orig_dim} → Reduced to 2D\n"
                     f"Total explained variance: {explained_variance:.2%}\n"
                     f"PC1: {pca.explained_variance_ratio_[0]:.2%} variance, "
                     f"PC2: {pca.explained_variance_ratio_[1]:.2%} variance\n"
                     f"Number of raw VPEs: {num_raw}\n")
        
        if self.clustering_performed:
            info_text += f"Clustering performed with K={self.k_value}\n"
            info_text += f"Number of prototypes: {len(self.prototypes)}"
        else:
            if self.k_value == 0:
                info_text += f"No clustering (K=0): all {num_raw} raw VPEs used as prototypes"
            else:
                info_text += f"No clustering performed (K={self.k_value} >= {num_raw}): all raw VPEs used as prototypes"

        self.info_label.setText(info_text)
    
    def generate_distinct_colors(self, num_colors):
        """Generates visually distinct colors."""
        import random
        from colorsys import hsv_to_rgb
        
        colors = []
        for i in range(num_colors):
            hue = (i * 0.618033988749895) % 1.0
            saturation = random.uniform(0.6, 1.0)
            value = random.uniform(0.7, 1.0)
            r, g, b = hsv_to_rgb(hue, saturation, value)
            hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            colors.append(hex_color)
            
        return colors