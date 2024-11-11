import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import gc

import torch
from autodistill.detection import CaptionOntology
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                             QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                             QMessageBox, QPushButton, QSlider, QSpinBox,
                             QVBoxLayout)
from torch.cuda import empty_cache

from toolbox.QtProgressBar import ProgressBar
from toolbox.QtRangeSlider import QRangeSlider
from toolbox.ResultsProcessor import ResultsProcessor
from toolbox.utilities import rasterio_to_numpy

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DeployModelDialog(QDialog):
    """
    Dialog for deploying and managing AutoDistill models.
    Allows users to load, configure, and deactivate models, as well as make predictions on images.
    """

    def __init__(self, main_window, parent=None):
        """
        Initialize the AutoDistillDeployModelDialog.

        Args:
            main_window: The main application window.
            parent: The parent widget, default is None.
        """
        super().__init__(parent)
        self.main_window = main_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

        # Initialize instance variables
        self.imgsz = 1024
        self.uncertainty_thresh = 0.25
        self.area_thresh_min = 0.01
        self.area_thresh_max = 0.75
        self.loaded_model = None
        self.model_name = None
        self.ontology = None
        self.class_mapping = {}
        self.ontology_pairs = []
        
        self.use_sam = False

        self.setup_ui()
        
    def showEvent(self, event):
        """
        Handle the show event to update label options and sync uncertainty threshold.

        Args:
            event: The event object.
        """
        super().showEvent(event)
        self.update_label_options()
        self.initialize_uncertainty_threshold()

    def setup_ui(self):
        """Setup the user interface components."""
        # Window configuration
        self.setWindowTitle("AutoDistill Deploy Model")
        self.resize(300, 250)
        
        # Create main layout
        self.main_layout = QVBoxLayout(self)
        # Setup model selection
        self.setup_model_selection()
        # Setup ontology mapping 
        self.setup_ontology_section()
        # Setup parameter controls
        self.setup_parameter_controls()
        # Setup action buttons
        self.setup_action_buttons()
        # Setup status bar
        self.status_bar = QLabel("No model loaded")
        self.main_layout.addWidget(self.status_bar)

    def setup_model_selection(self):
        """
        Setup model selection dropdown.
        """
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["GroundingDINO"])
        self.main_layout.addWidget(self.model_dropdown)

    def setup_ontology_section(self):
        """
        Setup ontology mapping section.
        """
        self.ontology_layout = QVBoxLayout()
        
        # Add/Remove buttons layout
        add_remove_layout = QHBoxLayout()
        
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_ontology_pair)
        add_remove_layout.addWidget(self.remove_button)
        
        self.add_button = QPushButton("Add")  
        self.add_button.clicked.connect(self.add_ontology_pair)
        add_remove_layout.addWidget(self.add_button)
        
        self.ontology_layout.addLayout(add_remove_layout)
        self.main_layout.addLayout(self.ontology_layout)

    def setup_parameter_controls(self):
        """
        Setup parameter control sliders and inputs.
        """
        self.form_layout = QFormLayout()
        
        # Resize image dropdown
        self.resize_image_dropdown = QComboBox()
        self.resize_image_dropdown.addItems(["True", "False"])
        self.resize_image_dropdown.setCurrentIndex(0)
        self.form_layout.addRow("Resize Image:", self.resize_image_dropdown)
        
        # Image size control
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(512, 4096)
        self.imgsz_spinbox.setSingleStep(1024)
        self.imgsz_spinbox.setValue(self.imgsz)
        self.form_layout.addRow("Image Size (imgsz):", self.imgsz_spinbox)
        
        # Area threshold controls
        self.setup_area_threshold_controls() 
            
        # Uncertainty threshold controls
        self.setup_uncertainty_threshold_controls()
        
        # Add the form to the layout
        self.main_layout.addLayout(self.form_layout)
        
        # SAM checkbox
        self.use_sam_checkbox = QCheckBox("Use SAM for creating Polygons")
        self.use_sam_checkbox.stateChanged.connect(self.is_sam_model_deployed)
        self.main_layout.addWidget(self.use_sam_checkbox)
        self.use_sam = self.use_sam_checkbox

    def setup_area_threshold_controls(self):
        """
        Setup area threshold slider and label.
        """
        self.area_threshold_slider = QRangeSlider()
        self.area_threshold_slider.setRange(0, 100)
        self.area_threshold_slider.setValue((int(self.area_thresh_min * 100), int(self.area_thresh_max * 100)))
        self.area_threshold_slider.rangeChanged.connect(self.update_area_label)

        self.area_threshold_label = QLabel(f"{self.area_thresh_min:.2f} - {self.area_thresh_max:.2f}")
        self.form_layout.addRow("Area Threshold", self.area_threshold_slider)
        self.form_layout.addRow("", self.area_threshold_label)

    def setup_uncertainty_threshold_controls(self):
        """
        Setup uncertainty threshold slider and label.
        """
        self.uncertainty_threshold_slider = QSlider(Qt.Horizontal)
        self.uncertainty_threshold_slider.setRange(0, 100)
        self.uncertainty_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.uncertainty_threshold_slider.setTickInterval(5)
        self.uncertainty_threshold_slider.valueChanged.connect(self.update_uncertainty_label)

        self.uncertainty_threshold_label = QLabel("")
        self.form_layout.addRow("Uncertainty Threshold", self.uncertainty_threshold_slider)
        self.form_layout.addRow("", self.uncertainty_threshold_label)

    def setup_action_buttons(self):
        """
        Setup load and deactivate buttons.
        """
        button_layout = QHBoxLayout()
        
        load_button = QPushButton("Load Model")
        load_button.clicked.connect(self.load_model)
        button_layout.addWidget(load_button)
        
        deactivate_button = QPushButton("Deactivate Model")
        deactivate_button.clicked.connect(self.deactivate_model)
        button_layout.addWidget(deactivate_button)
        
        self.main_layout.addLayout(button_layout)

    def update_area_label(self, min_val, max_val):
        """
        Update the area threshold label based on slider values.

        Args:
            min_val: Minimum slider value.
            max_val: Maximum slider value.
        """
        # Update the area threshold values
        self.area_thresh_min = min_val / 100.0
        self.area_thresh_max = max_val / 100.0
        self.area_threshold_label.setText(f"Area Threshold: {self.area_thresh_min:.2f} - {self.area_thresh_max:.2f}")

    def update_uncertainty_label(self):
        """
        Update the uncertainty threshold label based on slider value.
        """
        # Convert the slider value to a ratio (0-1)
        value = self.uncertainty_threshold_slider.value() / 100.0
        self.main_window.update_uncertainty_thresh(value)
        self.uncertainty_threshold_label.setText(f"{value:.2f}")
        self.uncertainty_thresh = self.uncertainty_threshold_slider.value() / 100.0
        
    def initialize_uncertainty_threshold(self):
        """Initialize the uncertainty threshold slider with the current value from main window"""
        current_value = self.main_window.get_uncertainty_thresh()
        self.uncertainty_threshold_slider.setValue(int(current_value * 100))
        self.uncertainty_threshold_label.setText(f"{current_value:.2f}")
        self.uncertainty_thresh = current_value

    def on_uncertainty_changed(self, value):
        """
        Update the slider and label when the uncertainty threshold changes.

        Args:
            value: New uncertainty threshold value.
        """
        # Update the slider and label when the shared data changes
        self.uncertainty_threshold_slider.setValue(int(value * 100))
        self.uncertainty_threshold_label.setText(f"{value:.2f}")
        self.uncertainty_thresh = self.uncertainty_threshold_slider.value() / 100.0

    def update_label_options(self):
        """
        Update the label options in ontology pairs based on available labels.
        """
        label_options = [label.short_label_code for label in self.label_window.labels]
        for _, label_dropdown in self.ontology_pairs:
            previous_label = label_dropdown.currentText()
            label_dropdown.clear()
            label_dropdown.addItems(label_options)
            if previous_label in label_options:
                label_dropdown.setCurrentText(previous_label)

    def add_ontology_pair(self):
        """
        Add a new ontology pair input (text input and label dropdown).
        """
        pair_layout = QHBoxLayout()

        text_input = QLineEdit()
        text_input.setMaxLength(100)  # Cap the width at 100 characters
        label_dropdown = QComboBox()
        label_dropdown.addItems([label.short_label_code for label in self.label_window.labels])

        pair_layout.addWidget(text_input)
        pair_layout.addWidget(label_dropdown)

        self.ontology_pairs.append((text_input, label_dropdown))
        self.ontology_layout.insertLayout(self.ontology_layout.count() - 1, pair_layout)

    def remove_ontology_pair(self):
        """
        Remove the last ontology pair input if more than one exists.
        """
        if len(self.ontology_pairs) > 1:
            pair = self.ontology_pairs.pop()
            pair[0].deleteLater()
            pair[1].deleteLater()

            # Remove the layout
            item = self.ontology_layout.itemAt(self.ontology_layout.count() - 2)
            item.layout().deleteLater()
            
    def is_sam_model_deployed(self):
        """
        Check if the SAM model is deployed and update the checkbox state accordingly.
        """
        self.sam_dialog = self.main_window.sam_deploy_model_dialog

        if not self.sam_dialog.loaded_model:
            # Ensure that the checkbox is not checked
            self.sender().setChecked(False)
            QMessageBox.warning(self, "SAM Model", "SAM model not currently deployed")
            return False

        return True

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
            # Get the ontology mapping
            ontology_mapping = self.get_ontology_mapping()

            # Set the ontology
            self.ontology = CaptionOntology(ontology_mapping)
            # Set the class mapping
            self.class_mapping = {k: v for k, v in enumerate(self.ontology.classes())}

            # Threshold for confidence
            uncertainty_thresh = self.get_uncertainty_threshold()

            # Get the name of the model to load
            model_name = self.model_dropdown.currentText()

            if model_name != self.model_name:
                self.load_new_model(model_name, uncertainty_thresh)
            else:
                # Update the model with the new ontology
                self.loaded_model.ontology = self.ontology

            self.status_bar.setText(f"Model loaded: {model_name}")
            QMessageBox.information(self, "Model Loaded", "Model loaded successfully")

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Model", str(e))

        # Stop the progress bar
        progress_bar.stop_progress()
        progress_bar.close()
        # Restore cursor
        QApplication.restoreOverrideCursor()
        # Exit the dialog box
        self.accept()

    def get_ontology_mapping(self):
        """
        Retrieve the ontology mapping from user inputs.

        Returns:
            Dictionary mapping texts to label codes.
        """
        ontology_mapping = {}
        for text_input, label_dropdown in self.ontology_pairs:
            if text_input.text() != "":
                ontology_mapping[text_input.text()] = label_dropdown.currentText()
        return ontology_mapping
    
    def get_area_thresh(self, image):
        """
        Calculate area thresholds based on image dimensions.

        Args:
            image_path: Path to the image.

        Returns:
            Tuple of (min_area_thresh, max_area_thresh).
        """
        h, w, _ = image.shape
        area_thresh_min = (h * w) * self.area_thresh_min
        area_thresh_max = (h * w) * self.area_thresh_max
        return area_thresh_min, area_thresh_max

    def get_uncertainty_threshold(self):
        """
        Get the uncertainty threshold, limiting it to a maximum of 0.10.

        Returns:
            Adjusted uncertainty threshold value.
        """
        if self.main_window.get_uncertainty_thresh() < 0.10:
            return self.main_window.get_uncertainty_thresh()
        else:
            return 0.10  # Arbitrary value to prevent too many detections

    def load_new_model(self, model_name, uncertainty_thresh):
        """
        Load a new model based on the selected model name.

        Args:
            model_name: Name of the model to load.
            uncertainty_thresh: Threshold for uncertainty.
        """
        if model_name == "GroundingDINO":
            from autodistill_grounding_dino import GroundingDINO
            self.model_name = model_name
            self.loaded_model = GroundingDINO(ontology=self.ontology,
                                              box_threshold=uncertainty_thresh,
                                              text_threshold=uncertainty_thresh)

    def predict(self, image_paths=None):
        """
        Make predictions on the given image paths using the loaded model.

        Args:
            image_paths: List of image paths to process. If None, uses the current image.
        """
        if not self.loaded_model:
            QMessageBox.critical(self, "Error", "No model loaded")
            return
   
        if not image_paths:
            image_paths = [self.annotation_window.current_image_path]
            
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
            
        progress_bar = ProgressBar(self.annotation_window, title=f"Making {self.model_name} Predictions")
        progress_bar.show()
        progress_bar.start_progress(len(image_paths))

        for image_path in image_paths:
            # Open the image
            image = self.main_window.image_window.rasterio_open(image_path)
            image = rasterio_to_numpy(image)
            # Predict the image
            results = self.loaded_model.predict(image)
            # Perform NMS thresholding
            results = results.with_nms(self.main_window.get_iou_thresh())
            # Perform area thresholding
            min_area_thresh, max_area_thresh = self.get_area_thresh(image)
            results = results[results.area >= min_area_thresh]
            results = results[results.area <= max_area_thresh]
            # Create a results processor
            results_processor = ResultsProcessor(self.main_window, self.class_mapping)
            results = results_processor.from_supervision(results, image, image_path, self.class_mapping)
            
            # Update the progress bar
            progress_bar.update_progress()

            if self.use_sam.isChecked():
                # Apply SAM to the detection results
                results = self.sam_dialog.predict_from_results(results, self.class_mapping)
                # Process the segmentation results
                results_processor.process_segmentation_results(results)
            else:
                # Process the detection results
                results_processor.process_detection_results(results)
                
        # Stop the progress bar
        progress_bar.stop_progress()
        progress_bar.close()
                
        # Make cursor normal
        QApplication.restoreOverrideCursor()
        gc.collect()
        empty_cache()

    def deactivate_model(self):
        """
        Deactivate the currently loaded model and clean up resources.
        """
        self.loaded_model = None
        self.model_name = None
        gc.collect()
        torch.cuda.empty_cache()
        self.main_window.untoggle_all_tools()
        self.status_bar.setText("No model loaded")
        QMessageBox.information(self, "Model Deactivated", "Model deactivated")