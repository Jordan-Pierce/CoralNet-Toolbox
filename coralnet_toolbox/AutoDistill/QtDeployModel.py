import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import gc
import traceback

import torch
from torch.cuda import empty_cache
from autodistill.detection import CaptionOntology

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog,
                             QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                             QMessageBox, QPushButton, QSlider, QVBoxLayout, QGroupBox)


from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.ResultsProcessor import ResultsProcessor

from coralnet_toolbox.utilities import rasterio_open
from coralnet_toolbox.utilities import rasterio_to_numpy

from coralnet_toolbox.Icons import get_icon


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

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("AutoDistill Deploy Model (Ctrl + 5)")
        self.resize(400, 325)

        # Initialize variables
        self.imgsz = 1024
        self.iou_thresh = 0.20
        self.uncertainty_thresh = 0.30
        self.area_thresh_min = 0.00
        self.area_thresh_max = 0.40
        self.loaded_model = None
        self.model_name = None
        self.ontology = None
        self.class_mapping = {}
        self.ontology_pairs = []

        # Create the layout
        self.layout = QVBoxLayout(self)

        # Setup the info layout
        self.setup_info_layout()
        # Setup the model layout
        self.setup_models_layout()
        # Setup the ontology layout
        self.setup_ontology_layout()
        # Setup the parameter layout
        self.setup_parameters_layout()
        # Setup the SAM layout
        self.setup_sam_layout()
        # Setup the button layout
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
        self.update_label_options()
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
        info_label = QLabel("Choose a model to deploy and use.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_models_layout(self):
        """
        Setup model selection dropdown in a group box.
        """
        group_box = QGroupBox("Model Selection")
        layout = QVBoxLayout()

        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems([
            # "OmDetTurbo-SwinT",
            "OWLViT",
            "GroundingDINO-SwinT",
            "GroundingDINO-SwinB",
        ])

        layout.addWidget(self.model_dropdown)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_ontology_layout(self):
        """
        Setup ontology mapping section in a group box with a single fixed pair.
        """
        group_box = QGroupBox("Ontology Mapping")
        layout = QVBoxLayout()

        # Create a single pair of text input and dropdown
        pair_layout = QHBoxLayout()

        self.text_input = QLineEdit()
        self.text_input.setMaxLength(100)  # Cap the width at 100 characters
        self.text_input.setPlaceholderText("Enter keyword or description")

        self.label_dropdown = QComboBox()
        self.label_dropdown.addItems([label.short_label_code for label in self.label_window.labels])

        pair_layout.addWidget(self.text_input)
        pair_layout.addWidget(self.label_dropdown)

        layout.addLayout(pair_layout)

        # Store the single pair for later reference
        self.ontology_pairs = [(self.text_input, self.label_dropdown)]

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

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

    def update_label_options(self):
        """
        Update the label options in the single ontology pair based on available labels.
        """
        label_options = [label.short_label_code for label in self.label_window.labels]
        previous_label = self.label_dropdown.currentText()
        self.label_dropdown.clear()
        self.label_dropdown.addItems(label_options)
        if previous_label in label_options:
            self.label_dropdown.setCurrentText(previous_label)

    def get_ontology_mapping(self):
        """
        Retrieve the ontology mapping from the single user input.

        Returns:
            Dictionary mapping text to label code.
        """
        ontology_mapping = {}
        if self.text_input.text() != "":
            ontology_mapping[self.text_input.text()] = self.label_dropdown.currentText()
        return ontology_mapping

    def is_sam_model_deployed(self):
        """
        Check if the SAM model is deployed and update the checkbox state accordingly.
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
        Load the selected model with the current configuration.
        """
        # Get the ontology mapping
        ontology_mapping = self.get_ontology_mapping()

        if not ontology_mapping:
            QMessageBox.critical(self,
                                 "Error",
                                 "Please provide at least one ontology mapping.")
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        # Show a progress bar
        progress_bar = ProgressBar(self.annotation_window, title="Loading Model")
        progress_bar.show()
        try:
            # Set the ontology
            self.ontology = CaptionOntology(ontology_mapping)
            # Set the class mapping
            self.class_mapping = {k: v for k, v in enumerate(self.ontology.classes())}

            # Get the name of the model to load
            model_name = self.model_dropdown.currentText()

            if model_name != self.model_name:
                self.load_new_model(model_name)
            else:
                # Update the model with the new ontology
                self.loaded_model.ontology = self.ontology

            progress_bar.finish_progress()
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

    def load_new_model(self, model_name):
        """
        Load a new model based on the selected model name.

        Args:
            model_name: Name of the model to load.
            uncertainty_thresh: Threshold for uncertainty.
        """
        if "GroundingDINO" in model_name:
            from coralnet_toolbox.AutoDistill.Models.GroundingDINO import GroundingDINOModel

            model = model_name.split("-")[1].strip()
            self.model_name = model_name
            self.loaded_model = GroundingDINOModel(ontology=self.ontology,
                                                   model=model,
                                                   device=self.main_window.device)

        elif "OmDetTurbo" in model_name:
            from coralnet_toolbox.AutoDistill.Models.OmDetTurbo import OmDetTurboModel

            self.model_name = model_name
            self.loaded_model = OmDetTurboModel(ontology=self.ontology,
                                                device=self.main_window.device)

        elif "OWLViT" in model_name:
            from coralnet_toolbox.AutoDistill.Models.OWLViT import OWLViTModel

            self.model_name = model_name
            self.loaded_model = OWLViTModel(ontology=self.ontology,
                                            device=self.main_window.device)

    def predict(self, image_paths=None):
        """
        Make Autodistill predictions on the given inputs
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

        # Start a progress bar
        progress_bar = ProgressBar(self.annotation_window, title="Making Predictions")
        progress_bar.show()
        progress_bar.start_progress(len(image_paths))

        try:
            # Loop through the image paths
            for image_path in image_paths:
                progress_bar.update_progress()
                inputs = self._get_inputs(image_path)
                if inputs is None:
                    continue

                results = self._apply_model(inputs)
                results = self._update_results(results_processor, results, inputs, image_path)
                results = self._apply_sam(results, image_path)
                self._process_results(results_processor, results)
        except Exception as e:
            print("An error occurred during prediction:", e)
            print(traceback.format_exc())
        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.finish_progress()
            progress_bar.stop_progress()
            progress_bar.close()

        gc.collect()
        empty_cache()  # Assuming this function is defined elsewhere

    def _get_inputs(self, inputs):
        """Get the inputs for the model prediction."""
        if isinstance(inputs, str):
            inputs = rasterio_to_numpy(rasterio_open(inputs))
        return inputs

    def _apply_model(self, inputs):
        """Apply the model to the inputs."""
        return self.loaded_model.predict(inputs)

    def _update_results(self, results_processor, results, inputs, image_path):
        """Update the results to match Ultralytics format."""           
        return results_processor.from_supervision(results,
                                                  inputs,
                                                  image_path,
                                                  self.class_mapping)

    def _apply_sam(self, results, image_path):
        """Apply SAM to the results if needed."""
        # Check if SAM model is deployed
        if self.use_sam_dropdown.currentText() == "True":
            self.task = 'segment'
            results = self.sam_dialog.predict_from_results(results, image_path)
        else:
            self.task = 'detect'

        return results

    def _process_results(self, result_processor, results):
        """Process the results using the result processor."""
        # Process the detections
        if self.task == 'segment':
            result_processor.process_segmentation_results(results)
        else:
            result_processor.process_detection_results(results)

    def deactivate_model(self):
        """
        Deactivate the currently loaded model and clean up resources.
        """
        # Clear the model
        self.loaded_model = None
        self.model_name = None
        # Clear cache
        gc.collect()
        torch.cuda.empty_cache()
        # Untoggle all tools
        self.main_window.untoggle_all_tools()
        # Update status bar
        self.status_bar.setText("No model loaded")
        QMessageBox.information(self, "Model Deactivated", "Model deactivated")
