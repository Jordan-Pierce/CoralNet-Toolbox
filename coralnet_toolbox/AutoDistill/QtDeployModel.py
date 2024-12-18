import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import gc

import torch
from autodistill.detection import CaptionOntology

from qtrangeslider import QRangeSlider
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog,
                             QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                             QMessageBox, QPushButton, QSlider, QVBoxLayout, QGroupBox)

from torch.cuda import empty_cache

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.ResultsProcessor import ResultsProcessor
from coralnet_toolbox.utilities import rasterio_to_numpy


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
        
        self.setWindowTitle("AutoDistill Deploy Model")
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
        self.model_dropdown.addItems(["GroundingDINO-SwinT", "GroundingDINO-SwinB"])
        layout.addWidget(self.model_dropdown)
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_ontology_layout(self):
        """
        Setup ontology mapping section in a group box.
        """
        group_box = QGroupBox("Ontology Mapping")
        layout = QVBoxLayout()
        
        add_remove_layout = QHBoxLayout()
        
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_ontology_pair)
        add_remove_layout.addWidget(self.remove_button)
        
        self.add_button = QPushButton("Add")  
        self.add_button.clicked.connect(self.add_ontology_pair)
        add_remove_layout.addWidget(self.add_button)
        
        layout.addLayout(add_remove_layout)
        self.ontology_layout = layout
        
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
        self.area_threshold_slider.setValue((int(current_min * 100), int(current_max * 100)))
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
        min_val, max_val = self.area_threshold_slider.value()  # Returns tuple of values
        self.area_thresh_min = min_val / 100.0
        self.area_thresh_max = max_val / 100.0
        self.main_window.update_area_thresh(self.area_thresh_min, self.area_thresh_max)
        self.area_threshold_label.setText(f"{self.area_thresh_min:.2f} - {self.area_thresh_max:.2f}")

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
        if not hasattr(self.main_window, 'sam_deploy_model_dialog'):
            return False
        
        self.sam_dialog = self.main_window.sam_deploy_model_dialog

        if not self.sam_dialog.loaded_model:
            self.use_sam_dropdown.setCurrentText("False")
            QMessageBox.critical(self, "Error", "Please deploy the SAM model first.")
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

            # Get the name of the model to load
            model_name = self.model_dropdown.currentText()

            if model_name != self.model_name:
                self.load_new_model(model_name)
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

    def load_new_model(self, model_name):
        """
        Load a new model based on the selected model name.

        Args:
            model_name: Name of the model to load.
            uncertainty_thresh: Threshold for uncertainty.
        """
        if "GroundingDINO" in model_name:
            from coralnet_toolbox.AutoDistill.Models.GroundingDINOModel import GroundingDINO

            model = model_name.split("-")[1].strip()
            self.model_name = model_name
            self.loaded_model = GroundingDINO(ontology=self.ontology,
                                              box_threshold=0.025,
                                              text_threshold=0.025,
                                              model=model)

    def predict(self, image_paths=None):
        """
        Make Autodistill predictions on the given inputs
        """
        if not self.loaded_model:
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
            
            # Create a results processor
            results_processor = ResultsProcessor(self.main_window, 
                                                 self.class_mapping,
                                                 uncertainty_thresh=self.main_window.get_uncertainty_thresh(),
                                                 iou_thresh=self.main_window.get_iou_thresh(),
                                                 min_area_thresh=self.main_window.get_area_thresh_min(),
                                                 max_area_thresh=self.main_window.get_area_thresh_max())
            
            results = results_processor.from_supervision(results, image, image_path, self.class_mapping)
            
            # Update the progress bar
            progress_bar.update_progress()

            if self.use_sam_dropdown.currentText() == "True":
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