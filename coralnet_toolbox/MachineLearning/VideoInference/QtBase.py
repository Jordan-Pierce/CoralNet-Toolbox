import os
import cv2
import numpy as np
import supervision as sv
from datetime import datetime

from shapely.geometry import Polygon

from ultralytics import YOLO

from PyQt5.QtCore import Qt, QTimer, QRect, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QLabel, QLineEdit, QPushButton, QFileDialog, 
                             QWidget, QListWidget, QListWidgetItem, QFrame,
                             QAbstractItemView, QFormLayout, QComboBox, QSizePolicy,
                             QMessageBox, QApplication)

from coralnet_toolbox.MachineLearning.VideoInference.QtVideoWidget import VideoRegionWidget

from coralnet_toolbox.Common import ThresholdsWidget

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

            
class Base(QDialog):
    """Dialog for video inference with region selection and parameter controls."""
    def __init__(self, main_window, parent=None):
        """Initialize the Video Inference dialog."""
        super().__init__(parent)
        self.main_window = main_window
        
        self.setWindowIcon(get_icon("coralnet.png"))
        self.setWindowTitle("Video Inference")
        
        # Optionally set a minimum size
        self.setMinimumSize(800, 600)
        
        # Allow maximizing
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Initialize device as 'cpu' by default
        self.device = 'cpu'
                
        # Initialize parameters
        self.video_path = ""
        self.output_dir = ""
        
        self.model_path = ""
        
        self.task = None                            # Task parameter, modified in subclasses
                 
        self.video_region_widget = None             # Initialized in setup_video_layout

        # Main layout
        self.layout = QHBoxLayout(self)
        self.controls_layout = QVBoxLayout()
        self.layout.addLayout(self.controls_layout, 30)
        self.video_layout = QVBoxLayout()
        self.layout.addLayout(self.video_layout, 70)

        # Setup the input layout
        self.setup_io_layout()
        # Setup the model and parameters layout
        self.setup_model_layout()
        # Setup Run/Cancel buttons
        self.setup_buttons_layout()
        # Setup the video player widget
        self.setup_video_layout()

    def setup_io_layout(self):
        """Setup the input video group with a file browser using QFormLayout."""
        group_box = QGroupBox("IO Parameters")
        layout = QFormLayout()

        self.input_edit = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_video)
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(browse_btn)
        input_widget = QWidget()
        input_widget.setLayout(input_layout)
        layout.addRow(QLabel("Input Video:"), input_widget)

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Provide directory to output video...")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_output)
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(browse_btn)
        output_widget = QWidget()
        output_widget.setLayout(output_layout)
        layout.addRow(QLabel("Output Directory:"), output_widget)

        group_box.setLayout(layout)
        self.controls_layout.addWidget(group_box)

    def setup_model_layout(self):
        """Setup the model input, parameters, and class filter in a single group using QFormLayout."""
        group_box = QGroupBox("Model and Parameters")
        form_layout = QFormLayout()

        # Model path input
        model_layout = QHBoxLayout()
        self.model_edit = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(browse_btn)
        model_widget = QWidget()
        model_widget.setLayout(model_layout)
        form_layout.addRow(QLabel("Model Path:"), model_widget)

        # Searchable Class Filter
        class_filter_layout = QVBoxLayout()
        self.class_filter_search = QLineEdit()
        self.class_filter_search.setPlaceholderText("Search classes...")
        self.class_filter_search.textChanged.connect(self.filter_class_list)
        class_filter_layout.addWidget(self.class_filter_search)
        self.class_filter_widget = QListWidget()
        self.class_filter_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        class_filter_layout.addWidget(self.class_filter_widget)
        class_filter_widget_container = QWidget()
        class_filter_widget_container.setLayout(class_filter_layout)
        form_layout.addRow(QLabel("Class Filter:"), class_filter_widget_container)

        # Select All / Deselect All buttons
        btn_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.deselect_all_btn = QPushButton("Deselect All")
        self.select_all_btn.clicked.connect(self.select_all_classes)
        self.deselect_all_btn.clicked.connect(self.deselect_all_classes)
        btn_layout.addWidget(self.select_all_btn)
        btn_layout.addWidget(self.deselect_all_btn)
        btn_widget = QWidget()
        btn_widget.setLayout(btn_layout)
        form_layout.addRow(QLabel(""), btn_widget)

        # Add ThresholdsWidget for all threshold controls
        self.thresholds_widget = ThresholdsWidget(
            self.main_window,
            show_max_detections=True,
            show_uncertainty=True,
            show_iou=True,
            show_area=True,
            title="Thresholds"
        )
        form_layout.addRow(self.thresholds_widget)
        
        # Connect threshold changes to update inference parameters
        if hasattr(self.thresholds_widget, 'max_detections_spinbox'):
            self.thresholds_widget.max_detections_spinbox.valueChanged.connect(
                lambda: self.update_inference_parameters()
            )
        if hasattr(self.thresholds_widget, 'uncertainty_threshold_slider'):
            self.thresholds_widget.uncertainty_threshold_slider.valueChanged.connect(
                lambda: self.update_inference_parameters()
            )
        if hasattr(self.thresholds_widget, 'iou_threshold_slider'):
            self.thresholds_widget.iou_threshold_slider.valueChanged.connect(
                lambda: self.update_inference_parameters()
            )
        if hasattr(self.thresholds_widget, 'area_threshold_min_slider'):
            self.thresholds_widget.area_threshold_min_slider.valueChanged.connect(
                lambda: self.update_inference_parameters()
            )
        if hasattr(self.thresholds_widget, 'area_threshold_max_slider'):
            self.thresholds_widget.area_threshold_max_slider.valueChanged.connect(
                lambda: self.update_inference_parameters()
            )
        
        # Add annotators section (child class specific)
        self.add_annotators_to_form(form_layout)
        
        # Add some spacing
        self.controls_layout.addSpacing(10)
        
        # Inference enable/disable buttons
        inference_button_layout = QHBoxLayout()
        self.enable_inference_btn = QPushButton("Enable Inference")
        self.enable_inference_btn.clicked.connect(self.enable_inference)
        self.enable_inference_btn.setFocusPolicy(Qt.NoFocus)  # Prevent focus/highlighting
        inference_button_layout.addWidget(self.enable_inference_btn)
        self.disable_inference_btn = QPushButton("Disable Inference")
        self.disable_inference_btn.clicked.connect(self.disable_inference)
        self.disable_inference_btn.setFocusPolicy(Qt.NoFocus)  # Prevent focus/highlighting
        self.disable_inference_btn.setEnabled(False)           # Initially disabled
        inference_button_layout.addWidget(self.disable_inference_btn)
        form_layout.addRow(inference_button_layout)

        group_box.setLayout(form_layout)
        self.controls_layout.addWidget(group_box)

    def add_annotators_to_form(self, form_layout):
        """Add annotators section to the form layout. To be implemented by subclasses."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def filter_class_list(self, text):
        """Filter the class filter QListWidget based on the search text."""
        text = text.lower()
        for i in range(self.class_filter_widget.count()):
            item = self.class_filter_widget.item(i)
            item.setHidden(text not in item.text().lower())

    def setup_video_layout(self):
        """Setup the video region widget directly without an external group box."""
        self.video_region_widget = VideoRegionWidget(self)
        self.video_layout.addWidget(self.video_region_widget)

    def setup_buttons_layout(self):
        """Setup the Exit button at the bottom of the controls layout."""
        btn_layout = QHBoxLayout()
        
        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.cleanup_and_exit)
        self.exit_btn.setFocusPolicy(Qt.NoFocus)                # Prevent focus/highlighting
        btn_layout.addWidget(self.exit_btn)
        
        self.controls_layout.addLayout(btn_layout)

    def cleanup_and_exit(self):
        """Perform cleanup operations and close the dialog."""
        self.cleanup()
        self.reject()

    def cleanup(self):
        """Clean up all resources before closing the dialog."""
        try:
            # Disable inference first
            if hasattr(self, 'video_region_widget') and self.video_region_widget:
                # Disable inference
                self.video_region_widget.enable_inference(False)
                
                # Stop video playback
                if hasattr(self.video_region_widget, 'stop_video'):
                    self.video_region_widget.stop_video()
                
                # Clear regions
                self.video_region_widget.clear_regions()
                
                # Clean up inference engine and model
                if hasattr(self.video_region_widget, 'inference_engine') and self.video_region_widget.inference_engine:
                    if hasattr(self.video_region_widget.inference_engine, 'cleanup'):
                        self.video_region_widget.inference_engine.cleanup()
                
                # Release video capture if exists
                if hasattr(self.video_region_widget, 'cap') and self.video_region_widget.cap:
                    self.video_region_widget.cap.release()
                    self.video_region_widget.cap = None
                    
            # Clear class filter
            if hasattr(self, 'class_filter_widget') and self.class_filter_widget:
                self.class_filter_widget.clear()
            
            # Reset UI state
            if hasattr(self, 'enable_inference_btn') and hasattr(self, 'disable_inference_btn'):
                self.enable_inference_btn.setEnabled(True)
                self.disable_inference_btn.setEnabled(False)
            
            # Clear file paths
            self.video_path = ""
            self.output_dir = ""
            self.model_path = ""
            
        except Exception as e:
            # Log the error but don't prevent closing
            print(f"Error during cleanup: {e}")

    def closeEvent(self, event):
        """Override closeEvent to ensure cleanup when window is closed with X button."""
        self.cleanup()
        event.accept()
        
    def browse_video(self):
        """Open file dialog to select input video (filtered to common formats)."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;All Files (*)"
        )
        if file_name:
            self.input_edit.setText(file_name)
            self.video_path = file_name
            self.video_region_widget.load_video(file_name, self.output_dir)

    def browse_output(self):
        """Open directory dialog to select output directory."""
        dir_name = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_name:
            self.output_edit.setText(dir_name)  # AttributeError - no output_edit attribute
            self.output_dir = dir_name          # Sets output_dir on wrong object
            # If video already loaded, update output dir for widget
            if self.video_path:
                self.video_region_widget.load_video(self.video_path, dir_name)

    def browse_model(self):
        """Open file dialog to select model file (filtered to .pt, .pth)."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "Model Files (*.pt *.pth);;All Files (*)"
        )
        if file_name:
            self.model_edit.setText(file_name)
            self.model_path = file_name
            self.video_region_widget.inference_engine.load_model(file_name, task=self.task)
            self.populate_class_filter()
        
    def get_selected_annotators(self):
        """Return a list of selected annotator class names from the QListWidget."""
        selected = []
        for i in range(self.annotator_list_widget.count()):
            item = self.annotator_list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))
        return selected

    def initialize_thresholds(self):
        """Initialize all threshold sliders with current values."""
        self.thresholds_widget.initialize_thresholds()

    def update_inference_parameters(self):
        """Update inference parameters in the video region widget."""
        self.video_region_widget.inference_engine.set_inference_params(
            self.thresholds_widget.get_uncertainty_thresh(),
            self.thresholds_widget.get_iou_thresh(),
            self.thresholds_widget.get_area_thresh_min(),
            self.thresholds_widget.get_area_thresh_max(),
            self.thresholds_widget.get_max_detections()
        )
        
    def populate_class_filter(self):
        """Populate the class filter widget with class names from the loaded model."""
        self.class_filter_widget.clear()
        for idx, name in enumerate(self.video_region_widget.inference_engine.class_names):
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, idx)
            item.setCheckState(Qt.Checked)  # Select all by default
            self.class_filter_widget.addItem(item)
        self.class_filter_widget.itemChanged.connect(self.update_selected_classes)
        self.update_selected_classes()  # Ensure selected_classes is up to date

    def update_selected_classes(self):
        """Update the selected classes based on the class filter widget."""
        selected = []
        for i in range(self.class_filter_widget.count()):
            item = self.class_filter_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))
        self.video_region_widget.inference_engine.set_selected_classes(selected)

    def select_all_classes(self):
        """Select all classes in the class filter widget."""
        for i in range(self.class_filter_widget.count()):
            item = self.class_filter_widget.item(i)
            item.setCheckState(Qt.Checked)
        self.update_selected_classes()

    def deselect_all_classes(self):
        """Deselect all classes in the class filter widget."""
        for i in range(self.class_filter_widget.count()):
            item = self.class_filter_widget.item(i)
            item.setCheckState(Qt.Unchecked)
        self.update_selected_classes()
        
    def clear_regions(self):
        """Clear all regions from the video region widget and update display."""
        self.video_region_widget.clear_regions()

    def enable_inference(self):
        """Enable inference on the video region."""
        if not self.video_region_widget.inference_engine.model:
            QMessageBox.warning(self, "Model Not Loaded", "Please load a model before enabling inference.")
            return
        
        # Only set the flag and update UI, do not change video state
        self.video_region_widget.enable_inference(True)
        self.enable_inference_btn.setEnabled(False)
        self.disable_inference_btn.setEnabled(True)
        
    def disable_inference(self):
        """Disable inference on the video region."""
        # Only set the flag and update UI, do not change video state
        self.video_region_widget.enable_inference(False)
        self.enable_inference_btn.setEnabled(True)
        self.disable_inference_btn.setEnabled(False)
        
        # Refresh the current frame without inference
        self.video_region_widget.seek(self.video_region_widget.current_frame_number)
