from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QVBoxLayout, QGroupBox, QCheckBox, QFormLayout, QAbstractButton,
                             QLabel, QSlider, QListWidget, QListWidgetItem, QHBoxLayout, QLineEdit, 
                             QPushButton, QAbstractItemView, QWidget)

from coralnet_toolbox.MachineLearning.VideoInference.QtBase import Base


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Classify(Base):
    """Dialog for classification video inference."""
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Classification Video Inference")
        
        self.task = "classify"
    
    def showEvent(self, event):
        self.device = self.main_window.device
        self.showMaximized()
        super().showEvent(event)
        
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

        # Class filter
        self.class_filter_widget = QListWidget()
        self.class_filter_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        form_layout.addRow(QLabel("Class Filter:"), self.class_filter_widget)

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

        # Parameter sliders (only uncertainty for classification)
        self.uncertainty_thresh_slider = QSlider(Qt.Horizontal)
        self.uncertainty_thresh_slider.setRange(0, 100)
        self.uncertainty_thresh_slider.setValue(int(self.uncertainty_thresh * 100))
        self.uncertainty_thresh_slider.valueChanged.connect(self.update_uncertainty_label)
        self.uncertainty_thresh_label = QLabel(f"{self.uncertainty_thresh:.2f}")
        uncertainty_layout = QHBoxLayout()
        uncertainty_layout.addWidget(self.uncertainty_thresh_slider)
        uncertainty_layout.addWidget(self.uncertainty_thresh_label)
        uncertainty_widget = QWidget()
        uncertainty_widget.setLayout(uncertainty_layout)
        form_layout.addRow(QLabel("Uncertainty Threshold:"), uncertainty_widget)
        
        # Add annotators section (child class specific)
        self.add_annotators_to_form(form_layout)
        
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
        """Add annotators section to the model form layout."""
        self.annotator_list_widget = QListWidget()
        # List of annotator types (LabelAnnotator now included and checked by default)
        self.annotator_types = [
            ("LabelAnnotator", "Label Annotator"),
            ("BoxAnnotator", "Box Annotator"),
            ("PercentageBarAnnotator", "Percentage Bar Annotator"),
        ]
        for key, label in self.annotator_types:
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            if key == "LabelAnnotator":
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, key)
            self.annotator_list_widget.addItem(item)
        
        form_layout.addRow(QLabel("Annotators:"), self.annotator_list_widget)

    def setup_annotators_layout(self):
        """Setup the annotator selection layout using a QListWidget with checkable items."""
        # This method is now handled by add_annotators_to_form
        pass
    
    def initialize_thresholds(self):
        """Initialize all threshold sliders with current values."""
        self.initialize_uncertainty_threshold()
