from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QVBoxLayout, QGroupBox, QCheckBox, QFormLayout, QAbstractButton,
                             QLabel, QSlider, QListWidget, QListWidgetItem, QHBoxLayout, QLineEdit, 
                             QPushButton, QAbstractItemView)

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
        self.showMaximized()
        super().showEvent(event)
        
    def setup_model_layout(self):
        """Setup the model and parameters layout using a QFormLayout within a group box."""
        group_box = QGroupBox("Model and Parameters")
        form_layout = QFormLayout()

        # Model path input
        model_path_layout = QHBoxLayout()
        self.model_edit = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_model)
        model_path_layout.addWidget(self.model_edit)
        model_path_layout.addWidget(browse_btn)
        form_layout.addRow(QLabel("Model Path:"), model_path_layout)

        # Class filter
        self.class_filter_widget = QListWidget()
        self.class_filter_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        class_filter_layout = QVBoxLayout()
        class_filter_layout.addWidget(self.class_filter_widget)
        btn_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.deselect_all_btn = QPushButton("Deselect All")
        self.select_all_btn.clicked.connect(self.select_all_classes)
        self.deselect_all_btn.clicked.connect(self.deselect_all_classes)
        btn_layout.addWidget(self.select_all_btn)
        btn_layout.addWidget(self.deselect_all_btn)
        class_filter_layout.addLayout(btn_layout)
        form_layout.addRow(QLabel("Class Filter:"), class_filter_layout)

        # Uncertainty threshold slider
        self.uncertainty_thresh_slider = QSlider(Qt.Horizontal)
        self.uncertainty_thresh_slider.setRange(0, 100)
        self.uncertainty_thresh_slider.setValue(int(self.uncertainty_thresh * 100))
        self.uncertainty_thresh_slider.valueChanged.connect(self.update_uncertainty_label)
        form_layout.addRow(QLabel("Uncertainty Threshold:"), self.uncertainty_thresh_slider)

        group_box.setLayout(form_layout)
        self.controls_layout.addWidget(group_box)
        
    def setup_annotators_layout(self):
        """Setup the annotator selection layout using a QListWidget with checkable items."""
        group_box = QGroupBox("Annotators to Use")
        layout = QVBoxLayout()

        self.annotator_list_widget = QListWidget()
        # List of annotator types (except label annotator, which is always on)
        self.annotator_types = [
            ("BoxAnnotator", "Box Annotator"),
            ("PercentageBarAnnotator", "Percentage Bar Annotator"),
        ]
        for key, label in self.annotator_types:
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, key)
            self.annotator_list_widget.addItem(item)
        layout.addWidget(self.annotator_list_widget)

        group_box.setLayout(layout)
        self.controls_layout.addWidget(group_box)
        
    def initialize_thresholds(self):
        """Initialize all threshold sliders with current values."""
        self.initialize_uncertainty_threshold()
