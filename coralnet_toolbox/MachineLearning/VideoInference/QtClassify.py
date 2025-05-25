from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QGroupBox, QCheckBox, QFormLayout, QLabel, QSlider

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
        
    def setup_parameters_layout(self):
        """
        Setup parameter control section in a group box.
        """
        group_box = QGroupBox("Model Parameters")
        layout = QFormLayout()

        # Confidence threshold controls (instead of uncertainty)
        self.uncertainty_thresh_slider = QSlider(Qt.Horizontal)
        self.uncertainty_thresh_slider.setRange(0, 100)
        self.uncertainty_thresh_slider.setValue(int(self.uncertainty_thresh * 100))
        self.uncertainty_thresh_slider.setTickPosition(QSlider.TicksBelow)
        self.uncertainty_thresh_slider.setTickInterval(10)
        self.uncertainty_thresh_slider.valueChanged.connect(self.update_uncertainty_label)
        self.uncertainty_thresh_label = QLabel(f"{self.uncertainty_thresh:.2f}")
        layout.addRow("Uncertainty Threshold", self.uncertainty_thresh_slider)
        layout.addRow("", self.uncertainty_thresh_label)
        
        group_box.setLayout(layout)
        self.controls_layout.addWidget(group_box)
        
    def setup_annotators_layout(self):
        """Setup the annotator selection layout."""
        group_box = QGroupBox("Annotators to Use")
        layout = QVBoxLayout()
        
        # Store checkboxes for later access
        self.annotator_checkboxes = {}
        
        # List of annotator types (except label annotator, which is always on)
        annotator_types = [
            ("BoxAnnotator", "Box Annotator"),
        ]
        for key, label in annotator_types:
            cb = QCheckBox(label)
            cb.setChecked(False)
            layout.addWidget(cb)
            self.annotator_checkboxes[key] = cb
        group_box.setLayout(layout)
        self.controls_layout.addWidget(group_box)
        
    def initialize_thresholds(self):
        """Initialize all threshold sliders with current values."""
        self.initialize_uncertainty_threshold()