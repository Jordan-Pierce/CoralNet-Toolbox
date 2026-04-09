from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QVBoxLayout, QGroupBox, QCheckBox, QFormLayout, 
                             QLabel, QSlider, QListWidget, QListWidgetItem)

from coralnet_toolbox.MachineLearning.VideoInference.QtBase import Base


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Detect(Base):
    """Dialog for detection video inference."""
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Detection Video Inference")

        self.task = "detect"

    def showEvent(self, event):
        self.device = self.main_window.device
        self.showMaximized()
        super().showEvent(event)

    def add_annotators_to_form(self, form_layout):
        """Add annotators section to the model form layout."""
        self.annotator_list_widget = QListWidget()
        self.annotator_types = [
            ("LabelAnnotator", "Label Annotator"),
            ("TrackerAnnotator", "Tracker Annotator"),
            ("BoxAnnotator", "Box Annotator"),
            ("RoundBoxAnnotator", "Round Box Annotator"),
            ("BoxCornerAnnotator", "Box Corner Annotator"),
            ("ColorAnnotator", "Color Annotator"),
            ("CircleAnnotator", "Circle Annotator"),
            ("DotAnnotator", "Dot Annotator"),
            ("TriangleAnnotator", "Triangle Annotator"),
            ("EllipseAnnotator", "Ellipse Annotator"),
            ("PercentageBarAnnotator", "Percentage Bar Annotator"),
            ("BlurAnnotator", "Blur Annotator"),
            ("PixelateAnnotator", "Pixelate Annotator"),
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
        self.initialize_iou_threshold()
        self.initialize_area_threshold()

