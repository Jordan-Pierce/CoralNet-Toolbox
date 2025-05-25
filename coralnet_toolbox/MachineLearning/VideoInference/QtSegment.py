from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QCheckBox

from coralnet_toolbox.MachineLearning.VideoInference.QtBase import Base


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Segment(Base):
    """Dialog for segmentation video inference."""
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Segmentation Video Inference")

        self.task = "segment"
        
    def showEvent(self, event):
        self.showMaximized()
        super().showEvent(event)
        
    def setup_annotators_layout(self):
        """Setup the annotator selection layout."""
        group_box = QGroupBox("Annotators to Use")
        layout = QVBoxLayout()
        
        # Store checkboxes for later access
        self.annotator_checkboxes = {}
        
        # List of annotator types (except label annotator, which is always on)
        annotator_types = [
            ("BoxAnnotator", "Box Annotator"),
            ("BoxCornerAnnotator", "Box Corner Annotator"),
            ("DotAnnotator", "Dot Annotator"),
            ("HaloAnnotator", "Halo Annotator"),
            ("PercentageBarAnnotator", "Percentage Bar Annotator"),
            ("MaskAnnotator", "Mask Annotator"),
            ("PolygonAnnotator", "Polygon Annotator"),
        ]
        for key, label in annotator_types:
            cb = QCheckBox(label)
            cb.setChecked(False)
            layout.addWidget(cb)
            self.annotator_checkboxes[key] = cb
        group_box.setLayout(layout)
        self.controls_layout.addWidget(group_box)
