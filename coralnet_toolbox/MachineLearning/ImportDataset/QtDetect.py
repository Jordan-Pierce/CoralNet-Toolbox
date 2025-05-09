from PyQt5.QtWidgets import (QGroupBox, QVBoxLayout, QLabel)

from coralnet_toolbox.MachineLearning.ImportDataset.QtBase import Base


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Detect(Base):
    """
    Dialog for importing datasets for object detection.
    """
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)        
        self.setWindowTitle('Import Object Detection Dataset')
        
    def setup_info_layout(self):
        """Setup the info layout"""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()
        
        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Import a YOLO-formatted Detection dataset.")
        
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)