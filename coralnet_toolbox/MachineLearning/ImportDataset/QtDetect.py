from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QGroupBox, QLabel, QFormLayout, QComboBox)

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
        self.task = 'detect'
        
    def setup_info_layout(self):
        group_box = QGroupBox("Import Options")
        layout = QFormLayout(group_box)

        import_as_label = QLabel("Import as:")
        self.import_as_combo = QComboBox()
        self.import_as_combo.addItems(["Rectangles (Default)", "Polygons"])
        layout.addRow(import_as_label, self.import_as_combo)

        self.layout.insertWidget(0, group_box)