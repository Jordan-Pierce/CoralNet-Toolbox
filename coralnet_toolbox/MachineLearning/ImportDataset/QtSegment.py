from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QGroupBox, QLabel, QFormLayout, QComboBox)

from coralnet_toolbox.MachineLearning.ImportDataset.QtBase import Base


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Segment(Base):
    """
    Dialog for importing datasets for instance segmentation.
    """
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)        
        self.setWindowTitle('Import Instance Segmentation Dataset')
        self.task = 'segment'
        
    def setup_info_layout(self):
        group_box = QGroupBox("Import Options")
        layout = QFormLayout(group_box)

        # Import as combo box
        import_as_label = QLabel("Import as:")
        self.import_as_combo = QComboBox()
        self.import_as_combo.addItems(["Polygons (Default)", "Rectangles"])
        layout.addRow(import_as_label, self.import_as_combo)

        self.layout.insertWidget(0, group_box)