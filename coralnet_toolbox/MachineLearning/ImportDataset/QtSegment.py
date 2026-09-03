from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QGroupBox, QLabel, QFormLayout, QComboBox, QVBoxLayout)

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
        """Setup the info layout"""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_text = ("Import a YOLO-formatted Instance Segmentation dataset. Polygon masks are converted to "
                     "Polygon or Rectangle annotations, and labels are created from the dataset's class names.")
        info_label = QLabel(info_text)

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        info_label.setToolTip("Every image in the dataset is copied into the project directory;\n"
                              "the original files are left untouched.")
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_options_layout(self):
        group_box = QGroupBox("Import Options")
        layout = QFormLayout(group_box)

        # Import as combo box
        import_as_label = QLabel("Import as:")
        self.import_as_combo = QComboBox()
        self.import_as_combo.addItems(["Polygons (Default)", "Rectangles"])
        self.import_as_combo.setToolTip("Format for imported annotations.\nPolygons (Default): Use precise polygon shapes from segmentation data.\nRectangles: Simplify to bounding boxes (faster, less precise).")
        layout.addRow(import_as_label, self.import_as_combo)

        self.layout.addWidget(group_box)
