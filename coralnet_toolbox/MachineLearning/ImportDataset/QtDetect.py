from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QGroupBox, QLabel, QFormLayout, QComboBox, QVBoxLayout)

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
        """Setup the info layout"""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_text = ("Import a YOLO-formatted Detection dataset. Bounding boxes are converted to Rectangle "
                     "or Polygon annotations, and labels are created from the dataset's class names.")
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

        import_as_label = QLabel("Import as:")
        self.import_as_combo = QComboBox()
        self.import_as_combo.addItems(["Rectangles (Default)", "Polygons"])
        self.import_as_combo.setToolTip("Format for imported annotations.\nRectangles (Default): Bounding boxes from detection data.\nPolygons: Convert bboxes to polygon corners (rectangular regions).")
        layout.addRow(import_as_label, self.import_as_combo)

        self.layout.addWidget(group_box)
