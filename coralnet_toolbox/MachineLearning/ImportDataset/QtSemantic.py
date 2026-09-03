from PyQt5.QtWidgets import (QGroupBox, QLabel, QFormLayout, QVBoxLayout)

from coralnet_toolbox.MachineLearning.ImportDataset.QtBase import Base


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Semantic(Base):
    """
    Dialog for importing datasets for semantic segmentation.
    """
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle('Import Semantic Segmentation Dataset')
        self.task = 'semantic'

    def setup_info_layout(self):
        """Setup the info layout"""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_text = ("Import a YOLO-formatted Semantic Segmentation dataset. Single-channel masks are converted "
                     "to one Mask annotation per image, where each pixel value is the class ID.\n"
                     "Masks must share the base name of their image, and only lossless formats are read "
                     "(PNG, TIF, TIFF). Pixel value 255 is the reserved ignore label.")
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

        # Unlike detection and instance segmentation there is no representation
        # to pick: a semantic mask imports as one MaskAnnotation per image, so
        # the "Import as" combo is deliberately absent. Base.start_processing
        # checks for it rather than assuming every task has one.
        import_as_label = QLabel("Import as:")
        description = QLabel("Masks (one per image)")
        description.setToolTip("Semantic masks are imported as a single MaskAnnotation per image.\n"
                               "Each pixel value is the class ID it belongs to.")
        layout.addRow(import_as_label, description)

        self.layout.addWidget(group_box)
