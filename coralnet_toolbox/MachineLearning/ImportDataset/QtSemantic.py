from PyQt5.QtWidgets import (QGroupBox, QLabel, QFormLayout)

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

        self.layout.insertWidget(0, group_box)
