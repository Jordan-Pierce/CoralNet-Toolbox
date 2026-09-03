from PyQt5.QtWidgets import QLabel

from coralnet_toolbox.MachineLearning.MergeDatasets.QtBase import Base


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Semantic(Base):
    # Read by Base during setup, so it has to be a class attribute rather than
    # something assigned after super().__init__ has already built the layout.
    task = 'semantic'

    def __init__(self, parent=None):
        """
        Initializes the MergeDatasetsDialog.

        :param parent: Parent widget, default is None.
        """
        super().__init__(parent)
        self.setWindowTitle("Merge Semantic Segmentation Datasets")

    def setup_info_layout(self):
        """Setup the info layout, with the notes specific to mask rasters.

        The mask note is added into the Information box rather than a box of its
        own, which would add a second frame's worth of height to a dialog that
        already has to fit on a laptop screen.
        """
        super().setup_info_layout()

        note = QLabel("Class IDs are stored as pixel values, so every mask is rewritten to the merged "
                      "numbering: 'background' is placed at ID 0, 255 stays ignore/unlabeled, and a "
                      "pixel holding a class no dataset contributes is set to ignore.")
        note.setWordWrap(True)
        note.setToolTip("Merging datasets whose masks were exported with different background settings\n"
                        "is supported, but the unlabeled regions of each keep the meaning they were\n"
                        "exported with: background (0) in one, ignore (255) in the other.")

        # The Information group box is the last widget Base added.
        self.layout.itemAt(self.layout.count() - 1).widget().layout().addWidget(note)
