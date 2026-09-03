from coralnet_toolbox.MachineLearning.MergeDatasets.QtBase import Base


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Detect(Base):
    # Read by Base during setup, so it has to be a class attribute rather than
    # something assigned after super().__init__ has already built the layout.
    task = 'detect'

    def __init__(self, parent=None):
        """
        Initializes the MergeDatasetsDialog.

        :param parent: Parent widget, default is None.
        """
        super().__init__(parent)
        self.setWindowTitle("Merge Detection Datasets")
