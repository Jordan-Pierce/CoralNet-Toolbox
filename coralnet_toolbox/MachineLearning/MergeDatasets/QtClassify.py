from coralnet_toolbox.MachineLearning.MergeDatasets.QtBase import Base


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Classify(Base):
    def __init__(self, parent=None):
        """
        Initializes the MergeDatasetsDialog.

        :param parent: Parent widget, default is None.
        """
        super().__init__(parent)
        self.setWindowTitle("Merge Classification Datasets")
        self.task = 'classify'