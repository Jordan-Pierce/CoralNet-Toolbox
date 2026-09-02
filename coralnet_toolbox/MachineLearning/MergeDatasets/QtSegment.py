from coralnet_toolbox.MachineLearning.MergeDatasets.QtBase import Base


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Segment(Base):
    # Instance segmentation shares detection's on-disk layout: both keep their
    # annotations in .txt files whose leading token is the class index, and the
    # merge only ever rewrites that token, never the geometry after it.
    task = 'segment'

    def __init__(self, parent=None):
        """
        Initializes the MergeDatasetsDialog.

        :param parent: Parent widget, default is None.
        """
        super().__init__(parent)
        self.setWindowTitle("Merge Instance Segmentation Datasets")
