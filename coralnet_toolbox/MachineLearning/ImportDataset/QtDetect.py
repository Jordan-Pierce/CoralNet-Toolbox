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