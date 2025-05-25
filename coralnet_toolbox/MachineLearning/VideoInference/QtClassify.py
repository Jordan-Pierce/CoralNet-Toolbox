from coralnet_toolbox.MachineLearning.VideoInference.QtBase import Base


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Classify(Base):
    """Dialog for classification video inference."""
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Classification Video Inference")
        
        self.task = "classify"
    
    def showEvent(self, event):
        self.showMaximized()
        super().showEvent(event)