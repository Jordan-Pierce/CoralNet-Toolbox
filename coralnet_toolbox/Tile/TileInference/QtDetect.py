import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from coralnet_toolbox.Tile.TileInference.QtBase import Base


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Detect(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Tile Detection Inference")
        
        self.annotation_type = "object_detection"
        self.deploy_model_dialog = main_window.detect_deploy_model_dialog

