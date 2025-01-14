import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from coralnet_toolbox.Tile.TileInference.QtBase import Base


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Segment(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Tile Segmentation Inference")
        
        self.annotation_type = "instance_segmentation"
        self.deploy_model_dialog = main_window.segment_deploy_model_dialog
