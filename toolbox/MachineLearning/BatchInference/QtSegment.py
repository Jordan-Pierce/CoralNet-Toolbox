from toolbox.MachineLearning.BatchInference.QtBase import Base

class Segment(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setup_generic_layout()

    def apply(self):
        """
        Apply batch inference for instance segmentation.
        """
        self.image_paths = self.get_selected_image_paths()
        self.batch_inference('segment')
