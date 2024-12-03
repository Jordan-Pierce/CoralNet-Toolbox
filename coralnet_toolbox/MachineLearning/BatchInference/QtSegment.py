import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from coralnet_toolbox.MachineLearning.BatchInference.QtBase import Base

from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Segment(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Segment Batch Inference")
        
        self.deploy_model_dialog = main_window.segment_deploy_model_dialog
        
    def setup_task_specific_layout(self):
        """
        Set up the layout with both generic and classification-specific options.
        """
        pass

    def batch_inference(self):
        """
        Perform batch inference on the selected images.
        """
        self.loaded_model = self.deploy_model_dialog.loaded_model
        
        # Make predictions on each image's annotations
        progress_bar = ProgressBar(self, title="Batch Inference")
        progress_bar.show()
        progress_bar.start_progress(len(self.image_paths))

        if self.loaded_model is not None:
            self.deploy_model_dialog.predict(inputs=self.image_paths)

        progress_bar.stop_progress()
        progress_bar.close()