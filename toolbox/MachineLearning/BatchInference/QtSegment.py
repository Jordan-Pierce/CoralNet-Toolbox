import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby
from operator import attrgetter

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox, QCheckBox, QVBoxLayout,
                             QLabel, QDialog, QDialogButtonBox, QGroupBox, QSlider, QButtonGroup)

from toolbox.MachineLearning.BatchInference.QtBase import Base

from toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Segment(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Segment Batch Inference")
        
        self.deploy_model_dialog = main_window.segment_deploy_model_dialog
        self.loaded_model = self.deploy_model_dialog.loaded_model
        
    def setup_task_specific_layout(self):
        """
        Set up the layout with both generic and classification-specific options.
        """
        pass

    def batch_inference(self):
        """
        Perform batch inference on the selected images.
        """
        # Make predictions on each image's annotations
        progress_bar = ProgressBar(self, title="Batch Inference")
        progress_bar.show()
        progress_bar.start_progress(len(self.image_paths))

        if self.loaded_model is not None:
            self.deploy_model_dialog.predict_segmentation(image_paths=self.image_paths)

        progress_bar.stop_progress()
        progress_bar.close()