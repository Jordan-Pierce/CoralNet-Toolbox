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
        self.setup_generic_layout("Segment Batch Inference")
        
    def apply(self):
        """
        Apply batch inference for instance segmentation.
        """
        self.image_paths = self.get_selected_image_paths()
        self.batch_inference()

    def batch_inference(self):
        """
        Perform batch inference on the selected images.
        """
        # Make predictions on each image's annotations
        progress_bar = ProgressBar(self, title="Batch Inference")
        progress_bar.show()
        progress_bar.start_progress(len(self.image_paths))

        if self.loaded_models['segment'] is not None:
            self.deploy_model_dialog.predict_segmentation(image_paths=self.image_paths)

        progress_bar.stop_progress()
        progress_bar.close()