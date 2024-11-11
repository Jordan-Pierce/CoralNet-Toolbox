import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QFileDialog, QScrollArea, QMessageBox, QCheckBox, QWidget, QVBoxLayout,
                             QLabel, QLineEdit, QDialog, QHBoxLayout, QPushButton, QComboBox, QSpinBox,
                             QFormLayout, QTabWidget, QDoubleSpinBox)

from toolbox.MachineLearning.TrainModel.QtBase import Base


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Segment(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setup_generic_layout("Train Segmentation Model")
        
    def load_model_combobox(self):
        """Load the model combobox with the available models."""
        self.model_combobox.clear()
        self.model_combo.setEditable(True)
        self.model_combobox.addItems(['yolov8n-seg.pt'
                                      'yolo11n-seg.pt',
                                      'yolov8s-seg.pt', 
                                      'yolo11s-seg.pt',
                                      'yolov8m-seg.pt', 
                                      'yolo11m-seg.pt',
                                      'yolov8l-seg.pt', 
                                      'yolo11l-seg.pt',
                                      'yolov8x-seg.pt'
                                      'yolo11x-seg.pt'])

