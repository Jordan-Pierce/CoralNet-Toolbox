import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                             QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                             QLineEdit, QMessageBox, QPushButton, QScrollArea,
                             QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from toolbox.MachineLearning.TrainModel.QtBase import Base

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Segment(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Train Segmentation Model")
        
    def setup_generic_layout(self):
        """
        Adopt the layout from the Base class but ensure task is set correctly
        """
        self.task = "segment"
        super().setup_generic_layout()
        
    def load_model_combobox(self):
        """Load the model combobox with the available models."""
        self.model_combo.clear()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(['yolov8n-seg.pt',
                                   'yolo11n-seg.pt',
                                   'yolov8s-seg.pt', 
                                   'yolo11s-seg.pt',
                                   'yolov8m-seg.pt', 
                                   'yolo11m-seg.pt',
                                   'yolov8l-seg.pt', 
                                   'yolo11l-seg.pt',
                                   'yolov8x-seg.pt',
                                   'yolo11x-seg.pt'])

