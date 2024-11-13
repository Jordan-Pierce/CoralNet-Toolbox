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


class Classify(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Train Classification Model")

    def setup_generic_layout(self):
        """
        Adopt the layout from the Base class but ensure task is set correctly
        """
        self.task = "classify"
        super().setup_generic_layout()
        
    def load_model_combobox(self):
        """Load the model combobox with the available models."""
        self.model_combo.clear()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(['yolov8n-cls.pt',
                                   'yolo11n-cls.pt',
                                   'yolov8s-cls.pt',
                                   'yolo11s-cls.pt', 
                                   'yolov8m-cls.pt', 
                                   'yolo11m-cls.pt',
                                   'yolov8l-cls.pt', 
                                   'yolo11l-cls.pt',
                                   'yolov8x-cls.pt',
                                   'yolo11x-cls.pt'])
