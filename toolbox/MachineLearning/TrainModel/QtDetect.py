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


class Detect(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Train Detection Model")
        
    def setup_generic_layout(self):
        """
        Adopt the layout from the Base class but ensure task is set correctly
        """
        self.task = "detect"
        super().setup_generic_layout()
        
        
    def load_model_combobox(self):
        """Load the model combobox with the available models."""
        self.model_combo.clear()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(['yolov8n.pt',
                                   'yolov8n-oiv7.pt',
                                   'yolo11n.pt',
                                   'yolov8s.pt', 
                                   'yolov8s-oiv7.pt',
                                   'yolo11s.pt',
                                   'yolov8m.pt', 
                                   'yolov8m-oiv7.pt',
                                   'yolo11m.pt',
                                   'yolov8l.pt', 
                                   'yolov8l-oiv7.pt',
                                   'yolo11l.pt',
                                   'yolov8x.pt',
                                   'yolov8x-oiv7.pt',
                                   'yolo11x.pt'])

