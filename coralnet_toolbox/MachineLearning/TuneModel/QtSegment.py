import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from PyQt5.QtWidgets import (QLineEdit, QHBoxLayout, QPushButton, QFormLayout, QGroupBox)

from coralnet_toolbox.MachineLearning.TuneModel.QtBase import Base
from coralnet_toolbox.MachineLearning.Community.cfg import get_available_configs


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Segment(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Tune Segmentation Model")

    def setup_dataset_layout(self):
        """Setup the dataset layout."""
        
        self.task = "segment"
        self.imgsz = 640
        self.batch = 4
        
        group_box = QGroupBox("Dataset")
        layout = QFormLayout()

        # Dataset YAML
        self.dataset_edit = QLineEdit()
        self.dataset_button = QPushButton("Browse...")
        self.dataset_button.clicked.connect(self.browse_dataset_yaml)

        dataset_yaml_layout = QHBoxLayout()
        dataset_yaml_layout.addWidget(self.dataset_edit)
        dataset_yaml_layout.addWidget(self.dataset_button)
        layout.addRow("Dataset YAML:", dataset_yaml_layout)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def load_model_combobox(self):
        """Load the model combobox with the available models."""
        self.model_combo.clear()
        self.model_combo.setEditable(True)

        standard_models = ['yolov5n-seg.pt',
                           'yolov5s-seg.pt',
                           'yolov5m-seg.pt',
                           'yolov5l-seg.pt',
                           'yolov5x-seg.pt',
                           'yolov8n-seg.pt',
                           'yolov8s-seg.pt',
                           'yolov8m-seg.pt',
                           'yolov8l-seg.pt',
                           'yolov8x-seg.pt',
                           'yolov9c-seg.pt',
                           'yolov9e-seg.pt',
                           'yolo11n-seg.pt',
                           'yolo11s-seg.pt',
                           'yolo11m-seg.pt',
                           'yolo11l-seg.pt',
                           'yolo11x-seg.pt',
                           'yolo12n-seg.pt',
                           'yolo12s-seg.pt',
                           'yolo12m-seg.pt',
                           'yolo12l-seg.pt',
                           'yolo12x-seg.pt']
        
        self.model_combo.addItems(standard_models)
                
        # Add community models
        community_configs = get_available_configs(task=self.task)
        if community_configs:
            self.model_combo.insertSeparator(len(standard_models))
            self.model_combo.addItems(list(community_configs.keys()))
            
        # Set the default model
        self.model_combo.setCurrentIndex(standard_models.index('yolov8n-seg.pt'))