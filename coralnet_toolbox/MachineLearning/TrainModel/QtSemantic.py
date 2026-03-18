import warnings
import os
import json
import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QLineEdit, QHBoxLayout, QPushButton, QFormLayout, QGroupBox,
                             QVBoxLayout, QWidget, QTabWidget, QFileDialog)

from coralnet_toolbox.MachineLearning.TrainModel.QtBase import Base
from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Semantic(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Train Semantic Model")

    def setup_dataset_layout(self):
        """Setup the dataset layout."""
        
        self.task = "semantic"
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

        # Class Mapping
        self.mapping_edit = QLineEdit()
        self.mapping_button = QPushButton("Browse...")
        self.mapping_button.clicked.connect(self.browse_class_mapping_file)

        class_mapping_layout = QHBoxLayout()
        class_mapping_layout.addWidget(self.mapping_edit)
        class_mapping_layout.addWidget(self.mapping_button)
        layout.addRow("Class Mapping:", class_mapping_layout)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def load_model_combobox(self):
        """Populate the model combobox with supported semseg YAMLs."""
        self.model_combo.clear()
        self.model_combo.setEditable(True)

        standard_models = [
            'yolo26n-semseg.yaml',
            'yolo26s-semseg.yaml',
            'yolo26m-semseg.yaml',
            'yolo26l-semseg.yaml',
            'yolo26x-semseg.yaml'
        ]

        self.model_combo.addItems(standard_models)

        # Default to the medium model if available
        try:
            self.model_combo.setCurrentIndex(standard_models.index('yolo26m-semseg.yaml'))
        except ValueError:
            if standard_models:
                self.model_combo.setCurrentIndex(0)

