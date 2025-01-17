import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from PyQt5.QtWidgets import (QFormLayout, QHBoxLayout, QLineEdit, QPushButton, QGroupBox)

from coralnet_toolbox.MachineLearning.TrainModel.QtBase import Base


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Segment(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Train Segmentation Model")

        self.task = "segment"

    def setup_dataset_layout(self):
        """Setup the dataset layout."""
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
        """Load the model combobox with the available models."""
        self.model_combo.clear()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(['yolov5n-seg.pt',
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
                                   'yolo11x-seg.pt'])
