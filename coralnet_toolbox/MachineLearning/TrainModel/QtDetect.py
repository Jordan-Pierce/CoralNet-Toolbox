import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from PyQt5.QtWidgets import (QLineEdit, QHBoxLayout, QPushButton, QFormLayout, QGroupBox)

from coralnet_toolbox.MachineLearning.TrainModel.QtBase import Base


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Detect(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Train Detection Model")

        self.task = "detect"
        self.imgsz = 640
        self.batch = 4

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
        self.model_combo.addItems(['yolov3u.pt',
                                   'yolov3-sppu.pt',
                                   'yolov3-tinyu.pt',
                                   'yolov5nu.pt',
                                   'yolov5su.pt',
                                   'yolov5mu.pt',
                                   'yolov5lu.pt',
                                   'yolov5xu.pt',
                                   'yolov5n6u.pt',
                                   'yolov5s6u.pt',
                                   'yolov5m6u.pt',
                                   'yolov5l6u.pt',
                                   'yolov5x6u.pt',
                                   'yolov8n.pt',
                                   'yolov8s.pt',
                                   'yolov8m.pt',
                                   'yolov8l.pt',
                                   'yolov8x.pt',
                                   'yolov8n-oiv7.pt',
                                   'yolov8s-oiv7.pt',
                                   'yolov8m-oiv7.pt',
                                   'yolov8l-oiv7.pt',
                                   'yolov8x-oiv7.pt',
                                   'yolov9t.pt',
                                   'yolov9s.pt',
                                   'yolov9m.pt',
                                   'yolov9c.pt',
                                   'yolov9e.pt',
                                   'yolov10n.pt',
                                   'yolov10s.pt',
                                   'yolov10m.pt',
                                   'yolov10l.pt',
                                   'yolov10x.pt',
                                   'yolo11n.pt',
                                   'yolo11s.pt',
                                   'yolo11m.pt',
                                   'yolo11l.pt',
                                   'yolo11x.pt',
                                   'rtdetr-l.pt',
                                   'rtdetr-x.pt'])
