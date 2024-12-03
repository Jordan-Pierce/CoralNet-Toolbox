import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from PyQt5.QtWidgets import (QLineEdit, QHBoxLayout, QPushButton, QFormLayout, QGroupBox)

from coralnet_toolbox.MachineLearning.TrainModel.QtBase import Base


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Classify(Base):
    def __init__(self, main_window, parent=None):
        self.task = "classify"
        
        super().__init__(main_window, parent)
        self.setWindowTitle("Train Classification Model")
        
    def setup_dataset_layout(self):
        """Setup the dataset layout."""
        
        group_box = QGroupBox("Dataset")
        layout = QFormLayout()

        # Dataset Directory
        self.dataset_edit = QLineEdit()
        self.dataset_button = QPushButton("Browse...")
        self.dataset_button.clicked.connect(self.browse_dataset_dir)

        dataset_dir_layout = QHBoxLayout()
        dataset_dir_layout.addWidget(self.dataset_edit)
        dataset_dir_layout.addWidget(self.dataset_button)
        layout.addRow("Dataset Directory:", dataset_dir_layout)

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