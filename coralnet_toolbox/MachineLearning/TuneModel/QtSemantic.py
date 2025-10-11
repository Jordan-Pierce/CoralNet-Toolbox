import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from PyQt5.QtWidgets import (QLineEdit, QHBoxLayout, QPushButton, QFormLayout, QGroupBox)

from coralnet_toolbox.MachineLearning.TuneModel.QtBase import Base


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Semantic(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Tune Semantic Segmentation Model")

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

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def load_model_combobox(self):
        """Load the model combobox with the available models."""
        self.model_combo.clear()
        self.model_combo.setEditable(True)

        standard_models = ['yolo11n-semseg.pt',
                           'yolo11s-semseg.pt',
                           'yolo11m-semseg.pt',
                           'yolo11l-semseg.pt',
                           'yolo11x-semseg.pt']

        self.model_combo.addItems(standard_models)
        # Set the default model
        self.model_combo.setCurrentIndex(standard_models.index('yolo11n-semseg.pt'))