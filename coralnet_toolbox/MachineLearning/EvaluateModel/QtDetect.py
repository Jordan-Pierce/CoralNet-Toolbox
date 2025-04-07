from PyQt5.QtWidgets import (QFormLayout, QGroupBox, QHBoxLayout, QLineEdit, QPushButton,
                             QLabel, QVBoxLayout)

from coralnet_toolbox.MachineLearning.EvaluateModel.QtBase import Base


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Detect(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Evaluate Detection Model")
        self.task = 'detection'
        self.imgsz = 640
        
        
    def setup_info_layout(self):
        """Set up the layout and widgets for the info layout."""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()
        
        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Evaluate a Detection model on a dataset, and the split you want to evaluate on.")
        
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
    def setup_dataset_layout(self):
        """Setup the dataset layout."""
        group_box = QGroupBox("Dataset")
        layout = QFormLayout()
        
        self.model_edit = QLineEdit()
        self.model_button = QPushButton("Browse...")
        self.model_button.clicked.connect(self.browse_model_file)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(self.model_button)
        layout.addRow("Existing Model:", model_layout)
            
        self.dataset_edit = QLineEdit()
        self.dataset_button = QPushButton("Browse...")
        self.dataset_button.clicked.connect(self.browse_dataset_yaml)
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(self.dataset_edit)
        dataset_layout.addWidget(self.dataset_button)
        layout.addRow("Dataset YAML:", dataset_layout)
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)