from PyQt5.QtWidgets import QFormLayout, QGroupBox, QHBoxLayout, QLineEdit, QPushButton

from coralnet_toolbox.MachineLearning.EvaluateModel.QtBase import Base


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Detect(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Evaluate Detection Model")
        self.task = 'detection'
        
    def setup_dataset_layout(self):
        """Setup the dataset layout."""
        group_box = QGroupBox("Dataset")
        layout = QFormLayout()
            
        self.dataset_edit = QLineEdit()
        self.dataset_button = QPushButton("Browse...")
        self.dataset_button.clicked.connect(self.browse_dataset_yaml)
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(self.dataset_edit)
        dataset_layout.addWidget(self.dataset_button)
        layout.addRow("Dataset YAML:", dataset_layout)
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)