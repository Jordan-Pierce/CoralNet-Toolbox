from PyQt5.QtWidgets import (QFormLayout, QGroupBox, QHBoxLayout, QLineEdit, QPushButton,
                             QLabel, QVBoxLayout, QSpinBox, QDoubleSpinBox, QComboBox)

from coralnet_toolbox.MachineLearning.EvaluateModel.QtBase import Base


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Classify(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Evaluate Classification Model")
        
    def setup_info_layout(self):
        """Set up the layout and widgets for the info layout."""
        
        self.resize(400, 400)  # Decreased height for less parameter
        self.task = 'classify'
        self.imgsz = 256
        
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()
        
        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Evaluate a Classification model on a dataset, and the split you want to evaluate on.")
        
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
        self.model_edit.setToolTip("Path to a trained classification model (.pt file).")
        self.model_button.setToolTip("Browse for a model file.")
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(self.model_button)
        layout.addRow("Existing Model:", model_layout)

        self.dataset_edit = QLineEdit()
        self.dataset_button = QPushButton("Browse...")
        self.dataset_button.clicked.connect(self.browse_dataset_dir)
        self.dataset_edit.setToolTip("Path to classification dataset for evaluation.\nDirectory should contain class subdirectories with test images.")
        self.dataset_button.setToolTip("Browse for a dataset directory.")
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(self.dataset_edit)
        dataset_layout.addWidget(self.dataset_button)
        layout.addRow("Dataset Directory:", dataset_layout)
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
    def setup_parameters_layout(self):
        """Setup the parameters layout."""
        group_box = QGroupBox("Parameters")
        layout = QFormLayout()
        
        # Image size
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setMinimum(16)
        self.imgsz_spinbox.setMaximum(4096)
        self.imgsz_spinbox.setValue(self.imgsz)
        self.imgsz_spinbox.setToolTip("Input image size for evaluation.\nMust match the training image size (default: 256 for classification).")
        layout.addRow("Image Size:", self.imgsz_spinbox)

        # Batch size
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setMinimum(1)
        self.batch_spinbox.setMaximum(1024)
        self.batch_spinbox.setValue(16)
        self.batch_spinbox.setToolTip("Number of images per batch during evaluation.\nLarger batches are faster but use more GPU memory.")
        layout.addRow("Batch:", self.batch_spinbox)

        # Confidence threshold
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setMinimum(0.0)
        self.conf_spinbox.setMaximum(1.0)
        self.conf_spinbox.setSingleStep(0.001)
        self.conf_spinbox.setDecimals(3)
        self.conf_spinbox.setValue(0.001)
        self.conf_spinbox.setToolTip("Minimum confidence threshold (0.0 to 1.0).\nPredictions below this threshold are discarded.\nDefault: 0.001 (very permissive).")
        layout.addRow("Confidence:", self.conf_spinbox)

        # Augment
        self.augment_combo = QComboBox()
        self.augment_combo.addItems(["True", "False"])
        self.augment_combo.setCurrentText("False")
        self.augment_combo.setToolTip("Apply data augmentation during evaluation.\nTrue: helps with robustness assessment. False: standard evaluation.")
        layout.addRow("Augment:", self.augment_combo)

        # Workers
        self.workers_spinbox = QSpinBox()
        self.workers_spinbox.setMinimum(0)
        self.workers_spinbox.setMaximum(64)
        self.workers_spinbox.setValue(8)
        self.workers_spinbox.setToolTip("Number of parallel workers for data loading.\n0 = main thread. Higher = faster loading if CPU is available.")
        layout.addRow("Workers:", self.workers_spinbox)        
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)