import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
import datetime
import gc
import json
import os
import shutil

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog, QScrollArea, QMessageBox, QCheckBox, QWidget, QVBoxLayout,
    QLabel, QLineEdit, QDialog, QHBoxLayout, QPushButton, QComboBox, QSpinBox,
    QFormLayout, QTabWidget, QDoubleSpinBox
)

from torch.cuda import empty_cache
from ultralytics import YOLO

from toolbox.MachineLearning.WeightedDataset import WeightedInstanceDataset
from toolbox.MachineLearning.WeightedDataset import WeightedClassificationDataset
from toolbox.MachineLearning.QtEvaluateModel import EvaluateModelWorker

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class TrainModelWorker(QThread):
    """
    Worker thread for training machine learning models.
    
    This class handles the training process in a separate thread to keep the UI responsive.
    """
    training_started = pyqtSignal()
    training_completed = pyqtSignal()
    training_error = pyqtSignal(str)

    def __init__(self, params, device):
        """
        Initializes the TrainModelWorker with training parameters and device.
        
        :param params: Dictionary of training parameters.
        :param device: Device to run the training on (e.g., 'cpu' or 'cuda').
        """
        super().__init__()
        self.params = params
        self.device = device
        self.model = None

    def run(self):
        """
        Executes the training process.
        
        Emits signals to indicate the start and completion of training,
        and handles any errors that occur during training.
        """
        try:
            # Emit signal to indicate training has started
            self.training_started.emit()
            # Initialize and train the model using provided parameters
            self.model = YOLO(self.params['model'])
            self.model.train(**self.params)
            # Emit signal to indicate training has completed
            self.training_completed.emit()
        except Exception as e:
            # Emit signal with the error message if training fails
            self.training_error.emit(str(e))
        finally:
            # Clean up resources
            self._cleanup()

    def evaluate_model(self):
        """
        Evaluates the trained model.
        
        This method can be extended to perform model evaluation after training.
        """
        try:
            # Placeholder for model evaluation logic
            pass
        except Exception as e:
            # Handle any errors during evaluation
            self.training_error.emit(str(e))

    def on_evaluation_started(self):
        """
        Slot to handle actions when evaluation starts.
        """
        pass

    def on_evaluation_completed(self):
        """
        Slot to handle actions when evaluation completes.
        """
        pass

    def on_evaluation_error(self, error_message):
        """
        Handles errors that occur during model evaluation.
        
        :param error_message: The error message to emit.
        """
        self.training_error.emit(error_message)

    def _cleanup(self):
        """
        Cleans up the model and releases GPU memory.
        """
        del self.model
        gc.collect()
        empty_cache()


class TrainModelDialog(QDialog):
    """
    Dialog window for configuring and initiating model training.
    
    Provides a user interface to set training parameters, select datasets,
    and manage the training process.
    """
    def __init__(self, main_window, parent=None):
        """
        Initializes the TrainModelDialog with UI components and default settings.
        
        :param main_window: Reference to the main application window.
        :param parent: Parent widget, default is None.
        """
        super().__init__(parent)
        self.main_window = main_window

        # Parameters for training
        self.params = {}
        self.custom_params = []
        # Path to the best model weights
        self.model_path = None
        # Mapping of class labels
        self.class_mapping = {}

        self.setWindowTitle("Train Model")

        # Set window flags for better UI control
        self.setWindowFlags(Qt.Window |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint |
                            Qt.WindowTitleHint)

        # Set initial size of the dialog
        self.resize(600, 800)

        # Main layout for the dialog
        self.main_layout = QVBoxLayout()

        # Setup UI components like tabs, forms, and output console
        self.setup_ui()

        # Wrap the main layout in a scrollable area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_widget.setLayout(self.main_layout)
        scroll_area.setWidget(scroll_widget)

        # Set the scroll area as the main layout of the dialog
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scroll_area)

    def setup_ui(self):
        """
        Sets up the user interface components including informational labels,
        tabs for different tasks, parameter forms, and action buttons.
        """
        # Informational label with a hyperlink to documentation
        info_label = QLabel("Details on different hyperparameters can be found "
                            "<a href='https://docs.ultralytics.com/modes/train/#train-settings'>here</a>.")
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        self.main_layout.addWidget(info_label)

        # Create tabs for different machine learning tasks
        self.tabs = QTabWidget()
        self.tab_classification = QWidget()
        self.tab_detection = QWidget()
        self.tab_segmentation = QWidget()

        self.tabs.addTab(self.tab_classification, "Image Classification")
        self.tabs.addTab(self.tab_detection, "Object Detection")
        self.tabs.addTab(self.tab_segmentation, "Instance Segmentation")

        # Setup each tab with its specific UI components
        self.setup_classification_tab()
        self.setup_detection_tab()
        self.setup_segmentation_tab()

        self.main_layout.addWidget(self.tabs)

        # Form layout for training parameters
        self.form_layout = QFormLayout()

        # Project Directory Selection
        self.project_edit = QLineEdit()
        self.project_button = QPushButton("Browse...")
        self.project_button.clicked.connect(self.browse_project_dir)
        project_layout = QHBoxLayout()
        project_layout.addWidget(self.project_edit)
        project_layout.addWidget(self.project_button)
        self.form_layout.addRow("Project:", project_layout)

        # Project Name Input
        self.name_edit = QLineEdit()
        self.form_layout.addRow("Name:", self.name_edit)

        # Existing Model File Selection
        self.model_edit = QLineEdit()
        self.model_button = QPushButton("Browse...")
        self.model_button.clicked.connect(self.browse_model_file)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(self.model_button)
        self.form_layout.addRow("Existing Model:", model_layout)

        # Number of Epochs SpinBox
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setMinimum(1)
        self.epochs_spinbox.setMaximum(1000)
        self.epochs_spinbox.setValue(100)
        self.form_layout.addRow("Epochs:", self.epochs_spinbox)

        # Early Stopping Patience SpinBox
        self.patience_spinbox = QSpinBox()
        self.patience_spinbox.setMinimum(1)
        self.patience_spinbox.setMaximum(1000)
        self.patience_spinbox.setValue(30)
        self.form_layout.addRow("Patience:", self.patience_spinbox)

        # Image Size SpinBox
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setMinimum(16)
        self.imgsz_spinbox.setMaximum(4096)
        self.imgsz_spinbox.setValue(256)
        self.form_layout.addRow("Image Size:", self.imgsz_spinbox)

        # Batch Size SpinBox
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setMinimum(1)
        self.batch_spinbox.setMaximum(1024)
        self.batch_spinbox.setValue(512)
        self.form_layout.addRow("Batch Size:", self.batch_spinbox)

        # Number of Workers SpinBox
        self.workers_spinbox = QSpinBox()
        self.workers_spinbox.setMinimum(1)
        self.workers_spinbox.setMaximum(64)
        self.workers_spinbox.setValue(8)
        self.form_layout.addRow("Workers:", self.workers_spinbox)

        # Save CheckBox
        self.save_checkbox = QCheckBox()
        self.save_checkbox.setChecked(True)
        self.form_layout.addRow("Save:", self.save_checkbox)

        # Save Period SpinBox
        self.save_period_spinbox = QSpinBox()
        self.save_period_spinbox.setMinimum(-1)
        self.save_period_spinbox.setMaximum(1000)
        self.save_period_spinbox.setValue(-1)
        self.form_layout.addRow("Save Period:", self.save_period_spinbox)

        # Pretrained CheckBox
        self.pretrained_checkbox = QCheckBox()
        self.pretrained_checkbox.setChecked(True)
        self.form_layout.addRow("Pretrained:", self.pretrained_checkbox)

        # Freeze Layers Input
        self.freeze_edit = QLineEdit()
        self.form_layout.addRow("Freeze Layers:", self.freeze_edit)

        # Weighted Dataset CheckBox
        self.weighted_checkbox = QCheckBox()
        self.weighted_checkbox.setChecked(False)
        self.form_layout.addRow("Weighted:", self.weighted_checkbox)

        # Dropout SpinBox
        self.dropout_spinbox = QDoubleSpinBox()
        self.dropout_spinbox.setMinimum(0.0)
        self.dropout_spinbox.setMaximum(1.0)
        self.dropout_spinbox.setValue(0.0)
        self.form_layout.addRow("Dropout:", self.dropout_spinbox)

        # Optimizer Selection ComboBox
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"])
        self.optimizer_combo.setCurrentText("auto")
        self.form_layout.addRow("Optimizer:", self.optimizer_combo)

        # Learning Rate SpinBox
        self.lr0_spinbox = QDoubleSpinBox()
        self.lr0_spinbox.setMinimum(0.0001)
        self.lr0_spinbox.setMaximum(1.0000)
        self.lr0_spinbox.setSingleStep(0.0001)
        self.lr0_spinbox.setValue(0.0100)
        self.form_layout.addRow("Learning Rate (lr0):", self.lr0_spinbox)

        # Validation CheckBox
        self.val_checkbox = QCheckBox()
        self.val_checkbox.setChecked(True)
        self.form_layout.addRow("Validation:", self.val_checkbox)

        # Fraction SpinBox
        self.fraction_spinbox = QDoubleSpinBox()
        self.fraction_spinbox.setMinimum(0.1)
        self.fraction_spinbox.setMaximum(1.0)
        self.fraction_spinbox.setValue(1.0)
        self.form_layout.addRow("Fraction:", self.fraction_spinbox)

        # Verbose CheckBox
        self.verbose_checkbox = QCheckBox()
        self.verbose_checkbox.setChecked(True)
        self.form_layout.addRow("Verbose:", self.verbose_checkbox)

        # Section for adding custom parameters
        self.custom_params_layout = QVBoxLayout()
        self.form_layout.addRow("Additional Parameters:", self.custom_params_layout)

        # Button to add new custom parameter pairs
        self.add_param_button = QPushButton("Add Parameter")
        self.add_param_button.clicked.connect(self.add_parameter_pair)
        self.form_layout.addRow("", self.add_param_button)

        self.main_layout.addLayout(self.form_layout)

        # OK and Cancel buttons to confirm or cancel training
        self.buttons = QPushButton("OK")
        self.buttons.clicked.connect(self.accept)
        self.main_layout.addWidget(self.buttons)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.main_layout.addWidget(self.cancel_button)

    def add_parameter_pair(self):
        """
        Adds a new pair of input fields for custom parameter name and value.
        """
        param_layout = QHBoxLayout()
        param_name = QLineEdit()
        param_value = QLineEdit()
        param_layout.addWidget(param_name)
        param_layout.addWidget(param_value)

        self.custom_params.append((param_name, param_value))
        self.custom_params_layout.addLayout(param_layout)

    def browse_dataset_dir(self):
        """
        Opens a dialog to select a dataset directory and sets the selected path.
        """
        dir_path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if dir_path:
            self.classify_dataset_edit.setText(dir_path)
            # Additional logic to handle the selected directory can be added here

    def browse_dataset_yaml(self):
        """
        Opens a dialog to select a dataset YAML file and sets the selected path.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset YAML File",
            "",
            "YAML Files (*.yaml *.yml)"
        )
        if file_path:
            # Logic to handle the selected YAML file can be added here
            pass

    def browse_class_mapping_file(self):
        """
        Opens a dialog to select a class mapping JSON file and sets the selected path.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Class Mapping File",
            "",
            "JSON Files (*.json)"
        )
        if file_path:
            # Logic to handle the selected class mapping file can be added here
            pass

    def browse_project_dir(self):
        """
        Opens a dialog to select a project directory and sets the selected path.
        """
        dir_path = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if dir_path:
            self.project_edit.setText(dir_path)
            # Additional logic for project directory can be added here

    def browse_model_file(self):
        """
        Opens a dialog to select a model file and sets the selected path.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File")
        if file_path:
            self.model_edit.setText(file_path)
            # Additional logic for model file can be added here

    def setup_classification_tab(self):
        """
        Sets up the UI components for the Image Classification tab.
        """
        layout = QVBoxLayout()

        # Dataset Directory Selection for Classification
        self.classify_dataset_edit = QLineEdit()
        self.classify_dataset_button = QPushButton("Browse...")
        self.classify_dataset_button.clicked.connect(self.browse_dataset_dir)
        dataset_dir_layout = QHBoxLayout()
        dataset_dir_layout.addWidget(QLabel("Dataset Directory:"))
        dataset_dir_layout.addWidget(self.classify_dataset_edit)
        dataset_dir_layout.addWidget(self.classify_dataset_button)
        layout.addLayout(dataset_dir_layout)

        # Class Mapping File Selection for Classification
        self.classify_mapping_edit = QLineEdit()
        self.classify_mapping_button = QPushButton("Browse...")
        self.classify_mapping_button.clicked.connect(self.browse_class_mapping_file)
        class_mapping_layout = QHBoxLayout()
        class_mapping_layout.addWidget(QLabel("Class Mapping:"))
        class_mapping_layout.addWidget(self.classify_mapping_edit)
        class_mapping_layout.addWidget(self.classify_mapping_button)
        layout.addLayout(class_mapping_layout)

        # Classification Model Selection Dropdown
        self.classification_model_combo = QComboBox()
        self.classification_model_combo.addItems([
            "yolov8n-cls.pt",
            "yolov8s-cls.pt",
            "yolov8m-cls.pt",
            "yolov8l-cls.pt",
            "yolov8x-cls.pt"
        ])
        self.classification_model_combo.setEditable(True)
        layout.addWidget(QLabel("Select or Enter Classification Model:"))
        layout.addWidget(self.classification_model_combo)

        self.tab_classification.setLayout(layout)

    def setup_detection_tab(self):
        """
        Sets up the UI components for the Object Detection tab.
        """
        layout = QVBoxLayout()

        # Dataset YAML File Selection for Detection
        self.detection_dataset_edit = QLineEdit()
        self.detection_dataset_button = QPushButton("Browse...")
        self.detection_dataset_button.clicked.connect(self.browse_dataset_yaml)
        dataset_yaml_layout = QHBoxLayout()
        dataset_yaml_layout.addWidget(QLabel("Dataset YAML:"))
        dataset_yaml_layout.addWidget(self.detection_dataset_edit)
        dataset_yaml_layout.addWidget(self.detection_dataset_button)
        layout.addLayout(dataset_yaml_layout)

        # Class Mapping File Selection for Detection
        self.detection_mapping_edit = QLineEdit()
        self.detection_mapping_button = QPushButton("Browse...")
        self.detection_mapping_button.clicked.connect(self.browse_class_mapping_file)
        class_mapping_layout = QHBoxLayout()
        class_mapping_layout.addWidget(QLabel("Class Mapping:"))
        class_mapping_layout.addWidget(self.detection_mapping_edit)
        class_mapping_layout.addWidget(self.detection_mapping_button)
        layout.addLayout(class_mapping_layout)

        # Detection Model Selection Dropdown
        self.detection_model_combo = QComboBox()
        self.detection_model_combo.addItems([
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8m.pt",
            "yolov8l.pt",
            "yolov8x.pt"
        ])
        self.detection_model_combo.setEditable(True)
        layout.addWidget(QLabel("Select or Enter Detection Model:"))
        layout.addWidget(self.detection_model_combo)

        self.tab_detection.setLayout(layout)

    def setup_segmentation_tab(self):
        """
        Sets up the UI components for the Instance Segmentation tab.
        """
        layout = QVBoxLayout()

        # Dataset YAML File Selection for Segmentation
        self.segmentation_dataset_edit = QLineEdit()
        self.segmentation_dataset_button = QPushButton("Browse...")
        self.segmentation_dataset_button.clicked.connect(self.browse_dataset_yaml)
        dataset_yaml_layout = QHBoxLayout()
        dataset_yaml_layout.addWidget(QLabel("Dataset YAML:"))
        dataset_yaml_layout.addWidget(self.segmentation_dataset_edit)
        dataset_yaml_layout.addWidget(self.segmentation_dataset_button)
        layout.addLayout(dataset_yaml_layout)

        # Class Mapping File Selection for Segmentation
        self.segmentation_mapping_edit = QLineEdit()
        self.segmentation_mapping_button = QPushButton("Browse...")
        self.segmentation_mapping_button.clicked.connect(self.browse_class_mapping_file)
        class_mapping_layout = QHBoxLayout()
        class_mapping_layout.addWidget(QLabel("Class Mapping:"))
        class_mapping_layout.addWidget(self.segmentation_mapping_edit)
        class_mapping_layout.addWidget(self.segmentation_mapping_button)
        layout.addLayout(class_mapping_layout)

        # Segmentation Model Selection Dropdown
        self.segmentation_model_combo = QComboBox()
        self.segmentation_model_combo.addItems([
            "yolov8n-seg.pt",
            "yolov8s-seg.pt",
            "yolov8m-seg.pt",
            "yolov8l-seg.pt",
            "yolov8x-seg.pt"
        ])
        self.segmentation_model_combo.setEditable(True)
        layout.addWidget(QLabel("Select or Enter Segmentation Model:"))
        layout.addWidget(self.segmentation_model_combo)

        self.tab_segmentation.setLayout(layout)

    def accept(self):
        """
        Overrides the accept method to initiate model training before closing the dialog.
        """
        self.train_model()
        super().accept()

    def get_parameters(self):
        """
        Gathers and constructs the training parameters from the dialog inputs.
        
        :return: Dictionary containing all training parameters.
        """
        # Determine the selected task based on the current tab
        selected_tab = self.tabs.currentWidget()
        if selected_tab == self.tab_classification:
            task = "classification"
            model = self.classification_model_combo.currentText()
            data = self.classify_dataset_edit.text()
        elif selected_tab == self.tab_detection:
            task = "detection"
            model = self.detection_model_combo.currentText()
            data = self.detection_dataset_edit.text()
        elif selected_tab == self.tab_segmentation:
            task = "segmentation"
            model = self.segmentation_model_combo.currentText()
            data = self.segmentation_dataset_edit.text()
        else:
            task = "unknown"
            model = ""
            data = ""

        # Extract values from dialog widgets
        params = {
            'task': task,
            'project': self.project_edit.text(),
            'name': self.name_edit.text(),
            'model': model,
            'data': data,
            'epochs': self.epochs_spinbox.value(),
            'patience': self.patience_spinbox.value(),
            'batch': self.batch_spinbox.value(),
            'imgsz': self.imgsz_spinbox.value(),
            'save': self.save_checkbox.isChecked(),
            'save_period': self.save_period_spinbox.value(),
            'workers': self.workers_spinbox.value(),
            'pretrained': self.pretrained_checkbox.isChecked(),
            'optimizer': self.optimizer_combo.currentText(),
            'verbose': self.verbose_checkbox.isChecked(),
            'fraction': self.fraction_spinbox.value(),
            'freeze': self.freeze_edit.text(),
            'lr0': self.lr0_spinbox.value(),
            'weighted': self.weighted_checkbox.isChecked(),
            'dropout': self.dropout_spinbox.value(),
            'val': self.val_checkbox.isChecked(),
            'exist_ok': True,
            'plots': True,
        }

        # Set default project folder if not specified
        project = 'Data/Training'
        params['project'] = params['project'] if params['project'] else project

        # Set default project name based on current timestamp if not specified
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d_%H-%M-%S")
        params['name'] = params['name'] if params['name'] else now

        # Add custom parameters to allow overriding of default parameters
        for param_name, param_value in self.custom_params:
            if param_name.text() and param_value.text():
                params[param_name.text()] = param_value.text()

        # Return the dictionary of parameters
        return params

    def train_model(self):
        """
        Initiates the training process by gathering parameters and starting the worker thread.
        """
        # Get training parameters from the dialog
        self.params = self.get_parameters()
        # Create and start the worker thread for training
        self.worker = TrainModelWorker(self.params, self.main_window.device)
        self.worker.training_started.connect(self.on_training_started)
        self.worker.training_completed.connect(self.on_training_completed)
        self.worker.training_error.connect(self.on_training_error)
        self.worker.start()

    def on_training_started(self):
        """
        Slot to handle actions when training starts.
        
        Can be used to update UI elements like disabling buttons or showing progress indicators.
        """
        QMessageBox.information(self, "Training Started", "Model training has started.")

    def on_training_completed(self):
        """
        Slot to handle actions when training completes.
        
        Can be used to update UI elements like enabling buttons or notifying the user.
        """
        QMessageBox.information(self, "Training Completed", "Model training has completed successfully.")

    def on_training_error(self, error_message):
        """
        Handles errors that occur during the training process.
        
        :param error_message: The error message to display to the user.
        """
        QMessageBox.critical(self, "Training Error", f"An error occurred during training:\n{error_message}")