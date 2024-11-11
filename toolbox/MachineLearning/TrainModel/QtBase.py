import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import datetime
import gc
import json
import os
from pathlib import Path

import ultralytics.data.build as build
import ultralytics.models.yolo.classify.train as train_build

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QFileDialog, QScrollArea, QMessageBox, QCheckBox, QWidget, QVBoxLayout,
                             QLabel, QLineEdit, QDialog, QHBoxLayout, QPushButton, QComboBox, QSpinBox,
                             QFormLayout, QTabWidget, QDoubleSpinBox)

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
    Worker thread for training a model.

    Signals:
        training_started: Emitted when the training starts.
        training_completed: Emitted when the training completes.
        training_error: Emitted when an error occurs during training.
    """
    training_started = pyqtSignal()
    training_completed = pyqtSignal()
    training_error = pyqtSignal(str)

    def __init__(self, params, device):
        """
        Initialize the TrainModelWorker.

        Args:
            params: A dictionary of parameters for training.
            device: The device to use for training (e.g., 'cpu' or 'cuda').
        """
        super().__init__()
        self.params = params
        self.device = device
        self.model = None

    def run(self):
        """
        Run the training process in a separate thread.
        """
        try:
            # Emit signal to indicate training has started
            self.training_started.emit()

            # Extract parameters
            model_path = self.params.pop('model', None)
            weighted = self.params.pop('weighted', False)

            # Use the custom dataset class for weighted sampling
            if weighted and self.params['task'] == 'classify':
                train_build.ClassificationDataset = WeightedClassificationDataset
            elif weighted and self.params['task'] in ['detect', 'segment']:
                build.YOLODataset = WeightedInstanceDataset

            # Load the model, train, and save the best weights
            self.model = YOLO(model_path)
            self.model.train(**self.params, device=self.device)
            # Evaluate the model after training
            self.evaluate_model()
            # Emit signal to indicate training has completed
            self.training_completed.emit()

        except Exception as e:
            self.training_error.emit(str(e))
        finally:
            self._cleanup()

    def evaluate_model(self):
        """
        Evaluate the model after training.
        """
        try:
            # Create an instance of EvaluateModelWorker and start it
            eval_params = {
                'data': self.params['data'],
                'imgsz': self.params['imgsz'],
                'split': 'test',  # Evaluate on the test set only
                'save_dir': Path(self.params['project']) / self.params['name'] / 'test'
            }

            # Create and start the worker thread
            eval_worker = EvaluateModelWorker(model=self.model,
                                              params=eval_params)

            eval_worker.evaluation_error.connect(self.on_evaluation_error)
            eval_worker.run()  # Run the evaluation synchronously (same thread)
        except Exception as e:
            self.training_error.emit(str(e))

    def on_evaluation_started(self):
        """
        Handle the event when the evaluation starts.
        """
        pass

    def on_evaluation_completed(self):
        """
        Handle the event when the evaluation completes.
        """
        pass

    def on_evaluation_error(self, error_message):
        """
        Handle the event when an error occurs during evaluation.

        Args:
            error_message (str): The error message.
        """
        self.training_error.emit(error_message)

    def _cleanup(self):
        """
        Clean up resources after training.
        """
        del self.model
        gc.collect()
        empty_cache()


class Base(QDialog):
    """
    Dialog for training machine learning models for image classification, object detection, 
    and instance segmentation.

    :param main_window: MainWindow object
    :param parent: Parent widget
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        # For holding parameters
        self.params = {}
        self.custom_params = []
        # Best model weights
        self.model_path = None
        # Class mapping
        self.class_mapping = {}

        self.setWindowTitle("Train Model")

        # Set window settings
        self.setWindowFlags(Qt.Window |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint |
                            Qt.WindowTitleHint)

        self.resize(600, 800)

        # Main layout
        self.main_layout = QVBoxLayout()

        # Create and set up the generic layout
        self.setup_generic_layout()

        # Wrap the main layout in a QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_widget.setLayout(self.main_layout)
        scroll_area.setWidget(scroll_widget)

        # Set the scroll area as the main layout of the dialog
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scroll_area)

    def setup_generic_layout(self):
        """
        Set up the layout and widgets for the generic layout.
        """
        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Details on different hyperparameters can be found "
                            "<a href='https://docs.ultralytics.com/modes/train/#train-settings'>here</a>.")
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        self.main_layout.addWidget(info_label)

        layout = QVBoxLayout()

        # Dataset Directory
        self.dataset_edit = QLineEdit()
        self.dataset_button = QPushButton("Browse...")
        self.dataset_button.clicked.connect(self.browse_dataset_dir)

        dataset_dir_layout = QHBoxLayout()
        dataset_dir_layout.addWidget(QLabel("Dataset Directory:"))
        dataset_dir_layout.addWidget(self.dataset_edit)
        dataset_dir_layout.addWidget(self.dataset_button)
        layout.addLayout(dataset_dir_layout)

        # Class Mapping
        self.mapping_edit = QLineEdit()
        self.mapping_button = QPushButton("Browse...")
        self.mapping_button.clicked.connect(self.browse_class_mapping_file)

        class_mapping_layout = QHBoxLayout()
        class_mapping_layout.addWidget(QLabel("Class Mapping:"))
        class_mapping_layout.addWidget(self.mapping_edit)
        class_mapping_layout.addWidget(self.mapping_button)
        layout.addLayout(class_mapping_layout)

        # Parameters Form
        self.form_layout = QFormLayout()

        # Project
        self.project_edit = QLineEdit()
        self.project_button = QPushButton("Browse...")
        self.project_button.clicked.connect(self.browse_project_dir)
        project_layout = QHBoxLayout()
        project_layout.addWidget(self.project_edit)
        project_layout.addWidget(self.project_button)
        self.form_layout.addRow("Project:", project_layout)

        # Name
        self.name_edit = QLineEdit()
        self.form_layout.addRow("Name:", self.name_edit)

        # Existing Model
        self.model_edit = QLineEdit()
        self.model_button = QPushButton("Browse...")
        self.model_button.clicked.connect(self.browse_model_file)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(self.model_button)
        self.form_layout.addRow("Existing Model:", model_layout)

        # Epochs
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setMinimum(1)
        self.epochs_spinbox.setMaximum(1000)
        self.epochs_spinbox.setValue(100)
        self.form_layout.addRow("Epochs:", self.epochs_spinbox)

        # Patience
        self.patience_spinbox = QSpinBox()
        self.patience_spinbox.setMinimum(1)
        self.patience_spinbox.setMaximum(1000)
        self.patience_spinbox.setValue(30)
        self.form_layout.addRow("Patience:", self.patience_spinbox)

        # Imgsz
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setMinimum(16)
        self.imgsz_spinbox.setMaximum(4096)
        self.imgsz_spinbox.setValue(256)
        self.form_layout.addRow("Image Size:", self.imgsz_spinbox)

        # Batch
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setMinimum(1)
        self.batch_spinbox.setMaximum(1024)
        self.batch_spinbox.setValue(512)
        self.form_layout.addRow("Batch Size:", self.batch_spinbox)

        # Workers
        self.workers_spinbox = QSpinBox()
        self.workers_spinbox.setMinimum(1)
        self.workers_spinbox.setMaximum(64)
        self.workers_spinbox.setValue(8)
        self.form_layout.addRow("Workers:", self.workers_spinbox)

        # Save
        self.save_checkbox = QCheckBox()
        self.save_checkbox.setChecked(True)
        self.form_layout.addRow("Save:", self.save_checkbox)

        # Save Period
        self.save_period_spinbox = QSpinBox()
        self.save_period_spinbox.setMinimum(-1)
        self.save_period_spinbox.setMaximum(1000)
        self.save_period_spinbox.setValue(-1)
        self.form_layout.addRow("Save Period:", self.save_period_spinbox)

        # Pretrained
        self.pretrained_checkbox = QCheckBox()
        self.pretrained_checkbox.setChecked(True)
        self.form_layout.addRow("Pretrained:", self.pretrained_checkbox)

        # Freeze
        self.freeze_edit = QLineEdit()
        self.form_layout.addRow("Freeze Layers:", self.freeze_edit)

        # Weighted Dataset
        self.weighted_checkbox = QCheckBox()
        self.weighted_checkbox.setChecked(False)
        self.form_layout.addRow("Weighted:", self.weighted_checkbox)

        # Dropout
        self.dropout_spinbox = QDoubleSpinBox()
        self.dropout_spinbox.setMinimum(0.0)
        self.dropout_spinbox.setMaximum(1.0)
        self.dropout_spinbox.setValue(0.0)
        self.form_layout.addRow("Dropout:", self.dropout_spinbox)

        # Optimizer
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"])
        self.optimizer_combo.setCurrentText("auto")
        self.form_layout.addRow("Optimizer:", self.optimizer_combo)

        # Lr0
        self.lr0_spinbox = QDoubleSpinBox()
        self.lr0_spinbox.setMinimum(0.0001)
        self.lr0_spinbox.setMaximum(1.0000)
        self.lr0_spinbox.setSingleStep(0.0001)
        self.lr0_spinbox.setValue(0.0100)
        self.form_layout.addRow("Learning Rate (lr0):", self.lr0_spinbox)

        # Val
        self.val_checkbox = QCheckBox()
        self.val_checkbox.setChecked(True)
        self.form_layout.addRow("Validation:", self.val_checkbox)

        # Fraction
        self.fraction_spinbox = QDoubleSpinBox()
        self.fraction_spinbox.setMinimum(0.1)
        self.fraction_spinbox.setMaximum(1.0)
        self.fraction_spinbox.setValue(1.0)
        self.form_layout.addRow("Fraction:", self.fraction_spinbox)

        # Verbose
        self.verbose_checkbox = QCheckBox()
        self.verbose_checkbox.setChecked(True)
        self.form_layout.addRow("Verbose:", self.verbose_checkbox)

        # Add custom parameters section
        self.custom_params_layout = QVBoxLayout()
        self.form_layout.addRow("Additional Parameters:", self.custom_params_layout)

        # Add button for new parameter pairs
        self.add_param_button = QPushButton("Add Parameter")
        self.add_param_button.clicked.connect(self.add_parameter_pair)
        self.form_layout.addRow("", self.add_param_button)

        self.main_layout.addLayout(self.form_layout)

        # Add OK and Cancel buttons
        self.buttons = QPushButton("OK")
        self.buttons.clicked.connect(self.accept)
        self.main_layout.addWidget(self.buttons)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.main_layout.addWidget(self.cancel_button)

    def add_parameter_pair(self):
        """
        Add a new pair of parameter name and value input fields.
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
        Browse and select a dataset directory.
        """
        dir_path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if dir_path:
            # Load the class mapping if it exists
            class_mapping_path = f"{dir_path}/class_mapping.json"
            if os.path.exists(class_mapping_path):
                self.class_mapping = json.load(open(class_mapping_path, 'r'))
                self.mapping_edit.setText(class_mapping_path)

            # Set the dataset path
            self.dataset_edit.setText(dir_path)

    def browse_dataset_yaml(self):
        """
        Browse and select a dataset YAML file.
        """
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Select Dataset YAML File",
                                                   "",
                                                   "YAML Files (*.yaml *.yml)")
        if file_path:
            # Load the class mapping if it exists
            dir_path = os.path.dirname(file_path)
            class_mapping_path = f"{dir_path}/class_mapping.json"
            if os.path.exists(class_mapping_path):
                self.class_mapping = json.load(open(class_mapping_path, 'r'))

            # Set the dataset and class mapping paths
            self.dataset_edit.setText(file_path)
            self.mapping_edit.setText(class_mapping_path)


    def browse_class_mapping_file(self):
        """
        Browse and select a class mapping file.
        """
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Select Class Mapping File",
                                                   "",
                                                   "JSON Files (*.json)")
        if file_path:
            # Load the class mapping
            self.class_mapping = json.load(open(file_path, 'r'))
            if self.tabs.currentWidget() == self.tab_classification:
                self.classify_mapping_edit.setText(file_path)
            elif self.tabs.currentWidget() == self.tab_detection:
                self.detection_mapping_edit.setText(file_path)
            elif self.tabs.currentWidget() == self.tab_segmentation:
                self.segmentation_mapping_edit.setText(file_path)

    def browse_project_dir(self):
        """
        Browse and select a project directory.
        """
        dir_path = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if dir_path:
            self.project_edit.setText(dir_path)

    def browse_model_file(self):
        """
        Browse and select a model file.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File")
        if file_path:
            self.model_edit.setText(file_path)

    def accept(self):
        """
        Handle the OK button click event.
        """
        self.train_model()
        super().accept()

    def get_parameters(self):
        """
        Get the training parameters from the dialog widgets.

        Returns:
            dict: A dictionary of training parameters.
        """
        # Determine the selected tab
        selected_tab = self.tabs.currentWidget()
        if selected_tab == self.tab_classification:
            task = 'classify'
            data = self.classify_dataset_edit.text()
            model = self.classification_model_combo.currentText()
        elif selected_tab == self.tab_detection:
            task = 'detect'
            data = self.detection_dataset_edit.text()
            model = self.detection_model_combo.currentText()
        elif selected_tab == self.tab_segmentation:
            task = 'segment'
            data = self.segmentation_dataset_edit.text()
            model = self.segmentation_model_combo.currentText()
        else:
            raise ValueError("Invalid tab selected.")

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
        # Default project folder
        project = 'Data/Training'
        params['project'] = params['project'] if params['project'] else project
        # Default project name
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d_%H-%M-%S")
        params['name'] = params['name'] if params['name'] else now

        # Add custom parameters (allows overriding the above parameters)
        for param_name, param_value in self.custom_params:
            name = param_name.text().strip()
            value = param_value.text().strip().lower()
            if name:
                if value == 'true':
                    params[name] = True
                elif value == 'false':
                    params[name] = False
                else:
                    try:
                        params[name] = int(value)
                    except ValueError:
                        try:
                            params[name] = float(value)
                        except ValueError:
                            params[name] = value

        # Return the dictionary of parameters
        return params

    def train_model(self):
        """
        Train the model based on the provided parameters.
        """
        # Get training parameters
        self.params = self.get_parameters()
        # Create and start the worker thread
        self.worker = TrainModelWorker(self.params, self.main_window.device)
        self.worker.training_started.connect(self.on_training_started)
        self.worker.training_completed.connect(self.on_training_completed)
        self.worker.training_error.connect(self.on_training_error)
        self.worker.start()

    def on_training_started(self):
        """
        Handle the event when the training starts.
        """
        # Save the class mapping JSON file
        output_dir_path = os.path.join(self.params['project'], self.params['name'])
        os.makedirs(output_dir_path, exist_ok=True)
        if self.class_mapping:
            # Write the json file to the output directory
            with open(f"{output_dir_path}/class_mapping.json", 'w') as json_file:
                json.dump(self.class_mapping, json_file, indent=4)

        message = "Model training has commenced.\nMonitor the console for real-time progress."
        QMessageBox.information(self, "Model Training Status", message)

    def on_training_completed(self):
        """
        Handle the event when the training completes.
        """
        message = "Model training has successfully been completed."
        QMessageBox.information(self, "Model Training Status", message)

    def on_training_error(self, error_message):
        """
        Handle the event when an error occurs during training.

        Args:
            error_message (str): The error message.
        """
        QMessageBox.critical(self, "Error", error_message)
        print(error_message)
