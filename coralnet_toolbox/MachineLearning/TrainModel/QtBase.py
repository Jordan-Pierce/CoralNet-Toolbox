import warnings

import os
import gc
import datetime
import traceback
import ujson as json
import yaml
from pathlib import Path

from ultralytics import YOLO
import ultralytics.data.build as detection_build
from ultralytics.data.dataset import YOLODataset
import ultralytics.models.yolo.classify.train as train_build
from ultralytics.data.dataset import ClassificationDataset

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QFileDialog, QScrollArea, QMessageBox, QCheckBox, QWidget, QVBoxLayout,
                             QLabel, QLineEdit, QDialog, QHBoxLayout, QPushButton, QComboBox, QSpinBox,
                             QFormLayout, QTabWidget, QDoubleSpinBox, QGroupBox, QFrame)

from torch.cuda import empty_cache

from coralnet_toolbox.MachineLearning.Community.cfg import get_available_configs
from coralnet_toolbox.MachineLearning.WeightedDataset import WeightedInstanceDataset
from coralnet_toolbox.MachineLearning.WeightedDataset import WeightedClassificationDataset
from coralnet_toolbox.MachineLearning.EvaluateModel.QtBase import EvaluateModelWorker

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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
        self.model_path = None
        self.weighted = False

    def pre_run(self):
        """
        Set up the model and prepare parameters for training.
        """
        try:
            # Extract model path
            self.model_path = self.params.pop('model', None)
            # Get the weighted flag
            self.weighted = self.params.pop('weighted', False)

            # Determine if ultralytics or community
            if self.model_path in get_available_configs(task=self.params['task']):
                self.model_path = get_available_configs(task=self.params['task'])[self.model_path]
                # Cannot use weighted sampling with community models
                self.weighted = False

            # Use the custom dataset class for weighted sampling
            if self.weighted and self.params['task'] == 'classify':
                train_build.ClassificationDataset = WeightedClassificationDataset
            elif self.weighted and self.params['task'] in ['detect', 'segment']:
                detection_build.YOLODataset = WeightedInstanceDataset

            # Load the model (8.3.141) YOLO handles RTDETR
            self.model = YOLO(self.model_path)

            # Set the task in the model itself
            self.model.task = self.params['task']

            # Freeze layers, freeze encoder
            freeze_layers = self.params.pop('freeze_layers', None)

            if freeze_layers:
                # Calculate the number of layers to freeze
                num_layers = len(self.model.model.model[0:-1])
                num_layers = int(num_layers * freeze_layers)
                freeze_layers = [_ for _ in range(0, num_layers)]
                print(f"Encoder layers frozen ({len(freeze_layers)})")
            else:
                freeze_layers = []

            # Set the freeze parameter for ultralytics
            self.params['freeze'] = freeze_layers

        except Exception as e:
            print(f"Error during setup: {e}\n\nTraceback:\n{traceback.format_exc()}")
            self.training_error.emit(f"Error during setup: {e} (see console log)")
            raise

    def run(self):
        """
        Run the training process in a separate thread.
        """
        try:
            # Emit signal to indicate training has started
            self.training_started.emit()

            # Set up the model and parameters
            self.pre_run()

            # Train the model
            self.model.train(**self.params, device=self.device)

            # Post-run cleanup
            self.post_run()

            # Evaluate the model after training
            self.evaluate_model()

            # Emit signal to indicate training has completed
            self.training_completed.emit()

        except Exception as e:
            print(f"Error during training: {e}\n\nTraceback:\n{traceback.format_exc()}")
            self.training_error.emit(f"Error during training: {e} (see console log)")
        finally:
            self._cleanup()

    def post_run(self):
        """
        Clean up resources after training.
        """
        # Revert to the original dataset class without weighted sampling
        if self.weighted and self.params['task'] == 'classify':
            train_build.ClassificationDataset = ClassificationDataset
        elif self.weighted and self.params['task'] in ['detect', 'segment']:
            detection_build.YOLODataset = YOLODataset

    def evaluate_model(self):
        """
        Evaluate the model after training.
        """
        try:
            # Do not evaluate if the user specifies 
            if not self.params.get('Validation', False):
                return
            
            # Check that there is a test folder
            test_folder = f"{self.params['data']}/test"
            print(f"Note: Looking for test folder: {test_folder}")
            if not os.path.exists(test_folder):
                print("Warning: No test folder found in that location. Skipping evaluation.")
                return
            
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
            print(f"Error during evaluation: {e}\n\nTraceback:\n{traceback.format_exc()}")
            self.training_error.emit(f"Error during evaluation: {e} (see console log)")

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

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Train Model")
        self.resize(600, 750)  

        # Set window settings
        self.setWindowFlags(Qt.Window |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint |
                            Qt.WindowTitleHint)

        # Task
        self.task = None
        # For holding parameters
        self.params = {}
        self.custom_params = []
        # Best model weights
        self.model_path = None
        # Class mapping
        self.class_mapping = {}

        # Task specific parameters
        self.imgsz = 640
        self.batch = 4

        # Create the layout
        self.layout = QVBoxLayout(self)

        # Create the info layout
        self.setup_info_layout()
        # Create the dataset layout
        self.setup_dataset_layout()
        # Create the model layout (new)
        self.setup_model_layout()
        # Create the output layout
        self.setup_output_layout()
        # Create and set up the parameters layout
        self.setup_parameters_layout()
        # Create the buttons layout
        self.setup_buttons_layout()

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Details on different hyperparameters can be found "
                            "<a href='https://docs.ultralytics.com/modes/train/#train-settings'>here</a>.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_dataset_layout(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def setup_model_layout(self):
        """
        Set up the layout and widgets for the model selection with a tabbed interface.
        """
        group_box = QGroupBox("Model Selection")
        layout = QVBoxLayout()

        # Create tabbed widget
        tab_widget = QTabWidget()

        # Tab 1: Select model from dropdown
        model_select_tab = QWidget()
        model_select_layout = QFormLayout(model_select_tab)

        # Model combo box
        self.model_combo = QComboBox()
        self.load_model_combobox()
        model_select_layout.addRow("Model:", self.model_combo)

        tab_widget.addTab(model_select_tab, "Select Model")

        # Tab 2: Use existing model
        model_existing_tab = QWidget()
        model_existing_layout = QFormLayout(model_existing_tab)

        # Existing Model
        self.model_edit = QLineEdit()
        self.model_button = QPushButton("Browse...")
        self.model_button.clicked.connect(self.browse_model_file)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(self.model_button)
        model_existing_layout.addRow("Existing Model:", model_layout)

        tab_widget.addTab(model_existing_tab, "Use Existing Model")

        layout.addWidget(tab_widget)
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
    def setup_output_layout(self):
        """
        Set up the layout and widgets for the output directory.
        """
        group_box = QGroupBox("Output Parameters")
        form_layout = QFormLayout()

        # Project
        self.project_edit = QLineEdit()
        self.project_button = QPushButton("Browse...")
        self.project_button.clicked.connect(self.browse_project_dir)
        project_layout = QHBoxLayout()
        project_layout.addWidget(self.project_edit)
        project_layout.addWidget(self.project_button)
        form_layout.addRow("Project:", project_layout)

        # Name
        self.name_edit = QLineEdit()
        form_layout.addRow("Name:", self.name_edit)

        group_box.setLayout(form_layout)
        self.layout.addWidget(group_box)

    def setup_parameters_layout(self):
        """
        Set up the layout and widgets for the generic layout.
        """
        # Create helper function for boolean dropdowns
        def create_bool_combo():
            combo = QComboBox()
            combo.addItems(["True", "False"])
            return combo

        # Create parameters group box
        group_box = QGroupBox("Training Parameters")
        group_layout = QVBoxLayout(group_box)

        # Add import/export buttons at the top
        import_export_layout = QHBoxLayout()
        
        self.import_button = QPushButton("Import YAML")
        self.import_button.clicked.connect(self.import_parameters)
        import_export_layout.addWidget(self.import_button)

        self.export_button = QPushButton("Export YAML")
        self.export_button.clicked.connect(self.export_parameters)
        import_export_layout.addWidget(self.export_button)
        
        # Add stretch to push buttons to the left
        import_export_layout.addStretch()
        
        group_layout.addLayout(import_export_layout)

        # Create a widget to hold the form layout
        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)

        # Create the scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(form_widget)
        
        group_layout.addWidget(scroll_area)

        # Create parameters
        # Epochs
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setMinimum(1)
        self.epochs_spinbox.setMaximum(1000)
        self.epochs_spinbox.setValue(100)
        form_layout.addRow("Epochs:", self.epochs_spinbox)

        # Patience
        self.patience_spinbox = QSpinBox()
        self.patience_spinbox.setMinimum(1)
        self.patience_spinbox.setMaximum(1000)
        self.patience_spinbox.setValue(30)
        form_layout.addRow("Patience:", self.patience_spinbox)

        # Imgsz
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setMinimum(16)
        self.imgsz_spinbox.setMaximum(4096)
        self.imgsz_spinbox.setValue(self.imgsz)
        form_layout.addRow("Image Size:", self.imgsz_spinbox)

        # Multi Scale
        self.multi_scale_combo = create_bool_combo()
        self.multi_scale_combo.setCurrentText("False")  # Default to False
        form_layout.addRow("Multi Scale:", self.multi_scale_combo)

        # Batch
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setMinimum(1)
        self.batch_spinbox.setMaximum(1024)
        self.batch_spinbox.setValue(self.batch)
        form_layout.addRow("Batch Size:", self.batch_spinbox)

        # Workers
        self.workers_spinbox = QSpinBox()
        self.workers_spinbox.setMinimum(1)
        self.workers_spinbox.setMaximum(64)
        self.workers_spinbox.setValue(8)
        form_layout.addRow("Workers:", self.workers_spinbox)

        # Save
        self.save_combo = create_bool_combo()
        form_layout.addRow("Save:", self.save_combo)

        # Save Period
        self.save_period_spinbox = QSpinBox()
        self.save_period_spinbox.setMinimum(-1)
        self.save_period_spinbox.setMaximum(1000)
        self.save_period_spinbox.setValue(-1)
        form_layout.addRow("Save Period:", self.save_period_spinbox)

        # Freeze Layers
        self.freeze_layers_spinbox = QDoubleSpinBox()
        self.freeze_layers_spinbox.setMinimum(0.0)
        self.freeze_layers_spinbox.setMaximum(1.0)
        self.freeze_layers_spinbox.setSingleStep(0.01)
        self.freeze_layers_spinbox.setValue(0.0)
        form_layout.addRow("Freeze Layers:", self.freeze_layers_spinbox)

        # Weighted Dataset
        self.weighted_combo = create_bool_combo()
        form_layout.addRow("Weighted Sampling:", self.weighted_combo)

        # Dropout
        self.dropout_spinbox = QDoubleSpinBox()
        self.dropout_spinbox.setMinimum(0.0)
        self.dropout_spinbox.setMaximum(1.0)
        self.dropout_spinbox.setValue(0.0)
        form_layout.addRow("Dropout:", self.dropout_spinbox)

        # Optimizer
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"])
        self.optimizer_combo.setCurrentText("auto")
        form_layout.addRow("Optimizer:", self.optimizer_combo)

        # Val
        self.val_combo = create_bool_combo()
        form_layout.addRow("Validation:", self.val_combo)

        # Verbose
        self.verbose_combo = create_bool_combo()
        form_layout.addRow("Verbose:", self.verbose_combo)

        # Add horizontal separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        form_layout.addRow("", separator)
        
        # Add parameter button at the top of custom parameters section
        self.add_param_button = QPushButton("Add Parameter")
        self.add_param_button.clicked.connect(self.add_parameter_pair)
        form_layout.addRow("", self.add_param_button)

        # Add custom parameters section
        self.custom_params_layout = QVBoxLayout()
        form_layout.addRow("", self.custom_params_layout)

        # Remove parameter button at the bottom
        self.remove_param_button = QPushButton("Remove Parameter")
        self.remove_param_button.clicked.connect(self.remove_parameter_pair)
        self.remove_param_button.setEnabled(False)  # Disabled until at least one parameter is added
        form_layout.addRow("", self.remove_param_button)

        self.layout.addWidget(group_box)

    def add_parameter_pair(self):
        """
        Add a new parameter input group with name, value, and type selector.
        """
        param_layout = QHBoxLayout()

        # Parameter name field
        param_name = QLineEdit()
        param_name.setPlaceholderText("Parameter name")

        # Parameter value field
        param_value = QLineEdit()
        param_value.setPlaceholderText("Value")

        # Parameter type selector
        param_type = QComboBox()
        param_type.addItems(["string", "int", "float", "bool"])

        # Add widgets to layout
        param_layout.addWidget(param_name)
        param_layout.addWidget(param_value)
        param_layout.addWidget(param_type)

        # Store the widgets for later retrieval
        self.custom_params.append((param_name, param_value, param_type))
        self.custom_params_layout.addLayout(param_layout)

        # Enable the remove button since we now have at least one parameter
        self.remove_param_button.setEnabled(True)

    def remove_parameter_pair(self):
        """
        Remove the most recently added parameter pair.
        """
        if not self.custom_params:
            return

        # Get the last parameter group
        param_name, param_value, param_type = self.custom_params.pop()

        # Remove the layout containing these widgets
        layout_to_remove = self.custom_params_layout.itemAt(self.custom_params_layout.count() - 1)

        if layout_to_remove:
            # Remove and delete widgets from the layout
            while layout_to_remove.count():
                widget = layout_to_remove.takeAt(0).widget()
                if widget:
                    widget.deleteLater()

            # Remove the layout itself
            self.custom_params_layout.removeItem(layout_to_remove)

        # Disable the remove button if no more parameters
        if not self.custom_params:
            self.remove_param_button.setEnabled(False)

    def setup_buttons_layout(self):
        """

        """
        # Add OK and Cancel buttons
        self.buttons = QPushButton("OK")
        self.buttons.clicked.connect(self.accept)
        self.layout.addWidget(self.buttons)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.layout.addWidget(self.cancel_button)

    def load_model_combobox(self):
        raise NotImplementedError("Subclasses must implement this method.")

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
                self.mapping_edit.setText(class_mapping_path)

            # Set the dataset and class mapping paths
            self.dataset_edit.setText(file_path)

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

            # Set the class mapping path
            self.mapping_edit.setText(file_path)

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

    def import_parameters(self):
        """
        Import parameters from a YAML file with automatic type inference.
        """
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Import Parameters from YAML",
                                                   "",
                                                   "YAML Files (*.yaml *.yml)")
        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)

            if not data:
                QMessageBox.warning(self, "Import Warning", "The YAML file appears to be empty or invalid.")
                return

            # Helper function to infer type from value
            def infer_type_and_value(value):
                """
                Infer the type and convert the value based on its content.
                Returns (type_string, converted_value)
                """
                if isinstance(value, bool):
                    return "bool", value
                elif isinstance(value, int):
                    return "int", value
                elif isinstance(value, float):
                    return "float", value
                elif isinstance(value, str):
                    # Check for boolean strings
                    if value.lower() in ['true', 'false']:
                        return "bool", value.lower() == 'true'
                    # Check for numeric strings
                    try:
                        # Try to convert to int first
                        if '.' not in value:
                            return "int", int(value)
                        else:
                            return "float", float(value)
                    except ValueError:
                        return "string", value
                else:
                    # For any other type, convert to string
                    return "string", str(value)

            # Clear existing custom parameters before importing
            while self.custom_params:
                self.remove_parameter_pair()

            # Map parameters to UI controls
            param_mapping = {
                'epochs': self.epochs_spinbox,
                'patience': self.patience_spinbox,
                'imgsz': self.imgsz_spinbox,
                'multi_scale': self.multi_scale_combo,
                'batch': self.batch_spinbox,
                'workers': self.workers_spinbox,
                'save_period': self.save_period_spinbox,
                'freeze_layers': self.freeze_layers_spinbox,
                'dropout': self.dropout_spinbox,
                'save': self.save_combo,
                'weighted': self.weighted_combo,
                'val': self.val_combo,
                'verbose': self.verbose_combo,
                'optimizer': self.optimizer_combo
            }

            # Update UI controls with imported values
            for param_name, value in data.items():
                param_type, converted_value = infer_type_and_value(value)
                
                if param_name in param_mapping:
                    widget = param_mapping[param_name]
                    
                    if isinstance(widget, QSpinBox):
                        if param_type in ['int', 'float'] and isinstance(converted_value, (int, float)):
                            widget.setValue(int(converted_value))
                    elif isinstance(widget, QDoubleSpinBox):
                        if param_type in ['int', 'float'] and isinstance(converted_value, (int, float)):
                            widget.setValue(float(converted_value))
                    elif isinstance(widget, QComboBox):
                        if param_name in ['multi_scale', 'save', 'weighted', 'val', 'verbose']:
                            # Boolean parameters
                            if param_type == 'bool':
                                widget.setCurrentText("True" if converted_value else "False")
                        else:
                            # String parameters like optimizer
                            if str(converted_value) in [widget.itemText(i) for i in range(widget.count())]:
                                widget.setCurrentText(str(converted_value))
                else:
                    # Add as custom parameter using inferred type
                    self.add_parameter_pair()
                    param_widgets = self.custom_params[-1]
                    param_name_widget, param_value_widget, param_type_widget = param_widgets
                    
                    param_name_widget.setText(param_name)
                    param_type_widget.setCurrentText(param_type)
                    
                    # Set value based on type
                    if param_type == "bool":
                        param_value_widget.setText("True" if converted_value else "False")
                    else:
                        param_value_widget.setText(str(converted_value))

            QMessageBox.information(self, 
                                    "Import Success", 
                                    "Parameters successfully imported with automatic type inference")

        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import parameters: {str(e)}")

    def export_parameters(self):
        """
        Export current parameters to a YAML file with explicit type information.
        """
        file_path, _ = QFileDialog.getSaveFileName(self,
                                                   "Export Parameters to YAML",
                                                   "training_parameters.yaml",
                                                   "YAML Files (*.yaml *.yml)")
        if not file_path:
            return

        try:
            # Structure: types section followed by parameters section
            export_data = {
                'types': {},
                'parameters': {}
            }
            
            # Standard parameters with their types
            standard_params = {
                'epochs': ('int', self.epochs_spinbox.value()),
                'patience': ('int', self.patience_spinbox.value()),
                'imgsz': ('int', self.imgsz_spinbox.value()),
                'batch': ('int', self.batch_spinbox.value()),
                'workers': ('int', self.workers_spinbox.value()),
                'save_period': ('int', self.save_period_spinbox.value()),
                'freeze_layers': ('float', self.freeze_layers_spinbox.value()),
                'dropout': ('float', self.dropout_spinbox.value()),
                'multi_scale': ('bool', self.multi_scale_combo.currentText() == "True"),
                'save': ('bool', self.save_combo.currentText() == "True"),
                'weighted': ('bool', self.weighted_combo.currentText() == "True"),
                'val': ('bool', self.val_combo.currentText() == "True"),
                'verbose': ('bool', self.verbose_combo.currentText() == "True"),
                'optimizer': ('string', self.optimizer_combo.currentText())
            }

            # Add standard parameters
            for param_name, (param_type, value) in standard_params.items():
                export_data['types'][param_name] = param_type
                export_data['parameters'][param_name] = value

            # Custom parameters
            for param_info in self.custom_params:
                param_name, param_value, param_type = param_info
                name = param_name.text().strip()
                value = param_value.text().strip()
                type_name = param_type.currentText()
                
                if name and value:
                    export_data['types'][name] = type_name
                    
                    if type_name == "bool":
                        export_data['parameters'][name] = value.lower() == "true"
                    elif type_name == "int":
                        try:
                            export_data['parameters'][name] = int(value)
                        except ValueError:
                            export_data['parameters'][name] = value
                            export_data['types'][name] = "string"  # Fallback to string
                    elif type_name == "float":
                        try:
                            export_data['parameters'][name] = float(value)
                        except ValueError:
                            export_data['parameters'][name] = value
                            export_data['types'][name] = "string"  # Fallback to string
                    else:  # string type
                        export_data['parameters'][name] = value

            # Write to YAML file
            with open(file_path, 'w') as f:
                yaml.dump(export_data, f, default_flow_style=False, indent=2)

            QMessageBox.information(self, 
                                    "Export Success", 
                                    "Parameters successfully exported")

        except Exception as e:
            QMessageBox.critical(self, 
                                 "Export Error", 
                                 f"Failed to export parameters: {str(e)}")

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
        # Extract values from dialog widgets
        params = {
            'task': self.task,
            'project': self.project_edit.text(),
            'name': self.name_edit.text(),
            'data': self.dataset_edit.text(),
            'epochs': self.epochs_spinbox.value(),
            'patience': self.patience_spinbox.value(),
            'batch': self.batch_spinbox.value(),
            'imgsz': self.imgsz_spinbox.value(),
            'multi_scale': self.multi_scale_combo.currentText() == "True",
            'save': self.save_combo.currentText() == "True",
            'save_period': self.save_period_spinbox.value(),
            'workers': self.workers_spinbox.value(),
            'optimizer': self.optimizer_combo.currentText(),
            'verbose': self.verbose_combo.currentText() == "True",
            'freeze_layers': self.freeze_layers_spinbox.value(),
            'weighted': self.weighted_combo.currentText() == "True",
            'dropout': self.dropout_spinbox.value(),
            'val': self.val_combo.currentText() == "True",
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
        # Either the model path, or the model name provided from combo box
        params['model'] = self.model_edit.text() if self.model_edit.text() else self.model_combo.currentText()
    
        # Add custom parameters (allows overriding the above parameters)
        for param_info in self.custom_params:
            param_name, param_value, param_type = param_info
            name = param_name.text().strip()
            value = param_value.text().strip()
            type_name = param_type.currentText()
            
            if name:
                if type_name == "bool":
                    params[name] = value.lower() == "true"
                elif type_name == "int":
                    try:
                        params[name] = int(value)
                    except ValueError:
                        print(f"Warning: Could not convert '{value}' to int for parameter '{name}', using as string")
                        params[name] = value
                elif type_name == "float":
                    try:
                        params[name] = float(value)
                    except ValueError:
                        print(f"Warning: Could not convert '{value}' to float for parameter '{name}', using as string")
                        params[name] = value
                else:  # string type
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
