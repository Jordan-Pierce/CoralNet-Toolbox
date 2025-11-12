import warnings

import os
import gc
import json
import yaml
import datetime
import traceback
from pathlib import Path

from torch.cuda import empty_cache

from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QLineEdit, QHBoxLayout, QPushButton, QFormLayout, QGroupBox, QComboBox,
                             QVBoxLayout, QWidget, QTabWidget, QSpinBox, QDoubleSpinBox,
                             QLabel, QMessageBox, QScrollArea, QFrame, QDialog, QFileDialog)

from coralnet_toolbox.MachineLearning.SMP import SemanticModel

from coralnet_toolbox.MachineLearning.EvaluateModel.QtSemantic import EvaluateModelWorker

from coralnet_toolbox.MachineLearning.SMP import get_segmentation_losses
from coralnet_toolbox.MachineLearning.SMP import get_segmentation_encoders
from coralnet_toolbox.MachineLearning.SMP import get_segmentation_decoders
from coralnet_toolbox.MachineLearning.SMP import get_segmentation_optimizers

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class TrainModelWorker(QThread):
    """
    Worker thread for training a SemanticModel.
    This is separate from the base TrainModelWorker to keep logic clean.
    """
    training_started = pyqtSignal()
    training_completed = pyqtSignal()
    training_error = pyqtSignal(str)

    def __init__(self, params, device):
        """
        Initialize the SemanticTrainModelWorker.

        Args:
            params: A dictionary of parameters for training.
            device: The device to use for training (e.g., 'cpu' or 'cuda').
        """
        super().__init__()
        self.params = params
        self.device = device
        self.model = None
        
    def pre_run(self):
        """
        Pre-run setup before starting the training process.
        """
        # Check if the imgsz is divisible by 32
        if self.params['imgsz'] % 32 != 0:
            raise ValueError("Image size must be divisible by 32.")

    def run(self):
        """
        Run the training process in a separate thread.
        """
        try:
            # Emit signal to indicate training has started
            self.training_started.emit()

            # Pre-run checks
            self.pre_run()

            # Initialize SemanticModel
            self.model = SemanticModel()
            
            # Pop 'task' as .train() doesn't accept it
            self.params.pop('task', None)
            
            # Pass device separately
            self.params['device'] = self.device
            
            # Run training, validate on validation set if applicable
            self.model.train(**self.params)
            
            # Evaluate model on test set, if applicable
            self.evaluate_model()

            # Emit signal to indicate training has completed
            self.training_completed.emit()

        except Exception as e:
            print(f"Error during training: {e}\n\nTraceback:\n{traceback.format_exc()}")
            self.training_error.emit(f"Error during training: {e} (see console log)")
        
    def evaluate_model(self):
        """
        Evaluate the model after training.
        """
        try:
            # Check that there is a test folder
            test_folder = f"{os.path.dirname(self.params['data_yaml'])}/test"
            print(f"Note: Looking for test folder: {test_folder}")
            if not os.path.exists(test_folder):
                print("Warning: No test folder found in that location. Skipping evaluation.")
                return
            
            # Create an instance of EvaluateModelWorker and start it
            eval_params = {
                'data_yaml': self.params['data_yaml'],
                'imgsz': self.params['imgsz'],
                'split': 'test',
                'save_dir': Path(self.params['project']) / self.params['name'] / 'test',
                'num_vis_samples': 5,  # Default number of visualization samples
                'device': self.device,  # Use the training device
                'batch': 1,  # Default batch size
                'conf': 0.5  # Default confidence threshold
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


# ----------------------------------------------------------------------------------------------------------------------
# Main UI Class
# ----------------------------------------------------------------------------------------------------------------------

class Semantic(QDialog):  # Does not inherit from Base due to major differences
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        self.setWindowIcon(get_icon("coralnet.png"))
        self.setWindowTitle("Train Semantic Segmentation Model")
        self.resize(600, 750)  

        # Set window settings
        self.setWindowFlags(Qt.Window |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint |
                            Qt.WindowTitleHint)

        # --- Variables (from Base) ---
        self.task = "semantic"  # Hardcoded for this class
        self.params = {}
        self.custom_params = []
        self.model_path = None
        self.class_mapping = {}
        self.class_mapping_path = ""
        self.imgsz = 640  # Task-specific default
        self.batch = 4    # Task-specific default

        # Create the layout
        self.layout = QVBoxLayout(self)

        # --- Setup UI components (from Base) ---
        self.setup_info_layout()
        self.setup_dataset_layout() 
        self.setup_output_layout()
        self.setup_model_layout()
        self.setup_parameters_layout()
        self.setup_buttons_layout()
        
    def setup_info_layout(self):
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()
        info_label = QLabel(
            "Details on different hyperparameters can be found "
            "<a href='https://segmentation-models-pytorch.readthedocs.io/en/latest/'>here</a>."
        )
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_dataset_layout(self):
        """Setup the dataset layout for Semantic Segmentation."""
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
        
    def setup_output_layout(self):
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

    def setup_model_layout(self):
        group_box = QGroupBox("Model Selection")
        layout = QVBoxLayout()
        tab_widget = QTabWidget()

        # Tab 1: Select encoder and decoder from dropdowns
        model_select_tab = QWidget()
        model_select_layout = QFormLayout(model_select_tab)
        
        # Encoder selection
        self.encoder_combo = QComboBox()
        encoders = get_segmentation_encoders()
        self.encoder_combo.addItems(encoders)
        if 'mit_b0' in encoders:
            self.encoder_combo.setCurrentText('mit_b0')
        elif encoders:
            self.encoder_combo.setCurrentIndex(0)
        model_select_layout.addRow("Encoder:", self.encoder_combo)
        
        # Decoder selection
        self.decoder_combo = QComboBox()
        decoders = get_segmentation_decoders()
        self.decoder_combo.addItems(decoders)
        if 'Segformer' in decoders:
            self.decoder_combo.setCurrentText('Segformer')
        elif decoders:
            self.decoder_combo.setCurrentIndex(0)
        model_select_layout.addRow("Decoder:", self.decoder_combo)
        
        tab_widget.addTab(model_select_tab, "Select Encoder and Decoder")

        # Tab 2: Use existing model (pre_trained_path)
        model_existing_tab = QWidget()
        model_existing_layout = QFormLayout(model_existing_tab)
        self.model_edit = QLineEdit()
        self.model_button = QPushButton("Browse...")
        self.model_button.clicked.connect(self.browse_model_file)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(self.model_button)
        model_existing_layout.addRow("Existing Model:", model_layout)
        tab_widget.addTab(model_existing_tab, "Use Existing Model")

        # Disable the pre-trained model tab
        tab_widget.setTabEnabled(1, False)

        layout.addWidget(tab_widget)
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

        # Connect signals to provide user feedback
        self.model_edit.textChanged.connect(self.on_model_path_changed)
        self.encoder_combo.currentTextChanged.connect(self.on_encoder_decoder_changed)
        self.decoder_combo.currentTextChanged.connect(self.on_encoder_decoder_changed)
        
    def on_model_path_changed(self, text):
        """Provide feedback when model path changes."""
        if text.strip():
            # Disable encoder/decoder selection when model path is set
            self.encoder_combo.setEnabled(False)
            self.decoder_combo.setEnabled(False)
        else:
            # Re-enable encoder/decoder selection
            self.encoder_combo.setEnabled(True)
            self.decoder_combo.setEnabled(True)

    def on_encoder_decoder_changed(self):
        """Provide feedback when encoder/decoder selection changes."""
        if self.encoder_combo.currentText() and self.decoder_combo.currentText():
            # Could optionally disable model path input here
            pass

    def setup_parameters_layout(self):
        """Set up the layout and widgets for the generic layout."""
        def create_bool_combo():
            combo = QComboBox()
            combo.addItems(["True", "False"])
            return combo

        group_box = QGroupBox("Training Parameters")
        group_layout = QVBoxLayout(group_box)

        import_export_layout = QHBoxLayout()
        self.import_button = QPushButton("Import YAML")
        self.import_button.clicked.connect(self.import_parameters)
        import_export_layout.addWidget(self.import_button)
        self.export_button = QPushButton("Export YAML")
        self.export_button.clicked.connect(self.export_parameters)
        import_export_layout.addWidget(self.export_button)
        import_export_layout.addStretch()
        group_layout.addLayout(import_export_layout)

        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(form_widget)
        group_layout.addWidget(scroll_area)

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
        self.imgsz_spinbox.setSingleStep(32)  # SMP models need 32-divisible
        form_layout.addRow("Image Size:", self.imgsz_spinbox)

        # Batch
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setMinimum(1)
        self.batch_spinbox.setMaximum(1024)
        self.batch_spinbox.setValue(self.batch)
        form_layout.addRow("Batch Size:", self.batch_spinbox)

        # Freeze Layers (Mapped to 'freeze' in get_parameters)
        self.freeze_layers_spinbox = QDoubleSpinBox()
        self.freeze_layers_spinbox.setMinimum(0.0)
        self.freeze_layers_spinbox.setMaximum(1.0)
        self.freeze_layers_spinbox.setSingleStep(0.01)
        self.freeze_layers_spinbox.setValue(0.50)  # Default for SMP
        form_layout.addRow("Freeze (Encoder %):", self.freeze_layers_spinbox)
        
        # Augmentation
        self.augmentation_combo = create_bool_combo()
        self.augmentation_combo.setCurrentText("True")
        form_layout.addRow("Augmentation:", self.augmentation_combo)

        # Dropout
        self.dropout_spinbox = QDoubleSpinBox()
        self.dropout_spinbox.setMinimum(0.0)
        self.dropout_spinbox.setMaximum(1.0)
        self.dropout_spinbox.setValue(0.50)
        form_layout.addRow("Dropout:", self.dropout_spinbox)

        # Optimizer
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(get_segmentation_optimizers())
        self.optimizer_combo.setCurrentText("Adam")  # Default for SMP
        form_layout.addRow("Optimizer:", self.optimizer_combo)
        
        # Learning rate
        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setMinimum(1e-6)
        self.lr_spinbox.setMaximum(1.0)
        self.lr_spinbox.setDecimals(6)
        self.lr_spinbox.setValue(0.0001)  # Default for SMP
        form_layout.addRow("Learning Rate:", self.lr_spinbox)
        
        # Loss function
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(get_segmentation_losses())
        self.loss_combo.setCurrentText("DiceLoss")  # Default for SMP
        form_layout.addRow("Loss Function:", self.loss_combo)

        # Ignore background
        self.ignore_index_combo = QComboBox()
        self.ignore_index_combo.addItems(["False", "True"])
        self.ignore_index_combo.setCurrentText("False")  # Default False
        form_layout.addRow("Ignore background:", self.ignore_index_combo)
        
        # Workers
        self.workers_spinbox = QSpinBox()
        self.workers_spinbox.setMinimum(1)
        self.workers_spinbox.setMaximum(64)
        self.workers_spinbox.setValue(8)
        form_layout.addRow("Workers:", self.workers_spinbox)
        
        # Validation during training
        self.val_combo = create_bool_combo()
        form_layout.addRow("Validation:", self.val_combo)
        
        # --- Custom Parameters Section ---
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        form_layout.addRow("", separator)
        
        self.add_param_button = QPushButton("Add Parameter")
        self.add_param_button.clicked.connect(self.add_parameter_pair)
        form_layout.addRow("", self.add_param_button)

        self.custom_params_layout = QVBoxLayout()
        form_layout.addRow("", self.custom_params_layout)

        self.remove_param_button = QPushButton("Remove Parameter")
        self.remove_param_button.clicked.connect(self.remove_parameter_pair)
        self.remove_param_button.setEnabled(False)
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
                try:
                    self.class_mapping = json.load(open(class_mapping_path, 'r'))
                    self.class_mapping_path = class_mapping_path
                    self.mapping_edit.setText(self.class_mapping_path)
                except Exception as e:
                    print(f"Warning: Failed to load class mapping from {class_mapping_path}: {e}")
                    self.class_mapping = {}
                    self.class_mapping_path = ""

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
            self.class_mapping_path = file_path
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

            # For backward compatibility, check if the old nested 'parameters' key exists.
            # If not, use the whole data dictionary.
            params_to_load = data.get('parameters', data)

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
                    if value.lower() in ['true', 'false']:
                        return "bool", value.lower() == 'true'
                    try:
                        if '.' not in value:
                            return "int", int(value)
                        else:
                            return "float", float(value)
                    except ValueError:
                        return "string", value
                else:
                    return "string", str(value)

            # Clear existing custom parameters before importing
            while self.custom_params:
                self.remove_parameter_pair()

            # Map standard parameters to their UI controls
            param_mapping = {
                'epochs': self.epochs_spinbox,
                'patience': self.patience_spinbox,
                'imgsz': self.imgsz_spinbox,
                'batch': self.batch_spinbox,
                'workers': self.workers_spinbox,
                'freeze': self.freeze_layers_spinbox,
                'dropout': self.dropout_spinbox,
                'lr': self.lr_spinbox,
                'augment_data': self.augmentation_combo,
                'val': self.val_combo,
                'optimizer': self.optimizer_combo,
                'loss_function': self.loss_combo,
                'ignore_index': self.ignore_index_combo
            }

            # Update UI controls with imported values
            for param_name, value in params_to_load.items():
                param_type, converted_value = infer_type_and_value(value)
                
                if param_name in param_mapping:
                    widget = param_mapping[param_name]
                    
                    if isinstance(widget, QSpinBox):
                        if isinstance(converted_value, (int, float)):
                            widget.setValue(int(converted_value))
                    elif isinstance(widget, QDoubleSpinBox):
                        if isinstance(converted_value, (int, float)):
                            widget.setValue(float(converted_value))
                    elif isinstance(widget, QComboBox):
                        if param_name in ['augment_data', 'val']:
                            widget.setCurrentText("True" if converted_value else "False")
                        elif param_name == 'ignore_index':
                            if converted_value == 0:
                                widget.setCurrentText("True")
                            else:
                                widget.setCurrentText("False")
                        elif str(converted_value) in [widget.itemText(i) for i in range(widget.count())]:
                            widget.setCurrentText(str(converted_value))
                    elif isinstance(widget, QLineEdit):
                        widget.setText(str(converted_value))
            else:
                # Add as a custom parameter
                self.add_parameter_pair()
                param_name_widget, param_value_widget, param_type_widget = self.custom_params[-1]
                
                param_name_widget.setText(param_name)
                param_type_widget.setCurrentText(param_type)
                
                if param_type == "bool":
                    param_value_widget.setText("True" if converted_value else "False")
                else:
                    param_value_widget.setText(str(converted_value))

            QMessageBox.information(self, 
                                    "Import Success", 
                                    "Parameters successfully imported with automatic type inference.")

        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import parameters: {str(e)}")

    def export_parameters(self):
        """
        Export current parameters to a flat YAML file.
        """
        file_path, _ = QFileDialog.getSaveFileName(self,
                                                   "Export Parameters to YAML",
                                                   "training_parameters.yaml",
                                                   "YAML Files (*.yaml *.yml)")
        if not file_path:
            return

        try:
            # Use a single flat dictionary for export
            export_data = {}

            # Standard parameters matching get_parameters()
            export_data['epochs'] = self.epochs_spinbox.value()
            export_data['patience'] = self.patience_spinbox.value()
            export_data['imgsz'] = self.imgsz_spinbox.value()
            export_data['batch'] = self.batch_spinbox.value()
            export_data['workers'] = self.workers_spinbox.value()
            export_data['freeze'] = self.freeze_layers_spinbox.value()
            export_data['dropout'] = self.dropout_spinbox.value()
            export_data['lr'] = self.lr_spinbox.value()
            export_data['augment_data'] = self.augmentation_combo.currentText() == "True"
            export_data['val'] = self.val_combo.currentText() == "True"
            export_data['optimizer'] = self.optimizer_combo.currentText()
            export_data['loss_function'] = self.loss_combo.currentText()
            export_data['ignore_index'] = 0 if self.ignore_index_combo.currentText() == "True" else None

            # Custom parameters
            for param_info in self.custom_params:
                param_name_widget, param_value_widget, param_type_widget = param_info
                name = param_name_widget.text().strip()
                value_str = param_value_widget.text().strip()
                type_name = param_type_widget.currentText()
                
                if name and value_str:
                    # Convert value to the correct type before exporting
                    try:
                        if type_name == "bool":
                            value = value_str.lower() == "true"
                        elif type_name == "int":
                            value = int(value_str)
                        elif type_name == "float":
                            value = float(value_str)
                        else:  # string type
                            value = value_str
                        export_data[name] = value
                    except ValueError:
                        # If conversion fails, save it as a string
                        print(f"Warning: Could not convert '{value_str}' to {type_name} for parameter '{name}'. "
                              "Saving as string.")
                        export_data[name] = value_str

            # Write the flat dictionary to the YAML file
            with open(file_path, 'w') as f:
                yaml.dump(export_data, f, default_flow_style=False, sort_keys=False, indent=2)

            QMessageBox.information(self, 
                                    "Export Success", 
                                    "Parameters successfully exported.")

        except Exception as e:
            QMessageBox.critical(self, 
                                 "Export Error", 
                                 f"Failed to export parameters: {str(e)}")

    def accept(self):
        """Handle the OK button click event with validation."""
        # Validate model selection
        pre_trained_path = self.model_edit.text().strip()
        encoder_selected = bool(self.encoder_combo.currentText())
        decoder_selected = bool(self.decoder_combo.currentText())
        
        if pre_trained_path and (encoder_selected or decoder_selected):
            reply = QMessageBox.question(
                self, 
                "Model Selection", 
                "You have selected both a pre-trained model and encoder/decoder. "
                "The pre-trained model will be used. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        if not pre_trained_path and not (encoder_selected and decoder_selected):
            QMessageBox.warning(
                self, 
                "Model Selection", 
                "Please either select a pre-trained model OR choose both encoder and decoder."
            )
            return
        
        self.train_model()
        super().accept()

    def get_parameters(self):
        """Get parameters, customized for SemanticModel."""
        # Extract values from dialog widgets
        params = {
            'task': self.task,
            'project': self.project_edit.text(),
            'name': self.name_edit.text(),
            'data_yaml': self.dataset_edit.text(),
            'epochs': self.epochs_spinbox.value(),
            'patience': self.patience_spinbox.value(),
            'batch': self.batch_spinbox.value(),
            'imgsz': self.imgsz_spinbox.value(),
            'augment_data': self.augmentation_combo.currentText() == "True",
            'workers': self.workers_spinbox.value(),
            'optimizer': self.optimizer_combo.currentText(),
            'freeze': self.freeze_layers_spinbox.value(),
            'dropout': self.dropout_spinbox.value(),
            'lr': self.lr_spinbox.value(),
            'loss_function': self.loss_combo.currentText(),
            'ignore_index': 0 if self.ignore_index_combo.currentText() == "True" else None,
            'val': self.val_combo.currentText() == "True",
            'exist_ok': True,
            'num_vis_samples': 5,
            'class_mapping': self.class_mapping_path,  # provide path to class mapping file
        }
        
        # Handle model selection logic
        pre_trained_path = self.model_edit.text().strip()
        
        if pre_trained_path:
            # Use existing model - don't pass encoder/decoder names
            params['pre_trained_path'] = pre_trained_path
            print(f"Using pre-trained model: {pre_trained_path}")
        else:
            # Use encoder/decoder selection - don't pass pre_trained_path
            params['encoder_name'] = self.encoder_combo.currentText()
            params['decoder_name'] = self.decoder_combo.currentText()
            print(f"Building new model: {params['decoder_name']}-{params['encoder_name']}")
        
        # Default project folder
        project = 'Data/Training'
        params['project'] = params['project'] if params['project'] else project
        
        # Default project name
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d_%H-%M-%S")
        params['name'] = params['name'] if params['name'] else now
        
        # Add custom parameters...
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
                        params[name] = value
                elif type_name == "float":
                    try: 
                        params[name] = float(value)
                    except ValueError: 
                        params[name] = value
                else:  # string type
                    params[name] = value
                        
        return params

    def train_model(self):
        """Train the model using the Semantic-specific worker."""
        # Get parameters
        self.params = self.get_parameters()
        
        # Create and start the training worker
        self.worker = TrainModelWorker(self.params, self.main_window.device)
        self.worker.training_started.connect(self.on_training_started)
        self.worker.training_completed.connect(self.on_training_completed)
        self.worker.training_error.connect(self.on_training_error)
        self.worker.start()

    def on_training_started(self):
        """Handle the event when training starts."""
        output_dir_path = os.path.join(self.params['project'], self.params['name'])
        os.makedirs(output_dir_path, exist_ok=True)
        if self.class_mapping:
            with open(f"{output_dir_path}/class_mapping.json", 'w') as json_file:
                json.dump(self.class_mapping, json_file, indent=4)
        message = "Model training has commenced.\nMonitor the console for real-time progress."
        QMessageBox.information(self, "Model Training Status", message)
        
    def on_training_error(self, error_message):
        """Handle the event when an error occurs during training."""
        QMessageBox.critical(self, "Error", error_message)
        print(error_message)

    def on_training_completed(self):
        """Handle the event when training completes."""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Model Training Status")
        msg_box.setText("Model training has successfully been completed.")
        msg_box.addButton(QMessageBox.Ok)
        deploy_button = msg_box.addButton("Deploy Model", QMessageBox.AcceptRole)
        msg_box.exec_()
        if msg_box.clickedButton() == deploy_button:
            self.deploy_trained_model()

    def deploy_trained_model(self):
        """Load the trained SMP model and class mapping into a deployment dialog."""
        if not self.main_window.image_window.raster_manager.image_paths:
            QMessageBox.warning(self, 
                                "Deploy Model", 
                                "No images found for deployment, you must import images first.")
            return

        output_folder = os.path.join(self.params['project'], self.params['name'])

        # SemanticModel saves weights in a structure like:
        # {project}/{name}/{encoder}_{decoder}/weights/best.pt
        best_weights = None
        for root, dirs, files in os.walk(output_folder):
            if "best.pt" in files and "weights" in root:
                best_weights = os.path.join(root, "best.pt")
                output_folder = os.path.dirname(os.path.dirname(best_weights))  # The run dir
                break
        
        if not best_weights:
            QMessageBox.warning(self, "Deploy Model", f"Could not find 'best.pt' in {output_folder}.")
            return

        # Load class mapping
        class_mapping_path = f"{output_folder}/class_mapping.json"
        class_mapping = {}
        if os.path.exists(class_mapping_path):
            try:
                with open(class_mapping_path, "r") as f:
                    class_mapping = json.load(f)
            except Exception as e:
                QMessageBox.warning(self, "Deploy Model", f"Failed to load class mapping: {str(e)}")

        # Use the existing deployment dialog instance from main_window
        if self.task == "semantic":
            if not hasattr(self.main_window, "semantic_deploy_model_dialog"):
                QMessageBox.warning(self, "Deploy Model", "No deployment dialog found for 'semantic' task.")
                return
            deploy_dialog = self.main_window.semantic_deploy_model_dialog
        else:
            QMessageBox.warning(self, "Deploy Model", "Unknown task type for deployment.")
            return

        # Set path and mapping, then load the model
        deploy_dialog.model_path = best_weights
        deploy_dialog.class_mapping = class_mapping  # use the loaded class mapping
        deploy_dialog.load_model()

        # Update label window and status bar
        if hasattr(deploy_dialog, "add_labels_to_label_window"):
            deploy_dialog.add_labels_to_label_window()
        if hasattr(deploy_dialog, "check_and_display_class_names"):
            deploy_dialog.check_and_display_class_names()
        if hasattr(deploy_dialog, "status_bar"):
            deploy_dialog.status_bar.setText(f"Model loaded: {os.path.basename(best_weights)}")


