import warnings

import os
import gc
import yaml
import datetime
import traceback
import ujson as json
from pathlib import Path

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOETrainerFromScratch

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QFileDialog, QScrollArea, QMessageBox, QCheckBox, QWidget, QVBoxLayout,
                             QLabel, QLineEdit, QDialog, QHBoxLayout, QPushButton, QComboBox, QSpinBox,
                             QFormLayout, QTabWidget, QDoubleSpinBox, QGroupBox, QFrame)

from torch.cuda import empty_cache

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

    def pre_run(self):
        """
        Set up the model and prepare parameters for training.
        """
        try:
            # Extract model path and task
            self.model_path = self.params.pop('model', None)
            task = self.params.get('task', 'segment')
            training_mode = self.params.pop('training_mode', 'linear-probing')

            # Load the model based on task
            if task == 'detect':
                # For detection: initialize from YAML config
                self.model = YOLOE(self.model_path)
                # Load pretrained weights from segmentation checkpoint (same scale)
                # Extract the scale (e.g., 's', 'm', 'l') from the YAML filename
                if 'v8' in self.model_path:
                    scale = self.model_path.split('-')[1].replace('.yaml', '')
                    pretrained_weights = f"yoloe-v8{scale}-seg.pt"
                else:  # v11
                    scale = self.model_path.split('-')[1].replace('.yaml', '')
                    pretrained_weights = f"yoloe-{scale}-seg.pt"
                
                try:
                    self.model.load(pretrained_weights)
                except Exception as e:
                    print(f"Warning: Could not load pretrained weights from {pretrained_weights}: {e}")
            else:
                # For segmentation: load pretrained model directly
                self.model = YOLOE(self.model_path)

            freeze = []
            if training_mode == 'linear-probing':
                head_index = len(self.model.model.model) - 1
                freeze = [str(f) for f in range(0, head_index)]
                for name, child in self.model.model.model[-1].named_children():
                    if "cv3" not in name:
                        freeze.append(f"{head_index}.{name}")

                freeze.extend(
                    [
                        f"{head_index}.cv3.0.0",
                        f"{head_index}.cv3.0.1",
                        f"{head_index}.cv3.1.0",
                        f"{head_index}.cv3.1.1",
                        f"{head_index}.cv3.2.0",
                        f"{head_index}.cv3.2.1",
                    ]
                )

            self.params['freeze'] = freeze
            
            self.data = self.params.pop('data', None)
            
            if self.data is None:
                raise ValueError("Dataset YAML file must be specified in parameters under 'data' key.")
            
            self.params['data'] = dict(
                train=dict(yolo_data=[self.data]),
                val=dict(yolo_data=[self.data]),
            )

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

            # Select the appropriate trainer based on task
            trainer = YOLOETrainerFromScratch

            # Train the model with the correct trainer
            self.model.train(**self.params,
                             trainer=trainer,
                             device=self.device)

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
        pass

    def evaluate_model(self):
        """
        Evaluate the model after training.
        """
        try:
            # Create an instance of EvaluateModelWorker and start it
            eval_params = {
                'data': self.data,
                'imgsz': self.params['imgsz'],
                'split': 'test',  # Evaluate on the test set only
                'save_dir': Path(self.params['project']) / self.params['name'] / 'test',
                'load_vp': True,
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


class TrainModelDialog(QDialog):
    """
    Dialog for training machine learning models for image classification, object detection,
    and instance segmentation.

    :param main_window: MainWindow object
    :param parent: Parent widget
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        self.setWindowIcon(get_icon("eye.png"))
        self.setWindowTitle("Train YOLOE Model")
        self.resize(600, 800)  # Increased height for new parameters

        # Set window settings
        self.setWindowFlags(Qt.Window |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint |
                            Qt.WindowTitleHint)

        # Task - default to segmentation
        self.task = "segment"
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
        
        # Training parameters with defaults for linear-probing
        self._lr0 = 1e-3
        self._warmup_bias_lr = 0.0
        self._weight_decay = 0.025
        self._momentum = 0.9
        self._close_mosaic = 0  # Default for linear-probing

        # Create the layout
        self.layout = QVBoxLayout(self)

        # Create the info layout
        self.setup_info_layout()
        # Create the dataset layout
        self.setup_dataset_layout()
        # Create the model layout (new)
        self.setup_model_layout()
        # Create the output parameters layout
        self.setup_output_parameters_layout()
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
                            "<a href='https://docs.ultralytics.com/models/yoloe/#train-usage'>here</a>.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_dataset_layout(self):
        """Setup the dataset layout."""

        group_box = QGroupBox("Dataset")
        layout = QFormLayout()

        # Task selection (detect or segment)
        self.task_combo = QComboBox()
        self.task_combo.addItems(["segment", "detect"])
        self.task_combo.setCurrentText("segment")
        self.task_combo.currentTextChanged.connect(self.on_task_changed)
        layout.addRow("Task:", self.task_combo)

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
        
    def setup_output_parameters_layout(self):
        """
        Set up the layout and widgets for output parameters (project directory and name).
        """
        group_box = QGroupBox("Output")
        layout = QFormLayout()

        # Project
        self.project_edit = QLineEdit()
        self.project_button = QPushButton("Browse...")
        self.project_button.clicked.connect(self.browse_project_dir)
        project_layout = QHBoxLayout()
        project_layout.addWidget(self.project_edit)
        project_layout.addWidget(self.project_button)
        layout.addRow("Project:", project_layout)

        # Name
        self.name_edit = QLineEdit()
        layout.addRow("Name:", self.name_edit)

        group_box.setLayout(layout)
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

        # Fine-tune or Linear Probing
        self.training_mode = QComboBox()
        self.training_mode.addItems(["linear-probe", "fine-tune"])
        self.training_mode.setCurrentText("linear-probe")
        self.training_mode.currentTextChanged.connect(self.update_by_training_mode)
        form_layout.addRow("Training Mode:", self.training_mode)

        # Epochs
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setMinimum(1)
        self.epochs_spinbox.setMaximum(1000)
        self.epochs_spinbox.setValue(2)
        form_layout.addRow("Epochs:", self.epochs_spinbox)

        # Patience
        self.patience_spinbox = QSpinBox()
        self.patience_spinbox.setMinimum(0)  # Changed minimum to 0 to allow for 0 patience
        self.patience_spinbox.setMaximum(1000)
        self.patience_spinbox.setValue(0)  # Default for linear-probing
        form_layout.addRow("Patience:", self.patience_spinbox)

        # Close Mosaic
        self.close_mosaic_spinbox = QSpinBox()
        self.close_mosaic_spinbox.setMinimum(0)
        self.close_mosaic_spinbox.setMaximum(1000)
        self.close_mosaic_spinbox.setValue(0)  # Default for linear-probing
        form_layout.addRow("Close Mosaic:", self.close_mosaic_spinbox)
        
        # Optimizer
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"])
        self.optimizer_combo.setCurrentText("AdamW")
        form_layout.addRow("Optimizer:", self.optimizer_combo)

        # Learning Rate (lr0)
        self.lr0_spinbox = QDoubleSpinBox()
        self.lr0_spinbox.setDecimals(6)
        self.lr0_spinbox.setMinimum(0.000001)
        self.lr0_spinbox.setMaximum(1.0)
        self.lr0_spinbox.setSingleStep(0.0001)
        self.lr0_spinbox.setValue(self._lr0)
        form_layout.addRow("Learning Rate:", self.lr0_spinbox)

        # Warmup Bias Learning Rate
        self.warmup_bias_lr_spinbox = QDoubleSpinBox()
        self.warmup_bias_lr_spinbox.setDecimals(6)
        self.warmup_bias_lr_spinbox.setMinimum(0.0)
        self.warmup_bias_lr_spinbox.setMaximum(1.0)
        self.warmup_bias_lr_spinbox.setSingleStep(0.0001)
        self.warmup_bias_lr_spinbox.setValue(self._warmup_bias_lr)
        form_layout.addRow("Warmup Bias LR:", self.warmup_bias_lr_spinbox)

        # Weight Decay
        self.weight_decay_spinbox = QDoubleSpinBox()
        self.weight_decay_spinbox.setDecimals(6)
        self.weight_decay_spinbox.setMinimum(0.0)
        self.weight_decay_spinbox.setMaximum(1.0)
        self.weight_decay_spinbox.setSingleStep(0.001)
        self.weight_decay_spinbox.setValue(self._weight_decay)
        form_layout.addRow("Weight Decay:", self.weight_decay_spinbox)

        # Momentum
        self.momentum_spinbox = QDoubleSpinBox()
        self.momentum_spinbox.setDecimals(2)
        self.momentum_spinbox.setMinimum(0.0)
        self.momentum_spinbox.setMaximum(1.0)
        self.momentum_spinbox.setSingleStep(0.01)
        self.momentum_spinbox.setValue(self._momentum)
        form_layout.addRow("Momentum:", self.momentum_spinbox)

        # Imgsz
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setMinimum(16)
        self.imgsz_spinbox.setMaximum(4096)
        self.imgsz_spinbox.setValue(self.imgsz)
        form_layout.addRow("Image Size:", self.imgsz_spinbox)

        # Batch
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setMinimum(1)
        self.batch_spinbox.setMaximum(1024)
        self.batch_spinbox.setValue(self.batch)
        form_layout.addRow("Batch Size:", self.batch_spinbox)

        # Multi Scale
        self.multi_scale_combo = create_bool_combo()
        self.multi_scale_combo.setCurrentText("False")
        form_layout.addRow("Multi Scale:", self.multi_scale_combo)

        # Single Class (cls)
        self.single_class_combo = create_bool_combo()
        self.single_class_combo.setCurrentText("False")
        form_layout.addRow("Single Class:", self.single_class_combo)

        # Dropout
        self.dropout_spinbox = QDoubleSpinBox()
        self.dropout_spinbox.setMinimum(0.0)
        self.dropout_spinbox.setMaximum(1.0)
        self.dropout_spinbox.setSingleStep(0.01)
        self.dropout_spinbox.setValue(0.0)
        form_layout.addRow("Dropout:", self.dropout_spinbox)

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

    def on_task_changed(self):
        """Handle task selection changes (detect vs segment)"""
        self.task = self.task_combo.currentText()
        # Update available models based on task
        self.load_model_combobox()

    def update_by_training_mode(self):
        """Update certain parameters depending on the fine-tune / linear probing"""
        # Get the current value of the fine-tune combo box
        training_mode = self.training_mode.currentText()

        if training_mode == "fine-tune":
            # Fine-tune mode (parameters from YOLOE documentation)
            self.epochs_spinbox.setValue(80)
            self.patience_spinbox.setValue(10)  # Changed from 20 to 10 per docs
            self.close_mosaic_spinbox.setValue(10)
            self._close_mosaic = 10
            
            # These parameters stay the same for both modes
            self.lr0_spinbox.setValue(1e-3)
            self.warmup_bias_lr_spinbox.setValue(0.0)
            self.weight_decay_spinbox.setValue(0.025)
            self.momentum_spinbox.setValue(0.9)
            
            # Ensure optimizer is set to AdamW
            self.optimizer_combo.setCurrentText("AdamW")
        else:
            # Linear-probing mode
            self.epochs_spinbox.setValue(2)
            self.patience_spinbox.setValue(0)
            self.close_mosaic_spinbox.setValue(0)
            self._close_mosaic = 0
            
            # These parameters stay the same for both modes
            self.lr0_spinbox.setValue(1e-3)
            self.warmup_bias_lr_spinbox.setValue(0.0)
            self.weight_decay_spinbox.setValue(0.025)
            self.momentum_spinbox.setValue(0.9)
            
            # Ensure optimizer is set to AdamW
            self.optimizer_combo.setCurrentText("AdamW")

    def load_model_combobox(self):
        """Load the model combobox with the available models based on task."""
        self.model_combo.clear()
        self.model_combo.setEditable(True)

        if self.task == "segment":
            standard_models = [
                'yoloe-v8s-seg.pt',
                'yoloe-v8m-seg.pt',
                'yoloe-v8l-seg.pt',
                'yoloe-11s-seg.pt',
                'yoloe-11m-seg.pt',
                'yoloe-11l-seg.pt',
            ]
            default_model = 'yoloe-v8s-seg.pt'
        else:  # detect
            standard_models = [
                'yoloe-11s.yaml',
                'yoloe-11m.yaml',
                'yoloe-11l.yaml',
                'yoloe-v8s.yaml',
                'yoloe-v8m.yaml',
                'yoloe-v8l.yaml',
            ]
            default_model = 'yoloe-11s.yaml'

        self.model_combo.addItems(standard_models)

        # Set the default model
        self.model_combo.setCurrentIndex(standard_models.index(default_model))

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
        Open a file dialog to browse for a model file.
        """
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Select Model File",
                                                   "",
                                                   "Model Files (*.pt *.pth);;All Files (*)")
        if file_path:
            self.model_path_edit.setText(file_path)

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
            'training_mode': self.training_mode.currentText(),
            'epochs': self.epochs_spinbox.value(),
            'patience': self.patience_spinbox.value(),
            'batch': self.batch_spinbox.value(),
            'imgsz': self.imgsz_spinbox.value(),
            'multi_scale': self.multi_scale_combo.currentText() == "True",
            'single_cls': self.single_class_combo.currentText() == "True",
            'dropout': self.dropout_spinbox.value(),
            'optimizer': self.optimizer_combo.currentText(),
            'lr0': self.lr0_spinbox.value(),
            'warmup_bias_lr': self.warmup_bias_lr_spinbox.value(),
            'weight_decay': self.weight_decay_spinbox.value(),
            'momentum': self.momentum_spinbox.value(),
            'close_mosaic': self.close_mosaic_spinbox.value(),
            'save': self.save_combo.currentText() == "True",
            'save_period': self.save_period_spinbox.value(),
            'workers': self.workers_spinbox.value(),
            'verbose': self.verbose_combo.currentText() == "True",
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

        # Add custom parameters with typed values (allows overriding the above parameters)
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
                        print(f"Warning: Could not convert '{value}' to int for parameter '{name}'. Skipping.")
                elif type_name == "float":
                    try:
                        params[name] = float(value)
                    except ValueError:
                        print(f"Warning: Could not convert '{value}' to float for parameter '{name}'. Skipping.")
                else:  # string
                    params[name] = value

        # Return the dictionary of parameters
        return params

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
                params_to_load = yaml.safe_load(f)

            if not params_to_load:
                QMessageBox.warning(self, "Import Warning", "The selected file is empty.")
                return

            # Helper function to infer type and convert value
            def infer_type_and_value(value):
                if isinstance(value, bool):
                    return "bool", value
                elif isinstance(value, int):
                    return "int", value
                elif isinstance(value, float):
                    return "float", value
                elif isinstance(value, str):
                    # Try to infer if it looks like a bool
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
                'multi_scale': self.multi_scale_combo,
                'batch': self.batch_spinbox,
                'single_cls': self.single_class_combo,
                'dropout': self.dropout_spinbox,
                'workers': self.workers_spinbox,
                'save_period': self.save_period_spinbox,
                'lr0': self.lr0_spinbox,
                'warmup_bias_lr': self.warmup_bias_lr_spinbox,
                'weight_decay': self.weight_decay_spinbox,
                'momentum': self.momentum_spinbox,
                'close_mosaic': self.close_mosaic_spinbox,
                'save': self.save_combo,
                'val': self.val_combo,
                'verbose': self.verbose_combo,
                'optimizer': self.optimizer_combo,
                'training_mode': self.training_mode
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
                        if param_name in ['multi_scale', 'save', 'val', 'verbose', 'single_cls']:
                            widget.setCurrentText("True" if converted_value else "False")
                        elif str(converted_value) in [widget.itemText(i) for i in range(widget.count())]:
                            widget.setCurrentText(str(converted_value))
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
                                                   "yoloe_training_parameters.yaml",
                                                   "YAML Files (*.yaml *.yml)")
        if not file_path:
            return

        try:
            # Use a single flat dictionary for export
            export_data = {}

            # Standard parameters
            export_data['training_mode'] = self.training_mode.currentText()
            export_data['epochs'] = self.epochs_spinbox.value()
            export_data['patience'] = self.patience_spinbox.value()
            export_data['close_mosaic'] = self.close_mosaic_spinbox.value()
            export_data['optimizer'] = self.optimizer_combo.currentText()
            export_data['lr0'] = self.lr0_spinbox.value()
            export_data['warmup_bias_lr'] = self.warmup_bias_lr_spinbox.value()
            export_data['weight_decay'] = self.weight_decay_spinbox.value()
            export_data['momentum'] = self.momentum_spinbox.value()
            export_data['imgsz'] = self.imgsz_spinbox.value()
            export_data['batch'] = self.batch_spinbox.value()
            export_data['multi_scale'] = self.multi_scale_combo.currentText() == "True"
            export_data['single_cls'] = self.single_class_combo.currentText() == "True"
            export_data['dropout'] = self.dropout_spinbox.value()
            export_data['workers'] = self.workers_spinbox.value()
            export_data['save'] = self.save_combo.currentText() == "True"
            export_data['save_period'] = self.save_period_spinbox.value()
            export_data['val'] = self.val_combo.currentText() == "True"
            export_data['verbose'] = self.verbose_combo.currentText() == "True"

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
