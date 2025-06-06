import warnings

import gc
import datetime
import traceback

from ultralytics import YOLO, RTDETR
import ultralytics.data.build as build
import ultralytics.models.yolo.classify.train as train_build
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.dataset import ClassificationDataset

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QFileDialog, QScrollArea, QMessageBox, QCheckBox, QWidget, QVBoxLayout,
                             QLabel, QLineEdit, QDialog, QHBoxLayout, QPushButton, QComboBox, QSpinBox,
                             QFormLayout, QTabWidget, QDoubleSpinBox, QGroupBox, QFrame)

from torch.cuda import empty_cache

from coralnet_toolbox.MachineLearning.Community.cfg import get_available_configs
from coralnet_toolbox.MachineLearning.WeightedDataset import WeightedInstanceDataset
from coralnet_toolbox.MachineLearning.WeightedDataset import WeightedClassificationDataset

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class TuneModelWorker(QThread):
    """
    Worker thread for tuning a model's hyperparameters.

    Signals:
        tuning_started: Emitted when the tuning starts.
        tuning_completed: Emitted when the tuning completes.
        tuning_error: Emitted when an error occurs during tuning.
        tuning_progress: Emitted to report progress during tuning.
    """
    tuning_started = pyqtSignal()
    tuning_completed = pyqtSignal()
    tuning_error = pyqtSignal(str)
    tuning_progress = pyqtSignal(int, int)  # current iteration, total iterations

    def __init__(self, params, device):
        """
        Initialize the TuneModelWorker.

        Args:
            params: A dictionary of parameters for tuning.
            device: The device to use for tuning (e.g., 'cpu' or 'cuda').
        """
        super().__init__()
        self.params = params
        self.device = device

        self.is_yolo = True
        self.model = None
        self.model_path = None
        self.weighted = False

    def pre_run(self):
        """
        Set up the model and prepare parameters for tuning.
        """
        try:
            # Extract model path
            self.model_path = self.params.pop('model', None)
            # Get the weighted flag
            self.weighted = self.params.pop('weighted', False)
            # Whether to use YOLO or RTDETR
            self.is_yolo = False if 'detr' in self.model_path.lower() else True

            # Determine if ultralytics or community
            if self.model_path in get_available_configs(task=self.params['task']):
                self.model_path = get_available_configs(task=self.params['task'])[self.model_path]
                # Cannot use weighted sampling with community models
                self.weighted = False

            # Use the custom dataset class for weighted sampling
            if self.weighted and self.params['task'] == 'classify':
                train_build.ClassificationDataset = WeightedClassificationDataset
            elif self.weighted and self.params['task'] in ['detect', 'segment']:
                build.YOLODataset = WeightedInstanceDataset

            # Load the model
            if self.is_yolo:
                self.model = YOLO(self.model_path)
            else:
                self.model = RTDETR(self.model_path)

            # Set the task in the model itself
            self.model.task = self.params['task']

        except Exception as e:
            print(f"Error during setup: {e}\n\nTraceback:\n{traceback.format_exc()}")
            self.tuning_error.emit(f"Error during setup: {e} (see console log)")
            raise

    def run(self):
        """
        Run the tuning process in a separate thread.
        """
        try:
            # Emit signal to indicate tuning has started
            self.tuning_started.emit()

            # Set up the model and parameters
            self.pre_run()

            # Tune the model using the built-in tune method
            self.model.tune(**self.params)

            # Post-run cleanup
            self.post_run()

            # Emit signal to indicate tuning has completed
            self.tuning_completed.emit()

        except Exception as e:
            print(f"Error during tuning: {e}\n\nTraceback:\n{traceback.format_exc()}")
            self.tuning_error.emit(f"Error during tuning: {e} (see console log)")
        finally:
            self._cleanup()

    def post_run(self):
        """
        Clean up resources after tuning.
        """
        # Revert to the original dataset class without weighted sampling
        if self.weighted and self.params['task'] == 'classify':
            train_build.ClassificationDataset = ClassificationDataset
        elif self.weighted and self.params['task'] in ['detect', 'segment']:
            build.YOLODataset = YOLODataset

    def _cleanup(self):
        """
        Clean up resources after tuning.
        """
        if self.model:
            del self.model
        gc.collect()
        empty_cache()


class Base(QDialog):
    """
    Dialog for tuning hyperparameters of machine learning models for image classification, 
    object detection, and instance segmentation.

    :param main_window: MainWindow object
    :param parent: Parent widget
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Tune Model")
        self.resize(700, 800)  

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
        self.custom_space_params = []
        
        # Best model weights
        self.model_path = None

        # Task specific parameters
        self.imgsz = 640
        self.batch = 4

        # Create the layout
        self.layout = QVBoxLayout(self)

        # Create the info layout
        self.setup_info_layout()
        # Create the dataset layout
        self.setup_dataset_layout()
        # Create the model layout
        self.setup_model_layout()
        # Create the output layout
        self.setup_output_layout()
        # Create the training parameters layout (combined tuning and base parameters)
        self.setup_parameters_layout()
        # Create the search space layout
        self.setup_search_space_layout()
        # Create the buttons layout
        self.setup_buttons_layout()

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Hyperparameter tuning systematically searches for optimal hyperparameters. "
                            "Details on tuning can be found "
                            "<a href='https://docs.ultralytics.com/guides/hyperparameter-tuning/'>here</a>.")

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
        Set up the layout and widgets for training parameters including tuning-specific parameters.
        """
        # Create helper function for boolean dropdowns
        def create_bool_combo():
            combo = QComboBox()
            combo.addItems(["True", "False"])
            return combo

        # Create parameters group box
        group_box = QGroupBox("Training Parameters")
        group_layout = QVBoxLayout(group_box)

        # Create a widget to hold the form layout
        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)

        # Create the scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(form_widget)
        
        group_layout.addWidget(scroll_area)

        # Tuning-specific parameters
        # Iterations
        self.iterations_spinbox = QSpinBox()
        self.iterations_spinbox.setMinimum(1)
        self.iterations_spinbox.setMaximum(10000)
        self.iterations_spinbox.setValue(100)
        form_layout.addRow("Iterations:", self.iterations_spinbox)

        # Base training parameters
        # Epochs (for each iteration)
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setMinimum(1)
        self.epochs_spinbox.setMaximum(1000)
        self.epochs_spinbox.setValue(10)  # Lower default for tuning
        form_layout.addRow("Epochs (per iteration):", self.epochs_spinbox)

        # Imgsz
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setMinimum(16)
        self.imgsz_spinbox.setMaximum(4096)
        self.imgsz_spinbox.setValue(self.imgsz)
        form_layout.addRow("Image Size:", self.imgsz_spinbox)
        
        # Multi Scale
        self.multi_scale_combo = QComboBox()
        self.multi_scale_combo.addItems(["True", "False"])
        self.multi_scale_combo.setCurrentText("False")
        form_layout.addRow("Multi Scale:", self.multi_scale_combo)

        # Batch
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setMinimum(1)
        self.batch_spinbox.setMaximum(1024)
        self.batch_spinbox.setValue(self.batch)
        form_layout.addRow("Batch Size:", self.batch_spinbox)
        
        # Weighted
        self.weighted_combo = QComboBox()
        self.weighted_combo.addItems(["True", "False"])
        self.weighted_combo.setCurrentText("True")
        form_layout.addRow("Weighted Sampling:", self.weighted_combo)
        
        # Optimizer
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"])
        self.optimizer_combo.setCurrentText("auto")
        form_layout.addRow("Optimizer:", self.optimizer_combo)
        
        # Validation
        self.val_combo = QComboBox()
        self.val_combo.addItems(["True", "False"])
        self.val_combo.setCurrentText("False")
        form_layout.addRow("Validation:", self.val_combo)

        # Workers
        self.workers_spinbox = QSpinBox()
        self.workers_spinbox.setMinimum(1)
        self.workers_spinbox.setMaximum(64)
        self.workers_spinbox.setValue(8)
        form_layout.addRow("Workers:", self.workers_spinbox)

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
        self.remove_param_button.setEnabled(False)
        form_layout.addRow("", self.remove_param_button)

        self.layout.addWidget(group_box)

    def setup_search_space_layout(self):
        """
        Set up the layout and widgets for hyperparameter search space configuration.
        """
        group_box = QGroupBox("Search Space Configuration")
        group_layout = QVBoxLayout(group_box)

        # Create a widget to hold the form layout
        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)

        # Create the scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(form_widget)
        
        group_layout.addWidget(scroll_area)

        # Default search space parameters
        default_space = {
            "lr0": (1e-5, 1e-1),
            "lrf": (0.0001, 0.1),
            "momentum": (0.7, 0.98, 0.3),
            "weight_decay": (0.0, 0.001),
            "warmup_epochs": (0.0, 5.0),
            "warmup_momentum": (0.0, 0.95),
            "box": (1.0, 20.0),
            "cls": (0.2, 4.0),
            "dfl": (0.4, 6.0),
            "hsv_h": (0.0, 0.1),
            "hsv_s": (0.0, 0.9),
            "hsv_v": (0.0, 0.9),
            "degrees": (0.0, 45.0),
            "translate": (0.0, 0.9),
            "scale": (0.0, 0.95),
            "shear": (0.0, 10.0),
            "perspective": (0.0, 0.001),
            "flipud": (0.0, 1.0),
            "fliplr": (0.0, 1.0),
            "bgr": (0.0, 1.0),
            "mosaic": (0.0, 1.0),
            "mixup": (0.0, 1.0),
            "cutmix": (0.0, 1.0),
            "copy_paste": (0.0, 1.0)
        }

        # Create UI elements for default search space
        self.space_widgets = {}
        for param_name, bounds in default_space.items():
            param_layout = QHBoxLayout()
            
            # Standard numeric parameters
            # Min value
            min_spinbox = QDoubleSpinBox()
            min_spinbox.setDecimals(2)
            min_spinbox.setSingleStep(0.01)
            min_spinbox.setMinimum(-1000000)
            min_spinbox.setMaximum(1000000)
            min_spinbox.setValue(bounds[0])
            param_layout.addWidget(min_spinbox)
            
            # Max value
            max_spinbox = QDoubleSpinBox()
            max_spinbox.setDecimals(2)
            max_spinbox.setSingleStep(0.01)
            max_spinbox.setMinimum(-1000000)
            max_spinbox.setMaximum(1000000)
            max_spinbox.setValue(bounds[1])
            param_layout.addWidget(max_spinbox)
            
            # Gain value (optional, third parameter)
            gain_spinbox = QDoubleSpinBox()
            gain_spinbox.setDecimals(2)
            gain_spinbox.setSingleStep(0.01)
            gain_spinbox.setMinimum(-1000000)
            gain_spinbox.setMaximum(1000000)
            if len(bounds) > 2:
                gain_spinbox.setValue(bounds[2])
            else:
                gain_spinbox.setValue(1.0)
            param_layout.addWidget(gain_spinbox)
            
            # Enabled checkbox
            enabled_checkbox = QCheckBox("Enabled")
            enabled_checkbox.setChecked(True)
            param_layout.addWidget(enabled_checkbox)
            
            self.space_widgets[param_name] = (min_spinbox, max_spinbox, gain_spinbox, enabled_checkbox)
        
            form_layout.addRow(f"{param_name}:", param_layout)

        # Add horizontal separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        form_layout.addRow("", separator)
        
        # Add custom space parameter button
        self.add_space_param_button = QPushButton("Add Space Parameter")
        self.add_space_param_button.clicked.connect(self.add_space_parameter_pair)
        form_layout.addRow("", self.add_space_param_button)

        # Add custom space parameters section
        self.custom_space_params_layout = QVBoxLayout()
        form_layout.addRow("", self.custom_space_params_layout)

        # Remove space parameter button
        self.remove_space_param_button = QPushButton("Remove Space Parameter")
        self.remove_space_param_button.clicked.connect(self.remove_space_parameter_pair)
        self.remove_space_param_button.setEnabled(False)
        form_layout.addRow("", self.remove_space_param_button)

        self.layout.addWidget(group_box)

    def add_space_parameter_pair(self):
        """
        Add a new search space parameter input group.
        """
        param_layout = QHBoxLayout()

        # Parameter name field
        param_name = QLineEdit()
        param_name.setPlaceholderText("Parameter name")

        # Min value
        min_spinbox = QDoubleSpinBox()
        min_spinbox.setDecimals(2)
        min_spinbox.setSingleStep(0.01)
        min_spinbox.setMinimum(-1000000)
        min_spinbox.setMaximum(1000000)
        min_spinbox.setValue(0.0)

        # Max value
        max_spinbox = QDoubleSpinBox()
        max_spinbox.setDecimals(2)
        max_spinbox.setSingleStep(0.01)
        max_spinbox.setMinimum(-1000000)
        max_spinbox.setMaximum(1000000)
        max_spinbox.setValue(1.0)

        # Gain value
        gain_spinbox = QDoubleSpinBox()
        gain_spinbox.setDecimals(2)
        gain_spinbox.setSingleStep(0.01)
        gain_spinbox.setMinimum(-1000000)
        gain_spinbox.setMaximum(1000000)
        gain_spinbox.setValue(1.0)

        # Add widgets to layout
        param_layout.addWidget(param_name)
        param_layout.addWidget(min_spinbox)
        param_layout.addWidget(max_spinbox)
        param_layout.addWidget(gain_spinbox)

        # Store the widgets for later retrieval
        self.custom_space_params.append((param_name, min_spinbox, max_spinbox, gain_spinbox))
        self.custom_space_params_layout.addLayout(param_layout)

        # Enable the remove button
        self.remove_space_param_button.setEnabled(True)

    def remove_space_parameter_pair(self):
        """
        Remove the most recently added search space parameter pair.
        """
        if not self.custom_space_params:
            return

        # Get the last parameter group
        self.custom_space_params.pop()

        # Remove the layout containing these widgets
        layout_to_remove = self.custom_space_params_layout.itemAt(self.custom_space_params_layout.count() - 1)

        if layout_to_remove:
            # Remove and delete widgets from the layout
            while layout_to_remove.count():
                widget = layout_to_remove.takeAt(0).widget()
                if widget:
                    widget.deleteLater()

            # Remove the layout itself
            self.custom_space_params_layout.removeItem(layout_to_remove)

        # Disable the remove button if no more parameters
        if not self.custom_space_params:
            self.remove_space_param_button.setEnabled(False)

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
        self.custom_params.pop()

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
        Set up the buttons layout.
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
            # Set the dataset path
            self.dataset_edit.setText(file_path)

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
        self.tune_model()
        super().accept()

    def get_parameters(self):
        """
        Get the tuning parameters from the dialog widgets.
    
        Returns:
            dict: A dictionary of tuning parameters.
        """
        # Extract values from dialog widgets
        params = {
            'task': self.task,
            'data': self.dataset_edit.text(),
            'iterations': self.iterations_spinbox.value(),
            'epochs': self.epochs_spinbox.value(),
            'batch': self.batch_spinbox.value(),
            'imgsz': self.imgsz_spinbox.value(),
            'multi_scale': self.multi_scale_combo.currentText().lower() == "true",
            'weighted': self.weighted_combo.currentText().lower() == "true",
            'optimizer': self.optimizer_combo.currentText(),
            'val': self.val_combo.currentText().lower() == "true",
            'workers': self.workers_spinbox.value(),
            'exist_ok': True,
        }
        
        # Get the current date and time for the project name
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d_%H-%M-%S")
        # Default project folder
        params['project'] = self.project_edit.text()
        params['project'] = params['project'] if params['project'] else "Data/Tuning"
        # Default project name
        params['name'] = self.name_edit.text()
        params['name'] = params['name'] if params['name'] else f"tune_{now}"
        # Combine project and name into a single parameter
        params['project'] = f"{params['project']}/{params['name']}"
        
        # Either the model path, or the model name provided from combo box
        params['model'] = self.model_edit.text() if self.model_edit.text() else self.model_combo.currentText()

        # Build search space
        space = {}
        
        # Add enabled default space parameters
        for param_name, widgets in self.space_widgets.items():
            # Handle numeric parameters
            min_spinbox, max_spinbox, gain_spinbox, enabled_checkbox = widgets
            if enabled_checkbox.isChecked():
                min_val = min_spinbox.value()
                max_val = max_spinbox.value()
                gain_val = gain_spinbox.value()
                
                if gain_val != 1.0:
                    space[param_name] = (min_val, max_val, gain_val)
                else:
                    space[param_name] = (min_val, max_val)

        # Add custom space parameters
        for param_info in self.custom_space_params:
            param_name_widget, min_spinbox, max_spinbox, gain_spinbox = param_info
            name = param_name_widget.text().strip()
            
            if name:
                min_val = min_spinbox.value()
                max_val = max_spinbox.value()
                gain_val = gain_spinbox.value()
                
                if gain_val != 1.0:
                    space[name] = (min_val, max_val, gain_val)
                else:
                    space[name] = (min_val, max_val)

        if space:
            params['space'] = space

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

    def tune_model(self):
        """
        Tune the model based on the provided parameters.
        """
        # Get tuning parameters
        self.params = self.get_parameters()

        # Create and start the worker thread
        self.worker = TuneModelWorker(self.params, self.main_window.device)
        self.worker.tuning_started.connect(self.on_tuning_started)
        self.worker.tuning_completed.connect(self.on_tuning_completed)
        self.worker.tuning_error.connect(self.on_tuning_error)
        self.worker.start()

    def on_tuning_started(self):
        """
        Handle the event when the tuning starts.
        """
        message = "Model hyperparameter tuning has commenced.\nMonitor the console for real-time progress."
        QMessageBox.information(self, "Model Tuning Status", message)

    def on_tuning_completed(self):
        """
        Handle the event when the tuning completes.
        """
        message = "Model hyperparameter tuning has successfully been completed."
        QMessageBox.information(self, "Model Tuning Status", message)

    def on_tuning_error(self, error_message):
        """
        Handle the event when an error occurs during tuning.

        Args:
            error_message (str): The error message.
        """
        QMessageBox.critical(self, "Error", error_message)
        print(error_message)