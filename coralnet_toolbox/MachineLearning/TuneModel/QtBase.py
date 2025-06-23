import warnings

import gc
import traceback

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QFileDialog, QScrollArea, QMessageBox, QCheckBox, QWidget, QVBoxLayout,
                             QLabel, QLineEdit, QDialog, QHBoxLayout, QPushButton, QComboBox, QSpinBox,
                             QFormLayout, QTabWidget, QDoubleSpinBox, QGroupBox, QFrame)

from torch.cuda import empty_cache

from coralnet_toolbox.MachineLearning.TuneModel.tuner import Tuner, DEFAULT_SPACE

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

    def __init__(self, params):
        """
        Initialize the TuneModelWorker.

        Args:
            params: A dictionary of parameters for tuning.
            device: The device to use for tuning (e.g., 'cpu' or 'cuda').
        """
        super().__init__()
        self.params = params
        self.tuner = None

    def pre_run(self):
        """
        Set up the model and prepare parameters for tuning.
        """
        try:
            # Prepare tuner arguments following the same pattern as model.tune()
            self.iterations = self.params.pop('iterations', None)  # Remove iterations, will pass separately
            self.patience = self.params.pop('patience', None)
            
            # Add mode parameter as done in model.tune()
            self.params['mode'] = 'train'
            
            # Create the custom tuner with the prepared arguments
            self.tuner = Tuner(args=self.params)

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

            # The model is already set up in the tuner through the args
            results = self.tuner(iterations=self.iterations, patience=self.patience)

            # Store results for potential future use
            self.tuning_results = results

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
        pass

    def _cleanup(self):
        """
        Clean up resources after tuning.
        """
        if self.tuner:
            del self.tuner
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
        self.setWindowFlags(Qt.Window
                            | Qt.WindowCloseButtonHint
                            | Qt.WindowMinimizeButtonHint
                            | Qt.WindowMaximizeButtonHint
                            | Qt.WindowTitleHint)

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
        info_label = QLabel(
            "Hyperparameter tuning systematically searches for optimal hyperparameters. "
            "Details on tuning can be found "
            "<a href='https://docs.ultralytics.com/guides/hyperparameter-tuning/'>here</a>, "
            "and the original script can be found "
            "<a href='https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/tuner.py'>here</a>."
        )

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
        
        # Iterations patience
        self.patience_spinbox = QSpinBox()
        self.patience_spinbox.setValue(50)
        self.patience_spinbox.setMinimum(3)
        form_layout.addRow("Patience:", self.patience_spinbox)

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

        # Batch
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setMinimum(1)
        self.batch_spinbox.setMaximum(1024)
        self.batch_spinbox.setValue(self.batch)
        form_layout.addRow("Batch Size:", self.batch_spinbox)
        
        # Optimizer
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"])
        self.optimizer_combo.setCurrentText("SGD")
        form_layout.addRow("Optimizer:", self.optimizer_combo)
        
        # Validation
        self.val_combo = QComboBox()
        self.val_combo.addItems(["True", "False"])
        self.val_combo.setCurrentText("False")
        form_layout.addRow("Validation:", self.val_combo)
        
        # Save 
        self.save_combo = QComboBox()
        self.save_combo.addItems(["True", "False"])
        self.save_combo.setCurrentText("False")
        form_layout.addRow("Save Checkpoints:", self.save_combo)

        # Workers
        self.workers_spinbox = QSpinBox()
        self.workers_spinbox.setMinimum(0)
        self.workers_spinbox.setMaximum(64)
        self.workers_spinbox.setValue(0)
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

        # Add section header for Min and Max
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel(""))
        header_layout.addWidget(QLabel("Min"))
        header_layout.addWidget(QLabel("Max"))
        header_layout.addWidget(QLabel(""))
        form_layout.addRow(header_layout)

        # Use imported default search space parameters from tuner.py
        # Create UI elements for all parameters (continuous and what was previously boolean)
        self.space_widgets = {}
        for param_name, bounds in DEFAULT_SPACE.items():
            param_layout = QHBoxLayout()
            
            # Min value
            min_spinbox = QDoubleSpinBox()
            min_spinbox.setDecimals(6)
            min_spinbox.setSingleStep(0.01)
            min_spinbox.setMinimum(-1000000)
            min_spinbox.setMaximum(1000000)
            min_spinbox.setValue(bounds[0])
            param_layout.addWidget(min_spinbox)
            
            # Max value
            max_spinbox = QDoubleSpinBox()
            max_spinbox.setDecimals(6)
            max_spinbox.setSingleStep(0.01)
            max_spinbox.setMinimum(-1000000)
            max_spinbox.setMaximum(1000000)
            max_spinbox.setValue(bounds[1])
            param_layout.addWidget(max_spinbox)
            
            # Enabled checkbox (default to checked for all parameters)
            enabled_checkbox = QCheckBox()
            enabled_checkbox.setChecked(True)  # Enable all by default
            param_layout.addWidget(enabled_checkbox)
            
            # Store widgets without gain spinbox
            self.space_widgets[param_name] = (min_spinbox, max_spinbox, enabled_checkbox)
            # Add the row to the form layout with the parameter name as label
            form_layout.addRow(param_name, param_layout)
        
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
            'patience': self.patience_spinbox.value(),
            'epochs': self.epochs_spinbox.value(),
            'batch': self.batch_spinbox.value(),
            'imgsz': self.imgsz_spinbox.value(),
            'optimizer': self.optimizer_combo.currentText(),
            'val': self.val_combo.currentText().lower() == "true",
            'workers': self.workers_spinbox.value(),
            'exist_ok': True,
        }
        
        # Handle project and name parameters properly
        project_path = self.project_edit.text().strip()
        name = self.name_edit.text().strip()
        
        # Set defaults if empty
        if not project_path:
            project_path = "Data/Tuning"
        
        if not name:
            name = ""
        
        # Use Ultralytics standard project/name structure
        params['project'] = f"{project_path}/{name}"
                
        # Either the model path, or the model name provided from combo box
        params['model'] = self.model_edit.text() if self.model_edit.text() else self.model_combo.currentText()

        # Build search space - ALWAYS include it, even if empty
        space = {}
        
        # Add enabled default space parameters
        for param_name, widgets in self.space_widgets.items():
            min_spinbox, max_spinbox, enabled_checkbox = widgets
            if enabled_checkbox.isChecked():
                min_val = min_spinbox.value()
                max_val = max_spinbox.value()
                
                # Store as simple tuple (min, max) since gain is no longer used
                space[param_name] = (min_val, max_val)

        # Always add space parameter to params
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
                    
        params['device'] = self.main_window.device  # Add device parameter

        # Return the dictionary of parameters
        return params

    def tune_model(self):
        """
        Tune the model based on the provided parameters.
        """
        # Get tuning parameters
        self.params = self.get_parameters()

        # Create and start the worker thread
        self.worker = TuneModelWorker(self.params)
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