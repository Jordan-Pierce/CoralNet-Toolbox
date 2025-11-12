import warnings

import gc
from torch.cuda import empty_cache

from ultralytics import YOLO

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog, QApplication, QMessageBox, QVBoxLayout,
    QLabel, QLineEdit, QDialog, QHBoxLayout, QPushButton,
    QComboBox, QFormLayout, QGroupBox, QScrollArea, QWidget,
    QSpinBox, QCheckBox, QDoubleSpinBox
)

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Worker Class
# ----------------------------------------------------------------------------------------------------------------------


class ExportModelWorker(QThread):
    """
    Worker thread for exporting a model.

    Signals:
        export_started: Emitted when the export starts.
        export_completed: Emitted when the export completes.
        export_error: Emitted when an error occurs during export.
    """
    export_started = pyqtSignal()
    export_completed = pyqtSignal()
    export_error = pyqtSignal(str)

    def __init__(self, model_path, params):
        """
        Initialize the ExportModelWorker.

        Args:
            model_path (str): Path to the .pt model file.
            params (dict): A dictionary of parameters for exporting.
        """
        super().__init__()
        self.model_path = model_path
        self.params = params

    def run(self):
        """
        Run the export process in a separate thread.
        """
        try:
            # Emit signal to indicate export has started
            self.export_started.emit()
            
            print(f"Starting model export with params: {self.params}")
            
            # Initialize the model
            model = YOLO(self.model_path)
            
            # Export the model
            model.export(**self.params)

            # Emit signal to indicate export has completed
            self.export_completed.emit()

        except Exception as e:
            self.export_error.emit(str(e))
        finally:
            # Clean up
            del model
            gc.collect()
            empty_cache()


# ----------------------------------------------------------------------------------------------------------------------
# Main Dialog Class
# ----------------------------------------------------------------------------------------------------------------------


class Base(QDialog):
    """
    A dialog for optimizing and exporting a YOLO model with specific parameters.
    Updated to mirror the structure of eval.py.
    """
    export_completed = pyqtSignal(str)

    def __init__(self, main_window, parent=None):
        """
        Initialize the dialog with the main window and optional parent widget.

        param main_window: The main window of the application.
        param parent: The parent widget of the dialog.
        """
        super().__init__(parent)
        self.main_window = main_window
        
        self.setWindowIcon(get_icon("coralnet.png"))
        self.setWindowTitle("Export Model")
        self.resize(500, 700)  # Increased size for new options

        self.params = {}
        self.worker = None

        self.layout = QVBoxLayout(self)
        
        # Setup the information layout
        self.setup_info_layout()
        # Setup the model layout
        self.setup_model_layout()
        # Setup the parameters layout
        self.setup_parameters_layout()
        # Setup the buttons layout
        self.setup_buttons_layout()

        # Connect signals
        self.int8_combo.currentTextChanged.connect(self.on_int8_changed)
        
    def setup_info_layout(self):
        """Setup information layout."""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        info_label = QLabel("Export the model to a production format (e.g., TensorRT, ONNX). "
                            "Details on formats and arguments can be found "
                            "<a href='https://docs.ultralytics.com/modes/export/#export-formats'>here</a>.")
        
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
    def setup_model_layout(self):
        """Setup model layout."""
        group_box = QGroupBox("Model and Format")
        layout = QFormLayout()

        # Model file selection
        model_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_browse_button = QPushButton("Browse...")
        self.model_browse_button.clicked.connect(self.browse_model_file)
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(self.model_browse_button)
        layout.addRow("Model Path:", model_layout)
        
        # Export Format Dropdown
        self.export_format_combo = QComboBox()
        # Common formats from export.md
        self.export_format_combo.addItems(["engine"])
        self.export_format_combo.setEditable(True)
        layout.addRow("Export Format:", self.export_format_combo)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
    
    def setup_parameters_layout(self):
        """Setup parameters layout with specific widgets, like eval.py."""
        
        # Create a widget and layout for the scroll area content
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # --- General Parameters ---
        general_groupbox = QGroupBox("General Parameters")
        general_layout = QFormLayout()

        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setMinimum(16)
        self.imgsz_spinbox.setMaximum(8192)
        self.imgsz_spinbox.setValue(640)
        self.imgsz_spinbox.setSingleStep(32)
        general_layout.addRow("Image Size (imgsz):", self.imgsz_spinbox)

        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setMinimum(1)
        self.batch_spinbox.setMaximum(1024)
        self.batch_spinbox.setValue(1)
        # Disable to make it stay as 1
        self.batch_spinbox.setEnabled(False)
        general_layout.addRow("Batch Size (batch):", self.batch_spinbox)
        
        self.device_edit = QLineEdit()
        self.device_edit.setPlaceholderText("e.g., 0 or cpu or dla:0")
        self.device_edit.setText("0")
        general_layout.addRow("Device (device):", self.device_edit)

        # --- Replaced Checkboxes with ComboBoxes ---
        
        self.half_combo = QComboBox()
        self.half_combo.addItems(["False", "True"])
        self.half_combo.setToolTip("Enable FP16 (half-precision) quantization")
        general_layout.addRow("FP16 (half):", self.half_combo)
        
        self.dynamic_combo = QComboBox()
        self.dynamic_combo.addItems(["False", "True"])
        self.dynamic_combo.setToolTip("Allow dynamic input sizes")
        general_layout.addRow("Dynamic Axes (dynamic):", self.dynamic_combo)
        
        self.simplify_combo = QComboBox()
        self.simplify_combo.addItems(["True", "False"])  # Default is True
        self.simplify_combo.setToolTip("Simplify the ONNX model graph")
        general_layout.addRow("Simplify (simplify):", self.simplify_combo)
        
        self.nms_combo = QComboBox()
        self.nms_combo.addItems(["False", "True"])
        self.nms_combo.setToolTip("Add Non-Maximum Suppression (NMS) to the model")
        general_layout.addRow("Add NMS (nms):", self.nms_combo)
        
        general_groupbox.setLayout(general_layout)
        scroll_layout.addWidget(general_groupbox)

        # --- TensorRT Parameters ---
        tensorrt_groupbox = QGroupBox("TensorRT Parameters")
        tensorrt_layout = QFormLayout()

        self.workspace_spinbox = QDoubleSpinBox()
        self.workspace_spinbox.setDecimals(1)
        self.workspace_spinbox.setMinimum(0.0)
        self.workspace_spinbox.setMaximum(64.0)
        self.workspace_spinbox.setValue(4.0)
        self.workspace_spinbox.setToolTip("GiB. Set to 0.0 for auto-allocation (default None).")
        tensorrt_layout.addRow("Workspace (GiB):", self.workspace_spinbox)
        
        tensorrt_groupbox.setLayout(tensorrt_layout)
        scroll_layout.addWidget(tensorrt_groupbox)

        # --- INT8 Quantization Parameters ---
        int8_layout = QHBoxLayout()
        int8_label = QLabel("Enable INT8 Quantization (int8):")
        int8_label.setToolTip("Activates INT8 quantization, requires calibration.")
        self.int8_combo = QComboBox()
        self.int8_combo.addItems(["False", "True"])
        int8_layout.addWidget(int8_label)
        int8_layout.addWidget(self.int8_combo)
        int8_layout.addStretch()
        scroll_layout.addLayout(int8_layout)

        self.int8_groupbox = QGroupBox("INT8 Calibration")
        int8_form_layout = QFormLayout()
        
        # Data file (YAML)
        data_layout = QHBoxLayout()
        self.data_edit = QLineEdit()
        self.data_edit.setPlaceholderText("Path to dataset.yaml")
        self.data_browse_button = QPushButton("Browse...")
        self.data_browse_button.clicked.connect(self.browse_data_yaml)
        data_layout.addWidget(self.data_edit)
        data_layout.addWidget(self.data_browse_button)
        int8_form_layout.addRow("Data (data):", data_layout)
        
        # Fraction
        self.fraction_spinbox = QDoubleSpinBox()
        self.fraction_spinbox.setDecimals(2)
        self.fraction_spinbox.setMinimum(0.01)
        self.fraction_spinbox.setMaximum(1.0)
        self.fraction_spinbox.setValue(1.0)
        self.fraction_spinbox.setSingleStep(0.1)
        self.fraction_spinbox.setToolTip("Fraction of dataset to use for calibration")
        int8_form_layout.addRow("Fraction (fraction):", self.fraction_spinbox)

        self.int8_groupbox.setLayout(int8_form_layout)
        self.int8_groupbox.setEnabled(False)  # Disabled by default
        scroll_layout.addWidget(self.int8_groupbox)
        
        # --- Create the scroll area ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_content)
        
        self.layout.addWidget(scroll_area)

    def setup_buttons_layout(self):
        """Setup buttons layout (like eval.py)."""
        self.button_layout = QHBoxLayout()
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.accept)
        
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.cancel_button)
        self.button_layout.addWidget(self.export_button)
        
        self.layout.addLayout(self.button_layout)

    def on_int8_changed(self, text):
        """Slot to enable/disable the INT8 groupbox."""
        self.int8_groupbox.setEnabled(text == "True")

    def browse_model_file(self):
        """
        Open a file dialog to select a model file and display its path.
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 
                                                   "Open Model File", 
                                                   "", 
                                                   "Model Files (*.pt)", 
                                                   options=options)
        if file_path:
            self.model_path_edit.setText(file_path)

    def browse_data_yaml(self):
        """Browse for the dataset YAML file (like eval.py)."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select YAML File", "", "YAML Files (*.yaml)")
        if file_path:
            self.data_edit.setText(file_path)

    def accept(self):
        """
        Override the accept method to export the model before closing.
        """
        self.export_model()
        super().accept()

    def get_export_parameters(self):
        """
        Extract and return the export parameters from the dialog widgets.
        """
        params = {'format': self.export_format_combo.currentText()}

        # General Params
        params['imgsz'] = self.imgsz_spinbox.value()
        params['batch'] = self.batch_spinbox.value()
        
        # --- Read values from ComboBoxes ---
        params['half'] = self.half_combo.currentText() == "True"
        params['dynamic'] = self.dynamic_combo.currentText() == "True"
        params['simplify'] = self.simplify_combo.currentText() == "True"
        params['nms'] = self.nms_combo.currentText() == "True"

        device_text = self.device_edit.text().strip()
        if device_text:
            params['device'] = device_text

        # TensorRT Params
        workspace_val = self.workspace_spinbox.value()
        if workspace_val > 0.0:
            params['workspace'] = workspace_val
        # If 0.0, we let Ultralytics use its default (None)

        # INT8 Params
        params['int8'] = self.int8_combo.currentText() == "True"
        if params['int8']:
            data_text = self.data_edit.text().strip()
            if data_text:
                params['data'] = data_text
            else:
                # INT8 requires data, show an error
                raise ValueError("INT8 quantization requires a 'data' YAML file.")
            params['fraction'] = self.fraction_spinbox.value()

        print(f"Export Parameters: {params}")
        return params

    def export_model(self):
        """
        Export the model using the worker thread.
        """
        try:
            self.params = self.get_export_parameters()
            model_path = self.model_path_edit.text()
            
            if not model_path:
                raise ValueError("Model Path must be specified.")

            # Set cursor to waiting
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # Start the worker thread
            self.worker = ExportModelWorker(model_path, self.params)
            self.worker.export_started.connect(self.on_export_started)
            self.worker.export_completed.connect(self.on_export_completed)
            self.worker.export_error.connect(self.on_export_error)
            self.worker.start()

        except Exception as e:
            # Restore cursor on error
            QApplication.restoreOverrideCursor()
            error_message = f"An error occurred when preparing to export model: {e}"
            QMessageBox.critical(self, "Error", error_message)
            print(error_message)

    # --- Worker Slots (like eval.py) ---
    
    def on_export_started(self):
        message = "Model export has started.\nThis may take a while, especially for INT8 calibration."
        "\nMonitor the console for real-time progress."
        QMessageBox.information(self, "Model Export Status", message)

    def on_export_completed(self):
        # Restore cursor
        QApplication.restoreOverrideCursor()
        message = "Model export has successfully completed."
        QMessageBox.information(self, "Model Export Status", message)
        
        # Compute exported path
        model_path = self.model_path_edit.text()
        format_ = self.params.get('format', 'engine')
        exported_path = model_path.replace('.pt', f'.{format_}')
        self.export_completed.emit(exported_path)

    def on_export_error(self, error_message):
        # Restore cursor
        QApplication.restoreOverrideCursor()
        QMessageBox.critical(self, "Export Error", error_message)
        print(error_message)