import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import datetime
import gc
from pathlib import Path

import ultralytics.engine.validator as validator

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QFileDialog, QMessageBox, QVBoxLayout,
                             QLineEdit, QDialog, QHBoxLayout, QPushButton,
                             QSpinBox, QFormLayout, QComboBox,
                             QGroupBox, QLabel)

from torch.cuda import empty_cache
from ultralytics import YOLO

from coralnet_toolbox.MachineLearning.ConfusionMatrix import ConfusionMatrixMetrics

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class EvaluateModelWorker(QThread):
    """
    Worker thread for evaluating a model.

    Signals:
        evaluation_started: Emitted when the evaluation starts.
        evaluation_completed: Emitted when the evaluation completes.
        evaluation_error: Emitted when an error occurs during evaluation.
    """
    evaluation_started = pyqtSignal()
    evaluation_completed = pyqtSignal()
    evaluation_error = pyqtSignal(str)

    def __init__(self, model, params):
        """
        Initialize the EvaluateModelWorker.

        Args:
            model: The model to be evaluated.
            params: A dictionary of parameters for evaluation.
        """
        super().__init__()
        self.model = model
        self.params = params

    def run(self):
        """
        Run the evaluation process in a separate thread.
        """
        try:
            # Emit signal to indicate evaluation has started
            self.evaluation_started.emit()

            # Modify the save directory
            save_dir = self.params['save_dir']
            validator.get_save_dir = lambda x: save_dir

            # Evaluate the model
            results = self.model.val(
                data=self.params['data'],
                imgsz=self.params['imgsz'],
                split=self.params['split'],
                save_json=True,
                plots=True
            )

            # Output confusion matrix metrics as json
            metrics = ConfusionMatrixMetrics(results, self.model.names)
            metrics.save_results(save_dir)

            # Emit signal to indicate evaluation has completed
            self.evaluation_completed.emit()

        except Exception as e:
            self.evaluation_error.emit(str(e))
            
            
class Base(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        
        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Evaluate Model")
        self.resize(400, 325)
        
        self.setWindowFlags(Qt.Window |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint |
                            Qt.WindowTitleHint)
        
        self.task = None
        self.params = {}
        self.class_mapping = {}
        
        self.layout = QVBoxLayout(self)
        
        # Setup the info layout
        self.setup_info_layout()
        # Setup the dataset layout
        self.setup_dataset_layout()
        # Setup the parameters layout
        self.setup_parameters_layout()
        # Setup the buttons layout
        self.setup_buttons_layout()
        
    def setup_info_layout(self):
        """Setup the info layout"""
        raise NotImplementedError("Subclasses must implement this method.")
        
    def setup_dataset_layout(self):
        """Setup the dataset layout."""
        raise NotImplementedError("Subclasses must implement this method.")
        
    def setup_parameters_layout(self):
        """Setup the layout."""
        group_box = QGroupBox("Parameters")
        layout = QFormLayout()
        
        self.split_combo = QComboBox()
        self.split_combo.addItems(["train", "val", "test"])
        self.split_combo.setCurrentText("test")
        layout.addRow("Split:", self.split_combo)
        
        self.save_dir_edit = QLineEdit()
        self.save_dir_button = QPushButton("Browse...")
        self.save_dir_button.clicked.connect(self.browse_save_dir)
        save_dir_layout = QHBoxLayout()
        save_dir_layout.addWidget(self.save_dir_edit)
        save_dir_layout.addWidget(self.save_dir_button)
        layout.addRow("Save Directory:", save_dir_layout)
        
        self.name_edit = QLineEdit()
        layout.addRow("Name:", self.name_edit)
        
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setMinimum(16)
        self.imgsz_spinbox.setMaximum(4096)
        self.imgsz_spinbox.setValue(256)
        layout.addRow("Image Size:", self.imgsz_spinbox)
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
    def setup_buttons_layout(self):
        """Setup the buttons layout."""        
        self.buttons = QPushButton("OK")
        self.buttons.clicked.connect(self.accept)
        self.layout.addWidget(self.buttons)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.layout.addWidget(self.cancel_button)

    def browse_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File")
        if file_path:
            self.model_edit.setText(file_path)

    def browse_dataset_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if dir_path:
            self.dataset_dir_edit.setText(dir_path)

    def browse_dataset_yaml(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select YAML File", "", "YAML Files (*.yaml)")
        if file_path:
            self.dataset_edit.setText(file_path)

    def browse_save_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if dir_path:
            self.save_dir_edit.setText(dir_path)

    def accept(self):
        self.evaluate_model()
        super().accept()

    def get_evaluation_parameters(self):
        params = {
            'verbose': True,
            'exist_ok': True,
            'plots': True,
        }
        params['task'] = self.task
        params['model'] = self.model_edit.text()
        params['data'] = self.dataset_edit.text()
        params['save_dir'] = self.save_dir_edit.text()
        params['name'] = self.name_edit.text()
        params['split'] = self.split_combo.currentText()
        params['imgsz'] = int(self.imgsz_spinbox.value())
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d_%H-%M-%S")
        params['name'] = params['name'] if params['name'] else now
        save_dir = params['save_dir']
        save_dir = Path(save_dir) / params['name']
        params['save_dir'] = save_dir
        return params

    def evaluate_model(self):
        self.params = self.get_evaluation_parameters()
        try:
            self.model = YOLO(self.params['model'], task=self.params['task'])
            self.worker = EvaluateModelWorker(self.model, self.params)
            self.worker.evaluation_started.connect(self.on_evaluation_started)
            self.worker.evaluation_completed.connect(self.on_evaluation_completed)
            self.worker.evaluation_error.connect(self.on_evaluation_error)
            self.worker.start()
            del self.model
            gc.collect()
            empty_cache()
        except Exception as e:
            error_message = f"An error occurred when evaluating model: {e}"
            QMessageBox.critical(self, "Error", error_message)
            print(error_message)

    def on_evaluation_started(self):
        message = "Model evaluation has commenced.\nMonitor the console for real-time progress."
        QMessageBox.information(self, "Model Evaluation Status", message)

    def on_evaluation_completed(self):
        message = "Model evaluation has successfully been completed."
        QMessageBox.information(self, "Model Evaluation Status", message)

    def on_evaluation_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        print(error_message)