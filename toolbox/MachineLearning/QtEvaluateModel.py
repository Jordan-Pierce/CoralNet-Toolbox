import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import datetime
import gc
import os
from pathlib import Path

import ultralytics.engine.validator as validator

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QFileDialog, QMessageBox, QWidget, QVBoxLayout,
                             QLineEdit, QDialog, QHBoxLayout, QPushButton, QComboBox, QSpinBox,
                             QFormLayout, QTabWidget)

from torch.cuda import empty_cache
from ultralytics import YOLO

from toolbox.MachineLearning.ConfusionMatrix import ConfusionMatrixMetrics


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


class EvaluateModelDialog(QDialog):
    """
    Dialog for evaluating machine learning models for image classification, object detection, 
    and instance segmentation.

    :param main_window: MainWindow object
    :param parent: Parent widget
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        # For holding parameters
        self.params = {}
        self.class_mapping = {}

        self.setWindowTitle("Evaluate Model")

        # Set window settings
        self.setWindowFlags(Qt.Window |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint |
                            Qt.WindowTitleHint)

        self.resize(400, 100)

        # Main layout
        self.main_layout = QVBoxLayout()

        # Create and set up the tabs, parameters form, and console output
        self.setup_ui()

        # Set the main layout as the layout of the dialog
        self.setLayout(self.main_layout)

    def setup_ui(self):
        """
        Set up the user interface for the dialog.
        """
        # Create a QTabWidget for different model types
        self.tab_widget = QTabWidget()

        # Add tabs for different model types
        self.tab_widget.addTab(self.setup_image_classification_tab(), "Image Classification")
        self.tab_widget.addTab(self.setup_object_detection_tab(), "Object Detection")
        self.tab_widget.addTab(self.setup_instance_segmentation_tab(), "Instance Segmentation")

        self.main_layout.addWidget(self.tab_widget)

        # Add OK and Cancel buttons
        self.buttons = QPushButton("OK")
        self.buttons.clicked.connect(self.accept)
        self.main_layout.addWidget(self.buttons)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.main_layout.addWidget(self.cancel_button)

    def setup_image_classification_tab(self):
        """
        Set up the layout and widgets for the image classification tab.

        Returns:
            QWidget: The image classification tab widget.
        """
        tab = QWidget()
        layout = QVBoxLayout()

        # Parameters Form
        form_layout = QFormLayout()

        # Existing Model
        self.model_edit_image_classification = QLineEdit()
        self.model_button_image_classification = QPushButton("Browse...")
        self.model_button_image_classification.clicked.connect(self.browse_model_file)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_edit_image_classification)
        model_layout.addWidget(self.model_button_image_classification)
        form_layout.addRow("Existing Model:", model_layout)

        # Dataset Directory
        self.dataset_dir_edit = QLineEdit()
        self.dataset_dir_button = QPushButton("Browse...")
        self.dataset_dir_button.clicked.connect(self.browse_dataset_dir)
        dataset_dir_layout = QHBoxLayout()
        dataset_dir_layout.addWidget(self.dataset_dir_edit)
        dataset_dir_layout.addWidget(self.dataset_dir_button)
        form_layout.addRow("Dataset Directory:", dataset_dir_layout)

        # Split
        self.split_combo_image_classification = QComboBox()
        self.split_combo_image_classification.addItems(["train", "val", "test"])
        self.split_combo_image_classification.setCurrentText("test")
        form_layout.addRow("Split:", self.split_combo_image_classification)

        # Save Directory
        self.save_dir_edit_image_classification = QLineEdit()
        self.save_dir_button_image_classification = QPushButton("Browse...")
        self.save_dir_button_image_classification.clicked.connect(self.browse_save_dir)
        save_dir_layout = QHBoxLayout()
        save_dir_layout.addWidget(self.save_dir_edit_image_classification)
        save_dir_layout.addWidget(self.save_dir_button_image_classification)
        form_layout.addRow("Save Directory:", save_dir_layout)

        # Name
        self.name_edit_image_classification = QLineEdit()
        form_layout.addRow("Name:", self.name_edit_image_classification)

        # Imgsz
        self.imgsz_spinbox_image_classification = QSpinBox()
        self.imgsz_spinbox_image_classification.setMinimum(16)
        self.imgsz_spinbox_image_classification.setMaximum(4096)
        self.imgsz_spinbox_image_classification.setValue(256)
        form_layout.addRow("Image Size:", self.imgsz_spinbox_image_classification)

        layout.addLayout(form_layout)
        tab.setLayout(layout)
        return tab

    def setup_object_detection_tab(self):
        """
        Set up the layout and widgets for the object detection tab.

        Returns:
            QWidget: The object detection tab widget.
        """
        tab = QWidget()
        layout = QVBoxLayout()

        # Parameters Form
        form_layout = QFormLayout()

        # Existing Model
        self.model_edit_object_detection = QLineEdit()
        self.model_button_object_detection = QPushButton("Browse...")
        self.model_button_object_detection.clicked.connect(self.browse_model_file)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_edit_object_detection)
        model_layout.addWidget(self.model_button_object_detection)
        form_layout.addRow("Existing Model:", model_layout)

        # YAML File for Object Detection
        self.object_detection_yaml_edit = QLineEdit()
        self.object_detection_yaml_button = QPushButton("Browse...")
        self.object_detection_yaml_button.clicked.connect(self.browse_object_detection_yaml)
        object_detection_yaml_layout = QHBoxLayout()
        object_detection_yaml_layout.addWidget(self.object_detection_yaml_edit)
        object_detection_yaml_layout.addWidget(self.object_detection_yaml_button)
        form_layout.addRow("Data YAML File:", object_detection_yaml_layout)

        # Split
        self.split_combo_object_detection = QComboBox()
        self.split_combo_object_detection.addItems(["train", "val", "test"])
        self.split_combo_object_detection.setCurrentText("test")
        form_layout.addRow("Split:", self.split_combo_object_detection)

        # Save Directory
        self.save_dir_edit_object_detection = QLineEdit()
        self.save_dir_button_object_detection = QPushButton("Browse...")
        self.save_dir_button_object_detection.clicked.connect(self.browse_save_dir)
        save_dir_layout = QHBoxLayout()
        save_dir_layout.addWidget(self.save_dir_edit_object_detection)
        save_dir_layout.addWidget(self.save_dir_button_object_detection)
        form_layout.addRow("Save Directory:", save_dir_layout)

        # Name
        self.name_edit_object_detection = QLineEdit()
        form_layout.addRow("Name:", self.name_edit_object_detection)

        # Imgsz
        self.imgsz_spinbox_object_detection = QSpinBox()
        self.imgsz_spinbox_object_detection.setMinimum(16)
        self.imgsz_spinbox_object_detection.setMaximum(4096)
        self.imgsz_spinbox_object_detection.setValue(256)
        form_layout.addRow("Image Size:", self.imgsz_spinbox_object_detection)

        layout.addLayout(form_layout)
        tab.setLayout(layout)
        return tab

    def setup_instance_segmentation_tab(self):
        """
        Set up the layout and widgets for the instance segmentation tab.

        Returns:
            QWidget: The instance segmentation tab widget.
        """
        tab = QWidget()
        layout = QVBoxLayout()

        # Parameters Form
        form_layout = QFormLayout()

        # Existing Model
        self.model_edit_instance_segmentation = QLineEdit()
        self.model_button_instance_segmentation = QPushButton("Browse...")
        self.model_button_instance_segmentation.clicked.connect(self.browse_model_file)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_edit_instance_segmentation)
        model_layout.addWidget(self.model_button_instance_segmentation)
        form_layout.addRow("Existing Model:", model_layout)

        # YAML File for Instance Segmentation
        self.instance_segmentation_yaml_edit = QLineEdit()
        self.instance_segmentation_yaml_button = QPushButton("Browse...")
        self.instance_segmentation_yaml_button.clicked.connect(self.browse_instance_segmentation_yaml)
        instance_segmentation_yaml_layout = QHBoxLayout()
        instance_segmentation_yaml_layout.addWidget(self.instance_segmentation_yaml_edit)
        instance_segmentation_yaml_layout.addWidget(self.instance_segmentation_yaml_button)
        form_layout.addRow("Data YAML File:", instance_segmentation_yaml_layout)

        # Split
        self.split_combo_instance_segmentation = QComboBox()
        self.split_combo_instance_segmentation.addItems(["train", "val", "test"])
        self.split_combo_instance_segmentation.setCurrentText("test")
        form_layout.addRow("Split:", self.split_combo_instance_segmentation)

        # Save Directory
        self.save_dir_edit_instance_segmentation = QLineEdit()
        self.save_dir_button_instance_segmentation = QPushButton("Browse...")
        self.save_dir_button_instance_segmentation.clicked.connect(self.browse_save_dir)
        save_dir_layout = QHBoxLayout()
        save_dir_layout.addWidget(self.save_dir_edit_instance_segmentation)
        save_dir_layout.addWidget(self.save_dir_button_instance_segmentation)
        form_layout.addRow("Save Directory:", save_dir_layout)

        # Name
        self.name_edit_instance_segmentation = QLineEdit()
        form_layout.addRow("Name:", self.name_edit_instance_segmentation)

        # Imgsz
        self.imgsz_spinbox_instance_segmentation = QSpinBox()
        self.imgsz_spinbox_instance_segmentation.setMinimum(16)
        self.imgsz_spinbox_instance_segmentation.setMaximum(4096)
        self.imgsz_spinbox_instance_segmentation.setValue(256)
        form_layout.addRow("Image Size:", self.imgsz_spinbox_instance_segmentation)

        layout.addLayout(form_layout)
        tab.setLayout(layout)
        return tab

    def browse_model_file(self):
        """
        Browse and select a model file.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File")
        if file_path:
            if self.tab_widget.currentIndex() == 0:
                self.model_edit_image_classification.setText(file_path)
            elif self.tab_widget.currentIndex() == 1:
                self.model_edit_object_detection.setText(file_path)
            elif self.tab_widget.currentIndex() == 2:
                self.model_edit_instance_segmentation.setText(file_path)

    def browse_dataset_dir(self):
        """
        Browse and select a dataset directory.
        """
        dir_path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if dir_path:
            self.dataset_dir_edit.setText(dir_path)

    def browse_object_detection_yaml(self):
        """
        Browse and select a YAML file for object detection.
        """
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Select Object Detection YAML File",
                                                   "",
                                                   "YAML Files (*.yaml)")
        if file_path:
            self.object_detection_yaml_edit.setText(file_path)

    def browse_instance_segmentation_yaml(self):
        """
        Browse and select a YAML file for instance segmentation.
        """
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Select Instance Segmentation YAML File",
                                                   "",
                                                   "YAML Files (*.yaml)")
        if file_path:
            self.instance_segmentation_yaml_edit.setText(file_path)

    def browse_save_dir(self):
        """
        Browse and select a save directory.
        """
        dir_path = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if dir_path:
            if self.tab_widget.currentIndex() == 0:
                self.save_dir_edit_image_classification.setText(dir_path)
            elif self.tab_widget.currentIndex() == 1:
                self.save_dir_edit_object_detection.setText(dir_path)
            elif self.tab_widget.currentIndex() == 2:
                self.save_dir_edit_instance_segmentation.setText(dir_path)

    def accept(self):
        """
        Handle the OK button click event.
        """
        if self.tab_widget.currentIndex() == 0:
            # Image Classification
            if not self.dataset_dir_edit.text():
                QMessageBox.critical(self,
                                     "Error",
                                     "Dataset Directory field cannot be empty.")
                return
        elif self.tab_widget.currentIndex() == 1:
            # Object Detection
            if not self.object_detection_yaml_edit.text():
                QMessageBox.critical(self,
                                     "Error",
                                     "Object Detection YAML File field cannot be empty.")
                return
        elif self.tab_widget.currentIndex() == 2:
            # Instance Segmentation
            if not self.instance_segmentation_yaml_edit.text():
                QMessageBox.critical(self,
                                     "Error",
                                     "Instance Segmentation YAML File field cannot be empty.")
                return

        self.evaluate_model()
        super().accept()

    def get_evaluation_parameters(self):
        """
        Get the evaluation parameters from the dialog widgets.

        Returns:
            dict: A dictionary of evaluation parameters.
        """
        # Extract values from dialog widgets based on the current tab
        params = {
            'verbose': True,
            'exist_ok': True,
            'plots': True,
        }

        if self.tab_widget.currentIndex() == 0:
            # Image Classification
            params['model'] = self.model_edit_image_classification.text()
            params['data'] = self.dataset_dir_edit.text()
            params['task'] = 'classify'
            params['save_dir'] = self.save_dir_edit_image_classification.text()
            params['name'] = self.name_edit_image_classification.text()
            params['split'] = self.split_combo_image_classification.currentText()
            params['imgsz'] = int(self.imgsz_spinbox_image_classification.value())

        elif self.tab_widget.currentIndex() == 1:
            # Object Detection
            params['model'] = self.model_edit_object_detection.text()
            params['data'] = self.object_detection_yaml_edit.text()
            params['task'] = 'detect'
            params['save_dir'] = self.save_dir_edit_object_detection.text()
            params['name'] = self.name_edit_object_detection.text()
            params['split'] = self.split_combo_object_detection.currentText()
            params['imgsz'] = int(self.imgsz_spinbox_object_detection.value())

        elif self.tab_widget.currentIndex() == 2:
            # Instance Segmentation
            params['model'] = self.model_edit_instance_segmentation.text()
            params['data'] = self.instance_segmentation_yaml_edit.text()
            params['task'] = 'segment'
            params['save_dir'] = self.save_dir_edit_instance_segmentation.text()
            params['name'] = self.name_edit_instance_segmentation.text()
            params['split'] = self.split_combo_instance_segmentation.currentText()
            params['imgsz'] = int(self.imgsz_spinbox_instance_segmentation.value())

        # Default project name
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d_%H-%M-%S")
        params['name'] = params['name'] if params['name'] else now

        save_dir = params['save_dir']
        save_dir = Path(save_dir) / params['name']
        params['save_dir'] = save_dir

        # Return the dictionary of parameters
        return params

    def evaluate_model(self):
        """
        Evaluate the model based on the provided parameters.
        """
        # Get evaluation parameters
        self.params = self.get_evaluation_parameters()

        try:
            # Load the model and start the worker thread
            self.model = YOLO(self.params['model'], task=self.params['task'])

            # Create and start the worker thread
            self.worker = EvaluateModelWorker(self.model, self.params)
            self.worker.evaluation_started.connect(self.on_evaluation_started)
            self.worker.evaluation_completed.connect(self.on_evaluation_completed)
            self.worker.evaluation_error.connect(self.on_evaluation_error)
            self.worker.start()

            # Empty cache
            del self.model
            gc.collect()
            empty_cache()
        except Exception as e:
            error_message = f"An error occurred when evaluating model: {e}"
            QMessageBox.critical(self, "Error", error_message)
            print(error_message)

    def on_evaluation_started(self):
        """
        Handle the event when the evaluation starts.
        """
        message = "Model evaluation has commenced.\nMonitor the console for real-time progress."
        QMessageBox.information(self, "Model Evaluation Status", message)

    def on_evaluation_completed(self):
        """
        Handle the event when the evaluation completes.
        """
        message = "Model evaluation has successfully been completed."
        QMessageBox.information(self, "Model Evaluation Status", message)

    def on_evaluation_error(self, error_message):
        """
        Handle the event when an error occurs during evaluation.

        Args:
            error_message (str): The error message.
        """
        QMessageBox.critical(self, "Error", error_message)
        print(error_message)
