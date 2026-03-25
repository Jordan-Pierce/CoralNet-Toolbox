import warnings

import os
import datetime
import traceback
from pathlib import Path

import lightly_train

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QGroupBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QFileDialog, QLabel, QAbstractItemView,
                             QFormLayout, QComboBox, QSpinBox, QDoubleSpinBox, 
                             QLineEdit, QWidget, QScrollArea, QFrame, QMessageBox)

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class PretrainModelWorker(QThread):
    """
    Worker thread for pre-training a YOLO encoder using the lightly_train API.

    Signals:
        training_started: Emitted when the training starts.
        training_completed: Emitted when the training completes.
        training_error: Emitted when an error occurs during training.
        training_status: Emitted with status message updates.
    """
    training_started = pyqtSignal()
    training_completed = pyqtSignal()
    training_error = pyqtSignal(str)
    training_status = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.output_dir = Path(self.params['project']) / self.params['name']

    def run(self):
        """Execute the LightlyTrain pretrain pipeline."""
        try:
            self.training_started.emit()
            self.training_status.emit("Parsing directories for images...")
            
            # Extract list of image strings
            dataset = UnlabeledMultiDirDataset(self.params['data_dirs'])
            image_list = dataset.get_image_paths()
            
            if not image_list:
                raise ValueError("No valid images found in the provided directories.")
            
            # Ensure the model string has the 'ultralytics/' prefix required by LightlyTrain
            model_name = self.params['model']
            if not model_name.startswith("ultralytics/"):
                model_name = f"ultralytics/{model_name}"

            # Map device string to LightlyTrain accelerator ('gpu', 'cpu', 'mps')
            accelerator = "auto"

            self.training_status.emit("Initializing LightlyTrain pre-training...")

            # Core arguments for lightly_train.pretrain
            pretrain_kwargs = {
                'out': str(self.output_dir),
                'model': model_name,
                'data': image_list,
                'method': self.params.get('ssl_method', 'dinov2').lower(),
                'epochs': self.params.get('epochs', 300),
                'batch_size': self.params.get('batch', 32),
                'accelerator': accelerator,
                'precision': self.params.get('precision', '16-mixed'),
                'num_workers': self.params.get('workers', 4),
                'overwrite': True,
            }
            
            # Handle resume/checkpoint logic
            resume_path = self.params.get('resume')
            if resume_path:
                if resume_path.endswith('.ckpt'):
                    # Resumes full training state (must be same 'out' directory)
                    pretrain_kwargs['resume_interrupted'] = True 
                else:
                    # Loads weights for a fresh training run
                    pretrain_kwargs['checkpoint'] = resume_path
            
            # Inject any custom UI parameters directly into kwargs
            for k, v in self.params.get('custom_kwargs', {}).items():
                pretrain_kwargs[k] = v

            # Start pre-training
            self.training_status.emit(
                f"Pre-training {model_name} for {pretrain_kwargs['epochs']} epochs. Check console for progress..."
            )
            
            # The highly abstracted API call
            lightly_train.pretrain(**pretrain_kwargs)
            
            self.training_status.emit("Pre-training completed successfully!")
            self.training_completed.emit()
            
        except Exception as e:
            err_msg = f"Error during pre-training: {e}\n\nTraceback:\n{traceback.format_exc()}"
            print(err_msg)
            self.training_error.emit(f"Error during pre-training: {e} (see console log)")


class UnlabeledMultiDirDataset:
    """
    Helper class that recursively loads unlabeled image paths from a list of directories.
    Provides a list of string paths to each image, which is passed directly to the 
    lightly_train.pretrain `data` argument.
    """
    def __init__(self, data_dirs):
        self.image_paths = []
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
        
        for directory in data_dirs:
            for root, _, files in os.walk(directory):
                for file in files:
                    if Path(file).suffix.lower() in valid_exts:
                        self.image_paths.append(os.path.join(root, file))

    def get_image_paths(self):
        """Returns the list of absolute image paths."""
        return self.image_paths
    

class Base(QDialog):
    """
    Dialog for pre-training machine learning models using self-supervised learning (SSL).
    This targets the encoder backbone using unlabeled image datasets via LightlyTrain.
    
    :param main_window: MainWindow object
    :param parent: Parent widget
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        self.setWindowIcon(get_icon("coralnet.svg"))
        self.setWindowTitle("Pre-train Encoder (Self-Supervised Learning)")
        self.resize(900, 600)  

        # Set window settings
        self.setWindowFlags(Qt.Window |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint |
                            Qt.WindowTitleHint)

        # For holding parameters and UI elements
        self.params = {}
        self.custom_params = []
        self.data_dirs = []  # List to track selected directory paths

        # Create the main layout
        self.layout = QVBoxLayout(self)

        # Build the UI sections
        self.setup_info_layout()
        
        # Create horizontal layout for data table (left) and parameters (right)
        content_layout = QHBoxLayout()
        
        # Left side: Data table
        self.data_group = self.create_data_layout()
        content_layout.addWidget(self.data_group, 1)  # stretch factor 1
        
        # Right side: Parameters in new order (Output, Architecture, Training)
        right_layout = QVBoxLayout()
        
        # Add Output Parameters first
        output_group = self.create_output_layout()
        right_layout.addWidget(output_group)
        
        # Add Architecture & Strategy
        model_ssl_group = self.create_model_ssl_layout()
        right_layout.addWidget(model_ssl_group)
        
        # Add Training Parameters
        params_group = self.create_parameters_layout()
        right_layout.addWidget(params_group)
        
        right_layout.addStretch()
        content_layout.addLayout(right_layout, 1)  # stretch factor 1
        
        self.layout.addLayout(content_layout)
        self.setup_buttons_layout()

    def setup_info_layout(self):
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()
        info_label = QLabel(
            "Pre-train an Ultralytics encoder using self-supervised learning (SSL) on unlabeled data. "
            "Add one or more directories containing raw images to build your dataset."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def create_data_layout(self):
        """Create and return the data section as a group box (not added to main layout)."""
        group_box = QGroupBox("Pre-training Data (Unlabeled Images)")
        layout = QVBoxLayout()

        # Buttons Layout
        buttons_layout = QHBoxLayout()
        self.add_folder_btn = QPushButton("Add Folder...")
        self.add_folder_btn.clicked.connect(self.add_data_folder)
        
        self.remove_folder_btn = QPushButton("Remove Selected")
        self.remove_folder_btn.clicked.connect(self.remove_data_folder)
        self.remove_folder_btn.setEnabled(False)  

        buttons_layout.addWidget(self.add_folder_btn)
        buttons_layout.addWidget(self.remove_folder_btn)
        buttons_layout.addStretch() 

        # Table Widget setup
        self.data_table = QTableWidget(0, 2)
        self.data_table.setHorizontalHeaderLabels(["Directory Path", "Image Count"])
        
        header = self.data_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        
        self.data_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.data_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.data_table.itemSelectionChanged.connect(self.on_table_selection_changed)

        self.total_images_label = QLabel("<b>Total Images for Pre-training: 0</b>")
        self.total_images_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addLayout(buttons_layout)
        layout.addWidget(self.data_table)
        layout.addWidget(self.total_images_label)

        group_box.setLayout(layout)
        return group_box

    def setup_data_layout(self):
        """Deprecated: kept for backward compatibility. Use create_data_layout() instead."""
        pass

    def create_model_ssl_layout(self):
        group_box = QGroupBox("Architecture & Strategy")
        layout = QFormLayout()

        # Model Selection
        self.model_combo = QComboBox()
        standard_models = [
            'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
            'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x'
        ]
        self.model_combo.addItems(standard_models)
        self.model_combo.setCurrentText('yolo11n')
        layout.addRow("Base Architecture:", self.model_combo)

        # Initialization
        self.weights_combo = QComboBox()
        self.weights_combo.addItems(["Initialize from Scratch (.yaml)", "Use Pre-trained Weights (.pt)"])
        layout.addRow("Initialization:", self.weights_combo)

        # SSL Method
        self.ssl_method_combo = QComboBox()
        self.ssl_method_combo.addItems(["DINOv2", "SimCLR", "MAE", "DINO", "BYOL", "MoCo", "Distillation"])
        self.ssl_method_combo.setCurrentText("SimCLR")
        layout.addRow("SSL Method:", self.ssl_method_combo)

        group_box.setLayout(layout)
        return group_box
    
    def setup_model_ssl_layout(self):
        """Deprecated: kept for backward compatibility. Use create_model_ssl_layout() instead."""
        pass

    def create_output_layout(self):
        group_box = QGroupBox("Output Parameters")
        form_layout = QFormLayout()

        self.project_edit = QLineEdit()
        self.project_button = QPushButton("Browse...")
        self.project_button.clicked.connect(self.browse_project_dir)
        project_layout = QHBoxLayout()
        project_layout.addWidget(self.project_edit)
        project_layout.addWidget(self.project_button)
        form_layout.addRow("Project:", project_layout)

        self.name_edit = QLineEdit()
        form_layout.addRow("Name:", self.name_edit)

        group_box.setLayout(form_layout)
        return group_box
    
    def setup_output_layout(self):
        """Deprecated: kept for backward compatibility. Use create_output_layout() instead."""
        pass

    def create_parameters_layout(self):
        group_box = QGroupBox("Training Parameters")
        group_layout = QVBoxLayout(group_box)

        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(form_widget)
        group_layout.addWidget(scroll_area)

        # Epochs
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setMinimum(1)
        self.epochs_spinbox.setMaximum(2000)
        self.epochs_spinbox.setValue(300)
        form_layout.addRow("Epochs:", self.epochs_spinbox)

        # Batch Size
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setMinimum(1)
        self.batch_spinbox.setMaximum(1024)
        self.batch_spinbox.setValue(32)
        form_layout.addRow("Batch Size:", self.batch_spinbox)

        # Precision
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["16-mixed", "32-true", "bf16-mixed"])
        self.precision_combo.setCurrentText("16-mixed")
        form_layout.addRow("Precision:", self.precision_combo)

        # Workers
        self.workers_spinbox = QSpinBox()
        self.workers_spinbox.setMinimum(0)
        self.workers_spinbox.setMaximum(64)
        self.workers_spinbox.setValue(4)
        form_layout.addRow("Dataloader Workers:", self.workers_spinbox)

        # Resume Checkpoint
        self.resume_edit = QLineEdit()
        self.resume_edit.setPlaceholderText("Optional path to .ckpt or .pt")
        self.resume_button = QPushButton("Browse...")
        self.resume_button.clicked.connect(self.browse_resume_checkpoint)
        resume_layout = QHBoxLayout()
        resume_layout.addWidget(self.resume_edit)
        resume_layout.addWidget(self.resume_button)
        form_layout.addRow("Resume/Checkpoint:", resume_layout)

        # Custom Parameters
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        form_layout.addRow("", separator)
        
        self.add_param_button = QPushButton("Add Custom Parameter")
        self.add_param_button.clicked.connect(self.add_parameter_pair)
        form_layout.addRow("", self.add_param_button)

        self.custom_params_layout = QVBoxLayout()
        form_layout.addRow("", self.custom_params_layout)

        self.remove_param_button = QPushButton("Remove Parameter")
        self.remove_param_button.clicked.connect(self.remove_parameter_pair)
        self.remove_param_button.setEnabled(False)
        form_layout.addRow("", self.remove_param_button)

        return group_box
    
    def setup_parameters_layout(self):
        """Deprecated: kept for backward compatibility. Use create_parameters_layout() instead."""
        pass

    def setup_buttons_layout(self):
        buttons_layout = QHBoxLayout()
        self.ok_button = QPushButton("Start Pre-training")
        self.ok_button.clicked.connect(self.accept)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.cancel_button)
        buttons_layout.addWidget(self.ok_button)
        self.layout.addLayout(buttons_layout)

    # --- Data Folder Logic ---
    def add_data_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if not dir_path or dir_path in self.data_dirs:
            return  

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
        image_count = 0
        
        if hasattr(self.main_window, 'statusBar'):
            self.main_window.statusBar().showMessage(f"Scanning {os.path.basename(dir_path)} for images...")
            
        for root, _, files in os.walk(dir_path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_count += 1
                    
        if hasattr(self.main_window, 'statusBar'):
            self.main_window.statusBar().clearMessage()
                    
        if image_count == 0:
            QMessageBox.warning(self, "No Images Found", f"No supported images found in {dir_path}")
            return

        self.data_dirs.append(dir_path)
        row_idx = self.data_table.rowCount()
        self.data_table.insertRow(row_idx)
        
        self.data_table.setItem(row_idx, 0, QTableWidgetItem(dir_path))
        
        count_item = QTableWidgetItem(f"{image_count:,}")
        count_item.setTextAlignment(Qt.AlignCenter)
        self.data_table.setItem(row_idx, 1, count_item)
        
        self.update_total_image_count()

    def remove_data_folder(self):
        selected_rows = sorted(set(item.row() for item in self.data_table.selectedItems()), reverse=True)
        for row in selected_rows:
            dir_path = self.data_table.item(row, 0).text()
            if dir_path in self.data_dirs:
                self.data_dirs.remove(dir_path)
            self.data_table.removeRow(row)
        self.update_total_image_count()

    def on_table_selection_changed(self):
        self.remove_folder_btn.setEnabled(len(self.data_table.selectedItems()) > 0)

    def update_total_image_count(self):
        total = sum(int(self.data_table.item(r, 1).text().replace(',', '')) for r in range(self.data_table.rowCount()))
        self.total_images_label.setText(f"<b>Total Images for Pre-training: {total:,}</b>")

    # --- File/Directory Browsers ---
    def browse_project_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if dir_path:
            self.project_edit.setText(dir_path)

    def browse_resume_checkpoint(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Checkpoint", "", "Checkpoints (*.ckpt *.pt)")
        if file_path:
            self.resume_edit.setText(file_path)

    # --- Custom Parameters ---
    def add_parameter_pair(self):
        param_layout = QHBoxLayout()
        param_name = QLineEdit()
        param_name.setPlaceholderText("Kwarg name")
        param_value = QLineEdit()
        param_value.setPlaceholderText("Value")
        param_type = QComboBox()
        param_type.addItems(["string", "int", "float", "bool"])

        param_layout.addWidget(param_name)
        param_layout.addWidget(param_value)
        param_layout.addWidget(param_type)

        self.custom_params.append((param_name, param_value, param_type))
        self.custom_params_layout.addLayout(param_layout)
        self.remove_param_button.setEnabled(True)

    def remove_parameter_pair(self):
        if not self.custom_params:
            return
        self.custom_params.pop()
        layout_to_remove = self.custom_params_layout.itemAt(self.custom_params_layout.count() - 1)
        if layout_to_remove:
            while layout_to_remove.count():
                widget = layout_to_remove.takeAt(0).widget()
                if widget:
                    widget.deleteLater()
            self.custom_params_layout.removeItem(layout_to_remove)
        if not self.custom_params:
            self.remove_param_button.setEnabled(False)

    # --- Execution Logic ---
    def get_parameters(self):
        use_pretrained = "Pre-trained" in self.weights_combo.currentText()
        base_ext = ".pt" if use_pretrained else ".yaml"
        model_name = f"{self.model_combo.currentText()}{base_ext}"

        params = {
            'data_dirs': self.data_dirs,
            'model': model_name,
            'ssl_method': self.ssl_method_combo.currentText(),
            'project': self.project_edit.text() or 'Data/Pretraining',
            'name': self.name_edit.text() or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            'epochs': self.epochs_spinbox.value(),
            'batch': self.batch_spinbox.value(),
            'precision': self.precision_combo.currentText(),
            'workers': self.workers_spinbox.value(),
            'resume': self.resume_edit.text() if self.resume_edit.text() else None,
            'custom_kwargs': {}
        }

        for param_name_widget, param_value_widget, param_type_widget in self.custom_params:
            name = param_name_widget.text().strip()
            value = param_value_widget.text().strip()
            type_name = param_type_widget.currentText()
            
            if name:
                if type_name == "bool":
                    params['custom_kwargs'][name] = value.lower() == "true"
                elif type_name == "int":
                    try: params['custom_kwargs'][name] = int(value)
                    except ValueError: params['custom_kwargs'][name] = value
                elif type_name == "float":
                    try: params['custom_kwargs'][name] = float(value)
                    except ValueError: params['custom_kwargs'][name] = value
                else:
                    params['custom_kwargs'][name] = value

        return params

    def accept(self):
        if not self.data_dirs:
            QMessageBox.warning(self, "Missing Data", "Please add at least one folder of images for pre-training.")
            return
        self.pretrain_model()
        super().accept()

    def pretrain_model(self):
        self.params = self.get_parameters()
        
        self.worker = PretrainModelWorker(self.params)
        
        self.worker.training_started.connect(self.on_training_started)
        self.worker.training_completed.connect(self.on_training_completed)
        self.worker.training_error.connect(self.on_training_error)
        self.worker.training_status.connect(self.on_training_status)
        
        self.worker.start()

    def on_training_started(self):
        QMessageBox.information(self, "Pre-training Initialized", 
                                "Lightly pre-training has commenced.\nMonitor the console for real-time progress.")
        
    def on_training_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
    
    def on_training_status(self, message):
        if hasattr(self.main_window, 'statusBar'):
            self.main_window.statusBar().showMessage(message, 5000)
            
    def on_training_completed(self):
        output_dir = Path(self.params['project']) / self.params['name']
        QMessageBox.information(
            self, 
            "Pre-training Completed", 
            f"Pre-training completed successfully."
        )