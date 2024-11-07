import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby
from operator import attrgetter

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox, QCheckBox, QWidget, QVBoxLayout,
                             QLabel, QDialog, QTabWidget, QDialogButtonBox, QGroupBox, QSlider, QButtonGroup)

from toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class BatchInferenceDialog(QDialog):
    """Dialog for performing batch inference on images using AutoDistill."""
    
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window
        self.deploy_model_dialog = main_window.auto_distill_deploy_model_dialog
        self.loaded_models = self.deploy_model_dialog.loaded_model
        
        self.setWindowTitle("Batch Inference")
        self.resize(300, 200)  # Width, height
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Create image selection group
        image_group = QGroupBox("Image Selection")
        image_layout = QVBoxLayout()
        
        # Create button group for image selection
        self.image_options_group = QButtonGroup(self)
        
        # Create image selection options
        self.apply_filtered = QCheckBox("Apply to filtered images")
        self.apply_prev = QCheckBox("Apply to previous images")
        self.apply_next = QCheckBox("Apply to next images") 
        self.apply_all = QCheckBox("Apply to all images")

        # Add options to button group
        self.image_options_group.addButton(self.apply_filtered)
        self.image_options_group.addButton(self.apply_prev)
        self.image_options_group.addButton(self.apply_next)
        self.image_options_group.addButton(self.apply_all)
        
        # Make selections exclusive
        self.image_options_group.setExclusive(True)
        
        # Default selection
        self.apply_all.setChecked(True)

        # Add widgets to layout
        image_layout.addWidget(self.apply_filtered)
        image_layout.addWidget(self.apply_prev)
        image_layout.addWidget(self.apply_next)
        image_layout.addWidget(self.apply_all)
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)

        # Add uncertainty threshold slider
        self.uncertainty_threshold_slider = QSlider(Qt.Horizontal)
        self.uncertainty_threshold_slider.setRange(0, 100)
        self.uncertainty_threshold_slider.setValue(int(self.main_window.get_uncertainty_thresh() * 100))
        self.uncertainty_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.uncertainty_threshold_slider.setTickInterval(10)
        self.uncertainty_threshold_slider.valueChanged.connect(self.update_uncertainty_label)

        self.uncertainty_threshold_label = QLabel(f"{self.main_window.get_uncertainty_thresh():.2f}")
        layout.addWidget(QLabel("Uncertainty Threshold"))
        layout.addWidget(self.uncertainty_threshold_slider)
        layout.addWidget(self.uncertainty_threshold_label)

        # Add OK/Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.apply)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def update_uncertainty_label(self):
        """Update uncertainty threshold label when slider changes."""
        value = self.uncertainty_threshold_slider.value() / 100.0
        self.main_window.update_uncertainty_thresh(value)
        self.uncertainty_threshold_label.setText(f"{value:.2f}")

    def get_selected_image_paths(self):
        """Get list of image paths based on selection."""
        current_path = self.annotation_window.current_image_path
        current_index = self.image_window.image_paths.index(current_path)
        
        if self.apply_filtered.isChecked():
            return self.image_window.filtered_image_paths
        elif self.apply_prev.isChecked():
            return self.image_window.image_paths[:current_index + 1]
        elif self.apply_next.isChecked():
            return self.image_window.image_paths[current_index:]
        else:
            return self.image_window.image_paths
        
    def apply(self):
        """Apply batch inference."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.deploy_model_dialog.predict(self.get_selected_image_paths())
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to make predictions: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()
            self.accept()
