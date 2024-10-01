import datetime
import gc
import json
import os
import random
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby
from operator import attrgetter
from pathlib import Path

import numpy as np

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF
from PyQt5.QtGui import QImage, QBrush, QColor, QShowEvent, QPixmap
from PyQt5.QtWidgets import (QFileDialog, QApplication, QScrollArea, QMessageBox, QCheckBox, QWidget, QVBoxLayout,
                             QLabel, QLineEdit, QDialog, QHBoxLayout, QTextEdit, QPushButton, QComboBox, QSpinBox,
                             QFormLayout, QTabWidget, QDialogButtonBox, QDoubleSpinBox, QGroupBox, QTableWidget,
                             QTableWidgetItem, QSlider, QButtonGroup)

from torch.cuda import empty_cache
from ultralytics.models.fastsam import FastSAMPredictor
from ultralytics.models.sam import Predictor as SAMPredictor

from toolbox.QtProgressBar import ProgressBar
from toolbox.utilities import pixmap_to_numpy

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SAMDeployModelDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window

        self.setWindowTitle("SAM Deploy Model")

        self.resize(300, 300)

        self.imgsz = 1024
        self.conf = 0.25
        self.model_path = None
        self.loaded_model = None

        # Main layout
        self.main_layout = QVBoxLayout(self)

        # Create and set up the tabs
        self.setup_tabs()

        # Custom parameters section
        self.form_layout = QFormLayout()

        # Add imgsz parameter
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(640, 2048)
        self.imgsz_spinbox.setValue(self.imgsz)
        self.form_layout.addRow("Image Size (imgsz):", self.imgsz_spinbox)

        # Set the threshold slider for uncertainty
        self.uncertainty_threshold_slider = QSlider(Qt.Horizontal)
        self.uncertainty_threshold_slider.setRange(0, 100)
        self.uncertainty_threshold_slider.setValue(int(self.main_window.get_uncertainty_thresh() * 100))
        self.uncertainty_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.uncertainty_threshold_slider.setTickInterval(10)
        self.uncertainty_threshold_slider.valueChanged.connect(self.update_uncertainty_label)

        self.uncertainty_threshold_label = QLabel(f"{self.main_window.get_uncertainty_thresh():.2f}")
        self.form_layout.addRow("Uncertainty Threshold", self.uncertainty_threshold_slider)
        self.form_layout.addRow("", self.uncertainty_threshold_label)

        # Load and Deactivate buttons
        button_layout = QHBoxLayout()
        load_button = QPushButton("Load Model")
        load_button.clicked.connect(self.load_model)
        button_layout.addWidget(load_button)

        deactivate_button = QPushButton("Deactivate Model")
        deactivate_button.clicked.connect(self.deactivate_model)
        button_layout.addWidget(deactivate_button)

        self.main_layout.addLayout(self.form_layout)
        self.main_layout.addLayout(button_layout)

        # Status bar label
        self.status_bar = QLabel("No model loaded")
        self.main_layout.addWidget(self.status_bar)

    def update_uncertainty_label(self):
        # Convert the slider value to a ratio (0-1)
        value = self.uncertainty_threshold_slider.value() / 100.0
        self.main_window.update_uncertainty_thresh(value)
        self.uncertainty_threshold_label.setText(f"{value:.2f}")

    def on_uncertainty_changed(self, value):
        # Update the slider and label when the shared data changes
        self.uncertainty_threshold_slider.setValue(int(value * 100))
        self.uncertainty_threshold_label.setText(f"{value:.2f}")

    def setup_tabs(self):
        self.tabs = QTabWidget()

        # Create tabs
        self.mobile_sam_tab = self.create_model_tab("MobileSAM")
        self.fast_sam_tab = self.create_model_tab("FastSAM")
        self.sam_tab = self.create_model_tab("SAM")
        self.sam2_tab = self.create_model_tab("SAM2")

        # Add tabs to the tab widget
        self.tabs.addTab(self.mobile_sam_tab, "MobileSAM")
        self.tabs.addTab(self.fast_sam_tab, "FastSAM")
        self.tabs.addTab(self.sam_tab, "SAM")
        self.tabs.addTab(self.sam2_tab, "SAM2")

        self.main_layout.addWidget(self.tabs)

    def create_model_tab(self, model_name):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        combo_box = QComboBox()
        combo_box.setEditable(True)

        # Define items for each model
        model_items = {
            "MobileSAM": ["mobile_sam.pt"],
            "FastSAM": ["FastSAM-s.pt", "FastSAM-x.pt"],
            "SAM": ["sam_b.pt", "sam_l.pt"],
            "SAM2": ["sam2_t.pt", "sam2_s.pt", "sam2_b.pt", "sam2_l.pt"]
        }

        # Add items to the combo box based on the model name
        if model_name in model_items:
            combo_box.addItems(model_items[model_name])

        layout.addWidget(QLabel(f"Select or Enter {model_name} Model:"))
        layout.addWidget(combo_box)
        return tab

    def get_parameters(self):
        # Get the parameters from the UI
        self.model_path = self.tabs.currentWidget().layout().itemAt(1).widget().currentText()
        self.imgsz = self.imgsz_spinbox.value()
        self.conf = self.uncertainty_threshold_slider.value() / 100.0

        parameters = {
            "model_path": self.model_path,
            "imgsz": self.imgsz,
            "conf": self.conf
        }

        return parameters

    def load_model(self):
        # Unpack the selected parameters
        parameters = self.get_parameters()

        try:
            # Make the cursor busy
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # Provide the necessary overrides for the model
            overrides = dict(conf=parameters["conf"],
                             task="segment",
                             mode="predict",
                             model=parameters["model_path"],
                             save=False,
                             imgsz=parameters["imgsz"])
            # Load the model
            if "fast" in parameters["model_path"].lower():
                self.loaded_model = FastSAMPredictor(overrides=overrides)
            else:
                self.loaded_model = SAMPredictor(overrides=overrides)

            # Set the current view
            self.set_image()

            QApplication.restoreOverrideCursor()
            self.status_bar.setText(f"Model loaded")
            QMessageBox.information(self, "Model Loaded", f"Model loaded successfully")

            # Exit the dialog box
            self.accept()

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error Loading Model", f"Error loading model: {e}")

    def deactivate_model(self):
        self.loaded_model = None
        self.model_path = None
        gc.collect()
        empty_cache()
        self.status_bar.setText("No model loaded")
        QMessageBox.information(self, "Model Deactivated", "Model deactivated")

    def set_image(self):
        if self.loaded_model is not None:
            # Get the current view of AnnotationWindow
            view = self.annotation_window.viewport()
            # Get the scene's visible area in the viewport
            scene_rect = self.annotation_window.mapToScene(view.rect()).boundingRect()
            # Convert the scene rect to a QRect in the viewport's coordinate system
            view_rect = self.annotation_window.mapFromScene(scene_rect).boundingRect()
            # Capture the current view of AnnotationWindow as a QImage
            qimage = self.annotation_window.grab(view_rect).toImage()
            # Convert QImage to QPixmap
            qpixmap = QPixmap.fromImage(qimage)
            # Convert QPixmap to numpy array
            image = pixmap_to_numpy(qpixmap)

            # Set the image in the model
            self.loaded_model.set_image(image)
        else:
            QMessageBox.critical(self, "Model Not Loaded", "Model not loaded")

    def predict(self, bboxes, positive, negative):
        if not self.loaded_model:
            QMessageBox.critical(self, "Model Not Loaded", "Model not loaded")
            return None
        try:
            # Combine the positive and negative points
            points = np.stack(positive + negative)
            labels = [1] * len(positive) + [0] * len(negative)
            # Make the prediction
            points = positive[0]
            labels = [labels[0]]
            results = self.loaded_model(points=points, labels=labels)[0]
            # Convert the results to a list of QPointF points
            polygon_points = results.masks.xy[0].tolist()
            polygon_points = [QPointF(*point) for point in polygon_points]
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"Error predicting: {e}")
            return None

        return polygon_points


class SAMBatchInferenceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAM Batch Inference")
        # Add additional initialization code here