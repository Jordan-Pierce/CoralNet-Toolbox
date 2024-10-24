import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import datetime
import gc
import uuid
import yaml
import glob
import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby
from operator import attrgetter
from pathlib import Path

import numpy as np
import pandas as pd

import ultralytics.engine.validator as validator
import ultralytics.data.build as build
import ultralytics.models.yolo.classify.train as train_build

from PyQt5.QtGui import QBrush, QColor, QShowEvent
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF
from PyQt5.QtWidgets import (QFileDialog, QApplication, QScrollArea, QMessageBox, QCheckBox, QWidget, QVBoxLayout,
                             QLabel, QLineEdit, QDialog, QHBoxLayout, QTextEdit, QPushButton, QComboBox, QSpinBox,
                             QFormLayout, QTabWidget, QDialogButtonBox, QDoubleSpinBox, QGroupBox, QTableWidget,
                             QTableWidgetItem, QSlider, QButtonGroup, QRadioButton, QGridLayout)

from torch.cuda import empty_cache
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.dataset import ClassificationDataset

from toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

from toolbox.QtProgressBar import ProgressBar
from toolbox.utilities import pixmap_to_numpy


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class BatchInferenceDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window
        self.deploy_model_dialog = main_window.deploy_model_dialog

        self.loaded_models = self.deploy_model_dialog.loaded_models

        self.annotations = []
        self.prepared_patches = []
        self.image_paths = []

        self.setWindowTitle("Batch Inference")
        self.resize(400, 100)

        self.layout = QVBoxLayout(self)

        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)

        self.classification_tab = QWidget()
        self.detection_tab = QWidget()
        self.segmentation_tab = QWidget()

        self.tab_widget.addTab(self.classification_tab, "Image Classification")
        self.tab_widget.addTab(self.detection_tab, "Object Detection")
        self.tab_widget.addTab(self.segmentation_tab, "Instance Segmentation")

        # Initialize the tabs
        self.setup_classification_tab()
        self.setup_detection_tab()
        self.setup_segmentation_tab()

        # Set the threshold slider for uncertainty
        self.uncertainty_threshold_slider = QSlider(Qt.Horizontal)
        self.uncertainty_threshold_slider.setRange(0, 100)
        self.uncertainty_threshold_slider.setValue(int(self.main_window.get_uncertainty_thresh() * 100))
        self.uncertainty_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.uncertainty_threshold_slider.setTickInterval(10)
        self.uncertainty_threshold_slider.valueChanged.connect(self.update_uncertainty_label)

        self.uncertainty_threshold_label = QLabel(f"{self.main_window.get_uncertainty_thresh():.2f}")
        self.layout.addWidget(QLabel("Uncertainty Threshold"))
        self.layout.addWidget(self.uncertainty_threshold_slider)
        self.layout.addWidget(self.uncertainty_threshold_label)

        # Add the "Okay" and "Cancel" buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.on_ok_clicked)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

        self.setLayout(self.layout)

        # Connect to the shared data signal
        self.main_window.uncertaintyChanged.connect(self.on_uncertainty_changed)

    def update_uncertainty_label(self):
        # Convert the slider value to a ratio (0-1)
        value = self.uncertainty_threshold_slider.value() / 100.0
        self.main_window.update_uncertainty_thresh(value)

    def on_uncertainty_changed(self, value):
        # Update the slider and label when the shared data changes
        self.uncertainty_threshold_slider.setValue(int(value * 100))
        self.uncertainty_threshold_label.setText(f"{value:.2f}")

    def setup_classification_tab(self):
        layout = QVBoxLayout()

        # Create a group box for annotation options
        annotation_group_box = QGroupBox("Annotation Options")
        annotation_layout = QVBoxLayout()

        # Create a button group for the annotation checkboxes
        self.annotation_options_group = QButtonGroup(self)

        self.classification_review_checkbox = QCheckBox("Predict Review Annotation")
        self.classification_all_checkbox = QCheckBox("Predict All Annotations")

        # Add the checkboxes to the button group
        self.annotation_options_group.addButton(self.classification_review_checkbox)
        self.annotation_options_group.addButton(self.classification_all_checkbox)

        # Ensure only one checkbox can be checked at a time
        self.annotation_options_group.setExclusive(True)

        # Set the default checkbox
        self.classification_review_checkbox.setChecked(True)

        annotation_layout.addWidget(self.classification_review_checkbox)
        annotation_layout.addWidget(self.classification_all_checkbox)
        annotation_group_box.setLayout(annotation_layout)

        layout.addWidget(annotation_group_box)

        # Create a group box for image options
        image_group_box = QGroupBox("Image Options")
        image_layout = QVBoxLayout()

        # Create a button group for the image checkboxes
        self.image_options_group = QButtonGroup(self)

        self.apply_filtered_checkbox = QCheckBox("Apply to filtered images")
        self.apply_prev_checkbox = QCheckBox("Apply to previous images")
        self.apply_next_checkbox = QCheckBox("Apply to next images")
        self.apply_all_checkbox = QCheckBox("Apply to all images")

        # Add the checkboxes to the button group
        self.image_options_group.addButton(self.apply_filtered_checkbox)
        self.image_options_group.addButton(self.apply_prev_checkbox)
        self.image_options_group.addButton(self.apply_next_checkbox)
        self.image_options_group.addButton(self.apply_all_checkbox)

        # Ensure only one checkbox can be checked at a time
        self.image_options_group.setExclusive(True)

        # Set the default checkbox
        self.apply_all_checkbox.setChecked(True)

        image_layout.addWidget(self.apply_filtered_checkbox)
        image_layout.addWidget(self.apply_prev_checkbox)
        image_layout.addWidget(self.apply_next_checkbox)
        image_layout.addWidget(self.apply_all_checkbox)
        image_group_box.setLayout(image_layout)

        layout.addWidget(image_group_box)

        self.classification_tab.setLayout(layout)

    def setup_detection_tab(self):
        layout = QVBoxLayout()

        # Create a group box for image options
        image_group_box = QGroupBox("Image Options")
        image_layout = QVBoxLayout()

        # Create a button group for the image checkboxes
        self.detection_image_options_group = QButtonGroup(self)

        self.detection_apply_filtered_checkbox = QCheckBox("Apply to filtered images")
        self.detection_apply_prev_checkbox = QCheckBox("Apply to previous images")
        self.detection_apply_next_checkbox = QCheckBox("Apply to next images")
        self.detection_apply_all_checkbox = QCheckBox("Apply to all images")

        # Add the checkboxes to the button group
        self.detection_image_options_group.addButton(self.detection_apply_filtered_checkbox)
        self.detection_image_options_group.addButton(self.detection_apply_prev_checkbox)
        self.detection_image_options_group.addButton(self.detection_apply_next_checkbox)
        self.detection_image_options_group.addButton(self.detection_apply_all_checkbox)

        # Ensure only one checkbox can be checked at a time
        self.detection_image_options_group.setExclusive(True)

        # Set the default checkbox
        self.detection_apply_all_checkbox.setChecked(True)

        image_layout.addWidget(self.detection_apply_filtered_checkbox)
        image_layout.addWidget(self.detection_apply_prev_checkbox)
        image_layout.addWidget(self.detection_apply_next_checkbox)
        image_layout.addWidget(self.detection_apply_all_checkbox)
        image_group_box.setLayout(image_layout)

        layout.addWidget(image_group_box)

        self.detection_tab.setLayout(layout)

    def setup_segmentation_tab(self):
        layout = QVBoxLayout()

        # Create a group box for image options
        image_group_box = QGroupBox("Image Options")
        image_layout = QVBoxLayout()

        # Create a button group for the image checkboxes
        self.segmentation_image_options_group = QButtonGroup(self)

        self.segmentation_apply_filtered_checkbox = QCheckBox("Apply to filtered images")
        self.segmentation_apply_prev_checkbox = QCheckBox("Apply to previous images")
        self.segmentation_apply_next_checkbox = QCheckBox("Apply to next images")
        self.segmentation_apply_all_checkbox = QCheckBox("Apply to all images")

        # Add the checkboxes to the button group
        self.segmentation_image_options_group.addButton(self.segmentation_apply_filtered_checkbox)
        self.segmentation_image_options_group.addButton(self.segmentation_apply_prev_checkbox)
        self.segmentation_image_options_group.addButton(self.segmentation_apply_next_checkbox)
        self.segmentation_image_options_group.addButton(self.segmentation_apply_all_checkbox)

        # Ensure only one checkbox can be checked at a time
        self.segmentation_image_options_group.setExclusive(True)

        # Set the default checkbox
        self.segmentation_apply_all_checkbox.setChecked(True)

        image_layout.addWidget(self.segmentation_apply_filtered_checkbox)
        image_layout.addWidget(self.segmentation_apply_prev_checkbox)
        image_layout.addWidget(self.segmentation_apply_next_checkbox)
        image_layout.addWidget(self.segmentation_apply_all_checkbox)
        image_group_box.setLayout(image_layout)

        layout.addWidget(image_group_box)

        self.segmentation_tab.setLayout(layout)

    def on_ok_clicked(self):
        if self.classification_all_checkbox.isChecked():
            reply = QMessageBox.warning(self,
                                        "Warning",
                                        "This will overwrite the existing labels. Are you sure?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return  # Do not accept the dialog if the user clicks "No"

        self.apply()
        self.accept()  # Close the dialog after applying the changes

    def apply(self):
        # Pause the cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # Get the selected image paths based on the current tab
            if self.tab_widget.currentIndex() == 0:  # Classification
                self.apply_classification()
            elif self.tab_widget.currentIndex() == 1:  # Detection
                self.apply_detection()
            elif self.tab_widget.currentIndex() == 2:  # Segmentation
                self.apply_segmentation()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to make predictions: {str(e)}")
        finally:
            self.annotations = []
            self.prepared_patches = []
            self.image_paths = []

        # Resume the cursor
        QApplication.restoreOverrideCursor()

    def apply_classification(self):
        # Get the Review Annotations
        if self.classification_review_checkbox.isChecked():
            for image_path in self.get_selected_image_paths():
                self.annotations.extend(self.annotation_window.get_image_review_annotations(image_path))
        else:
            # Get all the annotations
            for image_path in self.get_selected_image_paths():
                self.annotations.extend(self.annotation_window.get_image_annotations(image_path))

        # Crop them, if not already cropped
        self.preprocess_patch_annotations()
        self.batch_inference('classify')

    def apply_detection(self):
        self.image_paths = self.get_selected_image_paths()
        self.batch_inference('detect')

    def apply_segmentation(self):
        self.image_paths = self.get_selected_image_paths()
        self.batch_inference('segment')

    def get_selected_image_paths(self):
        if self.tab_widget.currentIndex() == 0:  # Classification
            if self.apply_filtered_checkbox.isChecked():
                return self.image_window.filtered_image_paths
            elif self.apply_prev_checkbox.isChecked():
                current_image_index = self.image_window.image_paths.index(self.annotation_window.current_image_path)
                return self.image_window.image_paths[:current_image_index + 1]
            elif self.apply_next_checkbox.isChecked():
                current_image_index = self.image_window.image_paths.index(self.annotation_window.current_image_path)
                return self.image_window.image_paths[current_image_index:]
            else:
                return self.image_window.image_paths
        elif self.tab_widget.currentIndex() == 1:  # Detection
            if self.detection_apply_filtered_checkbox.isChecked():
                return self.image_window.filtered_image_paths
            elif self.detection_apply_prev_checkbox.isChecked():
                current_image_index = self.image_window.image_paths.index(self.annotation_window.current_image_path)
                return self.image_window.image_paths[:current_image_index + 1]
            elif self.detection_apply_next_checkbox.isChecked():
                current_image_index = self.image_window.image_paths.index(self.annotation_window.current_image_path)
                return self.image_window.image_paths[current_image_index:]
            else:
                return self.image_window.image_paths
        elif self.tab_widget.currentIndex() == 2:  # Segmentation
            if self.segmentation_apply_filtered_checkbox.isChecked():
                return self.image_window.filtered_image_paths
            elif self.segmentation_apply_prev_checkbox.isChecked():
                current_image_index = self.image_window.image_paths.index(self.annotation_window.current_image_path)
                return self.image_window.image_paths[:current_image_index + 1]
            elif self.segmentation_apply_next_checkbox.isChecked():
                current_image_index = self.image_window.image_paths.index(self.annotation_window.current_image_path)
                return self.image_window.image_paths[current_image_index:]
            else:
                return self.image_window.image_paths

    def preprocess_patch_annotations(self):
        # Get unique image paths
        self.image_paths = list(set(a.image_path for a in self.annotations))
        if not self.image_paths:
            return

        progress_bar = ProgressBar(self, title=f"Cropping Annotations")
        progress_bar.show()
        progress_bar.start_progress(len(self.image_paths))

        def crop(image_path, image_annotations):
            # Crop the image based on the annotations
            return self.annotation_window.crop_these_image_annotations(image_path, image_annotations)

        # Group annotations by image path
        groups = groupby(sorted(self.annotations, key=attrgetter('image_path')), key=attrgetter('image_path'))

        with ThreadPoolExecutor() as executor:
            future_to_image = {}
            for path, group in groups:
                future = executor.submit(crop, path, list(group))
                future_to_image[future] = path

            for future in as_completed(future_to_image):
                image_path = future_to_image[future]
                try:
                    self.prepared_patches.extend(future.result())
                except Exception as exc:
                    print(f'{image_path} generated an exception: {exc}')
                finally:
                    progress_bar.update_progress()

        progress_bar.stop_progress()
        progress_bar.close()

    def batch_inference(self, task):
        # Make predictions on each image's annotations
        progress_bar = ProgressBar(self, title=f"Batch Inference")
        progress_bar.show()
        progress_bar.start_progress(len(self.image_paths))

        if task == 'classify':
            if self.loaded_models['classify'] is None:
                QMessageBox.warning(self, "Warning", "No classification model loaded")
                return

            # Group annotations by image path
            groups = groupby(sorted(self.prepared_patches, key=attrgetter('image_path')), key=attrgetter('image_path'))

            # Make predictions on each image's annotations
            for path, patches in groups:
                self.deploy_model_dialog.predict_classification(annotations=list(patches))
                progress_bar.update_progress()

        elif task == 'detect':
            if self.loaded_models['detect'] is None:
                QMessageBox.warning(self, "Warning", "No detection model loaded")
                return

            self.deploy_model_dialog.predict_detection(image_paths=self.image_paths)

        elif task == 'segment':
            if self.loaded_models['segment'] is None:
                QMessageBox.warning(self, "Warning", "No segmentation model loaded")
                return

            self.deploy_model_dialog.predict_segmentation(image_paths=self.image_paths)

        progress_bar.stop_progress()
        progress_bar.close()