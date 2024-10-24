import warnings

from toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import gc

import torch

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QSpinBox, QSlider, QLabel, QHBoxLayout, QPushButton,
                             QTabWidget, QComboBox, QMessageBox, QApplication, QWidget, QCheckBox, QLineEdit)

from torch.cuda import empty_cache
from ultralytics.utils import ops
from autodistill.detection import CaptionOntology

from toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class AutoDistillDeployModelDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

        self.setWindowTitle("AutoDistill Deploy Model")
        self.resize(300, 300)

        self.imgsz = 1024
        self.conf = 0.25
        self.model_path = None
        self.loaded_model = None
        self.ontology = None

        self.original_image = None
        self.resized_image = None

        # Main layout
        self.main_layout = QVBoxLayout(self)

        # Model selection dropdown
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItem("Grounding DINO")
        self.main_layout.addWidget(self.model_dropdown)

        # Ontology mapping form
        self.ontology_layout = QVBoxLayout()
        self.ontology_pairs = []  # To keep track of added pairs
        self.add_ontology_pair()  # Add the first pair

        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.add_ontology_pair)
        self.ontology_layout.addWidget(self.add_button)

        self.main_layout.addLayout(self.ontology_layout)

        # Custom parameters section
        self.form_layout = QFormLayout()

        # Add imgsz parameter
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(512, 4096)
        self.imgsz_spinbox.setSingleStep(1024)
        self.imgsz_spinbox.setValue(self.imgsz)
        self.form_layout.addRow("Image Size (imgsz):", self.imgsz_spinbox)

        # Add resize image checkbox
        self.resize_image_checkbox = QCheckBox("Resize Image")
        self.resize_image_checkbox.setChecked(True)
        self.form_layout.addRow("Resize Image:", self.resize_image_checkbox)

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

    def showEvent(self, event):
        super().showEvent(event)
        self.update_label_options()

    def update_uncertainty_label(self):
        # Convert the slider value to a ratio (0-1)
        value = self.uncertainty_threshold_slider.value() / 100.0
        self.main_window.update_uncertainty_thresh(value)
        self.uncertainty_threshold_label.setText(f"{value:.2f}")
        self.conf = self.uncertainty_threshold_slider.value() / 100.0

    def on_uncertainty_changed(self, value):
        # Update the slider and label when the shared data changes
        self.uncertainty_threshold_slider.setValue(int(value * 100))
        self.uncertainty_threshold_label.setText(f"{value:.2f}")
        self.conf = self.uncertainty_threshold_slider.value() / 100.0

    def update_label_options(self):
        label_options = [label.short_label_code for label in self.label_window.labels]
        for _, label_dropdown in self.ontology_pairs:
            label_dropdown.clear()
            label_dropdown.addItems(label_options)

    def add_ontology_pair(self):
        pair_layout = QHBoxLayout()

        text_input = QLineEdit()
        text_input.setMaxLength(100)  # Cap the width at 100 characters
        label_dropdown = QComboBox()
        label_dropdown.addItems([label.short_label_code for label in self.label_window.labels])

        pair_layout.addWidget(text_input)
        pair_layout.addWidget(label_dropdown)

        self.ontology_pairs.append((text_input, label_dropdown))
        self.ontology_layout.insertLayout(self.ontology_layout.count() - 1, pair_layout)

    def resize_image(self, image):
        # Get the current image size
        imgsz = self.imgsz_spinbox.value()
        # Determine the target shape based on the long side
        h, w = image.shape[:2]
        if h > w:
            target_shape = (imgsz, int(w * (imgsz / h)))
        else:
            target_shape = (int(h * (imgsz / w)), imgsz)

        # Use scale_image to resize and pad the image
        resized_image = ops.scale_image(image, target_shape)
        return resized_image

    def load_model(self):
        try:
            # Get the ontology mapping
            ontology_mapping = {}
            for text_input, label_dropdown in self.ontology_pairs:
                if text_input.text() != "":
                    ontology_mapping[text_input.text()] = label_dropdown.currentText()

            self.ontology = CaptionOntology(ontology_mapping)

            # Threshold for confidence
            if self.main_window.get_uncertainty_thresh() < 0.10:
                conf = self.main_window.get_uncertainty_thresh()
            else:
                conf = 0.10  # Arbitrary value to prevent too many detections

            # Get the name of the model to load
            model_name = self.model_dropdown.currentText()
            if model_name == "Grounding DINO":
                from autodistill_grounding_dino import GroundingDINO
                self.loaded_model = GroundingDINO(ontology=self.ontology,
                                                  box_threshold=conf,
                                                  text_threshold=conf)

            QMessageBox.information(self, "Model Loaded", "Model loaded successfully")

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Model", str(e))
            return

    def predict(self, image_paths=None):
        if not self.loaded_model:
            QMessageBox.critical(self, "Error", "No model loaded")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)

        if not image_paths:
            image_paths = [self.annotation_window.current_image_path]

        for image_path in image_paths:
            results = self.loaded_model.predict(image_path)
            self.process_results(image_path, results)

        QApplication.restoreOverrideCursor()
        gc.collect()
        empty_cache()

    def process_results(self, image_path, results):
        progress_bar = ProgressBar(self, title=f"Making Detection Predictions")
        progress_bar.show()
        progress_bar.start_progress(len(results))

        for result in results:
            try:
                x_min, y_min, x_max, y_max = map(float, result[0])
                mask = result[1]
                conf = result[2]
                cls = result[3]
                cls_name = self.ontology.classes()[cls]

                # Determine the short label
                short_label = 'Review'
                if conf > self.main_window.get_uncertainty_thresh():
                    short_label = cls_name

                # Prepare the annotation data
                label = self.label_window.get_label_by_short_code(short_label)
                top_left = QPointF(x_min, y_min)
                bottom_right = QPointF(x_max, y_max)

                # Create the rectangle annotation
                annotation = RectangleAnnotation(top_left,
                                                 bottom_right,
                                                 label.short_label_code,
                                                 label.long_label_code,
                                                 label.color,
                                                 image_path,
                                                 label.id,
                                                 self.main_window.get_transparency_value(),
                                                 show_msg=True)

                # Store the annotation and display the cropped image
                self.annotation_window.annotations_dict[annotation.id] = annotation

                # Connect update signals
                annotation.selected.connect(self.annotation_window.select_annotation)
                annotation.annotationDeleted.connect(self.annotation_window.delete_annotation)
                annotation.annotationUpdated.connect(self.main_window.confidence_window.display_cropped_image)

                # Add the prediction for the confidence window
                predictions = {self.label_window.get_label_by_short_code(cls_name): conf}
                annotation.update_machine_confidence(predictions)

                # Update label if confidence is below threshold
                if conf < self.main_window.get_uncertainty_thresh():
                    review_label = self.label_window.get_label_by_id('-1')
                    annotation.update_label(review_label)

                # Create the graphics and cropped image
                if image_path == self.annotation_window.current_image_path:
                    annotation.create_graphics_item(self.annotation_window.scene)
                    annotation.create_cropped_image(self.annotation_window.rasterio_image)
                    self.main_window.confidence_window.display_cropped_image(annotation)

                # Update the image annotations
                self.main_window.image_window.update_image_annotations(image_path)

                # Update the progress bar
                progress_bar.update_progress()

            except Exception as e:
                print(f"Warning: Failed to process detection result\n{e}")

        progress_bar.stop_progress()
        progress_bar.close()

    def deactivate_model(self):
        self.loaded_model = None
        self.model_path = None
        self.original_image = None
        self.resized_image = None
        gc.collect()
        torch.cuda.empty_cache()
        self.main_window.untoggle_all_tools()
        self.status_bar.setText("No model loaded")
        QMessageBox.information(self, "Model Deactivated", "Model deactivated")