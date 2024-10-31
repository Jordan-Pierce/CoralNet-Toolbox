import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import gc

import torch

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QSpinBox, QSlider, QLabel, QHBoxLayout, QPushButton,
                             QComboBox, QMessageBox, QApplication, QLineEdit)

from torch.cuda import empty_cache
from autodistill.detection import CaptionOntology

from toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

from toolbox.QtProgressBar import ProgressBar
from toolbox.QtRangeSlider import QRangeSlider


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
        self.resize(300, 250)

        self.imgsz = 1024
        self.uncertainty_thresh = 0.25
        self.area_thresh_min = 0.01
        self.area_thresh_max = 0.75
        self.loaded_model = None
        self.model_name = None
        self.ontology = None

        # Main layout
        self.main_layout = QVBoxLayout(self)

        # Model selection dropdown
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["GroundingDINO"])

        self.main_layout.addWidget(self.model_dropdown)

        # Ontology mapping form
        self.ontology_layout = QVBoxLayout()
        self.ontology_pairs = []  # To keep track of added pairs

        # Add and remove buttons
        add_remove_layout = QHBoxLayout()
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_ontology_pair)
        add_remove_layout.addWidget(self.remove_button)
        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.add_ontology_pair)
        add_remove_layout.addWidget(self.add_button)
        self.ontology_layout.addLayout(add_remove_layout)

        self.main_layout.addLayout(self.ontology_layout)

        # Custom parameters section
        self.form_layout = QFormLayout()

        # Add resize image dropdown (True / False)
        self.resize_image_dropdown = QComboBox()
        self.resize_image_dropdown.addItems(["True", "False"])
        self.resize_image_dropdown.setCurrentIndex(0)
        self.form_layout.addRow("Resize Image:", self.resize_image_dropdown)

        # Add imgsz parameter
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(512, 4096)
        self.imgsz_spinbox.setSingleStep(1024)
        self.imgsz_spinbox.setValue(self.imgsz)
        self.form_layout.addRow("Image Size (imgsz):", self.imgsz_spinbox)

        # Set the threshold slider for area
        self.area_threshold_slider = QRangeSlider()
        self.area_threshold_slider.setRange(0, 100)
        self.area_threshold_slider.setValue((int(self.area_thresh_min * 100), int(self.area_thresh_max * 100)))
        self.area_threshold_slider.rangeChanged.connect(self.update_area_label)

        self.area_threshold_label = QLabel(f"{self.area_thresh_min:.2f} - {self.area_thresh_max:.2f}")
        self.form_layout.addRow("Area Threshold", self.area_threshold_slider)
        self.form_layout.addRow("", self.area_threshold_label)

        # Set the threshold slider for uncertainty
        self.uncertainty_threshold_slider = QSlider(Qt.Horizontal)
        self.uncertainty_threshold_slider.setRange(0, 100)
        self.uncertainty_threshold_slider.setValue(int(self.main_window.get_uncertainty_thresh() * 100))
        self.uncertainty_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.uncertainty_threshold_slider.setTickInterval(5)
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

    def update_area_label(self, min_val, max_val):
        # Update the area threshold values
        self.area_thresh_min = min_val / 100.0
        self.area_thresh_max = max_val / 100.0
        self.area_threshold_label.setText(f"Area Threshold: {self.area_thresh_min:.2f} - {self.area_thresh_max:.2f}")

    def get_area_thresh(self, image_path):
        h, w = self.main_window.image_window.rasterio_open(image_path).shape
        area_thresh_min = (h * w) * self.area_thresh_min
        area_thresh_max = (h * w) * self.area_thresh_max
        return area_thresh_min, area_thresh_max

    def update_uncertainty_label(self):
        # Convert the slider value to a ratio (0-1)
        value = self.uncertainty_threshold_slider.value() / 100.0
        self.main_window.update_uncertainty_thresh(value)
        self.uncertainty_threshold_label.setText(f"{value:.2f}")
        self.uncertainty_thresh = self.uncertainty_threshold_slider.value() / 100.0

    def on_uncertainty_changed(self, value):
        # Update the slider and label when the shared data changes
        self.uncertainty_threshold_slider.setValue(int(value * 100))
        self.uncertainty_threshold_label.setText(f"{value:.2f}")
        self.uncertainty_thresh = self.uncertainty_threshold_slider.value() / 100.0

    def update_label_options(self):
        label_options = [label.short_label_code for label in self.label_window.labels]
        for _, label_dropdown in self.ontology_pairs:
            previous_label = label_dropdown.currentText()
            label_dropdown.clear()
            label_dropdown.addItems(label_options)
            if previous_label in label_options:
                label_dropdown.setCurrentText(previous_label)

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

    def remove_ontology_pair(self):
        if len(self.ontology_pairs) > 1:
            pair = self.ontology_pairs.pop()
            pair[0].deleteLater()
            pair[1].deleteLater()

            # Remove the layout
            item = self.ontology_layout.itemAt(self.ontology_layout.count() - 2)
            item.layout().deleteLater()

    def load_model(self):
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        # Show a progress bar
        progress_bar = ProgressBar(self.annotation_window, title="Loading Model")
        progress_bar.show()
        try:
            # Get the ontology mapping
            ontology_mapping = self.get_ontology_mapping()

            # Set the ontology
            self.ontology = CaptionOntology(ontology_mapping)

            # Threshold for confidence
            uncertainty_thresh = self.get_uncertainty_threshold()

            # Get the name of the model to load
            model_name = self.model_dropdown.currentText()

            if model_name != self.model_name:
                self.load_new_model(model_name, uncertainty_thresh)
            else:
                # Update the model with the new ontology
                self.loaded_model.ontology = self.ontology

            self.status_bar.setText(f"Model loaded: {model_name}")
            QMessageBox.information(self, "Model Loaded", "Model loaded successfully")

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Model", str(e))

        # Stop the progress bar
        progress_bar.stop_progress()
        progress_bar.close()
        # Restore cursor
        QApplication.restoreOverrideCursor()
        # Exit the dialog box
        self.accept()

    def get_ontology_mapping(self):
        ontology_mapping = {}
        for text_input, label_dropdown in self.ontology_pairs:
            if text_input.text() != "":
                ontology_mapping[text_input.text()] = label_dropdown.currentText()
        return ontology_mapping

    def get_uncertainty_threshold(self):
        if self.main_window.get_uncertainty_thresh() < 0.10:
            return self.main_window.get_uncertainty_thresh()
        else:
            return 0.10  # Arbitrary value to prevent too many detections

    def load_new_model(self, model_name, uncertainty_thresh):
        if model_name == "GroundingDINO":
            from autodistill_grounding_dino import GroundingDINO
            self.model_name = model_name
            self.loaded_model = GroundingDINO(ontology=self.ontology,
                                              box_threshold=uncertainty_thresh,
                                              text_threshold=uncertainty_thresh)

    def predict(self, image_paths=None):
        if not self.loaded_model:
            QMessageBox.critical(self, "Error", "No model loaded")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)

        if not image_paths:
            image_paths = [self.annotation_window.current_image_path]

        for image_path in image_paths:
            # Predict the image, use NMS, process the results
            results = self.loaded_model.predict(image_path)
            # Perform NMS thresholding
            results = results.with_nms(self.main_window.get_iou_thresh())
            # Perform area thresholding
            min_area_thresh, max_area_thresh = self.get_area_thresh(image_path)
            results = results[results.area >= min_area_thresh]
            results = results[results.area <= max_area_thresh]
            # Process the results
            self.process_results(image_path, results)

        QApplication.restoreOverrideCursor()
        gc.collect()
        empty_cache()

    def process_results(self, image_path, results):
        progress_bar = ProgressBar(self, title=f"Making Predictions")
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
        self.model_name = None
        gc.collect()
        torch.cuda.empty_cache()
        self.main_window.untoggle_all_tools()
        self.status_bar.setText("No model loaded")
        QMessageBox.information(self, "Model Deactivated", "Model deactivated")
