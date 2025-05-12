import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import gc
import os

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox, QLabel, QGroupBox, QFormLayout,
                             QSlider)

from torch.cuda import empty_cache
from ultralytics import YOLO, RTDETR

from coralnet_toolbox.MachineLearning.DeployModel.QtBase import Base

from coralnet_toolbox.Results import ResultsProcessor

from coralnet_toolbox.utilities import pixmap_to_numpy
from coralnet_toolbox.utilities import check_model_architecture


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Classify(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Deploy Classification Model (Ctrl + 1)")

        self.task = 'classify'

    def showEvent(self, event):
        """
        Handle the show event to update label options and sync uncertainty threshold.

        Args:
            event: The event object.
        """
        super().showEvent(event)
        self.initialize_uncertainty_threshold()

    def setup_parameters_layout(self):
        """
        Setup parameter control section in a group box.
        """
        group_box = QGroupBox("Parameters")
        layout = QFormLayout()

        # Uncertainty threshold controls
        self.uncertainty_thresh = self.main_window.get_uncertainty_thresh()
        self.uncertainty_threshold_slider = QSlider(Qt.Horizontal)
        self.uncertainty_threshold_slider.setRange(0, 100)
        self.uncertainty_threshold_slider.setValue(int(self.main_window.get_uncertainty_thresh() * 100))
        self.uncertainty_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.uncertainty_threshold_slider.setTickInterval(10)
        self.uncertainty_threshold_slider.valueChanged.connect(self.update_uncertainty_label)
        self.uncertainty_threshold_label = QLabel(f"{self.uncertainty_thresh:.2f}")
        layout.addRow("Uncertainty Threshold", self.uncertainty_threshold_slider)
        layout.addRow("", self.uncertainty_threshold_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_sam_layout(self):
        pass

    def load_model(self):
        """
        Load the classification model.
        """
        if not self.model_path:
            QMessageBox.warning(self, "Warning", "Please select a model file first")
            return

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # Get the model architecture and task
            model_architecture, task = check_model_architecture(self.model_path)

            if not model_architecture:
                # If architecture can't be determined, ask user to choose
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Model Architecture Selection")
                msg_box.setText("Model architecture could not be determined (is it still training?)\n"
                                "Please select how to load this model:")
                yolo_button = msg_box.addButton("Load as YOLO", QMessageBox.ActionRole)
                cancel_button = msg_box.addButton(QMessageBox.Cancel)

                msg_box.exec_()

                if msg_box.clickedButton() == yolo_button:
                    model_architecture = "yolo"
                else:
                    QApplication.restoreOverrideCursor()
                    return

            # Check if the model is supported
            if model_architecture == "yolo":
                self.loaded_model = YOLO(self.model_path)
            else:
                raise ValueError(f"Unsupported model architecture: {model_architecture}")

            try:
                imgsz = self.loaded_model.__dict__['overrides']['imgsz']
            except:
                imgsz = 640

            self.loaded_model(np.zeros((imgsz, imgsz, 3), dtype=np.uint8))
            self.class_names = list(self.loaded_model.names.values())

            if not self.class_mapping:
                self.handle_missing_class_mapping()
            else:
                self.add_labels_to_label_window()

            # Display the class names
            self.check_and_display_class_names()

            # Update the status bar
            self.status_bar.setText(f"Model loaded: {os.path.basename(self.model_path)}")
            QMessageBox.information(self, "Model Loaded", "Model loaded successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

    def predict(self, inputs=None):
        """
        Predict the classification results for the given inputs.
        """
        if self.loaded_model is None:
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        if not inputs:
            # Predict only the selected annotation
            inputs = self.annotation_window.selected_annotations.copy()
            # Unselect the annotations (regardless)
            self.annotation_window.unselect_annotations()

        if not inputs:
            # If no annotations are selected, predict all annotations in the image
            inputs = self.annotation_window.get_image_review_annotations()

        if not inputs:
            # If no annotations are available, return
            QApplication.restoreOverrideCursor()
            return

        # Create lists to store valid images and their corresponding annotations
        images_np = []
        valid_inputs = []

        # Only add images and inputs that have valid cropped images
        for annotation in inputs:
            if hasattr(annotation, 'cropped_image') and annotation.cropped_image:
                try:
                    img = pixmap_to_numpy(annotation.cropped_image)
                    images_np.append(img)
                    valid_inputs.append(annotation)
                except Exception as e:
                    print(f"Error converting pixmap to numpy: {str(e)}")
                    continue

        # Only proceed if we have valid images to process
        if images_np:
            # Predict the classification results
            results = self.loaded_model(images_np,
                                        conf=self.main_window.get_uncertainty_thresh(),
                                        device=self.main_window.device,
                                        half=True,
                                        stream=True)

            # Create a result processor
            results_processor = ResultsProcessor(self.main_window,
                                                 self.class_mapping,
                                                 uncertainty_thresh=self.main_window.get_uncertainty_thresh())

            # Process the classification results using the valid inputs
            results_processor.process_classification_results(results, valid_inputs)

        # Make cursor normal
        QApplication.restoreOverrideCursor()
        gc.collect()
        empty_cache()
