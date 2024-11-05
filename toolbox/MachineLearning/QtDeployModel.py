import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import gc
import json
import os
import random

import numpy as np

from PyQt5.QtGui import QColor, QShowEvent
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QWidget, QVBoxLayout,
                             QLabel, QDialog, QTextEdit, QPushButton, QTabWidget, QCheckBox)

from torch.cuda import empty_cache
from ultralytics import YOLO

from toolbox.ResultsProcessor import ResultsProcessor

from toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

from toolbox.QtProgressBar import ProgressBar
from toolbox.utilities import pixmap_to_numpy


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DeployModelDialog(QDialog):
    """
    Dialog for deploying machine learning models for image classification, object detection, 
    and instance segmentation.
    
    :param main_window: MainWindow object
    :param parent: Parent widget
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window
        self.sam_dialog = None

        self.setWindowTitle("Deploy Model")
        self.resize(400, 300)

        self.layout = QVBoxLayout(self)
        self.model_paths = {'classify': None, 'detect': None, 'segment': None}
        self.loaded_models = {'classify': None, 'detect': None, 'segment': None}
        self.class_mappings = {'classify': None, 'detect': None, 'segment': None}
        self.use_sam = {'classify': None, 'detect': None, 'segment': None}

        self.setup_tabs()
        self.setup_status_bars()

        self.tab_widget.currentChanged.connect(self.update_status_bar_visibility)
        self.setLayout(self.layout)

    def showEvent(self, event: QShowEvent):
        """
        Handle the show event to check and display class names and update status bar visibility.
        
        :param event: QShowEvent object
        """
        super().showEvent(event)
        self.check_and_display_class_names()
        self.update_status_bar_visibility()

    def setup_tabs(self):
        """
        Set up the tabs for different tasks (classification, detection, segmentation).
        """
        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)

        tasks = [("Image Classification", "classify"),
                 ("Object Detection", "detect"),
                 ("Instance Segmentation", "segment")]

        for label, task in tasks:
            tab = QWidget()
            text_area = self.setup_tab(tab, task)
            self.tab_widget.addTab(tab, label)
            setattr(self, f"{task}_text_area", text_area)

    def setup_tab(self, tab, task):
        """
        Set up a single tab with text area and buttons for the given task.
        
        :param tab: QWidget object
        :param task: Task identifier as a string
        :return: QTextEdit widget
        """
        layout = QVBoxLayout()
        text_area = QTextEdit()
        text_area.setReadOnly(True)
        layout.addWidget(text_area)

        buttons = [
            ("Browse Model", lambda: self.browse_file(task)),
            ("Browse Class Mapping", lambda: self.browse_class_mapping_file(task)),
            ("Load Model", lambda: self.load_model(task)),
            ("Deactivate Model", lambda: self.deactivate_model(task))
        ]

        for btn_text, callback in buttons:
            button = QPushButton(btn_text)
            button.clicked.connect(callback)
            layout.addWidget(button)

        use_sam_checkbox = QCheckBox("Use SAM for creating Polygons")
        use_sam_checkbox.stateChanged.connect(self.is_sam_model_deployed)
        use_sam_checkbox.setEnabled(task != 'classify')
        layout.addWidget(use_sam_checkbox)
        self.use_sam[task] = use_sam_checkbox

        tab.setLayout(layout)
        return text_area

    def setup_status_bars(self):
        """
        Set up status bars for each task.
        """
        self.status_bars = {
            'classify': QLabel("No model loaded"),
            'detect': QLabel("No model loaded"),
            'segment': QLabel("No model loaded")
        }
        for bar in self.status_bars.values():
            self.layout.addWidget(bar)

    def update_status_bar_visibility(self):
        """
        Update the visibility of status bars based on the selected tab.
        
        :param index: Index of the selected tab
        """
        current_task = self.get_current_task()
        for task, status_bar in self.status_bars.items():
            status_bar.setVisible(task == current_task)

    def browse_file(self, task):
        """
        Browse and select a model file for the given task.
        
        :param task: Task identifier as a string
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Open Model File", "",
                                                   "Model Files (*.pt *.onnx *.torchscript *.engine *.bin)",
                                                   options=options)
        if file_path:
            if ".bin" in file_path:
                # OpenVINO is a directory
                file_path = os.path.dirname(file_path)

            self.model_paths[task] = file_path
            self.get_text_area(task).setText("Model file selected")

            # Try to load the class mapping file if it exists in the directory above
            parent_dir = os.path.dirname(os.path.dirname(file_path))
            class_mapping_path = os.path.join(parent_dir, "class_mapping.json")
            if os.path.exists(class_mapping_path):
                self.load_class_mapping(task, class_mapping_path)

    def browse_class_mapping_file(self, task):
        """
        Browse and select a class mapping file for the given task.
        
        :param task: Task identifier as a string
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Open Class Mapping File", "",
                                                   "JSON Files (*.json)",
                                                   options=options)
        if file_path:
            self.load_class_mapping(task, file_path)

    def get_text_area(self, task):
        """
        Retrieves the text area widget associated with the given task.
        
        :param task: Task identifier as a string
        :return: QTextEdit widget corresponding to the task
        """
        if task == "classify":
            return self.classify_text_area
        elif task == "detect":
            return self.detect_text_area
        elif task == "segment":
            return self.segment_text_area
        else:
            raise ValueError(f"Unknown task: {task}")

    def get_current_task(self):
        """
        Retrieves the current task based on the selected tab in the deployment dialog.

        :return: Selected task identifier as a string
        """
        current_index = self.tab_widget.currentIndex()
        tasks = ["classify", "detect", "segment"]
        return tasks[current_index]

    def get_confidence_threshold(self):
        """
        Get the confidence threshold for predictions.

        :return: Confidence threshold as a float
        """
        threshold = self.main_window.get_uncertainty_thresh()
        return threshold if threshold < 0.10 else 0.10

    def load_class_mapping(self, task, file_path):
        """
        Load the class mapping file for the given task.

        :param task: Task identifier as a string
        :param file_path: Path to the class mapping file
        """
        try:
            with open(file_path, 'r') as f:
                self.class_mappings[task] = json.load(f)
            self.get_text_area(task).append("Class mapping file selected")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load class mapping file: {str(e)}")

    def is_sam_model_deployed(self):
        """
        Check if the SAM model is deployed and update the checkbox state accordingly.

        :return: Boolean indicating whether the SAM model is deployed
        """
        self.sam_dialog = self.main_window.sam_deploy_model_dialog

        if not self.sam_dialog.loaded_model:
            # Ensure that the checkbox is not checked
            self.sender().setChecked(False)
            QMessageBox.warning(self, "SAM Model", "SAM model not currently deployed")
            return False

        return True

    def load_model(self, task):
        """
        Load the model for the given task.

        :param task: Task identifier as a string
        """
        if not self.model_paths[task]:
            QMessageBox.warning(self, "Warning", f"No {task} model file selected")
            return

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.loaded_models[task] = YOLO(self.model_paths[task], task=task)
            self.loaded_models[task](np.zeros((224, 224, 3), dtype=np.uint8))

            if not self.class_mappings[task]:
                self.handle_missing_class_mapping(task)
            else:
                self.add_labels_to_label_window(task)
                self.check_and_display_class_names(task)
                QMessageBox.information(self, "Model Loaded", f"{task.capitalize()} model loaded successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load {task} model: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

    def handle_missing_class_mapping(self, task):
        """
        Handle the case when the class mapping file is missing.

        :param task: Task identifier as a string
        """
        reply = QMessageBox.question(self,
                                     'No Class Mapping Found',
                                     'Do you want to create generic labels automatically?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.create_generic_labels(task)
        else:
            self.check_and_display_class_names(task)
            QMessageBox.information(self, "Model Loaded", f"{task.capitalize()} model loaded successfully.")

    def add_labels_to_label_window(self, task):
        """
        Add labels to the label window based on the class mapping.

        :param task: Task identifier as a string
        """
        if self.class_mappings[task]:
            for label in self.class_mappings[task].values():
                self.label_window.add_label_if_not_exists(label['short_label_code'],
                                                          label['long_label_code'],
                                                          QColor(*label['color']))

    def create_generic_labels(self, task):
        """
        Create generic labels for the given task.

        :param task: Task identifier as a string
        """
        class_names = list(self.loaded_models[task].names.values())
        for class_name in class_names:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            self.label_window.add_label_if_not_exists(class_name,
                                                      class_name,
                                                      QColor(r, g, b))
        self.check_and_display_class_names(task)

        # Get the class mapping for the task
        labels = [self.label_window.get_label_by_short_code(class_name) for class_name in class_names]

        class_mapping = {}
        for label in labels:
            class_mapping[label.short_label_code] = label.to_dict()

        self.class_mappings[task] = class_mapping

    def check_and_display_class_names(self, task=None):
        """
        Check and display the class names for the given task.

        :param task: Task identifier as a string (optional)
        """
        if task is None:
            task = self.get_current_task()

        if self.loaded_models[task]:
            class_names = list(self.loaded_models[task].names.values())
            class_names_str = f"{task.capitalize()} Class Names: \n"
            missing_labels = []

            for class_name in class_names:
                label = self.label_window.get_label_by_short_code(class_name)
                if label:
                    class_names_str += f"✅ {label.short_label_code}: {label.long_label_code} \n"
                else:
                    class_names_str += f"❌ {class_name} \n"
                    missing_labels.append(class_name)

            self.get_text_area(task).setText(class_names_str)
            status_bar_text = f"{task.capitalize()} model loaded: {os.path.basename(self.model_paths[task])}"
            self.status_bars[task].setText(status_bar_text)

            if missing_labels:
                missing_labels_str = "\n".join(missing_labels)
                QMessageBox.warning(self,
                                    "Warning",
                                    f"The following short labels are missing for {task} and "
                                    f"cannot be predicted until added manually:"
                                    f"\n{missing_labels_str}")
        
    def predict_classification(self, annotations=None):
        """
        Predict the classification results for the given annotations.

        :param annotations: List of annotations (optional)
        """
        if self.loaded_models['classify'] is None:
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        if not annotations:
            # Predict only the selected annotation
            annotations = self.annotation_window.selected_annotations
        if not annotations:
            # If no annotations are selected, predict all annotations in the image
            annotations = self.annotation_window.get_image_review_annotations()

        images_np = []
        for annotation in annotations:
            images_np.append(pixmap_to_numpy(annotation.cropped_image))

        # Predict the classification results
        results = self.loaded_models['classify'](images_np,
                                                 device=self.main_window.device,
                                                 stream=True)
        # Create a result processor
        results_processor = ResultsProcessor(self.main_window,
                                             self.class_mappings['classify'],
                                             use_sam=False)
        # Process the classification results
        results_processor.process_classification_results(results, annotations)

        # Make cursor normal
        QApplication.restoreOverrideCursor()
        gc.collect()
        empty_cache()

    def predict_detection(self, image_paths=None):
        """
        Predict the detection results for the given image paths.

        :param image_paths: List of image paths (optional)
        """
        if self.loaded_models['detect'] is None:
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        if not image_paths:
            # Predict only the current image
            image_paths = [self.annotation_window.current_image_path]

        # Predict the detection results
        results = self.loaded_models['detect'](image_paths,
                                               agnostic_nms=True,
                                               conf=self.get_confidence_threshold(),
                                               iou=self.main_window.get_iou_thresh(),
                                               device=self.main_window.device,
                                               stream=True)

        # Create a result processor
        results_processor = ResultsProcessor(self.main_window,
                                             self.class_mappings['detect'],
                                             use_sam=self.use_sam['detect'].isChecked())

        # Process the detection results
        results_processor.process_detection_results(results)

        QApplication.restoreOverrideCursor()
        gc.collect()
        empty_cache()

    def predict_segmentation(self, image_paths=None):
        """
        Predict the segmentation results for the given image paths.

        :param image_paths: List of image paths (optional)
        """
        if self.loaded_models['segment'] is None:
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        if not image_paths:
            # Predict only the current image
            image_paths = [self.annotation_window.current_image_path]

        # Predict the segmentation results
        results = self.loaded_models['segment'](image_paths,
                                                agnostic_nms=True,
                                                conf=self.get_confidence_threshold(),
                                                iou=self.main_window.get_iou_thresh(),
                                                device=self.main_window.device,
                                                stream=True)

        # Create a result processor
        results_processor = ResultsProcessor(self.main_window,
                                             self.class_mappings['segment'],
                                             use_sam=self.use_sam['segment'].isChecked())

        # Process the segmentation results
        results_processor.process_segmentation_results(results)

        QApplication.restoreOverrideCursor()
        gc.collect()
        empty_cache()

    def deactivate_model(self, task):
        """
        Deactivate the model for the given task.

        :param task: Task identifier as a string
        """
        self.loaded_models[task] = None
        self.model_paths[task] = None
        self.class_mappings[task] = None
        gc.collect()
        empty_cache()
        self.status_bars[task].setText(f"No {task} model loaded")
        self.get_text_area(task).setText(f"No {task} model file selected")