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

from toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

from toolbox.QtProgressBar import ProgressBar
from toolbox.utilities import pixmap_to_numpy


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DeployModelDialog(QDialog):
    def __init__(self, main_window, parent=None):
        """
        Dialog for deploying machine learning models for image classification, object detection, 
        and instance segmentation.
        
        :param main_window: MainWindow object
        :param parent: Parent widget
        """
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
        super().showEvent(event)
        self.check_and_display_class_names()
        self.update_status_bar_visibility(self.tab_widget.currentIndex())

    def setup_tabs(self):
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
        self.status_bars = {
            'classify': QLabel("No model loaded"),
            'detect': QLabel("No model loaded"),
            'segment': QLabel("No model loaded")
        }
        for bar in self.status_bars.values():
            self.layout.addWidget(bar)

    def update_status_bar_visibility(self, index):
        current_task = self.get_current_task()
        for task, status_bar in self.status_bars.items():
            status_bar.setVisible(task == current_task)

    def browse_file(self, task):
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

    def load_class_mapping(self, task, file_path):
        try:
            with open(file_path, 'r') as f:
                self.class_mappings[task] = json.load(f)
            self.get_text_area(task).append("Class mapping file selected")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load class mapping file: {str(e)}")

    def is_sam_model_deployed(self):
        self.sam_dialog = self.main_window.sam_deploy_model_dialog

        if not self.sam_dialog.loaded_model:
            # Ensure that the checkbox is not checked
            self.sender().setChecked(False)
            QMessageBox.warning(self, "SAM Model", "SAM model not currently deployed")
            return False

        return True

    def load_model(self, task):
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
        if self.class_mappings[task]:
            for label in self.class_mappings[task].values():
                self.label_window.add_label_if_not_exists(label['short_label_code'],
                                                          label['long_label_code'],
                                                          QColor(*label['color']))

    def create_generic_labels(self, task):
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

    def deactivate_model(self, task):
        self.loaded_models[task] = None
        self.model_paths[task] = None
        self.class_mappings[task] = None
        gc.collect()
        empty_cache()
        self.status_bars[task].setText(f"No {task} model loaded")
        self.get_text_area(task).setText(f"No {task} model file selected")

    def predict_classification(self, annotations=None):
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

        # Preprocess the annotations
        self.preprocess_classification_annotations(annotations)
        # Unselect all annotations
        self.annotation_window.unselect_annotations()

        # Make cursor normal
        QApplication.restoreOverrideCursor()
        gc.collect()
        empty_cache()

    def preprocess_classification_annotations(self, annotations):
        if not annotations:
            return

        images_np = []
        for annotation in annotations:
            images_np.append(pixmap_to_numpy(annotation.cropped_image))

        progress_bar = ProgressBar(self, title="Making Classification Predictions")
        progress_bar.show()
        progress_bar.start_progress(len(annotations))

        # Predict the classification results
        results = self.loaded_models['classify'](images_np, stream=True, device=self.main_window.device)

        # Process the classification results
        for annotation, result in zip(annotations, results):
            self.process_classification_result(annotation, result)
            progress_bar.update_progress()

        # Update the image window and confidence window
        self.main_window.confidence_window.display_cropped_image(annotation)

        # Update the annotations for each image in the image window
        image_paths = list(set([annotation.image_path for annotation in annotations]))
        for image_path in image_paths:
            self.main_window.image_window.update_image_annotations(image_path)

        progress_bar.stop_progress()
        progress_bar.close()

    def process_classification_result(self, annotation, results):
        predictions = {}

        try:
            class_names = results.names
            top5 = results.probs.top5
            top5conf = results.probs.top5conf
            top1conf = top5conf[0].item()
        except Exception as e:
            print(f"Warning: Failed to process classification result\n{e}")
            return predictions

        for idx, conf in zip(top5, top5conf):
            class_name = class_names[idx]
            label = self.label_window.get_label_by_short_code(class_name)
            if label:
                predictions[label] = float(conf)

        if predictions:
            annotation.update_machine_confidence(predictions)
            if top1conf < self.main_window.get_uncertainty_thresh():
                label = self.label_window.get_label_by_id('-1')
                annotation.update_label(label)

    def predict_detection(self, image_paths=None):
        if self.loaded_models['detect'] is None:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)

        if not image_paths:
            image_paths = [self.annotation_window.current_image_path]

        conf = self.get_confidence_threshold()

        results = self.loaded_models['detect'](image_paths,
                                               agnostic_nms=True,
                                               conf=conf,
                                               iou=self.main_window.get_iou_thresh(),
                                               device=self.main_window.device,
                                               stream=True)

        if self.use_sam['detect'].isChecked() and self.sam_dialog.loaded_model:
            results = self.sam_dialog.boxes_to_masks(results)
            self.process_segmentation_results(results)
        else:
            self.process_detection_results(results)

        QApplication.restoreOverrideCursor()
        gc.collect()
        empty_cache()

    def get_confidence_threshold(self):
        threshold = self.main_window.get_uncertainty_thresh()
        return threshold if threshold < 0.10 else 0.10

    def process_detection_results(self, results_generator):
        class_mapping = self.class_mappings['detect']
        progress_bar = ProgressBar(self, title="Making Detection Predictions")
        progress_bar.show()

        for results in results_generator:
            progress_bar.start_progress(len(results))
            for result in results:
                self.process_single_detection_result(result, class_mapping)
                progress_bar.update_progress()

        progress_bar.stop_progress()
        progress_bar.close()

    def process_single_detection_result(self, result, class_mapping):
        try:
            image_path = result.path.replace("\\", "/")
            cls, cls_name, conf, x_min, y_min, x_max, y_max = self.extract_detection_result(result)
            short_label = self.get_short_label_for_detection(cls_name, conf, class_mapping)
            label = self.label_window.get_label_by_short_code(short_label)
            annotation = self.create_rectangle_annotation(x_min, y_min, x_max, y_max, label, image_path)
            self.store_and_display_annotation(annotation, image_path, cls_name, conf)
        except Exception as e:
            print(f"Warning: Failed to process detection result\n{e}")

    def extract_detection_result(self, result):
        cls = int(result.boxes.cls.cpu().numpy()[0])
        cls_name = result.names[cls]
        conf = float(result.boxes.conf.cpu().numpy()[0])
        x_min, y_min, x_max, y_max = map(float, result.boxes.xyxy.cpu().numpy()[0])
        return cls, cls_name, conf, x_min, y_min, x_max, y_max

    def get_short_label_for_detection(self, cls_name, conf, class_mapping):
        if conf <= self.main_window.get_uncertainty_thresh():
            return 'Review'
        return class_mapping.get(cls_name, {}).get('short_label_code', 'Review')

    def create_rectangle_annotation(self, x_min, y_min, x_max, y_max, label, image_path):
        top_left = QPointF(x_min, y_min)
        bottom_right = QPointF(x_max, y_max)
        return RectangleAnnotation(top_left, 
                                   bottom_right, 
                                   label.short_label_code, 
                                   label.long_label_code, 
                                   label.color, 
                                   image_path, 
                                   label.id, 
                                   self.main_window.get_transparency_value(), 
                                   show_msg=True)

    def store_and_display_annotation(self, annotation, image_path, cls_name, conf):
        self.annotation_window.annotations_dict[annotation.id] = annotation
        annotation.selected.connect(self.annotation_window.select_annotation)
        annotation.annotationDeleted.connect(self.annotation_window.delete_annotation)
        annotation.annotationUpdated.connect(self.main_window.confidence_window.display_cropped_image)
        predictions = {self.label_window.get_label_by_short_code(cls_name): conf}
        annotation.update_machine_confidence(predictions)
        if conf < self.main_window.get_uncertainty_thresh():
            review_label = self.label_window.get_label_by_id('-1')
            annotation.update_label(review_label)
        if image_path == self.annotation_window.current_image_path:
            annotation.create_graphics_item(self.annotation_window.scene)
            annotation.create_cropped_image(self.annotation_window.rasterio_image)
            self.main_window.confidence_window.display_cropped_image(annotation)
        self.main_window.image_window.update_image_annotations(image_path)

    def predict_segmentation(self, image_paths=None):
        if self.loaded_models['segment'] is None:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)

        if not image_paths:
            image_paths = [self.annotation_window.current_image_path]

        conf = self.get_confidence_threshold()

        results = self.loaded_models['segment'](image_paths,
                                                agnostic_nms=True,
                                                conf=conf,
                                                iou=self.main_window.get_iou_thresh(),
                                                device=self.main_window.device,
                                                stream=True)

        if self.use_sam['segment'].isChecked() and self.sam_dialog.loaded_model:
            results = self.sam_dialog.boxes_to_masks(results)

        self.process_segmentation_results(results)

        QApplication.restoreOverrideCursor()
        gc.collect()
        empty_cache()

    def process_segmentation_results(self, results_generator):
        class_mapping = self.class_mappings['segment']
        if not class_mapping:
            class_mapping = self.class_mappings['detect']

        progress_bar = ProgressBar(self, title=f"Making Segmentation Predictions")
        progress_bar.show()

        for results in results_generator:
            progress_bar.start_progress(len(results))
            for result in results:
                self.process_single_segmentation_result(result, class_mapping)
                progress_bar.update_progress()

        progress_bar.stop_progress()
        progress_bar.close()

    def process_single_segmentation_result(self, result, class_mapping):
        try:
            image_path = result.path.replace("\\", "/")
            cls, cls_name, conf, points = self.extract_segmentation_result(result)
            short_label = self.get_short_label_for_detection(cls_name, conf, class_mapping)
            label = self.label_window.get_label_by_short_code(short_label)
            annotation = self.create_polygon_annotation(points, label, image_path)
            self.store_and_display_annotation(annotation, image_path, cls_name, conf)
        except Exception as e:
            print(f"Warning: Failed to process segmentation result\n{e}")

    def extract_segmentation_result(self, result):
        cls = int(result.boxes.cls.cpu().numpy()[0])
        cls_name = result.names[cls]
        conf = float(result.boxes.conf.cpu().numpy()[0])
        points = result.masks.cpu().xy[0].astype(float)
        return cls, cls_name, conf, points

    def create_polygon_annotation(self, points, label, image_path):
        points = [QPointF(x, y) for x, y in points]
        return PolygonAnnotation(points, 
                                 label.short_label_code, 
                                 label.long_label_code, 
                                 label.color, 
                                 image_path, 
                                 label.id, 
                                 self.main_window.get_transparency_value(), 
                                 show_msg=True)
