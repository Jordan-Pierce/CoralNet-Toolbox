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

        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)

        self.classification_tab = QWidget()
        self.detection_tab = QWidget()
        self.segmentation_tab = QWidget()

        self.tab_widget.addTab(self.classification_tab, "Image Classification")
        self.tab_widget.addTab(self.detection_tab, "Object Detection")
        self.tab_widget.addTab(self.segmentation_tab, "Instance Segmentation")

        self.setup_classification_tab()
        self.setup_detection_tab()
        self.setup_segmentation_tab()

        # Task-specific status bars
        self.status_bars = {
            'classify': QLabel("No model loaded"),
            'detect': QLabel("No model loaded"),
            'segment': QLabel("No model loaded")
        }
        self.layout.addWidget(self.status_bars['classify'])
        self.layout.addWidget(self.status_bars['detect'])
        self.layout.addWidget(self.status_bars['segment'])

        self.tab_widget.currentChanged.connect(self.update_status_bar_visibility)

        self.setLayout(self.layout)

    def showEvent(self, event: QShowEvent):
        super().showEvent(event)
        self.check_and_display_class_names()
        self.update_status_bar_visibility(self.tab_widget.currentIndex())

    def setup_tab(self, tab, task):
        layout = QVBoxLayout()

        text_area = QTextEdit()
        text_area.setReadOnly(True)
        layout.addWidget(text_area)

        browse_model_button = QPushButton("Browse Model")
        browse_model_button.clicked.connect(lambda: self.browse_file(task))
        layout.addWidget(browse_model_button)

        browse_class_mapping_button = QPushButton("Browse Class Mapping")
        browse_class_mapping_button.clicked.connect(lambda: self.browse_class_mapping_file(task))
        layout.addWidget(browse_class_mapping_button)

        load_button = QPushButton("Load Model")
        load_button.clicked.connect(lambda: self.load_model(task))
        layout.addWidget(load_button)

        deactivate_button = QPushButton("Deactivate Model")
        deactivate_button.clicked.connect(lambda: self.deactivate_model(task))
        layout.addWidget(deactivate_button)

        use_sam_checkbox = QCheckBox("Use SAM for creating Polygons")
        use_sam_checkbox.stateChanged.connect(lambda: self.is_sam_model_deployed())

        if task == 'classify':
            use_sam_checkbox.setChecked(False)
            use_sam_checkbox.setEnabled(False)

        layout.addWidget(use_sam_checkbox)

        # Store the checkbox in the dictionary
        self.use_sam[task] = use_sam_checkbox

        tab.setLayout(layout)

        return text_area

    def setup_classification_tab(self):
        self.classification_text_area = self.setup_tab(self.classification_tab, 'classify')

    def setup_detection_tab(self):
        self.detection_text_area = self.setup_tab(self.detection_tab, 'detect')

    def setup_segmentation_tab(self):
        self.segmentation_text_area = self.setup_tab(self.segmentation_tab, 'segment')

    def get_current_task(self):
        index = self.tab_widget.currentIndex()
        return ['classify', 'detect', 'segment'][index]

    def get_text_area(self, task):
        if task == 'classify':
            return self.classification_text_area
        elif task == 'detect':
            return self.detection_text_area
        elif task == 'segment':
            return self.segmentation_text_area

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
        if not annotations:
            # If no annotations are found, return
            return

        # Preprocess the annotations
        self.preprocess_classification_annotations(annotations)
        # Unselect all annotations
        self.annotation_window.unselect_annotations()

        QApplication.restoreOverrideCursor()
        gc.collect()
        empty_cache()

    def preprocess_classification_annotations(self, annotations):
        if not annotations:
            return

        images_np = []
        for annotation in annotations:
            images_np.append(pixmap_to_numpy(annotation.cropped_image))

        progress_bar = ProgressBar(self, title=f"Making Classification Predictions")
        progress_bar.show()
        progress_bar.start_progress(len(annotations))

        results = self.loaded_models['classify'](images_np, stream=True, device=self.main_window.device)

        for annotation, result in zip(annotations, results):
            self.process_classification_result(annotation, result)
            progress_bar.update_progress()

        self.main_window.confidence_window.display_cropped_image(annotation)

        image_paths = list(set([annotation.image_path for annotation in annotations]))
        for image_path in image_paths:
            self.main_window.image_window.update_image_annotations(image_path)

        progress_bar.stop_progress()
        progress_bar.close()

    def process_classification_result(self, annotation, results):
        class_names = results.names
        top5 = results.probs.top5
        top5conf = results.probs.top5conf
        top1conf = top5conf[0].item()

        predictions = {}
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

        # Perform detection
        if self.main_window.get_uncertainty_thresh() < 0.10:
            conf = self.main_window.get_uncertainty_thresh()
        else:
            conf = 0.10  # Arbitrary value to prevent too many detections

        results = self.loaded_models['detect'](image_paths,
                                               agnostic_nms=True,
                                               conf=conf,
                                               iou=self.main_window.get_iou_thresh(),
                                               device=self.main_window.device,
                                               stream=True)

        # Check if the user selected to use SAM
        if self.use_sam['detect'].isChecked() and self.sam_dialog.loaded_model:
            # Convert the boxes to SAM masks, process as segmentations
            results = self.sam_dialog.boxes_to_masks(results)
            self.process_segmentation_results(results)
        else:
            # Process as detections
            self.process_detection_results(results)

        QApplication.restoreOverrideCursor()
        gc.collect()
        empty_cache()

    def process_detection_results(self, results_generator):
        # Get the class mapping for detection
        class_mapping = self.class_mappings['detect']

        progress_bar = ProgressBar(self, title=f"Making Detection Predictions")
        progress_bar.show()

        for results in results_generator:
            progress_bar.start_progress(len(results))
            for result in results:
                try:
                    # Get the image path
                    image_path = result.path.replace("\\", "/")

                    # Extract the results
                    cls = int(result.boxes.cls.cpu().numpy()[0])
                    cls_name = result.names[cls]
                    conf = float(result.boxes.conf.cpu().numpy()[0])
                    x_min, y_min, x_max, y_max = map(float, result.boxes.xyxy.cpu().numpy()[0])

                    # Determine the short label
                    short_label = 'Review'
                    if conf > self.main_window.get_uncertainty_thresh():
                        short_label = class_mapping.get(cls_name, {}).get('short_label_code', 'Review')

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

    def predict_segmentation(self, image_paths=None):
        if self.loaded_models['segment'] is None:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)

        if not image_paths:
            image_paths = [self.annotation_window.current_image_path]

        # Perform detection
        if self.main_window.get_uncertainty_thresh() < 0.10:
            conf = self.main_window.get_uncertainty_thresh()
        else:
            conf = 0.10  # Arbitrary value to prevent too many detections

        results = self.loaded_models['segment'](image_paths,
                                                agnostic_nms=True,
                                                conf=conf,
                                                iou=self.main_window.get_iou_thresh(),
                                                device=self.main_window.device,
                                                stream=True)

        # Check if the user selected to use SAM
        if self.use_sam['segment'].isChecked() and self.sam_dialog.loaded_model:
            # Convert the boxes to SAM masks
            results = self.sam_dialog.boxes_to_masks(results)

        # Process the segmentation results
        self.process_segmentation_results(results)

        QApplication.restoreOverrideCursor()
        gc.collect()
        empty_cache()

    def process_segmentation_results(self, results_generator):
        # If SAM is being used, and there is no class mapping for segmentation, use the detection class mapping
        class_mapping = self.class_mappings['segment']
        class_mapping = class_mapping if class_mapping else self.class_mappings['detect']

        progress_bar = ProgressBar(self, title=f"Making Segmentation Predictions")
        progress_bar.show()

        for results in results_generator:
            progress_bar.start_progress(len(results))
            for result in results:
                try:
                    # Get the image path
                    image_path = result.path.replace("\\", "/")

                    # Extract the results
                    cls = int(result.boxes.cls.cpu().numpy()[0])
                    cls_name = result.names[cls]
                    conf = float(result.boxes.conf.cpu().numpy()[0])
                    points = result.masks.cpu().xy[0].astype(float)

                    # Determine the short label
                    short_label = 'Review'
                    if conf > self.main_window.get_uncertainty_thresh():
                        short_label = class_mapping.get(cls_name, {}).get('short_label_code', 'Review')

                    # Prepare the annotation data
                    label = self.label_window.get_label_by_short_code(short_label)
                    points = [QPointF(x, y) for x, y in points]

                    # Create the rectangle annotation
                    annotation = PolygonAnnotation(points,
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