# TODO
#   - root output directory for create dataset
#   - update pre-warm inference based on input size of model
#   - convert thumbnails to rows?

import os
import uuid
import json
import random
import weakref
import datetime

import numpy as np
import pandas as pd
from ultralytics import YOLO

from PyQt5.QtWidgets import (QProgressBar, QMainWindow, QFileDialog, QApplication, QGridLayout, QGraphicsView,
                             QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, QToolBar, QAction, QScrollArea,
                             QSizePolicy, QMessageBox, QCheckBox, QDialog, QHBoxLayout, QWidget, QVBoxLayout, QLabel,
                             QPushButton, QColorDialog, QMenu, QLineEdit, QSpinBox, QDialog, QHBoxLayout, QTextEdit,
                             QPushButton, QComboBox, QSpinBox, QGraphicsPixmapItem, QGraphicsRectItem, QSlider,
                             QFormLayout, QInputDialog, QFrame, QTabWidget, QDialogButtonBox, QDoubleSpinBox,
                             QGroupBox, QListWidget, QListWidgetItem, QPlainTextEdit, QRadioButton, QTableWidget,
                             QTableWidgetItem)

from PyQt5.QtGui import (QMouseEvent, QIcon, QImage, QPixmap, QColor, QPainter, QPen, QBrush, QFontMetrics, QFont,
                         QCursor, QMovie)

from PyQt5.QtCore import (Qt, pyqtSignal, QSize, QObject, QThreadPool, QRunnable, QTimer, QEvent, QPointF, QRectF,
                          QThread)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class ProgressBar(QDialog):
    def __init__(self, parent=None, title="Progress"):
        super().__init__(parent)

        self.setWindowTitle(title)
        self.setModal(True)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.cancel)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.cancel_button)

        self.value = 0
        self.max_value = 100
        self.canceled = False

    def update_progress(self):
        if not self.canceled:
            self.value += 1
            self.progress_bar.setValue(self.value)
            if self.value >= self.max_value:
                self.stop_progress()

    def start_progress(self, max_value):
        self.value = 0
        self.max_value = max_value
        self.canceled = False
        self.progress_bar.setRange(0, max_value)

    def stop_progress(self):
        self.value = self.max_value
        self.progress_bar.setValue(self.value)

    def cancel(self):
        self.canceled = True
        self.stop_progress()

    def wasCanceled(self):
        return self.canceled


class AnnotationSamplingDialog(QDialog):
    annotationsSampled = pyqtSignal(list, bool)  # Signal to emit the sampled annotations and apply to all flag

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.annotation_window = main_window.annotation_window
        self.label_window = main_window.label_window
        self.image_window = main_window.image_window
        self.deploy_model_dialog = main_window.deploy_model_dialog

        self.setWindowTitle("Sample Annotations")

        self.layout = QVBoxLayout(self)

        # Sampling Method
        self.method_label = QLabel("Sampling Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Random", "Stratified Random", "Uniform"])
        self.layout.addWidget(self.method_label)
        self.layout.addWidget(self.method_combo)

        # Number of Annotations
        self.num_annotations_label = QLabel("Number of Annotations:")
        self.num_annotations_spinbox = QSpinBox()
        self.num_annotations_spinbox.setMinimum(1)
        self.num_annotations_spinbox.setMaximum(10000)  # Arbitrary large number for "infinite"
        self.layout.addWidget(self.num_annotations_label)
        self.layout.addWidget(self.num_annotations_spinbox)

        # Annotation Size
        self.annotation_size_label = QLabel("Annotation Size:")
        self.annotation_size_spinbox = QSpinBox()
        self.annotation_size_spinbox.setMinimum(32)
        self.annotation_size_spinbox.setMaximum(10000)  # Arbitrary large number for "infinite"
        self.annotation_size_spinbox.setValue(self.annotation_window.annotation_size)
        self.layout.addWidget(self.annotation_size_label)
        self.layout.addWidget(self.annotation_size_spinbox)

        # Margin Offsets using QFormLayout
        self.margin_form_layout = QFormLayout()
        self.margin_x_min_spinbox = self.create_margin_spinbox("X Min", self.margin_form_layout)
        self.margin_y_min_spinbox = self.create_margin_spinbox("Y Min", self.margin_form_layout)
        self.margin_x_max_spinbox = self.create_margin_spinbox("X Max", self.margin_form_layout)
        self.margin_y_max_spinbox = self.create_margin_spinbox("Y Max", self.margin_form_layout)
        self.layout.addLayout(self.margin_form_layout)

        # Apply to Next Images Checkbox
        self.apply_next_checkbox = QCheckBox("Apply to next images")
        self.layout.addWidget(self.apply_next_checkbox)
        # Apply to All Images Checkbox
        self.apply_all_checkbox = QCheckBox("Apply to all images")
        self.layout.addWidget(self.apply_all_checkbox)
        # Ensure only one of the apply checkboxes can be selected at a time
        self.apply_next_checkbox.stateChanged.connect(self.update_apply_next_checkboxes)
        self.apply_all_checkbox.stateChanged.connect(self.update_apply_all_checkboxes)

        # Make predictions on sampled annotations checkbox
        self.apply_predictions_checkbox = QCheckBox("Make predictions on sample annotations")
        self.layout.addWidget(self.apply_predictions_checkbox)
        # Ensure checkbox can only be selected if model is loaded
        self.apply_predictions_checkbox.stateChanged.connect(self.update_apply_predictions_checkboxes)

        # Preview Button
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.preview_annotations)
        self.layout.addWidget(self.preview_button)

        # Preview Area
        self.preview_view = QGraphicsView(self)
        self.preview_scene = QGraphicsScene(self)
        self.preview_view.setScene(self.preview_scene)
        self.preview_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.preview_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.layout.addWidget(self.preview_view)

        # Accept/Cancel Buttons
        self.button_box = QHBoxLayout()
        self.accept_button = QPushButton("Accept")
        self.accept_button.clicked.connect(self.accept_annotations)
        self.button_box.addWidget(self.accept_button)

        self.layout.addLayout(self.button_box)

        self.sampled_annotations = []

    def create_margin_spinbox(self, label_text, layout):
        label = QLabel(label_text + ":")
        spinbox = QSpinBox()
        spinbox.setMinimum(0)
        spinbox.setMaximum(1000)
        layout.addRow(label, spinbox)
        return spinbox

    def showEvent(self, event):
        super().showEvent(event)
        self.showMaximized()  # Maximize the dialog when it is shown
        self.reset_defaults()  # Reset settings to defaults

    def reset_defaults(self):
        self.preview_scene.clear()
        self.method_combo.setCurrentIndex(0)
        self.num_annotations_spinbox.setValue(1)
        self.annotation_size_spinbox.setValue(self.annotation_window.annotation_size)
        self.margin_x_min_spinbox.setValue(0)
        self.margin_y_min_spinbox.setValue(0)
        self.margin_x_max_spinbox.setValue(0)
        self.margin_y_max_spinbox.setValue(0)
        self.apply_all_checkbox.setChecked(False)
        self.apply_next_checkbox.setChecked(False)

    def update_apply_next_checkboxes(self):
        if self.apply_next_checkbox.isChecked():
            self.apply_next_checkbox.setChecked(True)
            self.apply_all_checkbox.setChecked(False)
            return

        if not self.apply_next_checkbox.isChecked():
            self.apply_next_checkbox.setChecked(False)
            return

    def update_apply_all_checkboxes(self):
        if self.apply_all_checkbox.isChecked():
            self.apply_all_checkbox.setChecked(True)
            self.apply_next_checkbox.setChecked(False)
            return

        if not self.apply_all_checkbox.isChecked():
            self.apply_all_checkbox.setChecked(False)
            return

    def update_apply_predictions_checkboxes(self):
        model_loaded = self.deploy_model_dialog.loaded_model is not None

        if not model_loaded:
            self.apply_predictions_checkbox.setChecked(False)
            QMessageBox.warning(self, "No model", "No model deployed to apply to predictions")
            return

        if self.apply_predictions_checkbox.isChecked():
            self.apply_predictions_checkbox.setChecked(True)
        else:
            self.apply_predictions_checkbox.setChecked(False)

    def sample_annotations(self, method, num_annotations, annotation_size, margins, image_width, image_height):
        # Extract the margins
        margin_x_min, margin_y_min, margin_x_max, margin_y_max = margins

        annotations = []

        if method == "Random":
            for _ in range(num_annotations):
                x = random.randint(margin_x_min, image_width - annotation_size - margin_x_max)
                y = random.randint(margin_y_min, image_height - annotation_size - margin_y_max)
                annotations.append((x, y, annotation_size))

        if method == "Uniform":
            grid_size = int(num_annotations ** 0.5)
            x_step = (image_width - margin_x_min - margin_x_max - annotation_size) / grid_size
            y_step = (image_height - margin_y_min - margin_y_max - annotation_size) / grid_size
            for i in range(grid_size):
                for j in range(grid_size):
                    x = margin_x_min + int(i * x_step + annotation_size / 2)
                    y = margin_y_min + int(j * y_step + annotation_size / 2)
                    annotations.append((x, y, annotation_size))

        if method == "Stratified Random":
            grid_size = int(num_annotations ** 0.5)
            x_step = (image_width - margin_x_min - margin_x_max - annotation_size) / grid_size
            y_step = (image_height - margin_y_min - margin_y_max - annotation_size) / grid_size
            for i in range(grid_size):
                for j in range(grid_size):
                    x = margin_x_min + int(
                        i * x_step + random.uniform(annotation_size / 2, x_step - annotation_size / 2))
                    y = margin_y_min + int(
                        j * y_step + random.uniform(annotation_size / 2, y_step - annotation_size / 2))
                    annotations.append((x, y, annotation_size))

        return annotations

    def preview_annotations(self):
        method = self.method_combo.currentText()
        num_annotations = self.num_annotations_spinbox.value()
        annotation_size = self.annotation_size_spinbox.value()
        margin_x_min = self.margin_x_min_spinbox.value()
        margin_y_min = self.margin_y_min_spinbox.value()
        margin_x_max = self.margin_x_max_spinbox.value()
        margin_y_max = self.margin_y_max_spinbox.value()

        margins = margin_x_min, margin_y_min, margin_x_max, margin_y_max

        self.sampled_annotations = self.sample_annotations(method,
                                                           num_annotations,
                                                           annotation_size,
                                                           margins,
                                                           self.annotation_window.image_item.pixmap().width(),
                                                           self.annotation_window.image_item.pixmap().height())

        self.draw_annotation_previews(margins)

    def draw_annotation_previews(self, margins):

        margin_x_min, margin_y_min, margin_x_max, margin_y_max = margins

        self.preview_scene.clear()
        pixmap = self.annotation_window.image_item.pixmap()
        if pixmap:
            # Add the image to the scene
            self.preview_scene.addItem(QGraphicsPixmapItem(pixmap))

            # Draw annotations
            for annotation in self.sampled_annotations:
                x, y, size = annotation
                rect_item = QGraphicsRectItem(x, y, size, size)
                rect_item.setPen(QPen(Qt.white, 4))
                brush = QBrush(Qt.white)
                brush.setStyle(Qt.SolidPattern)
                color = brush.color()
                color.setAlpha(50)
                brush.setColor(color)
                rect_item.setBrush(brush)
                self.preview_scene.addItem(rect_item)

            # Draw margin lines
            pen = QPen(QColor("red"), 5)
            pen.setStyle(Qt.DotLine)
            image_width = pixmap.width()
            image_height = pixmap.height()

            self.preview_scene.addLine(margin_x_min, 0, margin_x_min, image_height, pen)
            self.preview_scene.addLine(image_width - margin_x_max, 0, image_width - margin_x_max, image_height, pen)
            self.preview_scene.addLine(0, margin_y_min, image_width, margin_y_min, pen)
            self.preview_scene.addLine(0, image_height - margin_y_max, image_width, image_height - margin_y_max, pen)

            # Apply dark transparency outside the margins
            overlay_color = QColor(0, 0, 0, 150)  # Black with transparency

            # Left overlay
            left_overlay = QGraphicsRectItem(0, 0, margin_x_min, image_height)
            left_overlay.setBrush(QBrush(overlay_color))
            left_overlay.setPen(QPen(Qt.NoPen))
            self.preview_scene.addItem(left_overlay)

            # Right overlay
            right_overlay = QGraphicsRectItem(image_width - margin_x_max, 0, margin_x_max, image_height)
            right_overlay.setBrush(QBrush(overlay_color))
            right_overlay.setPen(QPen(Qt.NoPen))
            self.preview_scene.addItem(right_overlay)

            # Top overlay
            top_overlay = QGraphicsRectItem(margin_x_min, 0, image_width - margin_x_min - margin_x_max, margin_y_min)
            top_overlay.setBrush(QBrush(overlay_color))
            top_overlay.setPen(QPen(Qt.NoPen))
            self.preview_scene.addItem(top_overlay)

            # Bottom overlay
            bottom_overlay = QGraphicsRectItem(margin_x_min,
                                               image_height - margin_y_max,
                                               image_width - margin_x_min - margin_x_max,
                                               margin_y_max)

            bottom_overlay.setBrush(QBrush(overlay_color))
            bottom_overlay.setPen(QPen(Qt.NoPen))
            self.preview_scene.addItem(bottom_overlay)

            self.preview_view.fitInView(self.preview_scene.sceneRect(), Qt.KeepAspectRatio)

    def accept_annotations(self):

        margins = (self.margin_x_min_spinbox.value(),
                   self.margin_y_min_spinbox.value(),
                   self.margin_x_max_spinbox.value(),
                   self.margin_y_max_spinbox.value())

        self.add_sampled_annotations(self.method_combo.currentText(),
                                     self.num_annotations_spinbox.value(),
                                     self.annotation_size_spinbox.value(),
                                     margins)
        self.accept()

    def add_sampled_annotations(self, method, num_annotations, annotation_size, margins):

        self.apply_to_next = self.apply_next_checkbox.isChecked()
        self.apply_to_all = self.apply_all_checkbox.isChecked()
        self.make_predictions = self.apply_predictions_checkbox.isChecked()

        # Sets the LabelWindow and AnnotationWindow to Review
        self.label_window.set_selected_label("-1")
        review_label = self.annotation_window.selected_label

        if self.apply_to_next:
            current_image_path = self.annotation_window.current_image_path
            current_image_index = self.image_window.image_list.index(current_image_path)
            image_paths = self.image_window.image_list[current_image_index:]
        elif self.apply_to_all:
            image_paths = list(self.annotation_window.loaded_image_paths)
        else:
            image_paths = [self.annotation_window.current_image_path]

        # Create and show the progress bar
        progress_bar = ProgressBar(self, title="Sampling Annotations")
        progress_bar.show()
        progress_bar.start_progress(num_annotations)

        for image_path in image_paths:
            image = QImage(image_path)
            image_width = image.width()
            image_height = image.height()
            image_item = QGraphicsPixmapItem(QPixmap(image))
            # Sample the annotation, given params
            annotations = self.sample_annotations(method,
                                                  num_annotations,
                                                  annotation_size,
                                                  margins,
                                                  image_width,
                                                  image_height)

            for annotation in annotations:
                x, y, size = annotation
                new_annotation = Annotation(QPointF(x + size // 2, y + size // 2),
                                            size,
                                            review_label.short_label_code,
                                            review_label.long_label_code,
                                            review_label.color,
                                            image_path,
                                            review_label.id,
                                            transparency=self.annotation_window.transparency)

                if self.make_predictions:
                    # Create the cropped image now
                    new_annotation.create_cropped_image(image_item)

                # Add annotation to the dict
                self.annotation_window.annotations_dict[new_annotation.id] = new_annotation

                # Update the progress bar
                progress_bar.update_progress()
                QApplication.processEvents()  # Update GUI

        # Stop the progress bar
        progress_bar.stop_progress()
        progress_bar.close()

        # If selected, make predictions on all annotations
        self.make_predictions_on_sampled_annotations(image_paths)

        # Set / load the image / annotations of the last image
        self.image_window.load_image_by_path(image_path)

        QMessageBox.information(self,
                                "Annotations Sampled",
                                "Annotations have been sampled successfully.")

        self.reset_defaults()

    def make_predictions_on_sampled_annotations(self, image_paths):
        if not self.make_predictions:
            return

        # Create and show the progress bar
        progress_bar = ProgressBar(self, title="Making Predictions")
        progress_bar.show()
        progress_bar.start_progress(len(image_paths))

        for image_path in image_paths:
            annotations = self.annotation_window.get_image_annotations(image_path)
            self.deploy_model_dialog.predict(annotations)

            # Update the progress bar
            progress_bar.update_progress()
            QApplication.processEvents()  # Update GUI

        # Stop the progress bar
        progress_bar.stop_progress()
        progress_bar.close()


class GlobalEventFilter(QObject):
    def __init__(self, main_window):
        super().__init__()
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window
        self.deploy_model_dialog = main_window.deploy_model_dialog
        self.image_window = main_window.image_window

    def eventFilter(self, obj, event):
        # Check if the event is a wheel event
        if event.type() == QEvent.Wheel:
            # Handle Zoom wheel for setting annotation size
            if event.modifiers() & Qt.ControlModifier:
                delta = event.angleDelta().y()
                if delta > 0:
                    self.annotation_window.set_annotation_size(delta=16)  # Zoom in
                else:
                    self.annotation_window.set_annotation_size(delta=-16)  # Zoom out
                return True

        elif event.type() == QEvent.KeyPress:
            if event.modifiers() & Qt.ControlModifier:

                # Handle WASD keys for selecting Label
                if event.key() in [Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D]:
                    self.label_window.handle_wasd_key(event.key())
                    return True

                # Handle Delete key for Deleting selected annotation
                if event.key() == Qt.Key_Delete:
                    self.annotation_window.delete_selected_annotation()
                    return True

                # Handle hotkey for prediction
                if event.key() == Qt.Key_Z:
                    self.deploy_model_dialog.predict()
                    return True

                # Handle annotation cycling hotkeys
                if event.key() == Qt.Key_Left:
                    self.annotation_window.cycle_annotations(-1)
                    return True
                elif event.key() == Qt.Key_Right:
                    self.annotation_window.cycle_annotations(1)
                    return True

                # Handle thumbnail cycling hotkeys
                if event.key() == Qt.Key_Up:
                    self.image_window.cycle_previous_image()
                    return True
                elif event.key() == Qt.Key_Down:
                    self.image_window.cycle_next_image()
                    return True

            # Handle Escape key for exiting program
            if event.key() == Qt.Key_Escape:
                self.show_exit_confirmation_dialog()
                return True

        # Return False for other events to allow them to be processed by the target object
        return False

    def show_exit_confirmation_dialog(self):
        # noinspection PyTypeChecker
        reply = QMessageBox.question(None,
                                     'Confirm Exit',
                                     'Are you sure you want to exit?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            QApplication.quit()


class MainWindow(QMainWindow):
    toolChanged = pyqtSignal(str)  # Signal to emit the current tool state

    def __init__(self):
        super().__init__()

        root = os.path.dirname(os.path.abspath(__file__))

        # Define the icon path
        self.setWindowTitle("CoralNet Toolbox")
        # Set the window icon
        main_window_icon_path = f"{root}/icons/toolbox.png"
        self.setWindowIcon(QIcon(main_window_icon_path))

        # Set window flags for resizing, minimize, maximize, and customizing
        self.setWindowFlags(Qt.Window |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint |
                            Qt.WindowTitleHint)

        self.annotation_window = AnnotationWindow(self)
        self.label_window = LabelWindow(self)
        self.image_window = ImageWindow(self)
        self.confidence_window = ConfidenceWindow(self)

        self.create_dataset_dialog = CreateDatasetDialog(self)
        self.train_model_dialog = TrainModelDialog(self)
        self.optimize_model_dialog = OptimizeModelDialog(self)
        self.deploy_model_dialog = DeployModelDialog(self)

        self.annotation_sampling_dialog = AnnotationSamplingDialog(self)

        # Connect signals to update status bar
        self.annotation_window.imageLoaded.connect(self.update_image_dimensions)
        self.annotation_window.mouseMoved.connect(self.update_mouse_position)

        # Connect the toolChanged signal (to the AnnotationWindow)
        self.toolChanged.connect(self.annotation_window.set_selected_tool)
        # Connect the toolChanged signal (to the Toolbar)
        self.annotation_window.toolChanged.connect(self.handle_tool_changed)
        # Connect the selectedLabel signal to the LabelWindow's set_selected_label method
        self.annotation_window.labelSelected.connect(self.label_window.set_selected_label)
        # Connect the imageSelected signal to update_current_image_path in AnnotationWindow
        self.image_window.imageSelected.connect(self.annotation_window.update_current_image_path)
        # Connect the imageDeleted signal to delete_image in AnnotationWindow
        self.image_window.imageDeleted.connect(self.annotation_window.delete_image)
        # Connect the labelSelected signal from LabelWindow to update the selected label in AnnotationWindow
        self.label_window.labelSelected.connect(self.annotation_window.set_selected_label)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        # Menu bar
        self.menu_bar = self.menuBar()

        self.import_menu = self.menu_bar.addMenu("Import")

        self.import_images_action = QAction("Import Images", self)
        self.import_images_action.triggered.connect(self.import_images)
        self.import_menu.addAction(self.import_images_action)

        self.import_labels_action = QAction("Import Labels (JSON)", self)
        self.import_labels_action.triggered.connect(self.label_window.import_labels)
        self.import_menu.addAction(self.import_labels_action)

        self.import_annotations_action = QAction("Import Annotations (JSON)", self)
        self.import_annotations_action.triggered.connect(self.annotation_window.import_annotations)
        self.import_menu.addAction(self.import_annotations_action)

        self.import_coralnet_annotations_action = QAction("Import Annotations (CoralNet)", self)
        self.import_coralnet_annotations_action.triggered.connect(self.annotation_window.import_coralnet_annotations)
        self.import_menu.addAction(self.import_coralnet_annotations_action)

        self.export_menu = self.menu_bar.addMenu("Export")

        self.export_labels_action = QAction("Export Labels (JSON)", self)
        self.export_labels_action.triggered.connect(self.label_window.export_labels)
        self.export_menu.addAction(self.export_labels_action)

        self.export_annotations_action = QAction("Export Annotations (JSON)", self)
        self.export_annotations_action.triggered.connect(self.annotation_window.export_annotations)
        self.export_menu.addAction(self.export_annotations_action)

        self.export_coralnet_annotations_action = QAction("Export Annotations (CoralNet)", self)
        self.export_coralnet_annotations_action.triggered.connect(self.annotation_window.export_coralnet_annotations)
        self.export_menu.addAction(self.export_coralnet_annotations_action)

        self.annotation_sampling_action = QAction("Sample Annotations", self)
        self.annotation_sampling_action.triggered.connect(self.open_annotation_sampling_dialog)
        self.menu_bar.addAction(self.annotation_sampling_action)

        # CoralNet menu
        self.coralnet_menu = self.menu_bar.addMenu("CoralNet")

        self.coralnet_authenticate_action = QAction("Authenticate", self)
        self.coralnet_menu.addAction(self.coralnet_authenticate_action)

        self.coralnet_upload_action = QAction("Upload", self)
        self.coralnet_menu.addAction(self.coralnet_upload_action)

        self.coralnet_download_action = QAction("Download", self)
        self.coralnet_menu.addAction(self.coralnet_download_action)

        self.coralnet_model_api_action = QAction("Model API", self)
        self.coralnet_menu.addAction(self.coralnet_model_api_action)

        # Machine Learning menu
        self.ml_menu = self.menu_bar.addMenu("Machine Learning")

        self.ml_create_dataset_action = QAction("Create Dataset", self)
        self.ml_create_dataset_action.triggered.connect(self.open_create_dataset_dialog)
        self.ml_menu.addAction(self.ml_create_dataset_action)

        self.ml_train_model_action = QAction("Train Model", self)
        self.ml_train_model_action.triggered.connect(self.open_train_model_dialog)
        self.ml_menu.addAction(self.ml_train_model_action)

        self.ml_optimize_model_action = QAction("Optimize Model", self)
        self.ml_optimize_model_action.triggered.connect(self.open_optimize_model_dialog)
        self.ml_menu.addAction(self.ml_optimize_model_action)

        self.ml_deploy_model_action = QAction("Deploy Model", self)
        self.ml_deploy_model_action.triggered.connect(self.open_deploy_model_dialog)
        self.ml_menu.addAction(self.ml_deploy_model_action)

        # Create and add the toolbar
        self.toolbar = QToolBar("Tools", self)
        self.toolbar.setOrientation(Qt.Vertical)
        self.toolbar.setFixedWidth(40)
        self.toolbar.setMovable(False)  # Lock the toolbar in place
        self.addToolBar(Qt.LeftToolBarArea, self.toolbar)

        # Add a spacer before the first tool with a fixed height
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        spacer.setFixedHeight(10)  # Set a fixed height for the spacer
        self.toolbar.addWidget(spacer)

        # Define icon paths
        select_icon_path = f"{root}/icons/select.png"
        annotate_icon_path = f"{root}/icons/annotate.png"
        polygon_icon_path = f"{root}/icons/polygon.png"

        # Add tools here with icons
        self.select_tool_action = QAction(QIcon(select_icon_path), "Select", self)
        self.select_tool_action.setCheckable(True)
        self.select_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.select_tool_action)

        self.annotate_tool_action = QAction(QIcon(annotate_icon_path), "Annotate", self)
        self.annotate_tool_action.setCheckable(True)
        self.annotate_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.annotate_tool_action)

        self.polygon_tool_action = QAction(QIcon(polygon_icon_path), "Polygon", self)
        self.polygon_tool_action.setCheckable(False)
        self.polygon_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.polygon_tool_action)

        # Create status bar layout
        self.status_bar_layout = QHBoxLayout()

        # Labels for image dimensions and mouse position
        self.image_dimensions_label = QLabel("Image: 0 x 0")
        self.mouse_position_label = QLabel("Mouse: X: 0, Y: 0")

        # Set fixed width for labels to prevent them from resizing
        self.image_dimensions_label.setFixedWidth(150)
        self.mouse_position_label.setFixedWidth(150)

        # Transparency slider
        self.transparency_slider = QSlider(Qt.Horizontal)
        self.transparency_slider.setRange(0, 255)
        self.transparency_slider.setValue(128)  # Default transparency
        self.transparency_slider.valueChanged.connect(self.update_annotation_transparency)

        # Spin box for annotation size control
        self.annotation_size_spinbox = QSpinBox()
        self.annotation_size_spinbox.setMinimum(1)
        self.annotation_size_spinbox.setMaximum(1000)  # Adjust as needed
        self.annotation_size_spinbox.setValue(self.annotation_window.annotation_size)
        self.annotation_size_spinbox.valueChanged.connect(self.annotation_window.set_annotation_size)
        self.annotation_window.annotationSizeChanged.connect(self.annotation_size_spinbox.setValue)

        # Add widgets to status bar layout
        self.status_bar_layout.addWidget(self.image_dimensions_label)
        self.status_bar_layout.addWidget(self.mouse_position_label)
        self.status_bar_layout.addStretch()
        self.status_bar_layout.addWidget(QLabel("Transparency:"))
        self.status_bar_layout.addWidget(self.transparency_slider)
        self.status_bar_layout.addStretch()
        self.status_bar_layout.addWidget(QLabel("Annotation Size:"))
        self.status_bar_layout.addWidget(self.annotation_size_spinbox)

        self.imported_image_paths = set()  # Set to keep track of imported image paths

        # Create the main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Create left and right layouts
        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        # Add status bar layout to left layout above the AnnotationWindow
        self.left_layout.addLayout(self.status_bar_layout)
        self.left_layout.addWidget(self.annotation_window, 85)
        self.left_layout.addWidget(self.label_window, 15)

        # Add widgets to right layout
        self.right_layout.addWidget(self.image_window, 54)
        self.right_layout.addWidget(self.confidence_window, 46)

        # Add left and right layouts to main layout
        self.main_layout.addLayout(self.left_layout, 85)
        self.main_layout.addLayout(self.right_layout, 15)

        # Set up global event filter
        self.global_event_filter = GlobalEventFilter(self)
        QApplication.instance().installEventFilter(self.global_event_filter)

    def showEvent(self, event):
        super().showEvent(event)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            if self.windowState() & Qt.WindowMinimized:
                # Allow minimizing
                pass
            elif self.windowState() & Qt.WindowMaximized:
                # Window is maximized, do nothing
                pass
            else:
                # Restore to normal state
                pass  # Do nothing, let the OS handle the restore

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def toggle_tool(self, state):
        action = self.sender()
        if action == self.select_tool_action:
            if state:
                self.annotate_tool_action.setChecked(False)
                self.toolChanged.emit("select")
            else:
                self.toolChanged.emit(None)
        elif action == self.annotate_tool_action:
            if state:
                self.select_tool_action.setChecked(False)
                self.toolChanged.emit("annotate")
            else:
                self.toolChanged.emit(None)

    def handle_tool_changed(self, tool):
        if tool == "select":
            self.select_tool_action.setChecked(True)
            self.annotate_tool_action.setChecked(False)
        elif tool == "annotate":
            self.select_tool_action.setChecked(False)
            self.annotate_tool_action.setChecked(True)
        else:
            self.select_tool_action.setChecked(False)
            self.annotate_tool_action.setChecked(False)

    def update_image_dimensions(self, width, height):
        self.image_dimensions_label.setText(f"Image: {width} x {height}")

    def update_mouse_position(self, x, y):
        self.mouse_position_label.setText(f"Mouse: X: {x}, Y: {y}")

    def update_annotation_transparency(self, value):
        if self.annotation_window.selected_label:
            self.annotation_window.update_annotations_transparency(self.annotation_window.selected_label, value)

    def import_images(self):
        file_names, _ = QFileDialog.getOpenFileNames(self, "Open Image Files", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_names:
            progress_bar = ProgressBar(self, title="Importing Images")
            progress_bar.show()
            progress_bar.start_progress(len(file_names))

            for i, file_name in enumerate(file_names):
                if file_name not in self.imported_image_paths:
                    self.image_window.add_image(file_name)
                    self.imported_image_paths.add(file_name)
                    self.annotation_window.loaded_image_paths.add(file_name)
                    progress_bar.update_progress()
                    QApplication.processEvents()  # Update GUI

            progress_bar.stop_progress()
            progress_bar.close()

            if file_names:
                # Load the last image
                image_path = file_names[-1]
                self.image_window.load_image_by_path(image_path)

            QMessageBox.information(self,
                                    "Image(s) Imported",
                                    "Image(s) have been successfully exported.")

    def open_annotation_sampling_dialog(self):

        # Check if there are any loaded images
        if not self.annotation_window.loaded_image_paths:
            # Show a message box if no images are loaded
            QMessageBox.warning(self,
                                "No Images Loaded",
                                "Please load images into the project before sampling annotations.")
            return

        try:
            # Proceed to open the dialog if images are loaded
            self.annotation_sampling_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_create_dataset_dialog(self):
        # Check if there are loaded images
        if not len(self.annotation_window.loaded_image_paths):
            QMessageBox.warning(self,
                                "Create Dataset",
                                "No images are present in the project.")
            return

        # Check if there are annotations
        if not len(self.annotation_window.annotations_dict):
            QMessageBox.warning(self,
                                "Create Dataset",
                                "No annotations are present in the project.")
            return

        try:
            self.create_dataset_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_train_model_dialog(self):
        try:
            self.train_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_optimize_model_dialog(self):
        try:
            self.optimize_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_deploy_model_dialog(self):
        try:
            self.deploy_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")


class CreateDatasetDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.annotation_window = main_window.annotation_window
        self.image_window = main_window.image_window

        self.setWindowTitle("Create Dataset")
        self.layout = QVBoxLayout(self)

        # Create tabs
        self.tabs = QTabWidget()
        self.tab_classification = QWidget()
        self.tab_segmentation = QWidget()  # Future work

        self.tabs.addTab(self.tab_classification, "Image Classification")
        self.tabs.addTab(self.tab_segmentation, "Instance Segmentation")

        # Setup classification tab
        self.setup_classification_tab()
        # Setup segmentation tab
        self.setup_segmentation_tab()
        # Add the tabs to the layout
        self.layout.addWidget(self.tabs)

        # Add OK and Cancel buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

        # Connect signals to update summary statistics
        self.train_ratio_spinbox.valueChanged.connect(self.update_summary_statistics)
        self.val_ratio_spinbox.valueChanged.connect(self.update_summary_statistics)
        self.test_ratio_spinbox.valueChanged.connect(self.update_summary_statistics)
        self.class_filter_list.itemChanged.connect(self.update_summary_statistics)

        self.selected_labels = []
        self.selected_annotations = []

    def setup_classification_tab(self):
        layout = QVBoxLayout()

        # Dataset Name and Output Directory
        self.dataset_name_edit = QLineEdit()
        self.output_dir_edit = QLineEdit()
        self.output_dir_button = QPushButton("Browse...")
        self.output_dir_button.clicked.connect(self.browse_output_dir)

        form_layout = QFormLayout()
        form_layout.addRow("Dataset Name:", self.dataset_name_edit)
        form_layout.addRow("Output Directory:", self.output_dir_edit)
        form_layout.addRow(self.output_dir_button)

        layout.addLayout(form_layout)

        # Split Ratios
        split_layout = QHBoxLayout()
        self.train_ratio_spinbox = QDoubleSpinBox()
        self.train_ratio_spinbox.setRange(0.0, 1.0)
        self.train_ratio_spinbox.setSingleStep(0.1)
        self.train_ratio_spinbox.setValue(0.7)

        self.val_ratio_spinbox = QDoubleSpinBox()
        self.val_ratio_spinbox.setRange(0.0, 1.0)
        self.val_ratio_spinbox.setSingleStep(0.1)
        self.val_ratio_spinbox.setValue(0.2)

        self.test_ratio_spinbox = QDoubleSpinBox()
        self.test_ratio_spinbox.setRange(0.0, 1.0)
        self.test_ratio_spinbox.setSingleStep(0.1)
        self.test_ratio_spinbox.setValue(0.1)

        split_layout.addWidget(QLabel("Train Ratio:"))
        split_layout.addWidget(self.train_ratio_spinbox)
        split_layout.addWidget(QLabel("Validation Ratio:"))
        split_layout.addWidget(self.val_ratio_spinbox)
        split_layout.addWidget(QLabel("Test Ratio:"))
        split_layout.addWidget(self.test_ratio_spinbox)

        layout.addLayout(split_layout)

        # Label Code Selection
        self.label_code_combo = QComboBox()
        self.label_code_combo.addItems(["Short Label Codes", "Long Label Codes"])
        self.label_code_combo.currentTextChanged.connect(self.update_class_filter_list)
        self.label_code_combo.currentTextChanged.connect(self.update_summary_statistics)
        self.label_code_combo.setCurrentText("Short Label Codes")
        layout.addWidget(QLabel("Select Label Code Type:"))
        layout.addWidget(self.label_code_combo)

        # Class Filtering
        self.class_filter_group = QGroupBox("Class Filtering")
        self.class_filter_layout = QVBoxLayout()
        self.class_filter_list = QListWidget()
        self.class_filter_layout.addWidget(self.class_filter_list)
        self.class_filter_group.setLayout(self.class_filter_layout)
        layout.addWidget(self.class_filter_group)

        # Summary Statistics
        self.summary_label = QLabel()
        layout.addWidget(self.summary_label)

        # Ready Status
        self.ready_label = QLabel()
        layout.addWidget(self.ready_label)

        # Add the layout
        self.tab_classification.setLayout(layout)

        # Populate class filter list
        self.populate_class_filter_list()
        # Initial update of summary statistics
        self.update_summary_statistics()

    def setup_segmentation_tab(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Instance Segmentation tab (Future Work)"))
        self.tab_segmentation.setLayout(layout)

    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def get_selected_label_code_type(self):
        return self.label_code_combo.currentText()

    def update_class_filter_list(self):
        self.populate_class_filter_list()

    def populate_class_filter_list(self):
        self.class_filter_list.clear()
        label_code_type = self.get_selected_label_code_type()

        if label_code_type == 'Short Label Codes':
            unique_labels = set(a.label.short_label_code for a in self.annotation_window.annotations_dict.values())
        else:
            unique_labels = set(a.label.long_label_code for a in self.annotation_window.annotations_dict.values())

        for label in unique_labels:
            if label != 'Review':
                item = QListWidgetItem(label)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                self.class_filter_list.addItem(item)

    def split_data(self):
        self.train_ratio = self.train_ratio_spinbox.value()
        self.val_ratio = self.val_ratio_spinbox.value()
        self.test_ratio = self.test_ratio_spinbox.value()

        images = list(self.annotation_window.loaded_image_paths)
        random.shuffle(images)

        train_split = int(len(images) * self.train_ratio)
        val_split = int(len(images) * (self.train_ratio + self.val_ratio))

        self.train_images = images[:train_split]
        self.val_images = images[train_split:val_split]
        self.test_images = images[val_split:]

    def determine_splits(self):
        self.train_annotations = [a for a in self.selected_annotations if a.image_path in self.train_images]
        self.val_annotations = [a for a in self.selected_annotations if a.image_path in self.val_images]
        self.test_annotations = [a for a in self.selected_annotations if a.image_path in self.test_images]

    def check_label_distribution(self):
        for label in self.selected_labels:
            if self.get_selected_label_code_type() == "Short Label Codes":
                train_label_count = sum(1 for a in self.train_annotations if a.label.short_label_code == label)
                val_label_count = sum(1 for a in self.val_annotations if a.label.short_label_code == label)
                test_label_count = sum(1 for a in self.test_annotations if a.label.short_label_code == label)
            else:
                train_label_count = sum(1 for a in self.train_annotations if a.label.long_label_code == label)
                val_label_count = sum(1 for a in self.val_annotations if a.label.long_label_code == label)
                test_label_count = sum(1 for a in self.test_annotations if a.label.long_label_code == label)

            if train_label_count == 0 or val_label_count == 0 or test_label_count == 0:
                return False
        return True

    def update_summary_statistics(self):
        # Split the data by images
        self.split_data()

        # Selected labels based on user's selection
        matching_items = self.class_filter_list.findItems("", Qt.MatchContains)
        self.selected_labels = [item.text() for item in matching_items if item.checkState() == Qt.Checked]

        # All annotations in project
        annotations = list(self.annotation_window.annotations_dict.values())

        if self.get_selected_label_code_type() == "Short Label Codes":
            self.selected_annotations = [a for a in annotations if a.label.short_label_code in self.selected_labels]
        else:
            self.selected_annotations = [a for a in annotations if a.label.long_label_code in self.selected_labels]

        total_annotations = len(self.selected_annotations)
        total_classes = len(self.selected_labels)

        # Split the data by annotations
        self.determine_splits()

        train_count = len(self.train_annotations)
        val_count = len(self.val_annotations)
        test_count = len(self.test_annotations)

        self.summary_label.setText(f"Total Annotations: {total_annotations}\n"
                                   f"Total Classes: {total_classes}\n"
                                   f"Train Samples: {train_count}\n"
                                   f"Validation Samples: {val_count}\n"
                                   f"Test Samples: {test_count}")

        self.ready_status = self.check_label_distribution()
        self.split_status = abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-9
        self.ready_label.setText("Ready" if (self.ready_status and self.split_status) else "Not Ready")

    def accept(self):
        dataset_name = self.dataset_name_edit.text()
        output_dir = self.output_dir_edit.text()
        train_ratio = self.train_ratio_spinbox.value()
        val_ratio = self.val_ratio_spinbox.value()
        test_ratio = self.test_ratio_spinbox.value()

        if not self.ready_status:
            QMessageBox.warning(self, "Dataset Not Ready",
                                "Not all labels are present in all sets.\n"
                                "Please adjust your selections or sample more data.")
            return

        if not dataset_name or not output_dir:
            QMessageBox.warning(self, "Input Error", "All fields must be filled.")
            return

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
            QMessageBox.warning(self, "Input Error", "Train, Validation, and Test ratios must sum to 1.0")
            return

        output_dir_path = os.path.join(output_dir, dataset_name)
        if os.path.exists(output_dir_path):
            reply = QMessageBox.question(self,
                                         "Directory Exists",
                                         "The output directory already exists. Do you want to merge the datasets?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return

        os.makedirs(output_dir_path, exist_ok=True)
        train_dir = os.path.join(output_dir_path, 'train')
        val_dir = os.path.join(output_dir_path, 'val')
        test_dir = os.path.join(output_dir_path, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for label in self.selected_labels:
            os.makedirs(os.path.join(train_dir, label), exist_ok=True)
            os.makedirs(os.path.join(val_dir, label), exist_ok=True)
            os.makedirs(os.path.join(test_dir, label), exist_ok=True)

        # Set the cursor to waiting (busy) cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        self.process_annotations(self.train_annotations, train_dir, "Training")
        self.process_annotations(self.val_annotations, val_dir, "Validation")
        self.process_annotations(self.test_annotations, test_dir, "Testing")

        # Restore the cursor to the default cursor
        QApplication.restoreOverrideCursor()

        QMessageBox.information(self,
                                "Dataset Created",
                                "Dataset has been successfully created.")
        super().accept()

    def process_annotations(self, annotations, split_dir, split):
        progress_bar = ProgressBar(self, title=f"Creating {split} Dataset")
        progress_bar.show()

        progress_bar.start_progress(len(annotations))

        for annotation in annotations:
            if progress_bar.wasCanceled():
                break

            if not annotation.cropped_image:
                image = self.image_window.images[annotation.image_path]
                image_item = QGraphicsPixmapItem(QPixmap(image))
                annotation.create_cropped_image(image_item)

            cropped_image = annotation.cropped_image

            if cropped_image:
                if self.get_selected_label_code_type() == "Short Label Codes":
                    label_code = annotation.label.short_label_code
                else:
                    label_code = annotation.label.long_label_code

                output_path = os.path.join(split_dir, label_code)
                output_filename = f"{label_code}_{annotation.id}.jpg"
                cropped_image.save(os.path.join(output_path, output_filename))

            progress_bar.update_progress()
            QApplication.processEvents()  # Update GUI

        progress_bar.stop_progress()
        progress_bar.close()

    def showEvent(self, event):
        super().showEvent(event)
        self.populate_class_filter_list()
        self.update_summary_statistics()


class TrainModelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        # For holding parameters
        self.custom_params = []

        self.setWindowTitle("Train Model")

        # Set window settings
        self.setWindowFlags(Qt.Window |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint |
                            Qt.WindowTitleHint)

        self.resize(600, 800)

        # Main layout
        self.main_layout = QVBoxLayout()

        # Create and set up the tabs, parameters form, and console output
        self.setup_ui()

        # Wrap the main layout in a QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_widget.setLayout(self.main_layout)
        scroll_area.setWidget(scroll_widget)

        # Set the scroll area as the main layout of the dialog
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scroll_area)

    def setup_ui(self):

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Details on different hyperparameters can be found "
                            "<a href='https://docs.ultralytics.com/modes/train/#train-settings'>here</a>.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)

        # Add the label to the main layout
        self.main_layout.addWidget(info_label)

        # Create tabs
        self.tabs = QTabWidget()
        self.tab_classification = QWidget()
        self.tab_segmentation = QWidget()

        self.tabs.addTab(self.tab_classification, "Image Classification")
        self.tabs.addTab(self.tab_segmentation, "Instance Segmentation")

        # Setup tabs
        self.setup_classification_tab()
        self.setup_segmentation_tab()
        # Add to main layout
        self.main_layout.addWidget(self.tabs)

        # Parameters Form
        self.form_layout = QFormLayout()

        # Model
        self.model_edit = QLineEdit()
        self.model_button = QPushButton("Browse...")
        self.model_button.clicked.connect(self.browse_model_file)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(self.model_button)
        self.form_layout.addRow("Existing Model:", model_layout)

        # Epochs
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setMinimum(1)
        self.epochs_spinbox.setMaximum(1000)
        self.epochs_spinbox.setValue(100)
        self.form_layout.addRow("Epochs:", self.epochs_spinbox)

        # Patience
        self.patience_spinbox = QSpinBox()
        self.patience_spinbox.setMinimum(1)
        self.patience_spinbox.setMaximum(1000)
        self.patience_spinbox.setValue(100)
        self.form_layout.addRow("Patience:", self.patience_spinbox)

        # Batch
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setMinimum(-1)
        self.batch_spinbox.setMaximum(1024)
        self.batch_spinbox.setValue(16)
        self.form_layout.addRow("Batch:", self.batch_spinbox)

        # Imgsz
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setMinimum(16)
        self.imgsz_spinbox.setMaximum(4096)
        self.imgsz_spinbox.setValue(224)
        self.form_layout.addRow("Imgsz:", self.imgsz_spinbox)

        # Save
        self.save_checkbox = QCheckBox()
        self.save_checkbox.setChecked(True)
        self.form_layout.addRow("Save:", self.save_checkbox)

        # Save Period
        self.save_period_spinbox = QSpinBox()
        self.save_period_spinbox.setMinimum(-1)
        self.save_period_spinbox.setMaximum(1000)
        self.save_period_spinbox.setValue(-1)
        self.form_layout.addRow("Save Period:", self.save_period_spinbox)

        # Workers
        self.workers_spinbox = QSpinBox()
        self.workers_spinbox.setMinimum(1)
        self.workers_spinbox.setMaximum(64)
        self.workers_spinbox.setValue(8)
        self.form_layout.addRow("Workers:", self.workers_spinbox)

        # Exist Ok
        self.exist_ok_checkbox = QCheckBox()
        self.exist_ok_checkbox.setChecked(False)
        self.form_layout.addRow("Exist Ok:", self.exist_ok_checkbox)

        # Pretrained
        self.pretrained_checkbox = QCheckBox()
        self.pretrained_checkbox.setChecked(True)
        self.form_layout.addRow("Pretrained:", self.pretrained_checkbox)

        # Optimizer
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"])
        self.optimizer_combo.setCurrentText("auto")
        self.form_layout.addRow("Optimizer:", self.optimizer_combo)

        # Verbose
        self.verbose_checkbox = QCheckBox()
        self.verbose_checkbox.setChecked(True)
        self.form_layout.addRow("Verbose:", self.verbose_checkbox)

        # Fraction
        self.fraction_spinbox = QDoubleSpinBox()
        self.fraction_spinbox.setMinimum(0.1)
        self.fraction_spinbox.setMaximum(1.0)
        self.fraction_spinbox.setValue(1.0)
        self.form_layout.addRow("Fraction:", self.fraction_spinbox)

        # Freeze
        self.freeze_edit = QLineEdit()
        self.form_layout.addRow("Freeze:", self.freeze_edit)

        # Lr0
        self.lr0_spinbox = QDoubleSpinBox()
        self.lr0_spinbox.setMinimum(0.0001)
        self.lr0_spinbox.setMaximum(1.0)
        self.lr0_spinbox.setValue(0.01)
        self.form_layout.addRow("Lr0:", self.lr0_spinbox)

        # Dropout
        self.dropout_spinbox = QDoubleSpinBox()
        self.dropout_spinbox.setMinimum(0.0)
        self.dropout_spinbox.setMaximum(1.0)
        self.dropout_spinbox.setValue(0.0)
        self.form_layout.addRow("Dropout:", self.dropout_spinbox)

        # Val
        self.val_checkbox = QCheckBox()
        self.val_checkbox.setChecked(True)
        self.form_layout.addRow("Val:", self.val_checkbox)

        # Add custom parameters section
        self.custom_params_layout = QVBoxLayout()
        self.form_layout.addRow("Additional Parameters:", self.custom_params_layout)

        # Add button for new parameter pairs
        self.add_param_button = QPushButton("Add Parameter")
        self.add_param_button.clicked.connect(self.add_parameter_pair)
        self.form_layout.addRow("", self.add_param_button)

        self.main_layout.addLayout(self.form_layout)

        # Add OK and Cancel buttons
        self.buttons = QPushButton("OK")
        self.buttons.clicked.connect(self.accept)
        self.main_layout.addWidget(self.buttons)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.main_layout.addWidget(self.cancel_button)

    def add_parameter_pair(self):
        param_layout = QHBoxLayout()
        param_name = QLineEdit()
        param_value = QLineEdit()
        param_layout.addWidget(param_name)
        param_layout.addWidget(param_value)

        self.custom_params.append((param_name, param_value))
        self.custom_params_layout.addLayout(param_layout)

    def browse_dataset_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if dir_path:
            self.dataset_dir_edit.setText(dir_path)

    def browse_dataset_yaml(self):
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Select Dataset YAML File",
                                                   "",
                                                   "YAML Files (*.yaml *.yml)")
        if file_path:
            self.dataset_yaml_edit.setText(file_path)

    def browse_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File")
        if file_path:
            self.model_edit.setText(file_path)

    def setup_classification_tab(self):
        layout = QVBoxLayout()

        # Dataset Directory
        self.dataset_dir_edit = QLineEdit()
        self.dataset_dir_button = QPushButton("Browse...")
        self.dataset_dir_button.clicked.connect(self.browse_dataset_dir)

        dataset_dir_layout = QHBoxLayout()
        dataset_dir_layout.addWidget(QLabel("Dataset Directory:"))
        dataset_dir_layout.addWidget(self.dataset_dir_edit)
        dataset_dir_layout.addWidget(self.dataset_dir_button)
        layout.addLayout(dataset_dir_layout)

        # Classification Model Dropdown
        self.classification_model_combo = QComboBox()
        self.classification_model_combo.addItems(["yolov8n-cls.pt",
                                                  "yolov8s-cls.pt",
                                                  "yolov8m-cls.pt",
                                                  "yolov8l-cls.pt",
                                                  "yolov8x-cls.pt"])

        self.classification_model_combo.setEditable(True)
        layout.addWidget(QLabel("Select or Enter Classification Model:"))
        layout.addWidget(self.classification_model_combo)

        self.tab_classification.setLayout(layout)

    def setup_segmentation_tab(self):
        layout = QVBoxLayout()

        self.dataset_yaml_edit = QLineEdit()
        self.dataset_yaml_button = QPushButton("Browse...")
        self.dataset_yaml_button.clicked.connect(self.browse_dataset_yaml)

        dataset_yaml_layout = QHBoxLayout()
        dataset_yaml_layout.addWidget(QLabel("Dataset YAML:"))
        dataset_yaml_layout.addWidget(self.dataset_yaml_edit)
        dataset_yaml_layout.addWidget(self.dataset_yaml_button)
        layout.addLayout(dataset_yaml_layout)

        # Segmentation Model Dropdown
        self.segmentation_model_combo = QComboBox()
        self.segmentation_model_combo.addItems(["yolov8n-seg.pt",
                                                "yolov8s-seg.pt",
                                                "yolov8m-seg.pt",
                                                "yolov8l-seg.pt",
                                                "yolov8x-seg.pt"])

        self.segmentation_model_combo.setEditable(True)
        layout.addWidget(QLabel("Select or Enter Segmentation Model:"))
        layout.addWidget(self.segmentation_model_combo)

        self.tab_segmentation.setLayout(layout)

    def accept(self):
        self.train_classification_model()
        super().accept()

    def get_training_parameters(self):
        # Extract values from dialog widgets
        params = {
            'model': self.model_edit.text(),
            'data': self.dataset_dir_edit.text(),
            'epochs': self.epochs_spinbox.value(),
            'patience': self.patience_spinbox.value(),
            'batch': self.batch_spinbox.value(),
            'imgsz': self.imgsz_spinbox.value(),
            'save': self.save_checkbox.isChecked(),
            'save_period': self.save_period_spinbox.value(),
            'workers': self.workers_spinbox.value(),
            'exist_ok': self.exist_ok_checkbox.isChecked(),
            'pretrained': self.pretrained_checkbox.isChecked(),
            'optimizer': self.optimizer_combo.currentText(),
            'verbose': self.verbose_checkbox.isChecked(),
            'fraction': self.fraction_spinbox.value(),
            'freeze': self.freeze_edit.text(),
            'lr0': self.lr0_spinbox.value(),
            'dropout': self.dropout_spinbox.value(),
            'val': self.val_checkbox.isChecked(),
            'project': "Data/Training",
        }
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d_%H-%M-%S")
        params['name'] = now

        # Add custom parameters
        for param_name, param_value in self.custom_params:
            name = param_name.text().strip()
            value = param_value.text().strip().lower()
            if name:
                if value == 'true':
                    params[name] = True
                elif value == 'false':
                    params[name] = False
                else:
                    try:
                        params[name] = int(value)
                    except ValueError:
                        try:
                            params[name] = float(value)
                        except ValueError:
                            params[name] = value

        # Return the dictionary of parameters
        return params

    def train_classification_model(self):

        message = "Model training has commenced.\nMonitor the console for real-time progress."
        QMessageBox.information(self, "Model Training Status", message)

        # Minimization of windows
        self.showMinimized()
        self.parent().showMinimized()

        # Get training parameters
        params = self.get_training_parameters()

        try:
            # Initialize the model
            params['task'] = 'classify'
            model_path = params.pop('model', None)
            if not model_path:
                model_path = self.classification_model_combo.currentText()

            self.target_model = YOLO(model_path)

            # Train the model
            results = self.target_model.train(**params)

            # Restore the window after training is complete
            self.showNormal()

            message = "Model training has successfully been completed."
            QMessageBox.information(self, "Model Training Status", message)

        except Exception as e:
            # Restore the window after training is complete
            self.showNormal()

            # Display an error message box to the user
            error_message = f"An error occurred during model training: {e}"
            QMessageBox.critical(self, "Error", error_message)
            print(error_message)


class OptimizeModelDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        self.custom_params = []

        self.setWindowTitle("Optimize Model")
        self.resize(300, 200)

        self.layout = QVBoxLayout(self)

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Details on different production formats can be found "
                            "<a href='https://docs.ultralytics.com/modes/export/#export-formats'>here</a>.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        self.layout.addWidget(info_label)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_file)
        self.layout.addWidget(browse_button)

        self.model_text_area = QTextEdit("No model file selected")
        self.model_text_area.setReadOnly(True)
        self.layout.addWidget(self.model_text_area)

        # Export Format Dropdown
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["torchscript",
                                           "onnx",
                                           "openvino",
                                           "engine"])

        self.export_format_combo.setEditable(True)
        self.layout.addWidget(QLabel("Select or Enter Export Format:"))
        self.layout.addWidget(self.export_format_combo)

        # Parameters Form
        self.form_layout = QFormLayout()

        # Add custom parameters section
        self.custom_params_layout = QVBoxLayout()
        self.form_layout.addRow("Parameters:", self.custom_params_layout)

        # Add button for new parameter pairs
        self.add_param_button = QPushButton("Add Parameter")
        self.add_param_button.clicked.connect(self.add_parameter_pair)
        self.form_layout.addRow("", self.add_param_button)

        self.layout.addLayout(self.form_layout)

        accept_button = QPushButton("Accept")
        accept_button.clicked.connect(self.optimize_model)
        self.layout.addWidget(accept_button)

        self.setLayout(self.layout)

    def add_parameter_pair(self):
        param_layout = QHBoxLayout()
        param_name = QLineEdit()
        param_value = QLineEdit()
        param_layout.addWidget(param_name)
        param_layout.addWidget(param_value)

        self.custom_params.append((param_name, param_value))
        self.custom_params_layout.addLayout(param_layout)

    def browse_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Open Model File", "",
                                                   "Model Files (*.pt)", options=options)
        if file_path:
            self.model_path = file_path
            self.model_text_area.setText("Model file selected")

    def accept(self):
        self.optimize_model()
        super().accept()

    def get_optimization_parameters(self):
        # Extract values from dialog widgets
        params = {'format': self.export_format_combo.currentText()}

        for param_name, param_value in self.custom_params:
            name = param_name.text().strip()
            value = param_value.text().strip().lower()
            if name:
                if value == 'true':
                    params[name] = True
                elif value == 'false':
                    params[name] = False
                else:
                    try:
                        params[name] = int(value)
                    except ValueError:
                        try:
                            params[name] = float(value)
                        except ValueError:
                            params[name] = value

        # Return the dictionary of parameters
        return params

    def optimize_model(self):

        # Get training parameters
        params = self.get_optimization_parameters()

        # Set the cursor to waiting (busy) cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # Initialize the model, export given params
            YOLO(self.model_path).export(**params)

            message = "Model export successful."
            QMessageBox.information(self, "Model Export Status", message)

        except Exception as e:

            # Display an error message box to the user
            error_message = f"An error occurred when converting model: {e}"
            QMessageBox.critical(self, "Error", error_message)
            print(error_message)

        # Restore the cursor to the default cursor
        QApplication.restoreOverrideCursor()


class DeployModelDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

        self.setWindowTitle("Deploy Model")
        self.resize(300, 200)

        self.layout = QVBoxLayout(self)

        self.model_path = None
        self.loaded_model = None

        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)

        self.classification_tab = QWidget()
        self.segmentation_tab = QWidget()

        self.tab_widget.addTab(self.classification_tab, "Image Classification")
        self.tab_widget.addTab(self.segmentation_tab, "Instance Segmentation")

        self.init_classification_tab()
        self.init_segmentation_tab()

        self.status_bar = QLabel("No model loaded")
        self.layout.addWidget(self.status_bar)

        self.setLayout(self.layout)

    def init_classification_tab(self):
        layout = QVBoxLayout()

        self.classification_text_area = QTextEdit()
        self.classification_text_area.setReadOnly(True)
        layout.addWidget(self.classification_text_area)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_file)
        layout.addWidget(browse_button)

        load_button = QPushButton("Load Model")
        load_button.clicked.connect(self.load_model)
        layout.addWidget(load_button)

        deactivate_button = QPushButton("Deactivate Model")
        deactivate_button.clicked.connect(self.deactivate_model)
        layout.addWidget(deactivate_button)

        self.classification_tab.setLayout(layout)

    def init_segmentation_tab(self):
        layout = QVBoxLayout()

        self.segmentation_text_area = QTextEdit()
        self.segmentation_text_area.setReadOnly(True)
        layout.addWidget(self.segmentation_text_area)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_file)
        layout.addWidget(browse_button)

        load_button = QPushButton("Load Model")
        load_button.clicked.connect(self.load_model)
        layout.addWidget(load_button)

        deactivate_button = QPushButton("Deactivate Model")
        deactivate_button.clicked.connect(self.deactivate_model)
        layout.addWidget(deactivate_button)

        self.segmentation_tab.setLayout(layout)

    def browse_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Open Model File", "",
                                                   "Model Files (*.pt *.onnx *.torchscript *.engine *.bin)",
                                                   options=options)
        if file_path:
            if ".bin" in file_path:
                # OpenVINO is a directory
                file_path = os.path.dirname(file_path)

            self.model_path = file_path
            if self.tab_widget.currentIndex() == 0:
                self.classification_text_area.setText("Model file selected")
            else:
                self.segmentation_file_path.setText("Model file selected")

    def load_model(self):
        if self.model_path:
            try:
                # Set the cursor to waiting (busy) cursor
                QApplication.setOverrideCursor(Qt.WaitCursor)

                self.loaded_model = YOLO(self.model_path, task='classify')
                self.loaded_model(np.zeros((224, 224, 3), dtype=np.uint8))

                # Get the class names the model can predict
                class_names = list(self.loaded_model.names.values())
                class_names_str = "Class Names: \n"
                missing_labels = []

                for class_name in class_names:
                    class_names_str += f"{class_name} \n"
                    if not self.label_window.get_label_by_long_code(class_name):
                        missing_labels.append(class_name)

                self.classification_text_area.setText(class_names_str)
                self.status_bar.setText(f"Model loaded: {os.path.basename(self.model_path)}")

                if missing_labels:
                    missing_labels_str = "\n".join(missing_labels)
                    QMessageBox.warning(self,
                                        "Warning",
                                        f"The following classes are missing and cannot be predicted:"
                                        f"\n{missing_labels_str}")

                QMessageBox.information(self, "Model Loaded", "Model weights loaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            finally:
                # Restore the cursor to the default cursor
                QApplication.restoreOverrideCursor()
        else:
            QMessageBox.warning(self, "Warning", "No model file selected")

    def deactivate_model(self):
        self.loaded_model = None
        self.model_path = None
        self.status_bar.setText("No model loaded")
        if self.tab_widget.currentIndex() == 0:
            self.classification_text_area.setText("No model file selected")
        else:
            self.segmentation_file_path.setText("No model file selected")

    def pixmap_to_numpy(self, pixmap):
        # Convert QPixmap to QImage
        image = pixmap.toImage()
        # Get image dimensions
        width = image.width()
        height = image.height()

        # Convert QImage to numpy array
        byte_array = image.bits().asstring(width * height * 4)  # 4 for RGBA
        numpy_array = np.frombuffer(byte_array, dtype=np.uint8).reshape((height, width, 4))

        # If the image format is ARGB32, swap the first and last channels (A and B)
        if format == QImage.Format_ARGB32:
            numpy_array = numpy_array[:, :, [2, 1, 0, 3]]

        return numpy_array

    def predict(self, annotations=None):
        # If model isn't loaded
        if self.loaded_model is None:
            return

        # Set the cursor to waiting (busy) cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Get the selected annotation
        selected_annotation = self.annotation_window.selected_annotation

        if selected_annotation:
            # Make predictions on specific annotation
            self.predict_annotation(selected_annotation)
            # Update everything (essentially)
            self.main_window.annotation_window.unselect_annotation()
            self.main_window.annotation_window.select_annotation(selected_annotation)
        else:
            # Run predictions on multiple annotations
            if not annotations:
                # If not supplied with annotations, get those for current image
                annotations = self.annotation_window.get_image_annotations()

            # Filter annotations to only include those with 'Review' label
            review_annotations = [annotation for annotation in annotations if annotation.label.id == "-1"]

            if review_annotations:
                # Convert QImages to numpy arrays
                images_np = [self.pixmap_to_numpy(annotation.cropped_image) for annotation in review_annotations]

                # Perform batch prediction
                results = self.loaded_model(images_np)

                for annotation, result in zip(review_annotations, results):
                    self.process_prediction_result(annotation, result)

        # Restore the cursor to the default cursor
        QApplication.restoreOverrideCursor()

    def predict_annotation(self, annotation):
        # Get the cropped image
        image = annotation.cropped_image
        # Convert QImage to np
        image_np = self.pixmap_to_numpy(image)
        # Perform prediction
        results = self.loaded_model(image_np)[0]

        # Extract the results
        class_names = results.names
        top5 = results.probs.top5
        top5conf = results.probs.top5conf

        # Initialize an empty dictionary to store the results
        predictions = {}

        # Iterate over the top 5 predictions
        for idx, conf in zip(top5, top5conf):
            class_name = class_names[idx]
            label = self.label_window.get_label_by_long_code(class_name)

            if label:
                predictions[label] = float(conf)
            else:
                # Users does not have label loaded; skip.
                pass

        if predictions:
            # Update the machine confidence
            annotation.update_machine_confidence(predictions)

    def process_prediction_result(self, annotation, result):
        # Extract the results
        class_names = result.names
        top5 = result.probs.top5
        top5conf = result.probs.top5conf

        # Initialize an empty dictionary to store the results
        predictions = {}

        # Iterate over the top 5 predictions
        for idx, conf in zip(top5, top5conf):
            class_name = class_names[idx]
            label = self.label_window.get_label_by_long_code(class_name)

            if label:
                predictions[label] = float(conf)
            else:
                # User does not have label loaded; skip.
                pass

        if predictions:
            # Update the machine confidence
            annotation.update_machine_confidence(predictions)


class Annotation(QObject):
    color_changed = pyqtSignal(QColor)
    selected = pyqtSignal(object)
    annotation_deleted = pyqtSignal(object)
    annotation_updated = pyqtSignal(object)

    def __init__(self, center_xy: QPointF,
                 annotation_size: int,
                 short_label_code: str,
                 long_label_code: str,
                 color: QColor,
                 image_path: str,
                 label_id: str,
                 transparency: int = 128,
                 show_msg=True):
        super().__init__()
        self.id = str(uuid.uuid4())
        self.center_xy = center_xy
        self.annotation_size = annotation_size
        self.label = Label(short_label_code, long_label_code, color, label_id)
        self.image_path = image_path
        self.is_selected = False
        self.graphics_item = None
        self.transparency = transparency
        self.user_confidence = {self.label: 1.0}
        self.machine_confidence = {}
        self.cropped_image = None

        self.show_message = show_msg

    def show_warning_message(self):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("Warning")
        msg_box.setText("Altering an annotation with predictions will remove the machine suggestions.")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

        # Only show once
        self.show_message = False

    def contains_point(self, point: QPointF) -> bool:
        half_size = self.annotation_size / 2
        rect = QRectF(self.center_xy.x() - half_size,
                      self.center_xy.y() - half_size,
                      self.annotation_size,
                      self.annotation_size)
        return rect.contains(point)

    def select(self):
        self.is_selected = True
        self.update_graphics_item()

    def deselect(self):
        self.is_selected = False
        self.update_graphics_item()

    def delete(self):
        self.annotation_deleted.emit(self)
        if self.graphics_item:
            self.graphics_item.scene().removeItem(self.graphics_item)
            self.graphics_item = None

    def create_cropped_image(self, image_item: QGraphicsPixmapItem):

        pixmap = image_item.pixmap()
        half_size = self.annotation_size / 2
        rect = QRectF(self.center_xy.x() - half_size,
                      self.center_xy.y() - half_size,
                      self.annotation_size,
                      self.annotation_size).toRect()
        self.cropped_image = pixmap.copy(rect)
        self.annotation_updated.emit(self)  # Notify update

    def create_graphics_item(self, scene: QGraphicsScene):
        half_size = self.annotation_size / 2
        self.graphics_item = QGraphicsRectItem(self.center_xy.x() - half_size,
                                               self.center_xy.y() - half_size,
                                               self.annotation_size,
                                               self.annotation_size)
        self.update_graphics_item()
        self.graphics_item.setData(0, self.id)
        scene.addItem(self.graphics_item)

    def update_machine_confidence(self, prediction: dict):
        # Set user confidence to None
        self.user_confidence = {}
        # Update machine confidence
        self.machine_confidence = prediction
        # Pass the label with the largest confidence as the label
        self.label = max(prediction, key=prediction.get)
        # Create the graphic
        self.update_graphics_item()

    def update_user_confidence(self, new_label: 'Label'):
        # Set machine confidence to None
        self.machine_confidence = {}
        # Update user confidence
        self.user_confidence = {new_label: 1.0}
        # Pass the label with the largest confidence as the label
        self.label = new_label
        # Create the graphic
        self.update_graphics_item()

    def update_label(self, new_label: 'Label'):
        self.label = new_label
        self.update_graphics_item()

    def update_location(self, new_center_xy: QPointF):
        if self.machine_confidence and self.show_message:
            self.show_warning_message()
            return

        # Clear the machine confidence
        self.update_user_confidence(self.label)
        # Update the location, graphic
        self.center_xy = new_center_xy
        self.update_graphics_item()
        self.annotation_updated.emit(self)  # Notify update

    def update_annotation_size(self, size):
        if self.machine_confidence and self.show_message:
            self.show_warning_message()
            return

        # Clear the machine confidence
        self.update_user_confidence(self.label)
        # Update the size, graphic
        self.annotation_size = size
        self.update_graphics_item()
        self.annotation_updated.emit(self)  # Notify update

    def update_transparency(self, transparency: int):
        self.transparency = transparency
        self.update_graphics_item()

    def update_graphics_item(self):
        if self.graphics_item:
            half_size = self.annotation_size / 2
            self.graphics_item.setRect(self.center_xy.x() - half_size,
                                       self.center_xy.y() - half_size,
                                       self.annotation_size,
                                       self.annotation_size)
            color = QColor(self.label.color)
            color.setAlpha(self.transparency)

            if self.is_selected:
                inverse_color = QColor(255 - color.red(), 255 - color.green(), 255 - color.blue())
                pen = QPen(inverse_color, 4, Qt.DotLine)  # Inverse color, thicker border, and dotted line
            else:
                pen = QPen(color, 2, Qt.SolidLine)  # Default border color and thickness

            self.graphics_item.setPen(pen)
            brush = QBrush(color)
            self.graphics_item.setBrush(brush)
            self.graphics_item.update()

    def to_dict(self):
        return {
            'id': self.id,
            'center_xy': (self.center_xy.x(), self.center_xy.y()),
            'annotation_size': self.annotation_size,
            'label_short_code': self.label.short_label_code,
            'label_long_code': self.label.long_label_code,
            'annotation_color': self.label.color.getRgb(),
            'image_path': self.image_path,
            'label_id': self.label.id
        }

    @classmethod
    def from_dict(cls, data):
        return cls(QPointF(*data['center_xy']),
                   data['annotation_size'],
                   data['label_short_code'],
                   data['label_long_code'],
                   QColor(*data['annotation_color']),
                   data['image_path'],
                   data['label_id'])

    def to_coralnet_format(self):
        return [os.path.basename(self.image_path), int(self.center_xy.y()),
                int(self.center_xy.x()), self.label.short_label_code,
                self.label.long_label_code, self.annotation_size]

    def __repr__(self):
        return (f"Annotation(id={self.id}, center_xy={self.center_xy}, "
                f"annotation_size={self.annotation_size}, "
                f"annotation_color={self.label.color.name()}, "
                f"image_path={self.image_path}, "
                f"label={self.label.short_label_code})")


class AnnotationWindow(QGraphicsView):
    imageLoaded = pyqtSignal(int, int)  # Signal to emit when image is loaded
    mouseMoved = pyqtSignal(int, int)  # Signal to emit when mouse is moved
    imageDeleted = pyqtSignal(str)  # Signal to emit when an image is deleted
    toolChanged = pyqtSignal(str)  # Signal to emit when the tool changes
    labelSelected = pyqtSignal(str)  # Signal to emit when the label changes
    annotationSizeChanged = pyqtSignal(int)  # Signal to emit when annotation size changes

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.annotation_size = 224
        self.annotation_color = None
        self.transparency = 128

        self.zoom_factor = 1.0
        self.pan_active = False
        self.pan_start = None
        self.drag_start_pos = None
        self.cursor_annotation = None

        self.annotations_dict = {}  # Dictionary to store annotations by UUID

        self.selected_annotation = None  # Stores the selected annotation
        self.selected_label = None  # Flag to check if an active label is set
        self.selected_tool = None  # Store the current tool state

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.NoDrag)  # Disable default drag mode

        self.image_item = None
        self.active_image = False  # Flag to check if the image has been set
        self.current_image_path = None

        self.loaded_image_paths = set()  # Initialize the set to store loaded image paths

        self.toolChanged.connect(self.set_selected_tool)

    def export_annotations(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self,
                                                   "Save Annotations",
                                                   "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                export_dict = {}
                total_annotations = 0
                for annotation in self.annotations_dict.values():
                    image_path = annotation.image_path
                    if image_path not in export_dict:
                        export_dict[image_path] = []
                    export_dict[image_path].append(annotation.to_dict())
                    total_annotations += 1

                progress_bar = ProgressBar(self, title="Exporting Annotations")
                progress_bar.show()
                progress_bar.start_progress(total_annotations)

                with open(file_path, 'w') as file:
                    json.dump(export_dict, file, indent=4)
                    file.flush()  # Ensure the data is written to the file

                progress_bar.stop_progress()
                progress_bar.close()

                QMessageBox.information(self, "Annotations Exported",
                                        "Annotations have been successfully exported.")

            except Exception as e:
                QMessageBox.warning(self, "Error Exporting Annotations",
                                    f"An error occurred while exporting annotations: {str(e)}")

    def import_annotations(self):
        self.set_selected_tool(None)
        self.toolChanged.emit(None)

        if not self.active_image:
            QMessageBox.warning(self,
                                "No Images Loaded",
                                "Please load images first before importing annotations.")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Load Annotations",
                                                   "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    imported_annotations = json.load(file)

                total_annotations = sum(len(annotations) for annotations in imported_annotations.values())

                progress_bar = ProgressBar(self, title="Importing Annotations")
                progress_bar.show()
                progress_bar.start_progress(total_annotations)

                filtered_annotations = {p: a for p, a in imported_annotations.items() if p in self.loaded_image_paths}

                updated_annotations = False
                for image_path, annotations in filtered_annotations.items():
                    for annotation_data in annotations:
                        short_label_code = annotation_data['label_short_code']
                        long_label_code = annotation_data['label_long_code']
                        color = QColor(*annotation_data['annotation_color'])
                        label_id = annotation_data['label_id']
                        self.main_window.label_window.add_label_if_not_exists(short_label_code,
                                                                              long_label_code,
                                                                              color,
                                                                              label_id)

                        existing_color = self.main_window.label_window.get_label_color(short_label_code,
                                                                                       long_label_code)
                        if existing_color != color:
                            annotation_data['annotation_color'] = existing_color.getRgb()
                            updated_annotations = True

                        progress_bar.update_progress()
                        QApplication.processEvents()  # Update GUI

                if updated_annotations:
                    QMessageBox.information(self,
                                            "Annotations Updated",
                                            "Some annotations have been updated to match the "
                                            "color of the labels already in the project.")

                for image_path, annotations in filtered_annotations.items():
                    for annotation_data in annotations:
                        annotation = Annotation.from_dict(annotation_data)
                        self.annotations_dict[annotation.id] = annotation

                        progress_bar.update_progress()
                        QApplication.processEvents()  # Update GUI

                progress_bar.stop_progress()
                progress_bar.close()

                self.load_annotations()

                QMessageBox.information(self, "Annotations Imported",
                                        "Annotations have been successfully imported.")

            except Exception as e:
                QMessageBox.warning(self, "Error Importing Annotations",
                                    f"An error occurred while importing annotations: {str(e)}")

    def export_coralnet_annotations(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self,
                                                   "Export CoralNet Annotations",
                                                   "",
                                                   "CSV Files (*.csv);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                data = []
                total_annotations = len(self.annotations_dict)

                progress_bar = ProgressBar(self, title="Exporting CoralNet Annotations")
                progress_bar.show()
                progress_bar.start_progress(total_annotations)

                for annotation in self.annotations_dict.values():
                    data.append(annotation.to_coralnet_format())
                    progress_bar.update_progress()
                    QApplication.processEvents()  # Update GUI

                df = pd.DataFrame(data, columns=['Name', 'Row', 'Column', 'Label', 'Long Label', 'Patch Size'])
                df.to_csv(file_path, index=False)

                progress_bar.stop_progress()
                progress_bar.close()

                QMessageBox.information(self, "Annotations Exported",
                                        "Annotations have been successfully exported.")

            except Exception as e:
                QMessageBox.warning(self, "Error Exporting Annotations",
                                    f"An error occurred while exporting annotations: {str(e)}")

    def import_coralnet_annotations(self):
        self.set_selected_tool(None)
        self.toolChanged.emit(None)

        if not self.active_image:
            QMessageBox.warning(self,
                                "No Images Loaded",
                                "Please load images first before importing annotations.")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Import CoralNet Annotations",
                                                   "",
                                                   "CSV Files (*.csv);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                df = pd.read_csv(file_path)
                if not all(col in df.columns for col in ['Name', 'Row', 'Column', 'Label']):
                    QMessageBox.warning(self,
                                        "Invalid CSV Format",
                                        "The selected CSV file does not match the expected CoralNet format.")
                    return

                annotation_size, ok = QInputDialog.getInt(self,
                                                          "Annotation Size",
                                                          "Enter the annotation size for all imported annotations:",
                                                          224, 1, 10000, 1)
                if not ok:
                    return

                loaded_image_names = [os.path.basename(path) for path in list(self.loaded_image_paths)]
                df = df[df['Name'].isin(loaded_image_names)]

                total_annotations = len(df)

                if not total_annotations:
                    raise Exception("No annotations found for loaded images.")

                progress_bar = ProgressBar(self, title="Importing CoralNet Annotations")
                progress_bar.show()
                progress_bar.start_progress(total_annotations)

                for index, row in df.iterrows():
                    image_name = row['Name']
                    row_coord = row['Row']
                    col_coord = row['Column']
                    label_code = row['Label']

                    image_path = None
                    for loaded_image_path in self.loaded_image_paths:
                        if os.path.basename(loaded_image_path) == image_name:
                            image_path = loaded_image_path
                            break

                    if image_path is None:
                        continue

                    short_label_code = label_code
                    long_label_code = label_code

                    existing_label = self.main_window.label_window.get_label_by_codes(short_label_code, long_label_code)
                    if existing_label:
                        color = existing_label.color
                        label_id = existing_label.id
                    else:
                        color = QColor(random.randint(0, 255),
                                       random.randint(0, 255),
                                       random.randint(0, 255))

                        label_id = str(uuid.uuid4())
                        self.main_window.label_window.add_label(short_label_code, long_label_code, color, label_id)

                    annotation = Annotation(QPointF(col_coord, row_coord),
                                            annotation_size,
                                            short_label_code,
                                            long_label_code,
                                            color,
                                            image_path,
                                            label_id)

                    self.annotations_dict[annotation.id] = annotation

                    progress_bar.update_progress()
                    QApplication.processEvents()  # Update GUI

                progress_bar.stop_progress()
                progress_bar.close()

                self.load_annotations()

                QMessageBox.information(self, "Annotations Imported",
                                        "Annotations have been successfully imported.")

            except Exception as e:
                QMessageBox.warning(self, "Error Importing Annotations",
                                    f"An error occurred while importing annotations: {str(e)}")

    def set_selected_tool(self, tool):
        self.selected_tool = tool
        if self.selected_tool == "select":
            self.setCursor(Qt.PointingHandCursor)
        elif self.selected_tool == "annotate":
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        self.unselect_annotation()

    def set_selected_label(self, label):
        self.selected_label = label
        self.annotation_color = label.color

        if self.selected_annotation:
            if self.selected_annotation.label.id != label.id:
                self.selected_annotation.update_label(self.selected_label)
                self.selected_annotation.create_cropped_image(self.image_item)
                # Notify ConfidenceWindow the selected annotation has changed
                self.main_window.confidence_window.display_cropped_image(self.selected_annotation)

        if self.cursor_annotation:
            if self.cursor_annotation.label.id != label.id:
                self.toggle_cursor_annotation()

    def set_annotation_size(self, size=None, delta=0):
        if size is not None:
            self.annotation_size = size
        else:
            self.annotation_size += delta
            self.annotation_size = max(1, self.annotation_size)

        if self.selected_annotation:
            self.selected_annotation.update_annotation_size(self.annotation_size)
            self.selected_annotation.create_cropped_image(self.image_item)
            # Notify ConfidenceWindow the selected annotation has changed
            self.main_window.confidence_window.display_cropped_image(self.selected_annotation)

        if self.cursor_annotation:
            self.cursor_annotation.update_annotation_size(self.annotation_size)

        # Emit that the annotation size has changed
        self.annotationSizeChanged.emit(self.annotation_size)

    def set_annotation_location(self, annotation_id, new_center_xy: QPointF):
        if annotation_id in self.annotations_dict:
            annotation = self.annotations_dict[annotation_id]
            old_center_xy = annotation.center_xy
            annotation.update_location(new_center_xy)

    def set_transparency(self, transparency: int):
        self.transparency = transparency

    def toggle_cursor_annotation(self, scene_pos: QPointF = None):
        if scene_pos:
            if not self.selected_label or not self.annotation_color:
                return

            if not self.cursor_annotation:
                self.cursor_annotation = Annotation(scene_pos,
                                                    self.annotation_size,
                                                    self.selected_label.short_label_code,
                                                    self.selected_label.long_label_code,
                                                    self.selected_label.color,
                                                    self.current_image_path,
                                                    self.selected_label.id,
                                                    transparency=128,
                                                    show_msg=False)

                self.cursor_annotation.create_graphics_item(self.scene)
            else:
                self.cursor_annotation.update_location(scene_pos)
                self.cursor_annotation.update_graphics_item()
                self.cursor_annotation.update_transparency(128)
        else:
            if self.cursor_annotation:
                self.cursor_annotation.delete()
                self.cursor_annotation = None

    def set_image(self, image, image_path):

        self.unselect_annotation()

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.image_item = QGraphicsPixmapItem(QPixmap(image))
        self.current_image_path = image_path
        self.active_image = True

        self.imageLoaded.emit(image.width(), image.height())

        self.scene.addItem(self.image_item)
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.toggle_cursor_annotation()

        self.load_annotations()
        self.loaded_image_paths.add(image_path)

        # Clear the confidence window
        self.main_window.confidence_window.clear_display()

    def wheelEvent(self, event: QMouseEvent):
        if event.angleDelta().y() > 0:
            factor = 1.1
        else:
            factor = 0.9

        self.zoom_factor *= factor
        self.scale(factor, factor)

        if self.selected_tool == "select":
            self.setCursor(Qt.PointingHandCursor)
        elif self.selected_tool == "annotate":
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def mousePressEvent(self, event: QMouseEvent):
        if self.active_image:

            if event.button() == Qt.RightButton:
                self.pan_active = True
                self.pan_start = event.pos()
                self.setCursor(Qt.ClosedHandCursor)  # Change cursor to indicate panning

            elif self.selected_tool == "select" and event.button() == Qt.LeftButton:
                position = self.mapToScene(event.pos())
                items = self.scene.items(position)

                rect_items = [item for item in items if isinstance(item, QGraphicsRectItem)]
                rect_items.sort(key=lambda item: item.zValue(), reverse=True)

                for rect_item in rect_items:
                    annotation_id = rect_item.data(0)  # Retrieve the UUID from the graphics item's data
                    annotation = self.annotations_dict.get(annotation_id)
                    if annotation.contains_point(position):
                        self.select_annotation(annotation)
                        self.drag_start_pos = position  # Store the start position for dragging
                        break

            elif self.selected_tool == "annotate" and event.button() == Qt.LeftButton:
                self.add_annotation(self.mapToScene(event.pos()))

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.pan_active:
            self.pan(event.pos())
        elif self.selected_tool == "select" and self.selected_annotation:
            current_pos = self.mapToScene(event.pos())
            if hasattr(self, 'drag_start_pos'):
                delta = current_pos - self.drag_start_pos
                new_center = self.selected_annotation.center_xy + delta
                self.set_annotation_location(self.selected_annotation.id, new_center)
                self.selected_annotation.create_cropped_image(self.image_item)
                self.main_window.confidence_window.display_cropped_image(self.selected_annotation)
                self.drag_start_pos = current_pos  # Update the start position for smooth dragging
        elif (self.selected_tool == "annotate" and
              self.active_image and self.image_item and
              self.cursorInWindow(event.pos())):
            self.toggle_cursor_annotation(self.mapToScene(event.pos()))
        else:
            self.toggle_cursor_annotation()

        scene_pos = self.mapToScene(event.pos())
        self.mouseMoved.emit(int(scene_pos.x()), int(scene_pos.y()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.RightButton:
            self.pan_active = False
            self.setCursor(Qt.ArrowCursor)
        self.toggle_cursor_annotation()
        if hasattr(self, 'drag_start_pos'):
            del self.drag_start_pos  # Clean up the drag start position
        super().mouseReleaseEvent(event)

    def pan(self, pos):
        delta = pos - self.pan_start
        self.pan_start = pos
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())

    def cursorInWindow(self, pos, mapped=False):
        if self.image_item:
            image_rect = self.image_item.boundingRect()
            if not mapped:
                pos = self.mapToScene(pos)
            return image_rect.contains(pos)
        return False

    def cycle_annotations(self, direction):
        if self.selected_tool == "select" and self.active_image:
            annotations = self.get_image_annotations()
            if annotations:
                if self.selected_annotation:
                    current_index = annotations.index(self.selected_annotation)
                    new_index = (current_index + direction) % len(annotations)
                else:
                    new_index = 0
                self.select_annotation(annotations[new_index])

    def update_current_image_path(self, image_path):
        self.current_image_path = image_path

    def select_annotation(self, annotation):
        if self.selected_annotation != annotation:
            if self.selected_annotation:
                self.unselect_annotation()
            # Select the annotation
            self.selected_annotation = annotation
            self.selected_annotation.select()
            # Set the label and color for selected annotation
            self.selected_label = self.selected_annotation.label
            self.annotation_color = self.selected_annotation.label.color
            # Emit a signal indicating the selected annotations label to LabelWindow
            self.labelSelected.emit(annotation.label.id)
            # Crop the image from annotation using current image item
            if not self.selected_annotation.cropped_image:
                self.selected_annotation.create_cropped_image(self.image_item)
            # Display the selected annotation in confidence window
            self.main_window.confidence_window.display_cropped_image(self.selected_annotation)

    def unselect_annotation(self):
        if self.selected_annotation:
            self.selected_annotation.deselect()
            self.selected_annotation = None

        # Clear the confidence window
        self.main_window.confidence_window.clear_display()

    def update_annotations_transparency(self, label, transparency):
        self.set_transparency(transparency)
        for annotation in self.annotations_dict.values():
            if annotation.label.id == label.id:
                annotation.update_transparency(transparency)

    def load_annotations(self):
        for annotation_id, annotation in self.annotations_dict.items():
            if annotation.image_path == self.current_image_path:
                annotation.create_graphics_item(self.scene)
                annotation.create_cropped_image(self.image_item)

                # Connect update signals
                annotation.selected.connect(self.select_annotation)
                annotation.annotation_deleted.connect(self.delete_annotation)
                annotation.annotation_updated.connect(self.main_window.confidence_window.display_cropped_image)

    def get_image_annotations(self, image_path=None):
        if not image_path:
            image_path = self.current_image_path

        annotations = []
        for annotation_id, annotation in self.annotations_dict.items():
            if annotation.image_path == image_path:
                annotations.append(annotation)

        return annotations

    def add_annotation(self, scene_pos: QPointF, annotation=None):
        if not self.selected_label:
            QMessageBox.warning(self, "No Label Selected", "A label must be selected before adding an annotation.")
            return

        if not self.active_image or not self.image_item or not self.cursorInWindow(scene_pos, mapped=True):
            return

        if annotation is None:
            annotation = Annotation(scene_pos,
                                    self.annotation_size,
                                    self.selected_label.short_label_code,
                                    self.selected_label.long_label_code,
                                    self.selected_label.color,
                                    self.current_image_path,
                                    self.selected_label.id,
                                    transparency=self.transparency)

        annotation.create_graphics_item(self.scene)
        annotation.create_cropped_image(self.image_item)

        # Connect update signals
        annotation.selected.connect(self.select_annotation)
        annotation.annotation_deleted.connect(self.delete_annotation)
        annotation.annotation_updated.connect(self.main_window.confidence_window.display_cropped_image)

        self.annotations_dict[annotation.id] = annotation

        self.main_window.confidence_window.display_cropped_image(annotation)

    def delete_annotation(self, annotation_id):
        if annotation_id in self.annotations_dict:
            annotation = self.annotations_dict[annotation_id]
            annotation.delete()
            del self.annotations_dict[annotation_id]

    def delete_selected_annotation(self):
        if self.selected_annotation:
            self.delete_annotation(self.selected_annotation.id)
            self.selected_annotation = None
            # Clear the confidence window
            self.main_window.confidence_window.clear_display()

    def clear_annotations(self):
        for annotation_id in list(self.annotations_dict.keys()):
            self.delete_annotation(annotation_id)

    def delete_image(self, image_path):
        annotation_ids_to_delete = [i for i, a in self.annotations_dict.items() if a.image_path == image_path]

        for annotation_id in annotation_ids_to_delete:
            annotation = self.annotations_dict[annotation_id]
            annotation.delete()
            del self.annotations_dict[annotation_id]

        if self.current_image_path == image_path:
            self.scene.clear()
            self.current_image_path = None
            self.image_item = None
            self.active_image = False  # Reset image_set flag

        self.imageDeleted.emit(image_path)

    def delete_annotations_for_label(self, label):
        for annotation in list(self.annotations_dict.values()):
            if annotation.label.id == label.id:
                annotation.delete()
                del self.annotations_dict[annotation.id]


class ImageWindow(QWidget):
    imageSelected = pyqtSignal(str)
    imageDeleted = pyqtSignal(str)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Create a horizontal layout for the labels
        self.info_layout = QHBoxLayout()
        self.layout.addLayout(self.info_layout)

        # Add a label to display the index of the currently highlighted image
        self.current_image_index_label = QLabel("Current Image: None", self)
        self.current_image_index_label.setAlignment(Qt.AlignCenter)
        self.current_image_index_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # Set the desired height (to align with AnnotationWindow)
        self.current_image_index_label.setFixedHeight(24)
        self.info_layout.addWidget(self.current_image_index_label)

        # Add a label to display the total number of images
        self.image_count_label = QLabel("Total Images: 0", self)
        self.image_count_label.setAlignment(Qt.AlignCenter)
        self.image_count_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # Set the desired height (to align with AnnotationWindow)
        self.image_count_label.setFixedHeight(24)
        self.info_layout.addWidget(self.image_count_label)

        self.tableWidget = QTableWidget(self)
        self.tableWidget.setColumnCount(1)
        self.tableWidget.setHorizontalHeaderLabels(["Image Name"])
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.setSelectionBehavior(QTableWidget.SelectRows)
        self.tableWidget.setSelectionMode(QTableWidget.SingleSelection)
        self.tableWidget.cellClicked.connect(self.load_image)
        self.tableWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tableWidget.customContextMenuRequested.connect(self.show_context_menu)

        self.layout.addWidget(self.tableWidget)

        self.images = {}
        self.image_list = []
        self.selected_row = None
        self.show_confirmation_dialog = True

    def add_image(self, image_path):
        if image_path not in self.main_window.imported_image_paths:
            image = QImage(image_path)
            row_position = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row_position)
            self.tableWidget.setItem(row_position, 0, QTableWidgetItem(os.path.basename(image_path)))

            self.images[image_path] = image
            self.image_list.append(image_path)

            # Select and set the first image
            if row_position == 0:
                self.tableWidget.selectRow(0)
                self.load_image(0, 0)

            # Add to main window imported path
            self.main_window.imported_image_paths.add(image_path)

            # Update the image count label
            self.update_image_count_label()

    def load_image(self, row, column):
        image_path = self.image_list[row]
        self.load_image_by_path(image_path)

    def load_image_by_path(self, image_path):
        if self.selected_row is not None:
            self.tableWidget.item(self.selected_row, 0).setSelected(False)

        self.selected_row = self.image_list.index(image_path)
        self.tableWidget.selectRow(self.selected_row)
        self.annotation_window.set_image(self.images[image_path], image_path)
        self.imageSelected.emit(image_path)

        # Update the current image index label
        self.update_current_image_index_label()

    def delete_image(self, row):
        if self.show_confirmation_dialog:
            result = self._confirm_delete()
            if result == QMessageBox.No:
                return

        image_path = self.image_list[row]
        self.tableWidget.removeRow(row)
        del self.images[image_path]
        self.image_list.remove(image_path)
        self.main_window.imported_image_paths.discard(image_path)

        # Update the image count label
        self.update_image_count_label()

        # Update the selected row and load another image if possible
        if self.image_list:
            if row < len(self.image_list):
                self.selected_row = row
            else:
                self.selected_row = len(self.image_list) - 1
            self.load_image_by_path(self.image_list[self.selected_row])
        else:
            self.selected_row = None
            self.annotation_window.clear_image()  # Clear the annotation window if no images are left

        # Update the current image index label
        self.update_current_image_index_label()

    def _confirm_delete(self):
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setWindowTitle("Confirm Delete")
        msg_box.setText("Are you sure you want to delete this image?\n"
                        "This will delete all associated annotations.")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        checkbox = QCheckBox("Do not show this message again")
        msg_box.setCheckBox(checkbox)

        result = msg_box.exec_()

        if checkbox.isChecked():
            self.show_confirmation_dialog = False

        return result

    def show_context_menu(self, position):
        row = self.tableWidget.rowAt(position.y())
        if row >= 0:
            context_menu = QMenu(self)
            delete_action = context_menu.addAction("Delete")
            action = context_menu.exec_(self.tableWidget.mapToGlobal(position))

            if action == delete_action:
                self.delete_image(row)

    def update_image_count_label(self):
        total_images = len(self.images)
        self.image_count_label.setText(f"Total Images: {total_images}")

    def update_current_image_index_label(self):
        if self.selected_row is not None:
            self.current_image_index_label.setText(f"Current Image: {self.selected_row + 1}")
        else:
            self.current_image_index_label.setText("Current Image: None")

    def cycle_previous_image(self):
        if not self.image_list:
            return

        if self.selected_row is not None:
            new_row = (self.selected_row - 1) % len(self.image_list)
        else:
            new_row = 0

        self.tableWidget.selectRow(new_row)
        self.load_image(new_row, 0)

    def cycle_next_image(self):
        if not self.image_list:
            return

        if self.selected_row is not None:
            new_row = (self.selected_row + 1) % len(self.image_list)
        else:
            new_row = 0

        self.tableWidget.selectRow(new_row)
        self.load_image(new_row, 0)


class AddLabelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Label")
        self.setObjectName("AddLabelDialog")

        self.layout = QVBoxLayout(self)

        self.short_label_input = QLineEdit(self)
        self.short_label_input.setPlaceholderText("Short Label (max 10 characters)")
        self.short_label_input.setMaxLength(10)
        self.layout.addWidget(self.short_label_input)

        self.long_label_input = QLineEdit(self)
        self.long_label_input.setPlaceholderText("Long Label")
        self.layout.addWidget(self.long_label_input)

        self.color_button = QPushButton("Select Color", self)
        self.color_button.clicked.connect(self.select_color)
        self.layout.addWidget(self.color_button)

        self.button_box = QHBoxLayout()
        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.validate_and_accept)
        self.button_box.addWidget(self.ok_button)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.reject)
        self.button_box.addWidget(self.cancel_button)

        self.layout.addLayout(self.button_box)

        self.color = self.generate_random_color()
        self.update_color_button()

    def generate_random_color(self):
        return QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def update_color_button(self):
        self.color_button.setStyleSheet(f"background-color: {self.color.name()};")

    def select_color(self):
        color = QColorDialog.getColor(self.color, self, "Select Label Color")
        if color.isValid():
            self.color = color
            self.update_color_button()

    def get_label_details(self):
        return self.short_label_input.text(), self.long_label_input.text(), self.color

    def validate_and_accept(self):
        short_label_code = self.short_label_input.text().strip()
        long_label_code = self.long_label_input.text().strip()

        if not short_label_code or not long_label_code:
            QMessageBox.warning(self, "Input Error", "Both short and long label codes are required.")
        else:
            self.accept()


class EditLabelDialog(QDialog):
    def __init__(self, label, label_window, parent=None):
        super().__init__(parent)
        self.label = label
        self.label_window = label_window
        self.setWindowTitle("Edit Label")

        self.layout = QVBoxLayout(self)

        self.short_label_input = QLineEdit(self.label.short_label_code, self)
        self.short_label_input.setPlaceholderText("Short Label (max 10 characters)")
        self.short_label_input.setMaxLength(10)
        self.layout.addWidget(self.short_label_input)

        self.long_label_input = QLineEdit(self.label.long_label_code, self)
        self.long_label_input.setPlaceholderText("Long Label")
        self.layout.addWidget(self.long_label_input)

        self.color_button = QPushButton("Select Color", self)
        self.color_button.clicked.connect(self.select_color)
        self.layout.addWidget(self.color_button)

        self.button_box = QHBoxLayout()
        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.validate_and_accept)
        self.button_box.addWidget(self.ok_button)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.reject)
        self.button_box.addWidget(self.cancel_button)

        self.layout.addLayout(self.button_box)

        self.color = self.label.color
        self.update_color_button()

    def update_color_button(self):
        self.color_button.setStyleSheet(f"background-color: {self.color.name()};")

    def select_color(self):
        color = QColorDialog.getColor(self.color, self, "Select Label Color")
        if color.isValid():
            self.color = color
            self.update_color_button()

    def validate_and_accept(self):
        # Cannot edit Review
        if self.label.short_label_code == 'Review' and self.label.long_label_code == 'Review':
            QMessageBox.warning(self, "Cannot Edit Label", "The 'Review' label cannot be edited.")
            return

        # Can map other labels to Review
        short_label_code = self.short_label_input.text().strip()
        long_label_code = self.long_label_input.text().strip()

        if not short_label_code or not long_label_code:
            QMessageBox.warning(self, "Input Error", "Both short and long label codes are required.")
            return

        existing_label = self.label_window.get_label_by_codes(short_label_code, long_label_code)

        if existing_label and existing_label != self.label:
            text = (f"A label with the short code '{short_label_code}' "
                    f"and long code '{long_label_code}' already exists. "
                    f"Do you want to merge the labels?")

            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Question)
            msg_box.setText(text)
            msg_box.setWindowTitle("Merge Labels?")
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.No)
            result = msg_box.exec_()

            if result == QMessageBox.Yes:
                self.label_window.edit_labels(self.label, existing_label, delete_old=True)
                self.accept()

            return
        else:
            self.label.short_label_code = short_label_code
            self.label.long_label_code = long_label_code
            self.label.color = self.color
            self.label.update_label_color(self.color)
            self.accept()

            self.label_window.edit_labels(self.label, self.label, delete_old=False)


class Label(QWidget):
    color_changed = pyqtSignal(QColor)
    selected = pyqtSignal(object)  # Signal to emit the selected label
    label_deleted = pyqtSignal(object)  # Signal to emit when the label is deleted

    def __init__(self, short_label_code, long_label_code, color=QColor(255, 255, 255), label_id=None, fixed_width=80):
        super().__init__()

        self.id = str(uuid.uuid4()) if label_id is None else label_id
        self.short_label_code = short_label_code
        self.long_label_code = long_label_code
        self.color = color
        self.is_selected = False
        self.fixed_width = fixed_width

        self.setCursor(Qt.PointingHandCursor)
        self.setFixedWidth(self.fixed_width)

        # Set tooltip for long label
        self.setToolTip(self.long_label_code)

        # Context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def update_color(self):
        self.update()  # Trigger a repaint

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_selected = not self.is_selected
            self.update_selection()
            self.selected.emit(self)  # Emit the selected signal

    def update_selection(self):
        self.update()  # Trigger a repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Calculate the height based on the text height
        font_metrics = QFontMetrics(painter.font())
        text_height = font_metrics.height()
        # Add some padding
        self.setFixedHeight(text_height + 20)

        # Draw the main rectangle with a light transparent fill
        transparent_color = QColor(self.color)
        # Set higher transparency (0-255, where 255 is fully opaque)
        transparent_color.setAlpha(20)
        # Light transparent fill
        painter.setBrush(QBrush(transparent_color, Qt.SolidPattern))

        # Set the border color based on selection status
        if self.is_selected:
            # Lighter version of the label color
            selected_border_color = self.color.lighter(150)
            # Thicker border when selected
            painter.setPen(QPen(selected_border_color, 2, Qt.SolidLine))
        else:
            # Normal border with the color of the label
            painter.setPen(QPen(self.color, 1, Qt.SolidLine))

        painter.drawRect(0, 0, self.width(), self.height())

        # Draw the color rectangle only if selected
        if self.is_selected:
            # Width 5 pixels less than the main rectangle's width
            rectangle_width = self.width() - 10
            rectangle_height = 20
            inner_transparent_color = QColor(self.color)
            inner_transparent_color.setAlpha(100)
            painter.setBrush(QBrush(inner_transparent_color, Qt.SolidPattern))
            painter.drawRect(5, (self.height() - rectangle_height) // 2, rectangle_width, rectangle_height)

        # Draw the text
        if self.is_selected:
            painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
            painter.setFont(QFont(painter.font().family(), painter.font().pointSize(), QFont.Bold))
        else:
            painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))

        painter.drawText(12, 0, self.width() - 30, self.height(), Qt.AlignVCenter, self.short_label_code)

        super().paintEvent(event)

    def show_context_menu(self, pos):
        context_menu = QMenu(self)

    def delete_label(self):
        self.label_deleted.emit(self)
        self.deleteLater()

    def to_dict(self):
        return {
            'id': self.id,
            'short_label_code': self.short_label_code,
            'long_label_code': self.long_label_code,
            'color': self.color.getRgb()
        }

    @classmethod
    def from_dict(cls, data):
        color = QColor(*data['color'])
        return cls(data['short_label_code'], data['long_label_code'], color)

    def select(self):
        if not self.is_selected:
            self.is_selected = True
            self.update_selection()
            self.selected.emit(self)

    def deselect(self):
        if self.is_selected:
            self.is_selected = False
            self.update_selection()

    def update_label_color(self, new_color: QColor):
        if self.color != new_color:
            self.color = new_color
            self.update_color()
            self.color_changed.emit(new_color)

    def __repr__(self):
        return (f"Label(id={self.id}, "
                f"short_label_code={self.short_label_code}, "
                f"long_label_code={self.long_label_code}, "
                f"color={self.color.name()})")


class LabelWindow(QWidget):
    labelSelected = pyqtSignal(object)  # Signal to emit the entire Label object

    def __init__(self, main_window, label_width=100):
        super().__init__()

        self.annotation_window = main_window.annotation_window
        self.label_width = label_width
        self.labels_per_row = 1  # Initial value, will be updated

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Top bar with Add Label, Edit Label, and Delete Label buttons
        self.top_bar = QHBoxLayout()
        self.add_label_button = QPushButton("Add Label")
        self.add_label_button.setFixedSize(80, 30)
        self.top_bar.addWidget(self.add_label_button)

        self.edit_label_button = QPushButton("Edit Label")
        self.edit_label_button.setFixedSize(80, 30)
        self.edit_label_button.setEnabled(False)  # Initially disabled
        self.top_bar.addWidget(self.edit_label_button)

        self.delete_label_button = QPushButton("Delete Label")
        self.delete_label_button.setFixedSize(80, 30)
        self.delete_label_button.setEnabled(False)  # Initially disabled
        self.top_bar.addWidget(self.delete_label_button)

        self.top_bar.addStretch()  # Add stretch to the right side

        self.main_layout.addLayout(self.top_bar)

        # Scroll area for labels
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_content = QWidget()
        self.grid_layout = QGridLayout(self.scroll_content)
        self.grid_layout.setSpacing(0)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area.setWidget(self.scroll_content)
        self.main_layout.addWidget(self.scroll_area)

        self.add_label_button.clicked.connect(self.open_add_label_dialog)
        self.edit_label_button.clicked.connect(self.open_edit_label_dialog)
        self.delete_label_button.clicked.connect(self.delete_active_label)
        self.labels = []
        self.active_label = None

        # Add default label
        default_short_label_code = "Review"
        default_long_label_code = "Review"
        default_color = QColor(255, 255, 255)  # White color
        self.add_label(default_short_label_code, default_long_label_code, default_color, label_id="-1")
        # Deselect at first
        self.active_label.deselect()

        self.show_confirmation_dialog = True

    def export_labels(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self,
                                                   "Export Labels",
                                                   "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                labels_data = [label.to_dict() for label in self.labels]
                with open(file_path, 'w') as file:
                    json.dump(labels_data, file, indent=4)

                QMessageBox.information(self, "Labels Exported",
                                        "Labels have been successfully exported.")

            except Exception as e:
                QMessageBox.warning(self, "Error Importing Labels",
                                    f"An error occurred while importing labels: {str(e)}")

    def import_labels(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Import Labels",
                                                   "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    labels_data = json.load(file)

                for label_data in labels_data:
                    label = Label.from_dict(label_data)
                    if not self.label_exists(label.short_label_code, label.long_label_code):
                        self.add_label(label.short_label_code, label.long_label_code, label.color, label.id)

                # Set the Review label as active
                self.set_active_label(self.get_label_by_long_code("Review"))

                QMessageBox.information(self, "Labels Imported",
                                        "Annotations have been successfully imported.")

            except Exception as e:
                QMessageBox.warning(self, "Error Importing Labels",
                                    f"An error occurred while importing Labels: {str(e)}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_labels_per_row()
        self.reorganize_labels()

    def update_labels_per_row(self):
        available_width = self.scroll_area.width() - self.scroll_area.verticalScrollBar().width()
        self.labels_per_row = max(1, available_width // self.label_width)
        self.scroll_content.setFixedWidth(self.labels_per_row * self.label_width)

    def reorganize_labels(self):
        for i, label in enumerate(self.labels):
            row = i // self.labels_per_row
            col = i % self.labels_per_row
            self.grid_layout.addWidget(label, row, col)

    def open_add_label_dialog(self):
        dialog = AddLabelDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            short_label_code, long_label_code, color = dialog.get_label_details()
            if self.label_exists(short_label_code, long_label_code):
                QMessageBox.warning(self, "Label Exists", "A label with the same short and long name already exists.")
            else:
                new_label = self.add_label(short_label_code, long_label_code, color)
                self.set_active_label(new_label)

    def open_edit_label_dialog(self):
        if self.active_label:
            dialog = EditLabelDialog(self.active_label, self)
            if dialog.exec_() == QDialog.Accepted:
                # Update the tooltip with the new long label code
                self.active_label.setToolTip(self.active_label.long_label_code)
                self.update_labels_per_row()
                self.reorganize_labels()

    def add_label(self, short_label_code, long_label_code, color, label_id=None):
        # Create the label
        label = Label(short_label_code, long_label_code, color, label_id, fixed_width=self.label_width)
        # Connect
        label.selected.connect(self.set_active_label)
        label.label_deleted.connect(self.delete_label)
        self.labels.append(label)

        self.update_labels_per_row()
        self.reorganize_labels()
        self.set_active_label(label)

        return label

    def set_active_label(self, selected_label):
        if self.active_label and self.active_label != selected_label:
            self.deselect_active_label()

        self.active_label = selected_label
        self.active_label.select()
        self.labelSelected.emit(selected_label)

        # Enable or disable the Edit Label and Delete Label buttons based on whether a label is selected
        self.edit_label_button.setEnabled(self.active_label is not None)
        self.delete_label_button.setEnabled(self.active_label is not None)

        # Update annotations with the new label
        self.update_annotations_with_label(selected_label)

    def deselect_active_label(self):
        if self.active_label:
            self.active_label.deselect()

    def delete_active_label(self):
        if self.active_label:
            self.delete_label(self.active_label)

    def update_annotations_with_label(self, label):
        for annotation in self.annotation_window.annotations_dict.values():
            if annotation.label.id == label.id:
                annotation.update_label(label)

    def get_label_color(self, short_label_code, long_label_code):
        for label in self.labels:
            if short_label_code == label.short_label_code and long_label_code == label.long_label_code:
                return label.color
        return None

    def get_label_by_codes(self, short_label_code, long_label_code):
        for label in self.labels:
            if short_label_code == label.short_label_code and long_label_code == label.long_label_code:
                return label
        return None

    def get_label_by_long_code(self, long_label_code):
        for label in self.labels:
            if long_label_code == label.long_label_code:
                return label
        return None

    def label_exists(self, short_label_code, long_label_code, label_id=None):
        for label in self.labels:
            if label_id is not None and label.id == label_id:
                return True
            if label.short_label_code == short_label_code and label.long_label_code == long_label_code:
                return True
        return False

    def add_label_if_not_exists(self, short_label_code, long_label_code, color, label_id=None):
        if not self.label_exists(short_label_code, long_label_code, label_id):
            self.add_label(short_label_code, long_label_code, color, label_id)

    def set_selected_label(self, label_id):
        for lbl in self.labels:
            if lbl.id == label_id:
                self.set_active_label(lbl)
                break

    def edit_labels(self, old_label, new_label, delete_old=False):
        # Update annotations to use the new label
        for annotation in self.annotation_window.annotations_dict.values():
            if annotation.label.id == old_label.id:
                annotation.update_label(new_label)

        if delete_old:
            # Remove the old label
            self.labels.remove(old_label)
            old_label.deleteLater()

        # Update the active label if necessary
        if self.active_label == old_label:
            self.set_active_label(new_label)

        self.update_labels_per_row()
        self.reorganize_labels()

    def delete_label(self, label):
        if (label.short_label_code == "Review" and
                label.long_label_code == "Review" and
                label.color == QColor(255, 255, 255)):
            QMessageBox.warning(self, "Cannot Delete Label", "The 'Review' label cannot be deleted.")
            return

        if self.show_confirmation_dialog:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Question)
            msg_box.setWindowTitle("Confirm Delete")
            msg_box.setText("Are you sure you want to delete this label?\n"
                            "This will delete all associated annotations.")
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

            checkbox = QCheckBox("Do not show this message again")
            msg_box.setCheckBox(checkbox)

            result = msg_box.exec_()

            if checkbox.isChecked():
                self.show_confirmation_dialog = False

            if result == QMessageBox.No:
                return

        # Remove from the LabelWindow
        self.labels.remove(label)
        label.deleteLater()

        # Delete annotations associated with the label
        self.annotation_window.delete_annotations_for_label(label)

        # Reset active label if it was deleted
        if self.active_label == label:
            self.active_label = None
            if self.labels:
                self.set_active_label(self.labels[0])

        # Update the LabelWindow
        self.update_labels_per_row()
        self.reorganize_labels()

    def handle_wasd_key(self, key):
        if not self.active_label:
            return

        try:
            current_index = self.labels.index(self.active_label)
        except ValueError:
            # If the active label is not in the list, set it to None
            self.active_label = None
            return

        if key == Qt.Key_W:
            new_index = current_index - self.labels_per_row
        elif key == Qt.Key_S:
            new_index = current_index + self.labels_per_row
        elif key == Qt.Key_A:
            new_index = current_index - 1 if current_index % self.labels_per_row != 0 else current_index
        elif key == Qt.Key_D:
            new_index = current_index + 1 if (current_index + 1) % self.labels_per_row != 0 else current_index
        else:
            return

        if 0 <= new_index < len(self.labels):
            self.set_active_label(self.labels[new_index])


class ConfidenceBar(QFrame):
    barClicked = pyqtSignal(object)  # Define a signal that takes an object (label)

    def __init__(self, label, confidence, parent=None):
        super().__init__(parent)
        self.label = label
        self.confidence = confidence
        self.color = label.color
        self.setFixedHeight(20)  # Set a fixed height for the bars

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw the border
        painter.setPen(self.color)
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

        # Draw the filled part of the bar
        filled_width = int(self.width() * (self.confidence / 100))
        painter.setBrush(QColor(self.color.red(), self.color.green(), self.color.blue(), 192))  # 75% transparency
        painter.drawRect(0, 0, filled_width, self.height() - 1)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            self.handle_click()

    def handle_click(self):
        # Emit the signal with the label object
        self.barClicked.emit(self.label)

    def enterEvent(self, event):
        super().enterEvent(event)
        self.setCursor(QCursor(Qt.PointingHandCursor))

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.setCursor(QCursor(Qt.ArrowCursor))


class ConfidenceWindow(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.label_window = main_window.label_window

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.graphics_view = None
        self.scene = None

        self.bar_chart_widget = None
        self.bar_chart_layout = None

        self.init_graphics_view()
        self.init_bar_chart_widget()

        self.annotation = None
        self.user_confidence = None
        self.machine_confidence = None
        self.cropped_image = None
        self.chart_dict = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_blank_pixmap()
        self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def init_graphics_view(self):
        self.graphics_view = QGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.scene)
        self.layout.addWidget(self.graphics_view, 2)  # 2 for stretch factor
        self.update_blank_pixmap()

    def init_bar_chart_widget(self):
        self.bar_chart_widget = QWidget()
        self.bar_chart_layout = QVBoxLayout(self.bar_chart_widget)
        self.bar_chart_layout.setContentsMargins(0, 0, 0, 0)
        self.bar_chart_layout.setSpacing(2)  # Set spacing to make bars closer
        self.layout.addWidget(self.bar_chart_widget, 1)  # 1 for stretch factor

    def update_blank_pixmap(self):
        view_size = self.graphics_view.size()
        new_pixmap = QPixmap(view_size)
        new_pixmap.fill(Qt.transparent)
        self.scene.clear()
        self.scene.addPixmap(new_pixmap)

    def update_annotation(self, annotation):
        if annotation:
            self.annotation = annotation
            self.user_confidence = annotation.user_confidence
            self.machine_confidence = annotation.machine_confidence
            self.cropped_image = annotation.cropped_image.copy()
            self.chart_dict = self.machine_confidence if self.machine_confidence else self.user_confidence

    def display_cropped_image(self, annotation):
        self.clear_display()  # Clear the current display before updating
        self.update_annotation(annotation)
        if self.cropped_image:  # Ensure cropped_image is not None
            self.scene.addPixmap(self.cropped_image)
            self.scene.setSceneRect(QRectF(self.cropped_image.rect()))
            self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self.graphics_view.centerOn(self.scene.sceneRect().center())
            self.create_bar_chart()

    def create_bar_chart(self):
        self.clear_layout(self.bar_chart_layout)

        labels, confidences = self.get_chart_data()
        max_color = labels[confidences.index(max(confidences))].color
        self.graphics_view.setStyleSheet(f"border: 2px solid {max_color.name()};")

        for label, confidence in zip(labels, confidences):
            bar_widget = ConfidenceBar(label, confidence, self.bar_chart_widget)
            bar_widget.barClicked.connect(self.handle_bar_click)  # Connect the signal to the slot
            self.add_bar_to_layout(bar_widget, label, confidence)

    def get_chart_data(self):
        keys = list(self.chart_dict.keys())[:5]
        return (
            keys,
            [conf_value * 100 for conf_value in self.chart_dict.values()][:5]
        )

    def add_bar_to_layout(self, bar_widget, label, confidence):
        bar_layout = QHBoxLayout(bar_widget)
        bar_layout.setContentsMargins(5, 2, 5, 2)

        class_label = QLabel(label.short_label_code, bar_widget)
        class_label.setAlignment(Qt.AlignCenter)
        bar_layout.addWidget(class_label)

        percentage_label = QLabel(f"{confidence:.2f}%", bar_widget)
        percentage_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        bar_layout.addWidget(percentage_label)

        self.bar_chart_layout.addWidget(bar_widget)

    def clear_layout(self, layout):
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)

    def clear_display(self):
        """
        Clears the current scene and bar chart layout.
        """
        # Clear the scene
        self.scene.clear()
        # Clear the bar chart layout
        self.clear_layout(self.bar_chart_layout)
        # Reset the style sheet to default
        self.graphics_view.setStyleSheet("")

    def handle_bar_click(self, label):
        # Update the confidences to whichever bar was selected
        self.annotation.update_user_confidence(label)
        # Update the label to whichever bar was selected
        self.annotation.update_label(label)
        # Update everything (essentially)
        self.main_window.annotation_window.unselect_annotation()
        self.main_window.annotation_window.select_annotation(self.annotation)


def patch_extractor():
    app = QApplication([])
    app.setStyle('WindowsXP')
    main_window = MainWindow()
    main_window.show()
    app.exec_()


if __name__ == "__main__":
    patch_extractor()