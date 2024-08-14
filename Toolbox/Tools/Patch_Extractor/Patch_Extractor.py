# TODO
#   - If a user has annotation selected, and they zoom, update confidence window accordingly (message about scores lost)
#   - Update the undo and redo
#   - Move annotation crop as a method, store confidences, change confidence window to take annotation directly,
#       connect to labelwindow
#   - Resolution / resizing
#   - If an annotation' label is updated, re-draw / update crop, 1%

import os
import uuid
import json
import random
import weakref

import pandas as pd

from PyQt5.QtWidgets import (QProgressBar, QMainWindow, QFileDialog, QApplication, QGridLayout, QGraphicsView,
                             QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, QToolBar, QAction, QScrollArea,
                             QSizePolicy, QMessageBox, QCheckBox, QDialog, QHBoxLayout, QWidget, QVBoxLayout, QLabel,
                             QPushButton, QColorDialog, QMenu, QLineEdit, QSpinBox, QDialog, QHBoxLayout,
                             QPushButton, QComboBox, QSpinBox, QGraphicsPixmapItem, QGraphicsRectItem, QSlider,
                             QFormLayout, QInputDialog, QFrame)

from PyQt5.QtGui import QMouseEvent, QIcon, QImage, QPixmap, QColor, QPainter, QPen, QBrush, QFontMetrics, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QObject, QThreadPool, QRunnable, QTimer, QEvent, QPointF, QRectF



class GlobalEventFilter(QObject):
    def __init__(self, label_window, annotation_window):
        super().__init__()
        self.label_window = label_window
        self.annotation_window = annotation_window

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:  # Check if the event is a wheel event
            # Check for Ctrl modifier
            if event.modifiers() & Qt.ControlModifier:
                delta = event.angleDelta().y()
                if delta > 0:
                    self.annotation_window.set_annotation_size(delta=16)  # Zoom in
                else:
                    self.annotation_window.set_annotation_size(delta=-16)  # Zoom out
                return True

        elif event.type() == QEvent.KeyPress:
            # Check for Ctrl modifier
            if event.modifiers() & Qt.ControlModifier:
                if event.key() in [Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D]:
                    self.label_window.handle_wasd_key(event.key())
                    return True
                elif event.key() == Qt.Key_Z:
                    self.annotation_window.undo()
                    return True
                elif event.key() == Qt.Key_Y:
                    self.annotation_window.redo()
                    return True

            # Place holder
            if obj.objectName() in [""]:
                pass

        # Return False for other events to allow them to be processed by the target object
        return False


class ProgressBar(QDialog):
    def __init__(self, parent=None, title="Progress"):
        super().__init__(parent)

        self.setWindowTitle(title)
        self.setModal(True)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.progress_bar)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)

        self.value = 0

    def update_progress(self):
        self.value += 1
        self.progress_bar.setValue(self.value)

    def start_progress(self, max_value):
        self.value = 0
        self.progress_bar.setRange(0, max_value)
        self.timer.start(50)  # Update progress every 50 milliseconds

    def stop_progress(self):
        self.timer.stop()


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
        self.thumbnail_window = ThumbnailWindow(self)
        self.confidence_window = ConfidenceWindow(self)

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
        self.thumbnail_window.imageSelected.connect(self.annotation_window.update_current_image_path)
        # Connect the imageDeleted signal to delete_image in AnnotationWindow
        self.thumbnail_window.imageDeleted.connect(self.annotation_window.delete_image)
        # Connect the labelSelected signal from LabelWindow to update the selected label in AnnotationWindow
        self.label_window.labelSelected.connect(self.annotation_window.set_selected_label)

        # Set up global event filter
        self.global_event_filter = GlobalEventFilter(self.label_window, self.annotation_window)
        QApplication.instance().installEventFilter(self.global_event_filter)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

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

        # Add CoralNet menu
        self.coralnet_menu = self.menu_bar.addMenu("CoralNet")

        self.coralnet_authenticate_action = QAction("Authenticate", self)
        self.coralnet_menu.addAction(self.coralnet_authenticate_action)

        self.coralnet_upload_action = QAction("Upload", self)
        self.coralnet_menu.addAction(self.coralnet_upload_action)

        self.coralnet_download_action = QAction("Download", self)
        self.coralnet_menu.addAction(self.coralnet_download_action)

        self.coralnet_model_api_action = QAction("Model API", self)
        self.coralnet_menu.addAction(self.coralnet_model_api_action)

        # Add Machine Learning menu
        self.ml_menu = self.menu_bar.addMenu("Machine Learning")

        self.ml_create_dataset_action = QAction("Create Dataset", self)
        self.ml_menu.addAction(self.ml_create_dataset_action)

        self.ml_train_model_action = QAction("Train Model", self)
        self.ml_menu.addAction(self.ml_train_model_action)

        self.ml_make_predictions_action = QAction("Make Predictions", self)
        self.ml_menu.addAction(self.ml_make_predictions_action)

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

        # Spin box for annotation size control
        self.annotation_size_spinbox = QSpinBox()
        self.annotation_size_spinbox.setMinimum(1)
        self.annotation_size_spinbox.setMaximum(1000)  # Adjust as needed
        self.annotation_size_spinbox.setValue(self.annotation_window.annotation_size)
        self.annotation_size_spinbox.valueChanged.connect(self.annotation_window.set_annotation_size)
        self.annotation_window.annotationSizeChanged.connect(self.annotation_size_spinbox.setValue)

        # Transparency slider
        self.transparency_slider = QSlider(Qt.Horizontal)
        self.transparency_slider.setRange(0, 255)
        self.transparency_slider.setValue(128)  # Default transparency
        self.transparency_slider.valueChanged.connect(self.update_annotation_transparency)

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
        self.right_layout.addWidget(self.thumbnail_window, 54)
        self.right_layout.addWidget(self.confidence_window, 46)

        # Add left and right layouts to main layout
        self.main_layout.addLayout(self.left_layout, 85)
        self.main_layout.addLayout(self.right_layout, 15)

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
                    self.thumbnail_window.add_image(file_name)
                    self.imported_image_paths.add(file_name)
                    self.annotation_window.loaded_image_paths.add(file_name)
                    progress_bar.update_progress()
                    QApplication.processEvents()  # Update GUI

            progress_bar.stop_progress()
            progress_bar.close()

            if file_names:
                # Load the last image
                image_path = file_names[-1]
                image = QImage(image_path)
                self.annotation_window.set_image(image, image_path)

    def open_annotation_sampling_dialog(self):
        # Check if there are any loaded images
        if not self.annotation_window.loaded_image_paths:
            # Show a message box if no images are loaded
            QMessageBox.warning(self, "No Images Loaded",
                                "Please load images into the project before sampling annotations.")
            return

        # Proceed to open the dialog if images are loaded
        dialog = AnnotationSamplingDialog(self.annotation_window, self)
        dialog.annotationsSampled.connect(self.add_sampled_annotations)
        dialog.exec_()

    def add_sampled_annotations(self, method, num_annotations, annotation_size,
                                margin_x_min, margin_y_min, margin_x_max, margin_y_max,
                                apply_to_all):

        # Sets the LabelWindow and AnnotationWindow to Review
        self.label_window.set_selected_label("-1")
        review_label = self.annotation_window.selected_label

        if apply_to_all:
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
            # Sample the annotation, given params
            annotations = sample_annotations(method,
                                             num_annotations,
                                             annotation_size,
                                             margin_x_min,
                                             margin_y_min,
                                             margin_x_max,
                                             margin_y_max,
                                             image_width,
                                             image_height)

            for annotation in annotations:
                x, y, size = annotation
                new_annotation = Annotation(QPointF(x, y),
                                            size,
                                            review_label.short_label_code,
                                            review_label.long_label_code,
                                            review_label.color,
                                            image_path,
                                            review_label.id,
                                            transparency=self.annotation_window.transparency)

                # Add annotation to the dict
                self.annotation_window.annotations_dict[new_annotation.id] = new_annotation

                # Update the progress bar
                progress_bar.update_progress()
                QApplication.processEvents()  # Update GUI

        # Stop the progress bar
        progress_bar.stop_progress()
        progress_bar.close()

        # Reload the annotations on the image
        self.annotation_window.load_annotations()

        QMessageBox.information(self, "Annotations Sampled",
                                "Annotations have been sampled successfully.")


def sample_annotations(method, num_annotations, annotation_size,
                       margin_x_min, margin_y_min, margin_x_max, margin_y_max, image_width, image_height):
    annotations = []

    if method == "Random":
        for _ in range(num_annotations):
            x = random.randint(margin_x_min, image_width - annotation_size - margin_x_max)
            y = random.randint(margin_y_min, image_height - annotation_size - margin_y_max)
            annotations.append((x, y, annotation_size))
    elif method == "Uniform":
        grid_size = int(num_annotations ** 0.5)
        x_step = (image_width - margin_x_min - margin_x_max - annotation_size) // grid_size
        y_step = (image_height - margin_y_min - margin_y_max - annotation_size) // grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                x = margin_x_min + i * x_step
                y = margin_y_min + j * y_step
                annotations.append((x, y, annotation_size))
    elif method == "Stratified Random":
        grid_size = int(num_annotations ** 0.5)
        x_step = (image_width - margin_x_min - margin_x_max) // grid_size
        y_step = (image_height - margin_y_min - margin_y_max) // grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                x = margin_x_min + i * x_step + random.randint(0, x_step - annotation_size)
                y = margin_y_min + j * y_step + random.randint(0, y_step - annotation_size)
                annotations.append((x, y, annotation_size))

    return annotations


class AnnotationSamplingDialog(QDialog):
    annotationsSampled = pyqtSignal(list, bool)  # Signal to emit the sampled annotations and apply to all flag

    def __init__(self, annotation_window, parent=None):
        super().__init__(parent)
        self.annotation_window = annotation_window
        self.setWindowTitle("Sample Annotations")

        self.layout = QVBoxLayout(self)

        # Sampling Method
        self.method_label = QLabel("Sampling Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Random", "Uniform", "Stratified Random"])
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

        # Apply to All Images Checkbox
        self.apply_all_checkbox = QCheckBox("Apply to all images")
        self.layout.addWidget(self.apply_all_checkbox)

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

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.button_box.addWidget(self.cancel_button)

        self.layout.addLayout(self.button_box)

        self.sampled_annotations = []

    def create_margin_spinbox(self, label_text, layout):
        label = QLabel(label_text + ":")
        spinbox = QSpinBox()
        spinbox.setMinimum(0)
        spinbox.setMaximum(1000)
        layout.addRow(label, spinbox)
        return spinbox

    def preview_annotations(self):
        method = self.method_combo.currentText()
        num_annotations = self.num_annotations_spinbox.value()
        annotation_size = self.annotation_size_spinbox.value()
        margin_x_min = self.margin_x_min_spinbox.value()
        margin_y_min = self.margin_y_min_spinbox.value()
        margin_x_max = self.margin_x_max_spinbox.value()
        margin_y_max = self.margin_y_max_spinbox.value()

        self.sampled_annotations = sample_annotations(method,
                                                      num_annotations,
                                                      annotation_size,
                                                      margin_x_min,
                                                      margin_y_min,
                                                      margin_x_max,
                                                      margin_y_max,
                                                      self.annotation_window.image_item.pixmap().width(),
                                                      self.annotation_window.image_item.pixmap().height())

        self.draw_annotation_previews(margin_x_min, margin_y_min, margin_x_max, margin_y_max)

    def draw_annotation_previews(self, margin_x_min, margin_y_min, margin_x_max, margin_y_max):
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
        apply_to_all = self.apply_all_checkbox.isChecked()
        self.parent().add_sampled_annotations(self.method_combo.currentText(),
                                              self.num_annotations_spinbox.value(),
                                              self.annotation_size_spinbox.value(),
                                              self.margin_x_min_spinbox.value(),
                                              self.margin_y_min_spinbox.value(),
                                              self.margin_x_max_spinbox.value(),
                                              self.margin_y_max_spinbox.value(),
                                              apply_to_all)
        self.accept()

    def showEvent(self, event):
        super().showEvent(event)
        self.showMaximized()  # Maximize the dialog when it is shown


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
                 transparency: int = 128):
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

    def move(self, new_center_xy: QPointF):
        self.center_xy = new_center_xy
        self.update_graphics_item()

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

    def update_user_confidence(self, new_label: 'Label'):
        self.user_confidence = {new_label: 1.0}

    def update_machine_confidence(self, label: 'Label', confidence: float):
        self.machine_confidence[label] = confidence

    def update_label(self, new_label: 'Label'):
        self.label = new_label
        self.update_user_confidence(new_label)
        self.update_graphics_item()

    def update_annotation_size(self, size):
        self.annotation_size = size
        self.update_graphics_item()

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
        self.cursor_annotation = None
        self.undo_stack = []  # Stack to store undo actions
        self.redo_stack = []  # Stack to store redo actions

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
                                                    transparency=128)

                self.cursor_annotation.create_graphics_item(self.scene)
            else:
                self.cursor_annotation.move(scene_pos)
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
                        break

            elif self.selected_tool == "annotate" and event.button() == Qt.LeftButton:
                self.add_annotation(self.mapToScene(event.pos()))

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.pan_active:
            self.pan(event.pos())
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
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_Z:
                self.undo()
            elif event.key() == Qt.Key_Y:
                self.redo()
        elif event.key() == Qt.Key_Delete:
            self.delete_selected_annotation()
        super().keyPressEvent(event)

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

    def undo(self):
        if self.undo_stack:
            action, details = self.undo_stack.pop()
            if action == 'add':
                self.remove_annotation(details)
                self.redo_stack.append(('add', details))

    def redo(self):
        if self.redo_stack:
            action, details = self.redo_stack.pop()
            if action == 'add':
                self.add_annotation_from_details(details)
                self.undo_stack.append(('add', details))

    def update_current_image_path(self, image_path):
        self.current_image_path = image_path

    def select_annotation(self, annotation):
        if self.selected_annotation != annotation:
            if self.selected_annotation:
                self.selected_annotation.deselect()

            self.selected_annotation = annotation
            self.selected_annotation.select()

            self.selected_label = self.selected_annotation.label
            self.annotation_color = self.selected_annotation.label.color

            self.labelSelected.emit(annotation.label.id)

            if not self.selected_annotation.cropped_image:
                self.selected_annotation.create_cropped_image(self.image_item)

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

        self.undo_stack.append(('add', annotation.to_dict()))
        self.redo_stack.clear()  # Clear the redo stack

        self.main_window.confidence_window.display_cropped_image(annotation)

    def delete_selected_annotation(self):
        if self.selected_annotation:
            self.delete_annotation(self.selected_annotation.id)
            self.selected_annotation = None
            # Clear the confidence window
            self.main_window.confidence_window.clear_display()

    def delete_annotation(self, annotation_id):
        if annotation_id in self.annotations_dict:
            annotation = self.annotations_dict[annotation_id]
            annotation.delete()
            self.redo_stack.append(('add', annotation.to_dict()))
            del self.annotations_dict[annotation_id]

    def clear_annotations(self):
        for annotation_id in list(self.annotations_dict.keys()):
            self.delete_annotation(annotation_id)
        self.undo_stack.clear()
        self.redo_stack.clear()

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


class ThumbnailWidget(QFrame):
    def __init__(self, parent, image_path, image, size=100):
        super().__init__(parent)
        self.parent_window = parent
        self.image_path = image_path
        self.size = size
        self.is_selected = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.image_label = QLabel()
        self.image_label.setFixedSize(self.size, self.size)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)
        self.setImage(image)
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        self.text_label = QLabel(os.path.basename(image_path))
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setWordWrap(True)
        layout.addWidget(self.text_label)

        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)

    def setImage(self, image):
        scaled_image = image.scaled(self.size, self.size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(QPixmap.fromImage(scaled_image))

    def select(self):
        self.is_selected = True
        self.setStyleSheet("ThumbnailWidget { background-color: rgba(0, 0, 255, 0.1); }")
        self.parent_window.set_selected_widget_width(self)

    def deselect(self):
        self.is_selected = False
        self.setStyleSheet("")
        self.setFixedWidth(self.parent_window.thumbnail_container.width())

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.parent_window.load_image(self)


class ThumbnailWindow(QWidget):
    imageSelected = pyqtSignal(str)
    imageDeleted = pyqtSignal(str)

    def __init__(self, main_window):
        super().__init__()

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

        self.scrollArea = QScrollArea(self)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setWidgetResizable(True)

        self.thumbnail_container = QWidget()
        self.thumbnail_container_layout = QVBoxLayout(self.thumbnail_container)
        self.thumbnail_container_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.thumbnail_container_layout.setSpacing(5)

        self.scrollArea.setWidget(self.thumbnail_container)
        self.layout.addWidget(self.scrollArea)

        self.images = {}
        self.thumbnails = weakref.WeakValueDictionary()
        self.selected_thumbnail = None
        self.show_confirmation_dialog = True
        self.threadpool = QThreadPool()

    def add_image_async(self, image_path):
        loader = ThumbnailLoader(self, image_path)
        self.threadpool.start(loader)

    def add_image(self, image_path):
        if image_path not in self.annotation_window.main_window.imported_image_paths:
            image = QImage(image_path)
            thumbnail = ThumbnailWidget(self, image_path, image)
            thumbnail.setContextMenuPolicy(Qt.CustomContextMenu)
            thumbnail.customContextMenuRequested.connect(lambda pos, t=thumbnail: self.show_context_menu(pos, t))

            self.thumbnail_container_layout.addWidget(thumbnail)
            self.images[image_path] = image
            self.thumbnails[image_path] = thumbnail

            self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().maximum())

            # Select and set thumbnail widget
            self.load_image(thumbnail)

            self.annotation_window.main_window.imported_image_paths.add(image_path)

            # Update the image count label
            self.update_image_count_label()

    def load_image(self, thumbnail):
        if self.selected_thumbnail:
            self.selected_thumbnail.deselect()

        # Selects the thumbnail
        self.selected_thumbnail = thumbnail
        self.selected_thumbnail.select()
        # Sets the thumbnail widget
        self.set_selected_widget_width(self.selected_thumbnail)

        image_path = thumbnail.image_path
        self.annotation_window.set_image(self.images[image_path], image_path)
        self.imageSelected.emit(image_path)

        # Update the current image index label
        self.update_current_image_index_label()

    def set_selected_widget_width(self, widget):
        widget.setFixedWidth(self.thumbnail_container.width())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        new_width = self.width() - self.scrollArea.verticalScrollBar().width()
        self.thumbnail_container.setFixedWidth(new_width)
        for thumbnail in self.thumbnails.values():
            thumbnail.setFixedWidth(new_width)
        if self.selected_thumbnail:
            self.set_selected_widget_width(self.selected_thumbnail)

    def show_context_menu(self, pos, thumbnail):
        context_menu = QMenu(self)
        delete_action = context_menu.addAction("Delete")
        action = context_menu.exec_(thumbnail.mapToGlobal(pos))

        if action == delete_action:
            self.delete_image(thumbnail)

    def delete_image(self, thumbnail):
        if self.show_confirmation_dialog:
            result = self._confirm_delete()
            if result == QMessageBox.No:
                return

        self._remove_image(thumbnail)
        self.imageDeleted.emit(thumbnail.image_path)  # Emit the signal
        self.annotation_window.main_window.imported_image_paths.discard(thumbnail.image_path)

        # Update the image count label
        self.update_image_count_label()

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

    def _remove_image(self, thumbnail):
        self.thumbnail_container_layout.removeWidget(thumbnail)
        thumbnail.deleteLater()

        image_path = thumbnail.image_path
        if image_path in self.images:
            del self.images[image_path]

        if self.selected_thumbnail == thumbnail:
            self.selected_thumbnail = None
            if self.images:
                first_image_path = next(iter(self.images))
                first_thumbnail = self.thumbnails.get(first_image_path)
                if first_thumbnail:
                    self.load_image(first_thumbnail)

    def find_thumbnail_by_image_path(self, image_path):
        return self.thumbnails.get(image_path)

    def update_image_count_label(self):
        total_images = len(self.images)
        self.image_count_label.setText(f"Total Images: {total_images}")

    def update_current_image_index_label(self):
        if self.selected_thumbnail:
            image_paths = list(self.images.keys())
            current_index = image_paths.index(self.selected_thumbnail.image_path)
            self.current_image_index_label.setText(f"Current Image: {current_index + 1}")
        else:
            self.current_image_index_label.setText("Current Image: None")


class ThumbnailLoader(QRunnable):
    def __init__(self, thumbnail_window, image_path):
        super().__init__()
        self.thumbnail_window = thumbnail_window
        self.image_path = image_path

    def run(self):
        image = QImage(self.image_path)
        self.thumbnail_window.add_image(self.image_path)


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
        short_label_code = self.short_label_input.text().strip()
        long_label_code = self.long_label_input.text().strip()

        if not short_label_code or not long_label_code:
            QMessageBox.warning(self, "Input Error", "Both short and long label codes are required.")
            return

        if short_label_code == 'Review' and long_label_code == 'Review':
            QMessageBox.warning(self, "Cannot Edit Label", "The 'Review' label cannot be edited.")
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
            self.active_label.deselect()

        self.active_label = selected_label
        self.active_label.select()
        self.labelSelected.emit(selected_label)

        # Enable or disable the Edit Label and Delete Label buttons based on whether a label is selected
        self.edit_label_button.setEnabled(self.active_label is not None)
        self.delete_label_button.setEnabled(self.active_label is not None)

        # Update annotations with the new label
        self.update_annotations_with_label(selected_label)

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


class CustomBar(QFrame):
    def __init__(self, color, value, parent=None):
        super().__init__(parent)
        self.color = color
        self.value = value
        self.setFixedHeight(20)  # Set a fixed height for the bars

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw the border
        painter.setPen(self.color)
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

        # Draw the filled part of the bar
        filled_width = int(self.width() * (self.value / 100))
        painter.setBrush(QColor(self.color.red(), self.color.green(), self.color.blue(), 192))  # 75% transparency
        painter.drawRect(0, 0, filled_width, self.height() - 1)


class ConfidenceWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
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
            self.chart_dict = self.user_confidence or self.machine_confidence

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

        classes, colors, values = self.get_chart_data()
        max_color = colors[values.index(max(values))]
        self.graphics_view.setStyleSheet(f"border: 2px solid {max_color.name()};")

        for cls, color, value in zip(classes, colors, values):
            bar_widget = CustomBar(color, value, self.bar_chart_widget)
            self.add_bar_to_layout(bar_widget, cls, value)

    def get_chart_data(self):
        keys = list(self.chart_dict.keys())[:5]
        return (
            [label.short_label_code for label in keys],
            [label.color for label in keys],
            [conf_value * 100 for conf_value in self.chart_dict.values()][:5]
        )

    def add_bar_to_layout(self, bar_widget, cls, value):
        bar_layout = QHBoxLayout(bar_widget)
        bar_layout.setContentsMargins(5, 2, 5, 2)

        class_label = QLabel(cls, bar_widget)
        class_label.setAlignment(Qt.AlignCenter)
        bar_layout.addWidget(class_label)

        percentage_label = QLabel(f"{value:.2f}%", bar_widget)
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


def patch_extractor():
    app = QApplication([])
    app.setStyle('WindowsXP')
    main_window = MainWindow()
    main_window.show()
    app.exec_()


if __name__ == "__main__":
    patch_extractor()