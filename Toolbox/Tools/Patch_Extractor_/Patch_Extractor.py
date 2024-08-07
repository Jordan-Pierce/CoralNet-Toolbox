import os
import sys
import json
import random

import pandas as pd

from PyQt5.QtWidgets import (QProgressBar, QMainWindow, QFileDialog, QApplication, QGridLayout, QGraphicsView,
                             QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, QToolBar, QAction, QScrollArea,
                             QSizePolicy, QMessageBox, QCheckBox, QDialog, QHBoxLayout, QWidget, QVBoxLayout, QLabel,
                             QPushButton, QColorDialog, QMenu, QLineEdit)

from PyQt5.QtGui import QMouseEvent, QIcon, QImage, QPixmap, QColor, QPainter, QPen, QBrush, QFontMetrics
from PyQt5.QtCore import pyqtSignal, Qt, QTimer, QEvent, QObject, QPointF, QSize


class GlobalEventFilter(QObject):
    def __init__(self, label_window, annotation_window):
        super().__init__()
        self.label_window = label_window
        self.annotation_window = annotation_window

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:

            # Place holder
            if obj.objectName() in [""]:
                pass

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

        # Return False for other key events to allow them to be processed by the target object
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

        # Define the icon path

        self.setWindowTitle("CoralNet Toolbox")
        # Set the window icon
        main_window_icon_path = "Toolbox/Tools/Patch_Extractor_/icons/toolbox.png"
        self.setWindowIcon(QIcon(main_window_icon_path))

        # Set window flags for resizing, minimize, maximize, and customizing
        self.setWindowFlags(Qt.Window |
                            Qt.CustomizeWindowHint |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint |
                            Qt.WindowTitleHint)

        self.annotation_window = AnnotationWindow(self)
        self.label_window = LabelWindow(self.annotation_window)
        self.thumbnail_window = ThumbnailWindow(self.annotation_window)

        # Connect the label window to the annotation window for current selected label
        self.label_window.label_selected.connect(self.annotation_window.set_label_details)
        # Connect thumbnail window to the annotation window for current image selected
        self.annotation_window.imageDeleted.connect(self.thumbnail_window.handle_image_deletion)
        # Connect the imageSelected signal to update_current_image_path in AnnotationWindow
        self.thumbnail_window.imageSelected.connect(self.annotation_window.update_current_image_path)
        # Connect the imageDeleted signal to delete_image in AnnotationWindow
        self.thumbnail_window.imageDeleted.connect(self.annotation_window.delete_image)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        self.main_layout.addLayout(self.left_layout, 85)
        self.main_layout.addLayout(self.right_layout, 15)

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

        # TODO
        # Define icon paths
        select_icon_path = "Toolbox/Tools/Patch_Extractor_/icons/select.png"
        annotate_icon_path = "Toolbox/Tools/Patch_Extractor_/icons/annotate.png"

        # Add tools here with icons
        self.select_tool_action = QAction(QIcon(select_icon_path), "Select", self)
        self.select_tool_action.setCheckable(True)
        self.select_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.select_tool_action)

        self.annotate_tool_action = QAction(QIcon(annotate_icon_path), "Annotate", self)
        self.annotate_tool_action.setCheckable(True)
        self.annotate_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.annotate_tool_action)

        self.left_layout.addWidget(self.annotation_window, 85)
        self.left_layout.addWidget(self.label_window, 15)

        self.right_layout.addWidget(self.thumbnail_window)

        self.menu_bar = self.menuBar()
        self.import_menu = self.menu_bar.addMenu("Import")
        self.import_images_action = QAction("Import Images", self)
        self.import_images_action.triggered.connect(self.import_images)
        self.import_menu.addAction(self.import_images_action)

        self.import_annotations_action = QAction("Import Annotations (JSON)", self)
        self.import_annotations_action.triggered.connect(self.annotation_window.import_annotations)
        self.import_menu.addAction(self.import_annotations_action)

        self.export_menu = self.menu_bar.addMenu("Export")
        self.export_annotations_action = QAction("Export Annotations (JSON)", self)
        self.export_annotations_action.triggered.connect(self.annotation_window.export_annotations)
        self.export_menu.addAction(self.export_annotations_action)

        self.export_coralnet_annotations_action = QAction("Export Annotations (CoralNet)", self)
        self.export_coralnet_annotations_action.triggered.connect(self.annotation_window.export_coralnet_annotations)
        self.export_menu.addAction(self.export_coralnet_annotations_action)

        # Set up global event filter
        self.global_event_filter = GlobalEventFilter(self.label_window, self.annotation_window)
        QApplication.instance().installEventFilter(self.global_event_filter)

        self.imported_image_paths = set()  # Set to keep track of imported image paths

        # Connect the toolChanged signal to the AnnotationWindow
        self.toolChanged.connect(self.annotation_window.handle_tool_change)

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
                # Load the first image
                image_path = file_names[0]
                image = QImage(image_path)
                self.annotation_window.set_image(image, image_path)

    def showEvent(self, event):
        super().showEvent(event)
        self.showMaximized()  # Ensure the window is maximized when shown

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            if self.windowState() == Qt.WindowMinimized:
                pass  # Allow minimizing
            else:
                self.showMaximized()  # Restore maximized state

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


class AnnotationWindow(QGraphicsView):
    imageDeleted = pyqtSignal(str)  # Signal to emit when an image is deleted
    toolChanged = pyqtSignal(str)  # Signal to emit when the tool changes

    def __init__(self, main_window, parent=None, annotation_size=224, annotation_color=(255, 0, 0)):
        super().__init__(parent)
        self.main_window = main_window
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.zoom_factor = 1.0
        self.pan_active = False
        self.annotation_size = annotation_size
        self.set_annotation_color(annotation_color)
        self.selected_annotation = None
        self.temp_annotation = None
        self.image_set = False  # Flag to check if the image has been set
        self.annotations_dict = {}  # Dictionary to store annotations for each image
        self.undo_stack = []  # Stack to store undo actions
        self.redo_stack = []  # Stack to store redo actions
        self.active_label = None  # Flag to check if an active label is set
        self.current_tool = None  # Store the current tool state

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.NoDrag)  # Disable default drag mode

        self.image_item = None  # Initialize image_item to None
        self.loaded_image_paths = set()  # Initialize the set to store loaded image paths

        self.toolChanged.connect(self.handle_tool_change)

    def handle_tool_change(self, tool):
        self.current_tool = tool
        if self.current_tool == "select":
            self.setCursor(Qt.PointingHandCursor)
        elif self.current_tool == "annotate":
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
            self.hide_temp_annotation()

        # Unselect the annotation if the tool is changed
        if self.selected_annotation:
            self.selected_annotation.setPen(QPen(self.annotation_color, 2))
            self.selected_annotation = None

    def export_annotations(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Annotations", "", "JSON Files (*.json);;All Files (*)", options=options)
        if file_path:
            with open(file_path, 'w') as file:
                json.dump(self.annotations_dict, file, indent=4)

    def import_annotations(self):
        if not self.image_set:
            QMessageBox.warning(self, "No Images Loaded", "Please load images first before importing annotations.")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Annotations", "", "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_path:
            with open(file_path, 'r') as file:
                imported_annotations = json.load(file)

            # Count the total number of images with annotations in the JSON file
            total_images_with_annotations = len(imported_annotations)

            # Filter annotations to include only those for images already in the program
            self.annotations_dict = {k: v for k, v in imported_annotations.items() if k in self.loaded_image_paths}

            # Count the number of images loaded with annotations
            loaded_images_with_annotations = len(self.annotations_dict)

            # Display a message box with the information
            QMessageBox.information(self, "Annotations Loaded",
                                    f"Loaded annotations for {loaded_images_with_annotations} out of {total_images_with_annotations} images.")

            # Add labels to LabelWindow if they are not already present
            for image_path, annotations in self.annotations_dict.items():
                for annotation in annotations:
                    short_label = annotation['label_short_code']
                    long_label = annotation['label_long_code']
                    color = QColor(*annotation['annotation_color'])
                    self.main_window.label_window.add_label_if_not_exists(short_label, long_label, color)

            # Update annotations to match the color of the label already in the project
            updated_annotations = False
            for image_path, annotations in self.annotations_dict.items():
                for annotation in annotations:
                    short_label = annotation['label_short_code']
                    long_label = annotation['label_long_code']
                    existing_color = self.main_window.label_window.get_label_color(short_label, long_label)
                    if existing_color != QColor(*annotation['annotation_color']):
                        annotation['annotation_color'] = existing_color.getRgb()
                        updated_annotations = True

            if updated_annotations:
                QMessageBox.information(self, "Annotations Updated",
                                        "Some annotations have been updated to match the color of the labels already in the project.")

            # Load annotations for all images in the project
            for image_path in self.loaded_image_paths:
                self.load_annotations(image_path)

    def export_coralnet_annotations(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Export CoralNet Annotations", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            try:
                data = []
                for image_path, annotations in self.annotations_dict.items():
                    for annotation in annotations:
                        image_basename = os.path.basename(image_path)
                        center_xy = annotation['center_xy']
                        label = annotation['label_short_code']
                        patch_size = annotation['annotation_size']
                        data.append([image_basename, int(center_xy[1]), int(center_xy[0]), label, patch_size])

                df = pd.DataFrame(data, columns=['Name', 'Row', 'Column', 'Label', 'Patch Size'])
                df.to_csv(file_path, index=False)

            except Exception as e:
                QMessageBox.warning(self, "Error Exporting Annotations", f"An error occurred while exporting annotations: {str(e)}")

    def set_image(self, image, image_path):

        # Unselect the annotation if a new image is loaded
        if self.selected_annotation:
            self.selected_annotation.setPen(QPen(self.annotation_color, 2))
            self.selected_annotation = None

        self.image_item = QGraphicsPixmapItem(QPixmap(image))
        self.scene.clear()
        self.scene.addItem(self.image_item)
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.original_center = self.image_item.boundingRect().center()
        self.image_set = True  # Set the flag to True after the image has been set
        self.hide_temp_annotation()  # Hide temp annotation when a new image is set

        # Store the image path as a string
        self.current_image_path = image_path

        # Load annotations for the current image
        self.load_annotations(image_path)

        # Update the list of loaded image paths
        self.loaded_image_paths.add(image_path)

    def wheelEvent(self, event: QMouseEvent):
        if event.angleDelta().y() > 0:
            factor = 1.1
        else:
            factor = 0.9

        self.zoom_factor *= factor
        self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent):
        if self.image_set:
            if event.button() == Qt.RightButton:
                self.pan_active = True
                self.pan_start = event.pos()
                self.setCursor(Qt.ClosedHandCursor)  # Change cursor to indicate panning
            elif self.current_tool == "select" and event.button() == Qt.LeftButton:
                self.select_annotation(event)
            elif self.current_tool == "annotate" and event.button() == Qt.LeftButton:
                self.add_annotation(self.mapToScene(event.pos()))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.pan_active:
            self.pan(event.pos())
        elif self.current_tool == "annotate" and self.image_set and self.image_item and self.image_item.boundingRect().contains(
                self.mapToScene(event.pos())):
            self.show_temp_annotation(self.mapToScene(event.pos()))
        else:
            self.hide_temp_annotation()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.RightButton:
            self.pan_active = False
            self.setCursor(Qt.ArrowCursor)  # Reset cursor to default
        self.hide_temp_annotation()
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

    def add_annotation(self, scene_pos: QPointF):

        if not self.active_label:
            QMessageBox.warning(self, "No Label Selected", "A label must be selected before adding an annotation.")
            return

        # Check if the annotation's center point is within the image bounds
        if not self.image_item or not self.image_item.boundingRect().contains(scene_pos):
            return

        half_size = self.annotation_size / 2
        annotation = QGraphicsRectItem(scene_pos.x() - half_size, scene_pos.y() - half_size, self.annotation_size, self.annotation_size)
        annotation.setPen(QPen(self.annotation_color, 4))
        self.scene.addItem(annotation)

        # Store the annotation details in the dictionary
        image_path = self.current_image_path
        image_basename = os.path.basename(image_path)
        annotation_details = {
            'center_xy': (scene_pos.x(), scene_pos.y()),
            'annotation_size': self.annotation_size,
            'annotation_color': self.annotation_color.getRgb(),
            'image_path': image_path,
            'image_basename': image_basename,
            'label_short_code': self.label_short_code,  # Use the active label short code
            'label_long_code': self.label_long_code  # Use the active label long code
        }

        if image_path not in self.annotations_dict:
            self.annotations_dict[image_path] = []
        self.annotations_dict[image_path].append(annotation_details)

        # Push the action onto the undo stack
        self.undo_stack.append(('add', annotation_details))
        self.redo_stack.clear()  # Clear the redo stack

    def select_annotation(self, event):
        pos = self.mapToScene(event.pos())
        items = self.scene.items(pos)
        for item in items:
            if isinstance(item, QGraphicsRectItem):
                if self.selected_annotation:
                    self.selected_annotation.setPen(QPen(self.annotation_color, 4))
                self.selected_annotation = item
                self.selected_annotation.setPen(QPen(Qt.blue, 4))
                break

    def delete_selected_annotation(self):
        if self.selected_annotation:
            for annotation_details in self.annotations_dict.get(self.current_image_path, []):
                center_xy = annotation_details['center_xy']
                half_size = annotation_details['annotation_size'] / 2
                if self.selected_annotation.rect().x() == center_xy[0] - half_size and self.selected_annotation.rect().y() == center_xy[1] - half_size:
                    self.annotations_dict[self.current_image_path].remove(annotation_details)
                    self.scene.removeItem(self.selected_annotation)
                    self.selected_annotation = None
                    break

    def clear_annotations(self):
        for annotation in self.scene.items():
            if isinstance(annotation, QGraphicsRectItem):
                self.scene.removeItem(annotation)
        self.annotations_dict.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()

    def set_annotation_size(self, size):
        self.annotation_size = size

    def set_label_details(self, short_label, long_label, color):
        self.label_short_code = short_label
        self.label_long_code = long_label
        self.set_annotation_color(color.getRgb()[:3])
        self.active_label = True  # Set the active label flag

    def set_annotation_color(self, color):
        if isinstance(color, tuple) and len(color) == 3:
            self.annotation_color = QColor(color[0], color[1], color[2])
        else:
            raise ValueError("Annotation color must be a tuple of three integers (R, G, B)")

    def show_temp_annotation(self, scene_pos: QPointF):
        if not self.active_label:
            return

        self.hide_temp_annotation()
        half_size = self.annotation_size / 2
        self.temp_annotation = QGraphicsRectItem(scene_pos.x() - half_size, scene_pos.y() - half_size, self.annotation_size, self.annotation_size)
        self.temp_annotation.setPen(QPen(self.annotation_color, 4))
        self.scene.addItem(self.temp_annotation)

    def hide_temp_annotation(self):
        if self.temp_annotation and self.scene.items().__contains__(self.temp_annotation):
            self.scene.removeItem(self.temp_annotation)
        self.temp_annotation = None

    def load_annotations(self, image_path):
        if image_path in self.annotations_dict:
            for annotation_details in self.annotations_dict[image_path]:
                center_xy = annotation_details['center_xy']
                half_size = annotation_details['annotation_size'] / 2
                annotation = QGraphicsRectItem(center_xy[0] - half_size, center_xy[1] - half_size, annotation_details['annotation_size'], annotation_details['annotation_size'])
                annotation.setPen(QPen(QColor(*annotation_details['annotation_color']), 4))
                self.scene.addItem(annotation)

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

    def remove_annotation(self, details):
        image_path = details['image_path']
        if image_path in self.annotations_dict:
            self.annotations_dict[image_path] = [anno for anno in self.annotations_dict[image_path] if anno != details]
            for item in self.scene.items():
                if isinstance(item, QGraphicsRectItem):
                    center_xy = details['center_xy']
                    half_size = details['annotation_size'] / 2
                    if item.rect().x() == center_xy[0] - half_size and item.rect().y() == center_xy[1] - half_size:
                        self.scene.removeItem(item)
                        break

    def add_annotation_from_details(self, details):
        center_xy = details['center_xy']
        half_size = details['annotation_size'] / 2
        annotation = QGraphicsRectItem(center_xy[0] - half_size, center_xy[1] - half_size, details['annotation_size'], details['annotation_size'])
        annotation.setPen(QPen(QColor(*details['annotation_color']), 4))
        self.scene.addItem(annotation)

        image_path = details['image_path']
        if image_path not in self.annotations_dict:
            self.annotations_dict[image_path] = []
        self.annotations_dict[image_path].append(details)

    def delete_image(self, image_path):
        if image_path in self.annotations_dict:
            del self.annotations_dict[image_path]
        if self.current_image_path == image_path:
            self.scene.clear()
            self.current_image_path = None
            self.image_item = None  # Reset image_item to None
            self.image_set = False  # Reset image_set flag
        self.imageDeleted.emit(image_path)  # Emit the signal when an image is deleted

    def update_current_image_path(self, image_path):
        self.current_image_path = image_path

    def delete_annotations_for_label(self, short_label, long_label):
        for image_path, annotations in list(self.annotations_dict.items()):
            self.annotations_dict[image_path] = [anno for anno in annotations if not (anno['label_short_code'] == short_label and anno['label_long_code'] == long_label)]
            if not self.annotations_dict[image_path]:
                del self.annotations_dict[image_path]
            self.load_annotations(image_path)


class ThumbnailWindow(QWidget):
    imageSelected = pyqtSignal(str)  # Signal to emit the selected image path
    imageDeleted = pyqtSignal(str)  # Signal to emit the deleted image path

    def __init__(self, annotation_window):
        super().__init__()

        self.annotation_window = annotation_window
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

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
        self.selected_thumbnail = None
        self.show_confirmation_dialog = True

    def add_image(self, image_path):
        if image_path not in self.annotation_window.main_window.imported_image_paths:
            image = QImage(image_path)

            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)

            label = ThumbnailLabel(image_path, image)
            label.mousePressEvent = lambda event, img=image, img_path=image_path, lbl=label: self.load_image(img, img_path, lbl)
            label.setContextMenuPolicy(Qt.CustomContextMenu)
            label.customContextMenuRequested.connect(lambda pos, lbl=label: self.show_context_menu(pos, lbl))

            basename_label = QLabel(os.path.basename(image_path))
            basename_label.setAlignment(Qt.AlignCenter)
            basename_label.setWordWrap(True)

            container_layout.addWidget(label)
            container_layout.addWidget(basename_label)

            self.thumbnail_container_layout.addWidget(container)
            self.images[image_path] = image

            self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().maximum())

            # Automatically select the first image added
            if not self.selected_thumbnail:
                self.load_image(image, image_path, label)

            self.annotation_window.main_window.imported_image_paths.add(image_path)

    def load_image(self, image, image_path, label):
        if self.selected_thumbnail:
            self.selected_thumbnail.setStyleSheet("")  # Reset the previous selection

        self.selected_thumbnail = label
        self.selected_thumbnail.setStyleSheet("border: 2px solid blue;")  # Highlight the selected thumbnail

        self.annotation_window.set_image(image, image_path)
        self.imageSelected.emit(image_path)  # Emit the signal with the selected image path

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.thumbnail_container.setFixedWidth(self.width() - self.scrollArea.verticalScrollBar().width())

    def show_context_menu(self, pos, label):
        context_menu = QMenu(self)
        delete_action = context_menu.addAction("Delete")
        action = context_menu.exec_(label.mapToGlobal(pos))

        if action == delete_action:
            self.delete_image(label)

    def delete_image(self, label):
        if self.show_confirmation_dialog:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Question)
            msg_box.setWindowTitle("Confirm Delete")
            msg_box.setText("Are you sure you want to delete this image?")
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

            checkbox = QCheckBox("Do not show this message again")
            msg_box.setCheckBox(checkbox)

            result = msg_box.exec_()

            if checkbox.isChecked():
                self.show_confirmation_dialog = False

            if result == QMessageBox.No:
                return

        # Remove the image from the layout and dictionary
        for i in range(self.thumbnail_container_layout.count()):
            item = self.thumbnail_container_layout.itemAt(i)
            widget = item.widget()
            if widget and widget.layout().itemAt(0).widget() == label:
                self.thumbnail_container_layout.removeItem(item)
                widget.deleteLater()
                break

        # Remove the image from the dictionary
        image_path = label.image_path
        if image_path in self.images:
            del self.images[image_path]

        # Reset the selected thumbnail if it was deleted
        if self.selected_thumbnail == label:
            self.selected_thumbnail = None
            if self.images:
                first_image_path = next(iter(self.images))
                first_label = self.find_label_by_image_path(first_image_path)
                if first_label:
                    self.load_image(self.images[first_image_path], first_image_path, first_label)

        self.imageDeleted.emit(image_path)  # Emit the signal when an image is deleted
        self.annotation_window.main_window.imported_image_paths.discard(image_path)

    def find_label_by_image_path(self, image_path):
        for i in range(self.thumbnail_container_layout.count()):
            item = self.thumbnail_container_layout.itemAt(i)
            widget = item.widget()
            if widget:
                label = widget.layout().itemAt(0).widget()
                if isinstance(label, ThumbnailLabel) and label.image_path == image_path:
                    return label
        return None

    def handle_image_deletion(self, image_path):
        if image_path in self.images:
            del self.images[image_path]
        for i in range(self.thumbnail_container_layout.count()):
            item = self.thumbnail_container_layout.itemAt(i)
            widget = item.widget()
            if widget:
                label = widget.layout().itemAt(0).widget()
                if isinstance(label, ThumbnailLabel) and label.image_path == image_path:
                    self.thumbnail_container_layout.removeItem(item)
                    widget.deleteLater()
                    break

class ThumbnailLabel(QLabel):
    def __init__(self, image_path, image, size=100):
        super().__init__()
        self.image_path = image_path
        self.size = size
        self.setFixedSize(self.size, self.size)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)
        self.setImage(image_path, image)

    def setImage(self, image_path, image):
        scaled_image = image.scaled(self.size, self.size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(QPixmap.fromImage(scaled_image))

    def sizeHint(self):
        return QSize(self.size, self.size)


class AddLabelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Label")
        self.setObjectName("AddLabelDialog")

        self.layout = QVBoxLayout(self)

        self.short_label_input = QLineEdit(self)
        self.short_label_input.setPlaceholderText("Short Label")
        self.layout.addWidget(self.short_label_input)

        self.long_label_input = QLineEdit(self)
        self.long_label_input.setPlaceholderText("Long Label")
        self.layout.addWidget(self.long_label_input)

        self.color_button = QPushButton("Select Color", self)
        self.color_button.clicked.connect(self.select_color)
        self.layout.addWidget(self.color_button)

        self.button_box = QHBoxLayout()
        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.accept)
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


class Label(QWidget):
    color_changed = pyqtSignal(QColor)
    selected = pyqtSignal(object)  # Signal to emit the selected label
    label_deleted = pyqtSignal(object)  # Signal to emit when the label is deleted

    def __init__(self, short_label, long_label, color=QColor(255, 255, 255), fixed_width=80):
        super().__init__()

        self.short_label = short_label
        self.long_label = long_label
        self.color = color
        self.is_selected = False
        self.fixed_width = fixed_width

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.short_label_widget = QLabel(self.short_label)
        self.short_label_widget.setAlignment(Qt.AlignCenter)
        self.short_label_widget.setStyleSheet("color: black; background-color: transparent;")
        self.color_button = QPushButton()
        self.color_button.setFixedSize(20, 20)
        self.update_color()

        self.setContentsMargins(0, 0, 0, 0)  # Remove internal margins
        self.layout.setContentsMargins(0, 0, 0, 0)  # Remove layout margins
        self.layout.setSpacing(0)  # Remove spacing in the layout

        self.layout.addWidget(self.short_label_widget)
        self.layout.addWidget(self.color_button, alignment=Qt.AlignCenter)

        self.setCursor(Qt.PointingHandCursor)

        # Calculate the height based on the text height
        font_metrics = QFontMetrics(self.short_label_widget.font())
        text_height = font_metrics.height()
        self.setFixedSize(self.fixed_width, text_height + 40)  # Add some padding

        # Set tooltip for long label
        self.setToolTip(self.long_label)

        # Disable color change
        self.color_button.setEnabled(False)

        # Context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def update_color(self):
        self.color_button.setStyleSheet(
            f"background-color: {self.color.name()}; border: none; border-radius: 10px;")
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

        # Draw the main rectangle
        painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))  # Thinner border
        painter.setBrush(QBrush(self.color, Qt.SolidPattern))
        painter.drawRect(0, 0, self.width(), self.height())  # Fill the entire widget

        # Draw selection border if selected
        if self.is_selected:
            painter.setPen(QPen(Qt.blue, 2, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(1, 1, self.width() - 2, self.height() - 2)

        super().paintEvent(event)

    def show_context_menu(self, pos):
        context_menu = QMenu(self)
        delete_action = context_menu.addAction("Delete")
        action = context_menu.exec_(self.mapToGlobal(pos))

        if action == delete_action:
            self.delete_label()

    def delete_label(self):
        self.label_deleted.emit(self)
        self.deleteLater()


class LabelWindow(QWidget):
    label_selected = pyqtSignal(str, str, QColor)  # Signal to emit label details

    def __init__(self, annotation_window, label_width=80):
        super().__init__()

        self.annotation_window = annotation_window
        self.label_width = label_width
        self.labels_per_row = 1  # Initial value, will be updated

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Top bar with Add Label button
        self.top_bar = QHBoxLayout()
        self.top_bar.addStretch()
        self.add_label_button = QPushButton("Add Label")
        self.add_label_button.setFixedSize(80, 30)
        self.top_bar.addWidget(self.add_label_button)
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
        self.labels = []
        self.active_label = None

        # Add default label
        default_short_label = "Review"
        default_long_label = "Review"
        default_color = QColor(255, 255, 255)  # White color
        self.add_label(default_short_label, default_long_label, default_color)
        # Do not set the default label as active

        self.show_confirmation_dialog = True

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
            short_label, long_label, color = dialog.get_label_details()
            if self.label_exists(short_label, long_label):
                QMessageBox.warning(self, "Label Exists", "A label with the same short and long name already exists.")
            else:
                new_label = self.add_label(short_label, long_label, color)
                self.set_active_label(new_label)

    def add_label(self, short_label, long_label, color):
        label = Label(short_label, long_label, color, self.label_width)
        label.selected.connect(self.set_active_label)
        label.label_deleted.connect(self.delete_label)
        self.labels.append(label)

        self.update_labels_per_row()
        self.reorganize_labels()

        return label

    def get_label_color(self, short_label, long_label):
        label_color = None
        for label in self.labels:
            if short_label == label.short_label and long_label == label.long_label:
                label_color = label.color
        return label_color

    def label_exists(self, short_label, long_label):
        for label in self.labels:
            if label.short_label == short_label and label.long_label == long_label:
                return True
        return False

    def add_label_if_not_exists(self, short_label, long_label, color):
        if not self.label_exists(short_label, long_label):
            self.add_label(short_label, long_label, color)

    def set_active_label(self, selected_label):
        if self.active_label and self.active_label != selected_label:
            self.active_label.is_selected = False
            self.active_label.update_selection()

        self.active_label = selected_label
        self.active_label.is_selected = True
        self.active_label.update_selection()
        print(f"Active label: {self.active_label.short_label}")
        self.label_selected.emit(selected_label.short_label, selected_label.long_label, selected_label.color)

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

    def delete_label(self, label):
        if label.short_label == "Review" and label.long_label == "Review" and label.color == QColor(255, 255, 255):
            QMessageBox.warning(self, "Cannot Delete Label", "The 'Review' label cannot be deleted.")
            return

        if self.show_confirmation_dialog:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Question)
            msg_box.setWindowTitle("Confirm Delete")
            msg_box.setText("Are you sure you want to delete this label?")
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

            checkbox = QCheckBox("Do not show this message again")
            msg_box.setCheckBox(checkbox)

            result = msg_box.exec_()

            if checkbox.isChecked():
                self.show_confirmation_dialog = False

            if result == QMessageBox.No:
                return

        self.labels.remove(label)
        self.update_labels_per_row()
        self.reorganize_labels()

        # Delete annotations associated with the label
        self.annotation_window.delete_annotations_for_label(label.short_label, label.long_label)

        # Reset active label if it was deleted
        if self.active_label == label:
            self.active_label = None
            if self.labels:
                self.set_active_label(self.labels[0])


if __name__ == "__main__":
    app = QApplication([])
    app.setStyle('WindowsXP')
    main_window = MainWindow()
    main_window.show()
    app.exec_()