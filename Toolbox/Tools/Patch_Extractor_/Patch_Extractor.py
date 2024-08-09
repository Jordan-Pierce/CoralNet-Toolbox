import os
import sys
import uuid
import json
import random

import pandas as pd

from PyQt5.QtWidgets import (QProgressBar, QMainWindow, QFileDialog, QApplication, QGridLayout, QGraphicsView,
                             QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, QToolBar, QAction, QScrollArea,
                             QSizePolicy, QMessageBox, QCheckBox, QDialog, QHBoxLayout, QWidget, QVBoxLayout, QLabel,
                             QPushButton, QColorDialog, QMenu, QLineEdit)

from PyQt5.QtGui import QMouseEvent, QIcon, QImage, QPixmap, QColor, QPainter, QPen, QBrush, QFontMetrics
from PyQt5.QtCore import pyqtSignal, Qt, QTimer, QEvent, QObject, QPointF, QSize, QRectF


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
        self.label_window = LabelWindow(self)
        self.thumbnail_window = ThumbnailWindow(self)

        # Connect the imageSelected signal to update_current_image_path in AnnotationWindow
        self.thumbnail_window.imageSelected.connect(self.annotation_window.update_current_image_path)
        # Connect the imageDeleted signal to delete_image in AnnotationWindow
        self.thumbnail_window.imageDeleted.connect(self.annotation_window.delete_image)
        # Connect thumbnail window to the annotation window for current image selected
        self.annotation_window.imageDeleted.connect(self.thumbnail_window.handle_image_deletion)

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
        self.toolChanged.connect(self.annotation_window.set_selected_tool)

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


class Annotation(QObject):
    color_changed = pyqtSignal(QColor)
    selected = pyqtSignal(object)  # Signal to emit the selected annotation
    annotation_deleted = pyqtSignal(object)  # Signal to emit when the annotation is deleted

    def __init__(self, center_xy: QPointF, annotation_size: int, short_label_code: str, long_label_code: str,
                 color: QColor, image_path: str):
        super().__init__()
        self.id = str(uuid.uuid4())  # Unique identifier
        self.center_xy = center_xy
        self.annotation_size = annotation_size
        self.label = Label(short_label_code, long_label_code, color)  # Create a Label object
        self.image_path = image_path
        self.is_selected = False
        self.graphics_item = None  # QGraphicsRectItem for the annotation

    def to_dict(self):
        return {
            'id': self.id,
            'center_xy': (self.center_xy.x(), self.center_xy.y()),
            'annotation_size': self.annotation_size,
            'label_short_code': self.label.short_label_code,
            'label_long_code': self.label.long_label_code,
            'annotation_color': self.label.color.getRgb(),
            'image_path': self.image_path
        }

    @classmethod
    def from_dict(cls, data):
        center_xy = QPointF(data['center_xy'][0], data['center_xy'][1])
        return cls(center_xy,
                   data['annotation_size'],
                   data['label_short_code'],
                   data['label_long_code'],
                   QColor(*data['annotation_color']),
                   data['image_path'])

    def change_label(self, new_label: 'Label'):
        self.label = new_label
        self.update_graphics_item_color(self.label.color)

    def select(self):
        if not self.is_selected:
            self.is_selected = True
            self.selected.emit(self)
            self.update_graphics_item_color(self.get_inverse_color(self.label.color))

    def deselect(self):
        if self.is_selected:
            self.is_selected = False
            self.update_graphics_item_color(self.label.color)

    def delete(self):
        self.annotation_deleted.emit(self)
        if self.graphics_item:
            self.graphics_item.scene().removeItem(self.graphics_item)
            self.graphics_item = None

    def get_inverse_color(self, color: QColor):
        return QColor(255 - color.red(), 255 - color.green(), 255 - color.blue())

    def to_coralnet_format(self):
        return [os.path.basename(self.image_path),
                int(self.center_xy.y()),
                int(self.center_xy.x()),
                self.label.short_label_code,
                self.annotation_size]

    def create_graphics_item(self, scene):
        half_size = self.annotation_size / 2
        self.graphics_item = QGraphicsRectItem(self.center_xy.x() - half_size,
                                               self.center_xy.y() - half_size,
                                               self.annotation_size,
                                               self.annotation_size)
        self.graphics_item.setPen(QPen(self.label.color, 4))
        self.graphics_item.setData(0, self.id)  # Store the UUID in the graphics item's data
        scene.addItem(self.graphics_item)

    def update_graphics_item(self):
        if self.graphics_item:
            half_size = self.annotation_size / 2
            self.graphics_item.setRect(self.center_xy.x() - half_size,
                                       self.center_xy.y() - half_size,
                                       self.annotation_size,
                                       self.annotation_size)
            self.graphics_item.setPen(QPen(self.label.color, 4))

    def update_graphics_item_color(self, color: QColor):
        if self.graphics_item:
            self.graphics_item.setPen(QPen(color, 4))
            self.graphics_item.update()

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

    def __repr__(self):
        return (f"Annotation(id={self.id}, "
                f"center_xy={self.center_xy}, "
                f"annotation_size={self.annotation_size}, "
                f"annotation_color={self.label.color.name()}, "
                f"image_path={self.image_path}, "
                f"label={self.label.short_label})")


class AnnotationWindow(QGraphicsView):
    imageDeleted = pyqtSignal(str)  # Signal to emit when an image is deleted
    toolChanged = pyqtSignal(str)  # Signal to emit when the tool changes

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.annotation_size = 224
        self.annotation_color = None

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

        self.active_image = False  # Flag to check if the image has been set
        self.selected_image = None  # Initialize selected_image to None
        self.loaded_image_paths = set()  # Initialize the set to store loaded image paths

        self.toolChanged.connect(self.set_selected_tool)

    def set_selected_tool(self, tool):
        self.selected_tool = tool
        if self.selected_tool == "select":
            self.setCursor(Qt.PointingHandCursor)
        elif self.selected_tool == "annotate":
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
            self.hide_cursor_annotation()
        self.unselect_annotation()

    def export_annotations(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Annotations", "", "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_path:
            # Create a dictionary to hold the annotations grouped by image path
            export_dict = {}
            for annotation in self.annotations_dict.values():
                image_path = annotation.image_path
                if image_path not in export_dict:
                    export_dict[image_path] = []
                export_dict[image_path].append(annotation.to_dict())

            with open(file_path, 'w') as file:
                json.dump(export_dict, file, indent=4)

    def import_annotations(self):
        if not self.active_image:
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
            filtered_annotations = {image_path: annotations for image_path, annotations in imported_annotations.items()
                                    if image_path in self.loaded_image_paths}

            # Count the number of images loaded with annotations
            loaded_images_with_annotations = len(filtered_annotations)

            # Display a message box with the information
            QMessageBox.information(self, "Annotations Loaded",
                                    f"Loaded annotations for {loaded_images_with_annotations} "
                                    f"out of {total_images_with_annotations} images.")

            # Add labels to LabelWindow if they are not already present and update annotation colors
            updated_annotations = False
            for image_path, annotations in filtered_annotations.items():
                for annotation_data in annotations:
                    short_label_code = annotation_data['label_short_code']
                    long_label_code = annotation_data['label_long_code']
                    color = QColor(*annotation_data['annotation_color'])
                    # TODO update this
                    self.main_window.label_window.add_label_if_not_exists(short_label_code, long_label_code, color)

                    existing_color = self.main_window.label_window.get_label_color(short_label_code, long_label_code)
                    if existing_color != color:
                        annotation_data['annotation_color'] = existing_color.getRgb()
                        updated_annotations = True

            if updated_annotations:
                QMessageBox.information(self, "Annotations Updated",
                                        "Some annotations have been updated to match the "
                                        "color of the labels already in the project.")

            # Convert annotation data to Annotation objects and store in annotations_dict
            for image_path, annotations in filtered_annotations.items():
                for annotation_data in annotations:
                    annotation = Annotation.from_dict(annotation_data)
                    self.annotations_dict[annotation.id] = annotation

            # Load annotations for all images in the project
            for image_path in self.loaded_image_paths:
                self.load_annotations(image_path)

    def export_coralnet_annotations(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Export CoralNet Annotations", "",
                                                   "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            try:
                data = []
                for annotation in self.annotations_dict.values():
                    data.append(annotation.to_coralnet_format())

                df = pd.DataFrame(data, columns=['Name', 'Row', 'Column', 'Label', 'Patch Size'])
                df.to_csv(file_path, index=False)

            except Exception as e:
                QMessageBox.warning(self, "Error Exporting Annotations",
                                    f"An error occurred while exporting annotations: {str(e)}")

    def set_image(self, image, image_path):
        # Unselect annotation
        self.unselect_annotation()

        # Create a new scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.image_item = QGraphicsPixmapItem(QPixmap(image))
        self.scene.addItem(self.image_item)

        # Fit the image in the view while maintaining the aspect ratio
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        self.active_image = True  # Set the flag to True after the image has been set
        self.toggle_cursor_annotation()  # Hide temp annotation when a new image is set

        # Store the image path as a string
        self.current_image_path = image_path

        # Draw annotations for the current image
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
        if self.active_image:

            if event.button() == Qt.RightButton:
                self.pan_active = True
                self.pan_start = event.pos()
                self.setCursor(Qt.ClosedHandCursor)  # Change cursor to indicate panning

            elif self.selected_tool == "select" and event.button() == Qt.LeftButton:
                position = self.mapToScene(event.pos())
                items = self.scene.items(position)

                # Filter and sort items by z-value
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
        elif (self.selected_tool == "annotate" and self.active_image and self.image_item and
              self.image_item.boundingRect().contains(self.mapToScene(event.pos()))):
            self.toggle_cursor_annotation(self.mapToScene(event.pos()))
        else:
            self.toggle_cursor_annotation()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.RightButton:
            self.pan_active = False
            self.setCursor(Qt.ArrowCursor)  # Reset cursor to default
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

    def cursorInWindow(self, pos):
        return self.selected_image.boundingRect().contains(pos)

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

    def unselect_annotation(self):
        if self.selected_annotation:
            self.selected_annotation.deselect()
            self.selected_annotation = None

    def load_annotations(self, image_path):
        for annotation_id, annotation in self.annotations_dict.items():
            if annotation.image_path == image_path:
                annotation.create_graphics_item(self.scene)
                annotation.color_changed.connect(annotation.update_graphics_item_color)
                annotation.selected.connect(self.select_annotation)
                annotation.annotation_deleted.connect(self.delete_annotation)

    def add_annotation(self, scene_pos: QPointF):

        if not self.selected_label:
            QMessageBox.warning(self, "No Label Selected", "A label must be selected before adding an annotation.")
            return

        # Check if the annotation's center point is within the image bounds
        if not self.active_image or not self.selected_image or not self.cursorInWindow(scene_pos):
            return

        # Create an Annotation object
        annotation = Annotation(scene_pos,
                                self.annotation_size,
                                self.selected_label.color,
                                self.current_image_path,
                                self.selected_label.short_label_code,
                                self.selected_label.long_label_code)

        # Create the graphics item for the annotation
        annotation.create_graphics_item(self.scene)

        # Connect signals
        annotation.color_changed.connect(lambda color: annotation.graphics_item.setPen(QPen(color, 4)))
        annotation.selected.connect(self.select_annotation)
        annotation.annotation_deleted.connect(self.delete_annotation)

        # Store the annotation in the dictionary
        self.annotations_dict[annotation.id] = annotation

        # Push the action onto the undo stack
        self.undo_stack.append(('add', annotation.to_dict()))
        self.redo_stack.clear()  # Clear the redo stack

    def delete_selected_annotation(self):
        if self.selected_annotation:
            self.delete_annotation(self.selected_annotation.id)
            self.selected_annotation = None

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

    def toggle_cursor_annotation(self, scene_pos: QPointF = None):
        if self.cursor_annotation:
            # Hide the cursor annotation if it exists
            self.scene.removeItem(self.cursor_annotation)
            self.cursor_annotation = None
        elif scene_pos:
            # Show the cursor annotation if a position is provided
            if not self.selected_label:
                return

            half_size = self.annotation_size / 2
            self.cursor_annotation = QGraphicsRectItem(scene_pos.x() - half_size,
                                                       scene_pos.y() - half_size,
                                                       self.annotation_size,
                                                       self.annotation_size)

            self.cursor_annotation.setPen(QPen(self.annotation_color, 4))
            self.scene.addItem(self.cursor_annotation)

    def delete_image(self, image_path):
        if image_path in self.annotations_dict:
            for annotation_id in list(self.annotations_dict.keys()):
                annotation = self.annotations_dict[annotation_id]
                if annotation.image_path == image_path:
                    annotation.delete()
                    del self.annotations_dict[annotation_id]

        if self.current_image_path == image_path:
            self.scene.clear()
            self.current_image_path = None
            self.selected_image = None  # Reset selected_image to None
            self.active_image = False  # Reset image_set flag

        self.imageDeleted.emit(image_path)  # Emit the signal when an image is deleted

    def delete_annotations_for_label(self, short_label_code, long_label):
        for annotation_id in list(self.annotations_dict.keys()):
            annotation = self.annotations_dict[annotation_id]
            if annotation.label_short_code == short_label_code and annotation.label_long_code == long_label:
                annotation.delete()
                del self.annotations_dict[annotation_id]


class ThumbnailWindow(QWidget):
    imageSelected = pyqtSignal(str)  # Signal to emit the selected image path
    imageDeleted = pyqtSignal(str)  # Signal to emit the deleted image path

    def __init__(self, main_window):
        super().__init__()

        self.annotation_window = main_window.annotation_window
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


from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QScrollArea, QGridLayout, QMessageBox, QDialog, QLineEdit, QColorDialog, QLabel, QCheckBox
from PyQt5.QtGui import QColor, QFontMetrics
from PyQt5.QtCore import Qt, pyqtSignal
import random
import uuid

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

    def __init__(self, short_label_code, long_label_code, color=QColor(255, 255, 255), fixed_width=80):
        super().__init__()

        self.id = str(uuid.uuid4())  # Unique identifier
        self.short_label_code = short_label_code
        self.long_label_code = long_label_code
        self.color = color
        self.is_selected = False
        self.fixed_width = fixed_width

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.short_label_widget = QLabel(self.short_label_code)
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
        self.setToolTip(self.long_label_code)

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
    label_selected = pyqtSignal(str, str, QColor)  # Signal to emit label details

    def __init__(self, main_window, label_width=80):
        super().__init__()

        self.annotation_window = main_window.annotation_window
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
        default_short_label_code = "Review"
        default_long_label_code = "Review"
        default_color = QColor(255, 255, 255)  # White color
        self.add_label(default_short_label_code, default_long_label_code, default_color)
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
            short_label_code, long_label_code, color = dialog.get_label_details()
            if self.label_exists(short_label_code, long_label_code):
                QMessageBox.warning(self, "Label Exists", "A label with the same short and long name already exists.")
            else:
                new_label = self.add_label(short_label_code, long_label_code, color)
                self.set_active_label(new_label)

    def add_label(self, short_label_code, long_label_code, color):
        label = Label(short_label_code, long_label_code, color, self.label_width)
        label.selected.connect(self.set_active_label)
        label.label_deleted.connect(self.delete_label)
        self.labels.append(label)

        self.update_labels_per_row()
        self.reorganize_labels()

        return label

    def get_label_color(self, short_label_code, long_label_code):
        label_color = None
        for label in self.labels:
            if short_label_code == label.short_label_code and long_label_code == label.long_label_code:
                label_color = label.color
        return label_color

    def label_exists(self, short_label_code, long_label_code):
        for label in self.labels:
            if label.short_label_code == short_label_code and label.long_label_code == long_label_code:
                return True
        return False

    def add_label_if_not_exists(self, short_label_code, long_label_code, color):
        if not self.label_exists(short_label_code, long_label_code):
            self.add_label(short_label_code, long_label_code, color)

    def set_active_label(self, selected_label):
        if self.active_label and self.active_label != selected_label:
            self.active_label.deselect()

        self.active_label = selected_label
        self.active_label.select()
        print(f"Active label: {self.active_label.short_label_code}")
        self.label_selected.emit(selected_label.short_label_code, selected_label.long_label_code, selected_label.color)

    def select_label_for_annotation(self, label_id):
        for lbl in self.labels:
            if lbl.id == label_id:
                self.set_active_label(lbl)
                break

    def delete_label(self, label):
        if label.short_label_code == "Review" and label.long_label_code == "Review" and label.color == QColor(255, 255, 255):
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
        self.annotation_window.delete_annotations_for_label(label.short_label_code, label.long_label_code)

        # Reset active label if it was deleted
        if self.active_label == label:
            self.active_label = None
            if self.labels:
                self.set_active_label(self.labels[0])

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


if __name__ == "__main__":
    app = QApplication([])
    app.setStyle('WindowsXP')
    main_window = MainWindow()
    main_window.show()
    app.exec_()