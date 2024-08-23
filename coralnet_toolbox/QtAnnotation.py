import os
import uuid
import json
import random

import pandas as pd

from coralnet_toolbox.QtLabel import Label
from coralnet_toolbox.QtProgressBar import ProgressBar

from PyQt5.QtWidgets import (QFileDialog, QApplication, QGraphicsView, QGraphicsScene, QMessageBox, QCheckBox,
                             QVBoxLayout, QLabel, QDialog, QHBoxLayout, QPushButton, QComboBox, QSpinBox,
                             QGraphicsPixmapItem, QGraphicsRectItem, QFormLayout, QInputDialog)

from PyQt5.QtGui import QMouseEvent, QImage, QPixmap, QColor, QPen, QBrush, QPixmapCache
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QPointF, QRectF

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


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

    def create_cropped_image(self, pixmap: QPixmap):
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

        self.image_pixmap = None
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

                total_annotations = len(list(self.annotations_dict.values()))

                progress_bar = ProgressBar(self, title="Exporting Annotations")
                progress_bar.show()
                progress_bar.start_progress(total_annotations)

                export_dict = {}
                for annotation in self.annotations_dict.values():
                    image_path = annotation.image_path
                    if image_path not in export_dict:
                        export_dict[image_path] = []
                    export_dict[image_path].append(annotation.to_dict())

                    progress_bar.update_progress()
                    QApplication.processEvents()  # Update GUI

                with open(file_path, 'w') as file:
                    json.dump(export_dict, file, indent=4)
                    file.flush()  # Ensure the data is written to the file

                progress_bar.stop_progress()
                progress_bar.close()

                QMessageBox.information(self,
                                        "Annotations Exported",
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
                self.selected_annotation.create_cropped_image(self.image_pixmap)
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
            self.selected_annotation.create_cropped_image(self.image_pixmap)
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

        # Clean up
        self.unselect_annotation()
        self.scene.clear()
        QPixmapCache.clear()

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.image_pixmap = QPixmap(image)
        self.current_image_path = image_path
        self.active_image = True

        self.imageLoaded.emit(image.width(), image.height())

        self.scene.addItem(QGraphicsPixmapItem(self.image_pixmap))
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
                # Annotation cannot be selected in annotate mode
                self.unselect_annotation()
                # Add annotation to the scene
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
                self.selected_annotation.create_cropped_image(self.image_pixmap)
                self.main_window.confidence_window.display_cropped_image(self.selected_annotation)
                self.drag_start_pos = current_pos  # Update the start position for smooth dragging
        elif (self.selected_tool == "annotate" and
              self.active_image and self.image_pixmap and
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
        if self.image_pixmap:
            image_rect = QGraphicsPixmapItem(self.image_pixmap).boundingRect()
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
                self.selected_annotation.create_cropped_image(self.image_pixmap)
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
                annotation.create_cropped_image(self.image_pixmap)

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

        if not self.active_image or not self.image_pixmap or not self.cursorInWindow(scene_pos, mapped=True):
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
        annotation.create_cropped_image(self.image_pixmap)

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
        # Called by ImageWindow
        annotation_ids_to_delete = [i for i, a in self.annotations_dict.items() if a.image_path == image_path]

        for annotation_id in annotation_ids_to_delete:
            annotation = self.annotations_dict[annotation_id]
            annotation.delete()
            del self.annotations_dict[annotation_id]

        if self.current_image_path == image_path:
            self.scene.clear()
            self.current_image_path = None
            self.image_pixmap = None
            self.active_image = False  # Reset image_set flag

    def delete_annotations_for_label(self, label):
        for annotation in list(self.annotations_dict.values()):
            if annotation.label.id == label.id:
                annotation.delete()
                del self.annotations_dict[annotation.id]


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
                                                           self.annotation_window.image_pixmap.width(),
                                                           self.annotation_window.image_pixmap.height())

        self.draw_annotation_previews(margins)

    def draw_annotation_previews(self, margins):

        margin_x_min, margin_y_min, margin_x_max, margin_y_max = margins

        self.preview_scene.clear()
        pixmap = self.annotation_window.image_pixmap
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
            current_image_index = self.image_window.image_paths.index(current_image_path)
            image_paths = self.image_window.image_paths[current_image_index:]
        elif self.apply_to_all:
            image_paths = list(self.annotation_window.loaded_image_paths)
        else:
            image_paths = [self.annotation_window.current_image_path]

        # Create and show the progress bar
        progress_bar = ProgressBar(self, title="Sampling Annotations")
        progress_bar.show()
        progress_bar.start_progress(num_annotations)

        for image_path in image_paths:
            # Load the QImage
            image = QImage(image_path)
            # Get the pixmap once
            image_pixmap = QPixmap(image)

            # Sample the annotation, given params
            annotations = self.sample_annotations(method,
                                                  num_annotations,
                                                  annotation_size,
                                                  margins,
                                                  image_pixmap.width(),
                                                  image_pixmap.height())

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
                    new_annotation.create_cropped_image(image_pixmap)

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