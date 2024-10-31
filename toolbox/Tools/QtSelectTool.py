import warnings
import math
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QMouseEvent, QPen, QBrush
from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsPolygonItem, QGraphicsEllipseItem
from toolbox.Tools.QtTool import Tool
from toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SelectTool(Tool):
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.cursor = Qt.PointingHandCursor
        self.resizing = False
        self.moving = False
        self.resize_handle = None
        self.resize_start_pos = None
        self.move_start_pos = None
        self.resize_handles = []  # List to store resize handles

        self.buffer = 50

        self.selected_annotations = []

        # Listen for the annotation changed signals
        self.annotation_window.annotationSelected.connect(self.annotation_changed)
        self.annotation_window.annotationSizeChanged.connect(self.annotation_changed)
        self.annotation_window.annotationDeleted.connect(self.clear_resize_handles)

    def mousePressEvent(self, event: QMouseEvent):
        if not self.annotation_window.cursorInWindow(event.pos()):
            return

        if event.button() == Qt.LeftButton:
            position = self.annotation_window.mapToScene(event.pos())
            items = self.annotation_window.scene.items(position)
            rect_items = [item for item in items if isinstance(item, QGraphicsRectItem)]
            polygon_items = [item for item in items if isinstance(item, QGraphicsPolygonItem)]
            all_items = rect_items + polygon_items
            all_items.sort(key=lambda item: item.zValue(), reverse=True)

            # Filter items based on proximity to the center
            center_proximity_items = []
            for item in all_items:
                annotation_id = item.data(0)
                annotation = self.annotation_window.annotations_dict.get(annotation_id)
                if annotation and annotation.contains_point(position):
                    center = annotation.get_center_xy()
                    distance_to_center = (position - center).manhattanLength()
                    center_proximity_items.append((item, distance_to_center))

            # Sort items by proximity to the center
            center_proximity_items.sort(key=lambda x: x[1])

            # Select the closest item to the center
            for item, _ in center_proximity_items:
                annotation_id = item.data(0)
                selected_annotation = self.annotation_window.annotations_dict.get(annotation_id)
                if selected_annotation:
                    if selected_annotation in self.annotation_window.selected_annotations:
                        if event.modifiers() & Qt.ControlModifier:
                            self.annotation_window.unselect_annotation(selected_annotation)
                            return
                        else:
                            self.annotation_window.unselect_annotations()
                            self.annotation_window.select_annotation(selected_annotation)
                            self.annotation_window.drag_start_pos = position
                            self.moving = True
                            self.move_start_pos = position
                            break
                    else:
                        if not (event.modifiers() & Qt.ControlModifier):
                            self.annotation_window.unselect_annotations()
                        self.annotation_window.select_annotation(selected_annotation, ctrl_pressed=event.modifiers() & Qt.ControlModifier)
                        self.annotation_window.drag_start_pos = position

                        if event.modifiers() & Qt.ControlModifier:
                            self.resize_handle = self.detect_resize_handle(selected_annotation, position)
                            if self.resize_handle:
                                self.resizing = True
                                self.resize_start_pos = position
                                break
                        else:
                            self.moving = True
                            self.move_start_pos = position
                            break

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.annotation_window.cursorInWindow(event.pos()):
            return

        current_pos = self.annotation_window.mapToScene(event.pos())

        if self.resizing and self.resize_handle and event.modifiers() & Qt.ControlModifier:
            if len(self.annotation_window.selected_annotations) == 1:
                for selected_annotation in self.annotation_window.selected_annotations:
                    self.resize_annotation(selected_annotation, current_pos)
                    self.display_resize_handles(selected_annotation)

            self.resize_start_pos = current_pos

        elif self.moving and event.buttons() & Qt.LeftButton and not event.modifiers() & Qt.ControlModifier:
            if self.annotation_window.selected_annotations:
                for selected_annotation in self.annotation_window.selected_annotations:
                    delta = current_pos - self.move_start_pos
                    new_center = selected_annotation.center_xy + delta

                    if self.annotation_window.cursorInWindow(current_pos, mapped=True):
                        rasterio_image = self.annotation_window.rasterio_image
                        self.annotation_window.set_annotation_location(selected_annotation.id, new_center)
                        selected_annotation.create_cropped_image(rasterio_image)
                        self.annotation_window.main_window.confidence_window.display_cropped_image(selected_annotation)
                        self.move_start_pos = current_pos

    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self.annotation_window.cursorInWindow(event.pos()):
            return

        self.selected_annotations = self.annotation_window.selected_annotations

        if self.resizing:
            self.resizing = False
            self.resize_handle = None
            self.resize_start_pos = None
            for selected_annotation in self.selected_annotations:
                self.display_resize_handles(selected_annotation)
        if self.moving:
            self.moving = False
            self.move_start_pos = None
        self.annotation_window.drag_start_pos = None

    def keyPressEvent(self, event):
        if not self.annotation_window.selected_annotations:
            return

        self.selected_annotations = self.annotation_window.selected_annotations
        if event.modifiers() & Qt.ControlModifier:
            if len(self.selected_annotations) == 1:
                self.display_resize_handles(self.selected_annotations[0])

    def keyReleaseEvent(self, event):
        if not self.annotation_window.selected_annotations:
            return

        if not event.modifiers() & Qt.ControlModifier:
            self.remove_resize_handles()

    def wheelEvent(self, event: QMouseEvent):
        # Handle Zoom wheel for setting annotation size
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.annotation_window.set_annotation_size(delta=16)  # Zoom in
            else:
                self.annotation_window.set_annotation_size(delta=-16)  # Zoom out

    def annotation_changed(self, annotation_id):
        # Clear the resize handles if the selected annotation changed
        # via clicking or cycling through the annotations
        self.clear_resize_handles()

    def get_rectangle_handles(self, annotation):
        top_left = annotation.top_left
        bottom_right = annotation.bottom_right
        handles = {
            "left": QPointF(top_left.x(), (top_left.y() + bottom_right.y()) / 2),
            "right": QPointF(bottom_right.x(), (top_left.y() + bottom_right.y()) / 2),
            "top": QPointF((top_left.x() + bottom_right.x()) / 2, top_left.y()),
            "bottom": QPointF((top_left.x() + bottom_right.x()) / 2, bottom_right.y()),
            "top_left": QPointF(top_left.x(), top_left.y()),
            "top_right": QPointF(bottom_right.x(), top_left.y()),
            "bottom_left": QPointF(top_left.x(), bottom_right.y()),
            "bottom_right": QPointF(bottom_right.x(), bottom_right.y())
        }
        return handles

    def get_polygon_handles(self, annotation):
        handles = {f"point_{i}": QPointF(point.x(), point.y()) for i, point in enumerate(annotation.points)}
        return handles

    def detect_resize_handle(self, annotation, current_pos):
        if isinstance(annotation, RectangleAnnotation):
            handles = self.get_rectangle_handles(annotation)
        elif isinstance(annotation, PolygonAnnotation):
            handles = self.get_polygon_handles(annotation)
        else:
            return None

        closest_handle = None
        min_distance = float('inf')

        for handle, point in handles.items():
            # Calculate the distance from the current position to the handle
            distance = math.hypot(point.x() - current_pos.x(), point.y() - current_pos.y())
            # Check if the distance is within the buffer
            if distance <= self.buffer * 2 and distance < min_distance:
                closest_handle = handle
                min_distance = distance

        return closest_handle

    def display_resize_handles(self, annotation):
        self.remove_resize_handles()
        if isinstance(annotation, RectangleAnnotation):
            handles = self.get_rectangle_handles(annotation)
        elif isinstance(annotation, PolygonAnnotation):
            handles = self.get_polygon_handles(annotation)
        else:
            return

        handle_size = 10
        for handle, point in handles.items():
            ellipse = QGraphicsEllipseItem(point.x() - handle_size//2,
                                           point.y() - handle_size//2,
                                           handle_size,
                                           handle_size)

            ellipse.setPen(QPen(annotation.label.color))
            ellipse.setBrush(QBrush(annotation.label.color))
            self.annotation_window.scene.addItem(ellipse)
            self.resize_handles.append(ellipse)

    def remove_resize_handles(self):
        for handle in self.resize_handles:
            self.annotation_window.scene.removeItem(handle)
        self.resize_handles.clear()

    def resize_annotation(self, annotation, new_pos):
        if annotation is None:
            print("Warning: Attempted to resize a None annotation.")
            return

        if not hasattr(annotation, 'resize'):
            print(f"Warning: Annotation of type {type(annotation)} does not have a resize method.")
            return

        annotation.resize(self.resize_handle, new_pos)

    def clear_resize_handles(self):
        self.remove_resize_handles()