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

        self.buffer = 10

        # Listen for the annotationDeleted signal
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

            for item in all_items:
                annotation_id = item.data(0)
                annotation = self.annotation_window.annotations_dict.get(annotation_id)
                if annotation and annotation.contains_point(position):
                    self.annotation_window.select_annotation(annotation)
                    self.annotation_window.drag_start_pos = position

                    if event.modifiers() & Qt.ControlModifier:
                        self.resize_handle = self.detect_resize_handle(annotation, position)
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
            if self.annotation_window.selected_annotation is not None:
                annotation = self.annotation_window.selected_annotation
                self.resize_annotation(annotation, current_pos)
                self.display_resize_handles(annotation)

            self.resize_start_pos = current_pos
            self.resize_annotation(self.annotation_window.selected_annotation, current_pos)

        elif self.moving and event.buttons() & Qt.LeftButton and not event.modifiers() & Qt.ControlModifier:
            if self.annotation_window.selected_annotation:
                delta = current_pos - self.move_start_pos
                new_center = self.annotation_window.selected_annotation.center_xy + delta

                if self.annotation_window.cursorInWindow(current_pos, mapped=True):
                    selected_annotation = self.annotation_window.selected_annotation
                    rasterio_image = self.annotation_window.rasterio_image
                    self.annotation_window.set_annotation_location(selected_annotation.id, new_center)
                    self.annotation_window.selected_annotation.create_cropped_image(rasterio_image)
                    self.annotation_window.main_window.confidence_window.display_cropped_image(selected_annotation)
                    self.move_start_pos = current_pos

    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self.annotation_window.cursorInWindow(event.pos()):
            return

        annotation = self.annotation_window.selected_annotation

        if self.resizing:
            self.resizing = False
            self.resize_handle = None
            self.resize_start_pos = None
            self.display_resize_handles(annotation)
        if self.moving:
            self.moving = False
            self.move_start_pos = None
        self.annotation_window.drag_start_pos = None

    def keyPressEvent(self, event):
        if self.annotation_window.selected_annotation is None:
            return

        annotation = self.annotation_window.selected_annotation
        if event.modifiers() & Qt.ControlModifier:
            self.display_resize_handles(annotation)

    def keyReleaseEvent(self, event):
        if self.annotation_window.selected_annotation is None:
            return

        if not event.modifiers() & Qt.ControlModifier:
            self.remove_resize_handles()

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

        for handle, point in handles.items():
            # Calculate the distance between the clicked position and the point
            if math.hypot(point.x() - current_pos.x(), point.y() - current_pos.y()) <= self.buffer:
                return handle

        return None

    def display_resize_handles(self, annotation):
        self.remove_resize_handles()
        if isinstance(annotation, RectangleAnnotation):
            handles = self.get_rectangle_handles(annotation)
        elif isinstance(annotation, PolygonAnnotation):
            handles = self.get_polygon_handles(annotation)
        else:
            return

        for handle, point in handles.items():
            ellipse = QGraphicsEllipseItem(point.x() - self.buffer//2,
                                           point.y() - self.buffer//2,
                                           self.buffer,
                                           self.buffer)

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

    def clear_resize_handles(self, annotation_id):
        self.remove_resize_handles()
