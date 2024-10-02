import warnings

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsPolygonItem

from toolbox.Tools.QtTool import Tool

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SelectTool(Tool):
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.cursor = Qt.PointingHandCursor

    def mousePressEvent(self, event: QMouseEvent):
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
                    break

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.LeftButton and self.annotation_window.selected_annotation:
            # Get the current position of the mouse
            current_pos = self.annotation_window.mapToScene(event.pos())
            # If the drag start position is not set, set it
            if not self.annotation_window.drag_start_pos:
                self.annotation_window.drag_start_pos = current_pos

            # Calculate the delta between the current position and the drag start position
            delta = current_pos - self.annotation_window.drag_start_pos
            new_center = self.annotation_window.selected_annotation.center_xy + delta

            # Check if the current position is in the window, update annotation, graphics
            if self.annotation_window.cursorInWindow(current_pos, mapped=True):
                selected_annotation = self.annotation_window.selected_annotation
                rasterio_image = self.annotation_window.rasterio_image
                self.annotation_window.set_annotation_location(selected_annotation.id, new_center)
                self.annotation_window.selected_annotation.create_cropped_image(rasterio_image)
                self.annotation_window.main_window.confidence_window.display_cropped_image(selected_annotation)
                self.annotation_window.drag_start_pos = current_pos

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.annotation_window.drag_start_pos = None