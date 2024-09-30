import warnings

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QGraphicsRectItem

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
            rect_items.sort(key=lambda item: item.zValue(), reverse=True)

            for rect_item in rect_items:
                annotation_id = rect_item.data(0)
                annotation = self.annotation_window.annotations_dict.get(annotation_id)
                if annotation.contains_point(position):
                    self.annotation_window.select_annotation(annotation)
                    self.annotation_window.drag_start_pos = position
                    break

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.annotation_window.selected_annotation:
            if event.buttons() & Qt.LeftButton:
                current_pos = self.annotation_window.mapToScene(event.pos())
                if hasattr(self.annotation_window, 'drag_start_pos'):
                    if not self.annotation_window.drag_start_pos:
                        self.annotation_window.drag_start_pos = current_pos
                    delta = current_pos - self.annotation_window.drag_start_pos
                    new_center = self.annotation_window.selected_annotation.center_xy + delta
                    if self.annotation_window.cursorInWindow(current_pos, mapped=True) and self.annotation_window.selected_annotation:
                        self.annotation_window.set_annotation_location(self.annotation_window.selected_annotation.id, new_center)
                        self.annotation_window.selected_annotation.create_cropped_image(self.annotation_window.rasterio_image)
                        self.annotation_window.main_window.confidence_window.display_cropped_image(self.annotation_window.selected_annotation)
                        self.annotation_window.drag_start_pos = current_pos

    def mouseReleaseEvent(self, event: QMouseEvent):
        if hasattr(self.annotation_window, 'drag_start_pos'):
            del self.annotation_window.drag_start_pos