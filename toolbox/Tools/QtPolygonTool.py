import warnings

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QMouseEvent, QKeyEvent
from PyQt5.QtWidgets import QMessageBox

from toolbox.Tools.QtTool import Tool
from toolbox.QtPolygonAnnotation import PolygonAnnotation

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class PolygonTool(Tool):
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.cursor = Qt.CrossCursor
        self.points = []
        self.complete = False

    def activate(self):
        self.active = True
        self.annotation_window.setCursor(Qt.CrossCursor)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.annotation_window.unselect_annotation()
            self.annotation_window.add_annotation(self.annotation_window.mapToScene(event.pos()))

    def mouseMoveEvent(self, event: QMouseEvent):
        active_image = self.annotation_window.active_image
        image_pixmap = self.annotation_window.image_pixmap
        cursor_in_window = self.annotation_window.cursorInWindow(event.pos())
        if active_image and image_pixmap and cursor_in_window and self.points:
            self.annotation_window.toggle_cursor_annotation(self.annotation_window.mapToScene(event.pos()))

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.annotation_window.toggle_cursor_annotation()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space and len(self.points) > 2:
            self.annotation_window.add_annotation(None)

    def create_annotation(self, scene_pos: QPointF, finished: bool = False):

        if not self.annotation_window.selected_label:
            QMessageBox.warning(self.annotation_window,
                                "No Label Selected",
                                "A label must be selected before adding an annotation.")
            return None

        if not self.annotation_window.active_image or not self.annotation_window.image_pixmap:
            return None

        if finished and len(self.points) > 2:
            # Close the polygon
            self.points.append(self.points[0])
            self.complete = True
        else:
            self.points.append(scene_pos)

        # Create the annotation
        annotation = PolygonAnnotation(self.points,
                                       self.annotation_window.selected_label.short_label_code,
                                       self.annotation_window.selected_label.long_label_code,
                                       self.annotation_window.selected_label.color,
                                       self.annotation_window.current_image_path,
                                       self.annotation_window.selected_label.id,
                                       self.annotation_window.main_window.label_window.active_label.transparency,
                                       show_msg=False)

        if self.complete:
            self.points = []
            self.complete = False

        return annotation