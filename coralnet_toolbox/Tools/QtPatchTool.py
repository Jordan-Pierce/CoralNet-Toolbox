import warnings

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QMessageBox

from coralnet_toolbox.Tools.QtTool import Tool
from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class PatchTool(Tool):
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.cursor = Qt.CrossCursor

    def activate(self):
        self.active = True
        self.annotation_window.setCursor(Qt.CrossCursor)

    def mousePressEvent(self, event: QMouseEvent):

        if not self.annotation_window.selected_label:
            QMessageBox.warning(self.annotation_window,
                                "No Label Selected",
                                "A label must be selected before adding an annotation.")
            return None

        if event.button() == Qt.LeftButton:
            self.annotation_window.unselect_annotations()
            self.annotation_window.add_annotation(self.annotation_window.mapToScene(event.pos()))

    def mouseMoveEvent(self, event: QMouseEvent):
        active_image = self.annotation_window.active_image
        image_pixmap = self.annotation_window.image_pixmap
        cursor_in_window = self.annotation_window.cursorInWindow(event.pos())
        if active_image and image_pixmap and cursor_in_window:
            self.annotation_window.toggle_cursor_annotation(self.annotation_window.mapToScene(event.pos()))

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.annotation_window.toggle_cursor_annotation()

    def wheelEvent(self, event: QMouseEvent):
        # Handle Zoom wheel for setting annotation size
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.annotation_window.set_annotation_size(delta=16)  # Zoom in
            else:
                self.annotation_window.set_annotation_size(delta=-16)  # Zoom out

            self.annotation_window.toggle_cursor_annotation(self.annotation_window.mapToScene(event.pos()))

    def create_annotation(self, scene_pos: QPointF, finished: bool = False):

        annotation = PatchAnnotation(scene_pos,
                                     self.annotation_window.annotation_size,
                                     self.annotation_window.selected_label.short_label_code,
                                     self.annotation_window.selected_label.long_label_code,
                                     self.annotation_window.selected_label.color,
                                     self.annotation_window.current_image_path,
                                     self.annotation_window.selected_label.id,
                                     transparency=self.annotation_window.selected_label.transparency)
        return annotation