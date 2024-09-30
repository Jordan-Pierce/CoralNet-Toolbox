import warnings

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent

from toolbox.Tools.QtTool import Tool

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
        if event.button() == Qt.LeftButton:
            self.annotation_window.unselect_annotation()
            self.annotation_window.add_annotation(self.annotation_window.mapToScene(event.pos()))

    def mouseMoveEvent(self, event: QMouseEvent):
        if (self.annotation_window.active_image and
                self.annotation_window.image_pixmap and
                self.annotation_window.cursorInWindow(event.pos())):
            self.annotation_window.toggle_cursor_annotation(self.annotation_window.mapToScene(event.pos()))
        else:
            self.annotation_window.toggle_cursor_annotation()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.annotation_window.toggle_cursor_annotation()