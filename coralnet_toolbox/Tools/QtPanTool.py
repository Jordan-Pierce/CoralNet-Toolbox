import warnings

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent

from coralnet_toolbox.Tools.QtTool import Tool

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class PanTool(Tool):
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.cursor = Qt.ClosedHandCursor

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.RightButton:
            self.annotation_window.pan_active = True
            self.annotation_window.pan_start = event.pos()
            self.annotation_window.setCursor(self.cursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.annotation_window.pan_active:
            self.pan(event.pos())

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.RightButton:
            self.annotation_window.pan_active = False
            self.annotation_window.setCursor(Qt.ArrowCursor)

    def pan(self, pos):
        delta = pos - self.annotation_window.pan_start
        self.annotation_window.pan_start = pos
        x_value = self.annotation_window.horizontalScrollBar().value() - delta.x()
        y_value = self.annotation_window.verticalScrollBar().value() - delta.y()
        self.annotation_window.horizontalScrollBar().setValue(x_value)
        self.annotation_window.verticalScrollBar().setValue(y_value)