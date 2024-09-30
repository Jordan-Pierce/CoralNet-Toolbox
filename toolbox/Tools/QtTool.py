import warnings

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class Tool:
    def __init__(self, annotation_window):
        self.annotation_window = annotation_window
        self.active = False
        self.cursor = Qt.ArrowCursor
        self.default_cursor = Qt.ArrowCursor

    def activate(self):
        self.active = True
        self.annotation_window.setCursor(self.cursor)

    def deactivate(self):
        self.active = False
        self.annotation_window.setCursor(self.default_cursor)

    def mousePressEvent(self, event: QMouseEvent):
        pass

    def mouseMoveEvent(self, event: QMouseEvent):
        pass

    def mouseReleaseEvent(self, event: QMouseEvent):
        pass

    def keyPressEvent(self, event):
        pass

    def keyReleaseEvent(self, event):
        pass

    def wheelEvent(self, event: QMouseEvent):
        pass