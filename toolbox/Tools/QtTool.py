import warnings

from PyQt5.QtGui import QMouseEvent

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class Tool:
    def __init__(self, annotation_window):
        self.annotation_window = annotation_window

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