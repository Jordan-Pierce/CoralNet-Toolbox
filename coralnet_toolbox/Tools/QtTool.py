import warnings

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QMouseEvent

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Tool:
    def __init__(self, annotation_window):
        self.annotation_window = annotation_window
        self.main_window = annotation_window.main_window
        self.graphics_utility = self.annotation_window.graphics_utility

        self.active = False
        self.cursor = Qt.ArrowCursor
        self.default_cursor = Qt.ArrowCursor
        self.cursor_annotation = None

    def activate(self):
        self.active = True
        self.annotation_window.setCursor(self.cursor)

    def deactivate(self):
        self.active = False
        self.annotation_window.setCursor(self.default_cursor)
        self.clear_cursor_annotation()

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
        
    def create_cursor_annotation(self, scene_pos: QPointF = None):
        """
        Create and display a cursor annotation at the given position.
        Subclasses should override this method to create the appropriate annotation type.
        
        Args:
            scene_pos: Position in scene coordinates where to create the annotation
        """
        pass
        
    def update_cursor_annotation(self, scene_pos: QPointF = None):
        """
        Update the existing cursor annotation to a new position or with new properties.
        Subclasses should override this method to update the annotation appropriately.
        
        Args:
            scene_pos: New position for the cursor annotation
        """
        if self.cursor_annotation:
            self.clear_cursor_annotation()
            self.create_cursor_annotation(scene_pos)
    
    def clear_cursor_annotation(self):
        """
        Clear the current cursor annotation if it exists.
        """
        if self.cursor_annotation:
            self.cursor_annotation.delete()
            self.cursor_annotation = None