import warnings

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QMouseEvent, QColor
from PyQt5.QtWidgets import QGraphicsPixmapItem

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
        
        # Crosshair settings
        self.show_crosshair = True  # Flag to toggle crosshair visibility for this tool
        self.h_crosshair_line = None
        self.v_crosshair_line = None

    def activate(self):
        self.active = True
        self.annotation_window.setCursor(self.cursor)

    def deactivate(self):
        self.active = False
        self.annotation_window.setCursor(self.default_cursor)
        self.clear_cursor_annotation()
        
        # Ensure crosshair is properly cleared when deactivating tool
        self.clear_crosshair()
        
        # Stop any current drawing operation
        self.stop_current_drawing()

    def mousePressEvent(self, event: QMouseEvent):
        pass

    def mouseMoveEvent(self, event: QMouseEvent):
        """
        Base implementation of mouseMoveEvent that handles crosshair display.
        Child classes should call super().mouseMoveEvent(event) in their implementation.
        """
        # Handle crosshair display
        scene_pos = self.annotation_window.mapToScene(event.pos())
        cursor_in_window = self.annotation_window.cursorInWindow(event.pos())
        
        if (cursor_in_window and self.active and 
            self.annotation_window.selected_label and 
            self.show_crosshair):
            self.update_crosshair(scene_pos)
        else:
            self.clear_crosshair()

    def mouseReleaseEvent(self, event: QMouseEvent):
        pass

    def keyPressEvent(self, event):
        pass

    def keyReleaseEvent(self, event):
        pass

    def wheelEvent(self, event: QMouseEvent):
        pass
        
    def stop_current_drawing(self):
        """
        Force stop of the current drawing operation if one is in progress.
        Subclasses should override this to implement tool-specific stopping logic.
        """
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
            
    def draw_crosshair(self, scene_pos):
        """
        Draw crosshair guides at the current cursor position.
        
        Args:
            scene_pos: Position in scene coordinates where to draw the crosshair
        """
        # Only draw if we have an active image and scene position
        if (
            not self.show_crosshair
            or not self.annotation_window.active_image
            or not scene_pos
            or not self.annotation_window.pixmap_image
        ):
            return

        # Remove any existing crosshair lines
        self.clear_crosshair()
        
        # Get image bounds
        image_rect = QGraphicsPixmapItem(self.annotation_window.pixmap_image).boundingRect()
        
        # Create horizontal line across the full width of the image
        self.h_crosshair_line = self.graphics_utility.create_guide_line(
            QPointF(image_rect.left(), scene_pos.y()),
            QPointF(image_rect.right(), scene_pos.y())
        )
        self.annotation_window.scene.addItem(self.h_crosshair_line)
        
        # Create vertical line across the full height of the image
        self.v_crosshair_line = self.graphics_utility.create_guide_line(
            QPointF(scene_pos.x(), image_rect.top()),
            QPointF(scene_pos.x(), image_rect.bottom())
        )
        self.annotation_window.scene.addItem(self.v_crosshair_line)
        
    def clear_crosshair(self):
        """Remove any crosshair guide lines from the scene."""
        if self.h_crosshair_line and self.h_crosshair_line.scene():
            self.annotation_window.scene.removeItem(self.h_crosshair_line)
            self.h_crosshair_line = None
        if self.v_crosshair_line and self.v_crosshair_line.scene():
            self.annotation_window.scene.removeItem(self.v_crosshair_line)
            self.v_crosshair_line = None

    def update_crosshair(self, scene_pos):
        """
        Update the crosshair position. This is a convenience method that
        clears and redraws the crosshair.
        
        Args:
            scene_pos: New position for the crosshair
        """
        self.clear_crosshair()
        self.draw_crosshair(scene_pos)