import warnings

from PyQt5.QtCore import Qt, QPointF, QTimer
from PyQt5.QtGui import QMouseEvent, QColor, QPen
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsLineItem

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Tool:
    def __init__(self, annotation_window):
        self.annotation_window = annotation_window
        self.main_window = annotation_window.main_window

        self.active = False
        self.cursor = Qt.ArrowCursor
        self.default_cursor = Qt.ArrowCursor
        self.cursor_annotation = None

        # Optional callbacks for cursor preview propagation (set by MVATManager or any consumer).
        # Subclasses that support live cursor previews should call these in their mouseMoveEvent.
        # cursor_move_callback(scene_pos: QPointF, item_factory: callable)
        #   item_factory(u: float, v: float) -> QGraphicsItem  (positioned at image coords u,v)
        self.cursor_move_callback = None
        # cursor_clear_callback()
        self.cursor_clear_callback = None

        # Debounce timer — throttle MVAT context-canvas projection to ~10 Hz.
        # Subclasses call _schedule_cursor_preview_update(scene_pos) in mouseMoveEvent;
        # the base deactivate() and _fire_cursor_preview_update() handle the rest.
        self._cursor_update_timer = QTimer()
        self._cursor_update_timer.setSingleShot(True)
        self._cursor_update_timer.timeout.connect(self._fire_cursor_preview_update)
        self._pending_cursor_pos = None

        # Crosshair settings
        self.show_crosshair = True  # Flag to toggle crosshair visibility for this tool
        self.h_crosshair_line = None
        self.v_crosshair_line = None

    def activate(self):
        self.active = True
        self.annotation_window.setCursor(self.cursor)

    def deactivate(self):
        self._cursor_update_timer.stop()
        self._pending_cursor_pos = None
        self.active = False
        self.annotation_window.setCursor(self.default_cursor)
        self.clear_cursor_annotation()
        if self.cursor_clear_callback:
            self.cursor_clear_callback()

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
        
    def _schedule_cursor_preview_update(self, scene_pos: QPointF):
        """Queue a debounced MVAT context-canvas update (call from mouseMoveEvent when Multi-Annotate is on)."""
        self._pending_cursor_pos = scene_pos
        if not self._cursor_update_timer.isActive():
            self._cursor_update_timer.start(100)  # ~10 Hz throttle

    def _fire_cursor_preview_update(self):
        """Timer callback — fire cursor_move_callback if Multi-Annotate is still enabled."""
        mvat_manager = getattr(self.main_window, 'mvat_manager', None)
        if (self._pending_cursor_pos is not None
                and self.cursor_move_callback
                and mvat_manager
                and getattr(mvat_manager, 'multi_annotate_enabled', False)):
            self.cursor_move_callback(self._pending_cursor_pos, self.create_cursor_preview_item)

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

    def create_cursor_preview_item(self, u: float, v: float):
        """
        Create a QGraphicsItem representing this tool's cursor at image pixel (u, v).
        Used by the MVAT cursor preview propagation system to display the cursor
        on context canvases at the projected position.

        Subclasses that participate in cursor preview propagation should override
        this method and return a fully styled, positioned QGraphicsItem.
        The item must NOT be added to a scene — BaseCanvas does that.

        Returns:
            QGraphicsItem, or None if this tool does not support cursor preview.
        """
        return None
        
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
        self.h_crosshair_line = QGraphicsLineItem(image_rect.left(), scene_pos.y(), image_rect.right(), scene_pos.y())
        pen = QPen(QColor(255, 255, 255, 180), 1, Qt.DashLine)
        pen.setCosmetic(True)
        self.h_crosshair_line.setPen(pen)
        self.h_crosshair_line.setZValue(1000)
        self.annotation_window.scene.addItem(self.h_crosshair_line)
        
        # Create vertical line across the full height of the image
        self.v_crosshair_line = QGraphicsLineItem(scene_pos.x(), image_rect.top(), scene_pos.x(), image_rect.bottom())
        pen = QPen(QColor(255, 255, 255, 180), 1, Qt.DashLine)
        pen.setCosmetic(True)
        self.v_crosshair_line.setPen(pen)
        self.v_crosshair_line.setZValue(1000)
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