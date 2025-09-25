import numpy as np

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtWidgets import QGraphicsEllipseItem

from coralnet_toolbox.Tools.QtTool import Tool


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class EraseTool(Tool):
    """A tool for erasing pixels on a MaskAnnotation layer."""
    def __init__(self, annotation_window):
        # Call the parent constructor to set up annotation_window, etc.
        super().__init__(annotation_window)
        
        # Disable crosshair for this tool
        self.show_crosshair = False
        
        # You can set a specific cursor for this tool
        self.cursor = Qt.CrossCursor 
        
        self.brush_size = 90
        self.shape = 'circle'  # 'circle' or 'square'
        self.brush_mask = self._create_brush_mask()
        self.erasing = False  # Flag to track if erasing mode is active

    def _create_brush_mask(self):
        """Creates a boolean numpy array for the brush shape."""
        if self.shape == 'circle':
            return self._create_circular_brush()
        elif self.shape == 'square':
            return self._create_square_brush()

    def _create_circular_brush(self):
        """Creates a circular boolean numpy array to use as the brush shape."""
        radius = self.brush_size // 2
        y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
        return x**2 + y**2 <= radius**2

    def _create_square_brush(self):
        """Creates a square boolean numpy array to use as the brush shape."""
        size = self.brush_size
        return np.ones((size, size), dtype=bool)

    def mousePressEvent(self, event):
        """Handles left-click to toggle erasing mode and apply eraser if starting."""
        if event.button() == Qt.LeftButton:
            self.erasing = not self.erasing
            if self.erasing:
                self._apply_eraser(event)

    def mouseMoveEvent(self, event):
        """Handles mouse dragging, shows the eraser circle, and applies eraser if erasing is active."""
        # Call the parent method to handle any base logic (crosshair is disabled)
        super().mouseMoveEvent(event)
        
        # Handle eraser circle display
        scene_pos = self.annotation_window.mapToScene(event.pos())
        cursor_in_window = self.annotation_window.cursorInWindow(event.pos())
        
        if cursor_in_window and self.active and self.annotation_window.selected_label:
            self.update_cursor_annotation(scene_pos)
        else:
            self.clear_cursor_annotation()
        
        # Apply eraser if erasing is active
        if self.erasing:
            self._apply_eraser(event)
    
    def keyPressEvent(self, event):
        """Handles key press events, toggle shape with Ctrl."""
        if event.key() == Qt.Key_Control and self.active:
            self._toggle_shape()
        super().keyPressEvent(event)

    def _toggle_shape(self):
        """Toggles between circle and square eraser shapes."""
        self.shape = 'square' if self.shape == 'circle' else 'circle'
        self.brush_mask = self._create_brush_mask()
        # Update cursor if visible
        if self.cursor_annotation:
            cursor_pos = self.annotation_window.mapFromGlobal(self.annotation_window.cursor().pos())
            scene_pos = self.annotation_window.mapToScene(cursor_pos)
            self.update_cursor_annotation(scene_pos)

    def wheelEvent(self, event):
        """Handles mouse wheel events for adjusting eraser size when Ctrl is held."""
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.brush_size = max(1, self.brush_size + 5)  # Increase size, min 1
            else:
                self.brush_size = max(1, self.brush_size - 5)  # Decrease size, min 1
            
            # Recreate the brush mask with the new size
            self.brush_mask = self._create_circular_brush()
            
            # Update the cursor annotation to reflect the new eraser size
            scene_pos = self.annotation_window.mapToScene(event.pos())
            self.update_cursor_annotation(scene_pos)

    def create_cursor_annotation(self, scene_pos: QPointF = None):
        """Create a circular cursor annotation representing the eraser."""
        if not (scene_pos and self.annotation_window.selected_label and
                self.annotation_window.active_image and
                self.annotation_window.main_window.label_window.active_label):
            self.clear_cursor_annotation()
            return
            
        # First ensure any existing cursor annotation is removed
        self.clear_cursor_annotation()
        
        # Create the item for the eraser shape
        radius = self.brush_size / 2.0
        rect = QRectF(scene_pos.x() - radius, scene_pos.y() - radius, self.brush_size, self.brush_size)
        if self.shape == 'circle':
            self.cursor_annotation = QGraphicsEllipseItem(rect)
        else:  # square
            from PyQt5.QtWidgets import QGraphicsRectItem
            self.cursor_annotation = QGraphicsRectItem(rect)
        
        # Set transparent fill
        brush_color = QColor(0, 0, 0, 0)  # Fully transparent
        self.cursor_annotation.setBrush(brush_color)
        
        # Set thick black border
        border_color = QColor(0, 0, 0)  # Black
        pen = QPen(border_color)
        pen.setWidth(4)  # Thicker border
        self.cursor_annotation.setPen(pen)
        
        # Add to the scene
        self.annotation_window.scene.addItem(self.cursor_annotation)

    def update_cursor_annotation(self, scene_pos: QPointF = None):
        """Update the cursor annotation position."""
        self.clear_cursor_annotation()
        self.create_cursor_annotation(scene_pos)

    def clear_cursor_annotation(self):
        """Clear the current cursor annotation if it exists."""
        if self.cursor_annotation and self.cursor_annotation.scene():
            self.annotation_window.scene.removeItem(self.cursor_annotation)
            self.cursor_annotation = None

    def stop_current_drawing(self):
        """Force stop of current drawing by stopping erasing mode."""
        self.erasing = False

    def _apply_eraser(self, event):
        """Applies the eraser mask to the main mask_annotation, setting pixels to background (0)."""
        # Get the current mask annotation from the annotation window
        mask_annotation = self.annotation_window.current_mask_annotation
        
        # Ensure that a mask is active
        if not mask_annotation:
            return
            
        # Get the mouse position in the scene's coordinate system
        scene_pos = self.annotation_window.mapToScene(event.pos())
        
        # Adjust brush_location to center the eraser at the cursor position
        radius = self.brush_size / 2.0
        brush_location = QPointF(scene_pos.x() - radius, scene_pos.y() - radius)
        
        # Call the update_mask method on the MaskAnnotation object with class_id 0 (background)
        mask_annotation.update_mask(
            brush_location=brush_location,
            brush_mask=self.brush_mask,
            new_class_id=0
        )
