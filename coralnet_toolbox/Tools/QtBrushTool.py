import numpy as np

from PyQt5.QtGui import QColor, QPen
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtWidgets import QGraphicsEllipseItem, QMessageBox, QGraphicsRectItem


from coralnet_toolbox.Tools.QtTool import Tool


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class BrushTool(Tool):
    """A tool for painting on a MaskAnnotation layer."""
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
        self.painting = False  # Flag to track if painting mode is active

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
        """Handles left-click to toggle painting mode and apply brush if starting."""
        if event.button() != Qt.LeftButton:
            return
            
        if not self.annotation_window.selected_label:
            QMessageBox.warning(self.annotation_window,
                                "No Label Selected",
                                "A label must be selected before using the brush tool.")
            return
        
        if not self.annotation_window.cursorInWindow(event.pos()):
            return
        
        self.painting = not self.painting
        if self.painting:
            self._apply_brush(event)

    def mouseMoveEvent(self, event):
        """Handles mouse dragging, shows the brush circle, and applies brush if painting is active."""
        # Call the parent method to handle any base logic (crosshair is disabled)
        super().mouseMoveEvent(event)
        
        # Handle brush circle display
        scene_pos = self.annotation_window.mapToScene(event.pos())
        cursor_in_window = self.annotation_window.cursorInWindow(event.pos())
        
        if (cursor_in_window and self.active and 
            self.annotation_window.selected_label):
            self.update_cursor_annotation(scene_pos)
        else:
            self.clear_cursor_annotation()
        
        # Apply brush if painting is active
        if self.painting:
            self._apply_brush(event)
    
    def keyPressEvent(self, event):
        """Handles key press events, toggle shape with Ctrl+Shift."""
        modifiers = event.modifiers()
        if ((modifiers & Qt.ControlModifier) and (modifiers & Qt.ShiftModifier)) and self.active:
            self._toggle_shape()
        super().keyPressEvent(event)

    def _toggle_shape(self):
        """Toggles between circle and square brush shapes."""
        self.shape = 'square' if self.shape == 'circle' else 'circle'
        self.brush_mask = self._create_brush_mask()
        # Update cursor if visible
        if self.cursor_annotation:
            cursor_pos = self.annotation_window.mapFromGlobal(self.annotation_window.cursor().pos())
            scene_pos = self.annotation_window.mapToScene(cursor_pos)
            self.update_cursor_annotation(scene_pos)

    def wheelEvent(self, event):
        """Handles mouse wheel events for adjusting brush size when Ctrl is held."""
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.brush_size = max(1, self.brush_size + 5)  # Increase size, min 1
            else:
                self.brush_size = max(1, self.brush_size - 5)  # Decrease size, min 1
            
            # Recreate the brush mask with the new size
            self.brush_mask = self._create_circular_brush()
            
            # Update the cursor annotation to reflect the new brush size
            scene_pos = self.annotation_window.mapToScene(event.pos())
            self.update_cursor_annotation(scene_pos)

    def create_cursor_annotation(self, scene_pos: QPointF = None):
        """Create a circular cursor annotation representing the brush."""
        if (not scene_pos or not self.annotation_window.selected_label or 
            not self.annotation_window.active_image or 
            not self.annotation_window.main_window.label_window.active_label):
            self.clear_cursor_annotation()
            return
            
        # First ensure any existing cursor annotation is removed
        self.clear_cursor_annotation()
        
        # Get the label's color and transparency
        label_color = self.annotation_window.selected_label.color
        transparency = self.annotation_window.main_window.label_window.active_label.transparency
        
        # Create the item for the brush shape
        radius = self.brush_size / 2.0
        rect = QRectF(scene_pos.x() - radius, scene_pos.y() - radius, self.brush_size, self.brush_size)
        if self.shape == 'circle':
            self.cursor_annotation = QGraphicsEllipseItem(rect)
        else:  # square
            self.cursor_annotation = QGraphicsRectItem(rect)
        
        # Set the color with transparency
        brush_color = QColor(label_color)
        brush_color.setAlpha(transparency)
        self.cursor_annotation.setBrush(brush_color)
        
        # Create a darker border color by darkening the label color
        border_color = QColor(label_color)
        border_color = border_color.darker(150)  # Make the border 50% darker than the fill        
        # Set the pen for the border
        pen = QPen(border_color)
        pen.setWidth(2)  # Set border width to 2 pixels
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

    def deactivate(self):
        """Deactivate the brush tool and stop any current operations."""
        self.painting = False
        super().deactivate()
        
    def stop_current_drawing(self):
        """Force stop of current drawing by stopping painting mode."""
        self.painting = False

    def _apply_brush(self, event):
        """Applies the brush mask to the main mask_annotation."""
        # Get the current mask annotation from the annotation window
        mask_annotation = self.annotation_window.current_mask_annotation
        
        # Ensure that a mask and a selected label are active
        if not mask_annotation or not self.annotation_window.selected_label:
            return
            
        # Get the mouse position in the scene's coordinate system
        scene_pos = self.annotation_window.mapToScene(event.pos())
        
        # The old loop is replaced with a fast, direct lookup using the new mapping system.
        selected_label_id = self.annotation_window.selected_label.id
        class_id = mask_annotation.label_id_to_class_id_map.get(selected_label_id)
        
        if class_id is None:
            return  # Label not found in map, cannot apply brush
        
        # Adjust brush_location to center the brush at the cursor position
        radius = self.brush_size / 2.0
        brush_location = QPointF(scene_pos.x() - radius, scene_pos.y() - radius)
        
        # Call the update_mask method, now passing the annotation_window for context
        mask_annotation.update_mask(brush_location, self.brush_mask, class_id, self.annotation_window)
