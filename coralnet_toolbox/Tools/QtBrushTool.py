import warnings

import numpy as np

from PyQt5.QtGui import QColor, QPen, QBrush
from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer
from PyQt5.QtWidgets import QGraphicsEllipseItem, QMessageBox, QGraphicsRectItem

from coralnet_toolbox.Tools.QtTool import Tool

warnings.filterwarnings("ignore", category=DeprecationWarning)


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

        # Optional callback fired after each successful brush stroke:
        # callback(scene_pos: QPointF, label_id: str, brush_mask: np.ndarray)
        self.post_stroke_callback = None
        
        # Throttling setup for 3D propagation
        self._sync_timer = QTimer()
        self._sync_timer.setSingleShot(True)
        self._sync_timer.timeout.connect(self._flush_stroke)
        self._accumulated_points = []
        
        # Callbacks fired when painting starts/stops (used by MVATManager
        # to suppress expensive hover processing during active brush strokes)
        self.paint_start_callback = None
        self.paint_stop_callback = None

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

        # Check if the label is hidden when the user starts painting.
        # If it is, check the box in the LabelWindow UI automatically.
        if not self.annotation_window.selected_label.is_visible:
            self.annotation_window.selected_label.visibility_checkbox.setChecked(True)

        self.painting = not self.painting
        if self.painting:
            if self.paint_start_callback:
                self.paint_start_callback()
            self._apply_brush(event)
        else:
            if self.paint_stop_callback:
                self.paint_stop_callback()

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
            if self.cursor_move_callback:
                self.cursor_move_callback(scene_pos, self.create_cursor_preview_item)
        else:
            self.clear_cursor_annotation()
            if self.cursor_clear_callback:
                self.cursor_clear_callback()

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

    def create_cursor_preview_item(self, u: float, v: float):
        """Return a styled brush shape QGraphicsItem centred at image pixel (u, v)."""
        if not self.annotation_window.selected_label:
            return None
        
        label = self.annotation_window.selected_label
        transparency = self.annotation_window.main_window.get_transparency_value()
        radius = self.brush_size / 2.0
        c = QColor(label.color)
        fill = QColor(c)
        fill.setAlpha(transparency)
        pen = QPen(c.darker(150), 2)
        pen.setCosmetic(True)
        
        if self.shape == 'circle':
            item = QGraphicsEllipseItem(u - radius, v - radius, self.brush_size, self.brush_size)
        else:
            item = QGraphicsRectItem(u - radius, v - radius, self.brush_size, self.brush_size)
        
        item.setBrush(QBrush(fill))
        item.setPen(pen)
        return item

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
        transparency = self.annotation_window.main_window.get_transparency_value()
        
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
        was_painting = self.painting
        self.painting = False
        self._sync_timer.stop()
        if was_painting and self.paint_stop_callback:
            self.paint_stop_callback()
        if self._accumulated_points:
            self._flush_stroke()
        super().deactivate()
        
    def stop_current_drawing(self):
        """Force stop of current drawing by stopping painting mode."""
        was_painting = self.painting
        self.painting = False
        if was_painting and self.paint_stop_callback:
            self.paint_stop_callback()

    def _apply_brush(self, event):
        """Applies the brush locally and queues it for 3D sync."""
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
        
        # 1. Apply locally immediately for smooth 60Hz UX
        mask_annotation.update_mask(brush_location, self.brush_mask, class_id)

        # 2. Accumulate for 3D sync
        self._accumulated_points.append(scene_pos)
        
        # 3. Start throttle timer if not running (~15 FPS = 66ms)
        if not self._sync_timer.isActive():
            self._sync_timer.start(66)

    def _flush_stroke(self):
        """Builds a combined mask of recent strokes and sends to MVATManager."""
        if not self.post_stroke_callback or not self._accumulated_points:
            return

        selected_label_id = self.annotation_window.selected_label.id
        
        # Calculate bounding box of all accumulated points
        xs = [p.x() for p in self._accumulated_points]
        ys = [p.y() for p in self._accumulated_points]
        
        min_x = int(min(xs) - self.brush_size / 2.0)
        max_x = int(max(xs) + self.brush_size / 2.0)
        min_y = int(min(ys) - self.brush_size / 2.0)
        max_y = int(max(ys) + self.brush_size / 2.0)
        
        # Create a combined mask for the flushed batch
        w, h = max_x - min_x, max_y - min_y
        combined_mask = np.zeros((h, w), dtype=bool)
        
        for p in self._accumulated_points:
            px_local = int(p.x() - self.brush_size / 2.0) - min_x
            py_local = int(p.y() - self.brush_size / 2.0) - min_y
            
            # Stamp the brush mask into the combined mask
            bh, bw = self.brush_mask.shape
            if (0 <= px_local < w and 0 <= py_local < h and
                px_local + bw > 0 and py_local + bh > 0):
                ystart = max(0, py_local)
                yend = min(h, py_local + bh)
                xstart = max(0, px_local)
                xend = min(w, px_local + bw)
                
                brush_ystart = ystart - py_local
                brush_yend = brush_ystart + (yend - ystart)
                brush_xstart = xstart - px_local
                brush_xend = brush_xstart + (xend - xstart)
                
                combined_mask[ystart:yend, xstart:xend] |= self.brush_mask[brush_ystart:brush_yend, brush_xstart:brush_xend]
        
        # The new center of this combined mask
        center_pos = QPointF(min_x + w / 2.0, min_y + h / 2.0)
        
        # Send to MVATManager
        self.post_stroke_callback(center_pos, selected_label_id, combined_mask)
        
        # Clear accumulation buffer
        self._accumulated_points.clear()