import warnings
import numpy as np

from PyQt5.QtGui import QColor, QPen, QBrush, QPainterPath
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtWidgets import QGraphicsEllipseItem, QMessageBox, QGraphicsRectItem, QGraphicsPathItem

from coralnet_toolbox.Tools.QtTool import Tool

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class BrushTool(Tool):
    """A tool for painting on a MaskAnnotation layer."""
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        
        self.show_crosshair = False
        self.cursor = Qt.CrossCursor 
        
        self.brush_size = 90
        self.shape = 'circle'  # 'circle' or 'square'
        self.brush_mask = self._create_brush_mask()
        self.painting = False

        self.post_stroke_callback = None
        self._accumulated_points = []
        
        # NEW: The Qt Scratchpad elements for smooth UI rendering
        self.scratchpad_item = None
        self.scratchpad_path = QPainterPath()

    def _create_brush_mask(self):
        """Creates a boolean numpy array for the brush shape."""
        if self.shape == 'circle':
            radius = self.brush_size // 2
            y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
            return x**2 + y**2 <= radius**2
        elif self.shape == 'square':
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

        if not self.annotation_window.selected_label.is_visible:
            self.annotation_window.selected_label.visibility_checkbox.setChecked(True)

        self.painting = not self.painting
        
        if self.painting:
            # --- START OF STROKE: Initialize the Scratchpad ---
            self.scratchpad_path = QPainterPath()
            self.scratchpad_path.setFillRule(Qt.WindingFill)  # Ensure proper filling of complex shapes
            self.scratchpad_item = QGraphicsPathItem()
            
            label = self.annotation_window.selected_label
            transparency = self.annotation_window.main_window.get_transparency_value()
            
            c = QColor(label.color)
            fill = QColor(c)
            fill.setAlpha(transparency)
            
            self.scratchpad_item.setBrush(QBrush(fill))
            self.scratchpad_item.setPen(QPen(Qt.NoPen))
            self.scratchpad_item.setZValue(3)  # Hover above the existing mask layer
            
            self.annotation_window.scene.addItem(self.scratchpad_item)
            
            self._apply_brush(event)
        else:
            # --- END OF STROKE: Bake to NumPy ---
            self._flush_stroke()

    def mouseMoveEvent(self, event):
        """Handles mouse dragging, shows the brush circle, and applies brush if painting is active."""
        super().mouseMoveEvent(event)
        
        scene_pos = self.annotation_window.mapToScene(event.pos())
        cursor_in_window = self.annotation_window.cursorInWindow(event.pos())
        
        if (cursor_in_window and self.active and self.annotation_window.selected_label):
            self.update_cursor_annotation(scene_pos)
            if self.cursor_move_callback:
                self.cursor_move_callback(scene_pos, self.create_cursor_preview_item)
        else:
            self.clear_cursor_annotation()
            if self.cursor_clear_callback:
                self.cursor_clear_callback()

        if self.painting:
            self._apply_brush(event)
    
    def keyPressEvent(self, event):
        modifiers = event.modifiers()
        if ((modifiers & Qt.ControlModifier) and (modifiers & Qt.ShiftModifier)) and self.active:
            self._toggle_shape()
        super().keyPressEvent(event)

    def _toggle_shape(self):
        self.shape = 'square' if self.shape == 'circle' else 'circle'
        self.brush_mask = self._create_brush_mask()
        if self.cursor_annotation:
            cursor_pos = self.annotation_window.mapFromGlobal(self.annotation_window.cursor().pos())
            scene_pos = self.annotation_window.mapToScene(cursor_pos)
            self.update_cursor_annotation(scene_pos)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            self.brush_size = max(1, self.brush_size + (5 if delta > 0 else -5))
            self.brush_mask = self._create_brush_mask()
            
            scene_pos = self.annotation_window.mapToScene(event.pos())
            self.update_cursor_annotation(scene_pos)

    def create_cursor_preview_item(self, u: float, v: float):
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
        if (not scene_pos or not self.annotation_window.selected_label or 
            not self.annotation_window.active_image or 
            not self.annotation_window.main_window.label_window.active_label):
            self.clear_cursor_annotation()
            return
            
        self.clear_cursor_annotation()
        
        label_color = self.annotation_window.selected_label.color
        transparency = self.annotation_window.main_window.get_transparency_value()
        
        radius = self.brush_size / 2.0
        rect = QRectF(scene_pos.x() - radius, scene_pos.y() - radius, self.brush_size, self.brush_size)
        
        if self.shape == 'circle':
            self.cursor_annotation = QGraphicsEllipseItem(rect)
        else:
            self.cursor_annotation = QGraphicsRectItem(rect)
        
        brush_color = QColor(label_color)
        brush_color.setAlpha(transparency)
        self.cursor_annotation.setBrush(brush_color)
        
        border_color = QColor(label_color).darker(150)
        pen = QPen(border_color)
        pen.setWidth(2)
        self.cursor_annotation.setPen(pen)
        
        self.annotation_window.scene.addItem(self.cursor_annotation)
        
        self._last_brush_size = self.brush_size
        self._last_shape = self.shape

    def update_cursor_annotation(self, scene_pos: QPointF = None):
        if scene_pos is None:
            self.clear_cursor_annotation()
            return
        
        if (self.cursor_annotation and 
            hasattr(self, '_last_brush_size') and self._last_brush_size == self.brush_size and
            hasattr(self, '_last_shape') and self._last_shape == self.shape):
            radius = self.brush_size / 2.0
            rect = QRectF(scene_pos.x() - radius, scene_pos.y() - radius, self.brush_size, self.brush_size)
            self.cursor_annotation.setRect(rect)
        else:
            self.clear_cursor_annotation()
            self.create_cursor_annotation(scene_pos)

    def clear_cursor_annotation(self):
        if self.cursor_annotation and self.cursor_annotation.scene():
            self.annotation_window.scene.removeItem(self.cursor_annotation)
            self.cursor_annotation = None

    def deactivate(self):
        self.painting = False
        if self._accumulated_points:
            self._flush_stroke()
        super().deactivate()
        
    def stop_current_drawing(self):
        self.painting = False
        if self._accumulated_points:
            self._flush_stroke()

    def _apply_brush(self, event):
        """Draws visually on the Qt Scratchpad and accumulates points (Zero NumPy)."""
        scene_pos = self.annotation_window.mapToScene(event.pos())
        radius = self.brush_size / 2.0
        
        # 1. Update the C++ Vector visual layer instantly
        if self.scratchpad_item:
            if self.shape == 'circle':
                self.scratchpad_path.addEllipse(scene_pos.x() - radius, scene_pos.y() - radius, self.brush_size, self.brush_size)
            else:
                self.scratchpad_path.addRect(scene_pos.x() - radius, scene_pos.y() - radius, self.brush_size, self.brush_size)
                
            self.scratchpad_item.setPath(self.scratchpad_path)

        # 2. Accumulate the point for the eventual NumPy flush
        self._accumulated_points.append(scene_pos)

    def _flush_stroke(self):
        """Bakes the accumulated stroke into the NumPy array and triggers 3D sync."""
        if not self._accumulated_points:
            self._cleanup_scratchpad()
            return

        mask_annotation = self.annotation_window.current_mask_annotation
        if not mask_annotation or not self.annotation_window.selected_label:
            self._cleanup_scratchpad()
            return

        selected_label_id = self.annotation_window.selected_label.id
        class_id = mask_annotation.label_id_to_class_id_map.get(selected_label_id)
        
        if class_id is None:
            self._cleanup_scratchpad()
            return

        # 1. Calculate the boolean footprint of the entire stroke
        xs = [p.x() for p in self._accumulated_points]
        ys = [p.y() for p in self._accumulated_points]
        
        min_x = int(min(xs) - self.brush_size / 2.0)
        max_x = int(max(xs) + self.brush_size / 2.0)
        min_y = int(min(ys) - self.brush_size / 2.0)
        max_y = int(max(ys) + self.brush_size / 2.0)
        
        w, h = max_x - min_x, max_y - min_y
        combined_mask = np.zeros((h, w), dtype=bool)
        
        for p in self._accumulated_points:
            px_local = int(p.x() - self.brush_size / 2.0) - min_x
            py_local = int(p.y() - self.brush_size / 2.0) - min_y
            
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

        # 2. Convert the boolean footprint to flat global indices
        local_ys, local_xs = np.where(combined_mask)
        global_xs = local_xs + min_x
        global_ys = local_ys + min_y
        
        img_h, img_w = mask_annotation.mask_data.shape
        valid = (global_xs >= 0) & (global_xs < img_w) & (global_ys >= 0) & (global_ys < img_h)
        
        flat_indices = (global_ys[valid] * img_w + global_xs[valid]).astype(np.int64)

        # 3. Write to the NumPy Array exactly ONCE using the highly optimized index injection
        if len(flat_indices) > 0:
            mask_annotation.update_mask_at_indices(flat_indices, class_id, silent=False)

        # 4. Trigger the Multi-Camera 3D Sync
        if self.post_stroke_callback:
            center_pos = QPointF(min_x + w / 2.0, min_y + h / 2.0)
            self.post_stroke_callback(center_pos, selected_label_id, combined_mask)
        
        # 5. Destroy the visual Scratchpad (the pixels are now safely in the Mask)
        self._cleanup_scratchpad()

    def _cleanup_scratchpad(self):
        """Safely removes the temporary vector stroke from the UI."""
        if self.scratchpad_item and self.scratchpad_item.scene():
            self.annotation_window.scene.removeItem(self.scratchpad_item)
        self.scratchpad_item = None
        self.scratchpad_path = QPainterPath()
        self._accumulated_points.clear()