import warnings
import numpy as np

from PyQt5.QtGui import QColor, QPen, QBrush
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsRectItem, QApplication

from coralnet_toolbox.Tools.QtBrushTool import BrushTool

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class EraseTool(BrushTool):
    """A tool for erasing pixels on a MaskAnnotation layer. Inherits streaming/threading from BrushTool."""
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        # Note: Erase tool uses the exact same 'self.painting' flag from BrushTool to track the stroke.

    def keyPressEvent(self, event):
        """Handles key press events, toggle shape with Ctrl+Shift, clear with Ctrl+Delete/Backspace."""
        modifiers = event.modifiers()
        key = event.key()
        
        if ((modifiers & Qt.ControlModifier) and (key == Qt.Key_Delete or key == Qt.Key_Backspace)) and self.active:
            self._clear_non_locked_pixels()
        else:
            super().keyPressEvent(event)

    def _clear_non_locked_pixels(self):
        """Clears all non-locked pixels in the current mask annotation."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        mask_annotation = self.annotation_window.current_mask_annotation
        if mask_annotation:
            mask_annotation.mask_data[mask_annotation.mask_data < mask_annotation.LOCK_BIT] = 0
            mask_annotation._update_full_canvas()
            if mask_annotation.graphics_item:
                mask_annotation.graphics_item.update()
            mask_annotation.annotationUpdated.emit(mask_annotation)
        QApplication.restoreOverrideCursor()

    def create_cursor_preview_item(self, u: float, v: float):
        """Return a styled eraser shape QGraphicsItem centred at image pixel (u, v)."""
        radius = self.brush_size / 2.0
        
        if self.shape == 'circle':
            item = QGraphicsEllipseItem(u - radius, v - radius, self.brush_size, self.brush_size)
        else:
            item = QGraphicsRectItem(u - radius, v - radius, self.brush_size, self.brush_size)
        
        item.setBrush(QColor(0, 0, 0, 0))
        pen = QPen(QColor(255, 0, 0, 200), 2)
        pen.setStyle(Qt.DashLine)
        pen.setCosmetic(True)
        item.setPen(pen)
        return item

    def create_cursor_annotation(self, scene_pos: QPointF = None):
        """Create a cursor annotation showing the eraser shape."""
        if not scene_pos:
            self.clear_cursor_annotation()
            return
            
        self.clear_cursor_annotation()
        
        radius = self.brush_size / 2.0
        rect = QRectF(scene_pos.x() - radius, scene_pos.y() - radius, self.brush_size, self.brush_size)
        if self.shape == 'circle':
            self.cursor_annotation = QGraphicsEllipseItem(rect)
        else:
            self.cursor_annotation = QGraphicsRectItem(rect)
        
        self.cursor_annotation.setBrush(QColor(0, 0, 0, 0))
        pen = QPen(QColor(0, 0, 0), 4)
        self.cursor_annotation.setPen(pen)
        
        self.annotation_window.scene.addItem(self.cursor_annotation)
        self._last_brush_size = self.brush_size
        self._last_shape = self.shape

    def _apply_brush(self, event):
        """Draw a lightweight eraser scratchpad and accumulate points."""
        # Reuse the brush scratchpad path but style it for erasing so users get instant feedback
        scene_pos = self.annotation_window.mapToScene(event.pos())
        # Let the parent append the point and update the path (if available)
        try:
            super()._apply_brush(event)
        except Exception:
            # Fallback to manual accumulation if super fails
            self._accumulated_points.append(scene_pos)

        # If the parent created a scratchpad item, restyle it as an eraser (transparent fill + outline)
        try:
            if self.scratchpad_item:
                self.scratchpad_item.setBrush(QBrush(QColor(0, 0, 0, 0)))
                pen = QPen(QColor(0, 0, 0, 160), 2)
                pen.setCosmetic(True)
                pen.setStyle(Qt.SolidLine)
                self.scratchpad_item.setPen(pen)
        except Exception:
            pass

        # Stream a lightweight live stroke to MVAT (if hooked). Use a semi-transparent red to indicate erasing.
        if hasattr(self, 'live_stroke_callback') and callable(self.live_stroke_callback):
            try:
                eraser_color = QColor(255, 0, 0, 120)
                self.live_stroke_callback(scene_pos, self.brush_size, self.shape, eraser_color)
            except Exception:
                pass

    def _on_math_finished(self, flat_indices, center_pos, combined_mask, mask_annotation, selected_label_id):
        """Executes on the Main Thread: Writes 0 (background) and triggers 3D sync."""
        self._active_workers -= 1
        
        if len(flat_indices) > 0:
            # ERASER FIX: Hardcode class_id = 0
            mask_annotation.update_mask_at_indices(flat_indices, 0, silent=True)

        if self.post_stroke_callback:
            # We still pass selected_label_id for context routing, even though the value written is 0
            self.post_stroke_callback(center_pos, selected_label_id, combined_mask)
        
        if self._is_finishing_stroke and self._active_workers == 0 and not self._accumulated_points:
            self._cleanup_scratchpad()
            self.annotation_window.viewport().update()