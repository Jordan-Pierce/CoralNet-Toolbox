import warnings

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QApplication

from coralnet_toolbox.QtActions import MaskEditAction
from coralnet_toolbox.Tools.QtTool import Tool

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class FillTool(Tool):
    """A tool for filling contiguous regions in a MaskAnnotation layer with the selected label."""
    def __init__(self, annotation_window):
        # Call the parent constructor to set up annotation_window, etc.
        super().__init__(annotation_window)
        
        # Disable crosshair for this tool
        self.show_crosshair = True
        
        # You can set a specific cursor for this tool
        self.cursor = Qt.CrossCursor

        # Optional callbacks for cursor preview propagation
        self.cursor_move_callback = None
        self.cursor_clear_callback = None

    def mousePressEvent(self, event):
        """Handles left-click to fill the region under the cursor."""
        if event.button() != Qt.LeftButton:
            return
            
        if not self.annotation_window.selected_label:
            self.annotation_window.main_window.status_bar.showMessage(
                "A label must be selected before using the fill tool.", 4000)
            return
        
        if not self.annotation_window.cursorInWindow(event.pos()):
            return

        # Check if the label is hidden when the user clicks to fill.
        # If it is, check the box in the LabelWindow UI automatically.
        if not self.annotation_window.selected_label.is_visible:
            self.annotation_window.selected_label.visibility_checkbox.setChecked(True)

        self._apply_fill(event)

    def mouseMoveEvent(self, event):
        """Handles mouse movement, shows crosshair and propagates cursor preview for Multi-Annotate."""
        # Call the parent method to handle crosshair
        super().mouseMoveEvent(event)
        
        # Handle cursor preview propagation for Multi-Annotate
        scene_pos = self.annotation_window.mapToScene(event.pos())
        cursor_in_window = self.annotation_window.cursorInWindow(event.pos())
        
        if cursor_in_window and self.active and self.annotation_window.selected_label:
            if self.cursor_move_callback:
                self.cursor_move_callback(scene_pos, self.create_cursor_preview_item)
        else:
            if self.cursor_clear_callback:
                self.cursor_clear_callback()
    
    def mouseReleaseEvent(self, event):
        """Called when the mouse is released."""
        pass  # No action needed on release

    def wheelEvent(self, event):
        """No wheel event handling for fill tool."""
        pass

    def create_cursor_annotation(self, scene_pos: QPointF = None):
        """No special cursor annotation for fill tool."""
        pass

    def update_cursor_annotation(self, scene_pos: QPointF = None):
        """No update needed."""
        pass

    def clear_cursor_annotation(self):
        """No cursor annotation to clear."""
        pass
    
    def create_cursor_preview_item(self, u: float, v: float):
        """Return a styled cursor indicator for the fill tool (simple crosshair preview)."""
        if not self.annotation_window.selected_label:
            return None
        
        from PyQt5.QtGui import QColor, QPen
        from PyQt5.QtWidgets import QGraphicsEllipseItem
        
        label = self.annotation_window.selected_label
        
        # Create a small circle indicator to show the fill point
        c = QColor(label.color)
        pen = QPen(c.darker(150), 2)
        pen.setCosmetic(True)
        
        size = 8  # Small circle for fill tool cursor
        item = QGraphicsEllipseItem(u - size/2, v - size/2, size, size)
        item.setPen(pen)
        item.setBrush(QColor(c))
        return item

    def stop_current_drawing(self):
        """No drawing to stop."""
        pass

    def _apply_fill(self, event):
        """Fills the contiguous region under the cursor with the selected label."""
        # Get the current mask annotation from the annotation window
        mask_annotation = self.annotation_window.current_mask_annotation
        
        # Ensure that a mask and a selected label are active
        if not mask_annotation or not self.annotation_window.selected_label:
            return
            
        # Get the mouse position in the scene's coordinate system
        scene_pos = self.annotation_window.mapToScene(event.pos())
        
        # Get the selected label's class ID
        selected_label_id = self.annotation_window.selected_label.id
        new_class_id = mask_annotation.label_id_to_class_id_map.get(selected_label_id)
        
        if new_class_id is None:
            return  # Label not found in map
        
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        history_action = MaskEditAction(mask_annotation, description="Fill region")
        
        # Call the fill_region method on the MaskAnnotation object
        mask_annotation.fill_region(scene_pos, new_class_id, history_action=history_action)

        # Ensure the label is visible in the mask (even if checkbox is unchecked)
        if selected_label_id not in mask_annotation.visible_label_ids:
            mask_annotation.visible_label_ids.add(selected_label_id)
            mask_annotation.update_graphics_item()

        if not history_action.is_empty():
            self.annotation_window.action_stack.push(history_action)
        
        # Restore the cursor
        QApplication.restoreOverrideCursor()
