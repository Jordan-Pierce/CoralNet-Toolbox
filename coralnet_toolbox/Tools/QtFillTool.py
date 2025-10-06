from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QMessageBox

from coralnet_toolbox.Tools.QtTool import Tool


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

    def mousePressEvent(self, event):
        """Handles left-click to fill the region under the cursor."""
        if event.button() != Qt.LeftButton:
            return
            
        if not self.annotation_window.selected_label:
            QMessageBox.warning(self.annotation_window,
                                "No Label Selected",
                                "A label must be selected before using the fill tool.")
            return
        
        if not self.annotation_window.cursorInWindow(event.pos()):
            return
        
        self._apply_fill(event)

    def mouseMoveEvent(self, event):
        """Handles mouse movement, shows crosshair if enabled."""
        # Call the parent method to handle crosshair
        super().mouseMoveEvent(event)
    
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
        
        # Call the fill_region method on the MaskAnnotation object
        mask_annotation.fill_region(scene_pos, new_class_id, self.annotation_window)
        
        # Update the display to reflect changes
        self.annotation_window.update_scene()
