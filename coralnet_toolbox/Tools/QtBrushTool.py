import numpy as np

from PyQt5.QtCore import Qt

from coralnet_toolbox.Tools.QtTool import Tool


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class BrushTool(Tool):
    """A tool for painting on a MaskAnnotation layer."""
    def __init__(self, annotation_window):
        # Call the parent constructor to set up annotation_window, etc.
        super().__init__(annotation_window)
        
        # You can set a specific cursor for this tool
        self.cursor = Qt.CrossCursor 
        
        self.brush_size = 30
        self.brush_mask = self._create_circular_brush()

    def _create_circular_brush(self):
        """Creates a circular boolean numpy array to use as the brush shape."""
        radius = self.brush_size // 2
        y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
        return x**2 + y**2 <= radius**2

    def mousePressEvent(self, event):
        """Handles the initial click to apply the brush."""
        if event.buttons() & Qt.LeftButton:
            self._apply_brush(event)

    def mouseMoveEvent(self, event):
        """Handles mouse dragging and shows the crosshair."""
        # Call the parent method to handle drawing the crosshair
        super().mouseMoveEvent(event)
        
        if event.buttons() & Qt.LeftButton:
            self._apply_brush(event)
    
    def mouseReleaseEvent(self, event):
        """Called when the mouse is released."""
        pass # No action needed on release

    def _apply_brush(self, event):
        """Applies the brush mask to the main mask_annotation."""
        # Ensure that a mask and a label are active
        if not self.annotation_window.mask_annotation or not self.annotation_window.selected_label:
            return
            
        # Get the mouse position in the scene's coordinate system
        scene_pos = self.annotation_window.mapToScene(event.pos())
        # Get the class ID from the currently selected label
        class_id = int(self.annotation_window.selected_label.id)

        # Call the update_mask method on the MaskAnnotation object
        self.annotation_window.mask_annotation.update_mask(
            brush_location=scene_pos,
            brush_mask=self.brush_mask,
            new_class_id=class_id
        )