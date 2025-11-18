import warnings

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from coralnet_toolbox.Tools.QtTool import Tool

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DropperTool(Tool):
    """A tool for selecting a label by picking a pixel from the mask annotation."""

    def __init__(self, annotation_window):
        # Call the parent constructor to set up annotation_window, etc.
        super().__init__(annotation_window)
        
        # Disable crosshair for this tool
        self.show_crosshair = True
        
        # Set a specific cursor for this tool
        self.cursor = Qt.CrossCursor

    def mousePressEvent(self, event):
        """Handles left-click to pick the label at the cursor position."""
        if event.button() != Qt.LeftButton:
            return
            
        if not self.annotation_window.cursorInWindow(event.pos()):
            return
        
        # Get the current mask annotation
        mask_annotation = self.annotation_window.current_mask_annotation
        if not mask_annotation:
            QMessageBox.warning(self.annotation_window,
                                "No Mask Available",
                                "A mask annotation must be loaded to use the dropper tool.")
            return
        
        # Get the mouse position in the scene's coordinate system
        scene_pos = self.annotation_window.mapToScene(event.pos())
        
        # Get the class ID at the cursor position
        class_id = mask_annotation.get_class_at_point(scene_pos)
        
        if class_id == 0:
            QMessageBox.information(self.annotation_window,
                                    "Background Selected",
                                    "The selected pixel is background (no label).")
            return
        
        # Get the label associated with this class ID
        label = mask_annotation.class_id_to_label_map.get(class_id)
        if not label:
            QMessageBox.warning(self.annotation_window,
                                "Label Not Found",
                                "No label found for the selected pixel. Is it a locked pixel?")
            return
        
        # Emit the signal to select the label in the label window
        self.annotation_window.labelSelected.emit(label.id)
