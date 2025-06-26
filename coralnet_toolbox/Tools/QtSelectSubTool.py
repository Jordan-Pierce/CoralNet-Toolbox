from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPen, QColor
from PyQt5.QtWidgets import QGraphicsRectItem

from coralnet_toolbox.Tools.QtSubTool import SubTool


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SelectSubTool(SubTool):
    """SubTool for selecting multiple annotations with a rectangle."""
    
    def __init__(self, parent_tool):
        super().__init__(parent_tool)
        self.selection_rectangle = None
        self.selection_start_pos = None

    def activate(self, event, **kwargs):
        super().activate(event)
        self.selection_start_pos = self.annotation_window.mapToScene(event.pos())
        
        # Create and style the selection rectangle
        self.selection_rectangle = QGraphicsRectItem()
        width = self.parent_tool.graphics_utility.get_rectangle_graphic_thickness(self.annotation_window)
        pen = QPen(QColor(255, 255, 255), 2, Qt.DashLine)
        pen.setWidth(width)
        self.selection_rectangle.setPen(pen)
        self.selection_rectangle.setRect(QRectF(self.selection_start_pos, self.selection_start_pos))
        self.annotation_window.scene.addItem(self.selection_rectangle)

    def deactivate(self):
        super().deactivate()
        if self.selection_rectangle:
            self.annotation_window.scene.removeItem(self.selection_rectangle)
            self.selection_rectangle = None
        self.selection_start_pos = None

    def mouseMoveEvent(self, event):
        """Update the selection rectangle while dragging."""
        if not self.is_active or not self.selection_rectangle:
            return
            
        current_pos = self.annotation_window.mapToScene(event.pos())
        if self.annotation_window.cursorInWindow(event.pos()):
            rect = QRectF(self.selection_start_pos, current_pos).normalized()
            self.selection_rectangle.setRect(rect)

    def mouseReleaseEvent(self, event):
        """Finalize the selection and then deactivate."""
        self.finalize_selection()
        self.parent_tool.deactivate_subtool()

    def finalize_selection(self):
        """Select annotations contained within the drawn rectangle."""
        if not self.selection_rectangle:
            return

        rect = self.selection_rectangle.rect()
        locked_label = self.parent_tool.get_locked_label()

        # Iterate through all annotations to check for inclusion
        for annotation in self.annotation_window.get_image_annotations():
            if rect.contains(annotation.center_xy):
                if locked_label and annotation.label.id != locked_label.id:
                    continue  # Skip if label is locked and doesn't match
                if annotation not in self.parent_tool.selected_annotations:
                    # Append to selection (multi-select is the default for marquee)
                    self.annotation_window.select_annotation(annotation, multi_select=True)