from coralnet_toolbox.Tools.QtSubTool import SubTool


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MoveSubTool(SubTool):
    """SubTool for moving one or more selected annotations."""

    def __init__(self, parent_tool):
        super().__init__(parent_tool)
        self.move_start_pos = None

    def activate(self, event, **kwargs):
        """
        Activates the moving operation.
        Expects 'annotations_to_move' in kwargs, but works with parent's selected annotations.
        """
        super().activate(event)
        self.move_start_pos = self.annotation_window.mapToScene(event.pos())
        # The parent tool is responsible for ensuring annotations are selected.

    def deactivate(self):
        super().deactivate()
        self.move_start_pos = None

    def mouseMoveEvent(self, event):
        """Handle moving the selected annotation."""
        if not self.is_active or not self.parent_tool.selected_annotations:
            return

        current_pos = self.annotation_window.mapToScene(event.pos())
        delta = current_pos - self.move_start_pos

        # For this implementation, we only move the first selected annotation
        # that is moveable. A more complex version could move all of them.
        selected_annotation = self.parent_tool.selected_annotations[0]
        
        if not self.annotation_window.is_annotation_moveable(selected_annotation):
            # If it's not moveable, stop the operation.
            self.parent_tool.deactivate_subtool()
            return
            
        new_center = selected_annotation.center_xy + delta

        if self.annotation_window.cursorInWindow(event.pos()):
            self.annotation_window.set_annotation_location(selected_annotation.id, new_center)
            self.move_start_pos = current_pos

    def mouseReleaseEvent(self, event):
        """Finalize the move and deactivate this sub-tool."""
        self.parent_tool.deactivate_subtool()