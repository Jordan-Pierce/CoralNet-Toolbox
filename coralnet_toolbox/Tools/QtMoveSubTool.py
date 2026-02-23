from coralnet_toolbox.Tools.QtSubTool import SubTool
from coralnet_toolbox.QtActions import MoveAnnotationAction
from PyQt5.QtCore import QPointF


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
        # Capture original center for undo
        try:
            selected_annotation = self.parent_tool.selected_annotations[0]
            # copy QPointF
            self.orig_center = QPointF(selected_annotation.center_xy.x(), selected_annotation.center_xy.y())
        except Exception:
            self.orig_center = None
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
        # Record action (single-selection expected)
        try:
            selected_annotation = self.parent_tool.selected_annotations[0]
            new_center = selected_annotation.center_xy
            if self.orig_center is not None:
                action = MoveAnnotationAction(self.annotation_window, selected_annotation.id, self.orig_center, new_center)
                try:
                    self.annotation_window.action_stack.push(action)
                except Exception:
                    pass
                try:
                    self.annotation_window.annotationMoved.emit(selected_annotation.id, {'old_center': self.orig_center, 'new_center': new_center})
                except Exception:
                    pass
        except Exception:
            pass

        self.parent_tool.deactivate_subtool()