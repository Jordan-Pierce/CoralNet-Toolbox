from coralnet_toolbox.Tools.QtSubTool import SubTool
from coralnet_toolbox.QtActions import MoveAnnotationAction
from PyQt5.QtCore import QPointF


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MoveSubTool(SubTool):
    """SubTool for moving one or more selected annotations.

    During the drag only the existing QGraphicsItemGroup is translated (cheap).
    The authoritative move — geometry recompute, cropped image, confidence window,
    undo action — happens once on mouse release.
    """

    def __init__(self, parent_tool):
        super().__init__(parent_tool)
        self.move_start_pos = None      # scene pos at press (never reset during drag)
        self._last_pos = None           # scene pos at previous move event
        self._pending_center = None     # final center to apply on release

    def activate(self, event, **kwargs):
        super().activate(event)
        self.move_start_pos = self.annotation_window.mapToScene(event.pos())
        self._last_pos = self.move_start_pos
        self._pending_center = None
        # Capture original center for undo
        try:
            selected_annotation = self.parent_tool.selected_annotations[0]
            # copy QPointF
            self.orig_center = QPointF(selected_annotation.center_xy.x(), selected_annotation.center_xy.y())
        except Exception:
            self.orig_center = None

    def deactivate(self):
        super().deactivate()
        self.move_start_pos = None
        self._last_pos = None
        self._pending_center = None

    def mouseMoveEvent(self, event):
        """Preview the move by translating the existing graphics group."""
        if not self.is_active or not self.parent_tool.selected_annotations:
            return

        current_pos = self.annotation_window.mapToScene(event.pos())
        selected_annotation = self.parent_tool.selected_annotations[0]

        if not self.annotation_window.is_annotation_moveable(selected_annotation, use_status_bar=True):
            self.parent_tool.deactivate_subtool()
            return

        if not self.annotation_window.cursorInWindow(event.pos()):
            return

        group = selected_annotation.graphics_item_group
        if group is not None and self.orig_center is not None:
            step = current_pos - self._last_pos
            group.moveBy(step.x(), step.y())
            self._last_pos = current_pos
            self._pending_center = self.orig_center + (current_pos - self.move_start_pos)
        else:
            # Fallback: no live group (shouldn't happen for a selected annotation) —
            # use the legacy heavy per-move path so the move still works.
            delta = current_pos - self._last_pos
            self._last_pos = current_pos
            new_center = selected_annotation.center_xy + delta
            self.annotation_window.set_annotation_location(selected_annotation.id, new_center)

    def mouseReleaseEvent(self, event):
        """Apply the deferred move, record the undo action, and deactivate."""
        try:
            selected_annotation = self.parent_tool.selected_annotations[0]
            if self._pending_center is not None:
                # One authoritative move: recomputes geometry, rebuilds the group
                # (clearing the temporary translation), crops, updates confidence.
                self.annotation_window.set_annotation_location(
                    selected_annotation.id, self._pending_center
                )
            new_center = selected_annotation.center_xy
            if self.orig_center is not None and new_center != self.orig_center:
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