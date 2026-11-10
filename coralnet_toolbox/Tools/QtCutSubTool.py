from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainterPath, QColor
from PyQt5.QtWidgets import QGraphicsPathItem

from coralnet_toolbox.Tools.QtSubTool import SubTool
from coralnet_toolbox.Tools.QtTool import TOOL_OVERLAY_Z, fine_dash_pen

from coralnet_toolbox.Annotations import MultiPolygonAnnotation
from coralnet_toolbox.QtActions import SplitAnnotationAction
from coralnet_toolbox.utilities import get_view_scale


# Minimum on-screen spacing between free-hand samples, in screen pixels.
CUT_SAMPLE_SCREEN_PX = 3.0
CUT_LINE_COLOR = QColor(230, 62, 0)
CUT_FAILED_MESSAGE = ("Nothing was cut: the line has to cross the annotation from one edge to the other"
                      "  |  Draw it again  |  Ctrl+X: leave cut mode")


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class CutSubTool(SubTool):
    """
    SubTool for managing the UI of cutting an annotation.
    """

    def __init__(self, parent_tool):
        """Initialize CutSubTool with parent tool."""
        super().__init__(parent_tool)
        self.target_annotation = None  # The annotation to be cut
        self.drawing_in_progress = False  # Whether the user is currently drawing a cut line
        self.cutting_path_item = None  # QGraphicsPathItem for the cut line
        self.cutting_points = []  # List of QPointF for the cut path
        self.ctrl_pressed = False  # Track Ctrl key state for straight-line mode
        self.last_click_point = None  # Store last clicked point for straight-line mode

    def activate(self, event, **kwargs):
        """
        Activate cutting mode. Expects 'annotation' in kwargs.
        """
        super().activate(event)
        self.target_annotation = kwargs.get('annotation')

        # --- Pre-activation Checks ---
        if not self.target_annotation:
            # No annotation selected, deactivate subtool
            self.parent_tool.deactivate_subtool()
            return

        if not self.target_annotation.verified:
            # Only verified annotations can be cut
            self.annotation_window.main_window.status_bar.showMessage(
                "Cannot cut: verify by selecting and pressing Ctrl+Space, "
                "clicking a label in the ConfidenceWindow, or updating the label manually.",
                5000,
            )
            self.parent_tool.deactivate_subtool()
            return

        # Special case: MultiPolygonAnnotations are immediately broken apart.
        if isinstance(self.target_annotation, MultiPolygonAnnotation):
            self._break_apart_multipolygon()
            return

        # --- Enter Line-Drawing Mode ---
        self.drawing_in_progress = False
        self.cutting_points = []
        self.ctrl_pressed = False
        self.last_click_point = None
        self.annotation_window.viewport().setCursor(Qt.CrossCursor)

    def deactivate(self):
        """Clean up all state and UI elements for the cutting tool."""
        super().deactivate()
        self.target_annotation = None
        self.drawing_in_progress = False
        self.cutting_points = []
        if self.cutting_path_item:
            self.annotation_window.scene.removeItem(self.cutting_path_item)
            self.cutting_path_item = None
        self.annotation_window.viewport().setCursor(self.parent_tool.cursor)
        self.annotation_window.scene.update()

    def mousePressEvent(self, event):
        """Handle mouse press events for cutting."""
        if not self.is_active or event.button() != Qt.LeftButton:
            return
        position = self.annotation_window.mapToScene(event.pos())
        if not self.drawing_in_progress:
            self._start_drawing_cut_line(position)
            self.last_click_point = position
            self.parent_tool.report_state()
        else:
            if self.ctrl_pressed and self.last_click_point:
                # Add a straight segment from last_click_point to position
                self.cutting_points.append(position)
                self.last_click_point = position
                self._redraw_cut_line(position)
                # Continue drawing (don't finish yet)
            else:
                # The finishing click is part of the line: without it the cut
                # stopped at the last sample, up to a sample spacing short.
                if self.cutting_points and position != self.cutting_points[-1]:
                    self.cutting_points.append(position)
                # On failure the outcome message stays up; report_state would
                # replace it with the hint straight away.
                if self._finish_and_perform_cut():
                    self.parent_tool.report_state()

    def mouseMoveEvent(self, event):
        """Handle mouse move events to update the cut line."""
        if not self.drawing_in_progress:
            return
        position = self.annotation_window.mapToScene(event.pos())
        if self.ctrl_pressed and self.last_click_point:
            # Preview a straight line from last_click_point to cursor
            self._redraw_cut_line(position)
        else:
            self._update_cut_line_path(position)

    def keyPressEvent(self, event):
        """Handle key press events for cutting operations."""
        # Check for Ctrl+X to toggle cutting mode off
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_X:
            self.parent_tool.deactivate_subtool()
            self.parent_tool.report_state()
            return

        # Handle Backspace or Esc to clear the current cutting line but stay in cutting mode
        if event.key() in (Qt.Key_Backspace, Qt.Key_Escape):
            self._clear_cutting_line()
            self.parent_tool.report_state()
            return

        if event.key() == Qt.Key_Control:
            if self.drawing_in_progress and not self.ctrl_pressed:
                self.ctrl_pressed = True
                cursor_pos = self.annotation_window.mapFromGlobal(self.annotation_window.cursor().pos())
                scene_pos = self.annotation_window.mapToScene(cursor_pos)
                if not self.cutting_points or scene_pos != self.cutting_points[-1]:
                    self.cutting_points.append(scene_pos)
                self.last_click_point = scene_pos

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            if self.drawing_in_progress and self.ctrl_pressed:
                self.ctrl_pressed = False
                cursor_pos = self.annotation_window.mapFromGlobal(self.annotation_window.cursor().pos())
                scene_pos = self.annotation_window.mapToScene(cursor_pos)
                if not self.cutting_points or scene_pos != self.cutting_points[-1]:
                    self.cutting_points.append(scene_pos)
                self.last_click_point = scene_pos

    def stop_current_drawing(self):
        """Force stop of current cutting if in progress."""
        if self.drawing_in_progress:
            self.drawing_in_progress = False
            self.cutting_points = []
            if self.cutting_path_item:
                self.annotation_window.scene.removeItem(self.cutting_path_item)
                self.cutting_path_item = None
            self.annotation_window.scene.update()

    def _start_drawing_cut_line(self, position):
        """Start drawing the cut line from the given position."""
        self.drawing_in_progress = True
        self.cutting_points = [position]
        path = QPainterPath()
        path.moveTo(position)
        self.cutting_path_item = QGraphicsPathItem(path)
        self.cutting_path_item.setPen(fine_dash_pen(CUT_LINE_COLOR))
        # Without a Z the line sat at 0, beneath the semi-transparent fill of
        # the very annotation being cut (20) -- zoomed in, that fill covers the
        # whole view and the line could not be seen at all.
        self.cutting_path_item.setZValue(TOOL_OVERLAY_Z)
        self.cutting_path_item.setAcceptedMouseButtons(Qt.NoButton)
        self.annotation_window.scene.addItem(self.cutting_path_item)

    def _redraw_cut_line(self, cursor_pos=None):
        """Rebuild the path from the committed points plus a live tail to the cursor."""
        if not self.cutting_path_item or not self.cutting_points:
            return
        path = QPainterPath(self.cutting_points[0])
        for point in self.cutting_points[1:]:
            path.lineTo(point)
        if cursor_pos is not None and cursor_pos != self.cutting_points[-1]:
            path.lineTo(cursor_pos)
        self.cutting_path_item.setPath(path)

    def _update_cut_line_path(self, position):
        """Update the cut line path as the mouse moves."""
        if not self.cutting_path_item:
            return
        # Sample spacing is measured on screen, not in scene pixels: a fixed
        # 5 scene px needed 100+ px of hand movement at high zoom before any
        # point was added, so the line lagged far behind the cursor.
        scale = get_view_scale(self.annotation_window.transform()) or 1.0
        min_scene_dist = CUT_SAMPLE_SCREEN_PX / scale
        if not self.cutting_points or (position - self.cutting_points[-1]).manhattanLength() > min_scene_dist:
            self.cutting_points.append(position)
        # Always draw through to the cursor so the line never trails it.
        self._redraw_cut_line(position)
            
    def _clear_cutting_line(self):
        """Clear the current cutting line but remain in cutting mode."""
        self.cutting_points = []
        self.drawing_in_progress = False
        if self.cutting_path_item:
            self.annotation_window.scene.removeItem(self.cutting_path_item)
            self.cutting_path_item = None
            self.annotation_window.scene.update()
            
    def _break_apart_multipolygon(self):
        """Handle the special case of 'cutting' a MultiPolygonAnnotation."""
        new_annotations = self.target_annotation.cut()
        # Add new annotations without recording individual actions, then remove original
        for new_anno in new_annotations:
            self.annotation_window.add_annotation_from_tool(new_anno, record_action=False)
        try:
            self.annotation_window.delete_annotation(self.target_annotation.id, record_action=False)
        except Exception:
            pass

        # Record as a single split action
        try:
            action = SplitAnnotationAction(self.annotation_window, self.target_annotation, new_annotations)
            self.annotation_window.action_stack.push(action)
        except Exception:
            pass

        # Emit split signal
        try:
            self.annotation_window.annotationSplit.emit(self.target_annotation.id, new_annotations)
        except Exception:
            pass
        self.parent_tool.deactivate_subtool()

    def _finish_and_perform_cut(self):
        """
        Finalize the line drawing and tell the parent SelectTool to perform the cut.

        Returns True when the annotation was cut and cut mode has ended. A line
        that did not divide the annotation is cleared and cut mode stays on, so
        the user can simply draw again; the annotation is left untouched.
        """
        if self.drawing_in_progress and len(self.cutting_points) >= 2:
            if self.parent_tool.cut_selected_annotation(self.cutting_points):
                # The operation is complete, so deactivate the tool.
                self.parent_tool.deactivate_subtool()
                return True

        self._clear_cutting_line()
        self.annotation_window.main_window.status_bar.showMessage(CUT_FAILED_MESSAGE, 6000)
        return False