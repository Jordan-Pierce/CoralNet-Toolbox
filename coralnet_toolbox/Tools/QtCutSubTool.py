from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen, QPainterPath
from PyQt5.QtWidgets import QGraphicsPathItem, QMessageBox

from coralnet_toolbox.Tools.QtSubTool import SubTool

from coralnet_toolbox.Annotations import MultiPolygonAnnotation


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
            QMessageBox.warning(
                self.annotation_window, "Cannot Cut",
                "Cannot cut unverified annotations. Confirm prediction (Ctrl+Space) first."
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
        else:
            self._finish_and_perform_cut()

    def mouseMoveEvent(self, event):
        """Handle mouse move events to update the cut line."""
        if not self.drawing_in_progress:
            return
        position = self.annotation_window.mapToScene(event.pos())
        self._update_cut_line_path(position)

    def keyPressEvent(self, event):
        """Handle key press events for canceling the cut."""
        if event.key() in (Qt.Key_Backspace, Qt.Key_Escape):
            self.parent_tool.deactivate_subtool()

    def _start_drawing_cut_line(self, position):
        """Start drawing the cut line from the given position."""
        self.drawing_in_progress = True
        self.cutting_points = [position]
        path = QPainterPath()
        path.moveTo(position)
        line_thickness = self.parent_tool.graphics_utility.get_selection_thickness(self.annotation_window)
        self.cutting_path_item = QGraphicsPathItem(path)
        pen = QPen(Qt.red, line_thickness, Qt.DashLine)
        self.cutting_path_item.setPen(pen)
        self.annotation_window.scene.addItem(self.cutting_path_item)

    def _update_cut_line_path(self, position):
        """Update the cut line path as the mouse moves."""
        if not self.cutting_path_item:
            return
        # Only add point if it's sufficiently far from the last point
        if not self.cutting_points or (position - self.cutting_points[-1]).manhattanLength() > 5:
            self.cutting_points.append(position)
            path = QPainterPath(self.cutting_points[0])
            for point in self.cutting_points[1:]:
                path.lineTo(point)
            self.cutting_path_item.setPath(path)
            
    def _break_apart_multipolygon(self):
        """Handle the special case of 'cutting' a MultiPolygonAnnotation."""
        new_annotations = self.target_annotation.cut()
        self.annotation_window.delete_annotation(self.target_annotation.id)
        for new_anno in new_annotations:
            self.annotation_window.add_annotation_from_tool(new_anno)
        self.parent_tool.deactivate_subtool()

    def _finish_and_perform_cut(self):
        """
        Finalize the line drawing and tell the parent SelectTool to perform the cut.
        """
        if not self.drawing_in_progress or len(self.cutting_points) < 2:
            # Not a valid cut, just cancel
            self.parent_tool.deactivate_subtool()
            return

        # Call back to the parent tool to execute the cut.
        self.parent_tool.cut_selected_annotation(self.cutting_points)

        # The operation is complete, so deactivate the tool.
        self.parent_tool.deactivate_subtool()