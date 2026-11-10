from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QMouseEvent, QKeyEvent, QColor, QBrush, QPainterPath
from PyQt5.QtWidgets import QGraphicsPathItem

from coralnet_toolbox.Tools.QtTool import Tool
from coralnet_toolbox.Annotations.QtAnnotation import create_pen
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.utilities import get_view_scale


# Same Z a hydrated annotation group gets: above the phantom layer (10).
PREVIEW_Z = 20
# Minimum on-screen spacing between free-hand vertices, in screen pixels
# (Manhattan). Every mouse event used to add one, however small the move.
FREEHAND_SAMPLE_SCREEN_PX = 2.0


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class PolygonTool(Tool):
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.cursor = Qt.CrossCursor
        self.default_cursor = Qt.ArrowCursor  # Explicitly set, if needed
        self.points = []
        self.drawing_continuous = False  # Flag to indicate continuous drawing mode
        self.epsilon = 1.0  # Default epsilon value for polygon simplification (in pixels)
        self.ctrl_pressed = False  # Flag to track Ctrl key state for straight line drawing
        self.last_click_point = None  # Store the last clicked point for straight line drawing

        # In-progress preview. A single path item updated in place: building a
        # full PolygonAnnotation per mouse move (group, crosshair, bounding box,
        # tag, centroid pass) made long traces lag, O(n) Python per move.
        self._preview_item = None
        # Path through self.points, extended as points are appended so each
        # move costs only the new vertices. _trace_src/_trace_len detect when
        # self.points was replaced or shrank and the path must be rebuilt.
        self._trace_path = None
        self._trace_src = None
        self._trace_len = 0

    def activate(self):
        self.active = True
        self.annotation_window.setCursor(self.cursor)
        self.report_state()

    def status_hint(self):
        """Say whether the next click starts or closes the polygon."""
        if self.drawing_continuous:
            return ("Move to trace, left click to close the polygon"
                    "  |  Hold Ctrl and left click for straight segments"
                    "  |  Backspace or Esc: cancel it")
        return "Left click to start a polygon, then move to trace its outline"

    def deactivate(self):
        # Base clears the preview and the crosshair, then stop_current_drawing().
        super().deactivate()
        self.points = []
        self.drawing_continuous = False
        self.ctrl_pressed = False
        self.last_click_point = None

    def mousePressEvent(self, event: QMouseEvent):
        """Handles mouse press events for starting, continuing, or finishing polygon drawing."""
        if not self.annotation_window.selected_label:
            self.annotation_window.main_window.status_bar.showMessage(
                "A label must be selected before adding an annotation.", 4000)
            return None

        # Add cursor bounds check
        if not self.annotation_window.cursorInWindow(event.pos()):
            return None

        if event.button() == Qt.LeftButton and not self.drawing_continuous:
            self.drawing_continuous = True
            self.annotation_window.unselect_annotations()
            scene_pos = self.annotation_window.mapToScene(event.pos())
            self.points.append(scene_pos)
            self.last_click_point = scene_pos
            self.create_cursor_annotation(scene_pos)
            self.report_state()
        elif event.button() == Qt.LeftButton and self.drawing_continuous:
            scene_pos = self.annotation_window.mapToScene(event.pos())
            if self.ctrl_pressed and self.last_click_point:
                # Add a straight line segment from last point to this click
                self.points.append(scene_pos)
                self.last_click_point = scene_pos
                self.update_cursor_annotation(scene_pos)
            else:
                # Free-hand: finish polygon
                self.points.append(scene_pos)
                self.annotation_window.unselect_annotations()
                annotation = self.create_annotation(scene_pos, finished=True)
                self.annotation_window.add_annotation_from_tool(annotation)
                self.drawing_continuous = False
                self.clear_cursor_annotation()
                self.report_state()
        elif event.button() == Qt.RightButton and self.drawing_continuous:
            pass
        else:
            self.cancel_annotation()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handles mouse move events for updating the polygon preview and crosshair."""
        # Crosshair, and last_scene_pos so a label change mid-draw can repaint
        # the preview in place (refresh_label_preview).
        super().mouseMoveEvent(event)

        if not self.drawing_continuous:
            return

        active_image = self.annotation_window.active_image
        cursor_in_window = self.annotation_window.cursorInWindow(event.pos())
        scene_pos = self.annotation_window.mapToScene(event.pos())

        if active_image and cursor_in_window and self.points:
            if not (self.ctrl_pressed and self.last_click_point):
                # Free-hand: commit a vertex once the cursor has moved far
                # enough on screen. The preview runs to the cursor regardless.
                if self._far_enough_on_screen(scene_pos):
                    self.points.append(scene_pos)
            # Ctrl held: straight-line preview from the last vertex to the cursor.
            self.update_cursor_annotation(scene_pos)

    def keyPressEvent(self, event: QKeyEvent):
        """Handles key press events for canceling annotation or toggling straight line mode."""
        if event.key() in (Qt.Key_Backspace, Qt.Key_Escape):
            self.cancel_annotation()
            self.report_state()
        elif event.key() == Qt.Key_Control:
            # Check if drawing is active and if Ctrl wasn't already pressed
            if self.drawing_continuous and not self.ctrl_pressed:
                self.ctrl_pressed = True
                cursor_pos = self.annotation_window.mapFromGlobal(self.annotation_window.cursor().pos())
                scene_pos = self.annotation_window.mapToScene(cursor_pos)
                # Add the current point when switching to straight-line mode
                if not self.points or scene_pos != self.points[-1]:
                    self.points.append(scene_pos)
                self.last_click_point = scene_pos  # Update the anchor for the straight line
                self.update_cursor_annotation(scene_pos)  # Update preview for straight line

    def keyReleaseEvent(self, event: QKeyEvent):
        """Handles key release events for toggling back to free-hand mode."""
        if event.key() == Qt.Key_Control:
            # Check if drawing is active and if Ctrl was actually pressed
            if self.drawing_continuous and self.ctrl_pressed:
                self.ctrl_pressed = False
                cursor_pos = self.annotation_window.mapFromGlobal(self.annotation_window.cursor().pos())
                scene_pos = self.annotation_window.mapToScene(cursor_pos)
                # Add the current point when switching back to free-hand mode
                if not self.points or scene_pos != self.points[-1]:
                    self.points.append(scene_pos)
                # Update last_click_point, although less critical for free-hand mode start
                self.last_click_point = scene_pos
                self.update_cursor_annotation(scene_pos)  # Update preview for free-hand

    def cancel_annotation(self):
        """Cancels the current polygon drawing operation."""
        self.points = []
        self.drawing_continuous = False
        self.clear_cursor_annotation()
        self.last_click_point = None

    def stop_current_drawing(self):
        """Force stop of current polygon drawing if in progress."""
        if self.drawing_continuous:
            self.points = []
            self.drawing_continuous = False
            self.clear_cursor_annotation()
            self.last_click_point = None

    def create_annotation(self, scene_pos: QPointF, finished: bool = False):
        """Creates a PolygonAnnotation from the current points."""
        if not self.annotation_window.active_image:
            return None

        # Create the annotation with current points
        if finished and len(self.points) > 2:
            # Close the polygon
            self.points.append(self.points[0])

            # --- Validation for polygon size and shape ---
            # Step 1: Remove duplicate or near-duplicate points
            filtered_points = []
            MIN_DISTANCE = 2.0  # Minimum distance between points in pixels

            for i, point in enumerate(self.points):
                # Skip if this point is too close to the previous one
                if i > 0:
                    prev_point = filtered_points[-1]
                    distance = ((point.x() - prev_point.x())**2 + (point.y() - prev_point.y())**2)**0.5
                    if distance < MIN_DISTANCE:
                        continue
                filtered_points.append(point)

            # Step 2: Ensure we have enough points for a valid polygon
            if len(filtered_points) < 4:  # Need at least 3 + 1 closing point
                # Create a small triangle/square if we don't have enough points
                if len(filtered_points) > 0:
                    center_x = sum(p.x() for p in filtered_points) / len(filtered_points)
                    center_y = sum(p.y() for p in filtered_points) / len(filtered_points)

                    # Create a small polygon centered on the average of existing points
                    MIN_SIZE = 5.0
                    filtered_points = [
                        QPointF(center_x - MIN_SIZE, center_y - MIN_SIZE),
                        QPointF(center_x + MIN_SIZE, center_y - MIN_SIZE),
                        QPointF(center_x + MIN_SIZE, center_y + MIN_SIZE),
                        QPointF(center_x - MIN_SIZE, center_y + MIN_SIZE),
                        QPointF(center_x - MIN_SIZE, center_y - MIN_SIZE)  # Close the polygon
                    ]

                    self.annotation_window.main_window.status_bar.showMessage(
                        "The polygon had too few unique points and has been adjusted to a minimum size.", 4000
                    )

            # Use the filtered points list instead of the original
            self.points = filtered_points

        # Create the annotation with validated points
        annotation = PolygonAnnotation(
            self.points,
            self.annotation_window.selected_label,
            self.annotation_window.current_image_path,
            transparency=self.annotation_window.main_window.get_transparency_value(),
            show_confidence=False,
        )

        if finished:
            # Reset the tool
            self.points = []
            self.drawing_continuous = False
            self.last_click_point = None

        return annotation

    def _far_enough_on_screen(self, scene_pos: QPointF) -> bool:
        """True once scene_pos is FREEHAND_SAMPLE_SCREEN_PX from the last vertex on screen."""
        if not self.points:
            return True
        scale = get_view_scale(self.annotation_window.transform()) or 1.0
        return (scene_pos - self.points[-1]).manhattanLength() * scale >= FREEHAND_SAMPLE_SCREEN_PX

    def _committed_path(self) -> QPainterPath:
        """Open path through self.points, extended in place with any new points."""
        if self._trace_src is not self.points or self._trace_len > len(self.points):
            self._trace_path = QPainterPath()
            self._trace_src = self.points
            self._trace_len = 0
        for point in self.points[self._trace_len:]:
            if self._trace_len == 0:
                self._trace_path.moveTo(point)
            else:
                self._trace_path.lineTo(point)
            self._trace_len += 1
        return self._trace_path

    def create_cursor_annotation(self, scene_pos: QPointF = None):
        """Draw the polygon in progress: the traced vertices, a live edge to the
        cursor, and the closing edge back to the start.

        The preview is drawn straight from the raw points, never through
        PolygonAnnotation's simplify pass. An open stroke closed back to its
        start is often a self-intersecting sliver (a vertical stroke with 1 px
        of x jitter is a zigzag); simplify_polygon runs buffer(0) on that and
        GEOS keeps only one lobe of a bowtie, so the preview used to collapse to
        a 2 px triangle and vanish mid-draw. The finished polygon is still
        simplified as before.
        """
        label = self.annotation_window.selected_label
        if (not scene_pos or not label or not self.annotation_window.active_image
                or not self.drawing_continuous or not self.points):
            self.clear_cursor_annotation()
            return

        path = QPainterPath(self._committed_path())
        if scene_pos != self.points[-1]:
            path.lineTo(scene_pos)
        path.closeSubpath()
        # Winding fill keeps both lobes of a self-crossing trace solid.
        path.setFillRule(Qt.WindingFill)

        item = self._preview_item
        try:
            alive = item is not None and item.scene() is self.annotation_window.scene
        except RuntimeError:
            alive = False  # The scene was cleared and deleted the item.
        if not alive:
            self.clear_cursor_annotation()
            item = QGraphicsPathItem()
            item.setZValue(PREVIEW_Z)
            item.setAcceptedMouseButtons(Qt.NoButton)
            self.annotation_window.scene.addItem(item)
            self._preview_item = item

        # Restyled every call (Qt no-ops an unchanged pen/brush) so a label or
        # transparency change mid-draw shows on the next move.
        color = QColor(label.color)
        fill = QColor(color)
        fill.setAlpha(self.annotation_window.main_window.get_transparency_value())
        item.setBrush(QBrush(fill))
        item.setPen(create_pen(color, is_selected=False))
        item.setPath(path)

    def update_cursor_annotation(self, scene_pos: QPointF = None):
        """Update the preview in place."""
        self.create_cursor_annotation(scene_pos)

    def clear_cursor_annotation(self):
        """Remove the preview path item."""
        super().clear_cursor_annotation()
        item, self._preview_item = self._preview_item, None
        if item is None:
            return
        try:
            if item.scene() is not None:
                item.scene().removeItem(item)
        except RuntimeError:
            pass  # Already deleted along with a cleared scene.
