import warnings

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QMouseEvent, QKeyEvent
from PyQt5.QtWidgets import QMessageBox

from coralnet_toolbox.Tools.QtTool import Tool
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class PolygonTool(Tool):
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.cursor = Qt.CrossCursor
        self.points = []
        self.drawing_continuous = False  # Flag to indicate continuous drawing mode
        self.epsilon = 1.0  # Default epsilon value for polygon simplification (in pixels)
        self.ctrl_pressed = False  # Flag to track Ctrl key state for straight line drawing
        self.last_click_point = None  # Store the last clicked point for straight line drawing

    def activate(self):
        self.active = True
        self.annotation_window.setCursor(Qt.CrossCursor)

    def deactivate(self):
        self.active = False
        self.annotation_window.setCursor(self.default_cursor)
        self.clear_cursor_annotation()
        self.points = []
        self.drawing_continuous = False
        self.ctrl_pressed = False
        self.last_click_point = None

    def mousePressEvent(self, event: QMouseEvent):
        if not self.annotation_window.selected_label:
            QMessageBox.warning(self.annotation_window,
                                "No Label Selected",
                                "A label must be selected before adding an annotation.")
            return None

        if event.button() == Qt.LeftButton and not self.drawing_continuous:
            self.drawing_continuous = True
            self.annotation_window.unselect_annotations()
            scene_pos = self.annotation_window.mapToScene(event.pos())
            self.points.append(scene_pos)
            self.last_click_point = scene_pos
            self.create_cursor_annotation(scene_pos)
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
        elif event.button() == Qt.RightButton and self.drawing_continuous:
            pass
        else:
            self.cancel_annotation()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing_continuous:
            scene_pos = self.annotation_window.mapToScene(event.pos())
            active_image = self.annotation_window.active_image
            pixmap_image = self.annotation_window.pixmap_image
            cursor_in_window = self.annotation_window.cursorInWindow(event.pos())
            if active_image and pixmap_image and cursor_in_window and self.points:
                if self.ctrl_pressed and self.last_click_point:
                    # Show a straight line preview from last point to cursor, do not modify self.points
                    self.update_cursor_annotation(scene_pos)
                else:
                    # Free-hand: add points as the mouse moves
                    self.points.append(scene_pos)
                    self.update_cursor_annotation(scene_pos)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Backspace:
            self.cancel_annotation()
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
        self.points = []
        self.drawing_continuous = False
        self.clear_cursor_annotation()
        self.last_click_point = None

    def create_annotation(self, scene_pos: QPointF, finished: bool = False):
        if not self.annotation_window.active_image or not self.annotation_window.pixmap_image:
            return None

        # Create the annotation with current points
        # The polygon simplification is now handled inside the PolygonAnnotation class
        if finished and len(self.points) > 2:
            # Close the polygon
            self.points.append(self.points[0])

        # Create the annotation - will be simplified in the constructor
        annotation = PolygonAnnotation(self.points,
                                       self.annotation_window.selected_label.short_label_code,
                                       self.annotation_window.selected_label.long_label_code,
                                       self.annotation_window.selected_label.color,
                                       self.annotation_window.current_image_path,
                                       self.annotation_window.selected_label.id,
                                       self.annotation_window.selected_label.transparency)

        if finished:
            # Reset the tool
            self.points = []
            self.drawing_continuous = False
            self.last_click_point = None

        return annotation
        
    def create_cursor_annotation(self, scene_pos: QPointF = None):
        """Create a polygon cursor annotation at the given position."""
        if not scene_pos or not self.annotation_window.selected_label or not self.annotation_window.active_image:
            self.clear_cursor_annotation()
            return

        if self.drawing_continuous and len(self.points) > 0:
            # Determine points for preview: always include current scene_pos
            preview_points = self.points + [scene_pos]

            # Create the preview annotation using preview_points
            annotation = PolygonAnnotation(
                preview_points,
                self.annotation_window.selected_label.short_label_code,
                self.annotation_window.selected_label.long_label_code,
                self.annotation_window.selected_label.color,
                self.annotation_window.current_image_path,
                self.annotation_window.selected_label.id,
                min(self.annotation_window.selected_label.transparency + 100, 200) # Increased transparency for preview
            )
            annotation.create_graphics_item(self.annotation_window.scene)
            self.cursor_annotation = annotation

    def update_cursor_annotation(self, scene_pos: QPointF = None):
        """Update the cursor annotation position."""
        # Clear previous preview first to avoid flickering or overlap
        self.clear_cursor_annotation()
        # Create the new preview at the current cursor position
        self.create_cursor_annotation(scene_pos)

