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

    def activate(self):
        self.active = True
        self.annotation_window.setCursor(Qt.CrossCursor)

    def deactivate(self):
        self.active = False
        self.annotation_window.setCursor(self.default_cursor)
        self.clear_cursor_annotation()
        self.points = []
        self.drawing_continuous = False

    def mousePressEvent(self, event: QMouseEvent):
        if not self.annotation_window.selected_label:
            QMessageBox.warning(self.annotation_window,
                                "No Label Selected",
                                "A label must be selected before adding an annotation.")
            return None

        if event.button() == Qt.LeftButton and not self.drawing_continuous:
            # Start continuous drawing mode
            self.drawing_continuous = True
            self.annotation_window.unselect_annotations()
            self.points.append(self.annotation_window.mapToScene(event.pos()))
            self.create_cursor_annotation(self.annotation_window.mapToScene(event.pos()))
        elif event.button() == Qt.LeftButton and self.drawing_continuous:
            # Finish the current annotation
            self.points.append(self.annotation_window.mapToScene(event.pos()))
            self.annotation_window.unselect_annotations()
            
            # Create and add the final annotation
            annotation = self.create_annotation(self.annotation_window.mapToScene(event.pos()), finished=True)
            self.annotation_window.add_annotation_from_tool(annotation)
            
            self.drawing_continuous = False
            self.clear_cursor_annotation()
        elif event.button() == Qt.RightButton and self.drawing_continuous:
            # Panning the image while drawing
            pass
        else:
            self.cancel_annotation()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing_continuous:
            # Update the last point in continuous drawing mode
            scene_pos = self.annotation_window.mapToScene(event.pos())
            
            # Update the annotation graphics
            active_image = self.annotation_window.active_image
            pixmap_image = self.annotation_window.pixmap_image
            cursor_in_window = self.annotation_window.cursorInWindow(event.pos())
            if active_image and pixmap_image and cursor_in_window and self.points:
                self.points.append(scene_pos)
                self.update_cursor_annotation(scene_pos)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Backspace:
            # Cancel the current annotation
            self.cancel_annotation()

    def cancel_annotation(self):
        self.points = []
        self.drawing_continuous = False
        self.clear_cursor_annotation()

    def create_annotation(self, scene_pos: QPointF, finished: bool = False):
        if not self.annotation_window.active_image or not self.annotation_window.pixmap_image:
            return None

        if finished and len(self.points) > 2:
            # Close the polygon
            self.points.append(self.points[0])

        # Create the annotation
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

        return annotation
        
    def create_cursor_annotation(self, scene_pos: QPointF = None):
        """Create a polygon cursor annotation at the given position."""
        if not scene_pos or not self.annotation_window.selected_label or not self.annotation_window.active_image:
            self.clear_cursor_annotation()
            return
            
        # Create a temporary polygon for visualization
        if self.drawing_continuous and len(self.points) > 0:
            # Create a new cursor annotation with semi-transparent appearance
            self.cursor_annotation = self.create_annotation(scene_pos)
            if self.cursor_annotation:
                # Make the cursor annotation semi-transparent to distinguish it from actual annotations
                self.cursor_annotation.transparency = min(self.cursor_annotation.transparency + 100, 200)
                self.cursor_annotation.create_graphics_item(self.annotation_window.scene)

    def update_cursor_annotation(self, scene_pos: QPointF = None):
        """Update the cursor annotation position."""
        self.clear_cursor_annotation()
        self.create_cursor_annotation(scene_pos)
