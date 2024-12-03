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
            self.annotation_window.toggle_cursor_annotation(self.annotation_window.mapToScene(event.pos()))
        elif event.button() == Qt.LeftButton and self.drawing_continuous:
            # Finish the current annotation
            self.points.append(self.annotation_window.mapToScene(event.pos()))
            self.annotation_window.unselect_annotations()
            self.annotation_window.add_annotation(self.annotation_window.mapToScene(event.pos()))
            self.drawing_continuous = False
        elif event.button() == Qt.RightButton and self.drawing_continuous:
            # Panning the image while drawing
            pass
        else:
            self.cancel_annotation()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing_continuous:
            # Update the last point in continuous drawing mode
            self.points.append(self.annotation_window.mapToScene(event.pos()))
            # Update the annotation graphics
            active_image = self.annotation_window.active_image
            image_pixmap = self.annotation_window.image_pixmap
            cursor_in_window = self.annotation_window.cursorInWindow(event.pos())
            if active_image and image_pixmap and cursor_in_window and self.points:
                self.annotation_window.toggle_cursor_annotation(self.annotation_window.mapToScene(event.pos()))

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space:
            # Cancel the current annotation
            self.points = []
            self.drawing_continuous = False
            self.annotation_window.toggle_cursor_annotation()

    def cancel_annotation(self):
        self.points = []
        self.drawing_continuous = False
        self.annotation_window.toggle_cursor_annotation()

    def create_annotation(self, scene_pos: QPointF, finished: bool = False):
        if not self.annotation_window.active_image or not self.annotation_window.image_pixmap:
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
                                       self.annotation_window.main_window.label_window.active_label.transparency)

        if finished:
            # Reset the tool
            self.points = []
            self.drawing_continuous = False

        return annotation