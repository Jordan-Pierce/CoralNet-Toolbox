import warnings

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QMouseEvent, QKeyEvent
from PyQt5.QtWidgets import QMessageBox

from coralnet_toolbox.Tools.QtTool import Tool
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class RectangleTool(Tool):
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.cursor = Qt.CrossCursor
        self.start_point = None
        self.end_point = None
        self.drawing_rectangle = False

    def activate(self):
        self.active = True
        self.annotation_window.setCursor(Qt.CrossCursor)

    def mousePressEvent(self, event: QMouseEvent):
        if not self.annotation_window.selected_label:
            QMessageBox.warning(self.annotation_window,
                                "No Label Selected",
                                "A label must be selected before adding an annotation.")
            return None

        if event.button() == Qt.LeftButton and not self.drawing_rectangle:
            # Start drawing the rectangle
            self.start_point = self.annotation_window.mapToScene(event.pos())
            self.drawing_rectangle = True
            self.annotation_window.unselect_annotations()
            self.annotation_window.toggle_cursor_annotation(self.start_point)
        elif event.button() == Qt.LeftButton and self.drawing_rectangle:
            # Finish drawing the rectangle
            self.end_point = self.annotation_window.mapToScene(event.pos())
            self.annotation_window.unselect_annotations()
            self.annotation_window.add_annotation(self.end_point)
            self.drawing_rectangle = False
        elif event.button() == Qt.RightButton and self.drawing_rectangle:
            # Panning the image while drawing
            pass
        else:
            self.cancel_annotation()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing_rectangle:
            # Update the end point while drawing the rectangle
            self.end_point = self.annotation_window.mapToScene(event.pos())
            # Update the annotation graphics
            active_image = self.annotation_window.active_image
            image_pixmap = self.annotation_window.image_pixmap
            cursor_in_window = self.annotation_window.cursorInWindow(event.pos())
            if active_image and image_pixmap and cursor_in_window and self.start_point:
                self.annotation_window.toggle_cursor_annotation(self.end_point)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space:
            # Cancel the current annotation
            self.start_point = None
            self.end_point = None
            self.drawing_rectangle = False
            self.annotation_window.toggle_cursor_annotation()

    def cancel_annotation(self):
        self.start_point = None
        self.end_point = None
        self.drawing_rectangle = False
        self.annotation_window.toggle_cursor_annotation()

    def create_annotation(self, scene_pos: QPointF, finished: bool = False):
        if not self.annotation_window.active_image or not self.annotation_window.image_pixmap:
            return None

        # Get the current end point of the rectangle
        end_point = self.end_point if finished else scene_pos

        # Ensure top_left and bottom_right are correctly calculated
        top_left = QPointF(min(self.start_point.x(), end_point.x()), min(self.start_point.y(), end_point.y()))
        bottom_right = QPointF(max(self.start_point.x(), end_point.x()), max(self.start_point.y(), end_point.y()))

        # Create the rectangle annotation
        annotation = RectangleAnnotation(top_left,
                                         bottom_right,
                                         self.annotation_window.selected_label.short_label_code,
                                         self.annotation_window.selected_label.long_label_code,
                                         self.annotation_window.selected_label.color,
                                         self.annotation_window.current_image_path,
                                         self.annotation_window.selected_label.id,
                                         self.annotation_window.main_window.label_window.active_label.transparency)
        if finished:
            self.start_point = None
            self.end_point = None
            self.drawing_rectangle = False

        return annotation