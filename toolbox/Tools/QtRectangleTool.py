import warnings

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QMouseEvent, QKeyEvent
from PyQt5.QtWidgets import QMessageBox

from toolbox.Tools.QtTool import Tool
from toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

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
        self.resizing_top_left = False
        self.resizing_bottom_right = False

    def activate(self):
        self.active = True
        self.annotation_window.setCursor(Qt.CrossCursor)

    def mousePressEvent(self, event: QMouseEvent):
        if not self.annotation_window.selected_label:
            QMessageBox.warning(self.annotation_window,
                                "No Label Selected",
                                "A label must be selected before adding an annotation.")
            return None

        if event.button() == Qt.LeftButton:
            position = self.annotation_window.mapToScene(event.pos())
            if self.annotation_window.selected_annotation and isinstance(self.annotation_window.selected_annotation, RectangleAnnotation):
                annotation = self.annotation_window.selected_annotation
                if annotation.top_left.x() - 5 <= position.x() <= annotation.top_left.x() + 5 and \
                   annotation.top_left.y() - 5 <= position.y() <= annotation.top_left.y() + 5:
                    self.resizing_top_left = True
                elif annotation.bottom_right.x() - 5 <= position.x() <= annotation.bottom_right.x() + 5 and \
                     annotation.bottom_right.y() - 5 <= position.y() <= annotation.bottom_right.y() + 5:
                    self.resizing_bottom_right = True
                else:
                    self.start_point = position
                    self.drawing_rectangle = True
                    self.annotation_window.unselect_annotation()
                    self.annotation_window.toggle_cursor_annotation(self.start_point)
            else:
                self.start_point = position
                self.drawing_rectangle = True
                self.annotation_window.unselect_annotation()
                self.annotation_window.toggle_cursor_annotation(self.start_point)
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
        elif self.resizing_top_left:
            new_top_left = self.annotation_window.mapToScene(event.pos())
            self.annotation_window.selected_annotation.update_top_left(new_top_left)
        elif self.resizing_bottom_right:
            new_bottom_right = self.annotation_window.mapToScene(event.pos())
            self.annotation_window.selected_annotation.update_bottom_right(new_bottom_right)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            if self.drawing_rectangle:
                self.end_point = self.annotation_window.mapToScene(event.pos())
                self.annotation_window.unselect_annotation()
                self.annotation_window.add_annotation(self.end_point)
                self.drawing_rectangle = False
            elif self.resizing_top_left or self.resizing_bottom_right:
                self.resizing_top_left = False
                self.resizing_bottom_right = False

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
                                         self.annotation_window.main_window.label_window.active_label.transparency,
                                         show_msg=False)
        if finished:
            self.start_point = None
            self.end_point = None
            self.drawing_rectangle = False

        return annotation
