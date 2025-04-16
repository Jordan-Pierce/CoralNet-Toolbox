import warnings

from PyQt5.QtCore import Qt, QPointF, QLineF
from PyQt5.QtGui import QMouseEvent, QKeyEvent, QPen, QColor
from PyQt5.QtWidgets import QMessageBox, QGraphicsLineItem

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
        self.ctrl_pressed = False  # Flag to track Ctrl key state
        self.last_point = None  # Store the last point for straight line drawing
        self.line_item = None  # Temporary line graphic for straight line mode

    def activate(self):
        self.active = True
        self.annotation_window.setCursor(Qt.CrossCursor)

    def deactivate(self):
        self.active = False
        self.remove_line_preview()
        self.annotation_window.setCursor(Qt.ArrowCursor)

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
            self.last_point = self.points[-1]  # Store the last point
            self.annotation_window.toggle_cursor_annotation(self.annotation_window.mapToScene(event.pos()))
        elif event.button() == Qt.LeftButton and self.drawing_continuous:
            # Finish the current annotation
            self.points.append(self.annotation_window.mapToScene(event.pos()))
            self.annotation_window.unselect_annotations()
            self.annotation_window.add_annotation(self.annotation_window.mapToScene(event.pos()))
            self.drawing_continuous = False
            self.last_point = None
            self.remove_line_preview()
        elif event.button() == Qt.RightButton and self.drawing_continuous:
            # Panning the image while drawing
            pass
        else:
            self.cancel_annotation()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing_continuous:
            active_image = self.annotation_window.active_image
            pixmap_image = self.annotation_window.pixmap_image
            cursor_in_window = self.annotation_window.cursorInWindow(event.pos())
            
            if active_image and pixmap_image and cursor_in_window:
                current_pos = self.annotation_window.mapToScene(event.pos())
                
                if self.ctrl_pressed and self.last_point:
                    # When Ctrl is held, only update the line preview without adding points
                    self.update_line_preview(self.last_point, current_pos)
                    self.annotation_window.toggle_cursor_annotation(current_pos)
                else:
                    # Normal continuous drawing mode - add points as the mouse moves
                    self.points.append(current_pos)
                    self.last_point = current_pos
                    self.annotation_window.toggle_cursor_annotation(current_pos)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Backspace:
            # Cancel the current annotation
            self.cancel_annotation()
        elif event.key() == Qt.Key_Control:
            # Start straight line mode
            self.ctrl_pressed = True
            if self.drawing_continuous and self.last_point:
                # Create a line preview when Ctrl is first pressed
                cursor_pos = self.annotation_window.mapFromGlobal(self.annotation_window.cursor().pos())
                if self.annotation_window.cursorInWindow(cursor_pos):
                    current_scene_pos = self.annotation_window.mapToScene(cursor_pos)
                    self.update_line_preview(self.last_point, current_scene_pos)

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Control:
            # End straight line mode
            self.ctrl_pressed = False
            if self.drawing_continuous:
                # When Ctrl is released, drop a point at current position
                cursor_pos = self.annotation_window.mapFromGlobal(self.annotation_window.cursor().pos())
                if self.annotation_window.cursorInWindow(cursor_pos):
                    current_scene_pos = self.annotation_window.mapToScene(cursor_pos)
                    self.points.append(current_scene_pos)
                    self.last_point = current_scene_pos
                    self.annotation_window.toggle_cursor_annotation(current_scene_pos)
                    self.remove_line_preview()

    def update_line_preview(self, start_point, end_point):
        """Update or create the temporary straight line preview"""
        # Remove any existing line preview
        self.remove_line_preview()
        
        # Create a new line preview
        line = QLineF(start_point, end_point)
        self.line_item = QGraphicsLineItem(line)
        
        # Style the line
        if self.annotation_window.selected_label:
            color = QColor(self.annotation_window.selected_label.color)
        else:
            color = QColor(Qt.white)
        
        # Make the line visible and matched to the annotation style
        pen = QPen(color, 3, Qt.DashLine)
        self.line_item.setPen(pen)
        
        # Add the line to the scene
        self.annotation_window.scene.addItem(self.line_item)

    def remove_line_preview(self):
        """Remove the temporary line preview"""
        if self.line_item:
            if self.line_item.scene():
                self.line_item.scene().removeItem(self.line_item)
            self.line_item = None

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
                                       self.annotation_window.main_window.label_window.active_label.transparency)

        if finished:
            # Reset the tool
            self.points = []
            self.drawing_continuous = False
            self.last_point = None
            self.remove_line_preview()

        return annotation
    
    def cancel_annotation(self):
        self.points = []
        self.drawing_continuous = False
        self.last_point = None
        self.remove_line_preview()
        self.annotation_window.toggle_cursor_annotation()
