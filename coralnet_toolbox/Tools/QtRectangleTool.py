import warnings

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QMouseEvent, QKeyEvent
from PyQt5.QtWidgets import QMessageBox, QGraphicsPixmapItem

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
        self.default_cursor = Qt.ArrowCursor  # Explicitly set, if needed
        self.start_point = None
        self.end_point = None
        self.drawing_continuous = False

    def activate(self):
        self.active = True
        self.annotation_window.setCursor(self.cursor)

    def deactivate(self):
        super().deactivate()
        self.start_point = None
        self.end_point = None
        self.drawing_continuous = False

    def mousePressEvent(self, event: QMouseEvent):
        
        if not self.annotation_window.selected_label:
            QMessageBox.warning(self.annotation_window,
                                "No Label Selected",
                                "A label must be selected before adding an annotation.")
            return None
        
        # Add cursor bounds check
        if not self.annotation_window.cursorInWindow(event.pos()):
            return None

        if event.button() == Qt.LeftButton and not self.drawing_continuous:
            # Start drawing the rectangle
            self.start_point = self.annotation_window.mapToScene(event.pos())
            self.drawing_continuous = True
            self.annotation_window.unselect_annotations()
            # Create the initial cursor annotation
            self.create_cursor_annotation(self.start_point)
        elif event.button() == Qt.LeftButton and self.drawing_continuous:
            # Finish drawing the rectangle
            self.end_point = self.annotation_window.mapToScene(event.pos())
            self.annotation_window.unselect_annotations()
            # Add the annotation and finish drawing
            annotation = self.create_annotation(self.end_point, finished=True)
            self.annotation_window.add_annotation_from_tool(annotation)
            self.drawing_continuous = False
            # Clear the cursor annotation when finished
            self.clear_cursor_annotation()
        elif event.button() == Qt.RightButton and self.drawing_continuous:
            # Panning the image while drawing
            pass
        else:
            self.cancel_annotation()

    def mouseMoveEvent(self, event: QMouseEvent):
        # Call parent implementation to handle crosshair
        super().mouseMoveEvent(event)
        
        # Continue with tool-specific behavior
        if self.drawing_continuous:
            self.end_point = self.annotation_window.mapToScene(event.pos())
            
            # Update the cursor annotation if we're in the window
            active_image = self.annotation_window.active_image
            pixmap_image = self.annotation_window.pixmap_image
            cursor_in_window = self.annotation_window.cursorInWindow(event.pos())
            if active_image and pixmap_image and cursor_in_window and self.start_point:
                self.update_cursor_annotation(self.end_point)
                
                # Show crosshair at current cursor position during drawing
                self.update_crosshair(self.end_point)
        else:
            # Show a preview rectangle at the cursor position when not drawing
            scene_pos = self.annotation_window.mapToScene(event.pos())
            cursor_in_window = self.annotation_window.cursorInWindow(event.pos())
            
            # Show crosshair guides when cursor is over the image
            if cursor_in_window and self.active and self.annotation_window.selected_label:
                self.update_crosshair(scene_pos)
            else:
                self.clear_crosshair()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Backspace:
            # Cancel the current annotation
            self.cancel_annotation()

    def cancel_annotation(self):
        self.start_point = None
        self.end_point = None
        self.drawing_continuous = False
        self.clear_cursor_annotation()

    def create_annotation(self, scene_pos: QPointF, finished: bool = False):
        if not self.annotation_window.active_image or not self.annotation_window.pixmap_image:
            return None

        # Get the current end point of the rectangle
        end_point = self.end_point if finished else scene_pos

        if not self.start_point or not end_point:
            return None

        # Ensure top_left and bottom_right are correctly calculated
        top_left = QPointF(min(self.start_point.x(), end_point.x()), min(self.start_point.y(), end_point.y()))
        bottom_right = QPointF(max(self.start_point.x(), end_point.x()), max(self.start_point.y(), end_point.y()))
        
        # Calculate width and height of the rectangle
        width = bottom_right.x() - top_left.x()
        height = bottom_right.y() - top_left.y()
        
        # Define minimum dimensions for a valid rectangle (e.g., 3x3 pixels)
        MIN_DIMENSION = 3.0
        
        # If rectangle is too small and we're finalizing it, enforce minimum size
        if finished and (width < MIN_DIMENSION or height < MIN_DIMENSION):
            if width < MIN_DIMENSION:
                # Expand width while maintaining center
                center_x = (top_left.x() + bottom_right.x()) / 2
                top_left.setX(center_x - MIN_DIMENSION / 2)
                bottom_right.setX(center_x + MIN_DIMENSION / 2)
                
            if height < MIN_DIMENSION:
                # Expand height while maintaining center
                center_y = (top_left.y() + bottom_right.y()) / 2
                top_left.setY(center_y - MIN_DIMENSION / 2)
                bottom_right.setY(center_y + MIN_DIMENSION / 2)
            
            # Show a message if we had to adjust a very small rectangle
            if width < 1 or height < 1:
                QMessageBox.information(
                    self.annotation_window, 
                    "Rectangle Adjusted",
                    f"The rectangle was too small and has been adjusted to a minimum size of "
                    f"{MIN_DIMENSION}x{MIN_DIMENSION} pixels."
                )

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
            self.drawing_continuous = False

        return annotation
        
    def create_cursor_annotation(self, scene_pos: QPointF = None):
        """Create a rectangle cursor annotation at the given position."""
        if not scene_pos or not self.start_point:
            self.clear_cursor_annotation()
            return
        
        if not self.annotation_window.active_image or not self.annotation_window.selected_label:
            self.clear_cursor_annotation()
            return
            
        # Create a new cursor annotation with semi-transparent appearance
        if self.drawing_continuous and self.start_point:
            # Create a rectangle from start_point to scene_pos
            self.cursor_annotation = self.create_annotation(scene_pos)
            if self.cursor_annotation:
                # Make the cursor annotation semi-transparent to distinguish it from actual annotations
                transparency = self.annotation_window.main_window.label_window.active_label.transparency
                self.cursor_annotation.transparency = transparency
                self.cursor_annotation.create_graphics_item(self.annotation_window.scene)

    def update_cursor_annotation(self, scene_pos: QPointF = None):
        """Update the rectangle cursor annotation."""
        self.clear_cursor_annotation()
        self.create_cursor_annotation(scene_pos)