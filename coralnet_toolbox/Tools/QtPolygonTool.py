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
        self.temp_points = []  # For tracking current mouse position without adding to points
        self.drawing_continuous = False  # Flag to indicate continuous drawing mode

    def activate(self):
        self.active = True
        self.annotation_window.setCursor(Qt.CrossCursor)
        # Reset drawing state
        self.points = []
        self.temp_points = []
        self.drawing_continuous = False

    def deactivate(self):
        self.active = False
        self.annotation_window.setCursor(Qt.ArrowCursor)
        self.cancel_annotation()

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
            self.points = []  # Clear any previous points
            scene_pos = self.annotation_window.mapToScene(event.pos())
            self.points.append(scene_pos)
            self.temp_points = list(self.points)  # Initialize temp_points with the first point
            self.create_cursor_annotation(scene_pos)
        elif event.button() == Qt.LeftButton and self.drawing_continuous:
            # Add another point to the polygon
            scene_pos = self.annotation_window.mapToScene(event.pos())
            self.points.append(scene_pos)
            self.temp_points = list(self.points)  # Update temp_points
            self.update_cursor_annotation(scene_pos)
            
            # If we have clicked near the first point, close the polygon and finish
            if len(self.points) > 2:
                # Calculate distance to first point
                first_point = self.points[0]
                dist_to_first = ((scene_pos.x() - first_point.x()) ** 2 + 
                                 (scene_pos.y() - first_point.y()) ** 2) ** 0.5
                if dist_to_first < 10:  # Close enough to finish (10 pixel threshold)
                    self.annotation_window.unselect_annotations()
                    # Close the polygon by using a copy of the first point rather than the current position
                    self.points[-1] = QPointF(self.points[0])
                    self.annotation_window.add_annotation(self.points[0])
                    self.drawing_continuous = False
                    self.points = []
                    self.temp_points = []
        elif event.button() == Qt.RightButton:
            # Finish the polygon if we have enough points
            if self.drawing_continuous and len(self.points) > 2:
                # Close the polygon
                self.points.append(QPointF(self.points[0]))  # Create a copy of the first point
                self.annotation_window.unselect_annotations()
                self.annotation_window.add_annotation(self.points[0])
                self.drawing_continuous = False
                self.points = []
                self.temp_points = []
            else:
                self.cancel_annotation()

    def mouseMoveEvent(self, event: QMouseEvent):
        active_image = self.annotation_window.active_image
        pixmap_image = self.annotation_window.pixmap_image
        cursor_in_window = self.annotation_window.cursorInWindow(event.pos())
        
        if self.drawing_continuous and active_image and pixmap_image and cursor_in_window:
            scene_pos = self.annotation_window.mapToScene(event.pos())
            
            # Update temp_points with current points plus current mouse position
            self.temp_points = list(self.points)
            self.temp_points.append(scene_pos)
            
            # If we have enough points, temporarily close the polygon for preview
            if len(self.points) > 1:
                temp_closed_points = list(self.temp_points)
                # Create a copy of the first point to close the polygon
                if self.points:
                    temp_closed_points.append(QPointF(self.points[0]))
                self.update_cursor_annotation_with_points(temp_closed_points)
            else:
                self.update_cursor_annotation_with_points(self.temp_points)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Backspace:
            # If we're drawing and have points, remove the last point
            if self.drawing_continuous and len(self.points) > 1:
                self.points.pop()
                self.temp_points = list(self.points)
                scene_pos = self.annotation_window.get_scene_pos()
                if scene_pos:
                    self.temp_points.append(scene_pos)
                # Update the preview
                self.update_cursor_annotation_with_points(self.temp_points)
            else:
                # Cancel the current annotation
                self.cancel_annotation()
        elif event.key() == Qt.Key_Escape:
            # Always cancel on escape
            self.cancel_annotation()
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # Finish polygon on Enter if we have enough points
            if self.drawing_continuous and len(self.points) > 2:
                # Close the polygon
                self.points.append(QPointF(self.points[0]))
                self.annotation_window.unselect_annotations()
                self.annotation_window.add_annotation(self.points[0])
                self.drawing_continuous = False
                self.points = []
                self.temp_points = []

    def create_annotation(self, scene_pos: QPointF, finished: bool = False):
        if not self.annotation_window.active_image or not self.annotation_window.pixmap_image:
            return None
            
        # Use points for finished annotation, or temp_points for preview
        points_to_use = self.points if finished else self.temp_points
        
        if len(points_to_use) < 2:
            return None
            
        # Only close the polygon if finished and we have enough points
        if finished and len(points_to_use) > 2:
            # Make sure the polygon is closed
            if points_to_use[0] != points_to_use[-1]:
                points_to_use.append(QPointF(points_to_use[0]))

        # Create the annotation
        annotation = PolygonAnnotation(points_to_use,
                                      self.annotation_window.selected_label.short_label_code,
                                      self.annotation_window.selected_label.long_label_code,
                                      self.annotation_window.selected_label.color,
                                      self.annotation_window.current_image_path,
                                      self.annotation_window.selected_label.id,
                                      self.annotation_window.main_window.label_window.active_label.transparency)

        if finished:
            # Reset the tool
            self.points = []
            self.temp_points = []
            self.drawing_continuous = False

        return annotation
    
    def create_cursor_annotation(self, scene_pos: QPointF = None):
        """Create a polygon cursor annotation based on current points."""
        if (not scene_pos or not self.annotation_window.selected_label or 
                not self.annotation_window.active_image):
            return
        
        # First clear any existing cursor annotation
        self.clear_cursor_annotation()
        
        # For initial point, just show a single point
        self.temp_points = [scene_pos]
        
        self.cursor_annotation = self.create_annotation(scene_pos)
        if self.cursor_annotation:
            # Make the cursor annotation semi-transparent to distinguish it from actual annotations
            self.cursor_annotation.transparency = min(self.cursor_annotation.transparency + 100, 200)
            self.cursor_annotation.create_graphics_item(self.annotation_window.scene)
    
    def update_cursor_annotation_with_points(self, points):
        """
        Update cursor annotation with specific points instead of using current scene position.
        This is needed for polygon preview with temporary points.
        """
        if not self.annotation_window.selected_label or not self.annotation_window.active_image:
            return
            
        # Clear existing cursor annotation
        self.clear_cursor_annotation()
        
        # Store current points for reference
        self.temp_points = list(points)
        
        # Create a new annotation with the provided points
        self.cursor_annotation = PolygonAnnotation(
            points,
            self.annotation_window.selected_label.short_label_code,
            self.annotation_window.selected_label.long_label_code,
            self.annotation_window.selected_label.color,
            self.annotation_window.current_image_path,
            self.annotation_window.selected_label.id,
            self.annotation_window.selected_label.transparency
        )
        
        if self.cursor_annotation:
            # Make the cursor annotation semi-transparent
            self.cursor_annotation.transparency = min(self.cursor_annotation.transparency + 100, 200)
            self.cursor_annotation.create_graphics_item(self.annotation_window.scene)
    
    def cancel_annotation(self):
        self.points = []
        self.temp_points = []
        self.drawing_continuous = False
        self.clear_cursor_annotation()