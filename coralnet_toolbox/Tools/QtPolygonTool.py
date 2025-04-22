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

    def activate(self):
        self.active = True
        self.annotation_window.setCursor(Qt.CrossCursor)

    def deactivate(self):
        self.active = False
        self.annotation_window.setCursor(self.default_cursor)
        self.clear_cursor_annotation()
        self.points = []
        self.drawing_continuous = False
    
    @staticmethod
    def simplify_polygon(points, epsilon):
        """
        Simplify polygon vertices using the Ramer-Douglas-Peucker algorithm.
        
        Args:
            points: List of QPointF vertices defining the polygon
            epsilon: Maximum distance for a point to be considered close enough to the simplified line
                    Higher values = more simplification, lower values = less simplification
        
        Returns:
            List of QPointF vertices defining the simplified polygon
        """
        if len(points) < 3:
            return points
            
        # Convert QPointF to numpy array for processing
        points_array = [(point.x(), point.y()) for point in points]
        
        def rdp(points_array, epsilon):
            """Recursive implementation of the Ramer-Douglas-Peucker algorithm"""
            if len(points_array) <= 2:
                return points_array
            
            # Find the point with the maximum distance from the line between first and last points
            line_start = points_array[0]
            line_end = points_array[-1]
            
            # Calculate the distance of all points to the line
            max_dist = 0
            max_idx = 0
            
            for i in range(1, len(points_array) - 1):
                # Line equation: ax + by + c = 0
                # Where a = y2-y1, b = x1-x2, c = x2*y1 - x1*y2
                a = line_end[1] - line_start[1]
                b = line_start[0] - line_end[0]
                c = line_end[0] * line_start[1] - line_start[0] * line_end[1]
                
                # Distance from point to line = |ax + by + c| / sqrt(a² + b²)
                dist = abs(a * points_array[i][0] + b * points_array[i][1] + c) / ((a * a + b * b) ** 0.5)
                
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i
            
            # If the maximum distance is greater than epsilon, recursively simplify
            if max_dist > epsilon:
                # Recursive call
                first_half = rdp(points_array[:max_idx + 1], epsilon)
                second_half = rdp(points_array[max_idx:], epsilon)
                
                # Build the result (avoiding duplicate points)
                return first_half[:-1] + second_half
            else:
                # All points are close to the line, keep only endpoints
                return [points_array[0], points_array[-1]]
        
        # Run the algorithm
        simplified_array = rdp(points_array, epsilon)
        
        # Convert back to QPointF
        return [QPointF(x, y) for x, y in simplified_array]

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
            
            # Simplify the polygon points before creating the final annotation
            simplified_points = self.simplify_polygon(self.points, self.epsilon)
            
            # Create and add the final annotation with simplified points
            annotation = self.create_annotation(self.annotation_window.mapToScene(event.pos()), 
                                                finished=True, 
                                                simplified_points=simplified_points)
            
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

    def create_annotation(self, scene_pos: QPointF, finished: bool = False, simplified_points=None):
        if not self.annotation_window.active_image or not self.annotation_window.pixmap_image:
            return None

        # Use the simplified points if provided, otherwise use the original points
        points = simplified_points if simplified_points else self.points

        if finished and len(points) > 2:
            # Close the polygon
            points.append(points[0])

        # Create the annotation
        annotation = PolygonAnnotation(points,
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

