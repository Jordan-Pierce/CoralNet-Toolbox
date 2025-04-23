import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import math
import numpy as np

from rasterio.windows import Window

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPixmap, QColor, QPen, QBrush, QPolygonF, QPainter
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPolygonItem

from coralnet_toolbox.Annotations.QtAnnotation import Annotation

from coralnet_toolbox.utilities import rasterio_to_cropped_image


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class PolygonAnnotation(Annotation):
    def __init__(self,
                 points: list,
                 short_label_code: str,
                 long_label_code: str,
                 color: QColor,
                 image_path: str,
                 label_id: str,
                 transparency: int = 128,
                 show_msg=False):  # Added epsilon parameter with default value
        super().__init__(short_label_code, long_label_code, color, image_path, label_id, transparency, show_msg)

        self.center_xy = QPointF(0, 0)
        self.cropped_bbox = (0, 0, 0, 0)
        self.annotation_size = 0
        
        self.epsilon = 1.0  # Store epsilon for polygon simplification

        self.set_precision(points, True)
        self.set_centroid()
        self.set_cropped_bbox()

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
            
            # Check if start and end points are the same (or very close)
            if (abs(line_start[0] - line_end[0]) < 1e-6 and 
                abs(line_start[1] - line_end[1]) < 1e-6):
                # If start and end are essentially the same point, keep only one
                return [line_start]
            
            # Calculate the distance of all points to the line
            max_dist = 0
            max_idx = 0
            
            for i in range(1, len(points_array) - 1):
                # Line equation: ax + by + c = 0
                # Where a = y2-y1, b = x1-x2, c = x2*y1 - x1*y2
                a = line_end[1] - line_start[1]
                b = line_start[0] - line_end[0]
                c = line_end[0] * line_start[1] - line_start[0] * line_end[1]
                
                # Check for division by zero (if denominator is zero, line is a point)
                denominator = (a * a + b * b) ** 0.5
                if denominator < 1e-6:
                    # The line is essentially a point, so distance is just standard distance to that point
                    dx = points_array[i][0] - line_start[0]
                    dy = points_array[i][1] - line_start[1]
                    dist = (dx * dx + dy * dy) ** 0.5
                else:
                    # Distance from point to line = |ax + by + c| / sqrt(a² + b²)
                    dist = abs(a * points_array[i][0] + b * points_array[i][1] + c) / denominator
                
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

    def set_precision(self, points: list, reduce: bool = True):
        """
        Set the precision of the points to 3 decimal places and apply polygon simplification.
        
        Args:
            points: List of QPointF vertices defining the polygon
            reduce: Whether to round coordinates to 3 decimal places
        """
        # First apply the polygon simplification if there are enough points
        if len(points) > 3:
            # points = self.simplify_polygon(points, self.epsilon)
            pass  # Simplification is not applied here, yet.
            
        # Then round the coordinates if requested
        if reduce:
            points = [QPointF(round(point.x(), 3), round(point.y(), 3)) for point in points]

        self.points = points

    def set_centroid(self):
        """Calculate the centroid of the polygon defined by the points."""
        centroid_x = sum(point.x() for point in self.points) / len(self.points)
        centroid_y = sum(point.y() for point in self.points) / len(self.points)
        self.center_xy = QPointF(centroid_x, centroid_y)

    def set_cropped_bbox(self):
        """Calculate the bounding box of the polygon defined by the points."""
        min_x = min(point.x() for point in self.points)
        min_y = min(point.y() for point in self.points)
        max_x = max(point.x() for point in self.points)
        max_y = max(point.y() for point in self.points)
        self.cropped_bbox = (min_x, min_y, max_x, max_y)
        self.annotation_size = int(max(max_x - min_x, max_y - min_y))

    def contains_point(self, point: QPointF) -> bool:
        """Check if the given point is inside the polygon defined by the points."""
        polygon = QPolygonF(self.points)
        return polygon.containsPoint(point, Qt.OddEvenFill)

    def create_cropped_image(self, rasterio_src):
        """Create a cropped image from the rasterio source based on the polygon points."""
        # Set the rasterio source for the annotation
        self.rasterio_src = rasterio_src
        # Set the cropped bounding box for the annotation
        self.set_cropped_bbox()
        # Get the bounding box of the polygon
        min_x, min_y, max_x, max_y = self.cropped_bbox

        # Calculate the window for rasterio
        window = Window(
            col_off=max(0, int(min_x)),
            row_off=max(0, int(min_y)),
            width=min(rasterio_src.width - int(min_x), int(max_x - min_x)),
            height=min(rasterio_src.height - int(min_y), int(max_y - min_y))
        )

        # Convert rasterio to QImage
        q_image = rasterio_to_cropped_image(self.rasterio_src, window)
        # Convert QImage to QPixmap
        self.cropped_image = QPixmap.fromImage(q_image)

        self.annotationUpdated.emit(self)  # Notify update
        
    def get_area(self):
        """Calculate the area of the polygon defined by the points."""
        if len(self.points) < 3:
            return 0.0

        # Use the shoelace formula to calculate the area of the polygon
        area = 0.0
        n = len(self.points)
        for i in range(n):
            j = (i + 1) % n
            area += self.points[i].x() * self.points[j].y()
            area -= self.points[j].x() * self.points[i].y()
        return abs(area) / 2.0
    
    def get_perimeter(self):
        """Calculate the perimeter of the polygon defined by the points."""
        if len(self.points) < 2:
            return 0.0

        perimeter = 0.0
        n = len(self.points)
        for i in range(n):
            j = (i + 1) % n
            # Calculate Euclidean distance between points manually
            dx = self.points[i].x() - self.points[j].x()
            dy = self.points[i].y() - self.points[j].y()
            distance = math.sqrt(dx * dx + dy * dy)
            perimeter += distance
        return perimeter

    def get_polygon(self):
        """Get the polygon representation of this polygon annotation."""
        return QPolygonF(self.points)

    def get_bounding_box_top_left(self):
        """Get the top-left corner of the annotation's bounding box."""
        return QPointF(self.cropped_bbox[0], self.cropped_bbox[1])

    def get_bounding_box_bottom_right(self):
        """Get the bottom-right corner of the annotation's bounding box."""
        return QPointF(self.cropped_bbox[2], self.cropped_bbox[3])

    def get_cropped_image_graphic(self):
        """Get the cropped image with the polygon mask applied."""
        if self.cropped_image is None:
            return None

        # Create a QImage with alpha channel for masking
        masked_image = QPixmap(self.cropped_image.size()).toImage()
        masked_image.fill(Qt.transparent)

        # Create a QPainter to draw the mask
        painter = QPainter(masked_image)
        painter.setRenderHint(QPainter.Antialiasing)

        # Create a black brush
        brush = QBrush(QColor(0, 0, 0))  # Black color
        painter.setBrush(brush)
        painter.setPen(Qt.NoPen)

        # Create a copy of the points that are transformed to be relative to the cropped_image
        cropped_points = [QPointF(point.x() - self.cropped_bbox[0],
                                  point.y() - self.cropped_bbox[1]) for point in self.points]

        # Create a polygon from the cropped points
        polygon = QPolygonF(cropped_points)

        # Fill the polygon with white color (the area we want to keep)
        painter.setBrush(QBrush(Qt.white))
        painter.drawPolygon(polygon)

        painter.end()

        # Convert the QImage back to a QPixmap
        mask_pixmap = QPixmap.fromImage(masked_image)

        # Apply the mask to a copy of the cropped image
        cropped_image_graphic = self.cropped_image.copy()
        cropped_image_graphic.setMask(mask_pixmap.mask())

        # Now draw the dotted line outline on top of the masked image
        painter = QPainter(cropped_image_graphic)
        painter.setRenderHint(QPainter.Antialiasing)

        # Create a dotted pen
        pen = QPen(self.label.color)
        pen.setStyle(Qt.DashLine)  # Creates a dotted/dashed line
        pen.setWidth(2)  # Line width
        painter.setPen(pen)

        # Draw the polygon outline with the dotted pen
        painter.drawPolygon(polygon)

        painter.end()

        return cropped_image_graphic

    def update_graphics_item(self, crop_image=True):
        """Update the graphical representation of the annotation using base class method."""
        super().update_graphics_item(crop_image)

        # Update the cropped image if necessary
        if self.rasterio_src and crop_image:
            self.create_cropped_image(self.rasterio_src)

    def update_location(self, new_center_xy: QPointF):
        """Update the location of the annotation by moving it to a new center point."""
        # Clear the machine confidence
        self.update_user_confidence(self.label)

        # Update the location, graphic
        delta = QPointF(round(new_center_xy.x() - self.center_xy.x(), 2),
                        round(new_center_xy.y() - self.center_xy.y(), 2))

        new_points = [point + delta for point in self.points]

        self.set_precision(new_points)
        self.set_centroid()
        self.set_cropped_bbox()
        self.update_graphics_item()
        self.annotationUpdated.emit(self)  # Notify update

    def update_annotation_size(self, delta: float):
        """Update the size of the annotation by a given delta value."""
        # Clear the machine confidence
        self.update_user_confidence(self.label)

        # Calculate the new points for erosion or dilation
        new_points = []
        num_points = len(self.points)

        for i in range(num_points):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % num_points]

            # Calculate the vector from p1 to p2
            edge_vector = QPointF(p2.x() - p1.x(), p2.y() - p1.y())

            # Calculate the normal vector (perpendicular to the edge)
            normal_vector = QPointF(-edge_vector.y(), edge_vector.x())

            # Normalize the normal vector
            length = math.sqrt(normal_vector.x() ** 2 + normal_vector.y() ** 2)
            if length != 0:
                normal_vector = QPointF(normal_vector.x() / length, normal_vector.y() / length)
            else:
                normal_vector = QPointF(0, 0)

            # Move the point along the normal vector by the delta amount
            if delta < 1:
                new_point = QPointF(p1.x() - normal_vector.x() * (1 - delta),
                                    p1.y() - normal_vector.y() * (1 - delta))
            else:
                new_point = QPointF(p1.x() + normal_vector.x() * (delta - 1),
                                    p1.y() + normal_vector.y() * (delta - 1))
            new_points.append(new_point)

        # Update the points
        self.set_precision(new_points)
        self.set_centroid()
        self.set_cropped_bbox()
        self.update_graphics_item()
        self.annotationUpdated.emit(self)  # Notify update

    def resize(self, handle, new_pos):
        """Resize the annotation by moving a specific handle to a new position."""
        # Clear the machine confidence
        self.update_user_confidence(self.label)

        # Extract the point index from the handle string (e.g., "point_0" -> 0)
        if handle.startswith("point_"):
            new_points = self.points.copy()
            point_index = int(handle.split("_")[1])
            num_points = len(new_points)

            # Update the selected point
            delta = new_pos - new_points[point_index]
            new_points[point_index] = new_pos

            # Define decay factor (controls how quickly influence diminishes)
            # Higher values mean faster decay
            decay_factor = 0.1

            # Update all other points with exponentially decreasing influence
            for i in range(num_points):
                if i != point_index:
                    # Calculate minimum distance considering the circular nature
                    dist_clockwise = (i - point_index) % num_points
                    dist_counterclockwise = (point_index - i) % num_points
                    distance = min(dist_clockwise, dist_counterclockwise)

                    # Calculate influence using exponential decay
                    influence = math.exp(-decay_factor * distance)

                    # Update point position
                    new_points[i] += delta * influence

            # Recalculate centroid and bounding box
            self.set_precision(new_points)
            self.set_centroid()
            self.set_cropped_bbox()
            self.update_graphics_item()

            # Notify that the annotation has been updated
            self.annotationUpdated.emit(self)
            
    @classmethod
    def combine(cls, annotations: list):
        """Cbomine multiple polygon annotations into a single polygon using convex hull.
        
        Args:
            annotations: List of PolygonAnnotation objects to combine.
            
        Returns:
            A new PolygonAnnotation that encompasses all input polygons
        """
        if not annotations:
            raise ValueError("Cannot combine empty list of annotations")
        
        # Collect all points from all polygons
        all_points = []
        for annotation in annotations:
            all_points.extend(annotation.points)
        
        # Convert to numpy array for convex hull calculation
        points_array = np.array([(point.x(), point.y()) for point in all_points])
        
        # Compute convex hull
        if len(points_array) > 2:
            hull = cv2.convexHull(np.array(points_array, dtype=np.float32))
            hull_points = [QPointF(point[0][0], point[0][1]) for point in hull]
        else:
            # Not enough points for convex hull, use original points
            hull_points = all_points
            
        # Extract info from the first annotation
        short_label_code = annotations[0].short_label_code
        long_label_code = annotations[0].long_label_code
        color = annotations[0].label.color
        image_path = annotations[0].image_path
        label_id = annotations[0].label_id
        
        # Create a new annotation with the combined points
        new_annotation = cls(
            points=hull_points,
            short_label_code=short_label_code,
            long_label_code=long_label_code,
            color=color,
            image_path=image_path,
            label_id=label_id
        )
        
        # # If all input annotations have the same rasterio source, use it for the new one
        # if all(hasattr(anno, 'rasterio_src') and anno.rasterio_src is not None for anno in annotations):
        #     if len(set(id(anno.rasterio_src) for anno in annotations)) == 1:
        #         new_annotation.rasterio_src = annotations[0].rasterio_src
        #         new_annotation.create_cropped_image(new_annotation.rasterio_src)
        
        return new_annotation

    def to_dict(self):
        """Convert the annotation to a dictionary representation for serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'points': [(point.x(), point.y()) for point in self.points]
        })
        return base_dict

    @classmethod
    def from_dict(cls, data, label_window):
        """Create a PolygonAnnotation object from a dictionary representation."""
        points = [QPointF(x, y) for x, y in data['points']]
        annotation = cls(points,
                         data['label_short_code'],
                         data['label_long_code'],
                         QColor(*data['annotation_color']),
                         data['image_path'],
                         data['label_id'])
        annotation.data = data.get('data', {})

        # Convert machine_confidence keys back to Label objects
        machine_confidence = {}
        for short_label_code, confidence in data.get('machine_confidence', {}).items():
            label = label_window.get_label_by_short_code(short_label_code)
            if label:
                machine_confidence[label] = confidence

        annotation.update_machine_confidence(machine_confidence)

        return annotation

    def __repr__(self):
        """Return a string representation of the PolygonAnnotation object."""
        return (f"PolygonAnnotation(id={self.id}, points={self.points}, "
                f"annotation_color={self.label.color.name()}, "
                f"image_path={self.image_path}, "
                f"label={self.label.short_label_code}, "
                f"data={self.data}, "
                f"machine_confidence={self.machine_confidence})")
