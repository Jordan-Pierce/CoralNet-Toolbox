import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import math
import numpy as np

from rasterio.windows import Window
from shapely.geometry import Point
from shapely.geometry import Polygon, LineString

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPolygonItem
from PyQt5.QtGui import (QPixmap, QColor, QPen, QBrush, QPolygonF, 
                         QPainter, QRegion, QImage)

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
            points = [QPointF(round(point.x(), 6), round(point.y(), 6)) for point in points]

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
        
    def get_centroid(self):
        """Get the centroid of the annotation."""
        return (float(self.center_xy.x()), float(self.center_xy.y()))
        
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
        """Get the cropped image with the polygon mask applied and black background."""
        if self.cropped_image is None:
            return None
            
        # Create a QImage with transparent background for the mask
        masked_image = QImage(self.cropped_image.size(), QImage.Format_ARGB32)
        masked_image.fill(Qt.transparent)  # Transparent background
        
        # Create a QPainter to draw the polygon onto the mask
        painter = QPainter(masked_image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(Qt.white))  # White fill for the mask area
        painter.setPen(Qt.NoPen)
        
        # Create a copy of the points that are transformed to be relative to the cropped_image
        cropped_points = [QPointF(point.x() - self.cropped_bbox[0],
                                  point.y() - self.cropped_bbox[1]) for point in self.points]
                                
        # Create a polygon from the cropped points
        polygon = QPolygonF(cropped_points)
        
        # Draw the polygon onto the mask
        painter.drawPolygon(polygon)
        painter.end()
        
        # Convert the mask QImage to QPixmap and create a bitmap mask
        # We want the inside of the polygon to show the image, so we DON'T use MaskInColor
        mask_pixmap = QPixmap.fromImage(masked_image)
        mask_bitmap = mask_pixmap.createMaskFromColor(Qt.white, Qt.MaskOutColor)
        
        # Convert bitmap to region for clipping
        mask_region = QRegion(mask_bitmap)
        
        # Create the result image
        cropped_image_graphic = QPixmap(self.cropped_image.size())
        
        # First draw the entire original image at 50% opacity (for area outside polygon)
        result_painter = QPainter(cropped_image_graphic)
        result_painter.setRenderHint(QPainter.Antialiasing)
        result_painter.setOpacity(0.5)  # 50% opacity for outside the polygon
        result_painter.drawPixmap(0, 0, self.cropped_image)
        
        # Then draw the full opacity image only in the masked area (inside the polygon)
        result_painter.setOpacity(1.0)  # Reset to full opacity
        result_painter.setClipRegion(mask_region)
        result_painter.drawPixmap(0, 0, self.cropped_image)
        
        # Draw the dotted line outline on top
        pen = QPen(self.label.color)
        pen.setStyle(Qt.DashLine)  # Creates a dotted/dashed line
        pen.setWidth(2)  # Line width
        result_painter.setPen(pen)
        result_painter.setClipping(False)  # Disable clipping for the outline
        result_painter.drawPolygon(polygon)
        
        result_painter.end()
        
        return cropped_image_graphic
    
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
        """Combine multiple polygon annotations into a single polygon using polygon union,
        as long as the polygons form a connected component (directly or indirectly connected).
        
        Args:
            annotations: List of PolygonAnnotation objects to combine.
            
        Returns:
            A new PolygonAnnotation that encompasses all input polygons if they form a connected component,
            or None if the polygons form disconnected groups.
        """
        if not annotations:
            return None
        
        if len(annotations) == 1:
            return annotations[0]
        
        # Build an adjacency graph where an edge represents polygon overlap
        overlap_graph = {}
        for i in range(len(annotations)):
            overlap_graph[i] = set()
        
        # Check for overlap between polygons
        for i in range(len(annotations) - 1):
            poly1_points = np.array([(p.x(), p.y()) for p in annotations[i].points], dtype=np.int32)
            
            # Create a mask for the first polygon
            poly1_bbox = annotations[i].cropped_bbox
            p1_min_x, p1_min_y = int(poly1_bbox[0]), int(poly1_bbox[1])
            p1_max_x, p1_max_y = int(poly1_bbox[2]), int(poly1_bbox[3])
            p1_width = p1_max_x - p1_min_x + 20  # Add padding
            p1_height = p1_max_y - p1_min_y + 20
            
            # Adjust polygon coordinates to mask
            poly1_adjusted = poly1_points.copy()
            poly1_adjusted[:, 0] -= p1_min_x - 10
            poly1_adjusted[:, 1] -= p1_min_y - 10
            
            mask1 = np.zeros((p1_height, p1_width), dtype=np.uint8)
            cv2.fillPoly(mask1, [poly1_adjusted], 255)
            
            for j in range(i + 1, len(annotations)):
                poly2_points = np.array([(p.x(), p.y()) for p in annotations[j].points], dtype=np.int32)
                
                # First check bounding box overlap for quick filtering
                min_x1, min_y1, max_x1, max_y1 = annotations[i].cropped_bbox
                min_x2, min_y2, max_x2, max_y2 = annotations[j].cropped_bbox
                
                has_overlap = False
                
                # Check if bounding boxes overlap
                if not (max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1):
                    # Create a mask for the second polygon in the same coordinate system as the first
                    poly2_adjusted = poly2_points.copy()
                    poly2_adjusted[:, 0] -= p1_min_x - 10
                    poly2_adjusted[:, 1] -= p1_min_y - 10
                    
                    mask2 = np.zeros_like(mask1)
                    cv2.fillPoly(mask2, [poly2_adjusted], 255)
                    
                    # Check for intersection
                    intersection = cv2.bitwise_and(mask1, mask2)
                    if np.any(intersection):
                        has_overlap = True
                
                # Fallback to point-in-polygon check if no overlap detected yet
                if not has_overlap:
                    # Check if any point of polygon i is inside polygon j
                    for point in annotations[i].points:
                        if annotations[j].contains_point(point):
                            has_overlap = True
                            break
                            
                    if not has_overlap:
                        # Check if any point of polygon j is inside polygon i
                        for point in annotations[j].points:
                            if annotations[i].contains_point(point):
                                has_overlap = True
                                break
                
                # If overlap is found, add an edge between i and j in the graph
                if has_overlap:
                    overlap_graph[i].add(j)
                    overlap_graph[j].add(i)
        
        # Check if all polygons form a connected component using BFS
        visited = [False] * len(annotations)
        queue = [0]  # Start from the first polygon
        visited[0] = True
        
        while queue:
            node = queue.pop(0)
            for neighbor in overlap_graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        
        # If any polygon is not visited, the annotations don't form a connected component
        if False in visited:
            return None
        
        # Combine polygons by creating a binary mask of all polygons
        # Determine the bounds of all polygons
        min_x = min(anno.cropped_bbox[0] for anno in annotations)
        min_y = min(anno.cropped_bbox[1] for anno in annotations)
        max_x = max(anno.cropped_bbox[2] for anno in annotations)
        max_y = max(anno.cropped_bbox[3] for anno in annotations)
        
        # Add padding
        padding = 20
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding
        
        # Create a mask for the combined shape
        width = int(max_x - min_x)
        height = int(max_y - min_y)
        if width <= 0 or height <= 0:
            width = max(1, width)
            height = max(1, height)
        
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Draw all polygons on the mask
        for annotation in annotations:
            polygon_points = np.array([(point.x() - min_x, point.y() - min_y) for point in annotation.points])
            cv2.fillPoly(combined_mask, [polygon_points.astype(np.int32)], 255)
        
        # Find contours of the combined shape
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Simplify the contour slightly to reduce point count
            epsilon = 0.0005 * cv2.arcLength(largest_contour, True)
            approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Convert back to original coordinate system and to QPointF
            hull_points = [QPointF(point[0][0] + min_x, point[0][1] + min_y) for point in approx_contour]
        else:
            # Fallback to all points if contour finding fails
            all_points = []
            for annotation in annotations:
                all_points.extend(annotation.points)
            hull_points = all_points
        
        # Extract info from the first annotation
        short_label_code = annotations[0].label.short_label_code
        long_label_code = annotations[0].label.long_label_code
        color = annotations[0].label.color
        image_path = annotations[0].image_path
        label_id = annotations[0].label.id
        
        # Create a new annotation with the combined points
        new_annotation = cls(
            points=hull_points,
            short_label_code=short_label_code,
            long_label_code=long_label_code,
            color=color,
            image_path=image_path,
            label_id=label_id
        )
        
        # All input annotations have the same rasterio source, use it for the new one
        if all(hasattr(anno, 'rasterio_src') and anno.rasterio_src is not None for anno in annotations):
            if len(set(id(anno.rasterio_src) for anno in annotations)) == 1:
                new_annotation.rasterio_src = annotations[0].rasterio_src
                new_annotation.create_cropped_image(new_annotation.rasterio_src)
        
        return new_annotation
    
    @classmethod
    def cut(cls, annotation, cutting_points: list):
        """Cut a polygon annotation where it intersects with a cutting line.
        
        Args:
            annotation: A PolygonAnnotation object.
            cutting_points: List of QPointF objects defining a cutting line.
            
        Returns:
            List of new PolygonAnnotation objects resulting from the cut.
        """
        if not annotation or not cutting_points or len(cutting_points) < 2:
            return [annotation] if annotation else []
        
        # Extract polygon points as (x,y) tuples
        polygon_points = [(point.x(), point.y()) for point in annotation.points]
        if len(polygon_points) < 3:
            return [annotation]  # Not a valid polygon
        
        # Create shapely polygon
        polygon = Polygon(polygon_points)
        
        # Create cutting line
        line_points = [(point.x(), point.y()) for point in cutting_points]
        cutting_line = LineString(line_points)
        
        # Check if the line intersects with the polygon
        if not polygon.intersects(cutting_line):
            return [annotation]  # No intersection, return original
        
        # Extend the cutting line to ensure it fully cuts through the polygon
        # This extends the cutting line by calculating its bearing and extending beyond the polygon bounds
        def extend_line(line, distance=1000):
            """Extend line in both directions by the given distance."""            
            # Get the coordinates of the first and last points
            coords = list(line.coords)
            
            # Calculate direction vectors for start and end
            if len(coords) >= 2:
                # For start point (extend backwards)
                start_x, start_y = coords[0]
                next_x, next_y = coords[1]
                start_dx = start_x - next_x
                start_dy = start_y - next_y
                
                # Normalize and scale the direction vector
                start_length = (start_dx**2 + start_dy**2)**0.5
                if start_length > 0:
                    start_dx = start_dx / start_length * distance
                    start_dy = start_dy / start_length * distance
                
                # For end point (extend forwards)
                end_x, end_y = coords[-1]
                prev_x, prev_y = coords[-2]
                end_dx = end_x - prev_x
                end_dy = end_y - prev_y
                
                # Normalize and scale the direction vector
                end_length = (end_dx**2 + end_dy**2)**0.5
                if end_length > 0:
                    end_dx = end_dx / end_length * distance
                    end_dy = end_dy / end_length * distance
                
                # Create new extended points
                new_start = (start_x + start_dx, start_y + start_dy)
                new_end = (end_x + end_dx, end_y + end_dy)
                
                # Create new extended line with all points
                new_coords = [new_start] + coords + [new_end]
                return LineString(new_coords)
            
            return line
        
        # Extend the cutting line
        extended_line = extend_line(cutting_line)
        
        # Cut the polygon with the extended line
        try:
            # Split the polygon along the cutting line
            from shapely.ops import split
            split_polygons = split(polygon, extended_line)
            
            # Convert the split geometries back to polygons
            result_annotations = []
            min_area = 10  # Minimum area threshold
            
            for geom in split_polygons.geoms:
                # Skip tiny fragments
                if geom.area < min_area or not isinstance(geom, Polygon):
                    continue
                    
                # Get the exterior coordinates of the polygon
                coords = list(geom.exterior.coords)
                
                # Convert coordinates to QPointF objects
                new_points = [QPointF(x, y) for x, y in coords[:-1]]  
                
                if len(new_points) < 3:  # Skip if we don't have enough points for a polygon
                    continue
                    
                # Create a new polygon annotation
                new_anno = cls(
                    points=new_points,
                    short_label_code=annotation.label.short_label_code,
                    long_label_code=annotation.label.long_label_code,
                    color=annotation.label.color,
                    image_path=annotation.image_path,
                    label_id=annotation.label.id
                )
                
                # Transfer rasterio source if available
                if hasattr(annotation, 'rasterio_src') and annotation.rasterio_src is not None:
                    new_anno.rasterio_src = annotation.rasterio_src
                    new_anno.create_cropped_image(new_anno.rasterio_src)
                    
                result_annotations.append(new_anno)
            
            # If no valid polygons were created, return the original
            return result_annotations if result_annotations else [annotation]
        
        except Exception as e:
            # Log the error and return the original polygon
            print(f"Error during polygon cutting: {e}")
            return [annotation]

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

        # Set the machine confidence
        annotation.update_machine_confidence(machine_confidence, from_import=True)
        
        # Override the verified attribute if it exists in the data
        if 'verified' in data:
            annotation.set_verified(data['verified'])

        return annotation

    def __repr__(self):
        """Return a string representation of the PolygonAnnotation object."""
        return (f"PolygonAnnotation(id={self.id}, points={self.points}, "
                f"annotation_color={self.label.color.name()}, "
                f"image_path={self.image_path}, "
                f"label={self.label.short_label_code}, "
                f"data={self.data}, "
                f"machine_confidence={self.machine_confidence})")
