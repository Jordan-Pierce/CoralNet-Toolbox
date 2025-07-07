import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import math
import numpy as np

from rasterio.windows import Window

from shapely.ops import split
from shapely.geometry import Point, Polygon, LineString

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPolygonItem
from PyQt5.QtGui import (QPixmap, QColor, QPen, QBrush, QPolygonF,
                         QPainter, QRegion, QImage)

from coralnet_toolbox.Annotations.QtAnnotation import Annotation
from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation

from coralnet_toolbox.utilities import densify_polygon
from coralnet_toolbox.utilities import simplify_polygon
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
                 show_msg: bool = False):
        super().__init__(short_label_code, long_label_code, color, image_path, label_id, transparency, show_msg)

        self.center_xy = QPointF(0, 0)
        self.cropped_bbox = (0, 0, 0, 0)
        self.annotation_size = 0

        self.set_precision(points, True)
        self.set_centroid()
        self.set_cropped_bbox()

    def set_precision(self, points: list, reduce: bool = True):
        """
        Set the precision of the points to 3 decimal places and apply polygon simplification.

        Args:
            points: List of QPointF vertices defining the polygon
            reduce: Whether to round coordinates to 3 decimal places
        """
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
        pen = QPen(Qt.black)
        pen.setStyle(Qt.SolidLine)  # Solid line
        pen.setWidth(1)  # Line width
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

    def create_graphics_item(self, scene: QGraphicsScene):
        """Create all graphics items for the polygon annotation and add them to the scene as a group."""
        # Use a QGraphicsPolygonItem as the main graphics item
        self.graphics_item = QGraphicsPolygonItem(QPolygonF(self.points))
        # Call parent to handle group and helpers
        super().create_graphics_item(scene)

    def update_graphics_item(self):
        """Update the graphical representation of the polygon annotation."""
        # Use a QGraphicsPolygonItem as the main graphics item
        self.graphics_item = QGraphicsPolygonItem(QPolygonF(self.points))
        # Call parent to handle group and helpers
        super().update_graphics_item()
    
    def update_polygon(self, delta):
        """
        Simplify or densify the polygon based on wheel movement.
        """
        xy_points = [(p.x(), p.y()) for p in self.points]

        # Adjust tolerance based on wheel direction
        if delta < 0:
            # Simplify: increase tolerance (less detail)
            self.tolerance = min(self.tolerance + 0.05, 2.0)
            updated_coords = simplify_polygon(xy_points, self.tolerance)
        elif delta > 0:
            # Densify: decrease segment length (more detail)
            updated_coords = densify_polygon(xy_points)
        else:
            updated_coords = xy_points

        updated_coords = [QPointF(x, y) for x, y in updated_coords]
        self.set_precision(updated_coords)
        self.set_centroid()
        self.set_cropped_bbox()

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
        """
        Grow/shrink the polygon by scaling each vertex radially from the centroid.
        delta > 1: grow, 0 < delta < 1: shrink.
        The amount of change is reduced for smoother interaction.
        """
        self.update_user_confidence(self.label)

        if len(self.points) < 3:
            return

        # Calculate centroid
        centroid_x = sum(p.x() for p in self.points) / len(self.points)
        centroid_y = sum(p.y() for p in self.points) / len(self.points)

        # Determine scale factor: small step for each call
        # If delta > 1, grow; if delta < 1, shrink; if delta == 1, no change
        step = 0.01  # You can adjust this value for finer or coarser changes
        if delta > 1.0:
            scale = 1.0 + step
        elif delta < 1.0:
            scale = 1.0 - step
        else:
            scale = 1.0

        # Move each point radially using the scale factor
        new_points = []
        for p in self.points:
            dx = p.x() - centroid_x
            dy = p.y() - centroid_y
            new_x = centroid_x + dx * scale
            new_y = centroid_y + dy * scale
            new_points.append(QPointF(new_x, new_y))

        self.set_precision(new_points)
        self.set_centroid()
        self.set_cropped_bbox()
        self.update_graphics_item()
        self.annotationUpdated.emit(self)

    def resize(self, handle, new_pos):
        """Resize the annotation by moving a specific handle (vertex) to a new position."""
        # Clear the machine confidence
        self.update_user_confidence(self.label)

        # Extract the point index from the handle string (e.g., "point_0" -> 0)
        if handle.startswith("point_"):
            new_points = self.points.copy()
            point_index = int(handle.split("_")[1])

            # Move only the selected point
            new_points[point_index] = new_pos

            # Recalculate centroid and bounding box
            self.set_precision(new_points)
            self.set_centroid()
            self.set_cropped_bbox()
            self.update_graphics_item()

            # Notify that the annotation has been updated
            self.annotationUpdated.emit(self)

    @classmethod
    def combine(cls, annotations: list):
        """Combine annotations. Returns PolygonAnnotation (merged) or MultiPolygonAnnotation (disjoint)."""
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

        # Check if there are any overlaps at all
        has_any_overlap = any(len(neighbors) > 0 for neighbors in overlap_graph.values())
        
        if not has_any_overlap:
            # No intersections at all - create MultiPolygonAnnotation
            polygons = [
                cls(
                    points=anno.points,
                    short_label_code=anno.label.short_label_code,
                    long_label_code=anno.label.long_label_code,
                    color=anno.label.color,
                    image_path=anno.image_path,
                    label_id=anno.label.id
                ) for anno in annotations
            ]
            new_anno = MultiPolygonAnnotation(
                polygons=polygons,
                short_label_code=annotations[0].label.short_label_code,
                long_label_code=annotations[0].label.long_label_code,
                color=annotations[0].label.color,
                image_path=annotations[0].image_path,
                label_id=annotations[0].label.id
            )
            # Transfer rasterio source if applicable
            if all(hasattr(anno, 'rasterio_src') and anno.rasterio_src is not None for anno in annotations):
                if len(set(id(anno.rasterio_src) for anno in annotations)) == 1:
                    new_anno.rasterio_src = annotations[0].rasterio_src
                    new_anno.create_cropped_image(new_anno.rasterio_src)
            return new_anno
        
        # Check if all polygons form a single connected component
        visited = [False] * len(annotations)
        stack = [0]  # Start from the first polygon
        visited[0] = True
        visited_count = 1

        while stack:
            node = stack.pop()
            for neighbor in overlap_graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    visited_count += 1
                    stack.append(neighbor)

        # If not all polygons are reachable, we have multiple disconnected components
        if visited_count != len(annotations):
            # Multiple disconnected components - return early doing nothing
            return None

        # All polygons form a single connected component - merge them
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

        # Create cutting line (do NOT extend)
        line_points = [(point.x(), point.y()) for point in cutting_points]
        cutting_line = LineString(line_points)

        # Check if the line intersects with the polygon
        if not polygon.intersects(cutting_line):
            return [annotation]  # No intersection, return original

        try:
            # Split the polygon along the cutting line (no extension)
            split_polygons = split(polygon, cutting_line)

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
