import warnings

import math
import numpy as np

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPixmap, QColor, QPen, QBrush, QPolygonF
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPolygonItem
from rasterio.windows import Window

from coralnet_toolbox.Annotations.QtAnnotation import Annotation

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def resample_polygon_points(points: list, target_num_points: int = None, target_spacing: float = None) -> list:
    """
    Resample points along a polygon to achieve more uniform spacing.

    Args:
        points: List of QPointF points defining the polygon
        target_num_points: Desired number of points after resampling (optional)
        target_spacing: Desired spacing between points in pixels (optional)

    Returns:
        List of resampled QPointF points
    """
    if len(points) < 3:
        return points.copy()

    # Convert points to numpy array for easier computation
    points_array = np.array([(p.x(), p.y()) for p in points], dtype=np.float64)

    # Calculate segments and their lengths
    segments = np.diff(points_array, axis=0)
    segment_lengths = np.sqrt(np.sum(segments ** 2, axis=1))

    # Find and remove duplicate/very close points
    EPSILON = 1e-6
    valid_segments = segment_lengths > EPSILON

    if not np.any(valid_segments):
        return points.copy()

    # Filter out zero-length segments
    filtered_points = [points_array[0]]
    for i, length in enumerate(segment_lengths):
        if length > EPSILON:
            filtered_points.append(points_array[i + 1])
    points_array = np.array(filtered_points)

    # Recalculate segments after filtering
    segments = np.diff(points_array, axis=0)
    segment_lengths = np.sqrt(np.sum(segments ** 2, axis=1))

    # Calculate total length including closing segment
    total_length = np.sum(segment_lengths)
    last_segment = points_array[0] - points_array[-1]
    last_segment_length = np.sqrt(np.sum(last_segment ** 2))

    if last_segment_length > EPSILON:
        total_length += last_segment_length

    # Determine number of points to sample
    if target_spacing is not None:
        target_spacing = max(target_spacing, EPSILON)
        num_points = max(3, int(np.ceil(total_length / target_spacing)))
    elif target_num_points is not None:
        num_points = max(3, int(target_num_points))
    else:
        num_points = len(filtered_points)

    # Create cumulative length array
    cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))

    # Generate evenly spaced points along the perimeter
    spaces = np.linspace(0, total_length * (1 - EPSILON), num_points, endpoint=False)
    resampled_points = []

    for space in spaces:
        try:
            # Find which segment this point belongs to
            if space <= cumulative_lengths[-1]:
                segment_idx = np.searchsorted(cumulative_lengths, space) - 1
                segment_idx = max(0, min(segment_idx, len(points_array) - 2))
                start_point = points_array[segment_idx]
                end_point = points_array[segment_idx + 1]
                segment_length = segment_lengths[segment_idx]
                segment_space = space - cumulative_lengths[segment_idx]
            else:
                # Handle the closing segment
                start_point = points_array[-1]
                end_point = points_array[0]
                segment_length = last_segment_length
                segment_space = space - cumulative_lengths[-1]

            # Skip zero-length segments
            if segment_length <= EPSILON:
                continue

            # Calculate interpolation factor with bounds checking
            factor = np.clip(segment_space / segment_length, 0, 1)

            # Interpolate the point
            new_point = start_point + factor * (end_point - start_point)

            # Ensure point coordinates are finite
            if np.all(np.isfinite(new_point)):
                resampled_points.append(QPointF(float(new_point[0]), float(new_point[1])))

        except Exception as e:
            continue

    # Ensure we have enough points
    if len(resampled_points) < 3:
        return points.copy()

    return resampled_points

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
                 show_msg=False):
        super().__init__(short_label_code, long_label_code, color, image_path, label_id, transparency, show_msg)
        self.center_xy = QPointF(0, 0)
        self.cropped_bbox = (0, 0, 0, 0)
        self.annotation_size = 0

        self._reduce_precision(points)
        self.calculate_centroid()
        self.set_cropped_bbox()

    def _reduce_precision(self, points: list):
        self.points = [QPointF(round(point.x(), 2), round(point.y(), 2)) for point in points]

    def calculate_centroid(self):
        centroid_x = sum(point.x() for point in self.points) / len(self.points)
        centroid_y = sum(point.y() for point in self.points) / len(self.points)
        self.center_xy = QPointF(centroid_x, centroid_y)

    def set_cropped_bbox(self):
        min_x = min(point.x() for point in self.points)
        min_y = min(point.y() for point in self.points)
        max_x = max(point.x() for point in self.points)
        max_y = max(point.y() for point in self.points)
        self.cropped_bbox = (min_x, min_y, max_x, max_y)
        self.annotation_size = int(max(max_x - min_x, max_y - min_y))

    def calculate_area(self):
        n = len(self.points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.points[i].x() * self.points[j].y()
            area -= self.points[j].x() * self.points[i].y()
        area = abs(area) / 2.0
        return area

    def calculate_perimeter(self):
        n = len(self.points)
        perimeter = 0.0
        for i in range(n):
            j = (i + 1) % n
            perimeter += ((self.points[j].x() - self.points[i].x()) ** 2 +
                          (self.points[j].y() - self.points[i].y()) ** 2) ** 0.5
        return perimeter

    def contains_point(self, point: QPointF) -> bool:
        polygon = QPolygonF(self.points)
        return polygon.containsPoint(point, Qt.OddEvenFill)

    def create_cropped_image(self, rasterio_src):
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

        # Read the data from rasterio
        data = rasterio_src.read(window=window)

        # Ensure the data is in the correct format for QImage
        data = self._prepare_data_for_qimage(data)

        # Convert numpy array to QImage
        q_image = self._convert_to_qimage(data)

        # Convert QImage to QPixmap
        self.cropped_image = QPixmap.fromImage(q_image)

        self.annotationUpdated.emit(self)  # Notify update

    def get_cropped_image(self, downscaling_factor=1.0):
        if self.cropped_image is None:
            return None

        # Downscale the cropped image if downscaling_factor is not 1.0
        if downscaling_factor != 1.0:
            new_size = (int(self.cropped_image.width() * downscaling_factor),
                        int(self.cropped_image.height() * downscaling_factor))
            self.cropped_image = self.cropped_image.scaled(new_size[0], new_size[1])

        return self.cropped_image

    def get_cropped_image_graphic(self):
        if self.cropped_image is None:
            return None

        # Create a copy of the points that are transformed to be relative to the cropped_image
        cropped_points = [QPointF(point.x() - self.cropped_bbox[0],
                                  point.y() - self.cropped_bbox[1]) for point in self.points]

        cropped_polygon = QPolygonF(cropped_points)
        cropped_image_graphic = QGraphicsPolygonItem(cropped_polygon)

        color = QColor(self.label.color)
        color.setAlpha(64)
        brush = QBrush(color)
        cropped_image_graphic.setBrush(brush)
        cropped_image_graphic.update()

        return cropped_image_graphic

    def create_graphics_item(self, scene: QGraphicsScene):
        polygon = QPolygonF(self.points)
        self.graphics_item = QGraphicsPolygonItem(polygon)
        self.update_graphics_item()
        self.graphics_item.setData(0, self.id)
        scene.addItem(self.graphics_item)

        # Create separate graphics items for center/centroid, bounding box, and brush/mask
        self.create_center_graphics_item(self.center_xy, scene)
        self.create_bounding_box_graphics_item(QPointF(self.cropped_bbox[0], self.cropped_bbox[1]),
                                               QPointF(self.cropped_bbox[2], self.cropped_bbox[3]),
                                               scene)
        self.create_polygon_graphics_item(self.points, scene)

    def update_graphics_item(self, crop_image=True):
        if self.graphics_item:
            scene = self.graphics_item.scene()
            if scene:
                scene.removeItem(self.graphics_item)

            # Update the graphic item
            polygon = QPolygonF(self.points)
            self.graphics_item = QGraphicsPolygonItem(polygon)
            color = QColor(self.label.color)
            color.setAlpha(self.transparency)

            if self.is_selected:
                inverse_color = QColor(255 - color.red(), 255 - color.green(), 255 - color.blue())
                pen = QPen(inverse_color, 6, Qt.DotLine)
            else:
                pen = QPen(color, 4, Qt.SolidLine)

            self.graphics_item.setPen(pen)
            brush = QBrush(color)
            self.graphics_item.setBrush(brush)

            if scene:
                scene.addItem(self.graphics_item)

            self.graphics_item.setData(0, self.id)
            self.graphics_item.update()

            # Update separate graphics items for center/centroid, bounding box, and brush/mask
            self.update_center_graphics_item(self.center_xy)
            self.update_bounding_box_graphics_item(QPointF(self.cropped_bbox[0], self.cropped_bbox[1]),
                                                   QPointF(self.cropped_bbox[2], self.cropped_bbox[3]))
            self.update_polygon_graphics_item(self.points)

            if self.rasterio_src and crop_image:
                self.create_cropped_image(self.rasterio_src)

    def update_location(self, new_center_xy: QPointF):
        # Clear the machine confidence
        self.update_user_confidence(self.label)
        
        # Update the location, graphic
        delta = QPointF(round(new_center_xy.x() - self.center_xy.x(), 2),
                        round(new_center_xy.y() - self.center_xy.y(), 2))

        new_points = [point + delta for point in self.points]

        self._reduce_precision(new_points)
        self.calculate_centroid()
        self.set_cropped_bbox()
        self.update_graphics_item()
        self.annotationUpdated.emit(self)  # Notify update

    def update_annotation_size(self, delta: float):
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
        self._reduce_precision(new_points)
        self.calculate_centroid()
        self.set_cropped_bbox()
        self.update_graphics_item()
        self.annotationUpdated.emit(self)  # Notify update

    def resize(self, handle, new_pos):
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
            self._reduce_precision(new_points)
            self.calculate_centroid()
            self.set_cropped_bbox()
            self.update_graphics_item()

            # Notify that the annotation has been updated
            self.annotationUpdated.emit(self)

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            'points': [(point.x(), point.y()) for point in self.points],
        })
        return base_dict

    def to_yolo_detection(self, image_width, image_height):
        x_min, y_min, x_max, y_max = self.cropped_bbox
        x_center = (x_min + x_max) / 2 / image_width
        y_center = (y_min + y_max) / 2 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        return self.label.short_label_code, f"{x_center} {y_center} {width} {height}"

    def to_yolo_segmentation(self, image_width, image_height):
        normalized_points = [(point.x() / image_width, point.y() / image_height) for point in self.points]
        return self.label.short_label_code, " ".join([f"{x} {y}" for x, y in normalized_points])

    @classmethod
    def from_dict(cls, data, label_window):
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
        return (f"PolygonAnnotation(id={self.id}, points={self.points}, "
                f"annotation_color={self.label.color.name()}, "
                f"image_path={self.image_path}, "
                f"label={self.label.short_label_code}, "
                f"data={self.data}, "
                f"machine_confidence={self.machine_confidence})")