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
                 show_msg=False):
        super().__init__(short_label_code, long_label_code, color, image_path, label_id, transparency, show_msg)
        self.center_xy = QPointF(0, 0)
        self.cropped_bbox = (0, 0, 0, 0)
        self.annotation_size = 0

        self._reduce_precision(points, True)
        self.calculate_centroid()
        self.set_cropped_bbox()

    def _reduce_precision(self, points: list, reduce: bool = True):
        if reduce:
            points = [QPointF(round(point.x(), 3), round(point.y(), 3)) for point in points]
        self.points = points

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

        # Convert rasterio to QImage
        q_image = rasterio_to_cropped_image(self.rasterio_src, window)
        # Convert QImage to QPixmap
        self.cropped_image = QPixmap.fromImage(q_image)

        self.annotationUpdated.emit(self)  # Notify update
    
    def get_cropped_image_graphic(self):
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
        if self.graphics_item and self.graphics_item.scene():
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
            'area': self.calculate_area(),
            'perimeter': self.calculate_perimeter()
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
