import warnings

import numpy as np
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QImage, QPixmap, QColor, QPen, QBrush, QPolygonF
from PyQt5.QtWidgets import (QGraphicsScene, QGraphicsRectItem, QGraphicsPolygonItem)
from rasterio.windows import Window

from toolbox.QtAnnotation import Annotation

warnings.filterwarnings("ignore", category=DeprecationWarning)


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
                 show_msg=True):
        super().__init__(short_label_code, long_label_code, color, image_path, label_id, transparency, show_msg)
        self.points = points
        self.center_xy = QPointF(0, 0)
        self.cropped_bbox = (0, 0, 0, 0)

        self.calculate_centroid()
        self.set_cropped_bbox()

    def contains_point(self, point: QPointF) -> bool:
        polygon = QPolygonF(self.points)
        return polygon.containsPoint(point, Qt.OddEvenFill)

    def calculate_centroid(self):
        centroid_x = sum(point.x() for point in self.points) / len(self.points)
        centroid_y = sum(point.y() for point in self.points) / len(self.points)
        self.center_xy = QPointF(centroid_x, centroid_y)

    def update_location(self, new_center_xy: QPointF):
        if self.machine_confidence and self.show_message:
            self.show_warning_message()
            return

        # Clear the machine confidence
        self.update_user_confidence(self.label)
        # Update the location, graphic
        delta = new_center_xy - self.center_xy
        self.points = [point + delta for point in self.points]
        self.calculate_centroid()
        self.update_graphics_item()
        self.annotation_updated.emit(self)  # Notify update

    def update_annotation_size(self, scale_factor: float):
        if self.machine_confidence and self.show_message:
            self.show_warning_message()
            return

        # Clear the machine confidence
        self.update_user_confidence(self.label)
        # Update the location, graphic
        centroid_x, centroid_y = self.center_xy.x(), self.center_xy.y()
        translated_points = [QPointF(point.x() - centroid_x, point.y() - centroid_y) for point in self.points]
        scaled_points = [QPointF(point.x() * scale_factor, point.y() * scale_factor) for point in translated_points]
        self.points = [QPointF(point.x() + centroid_x, point.y() + centroid_y) for point in scaled_points]
        self.calculate_centroid()
        self.update_graphics_item()
        self.annotation_updated.emit(self)  # Notify update

    def create_graphics_item(self, scene: QGraphicsScene):
        polygon = QPolygonF(self.points)
        self.graphics_item = QGraphicsPolygonItem(polygon)
        self.update_graphics_item()
        self.graphics_item.setData(0, self.id)
        scene.addItem(self.graphics_item)

    def update_graphics_item(self):
        if self.graphics_item:
            # Create a new polygon item
            polygon = QPolygonF(self.points)
            self.graphics_item.setPolygon(polygon)
            color = QColor(self.label.color)
            color.setAlpha(self.transparency)

            if self.is_selected:
                inverse_color = QColor(255 - color.red(), 255 - color.green(), 255 - color.blue())
                pen = QPen(inverse_color, 4, Qt.DotLine)  # Inverse color, thicker border, and dotted line
            else:
                pen = QPen(color, 2, Qt.SolidLine)  # Default border color and thickness

            self.graphics_item.setPen(pen)
            brush = QBrush(color)
            self.graphics_item.setBrush(brush)

            # Update the vertex items
            for point in self.points:
                vertex_item = QGraphicsRectItem(point.x() - 2, point.y() - 2, 4, 4, parent=self.graphics_item)
                vertex_color = QColor(self.label.color)
                vertex_color.setAlpha(self.transparency)
                vertex_item.setBrush(QBrush(vertex_color))
                vertex_item.setPen(QPen(vertex_color))

            self.graphics_item.update()

            # Update the cropped image
            if self.rasterio_src:
                self.create_cropped_image(self.rasterio_src)

    def set_cropped_bbox(self):
        min_x = min(point.x() for point in self.points)
        min_y = min(point.y() for point in self.points)
        max_x = max(point.x() for point in self.points)
        max_y = max(point.y() for point in self.points)
        self.cropped_bbox = (min_x, min_y, max_x, max_y)
        self.center_xy = QPointF((min_x + max_x) / 2, (min_y + max_y) / 2)

    def transform_points_to_cropped_image(self):
        # Get the bounding box of the cropped image in xyxy format
        min_x, min_y, max_x, max_y = self.cropped_bbox

        # Transform the points
        transformed_points = []
        for point in self.points:
            transformed_point = QPointF(point.x() - min_x, point.y() - min_y)
            transformed_points.append(transformed_point)

        return transformed_points

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
        if data.shape[0] == 3:  # RGB image
            data = np.transpose(data, (1, 2, 0))
        elif data.shape[0] == 1:  # Grayscale image
            data = np.squeeze(data)

        # Normalize data to 0-255 range if it's not already
        if data.dtype != np.uint8:
            data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)

        # Convert numpy array to QImage
        height, width = data.shape[:2]
        bytes_per_line = 3 * width if len(data.shape) == 3 else width
        image_format = QImage.Format_RGB888 if len(data.shape) == 3 else QImage.Format_Grayscale8

        # Convert numpy array to bytes
        if len(data.shape) == 3:
            data = data.tobytes()
        else:
            data = np.expand_dims(data, -1).tobytes()

        q_image = QImage(data, width, height, bytes_per_line, image_format)

        # Convert QImage to QPixmap
        self.cropped_image = QPixmap.fromImage(q_image)

        self.annotation_updated.emit(self)  # Notify update

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            'points': [(point.x(), point.y()) for point in self.points],
        })
        return base_dict

    @classmethod
    def from_dict(cls, data):
        points = [QPointF(x, y) for x, y in data['points']]
        annotation = cls(points,
                         data['label_short_code'],
                         data['label_long_code'],
                         QColor(*data['annotation_color']),
                         data['image_path'],
                         data['label_id'])
        annotation.data = data.get('data', {})
        annotation.machine_confidence = data.get('machine_confidence', {})
        return annotation

    def __repr__(self):
        return (f"PolygonAnnotation(id={self.id}, points={self.points}, "
                f"annotation_color={self.label.color.name()}, "
                f"image_path={self.image_path}, "
                f"label={self.label.short_label_code}, "
                f"data={self.data}, "
                f"machine_confidence={self.machine_confidence})")
