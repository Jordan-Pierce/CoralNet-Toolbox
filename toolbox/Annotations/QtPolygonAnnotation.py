import warnings

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPixmap, QColor, QPen, QBrush, QPolygonF
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPolygonItem
from rasterio.windows import Window

from toolbox.Annotations.QtAnnotation import Annotation

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
        self.points = self._reduce_precision(points)
        self.center_xy = QPointF(0, 0)
        self.cropped_bbox = (0, 0, 0, 0)

        self.calculate_centroid()
        self.set_cropped_bbox()

    def _reduce_precision(self, points: list) -> list:
        return [QPointF(round(point.x(), 2), round(point.y(), 2)) for point in points]

    def contains_point(self, point: QPointF) -> bool:
        polygon = QPolygonF(self.points)
        return polygon.containsPoint(point, Qt.OddEvenFill)

    def calculate_centroid(self):
        centroid_x = sum(point.x() for point in self.points) / len(self.points)
        centroid_y = sum(point.y() for point in self.points) / len(self.points)
        self.center_xy = QPointF(centroid_x, centroid_y)

    def calculate_polygon_area(self):
        n = len(self.points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.points[i].x() * self.points[j].y()
            area -= self.points[j].x() * self.points[i].y()
        area = abs(area) / 2.0
        return area

    def calculate_polygon_perimeter(self):
        n = len(self.points)
        perimeter = 0.0
        for i in range(n):
            j = (i + 1) % n
            perimeter += ((self.points[j].x() - self.points[i].x()) ** 2 +
                          (self.points[j].y() - self.points[i].y()) ** 2) ** 0.5
        return perimeter

    def set_cropped_bbox(self):
        min_x = min(point.x() for point in self.points)
        min_y = min(point.y() for point in self.points)
        max_x = max(point.x() for point in self.points)
        max_y = max(point.y() for point in self.points)
        self.cropped_bbox = (min_x, min_y, max_x, max_y)
        self.center_xy = QPointF((min_x + max_x) / 2, (min_y + max_y) / 2)

    def create_cropped_image(self, rasterio_src, downscale_factor=1.0):
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

        # Downscale the data if downscale_factor is not 1.0
        if downscale_factor != 1.0:
            new_size = (int(data.shape[1] * downscale_factor), int(data.shape[0] * downscale_factor))
            data = np.array(Image.fromarray(data).resize(new_size, Image.ANTIALIAS))

        # Convert numpy array to QImage
        q_image = self._convert_to_qimage(data)

        # Convert QImage to QPixmap
        self.cropped_image = QPixmap.fromImage(q_image)

        self.annotation_updated.emit(self)  # Notify update

    def get_cropped_image(self, downscaling_factor=1.0):
        if self.cropped_image is None:
            return None

        # Downscale the cropped image if downscaling_factor is not 1.0
        if downscaling_factor != 1.0:
            new_size = (int(self.cropped_image.width() * downscaling_factor),
                        int(self.cropped_image.height() * downscaling_factor))
            self.cropped_image = self.cropped_image.scaled(new_size[0], new_size[1])

        return self.cropped_image

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
        self.create_brush_graphics_item(QPolygonF(self.points), scene)

    def update_graphics_item(self):
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
                pen = QPen(inverse_color, 4, Qt.DotLine)
            else:
                pen = QPen(color, 2, Qt.SolidLine)

            self.graphics_item.setPen(pen)
            brush = QBrush(color)
            self.graphics_item.setBrush(brush)

            if scene:
                scene.addItem(self.graphics_item)

            self.graphics_item.setData(0, self.id)
            self.graphics_item.update()

            if self.rasterio_src:
                self.create_cropped_image(self.rasterio_src)

            # Update separate graphics items for center/centroid, bounding box, and brush/mask
            self.update_center_graphics_item(self.center_xy)
            self.update_bounding_box_graphics_item(QPointF(self.cropped_bbox[0], self.cropped_bbox[1]),
                                                   QPointF(self.cropped_bbox[2], self.cropped_bbox[3]))
            self.update_brush_graphics_item(QPolygonF(self.points))

    def update_location(self, new_center_xy: QPointF):
        if self.machine_confidence and self.show_message:
            self.show_warning_message()
            return

        # Clear the machine confidence
        self.update_user_confidence(self.label)
        # Update the location, graphic
        delta = QPointF(round(new_center_xy.x() - self.center_xy.x(), 2), round(new_center_xy.y() - self.center_xy.y(), 2))
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

    def transform_points_to_cropped_image(self):
        # Get the bounding box of the cropped image in xyxy format
        min_x, min_y, max_x, max_y = self.cropped_bbox

        # Transform the points
        transformed_points = []
        for point in self.points:
            transformed_point = QPointF(point.x() - min_x, point.y() - min_y)
            transformed_points.append(transformed_point)

        return transformed_points

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            'points': [(point.x(), point.y()) for point in self.points],
        })
        return base_dict

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
        annotation.machine_confidence = machine_confidence

        return annotation

    def __repr__(self):
        return (f"PolygonAnnotation(id={self.id}, points={self.points}, "
                f"annotation_color={self.label.color.name()}, "
                f"image_path={self.image_path}, "
                f"label={self.label.short_label_code}, "
                f"data={self.data}, "
                f"machine_confidence={self.machine_confidence})")