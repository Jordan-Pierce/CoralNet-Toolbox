import warnings

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPixmap, QColor, QPen, QBrush
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsRectItem
from rasterio.windows import Window

from toolbox.Annotations.QtAnnotation import Annotation

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class RectangleAnnotation(Annotation):
    def __init__(self,
                 top_left: QPointF,
                 bottom_right: QPointF,
                 short_label_code: str,
                 long_label_code: str,
                 color: QColor,
                 image_path: str,
                 label_id: str,
                 transparency: int = 128,
                 show_msg=True):
        super().__init__(short_label_code, long_label_code, color, image_path, label_id, transparency, show_msg)
        self.top_left = QPointF(min(top_left.x(), bottom_right.x()), min(top_left.y(), bottom_right.y()))
        self.bottom_right = QPointF(max(top_left.x(), bottom_right.x()), max(top_left.y(), bottom_right.y()))
        self.center_xy = QPointF((self.top_left.x() + self.bottom_right.x()) / 2, (self.top_left.y() + self.bottom_right.y()) / 2)
        self.cropped_bbox = (self.top_left.x(), self.top_left.y(), self.bottom_right.x(), self.bottom_right.y())

    def contains_point(self, point: QPointF) -> bool:
        return (self.top_left.x() <= point.x() <= self.bottom_right.x() and
                self.top_left.y() <= point.y() <= self.bottom_right.y())

    def calculate_centroid(self):
        self.center_xy = QPointF((self.top_left.x() + self.bottom_right.x()) / 2,
                                 (self.top_left.y() + self.bottom_right.y()) / 2)

    def set_cropped_bbox(self):
        self.cropped_bbox = (self.top_left.x(), self.top_left.y(), self.bottom_right.x(), self.bottom_right.y())
        self.center_xy = QPointF((self.top_left.x() + self.bottom_right.x()) / 2,
                                 (self.top_left.y() + self.bottom_right.y()) / 2)

    def create_cropped_image(self, rasterio_src):
        # Set the rasterio source for the annotation
        self.rasterio_src = rasterio_src
        # Set the cropped bounding box for the annotation
        self.set_cropped_bbox()
        # Get the bounding box of the rectangle
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

        self.annotation_updated.emit(self)  # Notify update

    def create_graphics_item(self, scene: QGraphicsScene):
        rect = QGraphicsRectItem(self.top_left.x(), self.top_left.y(),
                                 self.bottom_right.x() - self.top_left.x(),
                                 self.bottom_right.y() - self.top_left.y())
        self.graphics_item = rect
        self.update_graphics_item()
        self.graphics_item.setData(0, self.id)
        scene.addItem(self.graphics_item)

    def update_graphics_item(self):
        if self.graphics_item:
            scene = self.graphics_item.scene()
            if scene:
                scene.removeItem(self.graphics_item)

            # Update the graphic item
            rect = QGraphicsRectItem(self.top_left.x(), self.top_left.y(),
                                     self.bottom_right.x() - self.top_left.x(),
                                     self.bottom_right.y() - self.top_left.y())
            self.graphics_item = rect
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

    def update_location(self, new_center_xy: QPointF):
        if self.machine_confidence and self.show_message:
            self.show_warning_message()
            return

        # Clear the machine confidence
        self.update_user_confidence(self.label)
        # Update the location, graphic
        delta = new_center_xy - self.center_xy
        self.top_left += delta
        self.bottom_right += delta
        self.center_xy = new_center_xy
        self.update_graphics_item()
        self.annotation_updated.emit(self)  # Notify update

    def update_annotation_size(self, scale_factor: float):
        if self.machine_confidence and self.show_message:
            self.show_warning_message()
            return

        # Clear the machine confidence
        self.update_user_confidence(self.label)
        # Update the size, graphic
        width = (self.bottom_right.x() - self.top_left.x()) * scale_factor
        height = (self.bottom_right.y() - self.top_left.y()) * scale_factor
        self.top_left = QPointF(self.center_xy.x() - width / 2, self.center_xy.y() - height / 2)
        self.bottom_right = QPointF(self.center_xy.x() + width / 2, self.center_xy.y() + height / 2)
        self.update_graphics_item()
        self.annotation_updated.emit(self)  # Notify update

    def transform_points_to_cropped_image(self):
        min_x, min_y, max_x, max_y = self.cropped_bbox
        transformed_top_left = QPointF(self.top_left.x() - min_x, self.top_left.y() - min_y)
        transformed_bottom_right = QPointF(self.bottom_right.x() - min_x, self.bottom_right.y() - min_y)
        return [transformed_top_left, transformed_bottom_right]

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            'top_left': (self.top_left.x(), self.top_left.y()),
            'bottom_right': (self.bottom_right.x(), self.bottom_right.y()),
        })
        return base_dict

    @classmethod
    def from_dict(cls, data):
        top_left = QPointF(data['top_left'][0], data['top_left'][1])
        bottom_right = QPointF(data['bottom_right'][0], data['bottom_right'][1])
        annotation = cls(top_left, bottom_right,
                         data['label_short_code'],
                         data['label_long_code'],
                         QColor(*data['annotation_color']),
                         data['image_path'],
                         data['label_id'])
        annotation.data = data.get('data', {})
        annotation.machine_confidence = data.get('machine_confidence', {})
        return annotation

    def __repr__(self):
        return (f"RectangleAnnotation(id={self.id}, "
                f"top_left={self.top_left}, bottom_right={self.bottom_right}, "
                f"annotation_color={self.label.color.name()}, "
                f"image_path={self.image_path}, "
                f"label={self.label.short_label_code}, "
                f"data={self.data}, "
                f"machine_confidence={self.machine_confidence})")