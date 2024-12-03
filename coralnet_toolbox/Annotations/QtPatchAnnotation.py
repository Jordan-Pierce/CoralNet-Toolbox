import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QPixmap, QColor, QPen, QBrush
from PyQt5.QtWidgets import (QGraphicsScene, QGraphicsRectItem)
from rasterio.windows import Window

from coralnet_toolbox.Annotations.QtAnnotation import Annotation


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class PatchAnnotation(Annotation):
    def __init__(self,
                 center_xy: QPointF,
                 annotation_size: int,
                 short_label_code: str,
                 long_label_code: str,
                 color: QColor,
                 image_path: str,
                 label_id: str,
                 transparency: int = 128,
                 show_msg=False):
        super().__init__(short_label_code, long_label_code, color, image_path, label_id, transparency, show_msg)
        self.center_xy = QPointF(round(center_xy.x(), 2), round(center_xy.y(), 2))
        self.annotation_size = annotation_size

    def contains_point(self, point: QPointF):
        half_size = self.annotation_size / 2
        rect = QRectF(self.center_xy.x() - half_size,
                      self.center_xy.y() - half_size,
                      self.annotation_size,
                      self.annotation_size)
        return rect.contains(point)

    def create_cropped_image(self, rasterio_src):
        # Provide the rasterio source to the annotation for the first time
        self.rasterio_src = rasterio_src

        # Calculate the half size of the annotation
        half_size = self.annotation_size / 2

        # Convert center coordinates to pixel coordinates
        pixel_x = int(self.center_xy.x())
        pixel_y = int(self.center_xy.y())

        # Calculate the window for rasterio
        window = Window(
            col_off=max(0, pixel_x - half_size),
            row_off=max(0, pixel_y - half_size),
            width=min(rasterio_src.width - (pixel_x - half_size), self.annotation_size),
            height=min(rasterio_src.height - (pixel_y - half_size), self.annotation_size)
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

        half_size = self.annotation_size / 2
        cropped_rect = QRectF(self.center_xy.x() - half_size,
                              self.center_xy.y() - half_size,
                              self.annotation_size,
                              self.annotation_size)

        cropped_image_graphic = QGraphicsRectItem(cropped_rect)

        color = QColor(self.label.color)
        color.setAlpha(64)
        brush = QBrush(color)
        cropped_image_graphic.setBrush(brush)
        cropped_image_graphic.update()

        return cropped_image_graphic

    def create_graphics_item(self, scene: QGraphicsScene):
        half_size = self.annotation_size / 2
        self.graphics_item = QGraphicsRectItem(self.center_xy.x() - half_size,
                                               self.center_xy.y() - half_size,
                                               self.annotation_size,
                                               self.annotation_size)
        self.update_graphics_item()
        self.graphics_item.setData(0, self.id)
        scene.addItem(self.graphics_item)

        # Create separate graphics items for center/centroid, bounding box, and brush/mask
        self.create_center_graphics_item(self.center_xy, scene)
        self.create_bounding_box_graphics_item(QPointF(self.center_xy.x() - half_size, self.center_xy.y() - half_size),
                                               QPointF(self.center_xy.x() + half_size, self.center_xy.y() + half_size),
                                               scene)
        self.create_polygon_graphics_item([self.center_xy], scene)

    def update_graphics_item(self, crop_image=True):
        if self.graphics_item:
            # Update the graphic item
            half_size = self.annotation_size / 2
            self.graphics_item.setRect(self.center_xy.x() - half_size,
                                       self.center_xy.y() - half_size,
                                       self.annotation_size,
                                       self.annotation_size)
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
            self.graphics_item.update()

            # Update separate graphics items for center/centroid, bounding box, and brush/mask
            self.update_center_graphics_item(self.center_xy)
            self.update_bounding_box_graphics_item(QPointF(self.center_xy.x() - half_size,
                                                           self.center_xy.y() - half_size),
                                                   QPointF(self.center_xy.x() + half_size,
                                                           self.center_xy.y() + half_size))

            self.update_polygon_graphics_item([self.center_xy])

            # Update the cropped image
            if self.rasterio_src and crop_image:
                self.create_cropped_image(self.rasterio_src)

    def update_location(self, new_center_xy: QPointF):
        # Clear the machine confidence
        self.update_user_confidence(self.label)
        
        # Update the location, graphic
        self.center_xy = QPointF(round(new_center_xy.x(), 2), round(new_center_xy.y(), 2))
        self.update_graphics_item()
        self.annotationUpdated.emit(self)  # Notify update

    def update_annotation_size(self, size):
        # Clear the machine confidence
        self.update_user_confidence(self.label)
        
        # Update the size, graphic
        self.annotation_size = size
        self.update_graphics_item()
        self.annotationUpdated.emit(self)  # Notify update

    def resize(self, handle: str, new_pos: QPointF):
        pass

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            'center_xy': (self.center_xy.x(), self.center_xy.y()),
            'annotation_size': self.annotation_size,
        })
        return base_dict

    @classmethod
    def from_dict(cls, data, label_window):
        annotation = cls(QPointF(*data['center_xy']),
                         data['annotation_size'],
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
        return (f"PatchAnnotation(id={self.id}, center_xy={self.center_xy}, "
                f"annotation_size={self.annotation_size}, "
                f"annotation_color={self.label.color.name()}, "
                f"image_path={self.image_path}, "
                f"label={self.label.short_label_code}, "
                f"data={self.data}, "
                f"machine_confidence={self.machine_confidence})")