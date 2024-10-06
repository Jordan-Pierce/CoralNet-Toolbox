import os
import warnings

import numpy as np
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QPixmap, QColor, QPen, QBrush
from PyQt5.QtWidgets import (QGraphicsScene, QGraphicsRectItem)
from rasterio.windows import Window

from toolbox.Annotations.QtAnnotation import Annotation

warnings.filterwarnings("ignore", category=DeprecationWarning)


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
                 show_msg=True):
        super().__init__(short_label_code, long_label_code, color, image_path, label_id, transparency, show_msg)
        self.center_xy = QPointF(round(center_xy.x(), 2), round(center_xy.y(), 2))  # Reduce to two significant digits
        self.annotation_size = annotation_size

    def contains_point(self, point: QPointF):
        half_size = self.annotation_size / 2
        rect = QRectF(self.center_xy.x() - half_size,
                      self.center_xy.y() - half_size,
                      self.annotation_size,
                      self.annotation_size)
        return rect.contains(point)

    def create_cropped_image(self, rasterio_src, downscale_factor=1.0):
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
        self.create_brush_graphics_item(self.graphics_item.rect(), scene)

    def update_graphics_item(self):
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
                pen = QPen(inverse_color, 4, Qt.DotLine)  # Inverse color, thicker border, and dotted line
            else:
                pen = QPen(color, 2, Qt.SolidLine)  # Default border color and thickness

            self.graphics_item.setPen(pen)
            brush = QBrush(color)
            self.graphics_item.setBrush(brush)
            self.graphics_item.update()
            # Update the cropped image
            if self.rasterio_src:
                self.create_cropped_image(self.rasterio_src)

            # Update separate graphics items for center/centroid, bounding box, and brush/mask
            self.update_center_graphics_item(self.center_xy)
            self.update_bounding_box_graphics_item(QPointF(self.center_xy.x() - half_size, self.center_xy.y() - half_size),
                                                   QPointF(self.center_xy.x() + half_size, self.center_xy.y() + half_size))
            self.update_brush_graphics_item(self.graphics_item.rect())

    def update_location(self, new_center_xy: QPointF):
        if self.machine_confidence and self.show_message:
            self.show_warning_message()
            return

        # Clear the machine confidence
        self.update_user_confidence(self.label)
        # Update the location, graphic
        self.center_xy = QPointF(round(new_center_xy.x(), 2), round(new_center_xy.y(), 2))  # Reduce to two significant digits
        self.update_graphics_item()
        self.annotation_updated.emit(self)  # Notify update

    def update_annotation_size(self, size):
        if self.machine_confidence and self.show_message:
            self.show_warning_message()
            return

        # Clear the machine confidence
        self.update_user_confidence(self.label)
        # Update the size, graphic
        self.annotation_size = size
        self.update_graphics_item()
        self.annotation_updated.emit(self)  # Notify update

    def to_coralnet_format(self):
        # Extract machine confidence values and suggestions
        confidences = [f"{confidence:.3f}" for confidence in self.machine_confidence.values()]
        suggestions = [suggestion.short_label_code for suggestion in self.machine_confidence.keys()]

        # Pad with NaN if there are fewer than 5 values
        while len(confidences) < 5:
            confidences.append(np.nan)
        while len(suggestions) < 5:
            suggestions.append(np.nan)

        return {
            'Name': os.path.basename(self.image_path),
            'Row': int(self.center_xy.y()),
            'Column': int(self.center_xy.x()),
            'Label': self.label.short_label_code,
            'Long Label': self.label.long_label_code,
            'Patch Size': self.annotation_size,
            'Machine confidence 1': confidences[0],
            'Machine suggestion 1': suggestions[0],
            'Machine confidence 2': confidences[1],
            'Machine suggestion 2': suggestions[1],
            'Machine confidence 3': confidences[2],
            'Machine suggestion 3': suggestions[2],
            'Machine confidence 4': confidences[3],
            'Machine suggestion 4': suggestions[3],
            'Machine confidence 5': confidences[4],
            'Machine suggestion 5': suggestions[4],
            **self.data
        }

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
        annotation.machine_confidence = machine_confidence

        return annotation

    def __repr__(self):
        return (f"PatchAnnotation(id={self.id}, center_xy={self.center_xy}, "
                f"annotation_size={self.annotation_size}, "
                f"annotation_color={self.label.color.name()}, "
                f"image_path={self.image_path}, "
                f"label={self.label.short_label_code}, "
                f"data={self.data}, "
                f"machine_confidence={self.machine_confidence})")