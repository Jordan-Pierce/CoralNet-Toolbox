import warnings

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPixmap, QColor, QPen, QBrush, QPolygonF
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsRectItem
from rasterio.windows import Window

from coralnet_toolbox.Annotations.QtAnnotation import Annotation

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
                 show_msg=False):
        super().__init__(short_label_code, long_label_code, color, image_path, label_id, transparency, show_msg)
        self.center_xy = QPointF(0, 0)
        self.cropped_bbox = (0, 0, 0, 0)
        self.annotation_size = 0

        self._reduce_precision(top_left, bottom_right)
        self.calculate_centroid()
        self.set_cropped_bbox()

    def _reduce_precision(self, top_left: QPointF, bottom_right: QPointF):
        self.top_left = QPointF(round(min(top_left.x(), bottom_right.x()), 2),
                                round(min(top_left.y(), bottom_right.y()), 2))

        self.bottom_right = QPointF(round(max(top_left.x(), bottom_right.x()), 2),
                                    round(max(top_left.y(), bottom_right.y()), 2))

    def calculate_centroid(self):
        self.center_xy = QPointF((self.top_left.x() + self.bottom_right.x()) / 2,
                                 (self.top_left.y() + self.bottom_right.y()) / 2)

    def set_cropped_bbox(self):
        self.cropped_bbox = (self.top_left.x(), self.top_left.y(), self.bottom_right.x(), self.bottom_right.y())
        self.annotation_size = int(max(self.bottom_right.x() - self.top_left.x(),
                                       self.bottom_right.y() - self.top_left.y()))

    def calculate_area(self):
        return (self.bottom_right.x() - self.top_left.x()) * (self.bottom_right.y() - self.top_left.y())

    def calculate_perimeter(self):
        return 2 * (self.bottom_right.x() - self.top_left.x()) + 2 * (self.bottom_right.y() - self.top_left.y())

    def contains_point(self, point: QPointF) -> bool:
        return (self.top_left.x() <= point.x() <= self.bottom_right.x() and
                self.top_left.y() <= point.y() <= self.bottom_right.y())

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

        # Create a graphic item (perimeter) for the cropped image, transform the points
        cropped_image_graphic = QGraphicsRectItem(self.top_left.x() - self.cropped_bbox[0],
                                                  self.top_left.y() - self.cropped_bbox[1],
                                                  self.bottom_right.x() - self.cropped_bbox[0],
                                                  self.bottom_right.y() - self.cropped_bbox[1])

        color = QColor(self.label.color)
        color.setAlpha(64)
        pen = QPen(color, 4, Qt.SolidLine)
        cropped_image_graphic.setPen(pen)
        cropped_image_graphic.setBrush(QBrush(color))
        cropped_image_graphic.update()

        return cropped_image_graphic

    def create_graphics_item(self, scene: QGraphicsScene):
        rect = QGraphicsRectItem(self.top_left.x(), self.top_left.y(),
                                 self.bottom_right.x() - self.top_left.x(),
                                 self.bottom_right.y() - self.top_left.y())
        self.graphics_item = rect
        self.update_graphics_item()
        self.graphics_item.setData(0, self.id)
        scene.addItem(self.graphics_item)

        # Create separate graphics items for center/centroid, bounding box, and brush/mask
        self.create_center_graphics_item(self.center_xy, scene)
        self.create_bounding_box_graphics_item(self.top_left, self.bottom_right, scene)
        self.create_polygon_graphics_item(QPolygonF([self.top_left,
                                                     QPointF(self.bottom_right.x(), self.top_left.y()),
                                                     self.bottom_right,
                                                     QPointF(self.top_left.x(), self.bottom_right.y())]), scene)

    def update_graphics_item(self, crop_image=True):
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
            self.update_bounding_box_graphics_item(self.top_left, self.bottom_right)
            self.update_polygon_graphics_item(QPolygonF([self.top_left,
                                                         QPointF(self.bottom_right.x(), self.top_left.y()),
                                                         self.bottom_right,
                                                         QPointF(self.top_left.x(), self.bottom_right.y())]))

            if self.rasterio_src and crop_image:
                self.create_cropped_image(self.rasterio_src)

    def update_location(self, new_center_xy: QPointF):
        # Clear the machine confidence
        self.update_user_confidence(self.label)
        
        # Update the location, graphic
        delta = QPointF(round(new_center_xy.x() - self.center_xy.x(), 2),
                        round(new_center_xy.y() - self.center_xy.y(), 2))
        self.top_left += delta
        self.bottom_right += delta
        self.center_xy = new_center_xy
        self.update_graphics_item()
        self.annotationUpdated.emit(self)  # Notify update

    def update_annotation_size(self, scale_factor: float):
        # Clear the machine confidence
        self.update_user_confidence(self.label)
        
        # Update the size, graphic
        width = (self.bottom_right.x() - self.top_left.x()) * scale_factor
        height = (self.bottom_right.y() - self.top_left.y()) * scale_factor
        self.top_left = QPointF(self.center_xy.x() - width / 2, self.center_xy.y() - height / 2)
        self.bottom_right = QPointF(self.center_xy.x() + width / 2, self.center_xy.y() + height / 2)
        self.update_graphics_item()
        self.annotationUpdated.emit(self)  # Notify update

    def resize(self, handle: str, new_pos: QPointF):
        # Clear the machine confidence
        self.update_user_confidence(self.label)

        # Resize the annotation
        if handle == "left":
            self.top_left.setX(new_pos.x())
        elif handle == "right":
            self.bottom_right.setX(new_pos.x())
        elif handle == "top":
            self.top_left.setY(new_pos.y())
        elif handle == "bottom":
            self.bottom_right.setY(new_pos.y())
        elif handle == "top_left":
            self.top_left = new_pos
        elif handle == "top_right":
            self.top_left.setY(new_pos.y())
            self.bottom_right.setX(new_pos.x())
        elif handle == "bottom_left":
            self.top_left.setX(new_pos.x())
            self.bottom_right.setY(new_pos.y())
        elif handle == "bottom_right":
            self.bottom_right = new_pos

        self._reduce_precision(self.top_left, self.bottom_right)
        self.calculate_centroid()
        self.update_graphics_item()
        self.annotationUpdated.emit(self)

    def to_yolo_detection(self, image_width, image_height):
        min_x, min_y, max_x, max_y = self.cropped_bbox
        x_center = (min_x + max_x) / 2 / image_width
        y_center = (min_y + max_y) / 2 / image_height
        width = (max_x - min_x) / image_width
        height = (max_y - min_y) / image_height

        return self.label.short_label_code, f"{x_center} {y_center} {width} {height}"

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            'top_left': (self.top_left.x(), self.top_left.y()),
            'bottom_right': (self.bottom_right.x(), self.bottom_right.y()),
        })
        return base_dict

    @classmethod
    def from_dict(cls, data, label_window):
        top_left = QPointF(data['top_left'][0], data['top_left'][1])
        bottom_right = QPointF(data['bottom_right'][0], data['bottom_right'][1])
        annotation = cls(top_left, bottom_right,
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
        return (f"RectangleAnnotation(id={self.id}, "
                f"top_left={self.top_left}, bottom_right={self.bottom_right}, "
                f"annotation_color={self.label.color.name()}, "
                f"image_path={self.image_path}, "
                f"label={self.label.short_label_code}, "
                f"data={self.data}, "
                f"machine_confidence={self.machine_confidence})")