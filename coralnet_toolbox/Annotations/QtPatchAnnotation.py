import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from rasterio.windows import Window

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QPixmap, QColor, QPen, QBrush, QPainter, QPolygonF
from PyQt5.QtWidgets import (QGraphicsScene, QGraphicsRectItem)

from coralnet_toolbox.Annotations.QtAnnotation import Annotation

from coralnet_toolbox.utilities import rasterio_to_cropped_image


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

        self.center_xy = QPointF(0, 0)
        self.cropped_bbox = (0, 0, 0, 0)
        self.annotation_size = annotation_size

        self.set_precision(center_xy, False)
        self.set_centroid()
        self.set_cropped_bbox()

    def set_precision(self, center_xy: QPointF, reduce: bool = True):
        """Reduce precision of the center coordinates to avoid floating point issues."""
        if reduce:
            self.center_xy = QPointF(round(center_xy.x(), 2), round(center_xy.y(), 2))
        else:
            self.center_xy = center_xy

    def set_cropped_bbox(self):
        """Set the cropped bounding box coordinates based on center and size."""
        half_size = self.annotation_size / 2
        min_x = self.center_xy.x() - half_size
        min_y = self.center_xy.y() - half_size
        max_x = self.center_xy.x() + half_size
        max_y = self.center_xy.y() + half_size
        self.cropped_bbox = (min_x, min_y, max_x, max_y)

    def set_centroid(self):
        """Calculate the centroid of the annotation (for patch, this is the center_xy)."""
        self.center_xy = self.center_xy

    def get_area(self):
        """Calculate the area of the square patch."""
        return self.annotation_size * self.annotation_size

    def get_perimeter(self):
        """Calculate the perimeter of the square patch."""
        return 4 * self.annotation_size

    def contains_point(self, point: QPointF):
        """Check if the point is within the annotation's bounding box."""
        half_size = self.annotation_size / 2
        rect = QRectF(self.center_xy.x() - half_size,
                      self.center_xy.y() - half_size,
                      self.annotation_size,
                      self.annotation_size)
        return rect.contains(point)
    
    def get_polygon(self):
        """Get the polygon representation of this patch (a square)."""
        half_size = self.annotation_size / 2
        points = [
            QPointF(self.center_xy.x() - half_size, self.center_xy.y() - half_size),  # Top-left
            QPointF(self.center_xy.x() + half_size, self.center_xy.y() - half_size),  # Top-right
            QPointF(self.center_xy.x() + half_size, self.center_xy.y() + half_size),  # Bottom-right
            QPointF(self.center_xy.x() - half_size, self.center_xy.y() + half_size),  # Bottom-left
        ]
        return QPolygonF(points)

    def get_bounding_box_top_left(self):
        """Get the top-left corner of the bounding box."""
        half_size = self.annotation_size / 2
        return QPointF(self.center_xy.x() - half_size, self.center_xy.y() - half_size)

    def get_bounding_box_bottom_right(self):
        """Get the bottom-right corner of the bounding box."""
        half_size = self.annotation_size / 2
        return QPointF(self.center_xy.x() + half_size, self.center_xy.y() + half_size)

    def get_cropped_image_graphic(self):
        """Get the cropped image with a dotted outline."""
        if self.cropped_image is None:
            return None

        # Create a QImage with alpha channel for masking
        masked_image = QPixmap(self.cropped_image.size()).toImage()
        masked_image.fill(Qt.transparent)

        # Create a QPainter to draw the polygon onto the image
        painter = QPainter(masked_image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor(0, 0, 0, 255)))  # Opaque black
        painter.setPen(Qt.NoPen)

        # Define the square's corners as a polygon
        cropped_points = [
            QPointF(0, 0),
            QPointF(self.cropped_image.width(), 0),
            QPointF(self.cropped_image.width(), self.cropped_image.height()),
            QPointF(0, self.cropped_image.height())
        ]

        # Draw the polygon onto the image
        polygon = QPolygonF(cropped_points)
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

        # Draw the square outline with the dotted pen
        painter.drawPolygon(polygon)

        painter.end()

        return cropped_image_graphic

    def create_cropped_image(self, rasterio_src):
        """Create a cropped image from the rasterio source based on the annotation's bounding box."""
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

        # Convert rasterio to QImage
        q_image = rasterio_to_cropped_image(self.rasterio_src, window)
        # Convert QImage to QPixmap
        self.cropped_image = QPixmap.fromImage(q_image)

        self.annotationUpdated.emit(self)  # Notify update

    def update_graphics_item(self, crop_image=True):
        """Update the graphical representation of the annotation."""
        super().update_graphics_item(crop_image)

        # Update the cropped image if necessary
        if self.rasterio_src and crop_image:
            self.create_cropped_image(self.rasterio_src)

    def update_location(self, new_center_xy: QPointF):
        """Update the location of the annotation."""
        # Clear the machine confidence
        self.update_user_confidence(self.label)

        # Update the location using the set_precision method
        self.set_precision(new_center_xy)
        self.set_cropped_bbox()
        self.update_graphics_item()
        self.annotationUpdated.emit(self)  # Notify update

    def update_annotation_size(self, size):
        """Update the size of the annotation."""
        self.update_user_confidence(self.label)

        # Update the size, graphic
        self.annotation_size = size
        self.set_cropped_bbox()  # Update the bounding box
        self.update_graphics_item()
        self.annotationUpdated.emit(self)  # Notify update

    def resize(self, handle: str, new_pos: QPointF):
        """Resize the annotation based on the handle position."""
        pass
    
    @classmethod
    def combine(cls, annotations: list):
        """Combine multiple annotations into a single one."""
        pass
    
    @classmethod
    def cut(cls, annotations: list, cutting_points: list):
        """Cut the annotations based on the provided cutting points."""
        pass

    def to_dict(self):
        """Convert the annotation to a dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            'center_xy': (self.center_xy.x(), self.center_xy.y()),
            'annotation_size': self.annotation_size,
        })
        return base_dict

    @classmethod
    def from_dict(cls, data, label_window):
        """Create an annotation from a dictionary representation."""
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
        
        # Override the verified attribute if it exists in the data
        if 'verified' in data:
            annotation.update_verified(data['verified'])

        return annotation

    def __repr__(self):
        """Return a string representation of the annotation."""
        return (f"PatchAnnotation(id={self.id}, center_xy={self.center_xy}, "
                f"annotation_size={self.annotation_size}, "
                f"annotation_color={self.label.color.name()}, "
                f"image_path={self.image_path}, "
                f"label={self.label.short_label_code}, "
                f"data={self.data}, "
                f"machine_confidence={self.machine_confidence})")
