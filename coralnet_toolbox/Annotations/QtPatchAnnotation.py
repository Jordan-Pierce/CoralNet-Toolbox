import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from rasterio.windows import Window

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsRectItem
from PyQt5.QtGui import (QPixmap, QColor, QPen, QBrush, QPainter, 
                         QPolygonF, QImage, QRegion)

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

    def set_centroid(self):
        """Calculate the centroid of the annotation (for patch, this is the center_xy)."""
        self.center_xy = self.center_xy
        
    def set_cropped_bbox(self):
        """Set the cropped bounding box coordinates based on center and size."""
        half_size = self.annotation_size / 2
        min_x = self.center_xy.x() - half_size
        min_y = self.center_xy.y() - half_size
        max_x = self.center_xy.x() + half_size
        max_y = self.center_xy.y() + half_size
        self.cropped_bbox = (min_x, min_y, max_x, max_y)
        
    def contains_point(self, point: QPointF):
        """Check if the point is within the annotation's bounding box."""
        half_size = self.annotation_size / 2
        rect = QRectF(self.center_xy.x() - half_size,
                      self.center_xy.y() - half_size,
                      self.annotation_size,
                      self.annotation_size)
        return rect.contains(point)
    
    def get_centroid(self):
        """Get the centroid of the annotation."""
        return (float(self.center_xy.x()), float(self.center_xy.y()))

    def get_area(self):
        """Calculate the area of the square patch."""
        return self.annotation_size * self.annotation_size

    def get_perimeter(self):
        """Calculate the perimeter of the square patch."""
        return 4 * self.annotation_size

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
        """Get the cropped image with a dotted outline and black background."""
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

        # Define the square's corners as a polygon
        cropped_points = [
            QPointF(0, 0),
            QPointF(self.cropped_image.width(), 0),
            QPointF(self.cropped_image.width(), self.cropped_image.height()),
            QPointF(0, self.cropped_image.height())
        ]

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
        pen = QPen(self.label.color)
        pen.setStyle(Qt.DashLine)  # Creates a dotted/dashed line
        pen.setWidth(2)  # Line width
        result_painter.setPen(pen)
        result_painter.setClipping(False)  # Disable clipping for the outline
        result_painter.drawPolygon(polygon)
        
        result_painter.end()
        
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
