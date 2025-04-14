import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from rasterio.windows import Window

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPixmap, QColor, QPen, QBrush, QPolygonF, QPainter
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsRectItem

from coralnet_toolbox.Annotations.QtAnnotation import Annotation

from coralnet_toolbox.utilities import rasterio_to_cropped_image


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

        self.set_precision(top_left, bottom_right, False)
        self.set_centroid()
        self.set_cropped_bbox()

    def set_precision(self, top_left: QPointF, bottom_right: QPointF, reduce: bool = True):
        """Reduce the precision of the coordinates to 2 decimal places."""
        if reduce:
            top_left = QPointF(round(top_left.x(), 2), round(top_left.y(), 2))
            bottom_right = QPointF(round(bottom_right.x(), 2), round(bottom_right.y(), 2))

        self.top_left = top_left
        self.bottom_right = bottom_right
        
    def set_cropped_bbox(self):
        """Set the cropped bounding box for the annotation."""
        self.cropped_bbox = (self.top_left.x(), self.top_left.y(), self.bottom_right.x(), self.bottom_right.y())
        self.annotation_size = int(max(self.bottom_right.x() - self.top_left.x(),
                                       self.bottom_right.y() - self.top_left.y()))

    def set_centroid(self):
        """Set the centroid of the rectangle."""
        self.center_xy = QPointF((self.top_left.x() + self.bottom_right.x()) / 2,
                                 (self.top_left.y() + self.bottom_right.y()) / 2)

    def contains_point(self, point: QPointF) -> bool:
        """Check if the given point is within the rectangle."""
        return (self.top_left.x() <= point.x() <= self.bottom_right.x() and
                self.top_left.y() <= point.y() <= self.bottom_right.y())
        
    def create_cropped_image(self, rasterio_src):
        """Create a cropped image from the rasterio source."""
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

        # Convert rasterio to QImage
        q_image = rasterio_to_cropped_image(self.rasterio_src, window)
        # Convert QImage to QPixmap
        self.cropped_image = QPixmap.fromImage(q_image)

        self.annotationUpdated.emit(self)  # Notify update
        
    def get_area(self):
        """Calculate the area of the rectangle."""
        return (self.bottom_right.x() - self.top_left.x()) * (self.bottom_right.y() - self.top_left.y())

    def get_perimeter(self):
        """Calculate the perimeter of the rectangle."""
        return 2 * (self.bottom_right.x() - self.top_left.x()) + 2 * (self.bottom_right.y() - self.top_left.y())

    def get_polygon(self):
        """Get the polygon representation of this rectangle."""
        points = [
            self.top_left,
            QPointF(self.bottom_right.x(), self.top_left.y()),
            self.bottom_right,
            QPointF(self.top_left.x(), self.bottom_right.y())
        ]
        return QPolygonF(points)

    def get_bounding_box_top_left(self):
        """Get the top-left corner of the bounding box."""
        return self.top_left

    def get_bounding_box_bottom_right(self):
        """Get the bottom-right corner of the bounding box."""
        return self.bottom_right

    def get_cropped_image_graphic(self):
        """Create a cropped image with a mask and dotted outline."""
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

        # Define the rectangle points based on top_left and bottom_right
        rectangle_points = [
            self.top_left,
            QPointF(self.bottom_right.x(), self.top_left.y()),
            self.bottom_right,
            QPointF(self.top_left.x(), self.bottom_right.y())
        ]

        # Create a copy of the points that are transformed to be relative to the cropped_image
        cropped_points = [QPointF(point.x() - self.cropped_bbox[0],
                                  point.y() - self.cropped_bbox[1]) for point in rectangle_points]

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

        # Draw the rectangle outline with the dotted pen
        painter.drawPolygon(polygon)

        painter.end()

        return cropped_image_graphic

    def update_graphics_item(self, crop_image=True):
        """Update the graphical representation of the annotation using base class method."""
        super().update_graphics_item(crop_image)

        # Update the cropped image if necessary
        if self.rasterio_src and crop_image:
            self.create_cropped_image(self.rasterio_src)

    def update_location(self, new_center_xy: QPointF):
        """# Update the location of the annotation"""
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
        """"Update the size of the annotation"""
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
        """Resize the annotation based on the handle and new position."""
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

        self.set_precision(self.top_left, self.bottom_right)
        self.set_centroid()
        self.update_graphics_item()
        self.annotationUpdated.emit(self)

    def to_dict(self):
        """Convert the annotation to a dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            'top_left': (self.top_left.x(), self.top_left.y()),
            'bottom_right': (self.bottom_right.x(), self.bottom_right.y()),
        })
        return base_dict

    @classmethod
    def from_dict(cls, data, label_window):
        """Create a RectangleAnnotation from a dictionary representation."""
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
        """Represent the RectangleAnnotation as a string."""
        return (f"RectangleAnnotation(id={self.id}, "
                f"top_left={self.top_left}, bottom_right={self.bottom_right}, "
                f"annotation_color={self.label.color.name()}, "
                f"image_path={self.image_path}, "
                f"label={self.label.short_label_code}, "
                f"data={self.data}, "
                f"machine_confidence={self.machine_confidence})")
