import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from rasterio.windows import Window
from shapely.ops import split
from shapely.geometry import Polygon, LineString, box

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsRectItem
from PyQt5.QtGui import (QPixmap, QColor, QPen, QBrush, QPolygonF, QPainter,
                         QImage, QRegion)

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
        
    def set_centroid(self):
        """Set the centroid of the rectangle."""
        self.center_xy = QPointF((self.top_left.x() + self.bottom_right.x()) / 2,
                                 (self.top_left.y() + self.bottom_right.y()) / 2)

    def set_cropped_bbox(self):
        """Set the cropped bounding box for the annotation."""
        self.cropped_bbox = (self.top_left.x(), self.top_left.y(), self.bottom_right.x(), self.bottom_right.y())
        self.annotation_size = int(max(self.bottom_right.x() - self.top_left.x(),
                                       self.bottom_right.y() - self.top_left.y()))

    def contains_point(self, point: QPointF) -> bool:
        """Check if the given point is within the rectangle."""
        return (self.top_left.x() <= point.x() <= self.bottom_right.x() and
                self.top_left.y() <= point.y() <= self.bottom_right.y())
        
    def get_centroid(self):
        """Get the centroid of the annotation."""
        return (float(self.center_xy.x()), float(self.center_xy.y()))
        
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

        # Create a QImage with transparent background for the mask
        masked_image = QImage(self.cropped_image.size(), QImage.Format_ARGB32)
        masked_image.fill(Qt.transparent)  # Transparent background
        
        # Create a QPainter to draw the polygon onto the mask
        painter = QPainter(masked_image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(Qt.white))  # White fill for the mask area
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
        
    @classmethod
    def combine(cls, annotations: list):
        """Combine multiple rectangle annotations into a single encompassing rectangle,
        but only if every annotation overlaps with at least one other annotation.
        
        Args:
            annotations: List of RectangleAnnotations objects to combine.
            
        Returns:
            A new RectangleAnnotations that encompasses all input rectangles if every 
            annotation overlaps with at least one other, otherwise None.
        """
        if not annotations:
            return None
        
        if len(annotations) == 1:
            return annotations[0]
        
        # Check if each annotation overlaps with at least one other annotation
        for i, anno_i in enumerate(annotations):
            has_overlap = False
            for j, anno_j in enumerate(annotations):
                if i == j:
                    continue
                    
                # Check if these two rectangles overlap
                if (anno_i.top_left.x() < anno_j.bottom_right.x() and 
                    anno_i.bottom_right.x() > anno_j.top_left.x() and
                    anno_i.top_left.y() < anno_j.bottom_right.y() and
                    anno_i.bottom_right.y() > anno_j.top_left.y()):
                    has_overlap = True
                    break
            
            # If any annotation doesn't overlap with any other, return None
            if not has_overlap:
                return None
                
        # Find the minimum top-left and maximum bottom-right coordinates
        min_x = min(anno.top_left.x() for anno in annotations)
        min_y = min(anno.top_left.y() for anno in annotations)
        max_x = max(anno.bottom_right.x() for anno in annotations)
        max_y = max(anno.bottom_right.y() for anno in annotations)
        
        # Create new rectangle with these bounds
        top_left = QPointF(min_x, min_y)
        bottom_right = QPointF(max_x, max_y)

        # Extract info from the first annotation
        short_label_code = annotations[0].label.short_label_code
        long_label_code = annotations[0].label.long_label_code
        color = annotations[0].label.color
        image_path = annotations[0].image_path
        label_id = annotations[0].label.id
        
        # Create a new annotation with the merged points
        new_annotation = cls(
            top_left=top_left,
            bottom_right=bottom_right,
            short_label_code=short_label_code,
            long_label_code=long_label_code,
            color=color,
            image_path=image_path,
            label_id=label_id
        )
        
        # All input annotations have the same rasterio source, use it for the new one
        if all(hasattr(anno, 'rasterio_src') and anno.rasterio_src is not None for anno in annotations):
            if len(set(id(anno.rasterio_src) for anno in annotations)) == 1:
                new_annotation.rasterio_src = annotations[0].rasterio_src
                new_annotation.create_cropped_image(new_annotation.rasterio_src)
        
        return new_annotation
    
    @classmethod
    def cut(cls, annotation, cutting_points: list):
        """Cut a rectangle annotation where it intersects with a cutting line.
        
        Args:
            annotation: A RectangleAnnotation object to process.
            cutting_points: List of QPointF objects defining a cutting line (potentially non-linear).
            
        Returns:
            List of new RectangleAnnotation objects resulting from the cut.
            If the line doesn't intersect the rectangle, returns a list with the original annotation.
        """
        if not annotation or len(cutting_points) < 2:
            return [annotation] if annotation else []
        
        # Get rectangle bounds
        x1, y1 = annotation.top_left.x(), annotation.top_left.y()
        x2, y2 = annotation.bottom_right.x(), annotation.bottom_right.y()
        
        # Create a shapely box from rectangle coordinates
        rect_shapely = box(x1, y1, x2, y2)
        
        # Create a line from the cutting points
        line_points = [(point.x(), point.y()) for point in cutting_points]
        cutting_line = LineString(line_points)
        
        # Check if the line intersects the rectangle
        if not rect_shapely.intersects(cutting_line):
            return [annotation]  # No intersection, return original
        
        # Extend the cutting line to ensure it fully cuts through the rectangle
        def extend_line(line, distance=1000):
            """Extend line segments at both ends to ensure complete cutting."""
            coords = list(line.coords)
            if len(coords) < 2:
                return line
            
            # Extend the first segment
            first, second = coords[0], coords[1]
            dx, dy = first[0] - second[0], first[1] - second[1]
            length = (dx**2 + dy**2)**0.5
            if length > 0:
                dx, dy = dx / length * distance, dy / length * distance
            extended_first = (first[0] + dx, first[1] + dy)
            
            # Extend the last segment
            last, second_last = coords[-1], coords[-2]
            dx, dy = last[0] - second_last[0], last[1] - second_last[1]
            length = (dx**2 + dy**2)**0.5
            if length > 0:
                dx, dy = dx / length * distance, dy / length * distance
            extended_last = (last[0] + dx, last[1] + dy)
            
            # Create new line with extended endpoints
            return LineString([extended_first] + coords[1:-1] + [extended_last])
        
        # Extend the cutting line
        extended_line = extend_line(cutting_line)
        
        try:
            # Split the rectangle with the extended line
            split_result = split(rect_shapely, extended_line)
            
            result_annotations = []
            min_area = 10  # Minimum area threshold
            
            for geom in split_result.geoms:
                # Skip tiny fragments
                if geom.area < min_area:
                    continue
                
                # Get the bounds of the split geometry
                minx, miny, maxx, maxy = geom.bounds
                
                # Avoid creating degenerate rectangles
                if maxx - minx < 1 or maxy - miny < 1:
                    continue
                    
                # Create a new rectangle annotation with the bounds
                new_anno = cls(
                    top_left=QPointF(minx, miny),
                    bottom_right=QPointF(maxx, maxy),
                    short_label_code=annotation.label.short_label_code,
                    long_label_code=annotation.label.long_label_code,
                    color=annotation.label.color,
                    image_path=annotation.image_path,
                    label_id=annotation.label.id
                )
                
                # Transfer rasterio source if available
                if hasattr(annotation, 'rasterio_src') and annotation.rasterio_src is not None:
                    new_anno.rasterio_src = annotation.rasterio_src
                    new_anno.create_cropped_image(new_anno.rasterio_src)
                    
                result_annotations.append(new_anno)
            
            # If cutting didn't produce any results, return the original annotation
            return result_annotations if result_annotations else [annotation]
            
        except Exception as e:
            # Log the error and return the original rectangle
            print(f"Error during rectangle cutting: {e}")
            return [annotation]

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

        # Set the machine confidence
        annotation.update_machine_confidence(machine_confidence, from_import=True)
        
        # Override the verified attribute if it exists in the data
        if 'verified' in data:
            annotation.set_verified(data['verified'])

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
