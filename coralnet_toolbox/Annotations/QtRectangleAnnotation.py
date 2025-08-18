import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from rasterio.windows import Window

from shapely.ops import split, unary_union
from shapely.geometry import Point, LineString, box

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPathItem
from PyQt5.QtGui import (QPixmap, QColor, QPen, QBrush, QPolygonF,
                         QPainter, QImage, QRegion, QPainterPath)

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
                 show_msg: bool = False):
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
        # Store the raw coordinates, keeping the top_left and bottom_right as they are
        # to maintain the user's dragging experience
        self.cropped_bbox = (self.top_left.x(), self.top_left.y(), self.bottom_right.x(), self.bottom_right.y())
        
        # Calculate annotation_size using absolute differences to handle inverted coordinates
        width = abs(self.bottom_right.x() - self.top_left.x())
        height = abs(self.bottom_right.y() - self.top_left.y())
        self.annotation_size = int(max(width, height))

    def contains_point(self, point: QPointF) -> bool:
        """Check if the given point is within the rectangle using Shapely."""
        try:
            # Create a shapely box from the rectangle's corner coordinates
            shapely_rect = box(self.top_left.x(), 
                               self.top_left.y(),
                               self.bottom_right.x(), 
                               self.bottom_right.y())

            # Convert the input QPointF to a Shapely Point
            shapely_point = Point(point.x(), point.y())

            # Return Shapely's boolean result for the containment check
            return shapely_rect.contains(shapely_point)
        
        except Exception:
            # Fallback to the original implementation if Shapely fails
            return (self.top_left.x() <= point.x() <= self.bottom_right.x() and
                    self.top_left.y() <= point.y() <= self.bottom_right.y())

    def get_centroid(self):
        """Get the centroid of the annotation."""
        return (float(self.center_xy.x()), float(self.center_xy.y()))

    def get_area(self):
        """Calculate the area of the rectangle using Shapely."""
        try:
            # Create a shapely box from the rectangle's corner coordinates
            shapely_rect = box(self.top_left.x(), 
                               self.top_left.y(),
                               self.bottom_right.x(), 
                               self.bottom_right.y())
            return shapely_rect.area

        except Exception:
            # Fallback to the original implementation if Shapely fails
            width = self.bottom_right.x() - self.top_left.x()
            height = self.bottom_right.y() - self.top_left.y()
            return width * height

    def get_perimeter(self):
        """Calculate the perimeter of the rectangle using Shapely."""
        try:
            # Create a shapely box from the rectangle's corner coordinates
            shapely_rect = box(self.top_left.x(), 
                               self.top_left.y(),
                               self.bottom_right.x(), 
                               self.bottom_right.y())
            return shapely_rect.length
        
        except Exception:
            # Fallback to the original implementation if Shapely fails
            width = self.bottom_right.x() - self.top_left.x()
            height = self.bottom_right.y() - self.top_left.y()
            return 2 * width + 2 * height

    def get_polygon(self):
        """Get the polygon representation of this rectangle."""
        points = [
            self.top_left,
            QPointF(self.bottom_right.x(), self.top_left.y()),
            self.bottom_right,
            QPointF(self.top_left.x(), self.bottom_right.y())
        ]
        return QPolygonF(points)
    
    def get_painter_path(self) -> QPainterPath:
        """
        Get a QPainterPath representation of the annotation.
        """
        path = QPainterPath()
        polygon = self.get_polygon()
        path.addPolygon(polygon)
        return path

    def get_bounding_box_top_left(self):
        """Get the top-left corner of the bounding box."""
        return self.top_left

    def get_bounding_box_bottom_right(self):
        """Get the bottom-right corner of the bounding box."""
        return self.bottom_right

    def get_cropped_image_graphic(self):
        """Create a cropped image with a mask and solid outline."""
        if self.cropped_image is None:
            return None

        # Create a QImage with transparent background for the mask
        masked_image = QImage(self.cropped_image.size(), QImage.Format_ARGB32)
        masked_image.fill(Qt.transparent)

        # Create a painter to draw the path onto the mask
        mask_painter = QPainter(masked_image)
        mask_painter.setRenderHint(QPainter.Antialiasing)
        mask_painter.setBrush(QBrush(Qt.white))
        mask_painter.setPen(Qt.NoPen)

        # Create a path from the points relative to the cropped image
        path = QPainterPath()
        # The cropped image is the same size as the rectangle, so the path is from (0,0)
        path.addRect(0, 0, self.cropped_image.width(), self.cropped_image.height())

        # Draw the path onto the mask
        mask_painter.drawPath(path)
        mask_painter.end()

        # Convert the mask QImage to QPixmap and create a bitmap mask
        mask_pixmap = QPixmap.fromImage(masked_image)
        mask_bitmap = mask_pixmap.createMaskFromColor(Qt.white, Qt.MaskOutColor)
        mask_region = QRegion(mask_bitmap)

        # Create the result image
        cropped_image_graphic = QPixmap(self.cropped_image.size())
        result_painter = QPainter(cropped_image_graphic)
        result_painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw the background at 50% opacity
        result_painter.setOpacity(0.5)
        result_painter.drawPixmap(0, 0, self.cropped_image)

        # Draw the full-opacity image inside the masked region
        result_painter.setOpacity(1.0)
        result_painter.setClipRegion(mask_region)
        result_painter.drawPixmap(0, 0, self.cropped_image)

        # Draw the solid line outline on top
        pen = QPen(Qt.black)
        pen.setStyle(Qt.SolidLine)
        pen.setWidth(1)
        result_painter.setPen(pen)
        result_painter.setClipping(False)
        result_painter.drawPath(path)
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
        
        # Ensure min/max values are correctly ordered to avoid negative width/height
        min_x, max_x = min(min_x, max_x), max(min_x, max_x)
        min_y, max_y = min(min_y, max_y), max(min_y, max_y)

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

    def create_graphics_item(self, scene: QGraphicsScene):
        """Create all graphics items for the annotation and add them to the scene."""
        # Get the complete shape as a QPainterPath.
        path = self.get_painter_path()
        
        # Use a QGraphicsPathItem for rendering.
        self.graphics_item = QGraphicsPathItem(path)
        
        # Call the parent class method to handle grouping, styling, and adding to the scene.
        super().create_graphics_item(scene)
    
    def update_graphics_item(self):
        """Update the graphical representation of the rectangle annotation."""
        # Get the complete shape as a QPainterPath.
        path = self.get_painter_path()
        
        # Use a QGraphicsPathItem to correctly represent the shape.
        self.graphics_item = QGraphicsPathItem(path)
        
        # Call the parent class method to handle rebuilding the graphics group.
        super().update_graphics_item()
        
    def update_polygon(self, delta):
        """
        For rectangles, the polygon is always defined by top_left and bottom_right.
        This method can be used to update centroid and bounding box if needed.
        """
        self.set_precision(self.top_left, self.bottom_right)
        self.set_centroid()
        self.set_cropped_bbox()

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

        # Normalize coordinates to ensure top_left has smaller values than bottom_right
        self.normalize_coordinates()
        
        self.set_precision(self.top_left, self.bottom_right)
        self.set_centroid()
        self.update_graphics_item()
        self.annotationUpdated.emit(self)
        
    def normalize_coordinates(self):
        """Ensure that top_left has smaller coordinates than bottom_right."""
        # Create temporary points to store the normalized coordinates
        x_min = min(self.top_left.x(), self.bottom_right.x())
        y_min = min(self.top_left.y(), self.bottom_right.y())
        x_max = max(self.top_left.x(), self.bottom_right.x())
        y_max = max(self.top_left.y(), self.bottom_right.y())
        
        # Update the points
        self.top_left.setX(x_min)
        self.top_left.setY(y_min)
        self.bottom_right.setX(x_max)
        self.bottom_right.setY(y_max)
        
        # Update centroid after normalization
        self.set_centroid()

    @classmethod
    def combine(cls, annotations: list):
        """Combine multiple rectangle annotations into a single encompassing rectangle
        using Shapely, but only if every annotation overlaps with at least one other.
        """
        if not annotations:
            return None
        if len(annotations) == 1:
            return annotations[0]

        # Convert all annotations to Shapely boxes
        shapely_rects = []
        for anno in annotations:
            shaped_rect = box(anno.top_left.x(), 
                              anno.top_left.y(),
                              anno.bottom_right.x(), 
                              anno.bottom_right.y())
            
            shapely_rects.append(shaped_rect)
            
        # 1. Perform the overlap check using Shapely's `intersects`
        for i, rect_i in enumerate(shapely_rects):
            has_overlap = False
            for j, rect_j in enumerate(shapely_rects):
                if i != j and rect_i.intersects(rect_j):
                    has_overlap = True
                    break
            if not has_overlap:
                return None  # An annotation is isolated, cancel the combine

        # 2. Get the encompassing bounding box using Shapely's union and bounds
        merged_geom = unary_union(shapely_rects)
        min_x, min_y, max_x, max_y = merged_geom.bounds

        # Create new rectangle with these bounds
        top_left = QPointF(min_x, min_y)
        bottom_right = QPointF(max_x, max_y)

        # Extract info from the first annotation
        first_anno = annotations[0]
        new_annotation = cls(
            top_left=top_left,
            bottom_right=bottom_right,
            short_label_code=first_anno.label.short_label_code,
            long_label_code=first_anno.label.long_label_code,
            color=first_anno.label.color,
            image_path=first_anno.image_path,
            label_id=first_anno.label.id
        )

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

        # Create a line from the cutting points (do NOT extend)
        line_points = [(point.x(), point.y()) for point in cutting_points]
        cutting_line = LineString(line_points)

        # Check if the line intersects the rectangle
        if not rect_shapely.intersects(cutting_line):
            return [annotation]  # No intersection, return original

        try:
            # Split the rectangle with the user-supplied line only (no extension)
            split_result = split(rect_shapely, cutting_line)

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
