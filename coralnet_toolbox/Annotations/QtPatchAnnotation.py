import warnings

from rasterio.windows import Window

from shapely.ops import unary_union
from shapely.geometry import Point, Polygon 

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPathItem
from PyQt5.QtGui import (QPixmap, QColor, QPen, QBrush, QPainter,
                         QPolygonF, QImage, QRegion, QPainterPath)

from coralnet_toolbox.Annotations.QtAnnotation import Annotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation

from coralnet_toolbox.utilities import rasterio_to_cropped_image

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
                 show_msg: bool = False):
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
        """
        Check if the given point is inside the polygon using Shapely.
        """
        try:
            # Convert the patch's corners to coordinate tuples for Shapely
            qt_polygon = self.get_polygon()
            shell_coords = [(p.x(), p.y()) for p in qt_polygon]

            # Create a Shapely polygon
            shapely_polygon = Polygon(shell=shell_coords)

            # Convert the input QPointF to a Shapely Point
            shapely_point = Point(point.x(), point.y())

            # Return Shapely's boolean result for the containment check
            return shapely_polygon.contains(shapely_point)

        except Exception:
            # Fallback to the original QRectF implementation if Shapely fails
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
        """
        Calculate the net area of the polygon using Shapely.
        """
        try:
            # Convert the patch's corners to coordinate tuples for Shapely
            qt_polygon = self.get_polygon()
            shell_coords = [(p.x(), p.y()) for p in qt_polygon]

            # A valid polygon needs at least 3 points
            if len(shell_coords) < 3:
                return 0.0

            # Create a Shapely polygon and return its area
            shapely_polygon = Polygon(shell=shell_coords)
            return shapely_polygon.area

        except Exception:
            # Fallback to the original implementation if Shapely fails
            return self.annotation_size * self.annotation_size

    def get_perimeter(self):
        """
        Calculate the perimeter of the polygon using Shapely.
        """
        try:
            # Convert the patch's corners to coordinate tuples for Shapely
            qt_polygon = self.get_polygon()
            shell_coords = [(p.x(), p.y()) for p in qt_polygon]
            
            # A shape with fewer than 2 points has no length
            if len(shell_coords) < 2:
                return 0.0

            # Create a Shapely polygon and return its perimeter (length)
            shapely_polygon = Polygon(shell=shell_coords)
            return shapely_polygon.length

        except Exception:
            # Fallback to the original implementation if Shapely fails
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
    
    def get_painter_path(self) -> QPainterPath:
        """
        Get a QPainterPath representation of the annotation.
        """
        path = QPainterPath()

        # Get the square's corners from the existing get_polygon method
        polygon = self.get_polygon()

        # Add the polygon to the path
        path.addPolygon(polygon)

        return path

    def get_bounding_box_top_left(self):
        """Get the top-left corner of the bounding box."""
        half_size = self.annotation_size / 2
        return QPointF(self.center_xy.x() - half_size, self.center_xy.y() - half_size)

    def get_bounding_box_bottom_right(self):
        """Get the bottom-right corner of the bounding box."""
        half_size = self.annotation_size / 2
        return QPointF(self.center_xy.x() + half_size, self.center_xy.y() + half_size)

    def get_cropped_image_graphic(self):
        """Get the cropped image with a solid outline."""
        if self.cropped_image is None:
            return None

        # Create a QImage with transparent background for the mask
        masked_image = QImage(self.cropped_image.size(), QImage.Format_ARGB32)
        masked_image.fill(Qt.transparent)

        # Create a painter to draw the path onto the mask
        mask_painter = QPainter(masked_image)
        mask_painter.setRenderHint(QPainter.Antialiasing)
        mask_painter.setBrush(QBrush(Qt.white))  # White fill for the mask area
        mask_painter.setPen(Qt.NoPen)
        
        # Define the square's corners relative to the cropped image (0,0)
        path = QPainterPath()
        path.addRect(0, 0, self.cropped_image.width(), self.cropped_image.height())

        # Draw the path onto the mask
        mask_painter.drawPath(path)
        mask_painter.end()

        # Convert the mask to a QRegion for clipping
        mask_pixmap = QPixmap.fromImage(masked_image)
        mask_bitmap = mask_pixmap.createMaskFromColor(Qt.white, Qt.MaskOutColor)
        mask_region = QRegion(mask_bitmap)

        # Create the result image with the background at 50% opacity
        cropped_image_graphic = QPixmap(self.cropped_image.size())
        result_painter = QPainter(cropped_image_graphic)
        result_painter.setRenderHint(QPainter.Antialiasing)
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
        result_painter.setClipping(False)  # Disable clipping for the outline
        result_painter.drawPath(path)
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

    def create_graphics_item(self, scene: QGraphicsScene):
        """Create all graphics items for the annotation and add them to the scene."""
        # Get the complete shape as a QPainterPath.
        path = self.get_painter_path()
        
        # Use a QGraphicsPathItem for rendering.
        self.graphics_item = QGraphicsPathItem(path)
        
        # Call the parent class method to handle grouping, styling, and adding to the scene.
        super().create_graphics_item(scene)
    
    def update_graphics_item(self):
        """Update the graphical representation of the patch annotation."""
        # Get the complete shape as a QPainterPath.
        path = self.get_painter_path()
        
        # Use a QGraphicsPathItem to correctly represent the shape.
        self.graphics_item = QGraphicsPathItem(path)
        
        # Call the parent class method to handle rebuilding the graphics group.
        super().update_graphics_item()
        
    def update_polygon(self, delta):
        """
        For patches, the polygon is always defined by center_xy and annotation_size.
        This method can be used to update centroid and bounding box if needed.
        """
        self.set_precision(self.center_xy)
        self.set_centroid()
        self.set_cropped_bbox()

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
        """
        Combine annotations using Shapely's union operation.
        Returns a single PolygonAnnotation or a MultiPolygonAnnotation.
        """
        if not annotations:
            return None
        if len(annotations) == 1:
            return annotations[0]

        try:
            # 1. Convert all input annotations to Shapely Polygons.
            shapely_polygons = []
            for anno in annotations:
                # get_polygon() works for both PatchAnnotation and PolygonAnnotation
                qt_polygon = anno.get_polygon()
                points = [(p.x(), p.y()) for p in qt_polygon]
                shapely_polygons.append(Polygon(points))

            # 2. Perform the union operation.
            merged_geom = unary_union(shapely_polygons)
            
            # --- Get properties from the first annotation for the new one ---
            first_anno = annotations[0]
            common_args = {
                "short_label_code": first_anno.label.short_label_code,
                "long_label_code": first_anno.label.long_label_code,
                "color": first_anno.label.color,
                "image_path": first_anno.image_path,
                "label_id": first_anno.label.id
            }

            # 3. Build the appropriate new annotation based on the result.
            if merged_geom.geom_type == 'Polygon':
                exterior_points = [QPointF(x, y) for x, y in merged_geom.exterior.coords]
                return PolygonAnnotation(points=exterior_points, **common_args)

            elif merged_geom.geom_type == 'MultiPolygon':
                new_polygons = []
                for poly in merged_geom.geoms:
                    exterior_points = [QPointF(x, y) for x, y in poly.exterior.coords]
                    new_polygons.append(PolygonAnnotation(points=exterior_points, **common_args))
                return MultiPolygonAnnotation(polygons=new_polygons, **common_args)
            
            return None  # The geometry is empty or an unexpected type

        except Exception as e:
            print(f"Error during polygon combination: {e}")
            return None

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

        # Add any additional data from the dictionary
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
        """Return a string representation of the annotation."""
        return (f"PatchAnnotation(id={self.id}, center_xy={self.center_xy}, "
                f"annotation_size={self.annotation_size}, "
                f"annotation_color={self.label.color.name()}, "
                f"image_path={self.image_path}, "
                f"label={self.label.short_label_code}, "
                f"data={self.data}, "
                f"machine_confidence={self.machine_confidence})")
