import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import math
import numpy as np

from rasterio.windows import Window
from shapely.geometry import Point
from shapely.geometry import Polygon, LineString

from PyQt5.QtCore import Qt, QPointF

from PyQt5.QtWidgets import (QGraphicsScene, QGraphicsPolygonItem, QGraphicsPathItem,
                             QGraphicsItem, QGraphicsItemGroup, QMessageBox)

from PyQt5.QtGui import (QPixmap, QColor, QPen, QBrush, QPolygonF,
                         QPainter, QRegion, QImage, QPainterPath)

from coralnet_toolbox.Annotations.QtAnnotation import Annotation

from coralnet_toolbox.utilities import rasterio_to_cropped_image


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MultiPolygonAnnotation(Annotation):
    def __init__(self,
                 polygons: list,  # Now a list of PolygonAnnotation objects
                 short_label_code: str,
                 long_label_code: str,
                 color: QColor,
                 image_path: str,
                 label_id: str,
                 transparency: int = 128,
                 show_msg: bool = False):
        """Initialize a MultiPolygonAnnotation from a list of PolygonAnnotation objects."""
        # Use properties from the first polygon if not provided
        if polygons and hasattr(polygons[0], 'label'):
            first = polygons[0]
            short_label_code = short_label_code or first.label.short_label_code
            long_label_code = long_label_code or first.label.long_label_code
            color = color or first.label.color
            image_path = image_path or first.image_path
            label_id = label_id or first.label.id
            transparency = transparency if transparency is not None else first.transparency
            show_msg = show_msg if show_msg is not None else first.show_message
        super().__init__(short_label_code, long_label_code, color, image_path, label_id, transparency, show_msg)

        self.center_xy = QPointF(0, 0)
        self.cropped_bbox = (0, 0, 0, 0)
        self.annotation_size = 0

        # Store the PolygonAnnotation objects
        self.polygons = polygons

        self.graphics_items = []
        self.set_centroid()
        self.set_cropped_bbox()

    def set_precision(self, reduce: bool = True):
        """Round coordinates of all polygons to 3 decimal places."""
        for polygon in self.polygons:
            polygon.set_precision(reduce)

    def set_centroid(self):
        """Compute and set the centroid of all polygons."""
        total_x, total_y, count = 0, 0, 0
        for polygon in self.polygons:
            if hasattr(polygon, 'center_xy'):
                total_x += polygon.center_xy.x()
                total_y += polygon.center_xy.y()
                count += 1
        self.center_xy = QPointF(total_x / count, total_y / count) if count else QPointF(0, 0)

    def set_cropped_bbox(self):
        """Compute and set the bounding box that crops all polygons."""
        all_points = [p for poly in self.polygons for p in poly.points]
        if not all_points:
            self.cropped_bbox = (0, 0, 0, 0)
            return
        min_x = min(p.x() for p in all_points)
        min_y = min(p.y() for p in all_points)
        max_x = max(p.x() for p in all_points)
        max_y = max(p.y() for p in all_points)
        self.cropped_bbox = (min_x, min_y, max_x, max_y)
        self.annotation_size = int(max(max_x - min_x, max_y - min_y))

    def contains_point(self, point: QPointF) -> bool:
        """Return True if the point is inside any polygon."""
        return any(poly.contains_point(point) for poly in self.polygons)

    def get_area(self):
        """Return the total area of all polygons."""
        return sum(poly.get_area() for poly in self.polygons)

    def get_perimeter(self):
        """Return the total perimeter of all polygons."""
        return sum(poly.get_perimeter() for poly in self.polygons)

    def get_polygon(self):
        """Return the first polygon (for compatibility)."""
        return QPolygonF(self.polygons[0].points) if self.polygons else QPolygonF()

    def get_bounding_box_top_left(self):
        """Return the top-left corner of the bounding box."""
        return QPointF(self.cropped_bbox[0], self.cropped_bbox[1])

    def get_bounding_box_bottom_right(self):
        """Get the bottom-right corner of the annotation's bounding box."""
        return QPointF(self.cropped_bbox[2], self.cropped_bbox[3])

    def get_cropped_image_graphic(self):
        """Return a QPixmap of the cropped image masked by the annotation polygons."""
        if not self.cropped_image:
            return None

        # Create a transparent image to use as a mask
        masked_image = QImage(self.cropped_image.size(), QImage.Format_ARGB32)
        masked_image.fill(Qt.transparent)
        painter = QPainter(masked_image)
        painter.setBrush(QBrush(Qt.white))
        painter.setPen(Qt.NoPen)

        # Draw each polygon (shifted to cropped bbox) in white on the mask
        for poly in self.polygons:
            cropped_points = [QPointF(p.x() - self.cropped_bbox[0], p.y() - self.cropped_bbox[1]) for p in poly.points]
            painter.drawPolygon(QPolygonF(cropped_points))
        painter.end()

        # Convert the mask image to a pixmap and create a mask bitmap
        mask_pixmap = QPixmap.fromImage(masked_image)
        mask_bitmap = mask_pixmap.createMaskFromColor(Qt.white, Qt.MaskOutColor)
        mask_region = QRegion(mask_bitmap)

        # Prepare the final pixmap for the cropped image
        cropped_image_graphic = QPixmap(self.cropped_image.size())
        result_painter = QPainter(cropped_image_graphic)
        result_painter.setRenderHint(QPainter.Antialiasing)

        # Draw the cropped image at 50% opacity as a background
        result_painter.setOpacity(0.5)
        result_painter.drawPixmap(0, 0, self.cropped_image)

        # Draw the masked region at full opacity
        result_painter.setOpacity(1.0)
        result_painter.setClipRegion(mask_region)
        result_painter.drawPixmap(0, 0, self.cropped_image)

        # Draw dashed outline for each polygon
        pen = QPen(Qt.black)
        pen.setStyle(Qt.SolidLine)  # Solid line
        pen.setWidth(1)  # Line width
        result_painter.setPen(pen)
        result_painter.setClipping(False)
        for poly in self.polygons:
            cropped_points = [QPointF(p.x() - self.cropped_bbox[0], p.y() - self.cropped_bbox[1]) for p in poly.points]
            result_painter.drawPolygon(QPolygonF(cropped_points))
        result_painter.end()

        return cropped_image_graphic

    def create_cropped_image(self, rasterio_src):
        """Create a cropped image from the rasterio source based on the polygon points."""
        # Set the rasterio source for the annotation
        self.rasterio_src = rasterio_src
        # Set the cropped bounding box for the annotation
        self.set_cropped_bbox()
        # Get the bounding box of the polygon
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

    def create_graphics_item(self, scene: QGraphicsScene):
        """Create and add QGraphicsItems for all polygons to the scene."""
        # Remove old group if it exists
        if self.graphics_item_group and self.graphics_item_group.scene():
            self.graphics_item_group.scene().removeItem(self.graphics_item_group)
            self.center_graphics_item = None
            self.bounding_box_graphics_item = None
            self.polygon_graphics_item = None

        # Create a new group to hold all polygon items
        self.graphics_item_group = QGraphicsItemGroup()

        # Add each polygon as a QGraphicsPolygonItem to the group
        for poly in self.polygons:
            item = QGraphicsPolygonItem(QPolygonF(poly.points))
            color = QColor(self.label.color)
            color.setAlpha(self.transparency)
            
            # Use the consolidated pen creation method
            item.setPen(self._create_pen(color))
            item.setBrush(QBrush(color))
            item.setData(0, self.id)  # <-- Enable selection by id
            self.graphics_item_group.addToGroup(item)

        # Add centroid and bounding box helper graphics to the group
        self.create_center_graphics_item(self.center_xy, scene, add_to_group=True)
        self.create_bounding_box_graphics_item(
            self.get_bounding_box_top_left(),
            self.get_bounding_box_bottom_right(),
            scene,
            add_to_group=True
        )
        # Add the group to the scene
        scene.addItem(self.graphics_item_group)

    def update_graphics_item(self):
        """Update the QGraphicsItems for all polygons and helper graphics."""
        # If a graphics item group exists and is in a scene, remove it before updating
        if self.graphics_item_group and self.graphics_item_group.scene():
            scene = self.graphics_item_group.scene()
            scene.removeItem(self.graphics_item_group)
            self.graphics_item_group = QGraphicsItemGroup()
            self.center_graphics_item = None
            self.bounding_box_graphics_item = None
            self.polygon_graphics_item = None
        else:
            scene = None

        # Recreate QGraphicsPolygonItems for each polygon and add to the group
        for poly in self.polygons:
            item = QGraphicsPolygonItem(QPolygonF(poly.points))
            color = QColor(self.label.color)
            color.setAlpha(self.transparency)
            
            # Use the consolidated pen creation method
            item.setPen(self._create_pen(color))
            item.setBrush(QBrush(color))
            item.setData(0, self.id)  # <-- Enable selection by id
            self.graphics_item_group.addToGroup(item)

        # Add centroid and bounding box helper graphics to the group
        self.create_center_graphics_item(self.center_xy, scene, add_to_group=True)
        self.create_bounding_box_graphics_item(
            self.get_bounding_box_top_left(),
            self.get_bounding_box_bottom_right(),
            scene,
            add_to_group=True
        )

        # Add the updated group back to the scene if available
        if scene:
            scene.addItem(self.graphics_item_group)
    
    def _update_pen_styles(self):
        """Update pen styles with current animated line offset for all polygon items."""
        if not self.is_selected:
            return
            
        color = QColor(self.label.color)
        pen = self._create_pen(color)
        
        # Update all polygon items in the group
        if self.graphics_item_group:
            for item in self.graphics_item_group.childItems():
                if isinstance(item, QGraphicsPolygonItem):
                    item.setPen(pen)
        
        # Update helper graphics items
        if self.center_graphics_item:
            self.center_graphics_item.setPen(pen)
        if self.bounding_box_graphics_item:
            self.bounding_box_graphics_item.setPen(pen)
        if self.polygon_graphics_item:
            self.polygon_graphics_item.setPen(pen)
            
    def update_polygon(self, delta):
        """Show a warning that MultiPolygonAnnotations should be cut before updating."""
        pass  # No operation; this is a placeholder for future functionality

    def update_location(self, new_center_xy: QPointF):
        """Show a warning that MultiPolygonAnnotations should be cut before moving."""
        pass  # No operation; this is a placeholder for future functionality

    def update_annotation_size(self, delta: float):
        """Show a warning that MultiPolygonAnnotations should be cut before resizing."""
        pass  # No operation; this is a placeholder for future functionality

    def cut(self, cutting_points: list = None):
        """Break apart the MultiPolygonAnnotation into individual PolygonAnnotations."""
        return [poly for poly in self.polygons]

    def to_dict(self):
        """Return a dictionary representation of the annotation."""
        base = super().to_dict()
        base['polygons'] = [poly.to_dict() for poly in self.polygons]
        return base

    @classmethod
    def from_dict(cls, data, label_window):
        """Instantiate from a dictionary representation."""
        from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation

        # Convert polygon dictionaries to PolygonAnnotation objects
        polygons = [PolygonAnnotation.from_dict(poly_dict, label_window) for poly_dict in data['polygons']]

        # Create a new MultiPolygonAnnotation instance
        multi_polygon = cls(
            polygons=polygons,
            short_label_code=data.get('label_short_code'),
            long_label_code=data.get('label_long_code'),
            color=QColor(*data.get('annotation_color', (255, 0, 0))),
            image_path=data.get('image_path'),
            label_id=data.get('label_id')
        )

        # Restore additional properties from the dict
        multi_polygon.data = data.get('data', {})

        # Convert machine_confidence keys back to Label objects
        machine_confidence = {}
        for short_label_code, confidence in data.get('machine_confidence', {}).items():
            label = label_window.get_label_by_short_code(short_label_code)
            if label:
                machine_confidence[label] = confidence

        # Set the machine confidence
        multi_polygon.update_machine_confidence(machine_confidence, from_import=True)

        # Override the verified attribute if it exists in the data
        if 'verified' in data:
            multi_polygon.set_verified(data['verified'])

        return multi_polygon

    def to_yolo_segmentation(self, image_width, image_height):
        """Return YOLO segmentation format for the annotation.
        Each polygon is output as a separate line, with normalized coordinates.
        """
        lines = []
        for poly in self.polygons:
            if not poly.points:
                continue
            formatted_points = []
            for point in poly.points:
                formatted_points.append(f"{point.x()/image_width:.6f} {point.y()/image_height:.6f}")
            line = f"{self.label.short_label_code} " + " ".join(formatted_points)
            lines.append(line)
        return "\n".join(lines)

    def __repr__(self):
        """Return a string representation of the annotation."""
        return (f"MultiPolygonAnnotation({len(self.polygons)} polygons, "
                f"label={self.label.short_label_code}, "
                f"area={self.get_area():.2f}, "
                f"centroid=({self.center_xy.x():.1f}, {self.center_xy.y():.1f}))")
