import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import math
import numpy as np

from rasterio.windows import Window
from shapely.geometry import Point
from shapely.geometry import Polygon, LineString

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPolygonItem, QGraphicsPathItem, QGraphicsItem, QGraphicsItemGroup
from PyQt5.QtGui import QPixmap, QColor, QPen, QBrush, QPolygonF, QPainter, QRegion, QImage, QPainterPath

from coralnet_toolbox.Annotations.QtAnnotation import Annotation

from coralnet_toolbox.utilities import rasterio_to_cropped_image


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MultiPolygonAnnotation(Annotation):
    def __init__(self, 
                 polygons, 
                 short_label_code, 
                 long_label_code, 
                 color, 
                 image_path, 
                 label_id, 
                 transparency=128, 
                 show_msg=False):
        """Initialize a MultiPolygonAnnotation with multiple polygons and related metadata."""
        super().__init__(short_label_code, long_label_code, color, image_path, label_id, transparency, show_msg)

        self.center_xy = QPointF(0, 0)
        self.cropped_bbox = (0, 0, 0, 0)
        self.annotation_size = 0
        
        self.polygons = [QPolygonF(poly) for poly in polygons]
        self.graphics_items = []
        
        self.set_centroid()
        self.set_cropped_bbox()
        
    # def select(self):
    #     """Mark the annotation as selected and update all graphics items."""
    #     self.is_selected = True
        
    #     # Update appearance of all polygon graphics items
    #     color = QColor(self.label.color)
    #     inverse_color = QColor(255 - color.red(), 255 - color.green(), 255 - color.blue())
    #     pen = QPen(inverse_color, 6, Qt.DotLine)
        
    #     for item in self.graphics_items:
    #         item.setPen(pen)
        
    #     # Update helper graphics using existing method
    #     self.update_graphics_item(crop_image=False)
        
    #     # Emit selected signal
    #     self.selected.emit(self)

    # def deselect(self):
    #     """Mark the annotation as not selected and update all graphics items."""
    #     self.is_selected = False
        
    #     # Update appearance of all polygon graphics items
    #     color = QColor(self.label.color)
    #     color.setAlpha(self.transparency)
    #     pen = QPen(color, 4, Qt.SolidLine)
        
    #     for item in self.graphics_items:
    #         item.setPen(pen)
        
    #     # Update helper graphics using existing method
    #     self.update_graphics_item(crop_image=False)
        
    # def delete(self):
    #     """Remove all graphics items associated with this multi-polygon annotation."""
    #     # Emit deletion signal first
    #     self.annotationDeleted.emit(self)
        
    #     # Remove all polygon graphics items
    #     for item in self.graphics_items:
    #         if item and item.scene():
    #             item.scene().removeItem(item)
    #     self.graphics_items = []
        
    #     # Remove helper graphics if they exist
    #     if self.center_graphics_item and self.center_graphics_item.scene():
    #         self.center_graphics_item.scene().removeItem(self.center_graphics_item)
    #         self.center_graphics_item = None
            
    #     if self.bounding_box_graphics_item and self.bounding_box_graphics_item.scene():
    #         self.bounding_box_graphics_item.scene().removeItem(self.bounding_box_graphics_item)
    #         self.bounding_box_graphics_item = None
            
    #     if self.polygon_graphics_item and self.polygon_graphics_item.scene():
    #         self.polygon_graphics_item.scene().removeItem(self.polygon_graphics_item)
    #         self.polygon_graphics_item = None
        
    #     # Clean up resources
    #     if self.cropped_image:
    #         del self.cropped_image
    #         self.cropped_image = None                       
        
    def simplify_polygons(self, epsilon: float):
        """Apply polygon simplification to each sub-polygon."""
        # Importing here to avoid circular dependency
        from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation

        for i, polygon in enumerate(self.polygons):
            points = [polygon.at(j) for j in range(polygon.count())]
            simplified = PolygonAnnotation.simplify_polygon(points, epsilon)
            self.polygons[i] = QPolygonF(simplified)
        self.set_cropped_bbox()
        
    def set_precision(self, reduce: bool = True):
        """Round coordinates of all polygons to 3 decimal places."""
        for i, polygon in enumerate(self.polygons):
            rounded_points = [QPointF(round(p.x(), 6), round(p.y(), 6)) for p in polygon]
            self.polygons[i] = QPolygonF(rounded_points)

    def set_centroid(self):
        """Compute and set the centroid of all polygons."""
        total_x, total_y, count = 0, 0, 0
        for polygon in self.polygons:
            if polygon.count() == 0:
                continue
            centroid_x = sum(p.x() for p in polygon) / polygon.count()
            centroid_y = sum(p.y() for p in polygon) / polygon.count()
            total_x += centroid_x
            total_y += centroid_y
            count += 1
        self.center_xy = QPointF(total_x / count, total_y / count) if count else QPointF(0, 0)

    def set_cropped_bbox(self):
        """Compute and set the bounding box that crops all polygons."""
        all_points = [p for poly in self.polygons for p in poly]
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
        return any(poly.containsPoint(point, Qt.OddEvenFill) for poly in self.polygons)

    def get_area(self):
        """Return the total area of all polygons."""
        total = 0.0
        for poly in self.polygons:
            n = poly.count()
            if n < 3: 
                continue
            # Calculate the area using the shoelace formula
            for i in range(n):
                j = (i + 1) % n
                area = poly.at(i).x() * poly.at(j).y()
                area -= poly.at(j).x() * poly.at(i).y()
            # Sum the area of the polygon
            total += abs(area) / 2.0
        return total

    def get_perimeter(self):
        """Return the total perimeter of all polygons."""
        total = 0.0
        for poly in self.polygons:
            n = poly.count()
            if n < 2: 
                continue
            # Calculate the perimeter using Euclidean distance
            for i in range(n):
                j = (i + 1) % n
                total += ((poly.at(i).x() - poly.at(j).x())**2 +
                          (poly.at(i).y() - poly.at(j).y())**2)**0.5
        return total
    
    def get_polygon(self):
        """Return the first polygon (for compatibility; use with caution)."""
        return self.polygons[0] if self.polygons else QPolygonF()  # TODO
    
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
        for polygon in self.polygons:
            cropped_points = [QPointF(p.x() - self.cropped_bbox[0], p.y() - self.cropped_bbox[1]) for p in polygon]
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
        pen = QPen(self.label.color)
        pen.setStyle(Qt.DashLine)
        pen.setWidth(2)
        result_painter.setPen(pen)
        result_painter.setClipping(False)
        for polygon in self.polygons:
            cropped_points = [QPointF(p.x() - self.cropped_bbox[0], p.y() - self.cropped_bbox[1]) for p in polygon]
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
            item = QGraphicsPolygonItem(poly)
            color = QColor(self.label.color)
            color.setAlpha(self.transparency)
            pen = QPen(color, 4, Qt.SolidLine)
            # If selected, use an inverse color and dotted line
            if self.is_selected:
                inverse_color = QColor(255 - color.red(), 255 - color.green(), 255 - color.blue())
                pen = QPen(inverse_color, 6, Qt.DotLine)
            item.setPen(pen)
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

    def update_graphics_item(self, crop_image=True):
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
            item = QGraphicsPolygonItem(poly)
            color = QColor(self.label.color)
            color.setAlpha(self.transparency)
            pen = QPen(color, 4, Qt.SolidLine)
            # If selected, use an inverse color and dotted line
            if self.is_selected:
                inverse_color = QColor(255 - color.red(), 255 - color.green(), 255 - color.blue())
                pen = QPen(inverse_color, 6, Qt.DotLine)
            item.setPen(pen)
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

    def update_location(self, new_center_xy: QPointF):
        """Move the annotation so its centroid is at new_center_xy."""
        delta = new_center_xy - self.center_xy
        for i, poly in enumerate(self.polygons):
            self.polygons[i] = QPolygonF([p + delta for p in poly])
        self.set_centroid()
        self.set_cropped_bbox()
        self.update_graphics_item()

    def update_annotation_size(self, delta: float):
        """Scale the annotation size by a factor of delta."""
        scale_factor = 1 + (delta - 1)  # Delta is 1-based scale factor
        for i, poly in enumerate(self.polygons):
            scaled_points = [QPointF(p.x() * scale_factor, p.y() * scale_factor) for p in poly]
            self.polygons[i] = QPolygonF(scaled_points)
        self.set_cropped_bbox()
        self.update_graphics_item()
        
    def cut(self, cutting_points: list = None):
        """Break apart the MultiPolygonAnnotation into individual PolygonAnnotations."""
        # Importing here to avoid circular dependency
        from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
        
        individual_annotations = []
        
        # Iterate over each polygon in the multi-polygon
        for polygon in self.polygons:
            # Extract points from the polygon
            points = [QPointF(point.x(), point.y()) for point in polygon]
            
            # Create a new PolygonAnnotation with the same properties
            new_anno = PolygonAnnotation(
                points=points,
                short_label_code=self.label.short_label_code,
                long_label_code=self.label.long_label_code,
                color=self.label.color,
                image_path=self.image_path,
                label_id=self.label.id,
                transparency=self.transparency,
                show_msg=self.show_message
            )
            
            # Transfer rasterio source if available
            if hasattr(self, 'rasterio_src') and self.rasterio_src is not None:
                new_anno.rasterio_src = self.rasterio_src
                new_anno.create_cropped_image(new_anno.rasterio_src)
            
            individual_annotations.append(new_anno)
        
        return individual_annotations

    def to_dict(self):
        """Return a dictionary representation of the annotation."""
        base = super().to_dict()
        base['polygons'] = [[(p.x(), p.y()) for p in poly] for poly in self.polygons]
        return base

    @classmethod
    def from_dict(cls, data, label_window):
        """Instantiate from a dictionary representation."""
        polygons = [[QPointF(x, y) for x, y in poly] for poly in data['polygons']]
        return cls(
            polygons=polygons,
            short_label_code=data['label_short_code'],
            long_label_code=data['label_long_code'],
            color=QColor(*data['annotation_color']),
            image_path=data['image_path'],
            label_id=data['label_id'],
            transparency=data.get('transparency', 128),
            show_msg=data.get('show_msg', False)
        )
    
    def to_yolo_segmentation(self, image_width, image_height):
        """Return YOLO segmentation format for the annotation.
        Each polygon is output as a separate line, with normalized coordinates.
        """
        lines = []
        for poly in self.polygons:
            if poly.count() == 0:
                continue
            formatted_points = []
            for i in range(poly.count()):
                formatted_points.append(f"{poly.at(i).x()/image_width:.6f} {poly.at(i).y()/image_height:.6f}")
            line = f"{self.label.short_label_code} " + " ".join(formatted_points)
            lines.append(line)
        return "\n".join(lines)

    def __repr__(self):
        """Return a string representation of the annotation."""
        return (f"MultiPolygonAnnotation({len(self.polygons)} polygons, "
                f"label={self.label.short_label_code}, "
                f"area={self.get_area():.2f}, "
                f"centroid=({self.center_xy.x():.1f}, {self.center_xy.y():.1f}))")