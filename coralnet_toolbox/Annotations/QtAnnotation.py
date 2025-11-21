import os
import uuid
import warnings

import cv2
import math
import numpy as np

from PyQt5.QtGui import QColor, QPen, QBrush, QPolygonF
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QPointF, pyqtProperty
from PyQt5.QtWidgets import (QMessageBox, QGraphicsEllipseItem, QGraphicsRectItem,
                             QGraphicsScene, QGraphicsItemGroup)

from coralnet_toolbox.QtLabelWindow import Label

from coralnet_toolbox.utilities import convert_scale_units

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Annotation(QObject):
    selected = pyqtSignal(object)
    colorChanged = pyqtSignal(QColor)
    annotationDeleted = pyqtSignal(object)
    annotationUpdated = pyqtSignal(object)

    def __init__(self,
                 short_label_code: str,
                 long_label_code: str,
                 color: QColor,
                 image_path: str,
                 label_id: str,
                 transparency: int = 128,
                 show_msg: bool = False):
        """Initialize an annotation object with label and display properties."""
        super().__init__()
        self.id = str(uuid.uuid4())
        self.label = Label(short_label_code, long_label_code, color, label_id)
        self.image_path = image_path
        self.is_selected = False
        self.graphics_item = None
        self.transparency = transparency
        self.user_confidence = {self.label: 1.0}
        self.machine_confidence = {}
        self.verified = True
        self.data = {}
        self.rasterio_src = None
        self.cropped_image = None
        self._cached_cropped_image_graphic = None

        self.show_message = show_msg
    
        self.center_xy = None
        self.annotation_size = None
        self.tolerance = 0.1  # Default detail level for simplification/densification
        
        self.scale_x: float | None = None
        self.scale_y: float | None = None
        self.scale_units: str | None = None

        # Attributes to store the graphics items for center/centroid and bounding box
        self.center_graphics_item = None
        self.bounding_box_graphics_item = None

        # Group for all graphics items
        self.graphics_item_group = None
        
        # Animation attributes
        self.animation_manager = None
        self.is_animating = False
        
        # Animation properties
        self._pulse_alpha = 128  # Starting alpha for pulsing (semi-transparent)
        self._pulse_direction = 1  # 1 for increasing alpha, -1 for decreasing

    def contains_point(self, point: QPointF) -> bool:
        """Check if the annotation contains a given point."""
        raise NotImplementedError("Subclasses must implement this method.")

    def create_cropped_image(self, rasterio_src):
        """Create a cropped image from the annotation area."""
        # Clear cached graphic when creating new cropped image
        self._cached_cropped_image_graphic = None
        raise NotImplementedError("Subclasses must implement this method.")

    def get_area(self):
        """Calculate the area of the annotation."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_perimeter(self):
        """Calculate the perimeter of the annotation."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_scaled_area(self) -> tuple[float, str] | None:
        """
        Calculates the real-world area if scale is available.

        Returns:
            tuple (float, str): The scaled area and its units (e.g., 'm'),
                                or None if scale is not set.
        """
        if self.scale_x and self.scale_y and self.scale_units:
            try:
                pixel_area = self.get_area()
                scaled_area = pixel_area * (self.scale_x * self.scale_y)
                return scaled_area, self.scale_units
            except (NotImplementedError, TypeError):
                return None
        return None

    def get_scaled_perimeter(self) -> tuple[float, str] | None:
        """
        Calculates the real-world perimeter if scale is available.
        Assumes scale_x is the representative scale factor.

        Returns:
            tuple (float, str): The scaled perimeter and its units (e.g., 'm'),
                                or None if scale is not set.
        """
        if self.scale_x and self.scale_units:
            try:
                pixel_perimeter = self.get_perimeter()
                # Use scale_x as the primary factor.
                # Our ScaleTool sets x and y to be the same.
                scaled_perimeter = pixel_perimeter * self.scale_x
                return scaled_perimeter, self.scale_units
            except (NotImplementedError, TypeError):
                return None
        return None
    
    def _get_raster_slice_and_mask(self, full_raster_data: np.ndarray):
        """
        A helper method to extract a small slice of raster data (e.g., z-channel)
        and a corresponding boolean mask for the annotation's exact shape.

        Args:
            full_raster_data (np.ndarray): The full 2D raster data (e.g., z_channel).

        Returns:
            tuple (np.ndarray, np.ndarray):
                - data_slice: A 2D numpy array slice from the input data.
                - mask: A 2D boolean numpy array of the same shape as data_slice,
                        with True values inside the annotation's polygon.
        """
        # 1. Check for valid input
        if full_raster_data is None or full_raster_data.ndim != 2:
            dtype = full_raster_data.dtype if full_raster_data is not None else np.float32
            return np.array([[]], dtype=dtype), np.array([[]], dtype=bool)

        # 2. Get bounding box (as QPointF)
        top_left = self.get_bounding_box_top_left()
        bottom_right = self.get_bounding_box_bottom_right()

        if top_left is None or bottom_right is None:
            return np.array([[]], dtype=full_raster_data.dtype), np.array([[]], dtype=bool)

        # 3. Convert to integer pixel indices, ensuring we stay within raster bounds
        h, w = full_raster_data.shape
        x1 = max(0, int(math.floor(top_left.x())))
        y1 = max(0, int(math.floor(top_left.y())))
        x2 = min(w, int(math.ceil(bottom_right.x())))
        y2 = min(h, int(math.ceil(bottom_right.y())))

        # Check for invalid or zero-area slice
        if x1 >= x2 or y1 >= y2:
            return np.array([[]], dtype=full_raster_data.dtype), np.array([[]], dtype=bool)

        # 4. Slice the data
        data_slice = full_raster_data[y1:y2, x1:x2]

        # 5. Create the empty mask
        slice_h, slice_w = data_slice.shape
        mask = np.zeros((slice_h, slice_w), dtype=np.uint8)  # Use uint8 for cv2

        # 6. Get the polygon
        polygon = self.get_polygon()  # This is a QPolygonF
        if polygon is None or polygon.isEmpty():
            return data_slice, np.zeros_like(data_slice, dtype=bool)  # Return slice, but empty mask

        # 7. Translate polygon points
        # Convert QPolygonF to numpy array
        points = []
        for i in range(polygon.count()):
            p = polygon.at(i)
            points.append([p.x(), p.y()])
        np_points = np.array(points)

        # Translate points to be relative to the slice's origin (x1, y1)
        translated_points = np_points - np.array([x1, y1])

        # 8. Draw the polygon mask
        # cv2.fillPoly needs points in int32 format
        cv2.fillPoly(mask, [translated_points.astype(np.int32)], color=1)

        # 9. Return slice and boolean mask
        return data_slice, mask.astype(bool)

    def get_scaled_volume(self, z_channel: np.ndarray, scale_x: float, scale_y: float, z_unit: str = None) -> float | None:
        """
        Calculates the 'volume' under the annotation relative to a Z=0 plane.
        Requires the full z_channel (depth/elevation) data and scale factors.

        Args:
            z_channel (np.ndarray): The full 2D z_channel data from the Raster.
            scale_x (float): The horizontal scale (e.g., meters per pixel).
            scale_y (float): The vertical scale (e.g., meters per pixel).
            z_unit (str, optional): The unit of the z_channel data (e.g., 'mm', 'cm', 'ft'). 
                                   If provided, z-values will be converted to meters.

        Returns:
            float | None: The calculated volume in cubic meters, 0.0 for zero area, or None on input error.
        """
        # 1. Check for valid inputs
        if z_channel is None or scale_x is None or scale_y is None:
            return None

        try:
            # 2. Get the sliced data and mask
            z_slice, mask = self._get_raster_slice_and_mask(z_channel)

            # 3. Check for valid data
            if z_slice.size == 0 or mask.size == 0 or not np.any(mask):
                return 0.0  # No area, so volume is 0

            # 4. Convert z_slice to meters if necessary
            # This ensures all dimensions are in the same unit (meters)
            if z_unit:
                z_to_meters_factor = convert_scale_units(1.0, z_unit, 'metre')
                z_slice_meters = z_slice * z_to_meters_factor
            else:
                z_slice_meters = z_slice  # Assume already in meters if no unit provided

            # 5. Select Z-values inside the mask (now in meters)
            z_values_inside = z_slice_meters[mask]

            # 6. Calculate 2D pixel area (in square meters)
            pixel_area_2d = scale_x * scale_y

            # 7. Calculate total volume (in cubic meters)
            # This is the sum of (pixel_area * pixel_height)
            total_volume = np.sum(z_values_inside) * pixel_area_2d

            return total_volume
        except Exception as e:
            print(f"Error calculating scaled volume for annotation {self.id}: {e}")
            return None

    def get_scaled_surface_area(self, z_channel: np.ndarray, scale_x: float, 
                               scale_y: float, z_unit: str = None) -> float | None:
        """
        Calculates the 3D surface area of the annotation using gradients.
        Requires the full z_channel (depth/elevation) data and scale factors.
        If z_channel is not provided, falls back to 2D scaled area.

        Args:
            z_channel (np.ndarray): The full 2D z_channel data from the Raster.
            scale_x (float): The horizontal scale (e.g., meters per pixel).
            scale_y (float): The vertical scale (e.g., meters per pixel).
            z_unit (str, optional): The unit of the z_channel data (e.g., 'mm', 'cm', 'ft'). 
                                   If provided, z-values will be converted to meters.

        Returns:
            float | None: The calculated 3D surface area in square meters, 2D area as fallback, 
                         or None on error.
        """
        # 1. Check for scale. If no scale, we can't do anything.
        if scale_x is None or scale_y is None:
            return None

        # 2. Fallback to 2D area if Z-channel is missing
        if z_channel is None:
            scaled_area_data = self.get_scaled_area()
            if scaled_area_data:
                return scaled_area_data[0]  # Return just the value
            else:
                return None  # No scale, no 2D area, no 3D area

        try:
            # 3. Get the sliced data and mask
            z_slice, mask = self._get_raster_slice_and_mask(z_channel)

            # 4. Check for valid data
            if z_slice.size == 0 or mask.size == 0 or not np.any(mask):
                return 0.0  # No area, so surface area is 0

            # 5. Convert z_slice to meters if necessary
            # This ensures all dimensions are in the same unit (meters)
            if z_unit:
                from coralnet_toolbox.utilities import convert_scale_units
                z_to_meters_factor = convert_scale_units(1.0, z_unit, 'metre')
                z_slice_meters = z_slice * z_to_meters_factor
            else:
                z_slice_meters = z_slice  # Assume already in meters if no unit provided

            # 6. Calculate 2D pixel area (in square meters)
            pixel_area_2d = scale_x * scale_y

            # 7. Calculate gradients (slope) of the slice with proper spacing
            # All dimensions now in meters for dimensional consistency
            dz_dy, dz_dx = np.gradient(z_slice_meters, scale_y, scale_x)

            # 8. Calculate the 3D area multiplier for each pixel
            # This is the surface area of a 3D plane, derived from vector cross product
            # Area = sqrt(1 + (dz/dx)^2 + (dz/dy)^2) * dx * dy
            multiplier = np.sqrt(1.0 + dz_dx**2 + dz_dy**2)

            # 9. Calculate the 3D surface area for *all* pixels in the slice
            pixel_areas_3d = pixel_area_2d * multiplier

            # 10. Select only the 3D areas *inside* the polygon and sum them
            total_surface_area = np.sum(pixel_areas_3d[mask])

            return total_surface_area
        except Exception as e:
            print(f"Error calculating scaled surface area for annotation {self.id}: {e}")
            # Fallback to 2D area on error
            scaled_area_data = self.get_scaled_area()
            return scaled_area_data[0] if scaled_area_data else None

    def get_polygon(self):
        """Get the polygon representation of this annotation."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_painter_path(self):
        """Get the QPainterPath representation of this annotation."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_bounding_box_top_left(self):
        """Get the top-left corner of the annotation's bounding box."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_bounding_box_bottom_right(self):
        """Get the bottom-right corner of the annotation's bounding box."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_cropped_image_graphic(self):
        """Get graphical representation of the cropped image area."""
        # Return cached version if available
        if self._cached_cropped_image_graphic is not None:
            return self._cached_cropped_image_graphic
            
        # Create and cache the graphic
        graphic = self._create_cropped_image_graphic()
        self._cached_cropped_image_graphic = graphic
        return graphic
    
    def _create_cropped_image_graphic(self):
        """Create the graphical representation - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")

    def update_polygon(self, delta):
        """Update the polygon representation of the annotation."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def update_location(self, new_center_xy: QPointF):
        """Update the location of the annotation to a new center point."""
        raise NotImplementedError("Subclasses must implement this method.")

    def update_annotation_size(self, size_or_scale_factor):
        """Update the size of the annotation using a size or scale factor."""
        raise NotImplementedError("Subclasses must implement this method.")

    def resize(self, handle: str, new_pos: QPointF):
        """Resize the annotation based on handle position."""
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def combine(cls, annotations: list):
        """Combine multiple annotations into one."""
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def cut(cls, annotations: list, cutting_points: list):
        """Cut multiple annotations using specified cutting points."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    @classmethod
    def subtract(cls, base_annotation, cutter_annotations: list):
        """Subtract cutter annotations from a base annotation."""
        raise NotImplementedError("Subclasses must implement this method.")

    def show_warning_message(self):
        """Display a warning message about removing machine suggestions when altering an annotation."""
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("Warning")
        msg_box.setText("Altering an annotation with predictions will remove the machine suggestions.")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def select(self):
        """Mark the annotation as selected and update its visual appearance."""
        self.is_selected = True
        self.update_graphics_item()
        # Start animation
        self.animate()

    def deselect(self):
        """Mark the annotation as not selected and update its visual appearance."""
        self.is_selected = False
        # Stop animation
        self.deanimate()
        self.update_graphics_item()

    def delete(self):
        """Remove the annotation and all associated graphics items from the scene."""
        # Stop animation
        self.deanimate()
        
        # Emit the deletion signal first
        self.annotationDeleted.emit(self)

        # Remove the graphics item group if it exists
        if self.graphics_item_group and self.graphics_item_group.scene():
            self.graphics_item_group.scene().removeItem(self.graphics_item_group)
            self.graphics_item_group = None

        # Clean up resources
        if self.cropped_image:
            del self.cropped_image
            self.cropped_image = None

        # Remove references to individual items
        self.graphics_item = None
        self.center_graphics_item = None
        self.bounding_box_graphics_item = None

    def create_graphics_item(self, scene: QGraphicsScene):
        """Create all graphics items for the annotation and add them to the scene as a group."""
        # Remove old group if it exists
        if self.graphics_item_group and self.graphics_item_group.scene():
            self.graphics_item_group.scene().removeItem(self.graphics_item_group)
            self.center_graphics_item = None
            self.bounding_box_graphics_item = None
        self.graphics_item_group = QGraphicsItemGroup()

        # The subclass has already created self.graphics_item.
        # This parent method is now only responsible for styling and grouping it.
        if self.graphics_item:
            color = QColor(self.label.color)
            color.setAlpha(self.transparency)
            # Only set brush and pen if the item supports them (e.g., shape items, not pixmaps)
            if hasattr(self.graphics_item, 'setBrush'):
                self.graphics_item.setBrush(QBrush(color))
            if hasattr(self.graphics_item, 'setPen'):
                self.graphics_item.setPen(self._create_pen(color))
            
            self.graphics_item.setData(0, self.id)
            self.graphics_item_group.addToGroup(self.graphics_item)

        # Create and group the helper graphics (center, bbox, etc.)
        self.create_center_graphics_item(self.center_xy, scene, add_to_group=True)
        self.create_bounding_box_graphics_item(self.get_bounding_box_top_left(),
                                               self.get_bounding_box_bottom_right(),
                                               scene, add_to_group=True)
        
        # Add the final group to the scene
        scene.addItem(self.graphics_item_group)
        
    def is_graphics_item_valid(self):
        """
        Checks if the graphics item group is still valid and added to a scene.
        
        Returns:
            bool: True if the item exists and has a scene, False otherwise.
        """
        try:
            # Check the group, as it's the parent of all other items
            return self.graphics_item_group and self.graphics_item_group.scene()
        except RuntimeError:
            # This can happen if the C++ part of the item is deleted
            return False
        
    def set_visibility(self, visible):
        """Set the visibility of this annotation's graphics item."""
        if self.graphics_item_group:
            self.graphics_item_group.setVisible(visible)
            
    def set_animation_manager(self, manager):
        """
        Binds this object to the central AnimationManager.
        
        Args:
            manager (AnimationManager): The central animation manager instance.
        """
        self.animation_manager = manager
        
    def tick_animation(self):
        """
        Perform one 'tick' of the animation.
        This is the public entry point for the global manager.
        """
        # This just calls the existing private method that holds the logic
        self._update_pulse_alpha()
        
    @pyqtProperty(int)  # Changed to int for QColor compatibility
    def pulse_alpha(self):
        """Get the current pulse alpha for animation."""
        return self._pulse_alpha
    
    @pulse_alpha.setter
    def pulse_alpha(self, value):
        """Set the pulse alpha and update pen styles."""
        self._pulse_alpha = int(max(0, min(255, value)))  # Clamp to 0-255 and convert to int
        self._update_pen_styles()
    
    def _update_pulse_alpha(self):
        """Update the pulse alpha for a heartbeat-like effect: quick rise, slow fall."""
        if self._pulse_direction == 1:
            # Quick increase (systole-like)
            self._pulse_alpha += 30
        else:
            # Slow decrease (diastole-like)
            self._pulse_alpha -= 10  # <-- Corrected from += to -=

        # Check direction before clamping to ensure smooth transition
        if self._pulse_alpha >= 255:
            self._pulse_alpha = 255  # Clamp to max
            self._pulse_direction = -1
        elif self._pulse_alpha <= 50:
            self._pulse_alpha = 50   # Clamp to min
            self._pulse_direction = 1

        self._update_pen_styles()
    
    def animate(self, force=False):
        """
        Start the pulsing animation by registering with the global timer.
        
        Args:
            force (bool): If True, force animation even if annotation is not selected
        """
        if force or self.is_selected:
            self.is_animating = True
            if self.animation_manager:
                self.animation_manager.register_animating_object(self)
    
    def deanimate(self):
        """Stop the pulsing animation by de-registering from the global timer."""
        self.is_animating = False
        if self.animation_manager:
            self.animation_manager.unregister_animating_object(self)
            
        self._pulse_alpha = 128  # Reset to default
        self.update_graphics_item()  # Apply the default style
    
    def _create_pen(self, base_color: QColor) -> QPen:
        """Create a pen with appropriate style based on selection state."""
        if self.is_selected or self.is_animating:
            # Use same color if verified, black if not verified
            if self.verified:
                pen_color = QColor(base_color)  # Create a copy
            else:
                pen_color = QColor(0, 0, 0)  # Black
            
            pen_color.setAlpha(self._pulse_alpha)  # Apply pulsing alpha for animation
            
            pen = QPen(pen_color.lighter(150), 4)  # Changed to lighter for brighter selected appearance
            pen.setStyle(Qt.DotLine)  # Predefined dotted line
            pen.setCosmetic(True)
            return pen
        else:
            # Use label color with solid line for unselected items, always opaque
            pen_color = QColor(base_color)
            pen_color.setAlpha(255)  # Pen should always be fully opaque
            pen = QPen(pen_color, 2, Qt.SolidLine)  # Consistent width
            pen.setCosmetic(True)
            return pen
    
    def _update_pen_styles(self):
        """Update pen styles with current pulse alpha."""
        # Only update if selected OR if animation is running (for forced animation)
        if not self.is_selected and not self.is_animating:
            return
            
        color = QColor(self.label.color).lighter(150) if not self.verified else QColor(self.label.color)  
        pen = self._create_pen(color)
        
        # Update all graphics items with the pen
        if self.graphics_item:
            self.graphics_item.setPen(pen)
        if self.center_graphics_item:
            self.center_graphics_item.setPen(pen)
        if self.bounding_box_graphics_item:
            self.bounding_box_graphics_item.setPen(pen)

    def create_center_graphics_item(self, center_xy, scene, add_to_group=False):
        """Create a graphical item representing the annotation's center point."""
        # First safely check if the center_graphics_item is still valid
        try:
            has_scene = self.center_graphics_item and self.center_graphics_item.scene()
        except RuntimeError:
            self.center_graphics_item = None
            has_scene = False
    
        if has_scene:
            self.center_graphics_item.scene().removeItem(self.center_graphics_item)
    
        color = QColor(self.label.color)
        color.setAlpha(self.transparency)
    
        self.center_graphics_item = QGraphicsEllipseItem(center_xy.x() - 4, center_xy.y() - 4, 8, 8)
        self.center_graphics_item.setBrush(color)
    
        # Use the consolidated pen creation method
        self.center_graphics_item.setPen(self._create_pen(color))
    
        if add_to_group and self.graphics_item_group:
            self.graphics_item_group.addToGroup(self.center_graphics_item)
        else:
            scene.addItem(self.center_graphics_item)
    
    def create_bounding_box_graphics_item(self, top_left, bottom_right, scene, add_to_group=False):
        """Create a graphical item representing the annotation's bounding box."""
        try:
            has_scene = self.bounding_box_graphics_item and self.bounding_box_graphics_item.scene()
        except RuntimeError:
            self.bounding_box_graphics_item = None
            has_scene = False
    
        if has_scene:
            self.bounding_box_graphics_item.scene().removeItem(self.bounding_box_graphics_item)
    
        color = QColor(self.label.color)
        color.setAlpha(self.transparency)
    
        self.bounding_box_graphics_item = QGraphicsRectItem(
            top_left.x(), top_left.y(),
            bottom_right.x() - top_left.x(),
            bottom_right.y() - top_left.y()
        )
    
        # Use the consolidated pen creation method
        self.bounding_box_graphics_item.setPen(self._create_pen(color))
    
        if add_to_group and self.graphics_item_group:
            self.graphics_item_group.addToGroup(self.bounding_box_graphics_item)
        else:
            scene.addItem(self.bounding_box_graphics_item)

    def get_center_xy(self):
        """Get the center coordinates of the annotation."""
        return self.center_xy
    
    def get_class_statistics(self) -> dict:
        """
        Returns a dictionary with pixel counts and percentages for each class
        contained within this single annotation.
        
        For vector types (Patch, Rectangle, Polygon), this will simply be
        a dictionary with one entry for its own label. For mask types, this
        will return statistics for all classes found within the mask.
        """
        # Default implementation for simple, single-label vector annotations.
        # Subclasses like MaskAnnotation will provide a more complex override.
        pixel_count = self.get_area()
        return {
            self.label.short_label_code: {
                "pixel_count": int(pixel_count),
                "percentage": 100.0
            }
        }

    def get_cropped_image(self, max_size=None):
        """Retrieve the cropped image, optionally scaled to maximum size."""
        if self.cropped_image is None:
            return None

        if max_size is not None:
            current_width = self.cropped_image.width()
            current_height = self.cropped_image.height()

            # Calculate scaling factor while maintaining aspect ratio
            width_ratio = max_size / current_width
            height_ratio = max_size / current_height
            scale_factor = min(width_ratio, height_ratio)

            # Only scale if image is larger than max_size
            if scale_factor < 1.0:
                new_width = int(current_width * scale_factor)
                new_height = int(current_height * scale_factor)
                return self.cropped_image.scaled(new_width, new_height)

        return self.cropped_image

    def update_graphics_item(self):
        """Update the graphical representation of the annotation."""
        # Try to get the scene from the current group, else do nothing
        scene = None
        try:
            if self.graphics_item_group and self.graphics_item_group.scene():
                scene = self.graphics_item_group.scene()
                
        except RuntimeError:
            self.graphics_item_group = None
            scene = None
            
        if scene is not None:
            # Remove the old group from the scene
            scene.removeItem(self.graphics_item_group)
            self.graphics_item_group = None
            # Re-create all graphics using the existing scene
            self.create_graphics_item(scene)

    def update_center_graphics_item(self, center_xy):
        """Update the position and appearance of the center graphics item."""
        if self.center_graphics_item:
            color = QColor(self.label.color)
            color.setAlpha(self.transparency)
    
            self.center_graphics_item.setRect(center_xy.x() - 5, center_xy.y() - 5, 10, 10)
            self.center_graphics_item.setBrush(color)
    
            # Use the consolidated pen creation method
            self.center_graphics_item.setPen(self._create_pen(color))
    
    def update_bounding_box_graphics_item(self, top_left, bottom_right):
        """Update the position and appearance of the bounding box graphics item."""
        if self.bounding_box_graphics_item:
            color = QColor(self.label.color)
            color.setAlpha(self.transparency)
    
            self.bounding_box_graphics_item.setRect(top_left.x(), top_left.y(),
                                                    bottom_right.x() - top_left.x(),
                                                    bottom_right.y() - top_left.y())
    
            # Use the consolidated pen creation method
            self.bounding_box_graphics_item.setPen(self._create_pen(color))
    
    def update_transparency(self, transparency: int):
        """Update the transparency value of the annotation's graphical representation."""
        if self.transparency != transparency:
            # Update the transparency value
            self.transparency = transparency
            # If the annotation is visible, update the graphics item
            if self.graphics_item_group is not None and self.graphics_item_group.isVisible():
                self.update_graphics_item()

    def update_label(self, new_label: 'Label'):
        """
        Updates the annotation's label. This method correctly handles:
        1.  Label re-assignment (merges): Re-assigns the annotation and cleans up confidence scores.
        2.  Label property edits: Updates confidence dictionary keys if label codes change.
        3.  Simple property changes (e.g., color): Efficiently updates graphics.
        """
        if self.label is None:
            self.label = new_label
            self.update_graphics_item()
            return

        # Check if the label's semantic identity has changed (based on __eq__ in Label class)
        if self.label != new_label:
            
            old_label = self.label
            
            # --- Handle User Confidence ---
            # The user's choice is now `new_label`. We transfer the confidence value.
            if self.user_confidence:
                old_confidence_value = self.user_confidence.pop(old_label, None)
                if old_confidence_value is not None:
                    self.user_confidence = {new_label: old_confidence_value}
            
            # --- Handle Machine Confidence ---
            # The prediction for `old_label` is now irrelevant after the merge/edit.
            # We remove it, preserving any other machine predictions.
            if self.machine_confidence:
                # The .pop() method safely removes the key if it exists, and does nothing otherwise.
                self.machine_confidence.pop(old_label, None)

            # Finally, update the annotation's primary label reference
            self.label = new_label

        # If only the color or other non-identifying properties changed, the `if` block is skipped.
        # We still need to make sure the label object reflects the latest color.
        self.label.color = new_label.color

        # Always update the graphics to reflect any change.
        self.update_graphics_item()

    def update_user_confidence(self, new_label: 'Label'):
        """Update annotation with user-defined label and confidence."""
        # Mark as verified, keep machine confidence
        self.verified = True
        # Update user confidence
        self.user_confidence = {new_label: 1.0}
        # Pass the label with the largest confidence as the label
        self.label = new_label

        # Create the graphic
        self.update_graphics_item()
        self.show_message = False
        
    def update_machine_confidence(self, prediction: dict, from_import: bool = False):
        """Update annotation with machine-generated confidence scores."""
        if not prediction:
            return

        # Convert any numpy numeric types to regular Python float for JSON compatibility
        prediction = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in prediction.items()}
        # Sort the prediction by confidence (descending order)
        prediction = {k: v for k, v in sorted(prediction.items(), key=lambda item: item[1], reverse=True)}
        # Update machine confidence
        self.machine_confidence = prediction

        if not from_import:
            # Set user confidence to None
            self.user_confidence = {}
            # Pass the label with the largest confidence as the label
            self.label = max(prediction, key=prediction.get)
            # Mark as not verified
            self.verified = False

            # Create the graphic
            self.update_graphics_item()
            self.show_message = True

    def set_verified(self, verified: bool):
        """Set the verified status of the annotation.
        This method is called on importing annotations to set the verified status."""
        # Update the verified status
        self.verified = verified

    def update_verified(self, verified: bool):
        """Update the verified status of the annotation, and update user confidence if necessary.
        This method is called when the user verifies or un-verifies an annotation."""
        # If the verified status is the same as the current one, do nothing
        if verified == self.verified:
            return

        if verified:
            # If the annotation is being verified and there are machine confidence scores,
            # update the user confidence to the maximum machine confidence
            if self.machine_confidence:
                self.update_user_confidence(max(self.machine_confidence, key=self.machine_confidence.get))
            else:
                # If there are no machine confidence scores, just mark it as verified
                # and keep the current label
                self.verified = True
                self.user_confidence = {self.label: 1.0}
                self.update_graphics_item()
        else:
            # If the annotation is being unverified, set verified to false and clear user confidence
            self.verified = False
            self.user_confidence = {}
            # Keep machine confidence as the source of truth when unverified
            if self.machine_confidence:
                self.label = max(self.machine_confidence, key=self.machine_confidence.get)
            self.update_graphics_item()
            self.show_message = True
            
    def to_nms_detection(self):
        """Convert annotation to NMS-compatible detection format.
        
        Returns a dictionary with bounding box coordinates, confidence score,
        class information, and reference to the original annotation.
        """
        # Get bounding box directly from existing methods - much more efficient!
        top_left = self.get_bounding_box_top_left()
        bottom_right = self.get_bounding_box_bottom_right()
        
        # Get confidence score (prefer machine confidence if available, otherwise use 1.0)
        confidence = 1.0  # Default for existing annotations
        if self.machine_confidence:
            # Use the highest machine confidence score
            confidence = max(self.machine_confidence.values())
        elif self.user_confidence:
            # Use user confidence if no machine confidence
            confidence = max(self.user_confidence.values())
        
        return {
            'bbox': [top_left.x(), top_left.y(), bottom_right.x(), bottom_right.y()],  # xyxy format
            'confidence': float(confidence),
            'class_name': self.label.short_label_code,
            'class_id': self.label.id,
            'annotation': self,  # Keep reference to original
            'is_existing': True,
            'area': self.get_area()
        }

    def to_coralnet(self):
        """Convert annotation to CoralNet format for export."""
        # Extract machine confidence values and suggestions
        confidences = [f"{confidence:.3f}" for confidence in self.machine_confidence.values()]
        suggestions = [suggestion.short_label_code for suggestion in self.machine_confidence.keys()]

        # Pad with NaN if there are fewer than 5 values
        while len(confidences) < 5:
            confidences.append(np.nan)
        while len(suggestions) < 5:
            suggestions.append(np.nan)
            
        # Default to pixel values and "pixels" units
        area_val = self.get_area()
        perimeter_val = self.get_perimeter()
        units_val = "pixels"
        
        # Overwrite with scaled values if available
        scaled_area = self.get_scaled_area()
        if scaled_area:
            area_val = scaled_area[0]
            units_val = scaled_area[1]  # e.g., "m"

        scaled_perimeter = self.get_scaled_perimeter()
        if scaled_perimeter:
            perimeter_val = scaled_perimeter[0]

        return {
            'Name': os.path.basename(self.image_path),
            'Path': self.image_path,
            'Row': int(self.center_xy.y()),
            'Column': int(self.center_xy.x()),
            'Patch Size': self.annotation_size,
            'Area': area_val,
            'Perimeter': perimeter_val,
            'Units': units_val,
            'Annotation Type': type(self).__name__.replace('Annotation', ''),
            'Label': self.label.short_label_code,
            'Long Label': self.label.long_label_code,
            'Verified': self.verified,
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
        """Convert annotation to dictionary format for serialization."""
        # Convert machine_confidence keys to short_label_code
        machine_confidence = {label.short_label_code: confidence for label, confidence in
                              self.machine_confidence.items()}

        # Add common attributes
        result = {
            'id': self.id,
            'label_short_code': self.label.short_label_code,
            'label_long_code': self.label.long_label_code,
            'annotation_color': self.label.color.getRgb(),
            'image_path': self.image_path,
            'label_id': self.label.id,
            'data': self.data,
            'machine_confidence': machine_confidence,
            'verified': self.verified,
            'area': self.get_area(),
            'perimeter': self.get_perimeter(),
        }
        
        # Add scaled values if they exist
        scaled_area = self.get_scaled_area()
        if scaled_area:
            result['scaled_area'] = scaled_area[0]
            result['area_units'] = scaled_area[1]

        scaled_perimeter = self.get_scaled_perimeter()
        if scaled_perimeter:
            result['scaled_perimeter'] = scaled_perimeter[0]
            result['perimeter_units'] = scaled_perimeter[1]

        return result

    def to_yolo_detection(self, image_width, image_height):
        """Convert annotation to YOLO detection format.

        YOLO detection format is: class_id x_center y_center width height
        where all values are normalized to [0, 1].
        """
        # Get the bounding box
        top_left = self.get_bounding_box_top_left()
        bottom_right = self.get_bounding_box_bottom_right()

        # Calculate normalized center, width, and height
        x_center = (top_left.x() + bottom_right.x()) / 2 / image_width
        y_center = (top_left.y() + bottom_right.y()) / 2 / image_height
        width = (bottom_right.x() - top_left.x()) / image_width
        height = (bottom_right.y() - top_left.y()) / image_height

        return self.label.short_label_code, f"{x_center} {y_center} {width} {height}"

    def to_yolo_segmentation(self, image_width, image_height):
        """Convert annotation to YOLO segmentation format.

        YOLO segmentation format is: class_id x1 y1 x2 y2 ... xn yn
        where all coordinates are normalized to [0, 1].
        """
        # Get the polygon
        polygon = self.get_polygon()

        # Normalize the points to [0,1] range
        normalized_points = []
        for i in range(polygon.count()):
            point = polygon.at(i)
            normalized_points.append((point.x() / image_width, point.y() / image_height))

        # Format as a string with alternating x y coordinates
        points_str = " ".join([f"{x} {y}" for x, y in normalized_points])

        return self.label.short_label_code, points_str

    def __repr__(self):
        """Return a string representation of the annotation."""
        return (f"Annotation(id={self.id}, "
                f"annotation_color={self.label.color.name()}, "
                f"image_path={self.image_path}, "
                f"label={self.label.short_label_code}, "
                f"data={self.data}, "
                f"machine_confidence={self.machine_confidence})")
