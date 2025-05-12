import os
import uuid
import warnings

import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal, QObject, QPointF
from PyQt5.QtGui import QColor, QImage, QPolygonF, QPen, QBrush
from PyQt5.QtWidgets import QMessageBox, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsPolygonItem, QGraphicsScene

from coralnet_toolbox.QtLabelWindow import Label

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Annotation(QObject):
    selected = pyqtSignal(object)
    colorChanged = pyqtSignal(QColor)
    annotationDeleted = pyqtSignal(object)
    annotationUpdated = pyqtSignal(object)

    def __init__(self, short_label_code: str,
                 long_label_code: str,
                 color: QColor,
                 image_path: str,
                 label_id: str,
                 transparency: int = 128,
                 show_msg=False):
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

        self.show_message = show_msg

        self.center_xy = None
        self.annotation_size = None

        # Attributes to store the graphics items for center/centroid, bounding box, and polygon
        self.center_graphics_item = None
        self.bounding_box_graphics_item = None
        self.polygon_graphics_item = None
        
    def contains_point(self, point: QPointF) -> bool:
        """Check if the annotation contains a given point."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def create_cropped_image(self, rasterio_src):
        """Create a cropped image from the annotation area."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_area(self):
        """Calculate the area of the annotation."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_perimeter(self):
        """Calculate the perimeter of the annotation."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_polygon(self):
        """Get the polygon representation of this annotation."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_bounding_box_top_left(self):
        """Get the top-left corner of the annotation's bounding box."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_bounding_box_bottom_right(self):
        """Get the bottom-right corner of the annotation's bounding box."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_cropped_image_graphic(self):
        """Get graphical representation of the cropped image area."""
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

    def deselect(self):
        """Mark the annotation as not selected and update its visual appearance."""
        self.is_selected = False
        self.update_graphics_item()

    def delete(self):
        """Remove the annotation and all associated graphics items from the scene."""

        # Emit the deletion signal first
        self.annotationDeleted.emit(self)

        # Remove the main graphics item
        if self.graphics_item and self.graphics_item.scene():
            self.graphics_item.scene().removeItem(self.graphics_item)
            self.graphics_item = None

        # Remove the center graphics item
        if self.center_graphics_item and self.center_graphics_item.scene():
            self.center_graphics_item.scene().removeItem(self.center_graphics_item)
            self.center_graphics_item = None

        # Remove the bounding box graphics item
        if self.bounding_box_graphics_item and self.bounding_box_graphics_item.scene():
            self.bounding_box_graphics_item.scene().removeItem(self.bounding_box_graphics_item)
            self.bounding_box_graphics_item = None

        # Remove the polygon graphics item
        if self.polygon_graphics_item and self.polygon_graphics_item.scene():
            self.polygon_graphics_item.scene().removeItem(self.polygon_graphics_item)
            self.polygon_graphics_item = None

        # Clean up resources
        if self.cropped_image:
            del self.cropped_image
            self.cropped_image = None
            
    def create_graphics_item(self, scene: QGraphicsScene):
        """Create all graphics items for the annotation and add them to the scene."""
        # Create the main graphics item based on the polygon
        polygon = self.get_polygon()
        
        self.graphics_item = QGraphicsPolygonItem(polygon)
        self.update_graphics_item()
        self.graphics_item.setData(0, self.id)
        
        scene.addItem(self.graphics_item)

        # Create the center graphics item
        self.create_center_graphics_item(self.center_xy, scene)
        
        # Create the bounding box graphics item
        self.create_bounding_box_graphics_item(self.get_bounding_box_top_left(), 
                                               self.get_bounding_box_bottom_right(),
                                               scene)
        # Create the polygon graphics item
        # Convert polygon to list of points
        points = [polygon.at(i) for i in range(polygon.count())]
        self.create_polygon_graphics_item(points, scene)
        
    def create_center_graphics_item(self, center_xy, scene):
        """Create a graphical item representing the annotation's center point."""
        if self.center_graphics_item and self.center_graphics_item.scene():
            self.center_graphics_item.scene().removeItem(self.center_graphics_item)

        color = QColor(self.label.color)
        if self.is_selected:
            color = QColor(255 - color.red(), 255 - color.green(), 255 - color.blue())
        color.setAlpha(self.transparency)

        self.center_graphics_item = QGraphicsEllipseItem(center_xy.x() - 5, center_xy.y() - 5, 10, 10)
        self.center_graphics_item.setBrush(color)
        scene.addItem(self.center_graphics_item)

    def create_bounding_box_graphics_item(self, top_left, bottom_right, scene):
        """Create a graphical item representing the annotation's bounding box."""
        if self.bounding_box_graphics_item and self.bounding_box_graphics_item.scene():
            self.bounding_box_graphics_item.scene().removeItem(self.bounding_box_graphics_item)

        color = QColor(self.label.color)
        if self.is_selected:
            color = QColor(255 - color.red(), 255 - color.green(), 255 - color.blue())
        color.setAlpha(self.transparency)

        self.bounding_box_graphics_item = QGraphicsRectItem(top_left.x(), top_left.y(),
                                                            bottom_right.x() - top_left.x(),
                                                            bottom_right.y() - top_left.y())

        self.bounding_box_graphics_item.setPen(color)
        scene.addItem(self.bounding_box_graphics_item)

    def create_polygon_graphics_item(self, points, scene):
        """Create a graphical item representing the annotation's polygon outline."""
        if self.polygon_graphics_item and self.polygon_graphics_item.scene():
            self.polygon_graphics_item.scene().removeItem(self.polygon_graphics_item)

        color = QColor(self.label.color)
        if self.is_selected:
            color = QColor(255 - color.red(), 255 - color.green(), 255 - color.blue())
        color.setAlpha(self.transparency)

        polygon = QPolygonF(points)
        self.polygon_graphics_item = QGraphicsPolygonItem(polygon)
        self.polygon_graphics_item.setBrush(color)
        scene.addItem(self.polygon_graphics_item)
            
    def get_center_xy(self):
        """Get the center coordinates of the annotation."""
        return self.center_xy

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

    def update_graphics_item(self, crop_image=True):
        """Update the graphical representation of the annotation."""
        if self.graphics_item and self.graphics_item.scene():
            scene = self.graphics_item.scene()
            if scene:
                scene.removeItem(self.graphics_item)

                # Get the polygon representation
                polygon = self.get_polygon()
                self.graphics_item = QGraphicsPolygonItem(polygon)

                # Set color and style
                color = QColor(self.label.color)
                color.setAlpha(self.transparency)

                if self.is_selected:
                    inverse_color = QColor(255 - color.red(), 255 - color.green(), 255 - color.blue())
                    pen = QPen(inverse_color, 6, Qt.DotLine)
                else:
                    pen = QPen(color, 4, Qt.SolidLine)

                self.graphics_item.setPen(pen)
                self.graphics_item.setBrush(QBrush(color))
                self.graphics_item.setData(0, self.id)
                scene.addItem(self.graphics_item)

                # Update separate graphics items
                self.update_center_graphics_item(self.center_xy)
                self.update_bounding_box_graphics_item(
                    self.get_bounding_box_top_left(),
                    self.get_bounding_box_bottom_right()
                )
                # Convert polygon to list of points
                points = [polygon.at(i) for i in range(polygon.count())]
                self.update_polygon_graphics_item(points)
                
    def update_center_graphics_item(self, center_xy):
        """Update the position and appearance of the center graphics item."""
        if self.center_graphics_item:
            color = QColor(self.label.color)
            if self.is_selected:
                color = QColor(255 - color.red(), 255 - color.green(), 255 - color.blue())
            color.setAlpha(self.transparency)

            self.center_graphics_item.setRect(center_xy.x() - 5, center_xy.y() - 5, 10, 10)
            self.center_graphics_item.setBrush(color)

    def update_bounding_box_graphics_item(self, top_left, bottom_right):
        """Update the position and appearance of the bounding box graphics item."""
        if self.bounding_box_graphics_item:
            color = QColor(self.label.color)
            if self.is_selected:
                color = QColor(255 - color.red(), 255 - color.green(), 255 - color.blue())
            color.setAlpha(self.transparency)

            self.bounding_box_graphics_item.setRect(top_left.x(), top_left.y(),
                                                    bottom_right.x() - top_left.x(),
                                                    bottom_right.y() - top_left.y())

            self.bounding_box_graphics_item.setPen(color)

    def update_polygon_graphics_item(self, points):
        """Update the shape and appearance of the polygon graphics item."""
        if self.polygon_graphics_item:
            color = QColor(self.label.color)
            if self.is_selected:
                color = QColor(255 - color.red(), 255 - color.green(), 255 - color.blue())
            color.setAlpha(self.transparency)

            polygon = QPolygonF(points)
            self.polygon_graphics_item.setPolygon(polygon)
            self.polygon_graphics_item.setBrush(color)

    def update_transparency(self, transparency: int):
        """Update the transparency value of the annotation's graphical representation."""
        if self.transparency != transparency:
            self.transparency = transparency
            self.update_graphics_item(crop_image=False)

    def update_label(self, new_label: 'Label', set_review=False):
        """Update the annotation's label while preserving confidence values."""
        # Initializing
        if self.label is None:
            self.label = new_label

        # Updating (new label, or new color)
        elif self.label.id != new_label.id or self.label.color != new_label.color:

            if not set_review:
                # Update the label in user_confidence if it exists
                if self.user_confidence:
                    old_confidence = next(iter(self.user_confidence.values()))
                    self.user_confidence = {new_label: old_confidence}

                # Update the label in machine_confidence if it exists
                if self.machine_confidence:
                    new_machine_confidence = {}
                    for label, confidence in self.machine_confidence.items():
                        if label.id == self.label.id:
                            new_machine_confidence[new_label] = confidence
                        else:
                            new_machine_confidence[label] = confidence

                    # Update the machine confidence
                    self.machine_confidence = new_machine_confidence

            # Update the label
            self.label = new_label

        # Always update the graphics item
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

        return {
            'Name': os.path.basename(self.image_path),
            'Row': int(self.center_xy.y()),
            'Column': int(self.center_xy.x()),
            'Patch Size': self.annotation_size,
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