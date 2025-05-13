import warnings

import random

from PyQt5.QtCore import QRectF, QObject, pyqtSignal, Qt
from PyQt5.QtGui import QPen, QColor, QBrush, QPainterPath
from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsItemGroup, QGraphicsLineItem, QGraphicsPathItem

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class WorkArea(QObject):
    """
    Class representing a rectangular work area on an image.
    Stores the coordinates and provides methods for serialization and deserialization.
    Manages its own graphical representation.
    """
    
    removed = pyqtSignal(object)  # Signal emitted when work area is removed
    
    def __init__(self, x=0, y=0, width=0, height=0, image_path=None):
        """
        Initialize a work area with coordinates and dimensions.
        
        Args:
            x (float): X-coordinate of the top-left corner
            y (float): Y-coordinate of the top-left corner
            width (float): Width of the work area
            height (float): Height of the work area
            image_path (str): Path to the image this work area belongs to
        """
        super().__init__()
        
        self.rect = QRectF(x, y, width, height)
        self.image_path = image_path
        self.graphics_item = None  # Reference to the main graphics item in the scene
        self.remove_button = None  # Reference to the remove button graphics item
        self.shadow_area = None  # Reference to the shadow graphics item
        
        # Create a random color for the work area
        random_color = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.work_area_pen = QPen(random_color, 2, Qt.DashLine)  # Default style
        
    @classmethod
    def from_rect(cls, rect, image_path=None):
        """
        Create a WorkArea instance from a QRectF.
        
        Args:
            rect (QRectF): Rectangle defining the work area
            image_path (str): Path to the image this work area belongs to
            
        Returns:
            WorkArea: A new WorkArea instance
        """
        return cls(rect.x(), rect.y(), rect.width(), rect.height(), image_path)
        
    @classmethod
    def from_dict(cls, data, image_path=None):
        """
        Create a WorkArea instance from a dictionary.
        
        Args:
            data (dict): Dictionary containing x, y, width, and height values
            image_path (str): Path to the image this work area belongs to
            
        Returns:
            WorkArea: A new WorkArea instance
        """
        return cls(
            data.get('x', 0),
            data.get('y', 0),
            data.get('width', 0),
            data.get('height', 0),
            image_path or data.get('image_path')
        )
        
    def to_dict(self):
        """
        Convert the work area to a dictionary.
        
        Returns:
            dict: Dictionary representation of the work area
        """
        return {
            'x': self.rect.x(),
            'y': self.rect.y(),
            'width': self.rect.width(),
            'height': self.rect.height(),
            'image_path': self.image_path
        }
        
    def create_graphics(self, scene, pen_width=2, include_shadow=False):
        """
        Create and return the graphics representation of this work area.
        
        Args:
            scene: The QGraphicsScene to add the items to
            pen_width: Width of the pen for the rectangle
            include_shadow: Whether to include a shadow effect
            
        Returns:
            QGraphicsRectItem: The created rectangle item
        """
        # Create rectangle item if it doesn't exist
        if not self.graphics_item:
            self.graphics_item = QGraphicsRectItem(self.rect)
            self.work_area_pen.setWidth(pen_width)
            self.graphics_item.setPen(self.work_area_pen)
            
            # Store reference to the work area in the graphics item
            self.graphics_item.setData(0, "work_area")
            self.graphics_item.setData(1, self)
            
            # Add to scene
            scene.addItem(self.graphics_item)
        
        # Remove any existing shadow before creating a new one
        if self.shadow_area is not None and hasattr(self.shadow_area, "scene") and self.shadow_area.scene():
            self.shadow_area.scene().removeItem(self.shadow_area)
            self.shadow_area = None
            if self.graphics_item:
                self.graphics_item.setData(3, None)
        
        # Add shadow if requested
        if include_shadow:
            # Create a semi-transparent overlay for the shadow
            shadow_brush = QBrush(QColor(0, 0, 0, 150))  # Semi-transparent black
            shadow_path = QPainterPath()
            shadow_path.addRect(scene.sceneRect())  # Cover the entire scene
            shadow_path.addRect(self.rect)  # Add the work area rect
            # Subtract the work area from the overlay
            shadow_path = shadow_path.simplified()

            # Create the shadow item
            shadow_area = QGraphicsPathItem(shadow_path)
            shadow_area.setBrush(shadow_brush)
            shadow_area.setPen(QPen(Qt.NoPen))  # No outline for the shadow
            
            # Add shadow to scene with z-value below the work area
            scene.addItem(shadow_area)
            
            # Store reference to shadow
            self.shadow_area = shadow_area
            if self.graphics_item:
                self.graphics_item.setData(3, shadow_area)
        
        return self.graphics_item
        
    def create_remove_button(self, button_size=20, thickness=2):
        """
        Create a remove button (X) for this work area.
        
        Args:
            button_size: Size of the button
            thickness: Thickness of the X lines
            
        Returns:
            QGraphicsItemGroup: The button graphics group
        """
        if not self.graphics_item:
            return None
            
        # Create a group item to hold the X shape
        if not self.remove_button:
            self.remove_button = QGraphicsItemGroup(self.graphics_item)
            
            # Create two diagonal lines to form an X
            line1 = QGraphicsLineItem(0, 0, button_size, button_size, self.remove_button)
            line2 = QGraphicsLineItem(0, button_size, button_size, 0, self.remove_button)
            
            # Set the pen properties - thicker red lines
            red_pen = QPen(QColor(255, 0, 0), thickness)
            line1.setPen(red_pen)
            line2.setPen(red_pen)
            
            # Position in the top-right corner
            self.remove_button.setPos(self.rect.right() - button_size - 10, self.rect.top() + 10)
            
            # Store data to identify this button and its work area
            self.remove_button.setData(0, "remove_button")
            self.remove_button.setData(1, self)  # Store reference to the work area
            
            # Make the group item clickable
            self.remove_button.setAcceptedMouseButtons(Qt.LeftButton)
            
            # Store reference to the remove button in the graphics item
            self.graphics_item.setData(2, self.remove_button)
            
            # Override mousePressEvent for the button item
            def on_press(event):
                self.remove()
                
            self.remove_button.mousePressEvent = on_press
            
        return self.remove_button
    
    def add_to_scene(self, scene, pen_width=2, button_size=20):
        """
        Add this work area's graphics to the specified scene.
        Creates both the rectangle and remove button in one operation.
        
        Args:
            scene: The QGraphicsScene to add the items to
            pen_width: Width of the pen for the rectangle
            button_size: Size of the remove button
            
        Returns:
            bool: True if successfully added to the scene, False otherwise
        """
        # Create the main rectangle graphic
        graphics_item = self.create_graphics(scene, pen_width)
        if not graphics_item or not graphics_item.scene():
            return False
            
        # Create the remove button
        remove_button = self.create_remove_button(button_size, pen_width)
        if not remove_button:
            return False
            
        return True

    def remove_from_scene(self):
        """
        Remove this work area's graphics from its scene.
        
        Returns:
            bool: True if successfully removed, False if not in a scene
        """
        if self.shadow_area is not None and hasattr(self.shadow_area, "scene") and self.shadow_area.scene():
            self.shadow_area.scene().removeItem(self.shadow_area)
            self.shadow_area = None
        if self.graphics_item and self.graphics_item.scene():
            self.graphics_item.scene().removeItem(self.graphics_item)
            self.graphics_item = None
            self.remove_button = None
            return True
        return False
        
    def set_remove_button_visibility(self, visible):
        """
        Set the visibility of the remove button.
        
        Args:
            visible (bool): Whether the button should be visible
        """
        if self.remove_button:
            self.remove_button.setVisible(visible)
        
    def contains_point(self, point):
        """
        Check if the work area contains a point.
        
        Args:
            point: QPointF or point-like object with x() and y() methods
            
        Returns:
            bool: True if the point is inside the work area, False otherwise
        """
        return self.rect.contains(point)
        
    def intersects(self, other_rect):
        """
        Check if the work area intersects with another rectangle.
        
        Args:
            other_rect: QRectF or rect-like object
            
        Returns:
            bool: True if the rectangles intersect, False otherwise
        """
        return self.rect.intersects(other_rect)
        
    def is_valid(self):
        """
        Check if the work area has valid dimensions.
        
        Returns:
            bool: True if the work area is valid, False otherwise
        """
        return (self.rect.width() > 10 and 
                self.rect.height() > 10 and 
                self.image_path is not None)
                
    def remove(self):
        """Remove this work area and emit the removed signal."""
        # Remove graphics from scene
        self.remove_from_scene()
        
        # Emit the removed signal
        self.removed.emit(self)
            
    def __eq__(self, other):
        """
        Check if two work areas are equal.
        
        Args:
            other: Another WorkArea instance
            
        Returns:
            bool: True if the work areas have the same coordinates and image path
        """
        if not isinstance(other, WorkArea):
            return False
            
        return (
            abs(self.rect.x() - other.rect.x()) < 0.001 and
            abs(self.rect.y() - other.rect.y()) < 0.001 and
            abs(self.rect.width() - other.rect.width()) < 0.001 and
            abs(self.rect.height() - other.rect.height()) < 0.001 and
            self.image_path == other.image_path
        )
        
    def __hash__(self):
        """
        Generate a hash for the work area.
        
        Returns:
            int: Hash value
        """
        return hash((
            round(self.rect.x(), 3),
            round(self.rect.y(), 3),
            round(self.rect.width(), 3),
            round(self.rect.height(), 3),
            self.image_path
        ))