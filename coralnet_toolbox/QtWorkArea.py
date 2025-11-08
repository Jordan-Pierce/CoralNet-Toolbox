import warnings

from PyQt5.QtGui import QPen, QColor, QBrush, QPainterPath
from PyQt5.QtCore import QRectF, QObject, pyqtSignal, Qt, pyqtProperty
from PyQt5.QtWidgets import (QGraphicsRectItem, QGraphicsItemGroup, QGraphicsLineItem, QGraphicsPathItem)

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
        
        self.animation_manager = None  # Will hold the central manager
        self.is_animating = False      # Tracks animation state
        
        self.graphics_item = None  # Reference to the main graphics item in the scene
        self.remove_button = None  # Reference to the remove button graphics item
        self.shadow_area = None  # Reference to the shadow graphics item
        
        # Create a random color for the work area
        self.work_area_pen = QPen(QColor(0, 168, 230), 3, Qt.DotLine)  # Changed to static dotted line
        self.work_area_pen.setCosmetic(True)  # Ensure pen width is constant regardless of zoom
        
        # Store the original color for reverting
        self.original_color = QColor(0, 168, 230)
        # Animation properties (updated for pulsing)
        self._pulse_alpha = 128  # Starting alpha for pulsing (semi-transparent)
        self._pulse_direction = 1  # 1 for increasing alpha, -1 for decreasing
        
    @pyqtProperty(int)
    def pulse_alpha(self):
        """Get the current pulse alpha for animation."""
        return self._pulse_alpha
    
    @pulse_alpha.setter
    def pulse_alpha(self, value):
        """Set the pulse alpha and update pen styles."""
        self._pulse_alpha = int(max(0, min(255, value)))  # Clamp to 0-255 and convert to int
        self._update_pen_style()
    
    def tick_animation(self):
        """
        Update the pulse alpha for a heartbeat-like effect.
        This is now called by the GLOBAL timer in AnimationManager.
        """
        if self._pulse_direction == 1:
            # Quick increase (systole-like)
            self._pulse_alpha += 30
        else:
            # Slow decrease (diastole-like)
            self._pulse_alpha -= 10

        # Check direction before clamping to ensure smooth transition
        if self._pulse_alpha >= 255:
            self._pulse_alpha = 255  # Clamp to max
            self._pulse_direction = -1
        elif self._pulse_alpha <= 50:
            self._pulse_alpha = 50   # Clamp to min
            self._pulse_direction = 1
        
        # Update the pen style after the alpha is calculated
        self._update_pen_style()
    
    def _create_pen(self):
        """Create a pen with pulsing alpha and brighter color if animating."""
        pen = QPen(self.work_area_pen)
        pen.setCosmetic(True)
        
        # Check self.is_animating flag instead of timer
        if self.is_animating:
            pen_color = QColor(self.work_area_pen.color())
            pen_color.setAlpha(self._pulse_alpha)  # Apply pulsing alpha for animation
            pen.setColor(pen_color)
            
        pen.setStyle(Qt.DotLine)  # Predefined dotted line (static, no movement)
        return pen

    def animate(self):
        """Start the pulsing animation by registering with the global timer."""
        self.is_animating = True
        if self.animation_manager:
            self.animation_manager.register_animating_object(self)
    
    def deanimate(self):
        """Stop the pulsing animation by de-registering from the global timer."""
        self.is_animating = False
        if self.animation_manager:
            self.animation_manager.unregister_animating_object(self)
            
        self._pulse_alpha = 128  # Reset to default
        self._update_pen_style()  # Apply the default style
    
    def highlight(self):
        """Highlight the working area by turning its pen blood red."""
        self.work_area_pen.setColor(QColor(230, 62, 0))  # Blood red color
        self._update_pen_style()
        # Update immediately to reflect the change
        if self.graphics_item:
            self.graphics_item.update()
    
    def unhighlight(self):
        """Revert the working area pen back to the original color."""
        self.work_area_pen.setColor(self.original_color)
        self._update_pen_style()
        # Update immediately to reflect the change
        if self.graphics_item:
            self.graphics_item.update()

    def _update_pen_style(self):
        """Update the pen style of the graphics item with the current pulse alpha."""
        if self.graphics_item:
            self.graphics_item.setPen(self._create_pen())
            self.graphics_item.update()

    def set_animation_manager(self, manager):
        """
        Binds this object to the central AnimationManager.
        
        Args:
            manager (AnimationManager): The central animation manager instance.
        """
        self.animation_manager = manager

    def is_graphics_item_valid(self):
        """
        Checks if the graphics item is still valid and added to a scene.
        
        Returns:
            bool: True if the item exists and has a scene, False otherwise.
        """
        try:
            return self.graphics_item and self.graphics_item.scene()
        except RuntimeError:
            # This can happen if the C++ part of the item is deleted
            return False

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
        
    def create_graphics(self, scene, pen_width=3, include_shadow=False, animate=True):
        """
        Create and return the graphics representation of this work area.
        
        Args:
            scene: The QGraphicsScene to add the items to
            pen_width: Width of the pen for the rectangle
            include_shadow: Whether to include a shadow effect
            animate: Whether to start the pulsing animation
            
        Returns:
            QGraphicsRectItem: The created rectangle item
        """
        # Create rectangle item if it doesn't exist
        if not self.graphics_item:
            self.graphics_item = QGraphicsRectItem(self.rect)
            self.graphics_item.setPen(self._create_pen())
            
            # Store reference to the work area in the graphics item
            self.graphics_item.setData(0, "work_area")
            self.graphics_item.setData(1, self)
            
            # Add to scene
            scene.addItem(self.graphics_item)
            
            # It calls the new animate() which registers with the manager
            if animate:
                self.animate()
                
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
        
    def create_remove_button(self, button_size=10, thickness=4):
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
            red_pen = QPen(QColor(230, 62, 0), thickness)
            red_pen.setCosmetic(True)
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
    
    def add_to_scene(self, scene, pen_width=3, button_size=10):
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
        graphics_item = self.create_graphics(scene)
        if not graphics_item or not graphics_item.scene():
            return False
            
        # Create the remove button
        remove_button = self.create_remove_button()
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
            
            self.deanimate()
            return True
            
        return False
        
    def update_remove_button_appearance(self):
        """Update the remove button size and thickness to appear consistent at current zoom level."""
        if not self.remove_button or not self.graphics_item or not self.graphics_item.scene():
            return
            
        # Get the current view
        views = self.graphics_item.scene().views()
        if not views:
            return
        view = views[0]
        
        # Get current zoom scale
        scale = view.transform().m11()
        if scale == 0:
            scale = 1
            
        # Desired screen size for the button
        desired_screen_size = 20
        desired_margin = 10
        
        # Convert to scene coordinates
        scene_size = desired_screen_size / scale
        scene_margin = desired_margin / scale
        
        # Get the line items
        child_items = self.remove_button.childItems()
        if len(child_items) >= 2:
            line1 = child_items[0]  # Assuming first is line1
            line2 = child_items[1]  # Assuming second is line2
            
            # Update line positions
            line1.setLine(0, 0, scene_size, scene_size)
            line2.setLine(0, scene_size, scene_size, 0)
            
            # Update position relative to work area
            self.remove_button.setPos(
                self.rect.right() - scene_size - scene_margin, 
                self.rect.top() + scene_margin
            )
    
    def set_remove_button_visibility(self, visible):
        """
        Set the visibility of the remove button.
        
        Args:
            visible (bool): Whether the button should be visible
        """
        if self.remove_button:
            if visible:
                self.update_remove_button_appearance()
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
        self.deanimate()
        
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

    def __del__(self):
        """Clean up when the work area is deleted."""
        # Ensure it de-registers from the global timer if it's still animating
        if hasattr(self, 'is_animating') and self.is_animating:
            self.deanimate()