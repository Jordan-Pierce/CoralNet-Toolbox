import warnings

from PyQt5.QtCore import QRectF, QObject, pyqtSignal

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class WorkArea(QObject):
    """
    Class representing a rectangular work area on an image.
    Stores the coordinates and provides methods for serialization and deserialization.
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
        self.graphics_item = None  # Reference to the graphics item in the scene
        
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
        if self.graphics_item and self.graphics_item.scene():
            self.graphics_item.scene().removeItem(self.graphics_item)
            self.graphics_item = None
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