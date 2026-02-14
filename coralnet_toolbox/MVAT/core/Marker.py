"""
Marker Class for MVAT

Handles storing and emitting pixel positions (u, v) with associated colors for markers in the annotation window.
"""

from PyQt5.QtCore import QObject, Qt
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import QGraphicsItemGroup, QGraphicsEllipseItem, QGraphicsLineItem


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Marker(QObject):
    """
    Marker class for managing focal point markers in the annotation window.

    Stores the pixel location (u, v) and color, and manages the QGraphicsItemGroup for display.
    """

    def __init__(self):
        super().__init__()
        self.u = None
        self.v = None
        self.color = None
        
        # Create the QGraphicsItemGroup for the marker
        self.marker_item = QGraphicsItemGroup()
        
        # Create open circle (ellipse with no fill)
        self.ellipse = QGraphicsEllipseItem(-5, -5, 10, 10, self.marker_item)
        self.ellipse.setBrush(QBrush(Qt.NoBrush))  # No fill
        
        # Create crosshairs (lines extending outside the circle)
        # Horizontal line: from (-10, 0) to (10, 0) relative to center
        self.h_line = QGraphicsLineItem(-10, 0, 10, 0, self.marker_item)
        
        # Vertical line: from (0, -10) to (0, 10) relative to center
        self.v_line = QGraphicsLineItem(0, -10, 0, 10, self.marker_item)
        
        # Initially hide the marker
        self.marker_item.hide()

    def set_position(self, u: float, v: float, color: QColor):
        """
        Set the pixel position and color, and update the marker item.

        Args:
            u: X pixel coordinate.
            v: Y pixel coordinate.
            color: QColor for the marker.
        """
        self.u = u
        self.v = v
        self.color = color
        
        # Update the marker item
        self.marker_item.setPos(u, v)
        self.ellipse.setPen(color)
        self.h_line.setPen(color)
        self.v_line.setPen(color)
        self.marker_item.show()

    def hide(self):
        """
        Hide the marker.
        """
        self.u = None
        self.v = None
        self.color = None
        self.marker_item.hide()
