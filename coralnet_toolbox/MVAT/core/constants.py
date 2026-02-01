"""
MVAT Color Constants

Shared color definitions for consistent theming across MVAT components.
Colors are defined in multiple formats for compatibility with different rendering systems.
"""

from PyQt5.QtGui import QColor


# ----------------------------------------------------------------------------------------------------------------------
# Color Constants - QColor format (for Qt widgets)
# ----------------------------------------------------------------------------------------------------------------------

# Selection/Highlight Colors
HIGHLIGHT_COLOR = QColor(0, 168, 230)       # Cyan for multi-highlight
SELECT_COLOR = QColor(144, 238, 144)        # Lime Green for single select

# Marker Colors (for cross-camera position display)
# Now using highlight/select colors instead of generic magenta
MARKER_COLOR_SELECTED = SELECT_COLOR        # Lime Green for selected camera marker
MARKER_COLOR_HIGHLIGHTED = HIGHLIGHT_COLOR  # Cyan for highlighted camera markers
MARKER_COLOR_DEFAULT = SELECT_COLOR


# ----------------------------------------------------------------------------------------------------------------------
# Color Constants - RGB tuple format (for PyVista)
# PyVista accepts colors as RGB tuples with values 0-255 or 0.0-1.0
# Using 0-255 integer format for clarity
# ----------------------------------------------------------------------------------------------------------------------

# Selection/Highlight Colors for PyVista
HIGHLIGHT_COLOR_RGB = (0, 168, 230)         # Cyan
SELECT_COLOR_RGB = (144, 238, 144)          # Lime Green

# Marker/Ray Colors for PyVista
RAY_COLOR_SELECTED = SELECT_COLOR_RGB        # RGB tuple for selected camera ray
RAY_COLOR_HIGHLIGHTED = HIGHLIGHT_COLOR_RGB  # RGB tuple for highlighted camera rays

# ----------------------------------------------------------------------------------------------------------------------
# UI Sizing Constants
# ----------------------------------------------------------------------------------------------------------------------

# Border widths for camera grid thumbnails
HIGHLIGHT_WIDTH = 4                         # Width for highlight border
SELECT_WIDTH = 6                            # Width for selection border

# Marker display settings
MARKER_SIZE = 12                            # Diameter of marker circle
MARKER_LINE_WIDTH = 2                       # Line width for marker crosshairs
