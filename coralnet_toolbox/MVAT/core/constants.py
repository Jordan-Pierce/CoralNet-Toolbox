"""
MVAT Constants

Shared constants for consistent theming and configuration across MVAT components.
Includes colors, UI sizing, and other application-wide settings.
"""

from PyQt5.QtGui import QColor


# ----------------------------------------------------------------------------------------------------------------------
# Base Color Constants - RGB tuple format (0-255)
# These are the three primary colors used throughout MVAT
# ----------------------------------------------------------------------------------------------------------------------

CYAN_RGB = (0, 168, 230)         # Primary highlight color
LIME_RGB = (144, 238, 144)       # Primary selection color
BLOOD_RED_RGB = (230, 62, 0)     # Primary invalid/hover color


# ----------------------------------------------------------------------------------------------------------------------
# Color Constants - QColor format (for Qt widgets)
# ----------------------------------------------------------------------------------------------------------------------

# Selection/Highlight Colors
SELECT_COLOR = QColor(*LIME_RGB)          # Lime Green for single select
HIGHLIGHT_COLOR = QColor(*CYAN_RGB)       # Cyan for multi-highlight
INVALID_COLOR = QColor(*BLOOD_RED_RGB)    # Blood red for invalid state
HOVER_COLOR = QColor(*BLOOD_RED_RGB)      # Blood red

# Marker Colors (for cross-camera position display)
MARKER_COLOR_SELECTED = SELECT_COLOR        # Lime Green for selected camera marker
MARKER_COLOR_HIGHLIGHTED = HIGHLIGHT_COLOR  # Cyan for highlighted camera markers
MARKER_COLOR_INVALID = INVALID_COLOR        # Blood red for invalid camera markers
MARKER_COLOR_DEFAULT = INVALID_COLOR         # Default marker color


# ----------------------------------------------------------------------------------------------------------------------
# Color Constants - RGB tuple format (for PyVista)
# PyVista accepts colors as RGB tuples with values 0-255 or 0.0-1.0
# Using 0-255 integer format for clarity
# ----------------------------------------------------------------------------------------------------------------------

# Selection/Highlight Colors for PyVista
SELECT_COLOR_RGB = LIME_RGB               # Lime Green
HIGHLIGHT_COLOR_RGB = CYAN_RGB            # Cyan
HOVER_COLOR_RGB = BLOOD_RED_RGB           # Blood red
INVALID_COLOR_RGB = BLOOD_RED_RGB         # Blood red for invalid state

# Marker/Ray Colors for PyVista
RAY_COLOR_SELECTED = SELECT_COLOR_RGB           # RGB tuple for selected camera ray
RAY_COLOR_HIGHLIGHTED = HIGHLIGHT_COLOR_RGB     # RGB tuple for highlighted camera rays
RAY_COLOR_INVALID = INVALID_COLOR_RGB           # RGB tuple for invalid rays (blood red)

# ----------------------------------------------------------------------------------------------------------------------
# UI Sizing Constants
# ----------------------------------------------------------------------------------------------------------------------

# Border widths for camera grid thumbnails
HIGHLIGHT_WIDTH = 4                         # Width for highlight border
SELECT_WIDTH = 6                            # Width for selection border

# Marker display settings
MARKER_SIZE = 12                            # Diameter of marker circle
MARKER_LINE_WIDTH = 2                       # Line width for marker crosshairs

# Camera Grid settings
DEFAULT_THUMBNAIL_SIZE = 256
MIN_THUMBNAIL_SIZE = 256
MAX_THUMBNAIL_SIZE = 1024
GRID_SPACING = 5
BUFFER_ROWS = 1

# Mouse interaction settings
MOUSE_THROTTLE_MS = 16

# ----------------------------------------------------------------------------------------------------------------------
# State Constants for Scalar-based Coloring
# ----------------------------------------------------------------------------------------------------------------------

# State constants for scalar-based coloring
STATE_DEFAULT = 0
STATE_HIGHLIGHTED = 1
STATE_SELECTED = 2
STATE_HOVER = 3
STATE_INVALID = 4
# Colors for each state (RGB normalized 0-1)
STATE_COLORS = {
    STATE_DEFAULT: (0.8, 0.8, 0.8),      # Light gray/white
    STATE_HIGHLIGHTED: tuple(c / 255 for c in HIGHLIGHT_COLOR_RGB),  # Cyan
    STATE_SELECTED: tuple(c / 255 for c in SELECT_COLOR_RGB),        # Lime green
    STATE_HOVER: tuple(c / 255 for c in HOVER_COLOR_RGB),            # Blood red
    STATE_INVALID: tuple(c / 255 for c in INVALID_COLOR_RGB),        # Blood red for invalid state
}