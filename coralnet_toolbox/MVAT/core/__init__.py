"""
MVAT Core Module

Contains core classes for camera geometry and visualization.
"""

from coralnet_toolbox.MVAT.core.Model import PointCloud
from coralnet_toolbox.MVAT.core.Camera import Camera
from coralnet_toolbox.MVAT.core.Frustum import Frustum
from coralnet_toolbox.MVAT.managers.VisibilityManager import VisibilityManager
from coralnet_toolbox.MVAT.core.constants import (
    HIGHLIGHT_COLOR,
    SELECT_COLOR,
    HIGHLIGHT_COLOR_RGB,
    SELECT_COLOR_RGB,
    RAY_COLOR_SELECTED,
    RAY_COLOR_HIGHLIGHTED,
    MARKER_COLOR_SELECTED,
    MARKER_COLOR_HIGHLIGHTED,
    HIGHLIGHT_WIDTH,
    SELECT_WIDTH,
    MARKER_SIZE,
    MARKER_LINE_WIDTH,
)

__all__ = [
    'PointCloud',
    'Camera',
    'Frustum',
    'VisibilityManager',
    # Color constants
    'HIGHLIGHT_COLOR',
    'SELECT_COLOR',
    'HIGHLIGHT_COLOR_RGB',
    'SELECT_COLOR_RGB',
    'RAY_COLOR_SELECTED',
    'RAY_COLOR_HIGHLIGHTED',
    'MARKER_COLOR_SELECTED',
    'MARKER_COLOR_HIGHLIGHTED',
    'HIGHLIGHT_WIDTH',
    'SELECT_WIDTH',
    'MARKER_SIZE',
    'MARKER_LINE_WIDTH',
]
