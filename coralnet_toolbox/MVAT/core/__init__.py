"""
MVAT Core Module

Contains core classes for camera geometry, scene products, and visualization.
"""

# Scene Product Abstraction Layer
from coralnet_toolbox.MVAT.core.SceneProduct import (
    AbstractSceneProduct,
    BoundsType,
    ElementType,
    RenderStyle,
)
from coralnet_toolbox.MVAT.core.SceneContext import SceneContext

# Concrete Scene Products
from coralnet_toolbox.MVAT.core.Model import (
    PointCloudProduct,
    MeshProduct,
    DEMProduct,
)

# Camera and Visualization
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
    # Scene Product Abstraction
    'AbstractSceneProduct',
    'BoundsType',
    'ElementType',
    'RenderStyle',
    'SceneContext',
    # Concrete Products
    'PointCloudProduct',
    'MeshProduct',
    'DEMProduct',
    # Camera and Visualization
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
