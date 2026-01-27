"""
Multi-View Annotation Tool (MVAT) Module

Provides 3D visualization and navigation for multi-view imagery projects.
"""

from coralnet_toolbox.MVAT.QtMVATWindow import MVATWindow

from coralnet_toolbox.MVAT.core.Camera import Camera
from coralnet_toolbox.MVAT.core.Frustum import Frustum

__all__ = [
    'MVATWindow',
    'Camera', 
    'Frustum',
]
