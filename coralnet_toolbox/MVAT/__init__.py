"""Multi-View Annotation Tool (MVAT) package.

Expose core classes eagerly but avoid importing the UI submodules at
package import time to prevent circular imports. UI components (the
3D viewer and camera grid) are lazily imported when accessed as
attributes on the package (PEP 562 module __getattr__).
"""

# Core, safe-to-import pieces
from .core.Camera import Camera
from .core.Frustum import Frustum
from .core.Ray import CameraRay

# Manager (controller) is safe to import eagerly in normal use
from .managers.MVATManager import MVATManager

__all__ = [
    'Camera',
    'Frustum',
    'CameraRay',
    'MVATManager',
    # The following names are provided lazily via __getattr__:
    'MVATViewer',
    'CameraGrid',
]


def __getattr__(name: str):
    """Lazily import UI classes to avoid circular imports.

    Accessing `coralnet_toolbox.MVAT.MVATViewer` or
    `coralnet_toolbox.MVAT.CameraGrid` will import the corresponding
    UI module only when needed.
    """
    if name == 'MVATViewer':
        from .ui.QtMVATViewer import MVATViewer as _MVATViewer
        return _MVATViewer
    if name == 'CameraGrid':
        from .ui.QtCameraGrid import CameraGrid as _CameraGrid
        return _CameraGrid
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + ['MVATViewer', 'CameraGrid'])
