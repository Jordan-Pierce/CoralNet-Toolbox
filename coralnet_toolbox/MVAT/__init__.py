"""Multi-View Annotation Tool (MVAT) package.

Expose core classes eagerly but avoid importing the UI submodules at
package import time to prevent circular imports. UI components (the
3D viewer and context matrix) are lazily imported when accessed as
attributes on the package (PEP 562 module __getattr__).
"""

# Core, safe-to-import pieces
from .core.Cameras import Camera
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
    'ContextMatrixWidget',
]


def __getattr__(name: str):
    """Lazily import UI classes to avoid circular imports.

    Accessing `coralnet_toolbox.MVAT.MVATViewer` or
    `coralnet_toolbox.MVAT.ContextMatrixWidget` will import the corresponding
    UI module only when needed.
    """
    if name == 'MVATViewer':
        from .ui.QtMVATViewer import MVATViewer as _MVATViewer
        return _MVATViewer
    if name == 'ContextMatrixWidget':
        from .ui.QtContextMatrix import ContextMatrixWidget as _ContextMatrixWidget
        return _ContextMatrixWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + ['MVATViewer', 'ContextMatrixWidget'])
