"""Single import site for the Qt Advanced Docking System bindings.

The bindings are laid out differently depending on where PyQtAds came from:

    PyPI      `PyQtAds` 3.x   -- a package; the bindings live in `PyQtAds.ads`
    conda-forge `pyqtads` 4.x -- one PyQtAds.pyd exporting the same names flat

Both builds expose an identical symbol set (CDockManager, CDockWidget,
DockWidgetArea, the *DockWidgetArea constants, ...), so the difference is
purely where they are hung. Normalising here keeps the call sites free of
build-detection logic.
"""

# Critical: order of imports matter
import PyQt5.QtCore  # noqa: F401
import PyQtAds

try:
    # PyPI layout.
    from PyQtAds import ads
except ImportError:
    # conda-forge layout: the module *is* the namespace.
    ads = PyQtAds

__all__ = ['ads']
