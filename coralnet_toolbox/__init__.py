"""Top-level package for CoralNet-Toolbox."""

# Import torch before anything can pull in Qt.
#
# On Windows, loading PyQt5 first makes `import torch` fail outright:
#
#   OSError: [WinError 1114] A dynamic link library (DLL) initialization
#   routine failed. Error loading "...torch\lib\c10.dll"
#
# Qt brings in dozens of DLLs, and c10.dll needs a static TLS slot that is no
# longer available once they are loaded. Nothing is wrong with either library;
# the order alone decides it, and torch-first works in both directions. The
# reverse does not, and no environment variable rescues it -- KMP_DUPLICATE_LIB_OK
# makes no difference, because this is not an OpenMP clash.
#
# Every entry point reaches the application through this package, so importing
# torch here is what actually guarantees the ordering. Do not move it below any
# Qt import, and do not make it lazy.
import torch  # noqa: F401  (imported for load order, not for use here)

__version__ = "1.0.12"
__author__ = "Jordan Pierce"
__email__ = "jordan.pierce@noaa.gov"
__credits__ = "National Center for Coastal and Ocean Sciences (NCCOS)"
