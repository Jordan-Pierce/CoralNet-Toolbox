# coralnet_toolbox/Rasters/__init__.py

# Import the Raster class here to expose it at the package level
from coralnet_toolbox.Rasters.QtRaster import Raster
from coralnet_toolbox.Rasters.RasterManager import RasterManager
from coralnet_toolbox.Rasters.ImageFilter import ImageFilter
from coralnet_toolbox.Rasters.RasterTableModel import RasterTableModel

__all__ = ['Raster', 'RasterManager', 'ImageFilter', 'RasterTableModel']