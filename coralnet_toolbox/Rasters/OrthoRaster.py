import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from coralnet_toolbox.Rasters.QtRaster import Raster


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class OrthoRaster(Raster):
    """A Raster subclass representing a georeferenced orthomosaic.

    Extends the base Raster with geographic extent metadata extracted from the
    rasterio affine transform (ortho_left, ortho_top, resolution_x, resolution_y).
    """

    def __init__(self, image_path: str):
        # Set canonical type before base initialization
        self.raster_type = "OrthoRaster"
        super().__init__(image_path)

        # Extract geo extent from rasterio affine transform
        self._init_geo_metadata()

    # ------------------------------------------------------------------
    # Geo metadata
    # ------------------------------------------------------------------

    def _init_geo_metadata(self):
        """Populate geo extent fields from the rasterio affine transform."""
        self.ortho_left = None      # X of leftmost pixel centre (CRS units)
        self.ortho_top = None       # Y of topmost  pixel centre (CRS units)
        self.resolution_x = None    # Pixel width  in CRS units (positive)
        self.resolution_y = None    # Pixel height in CRS units (positive)

        src = getattr(self, '_rasterio_src', None)
        if src is None:
            return
        t = src.transform
        if t is None or t.is_identity:
            return

        self.ortho_left = t.c
        self.ortho_top = t.f
        self.resolution_x = abs(t.a)
        self.resolution_y = abs(t.e)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        data = super().to_dict()
        data['raster_type'] = 'OrthoRaster'
        return data
