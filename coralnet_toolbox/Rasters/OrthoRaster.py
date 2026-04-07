import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from coralnet_toolbox.Rasters.QtRaster import Raster


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class OrthoRaster(Raster):
    """A minimal Raster subclass representing an orthomosaic.

    Currently behaves identically to `Raster` but records
    `raster_type = "OrthoRaster"` for serialization and future hooks.
    """

    def __init__(self, image_path: str):
        # Set canonical type before base initialization
        self.raster_type = "OrthoRaster"
        super().__init__(image_path)

    def to_dict(self) -> dict:
        data = super().to_dict()
        data['raster_type'] = 'OrthoRaster'
        return data

    @classmethod
    def from_dict(cls, raster_dict: dict):
        image_path = raster_dict['path']
        raster = cls(image_path)
        # Let base loader populate fields
        raster.update_from_dict(raster_dict)
        return raster
