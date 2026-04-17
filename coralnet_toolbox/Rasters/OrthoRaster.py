import warnings

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

from coralnet_toolbox.Rasters.QtRaster import Raster


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class OrthoRaster(Raster):
    """A Raster subclass representing a georeferenced orthomosaic.

    Extends the base Raster with:
    - Geographic extent metadata extracted from the rasterio affine transform
      (ortho_left, ortho_top, resolution_x, resolution_y).
    - An ortho projection matrix (4×4, default identity) that the user can
      supply for Metashape local-coordinate-system projects where the
      orthomosaic CRS has a non-trivial projection relative to the chunk's
      internal coordinate system.
    """

    def __init__(self, image_path: str):
        # Set canonical type before base initialization
        self.raster_type = "OrthoRaster"
        super().__init__(image_path)

        # User-settable 4×4 ortho projection matrix (identity by default).
        # Mirrors Metashape's orthomosaic.projection.matrix for local CRS projects.
        self.ortho_projection_matrix: np.ndarray = np.eye(4, dtype=np.float64)

        # User-settable 4×4 chunk transform matrix (identity by default).
        # Stored so orthomosaic-specific edits can persist alongside the raster.
        self.chunk_transform_matrix: np.ndarray = np.eye(4, dtype=np.float64)

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
        if self.ortho_projection_matrix is not None:
            data['ortho_projection_matrix'] = self.ortho_projection_matrix.tolist()
        if self.chunk_transform_matrix is not None:
            data['chunk_transform_matrix'] = self.chunk_transform_matrix.tolist()
        return data

    @classmethod
    def from_dict(cls, raster_dict: dict):
        image_path = raster_dict['path']
        raster = cls(image_path)
        raster.update_from_dict(raster_dict)
        proj_mat = raster_dict.get('ortho_projection_matrix')
        if proj_mat is not None:
            raster.ortho_projection_matrix = np.array(proj_mat, dtype=np.float64)
        chunk_mat = raster_dict.get('chunk_transform_matrix')
        if chunk_mat is not None:
            raster.chunk_transform_matrix = np.array(chunk_mat, dtype=np.float64)
        return raster
