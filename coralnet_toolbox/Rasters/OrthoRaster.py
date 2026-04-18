import warnings
from typing import Optional

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

        # Scale factor for the ortho index map (index_map.shape / native resolution).
        # None until the map is built.  Set by add_index_map when the map is smaller
        # than native resolution so the brush handler can convert coordinates.
        self.index_map_scale_factor: Optional[float] = None

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
    # Index map (overrides base to skip upscale for large orthos)
    # ------------------------------------------------------------------

    def add_index_map(self, index_map: np.ndarray,
                      index_map_path: Optional[str] = None,
                      visible_indices: Optional[np.ndarray] = None,
                      element_type: Optional[str] = 'face',
                      inverted_index: Optional[dict] = None):
        """Store the index map without upscaling to native ortho resolution.

        The base Raster.add_index_map always upscales to (height, width), which
        is infeasible for large orthomosaics (e.g. 40 k × 40 k = 6.4 GB int32).
        Here we store the map at whatever resolution it was built and record the
        scale factor so callers can convert native pixel coordinates.
        """
        if not isinstance(index_map, np.ndarray) or index_map.ndim != 2:
            raise ValueError("index_map must be a 2-D numpy array")
        if index_map.dtype != np.int32:
            raise ValueError("index_map must be int32")

        valid_types = {'point', 'face', 'cell'}
        if element_type is not None and element_type not in valid_types:
            raise ValueError(f"element_type must be one of {valid_types}")

        # Derive scale factor from the stored map dimensions vs native resolution
        if index_map.shape == (self.height, self.width):
            self.index_map_scale_factor = 1.0
        else:
            # Use the height ratio; width ratio should match for a correctly-built map
            self.index_map_scale_factor = index_map.shape[0] / self.height

        self.index_map = index_map.copy()
        self.index_map_path = index_map_path
        self.index_element_type = element_type

        if inverted_index is not None:
            self.inv_ids     = inverted_index['inv_ids']
            self.inv_offsets = inverted_index['inv_offsets']
            self.inv_pixels  = inverted_index['inv_pixels']
        else:
            self.inv_ids = self.inv_offsets = self.inv_pixels = None

        if visible_indices is not None:
            if not isinstance(visible_indices, np.ndarray) or visible_indices.ndim != 1:
                raise ValueError("visible_indices must be a 1-D numpy array")
            self.visible_indices = visible_indices.copy()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def update_from_dict(self, raster_dict: dict):
        """Extend the base update to restore OrthoRaster-specific matrices."""
        super().update_from_dict(raster_dict)
        proj_mat = raster_dict.get('ortho_projection_matrix')
        if proj_mat is not None:
            self.ortho_projection_matrix = np.array(proj_mat, dtype=np.float64)
        chunk_mat = raster_dict.get('chunk_transform_matrix')
        if chunk_mat is not None:
            self.chunk_transform_matrix = np.array(chunk_mat, dtype=np.float64)

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
