import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from coralnet_toolbox.Rasters.OrthoRaster import OrthoRaster


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class OrthoCamera:
    """
    Handles coordinate transformation from orthomosaic pixel space to 3D world space.

    Given an orthomosaic with known geographic extent (from GeoTIFF affine transform)
    and a Metashape chunk transform T, converts pixel coordinates to world-space
    rays that can be intersected against a 3D mesh to obtain accurate 3D positions.

    The transformation follows the Metashape convention:
        pixel (x, y)  →  geo (X, Y)  via affine geo transform
        geo (X, Y, Z) →  world p     via  p = T_inv @ proj_mat_inv @ [X, Y, Z, 1]

    For local coordinate systems (no reprojection needed) the ortho projection matrix
    defaults to identity.  For projected CRS with a non-trivial Metashape orthomosaic
    projection matrix, the user may supply it via the ImageWindow right-click menu.
    """

    def __init__(self, raster: 'OrthoRaster', chunk_transform: np.ndarray):
        """
        Args:
            raster: OrthoRaster with geo metadata populated from rasterio.
            chunk_transform: 4×4 Metashape chunk transform matrix (T).
        """
        self._raster = raster
        self.image_path = raster.image_path
        self.width = raster.width
        self.height = raster.height

        # Geo extent derived from rasterio affine transform
        self.ortho_left = getattr(raster, 'ortho_left', None)
        self.ortho_top = getattr(raster, 'ortho_top', None)
        self.resolution_x = getattr(raster, 'resolution_x', None)
        self.resolution_y = getattr(raster, 'resolution_y', None)

        # Chunk transform T and its inverse
        self._chunk_transform = np.asarray(chunk_transform, dtype=np.float64)
        self._T_inv = self._safe_inv(self._chunk_transform)

        # Ortho projection matrix (user-overridable; defaults to identity)
        proj_mat = getattr(raster, 'ortho_projection_matrix', None)
        self._proj_mat = (
            np.asarray(proj_mat, dtype=np.float64)
            if proj_mat is not None
            else np.eye(4, dtype=np.float64)
        )
        self._proj_mat_inv = self._safe_inv(self._proj_mat)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_inv(mat: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.inv(mat)
        except np.linalg.LinAlgError:
            return np.eye(4, dtype=np.float64)

    @staticmethod
    def _dehom(p: np.ndarray) -> np.ndarray:
        """Dehomogenise a 4-vector, guarding against near-zero w."""
        w = p[3]
        return p[:3] / w if abs(w) > 1e-12 else p[:3]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_valid(self) -> bool:
        """True when the geo metadata needed for pixel→world is available."""
        return all(
            v is not None
            for v in [self.ortho_left, self.ortho_top, self.resolution_x, self.resolution_y]
        )

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def pixel_to_geo(self, x: int, y: int):
        """Convert orthomosaic pixel (x, y) → geographic coordinates (X, Y)."""
        X = self.ortho_left + self.resolution_x * x
        Y = self.ortho_top - self.resolution_y * y
        return X, Y

    def geo_to_world(self, X: float, Y: float, Z: float) -> np.ndarray:
        """
        Convert geographic (X, Y, Z) → 3D world point.

        Applies:  p = T_inv @ proj_mat_inv @ [X, Y, Z, 1]
        """
        geo_hom = np.array([X, Y, Z, 1.0], dtype=np.float64)
        return self._dehom(self._T_inv @ (self._proj_mat_inv @ geo_hom))

    def pixel_to_xy_world(self, x: int, y: int) -> Optional[np.ndarray]:
        """
        Convert pixel (x, y) to a world-space base point with Z = 0 in the CRS.

        Used as the anchor for a vertical ray when querying mesh elevation.
        Returns None when geo metadata is unavailable.
        """
        if not self.is_valid:
            return None
        X, Y = self.pixel_to_geo(x, y)
        return self.geo_to_world(X, Y, 0.0)

    def get_vertical_direction_world(self) -> np.ndarray:
        """
        Return the unit vector in world space that corresponds to +Z in the ortho CRS.

        This is the direction a vertical ray (nadir-looking) travels when mapped
        through the combined T_inv @ proj_mat_inv transform.
        """
        z_crs = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        z_world = (self._T_inv @ (self._proj_mat_inv @ z_crs))[:3]
        norm = np.linalg.norm(z_world)
        return z_world / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def update_chunk_transform(self, T: np.ndarray):
        """Replace the chunk transform and recompute its inverse."""
        self._chunk_transform = np.asarray(T, dtype=np.float64)
        self._T_inv = self._safe_inv(self._chunk_transform)

    def update_ortho_projection_matrix(self, proj_mat: np.ndarray):
        """Replace the ortho projection matrix, sync back to the raster, recompute inverse."""
        self._proj_mat = np.asarray(proj_mat, dtype=np.float64)
        self._proj_mat_inv = self._safe_inv(self._proj_mat)
        self._raster.ortho_projection_matrix = self._proj_mat.copy()
