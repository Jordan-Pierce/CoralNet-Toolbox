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
        self._raster.chunk_transform_matrix = self._chunk_transform.copy()

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

    @property
    def visible_indices(self):
        """
        Get the visible point indices for this camera.

        Returns:
            np.ndarray or None: 1D array of visible point IDs, or None if not computed
        """
        return self._raster.visible_indices

    def get_pixels_for_elements(self, element_ids: np.ndarray, bbox: tuple = None) -> np.ndarray:
        """
        Return a 1D array of flat (row-major) pixel indices for all elements
        in ``element_ids`` that are visible in this orthomosaic.

        Args:
            element_ids: 1D int array of element IDs to query.
            bbox: Optional (u_min, u_max, v_min, v_max) to restrict the search area.
        """
        try:
            index_map = getattr(self._raster, 'index_map', None)
            if index_map is None:
                return np.empty(0, dtype=np.int64)
            if element_ids is None or not isinstance(element_ids, np.ndarray) or len(element_ids) == 0:
                return np.empty(0, dtype=np.int64)

            # --- LUT Setup ---
            current_map_id = id(index_map)
            if getattr(self, '_cached_map_id', None) != current_map_id:
                self._cached_max_id = int(np.max(index_map))
                self._cached_map_id = current_map_id
                self._lut_buf = None  # Invalidate buffer when index map changes

            max_id = self._cached_max_id
            valid_query_ids = element_ids[element_ids <= max_id]
            if len(valid_query_ids) == 0:
                return np.empty(0, dtype=np.int64)

            # Reuse pre-allocated buffer — zero-allocation fast path
            if getattr(self, '_lut_buf', None) is None or len(self._lut_buf) < max_id + 2:
                self._lut_buf = np.zeros(max_id + 2, dtype=bool)
            lut = self._lut_buf
            lut[valid_query_ids] = True

            map_h, map_w = index_map.shape

            # Native-resolution index maps can reuse the camera-style search.
            if (map_h, map_w) == (self.height, self.width):
                if bbox is not None:
                    u_min, u_max, v_min, v_max = bbox

                    u_min = max(0, int(u_min))
                    u_max = min(self.width, int(u_max))
                    v_min = max(0, int(v_min))
                    v_max = min(self.height, int(v_max))

                    if u_min >= u_max or v_min >= v_max:
                        lut[valid_query_ids] = False
                        return np.empty(0, dtype=np.int64)

                    sub_map = index_map[v_min:v_max, u_min:u_max].ravel()
                    valid_mask = lut[sub_map]
                    local_flat_indices = np.where(valid_mask)[0].astype(np.int64)

                    if len(local_flat_indices) == 0:
                        lut[valid_query_ids] = False
                        return np.empty(0, dtype=np.int64)

                    box_width = u_max - u_min
                    local_v, local_u = np.divmod(local_flat_indices, box_width)
                    global_flat_indices = (local_v + v_min) * self.width + (local_u + u_min)
                    lut[valid_query_ids] = False
                    return global_flat_indices

                stride = 8
                if map_h <= stride and map_w <= stride:
                    exact_map = index_map.ravel()
                    valid_mask = lut[exact_map]
                    local_flat_indices = np.where(valid_mask)[0].astype(np.int64)

                    if len(local_flat_indices) == 0:
                        lut[valid_query_ids] = False
                        return np.empty(0, dtype=np.int64)

                    lut[valid_query_ids] = False
                    return local_flat_indices

                sub_map = index_map[::stride, ::stride].ravel()
                valid_mask_sub = lut[sub_map]

                if not valid_mask_sub.any():
                    lut[valid_query_ids] = False
                    return np.empty(0, dtype=np.int64)

                sub_flat_indices = np.where(valid_mask_sub)[0]
                sub_w = (self.width + stride - 1) // stride
                sub_v, sub_u = np.divmod(sub_flat_indices, sub_w)

                u_min = max(0, (sub_u.min() - 1) * stride)
                u_max = min(self.width, (sub_u.max() + 2) * stride)
                v_min = max(0, (sub_v.min() - 1) * stride)
                v_max = min(self.height, (sub_v.max() + 2) * stride)

                exact_map = index_map[v_min:v_max, u_min:u_max].ravel()
                valid_mask = lut[exact_map]
                local_flat_indices = np.where(valid_mask)[0].astype(np.int64)

                if len(local_flat_indices) == 0:
                    lut[valid_query_ids] = False
                    return np.empty(0, dtype=np.int64)

                box_width = u_max - u_min
                local_v, local_u = np.divmod(local_flat_indices, box_width)
                global_flat_indices = (local_v + v_min) * self.width + (local_u + u_min)
                lut[valid_query_ids] = False
                return global_flat_indices

            # Lower-resolution ortho index maps are stored at a smaller grid.
            # Expand the matching cells back to native pixel coordinates.
            row_edges = np.round(np.linspace(0, self.height, map_h + 1)).astype(np.int64)
            col_edges = np.round(np.linspace(0, self.width, map_w + 1)).astype(np.int64)

            if bbox is not None:
                u_min, u_max, v_min, v_max = bbox
                u_min_lr = max(0, int(np.floor(u_min * map_w / self.width)))
                u_max_lr = min(map_w, int(np.ceil(u_max * map_w / self.width)))
                v_min_lr = max(0, int(np.floor(v_min * map_h / self.height)))
                v_max_lr = min(map_h, int(np.ceil(v_max * map_h / self.height)))

                if u_min_lr >= u_max_lr or v_min_lr >= v_max_lr:
                    lut[valid_query_ids] = False
                    return np.empty(0, dtype=np.int64)

                search_map = index_map[v_min_lr:v_max_lr, u_min_lr:u_max_lr]
                row_offset = v_min_lr
                col_offset = u_min_lr
            else:
                search_map = index_map
                row_offset = 0
                col_offset = 0

            valid_mask = lut[search_map]
            if not valid_mask.any():
                lut[valid_query_ids] = False
                return np.empty(0, dtype=np.int64)

            local_rows, local_cols = np.where(valid_mask)
            global_rows = local_rows + row_offset
            global_cols = local_cols + col_offset

            flat_chunks = []
            for row_idx, col_idx in zip(global_rows, global_cols):
                row_start = row_edges[row_idx]
                row_end = row_edges[row_idx + 1]
                col_start = col_edges[col_idx]
                col_end = col_edges[col_idx + 1]

                if row_start >= row_end or col_start >= col_end:
                    continue

                row_indices = np.arange(row_start, row_end, dtype=np.int64)[:, None]
                col_indices = np.arange(col_start, col_end, dtype=np.int64)[None, :]
                flat_chunks.append((row_indices * self.width + col_indices).ravel())

            lut[valid_query_ids] = False
            if not flat_chunks:
                return np.empty(0, dtype=np.int64)

            return np.concatenate(flat_chunks)

        except Exception as e:
            if hasattr(self, '_lut_buf') and getattr(self, '_lut_buf', None) is not None:
                try:
                    if 'valid_query_ids' in dir():
                        self._lut_buf[valid_query_ids] = False
                except Exception:
                    self._lut_buf = None
            print(f"⚠️ get_pixels_for_elements failed: {e}")
            return np.empty(0, dtype=np.int64)

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

    def world_to_geo(self, world_point: np.ndarray) -> Optional[np.ndarray]:
        """Convert a 3D world point back to geographic coordinates."""
        try:
            world = np.asarray(world_point, dtype=np.float64).reshape(-1)
            if world.size < 3:
                return None

            world_hom = np.array([world[0], world[1], world[2], 1.0], dtype=np.float64)
            geo_hom = self._proj_mat @ (self._chunk_transform @ world_hom)
            geo = self._dehom(geo_hom)
            return geo[:3]
        except Exception:
            return None

    def world_to_pixel(self, world_point: np.ndarray) -> Optional[np.ndarray]:
        """Convert a 3D world point to orthomosaic pixel coordinates."""
        geo = self.world_to_geo(world_point)
        if geo is None:
            return None

        if self.ortho_left is None or self.ortho_top is None:
            return None
        if self.resolution_x is None or self.resolution_y is None:
            return None
        if abs(self.resolution_x) < 1e-12 or abs(self.resolution_y) < 1e-12:
            return None

        X, Y = float(geo[0]), float(geo[1])
        u = (X - self.ortho_left) / self.resolution_x
        v = (self.ortho_top - Y) / self.resolution_y
        return np.array([u, v], dtype=np.float64)

    def project(self, world_point: np.ndarray) -> np.ndarray:
        """Project a world-space point into orthomosaic pixel coordinates."""
        pixel = self.world_to_pixel(world_point)
        if pixel is None:
            return np.array([np.nan, np.nan], dtype=np.float64)
        return np.asarray(pixel, dtype=np.float64)

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
    # Selection and highlighting
    # ------------------------------------------------------------------

    def select(self):
        """Mark as selected."""
        pass  # OrthoCamera is a geometric object without UI frustum

    def deselect(self):
        """Mark as deselected."""
        pass  # OrthoCamera is a geometric object without UI frustum

    def highlight(self):
        """Mark as highlighted."""
        pass  # OrthoCamera is a geometric object without UI frustum

    def unhighlight(self):
        """Mark as not highlighted."""
        pass  # OrthoCamera is a geometric object without UI frustum

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialize OrthoCamera state to a dictionary.

        Returns:
            dict with keys: 'chunk_transform', 'ortho_projection_matrix'
        """
        return {
            'chunk_transform': self._chunk_transform.tolist(),
            'ortho_projection_matrix': self._proj_mat.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict, raster: 'OrthoRaster') -> 'OrthoCamera':
        """
        Deserialize OrthoCamera from a dictionary.

        Args:
            data: dict with 'chunk_transform' and 'ortho_projection_matrix'
            raster: OrthoRaster instance to associate with the camera

        Returns:
            OrthoCamera instance with state restored from data
        """
        chunk_transform = np.asarray(data.get('chunk_transform', np.eye(4)), dtype=np.float64)
        camera = cls(raster, chunk_transform)

        if 'ortho_projection_matrix' in data:
            proj_mat = np.asarray(data['ortho_projection_matrix'], dtype=np.float64)
            camera.update_ortho_projection_matrix(proj_mat)

        return camera

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def update_chunk_transform(self, T: np.ndarray):
        """Replace the chunk transform and recompute its inverse."""
        self._chunk_transform = np.asarray(T, dtype=np.float64)
        self._T_inv = self._safe_inv(self._chunk_transform)
        self._raster.chunk_transform_matrix = self._chunk_transform.copy()

    def update_ortho_projection_matrix(self, proj_mat: np.ndarray):
        """Replace the ortho projection matrix, sync back to the raster, recompute inverse."""
        self._proj_mat = np.asarray(proj_mat, dtype=np.float64)
        self._proj_mat_inv = self._safe_inv(self._proj_mat)
        self._raster.ortho_projection_matrix = self._proj_mat.copy()
