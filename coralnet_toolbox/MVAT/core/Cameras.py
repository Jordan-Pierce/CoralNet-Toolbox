"""
Camera classes for MVAT.

Camera        — perspective camera wrapping a Raster object (intrinsics/extrinsics,
                projection/unprojection, occlusion, frustum visualization).
OrthoCamera   — orthographic camera wrapping an OrthoRaster (geo-to-world transforms,
                pixel ↔ world coordinate mapping via Metashape chunk transform).
"""
import os
import traceback

import numpy as np

from coralnet_toolbox.MVAT.core.Frustum import Frustum


def _clamp_bbox_to_image(bbox, width: int, height: int):
    if bbox is None:
        return None

    try:
        u_min, u_max, v_min, v_max = bbox
    except Exception:
        return None

    u_min = max(0, int(u_min))
    u_max = min(int(width), int(u_max))
    v_min = max(0, int(v_min))
    v_max = min(int(height), int(v_max))

    if u_min >= u_max or v_min >= v_max:
        return ()

    return u_min, u_max, v_min, v_max


def _query_pixels_from_csr_inverted_index(raster, element_ids: np.ndarray, width: int, height: int, bbox=None):
    """Return flat pixel indices from raster CSR arrays, or None when unavailable/invalid."""
    inv_ids = getattr(raster, 'inv_ids', None)
    inv_offsets = getattr(raster, 'inv_offsets', None)
    inv_pixels = getattr(raster, 'inv_pixels', None)

    if not isinstance(inv_ids, np.ndarray) or not isinstance(inv_offsets, np.ndarray) or not isinstance(inv_pixels, np.ndarray):
        # Build it lazily on first query for this camera (deferred from load time
        # to bound RAM across large camera sets), then re-read.
        if getattr(raster, 'ensure_inverted_index', None) and raster.ensure_inverted_index():
            inv_ids = getattr(raster, 'inv_ids', None)
            inv_offsets = getattr(raster, 'inv_offsets', None)
            inv_pixels = getattr(raster, 'inv_pixels', None)
        if not isinstance(inv_ids, np.ndarray) or not isinstance(inv_offsets, np.ndarray) or not isinstance(inv_pixels, np.ndarray):
            return None

    inv_ids_arr = np.asarray(inv_ids)
    inv_offsets_arr = np.asarray(inv_offsets)
    inv_pixels_arr = np.asarray(inv_pixels)

    if inv_ids_arr.ndim != 1 or inv_offsets_arr.ndim != 1 or inv_pixels_arr.ndim != 1:
        return None
    if inv_offsets_arr.size != inv_ids_arr.size + 1 or inv_offsets_arr.size == 0:
        return None

    inv_ids_arr = inv_ids_arr.astype(np.int64, copy=False)
    inv_offsets_arr = inv_offsets_arr.astype(np.int64, copy=False)

    if inv_offsets_arr[0] != 0:
        return None
    if inv_offsets_arr[-1] != inv_pixels_arr.size:
        return None
    if np.any(inv_offsets_arr < 0) or np.any(inv_offsets_arr[1:] < inv_offsets_arr[:-1]):
        return None
    if inv_ids_arr.size > 1 and np.any(inv_ids_arr[1:] < inv_ids_arr[:-1]):
        return None

    try:
        query_ids = np.asarray(element_ids, dtype=np.int64).ravel()
    except Exception:
        return None

    if query_ids.size == 0:
        return np.empty(0, dtype=np.int64)

    query_ids = query_ids[query_ids >= 0]
    if query_ids.size == 0:
        return np.empty(0, dtype=np.int64)

    query_ids = np.unique(query_ids)
    positions = np.searchsorted(inv_ids_arr, query_ids, side='left')
    in_bounds = positions < inv_ids_arr.size
    if not np.any(in_bounds):
        return np.empty(0, dtype=np.int64)

    candidate_rows = positions[in_bounds]
    candidate_ids = query_ids[in_bounds]
    row_matches = inv_ids_arr[candidate_rows] == candidate_ids
    if not np.any(row_matches):
        return np.empty(0, dtype=np.int64)

    matched_rows = candidate_rows[row_matches]
    chunks = []
    for row in matched_rows.tolist():
        start = int(inv_offsets_arr[row])
        end = int(inv_offsets_arr[row + 1])
        if end > start:
            chunks.append(inv_pixels_arr[start:end])

    if not chunks:
        return np.empty(0, dtype=np.int64)

    pixels = np.concatenate(chunks).astype(np.int64, copy=False)

    if bbox is not None:
        bbox_clamped = _clamp_bbox_to_image(bbox, width, height)
        if bbox_clamped is None:
            return None
        if len(bbox_clamped) == 0:
            return np.empty(0, dtype=np.int64)

        u_min, u_max, v_min, v_max = bbox_clamped
        u = pixels % int(width)
        v = pixels // int(width)
        keep = (u >= u_min) & (u < u_max) & (v >= v_min) & (v < v_max)
        pixels = pixels[keep]

    return pixels


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Camera:
    """
    A geometric controller class that wraps a 2D Raster object.
    
    This class handles all 3D operations including:
    - Managing Intrinsics (K) and Extrinsics (R, t) matrices
    - 3D-to-2D Projection (World -> Pixel)
    - 2D-to-3D Unprojection (Pixel -> World)
    - Occlusion testing using depth maps or meshes
    - Managing the visualization Frustum
    """

    def __init__(self, raster):
        """
        Initialize a Camera from a generic Raster object.

        Args:
            raster: A Raster object containing image data, intrinsics, and extrinsics.
        """
        self._raster = raster
        
        # --- Dimensions ---
        self.width = raster.width
        self.height = raster.height
        
        # --- Intrinsics (K) ---
        # The intrinsic matrix describes the internal camera parameters (focal length, principal point)
        if raster.intrinsics is not None:
            self.K = raster.intrinsics
        else:
            # Default to identity with a reasonable guess if missing to prevent crashes
            # Assumes the principal point is at the center
            self.K = np.eye(3)
            self.K[0, 0] = self.width   # fx approx width
            self.K[1, 1] = self.height  # fy approx height
            self.K[0, 2] = self.width / 2  # cx
            self.K[1, 2] = self.height / 2 # cy

        # Compute Inverse Intrinsics (K_inv) for unprojection
        try:
            self.K_inv = np.linalg.inv(self.K)
        except np.linalg.LinAlgError:
            self.K_inv = np.eye(3)

        # --- Extrinsics (R, t) ---
        # We expect Raster.extrinsics to be the World-to-Camera transformation matrix (4x4)
        # T_cam_world = [ R  | t ]
        #               [ 0  | 1 ]
        if raster.extrinsics is not None:
            self.extrinsics = raster.extrinsics
            self.R = self.extrinsics[:3, :3]  # Rotation (3x3)
            self.t = self.extrinsics[:3, 3]   # Translation (3x1)
        else:
            self.extrinsics = np.eye(4)
            self.R = np.eye(3)
            self.t = np.zeros(3)

        # Camera Position in World Coordinates (Optical Center)
        # Position C = -R^T * t
        self.position = -self.R.T @ self.t

        # Projection Matrix P = K [R | t]
        # This 3x4 matrix directly maps 3D homogeneous world points to 2D homogeneous image points
        self.P = self.K @ self.extrinsics[:3, :]

        # --- Distortion ---
        # K_linear: undistorted camera matrix used by 3D rendering engines.
        # When the raster is distorted, this is K_new from getOptimalNewCameraMatrix (wider FOV).
        # Falls back to K when undistorted or when cv2 is unavailable.
        self.dist_coeffs = getattr(raster, 'dist_coeffs', None)
        self.is_distorted = getattr(raster, 'is_distorted', False)
        self.is_fisheye = getattr(raster, 'is_fisheye', False)
        # Avoid using numpy arrays in boolean context (ambiguous truth value).
        intr_undist = getattr(raster, 'intrinsics_undistorted', None)
        self.K_linear = intr_undist if intr_undist is not None else self.K

        # Pre-compute float64 copies used repeatedly in project() to eliminate
        # per-call .astype(np.float64) allocations.
        self._t_f64 = self.t.astype(np.float64)
        self._K_f64 = self.K.astype(np.float64)
        self._dist_coeffs_f64 = (self.dist_coeffs.astype(np.float64)
                                 if self.dist_coeffs is not None else None)
        # Cache the Rodrigues rotation vector used by cv2.projectPoints.
        # R is fixed at construction time so this never needs recomputing.
        self._rvec = None
        if self.is_distorted and self._dist_coeffs_f64 is not None:
            try:
                import cv2 as _cv2
                self._rvec, _ = _cv2.Rodrigues(self.R.astype(np.float64))
            except Exception:
                pass

        # --- Visualization ---
        self.selected = False
        
        # Initialize the Frustum visualization object
        # The Frustum reads geometry directly from 'self' (this Camera instance)
        self.frustum = Frustum(self)

    # --------------------------------------------------------------------------
    # Properties (Delegated to Raster)
    # --------------------------------------------------------------------------

    @property
    def label(self):
        """Return the label/filename of the associated raster."""
        return self._raster.basename

    @property
    def image_path(self):
        """Return the file path of the source image."""
        return self._raster.image_path

    def get_thumbnail(self):
        """
        Get the image thumbnail for 3D visualization.
        
        Returns:
            QImage: The thumbnail image from the raster.
        """
        return self._raster.get_thumbnail()
    
    def get_native_size(self):
        """Return tuple of (width, height)."""
        return (self.width, self.height)

    # --------------------------------------------------------------------------
    # Visibility / Index Map Properties (Delegated to Raster)
    # --------------------------------------------------------------------------
    
    @property
    def visible_indices(self):
        """
        Get the visible point indices for this camera.
        
        Returns:
            np.ndarray or None: 1D array of visible point IDs, or None if not computed
        """
        return self._raster.visible_indices
    
    @property
    def index_map(self):
        """
        Get the index map for this camera.
        Uses lazy loading from Raster if available.
        
        Returns:
            np.ndarray or None: 2D index map (H x W), or None if not computed
        """
        return self._raster.index_map_lazy

    def get_pixels_for_element(self, element_id: int) -> list:
        """
        Return all (u, v) pixel coordinates where element_id is visible.

        Performs on-the-fly lookup using np.where() for single element.
        Latency: 5-10 ms for 16.8M pixel images (acceptable for UI clicks).

        Returns:
            list of (u, v) tuples, or [] if element_id is not visible or index_map unavailable.
        """
        if self._raster.index_map is None:
            return []
        
        try:
            # Fast single-element lookup
            y_coords, x_coords = np.where(self._raster.index_map == element_id)
            
            if len(y_coords) == 0:
                return []
            
            return list(zip(x_coords.tolist(), y_coords.tolist()))
        except Exception:
            # Safety: return empty list on any error to avoid hanging
            return []

    def get_pixels_for_elements(self, element_ids: np.ndarray, bbox: tuple = None) -> np.ndarray:
        """
        Return a 1D array of flat (row-major) pixel indices for all elements
        in ``element_ids`` that are visible in this camera.
        
        Args:
            element_ids: 1D int array of element IDs to query.
            bbox: Optional (u_min, u_max, v_min, v_max) to restrict the search area.
        """
        try:
            if self._raster.index_map is None:
                return np.empty(0, dtype=np.int64)
            if element_ids is None or not isinstance(element_ids, np.ndarray) or len(element_ids) == 0:
                return np.empty(0, dtype=np.int64)

            csr_pixels = _query_pixels_from_csr_inverted_index(
                self._raster,
                element_ids,
                self.width,
                self.height,
                bbox=bbox,
            )
            if csr_pixels is not None:
                return csr_pixels

            # --- LUT Setup ---
            current_map_id = id(self._raster.index_map)
            if getattr(self, '_cached_map_id', None) != current_map_id:
                self._cached_max_id = int(np.max(self._raster.index_map))
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
            
            # --- Localized Sub-grid Search ---
            if bbox is not None:
                # BBOX Clamping to image dimensions
                u_min, u_max, v_min, v_max = bbox
                
                u_min = max(0, int(u_min))
                u_max = min(self.width, int(u_max))
                v_min = max(0, int(v_min))
                v_max = min(self.height, int(v_max))
                
                if u_min >= u_max or v_min >= v_max:
                    return np.empty(0, dtype=np.int64)
                    
                sub_map = self._raster.index_map[v_min:v_max, u_min:u_max].ravel()
                valid_mask = lut[sub_map]
                local_flat_indices = np.where(valid_mask)[0].astype(np.int64)
                
                if len(local_flat_indices) == 0:
                    return np.empty(0, dtype=np.int64)
                    
                box_width = u_max - u_min
                local_v, local_u = np.divmod(local_flat_indices, box_width)
                
                global_flat_indices = (local_v + v_min) * self.width + (local_u + u_min)
                # CRITICAL: Reset only the bits we set before returning
                lut[valid_query_ids] = False
                return global_flat_indices
                
            # --- STRIDED PRE-SEARCH OPTIMIZATION (Fallback) ---
            else:
                # If we don't have a bbox, DO NOT scan 16M pixels! 
                # Scan a 1/64th resolution version to find the rough bounding box instantly.
                stride = 8
                if self.width <= stride and self.height <= stride:
                    exact_map = self._raster.index_map.ravel()
                    valid_mask = lut[exact_map]
                    local_flat_indices = np.where(valid_mask)[0].astype(np.int64)

                    if len(local_flat_indices) == 0:
                        lut[valid_query_ids] = False
                        return np.empty(0, dtype=np.int64)

                    lut[valid_query_ids] = False
                    return local_flat_indices

                sub_map = self._raster.index_map[::stride, ::stride].ravel()
                valid_mask_sub = lut[sub_map]
                
                if not valid_mask_sub.any():
                    lut[valid_query_ids] = False
                    return np.empty(0, dtype=np.int64)
                    
                sub_flat_indices = np.where(valid_mask_sub)[0]
                sub_w = (self.width + stride - 1) // stride
                sub_v, sub_u = np.divmod(sub_flat_indices, sub_w)
                
                # Convert back to full resolution with a generous safety margin
                u_min = max(0, (sub_u.min() - 1) * stride)
                u_max = min(self.width, (sub_u.max() + 2) * stride)
                v_min = max(0, (sub_v.min() - 1) * stride)
                v_max = min(self.height, (sub_v.max() + 2) * stride)
                
                # Now do the exact search ONLY within this tight bounding box
                exact_map = self._raster.index_map[v_min:v_max, u_min:u_max].ravel()
                valid_mask = lut[exact_map]
                local_flat_indices = np.where(valid_mask)[0].astype(np.int64)
                
                if len(local_flat_indices) == 0:
                    lut[valid_query_ids] = False
                    return np.empty(0, dtype=np.int64)
                    
                box_width = u_max - u_min
                local_v, local_u = np.divmod(local_flat_indices, box_width)
                
                global_flat_indices = (local_v + v_min) * self.width + (local_u + u_min)
                # CRITICAL: Reset only the bits we set before returning
                lut[valid_query_ids] = False
                return global_flat_indices
                
        except Exception as e:
            # Safety: ensure LUT is reset even on error
            if hasattr(self, '_lut_buf') and getattr(self, '_lut_buf') is not None:
                try:
                    if 'valid_query_ids' in dir():
                        self._lut_buf[valid_query_ids] = False
                except Exception:
                    self._lut_buf = None  # Force reallocation next call
            print(f"⚠️ get_pixels_for_elements failed: {e}")
            return np.empty(0, dtype=np.int64)

    # --------------------------------------------------------------------------
    # Geometric Methods
    # --------------------------------------------------------------------------
    
    def project(self, points_3d_world):
        """
        Project a 3D world point into a 2D pixel coordinate.

        When the source image has lens distortion, uses cv2.projectPoints so the
        result lands on the correct *distorted* pixel. Falls back to linear matrix
        multiplication otherwise.

        Args:
            points_3d_world (np.ndarray): 3D point [x, y, z] in world coordinates.

        Returns:
            np.ndarray: 2D pixel coordinate [u, v] or [nan, nan] if invalid.
        """
        if self.is_distorted and self._dist_coeffs_f64 is not None and self._rvec is not None:
            try:
                import cv2
                pts = np.asarray(points_3d_world, dtype=np.float64).reshape(1, 1, 3)
                # Check if the point is in front of the camera (fast, no allocation)
                pt_cam = self.R @ pts[0, 0] + self._t_f64
                if pt_cam[2] <= 0:
                    return np.array([np.nan, np.nan])
                # ---> FISHEYE BRANCH <---
                if self.is_fisheye:
                    projected, _ = cv2.fisheye.projectPoints(
                        pts, self._rvec, self._t_f64, self._K_f64, self._dist_coeffs_f64[:4]
                    )
                else:
                    projected, _ = cv2.projectPoints(
                        pts, self._rvec, self._t_f64, self._K_f64, self._dist_coeffs_f64
                    )
                return np.array([float(projected[0, 0, 0]), float(projected[0, 0, 1])])
            except ImportError:
                pass  # Fall through to linear path
            except Exception:
                return np.array([np.nan, np.nan])

        # Linear (undistorted) path
        points_hom = np.hstack((np.asarray(points_3d_world, dtype=np.float64), 1.0))
        points_cam_hom = (self.P @ points_hom.T).T
        points_pixel = np.full(2, np.nan)
        if points_cam_hom[2] > 0:
            points_pixel = points_cam_hom[:2] / points_cam_hom[2]
        return points_pixel

    def unproject(self, pixel_coord, depth=None):
        """
        Unproject a 2D pixel coordinate to a 3D world point.

        Requires valid depth data in the associated Raster (Z-channel).
        When the image is distorted, the raw pixel is first undistorted to its
        linear position in K_linear space before back-projection.

        Args:
            pixel_coord (tuple/list): 2D pixel [u, v] in distorted image space.
            depth (float, optional): Depth value at the pixel. If None, it will be fetched from the raster's Z-channel.

        Returns:
            np.ndarray: 3D world point [x, y, z] or None if depth is missing.
        """
        if depth is None:
            depth = self._get_depth_from_raster(int(pixel_coord[0]), int(pixel_coord[1]))
            
        if depth is None or depth <= 0 or np.isnan(depth):
            return None

        if self.is_distorted and self.dist_coeffs is not None and self.K_linear is not None:
            try:
                import cv2
                # Map the distorted pixel to its undistorted (linear) equivalent
                pts_in = np.array([[[float(pixel_coord[0]), float(pixel_coord[1])]]], dtype=np.float32)
                # ---> FISHEYE BRANCH <---
                if getattr(self, 'is_fisheye', False):
                    D = self.dist_coeffs[:4]
                    pts_out = cv2.fisheye.undistortPoints(
                        pts_in, self.K.astype(np.float64), D, P=self.K_linear.astype(np.float64)
                    )
                else:
                    pts_out = cv2.undistortPoints(
                        pts_in, self.K.astype(np.float64), self.dist_coeffs, P=self.K_linear.astype(np.float64)
                    )

                u_lin, v_lin = float(pts_out[0, 0, 0]), float(pts_out[0, 0, 1])
                K_linear_inv = np.linalg.inv(self.K_linear)
                pixel_hom = np.array([u_lin, v_lin, 1.0])
                point_cam = depth * (K_linear_inv @ pixel_hom)
                point_world = self.R.T @ (point_cam - self.t)
                return point_world
            except ImportError:
                pass  # Fall through to linear path
            except Exception:
                return None

        # Linear (undistorted) path
        pixel_hom = np.array([pixel_coord[0], pixel_coord[1], 1.0])
        point_cam = depth * (self.K_inv @ pixel_hom)
        point_world = self.R.T @ (point_cam - self.t)
        return point_world

    def _get_depth_from_raster(self, x, y):
        """
        Helper to safely fetch the Z-value from the raster.
        Assumes the Z-channel represents linear depth from the camera plane.
        """
        return self._raster.get_z_value(x, y)

    # --------------------------------------------------------------------------
    # Visibility / Occlusion Logic
    # --------------------------------------------------------------------------

    def is_point_occluded_depth_based(self, point_3d, depth_threshold=0.1):
        """
        Check if a 3D point is occluded using the Raster's Z-channel (Depth Map).
        
        Compares the actual distance of the point to the value stored in the depth map.
        
        Args:
            point_3d (np.ndarray): 3D world point [x, y, z].
            depth_threshold (float): Tolerance for depth comparison (meters).

        Returns:
            bool: True if occluded, False if visible.
        """
        # 1. Project point to find which pixel it falls onto
        pixel_coord = self.project(np.array(point_3d))
        
        # If projection fails or is behind camera
        if np.isnan(pixel_coord).any():
            return True
        
        # Check if pixel is within image bounds
        if not (0 <= pixel_coord[0] < self.width and 0 <= pixel_coord[1] < self.height):
            return True  # Outside FOV

        # 2. Calculate actual distance from camera center to the 3D point
        dist_to_point = np.linalg.norm(point_3d - self.position)
        
        # 3. Get the depth recorded in the raster at that pixel
        map_depth = self._get_depth_from_raster(int(pixel_coord[0]), int(pixel_coord[1]))
        
        if map_depth is None:
            # If we have no depth data, we generally assume visibility (or handle strictly)
            return False
        
        # 4. Compare: If the point is significantly farther than the map value, it is occluded
        if dist_to_point > (map_depth + depth_threshold):
            return True
            
        return False

    def is_point_occluded_ray_casting(self, point_3d, mesh):
        """
        Check occlusion by casting a ray from the camera to the point against a 3D mesh.
        
        Args:
            point_3d (np.ndarray): 3D world point [x, y, z].
            mesh: A PyVista mesh object or wrapper with .ray_trace method.

        Returns:
            bool: True if occluded, False if visible.
        """
        if mesh is None:
            return False
        
        # Vector from camera to point
        ray_direction = point_3d - self.position
        ray_length = np.linalg.norm(ray_direction)
        
        if ray_length == 0:
            return False
            
        ray_direction = ray_direction / ray_length  # Normalize
        
        try:
            # Cast ray
            # Note: Requires mesh to be a valid PyVista object
            intersection_points, _ = mesh.ray_trace(
                self.position, 
                ray_direction + self.position,  # Some ray tracers expect end point, others direction
                first_point=True
            )
            
            if len(intersection_points) > 0:
                # Check distance to intersection
                intersection_dist = np.linalg.norm(intersection_points[0] - self.position)
                
                # If intersection is closer than the target point (with small tolerance), it's occluded
                if intersection_dist < (ray_length * 0.99):
                    return True
            
        except Exception as e:
            print(f"Ray casting error: {e}")
            return False
        
        return False

    # --------------------------------------------------------------------------
    # Visualization Logic
    # --------------------------------------------------------------------------

    def create_actor(self, plotter, scale=0.1):
        """Delegates creation of the Frustum actor to the Frustum class."""
        return self.frustum.create_actor(plotter, scale)

    def update_appearance(self, plotter=None, opacity=0.0):
        """Update the Frustum appearance based on selection state."""
        self.frustum.update_appearance(plotter, opacity)

    def select(self):
        """Mark as selected and update appearance."""
        self.selected = True
        self.frustum.select()

    def deselect(self):
        """Mark as deselected and update appearance."""
        self.selected = False
        self.frustum.deselect()

    def highlight(self):
        """Mark as highlighted and update appearance."""
        if not self.selected:  # Only highlight if not selected
            self.frustum.highlight()

    def unhighlight(self):
        """Mark as not highlighted and update appearance."""
        self.frustum.unhighlight()

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

            map_h, map_w = index_map.shape
            if (map_h, map_w) == (self.height, self.width):
                csr_pixels = _query_pixels_from_csr_inverted_index(
                    self._raster,
                    element_ids,
                    self.width,
                    self.height,
                    bbox=bbox,
                )
                if csr_pixels is not None:
                    return csr_pixels

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
