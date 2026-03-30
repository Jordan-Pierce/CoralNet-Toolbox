import os
import traceback

import numpy as np

from coralnet_toolbox.MVAT.core.Frustum import Frustum


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
        # Avoid using numpy arrays in boolean context (ambiguous truth value).
        intr_undist = getattr(raster, 'intrinsics_undistorted', None)
        self.K_linear = intr_undist if intr_undist is not None else self.K

        # --- Visualization ---
        self.selected = False
        self.is_orthographic = False  # Flag for orthographic vs perspective camera
        
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
        if self.is_distorted and self.dist_coeffs is not None:
            try:
                import cv2
                pts = np.asarray(points_3d_world, dtype=np.float64).reshape(1, 1, 3)
                rvec, _ = cv2.Rodrigues(self.R.astype(np.float64))
                projected, _ = cv2.projectPoints(
                    pts,
                    rvec,
                    self.t.astype(np.float64),
                    self.K.astype(np.float64),
                    self.dist_coeffs,
                )
                u, v = float(projected[0, 0, 0]), float(projected[0, 0, 1])
                # Check if the point is in front of the camera
                pt_cam = self.R @ np.asarray(points_3d_world, dtype=np.float64) + self.t
                if pt_cam[2] <= 0:
                    return np.array([np.nan, np.nan])
                return np.array([u, v])
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

    def unproject(self, pixel_coord):
        """
        Unproject a 2D pixel coordinate to a 3D world point.

        Requires valid depth data in the associated Raster (Z-channel).
        When the image is distorted, the raw pixel is first undistorted to its
        linear position in K_linear space before back-projection.

        Args:
            pixel_coord (tuple/list): 2D pixel [u, v] in distorted image space.

        Returns:
            np.ndarray: 3D world point [x, y, z] or None if depth is missing.
        """
        depth = self._get_depth_from_raster(int(pixel_coord[0]), int(pixel_coord[1]))
        if depth is None or depth <= 0 or np.isnan(depth):
            return None

        if self.is_distorted and self.dist_coeffs is not None and self.K_linear is not None:
            try:
                import cv2
                # Map the distorted pixel to its undistorted (linear) equivalent
                pts_in = np.array([[[float(pixel_coord[0]), float(pixel_coord[1])]]], dtype=np.float32)
                pts_out = cv2.undistortPoints(
                    pts_in,
                    self.K.astype(np.float64),
                    self.dist_coeffs,
                    P=self.K_linear.astype(np.float64),
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
    # Visibility Computation (Index Maps)
    # --------------------------------------------------------------------------
    
    def ensure_visibility_data(self, point_cloud, cache_manager, compute_depth_map: bool = True, compute_index_maps: bool = True):
        """
        Ensure visibility data (index_map and visible_indices) is computed and cached.
        
        This method implements a three-tier loading strategy:
        1. Check if data is already in memory (self._raster.visible_indices)
        2. Try to load from disk cache
        3. Compute using VisibilityManager and save to cache
        
        Args:
            point_cloud: PointCloud object containing the 3D geometry
            cache_manager: CacheManager instance for disk caching
            
        Returns:
            bool: True if visibility data is available, False otherwise
        """
        # If index-map computation is disabled by the user, skip heavy work
        if not compute_index_maps:
            return True

        # Step 1: Check if already in memory
        if self._raster.visible_indices is not None:
            return True
        
        # Step 2: Try to load from cache
        if cache_manager is not None:
            cached_data = cache_manager.load_visibility(
                self._raster.extrinsics,
                point_cloud.file_path,
                point_cloud.get_element_type()
            )
            
            if cached_data is not None:
                # Store in Raster (restore inverted index if present)
                cache_path = cache_manager.get_cache_path(
                    self._raster.extrinsics,
                    point_cloud.file_path
                )
                self._raster.add_index_map(
                    cached_data['index_map'],
                    cache_path,
                    cached_data['visible_indices'],
                    inverted_index=cached_data.get('inverted_index')
                )
                # Merge or set depth map only if caller requested depth updates
                if compute_depth_map and 'depth_map' in cached_data and cached_data['depth_map'] is not None:
                    try:
                        self._raster.merge_or_set_depth_map(cached_data['depth_map'])
                    except Exception:
                        pass
                return True
        
        # Check if we have the required data for computation
        if self._raster.extrinsics is None or self._raster.intrinsics is None:
            print(f"Warning: Camera {self.label} missing calibration data for visibility computation")
            return False
        
        try:
            # Import here to avoid circular dependencies
            from coralnet_toolbox.MVAT.managers.VisibilityManager import VisibilityManager
            
            # Get point cloud points
            points_world = point_cloud.get_points_array()
            
            if points_world is None or len(points_world) == 0:
                print(f"Warning: Point cloud is empty or invalid")
                return False
            
            # Compute visibility
            result = VisibilityManager.compute_visibility(
                points_world=points_world,
                K=self.K,
                R=self.R,
                t=self.t,
                width=self.width,
                height=self.height,
                compute_depth_map=compute_depth_map
            )
            
            # Save to cache if manager is available
            cache_path = None
            if cache_manager is not None:
                cache_path = cache_manager.save_visibility(
                    self._raster.extrinsics,
                    point_cloud.file_path,
                    result['index_map'],
                    result['visible_indices'],
                    result.get('depth_map') if (isinstance(result, dict) and compute_depth_map) else None,
                    element_type=result.get('element_type', None) if isinstance(result, dict) else None,
                    inverted_index=result.get('inverted_index') if isinstance(result, dict) else None,
                )
            
            # Store in Raster (attach inverted index so inv_* arrays are available)
            self._raster.add_index_map(
                result['index_map'],
                cache_path,
                result['visible_indices'],
                inverted_index=result.get('inverted_index') if isinstance(result, dict) else None,
            )
            # Merge or set newly computed depth map only if requested
            if compute_depth_map and 'depth_map' in result and result['depth_map'] is not None:
                try:
                    self._raster.merge_or_set_depth_map(result['depth_map'])
                except Exception:
                    pass
            
            return True
            
        except Exception as e:
            print(f"Error computing visibility for {self.label}: {e}")
            traceback.print_exc()
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


class OrthographicCamera(Camera):
    """
    Orthographic camera for handling georeferenced orthomosaics.
    Uses affine transformation instead of pinhole camera model.

    Key differences from perspective Camera:
    - No intrinsics/extrinsics (K, R, t)
    - Uses affine transform matrix for pixel <-> world coordinate conversion
    - Requires DEM (z_channel) for accurate 3D unprojection
    - Rays are parallel (no perspective distortion)
    """
    def __init__(self, raster):
        self.is_orthographic = True
        self._raster = raster

        self.width = raster.width
        self.height = raster.height

        if raster.transform_matrix is None:
            raise ValueError(f"Orthomosaic {raster.basename} missing transform_matrix")

        self.transform_matrix = np.array(raster.transform_matrix, dtype=np.float64).reshape(3, 3)

        try:
            self.transform_matrix_inv = np.linalg.inv(self.transform_matrix)
        except np.linalg.LinAlgError:
            raise ValueError(f"Transform matrix for {raster.basename} is singular (non-invertible)")

        # --- NATIVE DEM PROPERTIES ---
        self.native_dem_data = None
        self.native_dem_transform_inv = None
        z_avg = 0.0

        if raster.z_channel_path and os.path.exists(raster.z_channel_path):
            try:
                import rasterio
                with rasterio.open(raster.z_channel_path) as dem_src:
                    self.native_dem_data = dem_src.read(1).astype(np.float32)
                    nodata = dem_src.nodata
                    if nodata is not None:
                        self.native_dem_data[self.native_dem_data == nodata] = np.nan
                    
                    t = dem_src.transform
                    dem_transform = np.array([
                        [t.a, t.b, t.c],
                        [t.d, t.e, t.f],
                        [0.0, 0.0, 1.0]
                    ], dtype=np.float64)
                    
                    self.native_dem_transform_inv = np.linalg.inv(dem_transform)
                    
                    valid_dem = self.native_dem_data[~np.isnan(self.native_dem_data)]
                    if valid_dem.size > 0:
                        z_avg = float(np.mean(valid_dem))
            except Exception as e:
                print(f"WARNING: Could not load native DEM for {raster.basename}: {e}")

        # Position (Conceptual)
        center_x, center_y = self.width / 2.0, self.height / 2.0
        center_vec = np.array([center_x, center_y, 1.0], dtype=np.float64)
        world_center = (self.transform_matrix @ center_vec).flatten()
        self.position = np.array([float(world_center[0]), float(world_center[1]), z_avg + 1000.0], dtype=np.float64)

        # Stubs
        self.K = np.eye(3)
        self.R = np.eye(3)
        self.t = np.array([0.0, 0.0, 0.0])
        self.K_inv = np.eye(3)
        self.extrinsics = np.eye(4)
        self.P = np.eye(3, 4)
        self.frustum = None
        self.selected = False

        # Placeholder for injected matrix
        self.chunk_transform_inv = None

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
        """Get the image thumbnail for 3D visualization."""
        return self._raster.get_thumbnail()

    def get_native_size(self):
        """Return tuple of (width, height)."""
        return (self.width, self.height)

    @property
    def visible_indices(self):
        """Get the visible point indices for this camera."""
        return self._raster.visible_indices
    
    @property
    def index_map(self):
        """Get the index map for this camera."""
        return self._raster.index_map_lazy   
    
    # --------------------------------------------------------------------------
    # Geometric Methods
    # --------------------------------------------------------------------------

    def project(self, points_3d_world):
        """
        Project 3D world points to 2D pixel coordinates.
        Math: Local XYZ -> chunk_transform -> World XYZ -> transform_matrix_inv -> Pixel UV
        """
        points = np.atleast_2d(points_3d_world)
        N = len(points)
        
        # 1. Reverse the Bridge: Convert Local XYZ -> World XYZ
        points_hom = np.column_stack([points[:, 0], points[:, 1], points[:, 2], np.ones(N)])
        
        if self.chunk_transform_inv is not None:
            # We invert the inverse to get the forward transform
            chunk_transform = np.linalg.inv(self.chunk_transform_inv)
            world_hom = (chunk_transform @ points_hom.T).T
        else:
            world_hom = points_hom
            
        # 2. Convert World XY -> Pixel UV
        world_xy_hom = np.column_stack([world_hom[:, 0], world_hom[:, 1], np.ones(N)])
        pixels_hom = (self.transform_matrix_inv @ world_xy_hom.T).T
        
        if N == 1:
            return pixels_hom[0, :2]
        return pixels_hom[:, :2]

    def unproject(self, pixel_coord):
        """
        Unproject Ortho pixel to 3D point.
        Math: Ortho Pixel -> World XYZ (via Native DEM) -> Local XYZ (via chunk_transform)
        """        
        # 1. Transform Ortho pixel to world X, Y
        pixel_hom = np.array([pixel_coord[0], pixel_coord[1], 1.0], dtype=np.float64)
        world_hom = (self.transform_matrix @ pixel_hom).flatten()
        X, Y = float(world_hom[0]), float(world_hom[1])

        # 2. Get World Z from Native DEM
        Z = 0.0
        if self.native_dem_data is not None and self.native_dem_transform_inv is not None:
            world_dem_hom = np.array([X, Y, 1.0], dtype=np.float64)
            dem_pixel_hom = (self.native_dem_transform_inv @ world_dem_hom).flatten()
            dem_u, dem_v = int(np.floor(dem_pixel_hom[0])), int(np.floor(dem_pixel_hom[1]))
            
            h, w = self.native_dem_data.shape
            if 0 <= dem_u < w and 0 <= dem_v < h:
                Z_val = self.native_dem_data[dem_v, dem_u]
                if not np.isnan(Z_val):
                    Z = float(Z_val)

        # --- THE BRIDGE ---
        # 3. Convert World XYZ to Local XYZ
        world_xyz_hom = np.array([X, Y, Z, 1.0], dtype=np.float64)
        
        if self.chunk_transform_inv is not None:
            local_xyz_hom = (self.chunk_transform_inv @ world_xyz_hom).flatten()
            return np.array([local_xyz_hom[0], local_xyz_hom[1], local_xyz_hom[2]], dtype=np.float64)
        
        # Fallback if no transform was loaded
        return np.array([X, Y, Z], dtype=np.float64)
    
    def unproject_ray(self, pixel_coord):
        """
        Unproject Ortho pixel to a full 3D Ray (origin, direction, terminal_point).
        Math: Ortho Pixel -> World XYZ (via Native DEM) -> Local XYZ (via chunk_transform)
        """        
        # 1. Transform Ortho pixel to world X, Y
        pixel_hom = np.array([pixel_coord[0], pixel_coord[1], 1.0], dtype=np.float64)
        world_hom = (self.transform_matrix @ pixel_hom).flatten()
        X, Y = float(world_hom[0]), float(world_hom[1])

        # 2. Get World Z from Native DEM
        Z = 0.0
        if self.native_dem_data is not None and self.native_dem_transform_inv is not None:
            world_dem_hom = np.array([X, Y, 1.0], dtype=np.float64)
            dem_pixel_hom = (self.native_dem_transform_inv @ world_dem_hom).flatten()
            dem_u, dem_v = int(np.floor(dem_pixel_hom[0])), int(np.floor(dem_pixel_hom[1]))
            
            h, w = self.native_dem_data.shape
            if 0 <= dem_u < w and 0 <= dem_v < h:
                Z_val = self.native_dem_data[dem_v, dem_u]
                if not np.isnan(Z_val):
                    Z = float(Z_val)

        # 3. Create Ray Points in World Space
        # The sky is high up in World Z
        terminal_world = np.array([X, Y, Z, 1.0], dtype=np.float64)
        origin_world = np.array([X, Y, Z + 1000.0, 1.0], dtype=np.float64)

        # 4. Bridge to Local Space
        if getattr(self, 'chunk_transform_inv', None) is not None:
            terminal_local = (self.chunk_transform_inv @ terminal_world)[:3]
            origin_local = (self.chunk_transform_inv @ origin_world)[:3]
        else:
            terminal_local = terminal_world[:3]
            origin_local = origin_world[:3]

        # 5. Calculate Local Direction
        direction_local = terminal_local - origin_local
        direction_local = direction_local / np.linalg.norm(direction_local)

        return origin_local, direction_local, terminal_local
    
    def is_point_occluded_depth_based(self, point_3d, depth_threshold=0.1):
        """
        Determine if a 3D point is occluded using the NATIVE DEM.
        """    
        if self.native_dem_data is None or self.native_dem_transform_inv is None:
            return False

        point_z = point_3d.flatten()[2]
        X, Y = point_3d.flatten()[0], point_3d.flatten()[1]

        # Transform World X, Y to Native DEM pixel coordinates
        world_dem_hom = np.array([X, Y, 1.0], dtype=np.float64)
        dem_pixel_hom = (self.native_dem_transform_inv @ world_dem_hom).flatten()
        
        dem_u = int(np.floor(dem_pixel_hom[0]))
        dem_v = int(np.floor(dem_pixel_hom[1]))
        
        h, w = self.native_dem_data.shape
        if 0 <= dem_u < w and 0 <= dem_v < h:
            Z_dem = self.native_dem_data[dem_v, dem_u]
            if np.isnan(Z_dem):
                return False 
            
            return point_z < (Z_dem - depth_threshold)

        return False

    def ensure_visibility_data(self, point_cloud, cache_manager, compute_depth_map=True, compute_index_maps=True):
        """
        Ensure visibility data is computed for this orthographic camera.
        Delegates to VisibilityManager.compute_orthographic_visibility
        """
        if not compute_index_maps:
            return True

        # Check if already in memory
        if self._raster.visible_indices is not None:
            return True

        # Try to load from cache
        if cache_manager is not None:
            # Use transform_matrix as unique identifier for orthomosaics
            cached_data = cache_manager.load_visibility(
                self.transform_matrix,
                point_cloud.file_path,
                point_cloud.get_element_type()
            )
            if cached_data is not None:
                cache_path = cache_manager.get_cache_path(
                    self.transform_matrix,
                    point_cloud.file_path
                )

                self._raster.add_index_map(
                    cached_data['index_map'],
                    cache_path,
                    cached_data['visible_indices'],
                    inverted_index=cached_data.get('inverted_index')
                )
            return True
            
        # Compute visibility using orthographic method
        try:
            from coralnet_toolbox.MVAT.managers.VisibilityManager import VisibilityManager
            points_world = point_cloud.get_points_array()
            if points_world is None or len(points_world) == 0:
                print(f"Warning: Point cloud is empty or invalid")
                return False
            
            result = VisibilityManager.compute_orthographic_visibility(
                points_world=points_world,
                transform_matrix_inv=self.transform_matrix_inv,
                width=self.width,
                height=self.height,
                chunk_transform_inv=self.chunk_transform_inv
            )
            # Save to cache
            cache_path = None
            if cache_manager is not None:
                cache_path = cache_manager.save_visibility(
                    self.transform_matrix,
                    point_cloud.file_path,
                    result['index_map'],
                    result['visible_indices'],
                    None,  # No depth map for orthographic
                    element_type=result.get('element_type', None) if isinstance(result, dict) else None,
                    inverted_index=result.get('inverted_index') if isinstance(result, dict) else None,
                )
            # Store in Raster (attach inverted index when present)
            self._raster.add_index_map(
                result['index_map'],
                cache_path,
                result['visible_indices'],
                inverted_index=result.get('inverted_index') if isinstance(result, dict) else None,
            )
            return True

        except Exception as e:
            print(f"Error computing orthographic visibility for {self.label}: {e}")
            traceback.print_exc()
            return False

    # --------------------------------------------------------------------------
    # Visualization Stubs (No frustum for orthomosaics)
    # --------------------------------------------------------------------------

    def create_actor(self, plotter, scale=0.1):
        """No frustum actor for orthomosaics."""
        return None

    def update_appearance(self, plotter=None, opacity=0.0):
        """No-op for orthomosaics."""
        pass

    def select(self):
        """Mark as selected."""
        self.selected = True

    def deselect(self):
        """Mark as deselected."""
        self.selected = False

    def highlight(self):
        """No-op for orthomosaics."""
        pass

    def unhighlight(self):
        """No-op for orthomosaics."""
        pass