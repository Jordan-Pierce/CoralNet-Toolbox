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

    # --------------------------------------------------------------------------
    # Geometric Methods
    # --------------------------------------------------------------------------
    
    def project(self, points_3d_world):
        """
        Project a 3D world point into a 2D pixel coordinate.
        
        Math: 
        $x_{pix} = P \cdot X_{world}$
        
        Args:
            points_3d_world (np.ndarray): 3D point [x, y, z] in world coordinates.

        Returns:
            np.ndarray: 2D pixel coordinate [u, v] or [nan, nan] if invalid.
        """
        # Add homogeneous coordinate (append 1.0)
        points_hom = np.hstack((points_3d_world, 1.0))
        
        # Project using the P matrix
        points_cam_hom = (self.P @ points_hom.T).T
        
        # Normalize by dividing by the 3rd component (depth Z in camera frame)
        points_pixel = np.full(2, np.nan)
        
        # Check if point is in front of the camera (Z > 0)
        if points_cam_hom[2] > 0:
            points_pixel = points_cam_hom[:2] / points_cam_hom[2]
            
        return points_pixel

    def unproject(self, pixel_coord):
        """
        Unproject a 2D pixel coordinate to a 3D world point.
        
        Requires valid depth data in the associated Raster (Z-channel).
        
        Args:
            pixel_coord (tuple/list): 2D pixel [u, v].

        Returns:
            np.ndarray: 3D world point [x, y, z] or None if depth is missing.
        """
        # 1. Get the depth value at this pixel
        depth = self._get_depth_from_raster(int(pixel_coord[0]), int(pixel_coord[1]))
        
        if depth is None or depth <= 0 or np.isnan(depth):
            return None
            
        # 2. Create homogeneous pixel coordinate
        pixel_hom = np.array([pixel_coord[0], pixel_coord[1], 1])
        
        # 3. Transform to Camera Coordinate System (Back-projection)
        # $X_{cam} = Z \cdot K^{-1} \cdot x_{pix}$
        point_cam = depth * (self.K_inv @ pixel_hom)
        
        # 4. Transform to World Coordinate System
        # $X_{world} = R^T \cdot (X_{cam} - t)$
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
            return True # Outside FOV

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
    
    def ensure_visibility_data(self, point_cloud, cache_manager):
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
        # Step 1: Check if already in memory
        if self._raster.visible_indices is not None:
            return True
        
        # Step 2: Try to load from cache
        if cache_manager is not None:
            cached_data = cache_manager.load_visibility(
                self._raster.extrinsics,
                point_cloud.file_path
            )
            
            if cached_data is not None:
                # Store in Raster
                cache_path = cache_manager.get_cache_path(
                    self._raster.extrinsics,
                    point_cloud.file_path
                )
                self._raster.add_index_map(
                    cached_data['index_map'],
                    cache_path,
                    cached_data['visible_indices']
                )
                return True
        
        # Step 3: Compute visibility using VisibilityManager
        # TODO: Implement frustum-based cone intersection as fallback when z_channel is None.
        # Could use conservative rasterization of frustum volume.
        
        # Check if we have the required data for computation
        if self._raster.extrinsics is None or self._raster.intrinsics is None:
            print(f"Warning: Camera {self.label} missing calibration data for visibility computation")
            return False
        
        try:
            # Import here to avoid circular dependencies
            from coralnet_toolbox.MVAT.core.VisibilityManager import VisibilityManager
            
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
                height=self.height
            )
            
            # Save to cache if manager is available
            cache_path = None
            if cache_manager is not None:
                cache_path = cache_manager.save_visibility(
                    self._raster.extrinsics,
                    point_cloud.file_path,
                    result['index_map'],
                    result['visible_indices']
                )
            
            # Store in Raster
            self._raster.add_index_map(
                result['index_map'],
                cache_path,
                result['visible_indices']
            )
            
            return True
            
        except Exception as e:
            print(f"Error computing visibility for {self.label}: {e}")
            import traceback
            traceback.print_exc()
            return False

    # --------------------------------------------------------------------------
    # Visualization Logic
    # --------------------------------------------------------------------------

    def create_actor(self, plotter, scale=0.1):
        """Delegates creation of the Frustum actor to the Frustum class."""
        return self.frustum.create_actor(plotter, scale)

    def update_appearance(self, plotter=None, opacity=0.8):
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