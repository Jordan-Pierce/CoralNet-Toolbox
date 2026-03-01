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
    
    def ensure_visibility_data(self, point_cloud, cache_manager, compute_depth_map: bool = True, 
                               compute_index_maps: bool = True):
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
                # Merge or set depth map only if caller requested depth updates
                if compute_depth_map and 'depth_map' in cached_data and cached_data['depth_map'] is not None:
                    try:
                        self._raster.merge_or_set_depth_map(cached_data['depth_map'])
                    except Exception:
                        pass
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
                    result.get('depth_map') if (isinstance(result, dict) and compute_depth_map) else None
                )
            
            # Store in Raster
            self._raster.add_index_map(
                result['index_map'],
                cache_path,
                result['visible_indices']
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
            import traceback
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
 
 
 
 #   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 
 #   O r t h o g r a p h i c C a m e r a   C l a s s 
 
 #   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 
 
 
 
 
 c l a s s   O r t h o g r a p h i c C a m e r a : 
 
         " " " 
 
         O r t h o g r a p h i c   c a m e r a   f o r   h a n d l i n g   g e o r e f e r e n c e d   o r t h o m o s a i c s . 
 
         U s e s   a f f i n e   t r a n s f o r m a t i o n   i n s t e a d   o f   p i n h o l e   c a m e r a   m o d e l . 
 
         
 
         K e y   d i f f e r e n c e s   f r o m   p e r s p e c t i v e   C a m e r a : 
 
         -   N o   i n t r i n s i c s / e x t r i n s i c s   ( K ,   R ,   t ) 
 
         -   U s e s   a f f i n e   t r a n s f o r m   m a t r i x   f o r   p i x e l   < - >   w o r l d   c o o r d i n a t e   c o n v e r s i o n 
 
         -   R e q u i r e s   D E M   ( z _ c h a n n e l )   f o r   a c c u r a t e   3 D   u n p r o j e c t i o n 
 
         -   R a y s   a r e   p a r a l l e l   ( n o   p e r s p e c t i v e   d i s t o r t i o n ) 
 
         " " " 
 
         
 
         d e f   _ _ i n i t _ _ ( s e l f ,   r a s t e r ) : 
 
                 " " " 
 
                 I n i t i a l i z e   o r t h o g r a p h i c   c a m e r a   f r o m   r a s t e r   w i t h   D E M . 
 
                 
 
                 A r g s : 
 
                         r a s t e r :   A   R a s t e r   o b j e c t   w i t h   t r a n s f o r m _ m a t r i x   a n d   z _ c h a n n e l   ( D E M ) 
 
                         
 
                 R a i s e s : 
 
                         V a l u e E r r o r :   I f   t r a n s f o r m _ m a t r i x   i s   m i s s i n g   o r   s i n g u l a r 
 
                 " " " 
 
                 i m p o r t   n u m p y   a s   n p 
 
                 
 
                 #   C o r e   f l a g 
 
                 s e l f . i s _ o r t h o g r a p h i c   =   T r u e 
 
                 s e l f . _ r a s t e r   =   r a s t e r 
 
                 
 
                 #   D i m e n s i o n s 
 
                 s e l f . w i d t h   =   r a s t e r . w i d t h 
 
                 s e l f . h e i g h t   =   r a s t e r . h e i g h t 
 
                 
 
                 #   E x t r a c t   a f f i n e   t r a n s f o r m 
 
                 i f   r a s t e r . t r a n s f o r m _ m a t r i x   i s   N o n e : 
 
                         r a i s e   V a l u e E r r o r ( f " O r t h o m o s a i c   { r a s t e r . b a s e n a m e }   m i s s i n g   t r a n s f o r m _ m a t r i x " ) 
 
                 
 
                 s e l f . t r a n s f o r m _ m a t r i x   =   r a s t e r . t r a n s f o r m _ m a t r i x . c o p y ( ) 
 
                 
 
                 #   C o m p u t e   a n d   c a c h e   i n v e r s e   ( m i c r o s e c o n d   o p e r a t i o n   f o r   3 x 3 ) 
 
                 t r y : 
 
                         s e l f . t r a n s f o r m _ m a t r i x _ i n v   =   n p . l i n a l g . i n v ( s e l f . t r a n s f o r m _ m a t r i x ) 
 
                 e x c e p t   n p . l i n a l g . L i n A l g E r r o r : 
 
                         r a i s e   V a l u e E r r o r ( f " T r a n s f o r m   m a t r i x   f o r   { r a s t e r . b a s e n a m e }   i s   s i n g u l a r   ( n o n - i n v e r t i b l e ) " ) 
 
                 
 
                 #   D E M   h a n d l i n g   w i t h   f a l l b a c k 
 
                 s e l f . z _ c h a n n e l   =   r a s t e r . z _ c h a n n e l 
 
                 i f   s e l f . z _ c h a n n e l   i s   N o n e : 
 
                         p r i n t ( f " � a� � � �   W A R N I N G :   { r a s t e r . b a s e n a m e }   h a s   n o   D E M .   A s s u m i n g   f l a t   t e r r a i n   a t   Z = 0 " ) 
 
                         s e l f . z _ c h a n n e l   =   n p . z e r o s ( ( s e l f . h e i g h t ,   s e l f . w i d t h ) ,   d t y p e = n p . f l o a t 3 2 ) 
 
                 e l s e : 
 
                         s e l f . z _ c h a n n e l   =   s e l f . z _ c h a n n e l . c o p y ( ) 
 
                 
 
                 #   P e r f o r m a n c e   c h e c k :   d o w n s a m p l e   i f   >   5 0 0 M B 
 
                 d e m _ s i z e _ m b   =   s e l f . z _ c h a n n e l . n b y t e s   /   1 _ 0 0 0 _ 0 0 0 
 
                 i f   d e m _ s i z e _ m b   >   5 0 0 : 
 
                         p r i n t ( f " � a� � � �   D E M   s i z e   { d e m _ s i z e _ m b : . 1 f } M B   e x c e e d s   5 0 0 M B   t h r e s h o l d .   D o w n s a m p l i n g   b y   f a c t o r   o f   2 . . . " ) 
 
                         s e l f . z _ c h a n n e l   =   s e l f . z _ c h a n n e l [ : : 2 ,   : : 2 ] 
 
                         s e l f . h e i g h t ,   s e l f . w i d t h   =   s e l f . z _ c h a n n e l . s h a p e 
 
                         #   S c a l e   t r a n s f o r m   m a t r i x   a c c o r d i n g l y   ( 2 x   p i x e l   s p a c i n g ) 
 
                         s c a l e _ m a t r i x   =   n p . a r r a y ( [ [ 2 ,   0 ,   0 ] ,   [ 0 ,   2 ,   0 ] ,   [ 0 ,   0 ,   1 ] ] ) 
 
                         s e l f . t r a n s f o r m _ m a t r i x   =   s e l f . t r a n s f o r m _ m a t r i x   @   s c a l e _ m a t r i x 
 
                         s e l f . t r a n s f o r m _ m a t r i x _ i n v   =   n p . l i n a l g . i n v ( s e l f . t r a n s f o r m _ m a t r i x ) 
 
                 
 
                 #   " C a m e r a "   p o s i t i o n   ( c o n c e p t u a l   -   d i r e c t l y   a b o v e   s c e n e   c e n t e r ) 
 
                 c e n t e r _ x ,   c e n t e r _ y   =   s e l f . w i d t h   /   2 ,   s e l f . h e i g h t   /   2 
 
                 w o r l d _ c e n t e r   =   s e l f . t r a n s f o r m _ m a t r i x   @   n p . a r r a y ( [ c e n t e r _ x ,   c e n t e r _ y ,   1 ] ) 
 
                 z _ a v g   =   f l o a t ( n p . n a n m e a n ( s e l f . z _ c h a n n e l ) ) 
 
                 s e l f . p o s i t i o n   =   n p . a r r a y ( [ w o r l d _ c e n t e r [ 0 ] ,   w o r l d _ c e n t e r [ 1 ] ,   z _ a v g   +   1 0 0 0 ] ) 
 
                 
 
                 #   N o   f r u s t u m   f o r   o r t h o m o s a i c s 
 
                 s e l f . f r u s t u m   =   N o n e 
 
                 s e l f . s e l e c t e d   =   F a l s e 
 
         
 
         #   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 
         #   P r o p e r t i e s   ( D e l e g a t e d   t o   R a s t e r ) 
 
         #   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 
         
 
         @ p r o p e r t y 
 
         d e f   l a b e l ( s e l f ) : 
 
                 " " " R e t u r n   t h e   l a b e l / f i l e n a m e   o f   t h e   a s s o c i a t e d   r a s t e r . " " " 
 
                 r e t u r n   s e l f . _ r a s t e r . b a s e n a m e 
 
         
 
         @ p r o p e r t y 
 
         d e f   i m a g e _ p a t h ( s e l f ) : 
 
                 " " " R e t u r n   t h e   f i l e   p a t h   o f   t h e   s o u r c e   i m a g e . " " " 
 
                 r e t u r n   s e l f . _ r a s t e r . i m a g e _ p a t h 
 
         
 
         d e f   g e t _ t h u m b n a i l ( s e l f ) : 
 
                 " " " G e t   t h e   i m a g e   t h u m b n a i l   f o r   3 D   v i s u a l i z a t i o n . " " " 
 
                 r e t u r n   s e l f . _ r a s t e r . g e t _ t h u m b n a i l ( ) 
 
         
 
         d e f   g e t _ n a t i v e _ s i z e ( s e l f ) : 
 
                 " " " R e t u r n   t u p l e   o f   ( w i d t h ,   h e i g h t ) . " " " 
 
                 r e t u r n   ( s e l f . w i d t h ,   s e l f . h e i g h t ) 
 
         
 
         @ p r o p e r t y 
 
         d e f   v i s i b l e _ i n d i c e s ( s e l f ) : 
 
                 " " " G e t   t h e   v i s i b l e   p o i n t   i n d i c e s   f o r   t h i s   c a m e r a . " " " 
 
                 r e t u r n   s e l f . _ r a s t e r . v i s i b l e _ i n d i c e s 
 
         
 
         @ p r o p e r t y 
 
         d e f   i n d e x _ m a p ( s e l f ) : 
 
                 " " " G e t   t h e   i n d e x   m a p   f o r   t h i s   c a m e r a . " " " 
 
                 r e t u r n   s e l f . _ r a s t e r . i n d e x _ m a p _ l a z y 
 
         
 
         #   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 
         #   G e o m e t r i c   M e t h o d s 
 
         #   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 
         
 
         d e f   p r o j e c t ( s e l f ,   p o i n t s _ 3 d _ w o r l d ) : 
 
                 " " " 
 
                 P r o j e c t   3 D   w o r l d   p o i n t s   t o   2 D   p i x e l   c o o r d i n a t e s   ( i g n o r i n g   Z ) . 
 
                 
 
                 U s e s   a f f i n e   t r a n s f o r m a t i o n :   [ u ,   v ,   1 ]   =   T _ i n v   @   [ X ,   Y ,   1 ] 
 
                 
 
                 A r g s : 
 
                         p o i n t s _ 3 d _ w o r l d   ( n p . n d a r r a y ) :   3 D   p o i n t ( s )   [ x ,   y ,   z ]   i n   w o r l d   c o o r d i n a t e s 
 
                         
 
                 R e t u r n s : 
 
                         n p . n d a r r a y :   2 D   p i x e l   c o o r d i n a t e ( s )   [ u ,   v ] 
 
                 " " " 
 
                 i m p o r t   n u m p y   a s   n p 
 
                 
 
                 p o i n t s   =   n p . a t l e a s t _ 2 d ( p o i n t s _ 3 d _ w o r l d ) 
 
                 N   =   l e n ( p o i n t s ) 
 
                 
 
                 #   H o m o g e n e o u s   c o o r d i n a t e s   [ X ,   Y ,   1 ]   ( i g n o r e   Z   f o r   o r t h o g r a p h i c ) 
 
                 p o i n t s _ h o m   =   n p . c o l u m n _ s t a c k ( [ p o i n t s [ : ,   0 ] ,   p o i n t s [ : ,   1 ] ,   n p . o n e s ( N ) ] ) 
 
                 
 
                 #   T r a n s f o r m   t o   p i x e l   s p a c e 
 
                 p i x e l s _ h o m   =   ( s e l f . t r a n s f o r m _ m a t r i x _ i n v   @   p o i n t s _ h o m . T ) . T 
 
                 
 
                 #   R e t u r n   [ u ,   v ]   ( d r o p   h o m o g e n e o u s   c o o r d i n a t e ) 
 
                 i f   N   = =   1 : 
 
                         r e t u r n   p i x e l s _ h o m [ 0 ,   : 2 ] 
 
                 r e t u r n   p i x e l s _ h o m [ : ,   : 2 ] 
 
         
 
         d e f   u n p r o j e c t ( s e l f ,   p i x e l _ c o o r d ) : 
 
                 " " " 
 
                 U n p r o j e c t   p i x e l   t o   3 D   w o r l d   p o i n t   u s i n g   D E M . 
 
                 
 
                 U s e s   a f f i n e   t r a n s f o r m a t i o n :   [ X ,   Y ,   1 ]   =   T   @   [ u ,   v ,   1 ] 
 
                 Z   i s   q u e r i e d   f r o m   t h e   D E M   a t   t h e   p i x e l   l o c a t i o n . 
 
                 
 
                 A r g s : 
 
                         p i x e l _ c o o r d   ( t u p l e / l i s t ) :   2 D   p i x e l   [ u ,   v ] 
 
                         
 
                 R e t u r n s : 
 
                         n p . n d a r r a y :   3 D   w o r l d   p o i n t   [ x ,   y ,   z ] 
 
                 " " " 
 
                 i m p o r t   n u m p y   a s   n p 
 
                 
 
                 u ,   v   =   p i x e l _ c o o r d 
 
                 
 
                 #   C l a m p   t o   v a l i d   i m a g e   b o u n d s   ( C R I T I C A L   f o r   P y V i s t a   p i c k s ) 
 
                 u   =   n p . c l i p ( i n t ( u ) ,   0 ,   s e l f . w i d t h   -   1 ) 
 
                 v   =   n p . c l i p ( i n t ( v ) ,   0 ,   s e l f . h e i g h t   -   1 ) 
 
                 
 
                 #   T r a n s f o r m   p i x e l   t o   w o r l d   X ,   Y 
 
                 p i x e l _ h o m   =   n p . a r r a y ( [ u ,   v ,   1 ] ) 
 
                 w o r l d _ h o m   =   s e l f . t r a n s f o r m _ m a t r i x   @   p i x e l _ h o m 
 
                 X ,   Y   =   w o r l d _ h o m [ 0 ] ,   w o r l d _ h o m [ 1 ] 
 
                 
 
                 #   Q u e r y   D E M   f o r   Z   ( h a n d l e   N a N / n o d a t a ) 
 
                 Z   =   s e l f . z _ c h a n n e l [ v ,   u ] 
 
                 i f   n p . i s n a n ( Z )   o r   ( s e l f . _ r a s t e r . z _ n o d a t a   i s   n o t   N o n e   a n d   Z   = =   s e l f . _ r a s t e r . z _ n o d a t a ) : 
 
                         Z   =   0 . 0     #   F a l l b a c k   t o   g r o u n d   l e v e l 
 
                 
 
                 r e t u r n   n p . a r r a y ( [ X ,   Y ,   Z ] ) 
 
         
 
         d e f   i s _ p o i n t _ o c c l u d e d _ d e p t h _ b a s e d ( s e l f ,   p o i n t _ 3 d ,   d e p t h _ t h r e s h o l d = 0 . 1 ) : 
 
                 " " " 
 
                 C h e c k   i f   3 D   p o i n t   i s   o c c l u d e d   ( u n d e r g r o u n d )   u s i n g   D E M . 
 
                 
 
                 A r g s : 
 
                         p o i n t _ 3 d   ( n p . n d a r r a y ) :   3 D   w o r l d   p o i n t   [ x ,   y ,   z ] 
 
                         d e p t h _ t h r e s h o l d   ( f l o a t ) :   T o l e r a n c e   b e l o w   g r o u n d   ( m e t e r s ) 
 
                         
 
                 R e t u r n s : 
 
                         b o o l :   T r u e   i f   o c c l u d e d ,   F a l s e   i f   v i s i b l e 
 
                 " " " 
 
                 i m p o r t   n u m p y   a s   n p 
 
                 
 
                 #   P r o j e c t   t o   p i x e l 
 
                 u v   =   s e l f . p r o j e c t ( p o i n t _ 3 d . r e s h a p e ( 1 ,   3 ) ) 
 
                 i f   n p . i s n a n ( u v ) . a n y ( ) : 
 
                         r e t u r n   T r u e 
 
                 
 
                 u ,   v   =   i n t ( n p . c l i p ( u v [ 0 ] ,   0 ,   s e l f . w i d t h   -   1 ) ) ,   i n t ( n p . c l i p ( u v [ 1 ] ,   0 ,   s e l f . h e i g h t   -   1 ) ) 
 
                 
 
                 #   C h e c k   i f   p i x e l   i s   w i t h i n   b o u n d s 
 
                 i f   n o t   ( 0   < =   u   <   s e l f . w i d t h   a n d   0   < =   v   <   s e l f . h e i g h t ) : 
 
                         r e t u r n   T r u e 
 
                 
 
                 #   G e t   D E M   h e i g h t 
 
                 Z _ d e m   =   s e l f . z _ c h a n n e l [ v ,   u ] 
 
                 i f   n p . i s n a n ( Z _ d e m ) : 
 
                         r e t u r n   F a l s e     #   C a n ' t   d e t e r m i n e   o c c l u s i o n 
 
                 
 
                 #   P o i n t   i s   o c c l u d e d   i f   b e l o w   g r o u n d 
 
                 r e t u r n   p o i n t _ 3 d [ 2 ]   <   ( Z _ d e m   -   d e p t h _ t h r e s h o l d ) 
 
         
 
         d e f   e n s u r e _ v i s i b i l i t y _ d a t a ( s e l f ,   p o i n t _ c l o u d ,   c a c h e _ m a n a g e r ,   c o m p u t e _ d e p t h _ m a p = T r u e ,   c o m p u t e _ i n d e x _ m a p s = T r u e ) : 
 
                 " " " 
 
                 E n s u r e   v i s i b i l i t y   d a t a   i s   c o m p u t e d   f o r   t h i s   o r t h o g r a p h i c   c a m e r a . 
 
                 
 
                 D e l e g a t e s   t o   V i s i b i l i t y M a n a g e r . c o m p u t e _ o r t h o g r a p h i c _ v i s i b i l i t y 
 
                 " " " 
 
                 i f   n o t   c o m p u t e _ i n d e x _ m a p s : 
 
                         r e t u r n   T r u e 
 
                 
 
                 #   C h e c k   i f   a l r e a d y   i n   m e m o r y 
 
                 i f   s e l f . _ r a s t e r . v i s i b l e _ i n d i c e s   i s   n o t   N o n e : 
 
                         r e t u r n   T r u e 
 
                 
 
                 #   T r y   t o   l o a d   f r o m   c a c h e 
 
                 i f   c a c h e _ m a n a g e r   i s   n o t   N o n e : 
 
                         #   U s e   t r a n s f o r m _ m a t r i x   a s   u n i q u e   i d e n t i f i e r   f o r   o r t h o m o s a i c s 
 
                         c a c h e d _ d a t a   =   c a c h e _ m a n a g e r . l o a d _ v i s i b i l i t y ( 
 
                                 s e l f . t r a n s f o r m _ m a t r i x , 
 
                                 p o i n t _ c l o u d . f i l e _ p a t h 
 
                         ) 
 
                         
 
                         i f   c a c h e d _ d a t a   i s   n o t   N o n e : 
 
                                 c a c h e _ p a t h   =   c a c h e _ m a n a g e r . g e t _ c a c h e _ p a t h ( 
 
                                         s e l f . t r a n s f o r m _ m a t r i x , 
 
                                         p o i n t _ c l o u d . f i l e _ p a t h 
 
                                 ) 
 
                                 s e l f . _ r a s t e r . a d d _ i n d e x _ m a p ( 
 
                                         c a c h e d _ d a t a [ ' i n d e x _ m a p ' ] , 
 
                                         c a c h e _ p a t h , 
 
                                         c a c h e d _ d a t a [ ' v i s i b l e _ i n d i c e s ' ] 
 
                                 ) 
 
                                 r e t u r n   T r u e 
 
                 
 
                 #   C o m p u t e   v i s i b i l i t y   u s i n g   o r t h o g r a p h i c   m e t h o d 
 
                 t r y : 
 
                         f r o m   c o r a l n e t _ t o o l b o x . M V A T . m a n a g e r s . V i s i b i l i t y M a n a g e r   i m p o r t   V i s i b i l i t y M a n a g e r 
 
                         
 
                         p o i n t s _ w o r l d   =   p o i n t _ c l o u d . g e t _ p o i n t s _ a r r a y ( ) 
 
                         i f   p o i n t s _ w o r l d   i s   N o n e   o r   l e n ( p o i n t s _ w o r l d )   = =   0 : 
 
                                 p r i n t ( f " W a r n i n g :   P o i n t   c l o u d   i s   e m p t y   o r   i n v a l i d " ) 
 
                                 r e t u r n   F a l s e 
 
                         
 
                         r e s u l t   =   V i s i b i l i t y M a n a g e r . c o m p u t e _ o r t h o g r a p h i c _ v i s i b i l i t y ( 
 
                                 p o i n t s _ w o r l d = p o i n t s _ w o r l d , 
 
                                 t r a n s f o r m _ m a t r i x _ i n v = s e l f . t r a n s f o r m _ m a t r i x _ i n v , 
 
                                 w i d t h = s e l f . w i d t h , 
 
                                 h e i g h t = s e l f . h e i g h t 
 
                         ) 
 
                         
 
                         #   S a v e   t o   c a c h e 
 
                         c a c h e _ p a t h   =   N o n e 
 
                         i f   c a c h e _ m a n a g e r   i s   n o t   N o n e : 
 
                                 c a c h e _ p a t h   =   c a c h e _ m a n a g e r . s a v e _ v i s i b i l i t y ( 
 
                                         s e l f . t r a n s f o r m _ m a t r i x , 
 
                                         p o i n t _ c l o u d . f i l e _ p a t h , 
 
                                         r e s u l t [ ' i n d e x _ m a p ' ] , 
 
                                         r e s u l t [ ' v i s i b l e _ i n d i c e s ' ] , 
 
                                         N o n e     #   N o   d e p t h   m a p   f o r   o r t h o g r a p h i c 
 
                                 ) 
 
                         
 
                         #   S t o r e   i n   R a s t e r 
 
                         s e l f . _ r a s t e r . a d d _ i n d e x _ m a p ( 
 
                                 r e s u l t [ ' i n d e x _ m a p ' ] , 
 
                                 c a c h e _ p a t h , 
 
                                 r e s u l t [ ' v i s i b l e _ i n d i c e s ' ] 
 
                         ) 
 
                         
 
                         r e t u r n   T r u e 
 
                         
 
                 e x c e p t   E x c e p t i o n   a s   e : 
 
                         p r i n t ( f " E r r o r   c o m p u t i n g   o r t h o g r a p h i c   v i s i b i l i t y   f o r   { s e l f . l a b e l } :   { e } " ) 
 
                         i m p o r t   t r a c e b a c k 
 
                         t r a c e b a c k . p r i n t _ e x c ( ) 
 
                         r e t u r n   F a l s e 
 
         
 
         #   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 
         #   V i s u a l i z a t i o n   S t u b s   ( N o   f r u s t u m   f o r   o r t h o m o s a i c s ) 
 
         #   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 
         
 
         d e f   c r e a t e _ a c t o r ( s e l f ,   p l o t t e r ,   s c a l e = 0 . 1 ) : 
 
                 " " " N o   f r u s t u m   a c t o r   f o r   o r t h o m o s a i c s . " " " 
 
                 r e t u r n   N o n e 
 
         
 
         d e f   u p d a t e _ a p p e a r a n c e ( s e l f ,   p l o t t e r = N o n e ,   o p a c i t y = 0 . 0 ) : 
 
                 " " " N o - o p   f o r   o r t h o m o s a i c s . " " " 
 
                 p a s s 
 
         
 
         d e f   s e l e c t ( s e l f ) : 
 
                 " " " M a r k   a s   s e l e c t e d . " " " 
 
                 s e l f . s e l e c t e d   =   T r u e 
 
         
 
         d e f   d e s e l e c t ( s e l f ) : 
 
                 " " " M a r k   a s   d e s e l e c t e d . " " " 
 
                 s e l f . s e l e c t e d   =   F a l s e 
 
         
 
         d e f   h i g h l i g h t ( s e l f ) : 
 
                 " " " N o - o p   f o r   o r t h o m o s a i c s . " " " 
 
                 p a s s 
 
         
 
         d e f   u n h i g h l i g h t ( s e l f ) : 
 
                 " " " N o - o p   f o r   o r t h o m o s a i c s . " " " 
 
                 p a s s 
 
 