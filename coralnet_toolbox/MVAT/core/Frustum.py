import numpy as np
import pyvista as pv

from PyQt5.QtGui import QImage

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Frustum:
    """
    A camera frustum object that encapsulates geometric representation only.
    
    This is a geometry-only class - all actor management, selection state,
    and appearance updates are handled by MVATWindow using merged mesh rendering.
    
    The frustum provides:
    - Wireframe mesh (pyramid) for 3D visualization
    - Image plane mesh (textured quad) for thumbnail display
    - Geometry caching with scale-based invalidation
    """
    
    def __init__(self, camera, near_plane=0.1, far_plane=10.0):
        """
        Initialize a camera frustum.
        
        Args:
            camera: The Camera object this frustum belongs to
            near_plane (float): Distance to near clipping plane
            far_plane (float): Distance to far clipping plane
        """
        self.camera = camera
        self.near_plane = near_plane
        self.far_plane = far_plane
        
        # Geometry caches
        self._frustum_mesh = None
        self._image_plane_mesh = None
        self._current_scale = None
    
    def _create_geometry(self, scale=0.1):
        """
        Create the geometric meshes (wireframe and image plane) based on camera parameters.
        
        Args:
            scale (float): The distance of the image plane from the camera center.
        """
        # 1. Define corners of the near plane in camera space
        # Camera Coordinate System: X=Right, Y=Down, Z=Forward
        # (0,0) is Top-Left in image space
        w, h = self.camera.width, self.camera.height
        
        corners_pix = np.array([
            [0, 0, 1],  # Top-Left
            [w, 0, 1],  # Top-Right
            [w, h, 1],  # Bottom-Right
            [0, h, 1]   # Bottom-Left
        ])
        
        # 2. Unproject corners to 3D rays in Camera Space
        # X_cam = scale * K_inv * u
        # We transpose for matrix multiplication: (K_inv @ corners.T).T
        frustum_points_cam = scale * (self.camera.K_inv @ corners_pix.T).T
        
        # 3. Add the camera center (origin)
        all_points_cam = np.vstack([frustum_points_cam, [0, 0, 0]])
        
        # 4. Transform all points to World Coordinates
        # X_world = R.T @ (X_cam - t)
        # Note: self.camera.R and self.camera.t are World-to-Camera
        R_inv = self.camera.R.T
        t_vec = self.camera.t.reshape(3, 1)
        
        all_points_world = (R_inv @ (all_points_cam.T - t_vec)).T

        # --- Create Wireframe Mesh (The Pyramid) ---
        # Vertices: 0=TL, 1=TR, 2=BR, 3=BL, 4=Center
        cells = np.array([
            4, 0, 1, 2, 3,  # Near plane quad
            3, 4, 0, 1,     # Side (Top)
            3, 4, 1, 2,     # Side (Right)
            3, 4, 2, 3,     # Side (Bottom)
            3, 4, 3, 0      # Side (Left)
        ])
        cell_types = np.array([
            pv.CellType.QUAD,
            pv.CellType.TRIANGLE,
            pv.CellType.TRIANGLE,
            pv.CellType.TRIANGLE,
            pv.CellType.TRIANGLE
        ])
        self._frustum_mesh = pv.UnstructuredGrid(cells, cell_types, all_points_world)

        # --- Create Image Plane Mesh (Textured Quad) ---
        # We use the first 4 points (the corners) which represent the image plane at 'scale' distance
        plane_points = all_points_world[:4]
        
        # Define the face (one quad)
        # PyVista/VTK faces are: [n_points, p0, p1, p2, p3]
        plane_faces = np.array([4, 0, 1, 2, 3])
        
        self._image_plane_mesh = pv.PolyData(plane_points, plane_faces)
        
        # Define Texture Coordinates (UV)
        # Mapping:
        # Pt 0 (TL) -> UV (0, 0) (Bottom-Left of Texture)
        # Pt 1 (TR) -> UV (1, 0) (Bottom-Right of Texture)
        # Pt 2 (BR) -> UV (1, 1) (Top-Right of Texture)
        # Pt 3 (BL) -> UV (0, 1) (Top-Left of Texture)
        # Note: We will flip the image array vertically later so that Row 0 (Image Top) 
        # maps to V=1 (Texture Top).
        self._image_plane_mesh.point_data.active_texture_coordinates = np.array([
            [0, 0],  # Bottom-Left
            [1, 0],  # Bottom-Right
            [1, 1],  # Top-Right
            [0, 1]   # Top-Left
        ])
        
        self._current_scale = scale

    def get_mesh(self, scale=0.1):
        """
        Get the geometric frustum wireframe mesh.
        
        Args:
            scale: Size scale of the frustum.
            
        Returns:
            pyvista.UnstructuredGrid: The wireframe mesh.
        """
        if self._frustum_mesh is None or self._current_scale != scale:
            self._create_geometry(scale)
        return self._frustum_mesh
    
    def get_image_plane_mesh(self, scale=0.1):
        """
        Get the geometric image plane mesh with UV coordinates.
        
        Args:
            scale: Size scale of the frustum.
            
        Returns:
            pyvista.PolyData: The image plane mesh with texture coordinates.
        """
        if self._image_plane_mesh is None or self._current_scale != scale:
            self._create_geometry(scale)
        return self._image_plane_mesh
    
    def get_texture(self):
        """
        Get a PyVista texture from the camera's thumbnail.
        
        Returns:
            pyvista.Texture or None: The texture, or None if unavailable.
        """
        qimg = self.camera.get_thumbnail()
        
        if qimg is None or qimg.isNull():
            return None
        
        try:
            # Convert QImage to RGBA8888 (Standard byte order R-G-B-A)
            qimg = qimg.convertToFormat(QImage.Format_RGBA8888)
            
            width = qimg.width()
            height = qimg.height()
            
            # Robustly convert to Numpy Array
            ptr = qimg.constBits()
            ptr.setsize(height * width * 4)
            
            # Create numpy array from buffer
            arr = np.frombuffer(ptr, np.uint8).reshape(height, width, 4)
            
            # Flip vertically for OpenGL texture coordinates
            arr = arr[::-1, :, :]
            
            # Create PyVista Texture (deep copy for memory safety)
            return pv.Texture(arr.copy())
            
        except Exception as e:
            print(f"Failed to create texture for frustum {self.camera.label}: {e}")
            return None
    
    def clear_cache(self):
        """Clear cached geometry to force regeneration on next access."""
        self._frustum_mesh = None
        self._image_plane_mesh = None
        self._current_scale = None