import numpy as np
import pyvista as pv

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Frustum:
    """
    A camera frustum object that encapsulates the geometric representation,
    visual properties, and PyVista actor management.
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
        
        # Visual properties
        self.selected = False
        self.color = 'cyan'  # Default color
        self.line_width = 1  # Default line width
        
        # PyVista actors for different plotters
        self.actors = {}        # plotter -> wireframe actor mapping
        self.image_actors = {}  # plotter -> image plane actor mapping
        
        # Geometry caches
        self._frustum_mesh = None
        self._image_plane_mesh = None
        self._current_scale = None
    
    def _create_geometry(self, scale=0.1):
        """
        Create the geometric meshes (wireframe and image plane) based on camera parameters.
        
        Args:
            scale (float): The distance of the near plane from the camera center.
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
        # We use the first 4 points (the corners)
        plane_points = all_points_world[:4]
        
        # Define the face (one quad)
        # PyVista/VTK faces are: [n_points, p0, p1, p2, p3]
        plane_faces = np.array([4, 0, 1, 2, 3])
        
        self._image_plane_mesh = pv.PolyData(plane_points, plane_faces)
        
        # Define Texture Coordinates (UV)
        # We need to map the image (0,0 at TL) to the mesh geometry
        # V=1 is usually "up" in texture space, but our Y=0 is "up" in image space.
        # Mapping:
        # Pt 0 (TL) -> UV (0, 1)
        # Pt 1 (TR) -> UV (1, 1)
        # Pt 2 (BR) -> UV (1, 0)
        # Pt 3 (BL) -> UV (0, 0)
        self._image_plane_mesh.point_data["t_coords"] = np.array([
            [0, 1], # Top-Left
            [1, 1], # Top-Right
            [1, 0], # Bottom-Right
            [0, 0]  # Bottom-Left
        ])

    def get_mesh(self, scale=0.1):
        """Get the geometric frustum wireframe mesh."""
        if self._frustum_mesh is None or self._current_scale != scale:
            self._create_geometry(scale)
            self._current_scale = scale
        return self._frustum_mesh
    
    def get_image_plane_mesh(self, scale=0.1):
        """Get the geometric image plane mesh with UV coordinates."""
        if self._image_plane_mesh is None or self._current_scale != scale:
            self._create_geometry(scale)
            self._current_scale = scale
        return self._image_plane_mesh

    def create_actor(self, plotter, scale=0.1):
        """
        Create the wireframe actor for the frustum.
        
        Args:
            plotter: The PyVista plotter
            scale: Size scale of the frustum
        """
        if plotter not in self.actors:
            mesh = self.get_mesh(scale)
            self.actors[plotter] = plotter.add_mesh(
                mesh,
                style='wireframe',
                color=self.color,
                line_width=self.line_width,
                name=f"frustum_wire_{id(self)}" # Unique name
            )
        return self.actors[plotter]

    def create_image_plane_actor(self, plotter, scale=0.1, opacity=0.8):
        """
        Create the textured image plane actor.
        
        Args:
            plotter: The PyVista plotter
            scale: Size scale of the frustum
            opacity: Opacity of the image plane
        """
        if plotter not in self.image_actors:
            mesh = self.get_image_plane_mesh(scale)
            
            # Get thumbnail from camera
            qimg = self.camera.get_thumbnail()
            texture = None
            
            if qimg and not qimg.isNull():
                # Convert QImage to numpy array for PyVista texture
                # Assumes QImage format is compatible (e.g. RGB/RGBA)
                try:
                    qimg = qimg.convertToFormat(4) # QImage.Format_RGB32
                    width = qimg.width()
                    height = qimg.height()
                    
                    ptr = qimg.bits()
                    ptr.setsize(qimg.byteCount())
                    # Reshape to (H, W, 4) for RGBA
                    arr = np.array(ptr).reshape(height, width, 4)
                    
                    # Create PyVista Texture
                    texture = pv.Texture(arr)
                except Exception as e:
                    print(f"Failed to create texture for frustum: {e}")
            
            # Add mesh to plotter
            self.image_actors[plotter] = plotter.add_mesh(
                mesh,
                texture=texture,
                opacity=opacity,
                show_edges=False,
                name=f"frustum_plane_{id(self)}"
            )
            
        return self.image_actors[plotter]

    def update_appearance(self, plotter=None):
        """
        Update the frustum appearance based on selection state.
        
        Args:
            plotter: Specific plotter to update, or None to update all
        """
        # Update visual properties based on selection
        self.color = 'red' if self.selected else 'cyan'
        self.line_width = 3 if self.selected else 1
        
        # Update actors
        plotters_to_update = [plotter] if plotter else list(self.actors.keys())
        
        for p in plotters_to_update:
            if p in self.actors:
                actor = self.actors[p]
                
                # Set color
                prop = actor.GetProperty()
                if self.color == 'red':
                    prop.SetColor(230 / 255, 62 / 255, 0 / 255)  # blood red (230, 62, 0)
                elif self.color == 'cyan':
                    prop.SetColor(0.0, 168 / 255, 230 / 255)  # cyan (0, 168, 230)
                else:
                    prop.SetColor(1.0, 1.0, 1.0)
                
                prop.SetLineWidth(self.line_width)

            # Update image plane border/highlight if needed
            if p in self.image_actors:
                # We could adjust opacity or add an outline here if selected
                pass

    def select(self):
        """Mark this frustum as selected and update appearance."""
        self.selected = True
        self.update_appearance()
    
    def deselect(self):
        """Mark this frustum as deselected and update appearance."""
        self.selected = False
        self.update_appearance()
    
    def remove_from_plotter(self, plotter):
        """Remove this frustum's actors from a specific plotter."""
        # Remove wireframe
        if plotter in self.actors:
            plotter.remove_actor(self.actors[plotter])
            del self.actors[plotter]
            
        # Remove image plane
        if plotter in self.image_actors:
            plotter.remove_actor(self.image_actors[plotter])
            del self.image_actors[plotter]