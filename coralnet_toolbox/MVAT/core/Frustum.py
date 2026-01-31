import numpy as np
import pyvista as pv

from PyQt5.QtGui import QImage

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
        self.highlighted = False
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
            [0, 0], # Bottom-Left
            [1, 0], # Bottom-Right
            [1, 1], # Top-Right
            [0, 1]  # Top-Left
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
                try:
                    # 1. Convert QImage to RGBA8888 (Standard byte order R-G-B-A)
                    # This avoids BGRA issues common with Format_RGB32 on Windows/LittleEndian
                    qimg = qimg.convertToFormat(QImage.Format_RGBA8888)
                    
                    width = qimg.width()
                    height = qimg.height()
                    
                    # 2. Robustly convert to Numpy Array
                    # constBits() returns a pointer to the first byte
                    ptr = qimg.constBits()
                    ptr.setsize(height * width * 4)
                    
                    # Create numpy array from buffer
                    arr = np.frombuffer(ptr, np.uint8).reshape(height, width, 4)
                    
                    # 3. Flip vertically
                    # QImage Top (Row 0) maps to OpenGL Bottom (V=0) by default.
                    # We flip it so Row 0 becomes the "Top" (V=1) to match our UVs.
                    arr = arr[::-1, :, :]
                    
                    # 4. Create PyVista Texture
                    # We create a deep copy to ensure memory safety if QImage is garbage collected
                    texture = pv.Texture(arr.copy())
                    
                except Exception as e:
                    print(f"Failed to create texture for frustum {self.camera.label}: {e}")
            
            # Add mesh to plotter
            actor = plotter.add_mesh(
                mesh,
                texture=texture,
                opacity=opacity,
                show_edges=False,
                name=f"frustum_plane_{id(self)}"
            )
            
            # 5. Disable Lighting
            # This ensures the image is shown as "emissive" (true colors) 
            # and doesn't get darkened by shadows inside the frustum.
            actor.GetProperty().SetLighting(False)
            
            self.image_actors[plotter] = actor
            
        return self.image_actors[plotter]

    def update_appearance(self, plotter=None):
        """
        Update the frustum appearance based on selection state and color.
        
        Args:
            plotter: Specific plotter to update, or None to update all
        """
        # Update visual properties based on selection and highlight
        if self.selected:
            display_color = 'lime'
            self.line_width = 3
        elif self.highlighted:
            display_color = 'cyan'
            self.line_width = 2
        else:
            display_color = 'white'
            self.line_width = 1
        
        # Update actors
        plotters_to_update = [plotter] if plotter else list(self.actors.keys())
        
        for p in plotters_to_update:
            if p in self.actors:
                actor = self.actors[p]
                
                # Set color
                prop = actor.GetProperty()
                if display_color == 'lime':
                    prop.SetColor(50 / 255, 205 / 255, 50 / 255)  # lime green
                elif display_color == 'cyan':
                    prop.SetColor(0.0, 168 / 255, 230 / 255)  # cyan
                elif display_color == 'white':
                    prop.SetColor(0.8, 0.8, 0.8)  # light gray/white
                else:
                    prop.SetColor(1.0, 1.0, 1.0)  # default white
                
                prop.SetLineWidth(self.line_width)

            # Update image plane selection highlight (Optional)
            if p in self.image_actors:
                pass 

    def select(self):
        """Mark this frustum as selected and update appearance."""
        self.selected = True
        self.highlighted = False  # Selection overrides highlight
        self.update_appearance()
    
    def deselect(self):
        """Mark this frustum as deselected and update appearance."""
        self.selected = False
        self.update_appearance()
    
    def highlight(self):
        """Mark this frustum as highlighted and update appearance."""
        if not self.selected:  # Only highlight if not selected
            self.highlighted = True
            self.update_appearance()
    
    def unhighlight(self):
        """Mark this frustum as not highlighted and update appearance."""
        self.highlighted = False
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