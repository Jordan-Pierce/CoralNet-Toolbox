from typing import Dict, List, Optional, Tuple

import numpy as np

import pyvista as pv

from PyQt5.QtGui import QImage

from coralnet_toolbox.MVAT.core.constants import (
    SELECT_COLOR_RGB,
    HIGHLIGHT_COLOR_RGB,
    STATE_DEFAULT,
    STATE_HIGHLIGHTED,
    STATE_SELECTED,
    STATE_HOVER,
    STATE_COLORS,
)

def _rgb_to_hex(rgb):
    """Convert RGB tuple (0-1) to hex string."""
    r, g, b = [int(c * 255) for c in rgb]
    return f'#{r:02x}{g:02x}{b:02x}'


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
            [0, 0],  # Bottom-Left
            [1, 0],  # Bottom-Right
            [1, 1],  # Top-Right
            [0, 1]   # Top-Left
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
                name=f"frustum_wire_{id(self)}"  # Unique name
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

    def update_appearance(self, plotter=None, opacity=0.8):
        """
        Update the frustum appearance based on selection state and color.
        
        Args:
            plotter: Specific plotter to update, or None to update all
            opacity: Opacity for the image plane (0.0 to 1.0)
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
                    prop.SetColor(144 / 255, 238 / 255, 144 / 255)  # lime green
                elif display_color == 'cyan':
                    prop.SetColor(0.0, 168 / 255, 230 / 255)  # cyan
                elif display_color == 'white':
                    prop.SetColor(0.8, 0.8, 0.8)  # light gray/white
                else:
                    prop.SetColor(1.0, 1.0, 1.0)  # default white
                
                prop.SetLineWidth(self.line_width)

            # Update image plane visibility based on selection/highlight
            if self.selected or self.highlighted:
                if p not in self.image_actors:
                    self.create_image_plane_actor(p, opacity=opacity)
            else:
                if p in self.image_actors:
                    p.remove_actor(self.image_actors[p])
                    del self.image_actors[p] 

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
            
            
class BatchedFrustumManager:
    """
    Manages batched rendering of camera frustums for efficient 3D visualization.
    
    Instead of creating individual actors per camera (O(N) draw calls), this class
    merges all frustum wireframes into a single PolyData mesh with scalar-based
    coloring, reducing draw calls to O(1).
    
    Selection and highlight states are managed via scalar arrays that map to
    a discrete color lookup table. Updating state only requires modifying the
    scalar array and calling mesh.Modified(), avoiding expensive actor property changes.
    
    Attributes:
        cameras: Dict mapping image_path -> Camera object
        camera_indices: Dict mapping image_path -> index in merged mesh
        merged_wireframe: Combined PyVista PolyData for all wireframes
        wireframe_actor: Single actor for the merged wireframe mesh
        point_counts: List of point counts per camera (for scalar array indexing)
    """
    
    def __init__(self):
        """Initialize the BatchedFrustumManager."""
        self.cameras: Dict[str, 'Camera'] = {}
        self.camera_indices: Dict[str, int] = {}
        self.camera_paths: List[str] = []  # Ordered list of camera paths
        
        # Merged geometry
        self.merged_wireframe: Optional[pv.PolyData] = None
        self.wireframe_actor = None
        
        # Track point ranges for each camera in the merged mesh
        self.point_ranges: List[Tuple[int, int]] = []  # (start_idx, end_idx) for each camera
        
        # Current scale for geometry generation
        self._current_scale = None
        
    def build_frustum_batch(self, 
                            cameras: Dict[str, 'Camera'], 
                            scale: float = 0.1) -> Optional[pv.PolyData]:
        """
        Build a merged PolyData mesh from all camera frustums.
        
        Args:
            cameras: Dict mapping image_path -> Camera object
            scale: Scale factor for frustum size
            
        Returns:
            pv.PolyData: Merged wireframe mesh with 'state' scalar array, or None if empty
        """
        if not cameras:
            return None
            
        self.cameras = cameras
        self.camera_indices.clear()
        self.camera_paths.clear()
        self.point_ranges.clear()
        self._current_scale = scale
        
        meshes = []
        point_offset = 0
        
        for idx, (path, camera) in enumerate(cameras.items()):
            self.camera_indices[path] = idx
            self.camera_paths.append(path)
            
            # Get wireframe mesh from frustum (geometry only, no actor creation)
            mesh = camera.frustum.get_mesh(scale)
            
            if mesh is not None:
                # Convert UnstructuredGrid to PolyData for merging
                # Extract the wireframe edges as lines
                wireframe_mesh = self._frustum_to_wireframe_polydata(camera, scale)
                
                if wireframe_mesh is not None:
                    n_points = wireframe_mesh.n_points
                    self.point_ranges.append((point_offset, point_offset + n_points))
                    point_offset += n_points
                    meshes.append(wireframe_mesh)
                else:
                    self.point_ranges.append((point_offset, point_offset))
            else:
                self.point_ranges.append((point_offset, point_offset))
        
        if not meshes:
            self.merged_wireframe = None
            return None
            
        # Merge all wireframe meshes into one
        if len(meshes) == 1:
            self.merged_wireframe = meshes[0]
        else:
            self.merged_wireframe = pv.merge(meshes)
        
        # Initialize state array (all default/white)
        n_points = self.merged_wireframe.n_points
        self.merged_wireframe['state'] = np.zeros(n_points, dtype=np.uint8)
        
        return self.merged_wireframe
    
    def _frustum_to_wireframe_polydata(self, camera: 'Camera', scale: float) -> Optional[pv.PolyData]:
        """
        Convert a camera's frustum to wireframe PolyData (lines only).
        
        Creates line segments for the frustum edges:
        - 4 lines for the near plane quad
        - 4 lines from center to corners
        
        Args:
            camera: Camera object with frustum geometry
            scale: Scale factor for frustum size
            
        Returns:
            pv.PolyData: Wireframe as line segments
        """
        # Get image dimensions
        w, h = camera.width, camera.height
        
        # Define pixel corners
        corners_pix = np.array([
            [0, 0, 1],   # Top-Left (0)
            [w, 0, 1],   # Top-Right (1)
            [w, h, 1],   # Bottom-Right (2)
            [0, h, 1]    # Bottom-Left (3)
        ])
        
        # Unproject to camera space
        frustum_points_cam = scale * (camera.K_inv @ corners_pix.T).T
        
        # Add camera center
        all_points_cam = np.vstack([frustum_points_cam, [0, 0, 0]])  # 5 points
        
        # Transform to world coordinates
        R_inv = camera.R.T
        t_vec = camera.t.reshape(3, 1)
        all_points_world = (R_inv @ (all_points_cam.T - t_vec)).T
        
        # Define line segments (VTK format: [n_pts, idx0, idx1, ...])
        # Near plane quad: 0-1, 1-2, 2-3, 3-0
        # Edges to center: 4-0, 4-1, 4-2, 4-3
        lines = np.array([
            2, 0, 1,  # Top edge
            2, 1, 2,  # Right edge
            2, 2, 3,  # Bottom edge
            2, 3, 0,  # Left edge
            2, 4, 0,  # Center to TL
            2, 4, 1,  # Center to TR
            2, 4, 2,  # Center to BR
            2, 4, 3,  # Center to BL
        ])
        
        return pv.PolyData(all_points_world, lines=lines)
    
    def add_to_plotter(self, 
                       plotter, 
                       line_width: float = 1.5) -> Optional['vtkActor']:
        """
        Add the merged wireframe mesh to a PyVista plotter.
        
        Uses a custom color lookup table to map state values to colors.
        
        Args:
            plotter: PyVista plotter instance
            line_width: Width of wireframe lines
            
        Returns:
            vtkActor: The wireframe actor, or None if no geometry
        """
        if self.merged_wireframe is None:
            return None
        
        # Remove existing actor if present
        if self.wireframe_actor is not None:
            try:
                plotter.remove_actor(self.wireframe_actor)
            except:
                pass
            
        # Create custom colormap for states
        # Map: 0=default(gray), 1=highlighted(cyan), 2=selected(lime), 3=hovered(red)
        cmap = [_rgb_to_hex(STATE_COLORS[STATE_DEFAULT]), 
                _rgb_to_hex(STATE_COLORS[STATE_HIGHLIGHTED]), 
                _rgb_to_hex(STATE_COLORS[STATE_SELECTED]),
                _rgb_to_hex(STATE_COLORS[STATE_HOVER])]
        
        self.wireframe_actor = plotter.add_mesh(
            self.merged_wireframe,
            scalars='state',
            cmap=cmap,
            clim=[0, 3],
            show_scalar_bar=False,
            line_width=line_width,
            render_lines_as_tubes=False,
            style='wireframe',
            name='_batched_frustums'
        )
        
        return self.wireframe_actor
    
    def update_camera_state(self, path: str, state: int):
        """
        Update the visual state of a single camera's frustum.
        
        Args:
            path: Image path of the camera
            state: STATE_DEFAULT (0), STATE_HIGHLIGHTED (1), or STATE_SELECTED (2)
        """
        if self.merged_wireframe is None:
            return
            
        if path not in self.camera_indices:
            return
            
        idx = self.camera_indices[path]
        if idx >= len(self.point_ranges):
            return
            
        start, end = self.point_ranges[idx]
        if start < end:
            self.merged_wireframe['state'][start:end] = state
    
    def update_camera_states(self, 
                             selected_path: Optional[str] = None,
                             highlighted_paths: Optional[List[str]] = None,
                             hovered_path: Optional[str] = None):
        """
        Batch update visual states for all cameras.
        
        More efficient than calling update_camera_state() repeatedly.
        
        Args:
            selected_path: Path of the selected camera (lime green)
            highlighted_paths: List of highlighted camera paths (cyan)
            hovered_path: Path of the hovered camera (red, with Ctrl)
        """
        if self.merged_wireframe is None:
            return
            
        highlighted_paths = highlighted_paths or []
        
        # Reset all to default
        self.merged_wireframe['state'][:] = STATE_DEFAULT
        
        # Set highlighted cameras
        for path in highlighted_paths:
            if path in self.camera_indices and path != selected_path and path != hovered_path:
                idx = self.camera_indices[path]
                if idx < len(self.point_ranges):
                    start, end = self.point_ranges[idx]
                    if start < end:
                        self.merged_wireframe['state'][start:end] = STATE_HIGHLIGHTED
        
        # Set selected camera (overrides highlight)
        if selected_path and selected_path in self.camera_indices and selected_path != hovered_path:
            idx = self.camera_indices[selected_path]
            if idx < len(self.point_ranges):
                start, end = self.point_ranges[idx]
                if start < end:
                    self.merged_wireframe['state'][start:end] = STATE_SELECTED
        
        # Set hovered camera (overrides all)
        if hovered_path and hovered_path in self.camera_indices:
            idx = self.camera_indices[hovered_path]
            if idx < len(self.point_ranges):
                start, end = self.point_ranges[idx]
                if start < end:
                    self.merged_wireframe['state'][start:end] = STATE_HOVER
    
    def mark_modified(self):
        """
        Mark the merged mesh as modified to trigger re-render.
        
        Call this after updating camera states, followed by plotter.render().
        """
        if self.merged_wireframe is not None:
            self.merged_wireframe.Modified()
    
    def set_visibility(self, visible: bool):
        """Set visibility of the batched wireframe actor."""
        if self.wireframe_actor is not None:
            self.wireframe_actor.SetVisibility(visible)
    
    def remove_from_plotter(self, plotter):
        """Remove the wireframe actor from the plotter."""
        if self.wireframe_actor is not None:
            try:
                plotter.remove_actor(self.wireframe_actor)
            except:
                pass
            self.wireframe_actor = None
    
    def clear(self):
        """Clear all cached data."""
        self.cameras.clear()
        self.camera_indices.clear()
        self.camera_paths.clear()
        self.point_ranges.clear()
        self.merged_wireframe = None
        self.wireframe_actor = None
        self._current_scale = None