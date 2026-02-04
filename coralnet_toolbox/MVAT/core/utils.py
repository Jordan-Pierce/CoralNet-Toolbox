import os
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap

from coralnet_toolbox.MVAT.core.constants import (
    SELECT_COLOR_RGB,
    HIGHLIGHT_COLOR_RGB,
    RAY_COLOR_SELECTED,
    RAY_COLOR_HIGHLIGHTED
)


# ----------------------------------------------------------------------------------------------------------------------
# Batched Geometry Manager for Efficient Rendering
# ----------------------------------------------------------------------------------------------------------------------


# State constants for scalar-based coloring
STATE_DEFAULT = 0
STATE_HIGHLIGHTED = 1
STATE_SELECTED = 2

# Colors for each state (RGB normalized 0-1)
STATE_COLORS = {
    STATE_DEFAULT: (0.8, 0.8, 0.8),      # Light gray/white
    STATE_HIGHLIGHTED: tuple(c / 255 for c in HIGHLIGHT_COLOR_RGB),  # Cyan
    STATE_SELECTED: tuple(c / 255 for c in SELECT_COLOR_RGB),        # Lime green
}


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
            
        # Create custom colormap for states using matplotlib ListedColormap
        # Map: 0=default(gray), 1=highlighted(cyan), 2=selected(lime)
        state_colors = [
            STATE_COLORS[STATE_DEFAULT], 
            STATE_COLORS[STATE_HIGHLIGHTED], 
            STATE_COLORS[STATE_SELECTED]
        ]
        cmap = ListedColormap(state_colors, name='frustum_states')
        
        self.wireframe_actor = plotter.add_mesh(
            self.merged_wireframe,
            scalars='state',
            cmap=cmap,
            clim=[0, 2],
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
                             highlighted_paths: Optional[List[str]] = None):
        """
        Batch update visual states for all cameras.
        
        More efficient than calling update_camera_state() repeatedly.
        
        Args:
            selected_path: Path of the selected camera (lime green)
            highlighted_paths: List of highlighted camera paths (cyan)
        """
        if self.merged_wireframe is None:
            return
            
        highlighted_paths = highlighted_paths or []
        
        # Reset all to default
        self.merged_wireframe['state'][:] = STATE_DEFAULT
        
        # Set highlighted cameras
        for path in highlighted_paths:
            if path in self.camera_indices and path != selected_path:
                idx = self.camera_indices[path]
                if idx < len(self.point_ranges):
                    start, end = self.point_ranges[idx]
                    if start < end:
                        self.merged_wireframe['state'][start:end] = STATE_HIGHLIGHTED
        
        # Set selected camera (overrides highlight)
        if selected_path and selected_path in self.camera_indices:
            idx = self.camera_indices[selected_path]
            if idx < len(self.point_ranges):
                start, end = self.point_ranges[idx]
                if start < end:
                    self.merged_wireframe['state'][start:end] = STATE_SELECTED
    
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


class BatchedRayManager:
    """
    Manages batched rendering of camera rays for efficient visualization.
    
    Instead of creating individual line and sphere actors per ray (2*N draw calls),
    this class maintains a single PolyData mesh containing all ray lines and updates
    point coordinates in-place when the mouse moves.
    
    Attributes:
        ray_mesh: PolyData containing all ray line segments
        sphere_mesh: PolyData containing terminal point spheres
        ray_actor: Single actor for all ray lines
        sphere_actor: Single actor for terminal spheres
    """
    
    def __init__(self):
        """Initialize the BatchedRayManager."""
        self.ray_mesh: Optional[pv.PolyData] = None
        self.sphere_mesh: Optional[pv.PolyData] = None
        self.ray_actor = None
        self.sphere_actor = None
        
        # Ray colors stored per ray
        self._ray_colors: Optional[np.ndarray] = None
        self._num_rays = 0
        
    def build_ray_batch(self, 
                        rays_with_colors: List[Tuple['CameraRay', tuple]]) -> Tuple[Optional[pv.PolyData], Optional[pv.PolyData]]:
        """
        Build merged meshes for multiple rays.
        
        Args:
            rays_with_colors: List of (CameraRay, color_rgb) tuples
                             Colors should be RGB tuples (0-255)
            
        Returns:
            Tuple of (ray_lines_mesh, sphere_mesh)
        """
        if not rays_with_colors:
            self.ray_mesh = None
            self.sphere_mesh = None
            self._num_rays = 0
            return None, None
        
        self._num_rays = len(rays_with_colors)
        
        # Build line segments
        points = []
        lines = []
        colors = []
        
        for i, (ray, color) in enumerate(rays_with_colors):
            if ray is None:
                continue
                
            # Add origin and terminal points
            pt_idx = len(points)
            points.append(ray.origin.tolist())
            points.append(ray.terminal_point.tolist())
            
            # Add line connectivity
            lines.extend([2, pt_idx, pt_idx + 1])
            
            # Add colors for both endpoints (same color per line)
            # Normalize to 0-1 if needed
            if isinstance(color, tuple) and any(c > 1 for c in color[:3]):
                norm_color = tuple(c / 255 for c in color[:3])
            else:
                norm_color = color[:3] if len(color) >= 3 else color
            colors.append(norm_color)
            colors.append(norm_color)
        
        if not points:
            self.ray_mesh = None
            self.sphere_mesh = None
            return None, None
        
        # Create lines mesh
        self.ray_mesh = pv.PolyData(np.array(points), lines=np.array(lines))
        self._ray_colors = np.array(colors)
        self.ray_mesh['RGB'] = (self._ray_colors * 255).astype(np.uint8)
        
        # Build spheres at terminal points
        first_ray = rays_with_colors[0][0]
        if first_ray is not None:
            base_radius = np.linalg.norm(first_ray.terminal_point - first_ray.origin) * 0.005
            base_radius = max(base_radius, 0.01)
        else:
            base_radius = 0.05
        
        spheres = []
        for i, (ray, color) in enumerate(rays_with_colors):
            if ray is None:
                continue
            # Primary ray gets larger sphere
            radius = base_radius if i == 0 else base_radius * 0.6
            sphere = pv.Sphere(radius=radius, center=ray.terminal_point.tolist())
            
            # Add color to sphere
            if isinstance(color, tuple) and any(c > 1 for c in color[:3]):
                norm_color = tuple(c / 255 for c in color[:3])
            else:
                norm_color = color[:3] if len(color) >= 3 else color
            sphere['RGB'] = np.tile(np.array(norm_color) * 255, (sphere.n_points, 1)).astype(np.uint8)
            spheres.append(sphere)
        
        if spheres:
            self.sphere_mesh = pv.merge(spheres) if len(spheres) > 1 else spheres[0]
        else:
            self.sphere_mesh = None
        
        return self.ray_mesh, self.sphere_mesh
    
    def add_to_plotter(self, plotter, line_width: float = 3) -> Tuple[Optional['vtkActor'], Optional['vtkActor']]:
        """
        Add the batched ray meshes to a plotter.
        
        Args:
            plotter: PyVista plotter instance
            line_width: Width of ray lines
            
        Returns:
            Tuple of (ray_actor, sphere_actor)
        """
        # Remove existing actors
        self.remove_from_plotter(plotter)
        
        if self.ray_mesh is not None:
            self.ray_actor = plotter.add_mesh(
                self.ray_mesh,
                scalars='RGB',
                rgb=True,
                line_width=line_width,
                render_lines_as_tubes=False,
                name='_batched_rays',
                pickable=False
            )
        
        if self.sphere_mesh is not None:
            self.sphere_actor = plotter.add_mesh(
                self.sphere_mesh,
                scalars='RGB',
                rgb=True,
                name='_batched_ray_spheres',
                pickable=False
            )
        
        return self.ray_actor, self.sphere_actor
    
    def update_ray_endpoints(self, 
                             rays_with_colors: List[Tuple['CameraRay', tuple]]):
        """
        Update ray endpoints in-place (more efficient than rebuilding).
        
        Only works if the number of rays hasn't changed.
        
        Args:
            rays_with_colors: List of (CameraRay, color_rgb) tuples
        """
        if self.ray_mesh is None or len(rays_with_colors) != self._num_rays:
            # Need to rebuild - ray count changed
            self.build_ray_batch(rays_with_colors)
            return
        
        # Update points in-place
        points = self.ray_mesh.points
        for i, (ray, _) in enumerate(rays_with_colors):
            if ray is not None:
                pt_idx = i * 2
                points[pt_idx] = ray.origin
                points[pt_idx + 1] = ray.terminal_point
        
        self.ray_mesh.Modified()
    
    def set_visibility(self, visible: bool):
        """Set visibility of ray actors."""
        if self.ray_actor is not None:
            self.ray_actor.SetVisibility(visible)
        if self.sphere_actor is not None:
            self.sphere_actor.SetVisibility(visible)
    
    def remove_from_plotter(self, plotter):
        """Remove ray actors from plotter."""
        if self.ray_actor is not None:
            try:
                plotter.remove_actor(self.ray_actor)
            except:
                pass
            self.ray_actor = None
            
        if self.sphere_actor is not None:
            try:
                plotter.remove_actor(self.sphere_actor)
            except:
                pass
            self.sphere_actor = None
    
    def clear(self):
        """Clear all cached data."""
        self.ray_mesh = None
        self.sphere_mesh = None
        self.ray_actor = None
        self.sphere_actor = None
        self._ray_colors = None
        self._num_rays = 0


# ----------------------------------------------------------------------------------------------------------------------
# Helper Functions for COLMAP parsing
# ----------------------------------------------------------------------------------------------------------------------


def parse_colmap_cameras(cameras_file):
    """
    Parse COLMAP cameras file (.txt or .bin format).

    Returns:
        dict: camera_id -> camera_params dict
    """
    if cameras_file.endswith('.bin'):
        return parse_colmap_cameras_bin(cameras_file)
    else:
        return parse_colmap_cameras_txt(cameras_file)


def parse_colmap_images(images_file):
    """
    Parse COLMAP images file (.txt or .bin format).

    Returns:
        dict: image_name -> image_data dict
    """
    if images_file.endswith('.bin'):
        return parse_colmap_images_bin(images_file)
    else:
        return parse_colmap_images_txt(images_file)
    

def parse_colmap_points3D(points3D_file):
    """
    Parse COLMAP points3D file (.txt or .bin format).

    Returns:
        tuple: (points, colors) where points is Nx3 array and colors is Nx3 uint8 array
    """
    if points3D_file.endswith('.bin'):
        return parse_colmap_points3D_bin(points3D_file)
    else:
        return parse_colmap_points3D_txt(points3D_file)


def parse_colmap_cameras_txt(cameras_txt_path):
    """
    Parse COLMAP cameras.txt file.

    Returns:
        dict: camera_id -> camera_params dict
    """
    cameras = {}
    with open(cameras_txt_path, 'r') as f:
        lines = f.readlines()

    # Skip comments and header
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or line == '':
            i += 1
            continue

        if line.startswith('Number of cameras:'):
            i += 1
            continue

        # Parse camera line: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        parts = line.split()
        if len(parts) < 4:
            i += 1
            continue

        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])

        # Parse camera parameters based on model
        if model == 'PINHOLE':
            # PARAMS: fx, fy, cx, cy
            fx, fy, cx, cy = map(float, parts[4:8])
            params = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
        elif model == 'SIMPLE_PINHOLE':
            # PARAMS: f, cx, cy
            f, cx, cy = map(float, parts[4:7])
            params = {'fx': f, 'fy': f, 'cx': cx, 'cy': cy}
        elif model == 'SIMPLE_RADIAL':
            # PARAMS: f, cx, cy, k
            f, cx, cy, k = map(float, parts[4:8])
            params = {'fx': f, 'fy': f, 'cx': cx, 'cy': cy, 'k1': k}
        elif model == 'RADIAL':
            # PARAMS: f, cx, cy, k1, k2
            f, cx, cy, k1, k2 = map(float, parts[4:9])
            params = {'fx': f, 'fy': f, 'cx': cx, 'cy': cy, 'k1': k1, 'k2': k2}
        else:
            # Default to pinhole with identity parameters
            params = {'fx': width * 0.8, 'fy': height * 0.8, 'cx': width / 2, 'cy': height / 2}

        cameras[camera_id] = {
            'model': model,
            'width': width,
            'height': height,
            'params': params
        }
        i += 1

    return cameras


def parse_colmap_images_txt(images_txt_path):
    """
    Parse COLMAP images.txt file.

    Returns:
        dict: image_name -> image_data dict
    """
    images = {}
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()

    # Skip comments and header
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or line == '':
            i += 1
            continue

        if line.startswith('Number of images:'):
            i += 1
            continue

        # Parse image line: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue

        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = ' '.join(parts[9:])

        # Skip the points2d line for now
        i += 2

        images[name] = {
            'image_id': image_id,
            'camera_id': camera_id,
            'quaternion': np.array([qw, qx, qy, qz]),
            'translation': np.array([tx, ty, tz])
        }

    return images


def parse_colmap_cameras_bin(cameras_bin_path):
    """
    Parse COLMAP cameras.bin file.

    Returns:
        dict: camera_id -> camera_params dict
    """
    cameras = {}
    
    try:
        with open(cameras_bin_path, 'rb') as f:
            # Read number of cameras
            num_cameras_data = f.read(8)
            if len(num_cameras_data) < 8:
                print(f"Warning: Could not read number of cameras from {cameras_bin_path}")
                return cameras
            num_cameras = struct.unpack('<Q', num_cameras_data)[0]
            print(f"Reading {num_cameras} cameras from binary file")
            
            for i in range(num_cameras):
                try:
                    # Read camera ID
                    camera_id_data = f.read(4)
                    if len(camera_id_data) < 4:
                        print(f"Warning: Could not read camera ID for camera {i}")
                        break
                    camera_id = struct.unpack('<I', camera_id_data)[0]
                    
                    # Read model (enum)
                    model_data = f.read(4)
                    if len(model_data) < 4:
                        print(f"Warning: Could not read model for camera {camera_id}")
                        break
                    model_id = struct.unpack('<I', model_data)[0]
                    
                    # Map model ID to string
                    model_map = {
                        0: 'SIMPLE_PINHOLE',
                        1: 'PINHOLE',
                        2: 'SIMPLE_RADIAL',
                        3: 'RADIAL',
                        4: 'OPENCV',
                        5: 'OPENCV_FISHEYE',
                        6: 'FULL_OPENCV',
                        7: 'FOV',
                        8: 'SIMPLE_RADIAL_FISHEYE',
                        9: 'RADIAL_FISHEYE',
                        10: 'THIN_PRISM_FISHEYE'
                    }
                    model = model_map.get(model_id, 'PINHOLE')
                    
                    # Read width and height
                    width_data = f.read(4)
                    height_data = f.read(4)
                    if len(width_data) < 4 or len(height_data) < 4:
                        print(f"Warning: Could not read dimensions for camera {camera_id}")
                        break
                    width = struct.unpack('<I', width_data)[0]
                    height = struct.unpack('<I', height_data)[0]
                    
                    # Read number of parameters
                    num_params_data = f.read(4)
                    if len(num_params_data) < 4:
                        print(f"Warning: Could not read num_params for camera {camera_id}")
                        break
                    num_params = struct.unpack('<I', num_params_data)[0]
                    
                    # Read parameters
                    params_data = []
                    for j in range(num_params):
                        param_data = f.read(8)
                        if len(param_data) < 8:
                            print(f"Warning: Could not read parameter {j} for camera {camera_id}")
                            break
                        params_data.append(struct.unpack('<d', param_data)[0])
                    
                    if len(params_data) < num_params:
                        print(f"Warning: Incomplete parameters for camera {camera_id}")
                        continue
                    
                    # Parse parameters based on model
                    if model == 'PINHOLE' and len(params_data) >= 4:
                        fx, fy, cx, cy = params_data[:4]
                        params = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
                    elif model == 'SIMPLE_PINHOLE' and len(params_data) >= 3:
                        f, cx, cy = params_data[:3]
                        params = {'fx': f, 'fy': f, 'cx': cx, 'cy': cy}
                    elif model == 'SIMPLE_RADIAL' and len(params_data) >= 4:
                        f, cx, cy, k = params_data[:4]
                        params = {'fx': f, 'fy': f, 'cx': cx, 'cy': cy, 'k1': k}
                    elif model == 'RADIAL' and len(params_data) >= 5:
                        f, cx, cy, k1, k2 = params_data[:5]
                        params = {'fx': f, 'fy': f, 'cx': cx, 'cy': cy, 'k1': k1, 'k2': k2}
                    else:
                        # Default to pinhole with reasonable parameters
                        print(f"Warning: Unknown or incomplete camera model {model} "
                              f"for camera {camera_id}, using defaults")
                        params = {'fx': width * 0.8, 'fy': height * 0.8, 'cx': width / 2, 'cy': height / 2}
                    
                    cameras[camera_id] = {
                        'model': model,
                        'width': width,
                        'height': height,
                        'params': params
                    }
                    
                except Exception as e:
                    print(f"Error parsing camera {i}: {e}")
                    break
                    
    except Exception as e:
        print(f"Error opening or reading binary cameras file: {e}")
        # Fall back to trying text format if binary fails
        try:
            txt_path = cameras_bin_path.replace('.bin', '.txt')
            if os.path.exists(txt_path):
                print(f"Falling back to text format: {txt_path}")
                return parse_colmap_cameras_txt(txt_path)
        except Exception as fallback_e:
            print(f"Fallback to text format also failed: {fallback_e}")
    
    return cameras


def parse_colmap_images_bin(images_bin_path):
    """
    Parse COLMAP images.bin file.

    Returns:
        dict: image_name -> image_data dict
    """
    images = {}
    
    try:
        with open(images_bin_path, 'rb') as f:
            # Read number of images
            num_images_data = f.read(8)
            if len(num_images_data) < 8:
                print(f"Warning: Could not read number of images from {images_bin_path}")
                return images
            num_images = struct.unpack('<Q', num_images_data)[0]
            print(f"Reading {num_images} images from binary file")
            
            for i in range(num_images):
                try:
                    # Read image ID
                    image_id_data = f.read(4)
                    if len(image_id_data) < 4:
                        print(f"Warning: Could not read image ID for image {i}")
                        break
                    image_id = struct.unpack('<I', image_id_data)[0]
                    
                    # Read quaternion (qw, qx, qy, qz)
                    quat_data = f.read(32)  # 4 doubles = 32 bytes
                    if len(quat_data) < 32:
                        print(f"Warning: Could not read quaternion for image {image_id}")
                        break
                    qw, qx, qy, qz = struct.unpack('<dddd', quat_data)
                    
                    # Read translation (tx, ty, tz)
                    trans_data = f.read(24)  # 3 doubles = 24 bytes
                    if len(trans_data) < 24:
                        print(f"Warning: Could not read translation for image {image_id}")
                        break
                    tx, ty, tz = struct.unpack('<ddd', trans_data)
                    
                    # Read camera ID
                    camera_id_data = f.read(4)
                    if len(camera_id_data) < 4:
                        print(f"Warning: Could not read camera ID for image {image_id}")
                        break
                    camera_id = struct.unpack('<I', camera_id_data)[0]
                    
                    # Read image name (null-terminated string)
                    name_bytes = b''
                    while True:
                        byte = f.read(1)
                        if not byte:
                            print(f"Warning: Unexpected end of file reading name for image {image_id}")
                            break
                        if byte == b'\x00':
                            break
                        name_bytes += byte
                    
                    if not name_bytes and not byte == b'\x00':
                        print(f"Warning: Could not read name for image {image_id}")
                        break
                        
                    try:
                        name = name_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        print(f"Warning: Could not decode name for image {image_id}")
                        name = f"image_{image_id}"
                    
                    # Read number of 2D points
                    num_points_data = f.read(4)
                    if len(num_points_data) < 4:
                        print(f"Warning: Could not read num_points for image {image_id}")
                        break
                    num_points = struct.unpack('<I', num_points_data)[0]
                    
                    # Skip 2D points data (each point: x, y, point3d_id)
                    points_data_size = num_points * 24  # 3 doubles per point
                    skipped_data = f.read(points_data_size)
                    if len(skipped_data) < points_data_size:
                        print(f"Warning: Could not skip 2D points data for image {image_id}")
                        break
                    
                    images[name] = {
                        'image_id': image_id,
                        'camera_id': camera_id,
                        'quaternion': np.array([qw, qx, qy, qz]),
                        'translation': np.array([tx, ty, tz])
                    }
                    
                except Exception as e:
                    print(f"Error parsing image {i}: {e}")
                    break
                    
    except Exception as e:
        print(f"Error opening or reading binary images file: {e}")
        # Fall back to trying text format if binary fails
        try:
            txt_path = images_bin_path.replace('.bin', '.txt')
            if os.path.exists(txt_path):
                print(f"Falling back to text format: {txt_path}")
                return parse_colmap_images_txt(txt_path)
        except Exception as fallback_e:
            print(f"Fallback to text format also failed: {fallback_e}")
    
    return images


def parse_colmap_points3D_txt(points3D_txt_path):
    """
    Parse COLMAP points3D.txt file.

    Returns:
        tuple: (points, colors) where points is Nx3 array and colors is Nx3 uint8 array
    """
    points = []
    colors = []
    
    try:
        with open(points3D_txt_path, 'r') as f:
            lines = f.readlines()
        
        # Skip comments and header
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#') or line == '':
                i += 1
                continue
            
            if line.startswith('Number of points:'):
                i += 1
                continue
            
            # Parse point line: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
            parts = line.split()
            if len(parts) < 8:
                i += 1
                continue
            
            # Extract X, Y, Z, R, G, B
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            
            points.append([x, y, z])
            colors.append([r, g, b])
            
            i += 1
    
    except Exception as e:
        print(f"Error parsing points3D.txt: {e}")
        return np.array([]), np.array([])
    
    return np.array(points), np.array(colors, dtype=np.uint8)


def parse_colmap_points3D_bin(points3D_bin_path):
    """
    Parse COLMAP points3D.bin file.

    Returns:
        tuple: (points, colors) where points is Nx3 array and colors is Nx3 uint8 array
    """
    points = []
    colors = []
    
    try:
        with open(points3D_bin_path, 'rb') as f:
            # Read number of points
            num_points_data = f.read(8)
            if len(num_points_data) < 8:
                print(f"Warning: Could not read number of points from {points3D_bin_path}")
                return np.array([]), np.array([])
            num_points = struct.unpack('<Q', num_points_data)[0]
            print(f"Reading {num_points} points from binary file")
            
            for i in range(num_points):
                try:
                    # Read point ID (u64)
                    point_id_data = f.read(8)
                    if len(point_id_data) < 8:
                        print(f"Warning: Could not read point ID for point {i}")
                        break
                    # point_id = struct.unpack('<Q', point_id_data)[0]  # We don't need the ID
                    
                    # Read X, Y, Z (3 doubles)
                    xyz_data = f.read(24)
                    if len(xyz_data) < 24:
                        print(f"Warning: Could not read XYZ for point {i}")
                        break
                    x, y, z = struct.unpack('<ddd', xyz_data)
                    
                    # Read R, G, B (3 uint8)
                    rgb_data = f.read(3)
                    if len(rgb_data) < 3:
                        print(f"Warning: Could not read RGB for point {i}")
                        break
                    r, g, b = struct.unpack('<BBB', rgb_data)
                    
                    # Read error (double)
                    error_data = f.read(8)
                    if len(error_data) < 8:
                        print(f"Warning: Could not read error for point {i}")
                        break
                    # error = struct.unpack('<d', error_data)[0]  # We don't need the error
                    
                    # Read number of track elements
                    num_track_data = f.read(8)
                    if len(num_track_data) < 8:
                        print(f"Warning: Could not read num_track for point {i}")
                        break
                    num_track = struct.unpack('<Q', num_track_data)[0]
                    
                    # Skip track data (each track: image_id u32, point2d_idx u32)
                    track_data_size = num_track * 8  # 2 u32 per track
                    skipped_data = f.read(track_data_size)
                    if len(skipped_data) < track_data_size:
                        print(f"Warning: Could not skip track data for point {i}")
                        break
                    
                    points.append([x, y, z])
                    colors.append([r, g, b])
                    
                except Exception as e:
                    print(f"Error parsing point {i}: {e}")
                    break
                    
    except Exception as e:
        print(f"Error opening or reading binary points3D file: {e}")
        # Fall back to trying text format if binary fails
        try:
            txt_path = points3D_bin_path.replace('.bin', '.txt')
            if os.path.exists(txt_path):
                print(f"Falling back to text format: {txt_path}")
                return parse_colmap_points3D_txt(txt_path)
        except Exception as fallback_e:
            print(f"Fallback to text format also failed: {fallback_e}")
    
    return np.array(points), np.array(colors, dtype=np.uint8)


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2]
    ])