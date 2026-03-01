"""
Camera Ray Module for MVAT

Provides ray casting functionality for projecting 2D pixel coordinates through
camera frustums into 3D space and back onto other camera views.
Uses PyVista for 3D ray-mesh intersection when available.
"""
from typing import Optional, Dict, Tuple, List

import numpy as np

import pyvista as pv

from coralnet_toolbox.MVAT.core.Camera import Camera


from coralnet_toolbox.MVAT.core.constants import (
    SELECT_COLOR_RGB,
    HIGHLIGHT_COLOR_RGB,
    STATE_DEFAULT,
    STATE_HIGHLIGHTED,
    STATE_SELECTED,
    STATE_COLORS,
)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class CameraRay:
    """
    Represents a ray cast from a camera through a pixel coordinate into 3D space.
    
    The ray consists of:
    - An origin point (camera position in world coordinates)
    - A direction vector (normalized ray direction from camera through pixel)
    - A terminal point (3D point where the ray "ends" - either from depth or estimated)
    - A flag indicating if the depth is accurate (from z-channel) or estimated
    
    Attributes:
        origin (np.ndarray): 3D camera position in world coordinates.
        direction (np.ndarray): Normalized direction vector of the ray.
        terminal_point (np.ndarray): 3D endpoint of the ray in world coordinates.
        has_accurate_depth (bool): True if terminal point calculated from actual depth data.
        pixel_coord (tuple): Original 2D pixel coordinate (u, v).
        source_camera (Camera): Reference to the camera that originated this ray.
    """
    
    def __init__(self, 
                 origin: np.ndarray, 
                 direction: np.ndarray, 
                 terminal_point: np.ndarray,
                 has_accurate_depth: bool = False,
                 pixel_coord: Optional[Tuple[int, int]] = None,
                 source_camera: Optional['Camera'] = None):
        """
        Initialize a CameraRay.
        
        Args:
            origin: 3D camera position in world coordinates.
            direction: Normalized direction vector of the ray.
            terminal_point: 3D endpoint of the ray in world coordinates.
            has_accurate_depth: Whether terminal_point is from actual depth data.
            pixel_coord: Original 2D pixel coordinate (u, v).
            source_camera: Reference to the originating camera.
        """
        self.origin = np.asarray(origin, dtype=np.float64)
        self.direction = np.asarray(direction, dtype=np.float64)
        # Ensure direction is normalized
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction = self.direction / norm
        self.terminal_point = np.asarray(terminal_point, dtype=np.float64)
        self.has_accurate_depth = has_accurate_depth
        self.pixel_coord = pixel_coord
        self.source_camera = source_camera
        # Visualized start/end points (may be offset slightly to avoid
        # coincident geometry with the viewer camera and near-plane clipping).
        # These are computed by factory methods; default to the raw values.
        self.visual_origin = self.origin.copy()
        self.visual_terminal = self.terminal_point.copy()
        
    @classmethod
    def from_pixel_and_camera(cls, pixel_xy: Tuple[int, int], camera: 'Camera', 
                              depth: Optional[float] = None, default_depth: float = 10.0) -> 'CameraRay':
        
        # --- ORTHOGRAPHIC RAYS ---
        if getattr(camera, 'is_orthographic', False):
            # ALWAYS use unproject to fetch actual DEM elevation natively,
            # completely bypassing the 'depth' arg from the UI.
            terminal_point = camera.unproject(pixel_xy)
            has_accurate_depth = False
            
            if terminal_point is not None:
                has_accurate_depth = True
            else:
                # Extreme fallback if DEM yields NaN (using flattened arrays to prevent shape mismatch)
                pixel_hom = np.array([pixel_xy, pixel_xy, 1.0])
                world_xy = np.asarray(camera.transform_matrix @ pixel_hom).flatten()
                terminal_point = np.array([float(world_xy), float(world_xy), 0.0])
            
            direction = np.array([0.0, 0.0, -1.0])
            origin = terminal_point + np.array([0.0, 0.0, 1000.0]) 
            
            ray = cls(
                origin=origin,
                direction=direction,
                terminal_point=terminal_point,
                has_accurate_depth=has_accurate_depth,
                pixel_coord=pixel_xy,
                source_camera=camera
            )
            ray.visual_origin = origin.copy()
            ray.visual_terminal = terminal_point.copy()
            return ray
        
        # EXISTING: Perspective camera logic
        # Camera origin is the optical center in world coordinates
        origin = camera.position.copy()
        
        # Check if we have valid depth
        has_accurate_depth = False
        actual_depth = default_depth
        
        if depth is not None and depth > 0 and not np.isnan(depth):
            actual_depth = depth
            has_accurate_depth = True
        
        # Calculate the 3D point by unprojecting the pixel with known depth
        # We need to manually unproject since camera.unproject() reads from raster
        pixel_hom = np.array([pixel_xy[0], pixel_xy[1], 1.0])
        
        # Transform to Camera Coordinate System (Back-projection)
        # X_cam = Z * K^{-1} * x_pix
        point_cam = actual_depth * (camera.K_inv @ pixel_hom)
        
        # Transform to World Coordinate System
        # X_world = R^T * (X_cam - t)
        terminal_point = camera.R.T @ (point_cam - camera.t)
        
        # Calculate direction (from camera position to terminal point)
        direction = terminal_point - origin
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        else:
            # Fallback to camera's forward direction
            direction = camera.R.T @ np.array([0, 0, 1])
            
        ray = cls(
            origin=origin,
            direction=direction,
            terminal_point=terminal_point,
            has_accurate_depth=has_accurate_depth,
            pixel_coord=pixel_xy,
            source_camera=camera
        )

        # Visual start/end default to the true geometry (no offset).
        ray.visual_origin = ray.origin.copy()
        ray.visual_terminal = ray.terminal_point.copy()

        return ray
    
    @classmethod
    def from_world_point_and_camera(cls, 
                                    world_point: np.ndarray, 
                                    camera: 'Camera') -> 'CameraRay':
        """
        Create a ray from a camera's origin to a known 3D world point.
        
        This is used to visualize rays from highlighted cameras to a point
        determined by another camera's ray (e.g., the selected camera).
        
        Args:
            world_point: 3D point in world coordinates (the target).
            camera: Camera object from which to cast the ray.
            
        Returns:
            CameraRay: A new ray from the camera to the world point.
            
        # TODO: When depth is fully incorporated, re-evaluate whether rays
        # from highlighted cameras should use solid or dashed line styling
        # based on depth accuracy at the projected point.
        """
        # BRANCH: Orthographic camera
        if getattr(camera, 'is_orthographic', False):
            world_point = np.asarray(world_point, dtype=np.float64)
            direction = np.array([0.0, 0.0, -1.0])
            origin = world_point + np.array([0.0, 0.0, 1000.0])
            
            ray = cls(
                origin=origin,
                direction=direction,
                terminal_point=world_point,
                has_accurate_depth=True,
                pixel_coord=None,
                source_camera=camera
            )
            ray.visual_origin = origin.copy()
            ray.visual_terminal = world_point.copy()
            return ray
        
        # EXISTING: Perspective camera logic
        origin = camera.position.copy()
        world_point = np.asarray(world_point, dtype=np.float64)
        
        # Calculate direction from camera to world point
        direction = world_point - origin
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        else:
            # Fallback to camera's forward direction
            direction = camera.R.T @ np.array([0, 0, 1])
            
        ray = cls(
            origin=origin,
            direction=direction,
            terminal_point=world_point,
            has_accurate_depth=True,  # World point is known precisely
            pixel_coord=None,  # Not originating from a pixel
            source_camera=camera
        )

        # Visual start/end default to the true geometry (no offset).
        ray.visual_origin = ray.origin.copy()
        ray.visual_terminal = ray.terminal_point.copy()

        return ray
    
    def cast_on_mesh(self, mesh) -> Optional[np.ndarray]:
        """
        Cast this ray onto a PyVista mesh to find intersection point.
        
        Uses PyVista's ray_trace method to find where the ray intersects
        the mesh surface.
        
        Args:
            mesh: A PyVista PolyData mesh to intersect with.
            
        Returns:
            np.ndarray or None: Intersection point if hit, None otherwise.
            
        # TODO: Add mesh intersection refinement when point cloud mesh is available
        # This can improve accuracy when z-channel is not available
        """
        if mesh is None:
            return None
            
        try:
            # Calculate a far end point for the ray
            ray_length = 1000.0  # Large distance to ensure we hit the mesh
            end_point = self.origin + self.direction * ray_length
            
            # Perform ray trace
            intersection_points, cell_ids = mesh.ray_trace(
                self.origin.tolist(), 
                end_point.tolist(),
                first_point=True
            )
            
            if len(intersection_points) > 0:
                # Update terminal point to intersection
                self.terminal_point = intersection_points[0]
                self.has_accurate_depth = True  # Mesh intersection provides accurate depth
                return self.terminal_point.copy()
                
        except Exception as e:
            print(f"Ray-mesh intersection error: {e}")
            
        return None
    
    def project_to_cameras(self, cameras: Dict[str, 'Camera']) -> Dict[str, Tuple[float, float, bool]]:
        """
        Project this ray's terminal point onto multiple camera views.
        
        For each camera, calculates the 2D pixel coordinate where the
        ray's terminal point would appear.
        
        Args:
            cameras: Dictionary mapping image_path to Camera objects.
            
        Returns:
            Dict mapping image_path to (pixel_x, pixel_y, is_valid) tuples.
            is_valid indicates if the projection is within the camera's FOV.
            
        # TODO: Add occlusion check here - skip cameras where point is occluded
        # Could use camera.is_point_occluded_depth_based() if z-channel available
        # or camera.is_point_occluded_ray_casting() if mesh available
        """
        projections = {}
        
        for path, camera in cameras.items():
            # Include all cameras including source camera
            # Source camera will show marker at cursor position
            try:
                # Project 3D point to 2D pixel
                pixel_coord = camera.project(self.terminal_point)
                
                if not np.isnan(pixel_coord).any():
                    # Check if within image bounds
                    is_valid = (0 <= pixel_coord[0] < camera.width and 
                                0 <= pixel_coord[1] < camera.height)
                    
                    projections[path] = (float(pixel_coord[0]), 
                                         float(pixel_coord[1]), 
                                         is_valid)
                                        
            except Exception as e:
                # Silently skip cameras that fail projection
                pass
                
        return projections
    
    def get_distance_from_camera(self) -> float:
        """
        Get the distance from the camera to the terminal point.
        
        Returns:
            float: Distance in world units.
        """
        return float(np.linalg.norm(self.terminal_point - self.origin))
    
    def to_line_segment(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the ray as a line segment for visualization.
        
        Returns:
            Tuple of (start_point, end_point) as numpy arrays.
        """
        # Return the visual segment by default (prevents near-plane clipping
        # when the viewer matches a camera). If raw geometry is required,
        # use .origin and .terminal_point directly.
        return (self.get_visual_start(), self.get_visual_end())
    
    def to_pyvista_line(self):
        """
        Create a PyVista Line mesh for visualization.
        
        Returns:
            pyvista.Line or None: Line mesh if PyVista available.
        """            
        start = self.get_visual_start().tolist()
        end = self.get_visual_end().tolist()
        return pv.Line(start, end)
    
    def to_pyvista_arrow(self, scale: float = 0.1):
        """
        Create a PyVista Arrow mesh for direction visualization.
        
        Args:
            scale: Scale factor for the arrow size.
            
        Returns:
            pyvista.Arrow or None: Arrow mesh if PyVista available.
        """
        # Arrow should originate from the visual start so it is visible
        # when the viewer is positioned at the camera.
        start = self.get_visual_start().tolist()
        return pv.Arrow(
            start=start,
            direction=self.direction.tolist(),
            scale=scale
        )

    def get_visual_start(self) -> np.ndarray:
        """Return the visualized start point for rendering."""
        return self.visual_origin.copy() if hasattr(self, 'visual_origin') else self.origin.copy()

    def get_visual_end(self) -> np.ndarray:
        """Return the visualized end point for rendering."""
        return self.visual_terminal.copy() if hasattr(self, 'visual_terminal') else self.terminal_point.copy()
    
    def __repr__(self) -> str:
        """String representation of the ray."""
        depth_str = "accurate" if self.has_accurate_depth else "estimated"
        return (f"CameraRay(origin={self.origin}, "
                f"terminal={self.terminal_point}, "
                f"depth={depth_str})")
        

class BatchedRayManager:
    """
    Manages batched rendering of camera rays for efficient visualization.
    
    Instead of creating individual line actors per ray (N draw calls),
    this class maintains a single PolyData mesh containing all ray lines and updates
    point coordinates in-place when the mouse moves.
    
    Attributes:
        ray_mesh: PolyData containing all ray line segments
        ray_actor: Single actor for all ray lines
    """
    
    def __init__(self):
        """Initialize the BatchedRayManager."""
        self.ray_mesh: Optional[pv.PolyData] = None
        self.ray_actor = None
        
        # Ray colors stored per ray
        self._ray_colors: Optional[np.ndarray] = None
        self._num_rays = 0
        
    def build_ray_batch(self, 
                        rays_with_colors: List[Tuple['CameraRay', tuple]]) -> Optional[pv.PolyData]:
        """
        Build merged mesh for multiple rays.
        
        Args:
            rays_with_colors: List of (CameraRay, color_rgb) tuples
                             Colors should be RGB tuples (0-255)
            
        Returns:
            ray_lines_mesh
        """
        if not rays_with_colors:
            self.ray_mesh = None
            self._num_rays = 0
            return None
        
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
            # Use visual start/end to avoid near-plane clipping / coincidence
            points.append(ray.get_visual_start().tolist())
            points.append(ray.get_visual_end().tolist())
            
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
            return None
        
        # Create lines mesh
        self.ray_mesh = pv.PolyData(np.array(points), lines=np.array(lines))
        self._ray_colors = np.array(colors)
        self.ray_mesh['RGB'] = (self._ray_colors * 255).astype(np.uint8)
        
        return self.ray_mesh
    
    def add_to_plotter(self, plotter, line_width: float = 3) -> Optional['vtkActor']:
        """
        Add the batched ray mesh to a plotter.
        
        Args:
            plotter: PyVista plotter instance
            line_width: Width of ray lines
            
        Returns:
            ray_actor
        """
        # Remove existing actors
        self.remove_from_plotter(plotter)
        
        if self.ray_mesh is not None:
            self.ray_actor = plotter.add_mesh(
                self.ray_mesh,
                scalars='RGB',
                rgb=True,
                line_width=line_width,
                render_lines_as_tubes=True,
                name='_batched_rays',
                pickable=False,
                smooth_shading=False,
                reset_camera=False
            )
        
        return self.ray_actor
    
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
        colors = []
        for i, (ray, color) in enumerate(rays_with_colors):
            if ray is not None:
                pt_idx = i * 2
                points[pt_idx] = ray.get_visual_start()
                points[pt_idx + 1] = ray.get_visual_end()

            # Normalize/convert color to 0-1 float tuple (store per-endpoint)
            if isinstance(color, tuple) and any(c > 1 for c in color[:3]):
                norm_color = tuple(c / 255 for c in color[:3])
            else:
                # assume already normalized or QColor-like sequence
                norm_color = color[:3] if len(color) >= 3 else color

            # Append same color for both endpoints of the line
            colors.append(norm_color)
            colors.append(norm_color)

        # Update the mesh color array
        try:
            self._ray_colors = np.array(colors)
            # PyVista expects uint8 RGB if using 0-255, so convert from 0-1 floats
            self.ray_mesh['RGB'] = (self._ray_colors * 255).astype(np.uint8)
        except Exception:
            # Fallback: leave existing colors if update fails
            pass

        # Mark mesh modified so the plotter updates
        self.ray_mesh.Modified()
    
    def set_visibility(self, visible: bool):
        """Set visibility of ray actor."""
        if self.ray_actor is not None:
            self.ray_actor.SetVisibility(visible)
    
    def remove_from_plotter(self, plotter):
        """Remove ray actor from plotter."""
        if self.ray_actor is not None:
            try:
                plotter.remove_actor(self.ray_actor)
            except:
                pass
            self.ray_actor = None
    
    def clear(self):
        """Clear all cached data."""
        self.ray_mesh = None
        self.ray_actor = None
        self._ray_colors = None
        self._num_rays = 0