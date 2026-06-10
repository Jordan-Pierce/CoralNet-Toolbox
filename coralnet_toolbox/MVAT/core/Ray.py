"""
Ray geometry and batch actor management for MVAT.

Contains:
- Ray: single ray geometry and VTK actor
- BatchedRayManager: manages all ray actors in the plotter
"""
from typing import Optional, Dict, Tuple, List, TYPE_CHECKING

import numpy as np

import pyvista as pv

from coralnet_toolbox.MVAT.core.Cameras import Camera

if TYPE_CHECKING:
    try:
        from vtkmodules.vtkRenderingCore import vtkActor
    except ImportError:
        from vtk import vtkActor


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
                 source_camera: Optional['Camera'] = None,
                 element_id: int = -1):
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
        self.element_id = element_id
        # Visualized start/end points (may be offset slightly to avoid
        # coincident geometry with the viewer camera and near-plane clipping).
        # These are computed by factory methods; default to the raw values.
        self.visual_origin = self.origin.copy()
        self.visual_terminal = self.terminal_point.copy()
        
    @classmethod
    def from_pixel_and_camera(cls, pixel_xy: Tuple[int, int], camera: 'Camera', 
                              depth: Optional[float] = None, default_depth: float = 10.0,
                              element_id: int = -1) -> 'CameraRay':
        
        # Perspective camera logic
        # Camera origin is the optical center in world coordinates
        origin = camera.position.copy()
        
        # Check if we have valid depth
        has_accurate_depth = False
        actual_depth = default_depth
        
        # For elevation models, accept negative values; for depth models, only accept positive
        z_data_type = getattr(camera._raster, 'z_data_type', None) if hasattr(camera, '_raster') else None
        is_elevation = z_data_type == 'elevation'
        
        depth_valid = (depth is not None and not np.isnan(depth) and (is_elevation or depth > 0))
        
        if depth_valid:
            actual_depth = depth
            has_accurate_depth = True
        
        # ---> THE UPDATE: Try proper camera unprojection first (handles distortion/fisheye) <---
        terminal_point = camera.unproject(pixel_xy, depth=actual_depth)
        
        # Fallback to linear math if the camera unprojection fails or isn't supported
        if terminal_point is None:
            pixel_hom = np.array([pixel_xy[0], pixel_xy[1], 1.0])
            point_cam = actual_depth * (camera.K_inv @ pixel_hom)
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
            source_camera=camera,
            element_id=element_id
        )

        # Visual start/end default to the true geometry (no offset).
        ray.visual_origin = ray.origin.copy()
        ray.visual_terminal = ray.terminal_point.copy()

        return ray
    
    @classmethod
    def from_world_point_and_camera(cls, 
                                    world_point: np.ndarray, 
                                    camera: 'Camera',
                                    element_id: int = -1) -> 'CameraRay':
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
        # Perspective camera logic
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
            source_camera=camera,
            element_id=element_id
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


# ---------------------------------------------------------------------------
# BatchedRayManager (inlined from former RayManager.py)
# ---------------------------------------------------------------------------
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    try:
        from vtkmodules.vtkRenderingCore import vtkActor
    except ImportError:
        from vtk import vtkActor

    from coralnet_toolbox.MVAT.core.Ray import CameraRay


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


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

        self._ray_colors: Optional[np.ndarray] = None
        self._num_rays = 0
        self._allocated_rays = 0  # Total pre-allocated slots (>= _num_rays)

    def build_ray_batch(self,
                        rays_with_colors: List[Tuple['CameraRay', tuple]],
                        pre_allocate_factor: float = 1.5) -> Optional[pv.PolyData]:
        """
        Build merged mesh for multiple rays with optional pre-allocation.

        Args:
            rays_with_colors: List of (CameraRay, color_rgb) tuples.
                              Colors should be RGB tuples (0-255).
            pre_allocate_factor: Factor to pre-allocate beyond the active count
                                 (e.g., 1.5 means allocate 50% more slots).
                                 This reduces actor churn when ray count varies.

        Returns:
            ray_lines_mesh
        """
        if not rays_with_colors:
            self.ray_mesh = None
            self._num_rays = 0
            self._allocated_rays = 0
            return None

        self._num_rays = len(rays_with_colors)
        self._allocated_rays = max(self._num_rays, int(self._num_rays * pre_allocate_factor))

        points = []
        lines = []
        colors = []

        # Add points and lines for actual rays
        for i, (ray, color) in enumerate(rays_with_colors):
            if ray is None:
                continue

            pt_idx = len(points)
            points.append(ray.get_visual_start().tolist())
            points.append(ray.get_visual_end().tolist())
            lines.extend([2, pt_idx, pt_idx + 1])

            if isinstance(color, tuple) and any(c > 1 for c in color[:3]):
                norm_color = tuple(c / 255 for c in color[:3])
            else:
                norm_color = color[:3] if len(color) >= 3 else color
            colors.append(norm_color)
            colors.append(norm_color)

        # Pad with degenerate rays (zero-length at origin) for pre-allocated slots
        origin = np.array([0, 0, 0], dtype=np.float64)
        for _ in range(self._allocated_rays - self._num_rays):
            pt_idx = len(points)
            points.append(origin.tolist())
            points.append(origin.tolist())
            lines.extend([2, pt_idx, pt_idx + 1])
            colors.append((0, 0, 0))
            colors.append((0, 0, 0))

        if not points:
            self.ray_mesh = None
            return None

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
            if self.ray_actor is not None:
                self.ray_actor.SetPickable(False)

        return self.ray_actor

    def update_ray_endpoints(self,
                             rays_with_colors: List[Tuple['CameraRay', tuple]]):
        """
        Update ray endpoints in-place (more efficient than rebuilding).

        If the new ray count fits within the pre-allocated capacity, updates
        points in-place and pads unused slots with degenerate rays.
        Only rebuilds if the new count exceeds the allocated capacity.

        Args:
            rays_with_colors: List of (CameraRay, color_rgb) tuples
        """
        if self.ray_mesh is None or len(rays_with_colors) > self._allocated_rays:
            self.build_ray_batch(rays_with_colors)
            return

        self._num_rays = len(rays_with_colors)
        points = self.ray_mesh.points
        colors = []
        origin = np.array([0, 0, 0], dtype=np.float64)

        # Update active rays
        for i, (ray, color) in enumerate(rays_with_colors):
            pt_idx = i * 2
            if ray is not None:
                points[pt_idx] = ray.get_visual_start()
                points[pt_idx + 1] = ray.get_visual_end()
            else:
                points[pt_idx] = origin
                points[pt_idx + 1] = origin

            if isinstance(color, tuple) and any(c > 1 for c in color[:3]):
                norm_color = tuple(c / 255 for c in color[:3])
            else:
                norm_color = color[:3] if len(color) >= 3 else color
            colors.append(norm_color)
            colors.append(norm_color)

        # Degenerate rays for unused pre-allocated slots
        for i in range(self._allocated_rays - self._num_rays):
            pt_idx = (self._num_rays + i) * 2
            points[pt_idx] = origin
            points[pt_idx + 1] = origin
            colors.append((0, 0, 0))
            colors.append((0, 0, 0))

        try:
            self._ray_colors = np.array(colors)
            self.ray_mesh['RGB'] = (self._ray_colors * 255).astype(np.uint8)
        except Exception:
            pass

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
        self._allocated_rays = 0
