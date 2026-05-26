"""
Batched Ray Manager for MVAT.

Manages batched rendering of camera rays for efficient visualization.
Extracted from core/Ray.py — geometry lives in core/Ray.CameraRay;
batch coordination lives here.
"""
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

    def build_ray_batch(self,
                        rays_with_colors: List[Tuple['CameraRay', tuple]]) -> Optional[pv.PolyData]:
        """
        Build merged mesh for multiple rays.

        Args:
            rays_with_colors: List of (CameraRay, color_rgb) tuples.
                              Colors should be RGB tuples (0-255).

        Returns:
            ray_lines_mesh
        """
        if not rays_with_colors:
            self.ray_mesh = None
            self._num_rays = 0
            return None

        self._num_rays = len(rays_with_colors)

        points = []
        lines = []
        colors = []

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

        Only works if the number of rays hasn't changed.

        Args:
            rays_with_colors: List of (CameraRay, color_rgb) tuples
        """
        if self.ray_mesh is None or len(rays_with_colors) != self._num_rays:
            self.build_ray_batch(rays_with_colors)
            return

        points = self.ray_mesh.points
        colors = []

        for i, (ray, color) in enumerate(rays_with_colors):
            if ray is not None:
                pt_idx = i * 2
                points[pt_idx] = ray.get_visual_start()
                points[pt_idx + 1] = ray.get_visual_end()

            if isinstance(color, tuple) and any(c > 1 for c in color[:3]):
                norm_color = tuple(c / 255 for c in color[:3])
            else:
                norm_color = color[:3] if len(color) >= 3 else color
            colors.append(norm_color)
            colors.append(norm_color)

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
