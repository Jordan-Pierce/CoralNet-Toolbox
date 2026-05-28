"""
Sphere Actor Manager for MVAT.

Manages a single hollow sphere or cube wireframe actor that follows the mouse
on the mesh surface, providing a visual brush-radius preview.
Extracted from core/Ray.py — batching helpers live in managers/.
"""
from typing import Optional

import numpy as np
import pyvista as pv


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SphereActorManager:
    """
    Manages a single hollow sphere or cube actor that follows the mouse on the mesh.

    Creates a preview actor once at startup and only updates its position
    when the mouse moves, following the same pattern as RayManager
    for efficient rendering.

    Attributes:
        sphere_actor: Single actor for the sphere
        sphere_mesh: PolyData mesh for the sphere
        current_position: Current position of the sphere (None if not set)
        radius: Radius of the sphere
    """

    def __init__(self, radius: float = 0.1):
        """
        Initialize the SphereActorManager.

        Args:
            radius: Radius of the sphere to create
        """
        self.sphere_actor = None
        self.sphere_mesh = None
        self._original_sphere_mesh = None
        self._preview_mesh_shape = None
        self.current_position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.radius = radius
        self.shape = 'circle'
        self._plotter = None
        self._last_color = (144, 238, 144)
        self._last_line_width = 1.0
        self._create_sphere()
        if self.sphere_mesh is not None:
            self.sphere_mesh.translate(self.current_position)

    def _create_sphere(self):
        """Create the hollow preview mesh (sphere or cube wireframe)."""
        if self.shape == 'square':
            base_sphere = pv.Cube(
                x_length=self.radius * 2.0,
                y_length=self.radius * 2.0,
                z_length=self.radius * 2.0,
            )
        else:
            base_sphere = pv.Sphere(radius=self.radius, theta_resolution=16, phi_resolution=16)
        try:
            self._original_sphere_mesh = base_sphere.extract_all_edges()
        except AttributeError:
            self._original_sphere_mesh = base_sphere.copy()
        self.sphere_mesh = self._original_sphere_mesh.copy()
        self._preview_mesh_shape = self.shape

    def set_position(self, position: np.ndarray):
        """
        Update the sphere position.

        Args:
            position: 3D position in world coordinates as numpy array
        """
        if position is None or self.sphere_mesh is None or self._original_sphere_mesh is None:
            return

        position = np.asarray(position, dtype=np.float64)

        if self.current_position is None or not np.allclose(self.current_position, position):
            self.current_position = position.copy()
            original_points = self._original_sphere_mesh.points
            self.sphere_mesh.points = original_points + position
            self.sphere_mesh.Modified()

    def add_to_plotter(self, plotter, color: tuple = (144, 238, 144),
                       line_width: float = 1.0) -> 'vtkActor':
        """
        Add the sphere actor to a plotter.

        Args:
            plotter: PyVista plotter instance
            color: RGB color tuple (0-255) for the sphere wireframe
            line_width: Width of the wireframe lines

        Returns:
            sphere_actor
        """
        self.remove_from_plotter(plotter)
        self._plotter = plotter
        self._last_color = color
        self._last_line_width = line_width

        if self.sphere_mesh is not None:
            if isinstance(color, tuple) and any(c > 1 for c in color[:3]):
                norm_color = tuple(c / 255.0 for c in color[:3])
            else:
                norm_color = color[:3] if len(color) >= 3 else color

            is_edges = self.sphere_mesh.n_cells == 0 or self.sphere_mesh.get_cell(0).GetNumberOfPoints() <= 2

            if is_edges:
                self.sphere_actor = plotter.add_mesh(
                    self.sphere_mesh,
                    color=norm_color,
                    line_width=line_width,
                    render_lines_as_tubes=False,
                    name='_sphere_actor',
                    pickable=False,
                    smooth_shading=False,
                    reset_camera=False
                )
            else:
                self.sphere_actor = plotter.add_mesh(
                    self.sphere_mesh,
                    color=norm_color,
                    style='wireframe',
                    line_width=line_width,
                    name='_sphere_actor',
                    pickable=False,
                    smooth_shading=False,
                    reset_camera=False
                )

            if self.sphere_actor is not None:
                self.sphere_actor.SetPickable(False)

        return self.sphere_actor

    def set_radius(self, radius: float):
        """Update the sphere radius and rebuild the cached geometry."""
        if radius is None:
            return
        try:
            radius = float(radius)
        except Exception:
            return
        if not np.isfinite(radius) or radius <= 0:
            return
        if np.isclose(self.radius, radius):
            return

        was_visible = False
        if self.sphere_actor is not None:
            try:
                was_visible = bool(self.sphere_actor.GetVisibility())
            except Exception:
                pass

        plotter = self._plotter
        self.radius = radius
        self._create_sphere()

        if self.current_position is not None and self.sphere_mesh is not None and self._original_sphere_mesh is not None:
            self.sphere_mesh.points = self._original_sphere_mesh.points + self.current_position
            self.sphere_mesh.Modified()

        if plotter is not None and self.sphere_actor is not None:
            try:
                plotter.remove_actor(self.sphere_actor)
            except Exception:
                pass
            self.sphere_actor = None
            self.add_to_plotter(plotter, color=self._last_color, line_width=self._last_line_width)
            if self.sphere_actor is not None and not was_visible:
                try:
                    self.sphere_actor.SetVisibility(False)
                except Exception:
                    pass

    def set_shape(self, shape: str, center: np.ndarray = None):
        """Update the preview mesh shape and rebuild the cached geometry."""
        if shape is None:
            return
        try:
            shape = str(shape).strip().lower()
        except Exception:
            return
        if shape not in ('circle', 'square'):
            return

        if center is not None:
            try:
                center = np.asarray(center, dtype=np.float64).reshape(-1)
                if center.size >= 3:
                    self.current_position = center[:3].copy()
            except Exception:
                center = None

        if self.shape == shape and self._preview_mesh_shape == shape and center is None:
            return

        was_visible = False
        if self.sphere_actor is not None:
            try:
                was_visible = bool(self.sphere_actor.GetVisibility())
            except Exception:
                pass

        plotter = self._plotter
        self.shape = shape
        self._create_sphere()

        if self.current_position is not None and self.sphere_mesh is not None and self._original_sphere_mesh is not None:
            self.sphere_mesh.points = self._original_sphere_mesh.points + self.current_position
            self.sphere_mesh.Modified()

        if plotter is not None and self.sphere_actor is not None:
            try:
                plotter.remove_actor(self.sphere_actor)
            except Exception:
                pass
            self.sphere_actor = None
            self.add_to_plotter(plotter, color=self._last_color, line_width=self._last_line_width)
            if self.sphere_actor is not None and not was_visible:
                try:
                    self.sphere_actor.SetVisibility(False)
                except Exception:
                    pass

    def set_visibility(self, visible: bool):
        """Set visibility of sphere actor."""
        if self.sphere_actor is not None:
            self.sphere_actor.SetVisibility(visible)

    def remove_from_plotter(self, plotter):
        """Remove sphere actor from plotter."""
        if self.sphere_actor is not None:
            try:
                plotter.remove_actor(self.sphere_actor)
            except:
                pass
            self.sphere_actor = None

    def clear(self):
        """Clear all cached data."""
        self.sphere_mesh = None
        self.sphere_actor = None
        self.current_position = None
        self._plotter = None
