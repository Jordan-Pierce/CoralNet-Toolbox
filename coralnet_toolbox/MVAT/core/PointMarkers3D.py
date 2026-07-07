"""
ColoredPointMarkers3D — a single VTK actor rendering N per-point-colored sphere
markers in the MVATViewer plotter.

A render primitive (sibling of ``CursorPreview3D`` and ``Ray.BatchedRayManager``):
one pyvista actor holds every marker point, each colored by a per-point RGB array,
so adding / moving markers is an in-place points + color update rather than actor
churn. Models the ``MVATManager._on_point_overlay_ready`` point-overlay pattern
(``pv.PolyData`` + ``point_data['OverlayColors']`` + ``render_points_as_spheres``,
``pickable=False``).

Two uses share this one class:
  * a single dynamic point that follows the 3D cursor (``QtMVATViewer``), and
  * the set of static, label-colored FeatureSelect prototype markers
    (``FeatureSelectTool3D``).

Each instance must be created with a UNIQUE ``name`` — pyvista keys actors by name,
so two instances sharing a name would clobber each other's actor.
"""
from typing import Optional

import numpy as np
import pyvista as pv


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ColoredPointMarkers3D:
    """
    Manages one actor that renders a set of colored point-sphere markers.

    Attributes:
        point_size: Screen-space size of each marker (pixels).
        _name: Unique pyvista actor name for this instance.
    """

    def __init__(self, point_size: float = 14.0, name: str = '_colored_point_markers'):
        self._plotter = None
        self._actor = None
        self._mesh: Optional[pv.PolyData] = None
        self._n = 0
        self.point_size = float(point_size)
        self._name = str(name)

    def add_to_plotter(self, plotter):
        """Register the plotter.

        The actor itself is created lazily on the first non-empty ``set_markers``
        so we never add an empty-geometry actor to the scene.
        """
        self._plotter = plotter
        return self._actor

    @staticmethod
    def _swap_mapper_input(actor, mesh_to_add) -> bool:
        """Replace an actor's mapper input in place. Returns False on failure.

        Mirrors ``MVATManager._swap_mapper_input`` so a changing marker count can
        reuse the existing actor (and its scalar/rgb config) without a remove+add.
        """
        try:
            mapper = actor.GetMapper()
        except Exception:
            return False
        if mapper is None:
            return False

        for method_name in ('SetInputDataObject', 'SetInputData'):
            method = getattr(mapper, method_name, None)
            if callable(method):
                try:
                    method(mesh_to_add)
                    try:
                        mapper.Update()
                    except Exception:
                        pass
                    return True
                except Exception:
                    continue
        return False

    def set_markers(self, points, colors):
        """Update the rendered marker set.

        Args:
            points: (N, 3) float world coordinates (None / empty hides the actor).
            colors: (N, 3) uint8 per-point RGB (must match ``points`` length).
        """
        if self._plotter is None:
            return

        # Normalize; empty (or mismatched) input hides the markers.
        try:
            if points is None:
                pts = np.empty((0, 3), dtype=np.float32)
            else:
                pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
            if colors is None:
                cols = np.empty((0, 3), dtype=np.uint8)
            else:
                cols = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        except Exception:
            return

        n = int(pts.shape[0])
        if n == 0 or cols.shape[0] != n:
            self.set_visibility(False)
            self._n = 0
            return

        # Fast path: same count → rewrite points + colors in place (no actor churn).
        # This is the hot path for the single cursor dot that moves every frame.
        if self._actor is not None and self._mesh is not None and self._n == n:
            try:
                self._mesh.points = pts
                self._mesh.point_data['OverlayColors'] = cols
                self._mesh.Modified()
                self.set_visibility(True)
                return
            except Exception:
                pass  # fall through to a full rebuild

        # Rebuild the geometry; swap the mapper input in place when possible.
        mesh = pv.PolyData(pts)
        mesh.point_data['OverlayColors'] = cols
        self._mesh = mesh
        self._n = n

        if self._actor is not None and self._swap_mapper_input(self._actor, mesh):
            self.set_visibility(True)
            return

        # (Re)create the actor.
        if self._actor is not None:
            try:
                self._plotter.remove_actor(self._actor, render=False)
            except Exception:
                pass
            self._actor = None

        try:
            self._actor = self._plotter.add_mesh(
                mesh,
                scalars='OverlayColors',
                rgb=True,
                style='points',
                point_size=self.point_size,
                render_points_as_spheres=True,
                copy_mesh=False,
                lighting=False,
                show_scalar_bar=False,
                pickable=False,
                reset_camera=False,
                name=self._name,
            )
            if self._actor is not None:
                try:
                    self._actor.SetPickable(False)
                except Exception:
                    pass
        except Exception:
            self._actor = None

    def set_visibility(self, visible: bool):
        """Show or hide the marker actor."""
        if self._actor is not None:
            try:
                self._actor.SetVisibility(bool(visible))
            except Exception:
                pass

    def clear(self):
        """Hide all markers (keeps the plotter registration)."""
        self.set_markers(None, None)

    def remove_from_plotter(self, plotter=None):
        """Remove the marker actor from the plotter and drop cached geometry."""
        plotter = plotter or self._plotter
        if self._actor is not None and plotter is not None:
            try:
                plotter.remove_actor(self._actor, render=False)
            except Exception:
                pass
        self._actor = None
        self._mesh = None
        self._n = 0
