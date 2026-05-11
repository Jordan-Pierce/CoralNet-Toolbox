"""
Base class for 3D mesh interaction tools in the MVATViewer.

Mirrors the structure of Tools/QtTool.py but operates in VTK viewport space:
  - Event handlers receive (event, face_id, world_pos) alongside the raw QMouseEvent.
  - No Qt scene, annotation window, or 2D crosshair concepts.
  - activate() / deactivate() are managed by MVATManager.set_selected_3d_tool(),
    exactly mirroring how AnnotationWindow.set_selected_tool() manages 2D tools.

Naming mirrors Tools/QtTool.py intentionally so the two hierarchies are easy
to read side-by-side.
"""

import warnings

import numpy as np
import pyvista as pv

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Tool3D:
    """
    Abstract base class for all 3D mesh interaction tools.

    Analogous to Tools/QtTool.Tool but designed for the MVATViewer (PyVista / VTK).
    The base class owns the shared preview sphere, hover batching, and label-
    colored highlight overlay.  Subclasses override only class-specific behavior
    such as preview color, label resolution, and paint/erase commits.

    Attributes:
        mvat_viewer:  The MVATViewer widget (owns the PyVista plotter and scene).
        mvat_manager: The MVATManager (owns labels, cameras, multi-annotate state).
        active (bool): True while this tool is the selected_3d_tool.
                       Mirrors Tool.active.
    """

    _DEFAULT_RADIUS_FRACTION = 0.015
    _PREVIEW_COLOR = 'white'
    _PREVIEW_OPACITY = 0.35

    def __init__(self, mvat_viewer, mvat_manager):
        """
        Args:
            mvat_viewer:  MVATViewer instance — provides plotter, scene_context,
                          and set_active_3d_tool().
            mvat_manager: MVATManager instance — provides annotation_window,
                          cameras, multi_annotate_enabled, and the label painter.
        """
        self.mvat_viewer = mvat_viewer
        self.mvat_manager = mvat_manager

        self.active = False
        self.preview_only = True
        self.brush_size = 0.1
        self.brush_shape = 'circle'

        self._preview_mesh = None            # pv.PolyData shared with the plotter
        self._preview_mesh_points_unit = None  # unit-scale points (radius=1) for fast transforms
        self._preview_actor = None
        self._preview_actor_shape = None
        self._last_hover_world_pos = None

    # ------------------------------------------------------------------
    # Lifecycle  (mirrors Tool.activate / Tool.deactivate)
    # ------------------------------------------------------------------

    def activate(self):
        """
        Activate this tool.
        Called by MVATManager.set_selected_3d_tool() after deactivating the
        previous tool — mirrors AnnotationWindow.set_selected_tool() calling
        tool.activate().
        """
        self.active = True
        self.preview_only = False
        calibrate_brush_size = getattr(self, '_calibrate_brush_size', None)
        if callable(calibrate_brush_size):
            try:
                calibrate_brush_size()
            except Exception:
                pass
        self._init_preview_actor()
        try:
            focal_point = np.array(self.mvat_viewer.plotter.camera.focal_point)
            self._update_preview_sphere(focal_point)
        except Exception:
            pass

    def deactivate(self):
        """
        Deactivate this tool and clean up any VTK actors / state.
        Called by MVATManager.set_selected_3d_tool() before switching to a
        different tool — mirrors AnnotationWindow.set_selected_tool() calling
        previous_tool.deactivate().
        """
        self.stop_current_drawing()
        self._last_hover_world_pos = None
        self._remove_preview_sphere()
        self.active = False
        self.preview_only = True

    def set_brush_size(self, brush_size, center=None):
        """Set the world-space brush radius and refresh the preview sphere."""
        try:
            self.brush_size = max(1e-6, float(brush_size))
        except Exception:
            return

        if not self.active:
            return

        if center is None:
            center = self._last_hover_world_pos

        if center is None:
            try:
                center = np.asarray(self.mvat_viewer.plotter.camera.focal_point, dtype=np.float64)
            except Exception:
                center = None

        if center is not None:
            self._update_preview_sphere(center)

    def set_brush_shape(self, brush_shape, center=None):
        """Set the preview shape and refresh the actor in-place."""
        try:
            brush_shape = str(brush_shape).strip().lower()
        except Exception:
            return

        if brush_shape not in ('circle', 'square'):
            return

        self.brush_shape = brush_shape

        if not self.active:
            return

        if center is None:
            center = self._last_hover_world_pos

        if center is None:
            try:
                center = np.asarray(self.mvat_viewer.plotter.camera.focal_point, dtype=np.float64)
            except Exception:
                center = None

        if center is not None:
            self._update_preview_sphere(center)

    def stop_current_drawing(self):
        """
        Force-stop any in-progress drawing / stroke operation.
        Subclasses should override this to commit or discard the current stroke.
        Mirrors Tool.stop_current_drawing().
        """
        pass

    # ------------------------------------------------------------------
    # Event handlers — called by MVATViewer.eventFilter when this tool is
    # the active_3d_tool.  Signatures differ from the 2D Tool equivalents
    # because VTK picking enriches each event with a resolved face_id and
    # world_pos before it reaches the tool.
    # ------------------------------------------------------------------

    def mousePressEvent(self, event, face_id: int, world_pos):
        """
        Handle a left-button press on the 3D viewport.

        Args:
            event:     The original QMouseEvent forwarded from eventFilter.
            face_id:   VTK cell ID of the mesh face under the cursor, or -1 if
                       the cursor is over empty space.
            world_pos: np.ndarray (3,) world-space coordinate of the pick, or
                       None when face_id == -1.
        """
        pass

    def mouseMoveEvent(self, event, face_id: int, world_pos):
        """
        Handle a mouse-move event in the 3D viewport.

        Args:
            event:     The original QMouseEvent forwarded from eventFilter.
            face_id:   VTK cell ID under the cursor, or -1.
            world_pos: np.ndarray (3,) world coordinate, or None.
        """
        if not self.active:
            return

        if world_pos is not None:
            self._last_hover_world_pos = np.asarray(world_pos, dtype=np.float64)
            self._update_preview_sphere(world_pos)
            manager = getattr(self, 'mvat_manager', None)
            if manager is not None:
                try:
                    manager.update_sphere_hover_overlay(world_pos, render=False)
                except Exception:
                    pass
        else:
            self._last_hover_world_pos = None
            self._hide_preview_sphere()
            manager = getattr(self, 'mvat_manager', None)
            if manager is not None:
                try:
                    manager.clear_sphere_hover_overlay(reset_context=True, render=False)
                except Exception:
                    pass

    def mouseReleaseEvent(self, event):
        """
        Handle a left-button release in the 3D viewport.

        Args:
            event: The original QMouseEvent forwarded from eventFilter.
        """
        pass

    def wheelEvent(self, event, delta_y: int):
        """
        Handle a Ctrl+wheel event forwarded from MVATViewer.eventFilter.
        Typically used to resize the brush radius.

        Args:
            event:   The original QWheelEvent.
            delta_y: angleDelta().y() (positive = scroll up / zoom in).
        """
        if not self.active or not (event.modifiers() & Qt.ControlModifier):
            return

        notches = delta_y / 120.0
        factor = 1.15 ** notches
        self.brush_size = max(1e-6, self.brush_size * factor)

        center = self._last_hover_world_pos
        if center is None:
            try:
                center = np.asarray(self.mvat_viewer.plotter.camera.focal_point, dtype=np.float64)
            except Exception:
                center = None

        if center is not None:
            self.set_brush_size(self.brush_size, center=center)
            manager = getattr(self, 'mvat_manager', None)
            if manager is not None:
                try:
                    manager.update_sphere_hover_overlay(center, render=False)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Shared preview / hover / highlight behavior
    # ------------------------------------------------------------------

    def _use_active_label_preview_color(self) -> bool:
        return True

    def _get_selected_label(self):
        try:
            return self.mvat_manager.annotation_window.selected_label
        except Exception:
            return None

    def _preview_color_rgb_float(self):
        """Return (r, g, b) in [0.0, 1.0] for the preview sphere."""
        if self._use_active_label_preview_color():
            selected_label = self._get_selected_label()
            if selected_label is not None:
                try:
                    c = QColor(selected_label.color)
                    return (c.redF(), c.greenF(), c.blueF())
                except Exception:
                    pass

        try:
            c = pv.Color(self._PREVIEW_COLOR)
            return c.float_rgb
        except Exception:
            return (1.0, 1.0, 1.0)

    def _init_preview_actor(self):
        """
        Create the wireframe preview mesh once and register it with the plotter.

        Uses a PyVista mesh so position/size updates are pure numpy point-array
        assignments rather than VTK pipeline re-executions.  The mesh is built at
        unit scale (radius = 1) so that a single multiply+add can apply both
        brush_size and world position without ever rebuilding geometry.
        """
        if self._preview_actor is not None:
            return

        try:
            if self.brush_shape == 'square':
                # Box with half-extents of ±1 — scale by brush_size at update time
                base = pv.Box(bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0))
            else:
                base = pv.Sphere(radius=1.0, theta_resolution=16, phi_resolution=16)

            mesh = base.extract_all_edges()
            self._preview_mesh = mesh
            self._preview_mesh_points_unit = mesh.points.copy()  # immutable reference

            r, g, b = self._preview_color_rgb_float()
            actor = self.mvat_viewer.plotter.add_mesh(
                mesh,
                color=(r, g, b),
                opacity=self._PREVIEW_OPACITY,
                style='wireframe',
                render=False,
            )
            actor.VisibilityOff()

            self._preview_actor = actor
            self._preview_actor_shape = self.brush_shape
        except Exception as e:
            print(f"⚠️  {self.__class__.__name__}: could not create preview actor: {e}")
            self._preview_mesh = None
            self._preview_mesh_points_unit = None
            self._preview_actor = None
            self._preview_actor_shape = None

    def _hide_preview_sphere(self):
        if self._preview_actor is None:
            return
        try:
            self._preview_actor.VisibilityOff()
            if self._preview_mesh is not None:
                self._preview_mesh.Modified()
        except Exception:
            pass

    def _update_preview_sphere(self, center: np.ndarray):
        """
        Move and resize the preview actor in-place.

        Translates and scales the pre-built unit mesh via a single numpy
        expression — no VTK pipeline re-execution, no forced render().
        The plotter's own render loop picks up the Modified() flag.
        """
        # Rebuild actor if shape changed or not yet initialised
        if (
            self._preview_mesh is None or
            self._preview_actor is None or
            self._preview_actor_shape != self.brush_shape
        ):
            if self._preview_actor is not None:
                try:
                    self.mvat_viewer.plotter.remove_actor(self._preview_actor, render=False)
                except Exception:
                    pass
            self._preview_mesh = None
            self._preview_mesh_points_unit = None
            self._preview_actor = None
            self._preview_actor_shape = None
            self._init_preview_actor()

        if self._preview_mesh is None:
            return

        try:
            c = np.asarray(center, dtype=np.float64)

            # Single numpy op: scale unit geometry by brush_size, then translate
            self._preview_mesh.points = self._preview_mesh_points_unit * self.brush_size + c

            # Update color in case the active label changed (cheap property set, not pipeline)
            r, g, b = self._preview_color_rgb_float()
            self._preview_actor.GetProperty().SetColor(r, g, b)
            self._preview_actor.VisibilityOn()

            # Notify VTK the points changed — render loop picks this up automatically
            self._preview_mesh.Modified()
        except Exception:
            pass

    def _remove_preview_sphere(self):
        """Detach the persistent actor from the renderer on deactivation."""
        if self._preview_actor is not None:
            try:
                manager = getattr(self, 'mvat_manager', None)
                if manager is not None:
                    try:
                        manager.clear_sphere_hover_overlay(reset_context=True, render=False)
                    except Exception:
                        pass
                self.mvat_viewer.plotter.remove_actor(self._preview_actor, render=False)
            except Exception:
                pass
            self._preview_mesh = None
            self._preview_mesh_points_unit = None
            self._preview_actor = None
            self._preview_actor_shape = None
