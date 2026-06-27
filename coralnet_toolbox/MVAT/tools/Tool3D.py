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
    tool_kind = None
    # Tools that need right-button presses (e.g. for a negative prototype) set
    # this True; the viewer only forwards right-clicks to opt-in tools so the
    # others keep normal right-button pan/double-click behavior.
    wants_right_button = False

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
        self._brush_size_customized = False

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
        if not self._brush_size_customized and callable(calibrate_brush_size):
            try:
                calibrate_brush_size()
            except Exception:
                pass
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

    def set_brush_size(self, brush_size, center=None, propagate: bool = True):
        """Set the world-space brush radius and refresh the preview sphere.

        When propagate=True the new radius is also pushed onto the sibling
        3D tool (Brush3DTool ↔ Erase3DTool), so toggling between paint and
        erase preserves the user's chosen size in 3D — mirroring the same
        sharing we do for the 2D BrushTool / EraseTool pair.
        """
        try:
            new_size = max(1e-6, float(brush_size))
        except Exception:
            return

        self.brush_size = new_size
        self._brush_size_customized = True

        if self.active:
            if center is None:
                center = self._last_hover_world_pos
            if center is None:
                try:
                    center = np.asarray(self.mvat_viewer.plotter.camera.focal_point, dtype=np.float64)
                except Exception:
                    center = None
            if center is not None:
                self._update_preview_sphere(center)

        if not propagate:
            return

        # Mirror the new size onto the sibling 3D tool (held by MVATViewer).
        viewer = getattr(self, 'mvat_viewer', None)
        if viewer is None:
            return

        sibling_attr = None
        current_kind = str(getattr(self, 'tool_kind', '')).strip().lower()
        if current_kind == 'brush':
            sibling_attr = '_erase_3d_tool'
        elif current_kind == 'erase':
            sibling_attr = '_brush_3d_tool'
        else:
            tool_name = type(self).__name__.strip().lower()
            if 'erase' in tool_name:
                sibling_attr = '_brush_3d_tool'
            elif 'brush' in tool_name:
                sibling_attr = '_erase_3d_tool'

        if sibling_attr is None:
            return

        sibling = getattr(viewer, sibling_attr, None)
        if sibling is not None and sibling is not self:
            try:
                sibling.set_brush_size(new_size, center=None, propagate=False)
            except Exception:
                try:
                    sibling.brush_size = new_size
                    sibling._brush_size_customized = True
                except Exception:
                    pass

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

    def keyPressEvent(self, event):
        """
        Handle a key press in the 3D viewport.

        Args:
            event: QKeyEvent from MVATViewer.eventFilter.
        """
        pass

    # ------------------------------------------------------------------
    # Shared preview / hover / highlight behavior
    # ------------------------------------------------------------------

    def _paint_shader_active(self) -> bool:
        """True when the GPU paint shader is the active label renderer.

        When it is, painting commits through the shader's O(painted) texture write
        (see MVATManager.submit_3d_face_paint / submit_3d_point_paint), so the
        LabelWorker overlay and the lazy-flush commit are dead work and the tools
        skip them. Falls back to False (overlay path) if the manager is missing.
        """
        psm = getattr(self.mvat_manager, 'paint_shader_manager', None)
        return psm is not None and getattr(psm, 'shader_enabled', False)

    def _use_active_label_preview_color(self) -> bool:
        return True

    def _get_selected_label(self):
        try:
            return self.mvat_manager.annotation_window.selected_label
        except Exception:
            return None

    def _resolve_label(self, label):
        """Resolve a label widget to its (class_id, color_rgb) for mesh painting.

        class_id is the canonical mesh class-id (project order, 'Review' excluded)
        resolved through PropagationEngine.canonical_class_id_for_label_id, so a
        direct brush/fill stroke and a multi-annotate projection paint the same
        label with the same id. Without this the two paths use different integer
        schemes and can collide (one label's faces get recolored to another's).
        Falls back to the mask annotation's discovery-order map only if the
        canonical resolver is unavailable.
        """
        try:
            label_id = getattr(label, 'id', None)

            class_id = None
            engine = getattr(self.mvat_manager, 'propagation_engine', None)
            if engine is not None and label_id is not None:
                try:
                    class_id = engine.canonical_class_id_for_label_id(label_id)
                except Exception:
                    class_id = None

            if class_id is None:
                # Legacy fallback: the mask annotation's discovery-order map.
                mask_annotation = (
                    self.mvat_manager.annotation_window.current_mask_annotation
                )
                if mask_annotation is not None:
                    class_id = mask_annotation.label_id_to_class_id_map.get(label_id)
                    if class_id is None:
                        mask_annotation.sync_label_map([label])
                        class_id = mask_annotation.label_id_to_class_id_map.get(label_id)

            if class_id is None:
                return None, None

            c = QColor(label.color)
            return class_id, (c.red(), c.green(), c.blue())

        except Exception:
            return None, None

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


    def _hide_preview_sphere(self):
        cursor = getattr(self.mvat_viewer, '_cursor_preview', None)
        if cursor is not None:
            cursor.set_visibility(False)

    def _update_preview_sphere(self, center: np.ndarray):
        """Move, resize, and recolor the shared cursor preview in-place.

        Delegates to MVATViewer._cursor_preview.update_transform() so that
        only one actor exists in the scene (no duplicate sphere when a tool
        is active).  Mirrors the 2D pattern where tools call
        annotation_window.update_cursor_preview() rather than owning an actor.
        """
        cursor = getattr(self.mvat_viewer, '_cursor_preview', None)
        if cursor is None:
            return
        try:
            cursor.update_transform(
                center=np.asarray(center, dtype=np.float64),
                radius=self.brush_size,
                shape=self.brush_shape,
                color_rgb_float=self._preview_color_rgb_float(),
                opacity=self._PREVIEW_OPACITY,
            )
        except Exception:
            pass

    def _remove_preview_sphere(self):
        """Return the shared cursor preview to its default viewer-owned state."""
        try:
            manager = getattr(self, 'mvat_manager', None)
            if manager is not None:
                manager.clear_sphere_hover_overlay(reset_context=True, render=False)
        except Exception:
            pass
        cursor = getattr(self.mvat_viewer, '_cursor_preview', None)
        if cursor is not None:
            cursor.set_visibility(False)
