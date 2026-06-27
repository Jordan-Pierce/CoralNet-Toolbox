"""
MVAT Viewer Widget

A PyVista-based 3D viewer for point clouds and camera frustums.
Customized interaction style:
- Left Double-Click: Set Focal Point (Pick)
- Left Drag: Rotate
- Right Drag: Pan (Strictly enforced via PyVista custom style)
- Scroll Wheel: Zoom
"""

import os
import time
import traceback
from time import perf_counter

import numpy as np

from pyvistaqt import QtInteractor
from PyQt5.QtCore import Qt, QEvent, QTimer, QObject, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QFrame, QVBoxLayout,
    QWidget, QHBoxLayout, QLabel, QSpinBox, QComboBox, QDoubleSpinBox,
    QToolBar, QToolButton, QMenu, QAction, QActionGroup, QStackedLayout,
    QVBoxLayout, QLabel, QHBoxLayout
)

from coralnet_toolbox.MVAT.core.Ray import CameraRay
from coralnet_toolbox.MVAT.core.Ray import BatchedRayManager
from coralnet_toolbox.MVAT.managers.CursorPreview3D import CursorPreview3D
from coralnet_toolbox.MVAT.core.Frustum import BatchedFrustumManager
from coralnet_toolbox.MVAT.core.Products import PointCloudProduct, MeshProduct, GaussianSplattingProduct
from coralnet_toolbox.MVAT.core.SceneContext import SceneContext
from coralnet_toolbox.MVAT.core.Products import AbstractSceneProduct
from coralnet_toolbox.MVAT.core.constants import RAY_COLOR_SELECTED
from coralnet_toolbox.MVAT.tools import BrushTool3D, EraseTool3D, FillTool3D, DropperTool3D
from coralnet_toolbox.MVAT.ui.QtCameraAnimator import CameraAnimator

from coralnet_toolbox.MVAT.utils.MVATLogger import get_visibility_logger

from coralnet_toolbox import theme as app_theme


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class _CameraInertiaController(QObject):
    """
    Post-interaction inertia layer for the MVAT camera.

    Handles two independent inertia channels:
      - Drag inertia  (rotate / pan): driven by VTK StartInteraction /
        Interaction / EndInteraction events bound to the active style.
      - Zoom inertia  (scroll wheel):  driven entirely at the Qt level via
        on_qt_wheel().  VTK never sees raw wheel events; this avoids the
        unreliable wheel routing that varies across PyVista / VTK versions
        and custom trackball styles.

    Usage
    -----
    1. Call bind(style) after enable_custom_trackball_style().
    2. Call on_qt_wheel(delta_y) from the Qt eventFilter for non-Ctrl wheel.
    3. Call stop() / unbind() on teardown.
    """

    # ── Drag inertia ──────────────────────────────────────────────────
    _DRAG_DECAY      = 0.84   # velocity retained per ~16 ms tick
    _MIN_POSITION    = 1e-4
    _MIN_FOCAL       = 1e-4
    _MIN_UP          = 1e-5
    _MIN_ANGLE       = 1e-4

    # ── Zoom inertia ──────────────────────────────────────────────────
    # Log-space step applied per standard wheel notch (Qt 120 units).
    # A value of ~0.12 ≈ ln(1.13) gives ≈13 % distance change per notch,
    # which is a comfortable default.
    _ZOOM_LOG_PER_NOTCH = 0.12
    _WHEEL_NOTCH_UNITS  = 120
    _ZOOM_DECAY          = 0.78   # slightly snappier than drag
    _MIN_ZOOM_VEL        = 5e-4

    def __init__(self, viewer):
        super().__init__(viewer)
        self.viewer = viewer

        # VTK observer bookkeeping
        self._style        = None
        self._observer_ids = []

        # Shared ~60 fps decay timer
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.setSingleShot(False)
        self._timer.timeout.connect(self._on_tick)

        # Drag state
        self._last_sample  = None
        self._drag_residual = None

        # Zoom state (log-space velocity; positive → zoom in)
        self._zoom_velocity = 0.0

    # ──────────────────────────────────────────────────────────────────
    # Binding / unbinding
    # ──────────────────────────────────────────────────────────────────

    def bind(self, style):
        """Attach to a VTK interaction style to capture drag events."""
        if style is self._style:
            return
        self.unbind()
        self._style = style
        if style is None:
            return

        # Only drag events — wheel is handled entirely at the Qt level.
        specs = [
            ("StartInteractionEvent", self._on_start,       -1.0),
            ("InteractionEvent",      self._on_interaction, -1.0),
            ("EndInteractionEvent",   self._on_end,         -1.0),
        ]
        for event_name, callback, priority in specs:
            try:
                oid = style.AddObserver(event_name, callback, priority)
            except TypeError:
                oid = style.AddObserver(event_name, callback)
            except Exception:
                continue
            if oid is not None:
                self._observer_ids.append(oid)

    def unbind(self):
        """Detach all VTK observers and stop the timer."""
        self.stop()
        if self._style is not None:
            for oid in self._observer_ids:
                try:
                    self._style.RemoveObserver(oid)
                except Exception:
                    pass
        self._observer_ids = []
        self._style = None

    # ──────────────────────────────────────────────────────────────────
    # Public control
    # ──────────────────────────────────────────────────────────────────

    def stop(self):
        """Halt all inertia immediately."""
        self._timer.stop()
        self._last_sample   = None
        self._drag_residual = None
        self._zoom_velocity = 0.0

    def on_qt_wheel(self, delta_y: int):
        """
        Entry point called from the Qt eventFilter on every non-Ctrl wheel event.

        Applies the zoom step immediately and seeds the zoom-inertia channel so
        the camera coasts to a stop after the user lifts their fingers.

        Args:
            delta_y: Qt angleDelta y-component.  Positive = scroll up = zoom in.
        """
        if delta_y == 0:
            return

        self._cancel_viewer_animation()

        # Stop drag inertia — the user has switched to zooming.
        self._drag_residual = None
        self._last_sample   = None
        # Do NOT stop the timer here; zoom inertia may already be running and
        # we want continued scrolls to accumulate naturally.

        # Convert to a log-space step.  Fractional notches (trackpads) are fine.
        notches   = delta_y / self._WHEEL_NOTCH_UNITS
        log_step  = notches * self._ZOOM_LOG_PER_NOTCH  # positive → zoom in

        # Apply immediately so the view responds on every tick of the wheel.
        self._apply_zoom_log(log_step)

        # Blend new step into the running velocity so rapid scrolling
        # accumulates while a single notch gives only gentle coasting.
        self._zoom_velocity = self._zoom_velocity * 0.45 + log_step * 0.35

        if abs(self._zoom_velocity) > self._MIN_ZOOM_VEL and not self._timer.isActive():
            self._timer.start()

    # ──────────────────────────────────────────────────────────────────
    # Shared helpers
    # ──────────────────────────────────────────────────────────────────

    def _cancel_viewer_animation(self):
        for name in ("_camera_animator", "_focal_point_animator"):
            animator = getattr(self.viewer, name, None)
            if animator is not None:
                try:
                    animator.cancel()
                except Exception:
                    pass

    def _normalize(self, vector, fallback):
        arr    = np.asarray(vector, dtype=float)
        length = float(np.linalg.norm(arr))
        return arr / length if length >= 1e-12 else np.asarray(fallback, dtype=float)

    def _trigger_render(self):
        try:
            self.viewer._update_clipping_range()
        except Exception:
            try:
                self.viewer.plotter.render()
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────
    # Zoom implementation
    # ──────────────────────────────────────────────────────────────────

    def _apply_zoom_log(self, log_step: float):
        """
        Apply a log-space zoom step to the camera.

        log_step > 0  →  zoom in  (camera moves toward focal point).
        log_step < 0  →  zoom out (camera moves away from focal point).

        Using log-space means equal notches produce equal *fractional* distance
        changes, giving perceptually uniform zoom at any depth.
        """
        if abs(log_step) < 1e-9:
            return
        try:
            cam         = self.viewer.plotter.camera
            is_parallel = bool(getattr(cam, 'GetParallelProjection', lambda: False)())

            if is_parallel:
                # Parallel projection: scale the ortho window instead.
                # log_step > 0 → zoom in → smaller scale.
                new_scale = float(cam.parallel_scale) * np.exp(-log_step)
                cam.parallel_scale = max(1e-6, new_scale)
            else:
                pos  = np.asarray(cam.position,    dtype=float)
                fp   = np.asarray(cam.focal_point, dtype=float)
                diff = pos - fp
                dist = float(np.linalg.norm(diff))
                if dist < 1e-8:
                    return
                # log_step > 0 → new_dist < dist  (zoom in)
                new_dist = dist * np.exp(-log_step)
                # Never let the camera pass through or clip the focal point.
                new_dist = max(1e-3, new_dist)
                cam.position = (fp + (diff / dist) * new_dist).tolist()

            self._trigger_render()
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────
    # Drag inertia implementation
    # ──────────────────────────────────────────────────────────────────

    def _snapshot(self):
        cam = self.viewer.plotter.camera
        return {
            "position":       np.asarray(cam.position,    dtype=float),
            "focal":          np.asarray(cam.focal_point, dtype=float),
            "up":             self._normalize(cam.up, [0.0, 0.0, 1.0]),
            "view_angle":     float(getattr(cam, 'view_angle',    30.0)),
            "parallel_scale": float(getattr(cam, 'parallel_scale', 1.0)),
        }

    def _diff(self, current, previous):
        return {k: current[k] - previous[k] for k in current}

    def _blend(self, base, update, alpha=0.35):
        return {k: base[k] * (1.0 - alpha) + update[k] * alpha for k in base}

    def _scale(self, delta, factor):
        return {k: delta[k] * factor for k in delta}

    def _has_drag_motion(self, delta):
        return (
            float(np.linalg.norm(delta["position"]))    > self._MIN_POSITION
            or float(np.linalg.norm(delta["focal"]))    > self._MIN_FOCAL
            or float(np.linalg.norm(delta["up"]))       > self._MIN_UP
            or abs(float(delta["view_angle"]))           > self._MIN_ANGLE
            or abs(float(delta["parallel_scale"]))       > self._MIN_ANGLE
        )

    def _capture_drag_delta(self):
        current = self._snapshot()
        if self._last_sample is None:
            self._last_sample = current
            return
        delta             = self._diff(current, self._last_sample)
        self._last_sample = current
        if not self._has_drag_motion(delta):
            return
        if self._drag_residual is None:
            self._drag_residual = delta
        else:
            self._drag_residual = self._blend(self._drag_residual, delta)

    def _apply_drag_delta(self, delta):
        cam  = self.viewer.plotter.camera
        pos  = np.asarray(cam.position,    dtype=float) + delta["position"]
        fp   = np.asarray(cam.focal_point, dtype=float) + delta["focal"]
        up   = self._normalize(np.asarray(cam.up, dtype=float) + delta["up"], [0.0, 0.0, 1.0])

        cam.position    = pos.tolist()
        cam.focal_point = fp.tolist()
        cam.up          = up.tolist()

        if bool(getattr(cam, 'GetParallelProjection', lambda: False)()):
            try:
                cam.parallel_scale = max(1e-6, float(cam.parallel_scale) + float(delta["parallel_scale"]))
            except Exception:
                pass
        else:
            try:
                cam.view_angle = float(np.clip(float(cam.view_angle) + float(delta["view_angle"]), 1.0, 175.0))
            except Exception:
                pass

        self._trigger_render()

    # ──────────────────────────────────────────────────────────────────
    # VTK drag event callbacks
    # ──────────────────────────────────────────────────────────────────

    def _on_start(self, *_args):
        """VTK drag started — cancel all running inertia and begin sampling."""
        self._timer.stop()
        self._cancel_viewer_animation()
        self._drag_residual = None
        self._zoom_velocity = 0.0
        self._last_sample   = self._snapshot()

    def _on_interaction(self, *_args):
        self._capture_drag_delta()

    def _on_end(self, *_args):
        self._capture_drag_delta()
        # Start the timer only if there is something to coast.
        has_drag = self._drag_residual is not None and self._has_drag_motion(self._drag_residual)
        has_zoom = abs(self._zoom_velocity) > self._MIN_ZOOM_VEL
        if has_drag or has_zoom:
            self._timer.start()
        else:
            self.stop()

    # ──────────────────────────────────────────────────────────────────
    # Shared decay tick
    # ──────────────────────────────────────────────────────────────────

    def _on_tick(self):
        has_drag = self._drag_residual is not None and self._has_drag_motion(self._drag_residual)
        has_zoom = abs(self._zoom_velocity) > self._MIN_ZOOM_VEL

        if not has_drag and not has_zoom:
            self.stop()
            return

        if has_drag:
            self._apply_drag_delta(self._drag_residual)
            self._drag_residual = self._scale(self._drag_residual, self._DRAG_DECAY)
            if not self._has_drag_motion(self._drag_residual):
                self._drag_residual = None

        if has_zoom:
            self._apply_zoom_log(self._zoom_velocity)
            self._zoom_velocity *= self._ZOOM_DECAY
            if abs(self._zoom_velocity) <= self._MIN_ZOOM_VEL:
                self._zoom_velocity = 0.0


class MVATViewer(QFrame):
    focalPointChanged = pyqtSignal(np.ndarray)  # Emits 3D point when focal point is set
    pointSizeChanged = pyqtSignal(int)
    computeIndexMapsToggled = pyqtSignal(bool)
    computeDepthMapsToggled = pyqtSignal(bool)
    primaryTargetChanged = pyqtSignal(str)      # Emits product_id when primary target changes

    def __init__(self, parent=None, point_size=1, show_rays=True):
        super().__init__(parent)
        self.mvat_manager = None
        self.setFrameShape(QFrame.NoFrame)
        self.setAcceptDrops(True)
        
        # Disable default context menu to prevent interference with right-click Pan
        self.setContextMenuPolicy(Qt.NoContextMenu)

        # Focus policy – important to receive key events
        self.setFocusPolicy(Qt.StrongFocus)

        # Layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Create PyVista QtInteractor
        self.plotter = QtInteractor(self, point_smoothing=False)

        # Optimizations TODO make configurable?
        self.plotter.set_background(app_theme.BACKGROUND_COLOR.name())
        self.plotter.disable_anti_aliasing()
        self.plotter.disable_eye_dome_lighting()
        self.plotter.disable_shadows()
        self.plotter.disable_depth_peeling()
        
        # Add observer for Left Click (for tool interaction / double-click detection)
        # This is added to the Interactor directly, so it persists
        self.plotter.interactor.AddObserver("LeftButtonPressEvent", self._on_left_press)
        self._last_click_time = 0
        self._last_right_click_time = 0

        # Scene context replaces single point_cloud with heterogeneous product collection
        self.scene_context = SceneContext()
        # Product actors keyed by product_id
        self._product_actors = {}
        # Translucent label-overlay actors (labels shown over a non-Labels base
        # array via the opacity slider), keyed by product_id.
        self._label_paint_actors = {}
        self.point_size = point_size
        self._splat_scale = 1.0          # Gaussian scale modifier (float, independent of point_size)
        self._gaussian_shading_idx = 7   # SH:0~3 default (index into RENDER_MODES)
        self._show_rays_enabled = show_rays
        self._ray_visible = True
        self._ray_manager = BatchedRayManager()
        self._ortho_ray_manager = BatchedRayManager()
        # Sphere actor for mouse tracking on mesh
        self._cursor_preview = CursorPreview3D(radius=0.1)
        self._sphere_visible = False
        self._brush_3d_tool = None
        self._erase_3d_tool = None
        self._fill_3d_tool = None
        self._dropper_3d_tool = None
        self._active_3d_tool = None
        self._mouse_sphere_observer_id = None
        self._sphere_hover_observer_bound = False
        self._sphere_hover_timer = QTimer(self)
        self._sphere_hover_timer.setSingleShot(True)
        self._sphere_hover_timer.setInterval(16)
        self._sphere_hover_timer.timeout.connect(self._process_sphere_hover_update)
        self._sphere_hover_pending_events = 0
        # Frustum and thumbnail management
        self._frustum_manager = BatchedFrustumManager()
        self._camera_animator = CameraAnimator(self.plotter, duration_ms=400)
        self._focal_point_animator = CameraAnimator(self.plotter, duration_ms=180)
        self._camera_inertia = _CameraInertiaController(self)
        self.thumbnail_actors = []
        self.thumbnail_opacity = 0.50
        self.frustum_scale = 0.1
        self._show_wireframes_enabled = True
        self._show_thumbnails_enabled = False
        
        # Scene product visibility by type
        self._show_point_clouds = True
        self._show_meshes = True
        self._show_gaussian_splats = True
        
        # Array selector widgets (created in create_top_toolbar)
        self.array_selector_combo = None

        # Top and bottom toolbar widgets (previously exposed for host composition)
        # Refactor: toolbars are owned by the viewer and inserted into its layout
        # to improve encapsulation while keeping attributes for backward compat.

        # Bottom toolbar widget
        self.bottom_toolbar_widget = QWidget()
        bottom_layout = QHBoxLayout(self.bottom_toolbar_widget)
        bottom_layout.setContentsMargins(6, 2, 6, 2)
        bottom_layout.setSpacing(12)
        
        # Use a stacked layout so we can show a centered placeholder when
        # no point cloud is loaded (matches pattern used in other viewers)
        self._stack_container = QWidget()
        self._stack = QStackedLayout(self._stack_container)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(self.plotter.interactor)

        # Placeholder shown when no scene products present
        self._placeholder_label = QLabel(
            "No 3D data loaded\nDrag a file here to load:\n• Point clouds (.pcd)\n• Meshes (.ply, .obj, .stl)"
        )
        self._placeholder_label.setStyleSheet(
            app_theme.scale_qss(
                f"color: {app_theme.TEXT_PRIMARY_COLOR.name()}; background-color: transparent; font-size: 14px; padding: 16px;"
            )
        )
        self._placeholder_label.setAlignment(Qt.AlignCenter)
        self._placeholder_label.setAutoFillBackground(True)
        self._placeholder_label.setWordWrap(True)
        self._stack.addWidget(self._placeholder_label)
        self._show_placeholder()  # Show placeholder initially
        self.layout.addWidget(self._stack_container)

        # Point size hint (spinbox removed; use Ctrl + mouse wheel)
        hint_label = QLabel("Ctrl + Mouse Wheel: resize the brush/erase preview sphere when those tools are active, otherwise point size")
        hint_label.setStyleSheet(f"color: {app_theme.TEXT_PRIMARY_COLOR.name()};")
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(hint_label)

        # Navigation constants
        self.move_speed = 0.01          # world units per key press
        self.rotate_speed = 0.25        # degrees per key press
        
        self.plotter.interactor.installEventFilter(self)
        QTimer.singleShot(100, self._configure_interaction)

        # Configure Interaction (Delayed)

    def refresh_scaling(self):
        """Refresh the placeholder styling after a UI scale change."""
        self._placeholder_label.setStyleSheet(
            app_theme.scale_qss(
                f"color: {app_theme.TEXT_PRIMARY_COLOR.name()}; background-color: transparent; font-size: 14px; padding: 16px;"
            )
        )
        # We run this on a timer to ensure it happens AFTER the parent window
        # has run its setup (like enable_point_picking), so we can override it.
        QTimer.singleShot(100, self._configure_interaction)
        
    # --------------------------------------------------------------------------
    # View Menu Actions
    # --------------------------------------------------------------------------
    
    def view_top(self):
        """Set camera to look down from the +Z-axis."""
        try:
            # preserve current zoom/distance when snapping to canonical axes
            # looking down from +Z: view direction (camera->focal) = [0,0,-1]
            self._set_view_preserve_zoom([0, 0, -1], up=[0, 1, 0])
        except Exception:
            print("Error setting top view: ", traceback.format_exc())

    def view_bottom(self):
        """Set camera to look up from the -Z-axis."""
        try:
            # looking up from -Z: view direction = [0,0,1]
            self._set_view_preserve_zoom([0, 0, 1], up=[0, 1, 0])
        except Exception:
            print("Error setting bottom view: ", traceback.format_exc())
            
    def view_front(self):
        """Set camera to look from the -Y-axis (Standard front)."""
        try:
            # front: camera at -Y looking towards +Y -> view direction = [0,1,0]
            self._set_view_preserve_zoom([0, 1, 0], up=[0, 0, 1])
        except Exception:
            print("Error setting front view: ", traceback.format_exc())

    def view_back(self):
        """Set camera to look from the +Y-axis."""
        try:
            # back: camera at +Y looking towards -Y -> view direction = [0,-1,0]
            self._set_view_preserve_zoom([0, -1, 0], up=[0, 0, 1])
        except Exception:
            print("Error setting back view: ", traceback.format_exc())

    def view_right(self):
        """Look at the YZ plane from the right (+X)."""
        # Z is UP, Y is LEFT
        # right: camera at +X looking towards -X -> view direction = [-1,0,0]
        try:
            self._set_view_preserve_zoom([-1, 0, 0], up=[0, 0, 1])
        except Exception:
            pass

    def view_left(self):
        """Look at the YZ plane from the left (-X)."""
        # Z is UP, Y is RIGHT
        # left: camera at -X looking towards +X -> view direction = [1,0,0]
        try:
            self._set_view_preserve_zoom([1, 0, 0], up=[0, 0, 1])
        except Exception:
            pass
        
    def view_isometric(self):
        try:
            self._cancel_camera_motion()
            self.plotter.view_isometric()
            self.plotter.render()
        except Exception:
            print("Error setting isometric view: ", traceback.format_exc())

    def _set_view_preserve_zoom(self, view_dir, up=None, animate=True):
        """Set camera looking along view_dir (from camera towards focal point)
        while preserving current camera distance (zoom) and view angle.

        view_dir: iterable-like 3-vector (direction from camera to focal point)
        up: optional up-vector to set on the camera
        """
        try:
            self._cancel_camera_motion()
            cam = self.plotter.camera
            fp = np.array(cam.focal_point)
            pos = np.array(cam.position)
            # preserve distance between camera and focal point; fallback to scene median
            dist = np.linalg.norm(pos - fp)
            if dist < 1e-6:
                dist = self.get_scene_median_depth(pos if pos is not None else np.array([0.0, 0.0, 0.0]))

            v = np.array(view_dir, dtype=float)
            n = np.linalg.norm(v)
            if n < 1e-12:
                return
            view_dir_norm = v / n

            # new camera position such that (fp - new_pos) is view_dir_norm * dist
            new_pos = fp - view_dir_norm * dist

            target_up = np.array(up, dtype=float) if up is not None else np.array(cam.up, dtype=float)

            if animate:
                animator = getattr(self, '_focal_point_animator', None)
                if animator is not None:
                    try:
                        animator.animate_to_camera_state(new_pos, fp, target_up, cam.view_angle)
                        QTimer.singleShot(int(getattr(animator, 'duration_ms', 180)) + 20, self._update_clipping_range)
                        return
                    except Exception:
                        pass

            cam.position = new_pos.tolist()
            cam.focal_point = fp.tolist()
            if up is not None:
                cam.up = target_up.tolist()
            # keep view_angle unchanged -> do not call reset/fit helpers
            try:
                self.plotter.render()
            except Exception:
                pass
            self._update_clipping_range()
        except Exception:
            pass

    def toggle_orthographic(self, state: bool):
        self._cancel_camera_motion()
        if state:
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()
        self.plotter.render()
        # Keep menu action in sync if present
        try:
            if hasattr(self, '_action_ortho') and self._action_ortho is not None:
                # Avoid re-triggering signals
                self._action_ortho.blockSignals(True)
                self._action_ortho.setChecked(bool(state))
                self._action_ortho.blockSignals(False)
        except Exception:
            pass

    def set_ortho_top_view(self):
        """Fit the scene and snap to the canonical top perspective view."""
        try:
            self.toggle_orthographic(False)
            self.fit_to_view()
            self.view_top()
        except Exception:
            print("Error setting ortho selection view: ", traceback.format_exc())
        
    # --------------------------------------------------------------------------
    # DockWrapper Hooks
    # --------------------------------------------------------------------------
    
    def create_top_toolbar(self) -> QToolBar:
        """Create the top toolbar with array selector and Gaussian splat controls."""
        toolbar = QToolBar("3D Viewer Tools")
        toolbar.setMovable(False)

        # Array selection dropdown
        self.array_selector_combo = QComboBox()
        self.array_selector_combo.addItem("Labels")
        self.array_selector_combo.setMinimumWidth(150)
        self.array_selector_combo.currentTextChanged.connect(self._on_array_selected)
        toolbar.addWidget(self.array_selector_combo)

        toolbar.addSeparator()

        # Gaussian splat shading mode (disabled until a Gaussian product is loaded)
        self._gaussian_shading_combo = QComboBox()
        self._gaussian_shading_combo.addItems([
            "Gaussian Ball", "Flat Ball", "Billboard",
            "Depth", "SH:0", "SH:0~1", "SH:0~2", "SH:0~3",
        ])
        self._gaussian_shading_combo.setCurrentIndex(self._gaussian_shading_idx)
        self._gaussian_shading_combo.setMinimumWidth(110)
        self._gaussian_shading_combo.setToolTip(
            "Gaussian splat shading / visualisation mode"
        )
        self._gaussian_shading_combo.setEnabled(False)
        self._gaussian_shading_combo.currentIndexChanged.connect(
            self._on_gaussian_shading_changed
        )
        toolbar.addWidget(self._gaussian_shading_combo)

        return toolbar

    def create_view_menu(self) -> QMenu:
        """Create a standalone QMenu for the viewer that can be attached to a menubar or dock."""
        view_menu = QMenu("3D Viewer")

        # Top actions: Fit and Reset
        action_fit = QAction("Fit All", self)
        action_fit.triggered.connect(self.fit_to_view)
        view_menu.addAction(action_fit)

        action_reset = QAction("Reset View", self)
        action_reset.triggered.connect(self.reset_view)
        view_menu.addAction(action_reset)

        view_menu.addSeparator()

        # Camera angles
        view_menu.addAction("Top (T)", self.view_top)
        view_menu.addAction("Bottom (C)", self.view_bottom)
        view_menu.addAction("Front (F)", self.view_front)
        view_menu.addAction("Back (B)", self.view_back)
        view_menu.addAction("Left (L)", self.view_left)
        view_menu.addAction("Right (R)", self.view_right)
        view_menu.addAction("Isometric (I)", self.view_isometric)

        view_menu.addSeparator()

        # Orthographic projection
        action_ortho = QAction("Orthographic (O)", self)
        action_ortho.setCheckable(True)
        action_ortho.toggled.connect(self.toggle_orthographic)
        self._action_ortho = action_ortho
        view_menu.addAction(action_ortho)

        view_menu.addSeparator()

        # Visibility toggles
        action_rays = QAction("Show Rays", self)
        action_rays.setCheckable(True)
        action_rays.setChecked(self._show_rays_enabled)
        action_rays.toggled.connect(self.set_ray_visible)
        view_menu.addAction(action_rays)

        action_wireframes = QAction("Show Wireframes", self)
        action_wireframes.setCheckable(True)
        action_wireframes.setChecked(self._show_wireframes_enabled)
        action_wireframes.toggled.connect(self.enable_wireframes)
        view_menu.addAction(action_wireframes)

        action_thumbnails = QAction("Show Thumbnails", self)
        action_thumbnails.setCheckable(True)
        action_thumbnails.setChecked(self._show_thumbnails_enabled)
        action_thumbnails.toggled.connect(self.enable_thumbnails)
        view_menu.addAction(action_thumbnails)

        view_menu.addSeparator()
        
        # Scene Products submenu
        products_menu = view_menu.addMenu("Scene Products")
        
        action_point_clouds = QAction("Show Point Clouds", self)
        action_point_clouds.setCheckable(True)
        action_point_clouds.setChecked(self._show_point_clouds)
        action_point_clouds.setToolTip("Toggle visibility of point cloud products")
        action_point_clouds.toggled.connect(self.set_point_clouds_visible)
        products_menu.addAction(action_point_clouds)
        
        action_meshes = QAction("Show Meshes", self)
        action_meshes.setCheckable(True)
        action_meshes.setChecked(self._show_meshes)
        action_meshes.setToolTip("Toggle visibility of mesh products")
        action_meshes.toggled.connect(self.set_meshes_visible)
        products_menu.addAction(action_meshes)

        action_gaussian_splats = QAction("Show Gaussian Splats", self)
        action_gaussian_splats.setCheckable(True)
        action_gaussian_splats.setChecked(self._show_gaussian_splats)
        action_gaussian_splats.setToolTip("Toggle visibility of 3D Gaussian Splatting products")
        action_gaussian_splats.toggled.connect(self.set_gaussian_splats_visible)
        products_menu.addAction(action_gaussian_splats)

        products_menu.addSeparator()

        action_show_all = QAction("Show All Products", self)
        action_show_all.triggered.connect(self.show_all_products)
        products_menu.addAction(action_show_all)

        action_hide_all = QAction("Hide All Products", self)
        action_hide_all.triggered.connect(self.hide_all_products)
        products_menu.addAction(action_hide_all)

        # Store actions for programmatic updates
        self._product_visibility_actions = {
            'point_clouds': action_point_clouds,
            'meshes': action_meshes,
            'gaussian_splats': action_gaussian_splats,
        }
        
        # Primary Target submenu - for selecting annotation target
        self._primary_target_menu = view_menu.addMenu("Primary Target")
        self._primary_target_menu.setToolTip(
            "Select which product's elements are indexed for annotations"
        )
        self._primary_target_action_group = QActionGroup(self)
        self._primary_target_action_group.setExclusive(True)
        self._update_primary_target_menu()  # Populate initially

        view_menu.addSeparator()

        # Background computation toggles
        action_index_maps = QAction("Compute Index Maps", self)
        action_index_maps.setCheckable(True)
        action_index_maps.setChecked(True)
        action_index_maps.setToolTip("Toggle background computation of visibility index maps")
        action_index_maps.toggled.connect(self.computeIndexMapsToggled.emit)
        view_menu.addAction(action_index_maps)

        # Settings
        action_depth = QAction("Compute Depth Maps", self)
        action_depth.setCheckable(True)
        action_depth.setChecked(True)
        action_depth.setToolTip("Toggle computing depth maps during visibility computation")
        action_depth.toggled.connect(self.computeDepthMapsToggled.emit)
        view_menu.addAction(action_depth)

        return view_menu

    def create_bottom_toolbar(self) -> QToolBar:
        """Create the bottom toolbar for opacity and point size."""
        toolbar = QToolBar("3D Display Settings")
        toolbar.setMovable(False)
        
        # Simply mount your existing bottom widget into the toolbar!
        toolbar.addWidget(self.bottom_toolbar_widget)
        
        return toolbar

    def initialize_3d_tools(self, mvat_manager):
        """Create the preview-only 3D brush, erase, fill, dropper, and feature select tools once the manager exists."""
        if self._brush_3d_tool is not None or self._erase_3d_tool is not None:
            return

        self._brush_3d_tool = BrushTool3D(self, mvat_manager)
        self._erase_3d_tool = EraseTool3D(self, mvat_manager)
        self._fill_3d_tool = FillTool3D(self, mvat_manager)
        self._dropper_3d_tool = DropperTool3D(self, mvat_manager)
        self._feature_3d_tool = None
        try:
            from coralnet_toolbox.MVAT.tools.FeatureSelectTool3D import FeatureSelectTool3D
            self._feature_3d_tool = FeatureSelectTool3D(self, mvat_manager)
        except Exception as e:
            print(f"Warning: Failed to initialize FeatureSelectTool3D: {e}")
        self._active_3d_tool = None

    def get_selected_3d_tool(self):
        return self._active_3d_tool

    def set_selected_3d_tool(self, tool_name):
        """Activate the BrushTool3D, EraseTool3D, FillTool3D, DropperTool3D, or FeatureSelectTool3D preview, or clear it."""
        tool_map = {
            'brush': self._brush_3d_tool,
            'erase': self._erase_3d_tool,
            'fill': getattr(self, '_fill_3d_tool', None),
            'dropper': getattr(self, '_dropper_3d_tool', None),
            'feature': getattr(self, '_feature_3d_tool', None),
        }
        next_tool = tool_map.get(tool_name)
        current_tool = self._active_3d_tool

        if current_tool is next_tool:
            if next_tool is not None:
                self._sphere_visible = True
                self._request_sphere_hover_refresh()
                try:
                    self._process_sphere_hover_update()
                except Exception:
                    pass
            else:
                self._restore_viewer_navigation_mode()
                sphere_manager = getattr(self, '_cursor_preview', None)
                if sphere_manager is not None:
                    try:
                        setter = getattr(sphere_manager, 'set_shape', None)
                        if callable(setter):
                            setter('circle')
                        else:
                            sphere_manager.shape = 'circle'
                    except Exception:
                        pass
            return

        if current_tool is not None:
            try:
                current_tool.deactivate()
            except Exception:
                pass

        self._active_3d_tool = next_tool

        if next_tool is not None:
            self._sphere_visible = True
            try:
                next_tool.activate()
            except Exception:
                pass
            self._sync_sphere_hover_binding()
            self._request_sphere_hover_refresh()
        else:
            sphere_manager = getattr(self, '_cursor_preview', None)
            if sphere_manager is not None:
                try:
                    setter = getattr(sphere_manager, 'set_shape', None)
                    if callable(setter):
                        setter('circle')
                    else:
                        sphere_manager.shape = 'circle'
                except Exception:
                    pass
            self._restore_viewer_navigation_mode()

        try:
            self.plotter.render()
        except Exception:
            pass

    def _restore_viewer_navigation_mode(self):
        """Return the MVAT viewer to the normal rotate / focal-point / pan interaction path."""
        self._sphere_visible = False

        if hasattr(self, '_sphere_hover_timer'):
            try:
                self._sphere_hover_timer.stop()
                self._sphere_hover_pending_events = 0
            except Exception:
                pass

        self._last_click_time = 0
        self._sync_sphere_hover_binding()

        if self._cursor_preview is not None:
            try:
                self._cursor_preview.set_visibility(False)
            except Exception:
                pass

        manager = getattr(self, 'mvat_manager', None)
        if manager is not None:
            try:
                manager.clear_sphere_hover_overlay(reset_context=True, render=False)
            except Exception:
                pass

        try:
            self.plotter.render()
        except Exception:
            pass

    # --------------------------------------------------------------------------
    # Custom Interaction Logic
    # --------------------------------------------------------------------------

    def _get_vtk_interaction_style(self, interactor):
        """Return the active VTK interaction style, or the interactor itself."""
        if interactor is None:
            return None

        try:
            style = interactor.GetInteractorStyle()
        except Exception:
            style = None

        return style if style is not None else interactor
    
    def _configure_interaction(self):
        """
        Configures the interaction style using PyVista's custom trackball API.
        """
        interactor = self.plotter.interactor
        if not interactor:
            return

        # 1. Clean Interactor Observers (CRITICAL)
        # We MUST remove these specific observers because PyVista's enable_point_picking 
        # attaches them directly to the interactor, bypassing the style mechanism.
        # If we don't remove them, single right-click will still trigger 'Pick'.
        interactor.RemoveObservers("RightButtonPressEvent")
        interactor.RemoveObservers("RightButtonReleaseEvent")
        
        # 2. Apply Custom Trackball Style. Left=rotate is the only binding we
        # really want VTK to manage; right=pan is set here for backward compat
        # but its observers are stripped immediately below so VTK never moves
        # the camera from a right-drag. Our Qt-level _apply_right_pan_delta
        # (see eventFilter) is the sole driver of right-button pan — that path
        # is decoupled from the camera-inertia controller, so a tiny right-drag
        # no longer triggers VTK's pan + the inertia coast that together
        # propelled the camera after the user released the button.
        self.plotter.enable_custom_trackball_style(
            left='rotate',
            control_left='rotate',
            right='pan',
            control_right='pan',
        )

        inertia_style = self._get_vtk_interaction_style(interactor)
        if hasattr(self, '_camera_inertia') and self._camera_inertia is not None:
            self._camera_inertia.bind(inertia_style)

        # Strip the right-button handlers that enable_custom_trackball_style
        # just re-installed. This removes the VTK-side pan and, because the
        # inertia controller listens for StartInteraction / InteractionEvent
        # on the style itself, also prevents inertia coasting from a
        # right-drag. Left-button rotate + inertia is untouched.
        try:
            interactor.RemoveObservers("RightButtonPressEvent")
            interactor.RemoveObservers("RightButtonReleaseEvent")
        except Exception:
            pass
        if inertia_style is not None and inertia_style is not interactor:
            try:
                inertia_style.RemoveObservers("RightButtonPressEvent")
                inertia_style.RemoveObservers("RightButtonReleaseEvent")
            except Exception:
                pass

        # Rebind the right-button observer after stripping the style-level
        # handlers so double-right-click focal-point picks still work.
        try:
            self.plotter.interactor.AddObserver("RightButtonPressEvent", self._on_right_press)
        except Exception:
            pass

        # 2a. Qt-driven right-button pan (see eventFilter). Track state here.
        self._right_pan_active = False
        self._right_pan_last_xy = None

        # 3. Add sphere actor to plotter and bind mouse move event
        if hasattr(self, '_cursor_preview') and self._cursor_preview is not None:
            # Add sphere actor (it's created empty initially)
            from coralnet_toolbox.MVAT.core.constants import SELECT_COLOR_RGB
            self._cursor_preview.add_to_plotter(self.plotter, color=SELECT_COLOR_RGB, line_width=1.5)
            self._cursor_preview.set_visibility(False)  # Hidden by default
            self._sync_sphere_hover_binding()

    def _scene_has_opaque_geometry(self) -> bool:
        """True if the scene has depth-writing geometry (point clouds / meshes).

        Splat-only scenes never populate the Z-buffer, so the neighborhood depth
        probe would scan an empty buffer there — skip it in that case.
        """
        try:
            return bool(self.scene_context.get_products_by_class(PointCloudProduct)
                        or self.scene_context.get_products_by_class(MeshProduct))
        except Exception:
            return False

    def _probe_depth_neighborhood(self, cx: int, cy: int, radius: int = 4):
        """Find the nearest pixel to (cx, cy) whose Z-buffer holds opaque geometry.

        Reads a small window of the depth buffer in one call and returns the
        display coords of the closest valid (non-empty) pixel, or None. Lets a
        point-cloud hover that lands in a gap between points snap to the nearest
        rendered point instead of dropping the pick.
        """
        try:
            import vtk
            try:
                from vtkmodules.util.numpy_support import vtk_to_numpy
            except Exception:
                from vtk.util.numpy_support import vtk_to_numpy
        except Exception:
            return None

        renderer = self.plotter.renderer
        win = renderer.GetRenderWindow()
        size = renderer.GetSize()
        w, h = max(int(size[0]), 1), max(int(size[1]), 1)
        x0, y0 = max(0, cx - radius), max(0, cy - radius)
        x1, y1 = min(w - 1, cx + radius), min(h - 1, cy + radius)
        if x1 < x0 or y1 < y0:
            return None
        try:
            zarr = vtk.vtkFloatArray()
            win.GetZbufferData(x0, y0, x1, y1, zarr)
            z = vtk_to_numpy(zarr)
        except Exception:
            return None

        nx, ny = (x1 - x0 + 1), (y1 - y0 + 1)
        if z is None or z.size != nx * ny:
            return None
        z = z.reshape(ny, nx)
        valid = z < 0.9999
        if not np.any(valid):
            return None
        ys, xs = np.nonzero(valid)
        abs_x = x0 + xs
        abs_y = y0 + ys
        d2 = (abs_x - cx) ** 2 + (abs_y - cy) ** 2
        k = int(np.argmin(d2))
        return int(abs_x[k]), int(abs_y[k])

    def _fast_hardware_pick(self, hover=False):
        import vtk

        # 1. Get raw event position
        pos = self.plotter.interactor.GetEventPosition()
        vtk_x, vtk_y = int(pos[0]), int(pos[1])

        # Z-buffer pick for opaque geometry (meshes, point clouds). A surface in
        # front of splats correctly wins here because it writes real depth.
        z_val = self.plotter.renderer.GetZ(vtk_x, vtk_y)

        if z_val is not None and not np.isclose(z_val, 1.0):
            picker = vtk.vtkWorldPointPicker()
            picker.Pick(vtk_x, vtk_y, 0, self.plotter.renderer)
            return np.array(picker.GetPickPosition())

        # Center pixel missed. For point clouds the cursor often lands in a
        # screen-space gap between points (worse when zoomed in), dropping the
        # pick and making the hover sphere jump. Snap to the nearest opaque pixel
        # in a small neighborhood. Only worth it when depth-writing geometry
        # exists (splats never populate the Z-buffer).
        if self._scene_has_opaque_geometry():
            probe = self._probe_depth_neighborhood(vtk_x, vtk_y)
            if probe is not None:
                picker = vtk.vtkWorldPointPicker()
                picker.Pick(probe[0], probe[1], 0, self.plotter.renderer)
                return np.array(picker.GetPickPosition())

        # Z-buffer miss: Gaussian splats disable depth writes so they never
        # populate the Z-buffer.
        from coralnet_toolbox.MVAT.core.Products import GaussianSplattingProduct
        gs_products = self.scene_context.get_products_by_class(GaussianSplattingProduct)
        if not gs_products:
            return None

        renderer = self.plotter.renderer

        renderer.SetDisplayPoint(vtk_x, vtk_y, 0.0)
        renderer.DisplayToWorld()
        wp = renderer.GetWorldPoint()
        w0 = wp[3] if wp[3] != 0 else 1.0
        ray_origin = np.array([wp[0] / w0, wp[1] / w0, wp[2] / w0], dtype=np.float64)

        renderer.SetDisplayPoint(vtk_x, vtk_y, 1.0)
        renderer.DisplayToWorld()
        wp = renderer.GetWorldPoint()
        w1 = wp[3] if wp[3] != 0 else 1.0
        ray_target = np.array([wp[0] / w1, wp[1] / w1, wp[2] / w1], dtype=np.float64)

        ray_dir = ray_target - ray_origin
        norm = np.linalg.norm(ray_dir)
        if norm < 1e-12:
            return None
        ray_dir /= norm

        # During hover (mouse-move), pick_gaussian uses a pre-built random subsample
        # of Gaussian centres (fast=True) so it runs in O(K) rather than O(N).
        # The same algorithm and tolerance are used for both hover and click, so
        # the sphere preview and the actual paint land on the same Gaussian.
        fovy_rad = np.radians(self.plotter.camera.view_angle)
        _, window_height = self.plotter.window_size

        # Size the pick gather to the brush sphere (world radius). This makes the
        # gather zoom-invariant (stable hover) while keeping the picked point inside
        # the region the brush paints, so paint coverage stays full. None when the
        # brush sphere isn't active (the pick falls back to the bare angular cone).
        min_radius = None
        try:
            cp = getattr(self, '_cursor_preview', None)
            if cp is not None and getattr(self, '_sphere_visible', False):
                r = float(getattr(cp, 'radius', 0.0))
                if r > 0.0:
                    min_radius = r
        except Exception:
            min_radius = None

        for gs_prod in gs_products:
            actor = self._product_actors.get(gs_prod.product_id)
            if actor is not None and not actor.GetVisibility():
                continue
            hit_pos = gs_prod.gaussian_actor.pick_gaussian(
                ray_origin, ray_dir, fovy_rad, window_height, fast=hover,
                min_world_radius=min_radius,
            )
            if hit_pos is not None:
                return hit_pos

        return None

    def _bind_sphere_hover_observer(self):
        """Bind the sphere hover observer if sphere tracking is enabled."""
        if self._sphere_hover_observer_bound:
            return

        interactor = self.plotter.interactor
        style = self._get_vtk_interaction_style(interactor)
        if style is None:
            return

        try:
            self._mouse_sphere_observer_id = style.AddObserver("MouseMoveEvent", self._on_mouse_move)
            self._sphere_hover_observer_bound = self._mouse_sphere_observer_id is not None
        except Exception as e:
            pass

    def _unbind_sphere_hover_observer(self):
        """Unbind the sphere hover observer so only camera interactions remain."""
        if not self._sphere_hover_observer_bound:
            return

        try:
            style = self._get_vtk_interaction_style(getattr(self.plotter, 'interactor', None))
            if style is not None and self._mouse_sphere_observer_id is not None:
                style.RemoveObserver(self._mouse_sphere_observer_id)
        except Exception:
            pass

        self._mouse_sphere_observer_id = None
        self._sphere_hover_observer_bound = False

    def _sync_sphere_hover_binding(self):
        """Keep the sphere hover observer aligned with the feature toggle."""
        if self._sphere_visible:
            self._bind_sphere_hover_observer()
        else:
            self._unbind_sphere_hover_observer()

    def _request_sphere_hover_refresh(self):
        """Queue a hover refresh for the current mouse position."""
        if not self._sphere_visible or self._cursor_preview is None:
            return

        try:
            self.plotter.store_mouse_position()
        except Exception:
            pass

        self._sphere_hover_pending_events = max(1, self._sphere_hover_pending_events)
        if not self._sphere_hover_timer.isActive():
            self._sphere_hover_timer.start()

    def _adjust_sphere_size_from_wheel(self, delta_y: int):
        """Scale the sphere radius from Ctrl+wheel input."""
        if self._cursor_preview is None:
            return

        current_radius = float(getattr(self._cursor_preview, 'radius', 0.1))
        wheel_step = float(delta_y) / 120.0
        scale_factor = float(np.exp(wheel_step * 0.12))
        new_radius = float(np.clip(current_radius * scale_factor, 0.01, 10.0))

        if np.isclose(current_radius, new_radius):
            return

        self._cursor_preview.set_radius(new_radius)
        manager = getattr(self, 'mvat_manager', None)
        if manager is not None:
            try:
                manager.refresh_sphere_hover_overlay(render=False)
            except Exception:
                pass

        try:
            self.plotter.render()
        except Exception:
            pass

    def is_sphere_tracking_enabled(self) -> bool:
        """Return whether sphere hover tracking is currently enabled."""
        return bool(self._sphere_visible)

    def eventFilter(self, obj, event):
        """Intercept key press events."""
        # Swallow ContextMenu events coming from the plotter interactor so the
        # default Qt context menu does not appear on single right-click.
        if event.type() == QEvent.ContextMenu:
            return True

        # ---- Right-button drag → manual camera pan -----------------
        # Qt mouse events are reliable in a way VTK's interactor observers
        # aren't when a 3D tool has captured the interaction style, so we
        # drive pan from here regardless of tool state.
        etype = event.type()
        if etype == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
            try:
                pos = event.pos()
                self._right_pan_last_xy = (int(pos.x()), int(pos.y()))
                self._right_pan_active = True
            except Exception:
                self._right_pan_active = False
                self._right_pan_last_xy = None
            # Don't consume — VTK may also want to react, and not consuming
            # keeps the trackball style's right=pan working as a backup.
        elif etype == QEvent.MouseMove and self._right_pan_active:
            try:
                pos = event.pos()
                self._apply_right_pan_delta(int(pos.x()), int(pos.y()))
            except Exception:
                pass
        elif etype in (QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick) and event.button() == Qt.RightButton:
            self._right_pan_active = False
            self._right_pan_last_xy = None

        if event.type() == QEvent.Leave:
            active_tool = getattr(self, '_active_3d_tool', None)
            if active_tool is not None:
                try:
                    active_tool.mouseMoveEvent(None, -1, None)
                except Exception:
                    pass
            else:
                manager = getattr(self, 'mvat_manager', None)
                if manager is not None:
                    try:
                        manager.clear_sphere_hover_overlay(reset_context=True, render=False)
                    except Exception:
                        pass
            manager = getattr(self, 'mvat_manager', None)
            if manager is not None:
                try:
                    manager.clear_all_markers()
                except Exception:
                    pass
            try:
                self.clear_ray()
            except Exception:
                pass
            try:
                self.clear_ortho_ray()
            except Exception:
                pass
            return False

        if event.type() == QEvent.Wheel:
            delta_y = event.angleDelta().y()

            # Ctrl + wheel → adjust sphere size when sphere tracking is enabled.
            if event.modifiers() & Qt.ControlModifier:
                if delta_y != 0:
                    active_tool = getattr(self, '_active_3d_tool', None)
                    if active_tool is not None:
                        try:
                            active_tool.wheelEvent(event, delta_y)
                        except Exception:
                            pass
                    elif self.is_sphere_tracking_enabled() and self._cursor_preview is not None:
                        self._adjust_sphere_size_from_wheel(delta_y)
                    else:
                        step = 1 if delta_y > 0 else -1
                        new_size = max(1, min(20, self.point_size + step))
                        if new_size != self.point_size:
                            self.set_point_size(new_size)
                            try:
                                self.pointSizeChanged.emit(new_size)
                            except Exception:
                                pass
                        # Also adjust Gaussian splat scale with float precision
                        try:
                            from coralnet_toolbox.MVAT.core.Products import GaussianSplattingProduct
                            if self.scene_context.get_products_by_class(GaussianSplattingProduct):
                                step_f = 0.1 if delta_y > 0 else -0.1
                                new_scale = round(max(0.01, min(10.0, self._splat_scale + step_f)), 2)
                                if new_scale != self._splat_scale:
                                    self.set_splat_scale(new_scale)
                        except Exception:
                            pass
                return True  # consumed

            # Regular wheel → zoom via inertia controller.
            # We always consume the event so VTK's default wheel handling
            # (which varies across versions / custom styles) never runs.
            if delta_y != 0:
                controller = getattr(self, '_camera_inertia', None)
                if controller is not None:
                    try:
                        controller.on_qt_wheel(delta_y)
                    except Exception:
                        pass
            return True  # consumed — VTK must not see raw wheel events

        # Forward key presses to keyPressEvent once; if handled, consume the event
        if event.type() == QEvent.KeyPress:
            self.keyPressEvent(event)
            if event.isAccepted():
                return True

        return super().eventFilter(obj, event)

    def _on_left_press(self, obj, event):
        """Handle Left Click to detect tool presses and legacy double-clicks."""
        active_tool = getattr(self, '_active_3d_tool', None)
        try:
            if active_tool is not None and not bool(getattr(active_tool, 'preview_only', True)):
                try:
                    world_pos = self._fast_hardware_pick()
                    active_tool.mousePressEvent(event, 1, world_pos)
                except Exception:
                    pass
                return

            if self.is_sphere_tracking_enabled():
                self._last_click_time = time.time() * 1000
                return

            current_time = time.time() * 1000
            dc_interval = QApplication.doubleClickInterval()
            if (current_time - self._last_click_time) < dc_interval:
                self._handle_double_click()

            self._last_click_time = current_time
        except Exception:
            pass

    def _on_right_press(self, obj, event):
        """Handle Right Click: forward to opt-in tools, else detect double-right focal pick."""
        # Forward right-button presses only to tools that opt in (e.g. the
        # FeatureSelectTool3D's Ctrl+right = negative). Other tools keep the
        # normal right-button pan / double-click focal-point behavior.
        active_tool = getattr(self, '_active_3d_tool', None)
        if (active_tool is not None
                and getattr(active_tool, 'wants_right_button', False)
                and not bool(getattr(active_tool, 'preview_only', True))):
            ctrl_held = False
            try:
                ctrl_held = bool(obj.GetControlKey())
            except Exception:
                ctrl_held = False
            try:
                world_pos = self._fast_hardware_pick()
                active_tool.mousePressEvent(event, 1, world_pos)
            except Exception:
                pass
            # A Ctrl+right is a query action (negative prototype) — don't let it
            # also drive the double-right-click focal-point pick.
            if ctrl_held:
                return

        try:
            current_time = time.time() * 1000
            dc_interval = QApplication.doubleClickInterval()
            if (current_time - self._last_right_click_time) < dc_interval:
                self._handle_double_click()

            self._last_right_click_time = current_time
        except Exception:
            pass

    def _apply_right_pan_delta(self, x: int, y: int):
        """Pan the camera based on the latest Qt mouse-move position.

        Coordinates are in Qt widget space (origin top-left). VTK's display
        coords have origin bottom-left, so dy is negated before being fed
        into the WorldToDisplay/DisplayToWorld round-trip.
        """
        last = self._right_pan_last_xy
        if last is None:
            self._right_pan_last_xy = (x, y)
            return

        dx = x - last[0]
        dy = y - last[1]
        self._right_pan_last_xy = (x, y)
        if dx == 0 and dy == 0:
            return

        # Qt top-left vs VTK bottom-left
        dy_vtk = -dy

        renderer = self.plotter.renderer
        cam = self.plotter.camera

        fp = np.asarray(cam.focal_point, dtype=np.float64)
        renderer.SetWorldPoint(fp[0], fp[1], fp[2], 1.0)
        renderer.WorldToDisplay()
        disp = renderer.GetDisplayPoint()
        focal_display = (disp[0] - dx, disp[1] - dy_vtk, disp[2])
        renderer.SetDisplayPoint(*focal_display)
        renderer.DisplayToWorld()
        new_fp_h = renderer.GetWorldPoint()
        w = new_fp_h[3] if new_fp_h[3] != 0 else 1.0
        new_fp = np.array([new_fp_h[0] / w, new_fp_h[1] / w, new_fp_h[2] / w])

        shift = new_fp - fp
        pos = np.asarray(cam.position, dtype=np.float64) + shift
        cam.position = tuple(pos)
        cam.focal_point = tuple(new_fp)

        try:
            self.plotter.render()
        except Exception:
            pass

    def _handle_double_click(self):
        """
        Perform a pick against the scene geometry and set the focal point to
        the picked world coordinate.

        Silently does nothing if no geometry is under the cursor (background click).
        """
        try:
            self._right_pan_active = False
            self._right_pan_last_xy = None
            # Stop any running inertia/animation so the animated focal-point
            # transition isn't immediately clobbered by the decay timer.
            self._cancel_camera_motion()

            picked = self._fast_hardware_pick()
            if picked is None:
                return

            self.set_focal_point(np.asarray(picked, dtype=float))
        except Exception:
            pass

    def _get_primary_target_actor(self):
        """Return the currently active actor for the primary mesh target, if any."""
        try:
            target = self.scene_context.get_primary_target()
            if target is None:
                return None
            return self._product_actors.get(getattr(target, 'product_id', None))
        except Exception:
            return None

    def _is_valid_scene_pick(self, picked, expected_actor=None) -> bool:
        """Return True only when PyVista reports a hit on the expected scene geometry."""
        if picked is None:
            return False

        try:
            picker = getattr(getattr(self.plotter, 'iren', None), 'picker', None)
            if picker is not None and hasattr(picker, 'GetDataSet'):
                try:
                    if picker.GetDataSet() is None:
                        return False
                except Exception:
                    pass
            if expected_actor is not None and picker is not None and hasattr(picker, 'GetActor'):
                try:
                    picked_actor = picker.GetActor()
                    if picked_actor is None or picked_actor is not expected_actor:
                        return False
                except Exception:
                    return False
        except Exception:
            pass

        try:
            picked = np.asarray(picked, dtype=float)
            if not np.all(np.isfinite(picked)):
                return False

            # Secondary sanity check: reject pathological values far outside the scene.
            cam_pos = np.asarray(self.plotter.camera.position, dtype=float)
            cam_fp = np.asarray(self.plotter.camera.focal_point, dtype=float)
            cam_dist = float(np.linalg.norm(cam_pos - cam_fp))
            pick_dist_from_cam = float(np.linalg.norm(picked - cam_pos))
            if cam_dist > 1e-4 and pick_dist_from_cam > cam_dist * 50.0:
                return False
        except Exception:
            pass

        return True

    def _process_sphere_hover_update(self):
        """Process the most recent queued mouse-move batch for the sphere actor."""
        start_time = perf_counter()
        try:
            pending_events = self._sphere_hover_pending_events
            self._sphere_hover_pending_events = 0

            if pending_events <= 0:
                return

            active_tool = getattr(self, '_active_3d_tool', None)

            # --- INSTANT HARDWARE PICK ---
            # hover=True: focal-plane projection for O(1) Gaussian miss path
            world_pos = self._fast_hardware_pick(hover=True)

            if active_tool is not None:
                if self._cursor_preview is None or not self._sphere_visible:
                    try:
                        active_tool.mouseMoveEvent(None, -1, None)
                    except Exception:
                        pass
                    return

                try:
                    active_tool.mouseMoveEvent(None, 1, world_pos)
                except Exception:
                    pass

                # Flush the frame — mesh.Modified() alone does not trigger a redraw.
                try:
                    self.plotter.render()
                except Exception:
                    pass

                if self._sphere_hover_pending_events > 0:
                    self._sphere_hover_timer.start()
                return

            if self._cursor_preview is None:
                return

            if world_pos is not None:
                self._cursor_preview.set_position(world_pos)
                self._cursor_preview.set_visibility(True)

                manager = getattr(self, 'mvat_manager', None)
                if manager is not None:
                    try:
                        manager.update_sphere_hover_overlay(world_pos, render=False)
                    except Exception:
                        pass
            else:
                self._cursor_preview.set_visibility(False)

                manager = getattr(self, 'mvat_manager', None)
                if manager is not None:
                    try:
                        manager.clear_sphere_hover_overlay(reset_context=True, render=False)
                    except Exception:
                        pass

            try:
                self.plotter.render()
            except Exception:
                pass

            # If more mouse moves arrived while we were processing this batch,
            # schedule another coalesced update for the latest cursor position.
            if self._sphere_hover_pending_events > 0:
                self._sphere_hover_timer.start()
        except Exception:
            # Silent failure
            pass

    def _on_mouse_move(self, obj, event):
        """
        Handle mouse movement to update the sphere position on mesh.

        Mouse moves are coalesced into a short timer so we only pick and render
        once per burst instead of on every raw event.
        """
        try:
            if self._cursor_preview is None:
                return

            # Only track if sphere feature is enabled
            if not self._sphere_visible:
                return

            # Refresh PyVista's stored mouse position immediately so the batch
            # processor picks the live hover location instead of the last click.
            try:
                self.plotter.store_mouse_position()
            except Exception:
                pass

            self._sphere_hover_pending_events += 1
            if not self._sphere_hover_timer.isActive():
                self._sphere_hover_timer.start()
        except Exception as e:
            # Silent failure
            pass

    # ------------------------------------------------------------------
    # Camera movement helpers
    # ------------------------------------------------------------------
    def _get_camera_vectors(self):
        """Return current camera position, focal point, view direction (normalized),
        right vector (normalized), and up vector (normalized)."""
        cam = self.plotter.camera
        pos = np.array(cam.position)
        fp = np.array(cam.focal_point)
        up = np.array(cam.up)
        up = up / np.linalg.norm(up)

        view_dir = fp - pos
        dist = np.linalg.norm(view_dir)
        if dist < 1e-6:
            view_dir_norm = np.array([0, 0, 1])
        else:
            view_dir_norm = view_dir / dist

        right = np.cross(view_dir_norm, up)
        right_norm = right / (np.linalg.norm(right) + 1e-12)
        return pos, fp, view_dir_norm, right_norm, up

    def _update_clipping_range(self):
        """Recompute near/far clipping planes based on current scene."""
        self.plotter.camera.reset_clipping_range()
        self.plotter.render()

    def _cancel_camera_motion(self):
        """Stop any active inertia or explicit camera animation."""
        controller = getattr(self, '_camera_inertia', None)
        if controller is not None:
            try:
                controller.stop()
            except Exception:
                pass

        for animator_name in ('_camera_animator', '_focal_point_animator'):
            animator = getattr(self, animator_name, None)
            if animator is not None:
                try:
                    animator.cancel()
                except Exception:
                    pass

    def _animate_focal_point(self, point):
        """Eased focal-point transition used for double-click picking."""
        target = np.asarray(point, dtype=float)
        try:
            self._cancel_camera_motion()
            cam = self.plotter.camera
            current_position = np.array(cam.position, dtype=float)
            current_up = np.array(cam.up, dtype=float)
            current_fov = float(cam.view_angle)

            # Preserve the signal contract: emit immediately, then animate.
            self.focalPointChanged.emit(target)

            animator = getattr(self, '_focal_point_animator', None)
            if animator is not None:
                try:
                    animator.animate_to_camera_state(current_position, target, current_up, current_fov)
                    return
                except Exception:
                    pass

            cam.focal_point = target.tolist()
            self.plotter.render()
        except Exception:
            try:
                cam = self.plotter.camera
                cam.focal_point = target.tolist()
                self.plotter.render()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Movement methods (each updates clipping)
    # ------------------------------------------------------------------
    def move_forward(self, speed=None):
        """Move camera forward along view direction."""
        self._cancel_camera_motion()
        if speed is None:
            speed = self.move_speed
        pos, fp, view_dir, _, _ = self._get_camera_vectors()
        delta = view_dir * speed
        self.plotter.camera.position = pos + delta
        self.plotter.camera.focal_point = fp + delta
        self._update_clipping_range()

    def move_backward(self, speed=None):
        """Move camera backward along view direction."""
        if speed is None:
            speed = self.move_speed
        self.move_forward(-speed)

    def strafe_left(self, speed=None):
        """Move camera left (perpendicular to view direction)."""
        self._cancel_camera_motion()
        if speed is None:
            speed = self.move_speed
        pos, fp, _, right, _ = self._get_camera_vectors()
        delta = -right * speed
        self.plotter.camera.position = pos + delta
        self.plotter.camera.focal_point = fp + delta
        self._update_clipping_range()

    def strafe_right(self, speed=None):
        """Move camera right."""
        self._cancel_camera_motion()
        if speed is None:
            speed = self.move_speed
        pos, fp, _, right, _ = self._get_camera_vectors()
        delta = right * speed
        self.plotter.camera.position = pos + delta
        self.plotter.camera.focal_point = fp + delta
        self._update_clipping_range()

    def rotate_left(self, angle_deg=None):
        """Orbit left (counter‑clockwise) around the focal point."""
        if angle_deg is None:
            angle_deg = self.rotate_speed
        self._orbit_yaw(np.radians(angle_deg))

    def rotate_right(self, angle_deg=None):
        """Orbit right (clockwise) around the focal point."""
        if angle_deg is None:
            angle_deg = self.rotate_speed
        self._orbit_yaw(-np.radians(angle_deg))

    def _orbit_yaw(self, angle_rad):
        """
        Rotate the camera position around the focal point,
        keeping the up vector fixed.
        """
        self._cancel_camera_motion()
        cam = self.plotter.camera
        pos = np.array(cam.position)
        fp = np.array(cam.focal_point)
        up = np.array(cam.up)
        up = up / np.linalg.norm(up)

        # Vector from focal point to camera
        vec = pos - fp
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            return

        # Decompose into parallel and perpendicular components relative to up
        v_parallel = np.dot(vec, up) * up
        v_perp = vec - v_parallel
        perp_len = np.linalg.norm(v_perp)
        if perp_len < 1e-6:
            return  # Camera is directly above/below focal point – no yaw

        v_perp_norm = v_perp / perp_len

        # Rotate the perpendicular component around up
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        v_perp_rot = cos_a * v_perp_norm + sin_a * np.cross(up, v_perp_norm)

        # New camera position
        new_pos = fp + v_parallel + v_perp_rot * perp_len

        # Update camera
        cam.position = new_pos.tolist()
        # Focal point and up remain unchanged
        self._update_clipping_range()

    def _orbit_pitch(self, angle_rad):
        """
        Pitch the camera up/down around the camera's right vector, keeping
        the focal point fixed. Positive angle pitches up.
        """
        self._cancel_camera_motion()
        cam = self.plotter.camera
        pos = np.array(cam.position)
        fp = np.array(cam.focal_point)
        up = np.array(cam.up)
        up = up / np.linalg.norm(up)

        # Vector from focal point to camera
        vec = pos - fp
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            return

        # Compute right vector
        view_dir = (fp - pos)
        view_dir_norm = view_dir / (np.linalg.norm(view_dir) + 1e-12)
        right = np.cross(view_dir_norm, up)
        right = right / (np.linalg.norm(right) + 1e-12)

        # Decompose into components parallel and perpendicular to right
        v_parallel = np.dot(vec, right) * right
        v_perp = vec - v_parallel
        perp_len = np.linalg.norm(v_perp)
        if perp_len < 1e-6:
            return

        v_perp_norm = v_perp / perp_len

        # Rotate the perpendicular component around the right vector
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        v_perp_rot = cos_a * v_perp_norm + sin_a * np.cross(right, v_perp_norm)

        # New camera position
        new_pos = fp + v_parallel + v_perp_rot * perp_len

        cam.position = new_pos.tolist()
        # focal point and up remain unchanged
        self._update_clipping_range()

    def _rotate_vector_around_axis(self, v, k, angle_rad):
        """Rotate vector v around axis k by angle_rad using Rodrigues' formula."""
        k = np.asarray(k, dtype=float)
        k = k / (np.linalg.norm(k) + 1e-12)
        v = np.asarray(v, dtype=float)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        return v * cos_a + np.cross(k, v) * sin_a + k * (np.dot(k, v)) * (1 - cos_a)

    def _rotate_yaw_inplace(self, angle_rad):
        """Rotate view direction around the camera's up vector (in-place rotation).

        Camera position remains fixed; focal_point is updated.
        """
        self._cancel_camera_motion()
        cam = self.plotter.camera
        pos = np.array(cam.position)
        fp = np.array(cam.focal_point)
        up = np.array(cam.up)
        if np.linalg.norm(up) < 1e-12:
            up = np.array([0, 0, 1])

        v = fp - pos
        if np.linalg.norm(v) < 1e-6:
            # No focal separation; choose sensible default distance
            dist = self.get_scene_median_depth(pos)
            v = np.array([0, 0, 1]) * dist

        v_rot = self._rotate_vector_around_axis(v, up, angle_rad)
        cam.focal_point = (pos + v_rot).tolist()
        # up remains unchanged for yaw
        self._update_clipping_range()

    def _rotate_pitch_inplace(self, angle_rad):
        """Rotate view direction around the camera's right vector (in-place rotation).

        Camera position remains fixed; focal_point and up vector are updated.
        """
        self._cancel_camera_motion()
        cam = self.plotter.camera
        pos = np.array(cam.position)
        fp = np.array(cam.focal_point)
        up = np.array(cam.up)
        up = up / (np.linalg.norm(up) + 1e-12)

        v = fp - pos
        if np.linalg.norm(v) < 1e-6:
            dist = self.get_scene_median_depth(pos)
            v = np.array([0, 0, 1]) * dist

        view_dir_norm = v / (np.linalg.norm(v) + 1e-12)
        right = np.cross(view_dir_norm, up)
        right = right / (np.linalg.norm(right) + 1e-12)

        v_rot = self._rotate_vector_around_axis(v, right, angle_rad)
        # Update focal point
        cam.focal_point = (pos + v_rot).tolist()

        # Recompute up vector to remain orthogonal
        new_view_dir = (pos + v_rot) - pos
        new_view_dir_norm = new_view_dir / (np.linalg.norm(new_view_dir) + 1e-12)
        new_up = np.cross(right, new_view_dir_norm)
        new_up = new_up / (np.linalg.norm(new_up) + 1e-12)
        cam.up = new_up.tolist()
        self._update_clipping_range()

    # ------------------------------------------------------------------
    # Array Selection
    # ------------------------------------------------------------------
    def _on_array_selected(self, array_name: str):
        """Handle array selection from the dropdown."""
        if self.array_selector_combo is None:
            return
            
        # FIX: Apply the selected array to ALL products in the scene, 
        # not just the primary target.
        needs_render = False
        
        for product in self.scene_context:
            if hasattr(product, 'set_selected_array'):
                # Only set it if the product actually has this array available
                if hasattr(product, 'get_available_arrays') and array_name in product.get_available_arrays():
                    product.set_selected_array(array_name)
                    needs_render = True
                    
        # Phase-2: when switching away from the Similarity heatmap, tear the GPU
        # shader off the live mesh actor first (render_scene rebuilds the actor
        # clean, but this also covers the no-rebuild path and releases textures).
        if array_name != "Similarity":
            fmm = getattr(self.mvat_manager, 'feature_mesh_manager', None)
            primary = self.scene_context.get_primary_target()
            if fmm is not None and primary is not None:
                actor = self._product_actors.get(getattr(primary, 'product_id', None))
                if actor is not None:
                    fmm.uninstall_shader(actor)

        # The label-overlay shader's install is managed entirely by render_scene: it
        # rebuilds each actor fresh and _sync_label_overlay_actor draws the paint
        # shader (discard mode) over the base array when the opacity slider is open,
        # and nothing on Similarity. An array switch always triggers render_scene
        # below, so there's no separate uninstall to do here.

        # The legacy floating paint overlays (LabelWorker fallback) are kept hidden;
        # they only show on the retired "Labels" base array, which is never selected.
        try:
            self.mvat_manager.set_paint_overlays_visible(array_name == "Labels")
        except Exception:
            pass

        # Re-render the whole scene once all products are updated
        if needs_render:
            self.render_scene()

    def _apply_paint_blend_opacity(self, opacity01: float):
        """Set the label-over-array blend opacity (0..1), driven by the annotation
        window's transparency slider.

        Crossing the 0 boundary (un)installs the blend shader, which needs a fresh
        actor → render_scene. Within (0, 1] we update the uniform live (no rebuild).
        Replace mode (the Labels array) ignores this — labels are always fully shown.
        """
        psm = getattr(self.mvat_manager, 'paint_shader_manager', None)
        if psm is None:
            return
        try:
            crossed_zero = psm.set_paint_opacity(opacity01)
        except Exception:
            return
        # The shared slider also fires during pure 2D work; skip 3D GPU work when the
        # viewer is hidden or empty. The new opacity is stored either way and applied
        # on the next MVAT render (e.g. array switch / load).
        try:
            if not self.isVisible() or not self.scene_context.has_any_product():
                return
        except Exception:
            pass
        # Splats render labels via their own GPU label channel (no overlay actor),
        # so drive that boost straight from the same slider — live, no rebuild.
        self._sync_splat_label_opacity(opacity01)
        if crossed_zero:
            # Crossing the 0 boundary (un)installs the translucent overlay actors.
            # We do NOT call render_scene() here: that rebuilds every base actor via
            # remove_actor() (which renders an empty frame with the mesh gone before
            # add_mesh re-adds it), producing a one-frame blackout right at the
            # ~1/255 slider position. Instead we add/remove ONLY the overlay actors
            # and leave the base actors untouched, then render once.
            self._sync_all_label_overlay_actors()
            try:
                self.plotter.render()
            except Exception:
                pass
        else:
            # Within (0, 1]: just retune each overlay actor's opacity in place.
            for la in list(self._label_paint_actors.values()):
                if la is not None:
                    try:
                        la.GetProperty().SetOpacity(opacity01)
                    except Exception:
                        pass
            try:
                self.plotter.render()
            except Exception:
                pass

    def _sync_splat_label_opacity(self, opacity01: float):
        """Push the shared label opacity into every Gaussian-splat product's GPU
        label channel (their analogue of the mesh/point label-overlay actor)."""
        try:
            products = self.scene_context.get_products_by_class(GaussianSplattingProduct)
        except Exception:
            return
        for p in products:
            try:
                p.apply_label_opacity(opacity01)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Label visibility (per-class show/hide in 3D)
    # ------------------------------------------------------------------

    def update_label_visibility(self, label, is_visible: bool):
        """Hide or show all 3D elements painted with a given label's class_id.

        Uses the PropagationEngine's canonical class_id mapping (which excludes
        the Review label with id='-1') to resolve the label to a class_id, then
        toggles visibility in the labels cache of every product and re-renders.
        """
        if not hasattr(self, 'mvat_manager') or self.mvat_manager is None:
            return

        # Resolve class_id via the canonical mapping (skips Review, 1-indexed)
        engine = getattr(self.mvat_manager, 'propagation_engine', None)
        if engine is None:
            return

        label_id = getattr(label, 'id', None)
        if label_id is None or label_id == '-1':
            return

        class_id = engine.canonical_class_id_for_label_id(label_id)
        if class_id is None:
            return

        # Update hidden set on each product and refresh
        needs_render = False
        for product in self.scene_context:
            hidden = getattr(product, '_hidden_class_ids', None)
            if hidden is None:
                product._hidden_class_ids = set()
                hidden = product._hidden_class_ids

            if is_visible:
                if class_id in hidden:
                    hidden.discard(class_id)
                    self._restore_hidden_elements(product, class_id)
                    needs_render = True
            else:
                if class_id not in hidden:
                    hidden.add(class_id)
                    self._hide_elements_by_class(product, class_id)
                    needs_render = True

        if needs_render:
            try:
                self.plotter.render()
            except Exception:
                pass

    def _hide_elements_by_class(self, product, class_id: int):
        """Set elements with the given class_id to white (255,255,255) in the
        labels cache and flush to GPU, effectively hiding them."""
        class_ids = getattr(product, 'class_ids', None)
        if class_ids is None:
            return

        mask = class_ids == class_id
        if not np.any(mask):
            return

        if isinstance(product, GaussianSplattingProduct):
            cache = getattr(product, '_label_color_cache', None)
            if cache is not None:
                product._labels_dirty = True
                product.flush_labels_to_gpu()
                # After flush, re-tint hidden splats to pristine (erase the label tint)
                try:
                    product.gaussian_actor.reset_colors(np.flatnonzero(mask))
                except Exception:
                    pass
        else:
            cache = getattr(product, '_labels_cache', None)
            if cache is None:
                return
            cache[mask] = (255, 255, 255)
            product._labels_dirty = True
            product.flush_labels_to_gpu()

            # Also update the shader texture so overlay actors match
            psm = getattr(self.mvat_manager, 'paint_shader_manager', None)
            if psm is not None:
                state = psm.get_state(product)
                if state is not None:
                    ids = np.flatnonzero(mask)
                    state.update_class_ids_subset(ids, 0)

    def _restore_hidden_elements(self, product, class_id: int):
        """Restore the original label color for elements with the given class_id."""
        class_ids = getattr(product, 'class_ids', None)
        if class_ids is None:
            return

        mask = class_ids == class_id
        if not np.any(mask):
            return

        # Look up the label color via the canonical mapping
        engine = getattr(self.mvat_manager, 'propagation_engine', None)
        if engine is None:
            return
        real_labels = engine._canonical_real_labels()
        if class_id < 1 or class_id > len(real_labels):
            return
        label = real_labels[class_id - 1]
        try:
            color_rgb = (label.color.red(), label.color.green(), label.color.blue())
        except Exception:
            return

        if isinstance(product, GaussianSplattingProduct):
            product._labels_dirty = True
            product.flush_labels_to_gpu()
        else:
            cache = getattr(product, '_labels_cache', None)
            if cache is None:
                return
            cache[mask] = color_rgb
            product._labels_dirty = True
            product.flush_labels_to_gpu()

            # Restore shader texture class_ids
            psm = getattr(self.mvat_manager, 'paint_shader_manager', None)
            if psm is not None:
                state = psm.get_state(product)
                if state is not None:
                    ids = np.flatnonzero(mask)
                    state.update_class_ids_subset(ids, class_id)

    # ------------------------------------------------------------------
    # Key event handling
    # ------------------------------------------------------------------
    def keyPressEvent(self, event):
        """Handle key presses.

                New mapping:
                - WASD: change view direction in-place (rotate viewing direction)
                    W/S: pitch up/down, A/D: yaw left/right
        - Arrow keys: move camera position (forward/back/strafe)
        - Q/E: rotate view left/right
        - View shortcuts: T=Top, F=Front, C=Bottom (Caboose), L=Left, R=Right, B=Back, I=Isometric
        """
        key = event.key()
        ang = np.radians(self.rotate_speed)

        if (event.modifiers() & Qt.ControlModifier) and (event.modifiers() & Qt.ShiftModifier):
            manager = getattr(self, 'mvat_manager', None)
            annotation_window = getattr(manager, 'annotation_window', None) if manager is not None else None
            selected_tool = None
            tool_map = None

            if annotation_window is not None:
                try:
                    selected_tool = annotation_window.get_selected_tool()
                except Exception:
                    selected_tool = None
                tool_map = getattr(annotation_window, 'tools', None)

            tool = None
            if isinstance(tool_map, dict) and selected_tool in ('brush', 'erase'):
                tool = tool_map.get(selected_tool)

            if tool is not None:
                toggle = getattr(tool, '_toggle_shape', None)
                if callable(toggle):
                    try:
                        toggle()
                        event.accept()
                        return
                    except Exception:
                        pass

            active_tool = getattr(self, '_active_3d_tool', None)
            if active_tool is not None:
                current_shape = str(getattr(active_tool, 'brush_shape', 'circle')).strip().lower()
                next_shape = 'square' if current_shape == 'circle' else 'circle'
                center = getattr(active_tool, '_last_hover_world_pos', None)
                if center is None:
                    try:
                        center = np.asarray(self.plotter.camera.focal_point, dtype=np.float64)
                    except Exception:
                        center = None

                setter = getattr(active_tool, 'set_brush_shape', None)
                if callable(setter):
                    try:
                        setter(next_shape, center=center)
                    except Exception:
                        pass
                else:
                    active_tool.brush_shape = next_shape

                manager = getattr(self, 'mvat_manager', None)
                if manager is not None and center is not None:
                    try:
                        manager.update_sphere_hover_overlay(center, render=False)
                    except Exception:
                        pass
            return

        if key == Qt.Key_W:
            # Pitch up (rotate view direction in-place)
            self._rotate_pitch_inplace(ang)
            event.accept()
        elif key == Qt.Key_S:
            # Pitch down (rotate view direction in-place)
            self._rotate_pitch_inplace(-ang)
            event.accept()
        elif key == Qt.Key_A:
            # Yaw left (rotate view direction in-place)
            self._rotate_yaw_inplace(ang)
            event.accept()
        elif key == Qt.Key_D:
            # Yaw right (rotate view direction in-place)
            self._rotate_yaw_inplace(-ang)
            event.accept()
        elif key == Qt.Key_Up:
            self.move_forward()
            event.accept()
        elif key == Qt.Key_Down:
            self.move_backward()
            event.accept()
        elif key == Qt.Key_Left:
            self.strafe_left()
            event.accept()
        elif key == Qt.Key_Right:
            self.strafe_right()
            event.accept()
        elif key == Qt.Key_Q:
            self.rotate_right()
            event.accept()
        elif key == Qt.Key_E:
            self.rotate_left()
            event.accept()
        elif key == Qt.Key_T:
            # Top view
            self.view_top()
            event.accept()
        elif key == Qt.Key_F:
            # Front view
            self.view_front()
            event.accept()
        elif key == Qt.Key_C:
            # Bottom view (Caboose)
            self.view_bottom()
            event.accept()
        elif key == Qt.Key_L:
            # Left view
            self.view_left()
            event.accept()
        elif key == Qt.Key_R:
            # Right view
            self.view_right()
            event.accept()
        elif key == Qt.Key_B:
            # Back view
            self.view_back()
            event.accept()
        elif key == Qt.Key_I:
            # Isometric view
            self.view_isometric()
            event.accept()
        elif key == Qt.Key_O:
            # Toggle orthographic projection via hotkey 'O'
            try:
                current = bool(getattr(self.plotter.camera, 'parallel_projection', False))
                self.toggle_orthographic(not current)
                event.accept()
            except Exception:
                pass
        elif key == Qt.Key_Space:
            # Space: launch MVAT-SAM dialog when the SAM tool is active
            manager = getattr(self, 'mvat_manager', None)
            if manager is not None:
                ann_win = getattr(manager, 'annotation_window', None)
                selected_tool = None
                if ann_win is not None:
                    try:
                        selected_tool = ann_win.get_selected_tool()
                    except Exception:
                        pass
                if selected_tool == 'sam':
                    try:
                        manager.launch_viewer_sam()
                    except Exception as _e:
                        print(f"MVAT-SAM launch error: {_e}")
                    event.accept()
                    return
            # Otherwise forward Space to the active 3D tool (e.g. the
            # FeatureSelectTool3D uses it to finalize the highlighted selection).
            active_tool = getattr(self, '_active_3d_tool', None)
            if active_tool is not None and hasattr(active_tool, 'keyPressEvent'):
                try:
                    active_tool.keyPressEvent(event)
                    if event.isAccepted():
                        return
                except Exception:
                    pass
            event.ignore()
        else:
            # Forward to active 3D tool (e.g., FeatureSelectTool3D for Enter/Escape)
            active_tool = getattr(self, '_active_3d_tool', None)
            if active_tool is not None and hasattr(active_tool, 'keyPressEvent'):
                try:
                    active_tool.keyPressEvent(event)
                    if event.isAccepted():
                        return
                except Exception:
                    pass
            # Let the parent widget handle other keys
            event.ignore()

    def set_focal_point(self, point):
        """Sets the camera focal point and re-renders."""
        self._cancel_camera_motion()
        target = np.asarray(point, dtype=float)
        self.plotter.camera.focal_point = target.tolist()
        self.plotter.render()
        self.focalPointChanged.emit(target)
        
    # --------------------------------------------------------------------------
    # Scene Product Loading
    # --------------------------------------------------------------------------
    
    # Supported file extensions for each product type
    _POINT_CLOUD_EXTENSIONS = ['.pcd']
    _MESH_EXTENSIONS = ['.ply', '.obj', '.stl', '.vtk']
    
    @property
    def point_cloud(self) -> 'PointCloudProduct':
        """
        Backward-compatible property to access the first point cloud product.
        
        Returns:
            First PointCloudProduct in the scene, or None if none loaded.
        """
        products = self.scene_context.get_products_by_class(PointCloudProduct)
        return products[0] if products else None
    
    @point_cloud.setter
    def point_cloud(self, value):
        """
        Backward-compatible setter for point_cloud.
        
        Clears existing point clouds and adds the new one to scene context.
        """
        if value is None:
            # Remove all point cloud products
            for p in list(self.scene_context.get_products_by_class(PointCloudProduct)):
                self.remove_product(p.product_id)
        else:
            # Add to scene context (remove existing point clouds first)
            for p in list(self.scene_context.get_products_by_class(PointCloudProduct)):
                self.remove_product(p.product_id)
            self.add_product(value)

    def dragEnterEvent(self, event):
        """Accept drag if a single supported 3D file is being dragged."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1:
                file_path = urls[0].toLocalFile().lower()
                supported_extensions = (
                    self._POINT_CLOUD_EXTENSIONS + 
                    self._MESH_EXTENSIONS
                )
                if any(file_path.endswith(ext) for ext in supported_extensions):
                    event.acceptProposedAction()
                else:
                    event.ignore()
            else:
                event.ignore()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Load the dropped 3D file via the shared ImportModel dialog."""
        file_path = event.mimeData().urls()[0].toLocalFile()

        top = self.window()
        import_model = getattr(top, 'import_model', None)
        if import_model is None:
            from coralnet_toolbox.IO.QtImportModel import ImportModel
            import_model = ImportModel(top)

        if import_model.import_model_file(file_path):
            event.acceptProposedAction()
        else:
            event.ignore()

    def _create_product_from_file(self, file_path: str, ply_type: str = None,
                                  sort_data: bool = True, simplification_ratio: float = 0.0,
                                  load_texture: bool = True) -> 'AbstractSceneProduct':
        """
        Create the appropriate scene product for a 3D data file.

        For .ply files the caller should pass ``ply_type`` (one of ``'Mesh'``,
        ``'Point Cloud'``, or ``'3D Gaussian Splatting'``) so that the dialog
        is not re-shown here.  All other extensions are auto-detected.

        Args:
            file_path: Path to the 3D data file.
            ply_type:  Pre-resolved PLY type string from the disambiguation
                       dialog, or None for non-PLY files.
            sort_data: When True, spatially sort mesh faces or point cloud points
                       for improved GPU cache coherence and index-map compression.
            simplification_ratio: Fraction of mesh faces or point cloud points to
                       remove before loading (0.0 = no simplification, 1.0 = remove all).
            load_texture: When True (default), attempt to load an associated texture
                       image for mesh products. Only relevant for mesh products.

        Returns:
            A concrete AbstractSceneProduct instance, or None if the type
            cannot be determined.
        """
        file_ext = os.path.splitext(file_path)[1].lower()

        _mesh_kwargs = dict(sort_data=sort_data, 
                            simplification_ratio=simplification_ratio,
                            load_texture=load_texture)
        # Point clouds mirror the mesh knobs: spatial sort + fractional decimation.
        _cloud_kwargs = dict(point_size=self.point_size, sort_data=sort_data,
                             simplification_ratio=simplification_ratio)

        # STL and OBJ are always meshes by format definition.
        if file_ext in ['.stl', '.obj']:
            print(f"📐 {file_ext.upper()} is a mesh-only format, creating MeshProduct")
            return MeshProduct.from_file(file_path, **_mesh_kwargs)

        # PCD is always a point cloud.
        if file_ext == '.pcd':
            print(f"☁️ PCD is a point-cloud-only format, creating PointCloudProduct")
            return PointCloudProduct.from_file(file_path, **_cloud_kwargs)

        # PLY: route based on the user-selected type from the dialog.
        if file_ext == '.ply':
            if ply_type == 'Mesh':
                print(f"📐 PLY → MeshProduct (user selection)")
                return MeshProduct.from_file(file_path, **_mesh_kwargs)
            elif ply_type == 'Point Cloud':
                print(f"☁️ PLY → PointCloudProduct (user selection)")
                return PointCloudProduct.from_file(file_path, **_cloud_kwargs)
            elif ply_type == '3D Gaussian Splatting':
                print(f"✨ PLY → GaussianSplattingProduct (user selection)")
                return GaussianSplattingProduct.from_file(file_path)
            else:
                print(f"📐 PLY type unknown — defaulting to MeshProduct")
                return MeshProduct.from_file(file_path, **_mesh_kwargs)

        # VTK: infer from structure.
        if file_ext == '.vtk':
            import pyvista as pv
            temp_mesh = pv.read(file_path)
            if temp_mesh.n_cells > 0 and temp_mesh.n_cells < temp_mesh.n_points:
                print(f"📐 VTK detected as mesh (cells < points)")
                return MeshProduct.from_file(file_path, **_mesh_kwargs)
            else:
                print(f"☁️ VTK detected as point cloud")
                return PointCloudProduct.from_file(file_path, **_cloud_kwargs)

        return None

    def add_product(self, product: 'AbstractSceneProduct') -> None:
        """
        Add a scene product to the viewer.
        
        Args:
            product: Scene product to add.
        """
        # If the scene already has products, make the new product inherit 
        # the currently active visualization style from the dropdown. 
        # (If the scene is empty, let it keep its default RGB and the UI will sync to it).
        if self.scene_context.has_any_product() and self.array_selector_combo is not None:
            current_array = self.array_selector_combo.currentText()
            if current_array and hasattr(product, 'set_selected_array'):
                if hasattr(product, 'get_available_arrays') and current_array in product.get_available_arrays():
                    product.set_selected_array(current_array)

        self.scene_context.add_product(product)
        
        # Hide placeholder once we have data
        try:
            self._hide_placeholder()
        except Exception:
            pass
        
        # Update primary target menu
        try:
            self._update_primary_target_menu()
        except Exception:
            pass
        
        # Update array selector based on new product
        try:
            self._update_array_selector()
        except Exception:
            pass

        self._sync_context_matrix_scene_controls()

    def remove_product(self, product_id: str) -> None:
        """
        Remove a scene product from the viewer.

        For GaussianSplattingProduct the OpenGL renderer is cleaned up before
        the actor is removed from the plotter so that GPU resources are freed
        in the correct order.

        Args:
            product_id: ID of the product to remove.
        """
        # Fetch the product before it is removed from the context so we can
        # call type-specific teardown logic below.
        product = self.scene_context.get_product(product_id)

        # Release GaussianActor OpenGL resources first while the actor is still
        # alive in the plotter, then remove the (now inert) actor handle.
        if isinstance(product, GaussianSplattingProduct):
            product.cleanup()

        actor = self._product_actors.pop(product_id, None)
        if actor is not None:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass

        self.scene_context.remove_product(product_id)
        self._update_splat_controls()

        # Show placeholder if scene is now empty
        if not self.scene_context.has_any_product():
            self._show_placeholder()
        
        # Update primary target menu
        try:
            self._update_primary_target_menu()
        except Exception:
            pass
        
        # Update array selector based on remaining products
        try:
            self._update_array_selector()
        except Exception:
            pass

        self._sync_context_matrix_scene_controls()

    def _sync_context_matrix_scene_controls(self) -> None:
        """Mirror scene-product availability into the ContextMatrix toolbar state."""
        manager = getattr(self, 'mvat_manager', None)
        context_matrix = getattr(manager, 'context_matrix', None) if manager is not None else None
        if context_matrix is None:
            return

        try:
            context_matrix.set_scene_controls_enabled(self.scene_context.has_any_product())
        except Exception:
            pass

    def render_scene(self) -> None:
        """
        Render all scene products into the viewer.
        
        Creates or updates PyVista actors for each product based on
        their render style preferences. Respects current visibility settings.
        """
        # Hide placeholder if we have any products
        if self.scene_context.has_any_product():
            try:
                self._hide_placeholder()
            except Exception:
                pass
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        was_empty = not self._product_actors  # True when no actors yet (first load)
        try:
            for product in self.scene_context:
                product_id = product.product_id
                mesh = product.get_render_mesh()
                style = product.get_render_style()

                if mesh is None:
                    continue

                # Deferred-paint barrier: sync the VTK Labels array from the
                # Python cache before (re)building the actor. No-op unless the
                # product's labels are dirty, so this is free in the common case.
                flush_labels = getattr(product, 'flush_labels_to_gpu', None)
                if callable(flush_labels):
                    try:
                        flush_labels()
                    except Exception:
                        pass

                should_be_visible = self._get_visibility_for_product(product)

                # ------------------------------------------------------------------
                # GaussianSplattingProduct: bind_to_plotter() installs the
                # VTK-native single-pass renderer (geometry shader + SSBOs) and
                # registers a camera observer for depth sorting.  Call it only
                # once; subsequent render_scene() calls only toggle visibility.
                # ------------------------------------------------------------------
                if isinstance(product, GaussianSplattingProduct):
                    if product_id not in self._product_actors:
                        product.gaussian_actor.bind_to_plotter(self.plotter)
                        try:
                            product.gaussian_actor.scale_modifier = self._splat_scale
                            product.gaussian_actor.render_mode = self._gaussian_shading_idx
                        except Exception:
                            pass
                        self._product_actors[product_id] = product.gaussian_actor.actor
                    actor = self._product_actors[product_id]
                    # Label-channel strength is left at the product's default
                    # (visible) and only changed when the user actually moves the
                    # transparency slider (_sync_splat_label_opacity). We do NOT
                    # force it to paint_opacity here: that value can be 0 (slider
                    # down / restored low), which would silently hide painted labels.
                    try:
                        actor.SetVisibility(should_be_visible)
                    except Exception:
                        pass
                    continue

                actor = self._product_actors.get(product_id)

                # FIX: If the actor exists, remove it so we can re-apply the fresh style
                if actor is not None:
                    self.plotter.remove_actor(actor)

                # Add mesh with the new style dictionary
                actor = self.plotter.add_mesh(
                    mesh,
                    render=False,
                    reset_camera=False,
                    **style
                )

                # Ensure actor opacity matches style
                try:
                    actor.GetProperty().SetOpacity(style.get('opacity', 1.0))
                except Exception:
                    pass

                self._product_actors[product_id] = actor

                # Phase-2: (re)install the feature-similarity GPU shader on the
                # freshly built actor when the Similarity array is active. The
                # actor is rebuilt here on every full render, so the shader must
                # be re-applied; the install is a no-op otherwise and falls back
                # to the uint8 cell scalar on any failure.
                fmm = getattr(self.mvat_manager, 'feature_mesh_manager', None)
                if fmm is not None:
                    try:
                        fmm.maybe_install_shader(actor, product)
                    except Exception:
                        pass

                # Label paint is rendered by the translucent label-overlay actor
                # below (the paint shader in discard mode), not on the base actor —
                # "Labels" is no longer a selectable base array. Rebuilt each render.
                try:
                    self._sync_label_overlay_actor(product, mesh, product_id,
                                                    should_be_visible, style)
                except Exception:
                    pass

                # Apply visibility setting
                try:
                    actor.SetVisibility(should_be_visible)
                except Exception:
                    pass

            # Drop label-overlay actors for products no longer in the scene.
            try:
                current_ids = {getattr(p, 'product_id', None) for p in self.scene_context}
                for pid in list(self._label_paint_actors.keys()):
                    if pid not in current_ids:
                        stale = self._label_paint_actors.pop(pid, None)
                        if stale is not None:
                            self.plotter.remove_actor(stale, render=False)
            except Exception:
                pass

            refresh_overlay = getattr(self.mvat_manager, 'refresh_primary_mesh_overlay', None)
            if callable(refresh_overlay):
                try:
                    refresh_overlay(force_recreate=True, render=False)
                except Exception:
                    pass

            # Point clouds keep their painted labels in a separate point overlay actor.
            refresh_point_overlay = getattr(self.mvat_manager, 'refresh_primary_point_overlay', None)
            if callable(refresh_point_overlay):
                try:
                    refresh_point_overlay(force_recreate=True, render=False)
                except Exception:
                    pass

            self._update_splat_controls()
            self.plotter.render()
            print(f"Rendered {len(self.scene_context)} scene products")

            # On first load (scene was empty), fit the camera to the new content.
            if was_empty and self._product_actors:
                try:
                    self.fit_to_view()
                except Exception:
                    pass

        finally:
            QApplication.restoreOverrideCursor()

    def _sync_all_label_overlay_actors(self):
        """Add/remove the translucent label-overlay actors for every product
        without rebuilding the base actors.

        Used when the transparency slider crosses the 0 opacity boundary: only the
        overlay actors need to appear/disappear, so rebuilding the whole scene (and
        flickering the base meshes) is both unnecessary and visually jarring.
        """
        for product in self.scene_context:
            product_id = getattr(product, 'product_id', None)
            if product_id is None:
                continue
            try:
                mesh = product.get_render_mesh()
                if mesh is None:
                    continue
                style = product.get_render_style()
                visible = self._get_visibility_for_product(product)
                self._sync_label_overlay_actor(product, mesh, product_id, visible, style)
            except Exception:
                pass

    def _sync_label_overlay_actor(self, product, mesh, product_id, visible: bool, base_style: dict):
        """Create/remove the translucent label-overlay actor for one product.

        On a non-Labels array with the opacity slider open, a second actor sharing
        the product's geometry is drawn over the base actor: it uses the paint shader
        in discard mode (unpainted fragments discarded) so only painted faces show,
        flat label colors blended over the base by the actor's opacity. This avoids
        reading VTK's base color in-shader (unreliable for direct-RGB scalars) and
        works uniformly for RGB / Texture / UV arrays.

        The overlay is added with the SAME style (scalars / texture) as the base
        actor: because the polydata is shared (copy_mesh=False), adding it with
        different/no scalars would mutate the shared active-scalar state and corrupt
        the base actor's coloring. The shader then does ScalarVisibilityOff, so the
        overlay's own scalars are never actually drawn.
        """
        # The geometry is rebuilt with the base actor each render, so drop any prior
        # overlay actor for this product first.
        existing = self._label_paint_actors.pop(product_id, None)
        if existing is not None:
            try:
                self.plotter.remove_actor(existing, render=False)
            except Exception:
                pass

        psm = getattr(self.mvat_manager, 'paint_shader_manager', None)
        if psm is None or not psm.should_show_label_overlay(product):
            return

        overlay_kwargs = dict(base_style)
        overlay_kwargs.update(
            render=False,
            reset_camera=False,
            copy_mesh=False,
            lighting=False,
            show_scalar_bar=False,
            opacity=float(psm.paint_opacity),
        )
        try:
            overlay_actor = self.plotter.add_mesh(mesh, **overlay_kwargs)
        except Exception:
            return

        if not psm.install_label_overlay_shader(overlay_actor, product):
            try:
                self.plotter.remove_actor(overlay_actor, render=False)
            except Exception:
                pass
            return

        try:
            overlay_actor.GetProperty().SetOpacity(float(psm.paint_opacity))
            # Pull the coincident overlay slightly toward the camera so it wins the
            # depth test over the base actor without z-fighting.
            mapper = overlay_actor.GetMapper()
            if mapper is not None:
                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                try:
                    mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-2.0, -2.0)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            overlay_actor.SetVisibility(bool(visible))
        except Exception:
            pass

        self._label_paint_actors[product_id] = overlay_actor

    def _get_visibility_for_product(self, product: 'AbstractSceneProduct') -> bool:
        """
        Determine if a product should be visible based on its type.
        
        Args:
            product: The scene product to check.
            
        Returns:
            True if the product should be visible.
        """
        if isinstance(product, PointCloudProduct):
            return self._show_point_clouds
        elif isinstance(product, MeshProduct):
            return self._show_meshes
        elif isinstance(product, GaussianSplattingProduct):
            return self._show_gaussian_splats
        return True  # Default to visible for unknown types

    # --------------------------------------------------------------------------
    # Primary Target Selection
    # --------------------------------------------------------------------------
    
    def _update_primary_target_menu(self):
        """
        Rebuild the Primary Target menu with current scene products.
        
        Called when products are added/removed or when the menu needs refresh.
        """
        if not hasattr(self, '_primary_target_menu'):
            return
            
        menu = self._primary_target_menu
        group = self._primary_target_action_group
        
        # Clear existing actions
        menu.clear()
        for action in group.actions():
            group.removeAction(action)
        
        # Add "None" option
        action_none = QAction("None (Auto)", self)
        action_none.setCheckable(True)
        action_none.setData(None)
        action_none.triggered.connect(lambda: self._on_primary_target_selected(None))
        group.addAction(action_none)
        menu.addAction(action_none)
        
        # Check if scene is empty
        if not self.scene_context.has_any_product():
            action_none.setChecked(True)
            return
        
        menu.addSeparator()
        
        # Group products by type
        current_target_id = self.scene_context.primary_target_id
        
        # Add each product as an option
        for product in self.scene_context:
            if not product.supports_index_mapping():
                continue
                
            element_type = product.get_element_type()
            element_count = product.get_element_count()
            label = f"{product.label} ({element_type}, {element_count:,} elements)"
            
            action = QAction(label, self)
            action.setCheckable(True)
            action.setData(product.product_id)
            action.triggered.connect(
                lambda checked, pid=product.product_id: self._on_primary_target_selected(pid)
            )
            group.addAction(action)
            menu.addAction(action)
            
            # Check if this is the current target
            if product.product_id == current_target_id:
                action.setChecked(True)
        
        # If no target selected, check "None"
        if current_target_id is None:
            action_none.setChecked(True)
    
    def _on_primary_target_selected(self, product_id: str):
        """
        Handle primary target selection from menu.
        
        Args:
            product_id: ID of selected product, or None for auto-select.
        """
        if product_id is None:
            # Auto-select based on priority
            self.scene_context.set_primary_target(None)
            # Re-select first indexable product if any
            for p in self.scene_context:
                if p.supports_index_mapping():
                    self.scene_context.set_primary_target(p.product_id)
                    break
        else:
            self.scene_context.set_primary_target(product_id)
        
        # Emit signal for manager to react
        target = self.scene_context.get_primary_target()
        target_id = target.product_id if target else ""
        self.primaryTargetChanged.emit(target_id)
        
        # Update the array selector dropdown to show available arrays for this product
        self._update_array_selector()
        
        print(f"🎯 Primary target changed to: {target_id or 'None'}")

    def _update_array_selector(self):
        """Update the array selector dropdown based on the current primary target."""
        # Check if combo has been created yet
        if self.array_selector_combo is None:
            return
            
        target = self.scene_context.get_primary_target()
        
        # Clear and ignore signals while updating
        self.array_selector_combo.blockSignals(True)
        self.array_selector_combo.clear()
        
        if target is None or not hasattr(target, 'get_available_arrays'):
            # No valid target or doesn't support arrays. "Labels" is no longer a
            # selectable base array (labels are shown via the translucent overlay
            # driven by the transparency slider), so fall back to RGB.
            self.array_selector_combo.addItem("RGB")
        else:
            # Add all available arrays from the target product, except "Labels":
            # labels are rendered as a translucent overlay (opacity slider), not as
            # a selectable base array, so there's nothing for the user to pick here.
            available = [a for a in target.get_available_arrays() if a != "Labels"]
            for array_name in available:
                self.array_selector_combo.addItem(array_name)

            # Select the currently selected array. If the product is somehow still
            # on "Labels" (e.g. a legacy fallback), findText returns -1 and we leave
            # the combo on its first entry without forcing a product change.
            selected = target.get_selected_array()
            index = self.array_selector_combo.findText(selected)
            if index >= 0:
                self.array_selector_combo.setCurrentIndex(index)
        
        self.array_selector_combo.blockSignals(False)

    def add_point_cloud(self):
        """
        Render the point cloud into the scene.
        
        Backward-compatible method that calls render_scene().
        """
        self.render_scene()

    def _show_placeholder(self, text: str = None):
        """Show the placeholder widget in the stacked layout."""
        try:
            if text:
                self._placeholder_label.setText(text)
            if hasattr(self, '_stack') and self._stack.indexOf(self._placeholder_label) >= 0:
                self._stack.setCurrentWidget(self._placeholder_label)
        except Exception:
            pass

    def _hide_placeholder(self):
        """Hide the placeholder and show the plotter interactor."""
        try:
            if hasattr(self, '_stack') and self._stack.indexOf(self.plotter.interactor) >= 0:
                self._stack.setCurrentWidget(self.plotter.interactor)
        except Exception:
            pass

    def set_point_size(self, size):
        """Update the point size for point cloud products."""
        self.point_size = size
        try:
            for p in self.scene_context.get_products_by_class(PointCloudProduct):
                actor = self._product_actors.get(p.product_id)
                if actor is not None:
                    try:
                        actor.GetProperty().SetPointSize(size)
                    except Exception:
                        pass
            try:
                self.plotter.render()
            except Exception:
                pass
        except Exception:
            pass

    def set_splat_scale(self, value: float):
        """Update the scale modifier for all Gaussian splat products."""
        self._splat_scale = float(value)
        try:
            from coralnet_toolbox.MVAT.core.Products import GaussianSplattingProduct
            for p in self.scene_context.get_products_by_class(GaussianSplattingProduct):
                gaussian_actor = getattr(p, 'gaussian_actor', None)
                if gaussian_actor is not None:
                    try:
                        gaussian_actor.scale_modifier = self._splat_scale
                    except Exception:
                        pass
            self.plotter.render()
        except Exception:
            pass

    def _on_gaussian_shading_changed(self, idx: int):
        """Update the shading/visualisation mode for all Gaussian splat products."""
        self._gaussian_shading_idx = idx
        try:
            from coralnet_toolbox.MVAT.core.Products import GaussianSplattingProduct
            for p in self.scene_context.get_products_by_class(GaussianSplattingProduct):
                gaussian_actor = getattr(p, 'gaussian_actor', None)
                if gaussian_actor is not None:
                    try:
                        gaussian_actor.render_mode = idx
                    except Exception:
                        pass
            self.plotter.render()
        except Exception:
            pass

    def _update_splat_controls(self):
        """Enable the shading combo when Gaussian products are present, grey it out when not."""
        combo = getattr(self, '_gaussian_shading_combo', None)
        if combo is None:
            return
        try:
            from coralnet_toolbox.MVAT.core.Products import GaussianSplattingProduct
            has_splats = bool(self.scene_context.get_products_by_class(GaussianSplattingProduct))
        except Exception:
            has_splats = False
        combo.setEnabled(has_splats)

    def _on_point_size_spin_changed(self, value):
        """Handle spinbox changes: set internal point size and emit signal."""
        try:
            self.set_point_size(value)
            self.pointSizeChanged.emit(value)
        except Exception:
            pass
    
    # Subsetting removed: viewer now renders the full point cloud immediately

    # --------------------------------------------------------------------------
    # Ray Visualization Methods (Using BatchedRayManager)
    # --------------------------------------------------------------------------
    
    def show_ray(self, ray: 'CameraRay', color: str = RAY_COLOR_SELECTED):
        """
        Display a single ray in the 3D viewer.
        
        Draws a line from the ray origin to the terminal point, with a
        sphere glyph at the terminal point.
        
        Args:
            ray: CameraRay object to visualize.
            color: Color for the ray (default: lime for selected camera).
            
        Note: For multiple rays, use show_rays() instead.
        """
        if ray is None:
            self.clear_ray()
            return
            
        # Use batched manager for single ray too (consistency)
        self.show_rays([(ray, color)])
    
    def show_rays(self, rays_with_colors: list):
        """
        Display multiple rays in the 3D viewer with distinct colors.

        Uses BatchedRayManager for efficient rendering - all rays are merged
        into a single PolyData mesh with one draw call instead of 2*N calls.
        Pre-allocates ray slots to minimize actor churn when ray count varies.

        Args:
            rays_with_colors: List of (CameraRay, color_tuple) tuples.
                              Colors should be RGB tuples (0-255 or 0-1).
        """
        if not self._show_rays_enabled:
            return

        if not rays_with_colors:
            self.clear_ray()
            return

        # Update or rebuild ray batch based on pre-allocated capacity
        if self._ray_manager.ray_actor is not None and len(rays_with_colors) <= self._ray_manager._allocated_rays:
            # Update existing rays in-place if they fit in allocated capacity
            self._ray_manager.update_ray_endpoints(rays_with_colors)
        else:
            # Build new batched ray geometry with pre-allocation
            self._ray_manager.build_ray_batch(rays_with_colors)
            # Add to plotter (removes old actors first)
            self._ray_manager.add_to_plotter(self.plotter, line_width=3)

        # Apply visibility state
        self._ray_manager.set_visibility(self._ray_visible)

        # Update display
        self.plotter.render()

    def show_ortho_ray(self, world_point: np.ndarray, ortho_direction: np.ndarray,
                       color: tuple = RAY_COLOR_SELECTED):
        """Display the orthomosaic debug ray as a single cached line actor."""
        if world_point is None:
            self.clear_ortho_ray()
            return

        world_point = np.asarray(world_point, dtype=np.float64)
        ortho_direction = np.asarray(ortho_direction, dtype=np.float64)
        direction_norm = np.linalg.norm(ortho_direction)
        if not np.isfinite(direction_norm) or direction_norm <= 1e-12:
            ortho_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            ortho_direction = ortho_direction / direction_norm

        bounds = None
        try:
            if self.scene_context is not None and self.scene_context.has_any_product():
                bounds = self.scene_context.unified_bounds()
        except Exception:
            bounds = None

        if bounds is None:
            bounds = self.get_bounds()

        if bounds is not None:
            try:
                scene_size = float(np.sqrt(
                    (bounds[1] - bounds[0]) ** 2
                    + (bounds[3] - bounds[2]) ** 2
                    + (bounds[5] - bounds[4]) ** 2
                ))
                ray_length = scene_size if np.isfinite(scene_size) and scene_size > 1e-6 else 100.0
            except Exception:
                ray_length = 100.0
        else:
            ray_length = 100.0

        sky_origin = world_point + (ortho_direction * ray_length)
        ortho_ray = CameraRay(
            origin=sky_origin,
            direction=world_point - sky_origin,
            terminal_point=world_point,
            has_accurate_depth=True,
            pixel_coord=None,
            source_camera=None,
        )

        rays_with_colors = [(ortho_ray, color)]
        if self._ortho_ray_manager.ray_actor is not None and self._ortho_ray_manager._num_rays == 1:
            self._ortho_ray_manager.update_ray_endpoints(rays_with_colors)
        else:
            self._ortho_ray_manager.build_ray_batch(rays_with_colors)
            self._ortho_ray_manager.add_to_plotter(self.plotter, line_width=4)

        self._ortho_ray_manager.set_visibility(self._ray_visible)
        self.plotter.render()

    def clear_ortho_ray(self):
        """Hide the orthomosaic debug ray without destroying its cached actor."""
        actor = self._ortho_ray_manager.ray_actor
        if actor is None:
            return

        try:
            if actor.GetVisibility():
                self._ortho_ray_manager.set_visibility(False)
                self.plotter.render()
        except Exception:
            try:
                self._ortho_ray_manager.set_visibility(False)
                self.plotter.render()
            except Exception:
                pass
        
    def clear_ray(self):
        """Remove any displayed ray visualization."""
        if self._ray_manager.ray_actor is None and self._ray_manager.ray_mesh is None:
            return
        self._ray_manager.remove_from_plotter(self.plotter)
        self._ray_manager.clear()
        self.plotter.render()

    def set_ray_visible(self, visible: bool):
        """
        Toggle ray visualization visibility.

        Args:
            visible: Whether the ray should be visible.
        """
        self._ray_visible = visible
        self._ray_manager.set_visibility(visible)
        self._ortho_ray_manager.set_visibility(visible)
        self.plotter.render()

    def set_sphere_visible(self, visible: bool):
        """
        Toggle sphere hover tracking.

        Args:
            visible: Whether the sphere tracking feature should be enabled.
        """
        self._sphere_visible = visible
        if not visible and hasattr(self, '_sphere_hover_timer'):
            try:
                self._sphere_hover_timer.stop()
                self._sphere_hover_pending_events = 0
            except Exception:
                pass
        if self._cursor_preview is not None:
            # Keep it invisible initially - it will show when mouse picks geometry
            self._cursor_preview.set_visibility(False)
        manager = getattr(self, 'mvat_manager', None)
        if manager is not None:
            try:
                manager.clear_sphere_hover_overlay(reset_context=not visible, render=False)
            except Exception:
                pass
        self._sync_sphere_hover_binding()
        if visible:
            self._request_sphere_hover_refresh()
        try:
            self.plotter.render()
        except Exception:
            pass
        
    def get_scene_median_depth(self, camera_position: np.ndarray) -> float:
        """
        Calculate median depth from camera to scene center.
        
        Used as default depth when z-channel is not available.
        Uses scene_context for unified bounds across all products.
        
        Args:
            camera_position: 3D position of the camera.
            
        Returns:
            float: Estimated median depth to scene.
        """
        try:
            # Prefer scene context for unified bounds across all products
            if self.scene_context.has_any_product():
                return self.scene_context.get_fallback_depth(camera_position)
            else:
                # Fallback to plotter bounds if no products
                bounds = self.plotter.bounds
                center = np.array([
                    (bounds[0] + bounds[1]) / 2,
                    (bounds[2] + bounds[3]) / 2,
                    (bounds[4] + bounds[5]) / 2
                ])
                depth = float(np.linalg.norm(center - camera_position))
                return depth if depth > 0 else 10.0
        except:
            return 10.0  # Default fallback depth

    # ------------------------------------------------------------------
    # Frustum & Thumbnail Management (moved from MVATWindow)
    # ------------------------------------------------------------------
    
    def add_frustums(self, cameras: dict, frustum_scale: float = None,
                     show_thumbnails: bool = None, selected_camera=None,
                     highlighted_paths: list = None, hovered_camera: str = None,
                     context_highlighted_paths: list = None):
        """
        Build and add batched frustums (wireframes) to the plotter.

        Args:
            cameras: Dict mapping image_path -> Camera
            frustum_scale: Scale to use for frustum geometry
            show_thumbnails: Whether to add image plane thumbnails for selected camera
            selected_camera: Camera object to render thumbnail for
            highlighted_paths: List of highlighted camera paths
            hovered_camera: Path of hovered camera
        """
        if frustum_scale is None:
            frustum_scale = self.frustum_scale
        if show_thumbnails is None:
            show_thumbnails = self._show_thumbnails_enabled

        # Remove old frustum actors from plotter before rebuilding
        try:
            self._frustum_manager.remove_from_plotter(self.plotter)
        except Exception:
            pass

        # Clear frustum manager's internal state
        try:
            self._frustum_manager.clear()
        except Exception:
            pass

        # Build merged mesh
        try:
            merged = self._frustum_manager.build_frustum_batch(cameras, scale=frustum_scale)

            if merged is not None and self._show_wireframes_enabled:
                self._frustum_manager.add_to_plotter(self.plotter, line_width=1.5)
                selected_path = selected_camera.image_path if selected_camera else None
                highlighted_paths = highlighted_paths or []
                self._frustum_manager.update_camera_states(
                    selected_path,
                    highlighted_paths,
                    hovered_camera,
                    context_highlighted_paths=context_highlighted_paths,
                )
                self._frustum_manager.mark_modified()
        except Exception:
            pass

        # Thumbnails (lazy): add for selected and highlighted cameras
        # Clear previous thumbnails first
        self.remove_thumbnails()
        if show_thumbnails:
            # Add thumbnail for selected camera first
            if selected_camera is not None:
                try:
                    self._add_thumbnail_for_camera(selected_camera, scale=frustum_scale)
                except Exception:
                    pass

            # Also add thumbnails for highlighted cameras (if any)
            try:
                for hp in (highlighted_paths or []):
                    try:
                        # cameras is the dict passed in mapping path->Camera
                        cam = cameras.get(hp) if isinstance(cameras, dict) else None
                        # Avoid duplicating the selected thumbnail
                        if cam is not None and (selected_camera is None or cam.image_path != selected_camera.image_path):
                            self._add_thumbnail_for_camera(cam, scale=frustum_scale)
                    except Exception:
                        pass
            except Exception:
                pass
        
        # Render update
        try:
            self.plotter.render()
        except Exception:
            pass

    def _add_thumbnail_for_camera(self, camera, scale: float = None):
        """Add a single image-plane thumbnail for the given camera."""
        if scale is None:
            scale = self.frustum_scale
        # Guard against cameras that do not have a Frustum 
        fr = getattr(camera, 'frustum', None)
        if fr is None:
            # No frustum for this camera — skip thumbnail
            return

        try:
            # Remove any existing image actors for this frustum from the plotter
            try:
                for a in list(fr.image_actors.values()):
                    try:
                        self.plotter.remove_actor(a)
                    except Exception:
                        pass
            except Exception:
                pass

            # Clear cached image actors mapping and create a new image plane actor
            fr.image_actors.clear()
            actor = fr.create_image_plane_actor(self.plotter, scale=scale, opacity=self.thumbnail_opacity)
            if actor is not None:
                self.thumbnail_actors.append(actor)
        except Exception as e:
            print(f"Failed to render thumbnail for {getattr(camera, 'image_path', '<unknown>')}: {e}")

    def remove_thumbnails(self):
        """Remove all thumbnail actors from the plotter and clear caches."""
        for actor in list(self.thumbnail_actors):
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.thumbnail_actors.clear()

        # Clear frustum image actor caches (safe attempt). If the viewer's frustum
        # manager exposes cameras, iterate and clear any image actors attached
        # to their Frustum objects, removing actors from the plotter first.
        try:
            frust_cams = getattr(self._frustum_manager, 'cameras', {})
            for _path, cam in frust_cams.items():
                fr = getattr(cam, 'frustum', None)
                if fr is None:
                    continue
                try:
                    for a in list(fr.image_actors.values()):
                        try:
                            self.plotter.remove_actor(a)
                        except Exception:
                            pass
                    fr.image_actors.clear()
                except Exception:
                    pass
        except Exception:
            pass

    def set_thumbnail_opacity(self, opacity: float):
        """Set opacity for any existing thumbnail image plane actors (0.0-1.0)."""
        try:
            self.thumbnail_opacity = float(opacity)
            # Update any existing frustum image actors (if they exist)
            # Try to update through frustum objects if accessible via frustum manager
            for path, cam in getattr(self._frustum_manager, 'cameras', {}).items():
                fr = getattr(cam, 'frustum', None)
                if fr is not None:
                    for actor in list(fr.image_actors.values()):
                        try:
                            actor.GetProperty().SetOpacity(self.thumbnail_opacity)
                        except Exception:
                            pass
            try:
                self.plotter.render()
            except Exception:
                pass
        except Exception:
            pass

    def fit_to_view(self):
        """Fit the current scene in view (wrapper)."""
        try:
            self._cancel_camera_motion()
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception:
            pass

    def reset_view(self):
        """Reset to default isometric view."""
        try:
            self._cancel_camera_motion()
            self.plotter.reset_camera()
            try:
                self.plotter.view_isometric()
            except Exception:
                pass
            self.plotter.render()
        except Exception:
            pass

    def match_camera_perspective(self, camera, focal_distance_ratio: float = 0.2, animate: bool = False):
        """Match the 3D viewer perspective to a camera's viewpoint with optional animation.

        Args:
            camera: Camera object with position, R, K, width/height
            focal_distance_ratio: Fraction of scene diagonal to use as focal distance
            animate: If True, smoothly animate the camera transition (default False)
        """
        try:
            self._cancel_camera_motion()
            # RESTORE: Perspective projection for normal cameras, but ONLY if currently
            # in parallel projection. PyVista's disable_parallel_projection() unconditionally
            # overwrites camera.position using stale parallel_scale, which would snap the
            # viewport to Reset View before the animation start state is captured.
            try:
                if self.plotter.camera.GetParallelProjection():
                    self.plotter.disable_parallel_projection()
            except Exception:
                pass
            
            # EXISTING: Perspective camera alignment
            position = camera.position

            # view direction: camera looks along +Z in camera frame
            view_direction = camera.R.T @ np.array([0, 0, 1])

            # up vector: -Y in camera frame
            up_vector = camera.R.T @ np.array([0, -1, 0])

            # Compute focal distance from scene bounds
            try:
                bounds = self.plotter.bounds
                scene_size = np.sqrt(
                    (bounds[1] - bounds[0])**2 +
                    (bounds[3] - bounds[2])**2 +
                    (bounds[5] - bounds[4])**2
                )
                focal_distance = scene_size * float(focal_distance_ratio)
            except Exception:
                focal_distance = 5.0

            focal_point = position + view_direction * focal_distance

            # Move the viewer slightly back from the camera optical center
            # to avoid exact coincidence between viewer and camera geometry.
            try:
                eps = max(1e-6, scene_size * 1e-4) if 'scene_size' in locals() else 1e-6
            except Exception:
                eps = 1e-6

            viewer_pos = (position - view_direction * eps)
            viewer_focal = viewer_pos + view_direction * focal_distance

            # Match vertical FOV from intrinsics if available
            fov_deg = 50.0  # default
            try:
                if getattr(camera, 'K', None) is not None:
                    fy = camera.K[1, 1]
                    height = camera.height
                    fov_rad = 2 * np.arctan(height / (2 * fy))
                    fov_deg = np.degrees(fov_rad)
                    fov_deg = np.clip(fov_deg, 10, 120)
            except Exception:
                pass

            # ANIMATION: If requested, use the animator
            if animate and self._camera_animator is not None:
                try:
                    self._camera_animator.animate_to_camera_state(
                        viewer_pos, viewer_focal, up_vector, fov_deg
                    )
                    return
                except Exception:
                    # Fall through to instant update on failure
                    pass
            
            # INSTANT UPDATE: Apply camera state immediately
            self.plotter.camera.position = viewer_pos.tolist()
            self.plotter.camera.focal_point = viewer_focal.tolist()
            self.plotter.camera.up = up_vector.tolist()
            self.plotter.camera.view_angle = fov_deg

            try:
                self.plotter.render()
            except Exception:
                pass

            # Reset/adjust clipping range so small visualized rays are not clipped
            try:
                # Let the renderer compute a reasonable clipping range first
                try:
                    self.plotter.renderer.ResetCameraClippingRange()
                except Exception:
                    pass

                # Then enforce a slightly smaller near plane relative to camera-focal distance
                cam = self.plotter.camera
                cam_pos = np.array(cam.position)
                cam_focal = np.array(cam.focal_point)
                dist = float(np.linalg.norm(cam_focal - cam_pos)) if cam_focal is not None else 1.0
                near = max(1e-6, dist * 1e-4)
                far = max(dist * 10.0, near + 1.0)
                try:
                    cam.SetClippingRange(near, far)
                except Exception:
                    # Fallback: try to reset clipping via plotter
                    try:
                        self.plotter.reset_camera()
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception as e:
            print(f"MVATViewer.match_camera_perspective failed: {e}")

    # ------------------------------------------------------------------
    # Convenience / Public API wrappers for MVATWindow
    # ------------------------------------------------------------------
    def enable_wireframes(self, enabled: bool):
        """Enable/disable drawing of frustum wireframes."""
        try:
            self._show_wireframes_enabled = bool(enabled)
            if hasattr(self, '_frustum_manager') and self._frustum_manager is not None:
                try:
                    self._frustum_manager.set_visibility(bool(enabled))
                except Exception:
                    pass
            try:
                self.plotter.render()
            except Exception:
                pass
        except Exception:
            pass

    def enable_thumbnails(self, enabled: bool):
        """Enable/disable thumbnail image plane actors."""
        try:
            self._show_thumbnails_enabled = bool(enabled)
            if enabled:
                # Build thumbnails from the frustum manager's current cameras
                self.remove_thumbnails()
                cameras = getattr(self._frustum_manager, 'cameras', {})
                for cam in cameras.values():
                    self._add_thumbnail_for_camera(cam)
            else:
                self.remove_thumbnails()
            try:
                self.plotter.render()
            except Exception:
                pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Scene Product Visibility Controls
    # ------------------------------------------------------------------
    
    def set_point_clouds_visible(self, visible: bool):
        """Toggle visibility of all point cloud products."""
        self._show_point_clouds = bool(visible)
        self._update_product_visibility_by_type(PointCloudProduct, visible)
    
    def set_meshes_visible(self, visible: bool):
        """Toggle visibility of all mesh products."""
        self._show_meshes = bool(visible)
        self._update_product_visibility_by_type(MeshProduct, visible)

    def set_gaussian_splats_visible(self, visible: bool):
        """Toggle visibility of all 3D Gaussian Splatting products."""
        self._show_gaussian_splats = bool(visible)
        self._update_product_visibility_by_type(GaussianSplattingProduct, visible)

    def show_all_products(self):
        """Show all scene products."""
        self._show_point_clouds = True
        self._show_meshes = True
        self._show_gaussian_splats = True

        # Update menu checkboxes if they exist
        if hasattr(self, '_product_visibility_actions'):
            for action in self._product_visibility_actions.values():
                action.blockSignals(True)
                action.setChecked(True)
                action.blockSignals(False)
        
        # Show all actors
        for actor in self._product_actors.values():
            try:
                actor.SetVisibility(True)
            except Exception:
                pass
        
        try:
            self.plotter.render()
        except Exception:
            pass
    
    def hide_all_products(self):
        """Hide all scene products."""
        self._show_point_clouds = False
        self._show_meshes = False
        self._show_gaussian_splats = False

        # Update menu checkboxes if they exist
        if hasattr(self, '_product_visibility_actions'):
            for action in self._product_visibility_actions.values():
                action.blockSignals(True)
                action.setChecked(False)
                action.blockSignals(False)
        
        # Hide all actors
        for actor in self._product_actors.values():
            try:
                actor.SetVisibility(False)
            except Exception:
                pass
        
        try:
            self.plotter.render()
        except Exception:
            pass
    
    def _update_product_visibility_by_type(self, product_class, visible: bool):
        """Update visibility for all products of a specific type."""
        try:
            for product in self.scene_context.get_products_by_class(product_class):
                actor = self._product_actors.get(product.product_id)
                if actor is not None:
                    try:
                        actor.SetVisibility(bool(visible))
                    except Exception:
                        pass
            self.plotter.render()
        except Exception:
            pass
    
    def set_product_visible(self, product_id: str, visible: bool):
        """
        Set visibility for a specific product by ID.
        
        Args:
            product_id: ID of the product to show/hide.
            visible: Whether the product should be visible.
        """
        actor = self._product_actors.get(product_id)
        if actor is not None:
            try:
                actor.SetVisibility(bool(visible))
                self.plotter.render()
            except Exception:
                pass
    
    def get_product_visible(self, product_id: str) -> bool:
        """
        Get visibility state for a specific product.
        
        Args:
            product_id: ID of the product.
            
        Returns:
            True if visible, False otherwise.
        """
        actor = self._product_actors.get(product_id)
        if actor is not None:
            try:
                return bool(actor.GetVisibility())
            except Exception:
                pass
        return False

    def set_frustum_scale(self, scale: float):
        """Set scale used for frustum geometry."""
        try:
            self.frustum_scale = float(scale)
        except Exception:
            pass

    def update_frustum_states(self, selected_path, highlighted_paths, hovered_camera,
                              context_highlighted_paths=None):
        """Update frustum manager camera states and mark modified."""
        try:
            if hasattr(self, '_frustum_manager') and self._frustum_manager is not None:
                try:
                    self._frustum_manager.update_camera_states(
                        selected_path,
                        highlighted_paths,
                        hovered_camera,
                        context_highlighted_paths=context_highlighted_paths,
                    )
                    self._frustum_manager.mark_modified()
                    self.plotter.render()
                except Exception:
                    pass
        except Exception:
            pass

    def close(self):
        """Clean up the plotter resources."""
        active_tool = getattr(self, '_active_3d_tool', None)
        if active_tool is not None:
            try:
                active_tool.deactivate()
            except Exception:
                pass
            self._active_3d_tool = None

        # Clean up ray manager
        if hasattr(self, '_ray_manager'):
            self._ray_manager.clear()
        if hasattr(self, '_ortho_ray_manager'):
            try:
                self._ortho_ray_manager.remove_from_plotter(self.plotter)
            except Exception:
                pass
            self._ortho_ray_manager.clear()

        # Clean up sphere manager
        if hasattr(self, '_cursor_preview'):
            try:
                self._cursor_preview.remove_from_plotter(self.plotter)
            except Exception:
                pass
            self._cursor_preview.clear()

        manager = getattr(self, 'mvat_manager', None)
        if manager is not None:
            try:
                manager.clear_sphere_hover_overlay(reset_context=True, render=False)
            except Exception:
                pass

        # Remove mouse move observer
        if (hasattr(self, '_mouse_sphere_observer_id') and
            self._mouse_sphere_observer_id is not None):
            try:
                self._unbind_sphere_hover_observer()
            except Exception:
                pass

        if hasattr(self, '_sphere_hover_timer'):
            try:
                self._sphere_hover_timer.stop()
            except Exception:
                pass

        try:
            self._cancel_camera_motion()
        except Exception:
            pass

        controller = getattr(self, '_camera_inertia', None)
        if controller is not None:
            try:
                controller.unbind()
            except Exception:
                pass

        if self.plotter:
            self.plotter.close()

    # ------------------------------------------------------------------
    # Small public helpers used by MVATWindow to avoid reaching into plotter
    # ------------------------------------------------------------------
    def clear_plotter(self):
        """Clear plotter actors (wrapper)."""
        try:
            self.plotter.clear()
        except Exception:
            pass

    def add_axes(self):
        """Add reference axes to the plotter (wrapper)."""
        try:
            self.plotter.add_axes()
        except Exception:
            pass

    def render(self):
        """Render the plotter (wrapper)."""
        try:
            self.plotter.render()
        except Exception:
            pass

    def update(self):
        """Update the plotter (alias to render/update as available)."""
        try:
            # Some PyVista versions have update(); prefer render()

            if hasattr(self.plotter, 'update'):
                try:
                    self.plotter.update()
                    return
                except Exception:
                    pass
            self.plotter.render()
        except Exception:
            pass


    def get_bounds(self):
        """Return the bounding box of all loaded scene products."""
        try:
            return self.scene_context.get_combined_bounds()
        except Exception:
            return None
