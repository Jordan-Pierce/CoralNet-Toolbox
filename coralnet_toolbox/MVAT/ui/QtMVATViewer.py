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

import numpy as np

from pyvistaqt import QtInteractor
from PyQt5.QtCore import Qt, QEvent, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QFrame, QVBoxLayout,
    QWidget, QHBoxLayout, QLabel, QSpinBox,
    QToolBar, QToolButton, QMenu, QAction, QActionGroup, QStackedLayout
)

from coralnet_toolbox.MVAT.core.Ray import CameraRay, BatchedRayManager
from coralnet_toolbox.MVAT.core.Frustum import BatchedFrustumManager
from coralnet_toolbox.MVAT.core.Model import PointCloudProduct, MeshProduct, DEMProduct
from coralnet_toolbox.MVAT.core.SceneContext import SceneContext
from coralnet_toolbox.MVAT.core.SceneProduct import AbstractSceneProduct
from coralnet_toolbox.MVAT.core.constants import RAY_COLOR_SELECTED


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MVATViewer(QFrame):
    focalPointChanged = pyqtSignal(np.ndarray)  # Emits 3D point when focal point is set
    pointSizeChanged = pyqtSignal(int)
    computeIndexMapsToggled = pyqtSignal(bool)
    computeDepthMapsToggled = pyqtSignal(bool)
    primaryTargetChanged = pyqtSignal(str)      # Emits product_id when primary target changes

    def __init__(self, parent=None, point_size=1, show_rays=True):
        super().__init__(parent)
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
        self.plotter.set_background('black')
        self.plotter.disable_anti_aliasing()
        self.plotter.disable_eye_dome_lighting()
        self.plotter.disable_shadows()
        self.plotter.disable_depth_peeling()
        
        # Add observer for Left Click (for Double-Click detection)
        # This is added to the Interactor directly, so it persists
        self.plotter.interactor.AddObserver("LeftButtonPressEvent", self._on_left_press)
        self._last_click_time = 0

        # Scene context replaces single point_cloud with heterogeneous product collection
        self.scene_context = SceneContext()
        # Product actors keyed by product_id
        self._product_actors = {}
        # Legacy filtered actor removed — picking now operates on visible scene actors
        self.point_size = point_size
        self._show_rays_enabled = show_rays
        self._ray_visible = True
        self._ray_manager = BatchedRayManager()
        # Frustum and thumbnail management
        self._frustum_manager = BatchedFrustumManager()
        self.thumbnail_actors = []
        self.thumbnail_opacity = 0.25
        self.frustum_scale = 0.1
        self._show_wireframes_enabled = True
        self._show_thumbnails_enabled = True
        
        # Scene product visibility by type
        self._show_point_clouds = True
        self._show_meshes = True
        self._show_dems = True

        # Top and bottom toolbar widgets (previously exposed for host composition)
        # Refactor: toolbars are owned by the viewer and inserted into its layout
        # to improve encapsulation while keeping attributes for backward compat.
        self.top_toolbar_widget = QWidget()
        self.top_toolbar_layout = QHBoxLayout(self.top_toolbar_widget)
        self.top_toolbar_layout.setContentsMargins(0, 0, 0, 0)

        # Bottom toolbar widget (point-size and opacity controls)
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
            "No 3D data loaded\nDrag a file here to load:\n• Point clouds (.ply, .pcd)\n• Meshes (.obj, .stl)"
        )
        self._placeholder_label.setStyleSheet("color: white; background-color: black; font-size: 14px; padding: 16px;")
        self._placeholder_label.setAlignment(Qt.AlignCenter)
        self._placeholder_label.setAutoFillBackground(True)
        self._placeholder_label.setWordWrap(True)
        self._show_placeholder()  # Show placeholder initially        
        self._stack.addWidget(self._placeholder_label)
        self._stack.setCurrentWidget(self._placeholder_label)
        self.layout.addWidget(self._stack_container)

        # Point size control
        point_size_label = QLabel("Point Size:")
        self.point_size_spinbox = QSpinBox()
        self.point_size_spinbox.setRange(1, 20)
        self.point_size_spinbox.setSingleStep(1)
        self.point_size_spinbox.setValue(self.point_size)
        self.point_size_spinbox.setToolTip("Adjust point cloud point size")
        self.point_size_spinbox.valueChanged.connect(self._on_point_size_spin_changed)

        # Add widgets to bottom layout (stretch, point size)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(point_size_label)
        bottom_layout.addWidget(self.point_size_spinbox)

        # Navigation constants
        self.move_speed = 0.01          # world units per key press
        self.rotate_speed = 0.25        # degrees per key press
        
        self.plotter.interactor.installEventFilter(self)

        # Configure Interaction (Delayed)
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
            self.plotter.view_isometric()
            self.plotter.render()
        except Exception:
            print("Error setting isometric view: ", traceback.format_exc())

    def _set_view_preserve_zoom(self, view_dir, up=None):
        """Set camera looking along view_dir (from camera towards focal point)
        while preserving current camera distance (zoom) and view angle.

        view_dir: iterable-like 3-vector (direction from camera to focal point)
        up: optional up-vector to set on the camera
        """
        try:
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

            cam.position = new_pos.tolist()
            cam.focal_point = fp.tolist()
            if up is not None:
                cam.up = np.array(up, dtype=float).tolist()
            # keep view_angle unchanged -> do not call reset/fit helpers
            try:
                self.plotter.render()
            except Exception:
                pass
            self._update_clipping_range()
        except Exception:
            pass

    def toggle_orthographic(self, state: bool):
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
        
    # --------------------------------------------------------------------------
    # DockWrapper Hooks
    # --------------------------------------------------------------------------
    
    def create_top_toolbar(self) -> QToolBar:
        """Create the top toolbar with the categorized View dropdown menu."""       
        toolbar = QToolBar("3D Viewer Tools")
        toolbar.setMovable(False)
        # The View menu has been moved to the application's menubar / dock menu.
        # Keep the toolbar present for other controls (bottom toolbar mounts widgets).
        return toolbar

    def create_view_menu(self) -> QMenu:
        """Create a standalone QMenu for the viewer that can be attached to a menubar or dock."""
        view_menu = QMenu("View")

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

        action_rays = QAction("Show Rays", self)
        action_rays.setCheckable(True)
        action_rays.setChecked(self._show_rays_enabled)
        action_rays.toggled.connect(self.set_ray_visible)
        view_menu.addAction(action_rays)

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
        
        action_dems = QAction("Show DEMs", self)
        action_dems.setCheckable(True)
        action_dems.setChecked(self._show_dems)
        action_dems.setToolTip("Toggle visibility of DEM products")
        action_dems.toggled.connect(self.set_dems_visible)
        products_menu.addAction(action_dems)
        
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
            'dems': action_dems,
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

    # --------------------------------------------------------------------------
    # Custom Interaction Logic
    # --------------------------------------------------------------------------
    
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
        
        # 2. Apply Custom Trackball Style
        # This cleanly maps Right Drag -> Pan without complex state management.
        # left='rotate' is implied default.
        self.plotter.enable_custom_trackball_style(right='pan')
        print("MVATViewer: Custom trackball style enabled (Right=Pan).")

    def eventFilter(self, obj, event):
        """Intercept key press events."""
        # Swallow ContextMenu events coming from the plotter interactor so the
        # default Qt context menu does not appear on single right-click.
        if event.type() == QEvent.ContextMenu:
            return True

        # Forward key presses to keyPressEvent once; if handled, consume the event
        if event.type() == QEvent.KeyPress:
            self.keyPressEvent(event)
            if event.isAccepted():
                return True

        return super().eventFilter(obj, event)

    def _on_left_press(self, obj, event):
        """Handle Left Click to detect Double Clicks."""
        # Get current time in milliseconds
        current_time = time.time() * 1000
        
        # Get system double click interval
        dc_interval = QApplication.doubleClickInterval()
        
        # Check if this click happened close enough to the last one
        if (current_time - self._last_click_time) < dc_interval:
            self._handle_double_click()
            
        self._last_click_time = current_time
        # Pass event through so standard rotation (Left Drag) still works

    def _handle_double_click(self):
        """
        Perform a pick explicitly against the Scene Geometry to set Focal Point.
        This is the ONLY way to change the focal point via mouse.
        """
        # Perform a normal pick against the scene; let the renderer choose the
        # visible actor under the mouse and return the picked world coordinate.
        try:
            picked_point = self.plotter.pick_mouse_position()
            if picked_point is not None:
                self.set_focal_point(picked_point)
        except Exception:
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

    # ------------------------------------------------------------------
    # Movement methods (each updates clipping)
    # ------------------------------------------------------------------
    def move_forward(self, speed=None):
        """Move camera forward along view direction."""
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
        if speed is None:
            speed = self.move_speed
        pos, fp, _, right, _ = self._get_camera_vectors()
        delta = -right * speed
        self.plotter.camera.position = pos + delta
        self.plotter.camera.focal_point = fp + delta
        self._update_clipping_range()

    def strafe_right(self, speed=None):
        """Move camera right."""
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
        else:
            # Let the parent widget handle other keys
            event.ignore()

    def set_focal_point(self, point):
        """Sets the camera focal point and re-renders."""
        # Animate the transition if desired, or just set it
        self.plotter.camera.focal_point = point
        self.plotter.render()
        self.focalPointChanged.emit(np.asarray(point))
        
    # --------------------------------------------------------------------------
    # Scene Product Loading
    # --------------------------------------------------------------------------
    
    # Supported file extensions for each product type
    _POINT_CLOUD_EXTENSIONS = ['.ply', '.pcd']
    _MESH_EXTENSIONS = ['.obj', '.stl', '.vtk']
    
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
        """Load the dropped 3D file into the viewer."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            file_path = event.mimeData().urls()[0].toLocalFile()
            file_ext = file_path.lower()
            
            # Notify user via status bar if available
            try:
                top = self.window()
                if hasattr(top, 'status_bar'):
                    top.status_bar.showMessage("Loading 3D data...", 0)
            except Exception:
                pass
            
            # Auto-detect product type and load
            product = self._create_product_from_file(file_path)
            
            if product is not None:
                self.add_product(product)
                self.render_scene()
                event.acceptProposedAction()
                
                # Trigger visibility filtering based on the model's current selections
                if self.parent() and hasattr(self.parent(), 'selection_model'):
                    mvat_window = self.parent()
                    model = mvat_window.selection_model
                    selected = model.get_selected_list() if model else []
                    if selected:
                        mvat_window._update_visibility_filter(selected)
            else:
                print(f"Failed to create product from file: {file_path}")
                event.ignore()
                
        except Exception as e:
            print(f"Failed to load 3D file: {e}")
            import traceback
            traceback.print_exc()
            event.ignore()
        finally:
            try:
                top = self.window()
                if hasattr(top, 'status_bar'):
                    top.status_bar.showMessage("3D data load finished.", 3000)
            except Exception:
                pass
            QApplication.restoreOverrideCursor()

    def _create_product_from_file(self, file_path: str) -> 'AbstractSceneProduct':
        """
        Auto-detect file type and create appropriate scene product.
        
        Args:
            file_path: Path to 3D data file.
            
        Returns:
            Scene product instance, or None if type cannot be determined.
        """
        from PyQt5.QtWidgets import QMessageBox
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # STL and OBJ are always meshes by format definition
        if file_ext in ['.stl', '.obj']:
            print(f"📐 {file_ext.upper()} is a mesh-only format, creating MeshProduct")
            return MeshProduct.from_file(file_path)
        
        # PCD is always point cloud
        if file_ext == '.pcd':
            print(f"☁️ PCD is a point-cloud-only format, creating PointCloudProduct")
            return PointCloudProduct.from_file(file_path, point_size=self.point_size)
        
        # PLY can be either - ask user
        if file_ext == '.ply':
            msg = QMessageBox(self)
            msg.setWindowTitle("PLY File Type")
            msg.setText(f"How should '{os.path.basename(file_path)}' be loaded?")
            msg.setInformativeText("PLY files can contain either mesh faces or point cloud vertices.")
            btn_mesh = msg.addButton("Mesh (faces)", QMessageBox.YesRole)
            btn_pc = msg.addButton("Point Cloud", QMessageBox.NoRole)
            msg.setDefaultButton(btn_mesh)
            msg.exec_()
            
            if msg.clickedButton() == btn_mesh:
                print(f"📐 User selected: PLY as MeshProduct")
                try:
                    return MeshProduct.from_file(file_path)
                except ValueError as e:
                    print(f"⚠️ MeshProduct creation failed: {e}, falling back to PointCloudProduct")
                    return PointCloudProduct.from_file(file_path, point_size=self.point_size)
            else:
                print(f"☁️ User selected: PLY as PointCloudProduct")
                return PointCloudProduct.from_file(file_path, point_size=self.point_size)
        
        # VTK - check structure
        if file_ext == '.vtk':
            import pyvista as pv
            temp_mesh = pv.read(file_path)
            # Check if it has non-vertex cells (faces/volumes)
            if temp_mesh.n_cells > 0 and temp_mesh.n_cells < temp_mesh.n_points:
                print(f"📐 VTK detected as mesh (cells < points)")
                return MeshProduct.from_file(file_path)
            else:
                print(f"☁️ VTK detected as point cloud")
                return PointCloudProduct.from_file(file_path, point_size=self.point_size)
        
        return None

    def add_product(self, product: 'AbstractSceneProduct') -> None:
        """
        Add a scene product to the viewer.
        
        Args:
            product: Scene product to add.
        """
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

    def remove_product(self, product_id: str) -> None:
        """
        Remove a scene product from the viewer.
        
        Args:
            product_id: ID of the product to remove.
        """
        # Remove actor if exists
        actor = self._product_actors.pop(product_id, None)
        if actor is not None:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        
        self.scene_context.remove_product(product_id)
        
        # Show placeholder if scene is now empty
        if not self.scene_context.has_any_product():
            self._show_placeholder()
        
        # Update primary target menu
        try:
            self._update_primary_target_menu()
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
        try:
            for product in self.scene_context:
                product_id = product.product_id
                mesh = product.get_render_mesh()
                style = product.get_render_style()
                
                if mesh is None:
                    continue
                
                # Determine visibility based on product type
                should_be_visible = self._get_visibility_for_product(product)
                
                # Get or create actor
                actor = self._product_actors.get(product_id)
                
                if actor is None:
                    actor = self.plotter.add_mesh(
                        mesh,
                        render=False,
                        **style
                    )
                    # Ensure actor opacity matches style (some PyVista versions ignore opacity in add_mesh)
                    try:
                        actor.GetProperty().SetOpacity(style.get('opacity', 1.0))
                    except Exception:
                        pass
                    self._product_actors[product_id] = actor
                else:
                    # Update existing actor's mesh
                    try:
                        actor.GetMapper().SetInputData(mesh)
                        # Update actor opacity from style in case persisted value changed
                        try:
                            actor.GetProperty().SetOpacity(style.get('opacity', 1.0))
                        except Exception:
                            pass
                    except Exception:
                        # Fallback: recreate actor
                        try:
                            self.plotter.remove_actor(actor)
                        except Exception:
                            pass
                        actor = self.plotter.add_mesh(mesh, render=False, **style)
                        try:
                            actor.GetProperty().SetOpacity(style.get('opacity', 1.0))
                        except Exception:
                            pass
                        self._product_actors[product_id] = actor
                
                # Apply visibility setting
                try:
                    actor.SetVisibility(should_be_visible)
                except Exception:
                    pass
            
            self.plotter.render()
            print(f"Rendered {len(self.scene_context)} scene products")
            
        finally:
            QApplication.restoreOverrideCursor()
    
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
        elif isinstance(product, DEMProduct):
            return self._show_dems
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
        
        print(f"🎯 Primary target changed to: {target_id or 'None'}")

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
            if hasattr(self, '_stack'):
                self._stack.setCurrentWidget(self._placeholder_label)
        except Exception:
            pass

    def _hide_placeholder(self):
        """Hide the placeholder and show the plotter interactor."""
        try:
            if hasattr(self, '_stack'):
                self._stack.setCurrentWidget(self.plotter.interactor)
        except Exception:
            pass

    def set_point_size(self, size):
        """Update the point size for point clouds."""
        self.point_size = size
        # Update all point-cloud actors' point size
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
            # Keep original behavior tolerant to failures
            pass

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
        
        Args:
            rays_with_colors: List of (CameraRay, color_tuple) tuples.
                              Colors should be RGB tuples (0-255 or 0-1).
        """
        if not self._show_rays_enabled:
            return
        
        if not rays_with_colors:
            self.clear_ray()
            return
        
        # Build or update batched ray geometry
        if self._ray_manager.ray_actor is not None and self._ray_manager._num_rays == len(rays_with_colors):
            # Update existing rays in-place for better performance
            self._ray_manager.update_ray_endpoints(rays_with_colors)
        else:
            # Build new batched ray geometry
            self._ray_manager.build_ray_batch(rays_with_colors)
            # Add to plotter (removes old actors first)
            self._ray_manager.add_to_plotter(self.plotter, line_width=3)
        
        # Apply visibility state
        self._ray_manager.set_visibility(self._ray_visible)
        
        # Update display
        self.plotter.render()
        
    def clear_ray(self):
        """Remove any displayed ray visualization."""
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
        self.plotter.render()
        
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
                     highlighted_paths: list = None, hovered_camera: str = None):
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

        print(f"🔧 MVATViewer.add_frustums called:")
        print(f"   - Cameras: {len(cameras)}")
        print(f"   - Scale: {frustum_scale}")
        print(f"   - Wireframes enabled: {self._show_wireframes_enabled}")
        print(f"   - Thumbnails enabled: {show_thumbnails}")
        print(f"   - Selected camera: {selected_camera.image_path if selected_camera else 'None'}")
        print(f"   - Highlighted paths: {len(highlighted_paths) if highlighted_paths else 0}")

        # Remove old frustum actors from plotter before rebuilding
        try:
            self._frustum_manager.remove_from_plotter(self.plotter)
            print(f"   ✅ Removed old frustum actors from plotter")
        except Exception as e:
            print(f"   ⚠️ Failed to remove old actors: {e}")
        
        # Clear frustum manager's internal state
        try:
            self._frustum_manager.clear()
            print(f"   ✅ Cleared frustum manager state")
        except Exception as e:
            print(f"   ⚠️ Failed to clear manager: {e}")
            
        # Build merged mesh
        try:
            merged = self._frustum_manager.build_frustum_batch(cameras, scale=frustum_scale)
            print(f"   - Built merged frustum mesh: {merged is not None}")
            
            if merged is not None:
                print(f"   - Mesh stats: n_points={merged.n_points}, n_cells={merged.n_cells}")
                
                if self._show_wireframes_enabled:
                    print(f"   - Adding frustums to plotter...")
                    self._frustum_manager.add_to_plotter(self.plotter, line_width=1.5)
                    selected_path = selected_camera.image_path if selected_camera else None
                    highlighted_paths = highlighted_paths or []
                    self._frustum_manager.update_camera_states(selected_path, highlighted_paths, hovered_camera)
                    self._frustum_manager.mark_modified()
                    print(f"   ✅ Frustums added to plotter successfully")
                else:
                    print(f"   ⚠️ Frustums NOT added - wireframes disabled")
            else:
                print(f"   ⚠️ Merged mesh is None - no frustums to add")
                
        except Exception as e:
            print(f"   ❌ Failed to build frustums: {e}")
            import traceback
            traceback.print_exc()

        # Thumbnails (lazy): add for selected and highlighted cameras
        # Clear previous thumbnails first
        self.remove_thumbnails()
        if show_thumbnails:
            # Add thumbnail for selected camera first
            if selected_camera is not None:
                try:
                    print(f"   - Adding thumbnail for selected camera")
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
            print(f"   ✅ Plotter rendered")
        except Exception as e:
            print(f"   ⚠️ Render failed: {e}")

    def _add_thumbnail_for_camera(self, camera, scale: float = None):
        """Add a single image-plane thumbnail for the given camera."""
        if scale is None:
            scale = self.frustum_scale
        try:
            # Clear any cached image actors for this frustum
            camera.frustum.image_actors.clear()
            actor = camera.frustum.create_image_plane_actor(self.plotter, scale=scale, opacity=self.thumbnail_opacity)
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

        # Clear frustum image actor caches
        try:
            # Cameras are not owned by viewer; safe to attempt if present on parent
            parent = getattr(self, 'parent', None)
        except Exception:
            parent = None
        # We cannot enumerate cameras here safely; callers should clear camera.frustum.image_actors if needed

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
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception:
            pass

    def reset_view(self):
        """Reset to default isometric view."""
        try:
            self.plotter.reset_camera()
            try:
                self.plotter.view_isometric()
            except Exception:
                pass
            self.plotter.render()
        except Exception:
            pass

    def match_camera_perspective(self, camera, focal_distance_ratio: float = 0.2):
        """Match the 3D viewer perspective to a camera's viewpoint.

        Args:
            camera: Camera object with position, R, K, width/height
            focal_distance_ratio: Fraction of scene diagonal to use as focal distance
        """
        try:
            # BRANCH: Orthographic camera
            if getattr(camera, 'is_orthographic', False):
                print(f"🗺️ Switching to orthographic projection for {camera.label}")
                self.view_top()  # Snap to top-down view
                self.plotter.enable_parallel_projection()
                return
            
            # RESTORE: Perspective projection for normal cameras
            try:
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

            self.plotter.camera.position = viewer_pos.tolist()
            self.plotter.camera.focal_point = viewer_focal.tolist()
            self.plotter.camera.up = up_vector.tolist()

            # Match vertical FOV from intrinsics if available
            try:
                if getattr(camera, 'K', None) is not None:
                    fy = camera.K[1, 1]
                    height = camera.height
                    fov_rad = 2 * np.arctan(height / (2 * fy))
                    fov_deg = np.degrees(fov_rad)
                    fov_deg = np.clip(fov_deg, 10, 120)
                    self.plotter.camera.view_angle = fov_deg
            except Exception:
                pass

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
            # Update visibility of existing thumbnail actors
            for actor in getattr(self, 'thumbnail_actors', []):
                try:
                    actor.SetVisibility(bool(enabled))
                except Exception:
                    pass
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
    
    def set_dems_visible(self, visible: bool):
        """Toggle visibility of all DEM products."""
        self._show_dems = bool(visible)
        self._update_product_visibility_by_type(DEMProduct, visible)
    
    def show_all_products(self):
        """Show all scene products."""
        self._show_point_clouds = True
        self._show_meshes = True
        self._show_dems = True
        
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
        self._show_dems = False
        
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

    def update_frustum_states(self, selected_path, highlighted_paths, hovered_camera):
        """Update frustum manager camera states and mark modified."""
        try:
            if hasattr(self, '_frustum_manager') and self._frustum_manager is not None:
                try:
                    self._frustum_manager.update_camera_states(selected_path, highlighted_paths, hovered_camera)
                    self._frustum_manager.mark_modified()
                except Exception:
                    pass
        except Exception:
            pass

    def close(self):
        """Clean up the plotter resources."""
        # Clean up ray manager
        if hasattr(self, '_ray_manager'):
            self._ray_manager.clear()
        
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
        """Return the plotter bounds (wrapper)."""
        try:
            return self.plotter.bounds
        except Exception:
            return None

    def update_camera_appearance(self, camera, opacity=None):
        """Update camera visual appearance via its helper method using this viewer's plotter."""
        try:
            if camera is None:
                return
            if opacity is not None:
                camera.update_appearance(self.plotter, opacity=opacity)
            else:
                camera.update_appearance(self.plotter)
        except Exception:
            pass