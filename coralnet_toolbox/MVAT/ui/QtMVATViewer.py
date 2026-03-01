"""
MVAT Viewer Widget

A PyVista-based 3D viewer for point clouds and camera frustums.
Customized interaction style:
- Left Double-Click: Set Focal Point (Pick)
- Left Drag: Rotate
- Right Drag: Pan (Strictly enforced via PyVista custom style)
- Scroll Wheel: Zoom
"""

import time
import numpy as np
from pyvistaqt import QtInteractor
from PyQt5.QtCore import Qt, QEvent, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QFrame, QVBoxLayout,
    QWidget, QHBoxLayout, QLabel, QSlider, QSpinBox,
    QToolBar, QToolButton, QMenu, QAction, QStackedLayout
)


from coralnet_toolbox.MVAT.core.Ray import CameraRay, BatchedRayManager
from coralnet_toolbox.MVAT.core.Frustum import BatchedFrustumManager
from coralnet_toolbox.MVAT.core.Model import PointCloud
from coralnet_toolbox.MVAT.core.constants import RAY_COLOR_SELECTED


class MVATViewer(QFrame):
    focalPointChanged = pyqtSignal(np.ndarray)  # Emits 3D point when focal point is set
    opacityChanged = pyqtSignal(int)            # percentage 0-100
    pointSizeChanged = pyqtSignal(int)
    computeIndexMapsToggled = pyqtSignal(bool)
    computeDepthMapsToggled = pyqtSignal(bool)

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
        self.plotter.set_background('white')
        self.plotter.disable_anti_aliasing()
        self.plotter.disable_eye_dome_lighting()
        self.plotter.disable_shadows()
        self.plotter.disable_depth_peeling()
        
        # Add observer for Left Click (for Double-Click detection)
        # This is added to the Interactor directly, so it persists
        self.plotter.interactor.AddObserver("LeftButtonPressEvent", self._on_left_press)
        self._last_click_time = 0

        # Point cloud and ray management
        self.point_cloud = None
        self._filtered_actor = None
        self.point_size = point_size
        self._show_rays_enabled = show_rays
        self._ray_visible = True
        self._ray_manager = BatchedRayManager()
        # Frustum and thumbnail management
        self._frustum_manager = BatchedFrustumManager()
        self.thumbnail_actors = []
        self.thumbnail_opacity = 0.0
        self.frustum_scale = 0.1
        self._show_wireframes_enabled = True
        self._show_thumbnails_enabled = True

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

        # Placeholder shown when no point cloud present
        self._placeholder_label = QLabel("No point cloud loaded")
        self._placeholder_label.setAlignment(Qt.AlignCenter)
        self._placeholder_label.setWordWrap(True)
        self._placeholder_label.setStyleSheet("color: #666;")
        self._stack.addWidget(self._placeholder_label)

        # Start showing placeholder by default
        self._stack.setCurrentWidget(self._placeholder_label)

        self.layout.addWidget(self._stack_container)

        # Opacity control (for thumbnail/frustum opacity)
        opacity_label = QLabel("Opacity:")
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(25)
        self.opacity_slider.setFixedWidth(120)
        self.opacity_slider.setToolTip("Adjust thumbnail opacity")
        self.opacity_slider.valueChanged.connect(lambda v: self.opacityChanged.emit(v))
        self.opacity_slider.valueChanged.connect(lambda v: self.set_thumbnail_opacity(v / 100.0))

        # Point size control
        point_size_label = QLabel("Point Size:")
        self.point_size_spinbox = QSpinBox()
        self.point_size_spinbox.setRange(1, 20)
        self.point_size_spinbox.setSingleStep(1)
        self.point_size_spinbox.setValue(self.point_size)
        self.point_size_spinbox.setToolTip("Adjust point cloud point size")
        self.point_size_spinbox.valueChanged.connect(self._on_point_size_spin_changed)

        # Initialize thumbnail opacity from slider value
        try:
            self.set_thumbnail_opacity(self.opacity_slider.value() / 100.0)
        except Exception:
            pass
        
        # Add widgets to bottom layout (left aligned: opacity, stretch, point size)
        bottom_layout.addWidget(opacity_label)
        bottom_layout.addWidget(self.opacity_slider)
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
        """Set camera to look down from the Z-axis."""
        self.plotter.camera_position = 'xy'
        self.plotter.render()

    def view_front(self):
        """Set camera to look from the Y-axis."""
        self.plotter.camera_position = 'xz'
        self.plotter.render()

    def view_side(self):
        """Set camera to look from the X-axis."""
        self.plotter.camera_position = 'yz'
        self.plotter.render()

    def view_isometric(self):
        """Set camera to a standard 3D isometric angle."""
        self.plotter.view_isometric()
        self.plotter.render()

    def toggle_orthographic(self, state: bool):
        """Toggle between perspective and orthographic projection."""
        if state:
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()
        self.plotter.render()
        
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
    
    def create_view_toolbar(self) -> QToolBar:
        """
        Create a toolbar containing a button with the View menu.
        This can be added directly to the dock's toolbar area.
        """
        toolbar = QToolBar("View Menu")
        toolbar.setMovable(False)
        
        # Create the menu (you can reuse the existing create_view_menu logic)
        view_menu = self.create_view_menu()
        
        # Create a tool button that shows the menu
        view_button = QToolButton()
        view_button.setText("View")
        view_button.setMenu(view_menu)
        view_button.setPopupMode(QToolButton.InstantPopup)
        view_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        
        toolbar.addWidget(view_button)
        return toolbar

    def create_view_menu(self) -> QMenu:
        """Create a standalone QMenu for the viewer that can be attached to a menubar or dock."""
        view_menu = QMenu("View")

        # Top actions: Fit and Reset
        action_fit = QAction("Fit All", self)
        action_fit.setShortcut("F")
        action_fit.triggered.connect(self.fit_to_view)
        view_menu.addAction(action_fit)

        action_reset = QAction("Reset View", self)
        action_reset.setShortcut("R")
        action_reset.triggered.connect(self.reset_view)
        view_menu.addAction(action_reset)

        view_menu.addSeparator()

        # Camera angles
        view_menu.addAction("Top (XY)", self.view_top)
        view_menu.addAction("Front (XZ)", self.view_front)
        view_menu.addAction("Side (YZ)", self.view_side)
        view_menu.addAction("Isometric", self.view_isometric)

        view_menu.addSeparator()

        # Orthographic projection
        action_ortho = QAction("Orthographic Projection", self)
        action_ortho.setCheckable(True)
        action_ortho.toggled.connect(self.toggle_orthographic)
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
        
        # Removed the old "Show Full Point Cloud" toggle - viewer now always
        # renders the full point cloud. Add a toggle to control whether index
        # map computation runs in the background.
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
        if self._filtered_actor is None:
            return

        # 1. Temporarily disable pickability for everything EXCEPT the filtered actor.
        restore_list = []
        for actor in self.plotter.actors.values():
            if actor != self._filtered_actor and actor.GetPickable():
                actor.SetPickable(False)
                restore_list.append(actor)
        
        try:
            # 2. Perform pick
            picked_point = self.plotter.pick_mouse_position()
            
            # 3. Update focal point if hit
            if picked_point is not None:
                self.set_focal_point(picked_point)
                
        finally:
            # 4. Restore pickability
            for actor in restore_list:
                try:
                    actor.SetPickable(True)
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
        - Q/E: keep existing rotate behavior
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
        else:
            # Let the parent widget handle keys like R, F, Escape, etc.
            event.ignore()

    def set_focal_point(self, point):
        """Sets the camera focal point and re-renders."""
        # Animate the transition if desired, or just set it
        self.plotter.camera.focal_point = point
        self.plotter.render()
        self.focalPointChanged.emit(np.asarray(point))
        
    # --------------------------------------------------------------------------
    # Point Cloud Loading and Subsetting
    # --------------------------------------------------------------------------

    def dragEnterEvent(self, event):
        """Accept drag if a single 3D file is being dragged."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1:
                file_path = urls[0].toLocalFile()
                if any(file_path.lower().endswith(ext) for ext in ['.ply', '.stl', '.obj', '.vtk', '.pcd']):
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
            # Notify user via status bar if available
            try:
                top = self.window()
                if hasattr(top, 'status_bar'):
                    top.status_bar.showMessage("Loading point cloud...", 0)
            except Exception:
                pass
            # Create PointCloud instance
            self.point_cloud = PointCloud.from_file(file_path, point_size=self.point_size)
            # Hide placeholder now that a point cloud exists
            try:
                self._hide_placeholder()
            except Exception:
                pass
            # Render the full point cloud immediately
            self.add_point_cloud()
            event.acceptProposedAction()
            
            # Trigger visibility filtering based on the model's current selections
            if self.parent() and hasattr(self.parent(), 'selection_model'):
                mvat_window = self.parent()
                model = mvat_window.selection_model

                selected = model.get_selected_list() if model else []
                if selected:
                    # The model already guarantees the active camera is in this set!
                    mvat_window._update_visibility_filter(selected)
        except Exception as e:
            print(f"Failed to load 3D file: {e}")
            event.ignore()
        finally:
            try:
                top = self.window()
                if hasattr(top, 'status_bar'):
                    top.status_bar.showMessage("Point cloud load finished.", 3000)
            except Exception:
                pass
            QApplication.restoreOverrideCursor()

    def add_point_cloud(self):
        """Render the full point cloud immediately into the scene."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if self.point_cloud is None:
                return

            mesh = self.point_cloud.get_mesh()
            if mesh is None:
                return

            # Create or replace the actor for the full cloud
            if self._filtered_actor is None:
                if 'RGB' in mesh.point_data:
                    self._filtered_actor = self.plotter.add_mesh(
                        mesh,
                        scalars='RGB',
                        rgb=True,
                        point_size=self.point_size,
                        style='points',
                        render_points_as_spheres=False,
                        lighting=False,
                        render=False
                    )
                else:
                    self._filtered_actor = self.plotter.add_mesh(
                        mesh,
                        color='black',
                        point_size=self.point_size,
                        style='points',
                        render_points_as_spheres=False,
                        lighting=False,
                        render=False
                    )
            else:
                try:
                    self._filtered_actor.GetMapper().SetInputData(mesh)
                    self._filtered_actor.SetVisibility(True)
                except Exception:
                    # Fallback: recreate actor
                    self._filtered_actor = self.plotter.add_mesh(mesh, 
                                                                 color='black', 
                                                                 point_size=self.point_size, 
                                                                 style='points', 
                                                                 render=False)

            # Hide placeholder and render
            try:
                self._hide_placeholder()
            except Exception:
                pass
            self.plotter.render()
            print("Rendered full point cloud into viewer")
        finally:
            QApplication.restoreOverrideCursor()

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
        # Update filtered actor if it exists
        if self._filtered_actor is not None:
            self._filtered_actor.GetProperty().SetPointSize(size)
            self.plotter.render()

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
        
        Args:
            camera_position: 3D position of the camera.
            
        Returns:
            float: Estimated median depth to scene.
        """
        try:
            if self.point_cloud is not None:
                # Use point cloud center
                center = np.array(self.point_cloud.get_mesh().center)
                return float(np.linalg.norm(center - camera_position))
            else:
                # Use scene bounds center
                bounds = self.plotter.bounds
                center = np.array([
                    (bounds[0] + bounds[1]) / 2,
                    (bounds[2] + bounds[3]) / 2,
                    (bounds[4] + bounds[5]) / 2
                ])
                depth = float(np.linalg.norm(center - camera_position))
                return depth if depth > 0 else 10.0  # Fallback
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

        # Thumbnails (lazy): only for selected camera to limit actors
        # Clear previous thumbnails first
        self.remove_thumbnails()
        if show_thumbnails and selected_camera is not None:
            print(f"   - Adding thumbnail for selected camera")
            self._add_thumbnail_for_camera(selected_camera, scale=frustum_scale)
        
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