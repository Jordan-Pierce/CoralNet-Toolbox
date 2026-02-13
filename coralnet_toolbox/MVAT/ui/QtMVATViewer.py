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
import vtk
from pyvistaqt import QtInteractor
from PyQt5.QtCore import Qt, QEvent, QTimer
from PyQt5.QtWidgets import QApplication, QFrame, QVBoxLayout

from coralnet_toolbox.MVAT.core.Ray import CameraRay, BatchedRayManager
from coralnet_toolbox.MVAT.core.Model import PointCloud
from coralnet_toolbox.MVAT.core.constants import RAY_COLOR_SELECTED


class MVATViewer(QFrame):
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
        self.plotter.set_background('white')

        # Optimizations
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

        self.layout.addWidget(self.plotter.interactor)

        # Navigation constants
        self.move_speed = 0.01          # world units per key press
        self.rotate_speed = 0.25        # degrees per key press
        
        self.plotter.interactor.installEventFilter(self)

        # Configure Interaction (Delayed)
        # We run this on a timer to ensure it happens AFTER the parent window
        # has run its setup (like enable_point_picking), so we can override it.
        QTimer.singleShot(100, self._configure_interaction)

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

    # ------------------------------------------------------------------
    # Key event handling
    # ------------------------------------------------------------------
    def keyPressEvent(self, event):
        """Handle key presses for WASD movement and QE rotation."""
        key = event.key()
        if key == Qt.Key_W:
            self.move_forward()
            event.accept()
        elif key == Qt.Key_S:
            self.move_backward()
            event.accept()
        elif key == Qt.Key_A:
            self.strafe_left()
            event.accept()
        elif key == Qt.Key_D:
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
            # Create PointCloud instance
            self.point_cloud = PointCloud.from_file(file_path, point_size=self.point_size)
            # Warm up GPU cache (no rendering yet)
            self.add_point_cloud()
            event.acceptProposedAction()
            
            # Trigger visibility filtering for the selected camera
            # This ensures the cloud transitions directly to filtered state
            if self.parent() and hasattr(self.parent(), 'selected_camera'):
                mvat_window = self.parent()
                if mvat_window.selected_camera:
                    # Always start with at least the selected camera
                    selected_path = mvat_window.selected_camera.image_path
                    highlighted_paths = [selected_path]
                    
                    # Add any other highlighted cameras
                    for cam in mvat_window.highlighted_cameras:
                        if cam.image_path not in highlighted_paths:
                            highlighted_paths.append(cam.image_path)
                    
                    # Trigger visibility filtering
                    mvat_window._update_visibility_filter(highlighted_paths)
        except Exception as e:
            print(f"Failed to load 3D file: {e}")
            event.ignore()
        finally:
            QApplication.restoreOverrideCursor()

    def add_point_cloud(self):
        """Warm up GPU cache for the point cloud.
        
        Does NOT add any actors to the plotter. The cloud will only be
        visualized through update_point_cloud_subset() which creates
        the single _filtered_actor on demand.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if self.point_cloud is not None:
                # Trigger GPU cache upload (Compute Space)
                self.point_cloud._ensure_gpu_cache()
                print("⚡ Point cloud GPU cache ready (no rendering yet)")
        finally:
            QApplication.restoreOverrideCursor()

    def set_point_size(self, size):
        """Update the point size for point clouds."""
        self.point_size = size
        # Update filtered actor if it exists
        if self._filtered_actor is not None:
            self._filtered_actor.GetProperty().SetPointSize(size)
            self.plotter.render()
    
    def update_point_cloud_subset(self, indices):
        """
        Update the viewer to show only a subset of points using GPU-accelerated extraction.
        
        Uses the hybrid \"Compute vs. Render\" architecture:
        - GPU slicing happens in Model.py (fast CUDA indexing)
        - Mapper swap happens here (minimal rendering overhead)
        
        Args:
            indices: Array of point indices to show. 
                    - None: show full cloud
                    - Empty list/array: hide cloud (show nothing)
                    - Array: show filtered subset
        """
        if self.point_cloud is None:
            return
        
        start_time = time.time()
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Extract subset mesh using GPU-accelerated slicing
            # This is where the magic happens - Model.py uses CUDA for fast indexing
            subset_mesh = self.point_cloud.get_subset_data(indices)
            
            # Handle empty results
            if subset_mesh is None or subset_mesh.n_points == 0:
                if self._filtered_actor is not None:
                    self._filtered_actor.SetVisibility(False)
                self.plotter.render()
                total_time = time.time() - start_time
                print(f"⏱️ update_point_cloud_subset: Hidden (empty subset) in {total_time:.3f}s")
                return
            
            # First time: Create the actor
            if self._filtered_actor is None:
                if 'RGB' in subset_mesh.point_data:
                    self._filtered_actor = self.plotter.add_mesh(
                        subset_mesh,
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
                        subset_mesh,
                        color='black',
                        point_size=self.point_size,
                        style='points',
                        render_points_as_spheres=False,
                        lighting=False,
                        render=False
                    )
                
                # Apply LOD optimization
                try:
                    self._filtered_actor.GetProperty().SetLODRenderThreshold(1000)
                except AttributeError:
                    pass
                
                update_type = "Initial Build"
            else:
                # Subsequent times: Swap the mapper input (FAST!)
                # This is the key optimization - no actor recreation
                self._filtered_actor.GetMapper().SetInputData(subset_mesh)
                self._filtered_actor.SetVisibility(True)
                update_type = "Mapper Swap"
            
            # Render the updated scene
            self.plotter.render()
            
            total_time = time.time() - start_time
            print(f"⏱️ update_point_cloud_subset ({update_type}): {subset_mesh.n_points:,} pts in {total_time:.3f}s")
            
        finally:
            QApplication.restoreOverrideCursor()

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
        
        # Build batched ray geometry
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

    def close(self):
        """Clean up the plotter resources."""
        # Clean up ray manager
        if hasattr(self, '_ray_manager'):
            self._ray_manager.clear()
        
        if self.plotter:
            self.plotter.close()