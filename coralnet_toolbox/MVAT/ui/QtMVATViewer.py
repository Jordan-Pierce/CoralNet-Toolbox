import time

import numpy as np

import vtk
from pyvistaqt import QtInteractor

from PyQt5.QtWidgets import QFrame, QVBoxLayout
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from coralnet_toolbox.MVAT.core.Ray import CameraRay
from coralnet_toolbox.MVAT.core.Model import PointCloud
from coralnet_toolbox.MVAT.core.Ray import BatchedRayManager
from coralnet_toolbox.MVAT.core.constants import (RAY_COLOR_SELECTED)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MVATInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """
    Custom VTK Interaction Style.
    - Right Click: Pan (Overrides default Zoom)
    - Left Click: Rotate (Inherited)
    - Scroll: Zoom (Inherited)
    """
    def __init__(self, parent=None):
        # We must initialize the parent class. 
        # Note: We do NOT need to add observers here for RightButton.
        # Overriding OnRightButtonDown/Up is sufficient.
        pass

    def OnRightButtonDown(self):
        """Override the default Right Button behavior (Zoom) to Pan."""
        self.StartPan()

    def OnRightButtonUp(self):
        """End the Pan interaction."""
        self.EndPan()
        

class MVATViewer(QFrame):
    """
    3D Viewer with custom mouse interactions:
    - Left Drag: Rotate (Default)
    - Scroll: Zoom (Default)
    - Right Drag: Pan (Custom Override)
    - Double Left Click: Set Focal Point (Custom Observer)
    """
    def __init__(self, parent=None, point_size=1, show_rays=True):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setAcceptDrops(True)
        
        # Layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create PyVista QtInteractor
        self.plotter = QtInteractor(self)
        self.plotter.set_background('white')
        self.plotter.enable_trackball_style()
        
        # --- CUSTOM INTERACTION SETUP ---
        
        # 1. Apply Custom Style (Right Click = Pan)
        # We replace the default style with our custom subclass
        self.style = MVATInteractorStyle()
        self.plotter.interactor.SetInteractorStyle(self.style)
        
        # 2. Double Click Handler (Observer)
        # We listen to the raw VTK LeftButtonPressEvent to detect double clicks.
        # This runs alongside the style's rotation logic.
        self.plotter.interactor.AddObserver("LeftButtonPressEvent", self._on_left_press)
        self._last_click_time = 0

        # Point cloud and Ray management
        self.point_cloud = None
        self._scene_actor = None
        
        self.point_size = point_size
        
        self._show_rays_enabled = show_rays
        self._ray_visible = True
        self._ray_manager = BatchedRayManager()
        
        self.layout.addWidget(self.plotter.interactor)

    # --------------------------------------------------------------------------
    # Custom Interaction Logic
    # --------------------------------------------------------------------------

    def _on_left_press(self, obj, event):
        """Handle Left Click to detect Double Clicks."""
        # Get current time in milliseconds
        current_time = time.time() * 1000
        
        # Get system double click interval (usually ~500ms)
        dc_interval = QApplication.doubleClickInterval()
        
        # Check if this click happened close enough to the last one
        if (current_time - self._last_click_time) < dc_interval:
            self._handle_double_click()
            
        self._last_click_time = current_time
        # Note: We do NOT abort the event here. We let it pass through so 
        # VTK can still start the rotation (Left Drag) logic if this turns out 
        # to be a drag instead of a click.

    def _handle_double_click(self):
        """
        Perform a pick explicitly against the Scene Geometry.
        Ignores frustums, rays, and other UI elements.
        """
        if self._scene_actor is None:
            return

        # 1. Temporarily disable pickability for everything EXCEPT the scene actor.
        # This ensures the picking ray passes through frustums to hit the mesh/cloud behind.
        restore_list = []
        for actor in self.plotter.actors.values():
            if actor != self._scene_actor and actor.GetPickable():
                actor.SetPickable(False)
                restore_list.append(actor)
        
        try:
            # 2. Perform the hardware pick at current mouse position
            # Since only the scene is pickable, this will hit the scene or nothing.
            picked_point = self.plotter.pick_mouse_position()
            
            # 3. Update focal point if we hit the scene
            if picked_point is not None:
                self.set_focal_point(picked_point)
                
        finally:
            # 4. Restore pickability for all other actors
            for actor in restore_list:
                try:
                    actor.SetPickable(True)
                except:
                    pass

    def _on_right_press(self, obj, event):
        """Force Right Click to Pan instead of Zoom."""
        # Get the interactor style (TrackballCamera)
        style = self.plotter.interactor.GetInteractorStyle()
        
        # Manually trigger the 'Pan' state on the style
        style.StartPan()
        
        # ABORT the event so the default interaction (Zoom) doesn't fire
        # This requires the observer to be added with high priority (1.0)
        self.plotter.interactor.SetAbortRender(1) 
        # Note: In some VTK bindings, you use event.AbortFlag = 1, 
        # but in Python wrappers, stopping propagation can be tricky.
        # If both Pan and Zoom happen, 'StartPan' usually overrides standard logic 
        # if called explicitly.

    def _on_right_release(self, obj, event):
        """End the Pan state."""
        style = self.plotter.interactor.GetInteractorStyle()
        style.EndPan()

    def set_focal_point(self, point):
        """Sets the camera focal point and re-renders."""
        # Animate the transition if desired, or just set it
        self.plotter.camera.focal_point = point
        self.plotter.render()

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
            # Add to plotter and reset camera
            self.add_point_cloud()
            self.plotter.reset_camera()
            event.acceptProposedAction()
            # Auto-select first camera after point cloud import
            if self.parent() and hasattr(self.parent(), '_auto_select_first_camera'):
                self.parent()._auto_select_first_camera()
        except Exception as e:
            print(f"Failed to load 3D file: {e}")
            event.ignore()
        finally:
            QApplication.restoreOverrideCursor()

    def add_point_cloud(self):
        """Add the point cloud to the plotter and capture its actor."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if self.point_cloud is not None:
                # Capture the set of actors BEFORE adding
                previous_actors = set(self.plotter.actors.values())
                
                self.point_cloud.add_to_plotter(self.plotter)
                
                # Identify the NEW actor (the scene geometry)
                current_actors = set(self.plotter.actors.values())
                new_actors = current_actors - previous_actors
                
                if new_actors:
                    self._scene_actor = list(new_actors)[0]
        finally:
            QApplication.restoreOverrideCursor()

    def set_point_cloud_visible(self, visible):
        """Set visibility of the point cloud actor."""
        if self.point_cloud is not None:
            self.point_cloud.set_visible(visible)

    def set_point_size(self, size):
        """Update the point size for point clouds."""
        self.point_size = size
        # If point cloud is loaded, update it
        if self.point_cloud is not None:
            self.point_cloud.set_point_size(size)
            self.plotter.render()  # Force re-render

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