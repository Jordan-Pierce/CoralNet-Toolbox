import time

import numpy as np

import vtk
import pyvista as pv
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
        self.plotter = QtInteractor(self, point_smoothing=False)
        self.plotter.set_background('white')
        self.plotter.enable_trackball_style()
        
        # Optimizations?
        self.plotter.disable_anti_aliasing()  # Disable default AA for better performance with large point clouds
        self.plotter.disable_eye_dome_lighting()  # Disable EDL (can cause issues with large clouds)
        self.plotter.enable_parallel_projection()  # Optional: Parallel projection for architectural scenes
        self.plotter.disable_shadows()  # Disable shadows for better performance (optional)      
        self.plotter.disable_depth_peeling()  # Disable depth peeling (transparency) for better performance (optional) 
        
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
        self._filtered_actor = None  # Separate actor for filtered point cloud
        self._filtered_mesh = None  # Persistent mesh for in-place updates
        self._filtered_mode = False  # Track if we're in filtered mode
        
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
        """Re-add the stored point cloud to the plotter."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if self.point_cloud is not None:
                # 1. Capture the actor returned by add_to_plotter
                actor = self.point_cloud.add_to_plotter(self.plotter)
                
                # 2. Add LOD optimization here
                if actor:
                    # Note: Ensure your VTK version supports this specific property method
                    # or use actor.SetEnableLOD(True) if using vtkLODActor
                    try:
                        # Your specific snippet:
                        actor.GetProperty().SetLODRenderThreshold(1000)  # ms
                    except AttributeError:
                        # Fallback for standard PyVista actors if property doesn't exist
                        pass
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
        # Also update filtered actor if in filtered mode
        if self._filtered_actor is not None:
            self._filtered_actor.GetProperty().SetPointSize(size)
            self.plotter.render()
    
    # def update_point_cloud_subset(self, indices):
    #     """
    #     Update the viewer to show only a subset of points based on visibility indices.
        
    #     Uses in-place mesh updates to avoid expensive actor teardown/rebuild cycles.
    #     Only creates/destroys actors when switching between full/filtered modes or
    #     when point count changes significantly.
        
    #     Args:
    #         indices (np.ndarray or None): Array of point indices to show. 
    #                                      If None or empty, shows full cloud.
    #     """
    #     if self.point_cloud is None:
    #         return
        
    #     import time
    #     start_time = time.time()
        
    #     QApplication.setOverrideCursor(Qt.WaitCursor)
    #     try:
    #         # If indices is None or empty, show full cloud
    #         if indices is None or len(indices) == 0:
    #             # Switch back to full cloud mode
    #             if self._filtered_mode:
    #                 self._filtered_mode = False
                    
    #                 # Remove filtered actor
    #                 if self._filtered_actor is not None:
    #                     try:
    #                         self.plotter.remove_actor(self._filtered_actor)
    #                     except:
    #                         pass
    #                     self._filtered_actor = None
    #                     self._filtered_mesh = None
                    
    #                 # Show full cloud
    #                 self.set_point_cloud_visible(True)
    #                 self.plotter.render()
    #             return
            
    #         # Extract subset mesh
    #         subset_mesh = self.point_cloud.extract_subset(indices)
            
    #         if subset_mesh is None or subset_mesh.n_points == 0:
    #             print("Warning: Filtered subset is empty")
    #             if self._filtered_actor is not None:
    #                 try:
    #                     self.plotter.remove_actor(self._filtered_actor)
    #                 except:
    #                     pass
    #                 self._filtered_actor = None
    #                 self._filtered_mesh = None
    #             self._filtered_mode = False
    #             self.plotter.render()
    #             return
            
    #         # Hide the full point cloud actor (only once when entering filtered mode)
    #         if not self._filtered_mode:
    #             self.set_point_cloud_visible(False)
    #             self._filtered_mode = True
            
    #         # Check if we need to create the filtered actor for the first time
    #         # OR if the point count changed (can't reuse buffer)
    #         need_rebuild = (
    #             self._filtered_actor is None or 
    #             self._filtered_mesh is None or
    #             self._filtered_mesh.n_points != subset_mesh.n_points
    #         )
            
    #         if need_rebuild:
    #             # REBUILD PATH: Remove old actor and create new one
    #             if self._filtered_actor is not None:
    #                 try:
    #                     self.plotter.remove_actor(self._filtered_actor)
    #                 except:
    #                     pass
                
    #             # Store reference to mesh for in-place updates
    #             self._filtered_mesh = subset_mesh
                
    #             # Create new actor
    #             if 'RGB' in subset_mesh.point_data:
    #                 self._filtered_actor = self.plotter.add_mesh(
    #                     subset_mesh,
    #                     scalars='RGB',
    #                     rgb=True,
    #                     point_size=self.point_size,
    #                     render=False  # Defer render until after LOD setup
    #                 )
    #             else:
    #                 point_size = self.point_size if subset_mesh.n_cells == 0 else None
    #                 self._filtered_actor = self.plotter.add_mesh(
    #                     subset_mesh,
    #                     color='black',
    #                     point_size=point_size,
    #                     render=False
    #                 )
                
    #             # Apply LOD optimization
    #             if self._filtered_actor:
    #                 try:
    #                     self._filtered_actor.GetProperty().SetLODRenderThreshold(1000)
    #                 except AttributeError:
    #                     pass
                
    #             render_time = time.time() - start_time
    #             print(f"⏱️ Rendered {subset_mesh.n_points:,} points in viewer (rebuild) in {render_time:.3f}s")
            
    #         else:
    #             # IN-PLACE UPDATE PATH (FAST!)
    #             # Reuse existing actor, just swap the data
                
    #             # Update point coordinates
    #             self._filtered_mesh.points = subset_mesh.points
                
    #             # Update colors if present
    #             if 'RGB' in subset_mesh.point_data and 'RGB' in self._filtered_mesh.point_data:
    #                 self._filtered_mesh['RGB'] = subset_mesh['RGB']
                
    #             # Mark geometry as modified so VTK knows to re-upload to GPU
    #             self._filtered_mesh.GetPoints().Modified()
                
    #             # If we have color data, mark that as modified too
    #             if 'RGB' in self._filtered_mesh.point_data:
    #                 self._filtered_mesh.GetPointData().GetScalars().Modified()
                
    #             render_time = time.time() - start_time
    #             print(f"⏱️ Rendered {subset_mesh.n_points:,} points in viewer (in-place update) in {render_time:.3f}s")
            
    #         # Single render call after all updates
    #         self.plotter.render()
            
    #     finally:
    #         QApplication.restoreOverrideCursor()
            
    def update_point_cloud_subset(self, indices):
        """
        Update the visible point cloud subset using in-place data replacement.
        Includes timing profile for the viewer update/render.
        """
        if self.point_cloud is None:
            return

        start_time = time.time()
            
        # --- CASE 1: Reverting to Full Cloud ---
        if indices is None:
            if self._filtered_mode:
                self._filtered_mode = False
                if self._filtered_actor:
                    self._filtered_actor.SetVisibility(False)
                self.set_point_cloud_visible(True)
                self.plotter.render()
            return

        # --- CASE 2: Switching to Filtered Mode ---
        if not self._filtered_mode:
            self.set_point_cloud_visible(False)
            self._filtered_mode = True
            
        # Get raw data arrays (Timing for this is handled inside get_subset_data)
        points, point_data = self.point_cloud.get_subset_data(indices)
        
        if points is None or len(points) == 0:
            if self._filtered_actor:
                self._filtered_actor.SetVisibility(False)
            self.plotter.render()
            return

        update_type = ""

        # --- CASE 3: Initial Creation (Rebuild) ---
        if self._filtered_actor is None:
            update_type = "Rebuild"
            
            # Create mesh container
            self._filtered_mesh = pv.PolyData(points)
            for name, data in point_data.items():
                self._filtered_mesh.point_data[name] = data
            
            # Create actor
            if 'RGB' in point_data:
                self._filtered_actor = self.plotter.add_mesh(
                    self._filtered_mesh, 
                    scalars='RGB', 
                    rgb=True, 
                    style='points',
                    point_size=self.point_size,
                    render=False,
                    render_points_as_spheres=False,
                    lighting=False
                )
            else:
                self._filtered_actor = self.plotter.add_mesh(
                    self._filtered_mesh, 
                    color='black', 
                    style='points',
                    point_size=self.point_size,
                    render=False,
                    render_points_as_spheres=False,
                    lighting=False
                )
            
            # LOD Optimization
            try:
                self._filtered_actor.GetProperty().SetLODRenderThreshold(1000)
            except:
                pass

        # --- CASE 4: Fast In-Place Update ---
        else:
            update_type = "In-Place Update"
            self._filtered_actor.SetVisibility(True)
            
            # 1. Update Geometry (Pointer Swap)
            self._filtered_mesh.points = points
            
            # 2. Update Data
            for name, data in point_data.items():
                self._filtered_mesh.point_data[name] = data
                
            # 3. Mark Modified (Triggers GPU Upload)
            self._filtered_mesh.GetPoints().Modified()
            if hasattr(self._filtered_mesh.GetPointData(), "GetScalars"):
                scalars = self._filtered_mesh.GetPointData().GetScalars()
                if scalars: 
                    scalars.Modified()
            
        # Render
        self.plotter.render()
        
        render_time = time.time() - start_time
        print(f"⏱️ Viewer: {update_type} for {len(points):,} points in {render_time:.3f}s")

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