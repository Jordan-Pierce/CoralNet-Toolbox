import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from PyQt5.QtWidgets import QFrame, QVBoxLayout
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from coralnet_toolbox.MVAT.core.Ray import CameraRay
from coralnet_toolbox.MVAT.core.Model import PointCloud
from coralnet_toolbox.MVAT.core.constants import RAY_COLOR_SELECTED, RAY_COLOR_HIGHLIGHTED


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MVATViewer(QFrame):
    """
    A dedicated widget for holding the PyVista 3D Interactor.
    
    Supports visualization of:
    - Point clouds (drag & drop)
    - Camera frustums
    - Ray casting visualization with accuracy indicators
    - Multiple simultaneous rays with distinct colors
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
        
        # Point cloud management
        self.point_cloud = None
        self.point_size = point_size
        
        # Ray visualization management
        self._show_rays_enabled = show_rays
        
        # Ray visualization management - single ray (legacy)
        self._ray_line_actor = None
        self._ray_point_actor = None
        self._ray_visible = True
        
        # Multiple ray visualization management
        self._ray_actors = []  # List of (line_actor, point_actor) tuples
        
        # Add to layout
        self.layout.addWidget(self.plotter.interactor)

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
                self.point_cloud.add_to_plotter(self.plotter)
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
    # Ray Visualization Methods
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
        
        # TODO: When depth is fully incorporated, re-evaluate solid vs dashed
        # line styling based on depth accuracy at the terminal point.
        """
        if ray is None:
            self.clear_ray()
            return
            
        # Clear existing ray visualization
        self._remove_ray_actors()
        
        # Create line mesh
        line_mesh = pv.Line(ray.origin.tolist(), ray.terminal_point.tolist())
        
        # Add line - using solid lines for now
        # TODO: Re-evaluate solid vs dashed styling when depth is fully incorporated
        self._ray_line_actor = self.plotter.add_mesh(
            line_mesh,
            color=color,
            line_width=3,
            name='_ray_line',
            pickable=False
        )
        
        # Create sphere at terminal point
        sphere_radius = np.linalg.norm(ray.terminal_point - ray.origin) * 0.02
        sphere_radius = max(sphere_radius, 0.01)  # Minimum size
        sphere = pv.Sphere(radius=sphere_radius, center=ray.terminal_point.tolist())
        
        self._ray_point_actor = self.plotter.add_mesh(
            sphere,
            color=color,
            name='_ray_point',
            pickable=False
        )
        
        # Update display
        self.plotter.render()
    
    def show_rays(self, rays_with_colors: list):
        """
        Display multiple rays in the 3D viewer with distinct colors.
        
        Each ray is drawn as a line from origin to terminal point with a
        sphere at the terminal point. All rays share the same terminal point
        (the 3D world point from the selected camera's ray).
        
        Args:
            rays_with_colors: List of (CameraRay, color_string) tuples.
                              Colors should be 'lime' for selected, 'cyan' for highlighted.
        
        # TODO: When depth is fully incorporated, re-evaluate solid vs dashed
        # line styling for rays based on depth accuracy.
        """
        if not self._show_rays_enabled:
            return
            
        # Clear all existing ray visualizations
        self._remove_ray_actors()
        self._remove_multi_ray_actors()
        
        if not rays_with_colors:
            self.plotter.render()
            return
        
        # Calculate sphere radius based on first ray's distance
        first_ray = rays_with_colors[0][0]
        sphere_radius = np.linalg.norm(first_ray.terminal_point - first_ray.origin) * 0.005
        sphere_radius = max(sphere_radius, 0.01)  # Minimum size
        
        for i, (ray, color) in enumerate(rays_with_colors):
            if ray is None:
                continue
                
            # Create line mesh
            line_mesh = pv.Line(ray.origin.tolist(), ray.terminal_point.tolist())
            
            # Add line - using solid lines for all rays
            # TODO: Re-evaluate solid vs dashed styling when depth is fully incorporated
            line_actor = self.plotter.add_mesh(
                line_mesh,
                color=color,
                line_width=3,
                name=f'_ray_line_{i}',
                pickable=False
            )
            
            # Create sphere at terminal point (smaller for non-primary rays)
            # Only the first (selected) ray gets a full-size sphere
            current_radius = sphere_radius if i == 0 else sphere_radius * 0.6
            sphere = pv.Sphere(radius=current_radius, center=ray.terminal_point.tolist())
            
            point_actor = self.plotter.add_mesh(
                sphere,
                color=color,
                name=f'_ray_point_{i}',
                pickable=False
            )
            
            self._ray_actors.append((line_actor, point_actor))
        
        # Update display
        self.plotter.render()
        
    def clear_ray(self):
        """Remove any displayed ray visualization."""
        self._remove_ray_actors()
        self._remove_multi_ray_actors()
        self.plotter.render()
        
    def _remove_ray_actors(self):
        """Internal method to remove single ray actors from the plotter."""
        if self._ray_line_actor is not None:
            try:
                self.plotter.remove_actor(self._ray_line_actor)
            except:
                pass
            self._ray_line_actor = None
            
        if self._ray_point_actor is not None:
            try:
                self.plotter.remove_actor(self._ray_point_actor)
            except:
                pass
            self._ray_point_actor = None
    
    def _remove_multi_ray_actors(self):
        """Internal method to remove multiple ray actors from the plotter."""
        for line_actor, point_actor in self._ray_actors:
            try:
                if line_actor is not None:
                    self.plotter.remove_actor(line_actor)
            except:
                pass
            try:
                if point_actor is not None:
                    self.plotter.remove_actor(point_actor)
            except:
                pass
        self._ray_actors.clear()
            
    def set_ray_visible(self, visible: bool):
        """
        Toggle ray visualization visibility.
        
        Args:
            visible: Whether the ray should be visible.
        """
        self._ray_visible = visible
        if self._ray_line_actor is not None:
            self._ray_line_actor.SetVisibility(visible)
        if self._ray_point_actor is not None:
            self._ray_point_actor.SetVisibility(visible)
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
        if self.plotter:
            self.plotter.close()