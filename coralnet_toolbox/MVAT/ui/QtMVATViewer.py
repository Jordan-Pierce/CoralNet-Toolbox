import numpy as np
import pyvista as pv

from pyvistaqt import QtInteractor

from PyQt5.QtWidgets import QFrame, QVBoxLayout
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from coralnet_toolbox.MVAT.core.Ray import CameraRay


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
    """
    def __init__(self, parent=None, point_size=1):
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
        self.point_cloud_mesh = None
        self.point_cloud_actor = None
        self.point_size = point_size
        
        # Ray visualization management
        self._ray_line_actor = None
        self._ray_point_actor = None
        self._ray_visible = True
        
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
            mesh = pv.read(file_path)
            # Remove existing point cloud if any
            if self.point_cloud_actor is not None:
                self.plotter.remove_actor(self.point_cloud_actor)
            # Handle styling for point cloud vs meshes
            if 'RGB' in mesh.point_data:
                self.point_cloud_actor = self.plotter.add_mesh(mesh, 
                                                               scalars='RGB', 
                                                               rgb=True, 
                                                               point_size=self.point_size)
            else:
                point_size = self.point_size if mesh.n_cells == 0 else None
                self.point_cloud_actor = self.plotter.add_mesh(mesh, 
                                                               color='cyan', 
                                                               point_size=point_size)
            # Store for re-adding after clears
            self.point_cloud_mesh = mesh
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
            if self.point_cloud_mesh is not None:
                if 'RGB' in self.point_cloud_mesh.point_data:
                    self.point_cloud_actor = self.plotter.add_mesh(self.point_cloud_mesh, 
                                                                   scalars='RGB', 
                                                                   rgb=True, 
                                                                   point_size=self.point_size)
                else:
                    point_size = self.point_size if self.point_cloud_mesh.n_cells == 0 else None
                    self.point_cloud_actor = self.plotter.add_mesh(self.point_cloud_mesh, 
                                                                   color='cyan', 
                                                                   point_size=point_size)
        finally:
            QApplication.restoreOverrideCursor()

    def set_point_cloud_visible(self, visible):
        """Set visibility of the point cloud actor."""
        if self.point_cloud_actor is not None:
            self.point_cloud_actor.SetVisibility(visible)

    def set_point_size(self, size):
        """Update the point size for point clouds."""
        self.point_size = size
        # If point cloud is loaded, update the actor
        if self.point_cloud_actor is not None:
            self.point_cloud_actor.GetProperty().SetPointSize(size)
            self.plotter.render()  # Force re-render

    # --------------------------------------------------------------------------
    # Ray Visualization Methods
    # --------------------------------------------------------------------------
    
    def show_ray(self, ray: 'CameraRay'):
        """
        Display a ray in the 3D viewer.
        
        Draws a line from the ray origin to the terminal point, with a
        sphere glyph at the terminal point. Uses solid line for accurate
        depth and dashed/stippled line for estimated depth.
        
        Args:
            ray: CameraRay object to visualize.
            
        # TODO: Add mesh intersection refinement when point cloud mesh is available
        """
        if ray is None:
            self.clear_ray()
            return
            
        # Clear existing ray visualization
        self._remove_ray_actors()
        
        # Create line mesh
        line_mesh = pv.Line(ray.origin.tolist(), ray.terminal_point.tolist())
        
        # Colors: magenta for visualization
        ray_color = 'magenta'
        
        # Set line style based on accuracy
        if ray.has_accurate_depth:
            # Solid line for accurate depth
            self._ray_line_actor = self.plotter.add_mesh(
                line_mesh,
                color=ray_color,
                line_width=3,
                name='_ray_line',
                pickable=False
            )
        else:
            # For estimated depth, create a dashed effect using multiple short segments
            # PyVista doesn't directly support dashed lines, so we use stippled tube
            # or just use a different visual (thinner, semi-transparent)
            self._ray_line_actor = self.plotter.add_mesh(
                line_mesh,
                color=ray_color,
                line_width=2,
                opacity=0.6,
                name='_ray_line',
                pickable=False
            )
        
        # Create sphere at terminal point
        sphere_radius = np.linalg.norm(ray.terminal_point - ray.origin) * 0.02
        sphere_radius = max(sphere_radius, 0.01)  # Minimum size
        sphere = pv.Sphere(radius=sphere_radius, center=ray.terminal_point.tolist())
        
        if ray.has_accurate_depth:
            # Solid sphere for accurate depth
            self._ray_point_actor = self.plotter.add_mesh(
                sphere,
                color=ray_color,
                name='_ray_point',
                pickable=False
            )
        else:
            # Semi-transparent sphere for estimated depth
            self._ray_point_actor = self.plotter.add_mesh(
                sphere,
                color=ray_color,
                opacity=0.5,
                name='_ray_point',
                pickable=False
            )
        
        # Update display
        self.plotter.render()
        
    def clear_ray(self):
        """Remove any displayed ray visualization."""
        self._remove_ray_actors()
        self.plotter.render()
        
    def _remove_ray_actors(self):
        """Internal method to remove ray actors from the plotter."""
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
            if self.point_cloud_mesh is not None:
                # Use point cloud center
                center = np.array(self.point_cloud_mesh.center)
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
