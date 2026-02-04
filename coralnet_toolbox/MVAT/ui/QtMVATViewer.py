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
# Constants
# ----------------------------------------------------------------------------------------------------------------------

# Default max rays for pre-allocation (will be updated based on camera count)
DEFAULT_MAX_RAYS = 100

# Color indices for scalar-based coloring
COLOR_TYPE_SELECTED = 0    # Lime green
COLOR_TYPE_HIGHLIGHTED = 1  # Cyan


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MVATViewer(QFrame):
    """
    A dedicated widget for holding the PyVista 3D Interactor.
    
    Supports visualization of:
    - Point clouds (drag & drop)
    - Camera frustums (merged mesh rendering)
    - Ray casting visualization with persistent actors (data-oriented)
    
    Ray visualization uses a data-oriented approach:
    - Persistent PolyData meshes created once at init
    - Geometry updated in-place via NumPy array overwrites
    - No actor creation/destruction during mouse tracking
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
        
        # Ray visualization settings
        self._show_rays_enabled = show_rays
        self._ray_visible = True
        self._max_rays = DEFAULT_MAX_RAYS
        
        # Initialize persistent ray meshes and actors
        self._init_ray_visualization()
        
        # Add to layout
        self.layout.addWidget(self.plotter.interactor)
    
    def _init_ray_visualization(self):
        """
        Initialize persistent ray visualization meshes and actors.
        
        Creates reusable PolyData objects for ray lines and tip spheres.
        These are added to the plotter once and their geometry is updated
        in-place during mouse tracking for optimal performance.
        """
        # Pre-allocate arrays for maximum expected rays
        # Each ray needs 2 points (origin + terminal), so max_points = 2 * max_rays
        max_points = 2 * self._max_rays
        
        # --- Ray Lines Mesh ---
        # Initialize with placeholder geometry (will be overwritten)
        self._ray_lines_points = np.zeros((max_points, 3), dtype=np.float64)
        self._ray_lines_mesh = pv.PolyData(self._ray_lines_points)
        
        # Pre-allocate lines connectivity array
        # Each line: [2, idx_start, idx_end]
        # Total size: max_rays * 3
        self._ray_lines_connectivity = np.zeros(self._max_rays * 3, dtype=np.int64)
        
        # Color type scalars (0=Selected/Lime, 1=Highlighted/Cyan)
        self._ray_lines_colors = np.zeros(max_points, dtype=np.uint8)
        self._ray_lines_mesh.point_data['ColorType'] = self._ray_lines_colors
        
        # Add to plotter with categorical LUT coloring
        self._ray_lines_actor = self.plotter.add_mesh(
            self._ray_lines_mesh,
            scalars='ColorType',
            cmap=['lime', 'cyan'],  # 0=lime (selected), 1=cyan (highlighted)
            clim=[0, 1],
            line_width=3,
            pickable=False,
            show_scalar_bar=False,
            name='_ray_lines_persistent'
        )
        self._ray_lines_actor.SetVisibility(False)
        
        # --- Ray Tips Mesh (Spheres as Glyphs) ---
        # We'll use a single PolyData with sphere glyphs for efficiency
        self._ray_tips_points = np.zeros((self._max_rays, 3), dtype=np.float64)
        self._ray_tips_mesh = pv.PolyData(self._ray_tips_points)
        
        # Color type scalars for tips
        self._ray_tips_colors = np.zeros(self._max_rays, dtype=np.uint8)
        self._ray_tips_mesh.point_data['ColorType'] = self._ray_tips_colors
        
        # Radius scalars for variable sphere sizes
        self._ray_tips_radii = np.full(self._max_rays, 0.01, dtype=np.float64)
        self._ray_tips_mesh.point_data['Radius'] = self._ray_tips_radii
        
        # Create sphere source for glyphing
        self._sphere_source = pv.Sphere(radius=1.0, theta_resolution=8, phi_resolution=8)
        
        # Add glyphed mesh to plotter
        self._ray_tips_actor = self.plotter.add_mesh(
            self._ray_tips_mesh.glyph(
                geom=self._sphere_source,
                scale='Radius',
                orient=False
            ),
            scalars='ColorType',
            cmap=['lime', 'cyan'],
            clim=[0, 1],
            pickable=False,
            show_scalar_bar=False,
            name='_ray_tips_persistent'
        )
        self._ray_tips_actor.SetVisibility(False)
        
        # Track current number of active rays
        self._active_ray_count = 0

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
    # Ray Visualization Methods (Data-Oriented)
    # --------------------------------------------------------------------------
    
    def set_max_rays(self, count: int):
        """
        Set the maximum number of rays for pre-allocation.
        
        Call this after loading cameras to match camera count.
        Reallocates internal arrays if count exceeds current capacity.
        
        Args:
            count: Maximum number of rays to support.
        """
        if count <= self._max_rays:
            return  # Already have enough capacity
        
        self._max_rays = count
        
        # Reinitialize with new capacity
        # Remove old actors first
        try:
            self.plotter.remove_actor(self._ray_lines_actor)
            self.plotter.remove_actor(self._ray_tips_actor)
        except:
            pass
        
        # Recreate with new size
        self._init_ray_visualization()
    
    def show_rays(self, rays_with_colors: list):
        """
        Display multiple rays in the 3D viewer using persistent geometry.
        
        Updates pre-allocated meshes in-place for optimal performance.
        No actors are created or destroyed - only geometry data is modified.
        
        Args:
            rays_with_colors: List of (CameraRay, color) tuples.
                              color should be RAY_COLOR_SELECTED or RAY_COLOR_HIGHLIGHTED
        """
        if not self._show_rays_enabled:
            return
        
        if not rays_with_colors:
            self.clear_ray()
            return
        
        # Filter out None rays
        valid_rays = [(ray, color) for ray, color in rays_with_colors if ray is not None]
        
        if not valid_rays:
            self.clear_ray()
            return
        
        num_rays = len(valid_rays)
        
        # Ensure we have capacity
        if num_rays > self._max_rays:
            self.set_max_rays(num_rays * 2)  # Double to avoid frequent reallocation
        
        # Calculate sphere radius based on first ray's distance
        first_ray = valid_rays[0][0]
        base_sphere_radius = np.linalg.norm(first_ray.terminal_point - first_ray.origin) * 0.005
        base_sphere_radius = max(base_sphere_radius, 0.01)
        
        # Build arrays for lines
        lines_points = np.zeros((num_rays * 2, 3), dtype=np.float64)
        lines_connectivity = np.zeros(num_rays * 3, dtype=np.int64)
        lines_colors = np.zeros(num_rays * 2, dtype=np.uint8)
        
        # Build arrays for tips
        tips_points = np.zeros((num_rays, 3), dtype=np.float64)
        tips_colors = np.zeros(num_rays, dtype=np.uint8)
        tips_radii = np.zeros(num_rays, dtype=np.float64)
        
        for i, (ray, color) in enumerate(valid_rays):
            # Line points: origin at 2*i, terminal at 2*i+1
            lines_points[2*i] = ray.origin
            lines_points[2*i + 1] = ray.terminal_point
            
            # Line connectivity: [2, start_idx, end_idx]
            lines_connectivity[3*i] = 2
            lines_connectivity[3*i + 1] = 2*i
            lines_connectivity[3*i + 2] = 2*i + 1
            
            # Color type: 0 for selected (lime), 1 for highlighted (cyan)
            color_type = COLOR_TYPE_SELECTED if color == RAY_COLOR_SELECTED else COLOR_TYPE_HIGHLIGHTED
            lines_colors[2*i] = color_type
            lines_colors[2*i + 1] = color_type
            
            # Tip point
            tips_points[i] = ray.terminal_point
            tips_colors[i] = color_type
            
            # Radius: larger for selected (first), smaller for highlighted
            tips_radii[i] = base_sphere_radius if i == 0 else base_sphere_radius * 0.6
        
        # Update lines mesh in-place
        self._ray_lines_mesh.points = lines_points
        self._ray_lines_mesh.lines = lines_connectivity
        self._ray_lines_mesh.point_data['ColorType'] = lines_colors
        
        # Update tips mesh - need to recreate glyphed mesh
        self._ray_tips_mesh.points = tips_points
        self._ray_tips_mesh.point_data['ColorType'] = tips_colors
        self._ray_tips_mesh.point_data['Radius'] = tips_radii
        
        # Remove and re-add tips actor with updated glyph
        # (glyphing doesn't update in-place well)
        try:
            self.plotter.remove_actor(self._ray_tips_actor)
        except:
            pass
        
        glyphed_tips = self._ray_tips_mesh.glyph(
            geom=self._sphere_source,
            scale='Radius',
            orient=False
        )
        
        self._ray_tips_actor = self.plotter.add_mesh(
            glyphed_tips,
            scalars='ColorType',
            cmap=['lime', 'cyan'],
            clim=[0, 1],
            pickable=False,
            show_scalar_bar=False,
            name='_ray_tips_persistent'
        )
        
        # Set visibility
        self._ray_lines_actor.SetVisibility(True)
        self._ray_tips_actor.SetVisibility(True)
        
        self._active_ray_count = num_rays
        
        # Render
        self.plotter.render()
        
    def clear_ray(self):
        """Hide ray visualization without destroying actors."""
        if self._ray_lines_actor is not None:
            self._ray_lines_actor.SetVisibility(False)
        if self._ray_tips_actor is not None:
            self._ray_tips_actor.SetVisibility(False)
        self._active_ray_count = 0
        
    def set_ray_visible(self, visible: bool):
        """
        Toggle ray visualization visibility.
        
        Args:
            visible: Whether the ray should be visible.
        """
        self._ray_visible = visible
        if self._ray_lines_actor is not None:
            self._ray_lines_actor.SetVisibility(visible and self._active_ray_count > 0)
        if self._ray_tips_actor is not None:
            self._ray_tips_actor.SetVisibility(visible and self._active_ray_count > 0)
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
