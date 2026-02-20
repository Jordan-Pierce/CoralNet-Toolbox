"""
MultiView Annotation Tool (MVAT) Window

A 3D viewer for visualizing camera frustums and navigating MultiView imagery.
Uses PyVista for 3D rendering and integrates with the main application's RasterManager.
"""

import warnings
import time

import numpy as np

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout,
    QToolBar, QAction,
    QMessageBox, QApplication,
    QSizePolicy, QDockWidget
)

from coralnet_toolbox.MVAT.ui.QtMVATViewer import MVATViewer
from coralnet_toolbox.MVAT.ui.QtCameraGrid import CameraGrid
from coralnet_toolbox.MVAT.core.Camera import Camera
from coralnet_toolbox.MVAT.managers.SelectionManager import SelectionManager
from coralnet_toolbox.MVAT.core.Ray import CameraRay
from coralnet_toolbox.MVAT.managers.VisibilityManager import VisibilityManager

from coralnet_toolbox.MVAT.core.constants import (
    MARKER_COLOR_SELECTED,
    MARKER_COLOR_INVALID,
    RAY_COLOR_SELECTED,
    RAY_COLOR_HIGHLIGHTED,
    RAY_COLOR_INVALID,
    MOUSE_THROTTLE_MS,
)


from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MousePositionBridge:
    """
    Bridges mouse position events from AnnotationWindow to MVATWindow.
    
    Handles:
    - Throttling mouse position updates to prevent performance issues
    - Creating rays from 2D pixel positions through the selected camera
    - Creating rays from highlighted cameras to the same 3D world point
    - Projecting rays onto other camera views
    - Updating markers on visible camera widgets with appropriate colors
    - Updating ray visualization in the 3D viewer
    """
    
    def __init__(self, mvat_window: 'MVATWindow'):
        """
        Initialize the MousePositionBridge.
        
        Args:
            mvat_window: Reference to the parent MVATWindow.
        """
        self.mvat_window = mvat_window
        self.enabled = True
        self._last_update_time = 0
        self._pending_position = None
        
        # Throttle timer
        self._throttle_timer = QTimer()
        self._throttle_timer.setSingleShot(True)
        self._throttle_timer.timeout.connect(self._process_pending_position)
        
    def on_mouse_moved(self, x: int, y: int):
        """
        Handle mouse moved signal from AnnotationWindow.
        
        Throttles updates to prevent performance issues with rapid mouse movement.
        
        Args:
            x: X pixel coordinate in image space.
            y: Y pixel coordinate in image space.
        """
        if not self.enabled:
            return
            
        # Store pending position
        self._pending_position = (x, y)
        
        # Start throttle timer if not running
        if not self._throttle_timer.isActive():
            self._throttle_timer.start(MOUSE_THROTTLE_MS)
            
    def _process_pending_position(self):
        """Process the pending mouse position after throttle delay."""
        if self._pending_position is None:
            return
            
        x, y = self._pending_position
        self._pending_position = None
        
        # Get the selected camera
        camera = self.mvat_window.selected_camera
        if camera is None:
            self.clear_all_markers()
            self.mvat_window.viewer.clear_ray()
            return
            
        # Check if position is within image bounds
        if not (0 <= x < camera.width and 0 <= y < camera.height):
            self.clear_all_markers()
            self.mvat_window.viewer.clear_ray()
            return
            
        # Get depth from z-channel if available
        raster = camera._raster
        depth = None
        
        if raster.z_channel is not None and raster.z_data_type == 'depth':
            depth = raster.get_z_value(x, y)
            # TODO: Improve depth estimation when z-channel unavailable
            # Could use mesh intersection or other depth estimation techniques
        
        # Get default depth from scene if no depth available
        if depth is None or depth <= 0 or np.isnan(depth):
            default_depth = self.mvat_window.viewer.get_scene_median_depth(camera.position)
        else:
            default_depth = 10.0  # Fallback
        
        # Create ray from pixel position (selected camera)
        ray = CameraRay.from_pixel_and_camera(
            pixel_xy=(x, y),
            camera=camera,
            depth=depth,
            default_depth=default_depth
        )
        
        # (mesh intersection checks removed - mesh variable unused)
        
        # Get highlighted cameras and create rays from them to the world point
        highlighted_cameras = self.mvat_window.highlighted_cameras
        
        # List to store rays for 3D viewer: [(Ray, Color), ...]
        rays_with_colors = [(ray, RAY_COLOR_SELECTED if ray.has_accurate_depth else RAY_COLOR_INVALID)]
        
        # Dictionary to store visibility status for 2D markers: {image_path: is_occluded}
        visibility_status = {}
        
        # Dictionary to store accuracies for markers: {image_path: has_accurate_depth}
        accuracies = {camera.image_path: ray.has_accurate_depth}
        
        for target_cam in highlighted_cameras:
            # Skip self
            if target_cam.image_path == camera.image_path:
                continue
            
            # --- OCCLUSION CHECK (for markers) ---
            # TODO: Handle case where target_cam does not have z-channel data.
            # Currently assumes visible, but consider user warning or alternative occlusion method (e.g., mesh-based).
            is_occluded = target_cam.is_point_occluded_depth_based(ray.terminal_point, depth_threshold=0.15)
            visibility_status[target_cam.image_path] = is_occluded
            
            # Create ray from target camera to the world point
            target_ray = CameraRay.from_world_point_and_camera(
                world_point=ray.terminal_point,
                camera=target_cam
            )
            
            # Color coding based on accuracy
            ray_color = RAY_COLOR_HIGHLIGHTED if target_ray.has_accurate_depth else RAY_COLOR_INVALID
                
            rays_with_colors.append((target_ray, ray_color))
            
            # Store accuracy for markers
            accuracies[target_cam.image_path] = target_ray.has_accurate_depth
        
        # Update 3D ray visualization with all rays
        self.mvat_window.viewer.show_rays(rays_with_colors)
        
        # Project ray onto other camera views (using selected camera's ray terminal point)
        projections = ray.project_to_cameras(self.mvat_window.cameras)
        
        # Update markers on visible camera widgets with appropriate colors
        self._update_camera_markers(projections, accuracies, highlighted_cameras, visibility_status)
        
    def _update_camera_markers(self, projections: dict, accuracies: dict, 
                               highlighted_cameras: list, visibility_status: dict):
        """
        Update marker positions on visible camera widgets.
        
        Only updates markers for widgets that are currently visible in the
        camera grid viewport. Uses lime color for selected camera marker if valid,
        cyan for highlighted camera markers if valid, blood red if invalid.
        
        Args:
            projections: Dict mapping image_path to (x, y, is_valid) tuples.
            accuracies: Dict mapping image_path to has_accurate_depth bool.
            highlighted_cameras: List of highlighted Camera objects.
            visibility_status: Dict mapping image_path to is_occluded bool.
        """        
        # Delegate marker updates to CameraGrid to encapsulate widget logic
        camera_grid = self.mvat_window.camera_grid
        highlighted_paths = {cam.image_path for cam in highlighted_cameras}
        selected_path = self.mvat_window.selected_camera.image_path if self.mvat_window.selected_camera else None
        try:
            camera_grid.update_markers(
                projections,
                accuracies,
                highlighted_paths,
                visibility_status,
                selected_path=selected_path,
            )
        except Exception:
            # Fallback: clear markers on error
            camera_grid.clear_all_markers()
                
    def clear_all_markers(self):
        """Clear markers from all visible camera widgets."""
        try:
            self.mvat_window.camera_grid.clear_all_markers()
        except Exception:
            # Fallback: iterate visible widgets
            try:
                camera_grid = self.mvat_window.camera_grid
                visible_widgets = camera_grid.get_visible_widgets()
                for widget in visible_widgets.values():
                    widget.clear_marker()
            except Exception:
                pass
            
    def set_enabled(self, enabled: bool):
        """
        Enable or disable the mouse position bridge.
        
        Args:
            enabled: Whether to enable mouse position tracking.
        """
        self.enabled = enabled
        if not enabled:
            self.clear_all_markers()
            self.mvat_window.viewer.clear_ray()
            
    def cleanup(self):
        """Clean up resources."""
        self._throttle_timer.stop()
        self.clear_all_markers()


class MVATWindow(QMainWindow):
    """
    MultiView Annotation Tool Window.
    
    Provides a 3D visualization of camera frustums for MultiView imagery projects.
    Allows users to navigate between views in 3D space and select cameras.
    
    Signals:
        cameraSelectedInMVAT: Emitted when a camera is selected in MVAT (image_path).
    """
    
    # Signal emitted when a camera is selected in MVAT
    cameraSelectedInMVAT = pyqtSignal(str)
    
    def __init__(self, main_window, parent=None):
        """
        Initialize the MVAT Window.
        
        Args:
            main_window: Reference to the main application window
            parent: Parent widget (default: None)
        """
        super().__init__(parent)
        
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.raster_manager = main_window.image_window.raster_manager
        
        # Camera management
        self.cameras = {}  # image_path -> Camera object
        self.selected_camera = None
        self.highlighted_cameras = []
        self.hovered_camera = None
        
        # Focal point management
        self.current_focal_point = None
        self.selected_camera_path = None
        
        # Batched geometry managers for efficient rendering (O(1) draw calls)
        # These are owned by the MVATViewer; initialize to None and bind to viewer-managed instances
        self.frustum_manager = None
        self.ray_manager = None
        
        # Cache manager for visibility data
        from coralnet_toolbox.MVAT.managers.CacheManager import CacheManager
        self.cache_manager = CacheManager("")
        
        # Actor lists for thumbnails (cannot be batched due to textures)
        # Owned by MVATViewer; placeholder until viewer is created
        self.thumbnail_actors = None
        
        # Display status
        self.point_size = 1
        self.frustum_scale = 0.1
        self.thumbnail_opacity = 0.25
        
        self._show_wireframes_enabled = True
        self._show_thumbnails_enabled = True
        self._show_rays_enabled = True
        
        # Mouse position bridge for cross-window sync
        self.mouse_bridge = None  # Initialized after UI setup
        
        # Reference to annotation window for signal connections
        self.annotation_window = main_window.annotation_window
        
        # Selection model (single source of truth for UI selection state)
        self.selection_model = SelectionManager(self)

        # Setup UI
        self._setup_window()
        self._setup_menubar()
        self._setup_toolbar()
        self._setup_central_layout()
        
        # Setup mouse position bridge and signal connections
        self._setup_signal_connections()
        
        # Flag to track initialization
        self._initialized = False
        
    def _setup_window(self):
        """Configure the main window properties."""
        self.setWindowTitle("MultiView Annotation Tool (MVAT)")
        self.setWindowIcon(QIcon(get_icon("camera.svg")))
        self.setMinimumSize(1200, 800)
        
    def _setup_menubar(self):
        """Create the menu bar."""
        self.menu_bar = self.menuBar()
        
        # ========== VIEW MENU ==========
        self.view_menu = self.menu_bar.addMenu("View")
        
        # Reset Camera View
        self.reset_view_action = QAction("Reset View", self)
        self.reset_view_action.setShortcut("R")
        self.reset_view_action.triggered.connect(self._reset_camera_view)
        self.view_menu.addAction(self.reset_view_action)
        
        # Toggle Wireframes
        self.toggle_wireframes_action = QAction("Show Wireframes", self)
        self.toggle_wireframes_action.setCheckable(True)
        self.toggle_wireframes_action.setChecked(True)  
        self.toggle_wireframes_action.triggered.connect(self._toggle_wireframes)
        self.view_menu.addAction(self.toggle_wireframes_action)
        
        # Toggle Thumbnails
        self.toggle_thumbnails_action = QAction("Show Thumbnails", self)
        self.toggle_thumbnails_action.setCheckable(True)
        self.toggle_thumbnails_action.setChecked(True)  
        self.toggle_thumbnails_action.triggered.connect(self._toggle_thumbnails)
        self.view_menu.addAction(self.toggle_thumbnails_action)
        
        # Toggle Full Cloud (disable visibility filtering)
        self.toggle_full_cloud_action = QAction("Show Full Point Cloud", self)
        self.toggle_full_cloud_action.setCheckable(True)
        self.toggle_full_cloud_action.setChecked(False)  # Default: filtering enabled
        self.toggle_full_cloud_action.setToolTip("Bypass visibility filtering and show entire point cloud")
        self.toggle_full_cloud_action.triggered.connect(self._toggle_full_cloud)
        self.view_menu.addAction(self.toggle_full_cloud_action)
        
        # Toggle Rays
        self.toggle_rays_action = QAction("Show Rays", self)
        self.toggle_rays_action.setCheckable(True)
        self.toggle_rays_action.setChecked(True)  
        self.toggle_rays_action.triggered.connect(self._toggle_rays)
        self.view_menu.addAction(self.toggle_rays_action)
        
        self.view_menu.addSeparator()
        
        # Fit to View
        self.fit_view_action = QAction("Fit All", self)
        self.fit_view_action.setShortcut("F")
        self.fit_view_action.triggered.connect(self._fit_to_view)
        self.view_menu.addAction(self.fit_view_action)
        
    def _setup_toolbar(self):
        """Create the left-side vertical toolbar."""
        self.toolbar = QToolBar("Tools")
        self.toolbar.setOrientation(Qt.Vertical)
        self.toolbar.setFixedWidth(40)
        self.toolbar.setMovable(False)
        self.addToolBar(Qt.LeftToolBarArea, self.toolbar)
        
        # Define spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        spacer.setFixedHeight(10)
        self.toolbar.addWidget(spacer)
        
        # Currently the toolbar is empty of actions, but the gutter exists 
        # to match the Main Window's look and feel.
        
    def _setup_central_layout(self):
        """Create the central widget and arrange MVAT viewer and camera grid as docks.

        Uses the same "vacant central widget" trick as MainWindow so docks can
        occupy the center column while remaining freely movable and resizable.
        """
        # Vacant central widget (zero height) so top/bottom dock areas behave like MainWindow
        self.central_widget = QWidget()
        self.central_widget.setFixedHeight(0)
        self.central_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCentralWidget(self.central_widget)
        self.setDockNestingEnabled(True)

        # Ensure corners trap left/right dock areas (center column remains for docks)
        self.setCorner(Qt.TopLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.TopRightCorner, Qt.RightDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)

        # ----------------------
        # Create MVAT Viewer dock
        # ----------------------
        # Create the viewer instance
        self.viewer = MVATViewer(self, point_size=self.point_size, show_rays=self._show_rays_enabled)
        
        try:
            # Enable picking for camera selection using the viewer's plotter
            self.viewer.plotter.enable_point_picking(
                callback=self._on_pick,
                show_message=False,
                use_picker=True,
                pickable_window=True
            )
        except Exception:
            # If plotter is not ready yet, it's fine; picking will be enabled later
            pass

        try:
            # Sync viewer controls and wire signals to MVAT window handlers
            # Initialize viewer controls from MVATWindow state
            if hasattr(self.viewer, 'opacity_slider'):
                self.viewer.opacity_slider.setValue(int(self.thumbnail_opacity * 100))
                self.viewer.opacityChanged.connect(self._on_opacity_changed)
            if hasattr(self.viewer, 'point_size_spinbox'):
                self.viewer.point_size_spinbox.setValue(self.point_size)
                # Connect viewer's point size changes to MVATWindow handler
                try:
                    self.viewer.pointSizeChanged.connect(self._on_point_size_changed)
                except Exception:
                    pass
        except Exception:
            pass

        # Viewer widget
        self.viewer_group_box = QWidget()
        vbox = QVBoxLayout(self.viewer_group_box)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(self.viewer)

        self.viewer_dock_container = QWidget()
        viewer_container_layout = QVBoxLayout(self.viewer_dock_container)
        viewer_container_layout.setContentsMargins(0, 0, 0, 0)
        viewer_container_layout.addWidget(self.viewer_group_box)

        self.viewer_dock = QDockWidget("MVAT Viewer", self)
        self.viewer_dock.setObjectName("MVATViewerDock")
        self.viewer_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.viewer_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.viewer_dock.setWidget(self.viewer_dock_container)

        # -------------------------
        # Create Camera Grid dock
        # -------------------------
        # Camera grid widget (pass shared selection model)
        self.camera_grid = CameraGrid(model=self.selection_model, mvat_window=self)
        self.camera_grid.camera_selected.connect(self._on_grid_camera_selected)
        self.camera_grid.cameras_highlighted.connect(self._on_grid_cameras_highlighted)
        self.camera_grid.camera_hovered.connect(self._on_camera_hovered)
        self.camera_grid.camera_unhovered.connect(self._on_camera_unhovered)

        try:
            # Wire CameraGrid intent signals to SelectionManager (SelectionManager is authoritative)
            self.camera_grid.selection_requested.connect(lambda paths: self.selection_model.set_selections(paths))
            self.camera_grid.toggle_requested.connect(lambda path: self.selection_model.toggle(path))
            self.camera_grid.active_requested.connect(lambda path: self.selection_model.set_active(path))
            self.camera_grid.clear_requested.connect(lambda: self.selection_model.clear_selections())
        except Exception:
            pass

        # Grid container
        self.grid_group_box = QWidget()
        gbox_layout = QVBoxLayout(self.grid_group_box)
        gbox_layout.setContentsMargins(2, 2, 2, 2)
        gbox_layout.addWidget(self.camera_grid)

        self.grid_dock_container = QWidget()
        grid_container_layout = QVBoxLayout(self.grid_dock_container)
        grid_container_layout.setContentsMargins(0, 0, 0, 0)
        grid_container_layout.addWidget(self.grid_group_box)

        self.grid_dock = QDockWidget("Camera Grid", self)
        self.grid_dock.setObjectName("CameraGridDock")
        self.grid_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.grid_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.grid_dock.setWidget(self.grid_dock_container)

        # Add docks to the window and split them side-by-side by default
        # Place the MVAT viewer in the center column (TopDockWidgetArea) so it
        # effectively takes the spot of the central widget by default (same trick as MainWindow).
        # Put the camera grid to the right and split horizontally so user may freely resize/swap.
        # Place the MVAT viewer in the center column (TopDockWidgetArea)
        # so it takes the spot of the central widget by default.
        self.addDockWidget(Qt.TopDockWidgetArea, self.viewer_dock)
        # Put the camera grid to the right and split horizontally so user may freely resize/swap.
        self.addDockWidget(Qt.RightDockWidgetArea, self.grid_dock)
        self.splitDockWidget(self.viewer_dock, self.grid_dock, Qt.Horizontal)
        # Bias initial sizes (viewer larger)
        self.resizeDocks([self.viewer_dock, self.grid_dock], [800, 800], Qt.Horizontal)
        # Give the viewer the maximum vertical space available in the center column
        self.resizeDocks([self.viewer_dock], [2000], Qt.Vertical)
        
        try:
            # Set sensible minimum widths to avoid collapse
            self.grid_group_box.setMinimumWidth(280)
            self.viewer_group_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.viewer_dock.setMinimumWidth(500)
            self.grid_dock.setMinimumWidth(240)
        except Exception:
            pass

        # Prefer viewer-owned managers/actors where available to avoid duplicated state.
        try:
            # MVATViewer owns the frustum and ray managers as well as thumbnail actors.
            self.frustum_manager = getattr(self.viewer, '_frustum_manager', self.frustum_manager)
            self.ray_manager = getattr(self.viewer, '_ray_manager', self.ray_manager)
            self.thumbnail_actors = getattr(self.viewer, 'thumbnail_actors', self.thumbnail_actors)
        except Exception:
            pass
    
    def _setup_signal_connections(self):
        """
        Set up signal connections between MVAT and main application windows.
        
        Establishes bi-directional synchronization:
        - Main app → MVAT: Image selection and mouse position
        - MVAT → Main app: Camera selection
        - Camera grid highlight changes → Ray clearing
        """
        # Create mouse position bridge
        self.mouse_bridge = MousePositionBridge(self)
        
        # Connect AnnotationWindow.mouseMoved to bridge
        if hasattr(self.annotation_window, 'mouseMoved'):
            self.annotation_window.mouseMoved.connect(self.mouse_bridge.on_mouse_moved)
        
        # Connect ImageWindow.imageLoaded to sync MVAT selection
        if hasattr(self.image_window, 'imageLoaded'):
            self.image_window.imageLoaded.connect(self._on_main_image_loaded)
        
        # Connect selection model signals to update MVAT state
        # This replaces direct grid-driven signals so window and grid share a model
        try:
            self.selection_model.active_changed.connect(self._on_active_camera_changed)
            self.selection_model.selection_changed.connect(self._on_selections_changed)
        except Exception:
            # Fall back to camera_grid signals if model isn't connected
            self.camera_grid.cameras_highlighted.connect(self._on_highlights_changed)
        # Connect viewer focal point changes
        self.viewer.focalPointChanged.connect(self._on_focal_point_changed)
    
    def _on_main_image_loaded(self, path: str):
        """
        Handle image loaded in main application.
        
        Updates MVAT selection to match main app's current image.
        
        Args:
            path: Image path that was loaded in the main app.
        """
        if path in self.cameras:
            # Use SelectionManager to set the active camera; model signals will drive UI sync
            self.selection_model.set_active(path)
    
    def _on_highlights_changed(self, highlighted_paths: list):
        """
        Handle camera highlight selection changes.
        
        Clears rays when highlights change to ensure stale rays from
        previously-highlighted cameras are removed. The rays will be
        recreated on the next mouse move event.
        
        Args:
            highlighted_paths: List of currently highlighted camera paths.
        """
        # Update highlight state for all cameras
        for path, camera in self.cameras.items():
            if path in highlighted_paths:
                camera.highlight()
            else:
                camera.unhighlight()
            # Update appearance for all cameras (selected/highlighted state may have changed)
            try:
                if hasattr(self.viewer, 'update_camera_appearance'):
                    self.viewer.update_camera_appearance(camera, opacity=self.thumbnail_opacity)
                else:
                    camera.update_appearance(self.viewer.plotter, opacity=self.thumbnail_opacity)
            except Exception:
                pass
        
        self.highlighted_cameras = [self.cameras.get(path) for path in highlighted_paths if path in self.cameras]
        
        # Update batched frustum colors via viewer API when possible
        selected_path = self.selected_camera.image_path if self.selected_camera else None
        try:
            if hasattr(self.viewer, 'update_frustum_states'):
                self.viewer.update_frustum_states(selected_path, highlighted_paths, self.hovered_camera)
            elif self.frustum_manager is not None:
                self.frustum_manager.update_camera_states(selected_path, highlighted_paths, self.hovered_camera)
                self.frustum_manager.mark_modified()
        except Exception:
            pass
        
        # Update camera grid visual state to match
        self.camera_grid.render_highlight_from_paths(highlighted_paths)
        
        self._clear_rays()
        
    def _on_focal_point_changed(self, point_3d):
        """Handle focal point changes from the 3D viewer."""
        # Set the current focal point
        self.current_focal_point = point_3d
        
        # Update the marker in the annotation window based on the new focal point
        if self.selected_camera_path and self.selected_camera_path in self.cameras:
            # Get the selected camera and project the 3D point to its image plane
            camera = self.cameras[self.selected_camera_path]
            pixel = camera.project(point_3d)
            # Check if the projected pixel is valid (not NaN)
            if not np.isnan(pixel).any():
                u, v = pixel[0], pixel[1]
                # Check z-channel for validity using get_z_value
                depth = camera._raster.get_z_value(int(u), int(v))
                color = MARKER_COLOR_SELECTED if depth is not None and depth > 0 else MARKER_COLOR_INVALID
                # Set the marker position and color in the annotation window
                self.main_window.annotation_window.set_incoming_marker(u, v, color)
            else:
                self.main_window.annotation_window.marker.hide()
        else:
            self.main_window.annotation_window.marker.hide()
    
    def _on_camera_hovered(self, path):
        """Handle camera hover start."""
        self.hovered_camera = path
        self._update_frustum_states()
        
    def _on_camera_unhovered(self, path):
        """Handle camera hover end."""
        if self.hovered_camera == path:
            self.hovered_camera = None
        self._update_frustum_states()
        
    def _update_frustum_states(self):
        """Update frustum colors for all cameras."""
        selected_path = self.selected_camera.image_path if self.selected_camera else None
        highlighted_paths = [cam.image_path for cam in self.highlighted_cameras]
        try:
            if hasattr(self.viewer, 'update_frustum_states'):
                self.viewer.update_frustum_states(selected_path, highlighted_paths, self.hovered_camera)
            elif self.frustum_manager is not None:
                self.frustum_manager.update_camera_states(selected_path, highlighted_paths, self.hovered_camera)
                self.frustum_manager.mark_modified()
        except Exception:
            pass
        
    def showEvent(self, event):
        """Handle show event - load cameras when window is shown."""
        super().showEvent(event)
        
        if not self._initialized:
            # Open window in fullscreen/maximized mode on first show
            self.showMaximized()
            # Use QTimer to load cameras after the window is fully shown
            QTimer.singleShot(100, self._load_cameras)
            self._initialized = True
            
    def closeEvent(self, event):
        """Handle close event - cleanup resources."""
        # Disconnect signal connections
        try:
            if hasattr(self.annotation_window, 'mouseMoved'):
                self.annotation_window.mouseMoved.disconnect(self.mouse_bridge.on_mouse_moved)
            if hasattr(self.image_window, 'imageLoaded'):
                self.image_window.imageLoaded.disconnect(self._on_main_image_loaded)
            # Disconnect model signals if connected
            if hasattr(self, 'selection_model'):
                try:
                    self.selection_model.active_changed.disconnect(self._on_active_camera_changed)
                except Exception:
                    pass
                try:
                    self.selection_model.selection_changed.disconnect(self._on_selections_changed)
                except Exception:
                    pass
            # Fall back: disconnect legacy grid signal if present
            if hasattr(self.camera_grid, 'cameras_highlighted'):
                try:
                    self.camera_grid.cameras_highlighted.disconnect(self._on_highlights_changed)
                except Exception:
                    pass
        except:
            pass  # Signals may already be disconnected
        
        # Cleanup mouse bridge
        if self.mouse_bridge:
            self.mouse_bridge.cleanup()
            self.mouse_bridge = None
        
        # Clear camera references
        self.cameras.clear()
        self.selected_camera = None
        self.highlighted_cameras.clear()
        
        # Clear batched geometry managers (prefer viewer-owned instances)
        try:
            if hasattr(self.viewer, '_frustum_manager') and self.viewer._frustum_manager is not None:
                self.viewer._frustum_manager.clear()
            elif self.frustum_manager is not None:
                self.frustum_manager.clear()
        except Exception:
            pass
        try:
            if hasattr(self.viewer, '_ray_manager') and self.viewer._ray_manager is not None:
                self.viewer._ray_manager.clear()
            elif self.ray_manager is not None:
                self.ray_manager.clear()
        except Exception:
            pass
        try:
            if hasattr(self.viewer, 'thumbnail_actors') and self.viewer.thumbnail_actors is not None:
                self.viewer.thumbnail_actors.clear()
            elif self.thumbnail_actors is not None:
                self.thumbnail_actors.clear()
        except Exception:
            pass
        
        # Close the viewer
        if self.viewer:
            self.viewer.close()
            
        # Close from the outside
        if hasattr(self.main_window, 'close_mvat_window'):
            self.main_window.close_mvat_window()

        # Clear the reference in the main_window to allow garbage collection
        self.main_window.mvat_window = None
                
        event.accept()
        
    def _load_cameras(self):
        """Load cameras from rasters with valid intrinsics/extrinsics."""
        # Get all raster paths
        all_paths = self.raster_manager.image_paths
        
        if not all_paths:
            QMessageBox.information(
                self,
                "No Images",
                "No images are loaded in the project."
            )
            self.close()
            return
            
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        # Start progress
        progress = ProgressBar(self, title="Loading Cameras")
        progress.show()
        progress.start_progress(len(all_paths))
        
        valid_count = 0
        
        try:
            for i, path in enumerate(all_paths):
                if progress.canceled:
                    break
                    
                raster = self.raster_manager.get_raster(path)
                
                if raster and raster.intrinsics is not None and raster.extrinsics is not None:
                    try:
                        # Create Camera object wrapping the raster
                        camera = Camera(raster)
                        self.cameras[path] = camera
                        valid_count += 1
                    except Exception as e:
                        # Silently skip cameras that fail to initialize
                        print(f"Failed to create camera for {path}: {e}")
                        
                progress.update_progress()
                
        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()
            # Close progress
            progress.finish_progress()
            progress.close()
            progress = None
            
        # Check if any cameras were loaded
        if valid_count == 0:
            QMessageBox.information(
                self,
                "No Camera Data",
                "No images with valid camera parameters (intrinsics/extrinsics) were found.\n\n"
                "Please import camera data using:\n"
                "  File → Import → Cameras → COLMAP or Metashape"
            )
            self.close()
            return
            
        # Update stats (CameraGrid owns the stats label)
        try:
            self.camera_grid.stats_label.setText(f"Cameras: {valid_count} / {len(all_paths)}")
        except Exception:
            pass
        
        # Populate camera grid
        self.camera_grid.set_cameras(self.cameras)
        
        # Render frustums
        self._render_frustums()
        
        # Fit view to show all cameras
        self._fit_to_view()
        
        # Fix Initial Synchronization**
        # Detect the active image from AnnotationWindow and select it immediately
        current_image_path = getattr(self.annotation_window, 'current_image_path', None)
        if current_image_path and current_image_path in self.cameras:
            # Select the active camera without emitting signals or navigating
            camera = self.cameras[current_image_path]
            self._select_camera(current_image_path, camera, emit_signal=False)
            self._match_camera_perspective(camera)
            self.camera_grid.render_selection_from_path(current_image_path)
            # Update visibility filtering for the selected camera if point cloud exists**
            if self.viewer.point_cloud:
                self._update_visibility_filter([current_image_path])
        else:
            # Fall back to auto-select first camera
            self._auto_select_first_camera()
        
    def _render_frustums(self):
        """Render all camera frustums in the 3D scene using batched geometry.

        Delegates work to MVATViewer to encapsulate 3D scene management.
        """
        if not self.viewer or not self.viewer.plotter:
            return

        # Clear existing actors in plotter
        try:
            self.viewer.plotter.clear()
        except Exception:
            pass

        # Keep MVATWindow thumbnail_actors in sync with viewer
        try:
            self.thumbnail_actors.clear()
        except Exception:
            pass

        # Let viewer warm up point cloud GPU cache and add reference axes via viewer API
        try:
            self.viewer.add_point_cloud()
        except Exception:
            pass
        try:
            self.viewer.add_axes()
        except Exception:
            pass

        # Delegate frustum creation to viewer
        try:
            # Use SelectionManager API to obtain highlighted/selected cameras
            highlighted = self.selection_model.get_selected_list() if self.selection_model else []
            self.viewer.add_frustums(
                self.cameras,
                frustum_scale=self.frustum_scale,
                show_thumbnails=self._show_thumbnails_enabled,
                selected_camera=self.selected_camera,
                highlighted_paths=highlighted,
                hovered_camera=self.hovered_camera
            )
        except Exception as e:
            print(f"Failed to render frustums via viewer: {e}")

        # Update the render via viewer
        try:
            self.viewer.update()
        except Exception:
            pass
            
    def _reset_camera_view(self):
        """Reset the 3D camera to default view."""
        if self.viewer:
            try:
                self.viewer.reset_view()
            except Exception:
                pass
            
    def _fit_to_view(self):
        """Fit all objects in the view."""
        if self.viewer:
            try:
                self.viewer.fit_to_view()
            except Exception:
                pass
            
    def _toggle_wireframes(self, checked=None):
        """Toggle wireframe visibility."""
        if checked is None:
            checked = self.toggle_wireframes_action.isChecked()
            
        self._show_wireframes_enabled = checked
        # Sync UI elements
        self.toggle_wireframes_action.setChecked(checked)

        # Delegate to viewer API to manage frustum visibility/state
        try:
            if hasattr(self.viewer, 'enable_wireframes'):
                self.viewer.enable_wireframes(checked)
            elif self.frustum_manager is not None:
                try:
                    self.frustum_manager.set_visibility(checked)
                except Exception:
                    pass
        except Exception:
            pass
        
    def _toggle_thumbnails(self, checked=None):
        """Toggle thumbnail visibility."""
        if checked is None:
            checked = self.toggle_thumbnails_action.isChecked()
            
        self._show_thumbnails_enabled = checked
        
        # Sync UI elements
        self.toggle_thumbnails_action.setChecked(checked)
        # Delegate to viewer to handle thumbnail visibility
        try:
            if hasattr(self.viewer, 'enable_thumbnails'):
                self.viewer.enable_thumbnails(checked)
            else:
                for actor in getattr(self.viewer, 'thumbnail_actors', []) or []:
                    try:
                        actor.SetVisibility(checked)
                    except Exception:
                        pass
        except Exception:
            pass
    
    def _toggle_full_cloud(self, checked):
        """Toggle between filtered and full point cloud view."""
        # If checked, show full cloud and bypass filtering
        if checked:
            # Show entire point cloud
            self.viewer.update_point_cloud_subset(None)
            
            # Update status bar
            if self.viewer and self.viewer.point_cloud:
                total_points = self.viewer.point_cloud.mesh.n_points
                try:
                    self.camera_grid.stats_label.setText(
                        f"Cameras: {len(self.cameras)} | Points: {total_points:,} / {total_points:,} (Full Cloud)"
                    )
                except Exception:
                    pass
        else:
            # Re-apply visibility filtering
            # Build list of paths to filter (always including selected camera)
            highlighted_paths = list(self.camera_grid.get_highlighted_cameras())
            
            # Ensure selected camera is always included for filtering
            # BUT don't modify the grid highlights - just use it for filtering
            if self.selected_camera:
                selected_path = self.selected_camera.image_path
                if selected_path not in highlighted_paths:
                    highlighted_paths.append(selected_path)
            
            self._update_visibility_filter(highlighted_paths)
        
    def _toggle_rays(self, checked=None):
        """Toggle ray visibility."""
        if checked is None:
            checked = self.toggle_rays_action.isChecked()
            
        self._show_rays_enabled = checked
        # Sync UI elements
        self.toggle_rays_action.setChecked(checked)

        # Delegate to viewer to toggle visibility
        try:
            # Use existing public API if available
            if hasattr(self.viewer, 'set_ray_visible'):
                self.viewer.set_ray_visible(checked)
            else:
                # Fallback: set internal flag and clear if disabling
                try:
                    self.viewer._show_rays_enabled = checked
                except Exception:
                    pass
                if not checked and self.viewer:
                    self.viewer.clear_ray()
        except Exception:
            pass
        
    def _on_opacity_changed(self, value):
        """Handle thumbnail opacity change."""
        self.thumbnail_opacity = value / 100.0
        # Delegate to viewer to update thumbnail opacity where possible
        try:
            if hasattr(self.viewer, 'set_thumbnail_opacity'):
                self.viewer.set_thumbnail_opacity(self.thumbnail_opacity)
            else:
                for actor in self.thumbnail_actors:
                    try:
                        actor.GetProperty().SetOpacity(self.thumbnail_opacity)
                    except Exception:
                        pass
        except Exception:
            pass

        # Update the render
        try:
            if self.viewer and self.viewer.plotter:
                self.viewer.plotter.update()
        except Exception:
            pass
    
    def _on_point_size_changed(self, value):
        """Handle point size change for point clouds."""
        self.point_size = value
        self.viewer.set_point_size(value)

    def _on_active_camera_changed(self, path):
        """Handle active camera changes coming from SelectionManager."""
        camera = self.cameras.get(path)
        if camera:
            # Clear any rendered rays before navigation
            self._clear_rays()

            # Select camera (updates frustum colors and thumbnails)
            self._select_camera(path, camera)

            # Match 3D view to camera perspective
            self._match_camera_perspective(camera)

            # Reorder cameras based on proximity to selected camera
            self._reorder_cameras(path, hide_distant_cameras=True)

            # Automatically navigate to the image (only on active change)
            self._goto_selected_image()

    def _on_selections_changed(self, selected_paths):
        """Handle selection set changes coming from SelectionManager.

        selected_paths is expected to be a set of image paths.
        """
        # Update highlight state for all cameras
        for path, camera in self.cameras.items():
            if path in selected_paths:
                camera.highlight()
            else:
                camera.unhighlight()
            # Delegate camera appearance update to viewer
            try:
                if hasattr(self.viewer, 'update_camera_appearance'):
                    self.viewer.update_camera_appearance(camera, opacity=self.thumbnail_opacity)
                else:
                    camera.update_appearance(self.viewer.plotter, opacity=self.thumbnail_opacity)
            except Exception:
                pass

        # Update local highlighted camera list for compatibility with existing code
        self.highlighted_cameras = [self.cameras.get(path) for path in selected_paths if path in self.cameras]

        # Update batched frustum colors via viewer
        try:
            highlighted = list(selected_paths)
            self.viewer.add_frustums(
                self.cameras,
                frustum_scale=self.frustum_scale,
                show_thumbnails=self._show_thumbnails_enabled,
                selected_camera=self.selected_camera,
                highlighted_paths=highlighted,
                hovered_camera=self.hovered_camera,
            )
        except Exception:
            # Fallback to legacy frustum manager update
            selected_path = self.selected_camera.image_path if self.selected_camera else None
            try:
                if hasattr(self.viewer, 'update_frustum_states'):
                    self.viewer.update_frustum_states(selected_path, list(selected_paths), self.hovered_camera)
                elif self.frustum_manager is not None:
                    self.frustum_manager.update_camera_states(selected_path, list(selected_paths), self.hovered_camera)
                    self.frustum_manager.mark_modified()
            except Exception:
                pass

        # Sync camera grid visuals
        try:
            self.camera_grid._sync_ui_to_model()
        except Exception:
            pass

        # Clear rays and update visibility filtering
        self._clear_rays()
        self._update_visibility_filter(list(selected_paths))
            
    def _on_grid_camera_selected(self, path):
        """Handle camera selection from the grid (double-click).
        
        Both changes the 3D view AND loads the image in the annotation window.
        """
        camera = self.cameras.get(path)
        if camera:
            # Clear any rendered rays before navigation
            self._clear_rays()
            
            # Select camera (updates frustum colors)
            self._select_camera(path, camera)
            
            # Highlight the selected camera to show its point cloud subset
            self.camera_grid.render_highlight_from_paths([path])
            self.highlighted_cameras = [camera]
            
            # Match 3D view to camera perspective
            self._match_camera_perspective(camera)
            
            # Reorder cameras based on proximity to selected camera
            self._reorder_cameras(path, hide_distant_cameras=True)
            
            # Automatically navigate to the image (only on double-click)
            self._goto_selected_image()
            
    def _on_grid_cameras_highlighted(self, paths):
        """Handle camera highlighting changes from the grid."""
        # Update frustum appearance using batched scalar updates
        selected_path = self.selected_camera.image_path if self.selected_camera else None
        
        # Batch update all camera states at once (O(1) instead of O(N) actor updates)
        try:
            if hasattr(self.viewer, 'update_frustum_states'):
                self.viewer.update_frustum_states(selected_path, paths, self.hovered_camera)
            elif self.frustum_manager is not None:
                self.frustum_manager.update_camera_states(selected_path, paths, self.hovered_camera)
                self.frustum_manager.mark_modified()
        except Exception:
            pass
            
        # Update the render via viewer
        try:
            if hasattr(self.viewer, 'render'):
                self.viewer.render()
            elif self.viewer and getattr(self.viewer, 'plotter', None):
                self.viewer.plotter.render()
        except Exception:
            pass
        
        # ALWAYS include the selected camera in visibility filtering
        # The selected camera (green) should always have its point cloud visible
        paths_for_filtering = list(paths) if paths else []
        if selected_path and selected_path not in paths_for_filtering:
            paths_for_filtering.append(selected_path)
        
        # Update visibility filtering based on highlighted cameras (including selected)
        self._update_visibility_filter(paths_for_filtering)
    
    def _update_visibility_filter(self, highlighted_paths):
        """
        Update point cloud visibility filtering based on highlighted/selected cameras.
        
        Computes the union of visible_indices from all cameras and updates the viewer
        to show only those points. Updates the status bar to show visibility stats.
        
        Args:
            highlighted_paths (list): List of image paths for highlighted cameras.
                                     Should always include the selected camera.
        """
        start_time = time.time()
        
        # TODO: Pre-compute visibility for all cameras using ThreadPoolExecutor on project load.
        # Shows progress bar, trades startup time for instant filtering.
        
        # Skip if no point cloud is loaded
        if not self.viewer or not self.viewer.point_cloud:
            total_time = time.time() - start_time
            print(f"⏱️ _update_visibility_filter: Skipped (no point cloud) in {total_time:.3f}s")
            return
        
        # Check if "Show Full Point Cloud" is enabled - if so, bypass filtering
        if self.toggle_full_cloud_action.isChecked():
            total_time = time.time() - start_time
            print(f"⏱️ _update_visibility_filter: Skipped (full cloud enabled) in {total_time:.3f}s")
            return
        
        # If no cameras provided, hide everything
        if not highlighted_paths:
            self.viewer.update_point_cloud_subset([])
            total_points = self.viewer.point_cloud.mesh.n_points
            try:
                self.camera_grid.stats_label.setText(f"Cameras: {len(self.cameras)} | Points: 0 / {total_points:,}")
            except Exception:
                pass
            total_time = time.time() - start_time
            print(f"⏱️ _update_visibility_filter: Hidden (no cameras) in {total_time:.3f}s")
            return
        
        # Collect visible_indices from all highlighted cameras
        all_visible_indices = []
        cameras_needing_visibility = []
        
        for path in highlighted_paths:
            camera = self.cameras.get(path)
            if camera:
                # Check if visibility data is already computed
                if camera.visible_indices is None:
                    cameras_needing_visibility.append(camera)
                else:
                    # Already have data
                    all_visible_indices.append(camera.visible_indices)
        
        # Batch compute visibility for cameras that need it
        if cameras_needing_visibility:
            # Show progress bar for visibility computation
            QApplication.setOverrideCursor(Qt.WaitCursor)
            progress = ProgressBar(self, title="Computing Visibility")
            progress.show()
            progress.start_progress(len(cameras_needing_visibility))
            
            try:
                # Prepare data for batch processing            
                points_world = self.viewer.point_cloud.get_points_array()
                camera_params = [(cam.K, cam.R, cam.t, cam.width, cam.height) for cam in cameras_needing_visibility]
                
                batch_results = VisibilityManager.compute_batch_visibility(points_world, camera_params)
                
                # Store results back to cameras and collect visible indices
                for i, (camera, result) in enumerate(zip(cameras_needing_visibility, batch_results)):
                    # Save to cache if manager is available
                    cache_path = None
                    if self.cache_manager is not None:
                        cache_path = self.cache_manager.save_visibility(
                            camera._raster.extrinsics,
                            self.viewer.point_cloud.file_path,
                            result['index_map'],
                            result['visible_indices']
                        )
                    
                    # Store in Raster
                    camera._raster.add_index_map(
                        result['index_map'],
                        cache_path,
                        result['visible_indices']
                    )
                    
                    all_visible_indices.append(result['visible_indices'])
                    progress.update_progress()
                    
            finally:
                QApplication.restoreOverrideCursor()
                progress.finish_progress()
                progress.close()
                progress = None
        
        # If no cameras have visibility data, trigger computation or hide cloud
        if not all_visible_indices:
            # **CHANGED: Trigger computation for cameras needing visibility data**
            if cameras_needing_visibility:
                # Compute visibility for cameras that need it
                points_world = self.viewer.point_cloud.get_points_array()
                camera_params = [(cam.K, cam.R, cam.t, cam.width, cam.height) for cam in cameras_needing_visibility]
                
                batch_results = VisibilityManager.compute_batch_visibility(points_world, camera_params)
                
                # Store results back to cameras and collect visible indices
                for camera, result in zip(cameras_needing_visibility, batch_results):
                    # Save to cache if manager is available
                    cache_path = None
                    if self.cache_manager is not None:
                        cache_path = self.cache_manager.save_visibility(
                            camera._raster.extrinsics,
                            self.viewer.point_cloud.file_path,
                            result['index_map'],
                            result['visible_indices']
                        )
                    
                    # Store in Raster
                    camera._raster.add_index_map(
                        result['index_map'],
                        cache_path,
                        result['visible_indices']
                    )
                    
                    all_visible_indices.append(result['visible_indices'])
            
            # If still no visibility data after computation attempt, hide the cloud
            if not all_visible_indices:
                # **CHANGED: Hide cloud instead of showing full cloud**
                self.viewer.update_point_cloud_subset([])  # Empty list hides the cloud
                total_points = self.viewer.point_cloud.mesh.n_points
                try:
                    self.camera_grid.stats_label.setText(
                        f"Cameras: {len(self.cameras)} | (No visibility data)"
                    )
                except Exception:
                    pass
                total_time = time.time() - start_time
                print(f"⏱️ _update_visibility_filter: Hidden (no visibility data) in {total_time:.3f}s")
                return
        
        # Compute union of all visible indices
        # Use np.union1d or concatenate + unique
        if len(all_visible_indices) == 1:
            union_indices = all_visible_indices[0]
        else:
            # Concatenate all arrays and get unique values
            concatenated = np.concatenate(all_visible_indices)
            union_indices = np.unique(concatenated)
        
        # Update the viewer to show only visible points
        self.viewer.update_point_cloud_subset(union_indices)
        
        # Update status bar
        total_points = self.viewer.point_cloud.mesh.n_points
        visible_count = len(union_indices)
        percentage = (visible_count / total_points * 100) if total_points > 0 else 0
        try:
            self.camera_grid.stats_label.setText(
                f"Cameras: {len(self.cameras)} | Visible Points: {percentage:.2f}%"
            )
        except Exception:
            pass
        
        total_time = time.time() - start_time
        print(
            f"⏱️ _update_visibility_filter: Updated visibility for {len(highlighted_paths)} "
            f"cameras in {total_time:.3f}s"
        )
            
    def _match_camera_perspective(self, camera):
        """Adapter to MVATViewer.match_camera_perspective."""
        try:
            if self.viewer and hasattr(self.viewer, 'match_camera_perspective'):
                self.viewer.match_camera_perspective(camera)
        except Exception as e:
            print(f"Failed to match camera perspective via viewer: {e}")
    
    def _clear_rays(self):
        """Clear any rendered rays from the viewer."""
        if self.viewer:
            self.viewer.clear_ray()
            
        # Also clear markers from the camera grid
        if self.mouse_bridge:
            self.mouse_bridge.clear_all_markers()
    
    def _calculate_camera_proximity_score(self, reference_camera, candidate_camera):
        """Calculate proximity score between two cameras.
        
        Combines spatial distance and view overlap to rank cameras.
        Higher scores indicate cameras that are closer and have more similar views.
        
        Args:
            reference_camera: The selected/reference Camera object.
            candidate_camera: The candidate Camera object to score.
            
        Returns:
            float: Proximity score (higher = closer/more similar view).
                  Returns 0 if cameras have no shared view.
        """
        # 1. Calculate spatial distance (Euclidean distance between camera positions)
        spatial_distance = np.linalg.norm(
            reference_camera.position - candidate_camera.position
        )
        
        # 2. Calculate view similarity (dot product of view directions)
        # View direction is Z-axis in camera frame transformed to world
        ref_view_dir = reference_camera.R.T @ np.array([0, 0, 1])
        cand_view_dir = candidate_camera.R.T @ np.array([0, 0, 1])
        
        # Normalize to ensure unit vectors
        ref_view_dir = ref_view_dir / np.linalg.norm(ref_view_dir)
        cand_view_dir = cand_view_dir / np.linalg.norm(cand_view_dir)
        
        # Dot product gives cosine of angle between view directions
        # Range: [-1, 1] where 1 = same direction, -1 = opposite, 0 = perpendicular
        view_alignment = np.dot(ref_view_dir, cand_view_dir)
        
        # 3. Normalize spatial distance to a score (0 to 1)
        # Use exponential decay: closer cameras get higher scores
        # Adjust the decay rate based on your scene scale
        try:
            bounds = None
            if hasattr(self.viewer, 'get_bounds'):
                bounds = self.viewer.get_bounds()
            elif getattr(self.viewer, 'plotter', None) is not None:
                bounds = self.viewer.plotter.bounds
            scene_size = np.sqrt(
                (bounds[1] - bounds[0])**2 + 
                (bounds[3] - bounds[2])**2 + 
                (bounds[5] - bounds[4])**2
            )
            # Normalize distance by scene size
            normalized_distance = spatial_distance / (scene_size + 1e-6)
        except:
            # Fallback normalization
            normalized_distance = spatial_distance / 10.0
        
        # Distance score: 1 at distance 0, decays exponentially
        distance_score = np.exp(-2.0 * normalized_distance)
        
        # 4. Convert view alignment to score (0 to 1)
        # Map from [-1, 1] to [0, 1], where 1 = same direction
        view_score = (view_alignment + 1.0) / 2.0
        
        # 5. Combine scores with weights
        # 50% weight on distance, 50% weight on view direction
        combined_score = 0.5 * distance_score + 0.5 * view_score
        
        # 6. Filter out cameras with very different views (optional)
        # If view alignment is negative (> 90 degrees apart), set score to 0
        if view_alignment < 0:
            combined_score = 0.0
        
        return combined_score
    
    def _reorder_cameras(self, reference_path, hide_distant_cameras=True):
        """Reorder cameras in the grid based on proximity to a reference camera.
        
        Args:
            reference_path: Image path of the reference/selected camera.
            hide_distant_cameras (bool): If True, hide cameras with zero overlap score.
        """
        if not hasattr(self, 'camera_grid'):
            return
            
        reference_camera = self.cameras.get(reference_path)
        if not reference_camera:
            return
        
        # Calculate proximity scores for all cameras
        camera_scores = []
        for path, camera in self.cameras.items():
            if path == reference_path:
                # Reference camera gets highest score (always first)
                score = float('inf')
            else:
                score = self._calculate_camera_proximity_score(reference_camera, camera)
            
            # Filter based on hide_distant_cameras setting
            if hide_distant_cameras and score == 0.0 and path != reference_path:
                continue  # Skip cameras with no shared view
            
            camera_scores.append((path, score))
        
        # Sort by score (descending - highest score first)
        camera_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Extract ordered paths
        ordered_paths = [path for path, score in camera_scores]
        
        # Update camera grid order
        self.camera_grid.set_camera_order(ordered_paths)
    
    def _on_pick(self, point):
        """Handle picking in the 3D view."""
        if point is None:
            return
            
        # Find the closest camera to the picked point
        min_dist = float('inf')
        closest_camera = None
        closest_path = None
        
        for path, camera in self.cameras.items():
            dist = np.linalg.norm(camera.position - point)
            if dist < min_dist:
                min_dist = dist
                closest_camera = camera
                closest_path = path
                
        # Select if within reasonable distance (heuristic based on scale)
        if closest_camera and min_dist < self.frustum_scale * 2:
            # Delegate selection to the central model
            self.selection_model.set_active(closest_path)
        else:
            # Clear model selections
            self.selection_model.clear_all()
            
    def _select_camera(self, path, camera, emit_signal=True):
        """Select a camera and update UI."""
        previous_camera = self.selected_camera
        
        # Deselect previous camera
        if previous_camera:
            previous_camera.deselect()
            try:
                if hasattr(self.viewer, 'update_camera_appearance'):
                    self.viewer.update_camera_appearance(previous_camera, opacity=self.thumbnail_opacity)
                else:
                    previous_camera.update_appearance(self.viewer.plotter, opacity=self.thumbnail_opacity)
            except Exception:
                pass
        
        # Update selected camera reference
        self.selected_camera = camera
        self.selected_camera_path = path

        # Sync the camera grid's selection visuals (do not call render_selection_from_path to avoid recursion)
        try:
            self.camera_grid._sync_ui_to_model()
        except Exception:
            pass
        
        # Select new camera
        camera.select()
        try:
            if hasattr(self.viewer, 'update_camera_appearance'):
                self.viewer.update_camera_appearance(camera, opacity=self.thumbnail_opacity)
            else:
                camera.update_appearance(self.viewer.plotter, opacity=self.thumbnail_opacity)
        except Exception:
            pass
        
        # Get highlighted paths
        highlighted_paths = [cam.image_path for cam in self.highlighted_cameras]
        
        # Update batched frustum colors
        try:
            if hasattr(self.viewer, 'update_frustum_states'):
                self.viewer.update_frustum_states(path, highlighted_paths, self.hovered_camera)
            elif self.frustum_manager is not None:
                self.frustum_manager.update_camera_states(path, highlighted_paths, self.hovered_camera)
                self.frustum_manager.mark_modified()
        except Exception:
            pass
        
        # Lazy thumbnail loading: update thumbnail for new selection
        if self._show_thumbnails_enabled:
            # Remove previous camera's thumbnail if different
            if previous_camera and previous_camera != camera:
                try:
                    if hasattr(self.viewer, 'remove_thumbnails'):
                        self.viewer.remove_thumbnails()
                        self.thumbnail_actors = list(getattr(self.viewer, 'thumbnail_actors', []) or [])
                    elif self.thumbnail_actors is not None:
                        for actor in list(self.thumbnail_actors):
                            try:
                                if hasattr(self.viewer, 'plotter'):
                                    self.viewer.plotter.remove_actor(actor)
                            except Exception:
                                pass
                        self.thumbnail_actors.clear()
                except Exception:
                    pass

            # Add thumbnail for newly selected camera
            try:
                if not self.thumbnail_actors:
                    if hasattr(self.viewer, '_add_thumbnail_for_camera'):
                        self.viewer._add_thumbnail_for_camera(camera, scale=self.frustum_scale)
                        self.thumbnail_actors = list(getattr(self.viewer, 'thumbnail_actors', []) or [])
            except Exception:
                pass
        
        # Update the plotter to show selection
        try:
            if hasattr(self.viewer, 'render'):
                self.viewer.render()
            elif self.viewer and getattr(self.viewer, 'plotter', None):
                self.viewer.plotter.render()
        except Exception:
            pass
        
        # Emit signal for bi-directional sync (unless suppressed)
        if emit_signal:
            self.cameraSelectedInMVAT.emit(path)
            # Emit focal point projection if we have a focal point
            if self.current_focal_point is not None:
                pixel = camera.project(self.current_focal_point)
                # TODO
                # if not np.isnan(pixel).any():
                #     self.focalPointProjected.emit(path, pixel[0], pixel[1])
                # else:
                #     self.focalPointProjected.emit(path, np.nan, np.nan)
        
    def _deselect_camera(self):
        """Deselect the current camera."""
        if self.selected_camera:
            self.selected_camera.deselect()
            self.selected_camera.update_appearance(self.viewer.plotter, opacity=self.thumbnail_opacity)
            
            # Get highlighted paths
            highlighted_paths = [cam.image_path for cam in self.highlighted_cameras]
            
            # Update batched frustum colors (no selection, keep highlights)
            try:
                if hasattr(self.viewer, 'update_frustum_states'):
                    self.viewer.update_frustum_states(None, highlighted_paths, self.hovered_camera)
                elif self.frustum_manager is not None:
                    self.frustum_manager.update_camera_states(None, highlighted_paths, self.hovered_camera)
                    self.frustum_manager.mark_modified()
            except Exception:
                pass
            
            # Remove thumbnail actor (lazy unloading)
            if self._show_thumbnails_enabled:
                self._remove_thumbnails()
            
            self.selected_camera = None
        
        # Clear ray visualization when camera is deselected
        if self.viewer:
            self.viewer.clear_ray()
        
        # Clear markers
        if self.mouse_bridge:
            self.mouse_bridge.clear_all_markers()
            
        if self.viewer and self.viewer.plotter:
            self.viewer.plotter.render()
            
    def _auto_select_first_camera(self):
        """Auto-select the first camera if none is selected."""
        if self.selected_camera is None and self.cameras:
            first_path = next(iter(self.cameras))
            # Use the central selection model to set active camera
            self.selection_model.set_active(first_path)
        
    def _goto_selected_image(self):
        """Navigate to the selected camera's image in the main window."""
        if not self.selected_camera:
            return
            
        path = self.selected_camera.image_path
        
        # Use ImageWindow to select the image
        try:
            self.image_window.load_image_by_path(path)
        except Exception as e:
            QMessageBox.warning(self, "Navigation Error", f"Could not navigate to image: {e}")
            
    def _refresh_scene(self):
        """Apply changes and refresh the entire scene by reloading cameras."""
        # Read controls from MVATViewer toolbars
        try:
            self.thumbnail_opacity = self.viewer.opacity_slider.value() / 100.0
        except Exception:
            pass

        try:
            # viewer already updates its own point size; keep MVATWindow in sync
            self.point_size = int(getattr(self.viewer, 'point_size', self.point_size))
        except Exception:
            pass
        
        # Update show flags
        self._show_wireframes_enabled = self.toggle_wireframes_action.isChecked()
        self._show_thumbnails_enabled = self.toggle_thumbnails_action.isChecked()
        self._show_point_cloud_enabled = self.toggle_point_cloud_action.isChecked()
        self._show_rays_enabled = self.toggle_rays_action.isChecked()
        
        # Clear existing
        self.cameras.clear()
        self.selected_camera = None
        self.highlighted_cameras.clear()
        self._initialized = False
        
        # Reload
        self._load_cameras()
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.camera_grid.recalculate_layout()
        self.camera_grid._update_visible_widgets()
        
    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Escape:
            self._deselect_camera()
        elif event.key() == Qt.Key_R:
            self._reset_camera_view()
        elif event.key() == Qt.Key_F:
            self._fit_to_view()
        elif event.key() == Qt.Key_Control:
            self.camera_grid.update_hover_visuals(True)
        else:
            super().keyPressEvent(event)
            
    def keyReleaseEvent(self, event):
        """Handle key release events."""
        if event.key() == Qt.Key_Control:
            self.camera_grid.update_hover_visuals(False)
        else:
            super().keyReleaseEvent(event)