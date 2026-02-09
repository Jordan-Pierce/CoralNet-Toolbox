"""
MultiView Annotation Tool (MVAT) Window

A 3D viewer for visualizing camera frustums and navigating MultiView imagery.
Uses PyVista for 3D rendering and integrates with the main application's RasterManager.
"""

import warnings

import numpy as np

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QToolBar, QAction, QLabel, QSlider, QCheckBox,
    QGroupBox, QMessageBox, QApplication, QFrame, QDoubleSpinBox,
    QSizePolicy, QSpinBox
)

from coralnet_toolbox.MVAT.ui.QtMVATViewer import MVATViewer
from coralnet_toolbox.MVAT.ui.QtCameraGrid import CameraGrid
from coralnet_toolbox.MVAT.core.Camera import Camera
from coralnet_toolbox.MVAT.core.Ray import CameraRay, BatchedRayManager
from coralnet_toolbox.MVAT.core.Frustum import BatchedFrustumManager

from coralnet_toolbox.MVAT.core.constants import (MARKER_COLOR_SELECTED, 
                                                  MARKER_COLOR_HIGHLIGHTED, 
                                                  RAY_COLOR_SELECTED, 
                                                  RAY_COLOR_HIGHLIGHTED)

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

# Throttle interval for mouse position updates (milliseconds)
# 16ms is approximately 60fps for responsive mouse tracking
MOUSE_THROTTLE_MS = 16


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
        
        # Get highlighted cameras and create rays from them to the world point
        highlighted_cameras = self.mvat_window.camera_grid.get_highlighted_cameras()
        
        # List to store rays for 3D viewer: [(Ray, Color), ...]
        rays_with_colors = [(ray, RAY_COLOR_SELECTED)]
        
        # Dictionary to store visibility status for 2D markers: {image_path: is_occluded}
        visibility_status = {}
        
        for target_cam in highlighted_cameras:
            # Skip self
            if target_cam.image_path == camera.image_path:
                continue
            
            # --- OCCLUSION CHECK ---
            # TODO: Handle case where target_cam does not have z-channel data.
            # Currently assumes visible, but consider user warning or alternative occlusion method (e.g., mesh-based).
            is_occluded = target_cam.is_point_occluded_depth_based(ray.terminal_point, depth_threshold=0.15)
            visibility_status[target_cam.image_path] = is_occluded
            
            # Create ray from target camera to the world point
            target_ray = CameraRay.from_world_point_and_camera(
                world_point=ray.terminal_point,
                camera=target_cam
            )
            
            # Color coding: Red if blocked, Cyan if visible
            if is_occluded:
                ray_color = (255, 0, 0)  # Red
            else:
                ray_color = RAY_COLOR_HIGHLIGHTED  # Cyan
                
            rays_with_colors.append((target_ray, ray_color))
        
        # Update 3D ray visualization with all rays
        self.mvat_window.viewer.show_rays(rays_with_colors)
        
        # Project ray onto other camera views (using selected camera's ray terminal point)
        projections = ray.project_to_cameras(self.mvat_window.cameras)
        
        # Update markers on visible camera widgets with appropriate colors
        self._update_camera_markers(projections, ray.has_accurate_depth, highlighted_cameras, visibility_status)
        
    def _update_camera_markers(self, projections: dict, accurate: bool, 
                               highlighted_cameras: list, visibility_status: dict):
        """
        Update marker positions on visible camera widgets.
        
        Only updates markers for widgets that are currently visible in the
        camera grid viewport. Uses lime color for selected camera marker,
        cyan for highlighted camera markers.
        
        Args:
            projections: Dict mapping image_path to (x, y, is_valid) tuples.
            accurate: Whether the depth used was accurate.
            highlighted_cameras: List of highlighted Camera objects.
            visibility_status: Dict mapping image_path to is_occluded bool.
        """        
        camera_grid = self.mvat_window.camera_grid
        selected_camera = self.mvat_window.selected_camera
        
        # Get set of highlighted camera paths for quick lookup
        highlighted_paths = {cam.image_path for cam in highlighted_cameras}
        
        # Get selected camera path
        selected_path = selected_camera.image_path if selected_camera else None
        
        # Get visible widgets from the camera grid
        visible_widgets = camera_grid.get_visible_widgets()
        
        # Update each visible widget
        for path, widget in visible_widgets.items():
            # Only draw markers on selected or highlighted cameras
            if path not in (highlighted_paths | {selected_path} if selected_path else highlighted_paths):
                widget.clear_marker()
                continue
                
            if path in projections:
                px, py, is_valid = projections[path]
                
                if is_valid:
                    # Check occlusion status (Default to False if not in dict)
                    is_occluded = visibility_status.get(path, False)
                    
                    # Determine color
                    if path == selected_path:
                        color = MARKER_COLOR_SELECTED
                        is_occluded = False # Selected camera is never occluded from itself
                    elif path in highlighted_paths:
                        color = MARKER_COLOR_HIGHLIGHTED
                    else:
                        color = MARKER_COLOR_HIGHLIGHTED
                    
                    # Pass is_occluded to the widget
                    widget.set_marker_position(px, py, accurate=accurate, 
                                               color=color, is_occluded=is_occluded)
                else:
                    widget.clear_marker()
            else:
                widget.clear_marker()
                
    def clear_all_markers(self):
        """Clear markers from all visible camera widgets."""
        camera_grid = self.mvat_window.camera_grid
        visible_widgets = camera_grid.get_visible_widgets()
        
        for widget in visible_widgets.values():
            widget.clear_marker()
            
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
        
        # Batched geometry managers for efficient rendering (O(1) draw calls)
        self.frustum_manager = BatchedFrustumManager()
        self.ray_manager = BatchedRayManager()
        
        # Actor lists for thumbnails (cannot be batched due to textures)
        self.thumbnail_actors = []
        
        # Display status
        self.frustum_scale = 0.1
        self._show_wireframes_enabled = True
        self._show_thumbnails_enabled = True
        self.thumbnail_opacity = 0.25
        self._show_point_cloud_enabled = True
        self.point_size = 1
        self._show_rays_enabled = True
        
        # Mouse position bridge for cross-window sync
        self.mouse_bridge = None  # Initialized after UI setup
        
        # Reference to annotation window for signal connections
        self.annotation_window = main_window.annotation_window
        
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
        self.setWindowIcon(QIcon(get_icon("camera.png")))
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
        """Create the central widget, top status bar, and splitters."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main Vertical Layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(5)  # Reduce spacing between bar and splitter
        
        # 1. Top status Group Box (Mimics MainWindow Status/Param Bar)
        self.status_bar_group_box = QGroupBox("Status Bar")
        # FORCE HEIGHT: Match typical compact status bar height (~50-60px)
        self.status_bar_group_box.setMaximumHeight(65) 
        
        self.status_bar_layout = QVBoxLayout(self.status_bar_group_box)
        # TIGHT MARGINS: (left, top, right, bottom)
        self.status_bar_layout.setContentsMargins(5, 5, 5, 5) 
        self.status_bar_layout.setSpacing(0)  # No spacing in vertical layout
        
        # Horizontal layout for the widgets
        self.horizontal_layout = QHBoxLayout()
        self.horizontal_layout.setSpacing(10)  # Spacing between widgets
        
        # --- Widget: Stats Label ---
        self.stats_label = QLabel("Cameras: 0")
        self.horizontal_layout.addWidget(self.stats_label)
        
        # Vertical Separator
        self.horizontal_layout.addWidget(self._create_v_line())
        
        # --- Widget: Frustum Scale ---
        scale_label = QLabel("Scale:")
        self.scale_spinbox = QDoubleSpinBox()
        self.scale_spinbox.setRange(0.01, 10.0)
        self.scale_spinbox.setSingleStep(0.1)
        self.scale_spinbox.setValue(self.frustum_scale)
        self.scale_spinbox.setToolTip("Adjust camera frustum size")
        self.scale_spinbox.valueChanged.connect(self._on_scale_changed)  
        self.horizontal_layout.addWidget(scale_label)
        self.horizontal_layout.addWidget(self.scale_spinbox)
        
        # --- Widget: Opacity Slider ---
        opacity_label = QLabel("Opacity:")
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(self.thumbnail_opacity * 100))
        self.opacity_slider.setFixedWidth(100)
        self.opacity_slider.setToolTip("Adjust thumbnail opacity")
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)  
        self.horizontal_layout.addWidget(opacity_label)
        self.horizontal_layout.addWidget(self.opacity_slider)
        
        # --- Widget: Point Size ---
        point_size_label = QLabel("Point Size:")
        self.point_size_spinbox = QSpinBox()
        self.point_size_spinbox.setRange(1, 20)
        self.point_size_spinbox.setValue(self.point_size)
        self.point_size_spinbox.setToolTip("Adjust point cloud point size")
        self.point_size_spinbox.valueChanged.connect(self._on_point_size_changed)
        self.horizontal_layout.addWidget(point_size_label)
        self.horizontal_layout.addWidget(self.point_size_spinbox)
        
        # --- Widget: Checkboxes ---
        self.wireframe_checkbox = QCheckBox("Wireframes")
        self.wireframe_checkbox.setChecked(self._show_wireframes_enabled)
        self.wireframe_checkbox.toggled.connect(self._toggle_wireframes)  
        self.horizontal_layout.addWidget(self.wireframe_checkbox)
        
        self.thumbnail_checkbox = QCheckBox("Thumbnails")
        self.thumbnail_checkbox.setChecked(self._show_thumbnails_enabled)
        self.thumbnail_checkbox.toggled.connect(self._toggle_thumbnails)  
        self.horizontal_layout.addWidget(self.thumbnail_checkbox)
        
        self.point_cloud_checkbox = QCheckBox("Point cloud")
        self.point_cloud_checkbox.setChecked(self._show_point_cloud_enabled)
        self.point_cloud_checkbox.toggled.connect(self._toggle_point_cloud)  
        self.horizontal_layout.addWidget(self.point_cloud_checkbox)
        
        self.rays_checkbox = QCheckBox("Rays")
        self.rays_checkbox.setChecked(self._show_rays_enabled)
        self.rays_checkbox.toggled.connect(self._toggle_rays)  
        self.horizontal_layout.addWidget(self.rays_checkbox)
        
        # Push everything to the left
        self.horizontal_layout.addStretch()
        
        # Add stretches and horizontal layout to vertical layout for centering
        self.status_bar_layout.addStretch()
        self.status_bar_layout.addLayout(self.horizontal_layout)
        self.status_bar_layout.addStretch()
        
        # Add status box to main layout
        self.main_layout.addWidget(self.status_bar_group_box)
        
        # 2. Main Horizontal Splitter
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)
        
        # --- Left Panel: 3D Viewer ---
        # Create the viewer container class
        self.viewer = MVATViewer(self, point_size=self.point_size, show_rays=self._show_rays_enabled)

        # Enable picking for camera selection using the viewer's plotter
        self.viewer.plotter.enable_point_picking(
            callback=self._on_pick,
            show_message=False,
            use_picker=True,
            pickable_window=True
        )

        # Wrap the viewer in a groupbox
        left_groupbox = QGroupBox("3D Viewer")
        left_layout = QVBoxLayout(left_groupbox)
        left_layout.addWidget(self.viewer)
        self.splitter.addWidget(left_groupbox)  # Add the groupbox instead of the viewer directly
        
        # --- Right Panel: Camera Grid ---
        self.right_container = QGroupBox("Camera Grid")
        right_layout = QVBoxLayout(self.right_container)
        right_layout.setContentsMargins(2, 2, 2, 2)
        
        # Create the camera grid widget
        self.camera_grid = CameraGrid(mvat_window=self)
        self.camera_grid.camera_selected.connect(self._on_grid_camera_selected)
        self.camera_grid.camera_highlighted_single.connect(self._on_grid_camera_highlighted_single)
        self.camera_grid.cameras_highlighted.connect(self._on_grid_cameras_highlighted)
        right_layout.addWidget(self.camera_grid)
        
        self.splitter.addWidget(self.right_container)
        
        # Set minimum width for camera grid to prevent collapse
        # Ensure at least one column of thumbnails is always visible
        self.right_container.setMinimumWidth(512)  # Thumbnail + scrollbar/padding
        
        # Set stretch factors: 3D viewer gets 3x weight, grid gets 1x weight
        self.splitter.setStretchFactor(0, 3)  # Left panel (3D viewer)
        self.splitter.setStretchFactor(1, 1)  # Right panel (camera grid)
        
        # Prevent panels from being completely collapsed
        self.splitter.setChildrenCollapsible(False)
        
        # Connect splitter resize to grid layout update
        self.splitter.splitterMoved.connect(self._on_splitter_moved)
        
        # Set splitter proportions (75% Viewer, 25% Right Panel)
        self.splitter.setSizes([900, 300])

    def _create_v_line(self):
        """Helper to create a vertical separator line."""
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        return line
    
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
        
        # Connect our signal to navigate main app (for completeness)
        self.cameraSelectedInMVAT.connect(self._on_camera_selected_sync)
        
        # Connect camera grid highlight changes to clear rays
        # This ensures stale rays from previously-highlighted cameras are removed
        self.camera_grid.cameras_highlighted.connect(self._on_highlights_changed)
    
    def _on_main_image_loaded(self, path: str):
        """
        Handle image loaded in main application.
        
        Updates MVAT selection to match main app's current image.
        
        Args:
            path: Image path that was loaded in the main app.
        """
        if path in self.cameras:
            # Update camera grid selection without triggering navigation
            self.camera_grid.render_selection_from_path(path)
            
            # Update 3D view selection
            camera = self.cameras[path]
            self._select_camera(path, camera)
            self._match_camera_perspective(camera)
    
    def _on_camera_selected_sync(self, path: str):
        """
        Handle camera selection sync (internal slot).
        
        This is connected to our own signal for extensibility.
        
        Args:
            path: Image path of the selected camera.
        """
        # Navigation to main app is handled in _goto_selected_image
        pass
    
    def _on_highlights_changed(self, highlighted_paths: list):
        """
        Handle camera highlight selection changes.
        
        Clears rays when highlights change to ensure stale rays from
        previously-highlighted cameras are removed. The rays will be
        recreated on the next mouse move event.
        
        Args:
            highlighted_paths: List of currently highlighted camera paths.
        """
        self._clear_rays()
        
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
            if hasattr(self.camera_grid, 'cameras_highlighted'):
                self.camera_grid.cameras_highlighted.disconnect(self._on_highlights_changed)
        except:
            pass  # Signals may already be disconnected
        
        # Cleanup mouse bridge
        if self.mouse_bridge:
            self.mouse_bridge.cleanup()
            self.mouse_bridge = None
        
        # Clear camera references
        self.cameras.clear()
        self.selected_camera = None
        
        # Clear batched geometry managers
        self.frustum_manager.clear()
        self.ray_manager.clear()
        self.thumbnail_actors.clear()
        
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
            
        # Show progress bar
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
                QApplication.processEvents()
                
        finally:
            progress.close()
            
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
            
        # Update stats
        self.stats_label.setText(f"Cameras: {valid_count} / {len(all_paths)}")
        
        # Populate camera grid
        self.camera_grid.set_cameras(self.cameras)
        
        # Render frustums
        self._render_frustums()
        
        # Fit view to show all cameras
        self._fit_to_view()
        
        # Auto-select the current image if it exists in the loaded cameras
        if hasattr(self.annotation_window, 'current_image_path') and self.annotation_window.current_image_path:
            current_path = self.annotation_window.current_image_path
            if current_path in self.cameras:
                # Select the camera corresponding to the current image
                self._select_camera(current_path, self.cameras[current_path])
                # Update camera grid selection
                self.camera_grid.render_selection_from_path(current_path)
        
    def _render_frustums(self):
        """Render all camera frustums in the 3D scene using batched geometry."""
        if not self.viewer or not self.viewer.plotter:
            return
            
        # Clear existing actors
        self.viewer.plotter.clear()
        
        # Clear thumbnail actors (textures cannot be batched)
        self.thumbnail_actors.clear()
        
        # Clear batched managers
        self.frustum_manager.clear()
        self.ray_manager.clear()
        
        # Re-add point cloud
        self.viewer.point_cloud_actor = None
        self.viewer.add_point_cloud()
        self.viewer.set_point_cloud_visible(self._show_point_cloud_enabled)
        
        # Add a reference grid
        self.viewer.plotter.add_axes()
        
        # Build batched wireframe geometry (single mesh for all frustums)
        if self._show_wireframes_enabled and self.cameras:
            merged_mesh = self.frustum_manager.build_frustum_batch(
                self.cameras, 
                scale=self.frustum_scale
            )
            
            if merged_mesh is not None:
                # Add single merged actor to plotter
                self.frustum_manager.add_to_plotter(self.viewer.plotter, line_width=1.5)
                
                # Apply current selection state
                selected_path = self.selected_camera.image_path if self.selected_camera else None
                highlighted_paths = list(getattr(self.camera_grid, 'highlighted_paths', set()))
                self.frustum_manager.update_camera_states(selected_path, highlighted_paths)
                self.frustum_manager.mark_modified()
        
        # Thumbnails: Only render for selected camera (lazy loading)
        # This avoids creating N texture actors; just 1 for the selected camera
        if self._show_thumbnails_enabled and self.selected_camera:
            self._add_thumbnail_for_camera(self.selected_camera)
                
        # Update the render
        self.viewer.plotter.update()
    
    def _add_thumbnail_for_camera(self, camera):
        """Add thumbnail actor for a single camera (lazy loading)."""
        try:
            # Clear old frustum actor cache to allow recreation
            camera.frustum.image_actors.clear()
            
            actor = camera.frustum.create_image_plane_actor(
                self.viewer.plotter, 
                scale=self.frustum_scale,
                opacity=self.thumbnail_opacity
            )
            self.thumbnail_actors.append(actor)
        except Exception as e:
            print(f"Failed to render thumbnail for {camera.image_path}: {e}")
    
    def _remove_thumbnails(self):
        """Remove all thumbnail actors from the plotter."""
        for actor in self.thumbnail_actors:
            try:
                self.viewer.plotter.remove_actor(actor)
            except:
                pass
        self.thumbnail_actors.clear()
        
        # Clear frustum image actor caches
        for camera in self.cameras.values():
            camera.frustum.image_actors.clear()
        
    def _reset_camera_view(self):
        """Reset the 3D camera to default view."""
        if self.viewer and self.viewer.plotter:
            self.viewer.plotter.reset_camera()
            self.viewer.plotter.view_isometric()
            
    def _fit_to_view(self):
        """Fit all objects in the view."""
        if self.viewer and self.viewer.plotter:
            self.viewer.plotter.reset_camera()
            
    def _toggle_wireframes(self, checked=None):
        """Toggle wireframe visibility."""
        if checked is None:
            checked = self.toggle_wireframes_action.isChecked()
            
        self._show_wireframes_enabled = checked
        
        # Sync UI elements
        self.toggle_wireframes_action.setChecked(checked)
        self.wireframe_checkbox.blockSignals(True)
        self.wireframe_checkbox.setChecked(checked)
        self.wireframe_checkbox.blockSignals(False)
        
        # Update visibility of batched wireframe actor
        self.frustum_manager.set_visibility(checked)
        
        # Update the render
        if self.viewer and self.viewer.plotter:
            self.viewer.plotter.update()
        
    def _toggle_thumbnails(self, checked=None):
        """Toggle thumbnail visibility."""
        if checked is None:
            checked = self.toggle_thumbnails_action.isChecked()
            
        self._show_thumbnails_enabled = checked
        
        # Sync UI elements
        self.toggle_thumbnails_action.setChecked(checked)
        self.thumbnail_checkbox.blockSignals(True)
        self.thumbnail_checkbox.setChecked(checked)
        self.thumbnail_checkbox.blockSignals(False)
        
        # Update visibility of existing actors
        for actor in self.thumbnail_actors:
            actor.SetVisibility(checked)
        
        # Update the render
        if self.viewer and self.viewer.plotter:
            self.viewer.plotter.update()
        
    def _toggle_point_cloud(self, checked):
        """Toggle point cloud visibility."""
        self._show_point_cloud_enabled = checked
        self.viewer.set_point_cloud_visible(checked)
        
    def _toggle_rays(self, checked=None):
        """Toggle ray visibility."""
        if checked is None:
            checked = self.toggle_rays_action.isChecked()
            
        self._show_rays_enabled = checked
        self.viewer._show_rays_enabled = checked
        
        # Sync UI elements
        self.toggle_rays_action.setChecked(checked)
        self.rays_checkbox.blockSignals(True)
        self.rays_checkbox.setChecked(checked)
        self.rays_checkbox.blockSignals(False)
        
        # Clear rays if disabling
        if not checked and self.viewer:
            self.viewer.clear_ray()
        
    def _on_scale_changed(self, value):
        """Handle frustum scale change."""
        self.frustum_scale = value
        
        # Regenerate all frustums with new scale
        # This creates new geometry at the correct size rather than transforming existing geometry
        self._render_frustums()
        
    def _on_opacity_changed(self, value):
        """Handle thumbnail opacity change."""
        self.thumbnail_opacity = value / 100.0
        
        # Update opacity of existing thumbnail actors
        for actor in self.thumbnail_actors:
            actor.GetProperty().SetOpacity(self.thumbnail_opacity)
        
        # Update the render
        if self.viewer and self.viewer.plotter:
            self.viewer.plotter.update()
    
    def _on_point_size_changed(self, value):
        """Handle point size change for point clouds."""
        self.point_size = value
        self.viewer.set_point_size(value)
        
    def _on_splitter_moved(self, pos, index):
        """Handle splitter resize to update camera grid layout."""
        if hasattr(self, 'camera_grid'):
            self.camera_grid.recalculate_layout()
    
    def _on_grid_camera_highlighted_single(self, path):
        """Handle single camera highlight from the grid (single-click).
        
        Only updates frustum colors without changing the 3D view.
        Use double-click to both change view and load image.
        """
        # Note: We don't change the 3D view on single-click highlight
        # That only happens on double-click selection
        # The highlighting is handled by _on_grid_cameras_highlighted signal
        pass
            
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
        self.frustum_manager.update_camera_states(selected_path, paths)
        self.frustum_manager.mark_modified()
            
        # Update the render
        if self.viewer and self.viewer.plotter:
            self.viewer.plotter.render()
            
    def _match_camera_perspective(self, camera):
        """Match the 3D viewer perspective to a camera's viewpoint."""
        if not self.viewer or not self.viewer.plotter:
            return
            
        try:
            # Get camera position (optical center in world coordinates)
            position = camera.position
            
            # Calculate view direction: Z-axis in camera frame transformed to world
            # Camera looks along +Z in camera coordinates
            view_direction = camera.R.T @ np.array([0, 0, 1])
            
            # Calculate up vector: -Y in camera frame (Y points down in image)
            up_vector = camera.R.T @ np.array([0, -1, 0])
            
            # Calculate focal distance based on scene bounds for better viewing
            # This ensures we're not too zoomed in or out regardless of frustum scale
            try:
                bounds = self.viewer.plotter.bounds
                # Calculate scene diagonal for a reasonable focal distance
                scene_size = np.sqrt(
                    (bounds[1] - bounds[0])**2 + 
                    (bounds[3] - bounds[2])**2 + 
                    (bounds[5] - bounds[4])**2
                )
                # Use 20% of scene size as focal distance (can be tuned)
                focal_distance = scene_size * 0.2
            except:
                # Fallback to a fixed reasonable distance if bounds aren't available
                focal_distance = 5.0
            
            focal_point = position + view_direction * focal_distance
            
            # Set the plotter camera
            self.viewer.plotter.camera.position = position.tolist()
            self.viewer.plotter.camera.focal_point = focal_point.tolist()
            self.viewer.plotter.camera.up = up_vector.tolist()
            
            # Optional: Match camera field of view from intrinsics
            # This makes the 3D view more accurately represent what the camera sees
            try:
                if camera.K is not None:
                    # Calculate vertical field of view from intrinsics
                    # FOV = 2 * atan(height / (2 * fy))
                    fy = camera.K[1, 1]
                    height = camera.height
                    fov_rad = 2 * np.arctan(height / (2 * fy))
                    fov_deg = np.degrees(fov_rad)
                    # Clamp FOV to reasonable range
                    fov_deg = np.clip(fov_deg, 10, 120)
                    self.viewer.plotter.camera.view_angle = fov_deg
            except:
                pass  # Use default FOV if calculation fails
            
            # Update the render
            self.viewer.plotter.update()
            
        except Exception as e:
            print(f"Failed to match camera perspective: {e}")
    
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
            self._select_camera(closest_path, closest_camera)
            # Sync selection to grid
            if hasattr(self, 'camera_grid'):
                self.camera_grid.render_selection_from_path(closest_path)
        else:
            self._deselect_camera()
            if hasattr(self, 'camera_grid'):
                self.camera_grid.clear_all_selections()
            
    def _select_camera(self, path, camera):
        """Select a camera and update UI."""
        previous_camera = self.selected_camera
        
        # Update selected camera reference
        self.selected_camera = camera
        
        # Get highlighted paths
        highlighted_paths = list(getattr(self.camera_grid, 'highlighted_paths', set()))
        
        # Update batched frustum colors
        self.frustum_manager.update_camera_states(path, highlighted_paths)
        self.frustum_manager.mark_modified()
        
        # Lazy thumbnail loading: update thumbnail for new selection
        if self._show_thumbnails_enabled:
            # Remove previous camera's thumbnail if different
            if previous_camera and previous_camera != camera:
                self._remove_thumbnails()
            
            # Add thumbnail for newly selected camera
            if not self.thumbnail_actors:
                self._add_thumbnail_for_camera(camera)
        
        # Update the plotter to show selection
        self.viewer.plotter.render()
        
        # Emit signal for bi-directional sync
        self.cameraSelectedInMVAT.emit(path)
        
    def _deselect_camera(self):
        """Deselect the current camera."""
        if self.selected_camera:
            # Get highlighted paths
            highlighted_paths = list(getattr(self.camera_grid, 'highlighted_paths', set()))
            
            # Update batched frustum colors (no selection, keep highlights)
            self.frustum_manager.update_camera_states(None, highlighted_paths)
            self.frustum_manager.mark_modified()
            
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
        # Update scale
        self.frustum_scale = self.scale_spinbox.value()
        
        # Update opacity
        self.thumbnail_opacity = self.opacity_slider.value() / 100.0
        
        # Update point size
        self.point_size = self.point_size_spinbox.value()
        
        # Update show flags
        self._show_wireframes_enabled = self.wireframe_checkbox.isChecked()
        self._show_thumbnails_enabled = self.thumbnail_checkbox.isChecked()
        self._show_point_cloud_enabled = self.point_cloud_checkbox.isChecked()
        self._show_rays_enabled = self.rays_checkbox.isChecked()
        
        # Clear existing
        self.cameras.clear()
        self.selected_camera = None
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
        else:
            super().keyPressEvent(event)