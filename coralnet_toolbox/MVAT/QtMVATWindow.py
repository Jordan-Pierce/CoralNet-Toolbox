"""
Multi-View Annotation Tool (MVAT) Window

A 3D viewer for visualizing camera frustums and navigating multi-view imagery.
Uses PyVista for 3D rendering and integrates with the main application's RasterManager.
"""

import warnings

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QToolBar, QAction, QStatusBar, QLabel, QSlider, QCheckBox,
    QGroupBox, QMessageBox, QApplication, QFrame, QDoubleSpinBox,
    QPushButton, QComboBox
)

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: PyVista or PyVistaQt not installed. MVAT will not be available.")

from coralnet_toolbox.MVAT.core.Camera import Camera
from coralnet_toolbox.MVAT.core.Frustum import Frustum

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MVATWindow(QMainWindow):
    """
    Multi-View Annotation Tool Window.
    
    Provides a 3D visualization of camera frustums for multi-view imagery projects.
    Allows users to navigate between views in 3D space and select cameras.
    """
    
    def __init__(self, main_window, parent=None):
        """
        Initialize the MVAT Window.
        
        Args:
            main_window: Reference to the main application window
            parent: Parent widget (default: None)
        """
        super().__init__(parent)
        
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista and PyVistaQt are required for MVAT. "
                              "Install with: pip install pyvista pyvistaqt")
        
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.raster_manager = main_window.image_window.raster_manager
        
        # Camera management
        self.cameras = {}  # image_path -> Camera object
        self.selected_camera = None
        
        # Display settings
        self.frustum_scale = 0.5
        self.show_wireframes = True
        self.show_thumbnails = True
        self.thumbnail_opacity = 0.8
        
        # PyVista plotter reference
        self.plotter = None
        
        # Setup UI
        self._setup_window()
        self._setup_menubar()
        self._setup_toolbar()
        self._setup_central_widget()
        self._setup_statusbar()
        
        # Flag to track initialization
        self._initialized = False
        
    def _setup_window(self):
        """Configure the main window properties."""
        self.setWindowTitle("Multi-View Annotation Tool (MVAT)")
        mvat_icon_path = get_icon("camera.png")
        if mvat_icon_path:
            self.setWindowIcon(QIcon(mvat_icon_path))
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
        
        self.view_menu.addSeparator()
        
        # Fit to View
        self.fit_view_action = QAction("Fit All", self)
        self.fit_view_action.setShortcut("F")
        self.fit_view_action.triggered.connect(self._fit_to_view)
        self.view_menu.addAction(self.fit_view_action)
        
    def _setup_toolbar(self):
        """Create the toolbar."""
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setMovable(False)
        self.addToolBar(Qt.LeftToolBarArea, self.toolbar)
        
        # Reset View button
        reset_action = QAction("Reset", self)
        reset_action.setToolTip("Reset the 3D view (R)")
        reset_action.triggered.connect(self._reset_camera_view)
        self.toolbar.addAction(reset_action)
        
        # Fit View button
        fit_action = QAction("Fit", self)
        fit_action.setToolTip("Fit all cameras in view (F)")
        fit_action.triggered.connect(self._fit_to_view)
        self.toolbar.addAction(fit_action)
        
        self.toolbar.addSeparator()
        
        # Deselect button
        deselect_action = QAction("Deselect", self)
        deselect_action.setToolTip("Deselect current camera (Escape)")
        deselect_action.triggered.connect(self._deselect_camera)
        self.toolbar.addAction(deselect_action)
        
    def _setup_central_widget(self):
        """Create the central widget with splitter layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create splitter
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)
        
        # Left panel: 3D Viewport
        self.viewport_widget = QWidget()
        viewport_layout = QVBoxLayout(self.viewport_widget)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create PyVista QtInteractor
        self.plotter = QtInteractor(self.viewport_widget)
        self.plotter.set_background('white')
        self.plotter.enable_trackball_style()
        
        # Enable picking for camera selection
        self.plotter.enable_point_picking(
            callback=self._on_pick,
            show_message=False,
            use_picker=True,
            pickable_window=True
        )
        
        viewport_layout.addWidget(self.plotter.interactor)
        
        # Right panel: Control Panel
        self.control_panel = self._create_control_panel()
        
        # Add panels to splitter
        self.splitter.addWidget(self.viewport_widget)
        self.splitter.addWidget(self.control_panel)
        
        # Set splitter sizes (80% viewport, 20% controls)
        self.splitter.setSizes([800, 200])
        
    def _create_control_panel(self):
        """Create the right-side control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # ===== Display Settings Group =====
        display_group = QGroupBox("Display Settings")
        display_layout = QVBoxLayout(display_group)
        
        # Frustum Scale
        scale_layout = QHBoxLayout()
        scale_label = QLabel("Frustum Scale:")
        self.scale_spinbox = QDoubleSpinBox()
        self.scale_spinbox.setRange(0.01, 10.0)
        self.scale_spinbox.setSingleStep(0.1)
        self.scale_spinbox.setValue(self.frustum_scale)
        self.scale_spinbox.valueChanged.connect(self._on_scale_changed)
        scale_layout.addWidget(scale_label)
        scale_layout.addWidget(self.scale_spinbox)
        display_layout.addLayout(scale_layout)
        
        # Wireframe checkbox
        self.wireframe_checkbox = QCheckBox("Show Wireframes")
        self.wireframe_checkbox.setChecked(self.show_wireframes)
        self.wireframe_checkbox.toggled.connect(self._toggle_wireframes)
        display_layout.addWidget(self.wireframe_checkbox)
        
        # Thumbnail checkbox
        self.thumbnail_checkbox = QCheckBox("Show Thumbnails")
        self.thumbnail_checkbox.setChecked(self.show_thumbnails)
        self.thumbnail_checkbox.toggled.connect(self._toggle_thumbnails)
        display_layout.addWidget(self.thumbnail_checkbox)
        
        # Thumbnail Opacity
        opacity_layout = QHBoxLayout()
        opacity_label = QLabel("Thumbnail Opacity:")
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(self.thumbnail_opacity * 100))
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(opacity_label)
        opacity_layout.addWidget(self.opacity_slider)
        display_layout.addLayout(opacity_layout)
        
        layout.addWidget(display_group)
        
        # ===== Selection Info Group =====
        selection_group = QGroupBox("Selected Camera")
        selection_layout = QVBoxLayout(selection_group)
        
        self.selection_label = QLabel("No camera selected")
        self.selection_label.setWordWrap(True)
        selection_layout.addWidget(self.selection_label)
        
        # Go to image button
        self.goto_image_btn = QPushButton("Go to Image")
        self.goto_image_btn.setEnabled(False)
        self.goto_image_btn.clicked.connect(self._goto_selected_image)
        selection_layout.addWidget(self.goto_image_btn)
        
        layout.addWidget(selection_group)
        
        # ===== Statistics Group =====
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("Cameras: 0")
        stats_layout.addWidget(self.stats_label)
        
        layout.addWidget(stats_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        # ===== Refresh Button =====
        self.refresh_btn = QPushButton("Refresh Scene")
        self.refresh_btn.clicked.connect(self._refresh_scene)
        layout.addWidget(self.refresh_btn)
        
        return panel
        
    def _setup_statusbar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
    def showEvent(self, event):
        """Handle show event - load cameras when window is shown."""
        super().showEvent(event)
        
        if not self._initialized:
            # Use QTimer to load cameras after the window is fully shown
            QTimer.singleShot(100, self._load_cameras)
            self._initialized = True
            
    def closeEvent(self, event):
        """Handle close event - cleanup resources."""
        # Clear camera references
        self.cameras.clear()
        self.selected_camera = None
        
        # Close the plotter
        if self.plotter:
            try:
                self.plotter.close()
            except Exception:
                pass
                
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
        self.status_bar.showMessage(f"Loaded {valid_count} cameras")
        
        # Render frustums
        self._render_frustums()
        
        # Fit view to show all cameras
        self._fit_to_view()
        
    def _render_frustums(self):
        """Render all camera frustums in the 3D scene."""
        if not self.plotter:
            return
            
        # Clear existing actors
        self.plotter.clear()
        
        # Add a reference grid
        self.plotter.add_axes()
        
        for path, camera in self.cameras.items():
            try:
                # Create wireframe actor
                if self.show_wireframes:
                    camera.frustum.create_actor(self.plotter, scale=self.frustum_scale)
                    
                # Create thumbnail actor
                if self.show_thumbnails:
                    camera.frustum.create_image_plane_actor(
                        self.plotter, 
                        scale=self.frustum_scale,
                        opacity=self.thumbnail_opacity
                    )
            except Exception as e:
                print(f"Failed to render frustum for {path}: {e}")
                
        # Update the render
        self.plotter.update()
        
    def _reset_camera_view(self):
        """Reset the 3D camera to default view."""
        if self.plotter:
            self.plotter.reset_camera()
            self.plotter.view_isometric()
            
    def _fit_to_view(self):
        """Fit all objects in the view."""
        if self.plotter:
            self.plotter.reset_camera()
            
    def _toggle_wireframes(self, checked=None):
        """Toggle wireframe visibility."""
        if checked is None:
            checked = self.toggle_wireframes_action.isChecked()
            
        self.show_wireframes = checked
        self.toggle_wireframes_action.setChecked(checked)
        self.wireframe_checkbox.setChecked(checked)
        
        # Re-render the scene
        self._render_frustums()
        
    def _toggle_thumbnails(self, checked=None):
        """Toggle thumbnail visibility."""
        if checked is None:
            checked = self.toggle_thumbnails_action.isChecked()
            
        self.show_thumbnails = checked
        self.toggle_thumbnails_action.setChecked(checked)
        self.thumbnail_checkbox.setChecked(checked)
        
        # Re-render the scene
        self._render_frustums()
        
    def _on_scale_changed(self, value):
        """Handle frustum scale change."""
        self.frustum_scale = value
        
        # Invalidate geometry caches
        for camera in self.cameras.values():
            camera.frustum._frustum_mesh = None
            camera.frustum._image_plane_mesh = None
            
        # Re-render
        self._render_frustums()
        
    def _on_opacity_changed(self, value):
        """Handle thumbnail opacity change."""
        self.thumbnail_opacity = value / 100.0
        
        # Re-render to apply new opacity
        self._render_frustums()
        
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
                
        # Select if within reasonable distance
        if closest_camera and min_dist < self.frustum_scale * 2:
            self._select_camera(closest_path, closest_camera)
        else:
            self._deselect_camera()
            
    def _select_camera(self, path, camera):
        """Select a camera and update UI."""
        # Deselect previous
        if self.selected_camera:
            self.selected_camera.frustum.deselect()
            
        # Select new
        self.selected_camera = camera
        camera.frustum.select()
        
        # Update UI
        self.selection_label.setText(f"Selected: {camera.label}")
        self.goto_image_btn.setEnabled(True)
        
        # Update the plotter to show selection
        self.plotter.update()
        
        self.status_bar.showMessage(f"Selected: {camera.label}")
        
    def _deselect_camera(self):
        """Deselect the current camera."""
        if self.selected_camera:
            self.selected_camera.frustum.deselect()
            self.selected_camera = None
            
        self.selection_label.setText("No camera selected")
        self.goto_image_btn.setEnabled(False)
        
        if self.plotter:
            self.plotter.update()
            
        self.status_bar.showMessage("Ready")
        
    def _goto_selected_image(self):
        """Navigate to the selected camera's image in the main window."""
        if not self.selected_camera:
            return
            
        path = self.selected_camera.image_path
        
        # Use ImageWindow to select the image
        try:
            self.image_window.load_image_by_path(path)
            self.status_bar.showMessage(f"Navigated to: {self.selected_camera.label}")
        except Exception as e:
            QMessageBox.warning(self, "Navigation Error", f"Could not navigate to image: {e}")
            
    def _refresh_scene(self):
        """Refresh the entire scene by reloading cameras."""
        # Clear existing
        self.cameras.clear()
        self.selected_camera = None
        self._initialized = False
        
        # Reload
        self._load_cameras()
        
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


# Import numpy here to avoid issues if pyvista is not available
import numpy as np
