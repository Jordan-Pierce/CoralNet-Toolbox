"""
MultiView Annotation Tool (MVAT) Window

A 3D viewer for visualizing camera frustums and navigating MultiView imagery.
Uses PyVista for 3D rendering and integrates with the main application's RasterManager.
"""

import warnings
import numpy as np

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QToolBar, QAction, QLabel, QSlider, QCheckBox,
    QGroupBox, QMessageBox, QApplication, QFrame, QDoubleSpinBox,
    QPushButton, QSizePolicy, QSpacerItem, QSpinBox
)

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: PyVista or PyVistaQt not installed. MVAT will not be available.")

from coralnet_toolbox.MVAT.ui.QtMVATViewer import MVATViewer
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
    MultiView Annotation Tool Window.
    
    Provides a 3D visualization of camera frustums for MultiView imagery projects.
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
        
        # Actor lists for efficient updates
        self.wireframe_actors = []
        self.thumbnail_actors = []
        
        # Display status
        self.frustum_scale = 0.5
        self.show_wireframes = True
        self.show_thumbnails = True
        self.thumbnail_opacity = 0.8
        self.show_point_cloud = True
        self.point_size = 3
        
        # Setup UI
        self._setup_window()
        self._setup_menubar()
        self._setup_toolbar()
        self._setup_central_layout()
        
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
        self.wireframe_checkbox.setChecked(self.show_wireframes)
        self.wireframe_checkbox.toggled.connect(self._toggle_wireframes)  
        self.horizontal_layout.addWidget(self.wireframe_checkbox)
        
        self.thumbnail_checkbox = QCheckBox("Thumbnails")
        self.thumbnail_checkbox.setChecked(self.show_thumbnails)
        self.thumbnail_checkbox.toggled.connect(self._toggle_thumbnails)  
        self.horizontal_layout.addWidget(self.thumbnail_checkbox)
        
        self.point_cloud_checkbox = QCheckBox("Point cloud")
        self.point_cloud_checkbox.setChecked(self.show_point_cloud)
        self.point_cloud_checkbox.toggled.connect(self._toggle_point_cloud)  
        self.horizontal_layout.addWidget(self.point_cloud_checkbox)
        
        # Vertical Separator
        self.horizontal_layout.addWidget(self._create_v_line())
        
        # --- Widget: Selection Info & Button ---
        self.selection_label = QLabel("None selected")
        self.selection_label.setStyleSheet("color: #666;")
        self.horizontal_layout.addWidget(self.selection_label)
        
        self.goto_image_btn = QPushButton("Go to Image")
        self.goto_image_btn.setEnabled(False)
        self.goto_image_btn.setToolTip("Load selected camera in Main Window")
        self.goto_image_btn.clicked.connect(self._goto_selected_image)
        self.horizontal_layout.addWidget(self.goto_image_btn)
        
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
        self.viewer = MVATViewer(self)

        # Enable picking for camera selection using the viewer's plotter
        self.viewer.plotter.enable_point_picking(
            callback=self._on_pick,
            show_message=False,
            use_picker=True,
            pickable_window=True
        )

        # NEW: Wrap the viewer in a groupbox
        left_groupbox = QGroupBox("3D Viewer")
        left_layout = QVBoxLayout(left_groupbox)
        left_layout.addWidget(self.viewer)
        self.splitter.addWidget(left_groupbox)  # Add the groupbox instead of the viewer directly
        
        # --- Right Panel: Empty Container ---
        self.right_container = QGroupBox("Camera Grid")
        
        # Add a label just to denote it's the future container
        right_layout = QVBoxLayout(self.right_container)
        right_label = QLabel("Tools / Info Panel")
        right_label.setAlignment(Qt.AlignCenter)
        right_label.setEnabled(False) 
        right_layout.addStretch()
        right_layout.addWidget(right_label)
        right_layout.addStretch()
        
        self.splitter.addWidget(self.right_container)
        
        # Set splitter proportions (75% Viewer, 25% Right Panel)
        self.splitter.setSizes([900, 300])

    def _create_v_line(self):
        """Helper to create a vertical separator line."""
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        return line
        
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
        
        # Clear actor lists
        self.wireframe_actors.clear()
        self.thumbnail_actors.clear()
        
        # Close the viewer
        if self.viewer:
            self.viewer.close()
                
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
        
        # Render frustums
        self._render_frustums()
        
        # Fit view to show all cameras
        self._fit_to_view()
        
    def _render_frustums(self):
        """Render all camera frustums in the 3D scene."""
        if not self.viewer or not self.viewer.plotter:
            return
            
        # Clear existing actors
        self.viewer.plotter.clear()
        
        # Clear actor lists
        self.wireframe_actors.clear()
        self.thumbnail_actors.clear()
        
        # Re-add point cloud
        self.viewer.point_cloud_actor = None
        self.viewer.add_point_cloud()
        self.viewer.set_point_cloud_visible(self.show_point_cloud)
        
        # Add a reference grid
        self.viewer.plotter.add_axes()
        
        for path, camera in self.cameras.items():
            try:
                # Create wireframe actor
                if self.show_wireframes:
                    actor = camera.frustum.create_actor(self.viewer.plotter, scale=self.frustum_scale)
                    self.wireframe_actors.append(actor)
                    
                # Create thumbnail actor
                if self.show_thumbnails:
                    actor = camera.frustum.create_image_plane_actor(
                        self.viewer.plotter, 
                        scale=self.frustum_scale,
                        opacity=self.thumbnail_opacity
                    )
                    self.thumbnail_actors.append(actor)
                    
                # Re-apply selection if this camera is selected
                if camera == self.selected_camera:
                    camera.frustum.select()
            except Exception as e:
                print(f"Failed to render frustum for {path}: {e}")
                
        # Update the render
        self.viewer.plotter.update()
        
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
            
        self.show_wireframes = checked
        
        # Sync UI elements
        self.toggle_wireframes_action.setChecked(checked)
        self.wireframe_checkbox.blockSignals(True)
        self.wireframe_checkbox.setChecked(checked)
        self.wireframe_checkbox.blockSignals(False)
        
        # Update visibility of existing actors
        for actor in self.wireframe_actors:
            actor.SetVisibility(checked)
        
        # Update the render
        if self.viewer and self.viewer.plotter:
            self.viewer.plotter.update()
        
    def _toggle_thumbnails(self, checked=None):
        """Toggle thumbnail visibility."""
        if checked is None:
            checked = self.toggle_thumbnails_action.isChecked()
            
        self.show_thumbnails = checked
        
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
        self.show_point_cloud = checked
        self.viewer.set_point_cloud_visible(checked)
        
    def _on_scale_changed(self, value):
        """Handle frustum scale change."""
        self.frustum_scale = value
        
        # Update scale of existing actors
        for actor in self.wireframe_actors + self.thumbnail_actors:
            actor.SetScale(value)
        
        # Update the render
        if self.viewer and self.viewer.plotter:
            self.viewer.plotter.update()
        
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
        self.viewer.plotter.update()
        
    def _deselect_camera(self):
        """Deselect the current camera."""
        if self.selected_camera:
            self.selected_camera.frustum.deselect()
            self.selected_camera = None
            
        self.selection_label.setText("None selected")
        self.goto_image_btn.setEnabled(False)
        
        if self.viewer and self.viewer.plotter:
            self.viewer.plotter.update()
        
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
        self.show_wireframes = self.wireframe_checkbox.isChecked()
        self.show_thumbnails = self.thumbnail_checkbox.isChecked()
        self.show_point_cloud = self.point_cloud_checkbox.isChecked()
        
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