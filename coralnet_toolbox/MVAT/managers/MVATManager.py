"""
MultiView Annotation Tool (MVAT) Manager

The central controller for the MVAT workspace.
Handles the business logic, data synchronization, and signal routing between 
the MainWindow, RasterManager, MVATViewer (3D), and CameraGrid (2D).
"""

import time
import numpy as np

from PyQt5.QtCore import QObject, QTimer, pyqtSignal, Qt, QThread
from PyQt5.QtWidgets import QApplication, QMessageBox
from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.MVAT.core.Camera import Camera
from coralnet_toolbox.MVAT.core.Ray import CameraRay
from coralnet_toolbox.MVAT.managers.SelectionManager import SelectionManager
from coralnet_toolbox.MVAT.managers.VisibilityManager import VisibilityManager
from coralnet_toolbox.MVAT.managers.VisibilityWorker import VisibilityWorker
from coralnet_toolbox.MVAT.managers.CacheManager import CacheManager
from coralnet_toolbox.MVAT.core.constants import (
    MARKER_COLOR_SELECTED,
    MARKER_COLOR_INVALID,
    RAY_COLOR_SELECTED,
    RAY_COLOR_HIGHLIGHTED,
    RAY_COLOR_INVALID,
    MOUSE_THROTTLE_MS,
)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MousePositionBridge(QObject):
    """
    Bridges mouse position events from the AnnotationWindow to the MVAT
    controller.

    Responsibilities:
    - Throttle rapid mouse-move events to avoid UI/compute thrashing.
    - Build a 3D ray from a selected camera and a 2D image pixel (uses depth
        from a z-channel when available, otherwise falls back to a scene
        median/default depth).
    - Create corresponding rays from highlighted cameras to the same world
        point and choose colors based on depth accuracy.
    - Project the selected ray into other camera image spaces and forward
        marker/visibility updates to the CameraGrid for UI presentation.

    This class mirrors the behavior previously implemented on the window
    layer but is now manager-owned so it can operate without direct UI
    responsibilities.
    """
    def __init__(self, manager: 'MVATManager'):
        super().__init__()
        self.manager = manager
        self.enabled = True
        self._last_update_time = 0
        self._pending_position = None

        self._throttle_timer = QTimer()
        self._throttle_timer.setSingleShot(True)
        self._throttle_timer.timeout.connect(self._process_pending_position)
        
    def on_mouse_moved(self, x: int, y: int):
        if not self.enabled:
            return
        self._pending_position = (x, y)
        if not self._throttle_timer.isActive():
            self._throttle_timer.start(MOUSE_THROTTLE_MS)
            
    def _process_pending_position(self):
        if self._pending_position is None:
            return
            
        x, y = self._pending_position
        self._pending_position = None
        
        camera = self.manager.selected_camera
        if camera is None or not (0 <= x < camera.width and 0 <= y < camera.height):
            self.clear_all_markers()
            self.manager.viewer.clear_ray()
            return
            
        raster = camera._raster
        depth = None
        if raster.z_channel is not None and raster.z_data_type == 'depth':
            depth = raster.get_z_value(x, y)
        
        if depth is None or depth <= 0 or np.isnan(depth):
            default_depth = self.manager.viewer.get_scene_median_depth(camera.position)
        else:
            default_depth = 10.0
        
        ray = CameraRay.from_pixel_and_camera(
            pixel_xy=(x, y),
            camera=camera,
            depth=depth,
            default_depth=default_depth
        )
        
        highlighted_cameras = self.manager.highlighted_cameras
        rays_with_colors = [(ray, RAY_COLOR_SELECTED if ray.has_accurate_depth else RAY_COLOR_INVALID)]
        visibility_status = {}
        accuracies = {camera.image_path: ray.has_accurate_depth}
        
        for target_cam in highlighted_cameras:
            if target_cam.image_path == camera.image_path:
                continue
            
            is_occluded = target_cam.is_point_occluded_depth_based(ray.terminal_point, depth_threshold=0.15)
            visibility_status[target_cam.image_path] = is_occluded
            
            target_ray = CameraRay.from_world_point_and_camera(
                world_point=ray.terminal_point,
                camera=target_cam
            )
            ray_color = RAY_COLOR_HIGHLIGHTED if target_ray.has_accurate_depth else RAY_COLOR_INVALID
            rays_with_colors.append((target_ray, ray_color))
            accuracies[target_cam.image_path] = target_ray.has_accurate_depth
        
        self.manager.viewer.show_rays(rays_with_colors)
        projections = ray.project_to_cameras(self.manager.cameras)
        
        # Update markers on CameraGrid
        highlighted_paths = {cam.image_path for cam in highlighted_cameras}
        selected_path = camera.image_path
        try:
            self.manager.camera_grid.update_markers(projections, 
                                                    accuracies, 
                                                    highlighted_paths, 
                                                    visibility_status,
                                                    selected_path)
        except Exception:
            self.clear_all_markers()
                
    def clear_all_markers(self):
        try:
            self.manager.camera_grid.clear_all_markers()
        except Exception:
            pass
            
    def cleanup(self):
        self._throttle_timer.stop()
        self.clear_all_markers()


class MVATManager(QObject):
    """
    Core Controller for the MVAT Workspace.
    """
    cameraSelectedInMVAT = pyqtSignal(str)
    
    def __init__(self, main_window, viewer, grid):
        super().__init__()
        
        self.main_window = main_window
        self.raster_manager = main_window.image_window.raster_manager
        self.annotation_window = main_window.annotation_window
        self.image_window = main_window.image_window
        
        self.viewer = viewer
        self.camera_grid = grid
        
        # State
        self.cameras = {}
        self.selected_camera = None
        self.highlighted_cameras = []
        self.hovered_camera = None
        self.current_focal_point = None
        
        # Data Settings
        self.compute_depth_maps_enabled = True
        # New toggle: whether to compute index maps in background
        self.compute_index_maps_enabled = True
        # Safety flag to prevent concurrent visibility computations
        self._is_computing_visibility = False
        # Track active worker threads to prevent GC
        self._active_workers = []
        
        # Internal Managers
        self.selection_model = SelectionManager(self)
        self.cache_manager = CacheManager("")
        self.mouse_bridge = MousePositionBridge(self)
        
        self._setup_connections()

    def _setup_connections(self):
        """
        Bind all signals between UI views and this controller.

        Connections established:
        1. SelectionModel signals -> manager handlers (active/selection changed)
        2. CameraGrid intent signals -> SelectionModel methods (selection, toggle,
            active, clear)
        3. CameraGrid hover events -> manager hover handlers
        4. Viewer notifications (focal point, full-cloud toggle, compute-depths)
        5. Main window sync: wire the annotation window's mouseMoved and the
            image window's imageLoaded signals to manager handlers when present.
        """
        # 1. Selection Model (The Source of Truth)
        self.selection_model.active_changed.connect(self._on_active_camera_changed)
        self.selection_model.selection_changed.connect(self._on_selections_changed)
        
        # 2. CameraGrid Intents -> Selection Model
        self.camera_grid.selection_requested.connect(self.selection_model.set_selections)
        self.camera_grid.toggle_requested.connect(self.selection_model.toggle)
        self.camera_grid.active_requested.connect(self.selection_model.set_active)
        self.camera_grid.clear_requested.connect(self.selection_model.clear_selections)
        
        # 3. CameraGrid Hover States
        self.camera_grid.camera_hovered.connect(self._on_camera_hovered)
        self.camera_grid.camera_unhovered.connect(self._on_camera_unhovered)
        
        # 4. Viewer Signals
        self.viewer.focalPointChanged.connect(self._on_focal_point_changed)
        self.viewer.computeIndexMapsToggled.connect(self._on_compute_index_maps_toggled)
        self.viewer.computeDepthMapsToggled.connect(self._on_compute_depth_maps_toggled)
        
        # 5. Main Window Sync
        if hasattr(self.annotation_window, 'mouseMoved'):
            self.annotation_window.mouseMoved.connect(self.mouse_bridge.on_mouse_moved)
        if hasattr(self.image_window, 'imageLoaded'):
            self.image_window.imageLoaded.connect(self._on_main_image_loaded)
        # CameraGrid image selection -> load image in AnnotationWindow
        try:
            self.camera_grid.camera_selected.connect(self._on_camera_selected)
        except Exception:
            pass
        # Single highlight intent -> update viewer perspective
        try:
            self.camera_grid.camera_highlighted_single.connect(self._on_camera_highlighted_single)
        except Exception:
            pass

    def load_cameras(self):
        """
        Extract camera parameters from the RasterManager, construct Camera
        objects, and push them into the Grid and Viewer.

        Shows a progress dialog during load and reports if no valid camera
        parameters are found. After loading it synchronizes the active camera
        with the AnnotationWindow's currently displayed image (when possible)
        and triggers an initial frustum render and viewer fit.
        """
        all_paths = self.raster_manager.image_paths
        if not all_paths:
            return
            
        # Indicate busy state via cursor and status bar (no modal progress)
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        except Exception:
            pass
        try:
            try:
                self.main_window.status_bar.showMessage("Loading cameras...", 0)
            except Exception:
                pass

            valid_count = 0
            ortho_count = 0
            
            for path in all_paths:
                raster = self.raster_manager.get_raster(path)
                if not raster:
                    continue
                
                # CREATE ORTHOGRAPHIC CAMERA
                if raster.is_orthomosaic:
                    try:
                        from coralnet_toolbox.MVAT.core.Camera import OrthographicCamera
                        self.cameras[path] = OrthographicCamera(raster)
                        ortho_count += 1
                        valid_count += 1
                    except Exception as e:
                        print(f"❌ Failed to load orthomosaic {raster.basename}: {e}")
                        continue
                
                # CREATE PERSPECTIVE CAMERA
                elif raster.intrinsics is not None and raster.extrinsics is not None:
                    try:
                        self.cameras[path] = Camera(raster)
                        valid_count += 1
                    except Exception:
                        print(f"❌ Failed to load perspective camera {raster.basename}")
                        pass
        finally:
            try:
                QApplication.restoreOverrideCursor()
            except Exception:
                pass
            try:
                self.main_window.status_bar.showMessage(
                    f"Loaded cameras: {valid_count} total ({ortho_count} orthomosaics, "
                    f"{valid_count - ortho_count} perspective)",
                    3000
                )
            except Exception:
                pass
            
        if valid_count == 0:
            QMessageBox.information(self.main_window, "No Camera Data", "No valid camera parameters found.")
            return
        
        # FILTER: Only pass perspective cameras to grid UI
        perspective_cameras = {p: c for p, c in self.cameras.items() if not c.is_orthographic}
        
        try:
            self.camera_grid.stats_label.setText(
                f"Cameras: {len(perspective_cameras)} perspective" + 
                (f", {ortho_count} ortho" if ortho_count > 0 else "")
            )
        except Exception:
            pass
        
        self.camera_grid.set_cameras(perspective_cameras)
        self._render_frustums()
        self.viewer.fit_to_view()
        
        # Initial Synchronization
        current_image_path = getattr(self.annotation_window, 'current_image_path', None)
        if current_image_path and current_image_path in self.cameras:
            self.selection_model.set_active(current_image_path)
        elif perspective_cameras:
            self.selection_model.set_active(next(iter(perspective_cameras)))
        elif self.cameras:
            # Only orthomosaics loaded - activate the first one
            self.selection_model.set_active(next(iter(self.cameras)))

    def _render_frustums(self):
        """
        Update the 3D scene to render frustums, point cloud and axes.

        This prepares the viewer by ensuring the point cloud and axes are
        present, then asks the viewer to draw all camera frustums using the
        current selection/highlight/hover state.
        """
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.viewer.add_point_cloud()
        finally:
            QApplication.restoreOverrideCursor()
            
        self.viewer.add_axes()
        highlighted = self.selection_model.get_selected_list()
        
        self.viewer.add_frustums(
            self.cameras,
            selected_camera=self.selected_camera,
            highlighted_paths=highlighted,
            hovered_camera=self.hovered_camera
        )

    # --- Signal Handlers ---

    def _on_main_image_loaded(self, path: str):
        """
        Handler for when the main image window loads a new image.

        If the image corresponds to a camera known to the MVAT manager, set
        that camera as the active camera in the selection model so the
        manager and views synchronize.
        """
        if path in self.cameras:
            self.selection_model.set_active(path)

    def _on_focal_point_changed(self, point_3d):
        """
        Respond to viewer focal-point changes.

        When the 3D viewer reports a new focal point, attempt to project it
        into the active camera's image. If projection succeeds and a depth
        value exists, show an incoming marker in the AnnotationWindow using
        a color that indicates whether the depth is valid. If projection
        fails, hide the marker.
        """
        self.current_focal_point = point_3d
        if self.selected_camera and self.selected_camera.image_path in self.cameras:
            pixel = self.selected_camera.project(point_3d)
            if not np.isnan(pixel).any():
                u, v = pixel[0], pixel[1]
                depth = self.selected_camera._raster.get_z_value(int(u), int(v))
                color = MARKER_COLOR_SELECTED if depth is not None and depth > 0 else MARKER_COLOR_INVALID
                self.annotation_window.set_incoming_marker(u, v, color)
            else:
                self.annotation_window.marker.hide()

    def _on_camera_hovered(self, path):
        """
        Called when a camera is hovered in the CameraGrid.

        Stores the hovered camera and refreshes frustum appearances so the
        viewer can highlight the hovered frustum appropriately.
        """
        self.hovered_camera = path
        self._update_frustum_states()

    def _on_camera_unhovered(self, path):
        """
        Called when the hover state is removed from a camera in the grid.

        Clears the hovered state if it matches and refreshes frustum visuals.
        """
        if self.hovered_camera == path:
            self.hovered_camera = None
        self._update_frustum_states()

    # Note: full-cloud toggling and GPU-based subsetting have been removed.
    # The viewer now renders the full point cloud; background index-map
    # computation is controlled by the computeIndexMaps toggle.

    def _on_compute_depth_maps_toggled(self, state: bool):
        """
        Enable or disable computing and storing depth maps during visibility
        computation. Depth maps can improve occlusion checks but are more
        expensive to compute and merge into rasters.
        """
        self.compute_depth_maps_enabled = state

    def _on_compute_index_maps_toggled(self, state: bool):
        """Enable/disable background computation of index maps."""
        self.compute_index_maps_enabled = state
        try:
            msg = "Compute Index Maps: ON" if state else "Compute Index Maps: OFF"
            self.main_window.status_bar.showMessage(msg, 2000)
        except Exception:
            pass

    def _on_visibility_computed(self, results: dict):
        """Handle results emitted from VisibilityWorker (runs on main thread)."""
        try:
            # Get primary target file path for cache key
            primary_target = self.viewer.scene_context.get_primary_target()
            target_file_path = primary_target.file_path if primary_target else ""
            
            self._process_visibility_results(results, target_file_path)

        finally:
            self._is_computing_visibility = False
            try:
                self.main_window.status_bar.showMessage("Visibility maps updated.", 3000)
            except Exception:
                pass
            # Restore cursor when done
            try:
                QApplication.restoreOverrideCursor()
            except Exception:
                pass

    def _process_visibility_results(self, results: dict, target_file_path: str):
        """
        Process visibility computation results and store in cameras.
        
        Shared by both sync (VTK mesh) and async (worker) code paths.
        If the async worker already saved the cache to disk, it skips IO on the main thread.
        
        Args:
            results: Dict mapping image_path -> visibility result dict
            target_file_path: Path to primary target for cache key
        """
        for path, result in results.items():
            camera = self.cameras.get(path)
            if not camera:
                continue

            # Store index map with element type metadata
            element_type = result.get('element_type', 'point')
            
            # 1. Check if the background worker already handled the disk I/O
            cache_path = result.get('cache_path')
            
            # 2. Fallback for sync paths (like VTK) that run on the main thread
            if cache_path is None and self.cache_manager is not None and target_file_path:
                try:
                    # Use transform_matrix for orthomosaic, extrinsics for perspective
                    cache_key = camera.transform_matrix if camera.is_orthographic else camera._raster.extrinsics
                    
                    cache_path = self.cache_manager.save_visibility(
                        cache_key,
                        target_file_path,
                        result.get('index_map'),
                        result.get('visible_indices'),
                        result.get('depth_map') if self.compute_depth_maps_enabled else None,
                        element_type=element_type
                    )
                except Exception:
                    cache_path = None
                    
            # 3. Apply the results to the camera
            try:
                camera._raster.add_index_map(
                    result.get('index_map'), 
                    cache_path, 
                    result.get('visible_indices'),
                    element_type=element_type
                )
            except Exception:
                pass

            if self.compute_depth_maps_enabled and result.get('depth_map') is not None:
                try:
                    camera._raster.merge_or_set_depth_map(result.get('depth_map'))
                except Exception:
                    pass
            
    def _on_visibility_error(self, error_str: str):
        print(f"Visibility worker error:\n{error_str}")
        self._is_computing_visibility = False
        try:
            self.main_window.status_bar.showMessage("Visibility computation failed. See console for details.", 5000)
        except Exception:
            pass
        # Restore cursor on error
        try:
            QApplication.restoreOverrideCursor()
        except Exception:
            pass

    def _on_active_camera_changed(self, path):
        """
        Handler for when the selection model reports a new active camera.

        Updates internal selection state, clears any active ray visualization,
        instructs the viewer to match the selected camera perspective (when
        supported), reorders the grid to prioritize nearby cameras, and asks
        the image window to load the selected image.
        
        ENFORCES: Clean map view for orthomosaics by clearing all highlights.
        """
        camera = self.cameras.get(path)
        if camera:
            # ENFORCE: Clear all highlights when entering orthomosaic view
            if camera.is_orthographic:
                print(f"📍 Entering orthomosaic view: {camera.label}")
                self.selection_model.clear_selections(keep_active=True, emit=True)
            
            self.viewer.clear_ray()
            self._select_camera(path, camera)
            if hasattr(self.viewer, 'match_camera_perspective'):
                self.viewer.match_camera_perspective(camera)
            self._reorder_cameras(path)
            
            try:
                self.image_window.load_image_by_path(path)
            except Exception: 
                pass

    def _on_camera_selected(self, path: str):
        """Handle camera_selected from the grid (context menu 'Select Image').

        Preferred entry point to change the displayed image in the annotation
        window is `annotation_window.set_image(path)` per project convention.
        Fall back to older image_window loader if the method isn't present.
        """
        try:
            # Make this the sole selection: set active and clear other highlights
            try:
                self.selection_model.set_active(path)
                # Ensure only the active camera remains selected/highlighted
                self.selection_model.set_selections([path])
            except Exception:
                pass

            if hasattr(self.annotation_window, 'set_image'):
                self.annotation_window.set_image(path)
            else:
                # Legacy fallback
                self.image_window.load_image_by_path(path)
            # Note: status message moved to AnnotationWindow.set_image
        except Exception as e:
            print(f"Failed to load selected image '{path}': {e}")

    def _on_camera_highlighted_single(self, path: str):
        """Handle single-camera highlight intent (e.g., plain click).

        This updates the viewer perspective to match the clicked camera when
        supported.
        """
        try:
            cam = self.cameras.get(path)
            if cam and hasattr(self.viewer, 'match_camera_perspective'):
                self.viewer.match_camera_perspective(cam)
        except Exception:
            pass

    def _on_selections_changed(self, selected_paths):
        """
        Respond to selection model changes (highlight toggles).

        Applies highlight state to Camera objects, updates viewer frustums,
        synchronizes the CameraGrid UI to the model, clears any active ray
        visualization, and triggers visibility recomputation for the new set
        of highlighted cameras.
        """
        for path, camera in self.cameras.items():
            if path in selected_paths:
                camera.highlight()
            else:
                camera.unhighlight()
            
            try:
                if hasattr(self.viewer, 'update_camera_appearance'):
                    self.viewer.update_camera_appearance(camera)
            except Exception: 
                pass

        self.highlighted_cameras = [self.cameras.get(path) for path in selected_paths if path in self.cameras]
        self._update_frustum_states()
        
        try:
            self.camera_grid._sync_ui_to_model()
        except Exception: 
            pass

        self.viewer.clear_ray()
        try:
            print(f"MVATManager: selections changed -> {selected_paths}")
        except Exception:
            pass
        self._update_visibility_filter(list(selected_paths))

    # --- Core Logic Methods ---

    def _select_camera(self, path, camera):
        """
        Make the provided camera the currently selected camera.

        Handles deselection of the previously selected camera, updates the
        viewer appearance, optionally shows a thumbnail for the selected
        camera, refreshes frustum states, and emits the
        `cameraSelectedInMVAT` signal with the camera path.
        """
        if self.selected_camera:
            self.selected_camera.deselect()
            if hasattr(self.viewer, 'update_camera_appearance'):
                self.viewer.update_camera_appearance(self.selected_camera)
        
        self.selected_camera = camera
        camera.select()
        
        if hasattr(self.viewer, 'update_camera_appearance'):
            self.viewer.update_camera_appearance(camera)
        
        self._update_frustum_states()
        
        if hasattr(self.viewer, '_add_thumbnail_for_camera') and self.viewer._show_thumbnails_enabled:
            self.viewer.remove_thumbnails()
            self.viewer._add_thumbnail_for_camera(camera)
            
        self.viewer.update()
        self.cameraSelectedInMVAT.emit(path)

    def _update_frustum_states(self):
        """
        Refresh viewer frustum appearances based on selected, highlighted,
        and hovered camera state.

        Delegates to the viewer's `update_frustum_states` if available; errors
        are caught and ignored to avoid destabilizing the UI for non-critical
        failures.
        """
        selected_path = self.selected_camera.image_path if self.selected_camera else None
        highlighted_paths = [cam.image_path for cam in self.highlighted_cameras]
        try:
            if hasattr(self.viewer, 'update_frustum_states'):
                self.viewer.update_frustum_states(selected_path, highlighted_paths, self.hovered_camera)
        except Exception: 
            pass

    def _update_visibility_filter(self, highlighted_paths):
        """
        Compute visibility index maps for the supplied highlighted cameras.
        Intercepts and loads from disk cache if available before using the worker.
        """
        if not self.viewer.scene_context.has_any_product():
            return
        if not self.compute_index_maps_enabled:
            return
        if not highlighted_paths:
            return
        
        primary_target = self.viewer.scene_context.get_primary_target()
        if primary_target is None:
            return
            
        target_file_path = primary_target.file_path
        element_type = primary_target.get_element_type()

        cameras_needing_visibility = []
        for path in highlighted_paths:
            camera = self.cameras.get(path)
            if not camera:
                continue
                
            # 1. Check if already in active memory (RAM)
            if camera.visible_indices is not None:
                continue

            # 2. Check Disk Cache [New Logic]
            loaded_from_cache = False
            if self.cache_manager is not None and target_file_path:
                self.main_window.status_bar.showMessage(f"Checking cache for {camera.label}...", 1000)
                # Use transform_matrix for orthomosaics, extrinsics for perspective
                cache_key = camera.transform_matrix if camera.is_orthographic else camera._raster.extrinsics
                cached_data = self.cache_manager.load_visibility(cache_key, target_file_path, element_type)
                
                if cached_data is not None:
                    self.main_window.status_bar.showMessage(f"Loaded visibility from cache for {camera.label}", 2000)
                    cache_path = self.cache_manager.get_cache_path(cache_key, target_file_path, element_type)
                    
                    # Store results in the camera's raster object
                    camera._raster.add_index_map(
                        cached_data.get('index_map'), 
                        cache_path, 
                        cached_data.get('visible_indices'),
                        element_type=element_type
                    )
                    
                    # Also restore the depth map if it exists and is enabled
                    if self.compute_depth_maps_enabled and cached_data.get('depth_map') is not None:
                        self.main_window.status_bar.showMessage(
                            f"Restoring depth map from cache for {camera.label}", 2000
                        )
                        camera._raster.merge_or_set_depth_map(cached_data['depth_map'])
                            
                    loaded_from_cache = True
                    print(f"💽 Loaded visibility from disk cache: {camera.label}")

            # 3. Only if missing from both RAM and Disk, queue for computation
            if not loaded_from_cache:
                cameras_needing_visibility.append(camera)

        if not cameras_needing_visibility:
            return

        if self._is_computing_visibility:
            return

        # Proceed to async computation for only the remaining cameras
        self._compute_visibility_async(primary_target, cameras_needing_visibility)

    def _compute_mesh_visibility_sync(self, mesh_product, cameras):
        """
        Compute mesh visibility using VTK rasterization (main thread).
        
        VTK/PyVista rendering requires the GUI thread's OpenGL context.
        This method performs synchronous rasterization for accurate mesh depth maps.
        """
        from coralnet_toolbox.MVAT.managers.VisibilityManager import VisibilityManager
        
        self._is_computing_visibility = True
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        except Exception:
            pass
        
        try:
            target_file_path = mesh_product.file_path
            n_cameras = len(cameras)
            
            try:
                self.main_window.status_bar.showMessage(
                    f"Rasterizing mesh for {n_cameras} camera(s)..."
                )
            except Exception:
                pass
            
            print(f"MVATManager: VTK mesh rasterization for {n_cameras} cameras")
            
            results = {}
            for i, camera in enumerate(cameras):
                if camera.is_orthographic:
                    # Skip orthographic cameras for now (use fallback)
                    continue
                
                try:
                    result = VisibilityManager._compute_mesh_visibility(
                        mesh_product,
                        camera.K, camera.R, camera.t,
                        camera.width, camera.height,
                        compute_depth_map=self.compute_depth_maps_enabled
                    )
                    result['element_type'] = 'face'
                    results[camera.image_path] = result
                except Exception as e:
                    print(f"⚠️ Failed to compute mesh visibility for {camera.label}: {e}")
            
            # Process results (same logic as _on_visibility_computed)
            self._process_visibility_results(results, target_file_path)
            
        except Exception as e:
            print(f"⚠️ Mesh visibility computation failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._is_computing_visibility = False
            try:
                QApplication.restoreOverrideCursor()
            except Exception:
                pass
            try:
                self.main_window.status_bar.showMessage("Visibility maps updated.", 3000)
            except Exception:
                pass

    def _compute_visibility_async(self, primary_target, cameras):
        """
        Asynchronously compute visibility for a set of cameras using a worker thread.
        Supports both orthographic and perspective cameras, and leverages caching
        to avoid redundant computations.
        """
        # Prepare camera parameters and cache keys for asynchronous visibility computation.
        camera_params_dict = {}
        cache_keys_dict = {} 
        
        for cam in cameras:
            if cam.is_orthographic:
                camera_params_dict[cam.image_path] = ('ortho', cam.transform_matrix_inv, cam.width, cam.height)
                cache_keys_dict[cam.image_path] = cam.transform_matrix
            else:
                camera_params_dict[cam.image_path] = (cam.K, cam.R, cam.t, cam.width, cam.height)
                cache_keys_dict[cam.image_path] = cam._raster.extrinsics

        try:
            self._is_computing_visibility = True
            self.main_window.status_bar.showMessage(
                f"Computing occlusion maps for {len(camera_params_dict)} cameras..."
            )
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # Pass the cache data to the worker
            worker = VisibilityWorker(
                primary_target=primary_target, 
                camera_params_dict=camera_params_dict, 
                compute_depth_maps=self.compute_depth_maps_enabled,
                cache_manager=self.cache_manager,
                cache_keys_dict=cache_keys_dict,
                target_file_path=primary_target.file_path if primary_target else ""
            )
            
            thread = QThread()
            worker.moveToThread(thread)
            thread.started.connect(worker.run)

            # Connect signals
            worker.signals.finished.connect(self._on_visibility_computed)
            worker.signals.error.connect(self._on_visibility_error)

            # Cleanup when done
            worker.signals.finished.connect(thread.quit)
            worker.signals.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)

            # Keep references to avoid GC
            self._active_workers.append((thread, worker))

            thread.start()

        except Exception as e:
            print(f"Failed to start visibility worker: {e}")
            self._is_computing_visibility = False

    def _extract_visibility_geometry(self, primary_target):
        """
        Extract 3D geometry from the primary target for visibility computation.
        
        Handles different product types:
        - PointCloudProduct: Returns point coordinates directly
        - MeshProduct: Returns face center coordinates with face IDs
        - DEMProduct: Returns grid cell center coordinates with cell IDs
        
        Args:
            primary_target: AbstractSceneProduct instance
            
        Returns:
            tuple: (points_world, element_ids, element_type) where:
                - points_world: (N, 3) array of 3D coordinates
                - element_ids: (N,) array of element IDs or None for default indexing
                - element_type: str ('point', 'face', or 'cell')
        """
        from coralnet_toolbox.MVAT.core.Model import PointCloudProduct, MeshProduct, DEMProduct
        
        element_type = primary_target.get_element_type()
        
        # Strategy A: Point Cloud - use points directly
        if isinstance(primary_target, PointCloudProduct):
            points = primary_target.get_points_array()
            if points is not None and len(points) > 0:
                return points, None, 'point'
            return None, None, 'point'
        
        # Strategy B: Mesh - use face centers
        if isinstance(primary_target, MeshProduct):
            try:
                face_centers = primary_target.get_face_centers()
                face_ids = np.arange(len(face_centers), dtype=np.int32)
                print(f"📐 MeshProduct: Extracted {len(face_centers):,} face centers for visibility")
                return face_centers, face_ids, 'face'
            except Exception as e:
                print(f"⚠️ Failed to extract mesh face centers: {e}")
                return None, None, 'face'
        
        # Strategy C: DEM - use grid cell centers
        if isinstance(primary_target, DEMProduct):
            try:
                dem_height, dem_width = primary_target.elevation.shape
                rows, cols = np.mgrid[0:dem_height, 0:dem_width]
                
                # Convert pixel coords to world coords using affine transform
                transform = primary_target.transform
                x_world = transform[0, 0] * cols + transform[0, 1] * rows + transform[0, 2]
                y_world = transform[1, 0] * cols + transform[1, 1] * rows + transform[1, 2]
                z_world = primary_target.elevation
                
                # Flatten to point array
                points = np.column_stack([
                    x_world.flatten(),
                    y_world.flatten(),
                    z_world.flatten()
                ])
                
                # Cell IDs: row * width + col
                cell_ids = np.arange(dem_height * dem_width, dtype=np.int32)
                
                # Filter out NaN elevations
                valid_mask = ~np.isnan(points[:, 2])
                points = points[valid_mask]
                cell_ids = cell_ids[valid_mask]
                
                print(f"🗺️ DEMProduct: Extracted {len(points):,} valid cell centers for visibility")
                return points, cell_ids, 'cell'
            except Exception as e:
                print(f"⚠️ Failed to extract DEM cell centers: {e}")
                return None, None, 'cell'
        
        # Fallback: try get_points_array if available (generic interface)
        if hasattr(primary_target, 'get_points_array'):
            points = primary_target.get_points_array()
            if points is not None and len(points) > 0:
                return points, None, element_type
        
        print(f"⚠️ MVATManager: Cannot extract geometry from {type(primary_target).__name__}")
        return None, None, element_type

    def _calculate_camera_proximity_score(self, reference_camera, candidate_camera):
        """
        Calculate a scalar proximity score between two cameras used for
        ordering the camera grid.

        The score is an interpolation of a distance-based score (exponentially
        decaying with scene-normalized spatial distance) and a view-alignment
        score (dot product between viewing directions). Cameras behind the
        reference (negative alignment) are given a score of 0.
        """
        spatial_distance = np.linalg.norm(reference_camera.position - candidate_camera.position)
        ref_view_dir = reference_camera.R.T @ np.array([0, 0, 1])
        cand_view_dir = candidate_camera.R.T @ np.array([0, 0, 1])
        
        ref_view_dir = ref_view_dir / np.linalg.norm(ref_view_dir)
        cand_view_dir = cand_view_dir / np.linalg.norm(cand_view_dir)
        
        view_alignment = np.dot(ref_view_dir, cand_view_dir)
        
        try:
            bounds = self.viewer.get_bounds()
            scene_size = np.sqrt((bounds[1] - bounds[0])**2 + (bounds[3] - bounds[2])**2 + (bounds[5] - bounds[4])**2)
            normalized_distance = spatial_distance / (scene_size + 1e-6)
        except Exception:
            normalized_distance = spatial_distance / 10.0
            
        distance_score = np.exp(-2.0 * normalized_distance)
        view_score = (view_alignment + 1.0) / 2.0
        
        combined_score = 0.5 * distance_score + 0.5 * view_score
        if view_alignment < 0:
            combined_score = 0.0
            
        return combined_score

    def _reorder_cameras(self, reference_path, hide_distant_cameras=True):
        """Reorder cameras based on proximity to reference camera."""
        reference_camera = self.cameras.get(reference_path)
        if not reference_camera: 
            return
        
        # Skip reordering for orthomosaics (no meaningful view direction)
        if reference_camera.is_orthographic:
            return
        
        camera_scores = []
        for path, camera in self.cameras.items():
            if path == reference_path:
                score = float('inf')
            else:
                score = self._calculate_camera_proximity_score(reference_camera, camera)
            
            if hide_distant_cameras and score == 0.0 and path != reference_path:
                continue
            camera_scores.append((path, score))
            
        camera_scores.sort(key=lambda x: x[1], reverse=True)
        ordered_paths = [p for p, s in camera_scores]
        self.camera_grid.set_camera_order(ordered_paths)

    def cleanup(self):
        """Clean up resources before closing."""
        self.mouse_bridge.cleanup()
        if hasattr(self.viewer, 'close'):
            self.viewer.close()