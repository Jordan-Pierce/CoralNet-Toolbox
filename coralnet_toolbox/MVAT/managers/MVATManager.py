"""
MultiView Annotation Tool (MVAT) Manager

The central controller for the MVAT workspace.
Handles the business logic, data synchronization, and signal routing between 
the MainWindow, RasterManager, MVATViewer (3D), and CameraGrid (2D).
"""

import time
import numpy as np

from PyQt5.QtCore import QObject, QTimer, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMessageBox

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.MVAT.core.Camera import Camera
from coralnet_toolbox.MVAT.core.Ray import CameraRay
from coralnet_toolbox.MVAT.managers.SelectionManager import SelectionManager
from coralnet_toolbox.MVAT.managers.VisibilityManager import VisibilityManager
from coralnet_toolbox.MVAT.managers.CacheManager import CacheManager
from coralnet_toolbox.MVAT.core.constants import (
    MARKER_COLOR_SELECTED,
    MARKER_COLOR_INVALID,
    RAY_COLOR_SELECTED,
    RAY_COLOR_HIGHLIGHTED,
    RAY_COLOR_INVALID,
    MOUSE_THROTTLE_MS,
)


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
            self.manager.camera_grid.update_markers(projections, accuracies, highlighted_paths, visibility_status, selected_path)
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
        self._show_full_cloud = False
        
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
        self.viewer.showFullCloudToggled.connect(self._on_full_cloud_toggled)
        self.viewer.computeDepthMapsToggled.connect(self._on_compute_depth_maps_toggled)
        
        # 5. Main Window Sync
        if hasattr(self.annotation_window, 'mouseMoved'):
            self.annotation_window.mouseMoved.connect(self.mouse_bridge.on_mouse_moved)
        if hasattr(self.image_window, 'imageLoaded'):
            self.image_window.imageLoaded.connect(self._on_main_image_loaded)

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
            
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress = ProgressBar(self.main_window, title="Loading Cameras")
        progress.show()
        progress.start_progress(len(all_paths))
        
        valid_count = 0
        try:
            for path in all_paths:
                if progress.canceled: break
                raster = self.raster_manager.get_raster(path)
                if raster and raster.intrinsics is not None and raster.extrinsics is not None:
                    try:
                        self.cameras[path] = Camera(raster)
                        valid_count += 1
                    except Exception: pass
                progress.update_progress()
        finally:
            QApplication.restoreOverrideCursor()
            progress.finish_progress()
            progress.close()
            
        if valid_count == 0:
            QMessageBox.information(self.main_window, "No Camera Data", "No valid camera parameters found.")
            return
            
        try:
            self.camera_grid.stats_label.setText(f"Cameras: {valid_count} / {len(all_paths)}")
        except Exception: pass
        
        self.camera_grid.set_cameras(self.cameras)
        self._render_frustums()
        self.viewer.fit_to_view()
        
        # Initial Synchronization
        current_image_path = getattr(self.annotation_window, 'current_image_path', None)
        if current_image_path and current_image_path in self.cameras:
            self.selection_model.set_active(current_image_path)
        elif self.cameras:
            self.selection_model.set_active(next(iter(self.cameras)))

    def _render_frustums(self):
        """
        Update the 3D scene to render frustums, point cloud and axes.

        This prepares the viewer by ensuring the point cloud and axes are
        present, then asks the viewer to draw all camera frustums using the
        current selection/highlight/hover state.
        """
        self.viewer.add_point_cloud()
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

    def _on_full_cloud_toggled(self, state: bool):
        """
        Toggle between showing the full point cloud or a visibility-filtered
        subset.

        When showing the full cloud, any visibility filtering is bypassed. When
        returning to filtered mode, recompute the visibility subset for the
        currently highlighted cameras (and the selected camera if needed).
        """
        self._show_full_cloud = state
        if state:
            self.viewer.update_point_cloud_subset(None)
        else:
            highlighted_paths = list(self.camera_grid.get_highlighted_cameras())
            if self.selected_camera and self.selected_camera.image_path not in highlighted_paths:
                highlighted_paths.append(self.selected_camera.image_path)
            self._update_visibility_filter(highlighted_paths)

    def _on_compute_depth_maps_toggled(self, state: bool):
        """
        Enable or disable computing and storing depth maps during visibility
        computation. Depth maps can improve occlusion checks but are more
        expensive to compute and merge into rasters.
        """
        self.compute_depth_maps_enabled = state

    def _on_active_camera_changed(self, path):
        """
        Handler for when the selection model reports a new active camera.

        Updates internal selection state, clears any active ray visualization,
        instructs the viewer to match the selected camera perspective (when
        supported), reorders the grid to prioritize nearby cameras, and asks
        the image window to load the selected image.
        """
        camera = self.cameras.get(path)
        if camera:
            self.viewer.clear_ray()
            self._select_camera(path, camera)
            if hasattr(self.viewer, 'match_camera_perspective'):
                self.viewer.match_camera_perspective(camera)
            self._reorder_cameras(path)
            
            try:
                self.image_window.load_image_by_path(path)
            except Exception: pass

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
            except Exception: pass

        self.highlighted_cameras = [self.cameras.get(path) for path in selected_paths if path in self.cameras]
        self._update_frustum_states()
        
        try:
            self.camera_grid._sync_ui_to_model()
        except Exception: pass

        self.viewer.clear_ray()
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
        except Exception: pass

    def _update_visibility_filter(self, highlighted_paths):
        """
        Compute and apply a visibility filter on the point cloud based on the
        supplied highlighted cameras.

        Steps:
        - If the viewer has no point cloud or the manager is configured to
          show the full cloud, do nothing.
        - For each highlighted camera, either collect cached visible indices
          or queue cameras that require visibility computation.
        - If computation is required, batch compute visibility using
          VisibilityManager, optionally compute depth maps, cache results,
          and merge depth maps into rasters when requested.
        - Combine (union) the visible indices across cameras and ask the
          viewer to display the resulting subset.
        """
        if not self.viewer.point_cloud or self._show_full_cloud:
            return
            
        if not highlighted_paths:
            self.viewer.update_point_cloud_subset([])
            return
            
        all_visible_indices = []
        cameras_needing_visibility = []
        
        for path in highlighted_paths:
            camera = self.cameras.get(path)
            if camera:
                if camera.visible_indices is None:
                    cameras_needing_visibility.append(camera)
                else:
                    all_visible_indices.append(camera.visible_indices)
                    
        if cameras_needing_visibility:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            progress = ProgressBar(self.main_window, title="Computing Visibility")
            progress.show()
            progress.start_progress(len(cameras_needing_visibility))
            
            try:
                points_world = self.viewer.point_cloud.get_points_array()
                camera_params = [(cam.K, cam.R, cam.t, cam.width, cam.height) for cam in cameras_needing_visibility]
                
                batch_results = VisibilityManager.compute_batch_visibility(
                    points_world, 
                    camera_params, 
                    compute_depth_map=self.compute_depth_maps_enabled
                )
                
                for camera, result in zip(cameras_needing_visibility, batch_results):
                    cache_path = None
                    if self.cache_manager is not None:
                        cache_path = self.cache_manager.save_visibility(
                            camera._raster.extrinsics,
                            self.viewer.point_cloud.file_path,
                            result['index_map'],
                            result['visible_indices'],
                            result.get('depth_map') if self.compute_depth_maps_enabled else None
                        )
                    
                    camera._raster.add_index_map(result['index_map'], cache_path, result['visible_indices'])
                    
                    if self.compute_depth_maps_enabled and result.get('depth_map') is not None:
                        try: camera._raster.merge_or_set_depth_map(result.get('depth_map'))
                        except Exception: pass
                        
                    all_visible_indices.append(result['visible_indices'])
                    progress.update_progress()
            finally:
                QApplication.restoreOverrideCursor()
                progress.finish_progress()
                progress.close()
                
        if not all_visible_indices:
            self.viewer.update_point_cloud_subset([])
            return
            
        union_indices = np.unique(np.concatenate(all_visible_indices)) if len(all_visible_indices) > 1 else all_visible_indices[0]
        self.viewer.update_point_cloud_subset(union_indices)

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
            scene_size = np.sqrt((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + (bounds[5]-bounds[4])**2)
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
        reference_camera = self.cameras.get(reference_path)
        if not reference_camera: return
        
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