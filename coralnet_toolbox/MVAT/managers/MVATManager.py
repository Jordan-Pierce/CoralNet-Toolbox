"""
MultiView Annotation Tool (MVAT) Manager

The central controller for the MVAT workspace.
Handles the business logic, data synchronization, and signal routing between 
the MainWindow, RasterManager, MVATViewer (3D), and ContextMatrix (2D).
"""

import os
import time
import numpy as np

from PyQt5.QtCore import QObject, QTimer, pyqtSignal, Qt, QThread
from PyQt5.QtWidgets import QApplication, QMessageBox

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
)

# DEMProduct removed: orthomosaics/DEMs are no longer scene products

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtMaskAnnotation import MaskAnnotation

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.MVAT.core.Model import MeshProduct


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MousePositionBridge(QObject):
    """
    Bridges mouse position events from the AnnotationWindow to the MVAT
    controller.

    Responsibilities:
    - Build a 3D ray from a selected camera and a 2D image pixel (uses depth
        from a z-channel when available, otherwise falls back to a scene
        median/default depth).
    - Create corresponding rays from highlighted cameras to the same world
        point and choose colors based on depth accuracy.
    - Project the selected ray into other camera image spaces and forward
        marker/visibility updates to the ContextMatrix for UI presentation.

    This class mirrors the behavior previously implemented on the window
    layer but is now manager-owned so it can operate without direct UI
    responsibilities.
    """
    def __init__(self, manager: 'MVATManager'):
        super().__init__()
        self.manager = manager
        self.enabled = True
        self._last_update_time = 0
        
    def on_mouse_moved(self, x: int, y: int):
        if not self.enabled:
            return
        self._process_pending_position(x, y)
            
    def _process_pending_position(self, x: int, y: int):
        camera = self.manager.selected_camera
        if camera is None or not (0 <= x < camera.width and 0 <= y < camera.height):
            self.clear_all_markers()
            self.manager.viewer.clear_ray()
            return

        primary_target = self.manager.viewer.scene_context.get_primary_target()

        # --- Path A: Index Map (preferred) ---
        ray = None
        candidate_id = -1
        # Use the in-memory raster index_map to avoid triggering lazy disk loads
        # on frequent UI events (mouse move).
        index_map = camera._raster.index_map
        if index_map is not None:
            candidate_id = int(index_map[y, x])

        if candidate_id > -1 and primary_target is not None:
            coord = primary_target.get_element_coordinate(candidate_id)
            if coord is not None:
                if getattr(camera, 'is_orthographic', False):
                    # Get the correct world-up direction in local space
                    if getattr(camera, 'chunk_transform_inv', None) is not None:
                        world_up_hom = np.array([0.0, 0.0, 1.0, 0.0])
                        local_up = (camera.chunk_transform_inv @ world_up_hom)[:3]
                        n = np.linalg.norm(local_up)
                        local_up = local_up / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])
                    else:
                        local_up = np.array([0.0, 0.0, 1.0])
                    origin    = coord + local_up * 1000.0
                    direction = -local_up
                
                    ray = CameraRay(
                        origin=origin,
                        direction=direction,
                        terminal_point=coord,
                        has_accurate_depth=True,
                        pixel_coord=(x, y),
                        source_camera=camera,
                        element_id=candidate_id,
                    )

        # --- Path B: Z-channel / depth fallback ---
        if ray is None:
            raster = camera._raster
            depth = None
            z_data_type = raster.z_data_type if hasattr(raster, 'z_data_type') else None
            
            if raster.z_channel is not None:
                z_value = raster.get_z_value(x, y)
                # For depth maps, only accept positive values
                # For elevation maps, accept any value (including negative)
                if z_value is not None:
                    if z_data_type == 'elevation' or (z_data_type == 'depth' and z_value > 0) or z_data_type is None:
                        depth = z_value
            
            if depth is None or np.isnan(depth):
                default_depth = self.manager.viewer.get_scene_median_depth(camera.position)
            else:
                default_depth = depth
            
            ray = CameraRay.from_pixel_and_camera(
                pixel_xy=(x, y),
                camera=camera,
                depth=depth,
                default_depth=default_depth,
            )

        highlighted_cameras = self.manager.highlighted_cameras
        rays_with_colors = [(ray, RAY_COLOR_SELECTED if ray.has_accurate_depth else RAY_COLOR_INVALID)]
        
        # --- Short-Circuit Invalid Primary Rays ---
        # If the primary ray did not hit real scene geometry, skip secondary rays entirely
        # (no secondary rays should project into an arbitrary guessed point in empty space).
        primary_ray_valid = ray.has_accurate_depth or ray.element_id > -1
        if not primary_ray_valid:
            # Primary ray is invalid: send only the invalid primary ray (RED), clear markers, and return
            self.manager.viewer.show_rays(rays_with_colors)
            self.clear_all_markers()
            return
        
        # --- Primary ray is valid: proceed with secondary rays ---
        visibility_status = {}
        accuracies = {camera.image_path: ray.has_accurate_depth}

        for target_cam in highlighted_cameras:
            if target_cam.image_path == camera.image_path:
                continue

            # Orthographic secondary cameras: skip index-map occlusion check
            if target_cam.is_orthographic:
                target_ray = CameraRay.from_world_point_and_camera(
                    world_point=ray.terminal_point, camera=target_cam)
                rays_with_colors.append((target_ray, RAY_COLOR_HIGHLIGHTED))
                visibility_status[target_cam.image_path] = False
                accuracies[target_cam.image_path] = True
                continue

            # Project primary terminal point into this camera
            proj = target_cam.project(ray.terminal_point)
            u_proj = int(round(float(proj[0]))) if not np.isnan(proj[0]) else -1
            v_proj = int(round(float(proj[1]))) if not np.isnan(proj[1]) else -1
            in_bounds = (
                not np.isnan(proj[0])
                and 0 <= u_proj < target_cam.width
                and 0 <= v_proj < target_cam.height
            )

            # If the projected point is out of bounds, skip this camera entirely
            # (no secondary ray should be drawn into empty space outside the image)
            if not in_bounds:
                continue

            target_terminal = ray.terminal_point
            ray_color = RAY_COLOR_INVALID
            is_occluded = True
            found_id = -1

            if getattr(target_cam, '_raster', None) is not None and target_cam._raster.index_map is not None and ray.element_id > -1:
                found_id = int(target_cam._raster.index_map[v_proj, u_proj])

                # Determine visibility with spatial tolerance.
                # METHOD A: 3D Distance Threshold (ACTIVE)
                # Resolves false-positive occlusions caused by sub-pixel rounding when
                # projecting the primary 3D point into the secondary camera.  A found_id
                # that differs from primary_element_id may still belong to the same
                # physical surface patch; we accept it as visible when its 3D centre is
                # within 5% of the camera-to-point distance.
                is_visible = False
                if found_id == ray.element_id:
                    is_visible = True
                elif found_id > -1 and primary_target is not None:
                    found_coord = primary_target.get_element_coordinate(found_id)
                    if found_coord is not None:
                        surface_dist = np.linalg.norm(found_coord - ray.terminal_point)
                        cam_to_point_dist = np.linalg.norm(target_cam.position - ray.terminal_point)
                        tolerance = 0.05 * cam_to_point_dist
                        if surface_dist <= tolerance:
                            is_visible = True

                # METHOD B: 2D Neighbourhood Search (COMMENTED OUT — for comparison testing)
                # Checks whether primary_element_id appears anywhere in a 5×5 pixel window
                # around the projected pixel, catching cases where aliasing shifts the hit
                # by 1-2 pixels.
                # -----------------------------------------------------------------------
                # HALF = 2  # half-width of the search window (full window = 2*HALF+1)
                # v_lo = max(0, v_proj - HALF)
                # v_hi = min(target_cam.height, v_proj + HALF + 1)
                # u_lo = max(0, u_proj - HALF)
                # u_hi = min(target_cam.width,  u_proj + HALF + 1)
                # neighbourhood = target_cam.index_map[v_lo:v_hi, u_lo:u_hi]
                # is_visible = int(ray.element_id) in neighbourhood
                # -----------------------------------------------------------------------

                if is_visible:
                    ray_color = RAY_COLOR_HIGHLIGHTED
                    target_terminal = ray.terminal_point
                    is_occluded = False
                    accuracies[target_cam.image_path] = True

                elif found_id > -1:                                 # TRUE OCCLUSION
                    occluder = primary_target.get_element_coordinate(found_id) if primary_target else None
                    target_terminal = occluder if occluder is not None else ray.terminal_point
                    accuracies[target_cam.image_path] = False

                else:                                               # BACKGROUND (-1)
                    accuracies[target_cam.image_path] = False

            else:
                # Legacy fallback: depth-based occlusion test
                is_occluded = target_cam.is_point_occluded_depth_based(
                    ray.terminal_point, depth_threshold=0.15)
                ray_color = RAY_COLOR_HIGHLIGHTED if not is_occluded else RAY_COLOR_INVALID
                accuracies[target_cam.image_path] = target_cam._raster.z_channel is not None

            visibility_status[target_cam.image_path] = is_occluded

            # Build secondary ray directly
            t_origin = target_cam.position.copy()
            t_direction = target_terminal - t_origin
            t_norm = np.linalg.norm(t_direction)
            t_direction = t_direction / t_norm if t_norm > 0 else target_cam.R.T @ np.array([0, 0, 1])
            target_ray = CameraRay(
                origin=t_origin,
                direction=t_direction,
                terminal_point=target_terminal,
                has_accurate_depth=(ray_color == RAY_COLOR_HIGHLIGHTED),
                source_camera=target_cam,
                element_id=found_id,
            )
            rays_with_colors.append((target_ray, ray_color))

        self.manager.viewer.show_rays(rays_with_colors)
        projections = ray.project_to_cameras(self.manager.cameras)

        # Update context matrix canvases (Phase 4)
        if self.manager.context_matrix is not None:
            try:
                self.manager.context_matrix.update_dynamic_markers(
                    projections, accuracies, visibility_status
                )
            except Exception:
                self.manager.context_matrix.clear_all_dynamic_markers()
                
    def clear_all_markers(self):
        if self.manager.context_matrix is not None:
            try:
                self.manager.context_matrix.clear_all_dynamic_markers()
            except Exception:
                pass
            
    def cleanup(self):
        self.clear_all_markers()


class MVATManager(QObject):
    """
    Core Controller for the MVAT Workspace.
    """
    cameraSelectedInMVAT = pyqtSignal(str)
    
    def __init__(self, main_window, viewer):
        super().__init__()
        
        self.main_window = main_window
        self.raster_manager = main_window.image_window.raster_manager
        self.annotation_window = main_window.annotation_window
        self.image_window = main_window.image_window
        
        self.viewer = viewer
        self.context_matrix = getattr(main_window, 'context_matrix', None)
        
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
        # Scale factor for visibility map resolution (1.0 = native, 0.1 = lowest)
        self.visibility_scale_factor = 0.50
        # Safety flag to prevent concurrent visibility computations
        self._is_computing_visibility = False
        # Track active worker threads to prevent GC
        self._active_workers = []

        # Multi-camera annotation state
        self.multi_annotate_enabled = False
        self._propagating_annotation = False

        # Internal Managers
        self.selection_model = SelectionManager(self)
        self.cache_manager = CacheManager("")
        self.mouse_bridge = MousePositionBridge(self)

        self._setup_connections()

    def _request_viewer_update(self):
        """Requests a 3D redraw."""
        self._do_viewer_update()

    def _do_viewer_update(self):
        """Performs the actual synchronous PyVista render."""
        if self.viewer:
            try:
                self.viewer.update()
            except Exception:
                pass

    def _setup_connections(self):
        """
        Bind all signals between UI views and this controller.

        Connections established:
        1. SelectionModel signals -> manager handlers (active/selection changed)
        2. ContextMatrix intent signals (loadCamerasRequested, clearSelectionsRequested)
        3. ContextMatrix hover/promote events -> manager handlers
        4. Viewer notifications (focal point, full-cloud toggle, compute-depths)
        5. Main window sync: wire the annotation window's mouseMoved and the
            image window's imageLoaded signals to manager handlers when present.
        """
        # 1. Selection Model (The Source of Truth)
        self.selection_model.active_changed.connect(self._on_active_camera_changed)
        self.selection_model.selection_changed.connect(self._on_selections_changed)
        
        # 4. Viewer Signals
        self.viewer.focalPointChanged.connect(self._on_focal_point_changed)
        self.viewer.computeIndexMapsToggled.connect(self._on_compute_index_maps_toggled)
        self.viewer.computeDepthMapsToggled.connect(self._on_compute_depth_maps_toggled)
        self.viewer.visibilityQualityChanged.connect(self._on_visibility_quality_changed)
        
        # 5. Main Window Sync
        if hasattr(self.annotation_window, 'mouseMoved'):
            self.annotation_window.mouseMoved.connect(self.mouse_bridge.on_mouse_moved)
        if hasattr(self.image_window, 'imageLoaded'):
            self.image_window.imageLoaded.connect(self._on_main_image_loaded)
        # 6. Context Matrix Signals
        if self.context_matrix is not None:
            # Promote (double-click) -> load image as active camera
            self.context_matrix.contextImagePromoted.connect(self._on_camera_selected)
            # Toolbar buttons
            self.context_matrix.loadCamerasRequested.connect(self.load_cameras)
            self.context_matrix.clearSelectionsRequested.connect(self.selection_model.clear_selections)
            # Selection intent signals -> SelectionModel (source of truth)
            self.context_matrix.selection_requested.connect(self.selection_model.set_selections)
            self.context_matrix.toggle_requested.connect(self.selection_model.toggle)
            self.context_matrix.active_requested.connect(self.selection_model.set_active)
            self.context_matrix.camera_highlighted_single.connect(self._on_camera_highlighted_single)
            # Phase 5 / multi-annotate
            self.context_matrix.set_mvat_manager(self)
            self.context_matrix.multiAnnotateToggled.connect(self._on_multi_annotate_toggled)
        
        # 7. Target-Lock Sync (Phase 5): AnnotationWindow viewNavigated -> sync engine
        if hasattr(self.annotation_window, 'viewNavigated'):
            self.annotation_window.viewNavigated.connect(self._on_main_view_navigated)

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
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            self.main_window.status_bar.showMessage("Loading cameras...", 0)

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
            
            self.main_window.status_bar.showMessage(
                f"Loaded cameras: {valid_count} total ({ortho_count} orthomosaics, "
                f"{valid_count - ortho_count} perspective)",
                3000
            )
            
        if valid_count == 0:
            QMessageBox.information(self.main_window, "No Camera Data", "No valid camera parameters found.")
            return
        
        # =====================================================================
        # Pre-computation Cache Check and Dialog
        # =====================================================================
        primary_target = self.viewer.scene_context.get_primary_target()
        
        # We can only pre-compute if a 3D model is loaded FIRST.
        if primary_target is not None and self.cache_manager is not None and self.compute_index_maps_enabled:
            target_path = primary_target.file_path
            element_type = primary_target.get_element_type()
            uncached_cameras = []
            
            for path, cam in self.cameras.items():
                cache_key = cam.transform_matrix if cam.is_orthographic else cam._raster.extrinsics
                cache_path = self.cache_manager.get_cache_path(cache_key, target_path, element_type)
                
                # Check if the cache file exists on disk
                if not os.path.exists(cache_path):
                    uncached_cameras.append(cam)
            
            if uncached_cameras:
                reply = QMessageBox.question(
                    self.main_window,
                    "Pre-compute Visibility?",
                    f"Found {len(uncached_cameras)} cameras without cached visibility maps.\n\n"
                    "Would you like to compute them all now? (This will take time upfront, but makes navigating the scene instantly responsive.)\n\n"
                    "Select 'No' to compute them in the background as you click them.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    # Prompt for quality level
                    from PyQt5.QtWidgets import QInputDialog
                    qualities = ["Highest (100%)", "High (75%)", "Medium (50%)", "Low (25%)", "Lowest (10%)"]
                    quality_map = dict(zip(qualities, [1.0, 0.75, 0.50, 0.25, 0.10]))

                    current_idx = 0
                    for i, s in enumerate([1.0, 0.75, 0.50, 0.25, 0.10]):
                        if self.visibility_scale_factor == s:
                            current_idx = i
                            break

                    choice, ok = QInputDialog.getItem(
                        self.main_window,
                        "Select Quality",
                        "Choose the resolution scale for the maps:\n(Lower is faster but less accurate)",
                        qualities,
                        current_idx,
                        False
                    )

                    if ok and choice:
                        self.visibility_scale_factor = quality_map[choice]
                        # Sync the View menu checkmark
                        for action in self.viewer._quality_action_group.actions():
                            if action.text() == choice:
                                action.setChecked(True)
                                break
                        self._compute_visibility_async(primary_target, uncached_cameras)
        
        # FILTER: Only pass perspective cameras to grid UI
        perspective_cameras = {p: c for p, c in self.cameras.items() if not c.is_orthographic}    

        if self.context_matrix is not None:
            try:
                self.context_matrix.update_stats(len(perspective_cameras), ortho_count)
            except Exception:
                pass
            try:
                all_ordered = list(perspective_cameras.keys())
                self.context_matrix.set_camera_data(list(perspective_cameras.values()), all_ordered)
            except Exception:
                pass
        
        # Generate and load 3D elevation for any orthomosaics with DEMs
        self._populate_ortho_elevation()
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
        
    def _populate_ortho_elevation(self):
        """
        Previously populated the 3D scene with elevation meshes generated
        from orthomosaic DEMs. DEM-as-mesh support has been removed, so
        this is now a no-op to preserve call sites.
        """
        # No-op: elevation meshes are no longer generated or added to the scene.
        return

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
                # For elevation models, accept negative values; for depth, only positive values indicate valid data
                z_data_type = self.selected_camera._raster.z_data_type
                is_elevation = z_data_type == 'elevation'
                depth_valid = depth is not None and (is_elevation or depth > 0)
                color = MARKER_COLOR_SELECTED if depth_valid else MARKER_COLOR_INVALID
                self.annotation_window.set_incoming_marker(u, v, color)
            else:
                self.annotation_window.clear_static_marker()
        
        # Project focal point into context matrix canvases (Phase 4)
        if self.context_matrix is not None:
            try:
                self.context_matrix.update_static_markers_from_3d(point_3d, self.cameras)
            except Exception:
                pass

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
        self.main_window.status_bar.showMessage("Compute Index Maps: ON" if state else "Compute Index Maps: OFF", 2000)

    def _on_visibility_quality_changed(self, scale_factor: float):
        """Store the user-selected visibility resolution scale factor."""
        self.visibility_scale_factor = scale_factor
        print(f"Visibility quality scale set to {scale_factor}")

    def _on_visibility_computed(self, results: dict):
        """Handle results emitted from VisibilityWorker (runs on main thread)."""
        try:
            # Get primary target file path for cache key
            primary_target = self.viewer.scene_context.get_primary_target()
            target_file_path = primary_target.file_path if primary_target else ""
            
            self._process_visibility_results(results, target_file_path)

        finally:
            self._is_computing_visibility = False
            self.main_window.status_bar.showMessage("Visibility maps updated.", 3000)
            QApplication.restoreOverrideCursor()

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
                    element_type=element_type,
                    inverted_index=result.get('inverted_index')
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
        self.main_window.status_bar.showMessage("Visibility computation failed. See console for details.", 5000)
        QApplication.restoreOverrideCursor()

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
                # Double-click to set active: animate
                self.viewer.match_camera_perspective(camera, animate=True)
            self._reorder_cameras(path)

            if self.context_matrix is not None:
                try:
                    self.context_matrix.sync_selection_borders(
                        path, self.selection_model.selected_paths
                    )
                except Exception:
                    pass

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

        This updates the viewer perspective to match the clicked camera with
        smooth animation when supported.
        """
        try:
            cam = self.cameras.get(path)
            if cam and hasattr(self.viewer, 'match_camera_perspective'):
                self.viewer.match_camera_perspective(cam, animate=True)
        except Exception:
            pass

    def _on_selections_changed(self, selected_paths):
        """
        Respond to selection model changes (highlight toggles).

        Applies highlight state to Camera objects, updates viewer frustums,
        synchronizes the ContextMatrix UI to the model, clears any active ray
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
        
        if self.context_matrix is not None:
            try:
                active_path = self.selection_model.active_path or ""
                active_label = ""
                if active_path:
                    from pathlib import Path
                    active_label = Path(active_path).stem
                self.context_matrix.sync_selection_borders(active_path, selected_paths)
                self.context_matrix.update_selection_labels(active_label, len(selected_paths))
            except Exception:
                pass

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

        Updates:
        1. Wireframe state scalars via update_frustum_states()
        2. Thumbnail visibility to show selected and highlighted cameras

        Errors are caught and ignored to avoid destabilizing the UI for
        non-critical failures.
        """
        selected_path = self.selected_camera.image_path if self.selected_camera else None
        highlighted_paths = [cam.image_path for cam in self.highlighted_cameras]
        
        # Update wireframe state scalars (colors based on selection/highlight)
        try:
            if hasattr(self.viewer, 'update_frustum_states'):
                self.viewer.update_frustum_states(selected_path, highlighted_paths, self.hovered_camera)
        except Exception: 
            pass
        
        # Update thumbnails (show/hide based on selection/highlight state)
        try:
            if hasattr(self.viewer, '_show_thumbnails_enabled') and self.viewer._show_thumbnails_enabled:
                # Clear all existing thumbnails
                if hasattr(self.viewer, 'remove_thumbnails'):
                    self.viewer.remove_thumbnails()
                
                # Add thumbnail for selected camera
                if self.selected_camera is not None:
                    try:
                        if hasattr(self.viewer, '_add_thumbnail_for_camera'):
                            self.viewer._add_thumbnail_for_camera(self.selected_camera)
                    except Exception:
                        pass
                
                # Add thumbnails for highlighted cameras (excluding the selected camera to avoid duplication)
                for cam in self.highlighted_cameras:
                    if self.selected_camera is None or cam.image_path != self.selected_camera.image_path:
                        try:
                            if hasattr(self.viewer, '_add_thumbnail_for_camera'):
                                self.viewer._add_thumbnail_for_camera(cam)
                        except Exception:
                            pass
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
                        element_type=element_type,
                        inverted_index=cached_data.get('inverted_index')
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
            
            self.main_window.status_bar.showMessage(f"Rasterizing mesh for {n_cameras} camera(s)...")
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
            self.main_window.status_bar.showMessage("Visibility maps updated.", 3000)
            QApplication.restoreOverrideCursor()

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
                # Include chunk_transform_inv if available (for local->world bridge in visibility computation)
                camera_params_dict[cam.image_path] = ('ortho', cam.transform_matrix_inv, cam.width, cam.height, cam.chunk_transform_inv)
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

            # Pass the cache data and scale factor to the worker
            worker = VisibilityWorker(
                primary_target=primary_target, 
                camera_params_dict=camera_params_dict, 
                compute_depth_maps=self.compute_depth_maps_enabled,
                cache_manager=self.cache_manager,
                cache_keys_dict=cache_keys_dict,
                target_file_path=primary_target.file_path if primary_target else "",
                scale_factor=self.visibility_scale_factor
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
        
        Args:
            primary_target: AbstractSceneProduct instance
            
        Returns:
            tuple: (points_world, element_ids, element_type) where:
                - points_world: (N, 3) array of 3D coordinates
                - element_ids: (N,) array of element IDs or None for default indexing
                - element_type: str ('point', 'face', or 'cell')
        """
        from coralnet_toolbox.MVAT.core.Model import PointCloudProduct, MeshProduct
        
        element_type = primary_target.get_element_type()
        
        # Strategy A: Point Cloud - use points directly
        if isinstance(primary_target, PointCloudProduct):
            points = primary_target.get_points_array()
            if points is not None and len(points) > 0:
                return points, None, 'point'
            return None, None, 'point'
        
        # Strategy B: Mesh products - treat as solid triangulated surfaces
        if isinstance(primary_target, MeshProduct):
            try:
                # Ensure GPU tensors are built for the Bounding Volume Hierarchy
                if hasattr(primary_target, 'prepare_geometry'):
                    primary_target.prepare_geometry()
                
                # Ask the product for its true PyVista PolyData mesh
                mesh = primary_target.get_render_mesh()
                if mesh is None:
                    return None, None, 'face'
                
                # Extract the physical centers of the triangles
                face_centers = mesh.cell_centers().points
                face_ids = np.arange(len(face_centers), dtype=np.int32)
                
                print(f"📐 Extracted {len(face_centers):,} solid faces for {primary_target.label} visibility")
                return face_centers, face_ids, 'face'
                
            except Exception as e:
                print(f"⚠️ Failed to extract face centers for {primary_target.label}: {e}")
                return None, None, 'face'

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
        
        # Feed ContextMatrixWidget with proximity-ordered neighbors
        if self.context_matrix is not None:
            self.context_matrix.set_camera_order(ordered_paths, reference_path)

    # --- Target-Lock Sync Engine (Phase 5) ---

    def _on_main_view_navigated(self, center_x: float, center_y: float, zoom_factor: float):
        """Handle navigation events from the main AnnotationWindow.

        Projects the viewport center into 3D world space, then back into
        each visible context camera to synchronize their viewports.
        """
        if self.context_matrix is None:
            return
        if not self.context_matrix.target_lock_enabled:
            return
        if self.selected_camera is None:
            return

        # Step 1: Get the 3D world point at the viewport center
        world_point = self._get_world_point_at_pixel(
            self.selected_camera, center_x, center_y
        )
        if world_point is None:
            return

        # Step 2: Project into each visible context camera
        targets = {}
        capacity = self.context_matrix._get_visible_capacity()

        for i in range(capacity):
            canvas = self.context_matrix._canvas_pool[i]
            if not canvas.isVisible() or not canvas.current_image_path:
                continue

            camera = self.cameras.get(canvas.current_image_path)
            if not camera:
                continue

            try:
                pixel = camera.project(world_point)
            except Exception:
                continue

            if np.isnan(pixel).any():
                continue

            target_u, target_v = float(pixel[0]), float(pixel[1])

            # Bounds check: only sync if the point is within the image
            if 0 <= target_u < camera.width and 0 <= target_v < camera.height:
                targets[i] = (target_u, target_v)

        # Step 3: Compute relative zoom ratio (how far beyond fit-to-view)
        if self.selected_camera and hasattr(self.annotation_window, '_min_zoom'):
            min_zoom = self.annotation_window._min_zoom
            if min_zoom > 0:
                relative_zoom = zoom_factor / min_zoom
            else:
                relative_zoom = 1.0
        else:
            relative_zoom = 1.0

        # Step 4: Command the context matrix to sync
        self.context_matrix.request_sync(targets, relative_zoom)

    def _get_world_point_at_pixel(self, camera, px, py):
        """Get the 3D world point at a specific pixel coordinate.

        Attempts depth-based unprojection first, falls back to scene
        median depth for a rough estimate.

        Args:
            camera: Camera object for the active image.
            px, py: Pixel coordinates (float).

        Returns:
            np.ndarray [x,y,z] world point, or None if impossible.
        """
        # Clamp to image bounds
        px = max(0, min(px, camera.width - 1))
        py = max(0, min(py, camera.height - 1))

        # Try depth/elevation from Z-channel
        raster = camera._raster
        depth = None
        z_data_type = raster.z_data_type if hasattr(raster, 'z_data_type') else None
        
        if raster.z_channel is not None:
            z_value = raster.get_z_value(int(px), int(py))
            # For depth maps, only accept positive values
            # For elevation maps, accept any value (including negative)
            if z_value is not None:
                if z_data_type == 'elevation' or (z_data_type == 'depth' and z_value > 0) or z_data_type is None:
                    depth = z_value

        if depth is None or np.isnan(depth):
            # Fallback to scene median depth
            try:
                default_depth = self.viewer.get_scene_median_depth(camera.position)
            except Exception:
                default_depth = 10.0
        else:
            default_depth = depth

        try:
            ray = CameraRay.from_pixel_and_camera(
                pixel_xy=(px, py),
                camera=camera,
                depth=depth,
                default_depth=default_depth
            )
            return ray.terminal_point
        except Exception:
            return None

    # --- Multi-Camera Annotation ---

    def _on_multi_annotate_toggled(self, enabled: bool):
        """Connect or disconnect annotation propagation handlers when toggle changes."""
        self.multi_annotate_enabled = enabled
        brush_tool = self.annotation_window.tools.get('brush')
        patch_tool = self.annotation_window.tools.get('patch')
        sam_tool = self.annotation_window.tools.get('sam')
        fill_tool = self.annotation_window.tools.get('fill')
        erase_tool = self.annotation_window.tools.get('erase')

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        if enabled:
            self.annotation_window.annotationCreated.connect(self._on_patch_annotation_created)
            if brush_tool is not None:
                brush_tool.post_stroke_callback = self._on_brush_stroke_applied
                brush_tool.cursor_move_callback = self._on_cursor_preview_moved
                brush_tool.cursor_clear_callback = self._on_cursor_preview_cleared
            if patch_tool is not None:
                patch_tool.cursor_move_callback = self._on_cursor_preview_moved
                patch_tool.cursor_clear_callback = self._on_cursor_preview_cleared
            if fill_tool is not None:
                fill_tool.post_stroke_callback = self._on_fill_stroke_applied
                fill_tool.cursor_move_callback = self._on_cursor_preview_moved
                fill_tool.cursor_clear_callback = self._on_cursor_preview_cleared
            if erase_tool is not None:
                erase_tool.post_stroke_callback = self._on_erase_stroke_applied
                erase_tool.cursor_move_callback = self._on_cursor_preview_moved
                erase_tool.cursor_clear_callback = self._on_cursor_preview_cleared
            if sam_tool is not None:
                # Final-mask propagation callback (no live-hover propagation for now)
                sam_tool.post_prediction_callback = self._on_sam_prediction_applied
            # Proactively compute visibility/index maps for visible context cameras
            # so True 3D mapping will be available when the user paints or applies SAM.
            try:
                visible = list(self._get_visible_context_paths())
                if visible and self.compute_index_maps_enabled:
                    self.main_window.status_bar.showMessage("Preparing context visibility maps...", 2000)
                    # Ask the visibility system to compute index maps for these visible cameras
                    # _update_visibility_filter handles cache checks and async worker dispatch.
                    self._update_visibility_filter(visible)

                # --- Force Mask Canvas Allocation NOW ---
                # Don't wait for the first brush stroke to allocate canvases!
                project_labels = list(self.main_window.label_window.labels)
                for path in visible:
                    raster = self.raster_manager.get_raster(path)
                    if raster and raster.mask_annotation is None:
                        raster.get_mask_annotation(project_labels)

            except Exception:
                pass
        else:
            try:
                self.annotation_window.annotationCreated.disconnect(self._on_patch_annotation_created)
            except TypeError:
                pass

            if brush_tool is not None:
                brush_tool.post_stroke_callback = None
                brush_tool.cursor_move_callback = None
                brush_tool.cursor_clear_callback = None
            if patch_tool is not None:
                patch_tool.cursor_move_callback = None
                patch_tool.cursor_clear_callback = None
            if fill_tool is not None:
                fill_tool.post_stroke_callback = None
                fill_tool.cursor_move_callback = None
                fill_tool.cursor_clear_callback = None
            if erase_tool is not None:
                erase_tool.post_stroke_callback = None
                erase_tool.cursor_move_callback = None
                erase_tool.cursor_clear_callback = None
            if sam_tool is not None:
                sam_tool.post_prediction_callback = None
            self._on_cursor_preview_cleared()

        # Restore cursor
        QApplication.restoreOverrideCursor()

    def _on_cursor_preview_moved(self, scene_pos, item_factory):
        """Project the cursor position into visible context cameras and show previews.

        Uses the blazingly fast center-point projection to display brush previews
        in all visible context cameras. The tool factory already draws the correct
        brush size visually; we just need to tell it where the center is.
        """
        if self.selected_camera is None or self.context_matrix is None:
            return

        px, py = int(scene_pos.x()), int(scene_pos.y())
        visible_paths = self._get_visible_context_paths()

        # Use the blazingly fast center-point projection
        projections = self._build_projection(px, py)
        self.context_matrix.update_cursor_previews(projections, visible_paths, item_factory)

    def _on_cursor_preview_cleared(self):
        """Clear cursor previews from all context canvases."""
        if self.context_matrix is not None:
            self.context_matrix.clear_all_cursor_previews()

    def _get_context_canvas_for_path(self, image_path: str):
        """Return the context canvas currently displaying image_path, or None."""
        if self.context_matrix is None:
            return None
        for row in self.context_matrix._visible_canvases:
            for canvas in row:
                if canvas is not None and canvas.current_image_path == image_path:
                    return canvas
        return None

    def _get_visible_context_paths(self) -> set:
        """Return the set of image paths currently visible in the context matrix."""
        paths = set()
        if self.context_matrix is None:
            return paths
        for row in self.context_matrix._visible_canvases:
            for canvas in row:
                if canvas and canvas.active_image and canvas.current_image_path:
                    paths.add(canvas.current_image_path)
        return paths

    def _build_projection(self, px: int, py: int) -> dict:
        """Cast a ray from the selected camera at (px, py) and return projections.

        Returns:
            dict mapping image_path -> (u, v, is_valid), or empty dict on failure.
        """
        if self.selected_camera is None:
            return {}

        depth = None
        try:
            raster = self.selected_camera._raster
            if raster.z_channel is not None and raster.z_data_type == 'depth':
                depth = raster.get_z_value(px, py)
        except Exception:
            pass

        try:
            default_depth = self.viewer.get_scene_median_depth(self.selected_camera.position)
        except Exception:
            default_depth = 10.0
        if not default_depth or default_depth <= 0:
            default_depth = 10.0

        try:
            ray = CameraRay.from_pixel_and_camera(
                pixel_xy=(px, py),
                camera=self.selected_camera,
                depth=depth,
                default_depth=default_depth,
            )
            return ray.project_to_cameras(self.cameras)
        except Exception:
            return {}

    def _on_patch_annotation_created(self, annotation_id: str):
        """Propagate a newly created PatchAnnotation into all visible context cameras."""
        if self._propagating_annotation:
            return

        annotation = self.annotation_window.annotations_dict.get(annotation_id)
        if annotation is None or not isinstance(annotation, PatchAnnotation):
            return
        if self.selected_camera is None:
            return
        if annotation.image_path != self.selected_camera.image_path:
            return

        px = int(annotation.center_xy.x())
        py = int(annotation.center_xy.y())

        selected_paths = set(self.selection_model.selected_paths)
        if self.selected_camera and self.selected_camera.image_path in selected_paths:
            selected_paths.discard(self.selected_camera.image_path)

        # Quick exit: nothing to propagate to
        if not selected_paths:
            return

        from PyQt5.QtCore import QPointF
        self._propagating_annotation = True
        try:
            # Try True 3D mapping: get the element id at source pixel (if index map present)
            # Use the in-memory raster index_map to avoid triggering lazy disk loads.
            source_index_map = self.selected_camera._raster.index_map
            element_id = None
            use_3d = False
            if source_index_map is not None:
                try:
                    img_h, img_w = source_index_map.shape
                    if 0 <= px < img_w and 0 <= py < img_h:
                        eid = int(source_index_map[py, px])
                        if eid > -1:
                            element_id = eid
                            use_3d = True
                except Exception:
                    pass

            # Lazy projection cache for fallback
            projections = None

            for target_path in selected_paths:

                target_camera = self.cameras.get(target_path)
                if target_camera is None:
                    continue

                try:
                    # 3D centroid mapping if both source hit and target index map exist
                    target_has_index = getattr(target_camera, '_raster', None) is not None and target_camera._raster.index_map is not None
                    if use_3d and target_has_index and element_id is not None:
                        flat = target_camera.get_pixels_for_elements(np.array([element_id], dtype=np.int64))
                        if flat.size == 0:
                            # occluded in this view
                            continue
                        v_arr, u_arr = np.divmod(flat, target_camera.width)
                        u_centroid = float(np.mean(u_arr))
                        v_centroid = float(np.mean(v_arr))
                        if not (0 <= u_centroid < target_camera.width and 0 <= v_centroid < target_camera.height):
                            continue

                        new_annotation = PatchAnnotation(
                            center_xy=QPointF(u_centroid, v_centroid),
                            annotation_size=annotation.annotation_size,
                            short_label_code=annotation.label.short_label_code,
                            long_label_code=annotation.label.long_label_code,
                            color=annotation.label.color,
                            image_path=target_path,
                            label_id=annotation.label.id,
                            transparency=annotation.transparency,
                        )
                        try:
                            self.annotation_window.add_annotation(new_annotation, record_action=True)
                        except Exception:
                            pass
                    else:
                        # 2D center-stamp fallback: per-target projection (lazy)
                        if projections is None:
                            projections = self._build_projection(px, py)
                        proj = projections.get(target_path)
                        if proj is None:
                            continue
                        u, v, is_valid = proj
                        if not is_valid:
                            continue
                        if not (0 <= u < target_camera.width and 0 <= v < target_camera.height):
                            continue

                        new_annotation = PatchAnnotation(
                            center_xy=QPointF(u, v),
                            annotation_size=annotation.annotation_size,
                            short_label_code=annotation.label.short_label_code,
                            long_label_code=annotation.label.long_label_code,
                            color=annotation.label.color,
                            image_path=target_path,
                            label_id=annotation.label.id,
                            transparency=annotation.transparency,
                        )
                        try:
                            self.annotation_window.add_annotation(new_annotation, record_action=True)
                        except Exception:
                            pass
                except Exception:
                    pass
        finally:
            self._propagating_annotation = False

    # TODO Note: dense mesh hit fills in the face IDs when the quality of index map < Highest; otherwise VTK does this fine.
    # If we can find a way to not use Open3D always, then we don't need to calculate a BVH, which takes times to build.
    # Figure out how we can allow the user to use lower quality index maps, but still fill in the gaps.
    def _dense_mesh_hit_test(self, source_camera, pixel_mask: np.ndarray, px: int, py: int, mesh_product) -> np.ndarray:
        """Cast rays through every True pixel in pixel_mask against the mesh surface.

        Unlike the index_map approach (which captures face IDs from a downsampled
        raycasting pass), this method casts a ray through every individual painted
        pixel at full resolution, intersecting the actual triangle surface area of
        the mesh.  This guarantees that every triangle touched by the brush or SAM
        mask contributes its face ID to the output set, regardless of its projected
        pixel size.

        The Open3D RaycastingScene BVH is built once per mesh product and cached on
        the product object to amortise the cost across many brush strokes.

        Args:
            source_camera: Perspective Camera for the selected image.
            pixel_mask: (H, W) bool/uint8 array; True pixels are ray-cast targets.
            px: X coordinate of the mask centre in source image space.
            py: Y coordinate of the mask centre in source image space.
            mesh_product: MeshProduct whose cached geometry is used.

        Returns:
            np.ndarray[int32]: Unique face IDs that were hit, or empty array on
            failure, missing Open3D, or orthographic source camera.
        """
        # Orthographic cameras use an affine projection model without K / R —
        # pinhole ray casting is not applicable; let the caller fall back to
        # the index_map path instead.
        if getattr(source_camera, 'is_orthographic', False):
            return np.array([], dtype=np.int32)

        try:
            import open3d as o3d
        except ImportError:
            return np.array([], dtype=np.int32)

        try:
            # 1. Ensure the GPU tensor geometry cache exists (idempotent call).
            mesh_product.prepare_geometry()
            vertices  = mesh_product._cached_vertices                                      # (V, 3) float32
            triangles = mesh_product._cached_triangles_pt.cpu().numpy().astype(np.uint32)  # (T, 3) uint32

            if len(triangles) == 0:
                return np.array([], dtype=np.int32)

            # 2. Build (or reuse) the Open3D RaycastingScene BVH.
            #    Mesh vertex/face topology never changes during annotation, so the
            #    cached scene remains valid for the entire session.
            if not getattr(mesh_product, '_o3d_raycasting_scene', None):
                scene = o3d.t.geometry.RaycastingScene()
                scene.add_triangles(
                    o3d.core.Tensor(vertices,   dtype=o3d.core.Dtype.Float32),
                    o3d.core.Tensor(triangles,  dtype=o3d.core.Dtype.UInt32),
                )
                mesh_product._o3d_raycasting_scene = scene
            scene = mesh_product._o3d_raycasting_scene

            # 3. Map the True pixels in pixel_mask to source image coordinates.
            mask_h, mask_w = pixel_mask.shape
            x0 = px - mask_w // 2
            y0 = py - mask_h // 2

            ys, xs = np.where(pixel_mask.astype(bool))
            if len(xs) == 0:
                return np.array([], dtype=np.int32)

            u_img = (xs + x0).astype(np.float32)
            v_img = (ys + y0).astype(np.float32)

            # Discard pixels that fall outside the image frame.
            valid = (
                (u_img >= 0) & (u_img < source_camera.width) &
                (v_img >= 0) & (v_img < source_camera.height)
            )
            u_img = u_img[valid]
            v_img = v_img[valid]

            if len(u_img) == 0:
                return np.array([], dtype=np.int32)

            # 4. Unproject pixels to world-space ray directions.
            #    Pinhole camera model (row-vector convention):
            #      d_cam   = K_inv @ [u, v, 1]^T   →   d_cam_row   = [u,v,1] @ K_inv.T
            #      d_world = R.T   @ d_cam          →   d_world_row = d_cam_row  @ R
            ones        = np.ones(len(u_img), dtype=np.float32)
            pixel_homog = np.stack([u_img, v_img, ones], axis=1)     # (N, 3)
            K_inv       = source_camera.K_inv.astype(np.float32)     # (3, 3)
            R           = source_camera.R.astype(np.float32)         # (3, 3)

            dirs_cam   = pixel_homog @ K_inv.T    # (N, 3) camera-space directions
            dirs_world = dirs_cam   @ R            # (N, 3) world-space directions

            norms = np.linalg.norm(dirs_world, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            dirs_world /= norms

            # 5. Build and cast the ray batch against the BVH.
            cam_origin = source_camera.position.astype(np.float32)          # (3,)
            origins    = np.tile(cam_origin, (len(u_img), 1))               # (N, 3)
            rays_np    = np.concatenate([origins, dirs_world], axis=1)      # (N, 6)

            ans = scene.cast_rays(o3d.core.Tensor(rays_np, dtype=o3d.core.Dtype.Float32))

            # 6. Extract hit triangle indices (Open3D uses uint32 max = miss).
            INVALID_O3D = np.uint32(4294967295)
            prim_ids     = ans['primitive_ids'].numpy()
            hit_prim_ids = prim_ids[prim_ids != INVALID_O3D].astype(np.int64)

            if len(hit_prim_ids) == 0:
                return np.array([], dtype=np.int32)

            # 7. Remap sub-triangle primitive IDs to original PyVista cell IDs
            #    when the mesh was triangulated from non-triangular faces during
            #    prepare_geometry().
            original_cell_ids = getattr(mesh_product, '_original_cell_ids', None)
            if original_cell_ids is not None:
                in_range     = hit_prim_ids < len(original_cell_ids)
                hit_prim_ids = hit_prim_ids[in_range]
                face_ids     = original_cell_ids[hit_prim_ids].astype(np.int32)
            else:
                face_ids = hit_prim_ids.astype(np.int32)

            return np.unique(face_ids)

        except Exception as e:
            print(f"⚠️ Dense mesh hit test failed: {e}")
            return np.array([], dtype=np.int32)

    def _on_brush_stroke_applied(self, scene_pos, label_id: str, brush_mask):
        """Propagate a brush stroke into all visible context cameras.

        Uses True 3D Mapping when the source camera's index map is available:
        1. Extract the element IDs painted under the brush using the source
           camera's index_map.
        2. Update the 3D Scene Product directly with the new Class ID and Color.
        3. Query each target camera's inverted index for the same IDs.
        4. Paint exactly those pixels with update_mask_at_indices().

        Falls back to the legacy 2D center-stamp when the index map is absent
        (e.g., visibility not yet computed) or when no scene geometry was hit.
        """
        
        if self._propagating_annotation:
            return
        if self.selected_camera is None:
            return

        px = int(scene_pos.x())
        py = int(scene_pos.y())

        selected_paths = set(self.selection_model.selected_paths)
        if self.selected_camera and self.selected_camera.image_path in selected_paths:
            selected_paths.discard(self.selected_camera.image_path)
        project_labels = list(self.main_window.label_window.labels)

        # Quick exit: nothing to propagate to
        if not selected_paths:
            return

        source_label = next((lbl for lbl in project_labels if lbl.id == label_id), None)
        if source_label is None:
            return

        brush_h, brush_w = brush_mask.shape

        # ------------------------------------------------------------------
        # Phase 1: Source ID Extraction (2D → 3D)
        # ------------------------------------------------------------------
        painted_ids = None
        _p1_target = self.viewer.scene_context.get_primary_target()
        if isinstance(_p1_target, MeshProduct) and not getattr(self.selected_camera, 'is_orthographic', False):
            try:
                # Dense ray casting: cast through every True pixel in the brush mask
                # to intersect the full triangle surface area rather than relying on
                # the downsampled index_map (which only captures face centres at reduced
                # resolution, missing small or oblique triangles).
                painted_ids = self._dense_mesh_hit_test(
                    self.selected_camera, brush_mask, px, py, _p1_target
                )
                if painted_ids is None:
                    painted_ids = np.array([], dtype=np.int32)
            except Exception as e:
                print(f"⚠️ Dense mesh hit test failed: {e}. Falling back to index_map.")
                painted_ids = None
        
        # If mesh hit test didn't work, try index_map fallback
        if painted_ids is None:
            # PointCloud (or non-mesh) target: use the pre-computed index_map.
            # Use the in-memory raster index_map to avoid triggering lazy disk loads.
            source_index_map = self.selected_camera._raster.index_map
            if source_index_map is not None:
                # Bounding box of the brush in source image coordinates
                x0 = px - brush_w // 2
                y0 = py - brush_h // 2
                x1 = x0 + brush_w
                y1 = y0 + brush_h

                img_h, img_w = source_index_map.shape

                # Only proceed if the brush overlaps the image at all
                if x0 < img_w and y0 < img_h and x1 > 0 and y1 > 0:
                    # Clip to image bounds
                    cx0 = max(x0, 0)
                    cy0 = max(y0, 0)
                    cx1 = min(x1, img_w)
                    cy1 = min(y1, img_h)

                    # Corresponding slice of the brush mask
                    bx0 = cx0 - x0
                    by0 = cy0 - y0
                    bx1 = bx0 + (cx1 - cx0)
                    by1 = by0 + (cy1 - cy0)

                    index_slice = source_index_map[cy0:cy1, cx0:cx1]
                    brush_clip  = brush_mask[by0:by1, bx0:bx1]

                    # Extract the 3D Face IDs perfectly using the VTK raster
                    raw_ids = index_slice[brush_clip.astype(bool)]
                    unique_ids = np.unique(raw_ids)
                    painted_ids = unique_ids[unique_ids > -1]  # filter background

        # Whether the source camera has valid 3D geometry hits to propagate
        use_3d = painted_ids is not None and len(painted_ids) > 0 

        # ------------------------------------------------------------------
        # Paint the 3D Model directly
        # ------------------------------------------------------------------
        if use_3d:  # Slow
            primary_target = self.viewer.scene_context.get_primary_target()
            if primary_target and hasattr(primary_target, 'apply_labels'):
                # 1. Convert QColor to RGB tuple
                target_color = (source_label.color.red(), source_label.color.green(), source_label.color.blue())
                
                # 2. Get the integer class_id mapped to this label from the source mask
                source_raster = self.raster_manager.get_raster(self.selected_camera.image_path)
                source_mask = source_raster.get_mask_annotation(project_labels) if source_raster else None
                
                if source_mask:
                    source_class_id = source_mask.label_id_to_class_id_map.get(label_id)
                    # Sync if it's a brand new label
                    if source_class_id is None:
                        source_mask.sync_label_map([source_label])
                        source_class_id = source_mask.label_id_to_class_id_map.get(label_id)
                        
                    # 3. Paint the 3D model arrays
                    primary_target.apply_labels(painted_ids, source_class_id, target_color)
                    
                    # 4. Tell the 3D viewer to refresh to show the new colors instantly
                    # self._request_viewer_update()  # Still slow with commeneted out

        # Projections for 2D fallback — computed lazily inside the loop
        projections = None

        self._propagating_annotation = True
        try:
            loop_start = time.time()
            for target_path in selected_paths:

                target_camera = self.cameras.get(target_path)
                if target_camera is None:
                    continue

                try:
                    target_raster = self.raster_manager.get_raster(target_path)
                    if target_raster is None:
                        continue
                    
                    # Access the mask directly to bypass forced UI syncing overhead
                    target_mask = target_raster.mask_annotation
                    if target_mask is None:
                        target_mask = target_raster.get_mask_annotation(project_labels)
                    if target_mask is None:
                        continue

                    # Resolve target class_id via stable label_id
                    target_class_id = target_mask.label_id_to_class_id_map.get(label_id)
                    if target_class_id is None:
                        target_mask.sync_label_map([source_label])
                        target_class_id = target_mask.label_id_to_class_id_map.get(label_id)
                    if target_class_id is None:
                        continue

                    target_has_index = target_camera._raster.index_map is not None

                    if use_3d and target_has_index:  # Not slow
                        # --------------------------------------------------
                        # Phase 2: Target Pixel Injection (3D → 2D)
                        # --------------------------------------------------
                        try:
                            if not isinstance(painted_ids, np.ndarray) or len(painted_ids) == 0:
                                continue
                           
                            lookup_start = time.time()
                            
                            # --- OPTIMIZATION 3: Localized Search Window ---
                            # Project the center of the brush to the target camera
                            if projections is None:
                                projections = self._build_projection(px, py)
                                
                            proj = projections.get(target_path)
                            bbox = None
                            
                            # If the center of the brush successfully projected into this camera,
                            # restrict the search to a local window to prevent scanning 16M pixels.
                            if proj is not None and proj[2]: # is_valid
                                target_u, target_v = proj[0], proj[1]
                                
                                # Use a generous radius (2.5x the brush size) to safely capture
                                # the painted area even under extreme perspective distortion.
                                search_radius = max(brush_w, brush_h) * 2.5
                                bbox = (
                                    target_u - search_radius, 
                                    target_u + search_radius, 
                                    target_v - search_radius, 
                                    target_v + search_radius
                                )
                            
                            # Pass the bbox to the camera to slice the index map
                            flat_indices = target_camera.get_pixels_for_elements(painted_ids, bbox=bbox)
                            
                            if len(flat_indices) == 0:
                                continue
                            
                            # --- OPTIMIZATION 2: Pixel Diffing ---
                            if hasattr(target_mask, 'mask_data'):
                                current_vals = target_mask.mask_data.ravel()[flat_indices]
                                changed_mask = current_vals != target_class_id
                                flat_indices = flat_indices[changed_mask]
                                
                                if len(flat_indices) == 0:
                                    continue
                                                        
                            # Revert to 1D index updating. Pixel-diffing (above) already 
                            # ensures this list is small enough to not choke the system, 
                            # and guarantees no square bounding-box artifacts.
                            target_mask.update_mask_at_indices(flat_indices, target_class_id, silent=True)
                            
                        except Exception as e:
                            print(f"⚠️ Failed to propagate stroke to {target_path}: {e}")
                            continue
                    else:
                        # 2D center-stamp fallback
                        if projections is None:
                            projections = self._build_projection(px, py)
                        proj = projections.get(target_path)
                        if proj is None:
                            continue
                        u, v, is_valid = proj
                        if not is_valid:
                            continue
                        if not (0 <= u < target_camera.width and 0 <= v < target_camera.height):
                            continue
                        from PyQt5.QtCore import QPointF
                        brush_location = QPointF(u - brush_w / 2.0, v - brush_h / 2.0)
                        target_mask.update_mask(brush_location, brush_mask, target_class_id, silent=True)

                    # Ensure the label is visible in the target overlay
                    if label_id not in target_mask.visible_label_ids:
                        target_mask.visible_label_ids.add(label_id)
                        target_mask.update_graphics_item()

                    # Only mount the overlay if it isn't already attached to the canvas
                    context_canvas = self._get_context_canvas_for_path(target_path)
                    if context_canvas is not None and context_canvas._mask_overlay_item is None:
                        context_canvas.set_mask_overlay(target_mask)
                except Exception:
                    pass
        finally:
            self._propagating_annotation = False

    def _on_fill_stroke_applied(self, scene_pos, label_id: str, fill_mask=None):
        """Propagate a fill operation into all visible context cameras.
        
        Uses True 3D Mapping when the source camera's index map is available:
        1. Extract all element IDs under the filled region using the source
           camera's index_map and the fill_mask.
        2. Update the 3D Scene Product directly with the new Class ID and Color.
        3. Query each target camera's inverted index for the same IDs.
        4. Fill the connected regions in the target masks.
        
        Falls back to the legacy 2D center-point projection when the index map is absent.
        """
        if self._propagating_annotation:
            return
        if self.selected_camera is None:
            return
        
        px = int(scene_pos.x())
        py = int(scene_pos.y())
        
        selected_paths = set(self.selection_model.selected_paths)
        if self.selected_camera and self.selected_camera.image_path in selected_paths:
            selected_paths.discard(self.selected_camera.image_path)
        project_labels = list(self.main_window.label_window.labels)
        
        source_label = next((lbl for lbl in project_labels if lbl.id == label_id), None)
        if source_label is None:
            return
        
        # ------------------------------------------------------------------
        # Phase 1: Source ID Extraction (2D → 3D)
        # ------------------------------------------------------------------
        painted_ids = None
        _p1_target = self.viewer.scene_context.get_primary_target()
        if fill_mask is not None and isinstance(_p1_target, MeshProduct) and not getattr(self.selected_camera, 'is_orthographic', False):
            # Dense ray casting: fill_mask is full image-sized, so pass center coords
            # that produce a zero offset (x0=0, y0=0) aligning the mask to image space.
            mask_h_fill, mask_w_fill = fill_mask.shape
            painted_ids = self._dense_mesh_hit_test(
                self.selected_camera, fill_mask.astype(bool),
                mask_w_fill // 2, mask_h_fill // 2, _p1_target
            )
        else:
            # PointCloud (or non-mesh) target: use the pre-computed index_map.
            # Use the in-memory raster index_map to avoid triggering lazy disk loads.
            source_index_map = self.selected_camera._raster.index_map
            if source_index_map is not None and fill_mask is not None:
                try:
                    img_h, img_w = source_index_map.shape
                    # Get all element IDs where fill_mask is True
                    if fill_mask.shape == (img_h, img_w):
                        element_ids = source_index_map[fill_mask]
                        # Get unique element IDs (excluding -1 which means background)
                        painted_ids = np.unique(element_ids[element_ids > -1])
                        painted_ids = np.array(painted_ids, dtype=np.int64)
                except Exception:
                    pass

        use_3d = painted_ids is not None and len(painted_ids) > 0
        
        # ------------------------------------------------------------------
        # NEW Phase: Paint the 3D Model directly
        # ------------------------------------------------------------------
        if use_3d:
            primary_target = self.viewer.scene_context.get_primary_target()
            if primary_target and hasattr(primary_target, 'apply_labels'):
                # 1. Convert QColor to RGB tuple
                target_color = (source_label.color.red(), source_label.color.green(), source_label.color.blue())
                
                # 2. Get the integer class_id mapped to this label from the source mask
                source_raster = self.raster_manager.get_raster(self.selected_camera.image_path)
                source_mask = source_raster.get_mask_annotation(project_labels) if source_raster else None
                
                if source_mask:
                    source_class_id = source_mask.label_id_to_class_id_map.get(label_id)
                    # Sync if it's a brand new label
                    if source_class_id is None:
                        source_mask.sync_label_map([source_label])
                        source_class_id = source_mask.label_id_to_class_id_map.get(label_id)
                    
                    # 3. Paint the 3D model arrays
                    primary_target.apply_labels(painted_ids, source_class_id, target_color)
                    
                    # 4. Tell the 3D viewer to refresh to show the new colors instantly
                    self._request_viewer_update()
        
        # Projections for 2D fallback — computed lazily
        projections = None
        
        self._propagating_annotation = True
        try:
            for target_path in selected_paths:
                
                target_camera = self.cameras.get(target_path)
                if target_camera is None:
                    continue
                
                try:
                    target_raster = self.raster_manager.get_raster(target_path)
                    if target_raster is None:
                        continue
                        
                    # --- OPTIMIZATION 1: Bypass forced sync_label_map ---
                    target_mask = target_raster.mask_annotation
                    if target_mask is None:
                        target_mask = target_raster.get_mask_annotation(project_labels)
                    if target_mask is None:
                        continue
                    
                    target_class_id = target_mask.label_id_to_class_id_map.get(label_id)
                    if target_class_id is None:
                        target_mask.sync_label_map([source_label])
                        target_class_id = target_mask.label_id_to_class_id_map.get(label_id)
                    if target_class_id is None:
                        continue
                    
                    target_has_index = target_camera._raster is not None and target_camera._raster.index_map is not None
                    
                    if use_3d and target_has_index:
                        try:
                            if not isinstance(painted_ids, np.ndarray) or len(painted_ids) == 0:
                                continue
                                
                            # Fill can span the whole image, so we don't pass a bbox. 
                            # The Strided Pre-Search in Camera.py will handle bounds dynamically!
                            flat_indices = target_camera.get_pixels_for_elements(painted_ids)
                            if len(flat_indices) == 0:
                                continue
                            
                            # --- OPTIMIZATION 2: Pixel Diffing ---
                            if hasattr(target_mask, 'mask_data'):
                                current_vals = target_mask.mask_data.ravel()[flat_indices]
                                changed_mask = current_vals != target_class_id
                                flat_indices = flat_indices[changed_mask]
                                
                                if len(flat_indices) == 0:
                                    continue
                            
                            target_mask.update_mask_at_indices(flat_indices, target_class_id, silent=True)
                        except Exception as e:
                            print(f"⚠️ Failed to propagate fill to {target_path}: {e}")
                            continue
                    else:
                        # 2D fallback
                        if projections is None:
                            projections = self._build_projection(px, py)
                        proj = projections.get(target_path)
                        if proj is None:
                            continue
                        u, v, is_valid = proj
                        if not is_valid:
                            continue
                        if not (0 <= u < target_camera.width and 0 <= v < target_camera.height):
                            continue
                        from PyQt5.QtCore import QPointF
                        fill_pos = QPointF(u, v)
                        target_mask.fill_region(fill_pos, target_class_id)
                    
                    if label_id not in target_mask.visible_label_ids:
                        target_mask.visible_label_ids.add(label_id)
                        target_mask.update_graphics_item()
                    
                    target_mask.sync_label_map()
                    
                    context_canvas = self._get_context_canvas_for_path(target_path)
                    if context_canvas is not None and context_canvas._mask_overlay_item is None:
                        context_canvas.set_mask_overlay(target_mask)
                except Exception:
                    pass
        finally:
            self._propagating_annotation = False

    def _on_erase_stroke_applied(self, scene_pos, label_id: str, brush_mask: np.ndarray):
        """Propagate an erase operation into all visible context cameras.

        Uses True 3D Mapping when the source camera's index map is available:
        1. Extract the element IDs beneath the eraser brush using the source
           camera's index_map.
        2. Reset those elements on the 3D Scene Product to class_id=0 (white).
        3. Query each target camera's inverted index for the same IDs.
        4. Erase exactly those pixels with update_mask_at_indices(flat, 0).

        Falls back to the legacy 2D center-stamp when the index map is absent.
        """
        if self._propagating_annotation:
            return
        if self.selected_camera is None or self.context_matrix is None:
            return

        px, py = int(scene_pos.x()), int(scene_pos.y())
        selected_paths = set(self.selection_model.selected_paths)
        if self.selected_camera and self.selected_camera.image_path in selected_paths:
            selected_paths.discard(self.selected_camera.image_path)

        # Quick exit: nothing to propagate to
        if not selected_paths:
            return

        brush_h, brush_w = brush_mask.shape

        # ------------------------------------------------------------------
        # Phase 1: Source ID Extraction (2D → 3D)
        # ------------------------------------------------------------------
        painted_ids = None
        _p1_target = self.viewer.scene_context.get_primary_target()
        if isinstance(_p1_target, MeshProduct) and not getattr(self.selected_camera, 'is_orthographic', False):
            # Dense ray casting: cast through every True pixel in the eraser mask
            # to intersect the full triangle surface area, matching the brush approach.
            painted_ids = self._dense_mesh_hit_test(
                self.selected_camera, brush_mask, px, py, _p1_target
            )
        else:
            # PointCloud (or non-mesh) target: use the pre-computed index_map.
            # Use the in-memory raster index_map to avoid triggering lazy disk loads.
            source_index_map = self.selected_camera._raster.index_map
            if source_index_map is not None:
                x0 = px - brush_w // 2
                y0 = py - brush_h // 2
                x1 = x0 + brush_w
                y1 = y0 + brush_h

                img_h, img_w = source_index_map.shape

                if x0 < img_w and y0 < img_h and x1 > 0 and y1 > 0:
                    cx0 = max(x0, 0)
                    cy0 = max(y0, 0)
                    cx1 = min(x1, img_w)
                    cy1 = min(y1, img_h)

                    bx0 = cx0 - x0
                    by0 = cy0 - y0
                    bx1 = bx0 + (cx1 - cx0)
                    by1 = by0 + (cy1 - cy0)

                    index_slice = source_index_map[cy0:cy1, cx0:cx1]
                    brush_clip = brush_mask[by0:by1, bx0:bx1]

                    raw_ids = index_slice[brush_clip.astype(bool)]
                    unique_ids = np.unique(raw_ids)
                    painted_ids = unique_ids[unique_ids > -1]

        use_3d = painted_ids is not None and len(painted_ids) > 0

        # ------------------------------------------------------------------
        # Phase 2: Reset the 3D Model to default (white / class_id 0)
        # ------------------------------------------------------------------
        if use_3d:
            primary_target = self.viewer.scene_context.get_primary_target()
            if primary_target and hasattr(primary_target, 'apply_labels'):
                primary_target.apply_labels(painted_ids, 0, (255, 255, 255))
                try:
                    self.viewer.update()
                except Exception:
                    pass

        # Projections for 2D fallback — computed lazily inside the loop
        projections = None

        self._propagating_annotation = True
        try:
            for target_path in selected_paths:

                target_camera = self.cameras.get(target_path)
                if target_camera is None:
                    continue

                try:
                    target_raster = self.raster_manager.get_raster(target_path)
                    if target_raster is None:
                        continue
                        
                    # --- OPTIMIZATION 1: Bypass forced sync_label_map ---
                    target_mask = target_raster.mask_annotation
                    if target_mask is None:
                        target_mask = target_raster.get_mask_annotation(self.main_window.label_window.labels)
                    if target_mask is None:
                        continue

                    target_has_index = target_camera._raster.index_map is not None

                    if use_3d and target_has_index:
                        try:
                            if not isinstance(painted_ids, np.ndarray) or len(painted_ids) == 0:
                                continue
                                
                            # --- OPTIMIZATION 3: Localized Search Window ---
                            if projections is None:
                                projections = self._build_projection(px, py)
                            proj = projections.get(target_path)
                            bbox = None
                            
                            if proj is not None and proj[2]:
                                target_u, target_v = proj[0], proj[1]
                                search_radius = max(brush_w, brush_h) * 2.5
                                bbox = (
                                    target_u - search_radius, target_u + search_radius, 
                                    target_v - search_radius, target_v + search_radius
                                )
                                
                            flat_indices = target_camera.get_pixels_for_elements(painted_ids, bbox=bbox)
                            if len(flat_indices) == 0:
                                continue
                                
                            # --- OPTIMIZATION 2: Pixel Diffing (Compare to 0 for erase) ---
                            if hasattr(target_mask, 'mask_data'):
                                current_vals = target_mask.mask_data.ravel()[flat_indices]
                                changed_mask = current_vals != 0
                                flat_indices = flat_indices[changed_mask]
                                
                                if len(flat_indices) == 0:
                                    continue
                                    
                            target_mask.update_mask_at_indices(flat_indices, 0, silent=True)
                        except Exception as e:
                            print(f"⚠️ Failed to propagate erase to {target_path}: {e}")
                            continue
                    else:
                        # 2D center-stamp fallback
                        if projections is None:
                            projections = self._build_projection(px, py)
                        proj = projections.get(target_path)
                        if proj is None:
                            continue
                        u, v, is_valid = proj
                        if not is_valid:
                            continue
                        if not (0 <= u < target_camera.width and 0 <= v < target_camera.height):
                            continue
                        from PyQt5.QtCore import QPointF
                        brush_location = QPointF(u - brush_w / 2.0, v - brush_h / 2.0)
                        target_mask.update_mask(brush_location, brush_mask, 0, silent=True)

                    context_canvas = self._get_context_canvas_for_path(target_path)
                    if context_canvas is not None and context_canvas._mask_overlay_item is None:
                        context_canvas.set_mask_overlay(target_mask)

                except Exception:
                    pass
        finally:
            self._propagating_annotation = False

    def _on_sam_prediction_applied(self, scene_pos, label_id: str, binary_mask: np.ndarray):
        """Propagate a final SAM mask prediction into all visible context cameras.

        Uses True 3D Mapping when the source camera's index map is available:
        1. Extract the element IDs beneath the predicted binary_mask using the
           source camera's index_map (the mask is a crop centred at scene_pos).
        2. Update the 3D Scene Product directly with the new Class ID and Color.
        3. Query each target camera's inverted index for the same IDs.
        4. Paint exactly those pixels with update_mask_at_indices().

        Falls back to the legacy 2D stamp when the index map is absent or when
        no scene geometry was hit (e.g., sky prediction).

        Args:
            scene_pos: QPointF — centre of the prediction crop in source image pixels.
            label_id: UUID of the label used for the prediction.
            binary_mask: small (H,W) uint8 array with 1 for predicted pixels.
        """
        if self._propagating_annotation:
            return
        if self.selected_camera is None:
            return

        px = int(scene_pos.x())
        py = int(scene_pos.y())

        selected_paths = set(self.selection_model.selected_paths)
        if self.selected_camera and self.selected_camera.image_path in selected_paths:
            selected_paths.discard(self.selected_camera.image_path)
        project_labels = list(self.main_window.label_window.labels)

        source_label = next((lbl for lbl in project_labels if lbl.id == label_id), None)
        if source_label is None:
            return

        mask_h, mask_w = binary_mask.shape

        # ------------------------------------------------------------------
        # Phase 1: Source ID Extraction (2D → 3D)
        # ------------------------------------------------------------------
        painted_ids = None
        _p1_target = self.viewer.scene_context.get_primary_target()
        if isinstance(_p1_target, MeshProduct) and not getattr(self.selected_camera, 'is_orthographic', False):
            # Dense ray casting: cast through every True pixel in the SAM binary_mask
            # to intersect the full triangle surface area rather than relying on
            # the downsampled index_map (which only captures face centres at reduced
            # resolution, missing small or oblique triangles).
            painted_ids = self._dense_mesh_hit_test(
                self.selected_camera, binary_mask.astype(bool), px, py, _p1_target
            )
        else:
            # PointCloud (or non-mesh) target: use the pre-computed index_map.
            # Use the in-memory raster index_map to avoid triggering lazy disk loads.
            source_index_map = self.selected_camera._raster.index_map
            if source_index_map is not None:
                # The binary_mask is a crop centred at (px, py)
                x0 = px - mask_w // 2
                y0 = py - mask_h // 2
                x1 = x0 + mask_w
                y1 = y0 + mask_h

                img_h, img_w = source_index_map.shape

                if x0 < img_w and y0 < img_h and x1 > 0 and y1 > 0:
                    cx0 = max(x0, 0)
                    cy0 = max(y0, 0)
                    cx1 = min(x1, img_w)
                    cy1 = min(y1, img_h)

                    # Corresponding slice of the binary_mask
                    bx0 = cx0 - x0
                    by0 = cy0 - y0
                    bx1 = bx0 + (cx1 - cx0)
                    by1 = by0 + (cy1 - cy0)

                    index_slice = source_index_map[cy0:cy1, cx0:cx1]
                    mask_clip   = binary_mask[by0:by1, bx0:bx1]

                    raw_ids = index_slice[mask_clip.astype(bool)]
                    unique_ids = np.unique(raw_ids)
                    painted_ids = unique_ids[unique_ids > -1]

        use_3d = painted_ids is not None and len(painted_ids) > 0

        # ------------------------------------------------------------------
        # NEW Phase: Paint the 3D Model directly
        # ------------------------------------------------------------------
        if use_3d:
            primary_target = self.viewer.scene_context.get_primary_target()
            if primary_target and hasattr(primary_target, 'apply_labels'):
                # 1. Convert QColor to RGB tuple
                target_color = (source_label.color.red(), source_label.color.green(), source_label.color.blue())
                
                # 2. Get the integer class_id mapped to this label from the source mask
                source_raster = self.raster_manager.get_raster(self.selected_camera.image_path)
                source_mask = source_raster.get_mask_annotation(project_labels) if source_raster else None
                
                if source_mask:
                    source_class_id = source_mask.label_id_to_class_id_map.get(label_id)
                    if source_class_id is None:
                        source_mask.sync_label_map([source_label])
                        source_class_id = source_mask.label_id_to_class_id_map.get(label_id)
                        
                    # 3. Paint the 3D model arrays
                    primary_target.apply_labels(painted_ids, source_class_id, target_color)
                    
                    # 4. Tell the 3D viewer to refresh
                    self._request_viewer_update()

        # Projections for 2D fallback — computed lazily inside the loop
        projections = None

        self._propagating_annotation = True
        try:
            for target_path in selected_paths:

                target_camera = self.cameras.get(target_path)
                if target_camera is None:
                    continue

                try:
                    target_raster = self.raster_manager.get_raster(target_path)
                    if target_raster is None:
                        continue
                        
                    # --- OPTIMIZATION 1: Bypass forced sync_label_map ---
                    target_mask = target_raster.mask_annotation
                    if target_mask is None:
                        target_mask = target_raster.get_mask_annotation(project_labels)
                    if target_mask is None:
                        continue

                    target_class_id = target_mask.label_id_to_class_id_map.get(label_id)
                    if target_class_id is None:
                        target_mask.sync_label_map([source_label])
                        target_class_id = target_mask.label_id_to_class_id_map.get(label_id)
                    if target_class_id is None:
                        continue

                    target_has_index = target_camera._raster.index_map is not None

                    if use_3d and target_has_index:
                        try:
                            if not isinstance(painted_ids, np.ndarray) or len(painted_ids) == 0:
                                continue
                                
                            # --- OPTIMIZATION 3: Localized Search Window ---
                            if projections is None:
                                projections = self._build_projection(px, py)
                            proj = projections.get(target_path)
                            bbox = None
                            
                            if proj is not None and proj[2]:
                                target_u, target_v = proj[0], proj[1]
                                search_radius = max(mask_w, mask_h) * 2.5
                                bbox = (
                                    target_u - search_radius, target_u + search_radius, 
                                    target_v - search_radius, target_v + search_radius
                                )
                                
                            flat_indices = target_camera.get_pixels_for_elements(painted_ids, bbox=bbox)
                            if len(flat_indices) == 0:
                                continue
                                
                            # --- OPTIMIZATION 2: Pixel Diffing ---
                            if hasattr(target_mask, 'mask_data'):
                                current_vals = target_mask.mask_data.ravel()[flat_indices]
                                changed_mask = current_vals != target_class_id
                                flat_indices = flat_indices[changed_mask]
                                
                                if len(flat_indices) == 0:
                                    continue
                                    
                            target_mask.update_mask_at_indices(flat_indices, target_class_id, silent=True)
                        except Exception as e:
                            print(f"⚠️ Failed to propagate SAM mask to {target_path}: {e}")
                            continue
                    else:
                        # 2D center-stamp fallback
                        if projections is None:
                            projections = self._build_projection(px, py)
                        proj = projections.get(target_path)
                        if proj is None:
                            continue
                        u, v, is_valid = proj
                        if not is_valid:
                            continue
                        if not (0 <= u < target_camera.width and 0 <= v < target_camera.height):
                            continue
                        subset_class_mask = binary_mask.astype(np.uint8) * int(target_class_id)
                        top_left_x = int(u - mask_w / 2.0)
                        top_left_y = int(v - mask_h / 2.0)
                        target_mask.update_mask_with_mask(subset_class_mask, (top_left_x, top_left_y), silent=True)

                    if label_id not in target_mask.visible_label_ids:
                        target_mask.visible_label_ids.add(label_id)
                        target_mask.update_graphics_item()

                    context_canvas = self._get_context_canvas_for_path(target_path)
                    if context_canvas is not None and context_canvas._mask_overlay_item is None:
                        context_canvas.set_mask_overlay(target_mask)
                except Exception:
                    pass
        finally:
            self._propagating_annotation = False

    def cleanup(self):
        """Clean up resources before closing."""
        self._on_multi_annotate_toggled(False)  # Disconnect all propagation hooks
        self.mouse_bridge.cleanup()
        if hasattr(self.viewer, 'close'):
            self.viewer.close()