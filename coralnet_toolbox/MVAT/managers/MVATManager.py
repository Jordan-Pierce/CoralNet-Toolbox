"""
MultiView Annotation Tool (MVAT) Manager

The central controller for the MVAT workspace.
Handles the business logic, data synchronization, and signal routing between 
the MainWindow, RasterManager, MVATViewer (3D), and ContextMatrix (2D).
"""

import os
import time
import numpy as np
import traceback
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from PyQt5.QtCore import QObject, QTimer, pyqtSignal, Qt, QThread, QPointF
from PyQt5.QtWidgets import QApplication, QMessageBox

from coralnet_toolbox.MVAT.core.Camera import Camera
from coralnet_toolbox.MVAT.core.Ray import CameraRay

from coralnet_toolbox.MVAT.managers.SelectionManager import SelectionManager
from coralnet_toolbox.MVAT.managers.VisibilityManager import VisibilityManager
from coralnet_toolbox.MVAT.managers.VisibilityWorker import VisibilityWorker
from coralnet_toolbox.MVAT.managers.CacheManager import CacheManager
from coralnet_toolbox.MVAT.managers.LabelPainterThread import LabelPainterThread

from coralnet_toolbox.MVAT.core.constants import (
    MARKER_COLOR_SELECTED,
    MARKER_COLOR_INVALID,
    RAY_COLOR_SELECTED,
    RAY_COLOR_HIGHLIGHTED,
    RAY_COLOR_INVALID,
)

from coralnet_toolbox.MVAT.core.Model import MeshProduct

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation


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
        self._last_mouse_x = -1
        self._last_mouse_y = -1

    def on_mouse_moved(self, x: int, y: int):
        if not self.enabled:
            return
        # Skip duplicate pixels (Qt can fire multiple events for the same position)
        if x == self._last_mouse_x and y == self._last_mouse_y:
            return
        # Time-gate: cap at ~60 fps to avoid flooding the ortho ray-trace / perspective
        # projection loop on every raw mouse event.
        now = time.monotonic()
        if now - self._last_update_time < 0.016:
            return
        self._last_update_time = now
        self._last_mouse_x = x
        self._last_mouse_y = y
        self._process_pending_position(x, y)
            
    def _process_pending_position(self, x: int, y: int):
        # --- Ortho path: route to ortho handler when OrthoRaster is displayed ---
        ortho_camera = self.manager.ortho_camera
        if ortho_camera is not None:
            current_path = getattr(self.manager.annotation_window, 'current_image_path', None)
            if current_path == ortho_camera.image_path:
                self.manager.viewer.clear_ray()
                self._process_ortho_position(x, y, ortho_camera)
                return

        self.manager.viewer.clear_ortho_ray()

        camera = self.manager.selected_camera
        if camera is None or not (0 <= x < camera.width and 0 <= y < camera.height):
            self.clear_all_markers()
            self.manager.viewer.clear_ray()
            return

        visible_cameras = self.manager._get_visible_context_cameras()
        visible_paths = {cam.image_path for cam in visible_cameras}
        if camera.image_path not in visible_paths:
            self.clear_all_markers()
            self.manager.viewer.clear_ray()
            return

        primary_target = self.manager.viewer.scene_context.get_primary_target()

        # --- Path A: Index Map (preferred) ---
        ray = None
        candidate_id = -1
        index_map = camera._raster.index_map
        if index_map is not None:
            candidate_id = int(index_map[y, x])

        if candidate_id > -1 and primary_target is not None:
            coord = primary_target.get_element_coordinate(candidate_id)
            if coord is not None:
                # ---> Perspective logic <---
                origin = camera.position.copy()
                direction = coord - origin
                norm = np.linalg.norm(direction)
                direction = direction / norm if norm > 0 else camera.R.T @ np.array([0, 0, 1])
                
                ray = CameraRay(
                    origin=origin,
                    direction=direction,
                    terminal_point=coord,
                    has_accurate_depth=True,
                    pixel_coord=(x, y),
                    source_camera=camera,
                    element_id=candidate_id
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

        primary_ray_valid = ray.has_accurate_depth or ray.element_id > -1
        rays_with_colors = [(ray, RAY_COLOR_SELECTED if primary_ray_valid else RAY_COLOR_INVALID)]
        
        # --- Short-Circuit Invalid Primary Rays ---
        # If the primary ray did not hit real scene geometry, skip secondary rays entirely.
        if not primary_ray_valid:
            # Primary ray is invalid: keep only the selected-camera ray, clear markers, and return.
            self.manager.viewer.show_rays(rays_with_colors)
            self.clear_all_markers()
            return
        
        # --- Primary ray is valid: proceed with secondary rays ---
        visibility_status = {}
        accuracies = {camera.image_path: ray.has_accurate_depth}
        highlighted_cameras = visible_cameras

        for target_cam in highlighted_cameras:
            if target_cam.image_path == camera.image_path:
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
                accuracies[target_cam.image_path] = target_cam._raster.z_channel is not None

            visibility_status[target_cam.image_path] = is_occluded
            ray_color = RAY_COLOR_HIGHLIGHTED if (accuracies.get(target_cam.image_path, False) and not is_occluded) else RAY_COLOR_INVALID

            # Build secondary ray directly
            t_origin = target_cam.position.copy()
            t_direction = target_terminal - t_origin
            t_norm = np.linalg.norm(t_direction)
            t_direction = t_direction / t_norm if t_norm > 0 else target_cam.R.T @ np.array([0, 0, 1])
            is_ray_accurate = accuracies.get(target_cam.image_path, False) and not is_occluded
            target_ray = CameraRay(
                origin=t_origin,
                direction=t_direction,
                terminal_point=target_terminal,
                has_accurate_depth=is_ray_accurate,
                source_camera=target_cam,
                element_id=found_id,
            )
            rays_with_colors.append((target_ray, ray_color))

        self.manager.viewer.show_rays(rays_with_colors)
        # Project only into visible cameras — avoids O(all_cameras) work per frame
        visible_cam_dict = {cam.image_path: cam for cam in highlighted_cameras}
        visible_cam_dict[camera.image_path] = camera  # include the primary camera
        projections = ray.project_to_cameras(visible_cam_dict)

        # Update context matrix canvases (Phase 4)
        if self.manager.context_matrix is not None:
            try:
                self.manager.context_matrix.update_dynamic_markers(
                    projections, accuracies, visibility_status
                )
            except Exception:
                self.manager.context_matrix.clear_all_dynamic_markers()
                
    def _process_ortho_position(self, x: int, y: int, ortho_camera):
        """
        Handle mouse-move events when the AnnotationWindow is showing an OrthoRaster.

        Resolves the 3D world point via z-channel lookup (O(1)) instead of mesh
        ray tracing.  The z-channel is stored at full ortho resolution so pixel
        coords map directly; the raw CRS elevation is fed into geo_to_world.
        """

        if not (0 <= x < ortho_camera.width and 0 <= y < ortho_camera.height):
            self.manager.viewer.clear_ortho_ray()
            self.clear_all_markers()
            return

        # Geo XY from affine transform
        X, Y = ortho_camera.pixel_to_geo(x, y)

        # CRS elevation from z-channel (stored at full ortho resolution, raw units)
        Z = ortho_camera._raster.get_z_value(x, y)
        if Z is None:
            self.manager.viewer.clear_ortho_ray()
            self.clear_all_markers()
            return

        world_pt = ortho_camera.geo_to_world(X, Y, Z)

        try:
            self.manager.viewer.show_ortho_ray(
                world_pt,
                ortho_camera.get_vertical_direction_world(),
            )
        except Exception:
            pass

        # Project to all visible context cameras and build marker dicts
        projections = {}
        accuracies = {}
        visibility_status = {}

        for cam in self.manager._get_visible_context_cameras():
            try:
                proj = cam.project(world_pt)
                if np.isnan(proj).any():
                    continue
                u = float(proj[0])
                v = float(proj[1])
                in_bounds = 0 <= u < cam.width and 0 <= v < cam.height
                projections[cam.image_path] = (u, v, in_bounds)
                accuracies[cam.image_path] = True
                visibility_status[cam.image_path] = False
            except Exception:
                pass

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
    contextStatsComputed = pyqtSignal(int, str, int, int)
    _orthoIndexMapReady  = pyqtSignal(object)  # internal: carries result dict from worker thread
    _sam_repaint_signal  = pyqtSignal(list)    # internal: UI update tasks from background SAM worker
    
    def __init__(self, main_window, viewer):
        super().__init__()
        
        self.main_window = main_window
        self.raster_manager = main_window.image_window.raster_manager
        self.annotation_window = main_window.annotation_window
        self.image_window = main_window.image_window
        
        self.viewer = viewer
        try:
            self.viewer.mvat_manager = self
        except Exception:
            pass
        self.context_matrix = getattr(main_window, 'context_matrix', None)
        
        # State
        self.cameras = {}
        self.selected_camera = None
        self.highlighted_cameras = []
        self.hovered_camera = None
        self.current_focal_point = None
        self._context_view_path = None
        
        # Data Settings
        self.compute_depth_maps_enabled = True
        # New toggle: whether to compute index maps in background
        self.compute_index_maps_enabled = True
        # Scale factor for visibility map resolution (1.0 = native, 0.1 = lowest)
        self.visibility_scale_factor = 1.0
        # Safety flag to prevent concurrent visibility computations
        self._is_computing_visibility = False
        # Track active worker threads to prevent GC
        self._active_workers = []
        self._context_stats_request_id = 0
        self._latest_context_stats_request_id = 0

        # Multi-camera annotation state
        self.multi_annotate_enabled = False
        self._propagating_annotation = False

        # Ortho state: chunk transform T and OrthoCamera (set during load_cameras
        # when an OrthoRaster is present in the project)
        self._chunk_transform = None  # 4x4 Metashape chunk transform matrix
        self.ortho_camera = None      # OrthoCamera instance
        self._computing_ortho_index_map = False  # guard against concurrent builds

        # Internal Managers
        self.selection_model = SelectionManager(self)
        self.cache_manager = CacheManager("")
        self.mouse_bridge = MousePositionBridge(self)

        # Propagation thread pool for parallel camera updates
        self._propagation_executor = ThreadPoolExecutor(
            max_workers=min(8, os.cpu_count() or 4),
            thread_name_prefix='mvat_propagate'
        )

        # Single background thread for SAM propagation — keeps the main thread free
        # during Phase 1 (cv2 erode/distanceTransform) and mask array writes.
        self._sam_bg_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix='mvat_sam_bg'
        )
        self._sam_repaint_signal.connect(self._on_sam_repaint, Qt.QueuedConnection)

        # --- Label Painter Thread ---
        self._label_painter_thread = None

        # Overlay actor handle (tiny actor swapped during painting)
        # Note: overlay is treated as the authoritative visualization; we
        # no longer use a debounce flush to upload labels into the main mesh GPU buffers.

        # Overlay actor handle (tiny actor swapped during painting)
        self._label_overlay_actor = None
        self._hover_overlay_actor = None
        self._hover_overlay_context = None

        self.contextStatsComputed.connect(self._on_context_stats_computed)

        self._setup_connections()

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
        self.viewer.primaryTargetChanged.connect(self._on_primary_target_changed)
        self._orthoIndexMapReady.connect(self._on_ortho_index_map_computed)
        
        # 5. Main Window Sync
        if hasattr(self.annotation_window, 'mouseMoved'):
            self.annotation_window.mouseMoved.connect(self.mouse_bridge.on_mouse_moved)
        if hasattr(self.image_window, 'imageLoaded'):
            self.image_window.imageLoaded.connect(self._on_main_image_loaded)
        label_window = getattr(self.main_window, 'label_window', None)
        if label_window is not None and hasattr(label_window, 'labelSelected'):
            label_window.labelSelected.connect(self._on_label_window_selected)
        # 6. Context Matrix Signals
        if self.context_matrix is not None:
            # Toolbar buttons
            self.context_matrix.loadCamerasRequested.connect(self.load_cameras)
            self.context_matrix.clearSelectionsRequested.connect(self.selection_model.clear_selections)
            self.context_matrix.previousCameraRequested.connect(self._on_previous_camera_requested)
            self.context_matrix.nextCameraRequested.connect(self._on_next_camera_requested)
            self.context_matrix.visibleCamerasChanged.connect(self._on_context_visible_cameras_changed)
            # Canvas click intents
            self.context_matrix.camera_highlighted_single.connect(self._on_camera_highlighted_single)
            self.context_matrix.new_active_camera_requested.connect(self._on_camera_selected)
            # Phase 5 / multi-annotate
            self.context_matrix.set_mvat_manager(self)
            self.context_matrix.multiAnnotateToggled.connect(self._on_multi_annotate_toggled)
            self.context_matrix.semanticMaskPropagationRequested.connect(
                self.propagate_current_semantic_mask
            )
        
        # 7. Target-Lock Sync (Phase 5): AnnotationWindow viewNavigated -> sync engine
        if hasattr(self.annotation_window, 'viewNavigated'):
            self.annotation_window.viewNavigated.connect(self._on_main_view_navigated)

    def load_cameras(self):
        """
        Extract camera parameters from the RasterManager, construct Camera
        objects, and push them into the Grid and Viewer.

        Idempotent: cameras that are already loaded are skipped so that calling
        this method a second time (with no project changes) is effectively a
        no-op.  New cameras added to the project mid-session are detected and
        added incrementally without disturbing existing state (selected camera,
        hovered camera, frustum highlights, etc.).

        On the very first call (cameras dict was empty) the viewer is also
        fit-to-view and the active camera is synchronised with the annotation
        window.  On subsequent calls with new cameras, the frustums are
        refreshed but the view is not reset.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        all_paths = self.raster_manager.image_paths
        if not all_paths:
            QApplication.restoreOverrideCursor()
            return

        # ------------------------------------------------------------------
        # Partition paths: already-loaded perspective cameras, new perspective
        # cameras, and OrthoRasters (handled separately).
        # ------------------------------------------------------------------
        first_load = len(self.cameras) == 0
        new_perspective_rasters: list = []   # [(path, raster), ...]
        ortho_rasters: list = []

        for path in all_paths:
            raster = self.raster_manager.get_raster(path)
            if not raster:
                continue

            if getattr(raster, 'raster_type', '') == 'OrthoRaster':
                ortho_rasters.append(raster)
                continue

            # Skip cameras that have already been successfully loaded.
            if path in self.cameras:
                continue

            if raster.intrinsics is not None and raster.extrinsics is not None:
                new_perspective_rasters.append((path, raster))

        need_ortho = bool(ortho_rasters) and self.ortho_camera is None

        # Nothing genuinely new to do — report and exit cleanly.
        if not new_perspective_rasters and not need_ortho:
            self.main_window.status_bar.showMessage("All cameras already loaded.", 3000)
            QApplication.restoreOverrideCursor()
            return

        # ------------------------------------------------------------------
        # Build Camera objects only for new perspective cameras.
        # ------------------------------------------------------------------
        valid_count = 0
        try:
            self.main_window.status_bar.showMessage("Loading cameras...", 0)
            for path, raster in new_perspective_rasters:
                try:
                    self.cameras[path] = Camera(raster)
                    valid_count += 1
                except Exception as e:
                    print(f"❌ Failed to load perspective camera {raster.basename}: {e}")
                    print(traceback.format_exc())
        finally:
            self.main_window.status_bar.showMessage(
                f"Loaded {valid_count} new camera(s)", 3000
            )

        if valid_count == 0 and not need_ortho:
            QMessageBox.information(self.main_window, "No Camera Data", "No valid camera parameters found.")
            QApplication.restoreOverrideCursor()
            return

        # =====================================================================
        # OrthoRaster: build OrthoCamera only when one hasn't been created yet.
        # =====================================================================
        if need_ortho:
            from coralnet_toolbox.MVAT.core.OrthoCamera import OrthoCamera

            chunk_transform = getattr(ortho_rasters[0], 'chunk_transform_matrix', None)
            if chunk_transform is None:
                chunk_transform = self._chunk_transform
            if chunk_transform is None:
                chunk_transform = np.eye(4, dtype=np.float64)

            self._chunk_transform = np.asarray(chunk_transform, dtype=np.float64)
            for raster in ortho_rasters:
                raster_transform = getattr(raster, 'chunk_transform_matrix', None)
                if raster_transform is None:
                    raster_transform = self._chunk_transform

                raster.chunk_transform_matrix = np.asarray(raster_transform, dtype=np.float64).copy()
                oc = OrthoCamera(raster, raster.chunk_transform_matrix)
                if oc.is_valid:
                    self.ortho_camera = oc
                    self._maybe_compute_ortho_index_map()
                    break
                else:
                    print(f"⚠️ OrthoRaster {raster.basename} missing geo transform — skipping.")

        # =====================================================================
        # Pre-computation Cache Check — only for cameras that are new this call.
        # =====================================================================
        primary_target = self.viewer.scene_context.get_primary_target()
        newly_added_cameras = [self.cameras[p] for p, _ in new_perspective_rasters if p in self.cameras]

        if (primary_target is not None
                and self.cache_manager is not None
                and self.compute_index_maps_enabled
                and newly_added_cameras):
            target_path = primary_target.file_path
            element_type = primary_target.get_element_type()
            uncached_cameras = []

            for cam in newly_added_cameras:
                cache_key = cam._raster.extrinsics
                extra = (cam._raster.dist_coeffs.tobytes()
                         if cam.is_distorted
                         and cam._raster.dist_coeffs is not None else None)
                cache_path = self.cache_manager.get_cache_path(cache_key, target_path, element_type, extra)
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
                        self._compute_visibility_async(primary_target, uncached_cameras)

        # ------------------------------------------------------------------
        # Update the context matrix with the full (now-extended) camera set.
        # ------------------------------------------------------------------
        if self.context_matrix is not None:
            try:
                all_ordered = list(self.cameras.keys())
                self.context_matrix.set_camera_data(list(self.cameras.values()), all_ordered)
            except Exception:
                pass

        self._render_frustums()

        # Fit to view only on the very first load so we don't reset the user's
        # current pan/zoom when cameras are added incrementally.
        if first_load:
            self.viewer.fit_to_view()
            # Sync the active camera with whatever image is open in the annotation window.
            current_image_path = getattr(self.annotation_window, 'current_image_path', None)
            if current_image_path and current_image_path in self.cameras:
                self.selection_model.set_active(current_image_path)
            elif self.cameras:
                self.selection_model.set_active(next(iter(self.cameras)))

        QApplication.restoreOverrideCursor()

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
        visible_paths = self._get_visible_context_camera_paths()
        selected_path = None
        if self.selected_camera and self.selected_camera.image_path in visible_paths:
            selected_path = self.selected_camera.image_path
        
        self.viewer.add_frustums(
            self.cameras,
            selected_camera=self.selected_camera if selected_path else None,
            highlighted_paths=visible_paths,
            hovered_camera=self.hovered_camera,
            context_highlighted_paths=visible_paths,
        )

    # --- Signal Handlers ---

    def _on_main_image_loaded(self, path: str):
        """
        Handler for when the main image window loads a new image.

        For a perspective camera: sets it as the active camera so the manager
        and views synchronise (proximity reorder, frustum highlight, etc.).

        For the OrthoRaster: sets selected_camera to ortho_camera so that
        multi-annotation can project cursor previews and paint operations
        into visible context cameras.
        """
        if path in self.cameras:
            self.selection_model.set_active(path)
        else:
            # Check if this path is an OrthoRaster (works whether or not ortho_camera has been set)
            _raster = self.raster_manager.get_raster(path)
            _is_ortho = getattr(_raster, 'raster_type', '') == 'OrthoRaster'
            if _is_ortho or (self.ortho_camera is not None and path == self.ortho_camera.image_path):
                # Deselect any active perspective camera from selection model
                self.selection_model.set_active(None)
                # BUT: set selected_camera to ortho_camera so that painting/preview works
                self.selected_camera = self.ortho_camera
                self.viewer.clear_ray()
                # Show all cameras in the context matrix
                if self.context_matrix is not None and self.cameras:
                    all_paths = list(self.cameras.keys())
                    try:
                        self.context_matrix.update_stats_label(len(all_paths), len(all_paths))
                    except Exception:
                        pass
                    try:
                        self.context_matrix.set_target_camera_count(len(all_paths))
                    except Exception:
                        pass
                    try:
                        self.context_matrix.set_camera_order(all_paths)
                    except Exception:
                        pass
                # Fit the scene and snap to the canonical top perspective view.
                self.viewer.set_ortho_top_view()

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
        if self.selected_camera is not None:
            if hasattr(self.selected_camera, 'world_to_pixel'):
                pixel = self.selected_camera.world_to_pixel(point_3d)
            elif self.selected_camera.image_path in self.cameras:
                pixel = self.selected_camera.project(point_3d)
            else:
                pixel = None

            if pixel is not None and not np.isnan(pixel).any():
                u, v = pixel[0], pixel[1]
                # Show green whenever the point projects within the camera's FOV.
                # A depth-validity check caused false reds for MVATViewer double-clicks
                # whose picked 3D coordinates are accurate but lack a matching depth map entry.
                cam_w = getattr(self.selected_camera, 'width', 0)
                cam_h = getattr(self.selected_camera, 'height', 0)
                if cam_w and cam_h and (0 <= u < cam_w and 0 <= v < cam_h):
                    color = MARKER_COLOR_SELECTED
                else:
                    color = MARKER_COLOR_INVALID
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

        # After perspective maps are done, build the ortho index map if needed
        self._maybe_compute_ortho_index_map()

    def _on_primary_target_changed(self, product_id: str):
        """A new 3D model was loaded — clear stale ortho index map and rebuild."""
        self._computing_ortho_index_map = False  # reset any in-flight build
        self.clear_sphere_hover_overlay(reset_context=True)
        if self.ortho_camera is not None:
            self.ortho_camera._raster.index_map = None
            self.ortho_camera._raster.index_map_scale_factor = None
            self.ortho_camera._raster.index_map_path = None
        self._maybe_compute_ortho_index_map()

    def _maybe_compute_ortho_index_map(self):
        """Build the ortho face-ID index map if ortho camera + mesh are both ready."""
        if self._computing_ortho_index_map:
            return
        if self.ortho_camera is None or not self.ortho_camera.is_valid:
            return
        if not self.compute_index_maps_enabled:
            return

        primary_target = self.viewer.scene_context.get_primary_target()
        if primary_target is None or not isinstance(primary_target, MeshProduct):
            return

        ortho_raster = self.ortho_camera._raster

        current_scale = float(self.visibility_scale_factor)

        if ortho_raster.index_map is not None:
            return

        if self.cache_manager is not None:
            try:
                cached = self.cache_manager.load_ortho_index_map(
                    self.ortho_camera.image_path,
                    primary_target.file_path,
                    self.ortho_camera._chunk_transform,
                    getattr(self.ortho_camera, '_proj_mat', None),
                    current_scale,
                    (self.ortho_camera.width, self.ortho_camera.height),
                    element_type='face',
                )
                if cached is not None and cached.get('index_map') is not None:
                    ortho_raster.add_index_map(
                        cached['index_map'],
                        index_map_path=cached.get('cache_path'),
                        visible_indices=cached.get('visible_indices'),
                        element_type=cached.get('element_type', 'face'),
                    )
                    sf = getattr(ortho_raster, 'index_map_scale_factor', None)
                    print(f"💽 Loaded ortho index map from cache: {ortho_raster.index_map.shape}, scale_factor={sf:.4f}")
                    self.main_window.status_bar.showMessage("Loaded ortho index map from cache.", 3000)
                    return
            except Exception as e:
                print(f"⚠️ Failed to load ortho index map cache: {e}")

        self._computing_ortho_index_map = True
        self.main_window.status_bar.showMessage(f"Building ortho index map at {current_scale:.0%} quality…")
        QApplication.setOverrideCursor(Qt.WaitCursor)

        ortho_camera   = self.ortho_camera
        mesh_product   = primary_target
        requested_scale = current_scale

        def _build():
            from coralnet_toolbox.MVAT.managers.VisibilityManager import VisibilityManager
            result = VisibilityManager.compute_ortho_index_map_vtk(
                ortho_camera,
                mesh_product,
                scale_factor=requested_scale,
            )
            result['scale_factor'] = requested_scale
            return result

        def _done(future):
            # Called on the thread-pool thread — only emit a Qt signal (thread-safe).
            # Always emit so that _on_ortho_index_map_computed's finally block runs
            # and restores the busy cursor, even on failure.
            try:
                result = future.result()
            except Exception as e:
                print(f"⚠️ Ortho index map build failed: {e}")
                result = {}  # empty sentinel — _on_ortho_index_map_computed will early-exit cleanly
            try:
                self._orthoIndexMapReady.emit(result)
            except Exception as e:
                # Manager may have been deleted while the build was running.
                # Restore the cursor on the main thread so it doesn't get stuck.
                print(f"⚠️ Failed to emit ortho index map ready signal: {e}")
                try:
                    from PyQt5.QtCore import QMetaObject, Qt
                    from PyQt5.QtWidgets import QApplication
                    QMetaObject.invokeMethod(
                        QApplication.instance(),
                        "restoreOverrideCursor",
                        Qt.QueuedConnection,
                    )
                except Exception:
                    pass

        try:
            future = self._propagation_executor.submit(_build)
            future.add_done_callback(_done)
        except Exception as e:
            print(f"⚠️ Failed to submit ortho index map build: {e}")
            self._computing_ortho_index_map = False
            QApplication.restoreOverrideCursor()

    def _on_ortho_index_map_computed(self, result: dict):
        """Store the completed ortho index map on the OrthoRaster (runs on main thread via signal)."""
        try:
            result_scale = float(result.get('scale_factor', self.visibility_scale_factor))
            if not np.isclose(result_scale, float(self.visibility_scale_factor)):
                print(
                    f"⚠️ Discarding stale ortho index map at scale {result_scale:.4f}; "
                    f"current quality is {self.visibility_scale_factor:.4f}"
                )
                return

            index_map = result.get('index_map')
            if index_map is None:
                return
            visible_indices = result.get('visible_indices')
            # scale_factor is derived automatically inside OrthoRaster.add_index_map
            ortho_raster = self.ortho_camera._raster
            ortho_raster.add_index_map(
                index_map,
                index_map_path=result.get('cache_path'),
                visible_indices=visible_indices,
                element_type='face',
            )
            sf = ortho_raster.index_map_scale_factor
            print(f"✅ Ortho index map stored: {index_map.shape}, scale_factor={sf:.4f}")

            if self.cache_manager is not None:
                try:
                    primary_target = self.viewer.scene_context.get_primary_target()
                    target_path = primary_target.file_path if primary_target is not None else ""
                    cache_path = self.cache_manager.save_ortho_index_map(
                        self.ortho_camera.image_path,
                        target_path,
                        self.ortho_camera._chunk_transform,
                        getattr(self.ortho_camera, '_proj_mat', None),
                        result_scale,
                        (self.ortho_camera.width, self.ortho_camera.height),
                        index_map,
                        visible_indices if visible_indices is not None else np.array([], dtype=np.int32),
                        element_type='face',
                    )
                    if cache_path is not None:
                        ortho_raster.index_map_path = cache_path
                except Exception as e:
                    print(f"⚠️ Failed to save ortho index map cache: {e}")
        except Exception as e:
            print(f"⚠️ Failed to store ortho index map: {e}")
        finally:
            self._computing_ortho_index_map = False
            QApplication.restoreOverrideCursor()
            self.main_window.status_bar.showMessage("Ortho index map ready.", 3000)

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
                    # Use extrinsics for perspective
                    cache_key = camera._raster.extrinsics
                    extra = (camera._raster.dist_coeffs.tobytes()
                             if camera.is_distorted
                             and camera._raster.dist_coeffs is not None else None)
                    cache_path = self.cache_manager.save_visibility(
                        cache_key,
                        target_file_path,
                        result.get('index_map'),
                        result.get('visible_indices'),
                        result.get('depth_map') if self.compute_depth_maps_enabled else None,
                        element_type=element_type,
                        extra_hash_data=extra,
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

        When path is None or empty (e.g. when switching to the OrthoRaster),
        the selected camera is cleared without any further perspective logic.
        """
        if not path:
            self.selected_camera = None
            return

        camera = self.cameras.get(path)
        if camera:
            self.viewer.clear_ray()
            self._select_camera(path, camera)
            if hasattr(self.viewer, 'match_camera_perspective'):
                # Double-click to set active: animate
                self.viewer.match_camera_perspective(camera, animate=True)
            self._reorder_cameras(path)
            self._context_view_path = path

            try:
                self.image_window.load_image_by_path(path)
            except Exception:
                pass

            # Update the N / M stat when the active camera changes.
            self._update_context_stats()

    def _on_camera_selected(self, path: str):
        """Handle camera_selected from the grid (context menu 'Select Image').

        Selection state is the source of truth; the active-camera change
        handler performs the actual image load.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Make this the sole selection: set active and clear other highlights
            try:
                self.selection_model.set_active(path)
                # Ensure only the active camera remains selected/highlighted
                self.selection_model.set_selections([path])
            except Exception:
                pass
        except Exception as e:
            print(f"Failed to load selected image '{path}': {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def _on_camera_highlighted_single(self, path: str):
        """Handle viewer-only camera navigation from the context matrix."""
        self._focus_context_camera(path, animate=True)

    def _get_context_camera_order(self) -> list:
        ordered_paths = []
        if self.context_matrix is not None and hasattr(self.context_matrix, 'get_camera_order'):
            try:
                ordered_paths = list(self.context_matrix.get_camera_order())
            except Exception:
                ordered_paths = []

        if not ordered_paths:
            ordered_paths = list(self.cameras.keys())

        return [path for path in ordered_paths if path in self.cameras]

    def _focus_context_camera(self, path: str, animate: bool = True):
        camera = self.cameras.get(path)
        if camera is None:
            return

        self._context_view_path = path
        if hasattr(self.viewer, 'match_camera_perspective'):
            self.viewer.match_camera_perspective(camera, animate=animate)

    def _cycle_active_camera(self, step: int):
        """Cycle the current context-view camera through the proximity order."""
        ordered_paths = self._get_context_camera_order()
        if not ordered_paths:
            return

        current_path = self._context_view_path
        if current_path not in ordered_paths:
            if self.selected_camera and self.selected_camera.image_path in ordered_paths:
                current_path = self.selected_camera.image_path
            else:
                current_path = ordered_paths[0]

        try:
            current_index = ordered_paths.index(current_path)
        except ValueError:
            current_index = 0

        target_path = ordered_paths[(current_index + step) % len(ordered_paths)]
        self._focus_context_camera(target_path, animate=True)

    def _on_previous_camera_requested(self):
        self._cycle_active_camera(-1)

    def _on_next_camera_requested(self):
        self._cycle_active_camera(1)

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
        # Invalidate median depth cache when camera changes
        self._median_depth_cache_key = None
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
        highlighted_paths = self._get_visible_context_camera_paths()
        selected_path = None
        if self.selected_camera and self.selected_camera.image_path in highlighted_paths:
            selected_path = self.selected_camera.image_path
        
        # Update wireframe state scalars (colors based on selection/highlight)
        try:
            if hasattr(self.viewer, 'update_frustum_states'):
                self.viewer.update_frustum_states(
                    selected_path,
                    highlighted_paths,
                    self.hovered_camera,
                    context_highlighted_paths=highlighted_paths,
                )
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
        highlighted_paths = set(self._get_visible_context_camera_paths())
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

        # ------------------------------------------------------------------
        # Phase 1: Split cameras into RAM-hits and disk-cache candidates
        # ------------------------------------------------------------------
        cache_candidates = {}  # path -> camera  (need disk lookup)
        for path in highlighted_paths:
            camera = self.cameras.get(path)
            if not camera:
                continue
            # Already in active memory — nothing to do
            if camera.visible_indices is not None:
                continue
            cache_candidates[path] = camera

        # ------------------------------------------------------------------
        # Phase 2: Parallel disk-cache load for all candidates
        # ------------------------------------------------------------------
        cache_results = {}  # path -> cached_data (or None)
        if self.cache_manager is not None and target_file_path and cache_candidates:
            self.main_window.status_bar.showMessage(
                f"Checking cache for {len(cache_candidates)} camera(s)...", 1000
            )

            def _load_one(path, camera):
                cache_key = camera._raster.extrinsics
                extra = (camera._raster.dist_coeffs.tobytes()
                         if camera.is_distorted
                         and camera._raster.dist_coeffs is not None else None)
                try:
                    return path, self.cache_manager.load_visibility(
                        cache_key, target_file_path, element_type, extra
                    )
                except Exception as exc:
                    print(f"⚠️ Cache load error for {camera.label}: {exc}")
                    return path, None

            n_workers = min(8, max(1, len(cache_candidates)))
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futs = {
                    pool.submit(_load_one, path, cam): path
                    for path, cam in cache_candidates.items()
                }
                for fut in as_completed(futs):
                    path, data = fut.result()
                    cache_results[path] = data

        # ------------------------------------------------------------------
        # Phase 3: Apply cache results on the main (Qt) thread, queue misses
        # ------------------------------------------------------------------
        for path, camera in cache_candidates.items():
            cached_data = cache_results.get(path)

            if cached_data is not None:
                self.main_window.status_bar.showMessage(
                    f"Loaded visibility from cache for {camera.label}", 2000
                )
                cache_key = camera._raster.extrinsics
                extra = (camera._raster.dist_coeffs.tobytes()
                         if camera.is_distorted
                         and camera._raster.dist_coeffs is not None else None)
                cache_path = self.cache_manager.get_cache_path(
                    cache_key, target_file_path, element_type, extra
                )

                # Store index map on raster (Qt object — must be on main thread)
                camera._raster.add_index_map(
                    cached_data.get('index_map'),
                    cache_path,
                    cached_data.get('visible_indices'),
                    element_type=element_type,
                    inverted_index=cached_data.get('inverted_index')
                )

                # Restore depth map if enabled
                if self.compute_depth_maps_enabled and cached_data.get('depth_map') is not None:
                    camera._raster.merge_or_set_depth_map(cached_data['depth_map'])

                print(f"💽 Loaded visibility from disk cache: {camera.label}")
            else:
                # Miss — must be computed
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
                try:
                    # Use K_linear so the 3D engine renders a linear (undistorted) map
                    K_for_render = camera.K_linear
                    result = VisibilityManager._compute_mesh_visibility(
                        mesh_product,
                        K_for_render, camera.R, camera.t,
                        camera.width, camera.height,
                        compute_depth_map=self.compute_depth_maps_enabled
                    )
                    result['element_type'] = 'face'
                    # Warp result back to distorted-pixel space if needed
                    if camera.is_distorted and camera._raster.intrinsics_undistorted is not None:
                        warp_fn = camera._raster.warp_linear_map_to_distorted
                        if result.get('index_map') is not None:
                            result['index_map'] = warp_fn(result['index_map'], nodata=-1)
                        if result.get('depth_map') is not None:
                            result['depth_map'] = warp_fn(result['depth_map'], nodata=0.0)
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
        warp_callables_dict = {}
        dist_coeffs_bytes_dict = {}

        for cam in cameras:
            # Use K_linear so the 3D rendering engine operates in linear (undistorted) space
            camera_params_dict[cam.image_path] = (cam.K_linear, cam.R, cam.t, cam.width, cam.height)
            cache_keys_dict[cam.image_path] = cam._raster.extrinsics
            # Register a warp callable for cameras whose source image has lens distortion
            if cam.is_distorted and cam._raster.intrinsics_undistorted is not None:
                warp_callables_dict[cam.image_path] = cam._raster.warp_linear_map_to_distorted
                dist_coeffs_bytes_dict[cam.image_path] = cam._raster.dist_coeffs.tobytes()

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
                scale_factor=self.visibility_scale_factor,
                warp_callables_dict=warp_callables_dict,
                dist_coeffs_bytes_dict=dist_coeffs_bytes_dict,
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
            QApplication.restoreOverrideCursor()

    # --- Label painter management ------------------------------------------------
    def _ensure_label_painter(self, primary_target):
        """Start the painter thread the first time a mesh is annotated."""
        try:
            if primary_target is None or not isinstance(primary_target, MeshProduct):
                return

            # If already running, keep it
            if self._label_painter_thread is not None and getattr(self._label_painter_thread, 'isRunning', lambda: False)():
                return

            # Stop any previous thread first (best-effort)
            try:
                if self._label_painter_thread is not None:
                    self._label_painter_thread.stop()
                    self._label_painter_thread.wait(500)
            except Exception:
                pass

            mesh = primary_target.get_render_mesh()
            if mesh is None:
                return

            mesh_points = np.asarray(mesh.points, dtype=np.float32)
            mesh_faces_flat = np.asarray(mesh.faces.reshape(-1, 4), dtype=np.int32)

            # Use the product's python-owned label cache if available, otherwise materialize one now
            labels_cache = getattr(primary_target, '_labels_cache', None)
            if labels_cache is None:
                try:
                    labels_cache = np.asarray(mesh.cell_data['Labels']).copy()
                    primary_target._labels_cache = labels_cache
                except Exception:
                    labels_cache = None

            class_ids = getattr(primary_target, 'class_ids', None)

            if labels_cache is None or class_ids is None:
                return

            self._label_painter_thread = LabelPainterThread(
                mesh_points=mesh_points,
                mesh_faces_flat=mesh_faces_flat,
                labels_view=labels_cache,
                class_ids=class_ids,
            )
            self._label_painter_thread.overlay_ready.connect(self._on_overlay_ready, Qt.QueuedConnection)
            self._label_painter_thread.start()
        except Exception as e:
            print(f"⚠️ _ensure_label_painter failed: {e}")

    def _on_overlay_ready(self, overlay):
        """Main thread: swap the overlay actor. Only tiny PolyData hits the GPU."""
        try:
            if self._label_overlay_actor is not None:
                try:
                    # Remove without triggering a full upload
                    self.viewer.plotter.remove_actor(self._label_overlay_actor, render=False)
                except Exception:
                    pass
                self._label_overlay_actor = None
            # Add the new tiny overlay actor. The worker emits numpy arrays
            # (points, faces_flat, colors) to avoid VTK work off the GUI thread.
            try:
                # If overlay is a tuple/list from the worker: assemble PolyData here
                if isinstance(overlay, (list, tuple)) and len(overlay) == 3:
                    pts, faces_flat, colors = overlay
                    import pyvista as pv
                    pts_arr = np.asarray(pts, dtype=np.float32)
                    faces_arr = np.asarray(faces_flat, dtype=np.int32)
                    colors_arr = np.asarray(colors, dtype=np.uint8)

                    tiny = pv.PolyData(pts_arr, faces_arr)
                    tiny.cell_data['OverlayColors'] = colors_arr
                    mesh_to_add = tiny
                else:
                    # Backwards-compat: already a pv.PolyData
                    mesh_to_add = overlay

                self._label_overlay_actor = self.viewer.plotter.add_mesh(
                    mesh_to_add,
                    scalars='OverlayColors',
                    rgb=True,
                    copy_mesh=False,
                    lighting=False,
                    show_scalar_bar=False,
                )
            except Exception as e:
                print(f"⚠️ Failed to assemble overlay on main thread: {e}")
            try:
                self.viewer.plotter.render()
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️ Overlay swap failed: {e}")

    def _on_label_window_selected(self, *_args):
        """Refresh the hover overlay when the active label changes."""
        try:
            self.refresh_sphere_hover_overlay()
        except Exception:
            pass

    def _get_active_label_widget(self):
        label_window = getattr(self.main_window, 'label_window', None)
        label = getattr(label_window, 'active_label', None) if label_window is not None else None
        if label is None:
            label = getattr(self.annotation_window, 'selected_label', None)

        if label is not None and not hasattr(label, 'color'):
            label_id = getattr(label, 'id', label)
            if isinstance(label_id, str) and label_window is not None:
                try:
                    label = label_window.get_label_by_id(label_id, return_review=True)
                except Exception:
                    label = None

        return label if label is not None and hasattr(label, 'color') else None

    def _get_active_label_color_rgb(self):
        label = self._get_active_label_widget()
        if label is None:
            return None

        try:
            return (
                int(label.color.red()),
                int(label.color.green()),
                int(label.color.blue()),
            )
        except Exception:
            return None

    def _get_primary_mesh_target(self):
        try:
            primary_target = self.viewer.scene_context.get_primary_target()
        except Exception:
            primary_target = None

        if primary_target is None or not isinstance(primary_target, MeshProduct):
            return None
        return primary_target

    def _get_sphere_hover_radius(self):
        sphere_manager = getattr(self.viewer, '_sphere_manager', None)
        try:
            return float(getattr(sphere_manager, 'radius', 0.1))
        except Exception:
            return 0.1

    def _get_faces_within_sphere(self, primary_target, center, radius):
        try:
            primary_target.prepare_geometry()
        except Exception:
            pass

        centers = getattr(primary_target, '_element_centers_np', None)
        if centers is None or len(centers) == 0:
            return np.empty(0, dtype=np.int32)

        center = np.asarray(center, dtype=np.float64)

        try:
            from scipy.spatial import cKDTree
        except Exception:
            cKDTree = None

        tree = getattr(primary_target, '_hover_face_kdtree', None)
        tree_product_id = getattr(primary_target, '_hover_face_kdtree_product_id', None)
        if tree is None or tree_product_id != getattr(primary_target, 'product_id', None):
            tree = None
            if cKDTree is not None:
                try:
                    tree = cKDTree(np.asarray(centers, dtype=np.float32))
                    primary_target._hover_face_kdtree = tree
                    primary_target._hover_face_kdtree_product_id = getattr(primary_target, 'product_id', None)
                except Exception:
                    tree = None

        if tree is not None:
            try:
                face_ids = tree.query_ball_point(center, float(radius))
                return np.asarray(face_ids, dtype=np.int32)
            except Exception:
                pass

        radius_sq = float(radius) * float(radius)
        deltas = np.asarray(centers, dtype=np.float32) - center.astype(np.float32)
        distances_sq = np.einsum('ij,ij->i', deltas, deltas)
        return np.flatnonzero(distances_sq <= radius_sq).astype(np.int32)

    def _swap_hover_overlay_actor(self, overlay, render: bool = True):
        try:
            if self._hover_overlay_actor is not None:
                try:
                    self.viewer.plotter.remove_actor(self._hover_overlay_actor, render=False)
                except Exception:
                    pass
                self._hover_overlay_actor = None

            if overlay is not None:
                self._hover_overlay_actor = self.viewer.plotter.add_mesh(
                    overlay,
                    scalars='OverlayColors',
                    rgb=True,
                    copy_mesh=False,
                    lighting=False,
                    opacity=0.45,
                    show_scalar_bar=False,
                    smooth_shading=False,
                    pickable=False,
                    name='_sphere_hover_overlay',
                )

            if render:
                try:
                    self.viewer.plotter.render()
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ Hover overlay swap failed: {e}")

    def clear_sphere_hover_overlay(self, reset_context: bool = False, render: bool = True):
        if reset_context:
            self._hover_overlay_context = None
        self._swap_hover_overlay_actor(None, render=render)

    def refresh_sphere_hover_overlay(self, render: bool = True):
        context = self._hover_overlay_context
        if not context:
            self._swap_hover_overlay_actor(None, render=render)
            return

        try:
            if not bool(getattr(self.viewer, '_sphere_visible', True)):
                self._swap_hover_overlay_actor(None, render=render)
                return
            passthrough_active = getattr(self.viewer, '_is_sphere_passthrough_active', None)
            if callable(passthrough_active) and passthrough_active():
                self._swap_hover_overlay_actor(None, render=render)
                return
        except Exception:
            pass

        primary_target = self._get_primary_mesh_target()
        if primary_target is None or getattr(primary_target, 'product_id', None) != context.get('product_id'):
            self.clear_sphere_hover_overlay(reset_context=True, render=render)
            return

        color_rgb = self._get_active_label_color_rgb()
        if color_rgb is None:
            self._swap_hover_overlay_actor(None, render=render)
            return

        center = context.get('center')
        if center is None:
            self._swap_hover_overlay_actor(None, render=render)
            return

        radius = self._get_sphere_hover_radius()
        face_ids = self._get_faces_within_sphere(primary_target, center, radius)
        if face_ids is None or len(face_ids) == 0:
            self._swap_hover_overlay_actor(None, render=render)
            return

        mesh = primary_target.get_render_mesh()
        if mesh is None:
            self.clear_sphere_hover_overlay(reset_context=True, render=render)
            return

        mesh_points = np.asarray(mesh.points, dtype=np.float32)
        mesh_faces_flat = np.asarray(mesh.faces.reshape(-1, 4), dtype=np.int32)
        overlay = LabelPainterThread.build_overlay(mesh_points, mesh_faces_flat, face_ids, color_rgb)
        self._swap_hover_overlay_actor(overlay, render=render)

    def update_sphere_hover_overlay(self, center, render: bool = True):
        primary_target = self._get_primary_mesh_target()
        if primary_target is None:
            self.clear_sphere_hover_overlay(reset_context=True, render=render)
            return

        try:
            if not bool(getattr(self.viewer, '_sphere_visible', True)):
                self.clear_sphere_hover_overlay(reset_context=True, render=render)
                return
            passthrough_active = getattr(self.viewer, '_is_sphere_passthrough_active', None)
            if callable(passthrough_active) and passthrough_active():
                self.clear_sphere_hover_overlay(reset_context=False, render=render)
                return
        except Exception:
            pass

        try:
            center = np.asarray(center, dtype=np.float64)
        except Exception:
            self.clear_sphere_hover_overlay(reset_context=True, render=render)
            return

        self._hover_overlay_context = {
            'product_id': getattr(primary_target, 'product_id', None),
            'center': center,
        }
        self.refresh_sphere_hover_overlay(render=render)

    # Note: full-GPU flush is intentionally removed. The overlay actor
    # is treated as the authoritative visualization for painted faces
    # during the session; persistent GPU uploads are unnecessary.

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

    def _calculate_camera_proximity_score(self, reference_camera, candidate_camera, scene_size=None):
        """
        Calculate a scalar proximity score between two cameras used for
        ordering the camera grid.

        The score is an interpolation of a distance-based score (exponentially
        decaying with scene-normalized spatial distance) and a view-alignment
        score (dot product between viewing directions). Cameras behind the
        reference (negative alignment) are given a score of 0.
        """
        if (getattr(reference_camera, 'position', None) is None or
                getattr(reference_camera, 'R', None) is None or
                getattr(candidate_camera, 'position', None) is None or
                getattr(candidate_camera, 'R', None) is None):
            return 0.0

        spatial_distance = np.linalg.norm(reference_camera.position - candidate_camera.position)
        ref_view_dir = reference_camera.R.T @ np.array([0, 0, 1])
        cand_view_dir = candidate_camera.R.T @ np.array([0, 0, 1])
        
        ref_view_dir = ref_view_dir / np.linalg.norm(ref_view_dir)
        cand_view_dir = cand_view_dir / np.linalg.norm(cand_view_dir)
        
        view_alignment = np.dot(ref_view_dir, cand_view_dir)
        
        if scene_size is None:
            try:
                bounds = self.viewer.get_bounds()
                scene_size = np.sqrt(
                    (bounds[1] - bounds[0])**2
                    + (bounds[3] - bounds[2])**2
                    + (bounds[5] - bounds[4])**2
                )
            except Exception:
                scene_size = None

        if scene_size is None:
            normalized_distance = spatial_distance / 10.0
        else:
            normalized_distance = spatial_distance / (scene_size + 1e-6)
            
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
        each visible context camera to synchronize their viewports and rotation.
        """
        if self.context_matrix is None:
            return
        if not self.context_matrix.target_lock_enabled:
            return
        base_rotation = getattr(self.annotation_window, 'rotation_angle', 0.0)

        # Ortho path: derive world point via z-channel instead of index-map ray
        if self.ortho_camera is not None:
            current_path = getattr(self.annotation_window, 'current_image_path', None)
            if current_path == self.ortho_camera.image_path:
                self._on_ortho_view_navigated(center_x, center_y, zoom_factor, base_rotation)
                return

        if self.selected_camera is None:
            return

        # Fetch reference path and current rotation from the Annotation Window
        reference_path = self.selected_camera.image_path

        # Step 1: Get the 3D world point at the viewport center
        world_point = self._get_world_point_at_pixel(
            self.selected_camera, center_x, center_y
        )
        if world_point is None:
            return

        # Step 2: Project into each visible context camera.
        # targets_with_center: canvases where the world point falls inside the image
        # zoom_only: canvases that are visible but the world point falls outside their FOV
        targets_with_center = {}
        zoom_only = set()
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
                zoom_only.add(i)
                continue

            if np.isnan(pixel).any():
                zoom_only.add(i)
                continue

            target_u, target_v = float(pixel[0]), float(pixel[1])

            if 0 <= target_u < camera.width and 0 <= target_v < camera.height:
                targets_with_center[i] = (target_u, target_v)
            else:
                # World point outside this camera's FOV — still sync zoom level
                zoom_only.add(i)

        # Step 3: Compute relative zoom ratio (how far beyond fit-to-view)
        if self.selected_camera and hasattr(self.annotation_window, '_min_zoom'):
            min_zoom = self.annotation_window._min_zoom
            if min_zoom > 0:
                relative_zoom = zoom_factor / min_zoom
            else:
                relative_zoom = 1.0
        else:
            relative_zoom = 1.0

        # Step 4: Full snap and Zoom-only (Now passing rotation kwargs)
        try:
            self.context_matrix.request_sync(
                targets_with_center, relative_zoom,
                reference_path=reference_path, base_rotation=base_rotation
            )
            self.context_matrix.request_zoom_only(
                zoom_only, relative_zoom,
                reference_path=reference_path, base_rotation=base_rotation
            )
        except TypeError:
            # Fallback if ContextMatrix hasn't been updated to accept kwargs yet
            self.context_matrix.request_sync(targets_with_center, relative_zoom)
            self.context_matrix.request_zoom_only(zoom_only, relative_zoom)

    def _on_ortho_view_navigated(self, center_x: float, center_y: float, zoom_factor: float, base_rotation: float):
        """Sync context canvases when the user pans/zooms the OrthoRaster view.

        Mirrors _on_main_view_navigated but resolves the world point via
        z-channel lookup (O(1)) rather than an index-map / ray-trace.
        """
        ortho_camera = self.ortho_camera
        cx, cy = int(round(center_x)), int(round(center_y))

        X, Y = ortho_camera.pixel_to_geo(cx, cy)
        Z = ortho_camera._raster.get_z_value(cx, cy)
        if Z is None:
            return
        world_point = ortho_camera.geo_to_world(X, Y, Z)

        # Compute relative zoom (same logic as perspective path)
        min_zoom = getattr(self.annotation_window, '_min_zoom', 0)
        relative_zoom = (zoom_factor / min_zoom) if min_zoom > 0 else 1.0

        targets_with_center = {}
        zoom_only = set()
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
                zoom_only.add(i)
                continue
            if np.isnan(pixel).any():
                zoom_only.add(i)
                continue
            u, v = float(pixel[0]), float(pixel[1])
            if 0 <= u < camera.width and 0 <= v < camera.height:
                targets_with_center[i] = (u, v)
            else:
                zoom_only.add(i)

        self.context_matrix.request_sync(
            targets_with_center, relative_zoom,
            reference_path=None, base_rotation=base_rotation
        )
        self.context_matrix.request_zoom_only(
            zoom_only, relative_zoom,
            reference_path=None, base_rotation=base_rotation
        )

    def _get_world_point_at_pixel(self, camera, px, py):
        """Get the 3D world point at a specific pixel coordinate.

        Plan A: Index-map lookup (exact element coordinate — most accurate).
        Plan B: Z-channel depth/elevation unprojection.
        Plan C: Scene median depth fallback (rough estimate).

        Args:
            camera: Camera object for the active image.
            px, py: Pixel coordinates (float).

        Returns:
            np.ndarray [x,y,z] world point, or None if impossible.
        """
        # Clamp to image bounds
        px = max(0, min(px, camera.width - 1))
        py = max(0, min(py, camera.height - 1))

        # Plan A: Index-map lookup — exact element coordinate, same approach used by
        # AnnotationWindow double-click.  Provides the most accurate world point and
        # avoids the depth-buffer imprecision that plagues median-depth fallbacks.
        try:
            index_map = camera._raster.index_map
            primary_target = self.viewer.scene_context.get_primary_target()
            if index_map is not None and primary_target is not None:
                candidate_id = int(index_map[int(py), int(px)])
                if candidate_id > -1:
                    raw_coord = primary_target.get_element_coordinate(candidate_id)
                    if raw_coord is not None:
                        if hasattr(raw_coord, 'cpu'):
                            return raw_coord.cpu().numpy().astype(np.float64)
                        else:
                            return np.asarray(raw_coord, dtype=np.float64)
        except Exception:
            pass

        # Plan B: Z-channel depth/elevation unprojection
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
            # Plan C: Fallback to scene median depth
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
                # NEW: Attach the live stroke hook
                brush_tool.live_stroke_callback = self._on_live_stroke_applied 
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
                target_paths = set(visible)
                if self.ortho_camera is not None and not self._is_ortho_annotation_source():
                    target_paths.add(self.ortho_camera.image_path)
                if visible and self.compute_index_maps_enabled:
                    self.main_window.status_bar.showMessage("Preparing context visibility maps...", 2000)
                    # Ask the visibility system to compute index maps for these visible cameras
                    # _update_visibility_filter handles cache checks and async worker dispatch.
                    self._update_visibility_filter(visible)

                # --- Force Mask Canvas Allocation NOW ---
                # Don't wait for the first brush stroke to allocate canvases!
                project_labels = list(self.main_window.label_window.labels)
                for path in target_paths:
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
                brush_tool.live_stroke_callback = None
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

        When on an OrthoCamera (orthomosaic), projects the cursor into all visible
        perspective cameras. When on a perspective camera, projects into all visible
        context cameras.

        Uses the blazingly fast center-point projection to display brush previews
        in all visible context cameras. The tool factory already draws the correct
        brush size visually; we just need to tell it where the center is.
        """
        if self.selected_camera is None or self.context_matrix is None:
            return

        px, py = int(scene_pos.x()), int(scene_pos.y())

        # Determine which cameras should show previews
        visible_paths = self._get_annotation_target_paths()

        # Use the blazingly fast center-point projection
        projections = self._build_projection(px, py)
        self.context_matrix.update_cursor_previews(projections, visible_paths, item_factory)

    def _on_cursor_preview_cleared(self):
        """Clear cursor previews from all context canvases."""
        if self.context_matrix is not None:
            self.context_matrix.clear_all_cursor_previews()

    def _on_live_stroke_applied(self, scene_pos, size, shape, color):
        """Instantly projects the center of the brush to context cameras for a live trail.

        When on an OrthoCamera (orthomosaic), projects into all visible perspective cameras.
        When on a perspective camera, projects into visible context cameras.
        """
        if self.selected_camera is None or self.context_matrix is None:
            return

        px, py = int(scene_pos.x()), int(scene_pos.y())

        # We already have a blazingly fast center-point projector!
        projections = self._build_projection(px, py)
        try:
            self.context_matrix.update_live_scratchpads(projections, size, shape, color)
        except Exception:
            pass

    def _get_context_canvas_for_path(self, image_path: str):
        """Return the context canvas currently displaying image_path, or None."""
        if self.context_matrix is None:
            return None
        for canvas in self.context_matrix._visible_canvases:
            if canvas is not None and canvas.current_image_path == image_path:
                return canvas
        return None

    def _compute_dirty_rect_from_flat_indices(self, flat_indices, width: int, height: int, padding: int = 1):
        """Return an x/y dirty rectangle for a flat index set, or None if empty."""
        if flat_indices is None or width <= 0 or height <= 0:
            return None

        flat_indices = np.asarray(flat_indices, dtype=np.int64).ravel()
        if flat_indices.size == 0:
            return None

        y_coords, x_coords = np.divmod(flat_indices, width)
        return (
            max(0, int(x_coords.min()) - padding),
            max(0, int(y_coords.min()) - padding),
            min(width, int(x_coords.max()) + padding + 1),
            min(height, int(y_coords.max()) + padding + 1),
        )

    def _apply_mask_visual_update(self, target_path: str, target_mask, label_id: Optional[str] = None, update_rect=None):
        """Apply the minimal UI refresh needed after a silent mask write."""
        if target_mask is None:
            return

        if label_id is not None and label_id not in target_mask.visible_label_ids:
            target_mask.visible_label_ids.add(label_id)

        try:
            target_mask.update_graphics_item(update_rect=update_rect)
        except Exception:
            pass

        context_canvas = self._get_context_canvas_for_path(target_path)
        if context_canvas is not None and context_canvas._mask_overlay_item is None:
            try:
                context_canvas.set_mask_overlay(target_mask)
            except Exception:
                pass

    def _get_visible_context_camera_paths(self) -> list:
        """Return the ordered list of image paths currently visible in the context matrix."""
        if self.context_matrix is None:
            return []
        if hasattr(self.context_matrix, 'get_visible_camera_paths'):
            try:
                return list(self.context_matrix.get_visible_camera_paths())
            except Exception:
                pass

        visible_paths = []
        for canvas in self.context_matrix._visible_canvases:
            if canvas and canvas.active_image and canvas.current_image_path:
                visible_paths.append(canvas.current_image_path)
        return visible_paths

    def _get_visible_context_cameras(self) -> list:
        """Return the Camera objects currently visible in the context matrix."""
        return [self.cameras[path] for path in self._get_visible_context_camera_paths() if path in self.cameras]

    def _get_visible_context_target_paths(self) -> set:
        """Return visible context camera paths excluding the active annotation camera."""
        paths = set(self._get_visible_context_camera_paths())
        if self.selected_camera and self.selected_camera.image_path in paths:
            paths.discard(self.selected_camera.image_path)
        return paths

    def _get_semantic_target_paths(self, source_camera) -> set:
        """Return source-aware target paths for semantic prediction propagation."""
        if source_camera is None:
            return set()

        target_paths = set(self._get_visible_context_camera_paths())
        target_paths.discard(source_camera.image_path)

        if self.ortho_camera is not None and source_camera is not self.ortho_camera:
            target_paths.add(self.ortho_camera.image_path)

        return target_paths

    def _warn_semantic_propagation(self, message: str):
        """Show a short warning when semantic propagation cannot run."""
        print(f"⚠️ Semantic propagation skipped: {message}")

        status_bar = getattr(self.main_window, 'status_bar', None)
        if status_bar is not None:
            try:
                status_bar.showMessage(message, 5000)
                return
            except Exception:
                pass

        try:
            QMessageBox.warning(self.main_window, "Semantic Propagation", message)
        except Exception:
            pass

    def propagate_current_semantic_mask(self):
        """Propagate the active AnnotationWindow semantic mask to MVAT targets."""
        if self._propagating_annotation:
            self._warn_semantic_propagation("Semantic propagation is already in progress.")
            return

        annotation_window = getattr(self.main_window, 'annotation_window', None)
        if annotation_window is None:
            self._warn_semantic_propagation("AnnotationWindow is not available.")
            return

        image_path = getattr(annotation_window, 'current_image_path', None)
        if not image_path:
            self._warn_semantic_propagation("No image is currently active in the AnnotationWindow.")
            return

        source_camera = self._get_camera_for_path(image_path)
        if source_camera is None:
            self._warn_semantic_propagation("The active image is not loaded in MVAT.")
            return

        target_paths = self._get_semantic_target_paths(source_camera)
        if not target_paths:
            self._warn_semantic_propagation("No target cameras are currently visible for semantic propagation.")
            return

        status_bar = getattr(self.main_window, 'status_bar', None)
        if status_bar is not None:
            try:
                status_bar.showMessage(
                    f"Propagating semantic mask to {len(target_paths)} target camera(s)...",
                    0,
                )
            except Exception:
                pass

        source_raster = self.raster_manager.get_raster(image_path) if self.raster_manager is not None else None
        source_mask = getattr(source_raster, 'mask_annotation', None)
        if source_mask is None:
            self._warn_semantic_propagation("The active image does not have a semantic mask to propagate.")
            return

        label_window = getattr(self.main_window, 'label_window', None)
        project_labels = list(getattr(label_window, 'labels', [])) if label_window is not None else []
        if not project_labels:
            self._warn_semantic_propagation("No project labels are available for semantic propagation.")
            return

        try:
            source_mask.sync_label_map(project_labels)
        except Exception:
            pass

        try:
            source_mask.update_graphics_item()
        except Exception:
            pass

        mask_data = getattr(source_mask, 'mask_data', None)
        if mask_data is None:
            self._warn_semantic_propagation("The active semantic mask is missing mask data.")
            return

        lock_bit = getattr(source_mask, 'LOCK_BIT', None)
        try:
            if lock_bit is not None and int(lock_bit) > 1:
                semantic_values = np.unique(mask_data % int(lock_bit))
            else:
                semantic_values = np.unique(mask_data)
        except Exception:
            semantic_values = np.unique(mask_data)

        semantic_values = semantic_values[semantic_values > 0]
        if len(semantic_values) == 0:
            self._warn_semantic_propagation("The active semantic mask does not contain any labels to propagate.")
            return

        if getattr(source_camera, '_raster', None) is None or getattr(source_camera._raster, 'index_map', None) is None:
            self._warn_semantic_propagation(
                "The active camera does not have an index map, so semantic propagation is unavailable."
            )
            return

        try:
            self._on_semantic_prediction_applied(image_path, source_mask)
        except Exception as exc:
            print(f"Error while propagating semantic mask from {image_path}: {exc}")
            traceback.print_exc()
            self._warn_semantic_propagation("Semantic propagation failed. See console for details.")
            return

        if status_bar is not None:
            try:
                status_bar.showMessage(
                    f"Semantic mask propagated to {len(target_paths)} target camera(s).",
                    3000,
                )
            except Exception:
                pass

    def _is_ortho_annotation_source(self) -> bool:
        """Return True when the active annotation source is the ortho view."""
        return self.ortho_camera is not None and self.selected_camera == self.ortho_camera

    def _get_annotation_target_paths(self) -> set:
        """Return the target camera paths for the current annotation source."""
        if self._is_ortho_annotation_source():
            return self._get_ortho_target_cameras()
        target_paths = self._get_visible_context_target_paths()
        if self.ortho_camera is not None:
            target_paths.add(self.ortho_camera.image_path)
        return target_paths

    def _extract_source_ids_from_crop_mask(self,
                                           source_camera,
                                           source_mask: np.ndarray,
                                           px: int,
                                           py: int) -> Optional[np.ndarray]:
        """Extract visible element IDs from a crop-centred binary mask."""
        raster = getattr(source_camera, '_raster', None)
        if raster is None or source_mask is None:
            return None

        source_index_map = getattr(raster, 'index_map', None)
        if source_index_map is None:
            return None

        source_mask = np.asarray(source_mask)
        mask_h, mask_w = source_mask.shape
        scale_factor = getattr(raster, 'index_map_scale_factor', None)

        if scale_factor is not None and scale_factor != 1.0:
            map_px = int(px * scale_factor)
            map_py = int(py * scale_factor)
            map_bw = max(1, int(mask_w * scale_factor))
            map_bh = max(1, int(mask_h * scale_factor))
        else:
            map_px, map_py = px, py
            map_bw, map_bh = mask_w, mask_h

        x0 = map_px - map_bw // 2
        y0 = map_py - map_bh // 2
        x1 = x0 + map_bw
        y1 = y0 + map_bh

        img_h, img_w = source_index_map.shape
        if x0 >= img_w or y0 >= img_h or x1 <= 0 or y1 <= 0:
            return np.array([], dtype=np.int64)

        cx0 = max(x0, 0)
        cy0 = max(y0, 0)
        cx1 = min(x1, img_w)
        cy1 = min(y1, img_h)
        index_slice = source_index_map[cy0:cy1, cx0:cx1]

        if scale_factor is not None and scale_factor != 1.0:
            raw_ids = index_slice.ravel()
        else:
            bx0 = cx0 - x0
            by0 = cy0 - y0
            bx1 = bx0 + (cx1 - cx0)
            by1 = by0 + (cy1 - cy0)
            mask_clip = source_mask[by0:by1, bx0:bx1]
            raw_ids = index_slice[mask_clip.astype(bool)]

        unique_ids = np.unique(raw_ids)
        return unique_ids[unique_ids > -1].astype(np.int64, copy=False)

    def _extract_source_ids_from_full_mask(self,
                                           source_camera,
                                           source_mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract visible element IDs from a full-frame binary mask."""
        raster = getattr(source_camera, '_raster', None)
        if raster is None or source_mask is None:
            return None

        source_index_map = getattr(raster, 'index_map', None)
        if source_index_map is None:
            return None

        source_mask = np.asarray(source_mask)
        if source_mask.ndim != 2:
            return None

        needs_resize = (
            source_mask.shape != source_index_map.shape or
            (getattr(raster, 'index_map_scale_factor', None) not in (None, 1.0))
        )

        if needs_resize:
            import cv2
            mask_bool = cv2.resize(
                source_mask.astype(np.uint8),
                (source_index_map.shape[1], source_index_map.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            mask_bool = source_mask.astype(bool)

        if not np.any(mask_bool):
            return np.array([], dtype=np.int64)

        raw_ids = source_index_map[mask_bool]
        unique_ids = np.unique(raw_ids)
        return unique_ids[unique_ids > -1].astype(np.int64, copy=False)

    def _extract_source_element_ids_from_full_mask(self,
                                                   source_camera,
                                                   source_mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract raw visible element IDs from a full-frame binary mask.

        Unlike _extract_source_ids_from_full_mask, this preserves duplicates so
        callers can compute per-element class votes before collapsing to one
        class per element.
        """
        raster = getattr(source_camera, '_raster', None)
        if raster is None or source_mask is None:
            return None

        source_index_map = getattr(raster, 'index_map', None)
        if source_index_map is None:
            return None

        source_mask = np.asarray(source_mask)
        if source_mask.ndim != 2:
            return None

        needs_resize = (
            source_mask.shape != source_index_map.shape or
            (getattr(raster, 'index_map_scale_factor', None) not in (None, 1.0))
        )

        if needs_resize:
            import cv2
            mask_bool = cv2.resize(
                source_mask.astype(np.uint8),
                (source_index_map.shape[1], source_index_map.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            mask_bool = source_mask.astype(bool)

        if not np.any(mask_bool):
            return np.array([], dtype=np.int64)

        raw_ids = source_index_map[mask_bool]
        return raw_ids[raw_ids > -1].astype(np.int64, copy=False)

    def _extract_source_element_ids_from_region(self,
                                                 source_camera,
                                                 source_mask: np.ndarray,
                                                 top_left) -> Optional[np.ndarray]:
        """Extract raw visible element IDs from a partial mask region.

        This mirrors the full-mask helper, but only samples the work-area tile
        (or other region payload) so semantic propagation does not leak stale
        pixels from untouched parts of the image.
        """
        raster = getattr(source_camera, '_raster', None)
        if raster is None or source_mask is None:
            return None

        source_index_map = getattr(raster, 'index_map', None)
        if source_index_map is None:
            return None

        source_mask = np.asarray(source_mask)
        if source_mask.ndim != 2:
            return None

        x, y = top_left
        mask_h, mask_w = source_mask.shape
        scale_factor = getattr(raster, 'index_map_scale_factor', None)

        if scale_factor is not None and scale_factor != 1.0:
            map_x0 = int(x * scale_factor)
            map_y0 = int(y * scale_factor)
            map_w = max(1, int(mask_w * scale_factor))
            map_h = max(1, int(mask_h * scale_factor))
        else:
            map_x0 = int(x)
            map_y0 = int(y)
            map_w = mask_w
            map_h = mask_h

        map_x1 = map_x0 + map_w
        map_y1 = map_y0 + map_h

        img_h, img_w = source_index_map.shape
        if map_x0 >= img_w or map_y0 >= img_h or map_x1 <= 0 or map_y1 <= 0:
            return np.array([], dtype=np.int64)

        cx0 = max(map_x0, 0)
        cy0 = max(map_y0, 0)
        cx1 = min(map_x1, img_w)
        cy1 = min(map_y1, img_h)
        index_slice = source_index_map[cy0:cy1, cx0:cx1]

        if index_slice.size == 0:
            return np.array([], dtype=np.int64)

        if scale_factor is not None and scale_factor != 1.0:
            import cv2

            mask_resized = cv2.resize(
                source_mask.astype(np.uint8),
                (map_w, map_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            bx0 = cx0 - map_x0
            by0 = cy0 - map_y0
            bx1 = bx0 + (cx1 - cx0)
            by1 = by0 + (cy1 - cy0)
            mask_clip = mask_resized[by0:by1, bx0:bx1]
        else:
            bx0 = cx0 - map_x0
            by0 = cy0 - map_y0
            bx1 = bx0 + (cx1 - cx0)
            by1 = by0 + (cy1 - cy0)
            mask_clip = source_mask[by0:by1, bx0:bx1].astype(bool)

        if mask_clip.size == 0 or not np.any(mask_clip):
            return np.array([], dtype=np.int64)

        raw_ids = index_slice[mask_clip]
        return raw_ids[raw_ids > -1].astype(np.int64, copy=False)

    def _get_visible_context_paths(self) -> set:
        """Return the set of image paths currently visible in the context matrix."""
        return set(self._get_visible_context_camera_paths())

    def _get_ortho_target_cameras(self) -> set:
        """Get target camera paths when painting on OrthoCamera.

        Returns the set of visible perspective camera paths that should receive
        multi-annotations when painting on an orthomosaic. This is the same as
        the visible context target paths, since we want to paint all visible cameras
        that can see the orthomosaic's geometry.
        """
        if self.ortho_camera is None:
            return set()
        # When on orthomosaic, propagate to all visible context cameras
        # (the ContextMatrix handles filtering for which cameras are viewable)
        return self._get_visible_context_target_paths()

    def _get_camera_for_path(self, image_path: str):
        """Return the loaded camera object for a path, including the orthocamera."""
        if self.ortho_camera is not None and image_path == self.ortho_camera.image_path:
            return self.ortho_camera
        return self.cameras.get(image_path)

    def _is_ortho_path(self, image_path: str) -> bool:
        return self.ortho_camera is not None and image_path == self.ortho_camera.image_path

    def _on_context_visible_cameras_changed(self, visible_paths):
        """Refresh viewer state when the ContextMatrix changes its visible cameras."""
        try:
            self.viewer.clear_ray()
        except Exception:
            pass

        self._update_frustum_states()
        self._update_visibility_filter(list(visible_paths))

        # Update the N / M stat when the visible count changes.
        self._update_context_stats()

    def _get_scene_size_snapshot(self):
        """Capture the viewer scene size on the main thread for background proximity checks."""
        try:
            bounds = self.viewer.get_bounds()
            return float(np.sqrt(
                (bounds[1] - bounds[0])**2
                + (bounds[3] - bounds[2])**2
                + (bounds[5] - bounds[4])**2
            ))
        except Exception:
            return None

    def count_overlapping_cameras(self, active_camera, camera_items=None, scene_size=None):
        """
        Calculates how many cameras share a view of the same 3D geometry.
        Uses proximity scoring as a fast-reject to keep UI thread performance high.

        TODO (Threading): If this begins to block the UI on extreme datasets
        (e.g., >10M polygons and >1,000 cameras), move this loop into
        self._propagation_executor.submit(). Have the thread return the
        overlap_count and emit a PyQt signal back to the main thread to
        safely call self.context_matrix.update_stats_label().
        """
        overlap_count = 0
        min_overlap_ratio = 0.20  # Secondary camera must cover at least 20% of the active camera's view
        camera_items = tuple(camera_items if camera_items is not None else self.cameras.items())
        active_indices = active_camera.visible_indices
        active_visible_count = len(active_indices) if active_indices is not None else 0

        # OrthoCamera (and any other non-pose camera) does not expose a
        # perspective center / orientation. In ortho mode the UI already treats
        # the orthomosaic as overlapping with every loaded context camera, so we
        # return that count directly instead of running perspective heuristics.
        if (getattr(active_camera, 'position', None) is None or
                getattr(active_camera, 'R', None) is None):
            return len(camera_items)

        for path, cam in camera_items:
            if path == active_camera.image_path:
                overlap_count += 1  # Always counts itself
                continue

            # OPTIMIZATION 1: Fast Reject.
            # If the proximity score is 0 (facing away, or too far), they don't overlap.
            # Skip the expensive array math entirely!
            score = self._calculate_camera_proximity_score(active_camera, cam, scene_size=scene_size)
            if score == 0.0:
                continue

            # OPTIMIZATION 2: True Geometric Overlap
            if active_indices is not None and cam.visible_indices is not None:
                # Both arrays are pre-sorted and unique thanks to VisibilityWorker.
                # assume_unique=True makes this incredibly fast.
                shared = np.intersect1d(active_indices, cam.visible_indices, assume_unique=True)

                if active_visible_count > 0 and (len(shared) / active_visible_count) >= min_overlap_ratio:
                    overlap_count += 1

        return overlap_count

    def _count_overlapping_cameras_async(self, request_id: int, active_path: str, visible_count: int, active_camera, camera_items, scene_size):
        """Background worker wrapper for overlap counting."""
        try:
            overlap_count = self.count_overlapping_cameras(active_camera, camera_items=camera_items, scene_size=scene_size)
        except Exception as e:
            print(f"Failed to count overlapping cameras for {active_path}: {e}")
            return

        self.contextStatsComputed.emit(request_id, active_path, visible_count, overlap_count)

    def _on_context_stats_computed(self, request_id: int, active_path: str, visible_count: int, overlap_count: int):
        """Apply async overlap counts only if they belong to the latest active image."""
        if request_id != self._latest_context_stats_request_id:
            return

        if self.selected_camera is None or self.selected_camera.image_path != active_path:
            return

        if self.context_matrix is not None:
            self.context_matrix.update_stats_label(visible_count, overlap_count)

    def _update_context_stats(self):
        """Calculates overlap and pushes the string to the ContextMatrix UI."""
        if self.context_matrix is None or self.selected_camera is None:
            return

        # N: Total cameras visible in the matrix right now.
        n_visible = len(self._get_visible_context_camera_paths())

        self._context_stats_request_id += 1
        request_id = self._context_stats_request_id
        self._latest_context_stats_request_id = request_id

        active_camera = self.selected_camera
        active_path = active_camera.image_path
        camera_items = tuple(self.cameras.items())
        scene_size = self._get_scene_size_snapshot()

        try:
            future = self._propagation_executor.submit(
                self._count_overlapping_cameras_async,
                request_id,
                active_path,
                n_visible,
                active_camera,
                camera_items,
                scene_size,
            )
        except Exception:
            try:
                m_overlapping = self.count_overlapping_cameras(active_camera, camera_items=camera_items, scene_size=scene_size)
            except Exception:
                return
            self.contextStatsComputed.emit(request_id, active_path, n_visible, m_overlapping)

    def _build_projection(self, px: int, py: int) -> dict:
        """Cast a ray from the selected camera at (px, py) and return projections.

        Handles both perspective cameras and OrthoCamera (orthomosaic):
        - For perspective cameras: uses existing ray-projection logic
        - For OrthoCamera: converts orthomosaic pixel → geo → world space

        Returns:
            dict mapping image_path -> (u, v, is_valid), or empty dict on failure.
        """
        if self.selected_camera is None:
            return {}

        camera = self.selected_camera
        primary_target = self.viewer.scene_context.get_primary_target()
        ray = None

        # Special handling for OrthoCamera (orthomosaic)
        if self.ortho_camera is not None and camera == self.ortho_camera:
            try:
                if not camera.is_valid:
                    return {}

                # Convert orthomosaic pixel → geo → world space
                X, Y = camera.pixel_to_geo(px, py)
                Z = camera._raster.get_z_value(px, py)
                if Z is None or np.isnan(Z):
                    Z = 0.0

                world_pt = camera.geo_to_world(X, Y, Z)

                # Get element ID if index_map is available
                element_id = None
                index_map = camera._raster.index_map
                if index_map is not None and 0 <= px < camera.width and 0 <= py < camera.height:
                    sf = getattr(camera._raster, 'index_map_scale_factor', None)
                    map_px = int(px * sf) if sf else px
                    map_py = int(py * sf) if sf else py
                    map_px = min(map_px, index_map.shape[1] - 1)
                    map_py = min(map_py, index_map.shape[0] - 1)
                    candidate_id = int(index_map[map_py, map_px])
                    if candidate_id > -1:
                        element_id = candidate_id

                # Construct a vertical ray from the orthomosaic
                # The ray origin is slightly above the world point, direction points down
                vertical_dir = camera.get_vertical_direction_world()
                ray_origin = world_pt - vertical_dir * 0.1  # Slightly above the surface
                ray_direction = vertical_dir

                ray = CameraRay(
                    origin=ray_origin,
                    direction=ray_direction,
                    terminal_point=world_pt,
                    has_accurate_depth=True,
                    pixel_coord=(px, py),
                    source_camera=camera,
                    element_id=element_id
                )
            except Exception as e:
                print(f"Error building ortho projection: {e}")
                return {}
        else:
            # Standard perspective camera logic
            # --- PLAN A: Index Map (Flawless 3D Coordinate) ---
            index_map = camera._raster.index_map
            if index_map is not None and primary_target is not None:
                try:
                    # Ensure we are inside the image bounds
                    if 0 <= px < camera.width and 0 <= py < camera.height:
                        candidate_id = int(index_map[py, px])
                        if candidate_id > -1:
                            coord = primary_target.get_element_coordinate(candidate_id)
                            if coord is not None:
                                origin = camera.position.copy()
                                direction = coord - origin
                                norm = np.linalg.norm(direction)
                                direction = direction / norm if norm > 0 else camera.R.T @ np.array([0, 0, 1])

                                ray = CameraRay(
                                    origin=origin,
                                    direction=direction,
                                    terminal_point=coord,
                                    has_accurate_depth=True,
                                    pixel_coord=(px, py),
                                    source_camera=camera,
                                    element_id=candidate_id
                                )
                except Exception:
                    pass

            # --- PLAN B: Depth Map / Scene Median Fallback ---
            if ray is None:
                depth = None
                try:
                    raster = camera._raster
                    if raster.z_channel is not None and raster.z_data_type == 'depth':
                        depth = raster.get_z_value(px, py)
                except Exception:
                    pass

                # Cache median depth per camera — only recompute when active camera changes
                cache_key = id(camera)
                if getattr(self, '_median_depth_cache_key', None) != cache_key:
                    try:
                        self._cached_median_depth = self.viewer.get_scene_median_depth(camera.position)
                    except Exception:
                        self._cached_median_depth = 10.0
                    self._median_depth_cache_key = cache_key

                default_depth = self._cached_median_depth or 10.0

                try:
                    ray = CameraRay.from_pixel_and_camera(
                        pixel_xy=(px, py),
                        camera=camera,
                        depth=depth,
                        default_depth=default_depth,
                    )
                except Exception:
                    return {}

        if ray is None:
            return {}

        cameras_for_projection = self.cameras
        if self.ortho_camera is not None and camera != self.ortho_camera:
            cameras_for_projection = dict(self.cameras)
            cameras_for_projection[self.ortho_camera.image_path] = self.ortho_camera

        return ray.project_to_cameras(cameras_for_projection)

    def _on_patch_annotation_created(self, annotation_id: str):
        """Propagate a newly created PatchAnnotation into all target cameras (perspective and ortho-aware)."""
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

        selected_paths = self._get_annotation_target_paths()

        # Quick exit: nothing to propagate to
        if not selected_paths:
            return

        from PyQt5.QtCore import QPointF
        self._propagating_annotation = True
        try:
            # ------------------------------------------------------------------
            # Source element-ID extraction: sample a sparse grid within the
            # annotation bounding box so that get_pixels_for_elements has many
            # IDs to work with — not just the single center pixel.  More IDs
            # dramatically reduces stride false-negatives in the target cameras.
            # ------------------------------------------------------------------
            source_raster = getattr(self.selected_camera, '_raster', None)
            source_index_map = source_raster.index_map if source_raster is not None else None
            source_element_ids = None   # list[int] — passed to get_pixels_for_elements
            element_id = None           # center-pixel ID — used by _build_projection ray
            use_3d = False

            if source_index_map is not None:
                try:
                    sf = getattr(source_raster, 'index_map_scale_factor', None) or 1.0
                    img_h, img_w = source_index_map.shape
                    ann_size = annotation.annotation_size   # half-extent in image pixels

                    # Clamp the annotation bounding box to the index-map bounds
                    # (coordinates scaled by sf to match the index-map resolution).
                    x0 = max(0,       int((px - ann_size) * sf))
                    x1 = min(img_w,   int((px + ann_size) * sf) + 1)
                    y0 = max(0,       int((py - ann_size) * sf))
                    y1 = min(img_h,   int((py + ann_size) * sf) + 1)

                    if x0 < x1 and y0 < y1:
                        patch = source_index_map[y0:y1, x0:x1].ravel()
                        valid = patch[patch > -1]
                        if valid.size > 0:
                            source_element_ids = list(np.unique(valid).tolist())
                            # Prefer the exact center-pixel ID for the ray direction
                            cx = min(int(px * sf), img_w - 1)
                            cy = min(int(py * sf), img_h - 1)
                            center_eid = int(source_index_map[cy, cx])
                            element_id = center_eid if center_eid > -1 else source_element_ids[0]
                            use_3d = True
                except Exception:
                    pass

            # Lazy projection cache for fallback
            projections = None

            for target_path in selected_paths:

                target_camera = self._get_camera_for_path(target_path)
                if target_camera is None:
                    continue

                try:
                    placed = False

                    # ----------------------------------------------------------
                    # 3D centroid path: look up every sampled element ID in the
                    # target's index map and use the resulting pixel centroid.
                    # Falls through to 2D when the lookup returns empty (element
                    # too small / edge-on in target, or stride miss) rather than
                    # hard-skipping the camera.
                    # ----------------------------------------------------------
                    target_has_index = (getattr(target_camera, '_raster', None) is not None
                                        and target_camera._raster.index_map is not None)
                    if use_3d and target_has_index and source_element_ids:
                        flat = target_camera.get_pixels_for_elements(
                            np.array(source_element_ids, dtype=np.int64)
                        )
                        if flat.size > 0:
                            v_arr, u_arr = np.divmod(flat, target_camera.width)
                            u_centroid = float(np.mean(u_arr))
                            v_centroid = float(np.mean(v_arr))
                            if 0 <= u_centroid < target_camera.width and 0 <= v_centroid < target_camera.height:
                                new_annotation = PatchAnnotation(
                                    center_xy=QPointF(u_centroid, v_centroid),
                                    annotation_size=annotation.annotation_size,
                                    label=annotation.label,
                                    image_path=target_path,
                                    transparency=annotation.transparency,
                                )
                                try:
                                    self.annotation_window.add_annotation(new_annotation, record_action=True)
                                    placed = True
                                except Exception:
                                    pass

                    # ----------------------------------------------------------
                    # 2D fallback: used when no index map is available OR when
                    # the 3D lookup returned empty (element occluded / missed).
                    # ----------------------------------------------------------
                    if not placed:
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
                            label=annotation.label,
                            image_path=target_path,
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
            # Clean up the fake vector trails!
            if self.context_matrix is not None:
                try:
                    self.context_matrix.clear_all_scratchpads()
                except Exception:
                    pass

    # TODO Note: dense mesh hit fills in the face IDs when the quality of index map < Highest; otherwise VTK does this fine.
    # If we can find a way to not use Open3D always, then we don't need to calculate a BVH, which takes times to build.
    # Figure out how we can allow the user to use lower quality index maps, but still fill in the gaps.
    # 
    # Leagcy code; do not delete
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
        try:
            import open3d as o3d
        except ImportError:
            return np.array([], dtype=np.int32)

        try:
            # 1. Ensure the GPU tensor geometry cache exists (idempotent call).
            mesh_product.prepare_geometry()
            vertices  = mesh_product._cached_vertices                                      # (V, 3) float32
            # Use the CPU-cached triangle array to avoid GPU->CPU transfers
            # on every brush tick. The tensor copy was created for fast GPU ops
            # but we keep a numpy copy for fast CPU-side BVH/path queries.
            triangles = getattr(mesh_product, '_cached_triangles_np', None)
            if triangles is None:
                # Fallback: materialize from the tensor once
                triangles = mesh_product._cached_triangles_pt.cpu().numpy().astype(np.uint32)

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

    def _propagate_to_camera(self, target_path, painted_ids, target_class_id_map,
                              projections, brush_w, brush_h, brush_mask, use_3d):
        """Single-camera propagation — runs in thread pool, no Qt calls."""
        target_camera = self._get_camera_for_path(target_path)
        if target_camera is None:
            return target_path, False, None

        target_raster = self.raster_manager.get_raster(target_path)
        if target_raster is None:
            return target_path, False, None

        target_mask = target_raster.mask_annotation
        if target_mask is None:
            return target_path, False, None

        target_class_id = target_class_id_map.get(target_path)
        if target_class_id is None:
            return target_path, False, None

        target_has_index = target_camera._raster.index_map is not None

        if use_3d and target_has_index and target_camera is not self.ortho_camera:
            proj = projections.get(target_path)
            bbox = None
            if proj is not None and proj[2]:
                target_u, target_v = proj[0], proj[1]
                search_radius = max(brush_w, brush_h) * 2.5
                bbox = (target_u - search_radius, target_u + search_radius,
                        target_v - search_radius, target_v + search_radius)

            flat_indices = target_camera.get_pixels_for_elements(painted_ids, bbox=bbox)
            if len(flat_indices) == 0:
                return target_path, False, None

            if hasattr(target_mask, 'mask_data'):
                current_vals = target_mask.mask_data.ravel()[flat_indices]
                flat_indices = flat_indices[(current_vals < target_mask.LOCK_BIT) & 
                                            (current_vals != target_class_id)]
            if len(flat_indices) == 0:
                return target_path, False, None

            target_mask.update_mask_at_indices(flat_indices, target_class_id, silent=True)
            update_rect = self._compute_dirty_rect_from_flat_indices(
                flat_indices,
                target_camera.width,
                target_camera.height,
            )
        else:
            proj = projections.get(target_path)
            if proj is None:
                return target_path, False, None
            u, v, is_valid = proj
            if not is_valid or not (0 <= u < target_camera.width and 0 <= v < target_camera.height):
                return target_path, False, None
            brush_location = QPointF(u - brush_w / 2.0, v - brush_h / 2.0)
            target_mask.update_mask(brush_location, brush_mask, target_class_id, silent=True)
            x_start = max(0, int(u - brush_w / 2.0))
            y_start = max(0, int(v - brush_h / 2.0))
            update_rect = (
                x_start,
                y_start,
                min(target_camera.width, x_start + brush_w),
                min(target_camera.height, y_start + brush_h),
            )

        return target_path, True, update_rect

    def _on_brush_stroke_applied(self, scene_pos, label_id: str, brush_mask):
        """Propagate a brush stroke into all visible context cameras.

        When painting on an OrthoRaster/OrthoCamera, applies the brush to all visible
        perspective cameras that can see the same 3D geometry.

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

        selected_paths = self._get_annotation_target_paths()

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
        painted_ids = self._extract_source_ids_from_crop_mask(
            self.selected_camera,
            brush_mask,
            px,
            py,
        )

        # Whether the source camera has valid 3D geometry hits to propagate
        use_3d = painted_ids is not None and len(painted_ids) > 0

        # ------------------------------------------------------------------
        # Paint the 3D Model directly
        # ------------------------------------------------------------------
        if use_3d:  # Slow?
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
                    # 3. Paint the 3D model arrays — offload to background painter thread
                    self._ensure_label_painter(primary_target)
                    self._label_painter_thread.submit(painted_ids, target_color, source_class_id)

        # Projections for 2D fallback — computed lazily inside the loop
        projections = None

        self._propagating_annotation = True
        try:
            # Pre-resolve class IDs for ALL cameras before entering threads
            # (avoids repeated dict lookups and sync_label_map calls inside threads)
            target_class_id_map = {}
            for target_path in selected_paths:
                target_camera = self._get_camera_for_path(target_path)
                target_raster = self.raster_manager.get_raster(target_path)
                if target_raster is None:
                    continue
                target_mask = target_raster.mask_annotation
                if target_mask is None:
                    target_mask = target_raster.get_mask_annotation(project_labels)
                if target_mask is None:
                    continue
                class_id = target_mask.label_id_to_class_id_map.get(label_id)
                if class_id is None:
                    target_mask.sync_label_map([source_label])
                    class_id = target_mask.label_id_to_class_id_map.get(label_id)
                if class_id is not None:
                    target_class_id_map[target_path] = class_id
                    # Pre-warm color map cache to avoid cold cache hit in threads
                    target_mask._get_color_map()

            if projections is None:
                projections = self._build_projection(px, py)

            futures = {
                self._propagation_executor.submit(
                    self._propagate_to_camera,
                    target_path, painted_ids, target_class_id_map,
                    projections, brush_w, brush_h, brush_mask, use_3d
                ): target_path
                for target_path in target_class_id_map
            }
            for future in as_completed(futures):
                target_path, did_update, update_rect = future.result()
                if did_update:
                    target_raster = self.raster_manager.get_raster(target_path)
                    if target_raster and target_raster.mask_annotation:
                        target_mask = target_raster.mask_annotation
                        self._apply_mask_visual_update(
                            target_path,
                            target_mask,
                            label_id,
                            update_rect=update_rect,
                        )
        finally:
            self._propagating_annotation = False

    def _on_fill_stroke_applied(self, scene_pos, label_id: str, fill_mask=None):
        """Propagate a fill operation into all visible context cameras.

        When painting on an OrthoRaster/OrthoCamera, applies the fill to all visible
        perspective cameras that can see the same 3D geometry.

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

        selected_paths = self._get_annotation_target_paths()

        project_labels = list(self.main_window.label_window.labels)

        source_label = next((lbl for lbl in project_labels if lbl.id == label_id), None)
        if source_label is None:
            return

        # ------------------------------------------------------------------
        # Phase 1: Source ID Extraction (2D → 3D)
        # ------------------------------------------------------------------
        painted_ids = None
        if fill_mask is not None:
            painted_ids = self._extract_source_ids_from_full_mask(self.selected_camera, fill_mask)

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
                    
                    # 3. Paint the 3D model arrays — offload to background painter thread
                    self._ensure_label_painter(primary_target)
                    self._label_painter_thread.submit(painted_ids, target_color, source_class_id)
        
        # Projections for 2D fallback — computed lazily
        projections = None
        
        self._propagating_annotation = True
        try:
            # 1. Pre-resolve class IDs and sort cameras into 3D/2D buckets
            target_class_id_map = {}
            target_cameras_3d = []
            target_cameras_2d = []

            for target_path in selected_paths:
                target_camera = self._get_camera_for_path(target_path)
                if not target_camera: continue
                
                target_raster = self.raster_manager.get_raster(target_path)
                if not target_raster: continue
                
                # --- OPTIMIZATION 1: Bypass forced sync_label_map ---
                target_mask = target_raster.mask_annotation
                if target_mask is None:
                    target_mask = target_raster.get_mask_annotation(project_labels)
                if target_mask is None: continue
                
                target_class_id = target_mask.label_id_to_class_id_map.get(label_id)
                if target_class_id is None:
                    target_mask.sync_label_map([source_label])
                    target_class_id = target_mask.label_id_to_class_id_map.get(label_id)
                if target_class_id is None: continue
                
                target_class_id_map[target_path] = target_class_id
                
                target_has_index = target_camera._raster is not None and target_camera._raster.index_map is not None
                if use_3d and target_has_index:
                    target_cameras_3d.append((target_path, target_camera))
                else:
                    target_cameras_2d.append((target_path, target_camera))

            # 2. THREADED 3D INDEX SEARCH
            # Throw the massive 16MP array searches into the background pool
            from concurrent.futures import as_completed
            futures = {}
            
            if use_3d and isinstance(painted_ids, np.ndarray) and len(painted_ids) > 0:
                for target_path, target_camera in target_cameras_3d:
                    future = self._propagation_executor.submit(
                        target_camera.get_pixels_for_elements, painted_ids
                    )
                    futures[future] = target_path

            # 3. Process 2D Fallbacks immediately on the Main Thread 
            # (cv2.floodFill is fast enough to do inline while background threads compute)
            if target_cameras_2d:
                projections = self._build_projection(px, py)
                for target_path, target_camera in target_cameras_2d:
                    proj = projections.get(target_path)
                    if proj is None or not proj[2]: continue
                    u, v, is_valid = proj
                    
                    if 0 <= u < target_camera.width and 0 <= v < target_camera.height:
                        target_class_id = target_class_id_map[target_path]
                        target_raster = self.raster_manager.get_raster(target_path)
                        target_mask = target_raster.mask_annotation
                        
                        from PyQt5.QtCore import QPointF
                        fill_pos = QPointF(u, v)
                        fill_result = target_mask.fill_region(
                            fill_pos,
                            target_class_id,
                            silent=True,
                            return_update_rect=True,
                        )
                        if fill_result is None:
                            continue
                        fill_mask_result, fill_rect = fill_result
                        if fill_mask_result is None:
                            continue
                        self._apply_mask_visual_update(
                            target_path,
                            target_mask,
                            label_id,
                            update_rect=fill_rect,
                        )

            # 4. Catch 3D Thread results and repaint
            for future in as_completed(futures):
                target_path = futures[future]
                flat_indices = future.result()
                target_class_id = target_class_id_map[target_path]
                
                if len(flat_indices) > 0:
                    target_raster = self.raster_manager.get_raster(target_path)
                    target_mask = target_raster.mask_annotation
                    
                    # --- OPTIMIZATION 2: Pixel Diffing ---
                    if hasattr(target_mask, 'mask_data'):
                        current_vals = target_mask.mask_data.ravel()[flat_indices]
                        changed_mask = current_vals != target_class_id
                        flat_indices = flat_indices[changed_mask]
                        
                        if len(flat_indices) > 0:
                            target_mask.update_mask_at_indices(flat_indices, target_class_id, silent=True)
                            dirty_rect = self._compute_dirty_rect_from_flat_indices(
                                flat_indices,
                                target_mask.mask_data.shape[1],
                                target_mask.mask_data.shape[0],
                            )
                            self._apply_mask_visual_update(
                                target_path,
                                target_mask,
                                label_id,
                                update_rect=dirty_rect,
                            )

        except Exception as e:
            print(f"Error in multi-annotate fill: {e}")
        finally:
            self._propagating_annotation = False

    def _on_erase_stroke_applied(self, scene_pos, label_id: str, brush_mask: np.ndarray):
        """Propagate an erase operation into all visible context cameras.

        When painting on an OrthoRaster/OrthoCamera, applies the erase to all visible
        perspective cameras that can see the same 3D geometry.

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

        selected_paths = self._get_annotation_target_paths()

        # Quick exit: nothing to propagate to
        if not selected_paths:
            return

        brush_h, brush_w = brush_mask.shape

        # ------------------------------------------------------------------
        # Phase 1: Source ID Extraction (2D → 3D)
        # ------------------------------------------------------------------
        painted_ids = self._extract_source_ids_from_crop_mask(self.selected_camera, brush_mask, px, py)

        use_3d = painted_ids is not None and len(painted_ids) > 0

        # ------------------------------------------------------------------
        # Phase 2: Reset the 3D Model to default (white / class_id 0)
        # ------------------------------------------------------------------
        if use_3d:
            primary_target = self.viewer.scene_context.get_primary_target()
            if primary_target and hasattr(primary_target, 'apply_labels'):
                self._ensure_label_painter(primary_target)
                self._label_painter_thread.submit(painted_ids, (255, 255, 255), 0)

        # Projections for 2D fallback — computed lazily inside the loop
        projections = None

        self._propagating_annotation = True
        try:
            # ERASER FIX: Map all target cameras to Class ID 0 (background)
            target_class_id_map = {target_path: 0 for target_path in selected_paths}

            if projections is None:
                projections = self._build_projection(px, py)

            # Spawn the background workers using the exact same logic as BrushTool
            from concurrent.futures import as_completed
            futures = {
                self._propagation_executor.submit(
                    self._propagate_to_camera,
                    target_path, painted_ids, target_class_id_map,
                    projections, brush_w, brush_h, brush_mask, use_3d
                ): target_path
                for target_path in selected_paths
            }
            
            # Catch the results on the Main Thread and repaint
            for future in as_completed(futures):
                target_path, did_update = future.result()

                if did_update:
                    target_raster = self.raster_manager.get_raster(target_path)
                    if target_raster and target_raster.mask_annotation:
                        target_mask = target_raster.mask_annotation

                        # The silent update in _propagate_to_camera already wrote to
                        # colored_mask, so only a lightweight Qt repaint is needed here —
                        # not a full _update_full_canvas() rebuild.
                        self._apply_mask_visual_update(
                            target_path,
                            target_mask,
                            label_id,
                        )

        except Exception as e:
            print(f"Error in multi-annotate erase: {e}")
        finally:
            self._propagating_annotation = False
            
            # Clean up the live vector scratchpads across all cameras
            if self.context_matrix is not None:
                self.context_matrix.clear_all_scratchpads()

    def _on_sam_prediction_applied(self, scene_pos, label_id: str, binary_mask: np.ndarray):
        """Propagate a final SAM mask prediction into all visible context cameras.

        When painting on an OrthoRaster/OrthoCamera, applies the SAM prediction to all visible
        perspective cameras that can see the same 3D geometry.
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

        self._propagating_annotation = True

        # Snapshot all main-thread state the background worker needs.
        # Do NOT read self.selected_camera or self.ortho_camera inside the worker.
        selected_camera = self.selected_camera
        selected_paths  = self._get_annotation_target_paths()
        project_labels  = list(self.main_window.label_window.labels)
        primary_target  = self.viewer.scene_context.get_primary_target()
        is_ortho_source = (self.ortho_camera is not None and
                           selected_camera is self.ortho_camera)
        ortho_camera    = self.ortho_camera  # captured for bucket test in worker

        self._sam_bg_executor.submit(
            self._do_sam_propagation,
            scene_pos, label_id, binary_mask,
            selected_camera, selected_paths, project_labels,
            primary_target, is_ortho_source, ortho_camera,
        )

    # ------------------------------------------------------------------
    # SAM background worker  (runs off the main thread)
    # ------------------------------------------------------------------

    def _do_sam_propagation(
        self,
        scene_pos, label_id: str, binary_mask: np.ndarray,
        selected_camera, selected_paths, project_labels,
        primary_target, is_ortho_source: bool, ortho_camera,
    ):
        """Background worker for SAM prediction propagation.

        Runs entirely off the main thread.  All Qt UI updates are deferred and
        executed on the main thread via ``_sam_repaint_signal``.
        """
        repaint_tasks = []   # UI update tasks flushed to main thread via signal

        try:
            px = int(scene_pos.x())
            py = int(scene_pos.y())

            source_label = next(
                (lbl for lbl in project_labels if lbl.id == label_id), None)
            if source_label is None:
                return

            mask_h, mask_w = binary_mask.shape

            # ------------------------------------------------------------------
            # Phase 1: Source ID Extraction (2D → 3D)
            # ------------------------------------------------------------------
            painted_ids = None
            if is_ortho_source:
                painted_ids = self._extract_source_ids_from_crop_mask(
                    selected_camera,
                    binary_mask.astype(bool),
                    px, py,
                )
            else:
                import cv2
                source_index_map = selected_camera._raster.index_map
                if source_index_map is not None:
                    x0 = px - mask_w // 2
                    y0 = py - mask_h // 2
                    x1 = x0 + mask_w
                    y1 = y0 + mask_h
                    img_h, img_w = source_index_map.shape

                    if x0 < img_w and y0 < img_h and x1 > 0 and y1 > 0:
                        cx0 = max(x0, 0);  cy0 = max(y0, 0)
                        cx1 = min(x1, img_w); cy1 = min(y1, img_h)
                        bx0 = cx0 - x0;    by0 = cy0 - y0
                        bx1 = bx0 + (cx1 - cx0); by1 = by0 + (cy1 - cy0)

                        index_slice = source_index_map[cy0:cy1, cx0:cx1]
                        mask_clip   = binary_mask[by0:by1, bx0:bx1]
                        valid_mask  = mask_clip.astype(bool)

                        source_depth_map = selected_camera._raster.z_channel
                        if source_depth_map is not None:
                            depth_slice = source_depth_map[cy0:cy1, cx0:cx1]

                            # Gradient depth filtering at the mask perimeter
                            erosion_r = int(np.clip(min(mask_clip.shape) * 0.03, 2, 12))
                            kernel = cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE, (2 * erosion_r + 1, 2 * erosion_r + 1))
                            interior_mask = cv2.erode(
                                valid_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                            perimeter_mask = valid_mask & ~interior_mask

                            interior_depths = depth_slice[interior_mask]
                            interior_depths = interior_depths[~np.isnan(interior_depths)]

                            if len(interior_depths) >= 10 and perimeter_mask.any():
                                ref_depth       = np.median(interior_depths)
                                interior_spread = np.std(interior_depths)
                                abs_floor       = max(0.02, ref_depth * 0.005)
                                full_tol        = interior_spread * 2.0 + abs_floor
                                dist = cv2.distanceTransform(
                                    valid_mask.astype(np.uint8), cv2.DIST_L2, 5)
                                norm_dist = np.clip(dist / max(erosion_r, 1), 0.0, 1.0)
                                per_pixel_tol = abs_floor + (full_tol - abs_floor) * norm_dist
                                with np.errstate(invalid='ignore'):
                                    perimeter_depth_ok = (
                                        np.abs(depth_slice - ref_depth) <= per_pixel_tol)
                                valid_mask = interior_mask | (perimeter_mask & perimeter_depth_ok)

                        raw_ids = index_slice[valid_mask]
                        unique_ids = np.unique(raw_ids)
                        painted_ids = unique_ids[unique_ids > -1]

            use_3d = painted_ids is not None and len(painted_ids) > 0

            # ------------------------------------------------------------------
            # Queue the 3D Model paint — executed on main thread via signal
            # (_ensure_label_painter / QThread.start must be on the main thread)
            # ------------------------------------------------------------------
            if use_3d and primary_target and hasattr(primary_target, 'apply_labels'):
                target_color  = (source_label.color.red(),
                                 source_label.color.green(),
                                 source_label.color.blue())
                source_raster = self.raster_manager.get_raster(selected_camera.image_path)
                source_mask   = (source_raster.get_mask_annotation(project_labels)
                                 if source_raster else None)
                if source_mask:
                    source_class_id = source_mask.label_id_to_class_id_map.get(label_id)
                    if source_class_id is None:
                        source_mask.sync_label_map([source_label])
                        source_class_id = source_mask.label_id_to_class_id_map.get(label_id)
                    if source_class_id is not None:
                        repaint_tasks.append({
                            'type': '3d_paint',
                            'painted_ids':    painted_ids.copy(),
                            'target_color':   target_color,
                            'source_class_id': source_class_id,
                            'primary_target': primary_target,
                        })

            # ------------------------------------------------------------------
            # 1. Pre-resolve class IDs and sort cameras into 3D / 2D buckets
            # ------------------------------------------------------------------
            target_class_id_map = {}
            target_cameras_3d   = []
            target_cameras_2d   = []

            for target_path in selected_paths:
                target_camera = self._get_camera_for_path(target_path)
                if not target_camera: continue
                target_raster = self.raster_manager.get_raster(target_path)
                if not target_raster: continue
                target_mask = target_raster.mask_annotation
                if target_mask is None:
                    target_mask = target_raster.get_mask_annotation(project_labels)
                if target_mask is None: continue
                target_class_id = target_mask.label_id_to_class_id_map.get(label_id)
                if target_class_id is None:
                    target_mask.sync_label_map([source_label])
                    target_class_id = target_mask.label_id_to_class_id_map.get(label_id)
                if target_class_id is None: continue
                target_class_id_map[target_path] = target_class_id
                target_has_index = (target_camera._raster is not None and
                                    target_camera._raster.index_map is not None)
                if use_3d and target_has_index:
                    target_cameras_3d.append((target_path, target_camera))
                else:
                    target_cameras_2d.append((target_path, target_camera))

            # ------------------------------------------------------------------
            # 2. Submit per-camera 3D index searches
            # ------------------------------------------------------------------
            from concurrent.futures import as_completed
            futures    = {}
            projections = None  # computed lazily — shared by 3D bbox and 2D fallback

            if use_3d and len(painted_ids) > 0:
                projections = self._build_projection(px, py)
                for target_path, target_camera in target_cameras_3d:
                    proj = projections.get(target_path)
                    bbox = None
                    if proj is not None and proj[2]:
                        target_u, target_v = proj[0], proj[1]
                        search_radius = max(mask_w, mask_h) * 2.5
                        bbox = (target_u - search_radius, target_u + search_radius,
                                target_v - search_radius, target_v + search_radius)
                    future = self._propagation_executor.submit(
                        target_camera.get_pixels_for_elements, painted_ids, bbox=bbox)
                    futures[future] = target_path

            # ------------------------------------------------------------------
            # 3. Process 2D fallbacks — numpy writes here, Qt repaints via signal
            # ------------------------------------------------------------------
            if target_cameras_2d:
                if projections is None:
                    projections = self._build_projection(px, py)
                for target_path, target_camera in target_cameras_2d:
                    proj = projections.get(target_path)
                    if proj is None or not proj[2]: continue
                    u, v, _ = proj
                    if 0 <= u < target_camera.width and 0 <= v < target_camera.height:
                        target_class_id   = target_class_id_map[target_path]
                        target_raster     = self.raster_manager.get_raster(target_path)
                        target_mask_obj   = target_raster.mask_annotation
                        subset_class_mask = binary_mask.astype(np.uint8) * int(target_class_id)
                        top_left_x = int(u - mask_w / 2.0)
                        top_left_y = int(v - mask_h / 2.0)
                        # Pure numpy write — safe from background thread (silent=True)
                        target_mask_obj.update_mask_with_mask(
                            subset_class_mask, (top_left_x, top_left_y), silent=True)
                        # Compute tight dirty rect so the slot does a localized repaint
                        mh, mw = subset_class_mask.shape
                        dirty = (
                            max(0, top_left_x),
                            max(0, top_left_y),
                            min(target_mask_obj.mask_data.shape[1], top_left_x + mw),
                            min(target_mask_obj.mask_data.shape[0], top_left_y + mh),
                        )
                        repaint_tasks.append({
                            'type':        'repaint',
                            'path':        target_path,
                            'mask':        target_mask_obj,
                            'label_id':    label_id,
                            'update_rect': dirty,
                        })

            # ------------------------------------------------------------------
            # 4. Collect 3D thread results — numpy writes here, Qt via signal
            # ------------------------------------------------------------------
            for future in as_completed(futures):
                target_path     = futures[future]
                flat_indices    = future.result()
                target_class_id = target_class_id_map[target_path]
                if len(flat_indices) > 0:
                    target_raster = self.raster_manager.get_raster(target_path)
                    target_mask_obj = target_raster.mask_annotation
                    if hasattr(target_mask_obj, 'mask_data'):
                        current_vals = target_mask_obj.mask_data.ravel()[flat_indices]
                        changed_mask = current_vals != target_class_id
                        flat_indices = flat_indices[changed_mask]
                        if len(flat_indices) > 0:
                            target_mask_obj.update_mask_at_indices(
                                flat_indices, target_class_id, silent=True)
                            # Compute dirty rect from the actual changed pixels
                            mask_w_full = target_mask_obj.mask_data.shape[1]
                            y_c, x_c = np.divmod(flat_indices, mask_w_full)
                            dirty = (
                                max(0, int(x_c.min()) - 1),
                                max(0, int(y_c.min()) - 1),
                                min(mask_w_full, int(x_c.max()) + 2),
                                min(target_mask_obj.mask_data.shape[0], int(y_c.max()) + 2),
                            )
                            repaint_tasks.append({
                                'type':        'repaint',
                                'path':        target_path,
                                'mask':        target_mask_obj,
                                'label_id':    label_id,
                                'update_rect': dirty,
                            })

        except Exception as e:
            import traceback; traceback.print_exc()
        finally:
            # Always emit — the slot is responsible for clearing _propagating_annotation
            self._sam_repaint_signal.emit(repaint_tasks)

    # ------------------------------------------------------------------
    # SAM repaint slot  (runs on main thread via Qt.QueuedConnection)
    # ------------------------------------------------------------------

    def _on_sam_repaint(self, repaint_tasks: list):
        """Apply all deferred Qt UI updates produced by ``_do_sam_propagation``.

        Called on the main thread via ``_sam_repaint_signal``.  Executes
        localized repaints using the dirty rects computed in the worker,
        avoiding full-canvas rebuilds on large ortho masks.
        """
        try:
            for task in repaint_tasks:
                task_type = task.get('type')

                if task_type == '3d_paint':
                    # _ensure_label_painter starts a QThread — main thread only
                    self._ensure_label_painter(task['primary_target'])
                    if self._label_painter_thread is not None:
                        self._label_painter_thread.submit(
                            task['painted_ids'],
                            task['target_color'],
                            task['source_class_id'],
                        )

                elif task_type == 'repaint':
                    target_mask = task['mask']
                    label_id    = task['label_id']
                    update_rect = task.get('update_rect')
                    target_path = task['path']

                    if label_id not in target_mask.visible_label_ids:
                        target_mask.visible_label_ids.add(label_id)

                    # Localized repaint — avoids full np.copyto on gigapixel masks
                    target_mask.update_graphics_item(update_rect=update_rect)

                    context_canvas = self._get_context_canvas_for_path(target_path)
                    if context_canvas is not None and context_canvas._mask_overlay_item is None:
                        context_canvas.set_mask_overlay(target_mask)

        except Exception as e:
            print(f"Error in _on_sam_repaint: {e}")
        finally:
            self._propagating_annotation = False
            if self.context_matrix is not None:
                self.context_matrix.clear_all_scratchpads()

    def _on_semantic_prediction_applied(self, image_path: str, source_mask_annotation, prediction_regions=None):
        """Propagate a semantic segmentation prediction to all target cameras.

        Called by Semantic.predict() after each image is processed when multi-annotate
        is enabled. When region payloads are supplied, only those tiles contribute
        to the 3D element votes, which keeps work-area predictions scoped to the
        predicted region instead of the entire source mask.

        Perspective ↔ orthomosaic propagation is supported: when the source is an
        OrthoCamera, targets are all visible perspective cameras; when the source is a
        perspective camera, targets include visible context cameras plus the ortho.

        Cameras without a pre-computed index map are skipped — full-image warping
        without geometry is too imprecise to be useful for semantic masks.

        Args:
            image_path: Path of the image whose Semantic prediction just completed.
            source_mask_annotation: The MaskAnnotation whose mask_data was just
                                    updated by the Semantic model.
        """
        if self._propagating_annotation:
            return

        source_camera = self._get_camera_for_path(image_path)
        if source_camera is None:
            return

        selected_paths = self._get_semantic_target_paths(source_camera)

        if not selected_paths:
            return

        project_labels = list(self.main_window.label_window.labels)
        semantic_mask = source_mask_annotation.mask_data
        LOCK_BIT = source_mask_annotation.LOCK_BIT
        source_index_map = getattr(getattr(source_camera, '_raster', None), 'index_map', None)
        if source_index_map is None:
            return

        # Phase 1: Build per-element class votes on the main thread.
        # Each source element is assigned to the dominant semantic class once,
        # so we do not end up painting the same element with multiple classes.
        element_votes = {}
        class_labels = {}
        if prediction_regions is not None:
            for region_mask, top_left in prediction_regions:
                if region_mask is None:
                    continue

                region_mask = np.asarray(region_mask)
                if region_mask.ndim != 2:
                    continue

                unique_real_ids = np.unique(region_mask % LOCK_BIT)
                unique_real_ids = unique_real_ids[unique_real_ids > 0]
                if len(unique_real_ids) == 0:
                    continue

                for real_class_id in unique_real_ids:
                    label = source_mask_annotation.class_id_to_label_map.get(int(real_class_id))
                    if label is None:
                        continue
                    class_labels[int(real_class_id)] = label
                    binary_mask = (region_mask % LOCK_BIT == real_class_id).astype(bool)
                    if not np.any(binary_mask):
                        continue

                    source_element_ids = self._extract_source_element_ids_from_region(
                        source_camera,
                        binary_mask,
                        top_left,
                    )
                    if source_element_ids is None or len(source_element_ids) == 0:
                        continue

                    unique_element_ids, counts = np.unique(source_element_ids, return_counts=True)
                    for element_id, count in zip(unique_element_ids.tolist(), counts.tolist()):
                        vote_map = element_votes.setdefault(int(element_id), {})
                        vote_map[int(real_class_id)] = vote_map.get(int(real_class_id), 0) + int(count)
        else:
            # Unique real class IDs in the prediction (lock bit stripped, skip background)
            unique_real_ids = np.unique(semantic_mask % LOCK_BIT)
            unique_real_ids = unique_real_ids[unique_real_ids > 0]
            if len(unique_real_ids) == 0:
                return

            for real_class_id in unique_real_ids:
                label = source_mask_annotation.class_id_to_label_map.get(int(real_class_id))
                if label is None:
                    continue
                class_labels[int(real_class_id)] = label
                binary_mask = (semantic_mask % LOCK_BIT == real_class_id).astype(bool)
                if not np.any(binary_mask):
                    continue

                source_element_ids = self._extract_source_element_ids_from_full_mask(source_camera, binary_mask)
                if source_element_ids is None or len(source_element_ids) == 0:
                    continue

                unique_element_ids, counts = np.unique(source_element_ids, return_counts=True)
                for element_id, count in zip(unique_element_ids.tolist(), counts.tolist()):
                    vote_map = element_votes.setdefault(int(element_id), {})
                    vote_map[int(real_class_id)] = vote_map.get(int(real_class_id), 0) + int(count)

        if not element_votes:
            return

        class_data = {}
        for element_id, vote_map in element_votes.items():
            winner_class_id = max(vote_map.items(), key=lambda item: (item[1], -item[0]))[0]
            class_data.setdefault(winner_class_id, []).append(element_id)


        class_data = {
            class_id: (
                np.asarray(sorted(set(element_ids)), dtype=np.int64),
                class_labels.get(class_id),
            )
            for class_id, element_ids in class_data.items()
            if class_labels.get(class_id) is not None
        }

        if not class_data:
            return

        # Phase 2: Paint the scene mesh directly so semantic batches do not
        # lose classes to the live-stroke queue coalescing behavior.
        primary_target = self.viewer.scene_context.get_primary_target()
        if primary_target and hasattr(primary_target, 'apply_labels'):
            for real_class_id, (element_ids, label) in class_data.items():
                if element_ids is None or len(element_ids) == 0:
                    continue
                target_color = (label.color.red(), label.color.green(), label.color.blue())
                try:
                    primary_target.apply_labels(element_ids, real_class_id, target_color)
                except Exception:
                    continue

            try:
                primary_target.flush_labels_to_gpu()
            except Exception:
                pass

            try:
                self.viewer.plotter.render()
            except Exception:
                pass

            if isinstance(primary_target, MeshProduct):
                try:
                    mesh = primary_target.get_render_mesh()
                    class_ids = getattr(primary_target, 'class_ids', None)
                    labels_cache = getattr(primary_target, '_labels_cache', None)
                    if mesh is not None and class_ids is not None and labels_cache is not None:
                        painted_faces = np.where(class_ids != 0)[0]
                        if len(painted_faces) > 0:
                            mesh_faces_flat = np.asarray(mesh.faces.reshape(-1, 4), dtype=np.int32)
                            mesh_points = np.asarray(mesh.points, dtype=np.float32)

                            selected = mesh_faces_flat[painted_faces, 1:]
                            unique_vids, inverse = np.unique(selected, return_inverse=True)
                            overlay_points = mesh_points[unique_vids]
                            remapped = inverse.reshape(selected.shape)
                            vtk_faces = np.hstack([
                                np.full((len(painted_faces), 1), 3, dtype=np.int32),
                                remapped.astype(np.int32),
                            ]).ravel()

                            colors = np.asarray(labels_cache[painted_faces], dtype=np.uint8)
                            self._on_overlay_ready((overlay_points, vtk_faces, colors))
                except Exception:
                    pass

        self._propagating_annotation = True
        try:
            # Phase 3: Resolve target masks and class ID maps — only cameras with
            # index maps can receive accurate full-image 3D propagation.
            target_masks = {}        # target_path -> MaskAnnotation
            target_cameras_map = {}  # target_path -> Camera
            # (target_path, real_class_id) -> target_class_id
            target_class_id_map = {}

            for target_path in list(selected_paths):
                target_camera = self._get_camera_for_path(target_path)
                if target_camera is None:
                    continue
                if (target_camera._raster is None or
                        target_camera._raster.index_map is None):
                    continue  # Skip cameras without geometry data

                target_raster = self.raster_manager.get_raster(target_path)
                if target_raster is None:
                    continue
                target_mask = target_raster.mask_annotation
                if target_mask is None:
                    target_mask = target_raster.get_mask_annotation(project_labels)
                if target_mask is None:
                    continue

                target_masks[target_path] = target_mask
                target_cameras_map[target_path] = target_camera

                for real_class_id, (_, label) in class_data.items():
                    t_class_id = target_mask.label_id_to_class_id_map.get(label.id)
                    if t_class_id is None:
                        target_mask.sync_label_map([label])
                        t_class_id = target_mask.label_id_to_class_id_map.get(label.id)
                    if t_class_id is not None:
                        target_class_id_map[(target_path, real_class_id)] = t_class_id
                        target_mask._get_color_map()  # Pre-warm cache

            if not target_masks:
                return

            # Phase 4: Dispatch one future per target camera, batching ALL classes
            # together so that get_pixels_for_elements is never called concurrently
            # on the same camera object.
            def _lookup_classes_for_camera(camera, class_items):
                out = {}
                for real_class_id, element_ids in class_items:
                    out[real_class_id] = camera.get_pixels_for_elements(element_ids)
                return out

            from concurrent.futures import as_completed
            futures = {}
            for target_path, target_camera in target_cameras_map.items():
                class_items = [
                    (real_class_id, element_ids)
                    for real_class_id, (element_ids, _) in class_data.items()
                    if (target_path, real_class_id) in target_class_id_map
                    and element_ids is not None and len(element_ids) > 0
                ]
                if not class_items:
                    continue
                future = self._propagation_executor.submit(
                    _lookup_classes_for_camera, target_camera, class_items
                )
                futures[future] = target_path

            # Phase 5: Collect results and apply masks (silent, Qt update deferred)
            updated_targets = {}
            for future in as_completed(futures):
                target_path = futures[future]
                try:
                    class_pixel_map = future.result()
                except Exception:
                    continue

                target_mask = target_masks[target_path]

                for real_class_id, flat_indices in class_pixel_map.items():
                    if flat_indices is None or len(flat_indices) == 0:
                        continue

                    target_class_id = target_class_id_map.get((target_path, real_class_id))
                    if target_class_id is None:
                        continue

                    current_vals = target_mask.mask_data.ravel()[flat_indices]
                    changed_indices = flat_indices[
                        (current_vals < LOCK_BIT) & (current_vals != target_class_id)
                    ]
                    if len(changed_indices) == 0:
                        continue

                    target_mask.update_mask_at_indices(
                        changed_indices, target_class_id, silent=True
                    )

                    dirty_rect = self._compute_dirty_rect_from_flat_indices(
                        changed_indices,
                        target_mask.mask_data.shape[1],
                        target_mask.mask_data.shape[0],
                    )
                    existing_rect = updated_targets.get(target_path)
                    if existing_rect is None:
                        updated_targets[target_path] = dirty_rect
                    elif dirty_rect is not None:
                        updated_targets[target_path] = (
                            min(existing_rect[0], dirty_rect[0]),
                            min(existing_rect[1], dirty_rect[1]),
                            max(existing_rect[2], dirty_rect[2]),
                            max(existing_rect[3], dirty_rect[3]),
                        )

                    label = class_data[real_class_id][1]
                    if label.id not in target_mask.visible_label_ids:
                        target_mask.visible_label_ids.add(label.id)

            # Phase 6: Qt graphics update per modified target (main thread)
            for target_path, update_rect in updated_targets.items():
                target_mask = target_masks[target_path]
                self._apply_mask_visual_update(
                    target_path,
                    target_mask,
                    None,
                    update_rect=update_rect,
                )

        except Exception as e:
            print(f"Error in multi-annotate semantic propagation: {e}")
            traceback.print_exc()
        finally:
            self._propagating_annotation = False

    def cleanup(self):
        """Clean up resources before closing."""
        self._on_multi_annotate_toggled(False)  # Disconnect all propagation hooks
        self.mouse_bridge.cleanup()

        # Stop label painter thread if running
        try:
            if self._label_painter_thread is not None:
                try:
                    self._label_painter_thread.stop()
                    self._label_painter_thread.wait(1000)
                except Exception:
                    pass
                self._label_painter_thread = None
        except Exception:
            pass

        # Shutdown propagation thread pool
        try:
            if hasattr(self, '_propagation_executor'):
                self._propagation_executor.shutdown(wait=False)
        except Exception:
            pass

        # Shutdown SAM background executor
        try:
            if hasattr(self, '_sam_bg_executor'):
                self._sam_bg_executor.shutdown(wait=False)
        except Exception:
            pass

        # Remove overlay actor if present
        try:
            if self._label_overlay_actor is not None:
                try:
                    self.viewer.plotter.remove_actor(self._label_overlay_actor, render=False)
                except Exception:
                    pass
                self._label_overlay_actor = None
        except Exception:
            pass

        try:
            self.clear_sphere_hover_overlay(reset_context=True, render=False)
        except Exception:
            pass

        if hasattr(self.viewer, 'close'):
            self.viewer.close()
