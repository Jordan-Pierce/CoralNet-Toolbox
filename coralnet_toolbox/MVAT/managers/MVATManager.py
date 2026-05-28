"""
MultiView Annotation Tool (MVAT) Manager

The central controller for the MVAT workspace.
Handles the business logic, data synchronization, and signal routing between 
the MainWindow, RasterManager, MVATViewer (3D), and ContextMatrix (2D).
"""

import os
import time
import threading
import numpy as np
import traceback
from time import perf_counter
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from PyQt5.QtCore import QObject, QTimer, pyqtSignal, Qt, QThread, QPointF
from PyQt5.QtWidgets import QApplication, QMessageBox

from coralnet_toolbox.MVAT.core.Cameras import Camera
from coralnet_toolbox.MVAT.core.Ray import CameraRay

from coralnet_toolbox.MVAT.managers.SelectionManager import SelectionManager
from coralnet_toolbox.MVAT.managers.VisibilityManager import VisibilityManager
from coralnet_toolbox.MVAT.workers.VisibilityWorker import VisibilityWorker
from coralnet_toolbox.MVAT.managers.CacheManager import CacheManager
from coralnet_toolbox.MVAT.workers.LabelWorker import LabelWorker
from coralnet_toolbox.MVAT.utils.MVATLogger import (
    get_visibility_logger,
    log_cam_stage,
)

from coralnet_toolbox.MVAT.core.constants import (
    MARKER_COLOR_SELECTED,
    MARKER_COLOR_HIGHLIGHTED,
    MARKER_COLOR_INVALID,
    RAY_COLOR_SELECTED,
    RAY_COLOR_HIGHLIGHTED,
    RAY_COLOR_INVALID,
)

from coralnet_toolbox.MVAT.core.Products import MeshProduct

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation


logger = get_visibility_logger()


def resolve_class_conflicts_vectorized(element_ids: np.ndarray, class_ids: np.ndarray):
    """Resolve per-element class conflicts using vectorized vote counts."""
    try:
        element_ids = np.asarray(element_ids, dtype=np.int64).ravel()
        class_ids = np.asarray(class_ids, dtype=np.int64).ravel()
    except Exception:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if element_ids.size == 0 or class_ids.size == 0 or element_ids.size != class_ids.size:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    max_classes = max(100000, int(np.max(class_ids)) + 1)
    compound_ids = (element_ids * max_classes) + class_ids

    unique_compounds, vote_counts = np.unique(compound_ids, return_counts=True)
    unique_elements = unique_compounds // max_classes
    unique_classes = unique_compounds % max_classes

    # Within each element group, keep the highest vote count and prefer the
    # smaller class ID when the vote count is tied.
    sort_indices = np.lexsort((-unique_classes, vote_counts, unique_elements))
    sorted_elements = unique_elements[sort_indices]
    sorted_classes = unique_classes[sort_indices]

    _, winner_indices = np.unique(sorted_elements[::-1], return_index=True)
    winner_indices = (len(sorted_elements) - 1) - winner_indices

    return sorted_elements[winner_indices], sorted_classes[winner_indices]


def _merge_update_rects(existing_rect, new_rect):
    """Return the union of two update rects in (x1, y1, x2, y2) form."""
    if new_rect is None:
        return existing_rect
    if existing_rect is None:
        return new_rect

    return (
        min(existing_rect[0], new_rect[0]),
        min(existing_rect[1], new_rect[1]),
        max(existing_rect[2], new_rect[2]),
        max(existing_rect[3], new_rect[3]),
    )


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
    _universal_repaint_signal = pyqtSignal(list)  # internal: UI update tasks from unified propagation worker
    
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
        try:
            self.viewer.initialize_3d_tools(self)
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
        self._projected_cursor_context = None
        
        # Data Settings
        self.compute_depth_maps_enabled = True
        # New toggle: whether to compute index maps in background
        self.compute_index_maps_enabled = True
        # Maximum pixel budget for background index map computation
        self.pixel_budget = 4_000_000  # Default to ~4 Megapixels
        # Safety flag to prevent concurrent visibility computations
        self._is_computing_visibility = False
        # Track active worker threads to prevent GC
        self._active_workers = []
        self._context_stats_request_id = 0
        self._latest_context_stats_request_id = 0
        self._depth_build_lock = threading.Lock()
        self._pending_depth_build_paths = set()

        # Multi-camera annotation state
        self.multi_annotate_enabled = False
        self._propagating_annotation = False
        self._pending_unified_propagation_jobs = 0
        self._propagation_buffer_pool = {}

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

        # Single background worker for all 2D/3D propagation writes.
        self._unified_bg_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix='mvat_unified_bg'
        )
        self._universal_repaint_signal.connect(self._on_universal_repaint, Qt.QueuedConnection)

        # Lazy flush debounce timer: 3D GPU uploads happen only after the user pauses.
        self._lazy_flush_timer = QTimer(self)
        self._lazy_flush_timer.setSingleShot(True)
        self._lazy_flush_timer.setInterval(1000)
        self._lazy_flush_timer.timeout.connect(self._execute_lazy_flush)

        # --- Label Painter Thread ---
        self._label_painter_thread = None

        # Overlay actor handle (tiny actor swapped during painting)
        # Note: overlay is treated as the authoritative visualization; we
        # no longer use a debounce flush to upload labels into the main mesh GPU buffers.

        # Overlay actor handle (tiny actor swapped during painting)
        self._label_overlay_actor = None
        self._hover_overlay_actor = None
        self._hover_overlay_context = None
        self._hover_overlay_face_ids = None
        self._hover_overlay_color_rgb = None
        self._hover_overlay_last_state = None
        self._hover_overlay_enabled = False  # True

        self.contextStatsComputed.connect(self._on_context_stats_computed)

        self._setup_connections()

    @property
    def ortho_pixel_budget(self):
        """
        Dynamically scale the perspective pixel budget up for the orthomosaic.

        Ortho maps cover the whole site, so they use 16x the single-camera
        pixel budget without introducing extra state.
        """
        if self.pixel_budget is None:
            return None
        return self.pixel_budget * 16

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
        all_paths = self.raster_manager.image_paths
        if not all_paths:
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
            return

        # =====================================================================
        # OrthoRaster: build OrthoCamera only when one hasn't been created yet.
        # =====================================================================
        if need_ortho:
            from coralnet_toolbox.MVAT.core.Cameras import OrthoCamera

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
                    break
                else:
                    print(f"⚠️ OrthoRaster {raster.basename} missing geo transform — skipping.")

        # =====================================================================
        # Pre-computation Cache Check — only for cameras that are new this call.
        # =====================================================================
        primary_target = self.viewer.scene_context.get_primary_target()
        newly_added_cameras = [self.cameras[p] for p, _ in new_perspective_rasters if p in self.cameras]
        should_compute_visibility = False
        cameras_to_compute = []

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
                if not self.cache_manager.has_visibility_cache(
                        cache_key, target_path, element_type, extra,
                        pixel_budget=self.pixel_budget):
                    uncached_cameras.append(cam)

            if uncached_cameras:
                choice_mode, new_budget = self._prompt_visibility_quality_dialog(
                    len(uncached_cameras)
                )

                if choice_mode is None:
                    return

                previous_budget = getattr(self, 'pixel_budget', None)
                self.pixel_budget = new_budget

                # If the budget actually changed, the previously cached
                # visibility maps (in RAM) were produced at a different
                # resolution. Invalidate them so later visibility work does not
                # mix face-ID sets sampled at different resolutions.
                if previous_budget != new_budget:
                    self._invalidate_perspective_visibility_state()

                if choice_mode == 'compute':
                    # Recompute everything that was already loaded at the old
                    # quality, not just the brand-new uncached cameras.
                    if previous_budget != new_budget:
                        cameras_to_compute = list(newly_added_cameras) + [
                            cam for cam in self.cameras.values()
                            if cam not in newly_added_cameras
                        ]
                    else:
                        cameras_to_compute = uncached_cameras
                    should_compute_visibility = True

        # Build the ortho index map only after the quality budget has been
        # resolved, and before any perspective visibility maps are started.
        if need_ortho:
            self._maybe_compute_ortho_index_map()

        if should_compute_visibility:
            self._compute_visibility_async(primary_target, cameras_to_compute)

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

    def _prompt_visibility_quality_dialog(self, camera_count: int):
        """Prompt for visibility quality and whether to compute now or defer."""
        from PyQt5.QtWidgets import (
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QLabel,
            QVBoxLayout,
        )

        qualities = [
            "Native (Full Resolution)",
            "Highest (~12 Megapixels)",
            "High (~4 Megapixels)",
            "Medium (~2 Megapixels)",
            "Low (~1 Megapixel)",
            "Lowest (~0.5 Megapixel)",
        ]
        quality_map = {
            "Native (Full Resolution)": None,
            "Highest (~12 Megapixels)": 12_000_000,
            "High (~4 Megapixels)": 4_000_000,
            "Medium (~2 Megapixels)": 2_000_000,
            "Low (~1 Megapixel)": 1_000_000,
            "Lowest (~0.5 Megapixel)": 500_000,
        }

        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("Pre-compute Visibility")
        dialog.setModal(True)
        dialog.resize(520, 180)

        selected_mode = {'mode': None}

        def _accept_compute():
            selected_mode['mode'] = 'compute'
            dialog.accept()

        def _reject_background():
            selected_mode['mode'] = 'background'
            dialog.reject()

        layout = QVBoxLayout(dialog)

        message_label = QLabel(
            f"Found {camera_count} cameras without cached visibility maps.<br><br>"
            "Choose a visibility quality, then either compute them now or defer "
            "to background loading."
        )
        message_label.setWordWrap(True)
        layout.addWidget(message_label)

        form_layout = QFormLayout()
        quality_combo = QComboBox(dialog)
        quality_combo.addItems(qualities)

        current_idx = 2  # Default to "High (~4 Megapixels)"
        for i, budget in enumerate(quality_map.values()):
            if getattr(self, 'pixel_budget', 4_000_000) == budget:
                current_idx = i
                break
        quality_combo.setCurrentIndex(current_idx)
        form_layout.addRow("Visibility Quality:", quality_combo)
        layout.addLayout(form_layout)

        button_box = QDialogButtonBox(dialog)
        compute_button = button_box.addButton("Compute Now", QDialogButtonBox.AcceptRole)
        background_button = button_box.addButton("Background", QDialogButtonBox.RejectRole)
        compute_button.clicked.connect(_accept_compute)
        background_button.clicked.connect(_reject_background)
        layout.addWidget(button_box)

        dialog.exec_()

        mode = selected_mode['mode']
        if mode is None:
            return None, None

        chosen_quality = quality_combo.currentText()
        return mode, quality_map[chosen_quality]

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
        primary_target = self.viewer.scene_context.get_primary_target()
        self._prewarm_spatial_caches(primary_target)
        if self.ortho_camera is not None:
            self.ortho_camera._raster.index_map = None
            self.ortho_camera._raster.index_map_scale_factor = None
            self.ortho_camera._raster.index_map_path = None
        self._maybe_compute_ortho_index_map()

    def _prewarm_spatial_caches(self, primary_target):
        """Build the KD-Tree on the main UI thread for fast spatial queries."""
        if primary_target is None or not hasattr(primary_target, 'get_render_mesh'):
            return

        tree = getattr(primary_target, '_hover_face_kdtree', None)
        tree_product_id = getattr(primary_target, '_hover_face_kdtree_product_id', None)
        if tree is not None and tree_product_id == getattr(primary_target, 'product_id', None):
            return

        print("🌳 Building KD-Tree...")
        try:
            self.main_window.status_bar.showMessage("🌳 Building KD-Tree...", 0)
        except Exception:
            pass

        build_start = time.perf_counter()

        try:
            if getattr(primary_target, '_element_centers_np', None) is None:
                primary_target.prepare_geometry()

            centers = getattr(primary_target, '_element_centers_np', None)
            if centers is not None and len(centers) > 0:
                from scipy.spatial import cKDTree

                tree = cKDTree(np.asarray(centers, dtype=np.float32))
                primary_target._hover_face_kdtree = tree
                primary_target._hover_face_kdtree_product_id = getattr(primary_target, 'product_id', None)

            try:
                self.main_window.status_bar.showMessage("KD-Tree built.", 3000)
            except Exception:
                pass

            build_elapsed_s = time.perf_counter() - build_start
            print(f"🌳 KD-Tree built in {build_elapsed_s:.2f} s")

        except Exception as e:
            build_elapsed_s = time.perf_counter() - build_start
            print(f"🌳 KD-Tree build failed after {build_elapsed_s:.2f} s: {e}")

    def _query_kdtree_candidate_ids(self, tree, center, search_radius, total_count: int, initial_k: int = 256):
        try:
            center = np.asarray(center, dtype=np.float32).reshape(-1)
            search_radius = float(search_radius)
        except Exception:
            return np.empty(0, dtype=np.int32)

        if search_radius <= 0.0:
            return np.empty(0, dtype=np.int32)

        while True:
            try:
                candidate_ids = tree.query_ball_point(center, search_radius)
            except Exception:
                return np.empty(0, dtype=np.int32)

            if not candidate_ids:
                return np.empty(0, dtype=np.int32)

            return np.asarray(candidate_ids, dtype=np.int32)

            next_k = min(total_count, max(k * 2, k + 1))
            if next_k == k:
                return candidate_ids
            k = next_k

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

        native_pixels = self.ortho_camera.width * self.ortho_camera.height
        pixel_budget = self.ortho_pixel_budget
        if pixel_budget is None or native_pixels <= pixel_budget:
            current_scale = 1.0
        else:
            current_scale = float(np.sqrt(pixel_budget / native_pixels))

        existing_scale = getattr(ortho_raster, 'index_map_scale_factor', None)
        if ortho_raster.index_map is not None:
            if existing_scale is not None and np.isclose(float(existing_scale), current_scale):
                return
            ortho_raster.index_map = None
            ortho_raster.index_map_path = None
            ortho_raster.index_map_scale_factor = None
            ortho_raster.visible_indices = None
            ortho_raster.inv_ids = None
            ortho_raster.inv_offsets = None
            ortho_raster.inv_pixels = None

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
        requested_budget = self.ortho_pixel_budget

        def _build():
            from coralnet_toolbox.MVAT.managers.VisibilityManager import VisibilityManager
            return VisibilityManager.compute_ortho_index_map_vtk(
                ortho_camera,
                mesh_product,
                pixel_budget=requested_budget,
            )

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
            native_pixels = self.ortho_camera.width * self.ortho_camera.height
            pixel_budget = self.ortho_pixel_budget
            if pixel_budget is None or native_pixels <= pixel_budget:
                current_scale = 1.0
            else:
                current_scale = float(np.sqrt(pixel_budget / native_pixels))

            result_scale = float(result.get('scale_factor', current_scale))
            if not np.isclose(result_scale, current_scale):
                print(
                    f"⚠️ Discarding stale ortho index map at scale {result_scale:.4f}; "
                    f"current quality is {current_scale:.4f}"
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

    def _reconstruct_depth_map_for_camera(self, primary_target, camera, index_map):
        """Reconstruct the depth map for the given camera."""
        
        if not self.compute_depth_maps_enabled or primary_target is None or index_map is None:
            return None

        try:
            start_time = time.perf_counter()
            from coralnet_toolbox.MVAT.managers.VisibilityManager import VisibilityManager
            depth_map = VisibilityManager.reconstruct_depth_map(index_map, primary_target, camera.R, camera.t)
            if depth_map is not None:
                log_cam_stage(camera.label, "Depth Map", time.perf_counter() - start_time, logger)
            return depth_map
        except Exception as exc:
            print(f"⚠️ Failed to reconstruct depth map for {camera.label}: {exc}")
            return None

    def _reconstruct_depth_map_for_camera_fast(self, primary_target, camera):
        """Fast reconstruction of the depth map for the given camera."""
        if not self.compute_depth_maps_enabled or primary_target is None or camera is None:
            return None

        try:
            start_time = time.perf_counter()
            from coralnet_toolbox.MVAT.managers.VisibilityManager import VisibilityManager
            depth_map = VisibilityManager.reconstruct_depth_map_fast(camera, primary_target)
            if depth_map is not None:
                log_cam_stage(camera.label, "Depth Map (Fast)", time.perf_counter() - start_time, logger)
            return depth_map
        except Exception as exc:
            print(f"⚠️ Failed to fast-reconstruct depth map for {camera.label}: {exc}")
            return None

    def _queue_active_camera_depth_build(self, primary_target=None):
        """Queue the active camera for depth map reconstruction."""
        if not self.compute_depth_maps_enabled:
            return

        camera = self.selected_camera
        if camera is None:
            return

        if getattr(camera._raster, 'z_channel', None) is not None:
            return

        if camera.visible_indices is None or len(camera.visible_indices) == 0:
            return

        if getattr(camera._raster, 'index_map', None) is None:
            return

        if primary_target is None:
            try:
                primary_target = self.viewer.scene_context.get_primary_target()
            except Exception:
                primary_target = None

        if primary_target is None:
            return

        camera_path = camera.image_path
        with self._depth_build_lock:
            if camera_path in self._pending_depth_build_paths:
                return
            self._pending_depth_build_paths.add(camera_path)

        def _lazy_build_depth():
            try:
                depth_map = self._reconstruct_depth_map_for_camera_fast(primary_target, camera)
                if depth_map is None:
                    depth_map = self._reconstruct_depth_map_for_camera(
                        primary_target, camera, camera.index_map,
                    )
                if depth_map is not None:
                    try:
                        camera._raster.merge_or_set_depth_map(depth_map)
                    except Exception:
                        pass
            finally:
                with self._depth_build_lock:
                    self._pending_depth_build_paths.discard(camera_path)

        threading.Thread(target=_lazy_build_depth, daemon=True).start()

    def _process_visibility_results(self, results: dict, target_file_path: str):
        """
        Process visibility computation results and store in cameras.
        """
        primary_target = None
        try:
            primary_target = self.viewer.scene_context.get_primary_target()
        except Exception:
            pass

        for path, result in results.items():
            camera = self.cameras.get(path)
            if not camera:
                continue

            element_type = result.get('element_type', 'point')
            cache_path = result.get('cache_path')
            
            # --- Reload arrays from disk if stripped by the worker ---
            if result.get('index_map') is None and cache_path and self.cache_manager:
                cache_key = camera._raster.extrinsics
                extra = (camera._raster.dist_coeffs.tobytes()
                         if camera.is_distorted
                         and camera._raster.dist_coeffs is not None else None)
                
                # Loads using memory-mapping (mmap_mode='r') where possible
                loaded_data = self.cache_manager.load_visibility(
                    cache_key, target_file_path, element_type, extra,
                    pixel_budget=self.pixel_budget,
                )
                
                if loaded_data:
                    result['index_map'] = loaded_data.get('index_map')
                    result['depth_map'] = loaded_data.get('depth_map')

            # 2. Fallback for sync paths (like VTK) that run on the main thread
            if cache_path is None and self.cache_manager is not None and target_file_path:
                try:
                    cache_key = camera._raster.extrinsics
                    extra = (camera._raster.dist_coeffs.tobytes()
                             if camera.is_distorted
                             and camera._raster.dist_coeffs is not None else None)
                    cache_path = self.cache_manager.save_visibility(
                        cache_key, target_file_path, result.get('index_map'),
                        result.get('visible_indices'),
                        result.get('depth_map') if self.compute_depth_maps_enabled else None,
                        element_type=element_type, extra_hash_data=extra,
                        pixel_budget=self.pixel_budget,
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

            if self.compute_depth_maps_enabled:
                depth_map = result.get('depth_map')
                if depth_map is not None:
                    try:
                        camera._raster.merge_or_set_depth_map(depth_map)
                    except Exception:
                        pass

        self._queue_active_camera_depth_build(primary_target)

            
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

            self._queue_active_camera_depth_build()

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
        """Get the current order of cameras in the context matrix."""
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
        """Focus the context matrix on the specified camera."""
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
                        cache_key, target_file_path, element_type, extra,
                        pixel_budget=self.pixel_budget,
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
                # Prefer the path the loader actually used. Fall back to
                # rebuilding the canonical cache path when the loader didn't
                # supply one.
                cache_path = cached_data.get('cache_path') or self.cache_manager.get_cache_path(
                    cache_key, target_file_path, element_type, extra,
                    pixel_budget=self.pixel_budget,
                )

                # Store index map on raster (Qt object — must be on main thread)
                camera._raster.add_index_map(
                    cached_data.get('index_map'),
                    cache_path,
                    cached_data.get('visible_indices'),
                    element_type=element_type,
                    inverted_index=cached_data.get('inverted_index')
                )

                if self.compute_depth_maps_enabled:
                    depth_map = cached_data.get('depth_map')
                    if depth_map is not None:
                        try:
                            camera._raster.merge_or_set_depth_map(depth_map)
                        except Exception:
                            pass

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
                        compute_depth_map=False
                    )
                    result['element_type'] = 'face'
                    # Warp result back to distorted-pixel space if needed
                    if camera.is_distorted and camera._raster.intrinsics_undistorted is not None:
                        warp_fn = camera._raster.warp_linear_map_to_distorted
                        if result.get('index_map') is not None:
                            result['index_map'] = warp_fn(result['index_map'], nodata=-1)
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
                compute_depth_maps=False,
                cache_manager=self.cache_manager,
                cache_keys_dict=cache_keys_dict,
                target_file_path=primary_target.file_path if primary_target else "",
                pixel_budget=self.pixel_budget,
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

            # Drop finished workers so the retention list only tracks live work.
            self._active_workers = [
                (active_thread, active_worker)
                for active_thread, active_worker in self._active_workers
                if active_thread.isRunning()
            ]

            # Keep references to avoid GC
            self._active_workers.append((thread, worker))

            thread.start()

        except Exception as e:
            print(f"Failed to start visibility worker: {e}")
            self._is_computing_visibility = False
            QApplication.restoreOverrideCursor()

    # --- Label painter management ------------------------------------------------
    def submit_3d_face_paint(self, face_ids, color_rgb, class_id: int, primary_target=None):
        """Queue a 3D face paint update through the shared overlay painter.

        Tool code should compute the covered face IDs, then call this helper
        instead of reaching into ``_label_painter_thread`` directly.  The helper
        keeps thread lifecycle, mesh validation, and queue submission in one
        place while leaving geometry selection in the caller.
        """
        try:
            face_ids = np.asarray(face_ids, dtype=np.int32).ravel()
        except Exception:
            return

        if face_ids.size == 0:
            return

        if primary_target is None:
            primary_target = self._get_primary_mesh_target()

        if primary_target is None:
            return

        if int(class_id) == 0:
            color_rgb = (255, 255, 255)

        self._ensure_label_painter(primary_target)
        painter = self._label_painter_thread
        if painter is not None and painter.isRunning():
            painter.submit(face_ids, color_rgb, int(class_id))

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

            self._label_painter_thread = LabelWorker(
                mesh_points=mesh_points,
                mesh_faces_flat=mesh_faces_flat,
                labels_view=labels_cache,
                class_ids=class_ids,
            )
            self._label_painter_thread.overlay_ready.connect(self._on_overlay_ready, Qt.QueuedConnection)
            self._label_painter_thread.start()
        except Exception as e:
            print(f"⚠️ _ensure_label_painter failed: {e}")

    def request_lazy_flush(self):
        """Called on mouse-release or tool completion to start/reset the debounce timer."""
        self._lazy_flush_timer.start()
        status_bar = getattr(self.main_window, 'status_bar', None)
        if status_bar is not None:
            status_bar.showMessage("Waiting for pause to commit 3D paint...", 1500)

    def _execute_lazy_flush(self):
        """The actual heavy VTK upload. Runs only when the user pauses."""
        status_bar = getattr(self.main_window, 'status_bar', None)
        if status_bar is not None:
            status_bar.showMessage("Saving paint to 3D model...", 0)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # 1. Tell LabelWorker to clear its temporary overlay.
            painter = self._label_painter_thread
            if painter is not None and painter.isRunning():
                painter.finish_stroke()
            else:
                self._on_overlay_ready(None)

            # 2. Do the heavy VBO rebuild.
            primary_target = self._get_primary_mesh_target()
            if primary_target and hasattr(primary_target, 'flush_labels_to_gpu'):
                primary_target.flush_labels_to_gpu()

            # 3. Force the screen to update.
            try:
                self.viewer.plotter.render()
            except Exception:
                pass
        finally:
            QApplication.restoreOverrideCursor()
            if status_bar is not None:
                status_bar.showMessage("3D model updated.", 3000)

    def _build_primary_mesh_overlay(self):
        """Snapshot the current painted mesh faces into a tiny overlay payload."""
        try:
            primary_target = self.viewer.scene_context.get_primary_target()
        except Exception:
            primary_target = None

        if primary_target is None or not isinstance(primary_target, MeshProduct):
            return None

        try:
            mesh = primary_target.get_render_mesh()
        except Exception:
            mesh = None

        if mesh is None:
            return None

        class_ids = getattr(primary_target, 'class_ids', None)
        labels_cache = getattr(primary_target, '_labels_cache', None)

        if labels_cache is None:
            try:
                labels_cache = np.asarray(mesh.cell_data['Labels']).copy()
                primary_target._labels_cache = labels_cache
            except Exception:
                labels_cache = None

        if class_ids is None or labels_cache is None:
            return None

        painted_faces = np.flatnonzero(np.asarray(class_ids) != 0)
        if painted_faces.size == 0:
            return None

        try:
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
            return overlay_points, vtk_faces, colors
        except Exception:
            return None

    def refresh_primary_mesh_overlay(self, force_recreate: bool = False, render: bool = True):
        """Rebuild the visible mesh-label overlay from the current mesh state."""
        overlay = self._build_primary_mesh_overlay()
        if overlay is None:
            if self._label_overlay_actor is not None:
                self._on_overlay_ready(None, render=render)
            return

        self._on_overlay_ready(overlay, force_recreate=force_recreate, render=render)

    def _on_overlay_ready(self, overlay, force_recreate: bool = False, render: bool = True):
        """Main thread: update the overlay actor in place when possible."""
        start_time = perf_counter()
        try:
            if overlay is None:
                if self._label_overlay_actor is not None:
                    try:
                        self.viewer.plotter.remove_actor(self._label_overlay_actor, render=False)
                    except Exception:
                        pass
                    self._label_overlay_actor = None
                    if render:
                        try:
                            self.viewer.plotter.render()
                        except Exception:
                            pass
                return

            t1 = perf_counter()

            def _add_overlay_actor(mesh_to_add):
                return self.viewer.plotter.add_mesh(
                    mesh_to_add,
                    scalars='OverlayColors',
                    rgb=True,
                    copy_mesh=False,
                    lighting=False,
                    show_scalar_bar=False,
                )

            # If overlay is a tuple/list from the worker: assemble PolyData here.
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
                # Backwards-compat: already a pv.PolyData.
                mesh_to_add = overlay

            if force_recreate and self._label_overlay_actor is not None:
                try:
                    self.viewer.plotter.remove_actor(self._label_overlay_actor, render=False)
                except Exception:
                    pass
                self._label_overlay_actor = None

            if self._label_overlay_actor is None:
                self._label_overlay_actor = _add_overlay_actor(mesh_to_add)
            else:
                mapper = None
                try:
                    mapper = self._label_overlay_actor.GetMapper()
                except Exception:
                    mapper = None

                updated = False
                if mapper is not None:
                    for method_name in ('SetInputDataObject', 'SetInputData'):
                        method = getattr(mapper, method_name, None)
                        if callable(method):
                            try:
                                method(mesh_to_add)
                                try:
                                    mapper.Update()
                                except Exception:
                                    pass
                                updated = True
                                break
                            except Exception:
                                continue

                if not updated:
                    try:
                        self.viewer.plotter.remove_actor(self._label_overlay_actor, render=False)
                    except Exception:
                        pass
                    self._label_overlay_actor = _add_overlay_actor(mesh_to_add)

            try:
                self._label_overlay_actor.SetVisibility(True)
            except Exception:
                pass
            t2 = perf_counter()
            if render:
                try:
                    last_render_time = getattr(self, '_last_vtk_render_time', None)
                    if last_render_time is None or (perf_counter() - last_render_time) > 0.033:
                        self.viewer.plotter.render()
                        self._last_vtk_render_time = perf_counter()
                except Exception:
                    pass
            t3 = perf_counter()
            print(f"DEBUG [OverlayReady (Main Thread)]: VTK Swap: {(t1 - start_time) * 1000:.2f}ms | Render: {(t3 - t2) * 1000:.2f}ms")
        except Exception as e:
            print(f"⚠️ Overlay swap failed: {e}")
        finally:
            get_visibility_logger().info(f"_on_overlay_ready: {perf_counter() - start_time:.4f}s")

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
        active_tool = getattr(self.viewer, '_active_3d_tool', None)
        try:
            radius = getattr(active_tool, 'brush_size', None)
            if radius is not None:
                return float(radius)
        except Exception:
            pass

        sphere_manager = getattr(self.viewer, '_sphere_manager', None)
        try:
            return float(getattr(sphere_manager, 'radius', 0.1))
        except Exception:
            return 0.1

    def _get_sphere_hover_shape(self):
        active_tool = getattr(self.viewer, '_active_3d_tool', None)
        try:
            shape = getattr(active_tool, 'brush_shape', None)
            if shape is not None:
                shape = str(shape).strip().lower()
                if shape in ('circle', 'square'):
                    return shape
        except Exception:
            pass

        sphere_manager = getattr(self.viewer, '_sphere_manager', None)
        try:
            shape = getattr(sphere_manager, 'shape', None)
            if shape is not None:
                shape = str(shape).strip().lower()
                if shape in ('circle', 'square'):
                    return shape
        except Exception:
            pass

        return 'circle'

    def _get_active_3d_tool_kind(self):
        active_tool = getattr(self.viewer, '_active_3d_tool', None)
        if active_tool is None:
            return None, None

        try:
            tool_kind = getattr(active_tool, 'tool_kind', None)
            if isinstance(tool_kind, str):
                tool_kind = tool_kind.strip().lower()
                if tool_kind in ('brush', 'erase'):
                    return tool_kind, active_tool
        except Exception:
            pass

        tool_name = type(active_tool).__name__.strip().lower()
        if 'erase' in tool_name:
            return 'erase', active_tool
        if 'brush' in tool_name:
            return 'brush', active_tool
        return None, active_tool

    def _get_2d_tool(self, tool_kind: str):
        tools = getattr(self.annotation_window, 'tools', None)
        if not isinstance(tools, dict):
            return None
        return tools.get(tool_kind)

    def _get_camera_view_normal(self, camera, world_point):
        if camera is None:
            return None

        if hasattr(camera, 'get_vertical_direction_world'):
            try:
                normal = np.asarray(camera.get_vertical_direction_world(), dtype=np.float64)
                if normal.size >= 3:
                    normal = normal[:3]
                    length = float(np.linalg.norm(normal))
                    if length >= 1e-8:
                        return normal / length
            except Exception:
                pass

        camera_pos = getattr(camera, 'position', None)
        if camera_pos is not None and world_point is not None:
            try:
                camera_pos = np.asarray(camera_pos, dtype=np.float64).reshape(-1)
                world_point = np.asarray(world_point, dtype=np.float64).reshape(-1)
                if camera_pos.size >= 3 and world_point.size >= 3:
                    normal = world_point[:3] - camera_pos[:3]
                    length = float(np.linalg.norm(normal))
                    if length >= 1e-8:
                        return normal / length
            except Exception:
                pass

        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def _estimate_pixels_per_world_unit(self, camera, world_point):
        if camera is None or world_point is None:
            return None

        try:
            world_point = np.asarray(world_point, dtype=np.float64).reshape(-1)
            if world_point.size < 3:
                return None
            world_point = world_point[:3]

            center_pixel = np.asarray(camera.project(world_point), dtype=np.float64)
            if center_pixel is None or np.isnan(center_pixel).any():
                return None

            view_normal = self._get_camera_view_normal(camera, world_point)
            if view_normal is None:
                return None

            reference_vectors = (
                np.array([0.0, 0.0, 1.0], dtype=np.float64),
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
            )
            tangent = None
            for reference in reference_vectors:
                candidate = np.cross(view_normal, reference)
                candidate_norm = float(np.linalg.norm(candidate))
                if candidate_norm >= 1e-8:
                    tangent = candidate / candidate_norm
                    break
            if tangent is None:
                return None

            tangent_2 = np.cross(view_normal, tangent)
            tangent_2_norm = float(np.linalg.norm(tangent_2))
            if tangent_2_norm < 1e-8:
                return None
            tangent_2 = tangent_2 / tangent_2_norm

            camera_pos = getattr(camera, 'position', None)
            if camera_pos is not None:
                try:
                    camera_pos = np.asarray(camera_pos, dtype=np.float64).reshape(-1)
                except Exception:
                    camera_pos = None

            probe_distance = 0.01
            if camera_pos is not None and camera_pos.size >= 3:
                distance_to_camera = float(np.linalg.norm(world_point - camera_pos[:3]))
                probe_distance = max(1e-4, min(0.25, distance_to_camera * 0.001))

            samples = []
            for direction in (tangent, tangent_2):
                try:
                    projected = np.asarray(camera.project(world_point + direction * probe_distance), dtype=np.float64)
                except Exception:
                    continue
                if projected is None or np.isnan(projected).any():
                    continue
                delta = float(np.linalg.norm(projected - center_pixel))
                if np.isfinite(delta) and delta > 0.0:
                    samples.append(delta / probe_distance)

            if not samples:
                return None

            scale = float(np.mean(samples))
            if not np.isfinite(scale) or scale <= 0.0:
                return None
            return scale
        except Exception:
            return None

    def _project_cursor_preview_for_camera(self, camera, world_point, world_radius):
        if camera is None or world_point is None:
            return None

        try:
            world_radius = float(world_radius)
        except Exception:
            return None

        if world_radius <= 0.0:
            return None

        try:
            pixel = np.asarray(camera.project(world_point), dtype=np.float64)
        except Exception:
            return None

        if pixel is None or np.isnan(pixel).any():
            return None

        u = float(pixel[0])
        v = float(pixel[1])

        cam_w = getattr(camera, 'width', 0)
        cam_h = getattr(camera, 'height', 0)
        if cam_w and cam_h and not (0 <= u < cam_w and 0 <= v < cam_h):
            return None

        pixels_per_world = self._estimate_pixels_per_world_unit(camera, world_point)
        if pixels_per_world is None:
            return None

        radius_px = max(0.5, world_radius * pixels_per_world)
        return u, v, radius_px

    def _build_projected_cursor_factory(self, tool_kind: str, radius_px: float):
        tool = self._get_2d_tool(tool_kind)
        if tool is None:
            return None

        radius_px = max(0.5, float(radius_px))

        def factory(u, v):
            try:
                return tool.create_cursor_preview_item(u, v, radius=radius_px)
            except TypeError:
                return tool.create_cursor_preview_item(u, v)
            except Exception:
                return None

        return factory

    def _sync_projected_cursor_previews(self, world_point, render: bool = False, sync_2d_size: bool = True):
        """Update projected cursor previews on the AnnotationWindow + context canvases.

        ``sync_2d_size`` controls whether the active 2D tool's brush_size is
        rewritten to match the projected world_radius. Pass False when the
        2D size was just set by the user (e.g. 2D Ctrl+wheel); otherwise the
        projection round-trip will overwrite the value they just chose.
        """
        tool_kind, active_tool = self._get_active_3d_tool_kind()
        if tool_kind not in ('brush', 'erase') or active_tool is None or world_point is None:
            self._clear_projected_cursor_previews(render=render)
            return

        world_radius = self._get_sphere_hover_radius()
        if world_radius <= 0.0:
            self._clear_projected_cursor_previews(render=render)
            return

        try:
            world_point = np.asarray(world_point, dtype=np.float64)
        except Exception:
            self._clear_projected_cursor_previews(render=render)
            return

        selected_camera = self.selected_camera
        selected_camera_path = getattr(selected_camera, 'image_path', None)
        selected_label = getattr(self.annotation_window, 'selected_label', None)
        selected_label_id = getattr(selected_label, 'id', None)
        selected_label_color = getattr(selected_label, 'color', None)
        brush_shape = self._get_sphere_hover_shape()

        current_state = {
            'tool_kind': tool_kind,
            'selected_camera_path': selected_camera_path,
            'selected_label_id': selected_label_id,
            'selected_label_color': selected_label_color,
            'brush_shape': brush_shape,
            'world_radius': float(world_radius),
            'world_point': world_point.copy(),
        }

        previous_state = self._projected_cursor_context
        if previous_state is not None:
            try:
                same_state = (
                    previous_state.get('tool_kind') == tool_kind and
                    previous_state.get('selected_camera_path') == selected_camera_path and
                    previous_state.get('selected_label_id') == selected_label_id and
                    previous_state.get('selected_label_color') == selected_label_color and
                    previous_state.get('brush_shape') == brush_shape and
                    previous_state.get('world_radius') is not None and
                    np.isclose(float(previous_state.get('world_radius')), float(world_radius))
                )
            except Exception:
                same_state = False

            if same_state:
                try:
                    prev_world_point = np.asarray(previous_state.get('world_point'), dtype=np.float64)
                    center_delta = float(np.linalg.norm(world_point - prev_world_point))
                except Exception:
                    center_delta = None

                # Tiny cursor jitter is visually insignificant, but it still
                # forces a full cross-camera preview recompute. Skip that work
                # until the brush center has actually moved by a meaningful amount.
                if center_delta is not None and center_delta <= max(1e-6, float(world_radius) * 0.02):
                    self._projected_cursor_context = current_state
                    return

        self._projected_cursor_context = current_state

        projected_main = None
        if selected_camera is not None:
            projected_main = self._project_cursor_preview_for_camera(selected_camera, world_point, world_radius)

        # Keep the active 2D tool's internal size aligned with the current
        # selected camera projection so future 2D strokes reuse the same radius.
        # Suppressed when the call originated from a 2D wheel resize — that
        # path already wrote a deliberate 2D size and we must not clobber it
        # with the round-trip projection of the (about-to-be-updated) 3D radius.
        # _suspend_2d_size_sync covers nested internal callers (e.g. update_sphere_hover_overlay
        # → _sync_projected_cursor_previews) that pass the default sync_2d_size=True.
        if getattr(self, '_suspend_2d_size_sync', False):
            sync_2d_size = False
        if sync_2d_size and projected_main is not None:
            main_radius_px = projected_main[2]
            main_tool = self._get_2d_tool(tool_kind)
            if main_tool is not None:
                diameter_px = max(1, int(round(main_radius_px * 2.0)))
                try:
                    setter = getattr(main_tool, 'set_brush_size', None)
                    if callable(setter):
                        setter(diameter_px)
                    else:
                        main_tool.brush_size = diameter_px
                        brush_mask_factory = getattr(main_tool, '_create_brush_mask', None)
                        if callable(brush_mask_factory):
                            main_tool.brush_mask = brush_mask_factory()
                except Exception:
                    pass

        self._clear_projected_cursor_previews(render=False, reset_context=False)

        if projected_main is not None and selected_camera is not None:
            current_image_path = getattr(self.annotation_window, 'current_image_path', None)
            if current_image_path == selected_camera.image_path:
                factory = self._build_projected_cursor_factory(tool_kind, projected_main[2])
                if factory is not None:
                    try:
                        self.annotation_window.update_cursor_preview(projected_main[0], projected_main[1], factory)
                    except Exception:
                        pass

        if self.context_matrix is not None:
            canvas_map = {}
            try:
                canvas_map = self.context_matrix._get_canvas_camera_map()
            except Exception:
                canvas_map = {}

            for path, canvas in canvas_map.items():
                if canvas is None or not canvas.isVisible() or not canvas.current_image_path:
                    continue

                camera = self.cameras.get(path)
                projected = self._project_cursor_preview_for_camera(camera, world_point, world_radius)
                if projected is None:
                    try:
                        canvas.clear_cursor_preview()
                    except Exception:
                        pass
                    continue

                factory = self._build_projected_cursor_factory(tool_kind, projected[2])
                if factory is None:
                    continue

                try:
                    canvas.update_cursor_preview(projected[0], projected[1], factory)
                except Exception:
                    pass

        if render:
            try:
                self.viewer.plotter.render()
            except Exception:
                pass

    def _clear_projected_cursor_previews(self, render: bool = False, reset_context: bool = True):
        if reset_context:
            self._projected_cursor_context = None

        try:
            if hasattr(self.annotation_window, 'toggle_cursor_annotation'):
                self.annotation_window.toggle_cursor_annotation(None)
        except Exception:
            pass

        try:
            self.annotation_window.clear_cursor_preview()
        except Exception:
            pass

        if self.context_matrix is not None:
            try:
                self.context_matrix.clear_all_cursor_previews()
            except Exception:
                pass

        if render:
            try:
                self.viewer.plotter.render()
            except Exception:
                pass

    def on_2d_tool_size_changed(self, tool, scene_pos: QPointF = None):
        """Sync a 2D brush/erase size or shape change into the active 3D tool.

        The 2D wheel is the authoritative source of size during this call,
        so we suspend the 3D→2D auto-sync inside _sync_projected_cursor_previews
        for the duration. Without that guard, the projection round-trip
        (3D world_radius → projected pixel diameter → 2D set_brush_size)
        immediately overwrites the value the user just chose.
        """
        tool_kind, active_tool = self._get_active_3d_tool_kind()
        if tool_kind not in ('brush', 'erase') or active_tool is None:
            return

        self._suspend_2d_size_sync = True

        brush_shape = getattr(tool, 'shape', None)
        try:
            brush_shape = str(brush_shape).strip().lower() if brush_shape is not None else None
        except Exception:
            brush_shape = None

        selected_camera = self.selected_camera
        world_point = None
        if selected_camera is not None:
            if scene_pos is not None:
                try:
                    scene_x = int(round(scene_pos.x()))
                    scene_y = int(round(scene_pos.y()))
                    world_point = self._get_world_point_at_pixel(selected_camera, scene_x, scene_y)
                except Exception:
                    world_point = None

            if world_point is None:
                context = self._hover_overlay_context or {}
                world_point = context.get('center')

            if world_point is None:
                try:
                    world_point = np.asarray(self.viewer.plotter.camera.focal_point, dtype=np.float64)
                except Exception:
                    world_point = None

        if brush_shape in ('circle', 'square'):
            try:
                setter = getattr(active_tool, 'set_brush_shape', None)
                if callable(setter):
                    setter(brush_shape, center=world_point)
                else:
                    active_tool.brush_shape = brush_shape
            except Exception:
                pass

            if world_point is not None:
                try:
                    self.update_sphere_hover_overlay(world_point, render=False)
                except Exception:
                    pass

        if selected_camera is None:
            return

        if world_point is None:
            return

        pixels_per_world = self._estimate_pixels_per_world_unit(selected_camera, world_point)
        if pixels_per_world is None or pixels_per_world <= 0.0:
            return

        try:
            diameter_px = float(getattr(tool, 'brush_size', 1))
        except Exception:
            diameter_px = 1.0

        world_radius = max(1e-6, (diameter_px / 2.0) / pixels_per_world)

        try:
            setter = getattr(active_tool, 'set_brush_size', None)
            if callable(setter):
                setter(world_radius, center=world_point)
            else:
                active_tool.brush_size = world_radius
                updater = getattr(active_tool, '_update_preview_sphere', None)
                if callable(updater):
                    updater(world_point)
        except Exception:
            pass

        # The 3D tool's brush_size has now changed. Three previews depend on it
        # and must be refreshed explicitly, otherwise old-size artifacts linger
        # alongside the new wireframe sphere:
        #
        #  1. The label-colored hover overlay on the mesh. refresh_sphere_hover_overlay
        #     takes a fast path when face IDs are already populated, which keeps
        #     the overlay sized to the previous radius. Invalidate the cached
        #     face IDs so the new radius forces a real recompute.
        #  2. The projected cursor previews on the AnnotationWindow's BaseCanvas
        #     and every ContextMatrix canvas. These were updated with the old
        #     radius earlier in this method (when update_sphere_hover_overlay
        #     ran before set_brush_size). Re-sync them now at the new radius.
        if self._hover_overlay_context is not None:
            self._hover_overlay_face_ids = None
            self._hover_overlay_last_state = None
            try:
                self.refresh_sphere_hover_overlay(render=False)
            except Exception:
                pass

        try:
            self._sync_projected_cursor_previews(world_point, render=False)
        except Exception:
            pass

        # When the resize came from a 2D Ctrl+wheel, the AnnotationWindow's
        # BaseCanvas-side projected preview is redundant — the 2D brush tool
        # already paints its own cursor_annotation there. Worse, regular 2D
        # mouse moves don't refresh the BaseCanvas preview (cursor_move_callback
        # only updates the context-matrix canvases), so the projected ellipse
        # sticks at the wheel position until the user mouses out. Clear it now.
        try:
            self.annotation_window.clear_cursor_preview()
        except Exception:
            pass

        # Same problem in 3D: the wireframe preview sphere and the
        # label-colored hover overlay were repositioned to the projected
        # world_point during this resize, but the 3D viewer never receives a
        # mouse-move event from inside the AnnotationWindow, so they stay
        # parked there until the user actually hovers the 3D viewer. Hide
        # them now; mouseMoveEvent in Tool3D will bring them back the next
        # time the cursor enters the 3D viewport.
        try:
            hide_preview = getattr(active_tool, '_hide_preview_sphere', None)
            if callable(hide_preview):
                hide_preview()
        except Exception:
            pass
        try:
            self.clear_sphere_hover_overlay(reset_context=False, render=False)
        except Exception:
            pass

        # Re-arm the 3D→2D auto-sync for subsequent hover-driven updates.
        self._suspend_2d_size_sync = False

    def _normalize_color_rgb(self, color_rgb):
        try:
            return tuple(int(c) for c in color_rgb[:3])
        except Exception:
            return None

    def set_hover_overlay_enabled(self, enabled: bool):
        """Enable or disable the 3D label hover overlay without touching the rest of the brush path."""
        enabled = bool(enabled)
        if self._hover_overlay_enabled == enabled:
            return

        self._hover_overlay_enabled = enabled

        if not enabled:
            self._hover_overlay_face_ids = None
            self._hover_overlay_last_state = None
            self._clear_hover_dynamic_markers(render=False)
            if self._hover_overlay_actor is not None:
                try:
                    self._hover_overlay_actor.SetVisibility(False)
                except Exception:
                    pass
        elif self._hover_overlay_context is not None:
            try:
                self.refresh_sphere_hover_overlay(render=False)
            except Exception:
                pass

        try:
            self.viewer.plotter.render()
        except Exception:
            pass

    def is_hover_overlay_enabled(self) -> bool:
        """Return whether the 3D label hover overlay is currently enabled."""
        return bool(self._hover_overlay_enabled)

    def _apply_hover_overlay_color(self, color_rgb, render: bool = False):
        if not self._hover_overlay_enabled or self._hover_overlay_actor is None or color_rgb is None:
            return

        try:
            normalized = self._normalize_color_rgb(color_rgb)
            if normalized is None:
                return
            r, g, b = normalized
            prop = self._hover_overlay_actor.GetProperty()
            prop.SetColor(r / 255.0, g / 255.0, b / 255.0)
            prop.SetOpacity(0.45)
            self._hover_overlay_actor.SetVisibility(True)
            self._hover_overlay_color_rgb = (r, g, b)
            if render:
                try:
                    self.viewer.plotter.render()
                except Exception:
                    pass
        except Exception:
            pass

    def _set_hover_overlay_geometry(self, overlay, color_rgb, render: bool = True):
        try:
            if not self._hover_overlay_enabled:
                if self._hover_overlay_actor is not None:
                    try:
                        self._hover_overlay_actor.SetVisibility(False)
                    except Exception:
                        pass
                self._hover_overlay_color_rgb = self._normalize_color_rgb(color_rgb) if color_rgb is not None else self._hover_overlay_color_rgb
                if render:
                    try:
                        self.viewer.plotter.render()
                    except Exception:
                        pass
                return

            if overlay is None:
                if self._hover_overlay_actor is not None:
                    try:
                        self._hover_overlay_actor.SetVisibility(False)
                    except Exception:
                        pass
                self._hover_overlay_color_rgb = self._normalize_color_rgb(color_rgb) if color_rgb is not None else self._hover_overlay_color_rgb
                if render:
                    try:
                        self.viewer.plotter.render()
                    except Exception:
                        pass
                return

            if self._hover_overlay_actor is None:
                self._hover_overlay_actor = self.viewer.plotter.add_mesh(
                    overlay,
                    color=tuple(c / 255.0 for c in self._normalize_color_rgb(color_rgb) or (255, 255, 255)),
                    copy_mesh=False,
                    lighting=False,
                    opacity=0.45,
                    show_scalar_bar=False,
                    smooth_shading=False,
                    pickable=False,
                    name='_sphere_hover_overlay',
                    reset_camera=False,
                )
            else:
                mapper = None
                try:
                    mapper = self._hover_overlay_actor.GetMapper()
                except Exception:
                    mapper = None

                updated = False
                if mapper is not None:
                    for method_name in ('SetInputDataObject', 'SetInputData'):
                        method = getattr(mapper, method_name, None)
                        if callable(method):
                            try:
                                method(overlay)
                                updated = True
                                break
                            except Exception:
                                continue
                if not updated:
                    try:
                        self.viewer.plotter.remove_actor(self._hover_overlay_actor, render=False)
                    except Exception:
                        pass
                    self._hover_overlay_actor = self.viewer.plotter.add_mesh(
                        overlay,
                        color=tuple(c / 255.0 for c in self._normalize_color_rgb(color_rgb) or (255, 255, 255)),
                        copy_mesh=False,
                        lighting=False,
                        opacity=0.45,
                        show_scalar_bar=False,
                        smooth_shading=False,
                        pickable=False,
                        name='_sphere_hover_overlay',
                        reset_camera=False,
                    )

            self._apply_hover_overlay_color(color_rgb, render=False)
            self._hover_overlay_actor.SetVisibility(True)
            if render:
                try:
                    self.viewer.plotter.render()
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ Hover overlay update failed: {e}")

    def _project_world_to_view_pixel(self, world_point, image_height: int):
        try:
            point = np.asarray(world_point, dtype=np.float64).reshape(-1)
            if point.size < 3:
                return None

            renderer = self.viewer.plotter.renderer
            renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
            renderer.WorldToDisplay()
            display = renderer.GetDisplayPoint()

            u = float(display[0])
            y_vtk = float(display[1])
            if not np.isfinite(u) or not np.isfinite(y_vtk):
                return None

            # VTK display coordinates use a bottom-left origin; image arrays use top-left.
            v = float(image_height - 1 - y_vtk)
            return np.array([u, v], dtype=np.float64)
        except Exception:
            return None

    def _estimate_view_pixels_per_world_unit(self, world_point, image_height: int):
        center_px = self._project_world_to_view_pixel(world_point, image_height)
        if center_px is None:
            return None

        try:
            cam = self.viewer.plotter.camera
            camera_pos = np.asarray(cam.position, dtype=np.float64).reshape(-1)
            focal_point = np.asarray(cam.focal_point, dtype=np.float64).reshape(-1)
            world_point = np.asarray(world_point, dtype=np.float64).reshape(-1)
            if camera_pos.size < 3 or focal_point.size < 3 or world_point.size < 3:
                return None

            view_normal = focal_point[:3] - camera_pos[:3]
            view_norm = float(np.linalg.norm(view_normal))
            if view_norm < 1e-8:
                return None
            view_normal = view_normal / view_norm

            tangent = None
            for reference in (
                np.array([0.0, 0.0, 1.0], dtype=np.float64),
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
            ):
                candidate = np.cross(view_normal, reference)
                length = float(np.linalg.norm(candidate))
                if length >= 1e-8:
                    tangent = candidate / length
                    break
            if tangent is None:
                return None

            tangent_2 = np.cross(view_normal, tangent)
            tangent_2_norm = float(np.linalg.norm(tangent_2))
            if tangent_2_norm < 1e-8:
                return None
            tangent_2 = tangent_2 / tangent_2_norm

            distance_to_camera = float(np.linalg.norm(world_point[:3] - camera_pos[:3]))
            probe_distance = max(1e-4, min(0.25, distance_to_camera * 0.001))

            samples = []
            for direction in (tangent, tangent_2):
                projected = self._project_world_to_view_pixel(world_point[:3] + direction * probe_distance, image_height)
                if projected is None:
                    continue
                delta = float(np.linalg.norm(projected - center_px))
                if np.isfinite(delta) and delta > 0.0:
                    samples.append(delta / probe_distance)

            if not samples:
                return None

            scale = float(np.mean(samples))
            if not np.isfinite(scale) or scale <= 0.0:
                return None
            return scale
        except Exception:
            return None

    def _filter_face_ids_by_world_brush_volume(self, primary_target, candidate_face_ids, center, radius, shape: str = 'circle'):
        centers = getattr(primary_target, '_element_centers_np', None)
        if centers is None:
            return np.empty(0, dtype=np.int32)

        try:
            centers = np.asarray(centers, dtype=np.float32)
        except Exception:
            return np.empty(0, dtype=np.int32)

        if centers.ndim != 2 or centers.shape[0] == 0:
            return np.empty(0, dtype=np.int32)

        try:
            center = np.asarray(center, dtype=np.float32).reshape(-1)
        except Exception:
            return np.empty(0, dtype=np.int32)

        if center.size < centers.shape[1]:
            return np.empty(0, dtype=np.int32)
        if center.size != centers.shape[1]:
            center = center[:centers.shape[1]]

        try:
            radius = float(radius)
        except Exception:
            return np.empty(0, dtype=np.int32)

        if radius <= 0.0:
            return np.empty(0, dtype=np.int32)

        shape = str(shape).strip().lower()
        if shape not in ('circle', 'square'):
            shape = 'circle'

        if candidate_face_ids is None:
            return np.empty(0, dtype=np.int32)

        try:
            candidate_face_ids = np.asarray(candidate_face_ids, dtype=np.int64).reshape(-1)
        except Exception:
            return np.empty(0, dtype=np.int32)

        if candidate_face_ids.size == 0:
            return np.empty(0, dtype=np.int32)

        valid = (candidate_face_ids >= 0) & (candidate_face_ids < int(centers.shape[0]))
        if not np.any(valid):
            return np.empty(0, dtype=np.int32)

        candidate_face_ids = np.unique(candidate_face_ids[valid]).astype(np.int32, copy=False)
        candidate_centers = centers[candidate_face_ids]
        deltas = candidate_centers - center.astype(np.float32)

        if shape == 'square':
            within = np.max(np.abs(deltas), axis=1) <= radius
        else:
            radius_sq = radius * radius
            distances_sq = np.einsum('ij,ij->i', deltas, deltas)
            within = distances_sq <= radius_sq

        return candidate_face_ids[within].astype(np.int32, copy=False)

    def _get_faces_within_sphere(self, primary_target, center, radius, shape: str = 'circle'):
        def _finish(face_ids_result):
            face_ids_result = np.asarray(face_ids_result, dtype=np.int32).reshape(-1)
            return face_ids_result

        # 1. Input sanitization
        try:
            center = np.asarray(center, dtype=np.float64)
        except Exception:
            return np.empty(0, dtype=np.int32)

        try:
            radius = float(radius)
        except Exception:
            return np.empty(0, dtype=np.int32)

        if radius <= 0.0:
            return np.empty(0, dtype=np.int32)

        shape = str(shape).strip().lower()
        if shape not in ('circle', 'square'):
            shape = 'circle'

        # 2. Hard guardrail: If background thread hasn't finished KD-Tree, abort immediately
        tree = getattr(primary_target, '_hover_face_kdtree', None)
        if tree is None:
            return _finish(np.empty(0, dtype=np.int32))

        # 3. Fast Spatial Query
        try:
            centers = getattr(primary_target, '_element_centers_np', None)
            if centers is None or len(centers) == 0:
                return _finish(np.empty(0, dtype=np.int32))

            search_radius = float(radius) * np.sqrt(3.0) if shape == 'square' else float(radius)
            candidate_ids = self._query_kdtree_candidate_ids(tree, center, search_radius, int(centers.shape[0]))
            
            if candidate_ids.size == 0:
                return _finish(np.empty(0, dtype=np.int32))
                
            face_ids = self._filter_face_ids_by_world_brush_volume(primary_target, candidate_ids, center, radius, shape=shape)
            return _finish(face_ids)
            
        except Exception:
            return _finish(np.empty(0, dtype=np.int32))

    def clear_sphere_hover_overlay(self, reset_context: bool = False, render: bool = True):
        """Hide the sphere hover overlay and clear projected cursor previews."""
        if reset_context:
            self._hover_overlay_context = None

        self._hover_overlay_face_ids = None
        self._hover_overlay_last_state = None
        self._hover_overlay_color_rgb = None

        try:
            self._set_hover_overlay_geometry(None, None, render=False)
        except Exception:
            pass

        try:
            self._clear_projected_cursor_previews(render=False)
        except Exception:
            pass

        self._clear_hover_dynamic_markers(render=False)

        if render:
            try:
                self.viewer.plotter.render()
            except Exception:
                pass

    def _clear_hover_dynamic_markers(self, render: bool = False):
        """Hide dynamic marker overlays that mirror the 3D hover point."""
        try:
            if self.annotation_window is not None:
                self.annotation_window.clear_dynamic_marker()
        except Exception:
            pass

        try:
            if self.context_matrix is not None:
                self.context_matrix.clear_all_dynamic_markers()
        except Exception:
            pass

        if render:
            try:
                self.viewer.plotter.render()
            except Exception:
                pass

    def _project_hover_dynamic_marker(self, camera, world_point):
        """Project a 3D hover point into a camera for dynamic-marker display."""
        if camera is None or world_point is None:
            return None

        try:
            world_point = np.asarray(world_point, dtype=np.float64).reshape(-1)
        except Exception:
            return None

        if world_point.size < 3 or not np.all(np.isfinite(world_point[:3])):
            return None

        try:
            projected = camera.project(world_point[:3])
            if projected is None:
                return None
            projected = np.asarray(projected, dtype=np.float64).reshape(-1)
        except Exception:
            return None

        if projected.size < 2 or not np.all(np.isfinite(projected[:2])):
            return None

        u, v = float(projected[0]), float(projected[1])

        width = getattr(camera, 'width', None)
        height = getattr(camera, 'height', None)
        try:
            if width is not None and height is not None and width > 0 and height > 0:
                if not (0 <= u < float(width) and 0 <= v < float(height)):
                    return None
        except Exception:
            pass

        is_visible = True
        try:
            is_visible = not bool(camera.is_point_occluded_depth_based(world_point[:3], depth_threshold=0.15))
        except Exception:
            pass

        return u, v, is_visible

    def _sync_hover_dynamic_markers(self, world_point, render: bool = False):
        """Update dynamic markers so 2D views mirror the current 3D hover point."""
        if world_point is None:
            self._clear_hover_dynamic_markers(render=render)
            return

        selected_camera = getattr(self, 'selected_camera', None)
        if selected_camera is not None:
            projection = self._project_hover_dynamic_marker(selected_camera, world_point)
            if projection is None:
                try:
                    self.annotation_window.clear_dynamic_marker()
                except Exception:
                    pass
            else:
                u, v, is_visible = projection
                color = MARKER_COLOR_HIGHLIGHTED if is_visible else MARKER_COLOR_INVALID
                try:
                    self.annotation_window.update_dynamic_marker(u, v, color=color, is_valid=is_visible)
                except Exception:
                    pass
        else:
            try:
                self.annotation_window.clear_dynamic_marker()
            except Exception:
                pass

        context_matrix = getattr(self, 'context_matrix', None)
        if context_matrix is None:
            return

        projections = {}
        accuracies = {}
        visibility_status = {}
        for camera in self._get_visible_context_cameras():
            projection = self._project_hover_dynamic_marker(camera, world_point)
            if projection is None:
                continue

            image_path = getattr(camera, 'image_path', None)
            if not image_path:
                continue

            u, v, is_visible = projection
            projections[image_path] = (u, v, is_visible)
            accuracies[image_path] = True
            visibility_status[image_path] = not is_visible

        try:
            context_matrix.update_dynamic_markers(projections, accuracies, visibility_status)
        except Exception:
            try:
                context_matrix.clear_all_dynamic_markers()
            except Exception:
                pass

    def refresh_sphere_hover_overlay(self, render: bool = True):
        """Rebuild the hover overlay from the current hover context."""
        if not self._hover_overlay_enabled:
            try:
                self._clear_hover_dynamic_markers(render=False)
                self._set_hover_overlay_geometry(None, None, render=render)
            except Exception:
                pass
            return

        context = self._hover_overlay_context
        if not context:
            self.clear_sphere_hover_overlay(reset_context=True, render=render)
            return

        try:
            if not bool(getattr(self.viewer, '_sphere_visible', True)):
                self._set_hover_overlay_geometry(None, None, render=render)
                return

            passthrough_active = getattr(self.viewer, '_is_sphere_passthrough_active', None)
            if callable(passthrough_active) and passthrough_active():
                self._set_hover_overlay_geometry(None, None, render=render)
                return
        except Exception:
            pass

        primary_target = self._get_primary_mesh_target()
        if primary_target is None or getattr(primary_target, 'product_id', None) != context.get('product_id'):
            self.clear_sphere_hover_overlay(reset_context=True, render=render)
            return

        color_rgb = self._normalize_color_rgb(self._get_active_label_color_rgb())
        if color_rgb is None:
            self._clear_hover_dynamic_markers(render=False)
            self._set_hover_overlay_geometry(None, None, render=render)
            return

        center = context.get('center')
        if center is None:
            self.clear_sphere_hover_overlay(reset_context=True, render=render)
            return

        try:
            center = np.asarray(center, dtype=np.float64).reshape(-1)
        except Exception:
            self.clear_sphere_hover_overlay(reset_context=True, render=render)
            return

        if center.size < 3 or not np.all(np.isfinite(center[:3])):
            self.clear_sphere_hover_overlay(reset_context=True, render=render)
            return

        radius = self._get_sphere_hover_radius()
        try:
            radius = float(radius)
        except Exception:
            radius = 0.0

        if radius <= 0.0:
            self._set_hover_overlay_geometry(None, color_rgb, render=render)
            self._sync_hover_dynamic_markers(center[:3], render=False)
            return

        brush_shape = self._get_sphere_hover_shape()
        current_state = {
            'product_id': getattr(primary_target, 'product_id', None),
            'center': center[:3].copy(),
            'radius': radius,
            'brush_shape': brush_shape,
            'color_rgb': color_rgb,
        }

        previous_state = self._hover_overlay_last_state
        if previous_state is not None and self._hover_overlay_actor is not None:
            try:
                actor_visible = bool(self._hover_overlay_actor.GetVisibility())
            except Exception:
                actor_visible = False

            if actor_visible:
                prev_product_id = previous_state.get('product_id')
                prev_shape = previous_state.get('brush_shape')
                prev_radius = previous_state.get('radius')
                prev_color = previous_state.get('color_rgb')
                prev_center = previous_state.get('center')

                if (
                    prev_product_id == current_state['product_id'] and
                    prev_shape == brush_shape and
                    prev_color == color_rgb and
                    prev_center is not None and
                    prev_radius is not None and
                    np.isclose(float(prev_radius), radius)
                ):
                    try:
                        center_delta = float(np.linalg.norm(center[:3] - np.asarray(prev_center, dtype=np.float64).reshape(-1)[:3]))
                    except Exception:
                        center_delta = None

                    # Tiny cursor jitter usually keeps the same face set; skip the
                    # KD-tree lookup and overlay rebuild until the movement is meaningful.
                    if center_delta is not None and center_delta <= max(1e-6, radius * 0.02):
                        self._hover_overlay_context = current_state
                        self._hover_overlay_last_state = current_state.copy()
                        self._sync_projected_cursor_previews(center[:3], render=False)
                        self._sync_hover_dynamic_markers(center[:3], render=False)
                        return

        previous_face_ids = self._hover_overlay_face_ids
        self._hover_overlay_context = current_state
        face_ids = self._get_faces_within_sphere(primary_target, center[:3], radius, shape=brush_shape)
        if face_ids is None or len(face_ids) == 0:
            self._hover_overlay_face_ids = None
            self._hover_overlay_last_state = current_state
            self._set_hover_overlay_geometry(None, color_rgb, render=render)
            return

        face_ids = np.asarray(face_ids, dtype=np.int32)
        same_faces = previous_face_ids is not None and len(previous_face_ids) == len(face_ids) and np.array_equal(previous_face_ids, face_ids)
        same_color = self._hover_overlay_color_rgb == color_rgb

        self._hover_overlay_face_ids = face_ids
        self._hover_overlay_color_rgb = color_rgb

        if same_faces and same_color and self._hover_overlay_actor is not None:
            self._apply_hover_overlay_color(color_rgb, render=render)
            self._sync_projected_cursor_previews(center[:3], render=False)
            self._sync_hover_dynamic_markers(center[:3], render=False)
            self._hover_overlay_last_state = current_state
            return

        mesh = primary_target.get_render_mesh()
        if mesh is None:
            self.clear_sphere_hover_overlay(reset_context=True, render=render)
            return

        mesh_points = np.asarray(mesh.points, dtype=np.float32)
        mesh_faces_flat = np.asarray(mesh.faces.reshape(-1, 4), dtype=np.int32)
        overlay = LabelWorker.build_overlay(mesh_points, mesh_faces_flat, face_ids, color_rgb, attach_colors=False)
        self._set_hover_overlay_geometry(overlay, color_rgb, render=render)
        self._sync_projected_cursor_previews(center[:3], render=False)
        self._sync_hover_dynamic_markers(center[:3], render=False)
        self._hover_overlay_last_state = current_state

    def update_sphere_hover_overlay(self, center, render: bool = True):
        """Store the current hover center and refresh the sphere overlay."""
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
            center = np.asarray(center, dtype=np.float64).reshape(-1)
        except Exception:
            self.clear_sphere_hover_overlay(reset_context=True, render=render)
            return

        if center.size < 3 or not np.all(np.isfinite(center[:3])):
            self.clear_sphere_hover_overlay(reset_context=True, render=render)
            return

        self._hover_overlay_context = {
            'product_id': getattr(primary_target, 'product_id', None),
            'center': center[:3].copy(),
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
        from coralnet_toolbox.MVAT.core.Products import PointCloudProduct, MeshProduct
        
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

    def _acquire_propagation_buffer(self, shape, dtype=np.uint8):
        """Return a reusable NumPy buffer for background propagation work."""
        key = (tuple(shape), np.dtype(dtype).str)
        pool = self._propagation_buffer_pool.get(key)
        if pool:
            return pool.pop()
        return np.empty(shape, dtype=dtype)

    def _release_propagation_buffer(self, buffer):
        """Return a temporary propagation buffer to the local pool."""
        if buffer is None:
            return
        key = (tuple(buffer.shape), np.dtype(buffer.dtype).str)
        self._propagation_buffer_pool.setdefault(key, []).append(buffer)

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

    def _invalidate_perspective_visibility_state(self):
        """Clear cached visibility data on every perspective camera.

        Called when the user-selected pixel budget changes so the in-memory
        index_map / visible_indices left over from the previous quality
        setting don't get mixed with newly computed maps. The next visibility
        pass will repopulate each camera from the (quality-aware) disk cache
        or by recomputing.

        This intentionally also clears the OrthoCamera raster so that the
        next call to _maybe_compute_ortho_index_map sees a stale-scale state
        and rebuilds at the new budget.
        """
        for cam in self.cameras.values():
            raster = getattr(cam, '_raster', None)
            if raster is None:
                continue
            try:
                raster.visible_indices = None
                raster.index_map = None
                if hasattr(raster, 'index_map_path'):
                    raster.index_map_path = None
                if hasattr(raster, 'index_map_scale_factor'):
                    raster.index_map_scale_factor = None
                if hasattr(raster, 'inv_ids'):
                    raster.inv_ids = None
                if hasattr(raster, 'inv_offsets'):
                    raster.inv_offsets = None
                if hasattr(raster, 'inv_pixels'):
                    raster.inv_pixels = None
            except Exception:
                pass

        if self.ortho_camera is not None:
            ortho_raster = getattr(self.ortho_camera, '_raster', None)
            if ortho_raster is not None:
                try:
                    ortho_raster.visible_indices = None
                    ortho_raster.index_map = None
                    if hasattr(ortho_raster, 'index_map_path'):
                        ortho_raster.index_map_path = None
                    if hasattr(ortho_raster, 'index_map_scale_factor'):
                        ortho_raster.index_map_scale_factor = None
                    if hasattr(ortho_raster, 'inv_ids'):
                        ortho_raster.inv_ids = None
                    if hasattr(ortho_raster, 'inv_offsets'):
                        ortho_raster.inv_offsets = None
                    if hasattr(ortho_raster, 'inv_pixels'):
                        ortho_raster.inv_pixels = None
                except Exception:
                    pass

    def count_overlapping_cameras(self, active_camera, camera_items=None, scene_size=None):
        """
        Calculates how many cameras share a view of the same 3D geometry.
        Uses proximity scoring as a fast-reject to keep UI thread performance high.

        Important: the "true geometric overlap" branch compares unique mesh
        face IDs sampled by each camera's rasterized index map. That signal
        is reliable only at Native (full-resolution) rendering. At reduced
        pixel budgets the rasterizer aliases many faces into a single pixel,
        so each camera ends up with a sparse, *different* subset of the
        shared geometry's face IDs — the intersection collapses even when
        the cameras are genuinely overlapping. To avoid the matrix's
        camera-count cap shrinking to 1 at low quality, we use proximity
        alone (which is what _reorder_cameras uses to pick neighbors)
        whenever the pixel budget is non-Native.

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

        # When the user picked anything other than "Native (Full Resolution)"
        # the face-ID intersection is unreliable (see docstring). Fall back to
        # proximity-only counting so the matrix's camera-count cap reflects how
        # many neighbors _reorder_cameras would actually surface.
        pixel_budget = getattr(self, 'pixel_budget', None)
        use_proximity_only = pixel_budget is not None and pixel_budget > 0

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

            if use_proximity_only:
                # Non-Native budget: trust proximity, which is render-quality-
                # independent and matches what _reorder_cameras uses.
                overlap_count += 1
                continue

            # OPTIMIZATION 2: True Geometric Overlap (Native quality only)
            if (active_indices is not None
                    and cam.visible_indices is not None
                    and active_visible_count > 0):
                # Both arrays are pre-sorted and unique thanks to VisibilityWorker.
                # assume_unique=True makes this incredibly fast.
                shared = np.intersect1d(active_indices, cam.visible_indices, assume_unique=True)

                if (len(shared) / active_visible_count) >= min_overlap_ratio:
                    overlap_count += 1
            else:
                # Visibility maps not ready yet (e.g. cache miss still being
                # computed). Fall back to the proximity score we already
                # computed above so the cap doesn't collapse to 1.
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

    def _build_projection(self, px: int, py: int, source_camera=None) -> dict:
        """Cast a ray from the selected camera at (px, py) and return projections.

        Handles both perspective cameras and OrthoCamera (orthomosaic):
        - For perspective cameras: uses existing ray-projection logic
        - For OrthoCamera: converts orthomosaic pixel → geo → world space

        Returns:
            dict mapping image_path -> (u, v, is_valid), or empty dict on failure.
        """
        camera = source_camera if source_camera is not None else self.selected_camera
        if camera is None:
            return {}

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

    def _resolve_source_mask_class_context(self, source_camera, label_id: str, project_labels: list):
        """Resolve the source label, mask, and internal class ID for propagation."""
        if source_camera is None:
            return None, None

        source_label = next((lbl for lbl in project_labels if lbl.id == label_id), None)
        if source_label is None:
            return None, None

        source_raster = self.raster_manager.get_raster(source_camera.image_path)
        if source_raster is None:
            return source_label, None

        source_mask = source_raster.mask_annotation
        if source_mask is None:
            source_mask = source_raster.get_mask_annotation(project_labels)
        if source_mask is None:
            return source_label, None

        source_class_id = source_mask.label_id_to_class_id_map.get(label_id)
        if source_class_id is None:
            source_mask.sync_label_map([source_label])
            source_class_id = source_mask.label_id_to_class_id_map.get(label_id)

        return source_label, source_class_id

    def _extract_source_ids_from_sam_prediction(self,
                                                source_camera,
                                                binary_mask: np.ndarray,
                                                px: int,
                                                py: int) -> Optional[np.ndarray]:
        """Extract source element IDs for a SAM prediction on the active image."""
        if source_camera is None or binary_mask is None:
            return None

        binary_mask = np.asarray(binary_mask)
        if binary_mask.ndim != 2 or not np.any(binary_mask):
            return np.array([], dtype=np.int64)

        if self.ortho_camera is not None and source_camera is self.ortho_camera:
            return self._extract_source_ids_from_crop_mask(
                source_camera,
                binary_mask.astype(bool),
                px,
                py,
            )

        raster = getattr(source_camera, '_raster', None)
        source_index_map = getattr(raster, 'index_map', None)
        if source_index_map is None:
            return None

        mask_h, mask_w = binary_mask.shape
        x0 = px - mask_w // 2
        y0 = py - mask_h // 2
        x1 = x0 + mask_w
        y1 = y0 + mask_h

        img_h, img_w = source_index_map.shape
        if x0 >= img_w or y0 >= img_h or x1 <= 0 or y1 <= 0:
            return np.array([], dtype=np.int64)

        cx0 = max(x0, 0)
        cy0 = max(y0, 0)
        cx1 = min(x1, img_w)
        cy1 = min(y1, img_h)

        bx0 = cx0 - x0
        by0 = cy0 - y0
        bx1 = bx0 + (cx1 - cx0)
        by1 = by0 + (cy1 - cy0)

        index_slice = source_index_map[cy0:cy1, cx0:cx1]
        mask_clip = binary_mask[by0:by1, bx0:bx1]
        valid_mask = mask_clip.astype(bool)
        if not np.any(valid_mask):
            return np.array([], dtype=np.int64)

        source_depth_map = getattr(raster, 'z_channel', None)
        if source_depth_map is not None:
            try:
                import cv2

                depth_slice = source_depth_map[cy0:cy1, cx0:cx1]
                erosion_r = int(np.clip(min(mask_clip.shape) * 0.03, 2, 12))
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * erosion_r + 1, 2 * erosion_r + 1),
                )
                interior_mask = cv2.erode(
                    valid_mask.astype(np.uint8),
                    kernel,
                    iterations=1,
                ).astype(bool)
                perimeter_mask = valid_mask & ~interior_mask

                interior_depths = depth_slice[interior_mask]
                interior_depths = interior_depths[~np.isnan(interior_depths)]

                if len(interior_depths) >= 10 and perimeter_mask.any():
                    ref_depth = np.median(interior_depths)
                    interior_spread = np.std(interior_depths)
                    abs_floor = max(0.02, ref_depth * 0.005)
                    full_tol = interior_spread * 2.0 + abs_floor
                    dist = cv2.distanceTransform(valid_mask.astype(np.uint8), cv2.DIST_L2, 5)
                    norm_dist = np.clip(dist / max(erosion_r, 1), 0.0, 1.0)
                    per_pixel_tol = abs_floor + (full_tol - abs_floor) * norm_dist
                    with np.errstate(invalid='ignore'):
                        perimeter_depth_ok = np.abs(depth_slice - ref_depth) <= per_pixel_tol
                    valid_mask = interior_mask | (perimeter_mask & perimeter_depth_ok)
            except Exception:
                pass

        raw_ids = index_slice[valid_mask]
        unique_ids = np.unique(raw_ids)
        return unique_ids[unique_ids > -1].astype(np.int64, copy=False)

    def _extract_semantic_element_votes(self,
                                        source_camera,
                                        source_mask_annotation,
                                        prediction_regions=None):
        """Build raw element/class vote arrays for semantic propagation."""
        if source_camera is None or source_mask_annotation is None:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), {}

        raster = getattr(source_camera, '_raster', None)
        if getattr(raster, 'index_map', None) is None:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), {}

        lock_bit = source_mask_annotation.LOCK_BIT
        element_chunks = []
        class_chunks = []
        class_label_ids = {}

        def _append_votes(real_class_id, label, raw_element_ids):
            if label is None or raw_element_ids is None:
                return

            raw_element_ids = np.asarray(raw_element_ids, dtype=np.int64).ravel()
            raw_element_ids = raw_element_ids[raw_element_ids > -1]
            if raw_element_ids.size == 0:
                return

            real_class_id = int(real_class_id)
            class_label_ids[real_class_id] = label.id
            element_chunks.append(raw_element_ids)
            class_chunks.append(
                np.full(raw_element_ids.size, real_class_id, dtype=np.int64)
            )

        if prediction_regions is not None:
            for region_mask, top_left in prediction_regions:
                if region_mask is None:
                    continue

                region_mask = np.asarray(region_mask)
                if region_mask.ndim != 2:
                    continue

                unique_real_ids = np.unique(region_mask % lock_bit)
                unique_real_ids = unique_real_ids[unique_real_ids > 0]
                for real_class_id in unique_real_ids:
                    label = source_mask_annotation.class_id_to_label_map.get(int(real_class_id))
                    if label is None:
                        continue

                    binary_mask = (region_mask % lock_bit == real_class_id)
                    if not np.any(binary_mask):
                        continue

                    raw_element_ids = self._extract_source_element_ids_from_region(
                        source_camera,
                        binary_mask,
                        top_left,
                    )
                    _append_votes(real_class_id, label, raw_element_ids)
        else:
            semantic_mask = np.asarray(source_mask_annotation.mask_data)
            if semantic_mask.ndim != 2:
                return np.array([], dtype=np.int64), np.array([], dtype=np.int64), {}

            unique_real_ids = np.unique(semantic_mask % lock_bit)
            unique_real_ids = unique_real_ids[unique_real_ids > 0]
            for real_class_id in unique_real_ids:
                label = source_mask_annotation.class_id_to_label_map.get(int(real_class_id))
                if label is None:
                    continue

                binary_mask = (semantic_mask % lock_bit == real_class_id)
                if not np.any(binary_mask):
                    continue

                raw_element_ids = self._extract_source_element_ids_from_full_mask(
                    source_camera,
                    binary_mask,
                )
                _append_votes(real_class_id, label, raw_element_ids)

        if not element_chunks or not class_chunks:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), {}

        return (
            np.concatenate(element_chunks).astype(np.int64, copy=False),
            np.concatenate(class_chunks).astype(np.int64, copy=False),
            class_label_ids,
        )

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
        if self.selected_camera is None:
            return

        px = int(scene_pos.x())
        py = int(scene_pos.y())

        selected_paths = self._get_annotation_target_paths()

        project_labels = list(self.main_window.label_window.labels)

        # Quick exit: nothing to propagate to
        if not selected_paths:
            return

        source_label, source_class_id = self._resolve_source_mask_class_context(
            self.selected_camera,
            label_id,
            project_labels,
        )
        if source_label is None or source_class_id is None:
            return

        painted_ids = self._extract_source_ids_from_crop_mask(
            self.selected_camera,
            brush_mask,
            px,
            py,
        )
        painted_ids = np.asarray(painted_ids if painted_ids is not None else [], dtype=np.int64)
        class_ids = np.full(painted_ids.size, int(source_class_id), dtype=np.int64)

        self._execute_mask_propagation(
            source_camera=self.selected_camera,
            element_ids=painted_ids,
            class_ids=class_ids,
            target_paths=selected_paths,
            project_labels=project_labels,
            class_label_ids={int(source_class_id): label_id},
            fallback_payload={
                'mode': 'brush',
                'label_id': label_id,
                'source_class_id': int(source_class_id),
                'mask': np.asarray(brush_mask, dtype=bool),
                'center': (px, py),
                'projections': self._build_projection(px, py),
                'search_radius': float(max(brush_mask.shape) * 2.5),
            },
        )

    def _on_fill_stroke_applied(self, scene_pos, label_id: str, fill_mask=None):
        """Propagate a fill operation into all visible context cameras."""
        if self.selected_camera is None:
            return

        selected_paths = self._get_annotation_target_paths()
        if not selected_paths:
            return

        project_labels = list(self.main_window.label_window.labels)
        source_label = next((lbl for lbl in project_labels if lbl.id == label_id), None)
        if source_label is None:
            return

        source_raster = self.raster_manager.get_raster(self.selected_camera.image_path)
        source_mask = source_raster.get_mask_annotation(project_labels) if source_raster else None
        source_class_id = None
        if source_mask is not None:
            source_class_id = source_mask.label_id_to_class_id_map.get(label_id)
            if source_class_id is None:
                source_mask.sync_label_map([source_label])
                source_class_id = source_mask.label_id_to_class_id_map.get(label_id)

        px = int(scene_pos.x())
        py = int(scene_pos.y())

        painted_ids = None
        if fill_mask is not None:
            painted_ids = self._extract_source_ids_from_full_mask(self.selected_camera, fill_mask)

        painted_ids = np.asarray(painted_ids if painted_ids is not None else [], dtype=np.int64)
        class_ids = (
            np.full(painted_ids.size, int(source_class_id), dtype=np.int64)
            if source_class_id is not None
            else np.array([], dtype=np.int64)
        )

        self._execute_mask_propagation(
            source_camera=self.selected_camera,
            element_ids=painted_ids,
            class_ids=class_ids,
            target_paths=selected_paths,
            project_labels=project_labels,
            class_label_ids={int(source_class_id): label_id} if source_class_id is not None else {},
            fallback_payload={
                'mode': 'fill',
                'label_id': label_id,
                'source_class_id': int(source_class_id) if source_class_id is not None else None,
                'mask': np.asarray(fill_mask, dtype=bool) if fill_mask is not None else None,
                'center': (px, py),
                'projections': self._build_projection(px, py),
                'search_radius': float(max(fill_mask.shape) * 2.5) if fill_mask is not None else 0.0,
            },
        )

    def _on_erase_stroke_applied(self, scene_pos, label_id: str, brush_mask: np.ndarray):
        """Propagate an erase operation into all visible context cameras."""
        if self.selected_camera is None:
            return

        selected_paths = self._get_annotation_target_paths()
        if not selected_paths:
            return

        px, py = int(scene_pos.x()), int(scene_pos.y())
        painted_ids = self._extract_source_ids_from_crop_mask(self.selected_camera, brush_mask, px, py)
        painted_ids = np.asarray(painted_ids if painted_ids is not None else [], dtype=np.int64)

        self._execute_mask_propagation(
            source_camera=self.selected_camera,
            element_ids=painted_ids,
            class_ids=np.zeros(painted_ids.size, dtype=np.int64),
            target_paths=selected_paths,
            project_labels=list(self.main_window.label_window.labels),
            class_label_ids={},
            fallback_payload={
                'mode': 'erase',
                'source_class_id': 0,
                'mask': np.asarray(brush_mask, dtype=bool),
                'center': (px, py),
                'projections': self._build_projection(px, py),
                'search_radius': float(max(brush_mask.shape) * 2.5),
            },
        )

    def _propagate_3d_face_ids_to_context_cameras(self, face_ids, label, erase: bool = False):
        """Propagate a 3D brush/erase stroke into visible context cameras."""
        if self.selected_camera is None:
            return

        selected_paths = self._get_annotation_target_paths()
        annotation_window = getattr(self.main_window, 'annotation_window', None)
        primary_path = getattr(annotation_window, 'current_image_path', None)
        if primary_path:
            selected_paths.add(primary_path)
        if not selected_paths:
            return

        try:
            face_ids_arr = np.asarray(face_ids, dtype=np.int64)
            face_ids_arr = np.unique(face_ids_arr[face_ids_arr >= 0])
        except Exception:
            return

        if face_ids_arr.size == 0:
            return

        project_labels = list(self.main_window.label_window.labels)
        label_id = getattr(label, 'id', None)

        if erase:
            source_class_id = 0
        else:
            if label is None or label_id is None:
                return

            source_raster = self.raster_manager.get_raster(self.selected_camera.image_path)
            source_mask = source_raster.get_mask_annotation(project_labels) if source_raster else None
            if source_mask is None:
                return

            source_class_id = source_mask.label_id_to_class_id_map.get(label_id)
            if source_class_id is None:
                source_mask.sync_label_map([label])
                source_class_id = source_mask.label_id_to_class_id_map.get(label_id)
            if source_class_id is None:
                return

        self._execute_mask_propagation(
            source_camera=self.selected_camera,
            element_ids=face_ids_arr,
            class_ids=np.full(face_ids_arr.size, int(source_class_id), dtype=np.int64),
            target_paths=selected_paths,
            project_labels=project_labels,
            class_label_ids={int(source_class_id): label_id} if int(source_class_id) != 0 else {},
            fallback_payload=None,
            skip_3d_paint=True,
        )

    def _on_3d_brush_stroke_applied(self, face_ids, label):
        self._propagate_3d_face_ids_to_context_cameras(face_ids, label, erase=False)

    def _on_3d_erase_stroke_applied(self, face_ids, label=None):
        self._propagate_3d_face_ids_to_context_cameras(face_ids, label, erase=True)

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
        if self.selected_camera is None:
            return

        selected_camera = self.selected_camera
        selected_paths  = self._get_annotation_target_paths()
        if not selected_paths:
            return

        project_labels  = list(self.main_window.label_window.labels)
        source_label, source_class_id = self._resolve_source_mask_class_context(
            selected_camera,
            label_id,
            project_labels,
        )
        if source_label is None or source_class_id is None:
            return

        px = int(scene_pos.x())
        py = int(scene_pos.y())
        painted_ids = self._extract_source_ids_from_sam_prediction(
            selected_camera,
            binary_mask,
            px,
            py,
        )
        painted_ids = np.asarray(painted_ids if painted_ids is not None else [], dtype=np.int64)
        class_ids = np.full(painted_ids.size, int(source_class_id), dtype=np.int64)

        self._execute_mask_propagation(
            source_camera=selected_camera,
            element_ids=painted_ids,
            class_ids=class_ids,
            target_paths=selected_paths,
            project_labels=project_labels,
            class_label_ids={int(source_class_id): label_id},
            fallback_payload={
                'mode': 'sam',
                'label_id': label_id,
                'source_class_id': int(source_class_id),
                'mask': np.asarray(binary_mask, dtype=np.uint8),
                'center': (px, py),
                'projections': self._build_projection(px, py),
                'search_radius': float(max(binary_mask.shape) * 2.5),
            },
        )

    def _execute_mask_propagation(self,
                                  source_camera,
                                  element_ids: np.ndarray,
                                  class_ids: np.ndarray,
                                  target_paths: set,
                                  project_labels: list,
                                  class_label_ids: dict,
                                  fallback_payload=None,
                                  skip_3d_paint: bool = False):
        """Queue a propagation job onto the single unified background worker."""
        t0 = perf_counter()
        if source_camera is None or not target_paths:
            return

        try:
            element_ids = np.asarray(element_ids, dtype=np.int64).ravel()
        except Exception:
            element_ids = np.array([], dtype=np.int64)

        try:
            class_ids = np.asarray(class_ids, dtype=np.int64).ravel()
        except Exception:
            class_ids = np.array([], dtype=np.int64)

        has_3d_payload = (
            element_ids.size > 0 and
            class_ids.size > 0 and
            element_ids.size == class_ids.size
        )
        if not has_3d_payload:
            element_ids = np.array([], dtype=np.int64)
            class_ids = np.array([], dtype=np.int64)

        if not has_3d_payload and fallback_payload is None:
            return

        target_paths = tuple(sorted({path for path in target_paths if path}))
        if not target_paths:
            return

        primary_target = self.viewer.scene_context.get_primary_target()
        self._pending_unified_propagation_jobs += 1
        self._propagating_annotation = True

        try:
            payload = dict(fallback_payload) if isinstance(fallback_payload, dict) else fallback_payload
            self._unified_bg_executor.submit(
                self._do_universal_propagation,
                source_camera,
                element_ids.copy(),
                class_ids.copy(),
                target_paths,
                list(project_labels),
                dict(class_label_ids or {}),
                primary_target,
                payload,
                skip_3d_paint,
            )
        except Exception:
            self._pending_unified_propagation_jobs = max(
                0,
                self._pending_unified_propagation_jobs - 1,
            )
            self._propagating_annotation = self._pending_unified_propagation_jobs > 0
            traceback.print_exc()
        finally:
            print(f"DEBUG [Sync Dispatch]: Packaged and queued job in {(perf_counter() - t0) * 1000:.2f}ms")

    def _do_universal_propagation(self,
                                  source_camera,
                                  element_ids: np.ndarray,
                                  class_ids: np.ndarray,
                                  target_paths,
                                  project_labels,
                                  class_label_ids,
                                  primary_target,
                                  fallback_payload=None,
                                  skip_3d_paint: bool = False):
        """Background worker for brush, SAM, and semantic mask propagation."""
        from time import perf_counter
        t0 = perf_counter()
        repaint_tasks = []
        mask_time = 0.0

        try:
            labels_by_id = {
                getattr(label, 'id', None): label
                for label in project_labels
                if getattr(label, 'id', None) is not None
            }

            winning_elements = np.array([], dtype=np.int64)
            winning_classes = np.array([], dtype=np.int64)
            if element_ids is not None and class_ids is not None:
                winning_elements, winning_classes = resolve_class_conflicts_vectorized(
                    element_ids,
                    class_ids,
                )

            class_to_elements = {}
            if winning_elements.size > 0 and winning_classes.size > 0:
                for source_class_id in np.unique(winning_classes):
                    class_to_elements[int(source_class_id)] = winning_elements[
                        winning_classes == source_class_id
                    ]

            if primary_target and hasattr(primary_target, 'apply_labels') and not skip_3d_paint:
                for source_class_id, subset_elements in class_to_elements.items():
                    if subset_elements.size == 0:
                        continue

                    label_id = class_label_ids.get(int(source_class_id))
                    if int(source_class_id) == 0:
                        repaint_tasks.append({
                            'type': '3d_paint',
                            'painted_ids': subset_elements.copy(),
                            'target_color': (255, 255, 255),
                            'source_class_id': 0,
                            'primary_target': primary_target,
                        })
                        continue

                    label = labels_by_id.get(label_id)
                    if label is None:
                        continue

                    repaint_tasks.append({
                        'type': '3d_paint',
                        'painted_ids': subset_elements.copy(),
                        'target_color': (
                            label.color.red(),
                            label.color.green(),
                            label.color.blue(),
                        ),
                        'source_class_id': int(source_class_id),
                        'primary_target': primary_target,
                    })

            centers = getattr(primary_target, '_element_centers_np', None)

            fallback_mode = None
            fallback_mask = None
            fallback_center = None
            fallback_search_radius = 0.0
            fallback_skip_ortho_index_lookup = False
            fallback_label_id = None
            fallback_projections = {}

            if isinstance(fallback_payload, dict):
                fallback_mode = str(fallback_payload.get('mode', '')).strip().lower()
                if fallback_payload.get('mask') is not None:
                    fallback_mask = np.asarray(fallback_payload.get('mask'))
                fallback_center = fallback_payload.get('center')
                fallback_search_radius = float(fallback_payload.get('search_radius', 0.0) or 0.0)
                fallback_skip_ortho_index_lookup = bool(fallback_payload.get('skip_ortho_index_lookup', False))
                fallback_label_id = fallback_payload.get('label_id')
                fallback_projections = fallback_payload.get('projections', {}) or {}

            def _project_bbox_for_subset(target_camera, subset_elements):
                if centers is None or target_camera is None or target_camera is self.ortho_camera:
                    return None
                try:
                    subset_centers = np.asarray(centers[np.asarray(subset_elements, dtype=np.int64)], dtype=np.float64)
                    if subset_centers.size == 0:
                        return None

                    min_pt = np.min(subset_centers, axis=0)
                    max_pt = np.max(subset_centers, axis=0)
                    corners_3d = np.array([
                        [min_pt[0], min_pt[1], min_pt[2]],
                        [min_pt[0], min_pt[1], max_pt[2]],
                        [min_pt[0], max_pt[1], min_pt[2]],
                        [min_pt[0], max_pt[1], max_pt[2]],
                        [max_pt[0], min_pt[1], min_pt[2]],
                        [max_pt[0], min_pt[1], max_pt[2]],
                        [max_pt[0], max_pt[1], min_pt[2]],
                        [max_pt[0], max_pt[1], max_pt[2]],
                    ], dtype=np.float64)

                    projected = []
                    for corner in corners_3d:
                        uv = target_camera.project(corner)
                        if uv is None or np.isnan(uv).any():
                            continue
                        projected.append(uv)

                    if not projected:
                        return None

                    projected = np.asarray(projected, dtype=np.float64)
                    u_min = max(0, int(np.floor(np.min(projected[:, 0]))))
                    u_max = min(int(target_camera.width), int(np.ceil(np.max(projected[:, 0]))) + 1)
                    v_min = max(0, int(np.floor(np.min(projected[:, 1]))))
                    v_max = min(int(target_camera.height), int(np.ceil(np.max(projected[:, 1]))) + 1)
                    if u_min >= u_max or v_min >= v_max:
                        return None

                    margin_u = max(1, int(round((u_max - u_min) * 0.2)))
                    margin_v = max(1, int(round((v_max - v_min) * 0.2)))
                    return (
                        max(0, u_min - margin_u),
                        min(int(target_camera.width), u_max + margin_u),
                        max(0, v_min - margin_v),
                        min(int(target_camera.height), v_max + margin_v),
                    )
                except Exception:
                    return None

            for target_path in target_paths:
                target_camera = self._get_camera_for_path(target_path)
                if target_camera is None:
                    continue

                target_raster = self.raster_manager.get_raster(target_path)
                if target_raster is None:
                    continue

                target_mask = target_raster.mask_annotation
                if target_mask is None:
                    target_mask = target_raster.get_mask_annotation(project_labels)
                if target_mask is None:
                    continue

                target_has_index = (
                    getattr(target_camera, '_raster', None) is not None and
                    target_camera._raster.index_map is not None
                )
                use_index_lookup = (
                    winning_elements.size > 0 and
                    target_has_index and
                    not (fallback_skip_ortho_index_lookup and target_camera is self.ortho_camera)
                )

                target_rect = None
                target_label_ids = set()

                if use_index_lookup:
                    t_mask_start = perf_counter()
                    target_index_map = target_camera._raster.index_map
                    target_mask_data = target_mask.mask_data
                    max_idx = int(np.max(target_index_map))

                    for source_class_id, subset_elements in class_to_elements.items():
                        if subset_elements.size == 0:
                            continue

                        if int(source_class_id) == 0:
                            target_class_id = 0
                            label_id = None
                        else:
                            label_id = class_label_ids.get(int(source_class_id))
                            label = labels_by_id.get(label_id)
                            if label is None:
                                continue

                            target_class_id = target_mask.label_id_to_class_id_map.get(label_id)
                            if target_class_id is None:
                                target_mask.sync_label_map([label])
                                target_class_id = target_mask.label_id_to_class_id_map.get(label_id)
                            if target_class_id is None:
                                continue

                        cam_bbox = _project_bbox_for_subset(target_camera, subset_elements)
                        if cam_bbox is not None:
                            u_min, u_max, v_min, v_max = cam_bbox
                            u_min = max(0, int(u_min))
                            u_max = min(target_camera.width, int(u_max))
                            v_min = max(0, int(v_min))
                            v_max = min(target_camera.height, int(v_max))
                        else:
                            u_min, u_max = 0, target_camera.width
                            v_min, v_max = 0, target_camera.height

                        if u_min >= u_max or v_min >= v_max:
                            continue

                        valid_elements = subset_elements[(subset_elements >= 0) & (subset_elements <= max_idx)]
                        if valid_elements.size == 0:
                            continue

                        lut = np.zeros(max_idx + 2, dtype=bool)
                        lut[valid_elements] = True

                        sub_index = target_index_map[v_min:v_max, u_min:u_max]
                        sub_mask = target_mask_data[v_min:v_max, u_min:u_max]
                        paintable = lut[sub_index] & (sub_mask < target_mask.LOCK_BIT) & (sub_mask != target_class_id)

                        if not np.any(paintable):
                            continue

                        y_local, x_local = np.where(paintable)
                        y_global = y_local + v_min
                        x_global = x_local + u_min
                        flat_global = (y_global * target_camera.width + x_global).astype(np.int64, copy=False)

                        target_mask.update_mask_at_indices(
                            flat_global,
                            int(target_class_id),
                            silent=True,
                        )

                        new_rect = (
                            int(np.min(x_global)),
                            int(np.min(y_global)),
                            int(np.max(x_global)) + 1,
                            int(np.max(y_global)) + 1,
                        )
                        target_rect = _merge_update_rects(target_rect, new_rect)

                        if label_id is not None:
                            target_label_ids.add(label_id)

                    mask_time += perf_counter() - t_mask_start

                if fallback_mask is not None and fallback_center is not None and (not use_index_lookup or target_rect is None):
                    t_mask_start = perf_counter()
                    if fallback_mode == 'erase':
                        target_class_id = 0
                        label = None
                    else:
                        label = labels_by_id.get(fallback_label_id)
                        if label is not None:
                            target_class_id = target_mask.label_id_to_class_id_map.get(fallback_label_id)
                            if target_class_id is None:
                                target_mask.sync_label_map([label])
                                target_class_id = target_mask.label_id_to_class_id_map.get(fallback_label_id)
                        else:
                            target_class_id = None

                    if target_class_id is not None:
                        target_center = fallback_center
                        if fallback_projections:
                            proj = fallback_projections.get(target_path)
                            if proj is not None and len(proj) >= 3 and proj[2]:
                                target_center = (proj[0], proj[1])
                            else:
                                continue

                        if target_center is None:
                            target_center = (0, 0)
                        if fallback_mode in ('brush', 'erase'):
                            brush_mask = self._acquire_propagation_buffer(fallback_mask.shape, dtype=bool)
                            try:
                                np.copyto(brush_mask, np.asarray(fallback_mask, dtype=bool))
                                brush_h, brush_w = brush_mask.shape
                                brush_location = QPointF(
                                    target_center[0] - brush_w / 2.0,
                                    target_center[1] - brush_h / 2.0,
                                )
                                target_mask.update_mask(
                                    brush_location,
                                    brush_mask,
                                    int(target_class_id),
                                    silent=True,
                                )
                                target_rect = _merge_update_rects(
                                    target_rect,
                                    (
                                        max(0, int(target_center[0] - brush_w / 2.0)),
                                        max(0, int(target_center[1] - brush_h / 2.0)),
                                        min(target_camera.width, int(target_center[0] - brush_w / 2.0) + brush_w),
                                        min(target_camera.height, int(target_center[1] - brush_h / 2.0) + brush_h),
                                    ),
                                )
                                if fallback_label_id is not None:
                                    target_label_ids.add(fallback_label_id)
                            finally:
                                self._release_propagation_buffer(brush_mask)
                        elif fallback_mode == 'fill':
                            fill_pos = QPointF(target_center[0], target_center[1])
                            fill_result = target_mask.fill_region(
                                fill_pos,
                                int(target_class_id),
                                silent=True,
                                return_update_rect=True,
                            )
                            if fill_result is not None:
                                fill_mask_result, fill_rect = fill_result
                                if fill_mask_result is not None:
                                    target_rect = _merge_update_rects(target_rect, fill_rect)
                                    if fallback_label_id is not None:
                                        target_label_ids.add(fallback_label_id)
                        elif fallback_mode == 'sam':
                            subset_mask = self._acquire_propagation_buffer(fallback_mask.shape, dtype=np.uint8)
                            try:
                                subset_mask.fill(0)
                                subset_mask[np.asarray(fallback_mask, dtype=bool)] = int(target_class_id)
                                mask_h, mask_w = subset_mask.shape
                                top_left_x = int(target_center[0] - mask_w / 2.0)
                                top_left_y = int(target_center[1] - mask_h / 2.0)
                                target_mask.update_mask_with_mask(
                                    subset_mask,
                                    (top_left_x, top_left_y),
                                    silent=True,
                                )
                                target_rect = _merge_update_rects(
                                    target_rect,
                                    (
                                        max(0, top_left_x),
                                        max(0, top_left_y),
                                        min(target_mask.mask_data.shape[1], top_left_x + mask_w),
                                        min(target_mask.mask_data.shape[0], top_left_y + mask_h),
                                    ),
                                )
                                if fallback_label_id is not None:
                                    target_label_ids.add(fallback_label_id)
                            finally:
                                self._release_propagation_buffer(subset_mask)
                    mask_time += perf_counter() - t_mask_start

                if target_rect is not None:
                    repaint_tasks.append({
                        'type': 'repaint',
                        'path': target_path,
                        'mask': target_mask,
                        'label_ids': tuple(sorted(target_label_ids)),
                        'update_rect': target_rect,
                    })

        except Exception:
            traceback.print_exc()
        finally:
            self._universal_repaint_signal.emit(repaint_tasks)
            print(
                f"DEBUG [Sync Worker]: {len(target_paths)} Cams | Total: {(perf_counter() - t0) * 1000:.2f}ms | "
                f"Mask Gen: {mask_time * 1000:.2f}ms"
            )
            return repaint_tasks

    def _on_universal_repaint(self, repaint_tasks: list):
        """Apply localized UI updates produced by the unified propagation worker."""
        t0 = perf_counter()
        needs_3d_flush = False
        try:
            for task in repaint_tasks:
                task_type = task.get('type')
                if task_type == '3d_paint':
                    self.submit_3d_face_paint(
                        task['painted_ids'],
                        task['target_color'],
                        task['source_class_id'],
                        primary_target=task.get('primary_target'),
                    )
                    needs_3d_flush = True
                    continue

                if task_type != 'repaint':
                    continue

                target_mask = task.get('mask')
                if target_mask is None:
                    continue

                for label_id in task.get('label_ids', ()):
                    if label_id is not None and label_id not in target_mask.visible_label_ids:
                        target_mask.visible_label_ids.add(label_id)

                target_path = task.get('path')
                context_canvas = self._get_context_canvas_for_path(target_path)
                should_update_now = True
                if self.context_matrix is not None and context_canvas is not None:
                    should_update_now = self.context_matrix.is_canvas_on_screen(context_canvas)

                if should_update_now:
                    target_mask.update_graphics_item(update_rect=task.get('update_rect'))
                    if context_canvas is not None and context_canvas._mask_overlay_item is None:
                        context_canvas.set_mask_overlay(target_mask)
                elif self.context_matrix is not None:
                    self.context_matrix.queue_pending_repaint(
                        target_path,
                        target_mask,
                        update_rect=task.get('update_rect'),
                        label_ids=task.get('label_ids', ()),
                    )

        except Exception as e:
            print(f"Error in _on_universal_repaint: {e}")
        finally:
            self._pending_unified_propagation_jobs = max(
                0,
                self._pending_unified_propagation_jobs - 1,
            )
            self._propagating_annotation = self._pending_unified_propagation_jobs > 0
            if needs_3d_flush:
                request_flush = getattr(self, 'request_lazy_flush', None)
                if callable(request_flush):
                    request_flush()
            print(f"DEBUG [Sync Repaint (Main Thread)]: Pushed {len(repaint_tasks)} image updates to UI in {(perf_counter() - t0) * 1000:.2f}ms")

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
        source_camera = self._get_camera_for_path(image_path)
        if source_camera is None:
            return

        selected_paths = self._get_semantic_target_paths(source_camera)

        if not selected_paths:
            return

        project_labels = list(self.main_window.label_window.labels)
        element_ids, class_ids, class_label_ids = self._extract_semantic_element_votes(
            source_camera,
            source_mask_annotation,
            prediction_regions=prediction_regions,
        )
        if element_ids.size == 0 or class_ids.size == 0 or not class_label_ids:
            return

        self._execute_mask_propagation(
            source_camera=source_camera,
            element_ids=element_ids,
            class_ids=class_ids,
            target_paths=selected_paths,
            project_labels=project_labels,
            class_label_ids=class_label_ids,
        )

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

        # Shutdown unified propagation executor
        try:
            if hasattr(self, '_unified_bg_executor'):
                self._unified_bg_executor.shutdown(wait=False)
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
