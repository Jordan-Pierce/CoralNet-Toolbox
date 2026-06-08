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
)

from coralnet_toolbox.MVAT.core.Products import MeshProduct
from coralnet_toolbox.MVAT.managers.MousePositionBridge import MousePositionBridge
from coralnet_toolbox.MVAT.managers.PropagationEngine import PropagationEngine

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation


logger = get_visibility_logger()


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MVATManager(QObject):
    """
    Core Controller for the MVAT Workspace.
    """
    cameraSelectedInMVAT = pyqtSignal(str)
    contextStatsComputed = pyqtSignal(int, str, int, int)
    _orthoIndexMapReady  = pyqtSignal(object)  # internal: carries result dict from worker thread

    def __getattr__(self, name):
        """Delegate unknown attributes to propagation_engine.

        Allows tool code (e.g. BrushTool3D) and legacy call sites to call
        ``manager._on_3d_brush_stroke_applied`` etc. without knowing they now
        live on PropagationEngine.

        IMPORTANT: We deliberately do NOT use ``getattr(pe, name)`` here
        because PropagationEngine.__getattr__ delegates unknown names BACK to
        this manager, creating infinite recursion.  Instead we look up the
        attribute on pe using object.__getattribute__ (instance dict only) and
        then walk pe's MRO for class-level attributes (bound methods, etc.).
        If the name is not found on pe at all we raise AttributeError cleanly.
        """
        try:
            pe = object.__getattribute__(self, 'propagation_engine')
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # 1. Instance dict (fastest path — avoids triggering pe's __getattr__)
        try:
            return object.__getattribute__(pe, name)
        except AttributeError:
            pass

        # 2. Class hierarchy (covers methods defined on PropagationEngine itself)
        for klass in type(pe).__mro__:
            if name in klass.__dict__:
                val = klass.__dict__[name]
                # Bind descriptors (plain functions become bound methods)
                if hasattr(val, '__get__'):
                    return val.__get__(pe, type(pe))
                return val

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

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
        self._focal_point_locked = False  # True while ContextMatrix is pinned to a 3D static marker
        self._context_view_path = None
        self._projected_cursor_context = None
        
        # Data Settings
        self.compute_depth_maps_enabled = True
        # New toggle: whether to compute index maps in background
        self.compute_index_maps_enabled = True
        # Stage 1 densification: expand sparse index-map-sampled face IDs into a
        # dense surface patch via the face-center KD-tree before propagation.
        # See MeshProduct.gather_dense_face_ids / PropagationEngine._densify_source_ids.
        # Tunable from the Propagation Hub dropdown (Settings → Densify).
        self.densify_enabled = True
        self.densify_radius_mult = 1.5      # gather radius vs. median seed spacing
        self.densify_normal_dot_min = 0.3   # lower = include more curved surface
        self.densify_max_expansion = 40.0   # circuit-breaker ceiling (× seed count)
        # Maximum pixel budget for background index map computation
        self.pixel_budget = 4_000_000  # Default to ~4 Megapixels
        # Number of parallel workers for cache disk I/O
        self._cache_n_workers = 4  # Default to 4 workers
        # Safety factor for distortion VRAM allocation (0.8 = 80% of free VRAM)
        # Can be overridden via env var: MVAT_DISTORTION_VRAM_SAFETY=0.9
        try:
            env_vram_safety = os.environ.get('MVAT_DISTORTION_VRAM_SAFETY', None)
            self._distortion_vram_safety_factor = float(env_vram_safety) if env_vram_safety else 0.8
        except (ValueError, TypeError):
            self._distortion_vram_safety_factor = 0.8
        # Safety flag to prevent concurrent visibility computations
        self._is_computing_visibility = False
        # Track active worker threads to prevent GC
        self._active_workers = []
        self._context_stats_request_id = 0
        self._latest_context_stats_request_id = 0
        self._depth_build_lock = threading.Lock()
        self._pending_depth_build_paths = set()

        # PropagationEngine handles all mask propagation and multi-annotate logic
        self.propagation_engine = PropagationEngine(self)

        # Ortho state: chunk transform T and OrthoCamera (set during load_cameras
        # when an OrthoRaster is present in the project)
        self._chunk_transform = None  # 4x4 Metashape chunk transform matrix
        self.ortho_camera = None      # OrthoCamera instance
        self._computing_ortho_index_map = False  # guard against concurrent builds

        # Internal Managers
        self.selection_model = SelectionManager(self)
        self.cache_manager = CacheManager("")
        self.mouse_bridge = MousePositionBridge(self)

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
            self.context_matrix.loadIndexMapsRequested.connect(self.load_index_maps)
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
            self.context_matrix.maskPropagationRequested.connect(
                self._on_mask_propagation_requested
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
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
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
            QApplication.restoreOverrideCursor()
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
                # dtype slot is unused: face IDs are always int32 (tiered encoding removed)
                choice_mode, new_budget, n_workers, _dtype, enable_cache = self._prompt_visibility_quality_dialog(
                    len(uncached_cameras)
                )

                if choice_mode is None:
                    return

                previous_budget = getattr(self, 'pixel_budget', None)
                self.pixel_budget = new_budget
                self._cache_n_workers = n_workers  # Store for use in _compute_visibility_async
                self.debug_enable_cache = enable_cache  # Debug: enable/disable caching

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

    def load_index_maps(self):
        """Attempt to load pre-computed index maps from disk cache for all loaded cameras.

        Walks every loaded perspective camera and tries the cache manager.  Cameras
        that already have an index map in RAM are skipped.  This lets the user
        trigger a bulk cache-load without navigating to each camera individually.
        """
        if not self.cameras:
            self.main_window.status_bar.showMessage("No cameras loaded yet.", 3000)
            return

        primary_target = self.viewer.scene_context.get_primary_target()
        if primary_target is None or self.cache_manager is None:
            self.main_window.status_bar.showMessage(
                "No 3D target or cache manager available for index map loading.", 4000
            )
            return

        target_file_path = primary_target.file_path
        element_type = primary_target.get_element_type()

        candidates = [
            cam for cam in self.cameras.values()
            if getattr(cam._raster, 'index_map', None) is None
        ]

        if not candidates:
            self.main_window.status_bar.showMessage(
                "All cameras already have index maps loaded.", 3000
            )
            return

        self.main_window.status_bar.showMessage(
            f"Loading index maps for {len(candidates)} camera(s)...", 0
        )
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        loaded = 0
        skipped = 0
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _load_one(cam):
                cache_key = cam._raster.extrinsics
                extra = (
                    cam._raster.dist_coeffs.tobytes()
                    if cam.is_distorted and cam._raster.dist_coeffs is not None
                    else None
                )
                try:
                    return cam, self.cache_manager.load_visibility(
                        cache_key, target_file_path, element_type, extra,
                        pixel_budget=self.pixel_budget,
                    )
                except Exception as exc:
                    print(f"load_index_maps: cache load error for {cam.label}: {exc}")
                    return cam, None

            n_workers = min(8, max(1, len(candidates)))
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futs = {pool.submit(_load_one, cam): cam for cam in candidates}
                for fut in as_completed(futs):
                    cam, data = fut.result()
                    if data is not None:
                        cache_path = data.get('cache_path') or self.cache_manager.get_cache_path(
                            cam._raster.extrinsics, target_file_path, element_type,
                            (cam._raster.dist_coeffs.tobytes()
                             if cam.is_distorted and cam._raster.dist_coeffs is not None
                             else None),
                            pixel_budget=self.pixel_budget,
                        )
                        cam._raster.add_index_map(
                            data.get('index_map'),
                            cache_path,
                            data.get('visible_indices'),
                            element_type=element_type,
                            inverted_index=data.get('inverted_index'),
                        )
                        if self.compute_depth_maps_enabled:
                            depth_map = data.get('depth_map')
                            if depth_map is not None:
                                try:
                                    cam._raster.merge_or_set_depth_map(depth_map)
                                except Exception:
                                    pass
                        loaded += 1
                    else:
                        skipped += 1
        finally:
            QApplication.restoreOverrideCursor()
            self.main_window.status_bar.showMessage(
                f"Index maps: {loaded} loaded from cache, {skipped} not cached.", 5000
            )

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
        """Prompt for visibility quality, cache workers, and whether to compute now or defer."""
        from PyQt5.QtWidgets import (
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QSlider,
            QVBoxLayout,
        )
        from PyQt5.QtCore import Qt
        import os

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

        # Determine max workers (CPU count - 1, so user can never select max)
        try:
            import psutil
            max_workers = max(1, psutil.cpu_count() - 1)
        except Exception:
            max_workers = max(1, (os.cpu_count() or 4) - 1)

        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("Pre-compute Visibility")
        dialog.setModal(True)
        dialog.resize(520, 320)

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

        quality_combo.setCurrentIndex(0)  # Always default to "Native (Full Resolution)"
        form_layout.addRow("Visibility Quality:", quality_combo)

        # Add n_workers slider with tick labels
        workers_slider = QSlider(Qt.Horizontal)
        workers_slider.setMinimum(1)
        workers_slider.setMaximum(max_workers)
        workers_slider.setValue(max_workers)  # Default to max workers
        workers_slider.setTickPosition(QSlider.TicksBelow)

        # Set tick interval and add labels at key positions
        tick_interval = max(1, max_workers // 4)
        workers_slider.setTickInterval(tick_interval)

        workers_slider.setToolTip(
            "Controls parallel disk writes during cache phase. "
            "Higher = faster caching but more I/O concurrency."
        )

        # Create tick mark labels below slider
        workers_labels_layout = QHBoxLayout()
        workers_labels_layout.setContentsMargins(0, 5, 0, 0)  # Small top margin

        # Generate tick labels at intervals
        tick_positions = []
        for i in range(1, max_workers + 1, tick_interval):
            tick_positions.append(i)
        # Always include max if not already there
        if tick_positions[-1] != max_workers:
            tick_positions.append(max_workers)

        # Distribute labels evenly across the slider width
        for i, tick_val in enumerate(tick_positions):
            label = QLabel(str(tick_val))
            label.setStyleSheet("font-size: 10px; color: gray;")
            if i == 0:
                workers_labels_layout.addWidget(label, 0, Qt.AlignLeft)
            elif i == len(tick_positions) - 1:
                workers_labels_layout.addWidget(label, 0, Qt.AlignRight)
            else:
                workers_labels_layout.addWidget(label, 1, Qt.AlignCenter)

        workers_container = QVBoxLayout()
        workers_container.addWidget(workers_slider)
        workers_container.addLayout(workers_labels_layout)

        form_layout.addRow("Cache Workers:", workers_container)
        layout.addLayout(form_layout)

        # Debugging section (editable controls for experimentation)
        debug_groupbox = QGroupBox("Debugging")
        debug_groupbox.setCheckable(False)

        debug_layout = QFormLayout(debug_groupbox)

        dtype_combo = QComboBox(dialog)
        dtype_options = ["r8", "rg16", "rgb24", "int32"]
        dtype_combo.addItems(dtype_options)
        dtype_combo.setCurrentIndex(3)  # Default to int32 (most reliable, fastest)
        dtype_combo.setToolTip(
            "Fragment face ID dtype for index map generation.\n"
            "r8: 1 byte per pixel (256 max faces)\n"
            "rg16: 2 bytes per pixel (65K max faces)\n"
            "rgb24: 3 bytes per pixel (16M max faces)\n"
            "int32: 4 bytes per pixel (4B max faces)"
        )
        debug_layout.addRow("Index Map Dtype:", dtype_combo)

        cache_combo = QComboBox(dialog)
        cache_combo.addItems(["Enabled", "Disabled"])
        cache_combo.setCurrentIndex(0)
        cache_combo.setToolTip(
            "Enable or disable caching during visibility computation.\n"
            "Disabled: Measures pure generation time without I/O overhead"
        )
        debug_layout.addRow("Caching:", cache_combo)

        layout.addWidget(debug_groupbox)

        button_box = QDialogButtonBox(dialog)
        compute_button = button_box.addButton("Compute Now", QDialogButtonBox.AcceptRole)
        background_button = button_box.addButton("Background", QDialogButtonBox.RejectRole)
        compute_button.clicked.connect(_accept_compute)
        background_button.clicked.connect(_reject_background)
        layout.addWidget(button_box)

        dialog.exec_()

        mode = selected_mode['mode']
        if mode is None:
            return None, None, None, None, None

        chosen_quality = quality_combo.currentText()
        n_workers = workers_slider.value()
        dtype = dtype_combo.currentText()
        enable_cache = cache_combo.currentIndex() == 0
        return mode, quality_map[chosen_quality], n_workers, dtype, enable_cache

    # --- Signal Handlers ---

    def get_propagation_camera_counts(self):
        """Return counts used by the propagation option dialogs.

        Returns a dict with:
          total          -- total loaded cameras
          have_index_map -- cameras with a pre-computed index map
          have_mask      -- cameras with both an index map and a mask annotation
        """
        total = len(self.cameras)
        have_index_map = 0
        have_mask = 0
        for camera in self.cameras.values():
            raster = getattr(camera, '_raster', None)
            if raster is None:
                continue
            if getattr(raster, 'index_map', None) is not None:
                have_index_map += 1
                if getattr(raster, 'mask_annotation', None) is not None:
                    have_mask += 1
        return {
            'total': total,
            'have_index_map': have_index_map,
            'have_mask': have_mask,
        }

    def _on_mask_propagation_requested(self, mode: str):
        """Route maskPropagationRequested(mode) to the correct PropagationEngine method.

        The mode string is one of the PROPAGATE_* constants defined in
        QtContextMatrix, optionally suffixed with option flags separated by ':'.

        Examples
        --------
        ``"active_to_context"``               → propagate_current_semantic_mask()
        ``"active_camera_to_mesh"``           → aggregate_active_camera_mask_to_mesh()
        ``"active_to_all_cameras"``           → propagate_current_semantic_mask_to_all_cameras()
        ``"cameras_to_mesh"``                 → aggregate_camera_masks_to_mesh()
        ``"mesh_to_active_camera:skip_unlabeled"`` → project_mesh_labels_to_active_camera(skip_unlabeled=True)
        ``"mesh_to_visible_cameras"``         → project_mesh_labels_to_visible_cameras()
        ``"mesh_to_cameras:skip_unlabeled"``  → project_mesh_labels_to_cameras(skip_unlabeled=True)
        ``"mesh_to_cameras:keep_all"``        → project_mesh_labels_to_cameras(skip_unlabeled=False)
        """
        from coralnet_toolbox.MVAT.ui.QtContextMatrix import (
            PROPAGATE_ACTIVE_TO_CONTEXT,
            PROPAGATE_ACTIVE_CAMERA_TO_MESH,
            PROPAGATE_ACTIVE_TO_ALL_CAMERAS,
            PROPAGATE_CAMERAS_TO_MESH,
            PROPAGATE_MESH_TO_ACTIVE_CAMERA,
            PROPAGATE_MESH_TO_VISIBLE_CAMERAS,
            PROPAGATE_MESH_TO_CAMERAS,
        )

        parts = mode.split(":")
        base_mode = parts[0]
        flags = set(parts[1:])

        if base_mode == PROPAGATE_ACTIVE_TO_CONTEXT:
            # Already handled by semanticMaskPropagationRequested → propagate_current_semantic_mask
            pass

        elif base_mode == PROPAGATE_ACTIVE_CAMERA_TO_MESH:
            self.aggregate_active_camera_mask_to_mesh()

        elif base_mode == PROPAGATE_ACTIVE_TO_ALL_CAMERAS:
            self.propagate_current_semantic_mask_to_all_cameras()

        elif base_mode == PROPAGATE_CAMERAS_TO_MESH:
            self.aggregate_camera_masks_to_mesh(also_project_to_cameras=False)

        elif base_mode == PROPAGATE_MESH_TO_ACTIVE_CAMERA:
            skip_unlabeled = "keep_all" not in flags
            self.project_mesh_labels_to_active_camera(skip_unlabeled=skip_unlabeled)

        elif base_mode == PROPAGATE_MESH_TO_VISIBLE_CAMERAS:
            skip_unlabeled = "keep_all" not in flags
            self.project_mesh_labels_to_visible_cameras(skip_unlabeled=skip_unlabeled)

        elif base_mode == PROPAGATE_MESH_TO_CAMERAS:
            skip_unlabeled = "keep_all" not in flags  # default True; opt-out with :keep_all
            self.project_mesh_labels_to_cameras(skip_unlabeled=skip_unlabeled)

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

        # Lock the ContextMatrix to this 3D marker and sync ordering/viewports from it
        self._focal_point_locked = True
        self._sync_context_from_focal_point(point_3d)

    def _sync_context_from_focal_point(self, point_3d):
        """Sync ContextMatrix ordering and viewports to a 3D world point.

        Projects point_3d into every loaded context camera and calls
        request_sync / request_zoom_only so the canvases that can actually
        see the point are snapped to it and floated to the front of the grid.
        Used when the user double-right-clicks in the 3D viewer to set a
        static marker — at that moment the ContextMatrix should reflect the
        marker position rather than the AnnotationWindow viewport center.
        """
        if self.context_matrix is None:
            return
        if not self.context_matrix.target_lock_enabled:
            return

        reference_path = self.selected_camera.image_path if self.selected_camera else None
        base_rotation = getattr(self.annotation_window, 'rotation_angle', 0.0)

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
                pixel = camera.project(point_3d)
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

        # Use a neutral relative zoom of 1.0 (fit-to-view) when snapping to a new marker
        relative_zoom = 1.0

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
            self.context_matrix.request_sync(targets_with_center, relative_zoom)
            self.context_matrix.request_zoom_only(zoom_only, relative_zoom)

    def reset_focal_lock(self):
        """Release the focal-point lock and restore ContextMatrix to image-based navigation.

        Called when the user presses Ctrl+H or Home in the AnnotationWindow.
        Clears all static markers and re-syncs the ContextMatrix from the
        AnnotationWindow's current viewport center.
        """
        self._focal_point_locked = False
        self.current_focal_point = None

        # Clear static markers in context canvases and the annotation window
        if self.context_matrix is not None:
            try:
                self.context_matrix.clear_all_static_markers()
            except Exception:
                pass

        # Trigger an immediate re-sync from the AnnotationWindow's current view
        try:
            aw = self.annotation_window
            if aw.active_image and aw.pixmap_image:
                viewport_center = aw.mapToScene(aw.viewport().rect().center())
                self._on_main_view_navigated(
                    viewport_center.x(), viewport_center.y(), aw.zoom_factor
                )
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

                # Prewarm per-face normals too — the densify gather needs them,
                # and computing them lazily on the first stroke causes a hitch on
                # large meshes.
                if hasattr(primary_target, '_get_cached_face_normals'):
                    try:
                        primary_target._get_cached_face_normals()
                    except Exception:
                        pass

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
            # ModernGL-only path (VTK removed in Phase 3)
            return VisibilityManager.compute_ortho_index_map_moderngl(
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
            from PyQt5.QtCore import QMetaObject, Q_ARG

            def _show_status(msg, timeout=0):
                try:
                    QMetaObject.invokeMethod(
                        self.main_window.status_bar,
                        "showMessage",
                        Qt.QueuedConnection,
                        Q_ARG(str, msg),
                        Q_ARG(int, timeout),
                    )
                except Exception:
                    pass

            _show_status(f"Building depth map for {camera.label}...")
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
                    _show_status(f"Depth map ready for {camera.label}.", 3000)
                else:
                    _show_status(f"Depth map could not be built for {camera.label}.", 4000)
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

            # 2. Fallback to check cache if result is not yet computed
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
            self.main_window.status_bar.showMessage(
                f"Loading index maps for {len(cache_candidates)} camera(s) from cache...", 0
            )
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            try:
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futs = {
                        pool.submit(_load_one, path, cam): path
                        for path, cam in cache_candidates.items()
                    }
                    for fut in as_completed(futs):
                        path, data = fut.result()
                        cache_results[path] = data
            finally:
                QApplication.restoreOverrideCursor()

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
        """Compute mesh visibility synchronously on the calling thread via moderngl."""
        from coralnet_toolbox.MVAT.managers.VisibilityManager import VisibilityManager

        self._is_computing_visibility = True
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            n_cameras = len(cameras)
            self.main_window.status_bar.showMessage(
                f"Computing index maps for {n_cameras} camera(s)...", 0
            )

            camera_params = [
                (cam.K_linear, cam.R, cam.t, cam.width, cam.height)
                for cam in cameras
            ]

            mgl_ctx = None
            try:
                mgl_ctx = VisibilityManager.setup_batch_moderngl_context(
                    mesh_product, self.pixel_budget,
                    cameras[0].width, cameras[0].height,
                )
                batch = VisibilityManager.compute_batch_mesh_visibility_moderngl(
                    mesh_product, camera_params,
                    compute_depth_map=self.compute_depth_maps_enabled,
                    compute_visible_indices=False,
                    pixel_budget=self.pixel_budget,
                    mgl_context=mgl_ctx,
                )
            except Exception as mgl_err:
                print(f"⚠️ moderngl sync failed ({mgl_err})")
                raise
            finally:
                if mgl_ctx is not None:
                    try:
                        for fbo in mgl_ctx.get('_fbo_cache', {}).values():
                            fbo.release()
                        mgl_ctx['ctx'].release()
                    except Exception:
                        pass

            results = {}
            for camera, result in zip(cameras, batch):
                result['element_type'] = 'face'
                if camera.is_distorted and camera._raster.intrinsics_undistorted is not None:
                    warp_fn = camera._raster.warp_linear_map_to_distorted
                    if result.get('index_map') is not None:
                        result['index_map'] = warp_fn(result['index_map'], nodata=-1)
                results[camera.image_path] = result

            self._process_visibility_results(results, mesh_product.file_path)

        except Exception as e:
            print(f"⚠️ Mesh visibility computation failed: {e}")
            import traceback; traceback.print_exc()
        finally:
            self._is_computing_visibility = False
            self.main_window.status_bar.showMessage("Index maps ready.", 3000)
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

            # Pass the cache data and scale factors to the worker
            n_workers = getattr(self, '_cache_n_workers', 4)  # Default to 4 if not set
            distortion_vram_safety = getattr(self, '_distortion_vram_safety_factor', 0.8)  # Default to 0.8
            enable_cache = getattr(self, 'debug_enable_cache', True)
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
                n_workers=n_workers,
                distortion_vram_safety_factor=distortion_vram_safety,
                enable_cache=enable_cache,
            )
            
            thread = QThread()
            worker.moveToThread(thread)
            thread.started.connect(worker.run)

            # Connect result / error signals
            worker.signals.finished.connect(self._on_visibility_computed)
            worker.signals.error.connect(self._on_visibility_error)

            # Both finished and error must quit the thread so it can be cleaned up
            worker.signals.finished.connect(thread.quit)
            worker.signals.error.connect(thread.quit)
            worker.signals.finished.connect(worker.deleteLater)

            # Remove this entry from _active_workers when the thread finishes.
            # Using a closure avoids the "wrapped C/C++ object deleted" RuntimeError
            # that occurs when thread.deleteLater() is connected to thread.finished
            # and the Python wrapper is later accessed through _active_workers.
            def _remove_worker(t=thread, w=worker):
                self._active_workers = [
                    (ot, ow) for ot, ow in self._active_workers
                    if ot is not t
                ]

            thread.finished.connect(_remove_worker)

            # Drop any stale entries left by threads that finished without a
            # clean signal (e.g. after an unhandled exception in an older run).
            def _is_alive(t):
                try:
                    return t.isRunning()
                except RuntimeError:
                    return False

            self._active_workers = [
                (t, w) for t, w in self._active_workers if _is_alive(t)
            ]

            # Keep a strong reference so neither thread nor worker is GC'd
            self._active_workers.append((thread, worker))

            thread.start()

        except Exception as e:
            print(f"Failed to start visibility worker: {e}")
            self._is_computing_visibility = False
            QApplication.restoreOverrideCursor()

    # --- Label painter management ------------------------------------------------
    def submit_3d_face_paint(self, face_ids, color_rgb, class_id: int, primary_target=None, label_id=None):
        """Queue a 3D face paint update through the shared overlay painter.

        Tool code should compute the covered face IDs, then call this helper
        instead of reaching into ``_label_painter_thread`` directly.  The helper
        keeps thread lifecycle, mesh validation, and queue submission in one
        place while leaving geometry selection in the caller.

        Args:
            label_id: Optional UUID of the label for this class_id. When provided
                     (especially during programmatic mesh painting from semantic
                     predictions or aggregation), this prevents the code from
                     looking up the active UI label, which could be incorrect.
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

        # Keep the mesh class-label registry in sync so a mesh painted
        # directly (without a prior camera -> mesh projection) can still be
        # projected back out to the cameras.  Without this, the registry stays
        # empty and project_mesh_labels_to_cameras aborts with "paint the mesh
        # first" even though primary_target.class_ids is populated.
        if int(class_id) != 0:
            try:
                engine = getattr(self, 'propagation_engine', None)
                if engine is not None:
                    # CRITICAL FIX: Use provided label_id if available to avoid
                    # relying on the active UI label, which could overwrite
                    # the wrong mesh_class_label_ids entry during semantic prediction
                    # or mesh aggregation workflows.
                    if label_id is not None:
                        engine._mesh_class_label_ids[int(class_id)] = label_id
                    else:
                        # Fallback to active label only when no label_id provided
                        # (e.g., direct brush painting on the 3D mesh)
                        active_label = self._get_active_label_widget()
                        fallback_label_id = getattr(active_label, 'id', None)
                        if fallback_label_id is not None:
                            engine._mesh_class_label_ids[int(class_id)] = fallback_label_id
            except Exception as e:
                pass

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
        """Commit painted labels to the GPU and refresh the 3D view. Runs when the user pauses."""
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
            if render:
                try:
                    last_render_time = getattr(self, '_last_vtk_render_time', None)
                    if last_render_time is None or (perf_counter() - last_render_time) > 0.033:
                        self.viewer.plotter.render()
                        self._last_vtk_render_time = perf_counter()
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
        active_tool = getattr(self.viewer, '_active_3d_tool', None)
        try:
            radius = getattr(active_tool, 'brush_size', None)
            if radius is not None:
                return float(radius)
        except Exception:
            pass

        sphere_manager = getattr(self.viewer, '_cursor_preview', None)
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

        sphere_manager = getattr(self.viewer, '_cursor_preview', None)
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
        if not self.multi_annotate_enabled:
            self._clear_projected_cursor_previews(render=render)
            return

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
                self._set_hover_overlay_geometry(None, None, render=render)
            except Exception:
                pass

            context = self._hover_overlay_context or {}
            center = context.get('center')
            if center is not None:
                try:
                    center = np.asarray(center, dtype=np.float64).reshape(-1)
                except Exception:
                    center = None

            if center is not None and center.size >= 3 and np.all(np.isfinite(center[:3])):
                self._sync_projected_cursor_previews(center[:3], render=False)
                self._sync_hover_dynamic_markers(center[:3], render=False)
            else:
                self._clear_projected_cursor_previews(render=False)
                self._clear_hover_dynamic_markers(render=False)
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

    def _reorder_cameras(self, reference_path):
        """Reorder cameras based on proximity to reference camera.

        All loaded cameras are always included so the user can scroll to any
        camera regardless of its angle relative to the active one.  Nearby
        cameras (non-zero proximity score) are floated to the front; distant /
        facing-away cameras are appended at the end.
        """
        reference_camera = self.cameras.get(reference_path)
        if not reference_camera:
            return

        nearby = []    # (path, score) with score > 0
        distant = []   # paths with score == 0

        for path, camera in self.cameras.items():
            if path == reference_path:
                nearby.append((path, float('inf')))
            else:
                score = self._calculate_camera_proximity_score(reference_camera, camera)
                if score > 0.0:
                    nearby.append((path, score))
                else:
                    distant.append(path)

        nearby.sort(key=lambda x: x[1], reverse=True)
        ordered_paths = [p for p, _ in nearby] + distant

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

    # =========================================================================
    # MVAT-SAM  (Space-bar segmentation of the current 3D viewer screenshot)
    # =========================================================================

    def _capture_viewer_sam_context(self, scale: int = 2):
        """Capture the current MVATViewer frame and build an on-the-fly index map.

        Non-mesh actors (rays, frustums, sphere markers, hover/label overlays) are
        hidden before the screenshot and restored afterwards so that only the mesh /
        point-cloud geometry appears in the image used for SAM prompting.

        The screenshot is rendered at ``scale`` x the current window resolution so
        that the resulting masks are dense even on small viewer windows.

        Returns (rgb_image, index_map, depth_map, element_type).
        Raises RuntimeError on any failure so the caller can show a message.
        """
        from coralnet_toolbox.MVAT.managers.VisibilityManager import VisibilityManager

        viewer   = self.viewer
        plotter  = viewer.plotter

        # ------------------------------------------------------------------ #
        # 1. Hide every actor except the primary geometry actor
        #    (rays, frustums, overlays, axes, etc. must not appear in the
        #     screenshot used for SAM prompting)
        # ------------------------------------------------------------------ #
        primary_actor  = viewer._get_primary_target_actor()
        actors_to_hide = []

        try:
            vtk_actors = plotter.renderer.GetActors()
            vtk_actors.InitTraversal()
            while True:
                actor = vtk_actors.GetNextActor()
                if actor is None:
                    break
                if actor is primary_actor:
                    continue          # keep the mesh/point-cloud visible
                if actor.GetVisibility():
                    actor.SetVisibility(False)
                    actors_to_hide.append(actor)
        except Exception:
            pass

        # ------------------------------------------------------------------ #
        # 2. Switch primary product to RGB array, screenshot, switch back
        # ------------------------------------------------------------------ #
        primary_target   = viewer.scene_context.get_primary_target()
        prev_array       = None
        switched_to_rgb  = False
        if primary_target is not None and hasattr(primary_target, 'get_selected_array'):
            prev_array = primary_target.get_selected_array()
            if prev_array != 'RGB' and hasattr(primary_target, 'set_selected_array'):
                if primary_target.set_selected_array('RGB'):
                    switched_to_rgb = True
                    try:
                        viewer.render_scene()
                    except Exception:
                        pass

        try:
            win_w, win_h = plotter.window_size
            render_w = win_w * scale
            render_h = win_h * scale

            plotter.render()
            rgb = plotter.screenshot(return_img=True, window_size=(render_w, render_h))
        finally:
            # Restore array
            if switched_to_rgb and prev_array is not None:
                try:
                    primary_target.set_selected_array(prev_array)
                    viewer.render_scene()
                except Exception:
                    pass
            # Restore hidden actors
            for actor in actors_to_hide:
                actor.SetVisibility(True)
            plotter.render()

        if rgb is None:
            raise RuntimeError("plotter.screenshot() returned None")
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w = rgb.shape[:2]

        # 2. Derive K / R / t from the VTK camera
        cam      = plotter.camera
        pos      = np.asarray(cam.position,    dtype=np.float64)
        focal_pt = np.asarray(cam.focal_point, dtype=np.float64)
        view_up  = np.asarray(cam.up,          dtype=np.float64)

        fwd = focal_pt - pos
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, view_up)
        norm_r = np.linalg.norm(right)
        if norm_r < 1e-9:
            raise RuntimeError("Degenerate camera: forward parallel to view_up")
        right /= norm_r
        up = np.cross(right, fwd)

        # OpenCV convention: Z forward, Y down
        R = np.stack([right, -up, fwd], axis=0)  # 3x3 world->cam
        t = -R @ pos

        # Intrinsics from VTK view_angle (full vertical FOV in degrees)
        fov_v_rad = np.radians(float(getattr(cam, 'view_angle', 30.0)))
        fy = (h / 2.0) / np.tan(fov_v_rad / 2.0)
        fx = fy
        K = np.array([[fx, 0.0, w / 2.0],
                      [0.0, fy, h / 2.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)

        # 3. Build index map via moderngl — stays on GPU when CUDA is available.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = VisibilityManager.compute_visibility_from_scene(
                scene_context=self.viewer.scene_context,
                K=K, R=R, t=t,
                width=w, height=h,
                compute_depth_map=True,
            )
        finally:
            QApplication.restoreOverrideCursor()

        index_map     = result.get('index_map',     np.full((h, w), -1, dtype=np.int32))
        depth_map     = result.get('depth_map',     None)
        element_type  = result.get('element_type',  'point')

        return rgb, index_map, depth_map, element_type

    def launch_viewer_sam(self):
        """Launch the MVAT-SAM dialog for the current 3D view (called on Space)."""
        from coralnet_toolbox.MVAT.tools.SAMTool3D import SAMTool3D

        sam_dialog = getattr(self.main_window, 'sam_deploy_predictor_dialog', None)
        if sam_dialog is None or not getattr(sam_dialog, 'loaded_model', None):
            self.main_window.status_bar.showMessage(
                'MVAT-SAM: load a SAM model first (Machine Learning -> SAM).', 4000)
            return

        selected_label = getattr(self.annotation_window, 'selected_label', None)
        if selected_label is None:
            self.main_window.status_bar.showMessage(
                'MVAT-SAM: select a label before segmenting.', 4000)
            return

        if not self.viewer.scene_context.has_any_product():
            self.main_window.status_bar.showMessage(
                'MVAT-SAM: no 3D scene loaded.', 4000)
            return

        self.main_window.status_bar.showMessage(
            'MVAT-SAM: building index map (moderngl)...', 0)
        QApplication.processEvents()
        try:
            rgb, index_map, depth_map, element_type = (
                self._capture_viewer_sam_context(scale=1)
            )
        except Exception as e:
            self.main_window.status_bar.showMessage(
                f'MVAT-SAM: capture failed - {e}', 5000)
            return

        # SAM's set_image expects numpy HWC uint8 — it handles its own GPU upload
        # and preprocessing (resize, normalise, BCHW conversion) internally.
        sam_dialog.set_image(rgb, image_path=None)

        logger.debug(
            "🎯 [MVAT-SAM] Index map: %s | element_type: %s",
            index_map.shape,
            element_type,
        )

        dlg = SAMTool3D(
            viewer=self.viewer,
            rgb_image=rgb,
            index_map=index_map,
            element_type=element_type,
            sam_dialog=sam_dialog,
            label=selected_label,
        )

        def _on_accepted(mask):
            live_label = (getattr(self.annotation_window, 'selected_label', None)
                          or selected_label)
            self._on_viewer_sam_accepted(
                mask, index_map, depth_map, element_type, live_label
            )

        dlg.maskAccepted.connect(_on_accepted)
        dlg.exec_()


    def cleanup(self):
        """Clean up resources before closing."""
        self._on_multi_annotate_toggled(False)
        self.mouse_bridge.cleanup()

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

        for thread, worker in list(self._active_workers):
            try:
                thread.quit()
                thread.wait(2000)
            except Exception:
                pass
        self._active_workers.clear()
