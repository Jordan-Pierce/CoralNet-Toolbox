import traceback
import threading
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal

from coralnet_toolbox.MVAT.managers.VisibilityManager import VisibilityManager
from coralnet_toolbox.MVAT.utils.MVATLogger import (
    build_camera_labels,
    get_visibility_logger,
    label_for_path,
    log_cam_stage,
)
from coralnet_toolbox.MVAT.core.Products import (
    MeshProduct,
    PointCloudProduct,
    GaussianSplattingProduct,
)

DEBUG_EXPORT_RGB_INDEX_MAPS = False

logger = get_visibility_logger()


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class VisibilityWorkerSignals(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)


class VisibilityWorker(QObject):
    """
    Background worker for computing camera visibility (index) maps.
    Uses moderngl GPU rasterization as the primary path, with VTK as fallback.
    """
    def __init__(self, primary_target, camera_params_dict, compute_depth_maps=True,
                 cache_manager=None, cache_keys_dict=None, target_file_path="", pixel_budget=None,
                 warp_callables_dict=None, dist_coeffs_bytes_dict=None, n_workers=4,
                 distortion_vram_safety_factor=0.95, enable_cache=True, enable_compression=True,
                 splat_radius=1, splat_round=False, persistent_rasterizer=None):
        super().__init__()
        self.primary_target = primary_target
        # Warm GL-context service. Owns the moderngl context on its own thread and
        # keeps geometry uploaded across runs so incremental camera adds skip the
        # full context rebuild + geometry re-upload. When None (e.g. standalone
        # use), run() spins up a private one and shuts it down on exit.
        self.persistent_rasterizer = persistent_rasterizer
        self.camera_params_dict = camera_params_dict
        self.compute_depth_maps = compute_depth_maps
        self.pixel_budget = pixel_budget
        self.upsample_to_native = True
        # Point-cloud splatting (GL_POINTS sprite size, render-resolution pixels) and
        # shape (square vs round disc). Ignored for meshes. Exposed via the visibility dialog.
        self.splat_radius = splat_radius
        self.splat_round = splat_round
        self.warp_callables_dict = warp_callables_dict or {}
        # path -> dist_coeffs.tobytes() for distorted cameras; used for cache-key disambiguation
        self.dist_coeffs_bytes_dict = dist_coeffs_bytes_dict or {}
        self.n_workers = n_workers
        self.distortion_vram_safety_factor = distortion_vram_safety_factor

        # Debug: cache settings for experimentation
        self.enable_cache = enable_cache
        self.enable_compression = enable_compression

        # Store cache dependencies
        self.cache_manager = cache_manager
        self.cache_keys_dict = cache_keys_dict
        self.target_file_path = target_file_path
        self.signals = VisibilityWorkerSignals()

    # ------------------------------------------------------------------
    def _status(self, msg: str):
        """Thread-safe status bar update (best-effort)."""
        try:
            from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
            from PyQt5.QtWidgets import QApplication
            main_win = QApplication.instance().activeWindow()
            if main_win and hasattr(main_win, 'status_bar'):
                QMetaObject.invokeMethod(
                    main_win.status_bar, "showMessage",
                    Qt.QueuedConnection,
                    Q_ARG(str, msg)
                )
        except Exception:
            pass

    @staticmethod
    def _cam_label(path: str, camera_labels=None, fallback: str = "cam") -> str:
        return label_for_path(path, camera_labels, fallback)

    def _build_warp_maps_list(self, paths):
        """Per-path distortion warp maps for ``compute_batch_visibility_moderngl``.

        Returns a list aligned with ``paths`` of ``(map_x, map_y)`` for distorted
        cameras (``Raster._map_x/_map_y``) or ``None`` otherwise, so the manager can
        fuse the warp into the render and skip the separate cv2.remap / grid_sample
        round-trip. Returns ``None`` when no path is distorted (keeps the fast,
        non-warp render path untouched).
        """
        if not self.warp_callables_dict:
            return None
        maps = []
        any_warp = False
        for p in paths:
            fn = self.warp_callables_dict.get(p)
            if fn is None:
                maps.append(None)
                continue
            try:
                raster = fn.__self__
                raster._ensure_warp_maps()
                maps.append((raster._map_x, raster._map_y))
                any_warp = True
            except Exception as e:
                logger.warning(f"Warp maps unavailable for {self._cam_label(p)}; "
                               f"rendering undistorted: {e}")
                maps.append(None)
        return maps if any_warp else None

    def _compute_visible_indices(self, results: dict, paths: list) -> None:
        """Populate ``results[p]['visible_indices']`` (sorted unique visible element IDs).

        Used for distorted cameras, whose warp is fused into the render so the manager
        leaves visible indices uncomputed (``compute_visible_indices=False``). A serial
        ``np.unique`` over native-resolution maps is ~200 ms/cam (a multi-second
        per-chunk lull on large batches), so this uses GPU ``torch.unique`` when CUDA
        is available (~17 ms/cam incl. the re-upload — the same approach the old
        grid_sample path used) and falls back to a thread-parallel ``np.unique``.

        Note: the manager's GPU coverage pass is deliberately not used here — its
        buffer scales with element count (~304 MB for a 76M-face mesh), which is
        slower than unique on the visible pixels for large meshes.
        """
        if not paths:
            return
        try:
            import torch
            use_cuda = torch.cuda.is_available()
        except Exception:
            torch = None
            use_cuda = False

        if use_cuda:
            try:
                for p in paths:
                    idx = results[p].get('index_map')
                    if idx is None:
                        continue
                    t = torch.as_tensor(idx, device='cuda')
                    results[p]['visible_indices'] = (
                        torch.unique(t[t >= 0]).to(torch.int32).cpu().numpy()
                    )
                    del t
                return
            except Exception as e:
                logger.warning(f"GPU visible-indices failed ({e}); using CPU fallback")

        def _one(p):
            idx = results[p].get('index_map')
            if idx is not None:
                results[p]['visible_indices'] = np.unique(idx[idx >= 0]).astype(np.int32)

        with ThreadPoolExecutor(max_workers=min(8, max(1, len(paths)))) as pool:
            list(pool.map(_one, paths))

    def _measure_actual_camera_footprint(self, primary_target, first_params,
                                         warp_map=None):
        """Render the first camera and measure its actual memory footprint.

        Returns tuple: (footprint_bytes, result_dict) so we can reuse the render.
        ``warp_map`` is this camera's ``(map_x, map_y)`` (or None) so its distortion
        is fused into the render exactly like the main chunk loop. This first
        render also builds (and warms) the persistent GL context for the geometry.
        """
        import sys
        import gc
        try:
            gc.collect()

            result = self.persistent_rasterizer.render(
                primary_target, [first_params],
                compute_depth_map=self.compute_depth_maps,
                # When not caching to disk (aggressive recompute mode) the dense map
                # isn't persisted, so visible_indices can't be re-derived later — we
                # must compute them now for the context matrix (cheap coverage pass).
                compute_visible_indices=(warp_map is not None) or (not self.enable_cache),
                pixel_budget=self.pixel_budget,
                # This render is reused as the first camera's real result, so it
                # must match the main loop — a sub-native map would otherwise be
                # cached and later indexed with native pixel coords.
                upsample_to_native=self.upsample_to_native,
                warp_maps_list=[warp_map] if warp_map is not None else None,
                splat_radius=self.splat_radius,
                splat_round=self.splat_round,
            )

            if result and len(result) > 0:
                res = result[0]
                index_map = res.get('index_map')
                depth_map = res.get('depth_map')

                footprint = 0
                if index_map is not None:
                    footprint += sys.getsizeof(index_map) + (index_map.nbytes if hasattr(index_map, 'nbytes') else 0)
                if depth_map is not None:
                    footprint += sys.getsizeof(depth_map) + (depth_map.nbytes if hasattr(depth_map, 'nbytes') else 0)

                logger.debug(f"📊 [ACTUAL FOOTPRINT] Measured first camera: {footprint / (1024**2):.1f} MB")
                return footprint, res

        except Exception as e:
            logger.warning(f"Could not measure actual footprint: {e}")

        return None, None

    def _calculate_dynamic_chunk_size(self, params_list, measured_footprint=None, safety_factor=0.90):
        """Calculates a safe chunk size from available RAM.

        If measured_footprint is provided, use actual data. Otherwise fall back to estimation.
        """
        try:
            import psutil
        except Exception:
            return 32

        if not params_list:
            return 32

        try:
            available_ram = psutil.virtual_memory().available
            safe_ram = available_ram * safety_factor

            if measured_footprint is not None and measured_footprint > 0:
                raw_chunk_size = int(safe_ram / measured_footprint)
                calculated_chunk = max(8, min(raw_chunk_size, 512))

                logger.debug(
                    "🧠 [RAM SCALING] Available RAM: %.1f GB | Safe Budget: %.1f GB",
                    available_ram / (1024 ** 3),
                    safe_ram / (1024 ** 3),
                )
                logger.debug(
                    "🧠 [RAM SCALING] MEASURED Camera Footprint: %.1f MB | Selected Chunk Size: %s",
                    measured_footprint / (1024 ** 2),
                    calculated_chunk,
                )
                return calculated_chunk

            # Fallback: estimate based on image dimensions
            first_params = params_list[0]
            if len(first_params) < 5:
                return 32

            _, _, _, width, height = first_params
            width = int(width)
            height = int(height)

            if width <= 0 or height <= 0:
                return 32

            if self.pixel_budget is not None and self.pixel_budget > 0:
                native_pixels = width * height
                if native_pixels > self.pixel_budget:
                    scale = float(np.sqrt(self.pixel_budget / native_pixels))
                    width = max(1, int(round(width * scale)))
                    height = max(1, int(round(height * scale)))

            # Estimate based on what we actually compute
            bytes_per_pixel = 4  # index map
            if self.compute_depth_maps:
                bytes_per_pixel += 4  # depth map
            # Small object overhead (~200 bytes total, amortized per pixel for large images)
            overhead_per_pixel = max(0.001, 200.0 / (width * height))
            bytes_per_pixel += overhead_per_pixel

            camera_footprint = width * height * bytes_per_pixel
            if camera_footprint <= 0:
                return 32

            raw_chunk_size = int(safe_ram / camera_footprint)
            calculated_chunk = max(8, min(raw_chunk_size, 512))

            logger.debug(
                "🧠 [RAM SCALING] Available RAM: %.1f GB | Safe Budget: %.1f GB",
                available_ram / (1024 ** 3),
                safe_ram / (1024 ** 3),
            )
            logger.debug(
                "🧠 [RAM SCALING] ESTIMATED Camera Footprint: %.1f MB | Selected Chunk Size: %s",
                camera_footprint / (1024 ** 2),
                calculated_chunk,
            )

            return calculated_chunk
        except Exception as e:
            logger.warning(f"Chunk size calculation failed: {e}")
            return 32

    def run(self):
        owns_rasterizer = False
        try:
            import time
            import gc
            import torch
            from collections import defaultdict
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Fall back to a private warm context if the caller didn't supply one
            # (keeps the worker self-contained for standalone/test use). When the
            # manager provides a shared rasterizer, the context survives across
            # runs; the private one is torn down in finally below.
            if self.persistent_rasterizer is None:
                from coralnet_toolbox.MVAT.managers.PersistentRasterizer import PersistentRasterizer
                self.persistent_rasterizer = PersistentRasterizer()
                owns_rasterizer = True

            # Perspective cameras
            perspective_params = {}

            for path, params in self.camera_params_dict.items():
                try:
                    first = params[0]
                except Exception:
                    perspective_params[path] = params
                    continue

                if isinstance(first, np.ndarray) and first.shape == (3, 3):
                    perspective_params[path] = params

            camera_paths = list(perspective_params.keys())
            camera_labels = build_camera_labels(camera_paths)

            # This will hold ONLY the tiny metadata payloads for all cameras
            lightweight_final_results = {}
            element_type = self.primary_target.get_element_type()
            # Set early so the final wall-time log is always safe, even when a
            # product type is unsupported or there are no perspective cameras.
            total_cameras = 0

            def _export_mesh_sort_proof():
                """Optional debug export of RGB index maps for visual proof of correct mesh sorting."""
                if not DEBUG_EXPORT_RGB_INDEX_MAPS or not isinstance(self.primary_target, MeshProduct):
                    return

                try:
                    debug_root = os.path.dirname(self.target_file_path)
                    if not debug_root:
                        debug_root = os.path.dirname(getattr(self.primary_target, 'file_path', '') or '')
                    if not debug_root:
                        debug_root = os.getcwd()

                    debug_dir = os.path.join(debug_root, "DEBUG_INDEX_MAPS")
                    proof_path = self.primary_target.export_sort_proof(debug_dir)
                    if proof_path:
                        logger.debug(f"🧪 Wrote mesh sort proof: {proof_path}")
                except Exception as exc:
                    logger.warning(f"⚠️ Mesh sort proof export failed: {exc}")

            # =================================================================
            # Helper: Synchronous Disk Saver
            # =================================================================
            # Accumulators for the end-of-run cache summary
            _cache_total_start: float = 0.0
            _cache_wall_end:    float = 0.0   # updated inside save_to_disk_task, excludes VTK cleanup
            _cache_saved_count: int = 0
            _cache_total_count: int = 0

            def save_to_disk_task(save_results, cache_mgr, target_path, keys_dict, extra_bytes_dict, pixel_budget):
                nonlocal _cache_total_start, _cache_wall_end, _cache_saved_count, _cache_total_count
                # Start the global timer on the first chunk
                if _cache_total_count == 0:
                    _cache_total_start = time.perf_counter()
                _cache_total_count += len(save_results)

                # Create path list and index mapping for progress tracking
                paths_list = list(save_results.keys())
                path_to_idx = {p: i for i, p in enumerate(paths_list)}
                total_paths = len(paths_list)
                actual_workers = min(self.n_workers, max(1, total_paths))
                total_batches = (total_paths + actual_workers - 1) // actual_workers

                def _save_one(p, result_dict):
                    cache_key = keys_dict.get(p)
                    if cache_key is None:
                        return False, 0.0
                    extra = extra_bytes_dict.get(p)
                    save_start = time.perf_counter()
                    cache_mgr.save_visibility(
                        cache_key,
                        target_path,
                        result_dict.get('index_map'),
                        result_dict.get('visible_indices'),
                        None,  # Depth maps are now handled lazily on the main thread
                        element_type=result_dict.get('element_type', 'point'),
                        inverted_index=None,
                        compressed=self.enable_compression,
                        extra_hash_data=extra,
                        pixel_budget=pixel_budget,
                    )
                    elapsed = time.perf_counter() - save_start
                    return True, elapsed

                with ThreadPoolExecutor(max_workers=actual_workers) as pool:
                    futs = {
                        pool.submit(_save_one, p, save_results[p]): p
                        for p in paths_list
                    }
                    for fut in as_completed(futs):
                        p = futs[fut]
                        try:
                            success, elapsed = fut.result()
                            if success:
                                _cache_saved_count += 1
                                path_idx = path_to_idx[p]
                                batch_num = (path_idx // actual_workers) + 1
                                # Log: cam name (count/total, +time) | batch N/total
                                log_msg = f"({_cache_saved_count}/{_cache_total_count}, +{elapsed:.3f}s) batch {batch_num}/{total_batches}"
                                log_cam_stage(self._cam_label(p, camera_labels), log_msg, 0, logger)
                                self._status(f"Caching maps to disk... ({_cache_saved_count}/{_cache_total_count})")
                        except Exception as exc:
                            logger.warning(f"⚠️ Cache save failed for {self._cam_label(p, camera_labels)}: {exc}")
                # Snapshot the end-of-saves wall time before any post-save work
                _cache_wall_end = time.perf_counter()

            # ==========================================
            # PERSPECTIVE PIPELINE (meshes + point clouds, CHUNKED)
            # ==========================================
            # Meshes (TRIANGLES) and point clouds (GL_POINTS) share the SAME chunked
            # render → distortion → cache → payload loop below; only the moderngl
            # context differs. Point clouds therefore inherit viewport cropping, the
            # distortion warp, zero-PCIe CUDA-GL readback and disk caching for free.
            mesh_processing_start = time.perf_counter()
            if isinstance(self.primary_target, (MeshProduct, PointCloudProduct, GaussianSplattingProduct)):
                _export_mesh_sort_proof()  # no-op for point clouds

                if perspective_params:
                    paths = list(perspective_params.keys())
                    params_list = list(perspective_params.values())

                    total_cameras = len(paths)
                    vtk_context = None
                    sample_w = params_list[0][3]
                    sample_h = params_list[0][4]

                    # The moderngl context (geometry upload, shader programs, FBO
                    # cache, CUDA-GL interop) is owned by the PersistentRasterizer
                    # and built lazily on its dedicated thread by the first render
                    # below, then reused across chunks and subsequent passes. The
                    # worker never touches the GL context directly — it submits
                    # render jobs and receives CPU numpy results.
                    mgl_context = None
                    logger.debug("✅ Using moderngl rasterizer (persistent warm context)")

                    # Measure actual footprint from first camera to inform chunk size
                    measured_footprint = None
                    first_camera_result = None
                    try:
                        self._status("Measuring camera footprint...")
                        _first_warp = self._build_warp_maps_list([paths[0]])
                        measured_footprint, first_camera_result = self._measure_actual_camera_footprint(
                            self.primary_target, params_list[0],
                            warp_map=(_first_warp[0] if _first_warp else None),
                        )
                    except Exception as e:
                        logger.warning(f"⚠️ Could not measure actual footprint, using estimation: {e}")

                    CHUNK_SIZE = self._calculate_dynamic_chunk_size(params_list, measured_footprint=measured_footprint)
                    logger.info(f"📦 Using CHUNK_SIZE={CHUNK_SIZE} based on {'measured' if measured_footprint else 'estimated'} footprint")

                    try:
                        # Helper: process a result dict through distortion, normalization, and optional caching
                        def _process_result_pipeline(p, res, is_measured=False):
                            """Apply distortion, normalization, and caching to a single result."""
                            element_type_val = element_type
                            res['element_type'] = element_type_val

                            # Distortion step — the warp is fused into the render
                            # (warp_maps_list), so the index map is already distorted
                            # to native resolution here; we only derive visible indices.
                            if p in self.warp_callables_dict:
                                try:
                                    idx = res.get('index_map')
                                    raster = self.warp_callables_dict[p].__self__
                                    if idx is not None and getattr(raster, '_map_x', None) is None:
                                        # Rare: warp maps were unavailable at render time, so
                                        # fusion was skipped — CPU warp, then recompute the
                                        # visible indices on the warped map.
                                        res['index_map'] = self.warp_callables_dict[p](idx, border_value=-1)
                                        res['scale_factor'] = 1.0
                                        self._compute_visible_indices({p: res}, [p])
                                    # else: fused → manager already provided visible_indices.
                                except Exception as e:
                                    logger.warning(f"Distortion finalize failed for {self._cam_label(p, camera_labels)}: {e}")
                            VisibilityManager._normalize_result_dict(res, False)

                            # Caching step (can be disabled for debugging)
                            if self.enable_cache and self.cache_manager is not None and self.target_file_path:
                                cache_key = self.cache_keys_dict.get(p)
                                if cache_key is not None:
                                    res['cache_path'] = self.cache_manager.get_cache_path(
                                        cache_key, self.target_file_path, res.get('element_type', 'point'),
                                        self.dist_coeffs_bytes_dict.get(p), pixel_budget=self.pixel_budget
                                    )
                                    try:
                                        self._status("Caching maps to disk... (1/1)")
                                        save_start = time.perf_counter()
                                        self.cache_manager.save_visibility(
                                            cache_key, self.target_file_path,
                                            res.get('index_map'),
                                            res.get('visible_indices'),
                                            None,
                                            element_type=res.get('element_type', 'point'),
                                            inverted_index=None,
                                            compressed=self.enable_compression,
                                            extra_hash_data=self.dist_coeffs_bytes_dict.get(p),
                                            pixel_budget=self.pixel_budget,
                                        )
                                        elapsed = time.perf_counter() - save_start
                                        log_cam_stage(self._cam_label(p, camera_labels), "Cache", elapsed, logger)
                                    except Exception as e:
                                        logger.warning(f"Cache save failed for {self._cam_label(p, camera_labels)}: {e}")

                            # Add to lightweight results. In aggressive mode (no disk)
                            # we also ship the dense map so the main thread can seed it
                            # into the just-computed on-screen camera — avoiding an
                            # immediate recompute on first hover. cache_path stays None
                            # so the raster falls through to the recompute provider.
                            lightweight_final_results[p] = {
                                'cache_path': res.get('cache_path'),
                                'element_type': res.get('element_type', 'point'),
                                'visible_indices': res.get('visible_indices'),
                                'index_map': res.get('index_map') if not self.enable_cache else None,
                            }

                            if is_measured:
                                logger.debug(f"✅ Processed measured first camera: {self._cam_label(p, camera_labels)}")

                        # Process the measured first camera if available
                        if first_camera_result is not None:
                            self._status("Processing measured first camera...")
                            _process_result_pipeline(paths[0], first_camera_result, is_measured=True)

                        # Process remaining cameras in chunks
                        start_idx = 1 if first_camera_result is not None else 0

                        for i in range(start_idx, total_cameras, CHUNK_SIZE):
                            chunk_paths = paths[i : i + CHUNK_SIZE]
                            chunk_params = params_list[i : i + CHUNK_SIZE]
                            chunk_results = {}

                            def update_status(current, total):
                                budget_str = "Native" if not self.pixel_budget else f"{self.pixel_budget / 1_000_000:.1f}MP"
                                self._status(f"Computing 3D maps... (Chunk {i+current}/{total_cameras} at {budget_str})")

                            # --- A. RENDER THE CHUNK (ModernGL only) ---
                            # Distortion is fused into the render. For distorted cameras
                            # we also ask the manager for visible_indices: with CUDA-GL
                            # interop these are a free torch.unique on the resident
                            # readback tensor (no separate CPU pass). Non-distorted
                            # scenes keep visible_indices off, matching prior behavior.
                            chunk_warp = self._build_warp_maps_list(chunk_paths)
                            batch_results = self.persistent_rasterizer.render(
                                self.primary_target, chunk_params,
                                compute_depth_map=self.compute_depth_maps,
                                # See _measure_actual_camera_footprint: aggressive mode
                                # needs real visible_indices since maps aren't persisted.
                                compute_visible_indices=(chunk_warp is not None) or (not self.enable_cache),
                                pixel_budget=self.pixel_budget,
                                upsample_to_native=self.upsample_to_native,
                                progress_callback=update_status,
                                camera_index_offset=i,
                                warp_maps_list=chunk_warp,
                                splat_radius=self.splat_radius,
                                splat_round=self.splat_round,
                            )
                            
                            for p, r in zip(chunk_paths, batch_results):
                                r['element_type'] = element_type
                                chunk_results[p] = r

                            # --- B. FINALIZE DISTORTED CAMERAS ---
                            # The warp is fused into the render above (warp_maps_list),
                            # so distorted cameras already carry their native-resolution
                            # warped index/depth maps. Here we only derive visible
                            # indices; the rare case where fusion was skipped (warp maps
                            # unavailable at render) falls back to a serial CPU warp.
                            distorted_paths = [p for p in chunk_results if p in self.warp_callables_dict]
                            if distorted_paths:
                                self._status(f"Finalizing distortion... (Chunk {i+len(chunk_paths)}/{total_cameras})")
                                # Fused cameras already carry the warped index map AND
                                # visible_indices (computed in the render). Only the rare
                                # un-fused case (warp maps unavailable at render time)
                                # needs a CPU warp + a visible_indices recompute, since
                                # the manager's were taken on the unwarped map.
                                unfused = []
                                for p in distorted_paths:
                                    r = chunk_results[p]
                                    idx = r.get('index_map')
                                    raster = self.warp_callables_dict[p].__self__
                                    if idx is not None and getattr(raster, '_map_x', None) is None:
                                        try:
                                            r['index_map'] = self.warp_callables_dict[p](idx, border_value=-1)
                                            r['scale_factor'] = 1.0
                                            unfused.append(p)
                                        except Exception as e:
                                            logger.warning(f"CPU warp fallback failed for {self._cam_label(p, camera_labels)}: {e}")
                                if unfused:
                                    self._compute_visible_indices(chunk_results, unfused)
                                for p in distorted_paths:
                                    VisibilityManager._normalize_result_dict(chunk_results[p], False)

                            # --- C. CACHE THE CHUNK ---
                            # Gated on enable_cache: in aggressive (recompute) mode we
                            # never touch disk during a session, so cache_path stays
                            # None and the map is served from RAM / recomputed on demand.
                            if self.enable_cache and self.cache_manager is not None and self.target_file_path:
                                self._status(f"Caching chunk to disk...")
                                for path, res in chunk_results.items():
                                    cache_key = self.cache_keys_dict.get(path)
                                    if cache_key is not None:
                                        res['cache_path'] = self.cache_manager.get_cache_path(
                                            cache_key, self.target_file_path, res.get('element_type', 'point'),
                                            self.dist_coeffs_bytes_dict.get(path), pixel_budget=self.pixel_budget
                                        )

                                # Execute synchronous save for this chunk
                                save_to_disk_task(
                                    chunk_results, self.cache_manager, self.target_file_path,
                                    self.cache_keys_dict, self.dist_coeffs_bytes_dict, self.pixel_budget
                                )

                            # --- D. SAVE LIGHTWEIGHT PAYLOAD ---
                            # In aggressive mode also ship the dense map (cache_path is
                            # None) so the main thread can seed on-screen cameras; the
                            # recompute provider regenerates them after eviction.
                            for path, res in chunk_results.items():
                                lightweight_final_results[path] = {
                                    'cache_path': res.get('cache_path'),
                                    'element_type': res.get('element_type', 'point'),
                                    'visible_indices': res.get('visible_indices'),
                                    'index_map': res.get('index_map') if not self.enable_cache else None,
                                }

                            # --- E. FLUSH SYSTEM RAM ---
                            # Wipe the heavy dictionaries to keep RAM usage flat
                            del chunk_results
                            del batch_results
                            gc.collect()

                    except Exception as err:
                        logger.warning(f"Perspective visibility processing failed: {err}")
                        raise err
                    finally:
                        if vtk_context is not None:
                            try:
                                vtk_context['plotter'].close()
                            except Exception:
                                pass
                        # The GL context is owned by the PersistentRasterizer and
                        # intentionally NOT released here — it stays warm for reuse
                        # across chunks and later visibility passes. Lifecycle (LRU
                        # eviction / shutdown) is handled by the rasterizer; the
                        # per-call transient PBOs are freed inside the manager.

                    # Log the true total cache time after ALL chunks are saved.
                    # Use _cache_wall_end (set inside save_to_disk_task) so the
                    # elapsed is pure disk-write wall time and excludes VTK cleanup.
                    if _cache_total_count > 0 and _cache_wall_end > 0:
                        total_elapsed = _cache_wall_end - _cache_total_start
                        logger.debug(
                            f"✅ Cached {_cache_saved_count}/{_cache_total_count} maps to disk in {total_elapsed:.2f}s"
                        )

            # ==========================================
            # UNSUPPORTED PRODUCTS (e.g. Gaussian splats)
            # ==========================================
            # Meshes, point clouds and Gaussian splats are all handled above.
            # Any other (future) product type that lacks an index-map path simply
            # produces no maps here.
            else:
                logger.debug(
                    f"⚠️ Index mapping not supported for product type "
                    f"{type(self.primary_target).__name__}; skipping."
                )

            # =================================================================
            # FINAL: Log wall-clock time and emit results
            # =================================================================
            mesh_processing_elapsed = time.perf_counter() - mesh_processing_start
            logger.info(
                f"⏱️  [VISIBILITY PROCESSING WALL TIME] {mesh_processing_elapsed:.2f}s "
                f"({total_cameras} cameras, {mesh_processing_elapsed/max(1, total_cameras):.3f}s per camera avg)"
            )

            # Emit ONLY the lightweight results to the main thread
            self.signals.finished.emit(lightweight_final_results)

        except Exception as e:
            self.signals.error.emit(f"{e}\n{traceback.format_exc()}")
        finally:
            # Tear down a privately-owned rasterizer (releases its warm context on
            # the owner thread). A manager-provided rasterizer is left alive so its
            # context stays warm for the next pass.
            if owns_rasterizer and self.persistent_rasterizer is not None:
                try:
                    self.persistent_rasterizer.shutdown()
                except Exception:
                    pass
            # Ensure VRAM is cleared if a failure occurred mid-loop
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass