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
from coralnet_toolbox.MVAT.core.Products import MeshProduct, PointCloudProduct

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
                 splat_radius=1, splat_round=False):
        super().__init__()
        self.primary_target = primary_target
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

    def _measure_actual_camera_footprint(self, primary_target, first_params, mgl_context):
        """Render the first camera and measure its actual memory footprint.

        Returns tuple: (footprint_bytes, result_dict) so we can reuse the render.
        """
        import sys
        import gc
        try:
            gc.collect()

            result = VisibilityManager.compute_batch_visibility_moderngl(
                primary_target, [first_params],
                compute_depth_map=self.compute_depth_maps,
                compute_visible_indices=False,
                pixel_budget=self.pixel_budget,
                mgl_context=mgl_context,
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
        try:
            import time
            import gc
            import torch
            from collections import defaultdict
            from concurrent.futures import ThreadPoolExecutor, as_completed

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
            if isinstance(self.primary_target, (MeshProduct, PointCloudProduct)):
                _export_mesh_sort_proof()  # no-op for point clouds

                if perspective_params:
                    paths = list(perspective_params.keys())
                    params_list = list(perspective_params.values())

                    total_cameras = len(paths)
                    vtk_context = None
                    sample_w = params_list[0][3]
                    sample_h = params_list[0][4]

                    # Setup the moderngl context for this geometry type.
                    try:
                        if isinstance(self.primary_target, MeshProduct):
                            mgl_context = VisibilityManager.setup_batch_mesh_moderngl_context(
                                self.primary_target, self.pixel_budget, sample_w, sample_h,
                            )
                        else:
                            mgl_context = VisibilityManager.setup_batch_point_moderngl_context(
                                self.primary_target, self.pixel_budget, sample_w, sample_h,
                                splat_radius=self.splat_radius, splat_round=self.splat_round,
                            )
                        logger.debug("✅ Using moderngl rasterizer (zero-PCIe CUDA-GL path)")
                    except Exception as _mgl_err:
                        logger.error("❌ ModernGL unavailable (%s); cannot proceed (VTK removed in Phase 3)", _mgl_err)
                        raise

                    # Measure actual footprint from first camera to inform chunk size
                    measured_footprint = None
                    first_camera_result = None
                    try:
                        self._status("Measuring camera footprint...")
                        measured_footprint, first_camera_result = self._measure_actual_camera_footprint(
                            self.primary_target, params_list[0], mgl_context
                        )
                    except Exception as e:
                        logger.warning(f"⚠️ Could not measure actual footprint, using estimation: {e}")

                    CHUNK_SIZE = self._calculate_dynamic_chunk_size(params_list, measured_footprint=measured_footprint)
                    logger.info(f"📦 Using CHUNK_SIZE={CHUNK_SIZE} based on {'measured' if measured_footprint else 'estimated'} footprint")

                    # Cache for per-group VRAM measurements (distorted camera path)
                    # Keyed by distortion group key; values are actual_vram_per_map
                    vram_per_map_cache = {}

                    try:
                        # Helper: process a result dict through distortion, normalization, and optional caching
                        def _process_result_pipeline(p, res, is_measured=False):
                            """Apply distortion, normalization, and caching to a single result."""
                            element_type_val = element_type
                            res['element_type'] = element_type_val

                            # Distortion step
                            if p in self.warp_callables_dict:
                                try:
                                    if torch.cuda.is_available():
                                        # CUDA batch warp would need to be adapted for single result
                                        # For now, use CPU path
                                        warp_fn = self.warp_callables_dict[p]
                                        if res.get('index_map') is not None:
                                            warped = warp_fn(res['index_map'], border_value=-1)
                                            res['index_map'] = warped
                                            valid_mask = warped >= 0
                                            res['visible_indices'] = np.unique(warped[valid_mask]).astype(np.int32)
                                            VisibilityManager._normalize_result_dict(res, False)
                                except Exception as e:
                                    logger.warning(f"Distortion failed for {self._cam_label(p, camera_labels)}: {e}")
                            else:
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

                            # Add to lightweight results
                            lightweight_final_results[p] = {
                                'cache_path': res.get('cache_path'),
                                'element_type': res.get('element_type', 'point'),
                                'visible_indices': res.get('visible_indices')
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

                            # Distorted cameras (on a CUDA GPU) get their index map back
                            # as a GPU tensor so the warp consumes it without a CPU round
                            # trip. Positions are batch-relative to this chunk.
                            gpu_positions = (
                                {j for j, p in enumerate(chunk_paths) if p in self.warp_callables_dict}
                                if torch.cuda.is_available() else None
                            )

                            # --- A. RENDER THE CHUNK (ModernGL only) ---
                            batch_results = VisibilityManager.compute_batch_visibility_moderngl(
                                self.primary_target, chunk_params,
                                compute_depth_map=self.compute_depth_maps,
                                compute_visible_indices=False,
                                pixel_budget=self.pixel_budget,
                                upsample_to_native=self.upsample_to_native,
                                progress_callback=update_status,
                                mgl_context=mgl_context,
                                camera_index_offset=i,
                                gpu_index_positions=gpu_positions,
                            )
                            
                            for p, r in zip(chunk_paths, batch_results):
                                r['element_type'] = element_type
                                chunk_results[p] = r

                            # --- B. DISTORTION & NORMALIZATION FOR THE CHUNK ---
                            distorted_paths = [p for p in chunk_results if p in self.warp_callables_dict]
                            if distorted_paths:
                                self._status(f"Applying distortion corrections... (Chunk {i+len(chunk_paths)}/{total_cameras})")
                                cuda_ok = False
                                if torch.cuda.is_available():
                                    try:
                                        from coralnet_toolbox.Rasters.QtRaster import Raster

                                        # Warp input is the GPU tensor when interop produced
                                        # one, else the CPU numpy map. warp_batch_cuda accepts
                                        # either; tensors avoid the CPU→GPU re-upload.
                                        def _warp_input(p):
                                            r = chunk_results[p]
                                            g = r.get('index_map_gpu')
                                            return g if g is not None else r.get('index_map')

                                        groups = defaultdict(list)
                                        for path in distorted_paths:
                                            key = self.dist_coeffs_bytes_dict.get(path, id(self.warp_callables_dict[path]))
                                            groups[key].append(path)

                                        for group_key, group_paths in groups.items():
                                            rep_raster = self.warp_callables_dict[group_paths[0]].__self__
                                            rep_raster._ensure_warp_maps()
                                            if not hasattr(rep_raster, '_torch_grid_gpu'):
                                                dummy = np.zeros((1, 1), dtype=np.float32)
                                                rep_raster._warp_pytorch_cuda(dummy, 0)

                                            grid_gpu  = rep_raster._torch_grid_gpu
                                            oob_mask  = rep_raster._torch_oob_mask

                                            # Check if we've already measured VRAM cost for this group
                                            if group_key in vram_per_map_cache:
                                                actual_vram_per_map = vram_per_map_cache[group_key]
                                            else:
                                                # Measure per-map cost from a small test batch. Use the
                                                # peak *live* allocation (memory_allocated) rather than the
                                                # mem_get_info free-memory delta: the latter reflects the
                                                # caching allocator's reserved pool (incl. freed grid_sample
                                                # transients), which over-estimates the true marginal cost
                                                # by ~2x and left the GPU sitting at half the budget.
                                                test_paths = group_paths[:min(4, len(group_paths))]
                                                test_maps = [_warp_input(p) for p in test_paths]

                                                torch.cuda.synchronize()
                                                torch.cuda.empty_cache()
                                                torch.cuda.reset_peak_memory_stats()
                                                alloc_before = torch.cuda.memory_allocated()

                                                # Warp small batch to measure realistic per-map live cost
                                                warped_test, _ = Raster.warp_batch_cuda(
                                                    test_maps, [-1] * len(test_maps), grid_gpu, oob_mask
                                                )

                                                torch.cuda.synchronize()
                                                peak_alloc = torch.cuda.max_memory_allocated()
                                                total_vram_used = max(0, peak_alloc - alloc_before)
                                                actual_vram_per_map = total_vram_used / max(1, len(test_maps))
                                                # Cache for future chunks
                                                vram_per_map_cache[group_key] = actual_vram_per_map

                                            # Size the batch against currently-free VRAM (reserved-aware)
                                            # with the safety factor; warp_batch_cuda also re-checks its
                                            # own estimate against free VRAM as a backstop.
                                            free_vram, _ = torch.cuda.mem_get_info()
                                            safe_vram = free_vram * self.distortion_vram_safety_factor
                                            vram_batch_size = max(1, int(safe_vram / max(1, actual_vram_per_map)))

                                            logger.debug(
                                                f"🎯 [VRAM SCALING] Free VRAM: {free_vram / (1024**3):.1f} GB | "
                                                f"Safe budget (×{self.distortion_vram_safety_factor}): {safe_vram / (1024**3):.1f} GB | "
                                                f"Test batch: {len(test_maps)} maps = {total_vram_used / (1024**3):.2f} GB | "
                                                f"Per-map avg: {actual_vram_per_map / (1024**2):.1f} MB | "
                                                f"VRAM batch size: {vram_batch_size}"
                                            )
                                            
                                            idx_paths = [p for p in group_paths if _warp_input(p) is not None]

                                            for j in range(0, len(idx_paths), vram_batch_size):
                                                vram_chunk = idx_paths[j:j + vram_batch_size]
                                                maps = [_warp_input(p) for p in vram_chunk]

                                                chunk_start = time.perf_counter()
                                                # New API: returns maps AND unique visible indices directly from GPU
                                                warped_maps, visible_indices_list = Raster.warp_batch_cuda(
                                                    maps, [-1] * len(vram_chunk), grid_gpu, oob_mask
                                                )
                                                chunk_elapsed = time.perf_counter() - chunk_start
                                                per_cam_elapsed = chunk_elapsed / max(1, len(vram_chunk))
                                                
                                                for idx, p in enumerate(vram_chunk):
                                                    chunk_results[p]['index_map'] = warped_maps[idx]
                                                    chunk_results[p]['visible_indices'] = visible_indices_list[idx]
                                                    # Warp output is native numpy: release the
                                                    # GPU tensor and reset scale to native (1.0).
                                                    chunk_results[p]['index_map_gpu'] = None
                                                    chunk_results[p]['scale_factor'] = 1.0
                                                    # Enforce canonical dtypes
                                                    VisibilityManager._normalize_result_dict(chunk_results[p], False)
                                                    
                                                    cam_name = self._cam_label(p, camera_labels)
                                                    log_cam_stage(cam_name, "Distortion & Normalize", per_cam_elapsed, logger)

                                        cuda_ok = True
                                    except Exception as e:
                                        logger.warning(f"CUDA batch warp failed, falling back to CPU: {e}")

                                if not cuda_ok:
                                    # Fallback to CPU parallel remap and sort
                                    def _cpu_warp_and_sort(p):
                                        stage_start = time.perf_counter()
                                        warp_fn = self.warp_callables_dict[p]
                                        r = chunk_results[p]
                                        # If interop produced a GPU tensor, materialize it to
                                        # CPU numpy so cv2.remap can consume it.
                                        src = r.get('index_map')
                                        if src is None and r.get('index_map_gpu') is not None:
                                            src = r['index_map_gpu'].cpu().numpy()
                                            r['index_map_gpu'] = None
                                        if src is not None:
                                            warped = warp_fn(src, border_value=-1)
                                            r['index_map'] = warped
                                            r['scale_factor'] = 1.0
                                            valid_mask = warped >= 0
                                            r['visible_indices'] = np.unique(warped[valid_mask]).astype(np.int32)
                                            VisibilityManager._normalize_result_dict(r, False)
                                        elapsed = time.perf_counter() - stage_start
                                        log_cam_stage(self._cam_label(p, camera_labels), "CPU Warp/Norm", elapsed, logger)

                                    n_workers = min(8, len(distorted_paths))
                                    with ThreadPoolExecutor(max_workers=n_workers) as pool:
                                        futs = [pool.submit(_cpu_warp_and_sort, p) for p in distorted_paths]
                                        for fut in as_completed(futs):
                                            try:
                                                fut.result()
                                            except Exception:
                                                pass

                            # --- C. CACHE THE CHUNK ---
                            if self.cache_manager is not None and self.target_file_path:
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
                            for path, res in chunk_results.items():
                                lightweight_final_results[path] = {
                                    'cache_path': res.get('cache_path'),
                                    'element_type': res.get('element_type', 'point'),
                                    'visible_indices': res.get('visible_indices')
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
                        if mgl_context is not None:
                            try:
                                for fbo in mgl_context.get('_fbo_cache', {}).values():
                                    fbo.release()
                                mgl_context['ctx'].release()
                            except Exception:
                                pass

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
            # GaussianSplattingProduct reports supports_index_mapping()==False, so
            # SceneContext never makes it the primary target and it should not reach
            # this worker. If a future product type does, it simply produces no maps
            # here. A dedicated splat-ID pass may be added later.
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
            # Ensure VRAM is cleared if a failure occurred mid-loop
            if torch.cuda.is_available():
                torch.cuda.empty_cache()