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

DEBUG_EXPORT_RGB_INDEX_MAPS = True

logger = get_visibility_logger()


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class VisibilityWorkerSignals(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)


class VisibilityWorker(QObject):
    """
    Background worker for computing camera visibility maps.
    Now safely handles Meshes (via Open3D), PointClouds, and DEMs.
    """
    def __init__(self, primary_target, camera_params_dict, compute_depth_maps=True,
                 cache_manager=None, cache_keys_dict=None, target_file_path="", pixel_budget=None,
                 warp_callables_dict=None, dist_coeffs_bytes_dict=None):
        super().__init__()
        self.primary_target = primary_target
        self.camera_params_dict = camera_params_dict
        self.compute_depth_maps = compute_depth_maps
        self.pixel_budget = pixel_budget
        self.warp_callables_dict = warp_callables_dict or {}
        # path -> dist_coeffs.tobytes() for distorted cameras; used for cache-key disambiguation
        self.dist_coeffs_bytes_dict = dist_coeffs_bytes_dict or {}

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

    def _calculate_dynamic_chunk_size(self, params_list, safety_factor=0.90):
        """Calculates a safe chunk size from available RAM and camera dimensions."""
        try:
            import psutil
        except Exception:
            return 32

        if not params_list:
            return 32

        try:
            available_ram = psutil.virtual_memory().available
            safe_ram = available_ram * safety_factor

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

            bytes_per_pixel = 24
            camera_footprint = width * height * bytes_per_pixel
            if camera_footprint <= 0:
                return 32

            raw_chunk_size = int(safe_ram / camera_footprint)
            calculated_chunk = max(8, min(raw_chunk_size, 256))

            logger.info(
                "🧠 [RAM SCALING] Available RAM: %.1f GB | Safe Budget: %.1f GB",
                available_ram / (1024 ** 3),
                safe_ram / (1024 ** 3),
            )
            logger.info(
                "🧠 [RAM SCALING] Est. Camera Footprint: %.1f MB | Selected Chunk Size: %s",
                camera_footprint / (1024 ** 2),
                calculated_chunk,
            )

            return calculated_chunk
        except Exception:
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
                        logger.info(f"🧪 Wrote mesh sort proof: {proof_path}")
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

                def _save_one(p, result_dict):
                    cache_key = keys_dict.get(p)
                    if cache_key is None:
                        return False
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
                        extra_hash_data=extra,
                        pixel_budget=pixel_budget,
                    )
                    elapsed = time.perf_counter() - save_start
                    log_cam_stage(self._cam_label(p, camera_labels), "Cache", elapsed, logger)
                    return True

                n_workers = min(4, max(1, len(save_results)))
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futs = {
                        pool.submit(_save_one, p, res): p
                        for p, res in save_results.items()
                    }
                    for fut in as_completed(futs):
                        p = futs[fut]
                        try:
                            if fut.result():
                                _cache_saved_count += 1
                        except Exception as exc:
                            logger.warning(f"⚠️ Cache save failed for {self._cam_label(p, camera_labels)}: {exc}")
                # Snapshot the end-of-saves wall time before any post-save work
                _cache_wall_end = time.perf_counter()

            # ==========================================
            # STRATEGY A: MESH PROCESSING (CHUNKED)
            # ==========================================
            if isinstance(self.primary_target, MeshProduct):
                _export_mesh_sort_proof()

                if perspective_params:
                    paths = list(perspective_params.keys())
                    params_list = list(perspective_params.values())

                    CHUNK_SIZE = self._calculate_dynamic_chunk_size(params_list)
                    total_cameras = len(paths)
                    vtk_context = None
                    sample_w = params_list[0][3]
                    sample_h = params_list[0][4]

                    try:
                        vtk_context = VisibilityManager.setup_batch_vtk_context(
                            self.primary_target,
                            self.pixel_budget,
                            sample_w,
                            sample_h,
                        )

                        for i in range(0, total_cameras, CHUNK_SIZE):
                            chunk_paths = paths[i : i + CHUNK_SIZE]
                            chunk_params = params_list[i : i + CHUNK_SIZE]
                            chunk_results = {}

                            def update_status(current, total):
                                budget_str = "Native" if not self.pixel_budget else f"{self.pixel_budget / 1_000_000:.1f}MP"
                                self._status(f"Computing 3D maps... (Chunk {i+current}/{total_cameras} at {budget_str})")

                            # --- A. RENDER THE CHUNK ---
                            batch_results = VisibilityManager.compute_batch_mesh_visibility_vtk(
                                self.primary_target, chunk_params, False,
                                pixel_budget=self.pixel_budget,
                                progress_callback=update_status,
                                vtk_context=vtk_context,
                                camera_index_offset=i,
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

                                        groups = defaultdict(list)
                                        for path in distorted_paths:
                                            key = self.dist_coeffs_bytes_dict.get(path, id(self.warp_callables_dict[path]))
                                            groups[key].append(path)

                                        for group_paths in groups.values():
                                            rep_raster = self.warp_callables_dict[group_paths[0]].__self__
                                            rep_raster._ensure_warp_maps()
                                            if not hasattr(rep_raster, '_torch_grid_gpu'):
                                                dummy = np.zeros((1, 1), dtype=np.float32)
                                                rep_raster._warp_pytorch_cuda(dummy, 0)

                                            grid_gpu  = rep_raster._torch_grid_gpu
                                            oob_mask  = rep_raster._torch_oob_mask

                                            free_vram, _ = torch.cuda.mem_get_info()
                                            safe_vram = free_vram * 0.8
                                            
                                            test_map = chunk_results[group_paths[0]]['index_map']
                                            bytes_per_img = Raster._estimate_batch_warp_bytes([test_map], grid_gpu)
                                            vram_batch_size = max(1, int(safe_vram / max(1, bytes_per_img)))
                                            
                                            idx_paths = [p for p in group_paths if chunk_results[p].get('index_map') is not None]

                                            for j in range(0, len(idx_paths), vram_batch_size):
                                                vram_chunk = idx_paths[j:j + vram_batch_size]
                                                maps = [chunk_results[p]['index_map'] for p in vram_chunk]

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
                                        if r.get('index_map') is not None:
                                            warped = warp_fn(r['index_map'], border_value=-1)
                                            r['index_map'] = warped
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
                        logger.warning(f"Mesh processing failed: {err}")
                        raise err
                    finally:
                        if vtk_context is not None:
                            try:
                                vtk_context['plotter'].close()
                            except Exception:
                                pass

                    # Log the true total cache time after ALL chunks are saved.
                    # Use _cache_wall_end (set inside save_to_disk_task) so the
                    # elapsed is pure disk-write wall time and excludes VTK cleanup.
                    if _cache_total_count > 0 and _cache_wall_end > 0:
                        total_elapsed = _cache_wall_end - _cache_total_start
                        logger.info(
                            f"✅ Cached {_cache_saved_count}/{_cache_total_count} maps to disk in {total_elapsed:.2f}s"
                        )

            # ==========================================
            # STRATEGY B: POINT CLOUD (NOT YET CHUNKED)
            # ==========================================
            else:
                points_world, element_ids = self._extract_points(self.primary_target)

                if points_world is not None and len(points_world) > 0:
                    if perspective_params:
                        paths = list(perspective_params.keys())
                        params_list = list(perspective_params.values())

                        batch_results = VisibilityManager.compute_batch_visibility(
                            points_world=points_world,
                            camera_params_list=params_list,
                            point_ids=element_ids,
                            compute_depth_map=False
                        )
                        
                        # Populate lightweight results (Cache logic skipped here for brevity, 
                        # but follows the same pattern as above)
                        for p, r in zip(paths, batch_results):
                            lightweight_final_results[p] = {
                                'element_type': element_type,
                                'visible_indices': r.get('visible_indices')
                            }

            # =================================================================
            # FINAL: Emit ONLY the lightweight results to the main thread
            # =================================================================
            self.signals.finished.emit(lightweight_final_results)

        except Exception as e:
            self.signals.error.emit(f"{e}\n{traceback.format_exc()}")
        finally:
            # Ensure VRAM is cleared if a failure occurred mid-loop
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _extract_points(self, target):
        """Helper to extract point arrays for targets."""
        if isinstance(target, PointCloudProduct):
            return target.get_points_array(), None

        # Treat mesh products as solid surfaces for face extraction
        if isinstance(target, MeshProduct):
            try:
                mesh = target.get_render_mesh()

                if mesh is None:
                    logger.warning(f"Warning: Geometry not loaded for {target.product_id}")
                    return None, None

                face_centers = mesh.cell_centers().points
                face_ids = np.arange(len(face_centers), dtype=np.int32)
                return face_centers, face_ids

            except Exception as e:
                logger.warning(f"Warning: Could not extract face centers from {target.product_id}: {e}")
                return None, None

        return None, None
