import traceback
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal

from coralnet_toolbox.MVAT.managers.VisibilityManager import VisibilityManager
from coralnet_toolbox.MVAT.managers.visibility_logging import (
    build_camera_labels,
    get_visibility_logger,
    label_for_path,
    log_cam_stage,
)
from coralnet_toolbox.MVAT.core.Model import MeshProduct, PointCloudProduct


logger = get_visibility_logger()


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class WorkerSignals(QObject):
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
        self.signals = WorkerSignals()

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

    def run(self):
        try:
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

            results = {}
            element_type = self.primary_target.get_element_type()

            # ==========================================
            # STRATEGY A: MESH PROCESSING
            # ==========================================
            if isinstance(self.primary_target, MeshProduct):
                if perspective_params:
                    paths = list(perspective_params.keys())
                    params_list = list(perspective_params.values())

                    try:
                        # Primary: Batched VTK rasterization
                        def update_status(current, total):
                            if self.pixel_budget is None or self.pixel_budget <= 0:
                                budget_str = "Native"
                            else:
                                budget_str = f"{self.pixel_budget / 1_000_000:.1f}MP"
                            self._status(f"Computing 3D maps... ({current}/{total} cameras at {budget_str} budget)")

                        batch_results = VisibilityManager.compute_batch_mesh_visibility_vtk(
                            self.primary_target, params_list, False,
                            pixel_budget=self.pixel_budget,
                            progress_callback=update_status
                        )
                        for p, r in zip(paths, batch_results):
                            r['element_type'] = element_type
                            results[p] = r

                    except Exception as vtk_err:
                        logger.warning(f"Batch VTK rasterization failed. Trying Open3D fallback: {vtk_err}")
                        try:
                            # Fallback: Batched Open3D raycasting
                            batch_results = VisibilityManager.compute_batch_mesh_visibility_open3d(
                                self.primary_target, params_list, False
                            )
                            for p, r in zip(paths, batch_results):
                                r['element_type'] = element_type
                                results[p] = r

                        except Exception as o3d_err:
                            logger.warning(f"Open3D fallback failed. Falling back to sequential processing: {o3d_err}")
                            # Last Resort: Sequential processing per-camera
                            for path, params in perspective_params.items():
                                K, R, t, width, height = params
                                result = VisibilityManager._compute_mesh_visibility(
                                    self.primary_target, K, R, t, width, height, False
                                )
                                result['element_type'] = element_type
                                results[path] = result

            # ==========================================
            # STRATEGY B: POINT CLOUD
            # ==========================================
            else:
                points_world, element_ids = self._extract_points(self.primary_target)

                if points_world is not None and len(points_world) > 0:
                    # PERSPECTIVE CAMERAS
                    if perspective_params:
                        paths = list(perspective_params.keys())
                        params_list = list(perspective_params.values())

                        batch_results = VisibilityManager.compute_batch_visibility(
                            points_world=points_world,
                            camera_params_list=params_list,
                            point_ids=element_ids,
                            compute_depth_map=False
                        )

                        for p, r in zip(paths, batch_results):
                            r['element_type'] = element_type
                            results[p] = r

            # =================================================================
            # 0. Apply distortion warp to maps generated with K_linear
            #
            # Optimisations applied here:
            #   • Opt 4 – batch all cameras that share the same lens model into a
            #             single F.grid_sample call on CUDA.
            #   • Opt 1 – fall back to a parallel ThreadPoolExecutor of cv2.remap
            #             calls (each releases the GIL) when CUDA is unavailable.
            #   • Opt 3 – visible_indices is derived from the warped index map
            #             using a parallel np.unique pass instead of a serial loop.
            # =================================================================
            distorted_paths = [p for p in results if p in self.warp_callables_dict]

            if distorted_paths:
                self._status("Applying distortion corrections to visibility maps...")

                # --- Phase A: warp index maps -----------------------------
                cuda_ok = False
                try:
                    import torch
                    if torch.cuda.is_available():
                        from coralnet_toolbox.Rasters.QtRaster import Raster

                        # Group cameras by distortion model (same bytes → same grid)
                        groups = defaultdict(list)
                        for path in distorted_paths:
                            key = self.dist_coeffs_bytes_dict.get(path, id(self.warp_callables_dict[path]))
                            groups[key].append(path)

                        for group_paths in groups.values():
                            # Ensure warp maps cached on the representative raster
                            rep_raster = self.warp_callables_dict[group_paths[0]].__self__
                            rep_raster._ensure_warp_maps()
                            if not hasattr(rep_raster, '_torch_grid_gpu'):
                                dummy = np.zeros((1, 1), dtype=np.float32)
                                rep_raster._warp_pytorch_cuda(dummy, 0)

                            grid_gpu  = rep_raster._torch_grid_gpu
                            oob_mask  = rep_raster._torch_oob_mask

                            # --- DYNAMIC VRAM BATCHING ---
                            # 1. Check free VRAM on the GPU
                            free_vram, total_vram = torch.cuda.mem_get_info()
                            safe_vram = free_vram * 0.8  # Leave 20% headroom for safety

                            # 2. Calculate footprint of a single image using the shared warp estimate
                            test_path = group_paths[0]
                            test_map = results[test_path].get('index_map')
                            bytes_per_img = Raster._estimate_batch_warp_bytes([test_map], grid_gpu)

                            # 3. Determine an initial chunk size from the shared estimate
                            batch_size = max(1, int(safe_vram / max(1, bytes_per_img)))
                            
                            # Collect index maps
                            idx_paths = [p for p in group_paths if results[p].get('index_map') is not None]
                            if idx_paths:
                                self._status(
                                    f"Applying distortion corrections to visibility maps... ({len(idx_paths)} cameras in current group)"
                                )
                                # Chunk the list based on our dynamic batch size
                                for i in range(0, len(idx_paths), batch_size):
                                    chunk = idx_paths[i:i + batch_size]
                                    maps = [results[p]['index_map'] for p in chunk]
                                    should_split = False
                                    try:
                                        estimated_bytes = Raster._estimate_batch_warp_bytes(maps, grid_gpu)
                                        if estimated_bytes > safe_vram:
                                            should_split = True
                                    except Exception:
                                        should_split = False

                                    if should_split:
                                        if len(chunk) == 1:
                                            raise ValueError("Single-map batch still exceeds safe CUDA budget")
                                        batch_size = max(1, len(chunk) // 2)
                                        chunk = idx_paths[i:i + batch_size]
                                        maps = [results[p]['index_map'] for p in chunk]

                                    chunk_start = __import__('time').perf_counter()
                                    warped = Raster.warp_batch_cuda(maps, [-1] * len(chunk), grid_gpu, oob_mask)
                                    chunk_elapsed = __import__('time').perf_counter() - chunk_start
                                    per_cam_elapsed = chunk_elapsed / max(1, len(chunk))
                                    for p, w in zip(chunk, warped):
                                        results[p]['index_map'] = w

                                        cam_name = self._cam_label(p, camera_labels)
                                        log_cam_stage(cam_name, "Distortion", per_cam_elapsed, logger)

                        cuda_ok = True

                except Exception as e:
                    logger.warning(f"CUDA batch warp failed, falling back to parallel CPU remap: {e}")

                if not cuda_ok:
                    # Parallel CPU path — cv2.remap releases the GIL so threads help
                    def _cpu_warp(path):
                        warp_fn = self.warp_callables_dict[path]
                        r = results[path]
                        if r.get('index_map') is not None:
                            r['index_map'] = warp_fn(r['index_map'], border_value=-1)

                    n_workers = min(8, len(distorted_paths))
                    with ThreadPoolExecutor(max_workers=n_workers) as pool:
                        futs = [pool.submit(_cpu_warp, p) for p in distorted_paths]
                        for fut in as_completed(futs):
                            try:
                                fut.result()
                            except Exception as exc:
                                logger.warning(f"CPU warp failed: {exc}")

                # --- Phase B: visible_indices + normalize (parallel) ------
                # Opt 2: inverted index is NOT rebuilt here; add_index_map's
                # daemon thread handles it so we don't block the emit.
                def _post_warp(path):
                    import time

                    stage_start = time.perf_counter()
                    r = results[path]
                    idx_map = r.get('index_map')
                    if idx_map is not None:
                        valid_mask = idx_map >= 0
                        r['visible_indices'] = np.unique(idx_map[valid_mask]).astype(np.int32)
                    try:
                        VisibilityManager._normalize_result_dict(r, self.compute_depth_maps)
                    except Exception:
                        pass

                    elapsed = time.perf_counter() - stage_start
                    log_cam_stage(self._cam_label(path, camera_labels), "Normalize", elapsed, logger)

                n_workers = min(8, len(distorted_paths))
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futs = [pool.submit(_post_warp, p) for p in distorted_paths]
                    for fut in as_completed(futs):
                        try:
                            fut.result()
                        except Exception:
                            pass

            # =================================================================
            # 1. Reconstruct depth maps from the final index maps when enabled
            # =================================================================
            if self.compute_depth_maps:
                self._status("Reconstructing depth maps from visibility indices...")
                for path, result_dict in results.items():
                    if result_dict.get('depth_map') is not None:
                        continue
                    index_map = result_dict.get('index_map')
                    if index_map is None:
                        continue
                    try:
                        depth_start = __import__('time').perf_counter()
                        _, R, t, _, _ = self.camera_params_dict[path]
                        result_dict['depth_map'] = VisibilityManager.reconstruct_depth_map(
                            index_map,
                            self.primary_target,
                            R,
                            t,
                        )
                        elapsed = __import__('time').perf_counter() - depth_start
                        log_cam_stage(self._cam_label(path, camera_labels), "Depth", elapsed, logger)
                    except Exception as exc:
                        logger.warning(f"⚠️ Depth reconstruction failed for {self._cam_label(path, camera_labels)}: {exc}")

            # =================================================================
            # 2. Pre-fill cache paths so the main thread knows not to save them
            # =================================================================
            if self.cache_manager is not None and self.target_file_path and self.cache_keys_dict:
                for path, result_dict in results.items():
                    cache_key = self.cache_keys_dict.get(path)
                    if cache_key is not None:
                        element_type = result_dict.get('element_type', 'point')
                        extra = self.dist_coeffs_bytes_dict.get(path)
                        expected_cache_path = self.cache_manager.get_cache_path(
                            cache_key, self.target_file_path, element_type, extra,
                            pixel_budget=self.pixel_budget,
                        )
                        result_dict['cache_path'] = expected_cache_path

            # =================================================================
            # 3. Define the background saving task (parallel I/O)
            # =================================================================
            def save_to_disk_task(save_results, cache_mgr, target_path, keys_dict, extra_bytes_dict, pixel_budget):
                import time
                start_cache_time = time.perf_counter()
                total_to_save = len(save_results)
                saved_count = 0

                def _save_one(path, result_dict):
                    import time
                    import threading

                    cache_key = keys_dict.get(path)
                    if cache_key is None:
                        return False
                        
                    t_name = threading.current_thread().name
                    idx_map = result_dict.get('index_map')
                    idx_mb = idx_map.nbytes / (1024 * 1024) if idx_map is not None else 0
                    print(f"🚨 [DISK DEBUG] {t_name} STARTING disk write for {path} ({idx_mb:.1f} MB payload)")
                    
                    extra = extra_bytes_dict.get(path)
                    save_start = time.perf_counter()
                    cache_mgr.save_visibility(
                        cache_key,
                        target_path,
                        result_dict.get('index_map'),
                        result_dict.get('visible_indices'),
                        None,
                        element_type=result_dict.get('element_type', 'point'),
                        inverted_index=None,  # No longer storing inverted_index to save RAM
                        extra_hash_data=extra,
                        pixel_budget=pixel_budget,
                    )
                    elapsed = time.perf_counter() - save_start
                    print(f"🚨 [DISK DEBUG] {t_name} FINISHED disk write for {path}")
                    log_cam_stage(self._cam_label(path, camera_labels), "Cache", elapsed, logger)
                    return True

                n_workers = min(4, max(1, len(save_results)))
                logger.info(f"💽 Starting background cache save for {total_to_save} cameras...")
                
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futs = {
                        pool.submit(_save_one, path, result_dict): path
                        for path, result_dict in save_results.items()
                    }
                    for fut in as_completed(futs):
                        path = futs[fut]
                        try:
                            success = fut.result()
                            if success:
                                saved_count += 1
                                # Safe cross-thread UI update
                                self._status(f"Caching visibility maps to disk... ({saved_count}/{total_to_save})")
                        except Exception as exc:
                            logger.warning(f"⚠️ Cache save failed for {self._cam_label(path, camera_labels)}: {exc}")
                
                # Final cleanup messages
                elapsed = time.perf_counter() - start_cache_time
                logger.info(f"✅ Cached {saved_count}/{total_to_save} visibility maps to disk in {elapsed:.2f}s")
                self._status("Visibility maps cached successfully.")

            # =================================================================
            # 4. Fire and forget the disk writing on a separate daemon thread
            # =================================================================
            if self.cache_manager is not None and self.target_file_path and self.cache_keys_dict:
                io_thread = threading.Thread(
                    target=save_to_disk_task,
                    args=(results, self.cache_manager, self.target_file_path, self.cache_keys_dict, self.dist_coeffs_bytes_dict, self.pixel_budget),
                    daemon=True
                )
                self._status("Saving visibility maps to cache...")
                io_thread.start()

            # =================================================================
            # 5. Emit final results back to the main thread IMMEDIATELY
            # =================================================================
            self.signals.finished.emit(results)

        except Exception as e:
            self.signals.error.emit(f"{e}\n{traceback.format_exc()}")

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

                # Extract the true face centers for the solid mesh raycaster
                face_centers = mesh.cell_centers().points
                face_ids = np.arange(len(face_centers), dtype=np.int32)

                return face_centers, face_ids

            except Exception as e:
                logger.warning(f"Failed to extract faces in worker for {target.product_id}: {e}")
                return None, None

        # Fallback
        if hasattr(target, 'get_points_array'):
            return target.get_points_array(), None

        return None, None
