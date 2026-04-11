import traceback
import threading

import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal

from coralnet_toolbox.MVAT.managers.VisibilityManager import VisibilityManager
from coralnet_toolbox.MVAT.core.Model import MeshProduct, PointCloudProduct


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
                 cache_manager=None, cache_keys_dict=None, target_file_path="", scale_factor=1.0,
                 warp_callables_dict=None, dist_coeffs_bytes_dict=None):
        super().__init__()
        self.primary_target = primary_target
        self.camera_params_dict = camera_params_dict
        self.compute_depth_maps = compute_depth_maps
        self.scale_factor = scale_factor
        self.warp_callables_dict = warp_callables_dict or {}
        # path -> dist_coeffs.tobytes() for distorted cameras; used for cache-key disambiguation
        self.dist_coeffs_bytes_dict = dist_coeffs_bytes_dict or {}
        
        # Store cache dependencies
        self.cache_manager = cache_manager
        self.cache_keys_dict = cache_keys_dict
        self.target_file_path = target_file_path
        self.signals = WorkerSignals()

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
                            try:
                                from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
                                from PyQt5.QtWidgets import QApplication
                                main_win = QApplication.instance().activeWindow()
                                if main_win and hasattr(main_win, 'status_bar'):
                                    QMetaObject.invokeMethod(
                                        main_win.status_bar, "showMessage",
                                        Qt.QueuedConnection,
                                        Q_ARG(str, f"Computing 3D maps... ({current}/{total} cameras at {self.scale_factor}x)")
                                    )
                            except Exception:
                                pass

                        batch_results = VisibilityManager.compute_batch_mesh_visibility_vtk(
                            self.primary_target, params_list, self.compute_depth_maps,
                            scale_factor=self.scale_factor,
                            progress_callback=update_status
                        )
                        for p, r in zip(paths, batch_results):
                            r['element_type'] = element_type
                            results[p] = r
                            
                    except Exception as vtk_err:
                        print(f"⚠️ Batch VTK rasterization failed: {vtk_err}. Trying Open3D fallback...")
                        try:
                            # Fallback: Batched Open3D raycasting
                            batch_results = VisibilityManager.compute_batch_mesh_visibility_open3d(
                                self.primary_target, params_list, self.compute_depth_maps
                            )
                            for p, r in zip(paths, batch_results):
                                r['element_type'] = element_type
                                results[p] = r
                                
                        except Exception as o3d_err:
                            print(f"⚠️ Open3D fallback failed: {o3d_err}. Falling back to sequential processing...")
                            # Last Resort: Sequential processing per-camera
                            for path, params in perspective_params.items():
                                K, R, t, width, height = params
                                result = VisibilityManager._compute_mesh_visibility(
                                    self.primary_target, K, R, t, width, height, self.compute_depth_maps
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
                            compute_depth_map=self.compute_depth_maps
                        )
                        
                        for p, r in zip(paths, batch_results):
                            r['element_type'] = element_type
                            results[p] = r

            # =================================================================
            # 0. Apply distortion warp to maps generated with K_linear
            # =================================================================
            for path, result in results.items():
                warp_fn = self.warp_callables_dict.get(path)
                if warp_fn is None:
                    continue
                
                idx_map = result.get('index_map')
                depth_map = result.get('depth_map')
                
                if idx_map is not None:
                    # MUST use INTER_NEAREST: interpolation would invent fake element IDs
                    result['index_map'] = warp_fn(idx_map, border_value=-1)
                    
                    # UPDATE: Re-extract visible indices because the warp may have culled edges
                    valid_mask = result['index_map'] >= 0
                    result['visible_indices'] = np.unique(result['index_map'][valid_mask]).astype(np.int32)
                    
                    # UPDATE: Rebuild the inverted index for immediate RAM usage
                    result['inverted_index'] = VisibilityManager._build_inverted_index(result['index_map'])
                    
                if depth_map is not None:
                    # UPDATE: Use np.nan so the 3D occlusion logic ignores the curved borders
                    result['depth_map'] = warp_fn(depth_map, border_value=np.nan)

                # Normalize dtypes for downstream consumers and caching
                try:
                    VisibilityManager._normalize_result_dict(result, self.compute_depth_maps)
                except Exception:
                    pass

            # Update status for distortion corrections
            try:
                from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
                from PyQt5.QtWidgets import QApplication
                main_win = QApplication.instance().activeWindow()
                if main_win and hasattr(main_win, 'status_bar'):
                    QMetaObject.invokeMethod(
                        main_win.status_bar, "showMessage",
                        Qt.QueuedConnection,
                        Q_ARG(str, "Applying distortion corrections to visibility maps...")
                    )
            except Exception:
                pass

            # =================================================================
            # 1. Pre-fill cache paths so the main thread knows not to save them
            # =================================================================
            if self.cache_manager is not None and self.target_file_path and self.cache_keys_dict:
                for path, result_dict in results.items():
                    cache_key = self.cache_keys_dict.get(path)
                    if cache_key is not None:
                        element_type = result_dict.get('element_type', 'point')
                        extra = self.dist_coeffs_bytes_dict.get(path)
                        expected_cache_path = self.cache_manager.get_cache_path(
                            cache_key, self.target_file_path, element_type, extra
                        )
                        result_dict['cache_path'] = expected_cache_path

            # =================================================================
            # 2. Define the background saving task
            # =================================================================
            def save_to_disk_task(save_results, cache_mgr, target_path, keys_dict, extra_bytes_dict):
                for path, result_dict in save_results.items():
                    cache_key = keys_dict.get(path)
                    if cache_key is not None:
                        extra = extra_bytes_dict.get(path)
                        cache_mgr.save_visibility(
                            cache_key,
                            target_path,
                            result_dict.get('index_map'),
                            result_dict.get('visible_indices'),
                            result_dict.get('depth_map') if self.compute_depth_maps else None,
                            element_type=result_dict.get('element_type', 'point'),
                            inverted_index=None,  # No longer storing inverted_index to save RAM
                            extra_hash_data=extra,
                        )

            # =================================================================
            # 3. Fire and forget the disk writing on a separate daemon thread
            # =================================================================
            if self.cache_manager is not None and self.target_file_path and self.cache_keys_dict:
                io_thread = threading.Thread(
                    target=save_to_disk_task, 
                    args=(results, self.cache_manager, self.target_file_path, self.cache_keys_dict, self.dist_coeffs_bytes_dict),
                    daemon=True
                )
                # Update status for saving
                try:
                    from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
                    from PyQt5.QtWidgets import QApplication
                    main_win = QApplication.instance().activeWindow()
                    if main_win and hasattr(main_win, 'status_bar'):
                        QMetaObject.invokeMethod(
                            main_win.status_bar, "showMessage",
                            Qt.QueuedConnection,
                            Q_ARG(str, "Saving visibility maps to cache...")
                        )
                except Exception:
                    pass
                io_thread.start()

            # =================================================================
            # 4. Emit final results back to the main thread IMMEDIATELY
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
                    print(f"⚠️ Warning: Geometry not loaded for {target.product_id}")
                    return None, None
                
                # Extract the true face centers for the solid mesh raycaster
                face_centers = mesh.cell_centers().points
                face_ids = np.arange(len(face_centers), dtype=np.int32)
                
                return face_centers, face_ids
                
            except Exception as e:
                print(f"⚠️ Failed to extract faces in worker for {target.product_id}: {e}")
                return None, None
            
        # Fallback
        if hasattr(target, 'get_points_array'):
            return target.get_points_array(), None
            
        return None, None