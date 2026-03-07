import traceback

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from coralnet_toolbox.MVAT.managers.VisibilityManager import VisibilityManager
from coralnet_toolbox.MVAT.core.Model import MeshProduct, PointCloudProduct, DEMProduct


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
                 cache_manager=None, cache_keys_dict=None, target_file_path=""):
        super().__init__()
        self.primary_target = primary_target
        self.camera_params_dict = camera_params_dict
        self.compute_depth_maps = compute_depth_maps
        
        # Store cache dependencies
        self.cache_manager = cache_manager
        self.cache_keys_dict = cache_keys_dict
        self.target_file_path = target_file_path
        self.signals = WorkerSignals()

    def run(self):
        try:
            # Separate orthographic and perspective cameras
            ortho_params = {}
            perspective_params = {}
            
            for path, params in self.camera_params_dict.items():
                try:
                    first = params[0]
                except Exception:
                    perspective_params[path] = params
                    continue

                if isinstance(first, str) and first == 'ortho':
                    _, transform_inv, width, height = params
                    ortho_params[path] = (transform_inv, width, height)
                else:
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
                        # Fast Path: Batched Open3D
                        batch_results = VisibilityManager.compute_batch_mesh_visibility_open3d(
                            self.primary_target, params_list, self.compute_depth_maps
                        )
                        for p, r in zip(paths, batch_results):
                            r['element_type'] = element_type
                            results[p] = r
                            
                    except ImportError:
                        # Slow Fallback: Sequential VTK/Point Sampling
                        print("⚠️ Open3D not found. Falling back to sequential mesh processing.")
                        for path, params in perspective_params.items():
                            K, R, t, width, height = params
                            result = VisibilityManager._compute_mesh_visibility(
                                self.primary_target, K, R, t, width, height, self.compute_depth_maps
                            )
                            result['element_type'] = element_type
                            results[path] = result

            # ==========================================
            # STRATEGY B: POINT CLOUD / DEM PROCESSING
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
                    
                    # ORTHOGRAPHIC CAMERAS
                    if ortho_params:
                        for path, (transform_inv, width, height) in ortho_params.items():
                            result = VisibilityManager.compute_orthographic_visibility(
                                points_world=points_world,
                                transform_matrix_inv=transform_inv,
                                width=width,
                                height=height,
                                point_ids=element_ids
                            )
                            result['element_type'] = element_type
                            results[path] = result

            # Save to disk on the background thread before emitting!
            if self.cache_manager is not None and self.target_file_path and self.cache_keys_dict:
                for path, result_dict in results.items():
                    cache_key = self.cache_keys_dict.get(path)
                    if cache_key is not None:
                        cache_path = self.cache_manager.save_visibility(
                            cache_key,
                            self.target_file_path,
                            result_dict.get('index_map'),
                            result_dict.get('visible_indices'),
                            result_dict.get('depth_map') if self.compute_depth_maps else None,
                            element_type=result_dict.get('element_type', 'point')
                        )
                        # Store the file path in the result so the main thread knows where it went
                        result_dict['cache_path'] = cache_path

            # Emit final results back to the main thread
            self.signals.finished.emit(results)

        except Exception as e:
            self.signals.error.emit(f"{e}\n{traceback.format_exc()}")

    def _extract_points(self, target):
        """Helper to extract point arrays for non-mesh targets."""
        if isinstance(target, PointCloudProduct):
            return target.get_points_array(), None
            
        if isinstance(target, DEMProduct):
            try:
                # 1. Grab the 3D grid directly from the wrapper
                grid = target.get_render_mesh()
                
                # Safety check in case the lazy-loader hasn't fired
                if grid is None:
                    print(f"⚠️ Warning: DEM geometry not loaded for {target.product_id}")
                    return None, None
                
                # 2. Fuse the grid into triangles
                surface = grid.extract_surface().triangulate()
                
                # 3. Extract the center points of the physical triangles
                face_centers = surface.cell_centers().points
                face_ids = np.arange(len(face_centers), dtype=np.int32)
                
                return face_centers, face_ids
                
            except Exception as e:
                print(f"⚠️ Failed to triangulate DEM points: {e}")
                return None, None
            
        # Fallback
        if hasattr(target, 'get_points_array'):
            return target.get_points_array(), None
            
        return None, None