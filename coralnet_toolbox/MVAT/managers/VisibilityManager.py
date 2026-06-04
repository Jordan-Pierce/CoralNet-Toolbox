"""
Visibility Manager for MVAT

A stateless engine that determines which 3D elements are visible to a specific camera.
It generates 'Index Maps' (Pixel -> ElementID) and sets of visible ElementIDs.

Supports heterogeneous scene products:
- Point Clouds: Scatter-reduce Z-buffering (point IDs)
- Meshes: Ray-casting/Rasterization (face IDs) [Placeholder]
- DEMs: Direct projection (cell IDs) [Placeholder]

Hardware Acceleration:
- Uses PyTorch (CUDA or CPU) as the primary compute engine.
- Falls back to NumPy if PyTorch is unavailable.

"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from coralnet_toolbox.MVAT.utils.MVATLogger import (
    cam_label,
    get_visibility_logger,
    log_cam_breakdown,
    log_cam_complete,
    log_section,
    log_summary,
)

if TYPE_CHECKING:
    from coralnet_toolbox.MVAT.core.SceneContext import SceneContext
    from coralnet_toolbox.MVAT.core.Products import AbstractSceneProduct

# Try importing torch, but handle the case where it's not installed
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


FACE_ID_RGB_BASE = 1 << 24
FACE_ID_RGB_LIMIT = FACE_ID_RGB_BASE - 1


logger = get_visibility_logger()
    
    
# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class VisibilityManager:
    """
    Stateless engine for computing element visibility and generating index maps.
    
    Strategy Pattern Implementation:
    - Point Cloud Target: Scatter-reduce Z-buffering (existing algorithm)
    - Mesh Target: Ray-casting/rasterization (placeholder - falls back to point sampling)
    
    Results include 'element_type' metadata ('point', 'face', 'cell') for downstream
    annotation engines to properly interpret index map values.
    """

    @classmethod
    def _get_2d_bounding_box(cls, bounds, K, R, t, width, height):
        """Project 3D mesh bounds to 2D pixel coordinates to find the render crop."""
        import numpy as np
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        corners_3d = np.array([
            [xmin, ymin, zmin], [xmin, ymin, zmax],
            [xmin, ymax, zmin], [xmin, ymax, zmax],
            [xmax, ymin, zmin], [xmax, ymin, zmax],
            [xmax, ymax, zmin], [xmax, ymax, zmax],
        ], dtype=np.float32)

        cam_pos = -R.T @ t
        
        # If camera is inside the box, we can't project all corners safely.
        if (xmin <= cam_pos[0] <= xmax and 
            ymin <= cam_pos[1] <= ymax and 
            zmin <= cam_pos[2] <= zmax):
            return 0, width, 0, height, "FULL_SCREEN"

        corners_cam = corners_3d @ R.T + t
        
        # If any corner is behind the camera, the perspective divide will invert them.
        # This usually means the mesh surrounds the camera or covers the whole view.
        if np.any(corners_cam[:, 2] <= 0.1):
            return 0, width, 0, height, "FULL_SCREEN"

        # Project 3D corners to 2D pixels
        u = K[0, 0] * corners_cam[:, 0] / corners_cam[:, 2] + K[0, 2]
        v = K[1, 1] * corners_cam[:, 1] / corners_cam[:, 2] + K[1, 2]

        u_min = int(np.floor(np.min(u)))
        u_max = int(np.ceil(np.max(u)))
        v_min = int(np.floor(np.min(v)))
        v_max = int(np.ceil(np.max(v)))

        # Add a 5-pixel padding buffer
        pad = 5
        u_min = max(0, u_min - pad)
        u_max = min(width, u_max + pad)
        v_min = max(0, v_min - pad)
        v_max = min(height, v_max + pad)

        # If the box is entirely outside the image bounds, it's off-screen
        if u_max <= 0 or u_min >= width or v_max <= 0 or v_min >= height:
            return 0, 0, 0, 0, "OFF_SCREEN"

        return u_min, u_max, v_min, v_max, "CROP"

    @staticmethod
    def _build_inverted_index(index_map: np.ndarray):
        """
        Build a CSR-style inverted index from a 2D index map.

        Maps each visible element ID back to the set of flat pixel indices
        (row-major: flat_idx = v * width + u) where it appears.

        Returns:
            dict with keys:
                'inv_ids'     (int32): sorted unique element IDs with >= 1 pixel
                'inv_offsets' (int64): length len(inv_ids)+1; offsets into inv_pixels
                'inv_pixels'  (int32): concatenated flat pixel indices
            or None if no valid pixels exist.
        """
        flat = index_map.ravel()
        flat_pixel_positions = np.where(flat >= 0)[0].astype(np.int32)
        if len(flat_pixel_positions) == 0:
            return None
        
        element_ids_at_pixels = flat[flat_pixel_positions].astype(np.int32)
        sort_order = np.argsort(element_ids_at_pixels, kind='stable')
        sorted_ids    = element_ids_at_pixels[sort_order]
        sorted_pixels = flat_pixel_positions[sort_order]
        unique_ids, start_positions, _ = np.unique(sorted_ids, return_index=True, return_counts=True)
        offsets = np.empty(len(unique_ids) + 1, dtype=np.int64)
        offsets[:-1] = start_positions
        offsets[-1]  = len(sorted_pixels)
        
        return {
            'inv_ids':     unique_ids.astype(np.int32),
            'inv_offsets': offsets,
            'inv_pixels':  sorted_pixels,
        }

    @staticmethod
    def _normalize_result_dict(result: dict, compute_depth_map: bool = True) -> dict:
        """
        Ensure canonical dtypes for index/depth maps and visible indices.
        - index_map -> np.int32
        - visible_indices -> np.int32
        - depth_map -> np.float16 (final storage), only if compute_depth_map True
        This mutates the dict in-place and returns it for convenience.
        """
        try:
            if result is None:
                return result
            if 'index_map' in result and result['index_map'] is not None:
                result['index_map'] = result['index_map'].astype(np.int32, copy=False)
            if 'visible_indices' in result and result['visible_indices'] is not None:
                result['visible_indices'] = np.asarray(result['visible_indices']).astype(np.int32, copy=False)
            if compute_depth_map and 'depth_map' in result and result['depth_map'] is not None:
                # Accept float32 internally but store/cache as float16 to save RAM/disk
                result['depth_map'] = result['depth_map'].astype(np.float16, copy=False)
        except Exception:
            # Be conservative on failure: leave original arrays unchanged
            pass
        return result

    @staticmethod
    def _decode_face_id_screenshot(screenshot_low: np.ndarray,
                                   screenshot_high: Optional[np.ndarray] = None):
        """Decode one or two RGB screenshots into an int32 face-ID map.

        When `screenshot_high` is provided, the red channel holds the upper 8 bits
        of the encoded face ID and is combined with the lower 24 bits from the
        first pass.
        """
        low_rgb = np.ascontiguousarray(np.asarray(screenshot_low)[..., :3])
        high_rgb = None if screenshot_high is None else np.ascontiguousarray(np.asarray(screenshot_high)[..., :3])

        if HAS_TORCH and torch.cuda.is_available():
            low_tensor = torch.from_numpy(low_rgb).cuda()
            decoded = (
                low_tensor[..., 0].to(torch.int64)
                + low_tensor[..., 1].to(torch.int64) * 256
                + low_tensor[..., 2].to(torch.int64) * 65536
            )

            if high_rgb is not None:
                high_tensor = torch.from_numpy(high_rgb).cuda()
                decoded += high_tensor[..., 0].to(torch.int64) * FACE_ID_RGB_BASE

            index_map_tensor = (decoded - 1).to(torch.int32)
            valid_mask = index_map_tensor >= 0
            visible_indices = torch.unique(index_map_tensor[valid_mask]).cpu().numpy().astype(np.int32)
            return index_map_tensor.cpu().numpy(), visible_indices, index_map_tensor

        decoded = (
            low_rgb[..., 0].astype(np.int64, copy=False)
            + low_rgb[..., 1].astype(np.int64, copy=False) * 256
            + low_rgb[..., 2].astype(np.int64, copy=False) * 65536
        )

        if high_rgb is not None:
            decoded += high_rgb[..., 0].astype(np.int64, copy=False) * FACE_ID_RGB_BASE

        index_map = (decoded - 1).astype(np.int32, copy=False)
        visible_indices = np.unique(index_map[index_map >= 0]).astype(np.int32)
        return index_map, visible_indices, None

    @classmethod
    def _create_face_id_mesh_actors(cls, plotter, mesh, n_cells: int):
        """Create the RGB face-ID mesh actor(s) used by the VTK rasterizers."""
        face_ids = np.arange(n_cells, dtype=np.int64)
        encoded_ids = face_ids + 1

        r_low = (encoded_ids % 256).astype(np.uint8)
        g_low = ((encoded_ids // 256) % 256).astype(np.uint8)
        b_low = ((encoded_ids // 65536) % 256).astype(np.uint8)
        rgb_low = np.column_stack([r_low, g_low, b_low])

        is_dual_pass = n_cells > FACE_ID_RGB_LIMIT
        if is_dual_pass:
            logger.info(f"   ⚠️ Massive mesh detected ({n_cells:,} faces). Activating Dual-Pass 32-bit rasterization.")

        mesh_low = mesh.copy()
        low_scalar_name = 'FaceID_Low' if is_dual_pass else 'FaceID_RGB'
        mesh_low.cell_data[low_scalar_name] = rgb_low

        actor_low = plotter.add_mesh(
            mesh_low,
            scalars=low_scalar_name,
            rgb=True,
            lighting=False,
            interpolate_before_map=False,
            show_edges=False,
            style='surface'
        )

        actor_high = None
        if is_dual_pass:
            r_high = ((encoded_ids // FACE_ID_RGB_BASE) % 256).astype(np.uint8)
            rgb_high = np.column_stack([r_high, np.zeros_like(r_high), np.zeros_like(r_high)])

            mesh_high = mesh.copy()
            mesh_high.cell_data['FaceID_High'] = rgb_high

            actor_high = plotter.add_mesh(
                mesh_high,
                scalars='FaceID_High',
                rgb=True,
                lighting=False,
                interpolate_before_map=False,
                show_edges=False,
                style='surface'
            )
            actor_high.SetVisibility(False)
            del mesh_high

        del mesh_low
        import gc
        gc.collect()

        return actor_low, actor_high, is_dual_pass

    @classmethod
    def reconstruct_depth_map(cls,
                              index_map: np.ndarray,
                              scene_product: 'AbstractSceneProduct',
                              R: np.ndarray,
                              t: np.ndarray) -> np.ndarray:
        """
        Reconstruct a camera-space depth map from an element index map.

        The visible element IDs in ``index_map`` are used to gather the
        corresponding 3D element coordinates from the scene product. Their
        world coordinates are then transformed into camera space using only
        ``R`` and ``t``. Intrinsics and lens distortion are not required.
        """
        if index_map is None:
            raise ValueError("index_map must not be None")
        if scene_product is None:
            raise ValueError("scene_product must not be None")
        if R is None or t is None:
            raise ValueError("R and t must not be None")

        index_map_np = np.asarray(index_map)
        if index_map_np.ndim != 2:
            raise ValueError("index_map must be a 2-D numpy array")

        valid_mask = index_map_np >= 0
        if not np.any(valid_mask):
            return np.full(index_map_np.shape, np.nan, dtype=np.float32)

        element_ids = np.unique(index_map_np[valid_mask].astype(np.int64, copy=False))
        if element_ids.size == 0:
            return np.full(index_map_np.shape, np.nan, dtype=np.float32)

        try:
            element_count = int(scene_product.get_element_count())
        except Exception:
            element_count = None

        if element_count is not None and element_count <= 0:
            return np.full(index_map_np.shape, np.nan, dtype=np.float32)

        # Prepare cached geometry where available so repeated reconstructions
        # reuse the same underlying element-coordinate arrays/tensors.
        if hasattr(scene_product, 'prepare_geometry'):
            try:
                scene_product.prepare_geometry()
            except Exception:
                pass

        # Fast GPU path using cached coordinates when available.
        if HAS_TORCH and torch.cuda.is_available():
            coords_t = getattr(scene_product, '_cached_face_centers_pt', None)
            if coords_t is None:
                coords_np = getattr(scene_product, '_element_centers_np', None)
                if coords_np is None and hasattr(scene_product, 'get_points_array'):
                    coords_np = scene_product.get_points_array()
                if coords_np is None and hasattr(scene_product, 'get_face_centers'):
                    coords_np = scene_product.get_face_centers()
                if coords_np is not None:
                    coords_t = torch.as_tensor(np.asarray(coords_np, dtype=np.float32), device='cuda')

            if coords_t is not None:
                coords_t = coords_t.to(device='cuda', dtype=torch.float32)
                coords_count = int(coords_t.shape[0])
                if int(element_ids.max()) < coords_count:
                    element_ids_t = torch.as_tensor(element_ids, dtype=torch.long, device='cuda')
                    coords_selected = coords_t.index_select(0, element_ids_t)
                    R_t = torch.as_tensor(R, dtype=torch.float32, device='cuda')
                    t_t = torch.as_tensor(t, dtype=torch.float32, device='cuda')
                    z_values = torch.matmul(coords_selected, R_t.T)[:, 2] + t_t[2]
                    lookup = torch.full((coords_count + 1,), float('nan'), dtype=torch.float32, device='cuda')
                    lookup[element_ids_t] = z_values.to(torch.float32)
                    index_map_t = torch.as_tensor(index_map_np.astype(np.int64, copy=False), dtype=torch.long, device='cuda')
                    index_map_t = index_map_t.clone()
                    index_map_t[index_map_t < 0] = coords_count
                    return lookup[index_map_t].cpu().numpy().astype(np.float32, copy=False)

        # CPU path using cached arrays when available.
        coords_np = getattr(scene_product, '_element_centers_np', None)
        if coords_np is None and hasattr(scene_product, 'get_points_array'):
            coords_np = scene_product.get_points_array()
        if coords_np is None and hasattr(scene_product, 'get_face_centers'):
            coords_np = scene_product.get_face_centers()

        if coords_np is not None:
            coords_np = np.asarray(coords_np, dtype=np.float32)
            coords_count = int(coords_np.shape[0])
            if int(element_ids.max()) < coords_count:
                coords_selected = coords_np[element_ids]
                R_np = np.asarray(R, dtype=np.float32)
                t_np = np.asarray(t, dtype=np.float32)
                z_values = (coords_selected @ R_np.T)[:, 2] + t_np[2]
                lookup = np.full((coords_count + 1,), np.nan, dtype=np.float32)
                lookup[element_ids] = z_values.astype(np.float32, copy=False)
                index_map_safe = index_map_np.astype(np.int64, copy=True)
                index_map_safe[index_map_safe < 0] = coords_count
                return lookup[index_map_safe].astype(np.float32, copy=False)

        # Slow fallback for products that only expose per-element lookup.
        coords_list = []
        filtered_ids = []
        for element_id in element_ids:
            coord = scene_product.get_element_coordinate(int(element_id))
            if coord is None:
                continue
            coords_list.append(np.asarray(coord, dtype=np.float32))
            filtered_ids.append(int(element_id))

        if not coords_list:
            return np.full(index_map_np.shape, np.nan, dtype=np.float32)

        element_ids = np.asarray(filtered_ids, dtype=np.int64)
        coords_selected = np.vstack(coords_list)
        if element_count is None:
            element_count = int(element_ids.max())

        R_np = np.asarray(R, dtype=np.float32)
        t_np = np.asarray(t, dtype=np.float32)
        z_values = (coords_selected @ R_np.T)[:, 2] + t_np[2]
        lookup = np.full((element_count + 1,), np.nan, dtype=np.float32)
        lookup[element_ids] = z_values.astype(np.float32, copy=False)
        index_map_safe = index_map_np.astype(np.int64, copy=True)
        index_map_safe[index_map_safe < 0] = element_count
        return lookup[index_map_safe].astype(np.float32, copy=False)

    @classmethod
    def reconstruct_depth_map_fast(cls,
                                   camera,
                                   scene_product: 'AbstractSceneProduct') -> np.ndarray:
        """Fast depth reconstruction using cached visible indices and CPU NumPy."""
        if camera is None or scene_product is None:
            return None

        index_map = getattr(camera._raster, 'index_map_lazy', None)
        visible_ids = getattr(camera, 'visible_indices', None)
        if index_map is None or visible_ids is None:
            return None

        visible_ids = np.asarray(visible_ids, dtype=np.int64)
        if visible_ids.size == 0:
            return None

        coords = getattr(scene_product, '_element_centers_np', None)
        if coords is None:
            if hasattr(scene_product, 'get_face_centers'):
                coords = scene_product.get_face_centers()
            elif hasattr(scene_product, 'get_points_array'):
                coords = scene_product.get_points_array()

        if coords is None:
            return None

        coords = np.asarray(coords, dtype=np.float32)
        if coords.size == 0:
            return None

        max_visible_id = int(visible_ids.max()) if visible_ids.size else -1
        if max_visible_id >= coords.shape[0]:
            visible_ids = visible_ids[visible_ids < coords.shape[0]]
            if visible_ids.size == 0:
                return None

        visible_coords = coords[visible_ids]
        R = np.asarray(camera.R, dtype=np.float32)
        t = np.asarray(camera.t, dtype=np.float32)

        # Z-axis only: depth = dot(X, R[2, :]) + t[2]
        z_values = visible_coords @ R[2, :].astype(np.float32, copy=False) + t[2]

        element_count = int(coords.shape[0])
        lookup = np.full(element_count + 1, np.nan, dtype=np.float32)
        lookup[visible_ids] = z_values.astype(np.float32, copy=False)

        index_map_safe = np.asarray(index_map, dtype=np.int64).copy()
        index_map_safe[index_map_safe < 0] = element_count
        return lookup[index_map_safe].astype(np.float16, copy=False)

    @classmethod
    def compute_visibility_from_scene(cls,
                                      scene_context: 'SceneContext',
                                      K: np.ndarray,
                                      R: np.ndarray,
                                      t: np.ndarray,
                                      width: int,
                                      height: int,
                                      compute_depth_map: bool = True) -> dict:
        """
        Strategy dispatcher: compute visibility based on scene context.
        
        Queries the scene for the primary target and dispatches to the 
        appropriate visibility algorithm based on the target's element type.
        
        Args:
            scene_context: SceneContext containing loaded products.
            K: (3, 3) Intrinsic matrix.
            R: (3, 3) Rotation matrix (World -> Camera).
            t: (3,) Translation vector (World -> Camera).
            width: Image width in pixels.
            height: Image height in pixels.
            compute_depth_map: Whether to generate depth map.
            
        Returns:
            dict: {
                'index_map': (H, W) int32 array. Pixel value is Element ID or -1.
                'visible_indices': (M,) int32 array. Unique IDs of visible elements.
                'depth_map': (H, W) float32 array. Camera-space depth per pixel (optional).
                'element_type': str. One of 'point', 'face', 'cell'.
            }
        """
        primary_target = scene_context.get_primary_target()
        
        if primary_target is None:
            return cls._normalize_result_dict({
                'index_map': np.full((height, width), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32),
                'depth_map': np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None,
                'element_type': 'point',
                'inverted_index': None,
            }, compute_depth_map)
        
        element_type = primary_target.get_element_type()
        
        # Strategy dispatch based on element type
        if element_type == 'point':
            # Strategy A: Point Cloud - existing scatter-reduce algorithm
            from coralnet_toolbox.MVAT.core.Products import PointCloudProduct
            if isinstance(primary_target, PointCloudProduct):
                points = primary_target.get_points_array()
                if points is not None:
                    result = cls.compute_visibility(points, K, R, t, width, height, 
                                                    compute_depth_map=compute_depth_map)
                    result['element_type'] = 'point'
                    return result
                    
        elif element_type == 'face':
            # Strategy B: Mesh - ray-casting/rasterization
            result = cls._compute_mesh_visibility(primary_target, K, R, t, width, height, 
                                                  compute_depth_map=compute_depth_map)
            result['element_type'] = 'face'
            return result
        
        # Fallback: empty result
        return cls._normalize_result_dict({
            'index_map': np.full((height, width), -1, dtype=np.int32),
            'visible_indices': np.array([], dtype=np.int32),
            'depth_map': np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None,
            'element_type': element_type,
            'inverted_index': None,
        }, compute_depth_map)

    @classmethod
    def _compute_mesh_visibility(cls,
                                 mesh_product: 'AbstractSceneProduct',
                                 K: np.ndarray,
                                 R: np.ndarray,
                                 t: np.ndarray,
                                 width: int,
                                 height: int,
                                 compute_depth_map: bool = True,
                                 pixel_budget: Optional[int] = None) -> dict:
        """
        Strategy B: Compute visibility for mesh products.
        Attempts VTK rasterization first, falls back to Open3D raycasting,
        and finally falls back to face-center point sampling.
        """
        try:
            # First Choice: VTK (Pixel-perfect, supports all mesh types)
            # Now passes the pixel_budget down for dynamic scaling and fast caching!
            return cls._compute_mesh_visibility_vtk(
                mesh_product, K, R, t, width, height, compute_depth_map, pixel_budget=pixel_budget
            )
        except Exception as e:
            logger.warning(f"⚠️ VTK mesh rasterization failed: {e}. Trying Open3D raycasting...")
            try:
                # Second Choice: Open3D (Thread-safe, fast, no OpenGL context required)
                import open3d
                return cls._compute_mesh_visibility_open3d(
                    mesh_product, K, R, t, width, height, compute_depth_map
                )
            except Exception as o3d_err:
                logger.warning(f"⚠️ Open3D raycasting failed: {o3d_err}. Falling back to face-center sampling")
                # Last Resort: Point Sampling
                return cls._compute_mesh_visibility_fallback(
                    mesh_product, K, R, t, width, height, compute_depth_map
                )
                
    @classmethod
    def _build_subset_bvh(cls, mesh_product, camera_params_list):
        import open3d as o3d
        import time
        import numpy as np
        import torch
        
        start_time = time.perf_counter()
        mesh_product.prepare_geometry() 
        
        # Grab tensors and the designated device from the model
        centers = mesh_product._cached_face_centers_pt
        centers_sq_norm = mesh_product._cached_centers_sq_norm_pt
        triangles = mesh_product._cached_triangles_pt
        device = mesh_product.device
        
        # Initialize the global mask directly on the target device
        global_mask = torch.zeros(len(centers), dtype=torch.bool, device=device)
        
        ANGLE_THRESHOLD = 0.6
        angle_sq_threshold = ANGLE_THRESHOLD ** 2

        for K, R, t, w, h in camera_params_list:
            cam_pos = -R.T @ t
            cam_dir = R.T @ np.array([0, 0, 1]) 
            cam_dir = cam_dir / np.linalg.norm(cam_dir)
            
            # Pack the small camera matrix and push to the device
            cam_matrix = np.column_stack((cam_dir, -2.0 * cam_pos)).astype(np.float32)
            cam_matrix_pt = torch.tensor(cam_matrix, device=device)
            
            # Massive Matrix Multiplication (GPU or CPU!)
            proj = torch.matmul(centers, cam_matrix_pt)
            
            # Calculate standard scalars in python to avoid tensor device mismatches
            cam_pos_dir_dot = float(np.dot(cam_pos, cam_dir))
            cam_pos_sq_norm = float(np.dot(cam_pos, cam_pos))
            
            dot_prods = proj[:, 0] - cam_pos_dir_dot
            sq_dists = centers_sq_norm + cam_pos_sq_norm + proj[:, 1]
            
            front_mask = dot_prods > 0
            valid_mask = front_mask & (sq_dists > 1e-6)
            camera_mask = valid_mask & ((dot_prods**2) > (sq_dists * angle_sq_threshold))
            
            global_mask |= camera_mask

        # Pull ONLY the surviving tiny subset back to the CPU for Open3D
        subset_triangles = triangles[global_mask].cpu().numpy().astype(np.uint32)
        
        subset_cell_ids = None
        if getattr(mesh_product, '_original_cell_ids_pt', None) is not None:
            subset_cell_ids = mesh_product._original_cell_ids_pt[global_mask].cpu().numpy()
        else:
            # If the mesh was already triangles, we must still map the subset back 
            # to the global face IDs so the painter thread hits the right targets!
            global_indices = torch.arange(len(global_mask), device=device)
            subset_cell_ids = global_indices[global_mask].cpu().numpy().astype(np.int32)
            
        cull_time = time.time() - start_time
        logger.info(f"✂️ {device.upper()} Frustum Cull: Kept {len(subset_triangles):,} faces in {cull_time:.3f}s")
        
        # Build Open3D Scene
        scene = o3d.t.geometry.RaycastingScene()
        
        if len(subset_triangles) > 0:
            build_start = time.time()
            v_tensor = o3d.core.Tensor(mesh_product._cached_vertices)
            t_tensor = o3d.core.Tensor(subset_triangles)
            scene.add_triangles(v_tensor, t_tensor)
            
            dummy_ray = o3d.core.Tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=o3d.core.Dtype.Float32)
            scene.cast_rays(dummy_ray)
            logger.info(f"🎯 Built Sub-BVH in {time.time() - build_start:.3f}s")
            
        return scene, subset_cell_ids, len(subset_triangles)
    
    @classmethod
    def _compute_mesh_visibility_open3d(cls,
                                        mesh_product: 'AbstractSceneProduct',
                                        K: np.ndarray,
                                        R: np.ndarray,
                                        t: np.ndarray,
                                        width: int,
                                        height: int,
                                        compute_depth_map: bool = True) -> dict:
        """
        Single-camera wrapper for Open3D mesh visibility computation.
        Delegates to the optimized batched method.
        """
        # Package the single camera parameters into the expected list format
        camera_params_list = [(K, R, t, width, height)]
        
        # Call the existing batched Open3D method
        results = cls.compute_batch_mesh_visibility_open3d(
            mesh_product, 
            camera_params_list, 
            compute_depth_maps=compute_depth_map
        )
        
        # Return the single result dictionary
        if results and len(results) > 0:
            return results[0]
            
        # Fallback empty dictionary if something goes wrong
        return {
            'index_map': np.full((height, width), -1, dtype=np.int32),
            'visible_indices': np.array([], dtype=np.int32),
            'depth_map': np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None,
            'inverted_index': None,
        }
            
    @classmethod
    def compute_batch_mesh_visibility_open3d(cls, 
                                             mesh_product, 
                                             camera_params_list, 
                                             compute_depth_maps=True,
                                             use_global_bvh=False) -> list:
        """
        Batched Open3D raycasting with dynamic options for Frustum Culling or Global BVH.
        Includes detailed timing metrics for performance comparison.
        """
        import open3d as o3d
        import time
        import cv2
        
        start_time = time.time()
        log_section(f"🚀 STARTING BATCH VISIBILITY FOR {len(camera_params_list)} CAMERAS", logger)
        
        if use_global_bvh:
            # ---------------------------------------------------------
            # STRATEGY 1: Global BVH (Build once, cast deeper tree)
            # ---------------------------------------------------------
            logger.info("🌐 STRATEGY: Global BVH")
            mesh_product.prepare_geometry()
            
            # If the persistent BVH hasn't been built yet, build it now
            if not getattr(mesh_product, '_o3d_raycasting_scene', None):
                logger.info("   -> Building Global BVH from scratch...")
                build_start = time.time()
                scene = o3d.t.geometry.RaycastingScene()
                v_tensor = o3d.core.Tensor(mesh_product._cached_vertices, dtype=o3d.core.Dtype.Float32)
                
                # Get the full list of triangles (not culled)
                triangles = mesh_product._cached_triangles_pt.cpu().numpy().astype(np.uint32)
                t_tensor = o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.UInt32)
                
                scene.add_triangles(v_tensor, t_tensor)
                mesh_product._o3d_raycasting_scene = scene
                bvh_build_time = time.time() - build_start
                logger.info(f"   ✅ Global BVH built in {bvh_build_time:.4f}s (Faces: {len(triangles):,})")
            else:
                logger.info("   ✅ Using cached Global BVH")
                bvh_build_time = 0.0
            
            scene = mesh_product._o3d_raycasting_scene
            original_cell_ids = getattr(mesh_product, '_original_cell_ids', None)
            num_faces = len(mesh_product._cached_triangles_pt)
            
        else:
            # ---------------------------------------------------------
            # STRATEGY 2: Dynamic Sub-BVH (Cull on GPU, build shallow tree)
            # ---------------------------------------------------------
            logger.info("✂️ STRATEGY: Culled Sub-BVH")
            bvh_build_start = time.time()
            scene, original_cell_ids, num_faces = cls._build_subset_bvh(mesh_product, camera_params_list)
            bvh_build_time = time.time() - bvh_build_start
            logger.info(f"   ✅ Sub-BVH prep & build completed in {bvh_build_time:.4f}s")

        results = []
        
        # If the culler removed everything (or empty mesh), return empty maps
        if num_faces == 0:
            logger.info("   ⚠️ No faces visible. Returning empty maps.")
            return [{
                'index_map': np.full((h, w), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32),
                'depth_map': np.full((h, w), np.nan, dtype=np.float32) if compute_depth_maps else None,
                'inverted_index': None,
            } for _, _, _, w, h in camera_params_list]

        # 2. Fast Downsampled Raycasting
        SCALE_FACTOR = 0.25  # 1/4 resolution raycasting
        logger.info(f"\n   -> Starting Raycasting (Scale: {SCALE_FACTOR}x)...")
        
        raycast_start_time = time.time()
        
        for K, R, t, width, height in camera_params_list:
            E = np.eye(4, dtype=np.float64)
            E[:3, :3] = R
            E[:3, 3] = t
            
            small_w = int(width * SCALE_FACTOR)
            small_h = int(height * SCALE_FACTOR)
            
            K_small = K.copy()
            K_small[0, :3] *= SCALE_FACTOR
            K_small[1, :3] *= SCALE_FACTOR

            K_tensor = o3d.core.Tensor(K_small, dtype=o3d.core.Dtype.Float64)
            E_tensor = o3d.core.Tensor(E, dtype=o3d.core.Dtype.Float64)

            rays = scene.create_rays_pinhole(
                intrinsic_matrix=K_tensor,
                extrinsic_matrix=E_tensor,
                width_px=small_w,
                height_px=small_h
            )
            ans = scene.cast_rays(rays)

            # Extract Maps
            index_map_raw = ans['primitive_ids'].numpy()
            invalid_mask = (index_map_raw == 4294967295)
            
            index_map_small = index_map_raw.astype(np.int64)
            index_map_small[invalid_mask] = -1
            index_map_small = index_map_small.astype(np.int32)

            # Re-map triangle IDs
            if original_cell_ids is not None:
                valid_mask = (index_map_small != -1)
                index_map_small[valid_mask] = original_cell_ids[index_map_small[valid_mask]]

            # Upsample
            index_map = cv2.resize(index_map_small, (width, height), interpolation=cv2.INTER_NEAREST)
            visible_indices = np.unique(index_map_small[index_map_small != -1]).astype(np.int32)

            if compute_depth_maps:
                depth_map_small = ans['t_hit'].numpy().astype(np.float32)
                depth_map_small[invalid_mask] = np.nan
                depth_map = cv2.resize(depth_map_small, (width, height), interpolation=cv2.INTER_NEAREST)
            else:
                depth_map = None

            results.append({
                'index_map': index_map,
                'visible_indices': visible_indices,
                'depth_map': depth_map,
                'inverted_index': None,
            })
            results[-1] = cls._normalize_result_dict(results[-1], compute_depth_maps)
            # Normalize dtypes for consistency and memory savings
            results[-1] = cls._normalize_result_dict(results[-1], compute_depth_maps)
            
        raycast_time = time.time() - raycast_start_time
        total_time = time.time() - start_time
        
        logger.info(f"   ✅ Raycasting finished in {raycast_time:.4f}s (Avg: {raycast_time/len(camera_params_list):.4f}s per camera)")
        log_summary(
            f"{'Global BVH' if use_global_bvh else 'Sub-BVH'}",
            [
                f"   - BVH Build/Prep : {bvh_build_time:.4f}s",
                f"   - Raycast Loop   : {raycast_time:.4f}s",
                f"   - Total Time     : {total_time:.4f}s",
            ],
            logger,
        )
        
        return results

    @classmethod
    def _compute_mesh_visibility_vtk(cls,
                                     mesh_product: 'AbstractSceneProduct',
                                     K: np.ndarray,
                                     R: np.ndarray,
                                     t: np.ndarray,
                                     width: int,
                                     height: int,
                                     compute_depth_map: bool = True,
                                     pixel_budget: Optional[int] = None) -> dict:
        """
        VTK-based mesh rasterization for pixel-perfect face ID and depth maps.
        Now dynamically scales resolution and offloads decode/depth to PyTorch.
        """
        import pyvista as pv
        import time

        start_time = time.perf_counter()
        log_section("🎨 VTK MESH VISIBILITY RASTERIZATION", logger)

        mesh = mesh_product.get_mesh()
        n_cells = mesh.n_cells

        # --- DYNAMIC SCALING ---
        native_pixels = width * height
        if pixel_budget is None or native_pixels <= pixel_budget:
            dynamic_scale = 1.0
        else:
            dynamic_scale = float(np.sqrt(pixel_budget / native_pixels))

        render_w = max(1, int(width * dynamic_scale))
        render_h = max(1, int(height * dynamic_scale))

        K_scaled = K.copy()
        K_scaled[0, :3] *= dynamic_scale
        K_scaled[1, :3] *= dynamic_scale

        if n_cells == 0:
            logger.info("   ⚠️ No cells in mesh. Returning empty maps.")
            return cls._normalize_result_dict({
                'index_map': np.full((render_h, render_w), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32),
                'depth_map': np.full((render_h, render_w), np.nan, dtype=np.float32) if compute_depth_map else None,
                'inverted_index': None,
                'scale_factor': dynamic_scale
            }, compute_depth_map)

        logger.info(f"   Mesh: {n_cells:,} cells | Render: {render_w}x{render_h} pixels (Scale: {dynamic_scale:.4f})")

        # --- 1. Setup face-ID rendering actors ---
        setup_start = time.perf_counter()
        plotter = pv.Plotter(off_screen=True, window_size=(render_w, render_h))
        plotter.set_background('black')
        plotter.disable_anti_aliasing()
        actor_low, actor_high, is_dual_pass = cls._create_face_id_mesh_actors(plotter, mesh, n_cells)
        encode_time = time.perf_counter() - setup_start
        plotter_time = 0.0

        # --- 3. Camera / scene prep ---
        camera_start = time.perf_counter()

        # Prepare face centers for GPU depth mapping
        if compute_depth_map and HAS_TORCH and torch.cuda.is_available():
            face_centers_np = mesh.cell_centers().points.astype(np.float32)
            face_centers_pt = torch.from_numpy(face_centers_np).cuda()

        # Configure the VTK camera
        cls._configure_vtk_camera(plotter, K_scaled, R, t, render_w, render_h, mesh.bounds)
        camera_time = time.perf_counter() - camera_start

        # --- 4. Render and extract face IDs ---
        render_start = time.perf_counter()
        if is_dual_pass and actor_high is not None:
            actor_low.SetVisibility(True)
            actor_high.SetVisibility(False)
        plotter.render()
        screenshot_low = plotter.screenshot(return_img=True)
        if screenshot_low.shape[2] == 4:
            screenshot_low = screenshot_low[:, :, :3]

        screenshot_high = None
        if is_dual_pass and actor_high is not None:
            actor_low.SetVisibility(False)
            actor_high.SetVisibility(True)
            plotter.render()
            screenshot_high = plotter.screenshot(return_img=True)
            if screenshot_high.shape[2] == 4:
                screenshot_high = screenshot_high[:, :, :3]

        index_map, visible_indices, index_map_tensor = cls._decode_face_id_screenshot(
            screenshot_low,
            screenshot_high,
        )

        render_time = time.perf_counter() - render_start

        # --- 5. Extract depth buffer ---
        depth_start = time.perf_counter()
        depth_map = None
        if compute_depth_map:
            if HAS_TORCH and torch.cuda.is_available():
                R_pt = torch.from_numpy(R).to(torch.float32).cuda()
                t_pt = torch.from_numpy(t).to(torch.float32).cuda()
                
                # Z-depth in camera space
                z_cam = torch.matmul(face_centers_pt, R_pt.T)[:, 2] + t_pt[2]
                z_cam_padded = torch.cat([z_cam, torch.tensor([float('nan')], device='cuda')])
                
                # Map Face IDs directly to their computed depths
                depth_tensor = z_cam_padded[index_map_tensor.long()]
                # Keep small, NO resizing!
                depth_map = depth_tensor.cpu().numpy()
            else:
                try:
                    vtk_depth = plotter.get_image_depth(fill_value=np.nan)
                    # Keep small, NO resizing!
                    depth_map = -vtk_depth.astype(np.float32)
                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to extract depth: {e}")
                    depth_map = np.full((render_h, render_w), np.nan, dtype=np.float32)

        depth_time = time.perf_counter() - depth_start
        plotter.close()

        total_time = time.perf_counter() - start_time
        single_summary = [
            f"   - Setup          : {encode_time + plotter_time:.4f}s",
            f"   - Camera Setup   : {camera_time:.4f}s",
            f"   - Render & Decode: {render_time:.4f}s",
        ]
        if compute_depth_map:
            single_summary.append(f"   - Depth Tricks   : {depth_time:.4f}s")
        single_summary.append(f"   - Total Time     : {total_time:.4f}s")
        log_summary("Single VTK Rasterization", single_summary, logger)

        # Crucial: Include the dynamic_scale so caching knows the resolution mapping
        return cls._normalize_result_dict({
            'index_map': index_map,
            'visible_indices': visible_indices,
            'depth_map': depth_map,
            'inverted_index': None,
            'scale_factor': dynamic_scale
        }, compute_depth_map)
    
    @classmethod
    def setup_batch_vtk_context(cls, mesh_product, pixel_budget, sample_width, sample_height):
        """Pre-load the VTK plotter and upload mesh geometry once for all chunks."""
        import pyvista as pv
        import time

        start_time = time.perf_counter()
        logger.info("   -> Setting up Persistent VTK Plotter Context...")

        mesh = mesh_product.get_mesh()
        n_cells = mesh.n_cells

        native_pixels = sample_width * sample_height
        if pixel_budget is None or pixel_budget <= 0 or native_pixels <= pixel_budget:
            scale = 1.0
        else:
            scale = float(np.sqrt(pixel_budget / native_pixels))

        render_w = max(1, int(round(sample_width * scale)))
        render_h = max(1, int(round(sample_height * scale)))

        plotter = pv.Plotter(off_screen=True, window_size=(render_w, render_h))
        plotter.set_background('black')
        plotter.disable_anti_aliasing()

        actor_low, actor_high, is_dual_pass = cls._create_face_id_mesh_actors(plotter, mesh, n_cells)

        face_centers_pt = None
        if HAS_TORCH and torch.cuda.is_available():
            face_centers_np = mesh.cell_centers().points.astype(np.float32)
            face_centers_pt = torch.from_numpy(face_centers_np).cuda()

        # Force the geometry upload/VBO build once up front so later cameras reuse VRAM data.
        plotter.render()

        logger.info(f"   ✅ Persistent Plotter setup in {time.perf_counter() - start_time:.4f}s")

        return {
            'plotter': plotter,
            'actor_low': actor_low,
            'actor_high': actor_high,
            'is_dual_pass': is_dual_pass,
            'face_centers_pt': face_centers_pt,
            'mesh_bounds': mesh.bounds,
        }

    @classmethod
    def compute_batch_mesh_visibility_vtk(cls,
                                          mesh_product: 'AbstractSceneProduct',
                                          camera_params_list: list,
                                          compute_depth_map: bool = True,
                                          pixel_budget: Optional[int] = None,
                                          progress_callback=None,
                                          vtk_context: dict = None,
                                          camera_index_offset: int = 0,
                                          use_viewport_cropping: bool = True) -> list:
        """
        Batched VTK-based mesh rasterization with Viewport Cropping.
        Dynamically calculates the 2D bounding box of the mesh to avoid rendering
        and transferring empty sky pixels from the GPU.
        """
        import time
        import pyvista as pv
        import torch
        import numpy as np

        perf_counter = time.perf_counter

        def _scale_for_dimensions(width: int, height: int) -> float:
            native_pixels = width * height
            if pixel_budget is None or pixel_budget <= 0 or native_pixels <= pixel_budget:
                return 1.0
            return float(np.sqrt(pixel_budget / native_pixels))
        
        start_time = perf_counter()
        budget_str = "Native" if pixel_budget is None or pixel_budget <= 0 else f"{pixel_budget / 1_000_000:.1f}MP"
        log_section(f"🎨 STARTING BATCH VTK RASTERIZATION FOR {len(camera_params_list)} CAMERAS (Budget: {budget_str})", logger)
        
        mesh = mesh_product.get_mesh()
        n_cells = mesh.n_cells
        mesh_bounds = mesh.bounds
        
        if n_cells == 0:
            logger.info("   ⚠️ No cells in mesh. Returning empty maps.")
            return [{
                'index_map': np.full((h, w), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32),
                'depth_map': np.full((h, w), np.nan, dtype=np.float32) if compute_depth_map else None,
                'inverted_index': None,
                'scale_factor': _scale_for_dimensions(w, h),
            } for _, _, _, w, h in camera_params_list]
            
        logger.info(f"   Mesh: {n_cells:,} cells")
        
        # --- 1. SETUP PHASE ---
        if vtk_context is None:
            setup_start = perf_counter()
            logger.info("   -> Encoding face IDs as RGB colors...")
            logger.info("   -> Creating off-screen plotter...")

            first_w = camera_params_list[0][3]
            first_h = camera_params_list[0][4]
            first_scale = _scale_for_dimensions(first_w, first_h)
            first_w = max(1, int(round(first_w * first_scale)))
            first_h = max(1, int(round(first_h * first_scale)))
            plotter = pv.Plotter(off_screen=True, window_size=(first_w, first_h))
            plotter.set_background('black')
            plotter.disable_anti_aliasing()

            actor_low, actor_high, is_dual_pass = cls._create_face_id_mesh_actors(plotter, mesh, n_cells)
            encode_time = perf_counter() - setup_start
            plotter_setup_time = 0.0
            logger.info(f"   ✅ Setup completed in {encode_time + plotter_setup_time:.4f}s")

            logger.info("   -> Preparing face centers for GPU depth mapping...")
            face_center_prep_time = 0.0
            if HAS_TORCH and torch.cuda.is_available():
                face_center_prep_start = perf_counter()
                face_centers_np = mesh.cell_centers().points.astype(np.float32)
                face_centers_pt = torch.from_numpy(face_centers_np).cuda()
                face_center_prep_time = perf_counter() - face_center_prep_start
            logger.info(f"      Face center prep completed in {face_center_prep_time:.4f}s")
        else:
            plotter = vtk_context['plotter']
            actor_low = vtk_context['actor_low']
            actor_high = vtk_context['actor_high']
            is_dual_pass = vtk_context['is_dual_pass']
            face_centers_pt = vtk_context['face_centers_pt']
            mesh = mesh_product.get_mesh()
            encode_time = 0.0
            plotter_setup_time = 0.0
            face_center_prep_time = 0.0
            logger.info("   -> Reusing persistent VTK plotter context...")

        # --- 2. RENDER LOOP ---
        logger.info(f"\n   -> Rendering {len(camera_params_list)} cameras (Budget: {budget_str})...")
        render_start_time = perf_counter()
        results = []
        camera_total_sum = 0.0
        _t_render_total = 0.0      # GPU rasterization only
        _t_screenshot_total = 0.0  # GPU→CPU transfer only
        _t_decode_total = 0.0      # CPU→GPU decode only

        for i, (K, R, t, width, height) in enumerate(camera_params_list):
            cam_start = perf_counter()
            prep_start = perf_counter()
            t_crop = 0.0

            if progress_callback is not None:
                progress_callback(camera_index_offset + i + 1, camera_index_offset + len(camera_params_list))

            dynamic_scale = _scale_for_dimensions(width, height)
            render_w = max(1, int(round(width * dynamic_scale)))
            render_h = max(1, int(round(height * dynamic_scale)))

            # -----------------------------------------------------------
            # VIEWPORT CROPPING LOGIC
            # -----------------------------------------------------------
            K_scaled = K.copy()
            K_scaled[0, :3] *= dynamic_scale
            K_scaled[1, :3] *= dynamic_scale

            if use_viewport_cropping:
                t0_crop = perf_counter()
                u_min, u_max, v_min, v_max, crop_status = cls._get_2d_bounding_box(mesh_bounds, K_scaled, R, t, render_w, render_h)
                t_crop = perf_counter() - t0_crop
                
                if crop_status == "OFF_SCREEN":
                    # The mesh is completely off-screen. Skip rendering entirely!
                    results.append({
                        'index_map': np.full((render_h, render_w), -1, dtype=np.int32),
                        'visible_indices': np.array([], dtype=np.int32),
                        'depth_map': np.full((render_h, render_w), np.nan, dtype=np.float32) if compute_depth_map else None,
                        'inverted_index': None,
                        'scale_factor': dynamic_scale,
                    })
                    results[-1] = cls._normalize_result_dict(results[-1], compute_depth_map)
                    log_cam_breakdown(cam_label(camera_index_offset + i + 1), perf_counter() - cam_start, t_crop, 0, 0, 0, 0, 0, 0, logger)
                    continue
                elif crop_status == "FULL_SCREEN":
                    crop_w, crop_h = render_w, render_h
                    u_min, v_min = 0, 0
                else: # "CROP"
                    crop_w = u_max - u_min
                    crop_h = v_max - v_min
                    K_scaled[0, 2] -= u_min
                    K_scaled[1, 2] -= v_min
            else:
                crop_w, crop_h = render_w, render_h
                u_min, v_min = 0, 0

            # Resize VTK window to the crop size
            current_size = plotter.window_size
            if current_size[0] != crop_w or current_size[1] != crop_h:
                plotter.window_size = (crop_w, crop_h)

            # Config & Render
            t0 = perf_counter()
            cls._configure_vtk_camera(plotter, K_scaled, R, t, crop_w, crop_h, mesh_bounds)
            if is_dual_pass and actor_high is not None:
                actor_low.SetVisibility(True)
                actor_high.SetVisibility(False)
            plotter.render()
            t_render = perf_counter() - t0
            t_prep = (t0 - prep_start) - t_crop

            # Screenshot transfer (Tiny footprint if cropped)
            t0 = perf_counter()
            screenshot_low = plotter.screenshot(return_img=True)
            if screenshot_low.shape[2] == 4:
                screenshot_low = screenshot_low[:, :, :3]

            screenshot_high = None
            if is_dual_pass and actor_high is not None:
                actor_low.SetVisibility(False)
                actor_high.SetVisibility(True)
                plotter.render()
                screenshot_high = plotter.screenshot(return_img=True)
                if screenshot_high.shape[2] == 4:
                    screenshot_high = screenshot_high[:, :, :3]
            t_screenshot = perf_counter() - t0

            # Decoding (GPU Math)
            t0 = perf_counter()
            crop_index_map, visible_indices, crop_index_tensor = cls._decode_face_id_screenshot(
                screenshot_low,
                screenshot_high,
            )
            t_decode = perf_counter() - t0
            
            # Depth Extraction
            t0 = perf_counter()
            crop_depth_map = None
            if compute_depth_map:
                if HAS_TORCH and torch.cuda.is_available():
                    R_pt = torch.from_numpy(R).to(torch.float32).cuda()
                    t_pt = torch.from_numpy(t).to(torch.float32).cuda()
                    
                    z_cam = torch.matmul(face_centers_pt, R_pt.T)[:, 2] + t_pt[2]
                    z_cam_padded = torch.cat([z_cam, torch.tensor([float('nan')], device='cuda')])
                    
                    small_depth_tensor = z_cam_padded[crop_index_tensor.long()]
                    crop_depth_map = small_depth_tensor.cpu().numpy()
                else:
                    try:
                        vtk_depth = plotter.get_image_depth(fill_value=np.nan)
                        crop_depth_map = -vtk_depth.astype(np.float32)
                    except Exception as e:
                        crop_depth_map = np.full((crop_h, crop_w), np.nan, dtype=np.float32)
            t_depth = perf_counter() - t0

            # Finalize: Paste the crop back into the full-resolution canvas
            t0 = perf_counter()
            
            full_index_map = np.full((render_h, render_w), -1, dtype=np.int32)
            full_index_map[v_min:v_min + crop_h, u_min:u_min + crop_w] = crop_index_map
            
            if compute_depth_map:
                full_depth_map = np.full((render_h, render_w), np.nan, dtype=np.float32)
                full_depth_map[v_min:v_min + crop_h, u_min:u_min + crop_w] = crop_depth_map
            else:
                full_depth_map = None

            results.append({
                'index_map': full_index_map,
                'visible_indices': visible_indices,
                'depth_map': full_depth_map,
                'inverted_index': None,
                'scale_factor': dynamic_scale,
            })
            results[-1] = cls._normalize_result_dict(results[-1], compute_depth_map)
            
            try:
                from PyQt5.QtWidgets import QApplication
                app = QApplication.instance()
                if app:
                    app.processEvents()
            except ImportError:
                pass
            t_finalize = perf_counter() - t0
            
            cam_time = perf_counter() - cam_start
            camera_total_sum += cam_time
            _t_render_total     += t_render
            _t_screenshot_total += t_screenshot
            _t_decode_total     += t_decode
            accounted_time = t_prep + t_crop + t_render + t_screenshot + t_decode + t_depth + t_finalize
            residual_time = max(0.0, cam_time - accounted_time)
            
            log_cam_breakdown(
                cam_label(camera_index_offset + i + 1),
                cam_time,
                t_prep + t_crop,
                t_render,
                t_screenshot,
                t_decode,
                t_depth,
                t_finalize,
                residual_time,
                logger,
            )

        plotter_close_time = 0.0
        if vtk_context is None:
            close_start = perf_counter()
            plotter.close()
            plotter_close_time = perf_counter() - close_start
        
        total_render_time = perf_counter() - render_start_time
        total_time = perf_counter() - start_time
        loop_residual = max(0.0, total_render_time - camera_total_sum - plotter_close_time)

        n_cams = max(1, len(camera_params_list))
        print(
            f"\n⏱️  PCIe TRANSFER BREAKDOWN ({n_cams} cameras)\n"
            f"   GPU rasterize  : {_t_render_total:.3f}s total  ({_t_render_total/n_cams*1000:.1f}ms/cam)\n"
            f"   GPU→CPU snap   : {_t_screenshot_total:.3f}s total  ({_t_screenshot_total/n_cams*1000:.1f}ms/cam)  ← PCIe readback\n"
            f"   CPU→GPU decode : {_t_decode_total:.3f}s total  ({_t_decode_total/n_cams*1000:.1f}ms/cam)  ← upload + GPU math\n"
            f"   Round-trip overhead: {(_t_screenshot_total + _t_decode_total):.3f}s  "
            f"({(_t_screenshot_total + _t_decode_total) / max(0.001, _t_render_total + _t_screenshot_total + _t_decode_total) * 100:.1f}% of render+transfer+decode)\n"
        )

        log_summary(
            "Batch VTK Rasterization (Viewport Cropping)",
            [
                f"   - Setup Time     : {encode_time + plotter_setup_time:.4f}s",
                f"   - Face Center Prep: {face_center_prep_time:.4f}s",
                f"   - Camera Time Sum: {camera_total_sum:.4f}s",
                f"   - Plotter Close   : {plotter_close_time:.4f}s",
                f"   - Loop Residual   : {loop_residual:.4f}s",
                f"   - Render Loop    : {total_render_time:.4f}s",
                f"   - Total Time     : {total_time:.4f}s (Avg: {total_time/len(camera_params_list):.4f}s per camera)",
            ],
            logger,
        )
        
        return results

    @classmethod
    def _configure_vtk_camera(cls, plotter, K: np.ndarray, R: np.ndarray, t: np.ndarray,
                              width: int, height: int, bounds: tuple) -> None:
        
        position = -R.T @ t
        forward_cam = np.array([0.0, 0.0, 1.0])
        forward_world = R.T @ forward_cam
        focal_point = position + forward_world
        
        up_cam = np.array([0.0, -1.0, 0.0])  
        view_up = R.T @ up_cam
        
        camera = plotter.camera
        camera.position = position.tolist()
        camera.focal_point = focal_point.tolist()
        camera.up = view_up.tolist()
        
        # --- Set clipping range based on scene bounds ---
        scene_center = np.array([
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        ])
        scene_radius = np.linalg.norm([
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4]
        ]) / 2
        
        dist_to_center = np.linalg.norm(scene_center - position)
        near_clip = max(0.01, dist_to_center - scene_radius * 2)
        far_clip = dist_to_center + scene_radius * 2
        camera.clipping_range = (near_clip, far_clip)
        
        # ---> EXPLICIT PROJECTION MATRIX <---
        # Bypasses VTK's aspect ratio assumptions by explicitly mapping
        # the true horizontal (fx) and vertical (fy) OpenCV focal lengths.
        
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        import vtk
        mat = vtk.vtkMatrix4x4()
        mat.Zero()
        
        # Map X
        mat.SetElement(0, 0, 2.0 * fx / width)
        mat.SetElement(0, 2, (width - 2.0 * cx) / width)
        
        # Map Y (Y is flipped between OpenCV and OpenGL)
        mat.SetElement(1, 1, 2.0 * fy / height)
        mat.SetElement(1, 2, (2.0 * cy - height) / height)
        
        # Map Z (Clipping Range)
        mat.SetElement(2, 2, -(far_clip + near_clip) / (far_clip - near_clip))
        mat.SetElement(2, 3, -2.0 * far_clip * near_clip / (far_clip - near_clip))
        
        # W (Perspective divide)
        mat.SetElement(3, 2, -1.0)
        
        # Force VTK to use our raw matrix
        camera.SetUseExplicitProjectionTransformMatrix(True)
        camera.SetExplicitProjectionTransformMatrix(mat)

    @classmethod
    def compute_ortho_index_map_vtk(cls,
                                    ortho_camera,
                                    mesh_product: 'AbstractSceneProduct',
                                    pixel_budget: Optional[int] = None) -> dict:
        """Build a downsampled face-ID index map for an OrthoCamera.

        Renders the mesh off-screen with VTK orthographic projection looking
        straight down over the ortho's world-space extent.  The result is stored
        at a resolution derived from the shared MVAT pixel budget, where values
        at or above the native resolution keep the full image and lower budgets
        trade fidelity for speed.

        Args:
            ortho_camera:    OrthoCamera instance (must be is_valid).
            mesh_product:    MeshProduct whose faces are being indexed.
            scale_factor:    Quality scale factor in the range (0, 1].

        Returns:
            dict with keys:
                'index_map'      – (H_r, W_r) int32 face-ID map (-1 = no face)
                'visible_indices'– 1-D int32 array of unique visible face IDs
                'scale_factor'   – float, render_h / ortho_height
        """
        import pyvista as pv
        import time

        perf_counter = time.perf_counter
        start = perf_counter()
        log_section("🗺️  VTK ORTHO INDEX MAP RASTERIZATION", logger)

        if not ortho_camera.is_valid:
            logger.info("   ⚠️ OrthoCamera has no valid geo metadata — aborting.")
            return {'index_map': None, 'visible_indices': np.array([], dtype=np.int32), 'scale_factor': 1.0}

        mesh = mesh_product.get_mesh()
        if mesh is None or mesh.n_cells == 0:
            logger.info("   ⚠️ Mesh has no cells — aborting.")
            return {'index_map': None, 'visible_indices': np.array([], dtype=np.int32), 'scale_factor': 1.0}

        n_cells = mesh.n_cells
        ortho_w, ortho_h = ortho_camera.width, ortho_camera.height

        native_pixels = ortho_w * ortho_h
        if pixel_budget is None or pixel_budget <= 0 or native_pixels <= pixel_budget:
            scale = 1.0
        else:
            scale = float(np.sqrt(pixel_budget / native_pixels))
        render_w = max(1, int(round(ortho_w * scale)))
        render_h = max(1, int(round(ortho_h * scale)))
        logger.info(f"   Ortho: {ortho_w}×{ortho_h}  →  render: {render_w}×{render_h}  (scale={scale:.4f})")
        logger.info(f"   Mesh: {n_cells:,} cells")

        # --- 1. Setup face-ID rendering actors ---
        plotter = pv.Plotter(off_screen=True, window_size=(render_w, render_h))
        plotter.set_background('black')
        plotter.disable_anti_aliasing()
        actor_low, actor_high, is_dual_pass = cls._create_face_id_mesh_actors(plotter, mesh, n_cells)

        # --- 3. Configure orthographic camera ---
        cls._configure_vtk_camera_ortho(plotter, ortho_camera, mesh.bounds)

        # --- 4. Render and decode ---
        if is_dual_pass and actor_high is not None:
            actor_low.SetVisibility(True)
            actor_high.SetVisibility(False)
        plotter.render()
        screenshot_low = plotter.screenshot(return_img=True)
        if screenshot_low.shape[2] == 4:
            screenshot_low = screenshot_low[:, :, :3]

        screenshot_high = None
        if is_dual_pass and actor_high is not None:
            actor_low.SetVisibility(False)
            actor_high.SetVisibility(True)
            plotter.render()
            screenshot_high = plotter.screenshot(return_img=True)
            if screenshot_high.shape[2] == 4:
                screenshot_high = screenshot_high[:, :, :3]

        index_map, visible_indices, _ = cls._decode_face_id_screenshot(
            screenshot_low,
            screenshot_high,
        )
        index_map = np.fliplr(index_map)

        plotter.close()
        
        total = perf_counter() - start
        n_vis = len(visible_indices)
        cov   = np.sum(index_map >= 0) / (render_w * render_h) * 100
        logger.info(f"   ✅ Done in {total:.2f}s — {n_vis:,} visible faces, {cov:.1f}% coverage")
        logger.info(f"{'='*50}\n")

        return cls._normalize_result_dict({
            'index_map':       index_map,
            'visible_indices': visible_indices,
            'depth_map':       None,
            'inverted_index':  None,
            'scale_factor':    scale,
        }, compute_depth_map=False)

    @classmethod
    def _configure_vtk_camera_ortho(cls, plotter, ortho_camera, bounds: tuple) -> None:
        """Configure a PyVista plotter for a nadir orthographic view matching OrthoCamera.

        The camera looks straight down along OrthoCamera.get_vertical_direction_world(),
        covers the full ortho extent, and has parallel (orthographic) projection.
        """
        W, H = ortho_camera.width, ortho_camera.height

        # World-space corners at Z=0 CRS base plane
        TL = ortho_camera.pixel_to_xy_world(0,     0    )
        TR = ortho_camera.pixel_to_xy_world(W - 1, 0    )
        BL = ortho_camera.pixel_to_xy_world(0,     H - 1)
        BR = ortho_camera.pixel_to_xy_world(W - 1, H - 1)

        if any(c is None for c in [TL, TR, BL, BR]):
            return

        center       = (TL + TR + BL + BR) * 0.25
        top_center   = (TL + TR) * 0.5
        bot_center   = (BL + BR) * 0.5

        # View-up = direction from bottom-centre to top-centre (ortho "north")
        vu = top_center - bot_center
        vu_len = np.linalg.norm(vu)
        view_up = vu / vu_len if vu_len > 1e-12 else np.array([0., 1., 0.])

        # Parallel scale = half the world-space height of the ortho
        parallel_scale = vu_len * 0.5

        # Camera position: lift above scene along the vertical direction
        vertical_dir  = ortho_camera.get_vertical_direction_world()
        z_range       = max(abs(bounds[5] - bounds[4]), 1.0)
        lift          = z_range * 5.0
        cam_pos       = center - vertical_dir * lift

        camera = plotter.camera
        camera.position        = cam_pos.tolist()
        camera.focal_point     = center.tolist()
        camera.up              = view_up.tolist()
        camera.parallel_projection = True
        camera.parallel_scale  = float(parallel_scale)
        camera.clipping_range  = (lift * 0.05, lift + z_range * 4.0)

    @classmethod
    def _compute_mesh_visibility_fallback(cls,
                                          mesh_product: 'AbstractSceneProduct',
                                          K: np.ndarray,
                                          R: np.ndarray,
                                          t: np.ndarray,
                                          width: int,
                                          height: int,
                                          compute_depth_map: bool = True) -> dict:
        """
        Fallback: Compute mesh visibility using face-center point sampling.
        
        Used when VTK rasterization fails. Less accurate (sparse) but reliable.
        """
        logger.info("⚠️ Mesh visibility: Using face-center sampling (fallback)")
        
        try:
            face_centers = mesh_product.get_face_centers()
            face_ids = np.arange(len(face_centers), dtype=np.int32)
            
            result = cls.compute_visibility(
                face_centers, K, R, t, width, height,
                point_ids=face_ids,
                compute_depth_map=compute_depth_map
            )
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ Mesh visibility fallback failed: {e}")
            return {
                'index_map': np.full((height, width), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32),
                'depth_map': np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None,
                'inverted_index': None,
            }

    @classmethod
    def compute_visibility(cls, 
                           points_world: np.ndarray, 
                           K: np.ndarray, 
                           R: np.ndarray, 
                           t: np.ndarray, 
                           width: int, 
                           height: int,
                           point_ids: np.ndarray = None,
                           compute_depth_map: bool = True) -> dict:
        """
        Compute visibility for a cloud of points given camera parameters.

        Args:
            points_world (np.ndarray): (N, 3) array of 3D points in World Coordinates.
            K (np.ndarray): (3, 3) Intrinsic matrix.
            R (np.ndarray): (3, 3) Rotation matrix (World -> Camera).
            t (np.ndarray): (3,) Translation vector (World -> Camera).
            width (int): Image width.
            height (int): Image height.
            point_ids (np.ndarray, optional): (N,) array of global IDs. 
                                              If None, indices 0..N-1 are used.

        Returns:
            dict: {
                'index_map': (H, W) int32 array. Pixel value is Point ID or -1.
                'visible_indices': (M,) int32 array. Unique IDs of visible points.
            }
        """
        perf_counter = time.perf_counter
        start_time = perf_counter()
        log_section("👁️  POINT CLOUD VISIBILITY COMPUTATION", logger)
        
        # Default point IDs if not provided
        if point_ids is None:
            point_ids = np.arange(len(points_world), dtype=np.int32)
        
        logger.info(f"   Points: {len(points_world):,} | Render: {width}x{height} pixels")

        # 1. Prefer PyTorch (CUDA or CPU)
        if HAS_TORCH:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"   Using {device.upper()} backend")
            compute_start = perf_counter()
            result = cls._compute_torch(points_world, 
                                        point_ids, 
                                        K, R, t, 
                                        width,
                                        height, 
                                        device, 
                                        compute_depth_map=compute_depth_map)
            compute_time = perf_counter() - compute_start
        else:
            # 2. Fallback to NumPy if Torch is missing
            device = 'numpy'
            logger.info(f"   Using NUMPY backend (PyTorch not available)")
            compute_start = perf_counter()
            result = cls._compute_numpy(points_world, 
                                        point_ids, 
                                        K, R, t, 
                                        width, 
                                        height, 
                                        compute_depth_map=compute_depth_map)
            compute_time = perf_counter() - compute_start
        # Normalize result dtypes for consistency (index_map int32, depth_map float16)
        result = cls._normalize_result_dict(result, compute_depth_map)

        total_time = perf_counter() - start_time
        visible_count = len(result['visible_indices'])
        coverage = np.sum(result['index_map'] >= 0) / (width * height) * 100
        
        log_summary(
            "Point Cloud Visibility",
            [
                f"   - Computation (Z-buffer): {compute_time:.4f}s",
                f"   - Total Time            : {total_time:.4f}s",
                f"   - Result: {visible_count:,} visible points, {coverage:.1f}% pixel coverage",
            ],
            logger,
        )
        
        return result

    @classmethod
    def compute_batch_visibility(cls, 
                                 points_world: np.ndarray, 
                                 camera_params_list: list,
                                 point_ids: np.ndarray = None,
                                 compute_depth_map: bool = True) -> list:
        perf_counter = time.perf_counter
        start_time = perf_counter()
        log_section("👁️  BATCH POINT CLOUD VISIBILITY COMPUTATION (STREAMING MODE)", logger)
        
        N_total = len(points_world)
        if point_ids is None:
            point_ids = np.arange(N_total, dtype=np.int32)
        
        logger.info(f"   Points: {N_total:,} | Cameras: {len(camera_params_list)}")

        if not HAS_TORCH:
            # Fallback to numpy (omitted for brevity, keep your existing numpy fallback)
            pass 

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"   Using {device.upper()} backend")
        
        M = len(camera_params_list)
        results = []

        # Stream 100 million points (~1.2 GB) to the GPU at a time. 
        # Change this based on your hardware, but 100M is a very safe sweet spot.
        CHUNK_SIZE = 100_000_000 

        for i in range(M):
            cam_start = perf_counter()
            K_np, R_np, t_np, width, height = camera_params_list[i]
            
            # Load camera matrices to GPU
            K = torch.as_tensor(K_np, dtype=torch.float32, device=device)
            R = torch.as_tensor(R_np, dtype=torch.float32, device=device)
            t = torch.as_tensor(t_np, dtype=torch.float32, device=device)
            
            # Initialize the MASTER Z-buffer and Index Map for this camera
            # This takes very little memory (O(width * height))
            global_z_buffer = torch.full((height * width,), float('inf'), device=device, dtype=torch.float32)
            global_index_map = torch.full((height * width,), -1, device=device, dtype=torch.int32)

            # Local buffers reused each chunk (pre-allocated once per camera)
            local_z_buffer  = torch.empty((height * width,), device=device, dtype=torch.float32)
            local_index_map = torch.empty((height * width,), device=device, dtype=torch.int32)

            logger.info(f'   -> Processing {cam_label(i + 1)}/{M} in chunks...')

            for start_idx in range(0, N_total, CHUNK_SIZE):
                end_idx = min(start_idx + CHUNK_SIZE, N_total)

                chunk_pts = torch.as_tensor(points_world[start_idx:end_idx], dtype=torch.float32, device=device)
                chunk_ids = torch.as_tensor(point_ids[start_idx:end_idx], dtype=torch.int32, device=device)

                # Project: camera space
                points_cam = chunk_pts @ R.T + t
                x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

                # Project to pixel coords
                u = K[0, 0] * x / z + K[0, 2]
                v = K[1, 1] * y / z + K[1, 2]

                u_idx, v_idx = u.round().long(), v.round().long()

                valid_mask = (u_idx >= 0) & (u_idx < width) & (v_idx >= 0) & (v_idx < height) & (z > 0)

                valid_u, valid_v, valid_z = u_idx[valid_mask], v_idx[valid_mask], z[valid_mask]
                valid_ids = chunk_ids[valid_mask]

                if valid_ids.numel() == 0:
                    del chunk_pts, chunk_ids, points_cam, x, y, z, u, v, u_idx, v_idx, valid_mask
                    continue

                flat_indices = valid_v * width + valid_u

                # Z-buffer: keep minimum depth per pixel within this chunk
                local_z_buffer.fill_(float('inf'))
                try:
                    local_z_buffer.scatter_reduce_(0, flat_indices, valid_z, reduce='amin', include_self=True)
                except AttributeError:
                    raise RuntimeError('PyTorch version too old for scatter_reduce_.')

                is_closest = torch.abs(valid_z - local_z_buffer[flat_indices]) < 0.0001

                local_index_map.fill_(-1)
                local_index_map[flat_indices[is_closest]] = valid_ids[is_closest]

                # Merge chunk result into master buffers
                won_mask = local_z_buffer < global_z_buffer
                global_z_buffer[won_mask] = local_z_buffer[won_mask]
                global_index_map[won_mask] = local_index_map[won_mask]

                del chunk_pts, chunk_ids, points_cam, x, y, z, u, v, u_idx, v_idx, valid_mask
                del valid_u, valid_v, valid_z, valid_ids, flat_indices, is_closest, won_mask

            visible_indices = torch.unique(global_index_map[global_index_map != -1], sorted=True)

            if compute_depth_map:
                try:
                    global_z_buffer[global_z_buffer == float('inf')] = float('nan')
                    depth_map_np = global_z_buffer.view(height, width).cpu().numpy()
                except Exception:
                    depth_map_np = np.full((height, width), np.nan, dtype=np.float32)
            else:
                depth_map_np = None

            index_map_np = global_index_map.view(height, width).cpu().numpy()

            results.append({
                'index_map':       index_map_np,
                'visible_indices': visible_indices.cpu().numpy(),
                'depth_map':       depth_map_np,
                'inverted_index':  VisibilityManager._build_inverted_index(index_map_np),
            })

            log_cam_complete(cam_label(i + 1), perf_counter() - cam_start, logger)

            del K, R, t, global_z_buffer, global_index_map, visible_indices
            del local_z_buffer, local_index_map

        if device == 'cuda':
            torch.cuda.empty_cache()

        logger.info(f'\n   - Total Time: {perf_counter() - start_time:.4f}s')

        return results

    @staticmethod
    def _compute_torch(points_np, ids_np, K_np, R_np, t_np, width, height,
                       device='cpu', compute_depth_map=False):
        """
        PyTorch-based visibility computation.
        Uses scatter_reduce_ for efficient Z-buffering.
        Works on 'cuda' (fastest) and 'cpu' (via PyTorch tensors).
        """
        stage_times = {}
        overall_start = time.time()

        # --- Transfer ---
        xfer_start = time.time()
        points = torch.as_tensor(points_np, dtype=torch.float32, device=device)
        p_ids  = torch.as_tensor(ids_np,    dtype=torch.int32,   device=device)
        K = torch.as_tensor(K_np, dtype=torch.float32, device=device)
        R = torch.as_tensor(R_np, dtype=torch.float32, device=device)
        t = torch.as_tensor(t_np, dtype=torch.float32, device=device)
        stage_times['transfer'] = time.time() - xfer_start

        # --- Transform ---
        transform_start = time.time()
        points_cam = points @ R.T + t
        stage_times['transform'] = time.time() - transform_start

        x_cam = points_cam[:, 0]
        y_cam = points_cam[:, 1]
        z_cam = points_cam[:, 2]

        # --- Projection ---
        proj_start = time.time()
        u = K[0, 0] * x_cam / z_cam + K[0, 2]
        v = K[1, 1] * y_cam / z_cam + K[1, 2]
        stage_times['projection'] = time.time() - proj_start

        # --- Bounds filter ---
        bounds_start = time.time()
        u_idx = u.round().long()
        v_idx = v.round().long()

        valid_mask = (
            (u_idx >= 0) & (u_idx < width) &
            (v_idx >= 0) & (v_idx < height) &
            (z_cam > 0)
        )

        valid_u   = u_idx[valid_mask]
        valid_v   = v_idx[valid_mask]
        valid_z   = z_cam[valid_mask]
        valid_ids = p_ids[valid_mask]
        stage_times['bounds'] = time.time() - bounds_start

        if valid_ids.numel() == 0:
            return VisibilityManager._normalize_result_dict(
                np.full((height, width), -1, dtype=np.int32),
                np.array([], dtype=np.int32),
                np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None,
                None,
                compute_depth_map,
            )

        # --- Z-buffer ---
        zbuf_start = time.time()
        flat_indices = valid_v * width + valid_u

        z_buffer = torch.full((height * width,), float('inf'), device=device, dtype=torch.float32)
        try:
            z_buffer.scatter_reduce_(0, flat_indices, valid_z, reduce='amin', include_self=True)
        except AttributeError:
            warnings.warn('PyTorch version too old for scatter_reduce_. Falling back to NumPy implementation.')
            return VisibilityManager._compute_numpy(points_np, ids_np, K_np, R_np, t_np, width, height)

        min_z_at_pixel = z_buffer[flat_indices]
        is_closest = torch.abs(valid_z - min_z_at_pixel) < 0.0001

        final_pixel_indices = flat_indices[is_closest]
        final_ids           = valid_ids[is_closest]
        stage_times['zbuffer'] = time.time() - zbuf_start

        # --- Output ---
        output_start = time.time()
        index_map_tensor = torch.full((height * width,), -1, device=device, dtype=torch.int32)
        index_map_tensor[final_pixel_indices] = final_ids

        index_map_2d    = index_map_tensor.view(height, width)
        visible_indices = torch.unique(final_ids, sorted=True)

        if compute_depth_map:
            try:
                z_buffer[z_buffer == float('inf')] = float('nan')
                depth_map_2d = z_buffer.view(height, width)
                depth_map_np = depth_map_2d.cpu().numpy()
            except Exception:
                depth_map_np = np.full((height, width), np.nan, dtype=np.float32)
        else:
            depth_map_np = None

        index_map_np = index_map_2d.cpu().numpy()
        stage_times['output'] = time.time() - output_start

        if str(device) == 'cuda':
            torch.cuda.empty_cache()

        return VisibilityManager._normalize_result_dict(
            index_map_np,
            visible_indices.cpu().numpy(),
            depth_map_np,
            None,
            compute_depth_map,
        )

    @staticmethod
    def _compute_numpy(points, ids, K, R, t, width, height, compute_depth_map=False):
        """
        CPU-based visibility computation (Legacy / Fallback).
        Uses 'Sort by Depth' optimization to handle occlusion efficiently without loops.
        """
        stage_times = {}
        overall_start = time.time()

        # --- Transform ---
        transform_start = time.time()
        points_cam = points @ R.T + t
        stage_times['transform'] = time.time() - transform_start

        x_cam = points_cam[:, 0]
        y_cam = points_cam[:, 1]
        z_cam = points_cam[:, 2]

        # --- Projection ---
        proj_start = time.time()
        with np.errstate(divide='ignore', invalid='ignore'):
            u = K[0, 0] * x_cam / z_cam + K[0, 2]
            v = K[1, 1] * y_cam / z_cam + K[1, 2]
        stage_times['projection'] = time.time() - proj_start

        # --- Bounds filter ---
        bounds_start = time.time()
        u_idx = np.rint(u).astype(np.int32)
        v_idx = np.rint(v).astype(np.int32)

        valid_mask = (
            (u_idx >= 0) & (u_idx < width) &
            (v_idx >= 0) & (v_idx < height) &
            (z_cam > 0)
        )

        u_valid  = u_idx[valid_mask]
        v_valid  = v_idx[valid_mask]
        z_valid  = z_cam[valid_mask]
        id_valid = ids[valid_mask]
        stage_times['bounds'] = time.time() - bounds_start

        if len(id_valid) == 0:
            return VisibilityManager._normalize_result_dict(
                np.full((height, width), -1, dtype=np.int32),
                np.array([], dtype=np.int32),
                np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None,
                None,
                compute_depth_map,
            )

        # --- Z-buffer (sort-by-depth) ---
        zbuf_start = time.time()
        sort_order = np.argsort(z_valid)[::-1]   # farthest first so closer overwrites

        u_sorted  = u_valid[sort_order]
        v_sorted  = v_valid[sort_order]
        id_sorted = id_valid[sort_order]
        z_sorted  = z_valid[sort_order]

        index_map = np.full((height, width), -1, dtype=np.int32)
        depth_map = np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None

        index_map[v_sorted, u_sorted] = id_sorted
        if compute_depth_map:
            depth_map[v_sorted, u_sorted] = z_sorted.astype(np.float32)

        stage_times['zbuffer'] = time.time() - zbuf_start

        visible_indices = np.unique(index_map[index_map != -1])

        return VisibilityManager._normalize_result_dict(
            index_map,
            visible_indices,
            depth_map,
            None,
            compute_depth_map,
        )
