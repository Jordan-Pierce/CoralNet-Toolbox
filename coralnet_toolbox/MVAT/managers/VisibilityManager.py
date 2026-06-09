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

# Shader sources and GPU utilities
from coralnet_toolbox.MVAT.shaders import VERT as _MGL_VERT
from coralnet_toolbox.MVAT.shaders import FRAG_FACE_ID_INT as _MGL_FRAG_INT
from coralnet_toolbox.MVAT.shaders.gpu_interop import (
    _resolve_gl_fns,
    _build_mvp,
    _pbo_cuda_readback,
    _load_cudart,
)

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

        # Try fast GPU/CPU paths first using already-cached coordinates.
        # Only call prepare_geometry() if caches are missing.
        coords_cached_np = getattr(scene_product, '_element_centers_np', None)
        coords_cached_pt = getattr(scene_product, '_cached_face_centers_pt', None)

        # Fast GPU path using cached coordinates when available.
        if HAS_TORCH and torch.cuda.is_available():
            coords_t = coords_cached_pt
            if coords_t is None:
                coords_np = coords_cached_np
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
        coords_np = coords_cached_np
        if coords_np is None and hasattr(scene_product, 'get_points_array'):
            coords_np = scene_product.get_points_array()
        if coords_np is None and hasattr(scene_product, 'get_face_centers'):
            coords_np = scene_product.get_face_centers()

        # Only prepare geometry if we still don't have coordinates
        if coords_np is None and hasattr(scene_product, 'prepare_geometry'):
            try:
                scene_product.prepare_geometry()
                coords_np = getattr(scene_product, '_element_centers_np', None)
            except Exception:
                pass

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
                                      compute_depth_map: bool = False) -> dict:
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
                'depth_map':      np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None,
                'inverted_index': None,
            }, compute_depth_map)

        element_type = primary_target.get_element_type()

        # Strategy dispatch based on element type
        if element_type == 'point':
            from coralnet_toolbox.MVAT.core.Products import PointCloudProduct
            if isinstance(primary_target, PointCloudProduct):
                points = primary_target.get_points_array()
                if points is not None:
                    result = cls.compute_point_cloud_visibility(points, K, R, t, width, height,
                                                                compute_depth_map=compute_depth_map)
                    result['element_type'] = 'point'
                    return result

        elif element_type == 'face':
            # Mesh visibility with ModernGL (VTK removed in Phase 3)
            # Keep visible_indices=False to skip expensive computation in batch paths.
            # compute_depth_map is controlled by caller (True for interactive SAM, False for batch).
            results = cls.compute_batch_mesh_visibility_moderngl(
                primary_target, [(K, R, t, width, height)],
                compute_depth_map=compute_depth_map,
                compute_visible_indices=False,
                pixel_budget=None,
            )
            result = results[0]
            result['element_type'] = 'face'
            return result

        # Fallback: empty result
        return cls._normalize_result_dict({
            'index_map':      np.full((height, width), -1, dtype=np.int32),
            'visible_indices': np.array([], dtype=np.int32),
            'depth_map':      np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None,
            'element_type':   element_type,
            'inverted_index': None,
        }, compute_depth_map)


    # =========================================================================
    # moderngl batch rasterizer  (zero-PCIe primary path)
    # =========================================================================

    @classmethod
    def setup_batch_moderngl_context(cls, mesh_product, pixel_budget,
                                     sample_width, sample_height):
        """Create a moderngl offscreen context and upload mesh geometry once."""
        import moderngl

        ctx = moderngl.create_context(standalone=True)
        ctx.enable(moderngl.DEPTH_TEST)

        mesh = mesh_product.get_mesh()
        if not mesh.is_all_triangles:
            mesh = mesh.triangulate()

        n_cells  = mesh.n_cells
        vertices = np.asarray(mesh.points, dtype=np.float32)
        faces    = mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32)

        vbo = ctx.buffer(vertices.tobytes())
        ibo = ctx.buffer(faces.tobytes())

        # Single-channel R32I face-ID target (always int32; tiered encoding removed)
        prog_int = ctx.program(vertex_shader=_MGL_VERT, fragment_shader=_MGL_FRAG_INT)
        vao_int  = ctx.vertex_array(prog_int, [(vbo, '3f', 'position')], ibo)

        gl_fns = _resolve_gl_fns()

        face_centers_pt = None
        if HAS_TORCH and torch.cuda.is_available():
            face_centers_pt = torch.from_numpy(
                mesh.cell_centers().points.astype(np.float32)
            ).cuda()

        # Zero-PCIe CUDA-GL readback capability (for the distorted-camera path).
        # cudart is loaded eagerly; interop_ok is tentative and self-disables on the
        # first failed readback in the render loop.
        import os
        cudart = None
        interop_ok = False

        if HAS_TORCH and torch.cuda.is_available():
            try:
                cudart = _load_cudart()
                interop_ok = True
            except Exception as exc:
                logger.debug(f"   CUDA-GL interop unavailable ({exc}); using CPU readback")
                cudart = None
                interop_ok = False

        logger.debug(f"   ✅ moderngl context ready: {n_cells:,} faces (32-bit int encoding)")

        return {
            'ctx':             ctx,
            'prog_int':        prog_int,
            'vao_int':         vao_int,
            'n_cells':         n_cells,
            'gl_fns':          gl_fns,
            'pixel_budget':    pixel_budget,
            'face_centers_pt': face_centers_pt,
            'cudart':          cudart,
            'interop_ok':      interop_ok,
            '_fbo_cache':      {},
        }

    @classmethod
    def compute_batch_mesh_visibility_moderngl(
        cls,
        mesh_product,
        camera_params_list: list,
        compute_depth_map: bool = True,
        compute_visible_indices: bool = True,
        pixel_budget: Optional[int] = None,
        upsample_to_native: bool = False,
        progress_callback=None,
        mgl_context: dict = None,
        camera_index_offset: int = 0,
        use_viewport_cropping: bool = True,
        gpu_index_positions: Optional[set] = None,
    ) -> list:
        """GPU rasterization via moderngl with CPU framebuffer readback.

        Returns a list of result dicts with 'index_map', 'visible_indices', 'depth_map', etc.

        ``gpu_index_positions`` is an optional set of batch-relative camera indices
        (0..len-1) whose index map should be returned as a CUDA int32 tensor in
        ``result['index_map_gpu']`` (with ``index_map=None``) via zero-PCIe CUDA-GL
        readback, instead of a CPU numpy ``index_map``. Used by the distorted-camera
        path to feed the warp directly. Requires ``mgl_context['interop_ok']``; any
        failure silently falls back to the CPU numpy path for that camera.
        """
        import time
        import logging
        perf_counter = time.perf_counter

        # Set up file logging for detailed debug output (disabled by default)
        # To enable: set environment variable VISIBILITY_DEBUG=1
        debug_handler = logging.FileHandler('visibility_timing_debug.log', mode='w', encoding='utf-8')
        debug_handler.setLevel(logging.DEBUG)
        debug_formatter = logging.Formatter('%(message)s')
        debug_handler.setFormatter(debug_formatter)
        logger.addHandler(debug_handler)
        # Only enable debug logging if explicitly requested
        import os
        os.environ.setdefault('VISIBILITY_DEBUG', '1')
        if os.environ.get('VISIBILITY_DEBUG', '0') == '1':
            logger.setLevel(logging.DEBUG)

        def _scale_for(w, h):
            native = w * h
            if pixel_budget is None or pixel_budget <= 0 or native <= pixel_budget:
                return 1.0
            return float(np.sqrt(pixel_budget / native))

        budget_str = "Native" if not pixel_budget else f"{pixel_budget/1e6:.1f}MP"
        log_section(
            f"🎨 MODERNGL BATCH RASTERIZATION — {len(camera_params_list)} cameras "
            f"(budget: {budget_str})", logger
        )
        start_time = perf_counter()

        if mgl_context is None:
            mgl_context = cls.setup_batch_moderngl_context(
                mesh_product, pixel_budget,
                camera_params_list[0][3], camera_params_list[0][4],
            )

        ctx          = mgl_context['ctx']
        prog_int     = mgl_context['prog_int']
        vao_int      = mgl_context['vao_int']
        gl_fns       = mgl_context['gl_fns']
        fbo_cache    = mgl_context['_fbo_cache']
        cudart       = mgl_context.get('cudart')

        gpu_index_positions = gpu_index_positions or set()

        mesh        = mesh_product.get_mesh()
        mesh_bounds = mesh.bounds

        def _get_fbo(w, h):
            key = (w, h)
            if key not in fbo_cache:
                # Single-channel R32I face-ID target (always int32).
                fbo_cache[key] = ctx.framebuffer(
                    color_attachments=[ctx.texture((w, h), 1, dtype='i4')],
                    depth_attachment=ctx.depth_texture((w, h)),
                )
            return fbo_cache[key]

        results = []
        # Accumulators for totals
        _t_render_total     = 0.0
        _t_readback_total   = 0.0
        _t_decode_total     = 0.0
        _t_assembly_total   = 0.0
        _t_setup_total      = 0.0
        _t_crop_total       = 0.0
        _t_fbo_total        = 0.0
        _t_mvp_total        = 0.0
        _t_python_overhead  = 0.0
        camera_total_sum    = 0.0

        # Min/max trackers for statistics
        _stats = {
            'crop': {'min': float('inf'), 'max': 0.0},
            'fbo': {'min': float('inf'), 'max': 0.0},
            'mvp': {'min': float('inf'), 'max': 0.0},
            'render': {'min': float('inf'), 'max': 0.0},
            'readback': {'min': float('inf'), 'max': 0.0},
            'gpu_sync': {'min': float('inf'), 'max': 0.0},
            'fbo_read': {'min': float('inf'), 'max': 0.0},
            'data_proc': {'min': float('inf'), 'max': 0.0},
            'decode_data': {'min': float('inf'), 'max': 0.0},
            'flip': {'min': float('inf'), 'max': 0.0},
            'copy': {'min': float('inf'), 'max': 0.0},
            'decode': {'min': float('inf'), 'max': 0.0},
            'cpu_convert': {'min': float('inf'), 'max': 0.0},
            'indices_compute': {'min': float('inf'), 'max': 0.0},
            'assembly': {'min': float('inf'), 'max': 0.0},
            'paste': {'min': float('inf'), 'max': 0.0},
            'paste_alloc': {'min': float('inf'), 'max': 0.0},
            'paste_assign': {'min': float('inf'), 'max': 0.0},
            'dict_build': {'min': float('inf'), 'max': 0.0},
            'normalize': {'min': float('inf'), 'max': 0.0},
            'append': {'min': float('inf'), 'max': 0.0},
        }

        for i, (K, R, t, width, height) in enumerate(camera_params_list):
            cam_start = perf_counter()

            if progress_callback is not None:
                progress_callback(camera_index_offset + i + 1,
                                  camera_index_offset + len(camera_params_list))

            dynamic_scale = _scale_for(width, height)
            render_w = max(1, int(round(width  * dynamic_scale)))
            render_h = max(1, int(round(height * dynamic_scale)))

            K_scaled       = K.copy()
            K_scaled[0, :3] *= dynamic_scale
            K_scaled[1, :3] *= dynamic_scale

            # Viewport cropping calculation
            t0_crop = perf_counter()
            u_min = v_min = 0
            crop_w, crop_h = render_w, render_h
            if use_viewport_cropping:
                u_min, u_max, v_min, v_max, crop_status = cls._get_2d_bounding_box(
                    mesh_bounds, K_scaled, R, t, render_w, render_h
                )
                if crop_status == "OFF_SCREEN":
                    results.append(cls._normalize_result_dict({
                        'index_map':      np.full((render_h, render_w), -1, dtype=np.int32),
                        'visible_indices': np.array([], dtype=np.int32),
                        'depth_map':      np.full((render_h, render_w), np.nan, dtype=np.float32) if compute_depth_map else None,
                        'inverted_index': None,
                        'scale_factor':   dynamic_scale,
                    }, compute_depth_map))
                    camera_total_sum += perf_counter() - cam_start
                    continue
                elif crop_status == "CROP":
                    crop_w = u_max - u_min
                    crop_h = v_max - v_min
                    K_scaled[0, 2] -= u_min
                    K_scaled[1, 2] -= v_min
            t_crop = perf_counter() - t0_crop

            # FBO setup (retrieval/creation + binding)
            t0_fbo = perf_counter()
            fbo = _get_fbo(crop_w, crop_h)
            fbo.use()
            ctx.viewport = (0, 0, crop_w, crop_h)
            ctx.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
            t_fbo = perf_counter() - t0_fbo

            # MVP build + render
            t0_mvp = perf_counter()
            mvp = _build_mvp(K_scaled, R, t, crop_w, crop_h)
            t_mvp_build = perf_counter() - t0_mvp

            t0_render = perf_counter()
            prog_int['mvp'].write(mvp.tobytes())
            vao_int.render()
            # NOTE: Removed ctx.finish() — fbo.read() synchronizes implicitly.
            # This saves ~3.2ms per camera by avoiding redundant GPU stall.
            t_render = perf_counter() - t0_render

            t_setup = t_crop + t_fbo  # Total setup time

            # ================================================================
            # READBACK PHASE: GPU sync + texture transfer + data processing
            # ================================================================
            t0_readback_total = perf_counter()

            # GPU sync: ensure render is complete before readback
            t0_sync = perf_counter()
            ctx.finish()
            t_gpu_sync = perf_counter() - t0_sync

            # ----------------------------------------------------------------
            # GPU PATH: zero-PCIe CUDA-GL readback for distorted cameras.
            # Keep the index map on the GPU as a CUDA int32 tensor so the
            # distortion warp consumes it directly (no CPU round trip). Any
            # failure self-disables interop and falls through to the CPU path.
            # ----------------------------------------------------------------
            if i in gpu_index_positions and mgl_context.get('interop_ok') and cudart is not None:
                import torch
                gpu_crop = None
                try:
                    # flip=False: the vertex shader already orients top-to-bottom.
                    gpu_crop = _pbo_cuda_readback(gl_fns, cudart, crop_w, crop_h, flip=False)
                except Exception as exc:
                    logger.debug(f"      CUDA-GL readback raised ({exc}); disabling interop")
                    gpu_crop = None

                if gpu_crop is None:
                    mgl_context['interop_ok'] = False
                    logger.debug("      CUDA-GL readback unavailable; falling back to CPU readback")
                else:
                    # Reverse the +1 ID offset (background 0 → -1) on the GPU.
                    gpu_crop = gpu_crop - 1
                    # Paste the crop into the full (low-res) render canvas on the GPU.
                    # Upsampling is intentionally deferred: F.grid_sample fuses it into
                    # the warp using the native-resolution lens grid.
                    if u_min == 0 and v_min == 0 and crop_w == render_w and crop_h == render_h:
                        full_index_map_gpu = gpu_crop
                    else:
                        full_index_map_gpu = torch.full(
                            (render_h, render_w), -1, dtype=torch.int32, device=gpu_crop.device
                        )
                        full_index_map_gpu[v_min:v_min + crop_h, u_min:u_min + crop_w] = gpu_crop
                    results.append({
                        'index_map':       None,
                        'index_map_gpu':   full_index_map_gpu,
                        'visible_indices': np.array([], dtype=np.int32),
                        'depth_map':       None,
                        'inverted_index':  None,
                        'scale_factor':    dynamic_scale,
                    })
                    camera_total_sum += perf_counter() - cam_start
                    continue

            # The vertex shader bakes the vertical flip into clip space, so the
            # readback is already in top-to-bottom image order (no CPU [::-1]).
            fbo_to_read = fbo

            # Texture readback: single-channel R32I → int32 (no bit-unpack decode)
            t0_read = perf_counter()
            raw = fbo_to_read.read(components=1, dtype='i4')
            t_fbo_read = perf_counter() - t0_read

            # Data processing: reverse the +1 ID offset (0 = background → -1).
            # frombuffer is read-only; the subtract yields a fresh writable
            # contiguous int32 array — no flip and no ascontiguousarray needed.
            t0_proc_total = perf_counter()

            t0_decode_data = perf_counter()
            crop_index_map = np.frombuffer(raw, dtype=np.int32).reshape(crop_h, crop_w) - 1
            t_decode_data = perf_counter() - t0_decode_data

            # Flip + copy now happen on the GPU (MVP Y-flip), so these are no-ops.
            t_flip = 0.0
            t_copy = 0.0

            t_data_proc = perf_counter() - t0_proc_total

            # Store encoding info for later breakdown logging
            _decode_encoding = 'int32'

            t_readback = perf_counter() - t0_readback_total

            # ================================================================
            # DECODE PHASE: visible indices computation
            # ================================================================
            t0_decode_total = perf_counter()
            t_cpu_convert = 0.0
            t_offset_adj = 0.0

            # Compute visible indices only if requested (skip for ~47ms savings in batch mode)
            t0_indices = perf_counter()
            if compute_visible_indices:
                valid_indices = crop_index_map[crop_index_map >= 0]
                if valid_indices.size > 0:
                    visible_indices = np.where(np.bincount(valid_indices) > 0)[0].astype(np.int32)
                else:
                    visible_indices = np.array([], dtype=np.int32)
            else:
                visible_indices = np.array([], dtype=np.int32)
            t_indices_compute = perf_counter() - t0_indices

            t_decode = perf_counter() - t0_decode_total

            # Log detailed data-proc breakdown (DEBUG level) — only if debug is enabled
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"      Data-proc breakdown ({_decode_encoding} enc): Decode={t_decode_data*1000:.2f}ms | "
                    f"Flip={t_flip*1000:.2f}ms | Copy={t_copy*1000:.2f}ms | "
                    f"Total={t_data_proc*1000:.2f}ms"
                )

                # Log detailed decode breakdown (DEBUG level)
                logger.debug(
                    f"      Decode breakdown: CPU-convert={t_cpu_convert*1000:.2f}ms | "
                    f"Offset-adj={t_offset_adj*1000:.2f}ms | "
                    f"Indices={t_indices_compute*1000:.2f}ms | "
                    f"Total={t_decode*1000:.2f}ms"
                )

            crop_depth_map = None
            if compute_depth_map:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"      → Computing depth map for camera {camera_index_offset + i + 1}")
                # Read depth texture directly from FBO depth attachment
                try:
                    # ModernGL depth texture is read without components parameter
                    depth_raw = fbo.depth_attachment.read()
                    # No [::-1]: the vertex shader's Y-flip already orients the
                    # depth attachment top-to-bottom, matching the index map.
                    depth_buffer = np.frombuffer(depth_raw, dtype=np.float32).reshape(crop_h, crop_w)

                    # Linearize OpenGL depth [0, 1] to CV depth (meters)
                    # Using the exact near/far clipping planes defined in _build_mvp
                    near, far = 0.01, 100000.0
                    z_ndc = 2.0 * depth_buffer - 1.0

                    with np.errstate(divide='ignore', invalid='ignore'):
                        linear_depth = (2.0 * near * far) / (far + near - z_ndc * (far - near))

                    # Mask out the background (where the camera sees nothing)
                    linear_depth[crop_index_map == -1] = np.nan
                    crop_depth_map = linear_depth.astype(np.float32)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"      ✓ Depth map extracted: shape={crop_depth_map.shape}, range=[{np.nanmin(crop_depth_map):.2f}, {np.nanmax(crop_depth_map):.2f}]")
                except Exception as depth_err:
                    logger.warning(f"      ⚠️ Depth map extraction failed ({depth_err}); skipping depth")
                    crop_depth_map = None

            # ================================================================
            # ASSEMBLY PHASE: Paste, upsample, normalize (overhead timing)
            # ================================================================
            t0_assembly = perf_counter()

            # Paste crop back into full canvas (with detailed breakdown)
            t0_paste = perf_counter()

            t0_alloc = perf_counter()
            t_alloc = 0.0
            t0_assign = perf_counter()
            t_assign = 0.0

            # Fast path: if rendering full-screen, skip allocation and assignment overhead
            if u_min == 0 and v_min == 0 and crop_w == render_w and crop_h == render_h:
                full_index_map = crop_index_map
            else:
                t0_alloc = perf_counter()
                full_index_map = np.full((render_h, render_w), -1, dtype=np.int32)
                t_alloc = perf_counter() - t0_alloc

                t0_assign = perf_counter()
                full_index_map[v_min:v_min + crop_h, u_min:u_min + crop_w] = crop_index_map
                t_assign = perf_counter() - t0_assign

            t_paste = perf_counter() - t0_paste

            # Depth assembly
            t0_depth_assemble = perf_counter()
            full_depth_map = None
            if compute_depth_map:
                # Fast path: if rendering full-screen, use crop directly
                if u_min == 0 and v_min == 0 and crop_w == render_w and crop_h == render_h:
                    full_depth_map = crop_depth_map
                else:
                    full_depth_map = np.full((render_h, render_w), np.nan, dtype=np.float32)
                    if crop_depth_map is not None:
                        full_depth_map[v_min:v_min + crop_h, u_min:u_min + crop_w] = crop_depth_map
            t_depth_assemble = perf_counter() - t0_depth_assemble

            # Upsampling
            t0_upsample = perf_counter()
            result_scale = dynamic_scale
            if upsample_to_native and dynamic_scale < 1.0:
                import cv2 as _cv2
                full_index_map = _cv2.resize(full_index_map, (width, height), interpolation=_cv2.INTER_NEAREST)
                if full_depth_map is not None:
                    full_depth_map = _cv2.resize(full_depth_map, (width, height), interpolation=_cv2.INTER_NEAREST)
                result_scale = 1.0
            t_upsample = perf_counter() - t0_upsample

            # Result dict building and normalization (detailed timing)
            t0_result_dict = perf_counter()
            result_dict = {
                'index_map':       full_index_map,
                'visible_indices': visible_indices,
                'depth_map':       full_depth_map,
                'inverted_index':  None,
                'scale_factor':    result_scale,
            }
            t_dict_build = perf_counter() - t0_result_dict

            t0_normalize = perf_counter()
            normalized_result = cls._normalize_result_dict(result_dict, compute_depth_map)
            t_normalize = perf_counter() - t0_normalize

            t0_append = perf_counter()
            results.append(normalized_result)
            t_append = perf_counter() - t0_append

            t_result_dict = t_dict_build + t_normalize + t_append

            t_assembly = perf_counter() - t0_assembly

            # Log detailed assembly breakdown (DEBUG level) — only if debug is enabled
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"      Assembly breakdown: Paste={t_paste*1000:.2f}ms | "
                    f"Depth-assemble={t_depth_assemble*1000:.2f}ms | Upsample={t_upsample*1000:.2f}ms | "
                    f"Result-dict={t_result_dict*1000:.2f}ms | Total={t_assembly*1000:.2f}ms"
                )

            cam_time = perf_counter() - cam_start
            # Calculate unaccounted time properly: total - all explicit timers
            t_accounted = t_setup + t_render + t_readback + t_decode + t_assembly
            t_unaccounted = cam_time - t_accounted

            # Accumulate timers and track min/max
            camera_total_sum  += cam_time
            _t_setup_total    += t_setup
            _t_crop_total     += t_crop
            _t_fbo_total      += t_fbo
            _t_mvp_total      += t_mvp_build
            _t_render_total   += t_render
            _t_readback_total += t_readback
            _t_decode_total   += t_decode
            _t_assembly_total += t_assembly
            _t_python_overhead += t_unaccounted

            # Update min/max statistics
            _stats['crop']['min'] = min(_stats['crop']['min'], t_crop)
            _stats['crop']['max'] = max(_stats['crop']['max'], t_crop)
            _stats['fbo']['min'] = min(_stats['fbo']['min'], t_fbo)
            _stats['fbo']['max'] = max(_stats['fbo']['max'], t_fbo)
            _stats['mvp']['min'] = min(_stats['mvp']['min'], t_mvp_build)
            _stats['mvp']['max'] = max(_stats['mvp']['max'], t_mvp_build)
            _stats['render']['min'] = min(_stats['render']['min'], t_render)
            _stats['render']['max'] = max(_stats['render']['max'], t_render)
            _stats['readback']['min'] = min(_stats['readback']['min'], t_readback)
            _stats['readback']['max'] = max(_stats['readback']['max'], t_readback)
            _stats['gpu_sync']['min'] = min(_stats['gpu_sync']['min'], t_gpu_sync)
            _stats['gpu_sync']['max'] = max(_stats['gpu_sync']['max'], t_gpu_sync)
            _stats['fbo_read']['min'] = min(_stats['fbo_read']['min'], t_fbo_read)
            _stats['fbo_read']['max'] = max(_stats['fbo_read']['max'], t_fbo_read)
            _stats['data_proc']['min'] = min(_stats['data_proc']['min'], t_data_proc)
            _stats['data_proc']['max'] = max(_stats['data_proc']['max'], t_data_proc)
            _stats['decode_data']['min'] = min(_stats['decode_data']['min'], t_decode_data)
            _stats['decode_data']['max'] = max(_stats['decode_data']['max'], t_decode_data)
            _stats['flip']['min'] = min(_stats['flip']['min'], t_flip)
            _stats['flip']['max'] = max(_stats['flip']['max'], t_flip)
            _stats['copy']['min'] = min(_stats['copy']['min'], t_copy)
            _stats['copy']['max'] = max(_stats['copy']['max'], t_copy)
            _stats['cpu_convert']['min'] = min(_stats['cpu_convert']['min'], t_cpu_convert)
            _stats['cpu_convert']['max'] = max(_stats['cpu_convert']['max'], t_cpu_convert)
            _stats['indices_compute']['min'] = min(_stats['indices_compute']['min'], t_indices_compute)
            _stats['indices_compute']['max'] = max(_stats['indices_compute']['max'], t_indices_compute)
            _stats['decode']['min'] = min(_stats['decode']['min'], t_decode)
            _stats['decode']['max'] = max(_stats['decode']['max'], t_decode)
            _stats['assembly']['min'] = min(_stats['assembly']['min'], t_assembly)
            _stats['assembly']['max'] = max(_stats['assembly']['max'], t_assembly)
            _stats['paste']['min'] = min(_stats['paste']['min'], t_paste)
            _stats['paste']['max'] = max(_stats['paste']['max'], t_paste)
            _stats['paste_alloc']['min'] = min(_stats['paste_alloc']['min'], t_alloc)
            _stats['paste_alloc']['max'] = max(_stats['paste_alloc']['max'], t_alloc)
            _stats['paste_assign']['min'] = min(_stats['paste_assign']['min'], t_assign)
            _stats['paste_assign']['max'] = max(_stats['paste_assign']['max'], t_assign)
            _stats['dict_build']['min'] = min(_stats['dict_build']['min'], t_dict_build)
            _stats['dict_build']['max'] = max(_stats['dict_build']['max'], t_dict_build)
            _stats['normalize']['min'] = min(_stats['normalize']['min'], t_normalize)
            _stats['normalize']['max'] = max(_stats['normalize']['max'], t_normalize)
            _stats['append']['min'] = min(_stats['append']['min'], t_append)
            _stats['append']['max'] = max(_stats['append']['max'], t_append)

            # Log comprehensive per-camera breakdown (DEBUG level) — only if debug is enabled
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"      Camera {camera_index_offset + i + 1} timing breakdown:"
                )
                logger.debug(
                    f"        Setup: Crop={t_crop*1000:.2f}ms | FBO={t_fbo*1000:.2f}ms | "
                    f"MVP={t_mvp_build*1000:.2f}ms | Total={t_setup*1000:.2f}ms"
                )
                logger.debug(
                    f"        Render: {t_render*1000:.2f}ms"
                )
                logger.debug(
                    f"        Readback breakdown: GPU-sync={t_gpu_sync*1000:.2f}ms | "
                    f"FBO-read={t_fbo_read*1000:.2f}ms | Data-proc={t_data_proc*1000:.2f}ms | "
                    f"Total={t_readback*1000:.2f}ms"
                )

            log_cam_breakdown(
                cam_label(camera_index_offset + i + 1),
                cam_time, 0.0, t_render, t_readback, t_decode, 0.0, 0.0, 0.0, logger
            )

        n_cams = max(1, len(camera_params_list))
        total_time = perf_counter() - start_time
        accounted_time = _t_setup_total + _t_render_total + _t_readback_total + _t_decode_total
        unaccounted_time = total_time - accounted_time
        readback_mode = "CPU readback"

        # Format statistics with min/max/avg
        def fmt_stat(name, total, stat_dict):
            avg = total / n_cams * 1000
            min_ms = stat_dict['min'] * 1000 if stat_dict['min'] != float('inf') else 0
            max_ms = stat_dict['max'] * 1000
            return f"   - {name:16} : avg={avg:5.1f}ms  (min={min_ms:5.1f}ms, max={max_ms:5.1f}ms)"

        log_summary(
            f"moderngl Batch Rasterization ({readback_mode}) — Statistics (min/max/avg)",
            [
                f"🔧 SETUP PHASE:",
                fmt_stat("Crop calc", _t_crop_total, _stats['crop']),
                fmt_stat("FBO setup", _t_fbo_total, _stats['fbo']),
                fmt_stat("MVP calc", _t_mvp_total, _stats['mvp']),
                f"",
                f"🎨 RENDER PHASE:",
                fmt_stat("GPU Rasterize", _t_render_total, _stats['render']),
                f"",
                f"📥 READBACK + PROCESSING:",
                fmt_stat("GPU sync", _t_setup_total*0, _stats['gpu_sync']),
                fmt_stat("FBO read", _t_setup_total*0, _stats['fbo_read']),
                fmt_stat("Data-proc (total)", _t_setup_total*0, _stats['data_proc']),
                f"   ├─ Decode data  : {_stats['decode_data']['min']*1000:.1f}ms–{_stats['decode_data']['max']*1000:.1f}ms",
                f"   ├─ Flip array   : {_stats['flip']['min']*1000:.1f}ms–{_stats['flip']['max']*1000:.1f}ms",
                f"   └─ Copy array   : {_stats['copy']['min']*1000:.1f}ms–{_stats['copy']['max']*1000:.1f}ms",
                fmt_stat("Decode phase", _t_decode_total, _stats['decode']),
                f"   ├─ CPU convert  : {_stats['cpu_convert']['min']*1000:.1f}ms–{_stats['cpu_convert']['max']*1000:.1f}ms",
                f"   └─ Indices compute: {_stats['indices_compute']['min']*1000:.1f}ms–{_stats['indices_compute']['max']*1000:.1f}ms",
                fmt_stat("Assembly", _t_assembly_total, _stats['assembly']),
                f"   └─ Paste (total): {_stats['paste']['min']*1000:.1f}ms–{_stats['paste']['max']*1000:.1f}ms",
                f"      ├─ Alloc     : {_stats['paste_alloc']['min']*1000:.1f}ms–{_stats['paste_alloc']['max']*1000:.1f}ms",
                f"      └─ Assign    : {_stats['paste_assign']['min']*1000:.1f}ms–{_stats['paste_assign']['max']*1000:.1f}ms",
                f"",
                f"🔨 RESULT BUILDING & FINALIZATION:",
                f"   - Dict build    : {_stats['dict_build']['min']*1000:.1f}ms–{_stats['dict_build']['max']*1000:.1f}ms",
                f"   - Normalize     : {_stats['normalize']['min']*1000:.1f}ms–{_stats['normalize']['max']*1000:.1f}ms",
                f"   - List append   : {_stats['append']['min']*1000:.1f}ms–{_stats['append']['max']*1000:.1f}ms",
                f"",
                f"⚙️  OVERHEAD & OTHER:",
                f"   - Python/loops   : {_t_python_overhead/n_cams*1000:.1f}ms/cam (total)",
                f"   - Unaccounted    : {unaccounted_time/n_cams*1000:.1f}ms/cam (total)",
                f"",
                f"📊 TOTALS:",
                f"   - Total Time     : {total_time:.3f}s  ({n_cams} cameras)",
                f"   - Avg per cam    : {total_time/n_cams*1000:.1f}ms/cam",
            ],
            logger,
        )
        return results

    @classmethod
    def compute_ortho_index_map_moderngl(cls, ortho_camera, mesh_product, pixel_budget=None):
        """Build a downsampled face-ID index map for an OrthoCamera using ModernGL."""
        import numpy as np
        from time import perf_counter

        start = perf_counter()
        log_section("🗺️  MODERNGL ORTHO INDEX MAP RASTERIZATION", logger)

        if not ortho_camera.is_valid:
            logger.debug("   ⚠️ OrthoCamera has no valid geo metadata — aborting.")
            return {'index_map': None, 'visible_indices': np.array([], dtype=np.int32), 'scale_factor': 1.0}

        mesh = mesh_product.get_mesh()
        if mesh is None or mesh.n_cells == 0:
            logger.debug("   ⚠️ Mesh has no cells — aborting.")
            return {'index_map': None, 'visible_indices': np.array([], dtype=np.int32), 'scale_factor': 1.0}

        n_cells = mesh.n_cells
        ortho_w, ortho_h = ortho_camera.width, ortho_camera.height
        native_pixels = ortho_w * ortho_h
        scale = 1.0 if (pixel_budget is None or pixel_budget <= 0 or native_pixels <= pixel_budget) else float(np.sqrt(pixel_budget / native_pixels))
        render_w = max(1, int(round(ortho_w * scale)))
        render_h = max(1, int(round(ortho_h * scale)))
        logger.debug(f"   Ortho: {ortho_w}×{ortho_h}  →  render: {render_w}×{render_h}  (scale={scale:.4f})")
        logger.debug(f"   Mesh: {n_cells:,} cells")

        # Extract ortho camera geometry
        W, H = ortho_camera.width, ortho_camera.height
        TL = ortho_camera.pixel_to_xy_world(0, 0)
        TR = ortho_camera.pixel_to_xy_world(W-1, 0)
        BL = ortho_camera.pixel_to_xy_world(0, H-1)
        BR = ortho_camera.pixel_to_xy_world(W-1, H-1)

        if any(c is None for c in [TL, TR, BL, BR]):
            logger.warning("   ⚠️ Could not compute ortho camera corners")
            return None

        center = (TL + TR + BL + BR) * 0.25
        top_center = (TL + TR) * 0.5
        bot_center = (BL + BR) * 0.5
        vu = top_center - bot_center
        vu_len = np.linalg.norm(vu)
        view_up = vu / vu_len if vu_len > 1e-12 else np.array([0., 1., 0.])
        parallel_scale = vu_len * 0.5

        vertical_dir = ortho_camera.get_vertical_direction_world()
        bounds = mesh.bounds
        z_range = max(abs(bounds[5] - bounds[4]), 1.0)
        lift = z_range * 5.0
        cam_pos = center - vertical_dir * lift

        # Build view matrix (R, t) from camera position, focal point, and up vector
        # Forward vector: from camera to focal point
        forward = (center - cam_pos)
        forward_len = np.linalg.norm(forward)
        if forward_len < 1e-12:
            logger.warning("   ⚠️ Invalid camera geometry")
            return None
        forward = forward / forward_len

        # Right vector: cross(forward, view_up)
        right = np.cross(forward, view_up)
        right_len = np.linalg.norm(right)
        if right_len < 1e-12:
            logger.warning("   ⚠️ Invalid view-up direction")
            return None
        right = right / right_len

        # Corrected up vector: cross(right, forward)
        up = np.cross(right, forward)
        up_len = np.linalg.norm(up)
        if up_len < 1e-12:
            logger.warning("   ⚠️ Could not build view frame")
            return None
        up = up / up_len

        # Rotation matrix (column-major: [right, up, -forward])
        R = np.column_stack([right, up, -forward])
        t = -R @ cam_pos

        # Build orthographic intrinsic matrix
        # For ortho: K = [[2/width, 0, 0.5], [0, 2/height, 0.5], [0, 0, 1], [0, 0, 0]]
        # But we use the standard format with focal lengths replaced by ortho scaling
        ortho_scale_x = parallel_scale / render_w
        ortho_scale_y = parallel_scale / render_h
        K = np.array([
            [1.0 / ortho_scale_x, 0.0, 0.5],
            [0.0, 1.0 / ortho_scale_y, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ], dtype=np.float64)

        # Call batch moderngl with single ortho camera
        try:
            results = cls.compute_batch_mesh_visibility_moderngl(
                mesh_product,
                [(K, R, t, render_w, render_h)],
                compute_depth_map=False,
                compute_visible_indices=False,
                pixel_budget=None,
                upsample_to_native=False,
            )
            index_map, visible_indices, _, _ = (results[0].get(k) for k in ['index_map', 'visible_indices', 'depth_map', 'inverted_index'])

            # Flip horizontally (ModernGL convention)
            index_map = np.fliplr(index_map)

            total = perf_counter() - start
            cov = np.sum(index_map >= 0) / (render_w * render_h) * 100
            logger.debug(f"   ✅ Done in {total:.2f}s — {len(visible_indices):,} visible faces, {cov:.1f}% coverage")
            logger.debug(f"{'='*50}\n")

            return cls._normalize_result_dict({
                'index_map': index_map,
                'visible_indices': visible_indices,
                'depth_map': None,
                'inverted_index': None,
                'scale_factor': scale,
            }, compute_depth_map=False)
        except Exception as e:
            logger.warning(f"   ⚠️ ModernGL ortho rasterization failed ({e})")
            return None

    @classmethod
    def compute_point_cloud_visibility(cls, points_world, K, R, t, width, height,
                                       point_ids=None, compute_depth_map=True):
        """Compute visibility for a cloud of points given single camera parameters."""
        perf_counter = time.perf_counter
        start_time = perf_counter()
        log_section("👁️  POINT CLOUD VISIBILITY COMPUTATION", logger)
        if point_ids is None:
            point_ids = np.arange(len(points_world), dtype=np.int32)
        logger.debug(f"   Points: {len(points_world):,} | Render: {width}x{height} pixels")

        if HAS_TORCH:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.debug(f"   Using {device.upper()} backend")
            compute_start = perf_counter()
            result = cls._compute_torch(points_world, point_ids, K, R, t,
                                        width, height, device, compute_depth_map=compute_depth_map)
            compute_time = perf_counter() - compute_start
        else:
            logger.debug("   Using NUMPY backend (PyTorch not available)")
            compute_start = perf_counter()
            result = cls._compute_numpy(points_world, point_ids, K, R, t,
                                        width, height, compute_depth_map=compute_depth_map)
            compute_time = perf_counter() - compute_start

        result = cls._normalize_result_dict(result, compute_depth_map)
        total_time = perf_counter() - start_time
        visible_count = len(result['visible_indices'])
        coverage = np.sum(result['index_map'] >= 0) / (width * height) * 100
        log_summary("Point Cloud Visibility",
                    [f"   - Computation (Z-buffer): {compute_time:.4f}s",
                     f"   - Total Time            : {total_time:.4f}s",
                     f"   - Result: {visible_count:,} visible points, {coverage:.1f}% pixel coverage"],
                    logger)
        return result

    @classmethod
    def compute_batch_point_cloud_visibility(cls, points_world, camera_params_list,
                                             point_ids=None, compute_depth_map=True):
        perf_counter = time.perf_counter
        start_time = perf_counter()
        log_section("👁️  BATCH POINT CLOUD VISIBILITY COMPUTATION (STREAMING MODE)", logger)
        N_total = len(points_world)
        if point_ids is None:
            point_ids = np.arange(N_total, dtype=np.int32)
        logger.debug(f"   Points: {N_total:,} | Cameras: {len(camera_params_list)}")
        if not HAS_TORCH:
            pass
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.debug(f"   Using {device.upper()} backend")
        M = len(camera_params_list)
        results = []
        CHUNK_SIZE = 100_000_000

        for i in range(M):
            cam_start = perf_counter()
            K_np, R_np, t_np, width, height = camera_params_list[i]
            K = torch.as_tensor(K_np, dtype=torch.float32, device=device)
            R = torch.as_tensor(R_np, dtype=torch.float32, device=device)
            t = torch.as_tensor(t_np, dtype=torch.float32, device=device)
            global_z_buffer  = torch.full((height*width,), float('inf'), device=device, dtype=torch.float32)
            global_index_map = torch.full((height*width,), -1, device=device, dtype=torch.int32)
            local_z_buffer   = torch.empty((height*width,), device=device, dtype=torch.float32)
            local_index_map  = torch.empty((height*width,), device=device, dtype=torch.int32)
            logger.debug(f'   -> Processing {cam_label(i+1)}/{M} in chunks...')

            for start_idx in range(0, N_total, CHUNK_SIZE):
                end_idx   = min(start_idx + CHUNK_SIZE, N_total)
                chunk_pts = torch.as_tensor(points_world[start_idx:end_idx], dtype=torch.float32, device=device)
                chunk_ids = torch.as_tensor(point_ids[start_idx:end_idx], dtype=torch.int32, device=device)
                points_cam = chunk_pts @ R.T + t
                x, y, z = points_cam[:,0], points_cam[:,1], points_cam[:,2]
                u = K[0,0]*x/z + K[0,2]; v = K[1,1]*y/z + K[1,2]
                u_idx, v_idx = u.round().long(), v.round().long()
                valid_mask = (u_idx>=0)&(u_idx<width)&(v_idx>=0)&(v_idx<height)&(z>0)
                valid_u, valid_v, valid_z, valid_ids = u_idx[valid_mask], v_idx[valid_mask], z[valid_mask], chunk_ids[valid_mask]
                if valid_ids.numel() == 0:
                    continue
                flat_indices = valid_v * width + valid_u
                local_z_buffer.fill_(float('inf'))
                local_z_buffer.scatter_reduce_(0, flat_indices, valid_z, reduce='amin', include_self=True)
                is_closest = torch.abs(valid_z - local_z_buffer[flat_indices]) < 0.0001
                local_index_map.fill_(-1)
                local_index_map[flat_indices[is_closest]] = valid_ids[is_closest]
                won_mask = local_z_buffer < global_z_buffer
                global_z_buffer[won_mask]  = local_z_buffer[won_mask]
                global_index_map[won_mask] = local_index_map[won_mask]
                del chunk_pts, chunk_ids, points_cam, x, y, z, u, v, u_idx, v_idx
                del valid_mask, valid_u, valid_v, valid_z, valid_ids, flat_indices, is_closest, won_mask

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
            results.append({'index_map': index_map_np, 'visible_indices': visible_indices.cpu().numpy(),
                             'depth_map': depth_map_np,
                             'inverted_index': VisibilityManager._build_inverted_index(index_map_np)})
            log_cam_complete(cam_label(i+1), perf_counter() - cam_start, logger)
            del K, R, t, global_z_buffer, global_index_map, visible_indices, local_z_buffer, local_index_map

        if device == 'cuda':
            torch.cuda.empty_cache()
        logger.debug(f'\n   - Total Time: {perf_counter() - start_time:.4f}s')
        return results

    @staticmethod
    def _compute_torch(points_np, ids_np, K_np, R_np, t_np, width, height,
                       device='cpu', compute_depth_map=False):
        """PyTorch-based visibility via scatter_reduce_ Z-buffering."""
        points = torch.as_tensor(points_np, dtype=torch.float32, device=device)
        p_ids  = torch.as_tensor(ids_np,    dtype=torch.int32,   device=device)
        K = torch.as_tensor(K_np, dtype=torch.float32, device=device)
        R = torch.as_tensor(R_np, dtype=torch.float32, device=device)
        t = torch.as_tensor(t_np, dtype=torch.float32, device=device)

        points_cam = points @ R.T + t
        x_cam, y_cam, z_cam = points_cam[:,0], points_cam[:,1], points_cam[:,2]
        u = K[0,0]*x_cam/z_cam + K[0,2]
        v = K[1,1]*y_cam/z_cam + K[1,2]
        u_idx, v_idx = u.round().long(), v.round().long()
        valid_mask = (u_idx>=0)&(u_idx<width)&(v_idx>=0)&(v_idx<height)&(z_cam>0)
        valid_u, valid_v, valid_z, valid_ids = u_idx[valid_mask], v_idx[valid_mask], z_cam[valid_mask], p_ids[valid_mask]

        if valid_ids.numel() == 0:
            return VisibilityManager._normalize_result_dict({
                'index_map':      np.full((height, width), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32),
                'depth_map':      np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None,
                'inverted_index': None,
            }, compute_depth_map)

        flat_indices = valid_v * width + valid_u
        z_buffer = torch.full((height*width,), float('inf'), device=device, dtype=torch.float32)
        try:
            z_buffer.scatter_reduce_(0, flat_indices, valid_z, reduce='amin', include_self=True)
        except AttributeError:
            warnings.warn('PyTorch version too old for scatter_reduce_. Falling back to NumPy.')
            return VisibilityManager._compute_numpy(points_np, ids_np, K_np, R_np, t_np, width, height)

        is_closest = torch.abs(valid_z - z_buffer[flat_indices]) < 0.0001
        final_pixel_indices = flat_indices[is_closest]
        final_ids           = valid_ids[is_closest]

        index_map_tensor = torch.full((height*width,), -1, device=device, dtype=torch.int32)
        index_map_tensor[final_pixel_indices] = final_ids
        index_map_np    = index_map_tensor.view(height, width).cpu().numpy()
        visible_indices = torch.unique(final_ids, sorted=True)

        if compute_depth_map:
            try:
                z_buffer[z_buffer == float('inf')] = float('nan')
                depth_map_np = z_buffer.view(height, width).cpu().numpy()
            except Exception:
                depth_map_np = np.full((height, width), np.nan, dtype=np.float32)
        else:
            depth_map_np = None

        if str(device) == 'cuda':
            torch.cuda.empty_cache()

        return VisibilityManager._normalize_result_dict({
            'index_map':      index_map_np,
            'visible_indices': visible_indices.cpu().numpy(),
            'depth_map':      depth_map_np,
            'inverted_index': None,
        }, compute_depth_map)

    @staticmethod
    def _compute_numpy(points, ids, K, R, t, width, height, compute_depth_map=False):
        """CPU-based visibility via sort-by-depth Z-buffering (fallback)."""
        points_cam = points @ R.T + t
        x_cam, y_cam, z_cam = points_cam[:,0], points_cam[:,1], points_cam[:,2]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = K[0,0]*x_cam/z_cam + K[0,2]
            v = K[1,1]*y_cam/z_cam + K[1,2]
        u_idx = np.rint(u).astype(np.int32)
        v_idx = np.rint(v).astype(np.int32)
        valid_mask = (u_idx>=0)&(u_idx<width)&(v_idx>=0)&(v_idx<height)&(z_cam>0)
        u_valid, v_valid, z_valid, id_valid = u_idx[valid_mask], v_idx[valid_mask], z_cam[valid_mask], ids[valid_mask]

        if len(id_valid) == 0:
            return VisibilityManager._normalize_result_dict({
                'index_map':      np.full((height, width), -1, dtype=np.int32),
                'visible_indices': np.array([], dtype=np.int32),
                'depth_map':      np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None,
                'inverted_index': None,
            }, compute_depth_map)

        sort_order = np.argsort(z_valid)[::-1]
        index_map = np.full((height, width), -1, dtype=np.int32)
        depth_map = np.full((height, width), np.nan, dtype=np.float32) if compute_depth_map else None
        index_map[v_valid[sort_order], u_valid[sort_order]] = id_valid[sort_order]
        if compute_depth_map:
            depth_map[v_valid[sort_order], u_valid[sort_order]] = z_valid[sort_order].astype(np.float32)
        visible_indices = np.unique(index_map[index_map != -1])
        return VisibilityManager._normalize_result_dict({
            'index_map':      index_map,
            'visible_indices': visible_indices,
            'depth_map':      depth_map,
            'inverted_index': None,
        }, compute_depth_map)
