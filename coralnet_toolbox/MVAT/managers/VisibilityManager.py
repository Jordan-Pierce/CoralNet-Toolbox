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
            results = cls.compute_batch_mesh_visibility_moderngl(
                primary_target, [(K, R, t, width, height)],
                compute_depth_map=compute_depth_map,
                compute_visible_indices=True,
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

        prog_int = ctx.program(vertex_shader=_MGL_VERT, fragment_shader=_MGL_FRAG_INT)
        vao_int  = ctx.vertex_array(prog_int, [(vbo, '3f', 'position')], ibo)

        gl_fns = _resolve_gl_fns()

        face_centers_pt = None
        if HAS_TORCH and torch.cuda.is_available():
            face_centers_pt = torch.from_numpy(
                mesh.cell_centers().points.astype(np.float32)
            ).cuda()

        logger.info(
            f"   ✅ moderngl context ready: {n_cells:,} faces (32-bit int rendering)"
        )

        return {
            'ctx':             ctx,
            'prog_int':        prog_int,
            'vao_int':         vao_int,
            'n_cells':         n_cells,
            'gl_fns':          gl_fns,
            'pixel_budget':    pixel_budget,
            'face_centers_pt': face_centers_pt,
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
    ) -> list:
        """GPU rasterization via moderngl with zero-PCIe CUDA-GL framebuffer readback.

        Each result dict includes 'index_map_gpu' — a CUDA int32 tensor that stays
        on the GPU so callers (e.g. SAM) can do mask→face-ID lookups without a
        PCIe upload.
        """
        import time
        perf_counter = time.perf_counter

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

        mesh        = mesh_product.get_mesh()
        mesh_bounds = mesh.bounds

        def _get_fbo(w, h):
            key = (w, h)
            if key not in fbo_cache:
                fbo_cache[key] = ctx.framebuffer(
                    color_attachments=[ctx.texture((w, h), 1, dtype='i4')],
                    depth_attachment=ctx.depth_texture((w, h)),
                )
            return fbo_cache[key]

        results = []
        _t_render_total     = 0.0
        _t_readback_total   = 0.0
        _t_decode_total     = 0.0
        camera_total_sum    = 0.0

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

            u_min = v_min = 0
            crop_w, crop_h = render_w, render_h
            if use_viewport_cropping:
                u_min, u_max, v_min, v_max, crop_status = cls._get_2d_bounding_box(
                    mesh_bounds, K_scaled, R, t, render_w, render_h
                )
                if crop_status == "OFF_SCREEN":
                    results.append(cls._normalize_result_dict({
                        'index_map':      np.full((render_h, render_w), -1, dtype=np.int32),
                        'index_map_gpu':  None,
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

            fbo = _get_fbo(crop_w, crop_h)
            fbo.use()
            ctx.viewport = (0, 0, crop_w, crop_h)
            ctx.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)

            t0 = perf_counter()
            t0_mvp = perf_counter()
            mvp = _build_mvp(K_scaled, R, t, crop_w, crop_h)
            t_mvp = perf_counter() - t0_mvp

            t0_write = perf_counter()
            prog_int['mvp'].write(mvp.tobytes())
            t_write = perf_counter() - t0_write

            t0_draw = perf_counter()
            vao_int.render()
            t_draw = perf_counter() - t0_draw

            # NOTE: Removed ctx.finish() — fbo.read() synchronizes implicitly.
            # This saves ~3.2ms per camera by avoiding redundant GPU stall.
            t_finish = 0.0

            t_render = perf_counter() - t0
            logger.info(f"      [Render] MVP: {t_mvp*1000:.2f}ms | Write: {t_write*1000:.2f}ms | Draw: {t_draw*1000:.2f}ms | Finish: {t_finish*1000:.2f}ms")

            t0 = perf_counter()
            raw = fbo.read(components=1, dtype='i4')
            shot_int32 = np.frombuffer(raw, dtype=np.int32).reshape(crop_h, crop_w)[::-1].copy()
            t_readback = perf_counter() - t0

            t0 = perf_counter()
            crop_index_tensor = shot_int32 - 1

            # DIAGNOSTIC: Time the GPU→CPU transfer separately from the math
            t0_transfer = perf_counter()
            crop_index_map = crop_index_tensor.cpu().numpy() if hasattr(crop_index_tensor, 'cpu') else crop_index_tensor
            t_transfer = perf_counter() - t0_transfer

            # DIAGNOSTIC: Time the CPU math separately
            t0_math = perf_counter()

            # Compute visible indices only if requested (skip for ~47ms savings in batch mode)
            if compute_visible_indices:
                t0_mask = perf_counter()
                valid_indices = crop_index_map[crop_index_map >= 0]
                t_mask = perf_counter() - t0_mask

                t0_unique = perf_counter()
                if valid_indices.size > 0:
                    visible_indices = np.where(np.bincount(valid_indices) > 0)[0].astype(np.int32)
                else:
                    visible_indices = np.array([], dtype=np.int32)
                t_unique = perf_counter() - t0_unique
            else:
                visible_indices = np.array([], dtype=np.int32)
                t_mask = 0.0
                t_unique = 0.0

            t_math = perf_counter() - t0_math

            t_decode = perf_counter() - t0
            # Log the split for diagnostics
            logger.info(f"      [Decode Split] Transfer: {t_transfer*1000:.1f}ms | Mask: {t_mask*1000:.1f}ms | Unique: {t_unique*1000:.1f}ms | Total Math: {t_math*1000:.1f}ms")

            crop_depth_map = None
            crop_index_tensor_gpu = None
            if compute_depth_map:
                t0_depth = perf_counter()

                # 1. Read the raw Z-buffer directly from the ModernGL FBO (attachment=-1 is depth)
                depth_raw = fbo.read(attachment=-1, dtype='f4')
                depth_buffer = np.frombuffer(depth_raw, dtype=np.float32).reshape(crop_h, crop_w)[::-1]

                # 2. Linearize OpenGL depth [0, 1] to CV depth (meters)
                # Using the exact near/far clipping planes defined in _build_mvp
                near, far = 0.01, 100000.0
                z_ndc = 2.0 * depth_buffer - 1.0

                with np.errstate(divide='ignore', invalid='ignore'):
                    linear_depth = (2.0 * near * far) / (far + near - z_ndc * (far - near))

                # 3. Mask out the background (where the camera sees nothing)
                linear_depth[crop_index_map == -1] = np.nan
                crop_depth_map = linear_depth.astype(np.float32)

                t_depth = perf_counter() - t0_depth
                logger.info(f"      [Depth Z-Buffer Read] {t_depth*1000:.2f}ms")

                if HAS_TORCH and torch.cuda.is_available():
                    crop_index_tensor_gpu = torch.from_numpy(crop_index_map).int().cuda()

            # Paste crop back into full canvas
            full_index_map = np.full((render_h, render_w), -1, dtype=np.int32)
            full_index_map[v_min:v_min + crop_h, u_min:u_min + crop_w] = crop_index_map

            # GPU tensor — stays on CUDA for downstream consumers (e.g. SAM lookup)
            full_index_map_gpu = None
            if HAS_TORCH and torch.cuda.is_available():
                full_index_map_gpu = torch.full(
                    (render_h, render_w), -1, dtype=torch.int32, device='cuda'
                )
                if crop_index_tensor_gpu is None:
                    crop_index_tensor_gpu = torch.from_numpy(crop_index_map).int().cuda()
                full_index_map_gpu[v_min:v_min + crop_h, u_min:u_min + crop_w] = crop_index_tensor_gpu

            full_depth_map = None
            if compute_depth_map:
                full_depth_map = np.full((render_h, render_w), np.nan, dtype=np.float32)
                if crop_depth_map is not None:
                    full_depth_map[v_min:v_min + crop_h, u_min:u_min + crop_w] = crop_depth_map

            result_scale = dynamic_scale
            if upsample_to_native and dynamic_scale < 1.0:
                import cv2 as _cv2
                full_index_map = _cv2.resize(full_index_map, (width, height), interpolation=_cv2.INTER_NEAREST)
                if full_depth_map is not None:
                    full_depth_map = _cv2.resize(full_depth_map, (width, height), interpolation=_cv2.INTER_NEAREST)
                if full_index_map_gpu is not None:
                    full_index_map_gpu = full_index_map_gpu.unsqueeze(0).unsqueeze(0).float()
                    full_index_map_gpu = torch.nn.functional.interpolate(
                        full_index_map_gpu, size=(height, width), mode='nearest'
                    ).squeeze().to(torch.int32)
                result_scale = 1.0

            results.append(cls._normalize_result_dict({
                'index_map':       full_index_map,
                'index_map_gpu':   full_index_map_gpu,
                'visible_indices': visible_indices,
                'depth_map':       full_depth_map,
                'inverted_index':  None,
                'scale_factor':    result_scale,
            }, compute_depth_map))

            cam_time = perf_counter() - cam_start
            camera_total_sum  += cam_time
            _t_render_total   += t_render
            _t_readback_total += t_readback
            _t_decode_total   += t_decode

            log_cam_breakdown(
                cam_label(camera_index_offset + i + 1),
                cam_time, 0.0, t_render, t_readback, t_decode, 0.0, 0.0, 0.0, logger
            )

        n_cams = max(1, len(camera_params_list))
        total_time = perf_counter() - start_time
        readback_mode = "CPU readback"
        log_summary(
            f"moderngl Batch Rasterization ({readback_mode})",
            [
                f"   - GPU Rasterize  : {_t_render_total:.3f}s total  ({_t_render_total/n_cams*1000:.1f}ms/cam)",
                f"   - Readback       : {_t_readback_total:.3f}s total  ({_t_readback_total/n_cams*1000:.1f}ms/cam)",
                f"   - Decode         : {_t_decode_total:.3f}s total  ({_t_decode_total/n_cams*1000:.1f}ms/cam)",
                f"   - Total Time     : {total_time:.3f}s  (Avg: {total_time/n_cams*1000:.1f}ms/cam)",
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
            logger.info("   ⚠️ OrthoCamera has no valid geo metadata — aborting.")
            return {'index_map': None, 'visible_indices': np.array([], dtype=np.int32), 'scale_factor': 1.0}

        mesh = mesh_product.get_mesh()
        if mesh is None or mesh.n_cells == 0:
            logger.info("   ⚠️ Mesh has no cells — aborting.")
            return {'index_map': None, 'visible_indices': np.array([], dtype=np.int32), 'scale_factor': 1.0}

        n_cells = mesh.n_cells
        ortho_w, ortho_h = ortho_camera.width, ortho_camera.height
        native_pixels = ortho_w * ortho_h
        scale = 1.0 if (pixel_budget is None or pixel_budget <= 0 or native_pixels <= pixel_budget) else float(np.sqrt(pixel_budget / native_pixels))
        render_w = max(1, int(round(ortho_w * scale)))
        render_h = max(1, int(round(ortho_h * scale)))
        logger.info(f"   Ortho: {ortho_w}×{ortho_h}  →  render: {render_w}×{render_h}  (scale={scale:.4f})")
        logger.info(f"   Mesh: {n_cells:,} cells")

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
            logger.info(f"   ✅ Done in {total:.2f}s — {len(visible_indices):,} visible faces, {cov:.1f}% coverage")
            logger.info(f"{'='*50}\n")

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
        logger.info(f"   Points: {len(points_world):,} | Render: {width}x{height} pixels")

        if HAS_TORCH:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"   Using {device.upper()} backend")
            compute_start = perf_counter()
            result = cls._compute_torch(points_world, point_ids, K, R, t,
                                        width, height, device, compute_depth_map=compute_depth_map)
            compute_time = perf_counter() - compute_start
        else:
            logger.info("   Using NUMPY backend (PyTorch not available)")
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
        logger.info(f"   Points: {N_total:,} | Cameras: {len(camera_params_list)}")
        if not HAS_TORCH:
            pass
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"   Using {device.upper()} backend")
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
            logger.info(f'   -> Processing {cam_label(i+1)}/{M} in chunks...')

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
        logger.info(f'\n   - Total Time: {perf_counter() - start_time:.4f}s')
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
