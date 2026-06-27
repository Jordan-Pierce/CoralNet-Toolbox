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
from coralnet_toolbox.MVAT.shaders import VERT_POINT as _MGL_VERT_POINT
from coralnet_toolbox.MVAT.shaders import FRAG_POINT_ID_INT as _MGL_FRAG_POINT_INT
from coralnet_toolbox.MVAT.shaders import COVERAGE_CS as _MGL_COVERAGE_CS
from coralnet_toolbox.MVAT.shaders import WARP_VERT as _MGL_WARP_VERT
from coralnet_toolbox.MVAT.shaders import WARP_FRAG as _MGL_WARP_FRAG
from coralnet_toolbox.MVAT.shaders.gpu_interop import _build_mvp

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
        """Fast depth reconstruction from cached visible indices.

        The cost is the per-pixel gather over the full-resolution index map (tens
        of millions of pixels). Two things keep it cheap:

        * The gather runs on the GPU when CUDA is available — only the int32 index
          map crosses PCIe up and the float16 depth crosses back; the lookup table
          and the 40M+ element gather stay on the device (~15x faster than NumPy).
        * Background pixels (-1) index the trailing NaN slot directly via negative
          indexing, so there is no separate int64 conversion / copy / mask pass.

        Output is float16 camera-space depth (NaN = background), matching the prior
        CPU implementation's values exactly. Returns None when inputs are missing.
        """
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

        element_count = int(coords.shape[0])
        if int(visible_ids.max()) >= element_count:
            visible_ids = visible_ids[visible_ids < element_count]
            if visible_ids.size == 0:
                return None

        R = np.asarray(camera.R, dtype=np.float32)
        t = np.asarray(camera.t, dtype=np.float32)

        # GPU path: keep the lookup-table build and the full-frame gather on the
        # device so only the index map (up) and depth (down) cross PCIe.
        if HAS_TORCH and torch.cuda.is_available():
            try:
                coords_t = getattr(scene_product, '_cached_face_centers_pt', None)
                if coords_t is None or int(coords_t.shape[0]) != element_count:
                    coords_t = torch.as_tensor(coords, device='cuda')
                    try:
                        scene_product._cached_face_centers_pt = coords_t
                    except Exception:
                        pass
                coords_t = coords_t.to(device='cuda', dtype=torch.float32)
                vis_t = torch.as_tensor(visible_ids, dtype=torch.long, device='cuda')
                R2 = torch.as_tensor(R[2, :], dtype=torch.float32, device='cuda')
                # z and the lookup table stay float32 (matches the prior CPU values
                # bit-for-bit); only the gather output is cast to float16.
                z = coords_t.index_select(0, vis_t) @ R2 + float(t[2])
                lookup = torch.full((element_count + 1,), float('nan'),
                                    dtype=torch.float32, device='cuda')
                lookup[vis_t] = z
                # int32 → long device-side; -1 stays -1 and gathers the NaN slot.
                idx_t = torch.as_tensor(
                    np.ascontiguousarray(index_map, dtype=np.int32), device='cuda'
                ).to(torch.long)
                return lookup[idx_t].to(torch.float16).cpu().numpy()
            except Exception:
                pass  # fall through to CPU

        # CPU fallback: gather then cast. -1 indexes the trailing NaN slot, so the
        # int64 conversion / copy / mask of the old path are gone (5 passes → 2).
        z_values = coords[visible_ids] @ R[2, :] + t[2]
        lookup = np.full(element_count + 1, np.nan, dtype=np.float32)
        lookup[visible_ids] = z_values.astype(np.float32, copy=False)
        return lookup[index_map].astype(np.float16, copy=False)

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
            # Point clouds rasterize through the SAME moderngl GL_POINTS path as
            # meshes (see setup_batch_point_moderngl_context). GaussianSplattingProduct
            # also reports element_type 'point' but is routed to its own
            # solid-ellipsoid splat path (setup_batch_splat_moderngl_context).
            from coralnet_toolbox.MVAT.core.Products import (
                PointCloudProduct,
                GaussianSplattingProduct,
            )
            if isinstance(primary_target, GaussianSplattingProduct):
                # 3D Gaussian Splats: solid-ellipsoid index map via a dedicated
                # ModernGL program that reuses pyvista_gs' gau_vert.glsl.
                mgl_ctx = cls.setup_batch_splat_moderngl_context(
                    primary_target, width, height,
                )
                try:
                    results = cls.compute_batch_visibility_moderngl(
                        primary_target, [(K, R, t, width, height)],
                        compute_depth_map=compute_depth_map,
                        compute_visible_indices=True,
                        pixel_budget=None,
                        mgl_context=mgl_ctx,
                    )
                    result = results[0]
                    result['element_type'] = 'point'
                    return cls._normalize_result_dict(result, compute_depth_map)
                finally:
                    try:
                        for buf in mgl_ctx.get('buffers_to_release', []):
                            buf.release()
                        for fbo in mgl_ctx.get('_fbo_cache', {}).values():
                            fbo.release()
                        mgl_ctx['ctx'].release()
                    except Exception:
                        pass
            elif isinstance(primary_target, PointCloudProduct):
                # Interactive default: splat radius follows the product's display
                # point_size, square sprites. Batch caching overrides these via the dialog.
                splat_radius = float(getattr(primary_target, 'point_size', 1) or 1)
                mgl_ctx = cls.setup_batch_point_moderngl_context(
                    primary_target, None, width, height,
                    splat_radius=splat_radius, splat_round=False,
                )
                try:
                    results = cls.compute_batch_visibility_moderngl(
                        primary_target, [(K, R, t, width, height)],
                        compute_depth_map=compute_depth_map,
                        compute_visible_indices=True,
                        pixel_budget=None,
                        mgl_context=mgl_ctx,
                    )
                    result = results[0]
                    result['element_type'] = 'point'
                    return cls._normalize_result_dict(result, compute_depth_map)
                finally:
                    try:
                        for fbo in mgl_ctx.get('_fbo_cache', {}).values():
                            fbo.release()
                        mgl_ctx['ctx'].release()
                    except Exception:
                        pass

        elif element_type == 'face':
            # Mesh visibility with ModernGL (VTK removed in Phase 3)
            # Keep visible_indices=False to skip expensive computation in batch paths.
            # compute_depth_map is controlled by caller (True for interactive SAM, False for batch).
            results = cls.compute_batch_visibility_moderngl(
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
    # moderngl batch rasterizer  (CPU framebuffer readback)
    # =========================================================================

    # Clipping planes shared by _build_mvp (mesh/point) and the per-camera splat
    # projection. Kept here so the in-shader depth linearization (u_near/u_far) and
    # the matrix builders stay in lockstep.
    GL_NEAR = 0.01
    GL_FAR = 100000.0

    # Max number of distinct-resolution FBOs retained in a context's _fbo_cache.
    # Viewport cropping produces a different (crop_w, crop_h) per camera, so on a
    # warm (persistent) context the cache would otherwise grow one FBO per crop
    # size for the lifetime of the scene. FIFO-evict the oldest beyond this cap so
    # resident VRAM stays bounded. With a per-batch (released) context this is a
    # no-op in practice — sizes within one batch rarely exceed the cap.
    _FBO_CACHE_CAP = 12

    @staticmethod
    def _set_depth_uniforms(prog):
        """Upload the near/far clipping planes used by the in-shader depth
        linearization. No-op if the program was built without the depth output
        (older shader), so callers don't need to special-case that."""
        if 'u_near' in prog:
            prog['u_near'].value = VisibilityManager.GL_NEAR
        if 'u_far' in prog:
            prog['u_far'].value = VisibilityManager.GL_FAR

    @staticmethod
    def _setup_coverage(ctx, n_cells):
        """Best-effort build of the visible-element coverage compute pass.

        Returns ``(cov_prog, cov_buffer)`` on success or ``(None, None)`` when the
        driver/context cannot run compute shaders (e.g. legacy macOS GL 4.1). The
        render loop falls back to np.bincount in that case. Compilation is the only
        reliable capability probe: standalone contexts report GL 3.3 even on drivers
        that fully support compute (see COVERAGE_CS docstring)."""
        # Coverage stores one uint per element, so the buffer + per-camera readback
        # scale with element count. Beyond ~32M elements that readback (and the
        # np.flatnonzero over it) costs more than np.unique over the visible pixels,
        # so disable it and let the caller fall back to np.unique.
        if int(n_cells) > 32_000_000:
            logger.debug(f"   ℹ️ coverage disabled for {n_cells:,} elements; using np.unique fallback")
            return None, None
        try:
            cov_prog = ctx.compute_shader(_MGL_COVERAGE_CS)
            # One uint slot per element; reused across cameras (cleared per frame).
            cov_buffer = ctx.buffer(reserve=max(1, int(n_cells)) * 4)
            return cov_prog, cov_buffer
        except Exception as e:
            logger.debug(f"   ℹ️ coverage compute unavailable, using np.unique fallback: {e}")
            return None, None

    @staticmethod
    def _get_interop(mgl_context):
        """Lazily set up CUDA-GL interop for zero-PCIe FBO readback (NVIDIA only).

        Returns ``{gl, cudart, cache}`` when torch.cuda + the GL/cudart function
        pointers all resolve, else ``None`` (caller falls back to the portable
        moderngl PBO readback). The probe runs once per context and is cached. The
        D2D path reads the FBO into a CUDA tensor without a PCIe copy, so the index
        map and its visible-element ``torch.unique`` both stay on the GPU; only the
        final host copy crosses PCIe (via torch's faster ``.cpu()``). Must be called
        with the GL context current — it is, inside the render loop.
        """
        if '_interop' in mgl_context:
            return mgl_context['_interop']
        interop = None
        try:
            import torch
            if torch.cuda.is_available():
                import ctypes
                from coralnet_toolbox.MVAT.shaders.gpu_interop import _resolve_gl_fns, _load_cudart
                gl = _resolve_gl_fns()
                cudart = _load_cudart()
                # cudaError_t (int) return so _pbo_cuda_readback's `if err:` checks fire.
                for _fn in ('cudaGraphicsGLRegisterBuffer', 'cudaGraphicsMapResources',
                            'cudaGraphicsResourceGetMappedPointer', 'cudaMemcpy',
                            'cudaGraphicsUnmapResources', 'cudaGraphicsUnregisterResource'):
                    try:
                        getattr(cudart, _fn).restype = ctypes.c_int
                    except Exception:
                        pass
                interop = {'gl': gl, 'cudart': cudart, 'cache': {}}
                logger.debug("   ✅ CUDA-GL interop ready (zero-PCIe readback)")
        except Exception as e:
            logger.debug(f"   ℹ️ CUDA-GL interop unavailable, using moderngl readback: {e}")
            interop = None
        mgl_context['_interop'] = interop
        return interop

    @classmethod
    def _warp_fbo_gl(cls, mgl_context, src_fbo, in_w, in_h, map_x, map_y,
                     out_w, out_h, have_depth):
        """Distort the *resident* render-FBO (index + linear depth) into a native-res
        warped FBO via a GL fullscreen pass — no PCIe round-trip.

        A standalone GL warp (upload → warp → readback) is PCIe-bound and actually
        slower than cv2.remap; the win only materializes by warping the textures that
        are already on the GPU after rasterization, so distortion costs only the
        existing single readback. ``(map_x, map_y)`` are ``Raster._map_x/_map_y``
        (native-resolution source-pixel coords); the source index is 1-based so the
        normal `-1` decode on readback yields the background. Matches
        grid_sample(nearest, align_corners=True) + border fill (verified bit-exact
        vs cv2.remap and grid_sample).

        Returns the warped FBO (color 0 = 1-based R32I index, color 1 = R32F depth),
        cached per output size on ``mgl_context``. The caller then reads it back with
        the same code path as a freshly-rendered native FBO. GL is single-threaded —
        call this on the thread that owns the context.
        """
        import moderngl
        ctx = mgl_context['ctx']

        prog = mgl_context.get('_warp_prog')
        if prog is None:
            prog = ctx.program(vertex_shader=_MGL_WARP_VERT,
                               fragment_shader=_MGL_WARP_FRAG)
            mgl_context['_warp_prog'] = prog
            mgl_context['_warp_vao'] = ctx.vertex_array(prog, [])  # attribute-less fullscreen tri
        vao = mgl_context['_warp_vao']

        cache = mgl_context.setdefault('_warp_res', {})
        entry = cache.get((out_w, out_h))
        if entry is None:
            map_tex = ctx.texture((out_w, out_h), 2, dtype='f4')
            map_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            out_idx = ctx.texture((out_w, out_h), 1, dtype='i4')
            out_dep = ctx.texture((out_w, out_h), 1, dtype='f4')
            warp_fbo = ctx.framebuffer(color_attachments=[out_idx, out_dep])
            entry = {'map': map_tex, 'fbo': warp_fbo, 'map_id': None}
            cache[(out_w, out_h)] = entry
        map_tex, warp_fbo = entry['map'], entry['fbo']

        # Warp maps are shared per-lens (Raster caches them), so re-upload only when
        # the array object actually changes between cameras of different lenses.
        if entry['map_id'] != id(map_x):
            map_arr = np.ascontiguousarray(
                np.stack([np.asarray(map_x, np.float32), np.asarray(map_y, np.float32)], axis=-1)
            )
            map_tex.write(map_arr.tobytes())
            entry['map_id'] = id(map_x)

        warp_fbo.use()
        ctx.viewport = (0, 0, out_w, out_h)
        prog['srcIdx'].value = 0
        prog['srcDepth'].value = 1
        prog['mapTex'].value = 2
        prog['in_size'].value = (in_w, in_h)
        prog['out_size'].value = (out_w, out_h)
        prog['have_depth'].value = 1 if have_depth else 0
        src_fbo.color_attachments[0].use(0)   # isampler2D ← R32I index (texelFetch)
        src_fbo.color_attachments[1].use(1)   # sampler2D  ← R32F depth (texelFetch)
        map_tex.use(2)
        vao.render(mode=moderngl.TRIANGLES, vertices=3)
        return warp_fbo

    @classmethod
    def setup_batch_mesh_moderngl_context(cls, mesh_product, pixel_budget,
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

        # Clipping planes for the in-shader depth linearization (match _build_mvp).
        cls._set_depth_uniforms(prog_int)

        # GPU visible-element coverage pass (falls back to bincount if unavailable).
        cov_prog, cov_buffer = cls._setup_coverage(ctx, n_cells)

        face_centers_pt = None
        if HAS_TORCH and torch.cuda.is_available():
            face_centers_pt = torch.from_numpy(
                mesh.cell_centers().points.astype(np.float32)
            ).cuda()

        logger.debug(f"   ✅ moderngl context ready: {n_cells:,} faces (32-bit int encoding)")

        return {
            'ctx':             ctx,
            'prog_int':        prog_int,
            'vao_int':         vao_int,
            'n_cells':         n_cells,
            'pixel_budget':    pixel_budget,
            'face_centers_pt': face_centers_pt,
            '_fbo_cache':      {},
            'cov_prog':        cov_prog,
            'cov_buffer':      cov_buffer,
            # Geometry source descriptors consumed by the generic render loop so
            # the same loop drives both the mesh (TRIANGLES) and point (POINTS) paths.
            'bounds':          mesh.bounds,
            'render_mode':     moderngl.TRIANGLES,
        }

    @classmethod
    def setup_batch_point_moderngl_context(cls, point_product, pixel_budget,
                                           sample_width, sample_height,
                                           splat_radius: float = 1.0,
                                           splat_round: bool = False):
        """Create a moderngl offscreen context and upload point-cloud geometry once.

        Mirrors ``setup_batch_mesh_moderngl_context`` but for a GL_POINTS draw: point
        IDs are ``gl_VertexID`` and each point is rasterized as a sprite of
        ``splat_radius`` render-resolution pixels (square, or a disc when
        ``splat_round``). The returned context is consumed by the SAME
        ``compute_batch_visibility_moderngl`` render loop as meshes.
        """
        import moderngl

        ctx = moderngl.create_context(standalone=True)
        ctx.enable(moderngl.DEPTH_TEST)
        # gl_PointSize in the vertex shader is only honored when this is enabled.
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        points = np.asarray(point_product.get_points_array(), dtype=np.float32)
        n_points = int(points.shape[0])

        vbo = ctx.buffer(points.tobytes())

        # Single-channel R32I point-ID target (same encoding as the mesh path).
        prog_int = ctx.program(vertex_shader=_MGL_VERT_POINT,
                               fragment_shader=_MGL_FRAG_POINT_INT)
        vao_int  = ctx.vertex_array(prog_int, [(vbo, '3f', 'position')])

        # Splat uniforms are constant across cameras (radius is in render pixels),
        # so set them once here rather than per-frame in the render loop.
        prog_int['point_size'].value = float(max(1.0, splat_radius))
        prog_int['splat_round'].value = 1 if splat_round else 0

        # Clipping planes for the in-shader depth linearization (match _build_mvp).
        cls._set_depth_uniforms(prog_int)

        # GPU visible-element coverage pass (falls back to bincount if unavailable).
        cov_prog, cov_buffer = cls._setup_coverage(ctx, n_points)

        # Points are their own element centers (used by depth reconstruction).
        face_centers_pt = None
        if HAS_TORCH and torch.cuda.is_available():
            face_centers_pt = torch.from_numpy(points).cuda()

        logger.debug(
            f"   ✅ moderngl point context ready: {n_points:,} points "
            f"(splat={splat_radius}px {'round' if splat_round else 'square'})"
        )

        return {
            'ctx':             ctx,
            'prog_int':        prog_int,
            'vao_int':         vao_int,
            'n_cells':         n_points,
            'pixel_budget':    pixel_budget,
            'face_centers_pt': face_centers_pt,
            '_fbo_cache':      {},
            'cov_prog':        cov_prog,
            'cov_buffer':      cov_buffer,
            'bounds':          point_product.get_bounds(),
            'render_mode':     moderngl.POINTS,
        }

    @classmethod
    def setup_batch_splat_moderngl_context(cls, splat_product, sample_width, sample_height):
        """Create a ModernGL context for instanced 3D Gaussian Splats.

        Reuses pyvista_gs' ``gau_vert.glsl`` (which projects each splat's 3D
        covariance into a screen-space quad) but pairs it with a hard-cutoff
        integer Splat-ID fragment shader (``FRAG_SPLAT_ID_INT``). DEPTH_TEST is
        enabled so the nearest opaque ellipsoid wins per pixel without any CPU
        depth sorting — the per-camera ``gi`` order is therefore irrelevant and
        left as the identity ``arange``.

        ``gau_vert.glsl`` is patched in-memory to forward the resolved splat
        index ``boxid`` to the fragment stage via the flat varying
        ``v_splatID`` (``gl_InstanceID`` is not readable in the fragment stage).
        """
        import moderngl
        import os
        from coralnet_toolbox.MVAT.shaders import FRAG_SPLAT_ID_INT
        import pyvista_gs
        from pyvista_gs.data import GaussianData

        ctx = moderngl.create_context(standalone=True)
        # Depth test gives automatic solid occlusion without CPU sorting.
        ctx.enable(moderngl.DEPTH_TEST)

        # The GaussianActor keeps live splat data in its mesh point_data (it has
        # no standalone .gaussians); rebuild a GaussianData exactly as the actor
        # does when syncing to its renderer so the .flat() layout matches the
        # shader's std430 buffer (xyz, rot, scale, opacity, sh).
        mesh = splat_product.gaussian_actor._mesh
        opacity = np.asarray(mesh.point_data['opacity'], dtype=np.float32)
        if opacity.ndim == 1:
            opacity = opacity.reshape(-1, 1)
        gaussians = GaussianData(
            xyz=np.asarray(mesh.points, dtype=np.float32),
            rot=np.asarray(mesh.point_data['rot'], dtype=np.float32),
            scale=np.asarray(mesh.point_data['scale'], dtype=np.float32),
            opacity=opacity,
            sh=np.asarray(mesh.point_data['sh'], dtype=np.float32),
        )
        gaussian_data = gaussians.flat()
        num_gau = len(gaussians)

        # Load the upstream vertex shader and patch in a flat Splat-ID varying.
        vs_path = os.path.join(os.path.dirname(pyvista_gs.__file__), 'shaders', 'gau_vert.glsl')
        with open(vs_path, 'r', encoding='utf-8') as f:
            vs_source = f.read()
        vs_source = vs_source.replace(
            'out vec2 coordxy;  // local coordinate in quad, unit in pixel',
            'out vec2 coordxy;  // local coordinate in quad, unit in pixel\nflat out int v_splatID;',
        )
        vs_source = vs_source.replace(
            'int boxid = gi[gl_InstanceID];',
            'int boxid = gi[gl_InstanceID];\n\tv_splatID = boxid;',
        )

        prog_int = ctx.program(vertex_shader=vs_source, fragment_shader=FRAG_SPLAT_ID_INT)

        # Clipping planes for the in-shader depth linearization (match the P matrix
        # built per-camera in the render loop: near=0.01, far=100000).
        cls._set_depth_uniforms(prog_int)

        # Unit quad (two triangles) instanced once per splat.
        quad_v = np.array([-1, 1, 1, 1, 1, -1, -1, -1], dtype=np.float32).reshape(4, 2)
        quad_f = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32).reshape(2, 3)
        vbo = ctx.buffer(quad_v.tobytes())
        ibo = ctx.buffer(quad_f.tobytes())
        vao_int = ctx.vertex_array(prog_int, [(vbo, '2f', 'position')], ibo)

        # SSBOs: packed gaussian data (binding 0) + identity draw order (binding 1).
        gau_buffer = ctx.buffer(gaussian_data.tobytes())
        gau_buffer.bind_to_storage_buffer(binding=0)
        gi = np.arange(num_gau, dtype=np.int32)
        index_buffer = ctx.buffer(gi.tobytes())
        index_buffer.bind_to_storage_buffer(binding=1)

        # GPU visible-element coverage pass (falls back to bincount if unavailable).
        cov_prog, cov_buffer = cls._setup_coverage(ctx, num_gau)

        # Static uniforms (per-camera matrices are set in the render loop).
        if 'sh_dim' in prog_int:
            prog_int['sh_dim'].value = gaussians.sh_dim
        if 'scale_modifier' in prog_int:
            prog_int['scale_modifier'].value = 1.0
        if 'render_mod' in prog_int:
            prog_int['render_mod'].value = 0
        if 'crop_enabled' in prog_int:
            prog_int['crop_enabled'].value = 0
        if 'model_matrix' in prog_int:
            prog_int['model_matrix'].write(np.eye(4, dtype=np.float32).tobytes('F'))

        logger.debug(f"   ✅ moderngl splat context ready: {num_gau:,} splats (solid ellipsoids)")

        return {
            'ctx':             ctx,
            'prog_int':        prog_int,
            'vao_int':         vao_int,
            'n_cells':         num_gau,
            'pixel_budget':    None,
            'face_centers_pt': None,
            '_fbo_cache':      {},
            'bounds':          splat_product.get_bounds(),
            'render_mode':     moderngl.TRIANGLES,
            'instances':       num_gau,
            'is_splat':        True,
            'buffers_to_release': [gau_buffer, index_buffer, vbo, ibo],
        }

    @classmethod
    def compute_batch_visibility_moderngl(
        cls,
        geometry_product,
        camera_params_list: list,
        compute_depth_map: bool = True,
        compute_visible_indices: bool = True,
        pixel_budget: Optional[int] = None,
        upsample_to_native: bool = False,
        progress_callback=None,
        mgl_context: dict = None,
        camera_index_offset: int = 0,
        use_viewport_cropping: bool = True,
        warp_maps_list: Optional[list] = None,
    ) -> list:
        """GPU rasterization via moderngl with CPU framebuffer readback.

        ``warp_maps_list`` (optional, aligned with ``camera_params_list``) fuses the
        lens-distortion warp into the render: each entry is ``(map_x, map_y)``
        (``Raster._map_x/_map_y``, native-resolution source-pixel coords) or ``None``.
        For a camera with a warp map the full frame is rendered (viewport cropping is
        forced off — distortion samples across the whole frame), the still-resident
        index/depth textures are warped on the GPU, and the result is read back at
        native resolution — replacing the separate cv2.remap / grid_sample round-trip
        the caller would otherwise apply after readback.

        Despite the name, this drives BOTH meshes and point clouds: the geometry
        source (VAO, bounds, draw primitive) is read from ``mgl_context``, which is
        built by ``setup_batch_mesh_moderngl_context`` (mesh, TRIANGLES) or
        ``setup_batch_point_moderngl_context`` (point cloud, POINTS). ``geometry_product``
        is only used when ``mgl_context`` is None to auto-build a mesh context.

        Returns a list of result dicts with 'index_map', 'visible_indices', 'depth_map', etc.
        """
        import time
        import logging
        perf_counter = time.perf_counter

        # Set up file logging for detailed debug output (enabled by default)
        # To disable: set environment variable VISIBILITY_DEBUG=0
        import os
        import logging
        os.environ.setdefault('VISIBILITY_DEBUG', '1')
        if os.environ.get('VISIBILITY_DEBUG', '0') != '0':
            debug_handler = logging.FileHandler('visibility_timing_debug.log', mode='w', encoding='utf-8')
            debug_handler.setLevel(logging.DEBUG)
            debug_formatter = logging.Formatter('%(message)s')
            debug_handler.setFormatter(debug_formatter)
            logger.addHandler(debug_handler)
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
            mgl_context = cls.setup_batch_mesh_moderngl_context(
                geometry_product, pixel_budget,
                camera_params_list[0][3], camera_params_list[0][4],
            )

        ctx          = mgl_context['ctx']
        prog_int     = mgl_context['prog_int']
        vao_int      = mgl_context['vao_int']
        fbo_cache    = mgl_context['_fbo_cache']
        # Optional GPU coverage pass for visible-element extraction (None → bincount).
        cov_prog     = mgl_context.get('cov_prog')
        cov_buffer   = mgl_context.get('cov_buffer')

        # Geometry bounds + draw primitive come from the context so this loop is
        # geometry-agnostic: meshes supply TRIANGLES, point clouds supply POINTS.
        mesh_bounds = mgl_context['bounds']
        render_mode = mgl_context.get('render_mode')
        # 3DGS splats are drawn instanced (one quad per splat) and need the
        # gau_vert.glsl camera uniforms instead of a single packed MVP.
        is_splat    = mgl_context.get('is_splat', False)
        n_instances = mgl_context.get('instances', 1)

        def _get_fbo(w, h):
            key = (w, h)
            if key not in fbo_cache:
                # color 0: single-channel R32I face/point/splat-ID (always int32).
                # color 1: single-channel R32F linearized camera-space depth, written
                #   by the fragment shader so the CPU skips the z_ndc → linear math.
                # The depth *texture* is still attached for the hardware z-test.
                fbo_cache[key] = ctx.framebuffer(
                    color_attachments=[
                        ctx.texture((w, h), 1, dtype='i4'),
                        ctx.texture((w, h), 1, dtype='f4'),
                    ],
                    depth_attachment=ctx.depth_texture((w, h)),
                )
                # Bound the cache on a warm context: FIFO-evict the oldest FBO(s)
                # (and their textures) once past the cap so resident VRAM stays
                # flat across a long session of variably-cropped cameras. dicts
                # preserve insertion order, so the first key is the oldest.
                while len(fbo_cache) > cls._FBO_CACHE_CAP:
                    old_key = next(iter(fbo_cache))
                    if old_key == key:
                        break  # never evict the FBO we just created/returned
                    old_fbo = fbo_cache.pop(old_key)
                    try:
                        for _tex in old_fbo.color_attachments:
                            _tex.release()
                        if old_fbo.depth_attachment is not None:
                            old_fbo.depth_attachment.release()
                        old_fbo.release()
                    except Exception:
                        pass
            return fbo_cache[key]

        # Pre-allocate persistent PBOs for readback (sized to largest camera)
        # This avoids per-camera PBO allocation overhead; double-buffer enables
        # async readback overlap with decode of previous result
        max_pixels = max(w * h for (_,_,_,w,h) in camera_params_list)
        max_bytes = 4 * max_pixels  # int32 = 4 bytes per pixel
        pbos = [ctx.buffer(reserve=max_bytes) for _ in range(2)]

        results = []  # Results will be appended in camera order
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

            # Distortion warp map for this camera (fuses into the render below).
            wm = warp_maps_list[i] if warp_maps_list is not None else None

            dynamic_scale = _scale_for(width, height)
            render_w = max(1, int(round(width  * dynamic_scale)))
            render_h = max(1, int(round(height * dynamic_scale)))

            K_scaled       = K.copy()
            K_scaled[0, :3] *= dynamic_scale
            K_scaled[1, :3] *= dynamic_scale

            # Viewport cropping calculation.
            # Cropping is skipped for distorted cameras: the warp samples source
            # pixels across the whole undistorted frame, so the full frame must be
            # rendered (a crop would leave the out-of-crop source pixels missing).
            t0_crop = perf_counter()
            u_min = v_min = 0
            crop_w, crop_h = render_w, render_h
            if use_viewport_cropping and wm is None:
                u_min, u_max, v_min, v_max, crop_status = cls._get_2d_bounding_box(
                    mesh_bounds, K_scaled, R, t, render_w, render_h
                )
                if crop_status == "OFF_SCREEN":
                    # Honor upsample_to_native here too — this early-out skips
                    # the upsample step below, and a sub-native empty map would
                    # otherwise be cached and indexed with native pixel coords.
                    if upsample_to_native:
                        empty_h, empty_w, empty_scale = height, width, 1.0
                    else:
                        empty_h, empty_w, empty_scale = render_h, render_w, dynamic_scale
                    results.append(cls._normalize_result_dict({
                        'index_map':      np.full((empty_h, empty_w), -1, dtype=np.int32),
                        'visible_indices': np.array([], dtype=np.int32),
                        'depth_map':      np.full((empty_h, empty_w), np.nan, dtype=np.float32) if compute_depth_map else None,
                        'inverted_index': None,
                        'scale_factor':   empty_scale,
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
            if is_splat:
                # 3DGS path: feed gau_vert.glsl the separate view/projection
                # matrices, camera position and focal/tan-fov it expects.
                # OpenCV (y-down, +z forward) → OpenGL camera convention.
                V = np.eye(4, dtype=np.float32)
                V[:3, :3] = R
                V[:3, 3] = t
                flip_yz = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
                V = flip_yz @ V

                fx, fy = float(K_scaled[0, 0]), float(K_scaled[1, 1])
                cx, cy = float(K_scaled[0, 2]), float(K_scaled[1, 2])
                W_f, H_f = float(crop_w), float(crop_h)
                far, near = 100000.0, 0.01
                P = np.zeros((4, 4), dtype=np.float32)
                P[0, 0] = 2.0 * fx / W_f
                P[0, 2] = 1.0 - 2.0 * cx / W_f
                P[1, 1] = 2.0 * fy / H_f
                P[1, 2] = 2.0 * cy / H_f - 1.0
                P[2, 2] = -(far + near) / (far - near)
                P[2, 3] = -2.0 * far * near / (far - near)
                P[3, 2] = -1.0
                # Bake the vertical flip into projection so the framebuffer is
                # already top-to-bottom, matching the mesh/point readback (which
                # does no CPU [::-1]). gau_vert.glsl performs no Y-flip itself.
                P[1, :] = -P[1, :]

                cam_pos = (-R.T @ t).astype(np.float32)
                htany = (crop_h / 2.0) / fy
                htanx = (crop_w / 2.0) / fx

                if 'view_matrix' in prog_int:
                    prog_int['view_matrix'].write(V.T.tobytes())
                if 'projection_matrix' in prog_int:
                    prog_int['projection_matrix'].write(P.T.tobytes())
                if 'cam_pos' in prog_int:
                    prog_int['cam_pos'].value = tuple(float(c) for c in cam_pos)
                if 'hfovxy_focal' in prog_int:
                    prog_int['hfovxy_focal'].value = (htanx, htany, fy)
                mvp = None
            else:
                mvp = _build_mvp(K_scaled, R, t, crop_w, crop_h)
            t_mvp_build = perf_counter() - t0_mvp

            t0_render = perf_counter()
            if is_splat:
                vao_int.render(mode=render_mode, instances=n_instances)
            else:
                prog_int['mvp'].write(mvp.tobytes())
                vao_int.render(mode=render_mode) if render_mode is not None else vao_int.render()
            # NOTE: Removed ctx.finish() — fbo.read() synchronizes implicitly.
            # This saves ~3.2ms per camera by avoiding redundant GPU stall.
            t_render = perf_counter() - t0_render

            # Fused distortion: warp the still-resident index + depth textures into a
            # native-resolution warped FBO, then point the rest of the loop at it. The
            # warped FBO has the same layout as a freshly-rendered native FBO (1-based
            # R32I index in color 0, R32F depth in color 1), so the existing readback,
            # coverage and assembly code below runs unchanged. result_scale becomes 1.0
            # (output is native), so the paste/upsample fast-paths engage.
            if wm is not None:
                map_x, map_y = wm
                fbo = cls._warp_fbo_gl(mgl_context, fbo, render_w, render_h,
                                       map_x, map_y, width, height,
                                       have_depth=compute_depth_map)
                u_min = v_min = 0
                crop_w, crop_h = width, height
                render_w, render_h = width, height
                dynamic_scale = 1.0

            t_setup = t_crop + t_fbo  # Total setup time

            # ================================================================
            # READBACK PHASE: GPU sync + texture transfer + data processing
            # ================================================================
            t0_readback_total = perf_counter()

            # GPU sync: ensure render is complete before readback
            # NOTE: removed ctx.finish() — fbo.read_into() synchronizes implicitly
            t0_sync = perf_counter()
            t_gpu_sync = 0.0

            # The vertex shader bakes the vertical flip into clip space, so the
            # readback is already in top-to-bottom image order (no CPU [::-1]).
            fbo_to_read = fbo

            # Readback: CUDA-GL interop (zero-PCIe FBO→CUDA D2D) when available, else
            # the portable moderngl PBO readback. The interop path keeps the index map
            # on the GPU long enough to also compute visible_indices (torch.unique) for
            # free, then copies to host via torch's faster .cpu(); the standalone PBO
            # copy (glGetBufferSubData) runs at ~3.7 GB/s, torch's at ~6.7 GB/s.
            t0_read = perf_counter()
            interop = cls._get_interop(mgl_context)
            crop_index_map = None
            visible_indices = None  # may be filled here by interop
            if interop is not None:
                try:
                    import torch
                    from coralnet_toolbox.MVAT.shaders.gpu_interop import _pbo_cuda_readback
                    fbo_to_read.use()  # bind so glReadPixels targets this FBO's attachment 0
                    t_gpu = _pbo_cuda_readback(interop['gl'], interop['cudart'],
                                               crop_w, crop_h, flip=False, cache=interop['cache'])
                    if t_gpu is None:
                        raise RuntimeError("interop readback returned None")
                    t_gpu -= 1  # decode 1-based → 0-based on the GPU
                    if compute_visible_indices:
                        valid = t_gpu[t_gpu >= 0]
                        visible_indices = (torch.unique(valid).to(torch.int32).cpu().numpy()
                                           if valid.numel() else np.array([], dtype=np.int32))
                    crop_index_map = t_gpu.cpu().numpy()
                    del t_gpu
                except Exception as interop_err:
                    logger.debug(f"      interop readback failed, moderngl fallback: {interop_err}")
                    crop_index_map = None
                    visible_indices = None

            if crop_index_map is None:
                # Portable moderngl path: persistent PBO, decode in place.
                pbo = pbos[i % 2]  # Double-buffered PBO (round-robin)
                fbo_to_read.read_into(pbo, components=1, dtype='i4')
                raw_bytes = pbo.read(size=crop_h * crop_w * 4)
                crop_index_map = np.frombuffer(raw_bytes, dtype=np.int32).reshape(crop_h, crop_w).copy()
                np.subtract(crop_index_map, 1, out=crop_index_map)
            t_fbo_read = perf_counter() - t0_read

            # Data processing: timing attribution (read_into is now the only operation)
            t0_proc_total = perf_counter()
            t0_decode_data = perf_counter()
            t_decode_data = 0.0  # Decoding now happens in-place during read_into

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
            if not compute_visible_indices:
                visible_indices = np.array([], dtype=np.int32)
            elif visible_indices is not None:
                pass  # already computed on the GPU during the interop readback
            else:
                # GPU coverage: scan the still-resident index texture, scatter
                # cov[id]=1, then read back N uints + flatnonzero. Replaces a
                # bincount over millions of foreground pixels (~66-109ms at
                # production resolution) with an O(N_elements) readback (~0.4-18ms).
                if cov_prog is not None and cov_buffer is not None:
                    try:
                        cov_buffer.clear()
                        cov_buffer.bind_to_storage_buffer(5)
                        fbo.color_attachments[0].bind_to_image(0, read=True, write=False)
                        cov_prog['dims'].value = (crop_w, crop_h)
                        ctx.memory_barrier()  # rasterization writes → image reads
                        cov_prog.run(group_x=(crop_w + 15) // 16,
                                     group_y=(crop_h + 15) // 16)
                        ctx.memory_barrier()  # compute writes → buffer readback
                        cov = np.frombuffer(cov_buffer.read(), dtype=np.uint32)
                        visible_indices = np.flatnonzero(cov).astype(np.int32)
                    except Exception as cov_err:
                        logger.debug(f"      coverage compute failed, bincount fallback: {cov_err}")
                        visible_indices = None
                if visible_indices is None:
                    # np.unique (not bincount): scales with the visible pixels, not the
                    # element-id range, so it doesn't allocate an n_cells-sized array on
                    # huge meshes. Same sorted-unique result.
                    valid_indices = crop_index_map[crop_index_map >= 0]
                    visible_indices = (np.unique(valid_indices).astype(np.int32)
                                       if valid_indices.size else np.array([], dtype=np.int32))
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
                # Linear camera-space depth is computed in the fragment shader and
                # written to color attachment 1 (R32F), so the CPU only reshapes and
                # masks the background — the z_ndc → linear math is gone.
                try:
                    # glReadPixels on attachment 1 uses the same orientation as the
                    # index map (shader bakes the Y-flip), so this aligns pixel-for-
                    # pixel with crop_index_map. No [::-1] needed.
                    depth_raw = fbo.read(components=1, attachment=1, dtype='f4')
                    linear_depth = np.frombuffer(depth_raw, dtype=np.float32).reshape(crop_h, crop_w).copy()

                    # Background fragments never ran the shader, so they hold the
                    # framebuffer clear value (0.0); mask them to NaN via the index map.
                    linear_depth[crop_index_map == -1] = np.nan
                    crop_depth_map = linear_depth
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

        # Release the per-call readback PBOs. These were allocated above sized to
        # this batch's largest camera; with a per-batch context they vanished on
        # ctx.release(), but on a warm (persistent) context they would leak a few
        # hundred MB of VRAM every call. The FBO cache and uploaded geometry are
        # intentionally retained on the context for reuse — only these transient
        # PBOs are freed here.
        for _pbo in pbos:
            try:
                _pbo.release()
            except Exception:
                pass

        n_cams = max(1, len(camera_params_list))
        total_time = perf_counter() - start_time
        accounted_time = _t_setup_total + _t_render_total + _t_readback_total + _t_decode_total
        unaccounted_time = total_time - accounted_time
        readback_mode = "CPU readback"

        # Off-screen cameras early-out before the per-camera timers run, which
        # can leave a stat's 'min' at the initial +inf. Reset those to 0 so the
        # summary doesn't print "infms".
        for _s in _stats.values():
            if _s['min'] == float('inf'):
                _s['min'] = 0.0

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
            results = cls.compute_batch_visibility_moderngl(
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
