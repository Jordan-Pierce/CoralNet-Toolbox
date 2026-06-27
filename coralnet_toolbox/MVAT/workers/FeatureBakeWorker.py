"""
FeatureBakeWorker — Tier 2 feature buffer baking (lift→compress→scatter→finalize).

Mirrors VisibilityWorker structure: QObject+signals, runs on QThread, chunked over cameras.
Per-camera: id-resample, compress C→D, confidence-weight, and scatter into [N,D] accumulator.
"""

from __future__ import annotations

import gc
import warnings
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from PyQt5.QtCore import QObject, pyqtSignal

from coralnet_toolbox.MVAT.core.FeatureBuffer import FeatureBuffer

warnings.filterwarnings("ignore", category=DeprecationWarning)


class FeatureBakeWorkerSignals(QObject):
    """Signals for FeatureBakeWorker."""
    finished = pyqtSignal(object)  # FeatureBuffer
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)  # done, total, message


class FeatureBakeWorker(QObject):
    """
    Bake Tier-2 feature buffer by lifting per-camera feature maps onto the mesh.

    Inputs:
        - eligible_cameras: [(path, Camera)] with both feature_map and index_map
        - primary_target: MeshProduct or PointCloudProduct
        - compressor: fitted FeatureCompressor (NN, PCA, etc.)
        - weighting_config: dict with toggles (angle, inv_dist, edge_guard)
        - element_count: N (mesh.n_cells or point cloud n_points)

    Output:
        - FeatureBuffer: [N,D] features, coverage, valid, pca_rgb (optional), metadata
    """

    def __init__(self, eligible_cameras: List[Tuple[str, Any]], primary_target: Any,
                 compressor: Any, weighting_config: Dict[str, Any],
                 element_count: int, cache_manager: Any = None,
                 interpolation: str = "nearest"):
        super().__init__()
        self.signals = FeatureBakeWorkerSignals()
        self.eligible_cameras = eligible_cameras
        self.primary_target = primary_target
        self.compressor = compressor
        self.weighting_config = weighting_config
        self.element_count = element_count
        self.cache_manager = cache_manager
        # How the coarse patch-grid features are sampled up to full image
        # resolution. NOTE: this only affects *feature* sampling — element IDs
        # (index_map) are always taken nearest (IDs are categorical and must
        # never be interpolated).
        self.interpolation = str(interpolation or "nearest").lower()
        self._cancelled = False

    def cancel(self):
        """Request cancellation."""
        self._cancelled = True

    def run(self):
        """Main worker loop: bake the feature buffer."""
        try:
            buffer = self._bake_feature_buffer()
            if not self._cancelled:
                self.signals.finished.emit(buffer)
        except Exception as e:
            error_msg = f"FeatureBakeWorker error: {e}"
            print(f"ERROR: {error_msg}")
            self.signals.error.emit(error_msg)

    def _bake_feature_buffer(self) -> FeatureBuffer:
        """Execute the bake: lift, compress, scatter, finalize."""
        import time
        _bake_t0 = time.perf_counter()

        N = self.element_count
        D = self.compressor.out_dim
        C = None  # Inferred from first map

        # Use GPU if available, else CPU
        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        use_torch = torch is not None

        # Accumulators
        if use_torch:
            feature_sum = torch.zeros((N, D), dtype=torch.float32, device=device)
            weight_sum = torch.zeros(N, dtype=torch.float32, device=device)
        else:
            feature_sum = np.zeros((N, D), dtype=np.float32)
            weight_sum = np.zeros(N, dtype=np.float32)

        # Pre-load geometry once before the camera loop.
        # On CUDA, also upload to GPU so weight computation never crosses PCIe per chunk.
        centers, normals = self._get_geometry()
        centers_gpu = normals_gpu = None
        if use_torch and device == "cuda":
            if centers is not None:
                centers_gpu = torch.as_tensor(centers, dtype=torch.float32, device=device)
            if normals is not None:
                normals_gpu = torch.as_tensor(normals, dtype=torch.float32, device=device)

        # Per-camera loop
        total_cams = len(self.eligible_cameras)
        for cam_idx, (path, camera) in enumerate(self.eligible_cameras):
            if self._cancelled:
                raise RuntimeError("Worker cancelled")

            try:
                self._process_camera(
                    camera, feature_sum, weight_sum, N, device, use_torch,
                    centers, normals, centers_gpu, normals_gpu,
                )
                if C is None:
                    # Infer C from the first successful map
                    fm = camera._raster.feature_map
                    if fm is not None:
                        C = fm.shape[-1]
            except Exception as e:
                print(f"[FeatureBake] Camera {path} failed: {e}")
                continue

            if (cam_idx + 1) % max(1, total_cams // 10) == 0:
                self.signals.progress.emit(
                    cam_idx + 1, total_cams,
                    f"Lifted {cam_idx + 1}/{total_cams} camera(s)"
                )

        # Finalize
        if use_torch:
            valid = weight_sum > 0
            features = feature_sum / weight_sum[:, None].clamp(min=1e-8)
            features = torch.nn.functional.normalize(features, dim=1, p=2)
            features[~valid] = 0
            # Move back to CPU for storage
            features_np = features.cpu().numpy().astype(np.float16)
            coverage_np = weight_sum.cpu().numpy().astype(np.float32)
            valid_np = valid.cpu().numpy()
        else:
            valid = weight_sum > 0
            features = feature_sum / np.maximum(weight_sum[:, None], 1e-8)
            # L2 normalize
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            features = features / norms
            features[~valid] = 0
            features_np = features.astype(np.float16)
            coverage_np = weight_sum.astype(np.float32)
            valid_np = valid

        # --- DEBUG: final buffer coverage summary ----------------------------
        n_valid = int(np.count_nonzero(valid_np))
        cov_pos = coverage_np[coverage_np > 0]
        print(
            f"[FeatureDebug] BAKE FINALIZE: N={N}, valid={n_valid} "
            f"({100.0 * n_valid / max(N, 1):.2f}%), cameras={len(self.eligible_cameras)}, "
            f"coverage[min/mean/max over covered]="
            f"{(float(cov_pos.min()) if cov_pos.size else 0):.4g}/"
            f"{(float(cov_pos.mean()) if cov_pos.size else 0):.4g}/"
            f"{(float(cov_pos.max()) if cov_pos.size else 0):.4g}"
        )

        # Compute PCA-RGB if D >= 3
        pca_rgb = None
        if features_np.shape[1] >= 3:
            try:
                pca_rgb = self._compute_pca_rgb(features_np[valid_np])
                # Pad to full [N, 3]
                if pca_rgb is not None:
                    full_pca_rgb = np.zeros((N, 3), dtype=np.uint8)
                    full_pca_rgb[valid_np] = pca_rgb
                    pca_rgb = full_pca_rgb
            except Exception as e:
                print(f"[FeatureBake] PCA-RGB failed: {e}")
                pca_rgb = None

        # Assemble provenance
        model_id = (
            getattr(self.eligible_cameras[0][1]._raster, 'feature_map_model_id', 'unknown')
            if self.eligible_cameras else 'unknown'
        )
        provenance = {
            "model_id": model_id,
            "compressor_kind": self.compressor.kind,
            "compressor_dim": D,
            "weighting_config": self.weighting_config,
            "element_type": getattr(self.primary_target, 'get_element_type', lambda: 'face')(),
            "element_count": N,
            "num_valid": int(np.sum(valid_np)),
            "num_cameras": len(self.eligible_cameras),
        }

        self.signals.progress.emit(total_cams, total_cams, "Finalizing buffer…")

        buffer = FeatureBuffer(
            features=features_np,
            coverage=coverage_np,
            valid=valid_np,
            compressor_state=self.compressor.state_dict(),
            pca_rgb=pca_rgb,
            provenance=provenance,
        )

        # Cleanup
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        _bake_elapsed = time.perf_counter() - _bake_t0
        print(
            f"[FeatureDebug] BAKE TOTAL: {_bake_elapsed:.2f}s "
            f"({len(self.eligible_cameras)} cameras, {N} elements, device={device})"
        )

        return buffer

    def _process_camera(self, camera: Any, feature_sum, weight_sum, N: int,
                        device: str, use_torch: bool,
                        centers, normals,
                        centers_gpu, normals_gpu) -> None:
        """Lift, compress, weight, and scatter one camera's features.

        The feature grid is tiny (e.g. 14×14) relative to the full-resolution
        index_map (e.g. 3310×5104). Downsampling the index_map to the patch grid
        would tag only ~h*w faces per camera, leaving virtually the whole mesh
        uncovered. Instead we iterate at the *full index_map resolution* and give
        every visible face the feature of its nearest patch (nearest-upsample of
        the feature grid). Processed in row-chunks to bound memory.

        On CUDA, the index_map is transferred to the GPU once per camera and all
        per-chunk operations (masking, feature gather, weight computation, scatter)
        run purely on GPU. On CPU/MPS the legacy chunked NumPy path is used.
        """
        raster = camera._raster
        feature_map = raster.feature_map  # [h, w, C] fp16 via LRU
        index_map = raster.index_map      # [H, W] int32, -1 = background

        if feature_map is None or index_map is None:
            return

        feature_map = np.asarray(feature_map, dtype=np.float32)  # [h, w, C]
        h_feat, w_feat, C = feature_map.shape
        index_map_np = np.asarray(index_map, dtype=np.int32)
        H_idx, W_idx = index_map_np.shape

        # Compress the (small) patch grid ONCE → [h, w, D]; NaN/inf guard.
        F_patch = self.compressor.transform(
            feature_map.reshape(h_feat * w_feat, C)
        ).astype(np.float32)
        F_patch[~np.isfinite(F_patch)] = 0.0
        D = F_patch.shape[1]
        F_grid = F_patch.reshape(h_feat, w_feat, D)

        nearest = self.interpolation == "nearest"
        camera_pos = np.asarray(camera.position, dtype=np.float32)

        total_kept = 0

        if use_torch and device == "cuda":
            # ------------------------------------------------------------------
            # GPU path: single H2D transfer per camera, all ops on GPU.
            # ------------------------------------------------------------------

            # Build patch-coord lookup tables on GPU (H_idx / W_idx elements — tiny).
            if nearest:
                pr_lut = torch.as_tensor(
                    (np.arange(H_idx) * h_feat // H_idx).astype(np.int64), device=device
                )
                pc_lut = torch.as_tensor(
                    (np.arange(W_idx) * w_feat // W_idx).astype(np.int64), device=device
                )
            else:
                pr_lut = torch.as_tensor(
                    ((np.arange(H_idx) + 0.5) * h_feat / H_idx - 0.5).astype(np.float32),
                    device=device,
                )
                pc_lut = torch.as_tensor(
                    ((np.arange(W_idx) + 0.5) * w_feat / W_idx - 0.5).astype(np.float32),
                    device=device,
                )

            # Single H2D transfers per camera.
            idx_gpu = torch.as_tensor(index_map_np, dtype=torch.int32, device=device)
            F_grid_gpu = torch.as_tensor(F_grid, dtype=torch.float32, device=device)
            cam_pos_gpu = torch.as_tensor(camera_pos, dtype=torch.float32, device=device)

            weighting_config = self.weighting_config or {}
            use_angle = weighting_config.get("use_angle", True)
            use_inv_dist = weighting_config.get("use_inv_dist", True)

            # Larger chunks: no H2D overhead per chunk, bounded only by VRAM.
            chunk_rows = 2048
            for r0 in range(0, H_idx, chunk_rows):
                r1 = min(H_idx, r0 + chunk_rows)

                # Slice from GPU tensor — zero PCIe cost.
                ids_chunk = idx_gpu[r0:r1].reshape(-1).long()  # [chunk*W]
                keep = ids_chunk >= 0
                if not keep.any():
                    continue
                ids = ids_chunk[keep]  # [K]

                # Expand LUT entries to pixel positions for this chunk.
                rows_full = pr_lut[r0:r1].repeat_interleave(W_idx)  # [chunk*W]
                cols_full = pc_lut.repeat(r1 - r0)                  # [chunk*W]
                rows = rows_full[keep]
                cols = cols_full[keep]

                if nearest:
                    F = F_grid_gpu[rows, cols]  # [K, D]
                else:
                    F = self._sample_grid_bilinear_gpu(F_grid_gpu, rows, cols)

                # Confidence weights — pure GPU arithmetic.
                w_vec = torch.ones(ids.shape[0], dtype=torch.float32, device=device)
                if centers_gpu is not None:
                    view_dirs = cam_pos_gpu.unsqueeze(0) - centers_gpu[ids]   # [K, 3]
                    dists = torch.linalg.norm(view_dirs, dim=1)               # [K]
                    dists_safe = dists.clamp(min=1e-2)

                    if use_angle and normals_gpu is not None:
                        vd = view_dirs / dists_safe.unsqueeze(1)
                        angles = (normals_gpu[ids] * vd).sum(dim=1).clamp(min=0.0)
                        w_vec = w_vec * angles

                    if use_inv_dist:
                        w_vec = w_vec * (1.0 / dists_safe)

                w_vec = w_vec.clamp(0.0, 1.0)

                feature_sum.index_add_(0, ids, F * w_vec.unsqueeze(1))
                weight_sum.index_add_(0, ids, w_vec)

                total_kept += int(ids.shape[0])

        else:
            # ------------------------------------------------------------------
            # CPU / non-CUDA path: original chunked NumPy approach.
            # ------------------------------------------------------------------
            if nearest:
                patch_r_for_row = (np.arange(H_idx) * h_feat // H_idx).astype(np.int64)
                patch_c_for_col = (np.arange(W_idx) * w_feat // W_idx).astype(np.int64)
            else:
                patch_r_for_row = (
                    (np.arange(H_idx) + 0.5) * h_feat / H_idx - 0.5
                ).astype(np.float32)
                patch_c_for_col = (
                    (np.arange(W_idx) + 0.5) * w_feat / W_idx - 0.5
                ).astype(np.float32)

            index_map_i64 = index_map_np.astype(np.int64)

            chunk_rows = 256
            for r0 in range(0, H_idx, chunk_rows):
                r1 = min(H_idx, r0 + chunk_rows)
                ids = index_map_i64[r0:r1].ravel()
                keep = ids >= 0
                if not keep.any():
                    continue
                ids = ids[keep]

                rows = np.repeat(patch_r_for_row[r0:r1], W_idx)[keep]
                cols = np.tile(patch_c_for_col, r1 - r0)[keep]
                if nearest:
                    F = F_grid[rows, cols]
                else:
                    F = self._sample_grid_bilinear(F_grid, rows, cols)

                w_vec = self._compute_weights_for_ids(ids, centers, normals, camera_pos)

                if use_torch:
                    ids_t = torch.as_tensor(ids, dtype=torch.long, device=device)
                    F_t = torch.as_tensor(F, dtype=torch.float32, device=device)
                    w_t = torch.as_tensor(w_vec, dtype=torch.float32, device=device)
                    feature_sum.index_add_(0, ids_t, F_t * w_t[:, None])
                    weight_sum.index_add_(0, ids_t, w_t)
                else:
                    np.add.at(feature_sum, ids, F * w_vec[:, None])
                    np.add.at(weight_sum, ids, w_vec)

                total_kept += int(ids.size)


    @staticmethod
    def _sample_grid_bilinear(F_grid: np.ndarray, rr: np.ndarray,
                              cc: np.ndarray) -> np.ndarray:
        """Bilinearly sample a [h, w, D] feature grid at float coords (rr, cc).

        Returns [K, D]. Coordinates are clamped to the grid; corners outside the
        grid reuse the edge value (standard clamp-to-border behavior).
        """
        h, w, _ = F_grid.shape
        r0 = np.floor(rr).astype(np.int64)
        c0 = np.floor(cc).astype(np.int64)
        wr = (rr - r0).astype(np.float32)
        wc = (cc - c0).astype(np.float32)
        r0 = np.clip(r0, 0, h - 1)
        c0 = np.clip(c0, 0, w - 1)
        r1 = np.clip(r0 + 1, 0, h - 1)
        c1 = np.clip(c0 + 1, 0, w - 1)

        f00 = F_grid[r0, c0]
        f01 = F_grid[r0, c1]
        f10 = F_grid[r1, c0]
        f11 = F_grid[r1, c1]
        top = f00 * (1.0 - wc)[:, None] + f01 * wc[:, None]
        bot = f10 * (1.0 - wc)[:, None] + f11 * wc[:, None]
        return top * (1.0 - wr)[:, None] + bot * wr[:, None]

    @staticmethod
    def _sample_grid_bilinear_gpu(F_grid: "torch.Tensor", rr: "torch.Tensor",
                                  cc: "torch.Tensor") -> "torch.Tensor":
        """Bilinearly sample a [h, w, D] GPU feature grid at float coords (rr, cc).

        Returns [K, D]. Edge values are clamped (clamp-to-border).
        """
        h, w, _ = F_grid.shape
        r0 = rr.floor().long()
        c0 = cc.floor().long()
        wr = (rr - r0.float()).unsqueeze(1)   # [K, 1]
        wc = (cc - c0.float()).unsqueeze(1)   # [K, 1]
        r0 = r0.clamp(0, h - 1)
        c0 = c0.clamp(0, w - 1)
        r1 = (r0 + 1).clamp(0, h - 1)
        c1 = (c0 + 1).clamp(0, w - 1)

        f00 = F_grid[r0, c0]
        f01 = F_grid[r0, c1]
        f10 = F_grid[r1, c0]
        f11 = F_grid[r1, c1]
        top = f00 * (1.0 - wc) + f01 * wc
        bot = f10 * (1.0 - wc) + f11 * wc
        return top * (1.0 - wr) + bot * wr

    def _get_geometry(self):
        """Cache and return (element_centers [N,3], element_normals [N,3]) once.

        Centers come from the generic ``_element_centers_np`` array that every
        product type populates (mesh face centers, point-cloud points, splat
        centers). Normals are mesh-only, so they are fetched opportunistically
        and left as ``None`` for point clouds / Gaussian splats (which have no
        per-element normals); the weighting code degrades gracefully.
        """
        if getattr(self, "_centers_cache", None) is None:
            centers = getattr(self.primary_target, "_element_centers_np", None)
            if centers is None:
                get_centers = getattr(self.primary_target, "get_face_centers", None)
                if callable(get_centers):
                    try:
                        centers = get_centers()
                    except Exception as e:
                        print(f"[FeatureBake] get_face_centers failed: {e}")
                        centers = None
            self._centers_cache = (
                None if centers is None else np.asarray(centers, dtype=np.float32)
            )
        if getattr(self, "_normals_cache", None) is None:
            get_normals = getattr(self.primary_target, "_get_cached_face_normals", None)
            if callable(get_normals):
                try:
                    normals = get_normals()
                    self._normals_cache = (
                        None if normals is None
                        else np.asarray(normals, dtype=np.float32)
                    )
                except Exception as e:
                    print(f"[FeatureBake] face normals failed: {e}")
                    self._normals_cache = None
            else:
                self._normals_cache = None
        return self._centers_cache, self._normals_cache

    def _compute_weights_for_ids(self, ids: np.ndarray, centers, normals,
                                 camera_pos: np.ndarray) -> np.ndarray:
        """
        Confidence weight per pixel-occurrence, gathered by face id.

        Default: angle × 1/dist (edge-guard from the old grid path is dropped —
        it is off by default and incompatible with the full-res scatter).
        """
        weighting_config = self.weighting_config or {}
        use_angle = weighting_config.get("use_angle", True)
        use_inv_dist = weighting_config.get("use_inv_dist", True)

        weights = np.ones(ids.shape[0], dtype=np.float32)

        if centers is not None:
            view_dirs = camera_pos[None, :] - centers[ids]            # [K, 3]
            dists = np.linalg.norm(view_dirs, axis=1)
            dists_safe = np.maximum(dists, 1e-2)

            if use_angle and normals is not None:
                vd = view_dirs / dists_safe[:, None]
                angles = np.maximum(0.0, np.sum(normals[ids] * vd, axis=1))
                weights *= angles.astype(np.float32)

            if use_inv_dist:
                weights *= (1.0 / dists_safe).astype(np.float32)

        return np.clip(weights, 0.0, 1.0)

    def _compute_pca_rgb(self, features_valid: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute PCA-RGB for the top 3 dims.

        Args:
            features_valid: [M, D] where M = number of valid elements.

        Returns:
            [M, 3] uint8 RGB, or None on error.
        """
        try:
            from sklearn.decomposition import PCA
            D = features_valid.shape[1]
            if D < 3:
                return None
            pca = PCA(n_components=3)
            rgb_float = pca.fit_transform(features_valid).astype(np.float32)
            # Min-max scale to [0, 255]
            rgb_min = np.min(rgb_float, axis=0, keepdims=True)
            rgb_max = np.max(rgb_float, axis=0, keepdims=True)
            rgb_max = np.maximum(rgb_max - rgb_min, 1e-8)
            rgb_normalized = (rgb_float - rgb_min) / rgb_max
            rgb_uint8 = (rgb_normalized * 255).astype(np.uint8)
            return rgb_uint8
        except Exception as e:
            print(f"[FeatureBake] PCA-RGB computation failed: {e}")
            return None
