"""
FeatureMeshManager — Tier 2 feature buffer orchestration for MVAT.

Responsibilities:
  - Precondition checking (feature + index maps present)
  - Worker launch + caching
  - QueryEngine instantiation
  - Viewer integration (array attachment, recolor triggers)
"""

from __future__ import annotations

import hashlib
from typing import Optional, List, Tuple, Dict, Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from coralnet_toolbox.MVAT.core.FeatureBuffer import FeatureBuffer
from coralnet_toolbox.MVAT.workers.FeatureBakeWorker import FeatureBakeWorker
from coralnet_toolbox.MVAT.utils.FeatureBufferCodec import save_feature_buffer
from PyQt5.QtCore import QThread, Qt
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox,
    QPushButton, QGroupBox, QCheckBox, QDoubleSpinBox, QWidget, QFormLayout
)


class QueryEngine:
    """
    Real-time query engine: add prototypes, compute similarity, select by threshold.

    Two scoring modes, chosen automatically per query (same matvec cost either way):

    - **Max-pool cosine** (0/1-sided clicks): sim = best_pos - alpha*best_neg, where
      best_pos[i] / best_neg[i] are the max cosine of element i to ANY positive /
      negative prototype. Used whenever there isn't at least one positive AND one
      negative click.
    - **Linear head** (>=1 positive AND >=1 negative): fit a class-balanced,
      ridge-regularized linear discriminant in D-space over the clicked features
      and score sim = features @ w + b, mapped to [0,1] with the decision boundary
      at 0.5. Learns which feature dims separate the classes (cosine weights all
      dims equally) and folds pos/neg into one boundary instead of subtracting two
      max fields. The fit is a tiny (D+1)x(D+1) solve — microseconds.

    On GPU (torch) if available; fallback to CPU (numpy).
    """

    # Weight of the negative term in the max-pool score (best_pos - alpha*best_neg).
    _ALPHA = 1.0
    # Ridge strength for the linear head, scaled to the data (mean Gram diagonal).
    _RIDGE = 1.0

    def __init__(self, features: np.ndarray, valid: np.ndarray):
        """
        Args:
            features: [N, D] float16, L2-normalized.
            valid: [N] bool, which elements were seen by ≥1 camera.
        """
        self.valid = valid
        self.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        self.use_torch = torch is not None

        # Keep both CPU (numpy) and GPU (torch) copies for flexibility
        self.features_np = features.astype(np.float32)
        if self.use_torch:
            self.features_cuda = torch.tensor(
                self.features_np, dtype=torch.float32, device=self.device
            )
            self.valid_dev = torch.as_tensor(
                np.asarray(self.valid, dtype=bool), device=self.device
            )
        else:
            self.features_cuda = None
            self.valid_dev = None

        self.positive_ids = set()
        self.negative_ids = set()

        # Incremental running maxima of cosine similarity to ANY positive /
        # negative prototype, held on-device. Union (max) semantics mean adding
        # a prototype is a single [N] matvec + elementwise max — independent of
        # how many prototypes were already clicked, with no full [N,K] recompute.
        self._best_pos = None   # [N] device tensor / numpy array, or None
        self._best_neg = None

        # Lazy caches, invalidated whenever the prototype set changes.
        self._sim_dev = None           # [N] similarity on device (masked to -inf)
        self._similarity_cached = None  # [N] numpy mirror (host copy for select())

    def _invalidate(self) -> None:
        """Drop cached similarity products after a prototype-set change."""
        self._sim_dev = None
        self._similarity_cached = None

    def _proto_sim(self, eid: int):
        """[N] cosine similarity of every element to prototype ``eid`` (device)."""
        if self.use_torch:
            return self.features_cuda @ self.features_cuda[eid]
        return self.features_np @ self.features_np[eid]

    def _accumulate(self, eid: int, attr: str) -> None:
        """Fold prototype ``eid`` into the running max stored on ``attr``."""
        s = self._proto_sim(eid)
        cur = getattr(self, attr)
        if cur is None:
            cur = s
        elif self.use_torch:
            cur = torch.maximum(cur, s)
        else:
            cur = np.maximum(cur, s)
        setattr(self, attr, cur)
        self._invalidate()

    def add_positive(self, element_id: int) -> None:
        """Add a positive prototype element."""
        eid = int(element_id)
        if not (0 <= eid < self.features_np.shape[0]) or eid in self.positive_ids:
            return
        self.positive_ids.add(eid)
        self._accumulate(eid, "_best_pos")

    def add_negative(self, element_id: int) -> None:
        """Add a negative prototype element."""
        eid = int(element_id)
        if not (0 <= eid < self.features_np.shape[0]) or eid in self.negative_ids:
            return
        self.negative_ids.add(eid)
        self._accumulate(eid, "_best_neg")

    def clear(self) -> None:
        """Clear all prototypes."""
        self.positive_ids.clear()
        self.negative_ids.clear()
        self._best_pos = None
        self._best_neg = None
        self._invalidate()

    # -- scoring backends -------------------------------------------------

    def _maxpool_sim(self, best_pos, best_neg):
        """Unmasked max-pool score: best_pos - alpha*best_neg (fresh array)."""
        N = self.features_np.shape[0]
        if self.use_torch:
            sim = (torch.zeros(N, dtype=torch.float32, device=self.device)
                   if best_pos is None else best_pos.clone())
            if best_neg is not None:
                sim = sim - self._ALPHA * best_neg
            return sim
        sim = (np.zeros(N, dtype=np.float32)
               if best_pos is None else best_pos.astype(np.float32, copy=True))
        if best_neg is not None:
            sim = sim - self._ALPHA * best_neg
        return sim.astype(np.float32, copy=False)

    def _fit_linear_head(self, pos_ids, neg_ids):
        """
        Fit a class-balanced, ridge-regularized linear discriminant in D-space.

        Solves (Xaᵀ W Xa + λR) θ = Xaᵀ W y on ±1 targets, where Xa augments the
        clicked features with a bias column, W balances the two classes, and R is
        identity except the (un-regularized) bias term. Returns (w [D], b scalar).
        """
        pos = sorted(int(i) for i in pos_ids)
        neg = sorted(int(i) for i in neg_ids)
        D = self.features_np.shape[1]

        if self.use_torch:
            dev = self.device
            Xp = self.features_cuda[torch.as_tensor(pos, dtype=torch.long, device=dev)]
            Xn = self.features_cuda[torch.as_tensor(neg, dtype=torch.long, device=dev)]
            Np, Nn = Xp.shape[0], Xn.shape[0]
            X = torch.cat([Xp, Xn], 0)
            y = torch.cat([torch.ones(Np, device=dev), -torch.ones(Nn, device=dev)])
            wts = torch.cat([torch.full((Np,), 0.5 / Np, device=dev),
                             torch.full((Nn,), 0.5 / Nn, device=dev)])
            Xa = torch.cat([X, torch.ones((X.shape[0], 1), device=dev)], 1)
            A = Xa.t() @ (Xa * wts[:, None])          # [D+1, D+1]
            rhs = Xa.t() @ (wts * y)                  # [D+1]
            lam = self._RIDGE * (torch.diagonal(A)[:D].mean() + 1e-6)
            reg = torch.eye(D + 1, device=dev) * lam
            reg[D, D] = 0.0                           # don't regularize the bias
            sol = torch.linalg.solve(A + reg, rhs)
            return sol[:D], sol[D]

        Xp = self.features_np[pos]
        Xn = self.features_np[neg]
        Np, Nn = Xp.shape[0], Xn.shape[0]
        X = np.concatenate([Xp, Xn], 0)
        y = np.concatenate([np.ones(Np, np.float32), -np.ones(Nn, np.float32)])
        wts = np.concatenate([np.full(Np, 0.5 / Np, np.float32),
                              np.full(Nn, 0.5 / Nn, np.float32)])
        Xa = np.concatenate([X, np.ones((X.shape[0], 1), np.float32)], 1)
        A = Xa.T @ (Xa * wts[:, None])
        rhs = Xa.T @ (wts * y)
        lam = self._RIDGE * (np.diagonal(A)[:D].mean() + 1e-6)
        reg = np.eye(D + 1, dtype=np.float32) * lam
        reg[D, D] = 0.0
        sol = np.linalg.solve(A + reg, rhs)
        return sol[:D].astype(np.float32), np.float32(sol[D])

    def _head_sim(self, pos_ids, neg_ids):
        """
        Unmasked linear-head score [N], mapped to [0,1] with the decision
        boundary at 0.5 (so the [0,1] threshold UI is shared with max-pool).
        Returns None on a fit failure (caller falls back to max-pool).
        """
        try:
            w, b = self._fit_linear_head(pos_ids, neg_ids)
        except Exception as e:
            print(f"[QueryEngine] linear head fit failed: {e}; using max-pool")
            return None
        if self.use_torch:
            return 0.5 * (self.features_cuda @ w + b) + 0.5
        return (0.5 * (self.features_np @ w + b) + 0.5).astype(np.float32)

    def _mask_invalid(self, sim):
        """Set uncovered (~valid) elements to -inf (in place / via where)."""
        if self.use_torch:
            neg_inf = torch.full_like(sim, float("-inf"))
            return torch.where(self.valid_dev, sim, neg_inf)
        sim = np.asarray(sim, dtype=np.float32)
        sim[~self.valid] = -np.inf
        return sim

    def _similarity_device(self):
        """
        Compute (and cache) similarity [N] on-device.

        Uses the linear head when there is >=1 positive AND >=1 negative click,
        otherwise the max-pool score. Uncovered elements are set to -inf. Returns
        a torch tensor (GPU/CPU) when torch is available, else numpy.
        """
        if self._sim_dev is not None:
            return self._sim_dev

        sim = None
        if self.positive_ids and self.negative_ids:
            sim = self._head_sim(self.positive_ids, self.negative_ids)
        if sim is None:
            sim = self._maxpool_sim(self._best_pos, self._best_neg)

        self._sim_dev = self._mask_invalid(sim)
        return self._sim_dev

    def similarity(self) -> np.ndarray:
        """
        Cosine similarity [N] as a host numpy array (for select()/thresholding).

        Returns:
            [N] float32 similarity scores. Uncovered elements are -inf.
        """
        if self._similarity_cached is not None:
            return self._similarity_cached

        sim = self._similarity_device()
        if self.use_torch:
            sim = sim.detach().cpu().numpy()
        self._similarity_cached = np.asarray(sim, dtype=np.float32)
        return self._similarity_cached

    def _effective_sim_device(self, hover_id: Optional[int] = None):
        """
        Similarity [N] on the active backend, optionally folding in a TRANSIENT
        hover prototype (for live hover preview) as if it were a positive click —
        WITHOUT mutating committed state or caches (throwaway per-frame compute).

        The hovered face joins the positive set: if that makes both classes
        non-empty, the live preview re-fits the linear head (cheap); otherwise it
        unions the hover into the max-pool positive maxima. hover_id=None returns
        the cached committed similarity.
        """
        if hover_id is None:
            return self._similarity_device()

        eid = int(hover_id)
        if not (0 <= eid < self.features_np.shape[0]):
            return self._similarity_device()

        # Linear-head preview: hover becomes a positive; if negatives exist, re-fit.
        if self.negative_ids:
            eff_pos = self.positive_ids | {eid}
            sim = self._head_sim(eff_pos, self.negative_ids)
            if sim is not None:
                return self._mask_invalid(sim)

        # Max-pool preview: union the hovered face into the positive maxima.
        if self.use_torch:
            hsim = self.features_cuda @ self.features_cuda[eid]  # [N]
            best_pos = hsim if self._best_pos is None else torch.maximum(self._best_pos, hsim)
        else:
            hsim = self.features_np @ self.features_np[eid]
            best_pos = hsim if self._best_pos is None else np.maximum(self._best_pos, hsim)
        return self._mask_invalid(self._maxpool_sim(best_pos, self._best_neg))

    def _disp_unit01_device(self, threshold: Optional[float], sim=None):
        """[N] display value in [0,1] on-device (torch): gradient or thresholded."""
        if sim is None:
            sim = self._similarity_device()
        finite = torch.isfinite(sim)
        disp = torch.zeros_like(sim)
        if bool(finite.any()):
            fvals = sim[finite]
            fmin = fvals.min()
            fmax = fvals.max()
            if float(fmax) > float(fmin):
                norm = (sim - fmin) / (fmax - fmin)
            else:
                norm = torch.zeros_like(sim)
            if threshold is None:
                disp = torch.where(finite, norm, disp)
            else:
                selected = finite & (sim >= threshold)
                disp = torch.where(selected, 0.5 + 0.5 * norm, disp)
        return disp

    @staticmethod
    def _disp_unit01_numpy(sim: np.ndarray, threshold: Optional[float]) -> np.ndarray:
        """CPU mirror of _disp_unit01_device: [N] float32 in [0,1]."""
        finite = np.isfinite(sim)
        disp = np.zeros_like(sim, dtype=np.float32)
        if finite.any():
            sim_min = float(np.min(sim[finite]))
            sim_max = float(np.max(sim[finite]))
            if sim_max > sim_min:
                norm = (sim - sim_min) / (sim_max - sim_min)
            else:
                norm = np.zeros_like(sim)
            if threshold is None:
                disp[finite] = norm[finite].astype(np.float32)
            else:
                selected = finite & (sim >= threshold)
                disp[selected] = (0.5 + 0.5 * norm[selected]).astype(np.float32)
        return disp

    DEFAULT_COLORMAP = "plasma"

    def _ensure_colormap_lut(self) -> None:
        """Build the colormap LUT [256,3] uint8 once (numpy + device tensor)."""
        if getattr(self, "_cmap_np", None) is not None:
            return
        try:
            import matplotlib
            cmap = matplotlib.colormaps[self.DEFAULT_COLORMAP]
        except Exception:
            import matplotlib.cm as cm
            cmap = cm.get_cmap(self.DEFAULT_COLORMAP)
        lut = (np.asarray(cmap(np.linspace(0.0, 1.0, 256)))[:, :3] * 255.0)
        self._cmap_np = lut.round().astype(np.uint8)  # [256, 3]
        self._cmap_dev = (
            torch.as_tensor(self._cmap_np, device=self.device)
            if self.use_torch else None
        )

    def display_colors(self, threshold: Optional[float] = None,
                       hover_id: Optional[int] = None) -> np.ndarray:
        """
        Per-element colormap RGB [N, 3] uint8, computed on-device.

        This is the colormap step done on the GPU (a 256-entry LUT gather) so the
        mesh can be colored with DIRECT cell colors (rgb=True) — the same path the
        Labels array uses — bypassing VTK's slow CPU scalar->color MapScalars.

        Args:
            threshold: None -> full gradient over covered faces; set -> light only
                faces with similarity >= threshold (bright half), rest dark.
            hover_id: optional transient prototype to fold in for hover preview.

        Returns:
            [N, 3] uint8, ready to write into the mesh's "Similarity" cell array.
        """
        self._ensure_colormap_lut()
        sim = self._effective_sim_device(hover_id)
        if not self.use_torch:
            disp = self._disp_unit01_numpy(sim, threshold)
            ci = np.clip(np.rint(disp * 255.0), 0, 255).astype(np.int64)
            return self._cmap_np[ci]
        disp = self._disp_unit01_device(threshold, sim)
        ci = (disp * 255.0).round().clamp_(0, 255).to(torch.long)
        colors = self._cmap_dev[ci]  # [N, 3] uint8 on device
        return colors.detach().cpu().numpy()

    def display_scalars(self, threshold: Optional[float] = None,
                        hover_id: Optional[int] = None) -> np.ndarray:
        """[N] uint8 display scalar for the GPU value-texture (shader) path."""
        sim = self._effective_sim_device(hover_id)
        if not self.use_torch:
            disp = self._disp_unit01_numpy(sim, threshold)
            return np.clip(np.rint(disp * 255.0), 0, 255).astype(np.uint8)
        disp = self._disp_unit01_device(threshold, sim)
        return (disp * 255.0).round().clamp_(0, 255).to(torch.uint8).detach().cpu().numpy()

    def select(self, threshold: float = 0.5, top_k: Optional[int] = None,
               grow: bool = False) -> np.ndarray:
        """
        Select elements by threshold or top-k.

        Args:
            threshold: Similarity threshold (only used if top_k is None).
            top_k: If set, select top k valid elements by similarity.
            grow: If True and adjacency available, region-grow. (Deferred.)

        Returns:
            [M] int64 element IDs of selected elements.
        """
        sim = self.similarity()
        valid_sim = sim[self.valid]

        if top_k is not None:
            # Top-k within valid elements
            valid_ids = np.where(self.valid)[0]
            top_indices = np.argsort(-valid_sim)[:min(top_k, len(valid_sim))]
            selected_ids = valid_ids[top_indices]
        else:
            # Threshold
            selected_ids = np.where(sim >= threshold)[0].astype(np.int64)

        return selected_ids

    def prototypes(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the clicked prototype feature vectors for the GPU shader path.

        Returns:
            (pos, neg): [Kpos, D] / [Kneg, D] float32 L2-normalized prototypes.
            Either may be empty. IDs are sorted for deterministic packing.
        """
        D = self.features_np.shape[1]
        pos = (self.features_np[sorted(self.positive_ids)]
               if self.positive_ids else np.zeros((0, D), dtype=np.float32))
        neg = (self.features_np[sorted(self.negative_ids)]
               if self.negative_ids else np.zeros((0, D), dtype=np.float32))
        return pos.astype(np.float32, copy=False), neg.astype(np.float32, copy=False)

    def sim_minmax(self) -> Tuple[float, float]:
        """
        Min/max similarity over covered faces (for the shader's normalization).

        Computed as a device reduction over the cached similarity — only two
        scalars cross to the host, never the [N] array.
        """
        sim = self._similarity_device()
        if self.use_torch:
            finite = torch.isfinite(sim)
            if not bool(finite.any()):
                return 0.0, 0.0
            fv = sim[finite]
            return float(fv.min()), float(fv.max())
        finite = np.isfinite(sim)
        if not finite.any():
            return 0.0, 0.0
        fv = sim[finite]
        return float(fv.min()), float(fv.max())


class FeatureMeshManager:
    """
    Manages Tier-2 feature buffer baking, caching, and querying.

    Owns:
      - Current buffer (in memory)
      - QueryEngine instance
      - Worker thread
    """

    def __init__(self, mvat_manager):
        """
        Args:
            mvat_manager: MVATManager instance (access to cameras, cache_manager, etc.).
        """
        self.mvat_manager = mvat_manager
        self.viewer = mvat_manager.viewer
        self.main_window = mvat_manager.main_window
        self.cache_manager = mvat_manager.cache_manager

        self.buffer: Optional[FeatureBuffer] = None
        self.query_engine: Optional[QueryEngine] = None
        self.bake_worker_thread: Optional[QThread] = None

        # Phase-2 GPU colormap shader (SimilarityShader). Bypasses VTK's per-change
        # color-buffer rebuild (the ~125 ms-at-4M / seconds-at-76M cost — confirmed
        # by profiling: a no-change render is ~5 ms, a post-recolor render ~125 ms)
        # by colormapping in a fragment shader from a small value texture, with
        # ScalarVisibilityOff so VTK never rebuilds the cell colors. If the shader
        # can't install, we fall back to direct RGB cell colors (slower but works).
        self.shader_enabled = True
        self.shader_state = None

        self._weighting_config = {
            "use_angle": True,
            "use_inv_dist": True,
            "use_edge_guard": False,
        }
        # How the coarse patch features are sampled up to image resolution
        # during the bake ("nearest" | "bilinear"). Part of the cache key.
        self._interpolation = "nearest"

    def prepare(self, scope: str = "all") -> Tuple[List[Tuple[str, Any]], Dict[str, int]]:
        """
        Check preconditions: which cameras have both feature_map AND index_map loaded?

        Args:
            scope: "all" | "selected" | "visible".

        Returns:
            (eligible_cameras, stats): List of (path, Camera) with both maps.
                stats = {"with_feature_map": N, "with_index_map": M, "both": K, "missing_feature": ..., ...}
        """
        stats = {
            "with_feature_map": 0,
            "with_index_map": 0,
            "both": 0,
            "missing_feature": 0,
            "missing_index": 0,
        }

        eligible = []

        # Collect cameras
        if scope == "selected":
            cameras_to_check = [
                (p, self.mvat_manager.cameras[p])
                for p in getattr(self.mvat_manager, "selected_camera_paths", [])
                if p in self.mvat_manager.cameras
            ]
        elif scope == "visible":
            cameras_to_check = list(self.mvat_manager.cameras.items())  # TODO: filter by visibility
        else:  # "all"
            cameras_to_check = list(self.mvat_manager.cameras.items())

        for path, camera in cameras_to_check:
            raster = getattr(camera, "_raster", None)
            if raster is None:
                continue

            # Existence check ONLY — do NOT touch raster.has_feature_map() /
            # raster.index_map here: those read the lazy properties, which pull
            # every map off disk through the LRU (decompressing all N cameras'
            # maps just to count them — the dialog-open lag). The in-memory
            # array OR a configured disk path is sufficient to mark eligibility;
            # the bake worker loads the actual data later, on its own thread.
            has_fm = (getattr(raster, "_feature_map", None) is not None
                      or bool(getattr(raster, "feature_map_path", None)))
            has_im = (getattr(raster, "_index_map", None) is not None
                      or bool(getattr(raster, "index_map_path", None)))

            if has_fm:
                stats["with_feature_map"] += 1
            else:
                stats["missing_feature"] += 1

            if has_im:
                stats["with_index_map"] += 1
            else:
                stats["missing_index"] += 1

            if has_fm and has_im:
                stats["both"] += 1
                eligible.append((path, camera))

        return eligible, stats

    def bake(self, compressor_kind: str = "nn", compressor_dim: int = 32,
             scope: str = "all", interpolation: str = "nearest",
             nn_params: dict = None) -> None:
        """
        Launch the bake worker on a background thread.

        Args:
            compressor_kind: "nn" | "pca".
            compressor_dim: Target D.
            scope: "all" | "selected" | "visible".
            interpolation: "nearest" | "bilinear" — how the coarse patch
                features are sampled up to image resolution.
            nn_params: optional dict of NN autoencoder hyperparameters
                (hidden_dim, epochs, lr, beta) when compressor_kind == "nn".
        """
        self._interpolation = str(interpolation or "nearest").lower()
        eligible, stats = self.prepare(scope)

        if stats["both"] == 0:
            self.mvat_manager.main_window.status_bar.showMessage(
                "Bake failed: no cameras have both feature maps and index maps loaded.",
                5000
            )
            return

        # Check that all feature maps use the same model
        model_ids = set()
        for path, camera in eligible:
            mid = getattr(camera._raster, "feature_map_model_id", "unknown")
            model_ids.add(mid)

        if len(model_ids) > 1:
            self.mvat_manager.main_window.status_bar.showMessage(
                f"Bake failed: mixed feature models across cameras: {model_ids}",
                5000
            )
            return

        model_id = model_ids.pop() if model_ids else "unknown"

        # Baking replaces any existing feature mesh: clear the current buffer
        # (and detach it from the mesh / tool) before building the new one. Done
        # only after preconditions pass, so a no-op bake never wipes a good buffer.
        if self.buffer is not None:
            self.clear()

        # Create compressor
        if compressor_kind == "pca":
            from coralnet_toolbox.MVAT.core.FeatureBuffer import PCACompressor
            compressor = PCACompressor(compressor_dim)
        elif compressor_kind == "nn":
            from coralnet_toolbox.MVAT.core.FeatureBuffer import NNCompressor
            params = nn_params or {}
            compressor = NNCompressor(
                compressor_dim,
                hidden_dim=params.get("hidden_dim", 256),
                epochs=params.get("epochs", 30),
                lr=params.get("lr", 1e-3),
                beta=params.get("beta", 1.0),
            )
        else:
            raise ValueError(f"Unknown compressor: {compressor_kind}")

        # Show a busy cursor for the whole bake (compressor fit on the main
        # thread + the background scatter). Restored in _on_bake_finished /
        # _on_bake_error; restored here too if launch fails before then.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Fit compressor on sampled patches
            self._fit_compressor(compressor, eligible)

            # Launch worker
            primary_target = self.viewer.scene_context.get_primary_target()
            N = primary_target.get_element_count()
        except Exception:
            QApplication.restoreOverrideCursor()
            raise

        worker = FeatureBakeWorker(
            eligible_cameras=eligible,
            primary_target=primary_target,
            compressor=compressor,
            weighting_config=self._weighting_config,
            element_count=N,
            cache_manager=self.cache_manager,
            interpolation=self._interpolation,
        )

        thread = QThread()
        self.bake_worker_thread = thread
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        worker.signals.finished.connect(
            lambda buf: self._on_bake_finished(buf, model_id, eligible)
        )
        worker.signals.error.connect(
            lambda err: self._on_bake_error(err)
        )
        worker.signals.progress.connect(
            lambda done, total, msg: (
                self.mvat_manager.main_window.status_bar.showMessage(
                    f"Baking… {msg}", 0
                )
            )
        )

        # Both finished and error must quit the thread so it can be cleaned up
        worker.signals.finished.connect(thread.quit)
        worker.signals.error.connect(thread.quit)
        worker.signals.finished.connect(worker.deleteLater)

        # Remove this entry from _active_workers when the thread finishes
        # (mirrors MVATManager._run_visibility_worker's lifecycle pattern).
        def _remove_worker(t=thread, w=worker):
            self.mvat_manager._active_workers = [
                (ot, ow) for ot, ow in self.mvat_manager._active_workers
                if ot is not t
            ]

        thread.finished.connect(_remove_worker)

        def _is_alive(t):
            try:
                return t.isRunning()
            except RuntimeError:
                return False

        self.mvat_manager._active_workers = [
            (t, w) for t, w in self.mvat_manager._active_workers if _is_alive(t)
        ]
        self.mvat_manager._active_workers.append((thread, worker))

        thread.start()

    def _fit_compressor(self, compressor, eligible: List[Tuple[str, Any]]) -> None:
        """Fit the compressor on a sample of patches from loaded feature maps."""
        # Sample patches
        sample_list = []
        max_sample = 200_000
        per_camera = max(100, max_sample // max(len(eligible), 1))

        for path, camera in eligible:
            try:
                fm = camera._raster.feature_map
                if fm is None:
                    continue
                fm = np.asarray(fm, dtype=np.float32)
                h, w, C = fm.shape
                total = h * w
                if total > per_camera:
                    indices = np.random.choice(total, per_camera, replace=False)
                    sample = fm.reshape(total, C)[indices]
                else:
                    sample = fm.reshape(total, C)
                sample_list.append(sample)
            except Exception as e:
                print(f"[FeatureMeshManager] Fit sample from {path} failed: {e}")
                continue

        if sample_list:
            sample_feats = np.vstack(sample_list)
            compressor.fit(sample_feats)

    def _on_bake_finished(self, buffer: FeatureBuffer, model_id: str,
                          eligible: List[Tuple[str, Any]]) -> None:
        """Handle successful bake completion.

        The worker thread is done by now, but there's still heavy main-thread
        post-processing: the QueryEngine GPU upload, the shader value/LUT
        textures, and the cache write to disk. Keep the busy cursor up until ALL
        of that finishes (restore in finally) — otherwise the cursor goes normal
        while the UI is still frozen.
        """
        try:
            self.buffer = buffer
            self.query_engine = QueryEngine(buffer.features, buffer.valid)
            self._build_shader_state(buffer)

            # NOTE: automated disk caching of the feature buffer is intentionally
            # disabled — the compressed write of the [N,D] buffer dominated the
            # post-bake stall. The cache methods (_cache_buffer / load_from_cache,
            # CacheManager.*_feature_buffer, FeatureBufferCodec) are kept dormant
            # for possible future use.

            # Attach arrays to the mesh
            primary_target = self.viewer.scene_context.get_primary_target()
            primary_target.attach_feature_arrays(buffer)

            # Repopulate the array dropdown
            self.viewer._update_array_selector()

            # Status
            n_valid = int(np.sum(buffer.valid))
            msg = f"Baked {n_valid}/{buffer.features.shape[0]} elements. Query ready."
            self.mvat_manager.main_window.status_bar.showMessage(msg, 5000)
        finally:
            self.bake_worker_thread = None
            QApplication.restoreOverrideCursor()

    def _on_bake_error(self, error_msg: str) -> None:
        """Handle bake error."""
        QApplication.restoreOverrideCursor()
        self.mvat_manager.main_window.status_bar.showMessage(f"Bake error: {error_msg}", 5000)
        self.bake_worker_thread = None

    def _cache_buffer(self, buffer: FeatureBuffer, model_id: str,
                      eligible: List[Tuple[str, Any]]) -> None:
        """Save the buffer to the Tier-2 cache."""
        try:
            primary_target = self.viewer.scene_context.get_primary_target()
            mesh_path = getattr(primary_target, "path", "unknown")

            # Camera set hash
            camera_extrinsics = [
                camera.extrinsics.tobytes()
                for path, camera in eligible
            ]
            camera_set_hash = hashlib.md5(
                b"".join(sorted(camera_extrinsics))
            ).hexdigest().encode()

            # Weighting flags
            weighting_str = "_".join(
                f"{k}={int(v)}" for k, v in sorted(self._weighting_config.items())
            )
            weighting_str = f"{weighting_str}_interp={self._interpolation}"
            weighting_bytes = weighting_str.encode()

            element_type = primary_target.get_element_type()

            self.cache_manager.save_feature_buffer(
                mesh_path,
                camera_set_hash,
                model_id,
                buffer.provenance.get("compressor_kind", "nn"),
                buffer.provenance.get("compressor_dim", 32),
                weighting_bytes,
                buffer,
                element_type=element_type,
            )
        except Exception as e:
            print(f"[FeatureMeshManager] Cache save failed: {e}")

    def load_from_cache(self, model_id: str, compressor_kind: str = "nn",
                        compressor_dim: int = 32) -> bool:
        """
        Try to load a feature buffer from cache.

        Returns:
            True if successfully loaded, False otherwise.
        """
        try:
            primary_target = self.viewer.scene_context.get_primary_target()
            mesh_path = getattr(primary_target, "path", "unknown")

            # Reconstruct cache key (same as _cache_buffer)
            eligible, _ = self.prepare()
            camera_extrinsics = [
                camera.extrinsics.tobytes()
                for path, camera in eligible
            ]
            camera_set_hash = hashlib.md5(
                b"".join(sorted(camera_extrinsics))
            ).hexdigest().encode()

            weighting_str = "_".join(
                f"{k}={int(v)}" for k, v in sorted(self._weighting_config.items())
            )
            weighting_str = f"{weighting_str}_interp={self._interpolation}"
            weighting_bytes = weighting_str.encode()

            element_type = primary_target.get_element_type()

            buffer = self.cache_manager.load_feature_buffer(
                mesh_path,
                camera_set_hash,
                model_id,
                compressor_kind,
                compressor_dim,
                weighting_bytes,
                element_type=element_type,
            )

            if buffer is None:
                return False

            self.buffer = buffer
            self.query_engine = QueryEngine(buffer.features, buffer.valid)
            self._build_shader_state(buffer)
            primary_target.attach_feature_arrays(buffer)
            self.viewer._update_array_selector()

            return True
        except Exception as e:
            print(f"[FeatureMeshManager] Cache load failed: {e}")
            return False

    def clear(self) -> None:
        """Clear the current buffer and detach from the mesh."""
        self.buffer = None
        self.query_engine = None
        self.shader_state = None

        primary_target = self.viewer.scene_context.get_primary_target()
        if primary_target:
            # Tear the shader off the live mesh actor before the array detaches.
            actor = self._get_mesh_actor(primary_target)
            if actor is not None:
                self.uninstall_shader(actor)
            primary_target.clear_feature_arrays()
            self.viewer._update_array_selector()

        # Disable (and exit) the Feature Select tool, since there's no buffer
        # left to query.
        feature_tool_action = getattr(self.main_window, 'feature_tool_action', None)
        if feature_tool_action is not None:
            if feature_tool_action.isChecked():
                feature_tool_action.setChecked(False)
                self.viewer.set_selected_3d_tool(None)
                
    def recolor_by_similarity(self, sim: Optional[np.ndarray] = None,
                              threshold: Optional[float] = None,
                              hover_id: Optional[int] = None) -> None:
        """
        Recolor the primary target by similarity in place.

        Works across product types: meshes write a per-face "Similarity" RGB
        array (or drive the GPU colormap shader), point clouds write a per-point
        "Similarity" RGB array, and Gaussian splats push the colours straight to
        the splat SH. The per-element display colours are identical in every
        case — only the sink differs (``product.set_similarity_colors``).

        Args:
            sim: [N] float32 similarity scores, or None to use query_engine.similarity().
            threshold: When provided, render a live preview of the thresholded
                selection — only elements with raw similarity >= threshold (exactly
                what QueryEngine.select() would pick) are lit. When None, render
                the full similarity gradient.
            hover_id: optional transient prototype (element under the cursor) to
                fold into the query for live hover preview, without committing it.
        """
        if self.buffer is None or (sim is None and self.query_engine is None):
            return

        primary_target = self.viewer.scene_context.get_primary_target()
        if primary_target is None:
            return

        element_type = getattr(primary_target, "get_element_type", lambda: None)()
        # Only meshes (gl_PrimitiveID) and point clouds (injected gl_VertexID) are
        # rendered through a VTK mapper the colormap shader can replace. Gaussian
        # splats are drawn by their own ModernGL geometry shader (no VTK mapper to
        # patch — maybe_install_shader is never even called for them), so they MUST
        # take the direct-RGB path below, which pushes the colours into the splat
        # SH via set_similarity_colors. Routing splats through the shader path
        # silently no-ops (the disp texture nothing reads gets updated), which is
        # why similarity never appeared on splats.
        is_shader_capable = element_type in ("face", "point")

        # Both meshes and point clouds use the GPU colormap shader path:
        # push the [N] uint8 display value into the shader's value texture (raw
        # upload). ScalarVisibilityOff means VTK never rebuilds the color
        # buffer — the render stays ~5 ms. Splats take the direct-RGB path below.
        if sim is None and is_shader_capable and self._shader_in_play():
            try:
                disp = self.query_engine.display_scalars(threshold, hover_id=hover_id)
                self.shader_state.update_disp(disp)
            except Exception as e:
                self._disable_shader(f"disp update failed: {e}")
            else:
                return
            # On failure the shader is disabled → fall through to the RGB path.

        # Direct-RGB path: used by Gaussian Splats (straight to SH) or as a fallback.
        if sim is None:
            colors = self.query_engine.display_colors(threshold, hover_id=hover_id)
        else:
            self.query_engine._ensure_colormap_lut()
            disp = QueryEngine._disp_unit01_numpy(np.asarray(sim, dtype=np.float32),
                                                  threshold)
            ci = np.clip(np.rint(disp * 255.0), 0, 255).astype(np.int64)
            colors = self.query_engine._cmap_np[ci]

        # Dispatch to the product's similarity sink (mesh cell_data / point_data
        # in place, or the splat SH push).
        sink = getattr(primary_target, "set_similarity_colors", None)
        if callable(sink):
            sink(colors)

    # ------------------------------------------------------------------ #
    # Phase-2 shader plumbing
    # ------------------------------------------------------------------ #
    def _shader_in_play(self) -> bool:
        """True when the GPU shader path is enabled and its artifacts are built."""
        return bool(self.shader_enabled and self.shader_state is not None)

    def _disable_shader(self, reason: str) -> None:
        """Permanently fall back to the uint8 path for this session."""
        print(f"[FeatureMeshManager] shader disabled → uint8 fallback: {reason}")
        self.shader_enabled = False
        self.shader_state = None

    def _build_shader_state(self, buffer: FeatureBuffer) -> None:
        """Build the GPU colormap textures once per bake / cache-load (best-effort)."""
        self.shader_state = None
        if not self.shader_enabled:
            return
        try:
            from coralnet_toolbox.MVAT.shaders.SimilarityShader import build_state
            self.shader_state = build_state(int(buffer.features.shape[0]))
        except Exception as e:
            # Don't kill the whole feature — just note we'll use uint8.
            print(f"[FeatureMeshManager] shader artifacts unavailable: {e}")
            self.shader_state = None

    def _get_mesh_actor(self, primary_target):
        """Resolve the live VTK actor for the primary mesh product, if any."""
        try:
            product_actors = getattr(self.viewer, "_product_actors", {})
            return product_actors.get(getattr(primary_target, "product_id", None))
        except Exception:
            return None

    def maybe_install_shader(self, actor, product) -> None:
        """
        Install the similarity shader on a freshly built actor.

        Called from the viewer after add_mesh()/add_points() whenever the
        Similarity array is active. No-op unless the shader is in play. Any
        failure flips to the uint8 fallback for the rest of the session.
        """
        if actor is None or not self._shader_in_play():
            return

        element_type = getattr(product, "get_element_type", lambda: None)()
        # Splats are excluded: they have no VTK mapper for the colormap shader to
        # replace (they recolor via the SH direct-RGB path), and render_scene
        # short-circuits them before this is ever called.
        if element_type not in ("face", "point"):
            return

        sa = getattr(product, "selected_array", None)
        if sa != "Similarity":
            return
        try:
            from coralnet_toolbox.MVAT.shaders.SimilarityShader import (
                install_similarity_shader,
            )
            install_similarity_shader(actor, self.shader_state, element_type=element_type)
        except Exception as e:
            self._disable_shader(f"install failed: {e}")

    def uninstall_shader(self, actor) -> None:
        """Remove the shader from a mesh actor (tool deactivate / array switch)."""
        if actor is None:
            return
        try:
            from coralnet_toolbox.MVAT.shaders.SimilarityShader import (
                uninstall_similarity_shader,
            )
            uninstall_similarity_shader(actor)
        except Exception as e:
            print(f"[FeatureMeshManager] shader uninstall failed: {e}")


class BakeFeatureDialog(QDialog):
    """Simple dialog to configure and launch a feature buffer bake."""

    def __init__(self, feature_mesh_manager, parent=None):
        super().__init__(parent)
        self.feature_mesh_manager = feature_mesh_manager
        self.setWindowTitle("Bake Mesh Features")
        self.setMinimumWidth(400)

        layout = QVBoxLayout()

        # Precondition check
        eligible, stats = feature_mesh_manager.prepare()
        precond_label = QLabel(
            f"Cameras: {stats['both']}/{stats['with_feature_map']} have both feature & index maps\n"
            f"Missing feature maps: {stats['missing_feature']}\n"
            f"Missing index maps: {stats['missing_index']}"
        )
        layout.addWidget(precond_label)

        if stats["both"] == 0:
            reject_label = QLabel("Cannot proceed: no cameras have both maps loaded.")
            layout.addWidget(reject_label)
            self.setMinimumHeight(200)
            cancel_btn = QPushButton("Close")
            cancel_btn.clicked.connect(self.reject)
            layout.addWidget(cancel_btn)
            self.setLayout(layout)
            return

        # Compressor config
        config_group = QGroupBox("Compressor")
        config_layout = QVBoxLayout()

        comp_layout = QHBoxLayout()
        comp_layout.addWidget(QLabel("Type:"))
        self.compressor_combo = QComboBox()
        self.compressor_combo.addItems(["nn", "pca"])
        self.compressor_combo.setItemData(
            0, "EXPERIMENTAL (default): trains a scene-specific autoencoder. Slower bake "
               "(trains on the main thread, see console log); may separate look-alike "
               "classes better than PCA at small dimensions.", Qt.ToolTipRole
        )
        comp_layout.addWidget(self.compressor_combo)
        config_layout.addLayout(comp_layout)

        dim_layout = QHBoxLayout()
        dim_label = QLabel("Dimension:")
        dim_tooltip = (
            "Compressed feature size per element (default 32).\n"
            "Higher = separates look-alike classes better but uses more memory; "
            "lower = lighter but blurs fine distinctions."
        )
        dim_label.setToolTip(dim_tooltip)
        dim_layout.addWidget(dim_label)
        self.dim_spinbox = QSpinBox()
        self.dim_spinbox.setMinimum(1)
        self.dim_spinbox.setMaximum(512)
        self.dim_spinbox.setValue(32)
        self.dim_spinbox.setToolTip(dim_tooltip)
        dim_layout.addWidget(self.dim_spinbox)
        config_layout.addLayout(dim_layout)

        # Interpolation: how coarse patch features are upsampled to image res.
        interp_layout = QHBoxLayout()
        interp_label = QLabel("Interpolation:")
        interp_tooltip = (
            "How the coarse patch-grid features are sampled up to full image "
            "resolution when projecting onto the mesh.\n\n"
            "• Nearest — each surface point takes its closest patch's feature. "
            "Crisp boundaries, fastest, but blocky.\n"
            "• Bilinear — blends the 4 neighboring patches for smoother feature "
            "transitions across the surface (slightly slower).\n\n"
            "Element IDs are always matched nearest — only the features are "
            "interpolated."
        )
        interp_label.setToolTip(interp_tooltip)
        interp_layout.addWidget(interp_label)
        self.interp_combo = QComboBox()
        self.interp_combo.addItems(["nearest", "bilinear"])
        self.interp_combo.setToolTip(interp_tooltip)
        interp_layout.addWidget(self.interp_combo)
        config_layout.addLayout(interp_layout)

        # NN autoencoder training parameters (only shown when "nn" is selected).
        self.nn_params_widget = QWidget()
        nn_form = QFormLayout(self.nn_params_widget)
        nn_form.setContentsMargins(0, 0, 0, 0)

        self.nn_hidden_spinbox = QSpinBox()
        self.nn_hidden_spinbox.setRange(8, 4096)
        self.nn_hidden_spinbox.setValue(256)
        self.nn_hidden_spinbox.setToolTip(
            "Hidden layer width H of the encoder/decoder. Larger = more capacity.")
        nn_form.addRow("Hidden dim:", self.nn_hidden_spinbox)

        self.nn_epochs_spinbox = QSpinBox()
        self.nn_epochs_spinbox.setRange(1, 1000)
        self.nn_epochs_spinbox.setValue(30)
        self.nn_epochs_spinbox.setToolTip("Number of training epochs over the sampled patches.")
        nn_form.addRow("Epochs:", self.nn_epochs_spinbox)

        self.nn_lr_spinbox = QDoubleSpinBox()
        self.nn_lr_spinbox.setDecimals(5)
        self.nn_lr_spinbox.setRange(1e-5, 1.0)
        self.nn_lr_spinbox.setSingleStep(1e-4)
        self.nn_lr_spinbox.setValue(1e-3)
        self.nn_lr_spinbox.setToolTip("Adam learning rate.")
        nn_form.addRow("Learning rate:", self.nn_lr_spinbox)

        self.nn_beta_spinbox = QDoubleSpinBox()
        self.nn_beta_spinbox.setDecimals(2)
        self.nn_beta_spinbox.setRange(0.0, 100.0)
        self.nn_beta_spinbox.setSingleStep(0.1)
        self.nn_beta_spinbox.setValue(1.0)
        self.nn_beta_spinbox.setToolTip(
            "Weight of the cosine-preservation loss. Higher = prioritize keeping the "
            "compressed space angle-faithful (better for click-to-select queries).")
        nn_form.addRow("Cosine loss β:", self.nn_beta_spinbox)

        config_layout.addWidget(self.nn_params_widget)

        # Toggle NN params visibility with the compressor selection.
        self.compressor_combo.currentTextChanged.connect(self._on_compressor_changed)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # Apply the initial visibility for the default selection.
        self._on_compressor_changed(self.compressor_combo.currentText())

        # Scope
        scope_group = QGroupBox("Camera Scope")
        scope_layout = QVBoxLayout()
        self.scope_combo = QComboBox()
        self.scope_combo.addItems(["all", "selected", "visible"])
        scope_layout.addWidget(self.scope_combo)
        scope_group.setLayout(scope_layout)
        layout.addWidget(scope_group)

        # Weighting options
        weight_group = QGroupBox("Confidence Weighting")
        weight_layout = QVBoxLayout()
        self.angle_check = QCheckBox("View Angle")
        self.angle_check.setChecked(True)
        weight_layout.addWidget(self.angle_check)

        self.dist_check = QCheckBox("Inverse Distance")
        self.dist_check.setChecked(True)
        weight_layout.addWidget(self.dist_check)

        self.edge_check = QCheckBox("Edge Guard")
        self.edge_check.setChecked(False)
        weight_layout.addWidget(self.edge_check)

        weight_group.setLayout(weight_layout)
        layout.addWidget(weight_group)

        # Buttons
        button_layout = QHBoxLayout()
        bake_btn = QPushButton("Bake")
        bake_btn.clicked.connect(self._on_bake_clicked)
        button_layout.addWidget(bake_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def _on_compressor_changed(self, kind: str):
        """Show NN training params only when the NN compressor is selected."""
        self.nn_params_widget.setVisible(kind == "nn")

    def _on_bake_clicked(self):
        """Launch the bake with current settings."""
        compressor_kind = self.compressor_combo.currentText()
        compressor_dim = self.dim_spinbox.value()
        scope = self.scope_combo.currentText()
        interpolation = self.interp_combo.currentText()

        nn_params = None
        if compressor_kind == "nn":
            nn_params = {
                "hidden_dim": self.nn_hidden_spinbox.value(),
                "epochs": self.nn_epochs_spinbox.value(),
                "lr": self.nn_lr_spinbox.value(),
                "beta": self.nn_beta_spinbox.value(),
            }

        # Update weighting config
        self.feature_mesh_manager._weighting_config = {
            "use_angle": self.angle_check.isChecked(),
            "use_inv_dist": self.dist_check.isChecked(),
            "use_edge_guard": self.edge_check.isChecked(),
        }

        # Launch the bake
        self.feature_mesh_manager.bake(
            compressor_kind=compressor_kind,
            compressor_dim=compressor_dim,
            scope=scope,
            interpolation=interpolation,
            nn_params=nn_params,
        )

        self.accept()
