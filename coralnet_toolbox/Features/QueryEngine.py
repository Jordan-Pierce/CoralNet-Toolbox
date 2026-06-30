"""
QueryEngine — real-time cosine / linear-head feature-similarity query kernel.

A stateless compute kernel over a dense [N, D] feature buffer: the caller adds
positive / negative prototypes (clicked patches) and the engine returns per-element
similarity, display colors/scalars, and threshold selections.

Originally part of the 3D pipeline; relocated here so the 2D FeatureSelectTool can
reuse it without depending on the MVAT package. Self-contained — numpy on CPU,
torch on GPU when available.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None


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

    def class_scores(self, prototypes_by_class) -> Tuple[np.ndarray, list]:
        """Per-class max-pool cosine similarity over the feature buffer.

        The multi-class counterpart of the binary ``_best_pos`` field: for each
        class, the max cosine of every element to ANY of that class's clicked
        prototypes (the paper's FAISS ``k=1`` nearest-prototype, per class).

        Args:
            prototypes_by_class: mapping ``class_key -> list[element_id]``. Empty
                lists and out-of-range ids are ignored; a class with no surviving
                ids is dropped from the result.

        Returns:
            (best, keys): ``best`` is ``[C, N]`` float32 where row k is the
            similarity field for ``keys[k]``; ``keys`` is the class-key list in
            row order. Both empty when no class has prototypes.
        """
        N = self.features_np.shape[0]
        keys = []
        rows = []
        for key, ids in prototypes_by_class.items():
            valid_ids = [int(i) for i in ids if 0 <= int(i) < N]
            if not valid_ids:
                continue
            keys.append(key)
            if self.use_torch:
                idx = torch.as_tensor(valid_ids, dtype=torch.long, device=self.device)
                protos = self.features_cuda[idx]            # [P, D]
                sims = self.features_cuda @ protos.t()       # [N, P]
                rows.append(sims.max(dim=1).values)          # [N]
            else:
                protos = self.features_np[valid_ids]         # [P, D]
                sims = self.features_np @ protos.T           # [N, P]
                rows.append(sims.max(axis=1))                # [N]

        if not keys:
            return np.zeros((0, N), dtype=np.float32), []
        if self.use_torch:
            best = torch.stack(rows, dim=0).detach().cpu().numpy()
        else:
            best = np.stack(rows, axis=0)
        return best.astype(np.float32, copy=False), keys

    def classify(self, prototypes_by_class, threshold: Optional[float] = None
                 ) -> Tuple[np.ndarray, np.ndarray, list]:
        """Nearest-prototype multi-class labeling over the feature buffer.

        Each element is assigned the class with the highest max-pool cosine
        similarity (see :meth:`class_scores`). Uncovered (``~valid``) elements,
        and elements whose winning similarity is below ``threshold``, map to
        label index ``-1`` (unlabeled).

        Returns:
            (label_field, conf, keys): ``label_field`` [N] int32 indexes into
            ``keys`` (-1 = unlabeled); ``conf`` [N] float32 is the winning
            similarity (0 where unlabeled); ``keys`` is the class-key list.
        """
        best, keys = self.class_scores(prototypes_by_class)
        N = self.features_np.shape[0]
        if not keys:
            return np.full(N, -1, dtype=np.int32), np.zeros(N, dtype=np.float32), []
        label_field = np.argmax(best, axis=0).astype(np.int32)
        conf = np.max(best, axis=0).astype(np.float32)
        reject = ~np.asarray(self.valid, dtype=bool)
        if threshold is not None:
            reject = reject | (conf < float(threshold))
        label_field[reject] = -1
        conf[reject] = 0.0
        return label_field, conf, keys

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
