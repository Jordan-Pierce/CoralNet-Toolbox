"""
Feature buffer artifact and compressors for MVAT Tier 2.

FeatureBuffer holds the per-element feature representation [N, D] (normalized, fp16),
coverage diagnostics, precomputed PCA-RGB visualization, and compressor state for
round-trip persistence.

FeatureCompressor is a protocol that fits on a sample of features and transforms
[K, C] → [K, D]. Implementations: NN autoencoder (default) and PCA.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Protocol

import numpy as np


@dataclass
class FeatureBuffer:
    """
    Per-element feature representation and metadata after baking.

    Attributes:
        features: [N, D] float16, L2-normalized. At query time, loaded as CUDA tensor.
        coverage: [N] float32. Summed weight per element (fusion diagnostic).
        valid: [N] bool. True if coverage > 0 (element was seen by ≥1 camera).
        compressor_state: dict. Serialized compressor (mean, components, etc.) for state round-trip.
        pca_rgb: [N, 3] uint8 or None. Precomputed top-3 PCA dims → RGB for "colorful" view.
        provenance: dict. Metadata: model_id, compressor kind+dim, weighting flags, camera_set_hash,
                    element_type, N, timestamp.
    """
    features: np.ndarray          # [N, D] float16
    coverage: np.ndarray          # [N] float32
    valid: np.ndarray             # [N] bool
    compressor_state: Dict[str, Any]  # for state round-trip
    pca_rgb: Optional[np.ndarray] # [N, 3] uint8 or None
    provenance: Dict[str, Any]    # metadata


class FeatureCompressor(Protocol):
    """Protocol for feature compressors: C-dim → D-dim."""

    kind: str      # "nn" | "pca"
    out_dim: int   # D

    def fit(self, sample_feats: np.ndarray) -> None:
        """Fit on a sample of features [M, C]."""
        ...

    def transform(self, feats: np.ndarray) -> np.ndarray:
        """Transform [K, C] → [K, D]."""
        ...

    def state_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable state for persistence."""
        ...

    def load_state(self, s: Dict[str, Any]) -> None:
        """Restore from saved state."""
        ...


class PCACompressor:
    """
    PCA-based feature compressor using scikit-learn.

    Linear: (feats - mean) @ components.T [optionally / sqrt(var) if whitened].
    Applied per-camera before scatter (cheap, keeps accumulator [N,D] not [N,C]).
    """

    kind = "pca"

    def __init__(self, out_dim: int, whiten: bool = False):
        """
        Args:
            out_dim: Target dimensionality D.
            whiten: If True, divide by sqrt(explained_variance) after projection.
        """
        self.out_dim = out_dim
        self.whiten = whiten
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None

    def fit(self, sample_feats: np.ndarray) -> None:
        """
        Fit PCA on a sample of feature patches.

        Args:
            sample_feats: [M, C] array of feature vectors (M should be capped, e.g. ≤200k).
        """
        from sklearn.decomposition import PCA

        sample_feats = np.asarray(sample_feats, dtype=np.float32)
        if sample_feats.ndim != 2:
            raise ValueError(f"Expected 2-D array; got shape {sample_feats.shape}")

        C = sample_feats.shape[1]
        if self.out_dim > C:
            raise ValueError(f"out_dim {self.out_dim} > C {C}")

        pca = PCA(n_components=self.out_dim, whiten=False)
        pca.fit(sample_feats)

        self.mean_ = pca.mean_.astype(np.float32)
        self.components_ = pca.components_.astype(np.float32)  # [D, C]
        self.explained_variance_ = pca.explained_variance_.astype(np.float32) if self.whiten else None

    def transform(self, feats: np.ndarray) -> np.ndarray:
        """
        Project features [K, C] → [K, D].

        Args:
            feats: [K, C] feature array.

        Returns:
            [K, D] float32 transformed features.
        """
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCACompressor not fitted; call fit() first")

        feats = np.asarray(feats, dtype=np.float32)
        centered = feats - self.mean_
        projected = centered @ self.components_.T  # [K, D]

        if self.whiten and self.explained_variance_ is not None:
            var_safe = np.maximum(self.explained_variance_, 1e-8)
            projected = projected / np.sqrt(var_safe)

        return projected.astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable state."""
        return {
            "kind": self.kind,
            "out_dim": int(self.out_dim),
            "whiten": bool(self.whiten),
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "components": self.components_.tolist() if self.components_ is not None else None,
            "explained_variance": (
                self.explained_variance_.tolist()
                if self.explained_variance_ is not None else None
            ),
        }

    def load_state(self, s: Dict[str, Any]) -> None:
        """Restore from saved state."""
        self.kind = s.get("kind", "pca")
        self.out_dim = int(s.get("out_dim", 32))
        self.whiten = bool(s.get("whiten", False))
        self.mean_ = (
            np.array(s["mean"], dtype=np.float32)
            if s.get("mean") is not None else None
        )
        self.components_ = (
            np.array(s["components"], dtype=np.float32)
            if s.get("components") is not None else None
        )
        self.explained_variance_ = (
            np.array(s["explained_variance"], dtype=np.float32)
            if s.get("explained_variance") is not None else None
        )


class NNCompressor:
    """
    Scene-specific autoencoder compressor (EXPERIMENTAL).

    Trains a tiny undercomplete autoencoder C -> H -> D -> H -> C on a sample of the
    scene's patch features. Only the encoder (C -> D) is used at transform time; the
    decoder provides the reconstruction training signal.

    Loss = reconstruction MSE + beta * cosine-preservation, the latter penalizing the
    difference between input-space and bottleneck-space pairwise cosine similarity. This
    keeps the learned D-dim space angle-faithful so downstream cosine queries stay valid.

    The aggregated per-element buffer is L2-normalized in the bake's finalize step, so
    transform() does NOT normalize here (that would corrupt the weighted mean).

    Training runs synchronously (caller is responsible for a busy cursor); progress is
    logged to stdout.
    """

    kind = "nn"

    def __init__(self, out_dim: int, hidden_dim: int = 256, epochs: int = 30,
                 lr: float = 1e-3, batch_size: int = 4096, beta: float = 1.0,
                 seed: int = 0):
        """
        Args:
            out_dim: Bottleneck dimensionality D.
            hidden_dim: Hidden layer width H for both encoder and decoder.
            epochs: Number of training epochs.
            lr: Adam learning rate.
            batch_size: Minibatch size.
            beta: Weight of the cosine-preservation loss term.
            seed: RNG seed for reproducible training on a fixed sample.
        """
        self.out_dim = out_dim
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.beta = float(beta)
        self.seed = int(seed)

        self.input_dim = None      # C, set at fit time
        self._encoder = None       # torch.nn.Module (built lazily)
        self._device = "cpu"
        # Persisted encoder weights as nested lists (JSON-serializable).
        self._encoder_state = None

    # -- internals ---------------------------------------------------------

    def _build_modules(self, C: int):
        """Build encoder + decoder modules on the selected device."""
        import torch
        from torch import nn

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        H, D = self.hidden_dim, self.out_dim

        encoder = nn.Sequential(
            nn.Linear(C, H), nn.GELU(), nn.Linear(H, D)
        ).to(device)
        decoder = nn.Sequential(
            nn.Linear(D, H), nn.GELU(), nn.Linear(H, C)
        ).to(device)
        return encoder, decoder

    def fit(self, sample_feats: np.ndarray) -> None:
        """
        Train the autoencoder on a sample of feature patches [M, C].

        Logs per-epoch loss to stdout. Caller should set a busy cursor.
        """
        import torch
        from torch import nn

        sample_feats = np.asarray(sample_feats, dtype=np.float32)
        if sample_feats.ndim != 2:
            raise ValueError(f"Expected 2-D array; got shape {sample_feats.shape}")

        # Drop non-finite rows.
        finite = np.isfinite(sample_feats).all(axis=1)
        sample_feats = sample_feats[finite]

        M, C = sample_feats.shape
        if self.out_dim >= C:
            raise ValueError(
                f"NNCompressor requires out_dim ({self.out_dim}) < C ({C}); "
                f"use Identity for D == C."
            )
        if M < 5 * C:
            print(
                f"[NNCompressor] WARNING: small sample (M={M} < 5*C={5 * C}); "
                f"the autoencoder may overfit. Consider PCA."
            )

        self.input_dim = C
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        encoder, decoder = self._build_modules(C)
        device = self._device
        X = torch.as_tensor(sample_feats, dtype=torch.float32, device=device)

        params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        mse = nn.MSELoss()

        batch_size = min(self.batch_size, M)
        n_batches = max(1, M // batch_size)

        print(
            f"[NNCompressor] Training: M={M}, C={C}, D={self.out_dim}, H={self.hidden_dim}, "
            f"epochs={self.epochs}, lr={self.lr}, batch={batch_size}, beta={self.beta}, "
            f"device={device}"
        )
        _t0 = time.perf_counter()

        encoder.train()
        decoder.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(M, device=device)
            ep_recon = 0.0
            ep_cos = 0.0
            for b in range(n_batches):
                idx = perm[b * batch_size:(b + 1) * batch_size]
                xb = X[idx]                            # [B, C] (already L2-normalized)
                z = encoder(xb)                        # [B, D]
                xr = decoder(z)                        # [B, C]

                recon = mse(xr, xb)

                # Cosine-preservation: compare input cosine vs bottleneck cosine over a
                # shuffled pairing within the batch.
                shuffle = torch.randperm(xb.shape[0], device=device)
                cos_in = (xb * xb[shuffle]).sum(dim=1)              # inputs are unit-norm
                z_n = torch.nn.functional.normalize(z, dim=1)
                cos_out = (z_n * z_n[shuffle]).sum(dim=1)
                cos_loss = ((cos_out - cos_in) ** 2).mean()

                loss = recon + self.beta * cos_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ep_recon += float(recon.detach())
                ep_cos += float(cos_loss.detach())

            print(
                f"[NNCompressor]   epoch {epoch + 1:3d}/{self.epochs}  "
                f"recon={ep_recon / n_batches:.5f}  cos={ep_cos / n_batches:.5f}"
            )

        encoder.eval()
        self._encoder = encoder
        # Snapshot encoder weights as JSON-able lists for persistence.
        self._encoder_state = {
            k: v.detach().cpu().tolist() for k, v in encoder.state_dict().items()
        }
        print(f"[NNCompressor] Training done in {time.perf_counter() - _t0:.2f}s")

    def transform(self, feats: np.ndarray) -> np.ndarray:
        """
        Encode features [K, C] -> [K, D] (float32). Non-finite rows are zeroed.
        """
        import torch

        if self._encoder is None:
            self._rebuild_encoder()
        if self._encoder is None:
            raise RuntimeError("NNCompressor not fitted; call fit() first")

        feats = np.asarray(feats, dtype=np.float32)
        bad = ~np.isfinite(feats).all(axis=1)
        if bad.any():
            feats = feats.copy()
            feats[bad] = 0.0

        with torch.inference_mode():
            x = torch.as_tensor(feats, dtype=torch.float32, device=self._device)
            z = self._encoder(x).cpu().numpy().astype(np.float32)

        z[~np.isfinite(z).all(axis=1)] = 0.0
        return z

    def _rebuild_encoder(self):
        """Reconstruct the encoder module from persisted weights (after load_state)."""
        if self._encoder_state is None or self.input_dim is None:
            return
        import torch
        encoder, _decoder = self._build_modules(self.input_dim)
        state = {
            k: torch.as_tensor(np.array(v, dtype=np.float32), device=self._device)
            for k, v in self._encoder_state.items()
        }
        encoder.load_state_dict(state)
        encoder.eval()
        self._encoder = encoder

    def state_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable state (hyperparameters + encoder weights)."""
        return {
            "kind": self.kind,
            "out_dim": int(self.out_dim),
            "input_dim": int(self.input_dim) if self.input_dim is not None else None,
            "hidden_dim": int(self.hidden_dim),
            "epochs": int(self.epochs),
            "lr": float(self.lr),
            "batch_size": int(self.batch_size),
            "beta": float(self.beta),
            "seed": int(self.seed),
            "encoder_state": self._encoder_state,
        }

    def load_state(self, s: Dict[str, Any]) -> None:
        """Restore from saved state and rebuild the encoder."""
        self.kind = s.get("kind", "nn")
        self.out_dim = int(s.get("out_dim", 32))
        self.input_dim = s.get("input_dim")
        self.hidden_dim = int(s.get("hidden_dim", 256))
        self.epochs = int(s.get("epochs", 30))
        self.lr = float(s.get("lr", 1e-3))
        self.batch_size = int(s.get("batch_size", 4096))
        self.beta = float(s.get("beta", 1.0))
        self.seed = int(s.get("seed", 0))
        self._encoder_state = s.get("encoder_state")
        self._encoder = None
        self._rebuild_encoder()
