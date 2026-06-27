"""
AnyUp-based feature map upsampler.

Wraps the inference-time, feature-agnostic AnyUp upsampler
(https://github.com/wimmerth/anyup), loaded via torch.hub. Given the
high-resolution RGB image used as model input and a coarse [h, w, C]
feature grid, produces a denser [H, W, C] grid (same C) using the image
as structural guidance.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

# ImageNet normalization stats expected by AnyUp's RGB guidance branch.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Cache loaded AnyUp models, keyed by (variant, use_natten).
_ANYUP_CACHE = {}


class AnyUpUpsampler:
    """Loads and runs the AnyUp upsampler model."""

    def __init__(self, device: str = "cuda",
                 variant: str = "anyup_multi_backbone",
                 use_natten: bool = False):
        self.device = device
        self.variant = variant
        self.use_natten = use_natten
        self._model = None
        self._load()

    def _load(self):
        """Load or retrieve a cached AnyUp model from torch.hub."""
        key = (self.variant, self.use_natten)
        if key in _ANYUP_CACHE:
            self._model = _ANYUP_CACHE[key]
            return

        try:
            model = torch.hub.load(
                "wimmerth/anyup", self.variant, use_natten=self.use_natten,
                trust_repo=True,
            )
            model = model.to(self.device).eval()
            _ANYUP_CACHE[key] = model
            self._model = model
        except Exception as e:
            raise RuntimeError(
                f"Failed to load AnyUp model '{self.variant}' via torch.hub "
                f"(requires internet access on first use): {e}"
            )

    def upsample(self, hr_image_rgb: np.ndarray, lr_features: np.ndarray,
                  out_hw: Tuple[int, int]) -> np.ndarray:
        """
        Upsample a coarse feature grid using the RGB image as guidance.

        Args:
            hr_image_rgb: Guidance image, RGB uint8 [H, W, 3].
            lr_features: Coarse feature grid [h, w, C].
            out_hw: Target (height, width) for the upsampled grid.

        Returns:
            Upsampled feature grid [H_out, W_out, C] as float32.
        """
        H_out, W_out = out_hw
        n_pixels = H_out * W_out
        q_chunk_size = None if n_pixels <= 768 * 768 else 64

        image = hr_image_rgb.astype(np.float32) / 255.0
        mean = np.array(_IMAGENET_MEAN, dtype=np.float32)
        std = np.array(_IMAGENET_STD, dtype=np.float32)
        image = (image - mean) / std
        image_t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)

        feat_t = torch.from_numpy(lr_features.astype(np.float32))
        feat_t = feat_t.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            kwargs = {"output_size": (H_out, W_out)}
            if q_chunk_size is not None:
                kwargs["q_chunk_size"] = q_chunk_size
            out = self._model(image_t, feat_t, **kwargs)

        out = out[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        return out


def clear_cache():
    """Drop all cached AnyUp models and free CUDA memory."""
    global _ANYUP_CACHE
    _ANYUP_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
