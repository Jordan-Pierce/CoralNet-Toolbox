"""
Unified feature extraction system for transformers, YOLO, and color features.

Provides a single FeatureExtractor interface with model caching, dense extraction,
and pooled extraction suitable for both Raster products and Explorer embeddings.
"""

from __future__ import annotations

import gc
import warnings
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod

import numpy as np
import cv2
import torch
from PIL import Image

from coralnet_toolbox.Features.ModelRegistry import (
    TRANSFORMER_MODELS,
    YOLO_MODELS,
    is_transformer_model,
    is_yolo_model,
    is_timm_model,
    strip_timm_prefix,
    is_openclip_model,
    strip_openclip_prefix,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


def model_supports_dense(model_name: str) -> bool:
    """Whether a model can produce dense feature maps, *without loading it*.

    Used to gate the Features (Tier-1) UI to dense-capable backbones. YOLO and
    Color models are pooled-only (image-level embeddings for the Explorer); only
    transformer backbones (ViT token grids + ConvNext/ResNet spatial maps)
    produce the dense [h, w, C] maps that feature maps / the 3D Features tool
    require.
    """
    if not model_name or model_name == "Color Features":
        return False
    if is_openclip_model(model_name):
        return True
    if is_timm_model(model_name):
        return True
    if is_yolo_model(model_name):
        return False
    return is_transformer_model(model_name)


# Global model caches (one per model name to avoid reloading)
_TRANSFORMER_MODEL_CACHE = {}
_YOLO_MODEL_CACHE = {}
_TIMM_MODEL_CACHE = {}
_OPENCLIP_MODEL_CACHE = {}


class BaseExtractor(ABC):
    """Abstract base for feature extractors."""

    @abstractmethod
    def extract_dense(
        self,
        image_rgb: np.ndarray,
        *,
        out_hw: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Extract dense feature map [h, w, C] at patch-grid resolution.

        Args:
            image_rgb: Input image as RGB uint8 [H, W, 3].
            out_hw: Optional target (height, width). If None, use native patch grid.

        Returns:
            Dense feature map [h, w, C] as float16, L2-normalized.
        """
        pass

    @abstractmethod
    def extract_pooled(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Extract pooled feature vector [C] by global mean pooling.

        Args:
            image_rgb: Input image as RGB uint8 [H, W, 3].

        Returns:
            Pooled vector [C] as float16, L2-normalized.
        """
        pass

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Number of feature channels C."""
        pass

    @property
    @abstractmethod
    def patch_stride(self) -> Optional[int]:
        """Spatial stride of the feature map (16 for ViT/16, None for global-only)."""
        pass

    @property
    def supports_dense(self) -> bool:
        """Whether this backend can produce dense [h, w, C] feature maps."""
        return False


class TransformerExtractor(BaseExtractor):
    """DINOv2/v3 and other HF transformer-based extractors via pipeline."""

    def __init__(self, model_name: str, device: str = "cuda",
                 input_size: Optional[int] = None,
                 upsample_factor: Optional[int] = None):
        self.model_name = model_name
        self.device = device
        # Target square input edge in pixels. None → use the model's default
        # processor resolution (small, e.g. ~224 → coarse grid). A larger value
        # yields a proportionally larger patch grid (e.g. 768 → ~48×48 for /16).
        self._input_size = int(input_size) if input_size else None
        # Optional AnyUp densification factor (2, 4, or 8) applied to the
        # native patch grid, capped at the model's input resolution.
        self._upsample_factor = int(upsample_factor) if upsample_factor else None
        self._anyup = None
        self._channels = None
        self._stride = None
        self._pipeline = None
        self._load_model()

    def _load_anyup(self):
        """Lazily load the AnyUp upsampler (requires internet on first use)."""
        if self._anyup is None:
            from coralnet_toolbox.Features.Models.AnyUpUpsampler import AnyUpUpsampler
            self._anyup = AnyUpUpsampler(device=self.device)
        return self._anyup

    def _load_model(self):
        """Load or retrieve cached transformer model."""
        global _TRANSFORMER_MODEL_CACHE

        if self.model_name in _TRANSFORMER_MODEL_CACHE:
            self._pipeline = _TRANSFORMER_MODEL_CACHE[self.model_name]
            self._infer_architecture()
            self._configure_resolution()
            return

        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "image-feature-extraction",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                trust_remote_code=True,
            )
            _TRANSFORMER_MODEL_CACHE[self.model_name] = self._pipeline
            self._infer_architecture()
            self._configure_resolution()
        except Exception as e:
            raise RuntimeError(f"Failed to load transformer {self.model_name}: {e}")

    def _configure_resolution(self):
        """Override the pipeline's image processor to honor ``_input_size``.

        We pre-resize the image to a fixed square ourselves (in extract_dense),
        so the processor's own resize/center-crop is disabled here. That makes
        the resulting patch grid deterministic across the various HF image
        processors (which use different ``size``/``crop_size`` conventions).
        """
        if not self._input_size or self._pipeline is None:
            return
        try:
            proc = getattr(self._pipeline, "image_processor", None)
            if proc is None:
                return
            S = int(self._input_size)
            if hasattr(proc, "do_resize"):
                proc.do_resize = False
            if hasattr(proc, "do_center_crop"):
                proc.do_center_crop = False
            # Belt-and-suspenders for processors that still consult size dicts.
            if hasattr(proc, "size") and isinstance(getattr(proc, "size"), dict):
                proc.size = ({"shortest_edge": S}
                             if "shortest_edge" in proc.size
                             else {"height": S, "width": S})
            if hasattr(proc, "crop_size") and isinstance(getattr(proc, "crop_size"), dict):
                proc.crop_size = {"height": S, "width": S}
        except Exception as e:
            print(f"[FeatureExtractor] resolution config failed: {e}")

    def _infer_architecture(self):
        """Infer output channels and patch stride from model name."""
        # DINOv2 models have known architectures
        if "dinov2" in self.model_name.lower():
            if "small" in self.model_name.lower():
                self._channels = 384
            elif "base" in self.model_name.lower():
                self._channels = 768
            elif "large" in self.model_name.lower():
                self._channels = 1024
            elif "giant" in self.model_name.lower():
                self._channels = 1536
            self._stride = 16  # All ViT/16
        # DINOv3 models
        elif "dinov3" in self.model_name.lower():
            if "convnext" in self.model_name.lower():
                if "tiny" in self.model_name.lower():
                    self._channels = 768
                elif "small" in self.model_name.lower():
                    self._channels = 1024
                elif "base" in self.model_name.lower():
                    self._channels = 1280
                elif "large" in self.model_name.lower():
                    self._channels = 1600
                self._stride = None  # ConvNext spatial; query at runtime
            else:  # ViT
                if "small" in self.model_name.lower():
                    self._channels = 384
                elif "base" in self.model_name.lower():
                    self._channels = 768
                elif "large" in self.model_name.lower():
                    self._channels = 1024
                elif "7b" in self.model_name.lower():
                    self._channels = 2560
                self._stride = 16
        # ConvNeXt V2 (plain, non-DINOv3). Channels-first [C, H, W] spatial map,
        # so the forward-pass fallback can't be used (it reads shape[-1] = width,
        # not channels). Pin the final-stage dim by size; stride is recomputed
        # from the actual grid in extract_dense.
        elif "convnextv2" in self.model_name.lower():
            if "nano" in self.model_name.lower():
                self._channels = 640
            elif "tiny" in self.model_name.lower():
                self._channels = 768
            elif "base" in self.model_name.lower():
                self._channels = 1024
            self._stride = None
        # Fallback: infer from test forward pass
        else:
            self._infer_from_forward_pass()

    def _infer_from_forward_pass(self):
        """Run a small test image to infer architecture."""
        try:
            test_img = np.zeros((224, 224, 3), dtype=np.uint8)
            with torch.inference_mode():
                result = self._pipeline(Image.fromarray(test_img))
            result_arr = (
                result[0].cpu().numpy()
                if isinstance(result, (list, tuple))
                else result.cpu().numpy()
            )
            self._channels = result_arr.shape[-1]
            self._stride = 16  # Default to ViT/16
        except Exception:
            self._channels = 768
            self._stride = 16

    @staticmethod
    def _split_vit_tokens(feat: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Drop leading special tokens and return (patch_tokens, grid_h, grid_w).

        The HF image-feature-extraction pipeline resizes inputs to a fixed
        square resolution, so the remaining patch tokens form a square grid.
        The only unknown is how many leading special tokens to drop:
          - 1: CLS only (ViT, DINOv2, CLIP/BioCLIP)
          - 5: CLS + 4 DINOv3 register tokens
          - 0: no special tokens (Swin and other hierarchical/no-CLS backbones,
               whose sequence is exactly the square patch grid)
        Ordered so existing CLS/register models keep their prior behavior; the
        no-CLS (0) case is only reached when neither 1 nor 5 yields a square.
        """
        n_tokens = feat.shape[0]
        for n_special in (1, 5, 0):
            n_patch = n_tokens - n_special
            side = int(round(n_patch ** 0.5))
            if n_patch > 0 and side * side == n_patch:
                return feat[n_special:], side, side
        # Fallback: assume CLS-only and a square grid (best effort)
        side = max(1, int(round((n_tokens - 1) ** 0.5)))
        return feat[1:], side, side

    def extract_dense(
        self,
        image_rgb: np.ndarray,
        *,
        out_hw: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Extract dense features [h, w, C].

        Handles two output layouts the HF ``image-feature-extraction`` pipeline
        can emit:
          - **ViT token sequence** ``[T, C]`` (CLS/register tokens + patch grid)
            → drop special tokens, reshape to the square patch grid.
          - **CNN spatial map** ``[C, H, W]`` (ConvNext, ResNet) → transpose the
            channels-first map to ``[H, W, C]``.
        """
        try:
            # Feed a fixed square at the requested resolution (the processor's
            # own resize/crop was disabled in _configure_resolution). Larger
            # input → larger/denser patch grid.
            model_input = image_rgb
            if self._input_size:
                S = int(self._input_size)
                model_input = cv2.resize(
                    image_rgb, (S, S), interpolation=cv2.INTER_AREA
                )

            with torch.inference_mode():
                result = self._pipeline(Image.fromarray(model_input))

            # Convert result to numpy
            if isinstance(result, (list, tuple)):
                feat = result[0]
            else:
                feat = result

            if hasattr(feat, "cpu"):
                feat = feat.cpu().numpy()
            else:
                feat = np.array(feat)

            # Strip any leading singleton batch dims: [1, T, C] -> [T, C],
            # [1, C, H, W] -> [C, H, W].
            while feat.ndim > 3 and feat.shape[0] == 1:
                feat = feat[0]
            if feat.ndim == 3 and feat.shape[0] == 1:
                feat = feat[0]

            name = self.model_name.lower()
            is_spatial_cnn = ("convnext" in name) or ("resnet" in name)
            C = self.out_channels

            if feat.ndim == 3 and (is_spatial_cnn or feat.shape[0] == C):
                # CNN spatial, channels-first [C, H, W] -> [H, W, C].
                feat = np.transpose(feat, (1, 2, 0))
                h_patches, w_patches = feat.shape[0], feat.shape[1]
                self._stride = max(1, round(image_rgb.shape[0] / h_patches))
            elif feat.ndim == 3 and feat.shape[-1] == C:
                # Already spatial [H, W, C].
                h_patches, w_patches = feat.shape[0], feat.shape[1]
                self._stride = max(1, round(image_rgb.shape[0] / h_patches))
            elif feat.ndim == 2:
                # ViT token sequence [T, C]. The pipeline resizes inputs to a
                # fixed square, so the patch grid is square; infer it from the
                # token count (not image_rgb.shape).
                feat, h_patches, w_patches = self._split_vit_tokens(feat)
                feat = feat.reshape(h_patches, w_patches, -1)
                self._stride = max(1, round(image_rgb.shape[0] / h_patches))
            else:
                raise ValueError(f"Unexpected feature shape: {feat.shape}")

            # Optionally densify the patch grid with AnyUp, using the resized
            # model input as RGB guidance. Capped at the model input
            # resolution so memory stays bounded regardless of C.
            if self._upsample_factor:
                try:
                    target_h = min(h_patches * self._upsample_factor, model_input.shape[0])
                    target_w = min(w_patches * self._upsample_factor, model_input.shape[1])
                    anyup = self._load_anyup()
                    feat = anyup.upsample(model_input, feat, (target_h, target_w))
                    h_patches, w_patches = target_h, target_w
                    self._stride = max(1, round(image_rgb.shape[0] / h_patches))
                except Exception as e:
                    print(f"AnyUp upsampling failed, using native patch grid: {e}")

            # L2 normalize over the channel axis
            feat = feat.astype(np.float32)
            norms = np.linalg.norm(feat, axis=-1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            feat = feat / norms

            # Convert to float16
            feat = feat.astype(np.float16)
            return feat

        except Exception as e:
            print(f"Dense extraction failed: {e}")
            return np.zeros((1, 1, self.out_channels), dtype=np.float16)

    def extract_pooled(self, image_rgb: np.ndarray) -> np.ndarray:
        """Extract pooled vector [C] by mean pooling over spatial dims."""
        dense = self.extract_dense(image_rgb)
        pooled = np.mean(dense, axis=(0, 1))  # Mean over h, w
        return pooled.astype(np.float16)

    @property
    def out_channels(self) -> int:
        return self._channels or 768

    @property
    def patch_stride(self) -> Optional[int]:
        return self._stride

    @property
    def supports_dense(self) -> bool:
        return True


class TimmExtractor(BaseExtractor):
    """timm backbone extractor (EVA-02, FastViT, EfficientViT, ...).

    Pulls the last-stage spatial map via ``forward_intermediates`` (timm >= 1.0)
    in a uniform ``[B, C, H, W]`` layout for both ViT and CNN backbones, so there
    is no special-token bookkeeping. Two things differ from the HF pipeline path:
      - timm does NOT normalize inputs internally, so we apply each model's own
        mean/std (resolved from its data config).
      - channel count is taken from an actual forward probe, not
        ``model.num_features`` (which can disagree, e.g. FastViT reports 1024 but
        its last spatial map is 512).
    """

    def __init__(self, model_name: str, device: str = "cuda",
                 input_size: Optional[int] = None,
                 upsample_factor: Optional[int] = None):
        # model_name arrives WITHOUT the "timm:" prefix (stripped by the caller).
        self.model_name = model_name
        self.device = device
        self._input_size = int(input_size) if input_size else None
        self._upsample_factor = int(upsample_factor) if upsample_factor else None
        self._anyup = None
        self._model = None
        self._mean = None
        self._std = None
        self._channels = None
        self._native_size = 224
        # Input edge must be a multiple of this (the ViT/EVA patch size, e.g. 14
        # or 16). patch14 backbones HARD-ASSERT on non-divisible inputs, and none
        # of the dialog's resolutions are multiples of 14 — so we snap. 1 = no
        # constraint (CNN backbones tolerate arbitrary sizes).
        self._size_multiple = 1
        self._stride = None
        self._load_model()

    def _load_anyup(self):
        """Lazily load the AnyUp upsampler (requires internet on first use)."""
        if self._anyup is None:
            from coralnet_toolbox.Features.Models.AnyUpUpsampler import AnyUpUpsampler
            self._anyup = AnyUpUpsampler(device=self.device)
        return self._anyup

    def _load_model(self):
        """Load or retrieve a cached timm model and probe its true channel dim."""
        global _TIMM_MODEL_CACHE

        if self.model_name in _TIMM_MODEL_CACHE:
            (self._model, self._mean, self._std, self._channels,
             self._native_size, self._size_multiple) = _TIMM_MODEL_CACHE[self.model_name]
            return

        try:
            import timm

            # dynamic_img_size lets ViTs run at arbitrary input resolution
            # (interpolated position embeddings). CNN backbones don't accept the
            # kwarg, so fall back to a plain create_model for them.
            try:
                model = timm.create_model(
                    self.model_name, pretrained=True, num_classes=0,
                    dynamic_img_size=True,
                )
            except TypeError:
                model = timm.create_model(self.model_name, pretrained=True, num_classes=0)

            model.eval()
            if self.device == "cuda":
                model = model.cuda()

            cfg = timm.data.resolve_model_data_config(model)
            self._mean = np.asarray(cfg["mean"], dtype=np.float32).reshape(1, 1, 3)
            self._std = np.asarray(cfg["std"], dtype=np.float32).reshape(1, 1, 3)
            self._native_size = int(cfg["input_size"][-1])

            # Required input multiple = the patch size for ViT/EVA backbones
            # (patch_embed.patch_size); CNNs have no such attribute → 1.
            patch_embed = getattr(model, "patch_embed", None)
            patch_size = getattr(patch_embed, "patch_size", None)
            if isinstance(patch_size, (tuple, list)):
                self._size_multiple = int(patch_size[0])
            elif isinstance(patch_size, int):
                self._size_multiple = int(patch_size)
            else:
                self._size_multiple = 1

            # Probe the true last-stage channel count from a real forward pass:
            # forward_intermediates' last map is what extract_dense consumes, and
            # it can differ from model.num_features (FastViT). Snap the probe size
            # so patch14 backbones don't assert.
            probe_S = self._snap_size(self._native_size)
            probe = np.zeros((1, 3, probe_S, probe_S), dtype=np.float32)
            with torch.inference_mode():
                probe_t = torch.from_numpy(probe)
                if self.device == "cuda":
                    probe_t = probe_t.cuda()
                outs = model.forward_intermediates(
                    probe_t, indices=[-1], output_fmt="NCHW",
                    intermediates_only=True, norm=True
                )
            self._channels = int(outs[-1].shape[1])

            self._model = model
            _TIMM_MODEL_CACHE[self.model_name] = (
                model, self._mean, self._std, self._channels,
                self._native_size, self._size_multiple
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load timm model {self.model_name}: {e}")

    def _snap_size(self, size: int) -> int:
        """Round an input edge down to a multiple of the patch size.

        patch14 ViT/EVA backbones assert on non-divisible inputs, and none of the
        dialog resolutions are multiples of 14, so we snap (e.g. 768 -> 756).
        """
        m = self._size_multiple
        if m <= 1:
            return int(size)
        return max(m, (int(size) // m) * m)

    def extract_dense(
        self,
        image_rgb: np.ndarray,
        *,
        out_hw: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Extract dense features [h, w, C] from the last-stage spatial map."""
        try:
            S = self._snap_size(self._input_size or self._native_size)
            model_input = cv2.resize(image_rgb, (S, S), interpolation=cv2.INTER_AREA)

            # timm does not normalize internally — apply the model's mean/std.
            x = model_input.astype(np.float32) / 255.0
            x = (x - self._mean) / self._std
            x = np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None])  # [1, 3, S, S]
            tensor = torch.from_numpy(x)
            if self.device == "cuda":
                tensor = tensor.cuda()

            with torch.inference_mode():
                outs = self._model.forward_intermediates(
                    tensor, indices=[-1], output_fmt="NCHW",
                    intermediates_only=True, norm=True
                )
            # [1, C, h, w] -> [h, w, C]
            feat = outs[-1][0].float().cpu().numpy()
            feat = np.transpose(feat, (1, 2, 0))
            h_patches, w_patches = feat.shape[0], feat.shape[1]
            self._stride = max(1, round(image_rgb.shape[0] / h_patches))

            # Optional AnyUp densification, mirroring TransformerExtractor.
            if self._upsample_factor:
                try:
                    target_h = min(h_patches * self._upsample_factor, model_input.shape[0])
                    target_w = min(w_patches * self._upsample_factor, model_input.shape[1])
                    anyup = self._load_anyup()
                    feat = anyup.upsample(model_input, feat, (target_h, target_w))
                    h_patches, w_patches = target_h, target_w
                    self._stride = max(1, round(image_rgb.shape[0] / h_patches))
                except Exception as e:
                    print(f"AnyUp upsampling failed, using native patch grid: {e}")

            # L2 normalize over channels, return float16.
            feat = feat.astype(np.float32)
            norms = np.linalg.norm(feat, axis=-1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            feat = feat / norms
            return feat.astype(np.float16)

        except Exception as e:
            print(f"Dense extraction failed: {e}")
            return np.zeros((1, 1, self.out_channels), dtype=np.float16)

    def extract_pooled(self, image_rgb: np.ndarray) -> np.ndarray:
        """Extract pooled vector [C] by mean pooling over spatial dims."""
        dense = self.extract_dense(image_rgb)
        pooled = np.mean(dense, axis=(0, 1))
        return pooled.astype(np.float16)

    @property
    def out_channels(self) -> int:
        return self._channels or 768

    @property
    def patch_stride(self) -> Optional[int]:
        return self._stride

    @property
    def supports_dense(self) -> bool:
        return True


class YOLOExtractor(BaseExtractor):
    """YOLO-based feature extractor."""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._channels = None
        self._load_model()

    def _load_model(self):
        """Load or retrieve cached YOLO model."""
        global _YOLO_MODEL_CACHE

        if self.model_name in _YOLO_MODEL_CACHE:
            self._model = _YOLO_MODEL_CACHE[self.model_name]
            return

        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
            _YOLO_MODEL_CACHE[self.model_name] = self._model
            # Infer channels from model
            self._channels = 1024  # Typical YOLO embedding dim
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model {self.model_name}: {e}")

    def extract_dense(
        self,
        image_rgb: np.ndarray,
        *,
        out_hw: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """YOLO doesn't support dense maps; not implemented."""
        raise NotImplementedError(
            "YOLO extractor does not support dense extraction. Use extract_pooled."
        )

    def extract_pooled(self, image_rgb: np.ndarray) -> np.ndarray:
        """Extract pooled embedding vector."""
        try:
            kwargs = {
                "imgsz": 224,
                "quantize": self.device == "cuda",
                "device": self.device if self.device != "cpu" else "cpu",
                "verbose": False,
            }
            with torch.inference_mode():
                results = self._model.embed(image_rgb, **kwargs)

            if isinstance(results, list):
                result = results[0]
            else:
                result = results

            feat = result.cpu().numpy().flatten() if hasattr(result, "cpu") else np.array(result).flatten()

            # L2 normalize
            feat = feat.astype(np.float32)
            norm = np.linalg.norm(feat)
            norm = max(norm, 1e-12)
            feat = feat / norm

            return feat.astype(np.float16)

        except Exception as e:
            print(f"YOLO pooled extraction failed: {e}")
            return np.zeros(self.out_channels, dtype=np.float16)

    @property
    def out_channels(self) -> int:
        return self._channels or 1024

    @property
    def patch_stride(self) -> Optional[int]:
        return None  # YOLO only supports pooled


class ColorExtractor(BaseExtractor):
    """HSV-based color feature extraction."""

    def extract_dense(
        self,
        image_rgb: np.ndarray,
        *,
        out_hw: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Color extraction only supports pooled; not implemented."""
        raise NotImplementedError(
            "Color extractor does not support dense extraction. Use extract_pooled."
        )

    def extract_pooled(self, image_rgb: np.ndarray) -> np.ndarray:
        """Extract HSV color histogram + moments."""
        try:
            hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

            # 2D Hue-Saturation histogram
            hist = cv2.calcHist(
                [hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]
            )
            hist = hist.flatten().astype(np.float32)
            hist /= max(hist.sum(), 1e-7)

            # Saturation and Value moments
            s_channel = hsv[..., 1].astype(np.float32) / 255.0
            v_channel = hsv[..., 2].astype(np.float32) / 255.0

            s_mean = np.mean(s_channel)
            s_std = np.std(s_channel)
            v_mean = np.mean(v_channel)
            v_std = np.std(v_channel)

            # Concatenate: 256 (HS hist) + 4 (moments) = 260 dims
            feat = np.concatenate([[s_mean, s_std, v_mean, v_std], hist]).astype(np.float32)

            # L2 normalize
            norm = np.linalg.norm(feat)
            norm = max(norm, 1e-12)
            feat = feat / norm

            return feat.astype(np.float16)

        except Exception as e:
            print(f"Color extraction failed: {e}")
            return np.zeros(self.out_channels, dtype=np.float16)

    @property
    def out_channels(self) -> int:
        return 260  # 256 (HS hist) + 4 (moments)

    @property
    def patch_stride(self) -> Optional[int]:
        return None


class OpenCLIPExtractor(BaseExtractor):
    """open_clip-based extractor for CLIP variants (BioCLIP, MobileCLIP, etc.).

    Uses ``open_clip.create_model_and_transforms()`` for loading.  Supports both
    ViT (patch-token) and CNN/hybrid (spatial-map) vision encoders via a forward
    hook on the last block/stage of the visual encoder.
    """

    def __init__(self, model_name: str, device: str = "cuda",
                 input_size: Optional[int] = None,
                 upsample_factor: Optional[int] = None):
        self.model_name = model_name
        self.device = device
        self._input_size = int(input_size) if input_size else None
        self._upsample_factor = int(upsample_factor) if upsample_factor else None
        self._anyup = None
        self._model = None
        self._preprocess = None
        self._channels = None
        self._stride = None
        self._is_vit = None
        self._hook_layer = None
        self._load_model()

    def _load_anyup(self):
        if self._anyup is None:
            from coralnet_toolbox.Features.Models.AnyUpUpsampler import AnyUpUpsampler
            self._anyup = AnyUpUpsampler(device=self.device)
        return self._anyup

    def _load_model(self):
        global _OPENCLIP_MODEL_CACHE

        if self.model_name in _OPENCLIP_MODEL_CACHE:
            cached = _OPENCLIP_MODEL_CACHE[self.model_name]
            self._model = cached['model']
            self._preprocess = cached['preprocess']
            self._channels = cached['channels']
            self._stride = cached['stride']
            self._is_vit = cached['is_vit']
            self._hook_layer = cached['hook_layer']
            return

        import open_clip

        if self.model_name.startswith('hf-hub:'):
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, device=self.device,
            )
        else:
            pretrained = self._resolve_pretrained(self.model_name)
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=pretrained, device=self.device,
            )

        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._probe_architecture()

        _OPENCLIP_MODEL_CACHE[self.model_name] = {
            'model': self._model,
            'preprocess': self._preprocess,
            'channels': self._channels,
            'stride': self._stride,
            'is_vit': self._is_vit,
            'hook_layer': self._hook_layer,
        }

    @staticmethod
    def _resolve_pretrained(model_name: str) -> str:
        """Resolve the best pretrained checkpoint tag for a named open_clip model."""
        import open_clip
        tags = open_clip.list_pretrained_tags_by_model(model_name)
        if tags:
            for preferred in ('datacomp_xl_s13b_b90k', 'datacomp1b', 'openai'):
                if preferred in tags:
                    return preferred
            return tags[0]
        return ''

    def _probe_architecture(self):
        """Determine the vision encoder type and locate the hook target."""
        visual = self._model.visual

        has_trunk = hasattr(visual, 'trunk')
        has_transformer = hasattr(visual, 'transformer')

        if has_trunk:
            self._is_vit = self._has_vit_blocks(visual.trunk)
            if self._is_vit:
                blocks = getattr(visual.trunk, 'blocks', None)
                self._hook_layer = blocks[-1] if blocks and len(blocks) > 0 else None
            else:
                self._hook_layer = self._find_last_conv_stage(visual.trunk)
        elif has_transformer:
            self._is_vit = True
            resblocks = getattr(visual.transformer, 'resblocks', None)
            self._hook_layer = resblocks[-1] if resblocks and len(resblocks) > 0 else None
        else:
            self._is_vit = False
            self._hook_layer = self._find_last_conv_stage(visual)

        self._probe_forward()

    @staticmethod
    def _has_vit_blocks(trunk) -> bool:
        """Return True if the trunk looks like a ViT (has .blocks of attention layers)."""
        blocks = getattr(trunk, 'blocks', None)
        if blocks is None or not hasattr(blocks, '__len__') or len(blocks) == 0:
            return False
        first = blocks[0]
        return hasattr(first, 'attn') or hasattr(first, 'self_attn')

    @staticmethod
    def _find_last_conv_stage(module):
        """Find the last spatial stage in a CNN/hybrid backbone."""
        for attr in ('stages', 'features', 'layer4', 'blocks'):
            container = getattr(module, attr, None)
            if container is not None and hasattr(container, '__getitem__'):
                try:
                    return container[-1]
                except (IndexError, TypeError):
                    pass
        return module

    def _probe_forward(self):
        """Run a test image to determine channels and stride."""
        test_size = self._input_size or 224
        test_img = Image.new('RGB', (test_size, test_size))
        preprocessed = self._preprocess(test_img).unsqueeze(0)
        if self.device == 'cuda':
            preprocessed = preprocessed.cuda()

        captured = {}

        def hook_fn(_module, _input, output):
            captured['output'] = output

        handle = self._hook_layer.register_forward_hook(hook_fn) if self._hook_layer else None

        with torch.inference_mode():
            self._model.encode_image(preprocessed)

        if handle is not None:
            handle.remove()

        if 'output' not in captured:
            with torch.inference_mode():
                pooled = self._model.encode_image(preprocessed)
            self._channels = int(pooled.shape[-1])
            self._stride = None
            return

        feat = captured['output']
        if isinstance(feat, tuple):
            feat = feat[0]
        feat = feat.detach()

        if feat.ndim == 3:
            # ViT: [B, T, C]
            T, C = feat.shape[1], feat.shape[2]
            n_patches = T - 1
            side = max(1, int(round(n_patches ** 0.5)))
            self._channels = C
            self._stride = max(1, test_size // side) if side > 0 else 16
        elif feat.ndim == 4:
            # CNN: [B, C, H, W]
            C, H = feat.shape[1], feat.shape[2]
            self._channels = C
            self._stride = max(1, test_size // H) if H > 0 else 16
        else:
            self._channels = int(feat.shape[-1])
            self._stride = 16

    def extract_dense(
        self,
        image_rgb: np.ndarray,
        *,
        out_hw: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        if self._hook_layer is None:
            raise NotImplementedError(
                f"Dense extraction not available for {self.model_name} "
                "(no hookable layer found). Use extract_pooled."
            )
        try:
            model_input = image_rgb
            S = self._input_size
            if S:
                model_input = cv2.resize(image_rgb, (S, S), interpolation=cv2.INTER_AREA)

            pil_img = Image.fromarray(model_input)
            preprocessed = self._preprocess(pil_img).unsqueeze(0)
            if self.device == 'cuda':
                preprocessed = preprocessed.cuda()

            captured = {}

            def hook_fn(_module, _input, output):
                captured['output'] = output

            handle = self._hook_layer.register_forward_hook(hook_fn)
            with torch.inference_mode():
                self._model.encode_image(preprocessed)
            handle.remove()

            feat = captured['output']
            if isinstance(feat, tuple):
                feat = feat[0]
            feat = feat[0].float().cpu().numpy()  # remove batch dim

            if self._is_vit and feat.ndim == 2:
                feat, h_patches, w_patches = TransformerExtractor._split_vit_tokens(feat)
                feat = feat.reshape(h_patches, w_patches, -1)
            elif feat.ndim == 3 and feat.shape[0] == self._channels:
                # CNN: [C, H, W] -> [H, W, C]
                feat = np.transpose(feat, (1, 2, 0))
            elif feat.ndim == 2:
                feat, h_patches, w_patches = TransformerExtractor._split_vit_tokens(feat)
                feat = feat.reshape(h_patches, w_patches, -1)
            # else: already [H, W, C]

            h_patches, w_patches = feat.shape[0], feat.shape[1]
            self._stride = max(1, round(image_rgb.shape[0] / h_patches))

            if self._upsample_factor:
                try:
                    target_h = min(h_patches * self._upsample_factor, model_input.shape[0])
                    target_w = min(w_patches * self._upsample_factor, model_input.shape[1])
                    anyup = self._load_anyup()
                    feat = anyup.upsample(model_input, feat, (target_h, target_w))
                    h_patches, w_patches = target_h, target_w
                    self._stride = max(1, round(image_rgb.shape[0] / h_patches))
                except Exception as e:
                    print(f"AnyUp upsampling failed, using native patch grid: {e}")

            feat = feat.astype(np.float32)
            norms = np.linalg.norm(feat, axis=-1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            feat = feat / norms
            return feat.astype(np.float16)

        except Exception as e:
            print(f"OpenCLIP dense extraction failed: {e}")
            return np.zeros((1, 1, self.out_channels), dtype=np.float16)

    def extract_pooled(self, image_rgb: np.ndarray) -> np.ndarray:
        try:
            pil_img = Image.fromarray(image_rgb)
            preprocessed = self._preprocess(pil_img).unsqueeze(0)
            if self.device == 'cuda':
                preprocessed = preprocessed.cuda()

            with torch.inference_mode():
                feat = self._model.encode_image(preprocessed)

            feat = feat[0].float().cpu().numpy()
            feat = feat.astype(np.float32)
            norm = np.linalg.norm(feat)
            feat = feat / max(norm, 1e-12)
            return feat.astype(np.float16)

        except Exception as e:
            print(f"OpenCLIP pooled extraction failed: {e}")
            return np.zeros(self.out_channels, dtype=np.float16)

    @property
    def out_channels(self) -> int:
        return self._channels or 512

    @property
    def patch_stride(self) -> Optional[int]:
        return self._stride

    @property
    def supports_dense(self) -> bool:
        return self._hook_layer is not None


class FeatureExtractor:
    """
    Unified interface for feature extraction across model families.

    Supports transformer (DINOv2/v3), YOLO, and color features with
    model caching, batching, and both dense and pooled extraction.
    """

    def __init__(self, model_name: str, device: Optional[str] = None,
                 input_size: Optional[int] = None,
                 upsample_factor: Optional[int] = None):
        """
        Initialize the extractor.

        Args:
            model_name: Model identifier (HF path, YOLO name, or "Color Features").
            device: Compute device ("cuda" or "cpu"). Auto-detected if None.
            input_size: Optional square input edge (px) for dense transformer
                extraction. Larger → denser feature grid. None = model default.
            upsample_factor: Optional AnyUp densification factor (2, 4, or 8)
                applied to the native patch grid, capped at input_size.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = device
        self.input_size = int(input_size) if input_size else None
        self.upsample_factor = int(upsample_factor) if upsample_factor else None
        self._backend = None
        self._create_backend()

    def _create_backend(self):
        """Create the appropriate backend extractor."""
        if self.model_name == "Color Features":
            self._backend = ColorExtractor()
        elif is_openclip_model(self.model_name):
            self._backend = OpenCLIPExtractor(
                strip_openclip_prefix(self.model_name), device=self.device,
                input_size=self.input_size, upsample_factor=self.upsample_factor
            )
        elif is_timm_model(self.model_name):
            self._backend = TimmExtractor(
                strip_timm_prefix(self.model_name), device=self.device,
                input_size=self.input_size, upsample_factor=self.upsample_factor
            )
        elif is_yolo_model(self.model_name):
            self._backend = YOLOExtractor(self.model_name, device=self.device)
        elif is_transformer_model(self.model_name):
            self._backend = TransformerExtractor(
                self.model_name, device=self.device, input_size=self.input_size,
                upsample_factor=self.upsample_factor
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    @property
    def model_id(self) -> str:
        """Canonical model identifier."""
        return self.model_name

    @property
    def out_channels(self) -> int:
        """Number of output feature channels."""
        return self._backend.out_channels

    @property
    def patch_stride(self) -> Optional[int]:
        """Spatial stride for dense extraction (None if only pooled)."""
        return self._backend.patch_stride

    @property
    def supports_dense(self) -> bool:
        """Whether the active backend can produce dense feature maps."""
        return getattr(self._backend, "supports_dense", False)

    @property
    def upsample_descriptor(self) -> Optional[str]:
        """Human-readable AnyUp config (e.g. 'anyup_multi_backbone (4x)'), or None."""
        if not self.upsample_factor:
            return None
        return f"anyup_multi_backbone ({self.upsample_factor}x)"

    def extract_dense(
        self,
        image_rgb: np.ndarray,
        *,
        out_hw: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Extract dense feature map [h, w, C] at patch-grid resolution.

        Args:
            image_rgb: Input image as RGB uint8 [H, W, 3].
            out_hw: Optional target (height, width). If None, use native resolution.

        Returns:
            Dense feature map [h, w, C] as float16, L2-normalized.

        Raises:
            NotImplementedError: If the backend doesn't support dense extraction.
        """
        return self._backend.extract_dense(image_rgb, out_hw=out_hw)

    def extract_pooled(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Extract pooled feature vector [C] by global mean pooling.

        Args:
            image_rgb: Input image as RGB uint8 [H, W, 3].

        Returns:
            Pooled vector [C] as float16, L2-normalized.
        """
        return self._backend.extract_pooled(image_rgb)

    def extract_dense_batch(
        self, images: List[np.ndarray], *, out_hw: Optional[Tuple[int, int]] = None
    ) -> List[np.ndarray]:
        """
        Extract dense maps for a batch of images.

        Args:
            images: List of RGB images [H, W, 3].
            out_hw: Optional target resolution.

        Returns:
            List of dense feature maps [h, w, C].
        """
        return [self.extract_dense(img, out_hw=out_hw) for img in images]

    def extract_pooled_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract pooled vectors for a batch of images.

        Args:
            images: List of RGB images [H, W, 3].

        Returns:
            Array of pooled vectors [N, C].
        """
        vectors = [self.extract_pooled(img) for img in images]
        return np.array(vectors)

    def clear_cache(self):
        """Clear global model caches (memory cleanup)."""
        global _TRANSFORMER_MODEL_CACHE, _YOLO_MODEL_CACHE, _TIMM_MODEL_CACHE, _OPENCLIP_MODEL_CACHE
        _TRANSFORMER_MODEL_CACHE.clear()
        _YOLO_MODEL_CACHE.clear()
        _TIMM_MODEL_CACHE.clear()
        _OPENCLIP_MODEL_CACHE.clear()

        from coralnet_toolbox.Features.Models.AnyUpUpsampler import clear_cache as clear_anyup_cache
        clear_anyup_cache()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
