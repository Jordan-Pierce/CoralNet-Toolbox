# coralnet_toolbox/Explorer/core/feature_vectors.py
"""
Hand-made appearance features, and the helpers that turn backbone outputs into
one vector per crop.

Kept free of Qt so they can be tested and benchmarked on plain arrays. The
EmbeddingViewer calls these per crop.
"""

from math import isqrt

import cv2
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# Colour
# ----------------------------------------------------------------------------------------------------------------------

# A coarse 8x8 Hue-Saturation histogram (64 bins) is far less sparse than a
# fine 16x16 grid, so similar-but-not-identical patches no longer collapse onto
# the same bin. The histogram is paired with full H/S/V mean+std moments below.
COLOR_HS_HUE_BINS = 8
COLOR_HS_SAT_BINS = 8
# Weight applied to the 6 H/S/V mean+std moments so that block is comparable in
# magnitude to the L2-normalized (unit-norm) histogram block. Tune upward to let
# overall brightness/saturation drive the layout more than fine hue structure.
COLOR_MOMENT_WEIGHT = 2.0
COLOR_FEATURE_DIMS = COLOR_HS_HUE_BINS * COLOR_HS_SAT_BINS + 6

# ----------------------------------------------------------------------------------------------------------------------
# Texture
# ----------------------------------------------------------------------------------------------------------------------

# Texture is measured on a crop no larger than this, so a 2,000 px atlas chart
# and a 200 px patch are compared at a similar scale.
TEXTURE_MAX_SIDE = 256
# Rotation-invariant uniform LBP at two radii: (points, radius). riu2 codes
# take P+2 values, so 10 + 10 bins. P=16 at R=2 scored the same on Coralscapes
# and Shoe and costs half again as much.
TEXTURE_LBP_SCALES = ((8, 1), (8, 3))
# Weight of each LBP histogram block (unit norm before weighting), the same
# energy as the colour histogram block.
TEXTURE_LBP_WEIGHT = 1.0
# Weight of the 3 gradient statistics (mean, std, edge density, each ~[0, 1]).
TEXTURE_GRADIENT_WEIGHT = 2.0
# Normalised Sobel magnitude above which a pixel counts as an edge: 0.1 is a
# grey step of about a tenth of the range across one pixel.
TEXTURE_EDGE_THRESHOLD = 0.1
TEXTURE_FEATURE_DIMS = sum(p + 2 for p, _r in TEXTURE_LBP_SCALES) + 3
COLOR_TEXTURE_FEATURE_DIMS = COLOR_FEATURE_DIMS + TEXTURE_FEATURE_DIMS


def _mask_pixels(mask, shape):
    """Boolean pixel mask, or all-True when there is no (or an empty) mask."""
    if mask is None:
        return np.ones(shape[:2], dtype=bool)
    keep = np.asarray(mask) > 0
    if not keep.any():
        return np.ones(shape[:2], dtype=bool)
    return keep


def color_features(img_rgb, mask=None):
    """70-d colour vector: 8x8 Hue-Saturation histogram + 6 H/S/V moments.

    The histogram is Hellinger-transformed and L2-normalised; the moments
    (each in [0, 1]) are scaled by COLOR_MOMENT_WEIGHT. `mask` (uint8, >0 is
    inside) restricts both to the annotation's own pixels.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    cv_mask = None if mask is None else np.asarray(mask, dtype=np.uint8)

    # 2D Hue-Saturation histogram (8x8 = 64 bins). Hue 0-180, Saturation 0-256.
    hist = cv2.calcHist(
        [hsv], [0, 1], cv_mask,
        [COLOR_HS_HUE_BINS, COLOR_HS_SAT_BINS],
        [0, 180, 0, 256]
    )
    hist = hist.flatten().astype(np.float32)
    hist /= (hist.sum() + 1e-7)

    # Hellinger (square-root) transform: de-spikes the dominant bin so
    # sub-dominant colors carry weight. After the sqrt the vector is
    # L2-normalized, which makes the histogram block contribute a fixed amount
    # of "energy" regardless of how peaky it is.
    hist = np.sqrt(hist)
    hist /= (np.linalg.norm(hist) + 1e-7)

    # Full H/S/V mean+std moments: continuous statistics give the layout
    # something to spread on even when patches share a dominant bin.
    masked_pixels = np.asarray(mask) > 0 if mask is not None else np.ones(hsv.shape[:2], dtype=bool)

    h_channel = hsv[..., 0].astype(np.float32) / 180.0
    s_channel = hsv[..., 1].astype(np.float32) / 255.0
    v_channel = hsv[..., 2].astype(np.float32) / 255.0

    moments = np.array([
        np.mean(h_channel[masked_pixels]), np.std(h_channel[masked_pixels]),
        np.mean(s_channel[masked_pixels]), np.std(s_channel[masked_pixels]),
        np.mean(v_channel[masked_pixels]), np.std(v_channel[masked_pixels]),
    ], dtype=np.float32)
    moments *= COLOR_MOMENT_WEIGHT

    return np.concatenate([hist, moments]).astype(np.float32)


def _limit_size(img, mask, max_side):
    """Downscale img (area) and mask (nearest) so the longer side is <= max_side."""
    h, w = img.shape[:2]
    scale = max_side / float(max(h, w))
    if scale >= 1.0:
        return img, mask
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if mask is not None:
        mask = cv2.resize(np.asarray(mask, dtype=np.uint8), size, interpolation=cv2.INTER_NEAREST)
    return img, mask


def texture_features(img_rgb, mask=None):
    """Rotation-invariant texture vector (23-d).

    Two LBP riu2 histograms (P=8 at R=1 and R=3; Hellinger + L2, weighted)
    and three gradient statistics: mean and std of the Sobel magnitude, and the
    fraction of pixels above an edge threshold. All are invariant to rotation,
    which matters on a texture atlas where charts sit at arbitrary angles.
    """
    from skimage.feature import local_binary_pattern

    img, mask = _limit_size(img_rgb, mask, TEXTURE_MAX_SIDE)
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    keep = _mask_pixels(mask, grey.shape)
    if mask is not None and keep.sum() < keep.size:
        # Neighbourhoods that cross the outline would read the background as
        # texture; fill it with the inside's mean grey so the edge stays flat.
        grey = grey.copy()
        grey[~keep] = round(float(grey[keep].mean()))

    blocks = []
    for points, radius in TEXTURE_LBP_SCALES:
        codes = local_binary_pattern(grey, points, radius, method='uniform')
        hist = np.bincount(codes[keep].astype(np.int64), minlength=points + 2)[:points + 2]
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        hist = np.sqrt(hist)
        hist /= (np.linalg.norm(hist) + 1e-7)
        blocks.append(hist * TEXTURE_LBP_WEIGHT)

    g = grey.astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    # A 3x3 Sobel on [0, 1] input peaks at 4; divide so the stats sit in ~[0, 1].
    mag = np.sqrt(gx * gx + gy * gy)[keep] / 4.0
    gradient = np.array([
        mag.mean(), mag.std(), float((mag > TEXTURE_EDGE_THRESHOLD).mean()),
    ], dtype=np.float32)
    blocks.append(np.clip(gradient, 0.0, 1.0) * TEXTURE_GRADIENT_WEIGHT)
    return np.concatenate(blocks).astype(np.float32)


def color_texture_features(img_rgb, mask=None):
    """Colour (70-d, identical to color_features) followed by texture (23-d)."""
    return np.concatenate([color_features(img_rgb, mask), texture_features(img_rgb, mask)])


# ----------------------------------------------------------------------------------------------------------------------
# Crops
# ----------------------------------------------------------------------------------------------------------------------


def fill_background(img, mask):
    """Paint the pixels outside `mask` with the mean colour inside it.

    A polygon's crop is its bounding box, so a thin or diagonal chart can be
    mostly neighbour. A flat fill in the polygon's own mean colour removes the
    neighbour without adding an edge or a colour of its own. Returns img
    unchanged when the mask is missing, empty or covers everything.
    """
    if mask is None:
        return img
    keep = np.asarray(mask) > 0
    if keep.all() or not keep.any():
        return img
    out = np.array(img, copy=True)
    out[~keep] = np.round(out[keep].mean(axis=0)).astype(out.dtype)
    return out


def rotations(img):
    """The image and its 90, 180 and 270 degree rotations (numpy or PIL)."""
    if hasattr(img, 'transpose') and not isinstance(img, np.ndarray):
        from PIL import Image
        return [img, img.transpose(Image.ROTATE_90), img.transpose(Image.ROTATE_180),
                img.transpose(Image.ROTATE_270)]
    return [img] + [np.ascontiguousarray(np.rot90(img, k)) for k in (1, 2, 3)]


def mean_of_groups(vectors, group):
    """Average consecutive runs of `group` vectors (one run per original crop)."""
    V = np.asarray(vectors, dtype=np.float32)
    if group <= 1:
        return V
    return V.reshape(-1, group, V.shape[-1]).mean(axis=1)


# ----------------------------------------------------------------------------------------------------------------------
# Backbone outputs
# ----------------------------------------------------------------------------------------------------------------------

# HF model types whose token sequence starts with a [CLS] token; registers, when
# the config has them, follow it.
_CLS_MODEL_TYPES = {
    'vit', 'deit', 'beit', 'data2vec-vision', 'dinov2', 'dinov2_with_registers',
    'dinov3_vit', 'vit_mae', 'vit_msn', 'ijepa',
}


def prefix_token_count(config, n_tokens):
    """How many tokens precede the patch grid in a backbone's output.

    Known ViT types: one [CLS] plus config.num_register_tokens. Anything else
    (Swin, DINOv3 ConvNeXt's leading pooled token, ...) is inferred from the
    grid being square: whatever sits in front of the largest square count.
    """
    model_type = getattr(config, 'model_type', '') or ''
    registers = int(getattr(config, 'num_register_tokens', 0) or 0)
    if model_type in _CLS_MODEL_TYPES:
        return min(n_tokens - 1, 1 + registers) if n_tokens > 1 else 0
    side = isqrt(int(n_tokens))
    return int(n_tokens - side * side)


def pool_backbone_output(output, prefix_tokens=None):
    """One vector from a backbone's per-image output.

    - tokens (T, D), optionally with a leading batch of 1: mean of the patch
      tokens, skipping `prefix_tokens` ([CLS], registers). The patch mean
      clusters as well as the 98,688-d flattened sequence it replaces, and
      better than [CLS] alone.
    - feature map (C, H, W), optionally with a leading batch of 1: spatial mean.
    - a vector: returned as is.
    """
    if hasattr(output, 'detach'):
        output = output.detach().float().cpu().numpy()
    a = np.asarray(output, dtype=np.float32)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 3:
        if a.shape[0] == 1:
            a = a[0]
        else:
            return a.mean(axis=(1, 2))
    if a.ndim == 2:
        if prefix_tokens is None:
            side = isqrt(a.shape[0])
            prefix_tokens = a.shape[0] - side * side
        prefix_tokens = int(prefix_tokens)
        patches = a[prefix_tokens:] if a.shape[0] > prefix_tokens else a
        return patches.mean(axis=0)
    return a.ravel()
