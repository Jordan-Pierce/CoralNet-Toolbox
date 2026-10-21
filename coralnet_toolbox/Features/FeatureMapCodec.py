"""
Feature map storage and retrieval with bounded LRU caching.

Feature maps are dense float16 arrays [h, w, C] at patch-grid resolution,
stored as raw .npy files with JSON sidecars for metadata.
"""

from __future__ import annotations

import os
import json
import threading
from collections import OrderedDict
from typing import Optional, Dict, Any

import numpy as np


def feature_variant(model_id: str, input_size=None, upsample_factor=None, normalized: bool = True) -> str:
    """The extraction settings a cached map depends on, as one string.

    Part of the cache key (paths.feature_cache_path) and stored in the sidecar,
    so a map extracted with another model, input resolution, upsampler or
    normalization is never mistaken for — or overwritten by — this one.
    """
    return f"{model_id}|in={input_size or 'default'}|up={upsample_factor or 1}|norm={int(bool(normalized))}"


def extractor_variant(extractor, normalized: bool = True) -> str:
    """feature_variant() of a deployed FeatureExtractor."""
    return feature_variant(getattr(extractor, "model_id", None) or getattr(extractor, "model_name", "unknown"),
                           getattr(extractor, "input_size", None),
                           getattr(extractor, "upsample_factor", None), normalized)


def read_feature_metadata(npy_path: str) -> Optional[Dict[str, Any]]:
    """The sidecar metadata of a cached map, or None when the map or its sidecar
    is missing or unreadable."""
    npy_path = os.fspath(npy_path)
    json_path = os.path.splitext(npy_path)[0] + ".json"
    if not (os.path.exists(npy_path) and os.path.exists(json_path)):
        return None
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save_feature_map(
    npy_path: str,
    feature_map: np.ndarray,
    *,
    model_id: str,
    stride: int,
    dim: int,
    normalized: bool = True,
    feature_vector: Optional[np.ndarray] = None,
    upsampler: Optional[str] = None,
    variant: Optional[str] = None,
    complete: bool = True,
) -> str:
    """
    Save a dense feature map as a .npy file with a .json metadata sidecar.

    Args:
        npy_path: Path to write the .npy file (e.g., '.cache/features/image001.npy').
        feature_map: Dense array [h, w, C] as float16.
        model_id: Model identifier.
        stride: Patch grid stride (16 for ViT/16).
        dim: Number of feature channels C.
        normalized: Whether features are L2-normalized.
        feature_vector: Optional pooled vector [C] (cached for Explorer).
        upsampler: Optional descriptor of an AnyUp densification step applied
            to the native patch grid (e.g. "anyup_multi_backbone (4x)").
        variant: feature_variant() of the extraction settings (cache reuse check).
        complete: False for a map that only holds part of the image (the 2D
            tool's work-area persistence) — never reused as a whole-image map.

    Returns:
        Path to the written .npy file.
    """
    npy_path = os.fspath(npy_path)
    json_path = os.path.splitext(npy_path)[0] + ".json"

    # Ensure directory exists
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)

    # Write .npy file
    np.save(npy_path, feature_map)

    # Write .json sidecar
    metadata = {
        "model_id": model_id,
        "stride": stride,
        "dim": dim,
        "normalized": normalized,
        "h": int(feature_map.shape[0]),
        "w": int(feature_map.shape[1]),
        "upsampler": upsampler,
        "variant": variant,
        "complete": bool(complete),
    }
    if feature_vector is not None:
        # Store pooled vector as a list for JSON serialization
        metadata["feature_vector"] = feature_vector.tolist() if isinstance(feature_vector, np.ndarray) else list(feature_vector)

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return npy_path


def load_feature_map(npy_path: str) -> Dict[str, Any]:
    """
    Load a feature map and its metadata from disk.

    Returns a dict with keys:
        - 'feature_map': [h, w, C] array as float16
        - 'model_id': str
        - 'stride': int
        - 'dim': int
        - 'normalized': bool
        - 'feature_vector': Optional [C] array or None

    Raises ValueError if the .json sidecar is missing or invalid.
    """
    npy_path = os.fspath(npy_path)
    json_path = os.path.splitext(npy_path)[0] + ".json"

    # Load metadata
    if not os.path.exists(json_path):
        raise ValueError(f"Missing metadata sidecar: {json_path}")

    with open(json_path, "r") as f:
        metadata = json.load(f)

    # Load feature map
    feature_map = np.load(npy_path, allow_pickle=False)

    # Restore feature_vector if present
    feature_vector = None
    if "feature_vector" in metadata:
        feature_vector = np.array(metadata["feature_vector"], dtype=np.float16)

    return {
        "feature_map": feature_map,
        "model_id": metadata.get("model_id", "unknown"),
        "stride": metadata.get("stride", 16),
        "dim": metadata.get("dim", 768),
        "normalized": metadata.get("normalized", True),
        "feature_vector": feature_vector,
        "upsampler": metadata.get("upsampler"),
    }


def load_feature_vector(npy_path: str):
    """
    Load only the pooled [C] descriptor for a feature map, without its dense array.

    The vector lives in the .json sidecar, so reading it costs a few kilobytes
    while load_feature_map() pulls the whole [h, w, C] map into memory. A caller
    that ranks every image in a project by descriptor -- Active Learning's
    acquisition pass -- would otherwise load an entire project's dense maps to
    read one row of metadata each.

    Returns None when the sidecar is missing, unreadable, or carries no
    feature_vector: the descriptor is optional and its absence is not an error.
    """
    npy_path = os.fspath(npy_path)
    json_path = os.path.splitext(npy_path)[0] + ".json"

    try:
        with open(json_path, "r") as f:
            metadata = json.load(f)
    except (OSError, ValueError):
        return None

    values = metadata.get("feature_vector")
    if values is None or len(values) == 0:
        return None

    return np.array(values, dtype=np.float16)


class FeatureMapLRU:
    """
    Bounded, thread-safe cache of decompressed feature maps, keyed by .npy path.

    Evicts least-recently-used maps to stay under max_bytes budget. On access, hit maps
    are moved to end (most recent). On put, eviction happens from front (least recent).
    """

    def __init__(self, max_bytes: int):
        self._max_bytes = max_bytes
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path: str) -> Optional[np.ndarray]:
        """Get a map from cache (or load from disk if not cached), returning it on hit.

        Returns None if the path doesn't exist or can't be loaded.
        """
        with self._lock:
            arr = self._cache.get(path)
            if arr is not None:
                self._cache.move_to_end(path)
                return arr

        try:
            data = load_feature_map(path)
            arr = data["feature_map"]
        except Exception:
            return None

        self.put(path, arr)
        return arr

    def put(self, path: str, arr: np.ndarray) -> None:
        """Insert a map into the cache and evict LRU entries if needed."""
        with self._lock:
            self._cache[path] = arr
            self._cache.move_to_end(path)
            total = sum(a.nbytes for a in self._cache.values())
            while total > self._max_bytes and len(self._cache) > 1:
                _, evicted = self._cache.popitem(last=False)
                total -= evicted.nbytes

    def discard(self, path: str) -> None:
        """Remove a map from the cache (e.g. when feature_map is explicitly cleared)."""
        with self._lock:
            self._cache.pop(path, None)

    def clear(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()


# Global LRU cache (512 MB budget by default)
FEATURE_MAP_LRU = FeatureMapLRU(max_bytes=512 * 1024**2)
