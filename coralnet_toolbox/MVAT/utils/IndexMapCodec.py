"""Shared archive helpers for MVAT index maps.

Index maps are dense 2-D int32 arrays of element IDs (-1 = no content). They are
stored as DEFLATE-compressed ``.npz`` archives; this integer label data
compresses ~15-30x on its own, so no application-level run-length/palette
encoding is layered on top of it.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np


INDEX_MAP_DENSE_FORMAT = "index_map_dense_v1"


def _npz_temp_path(archive_path: str) -> str:
    base, ext = os.path.splitext(archive_path)
    if ext.lower() == ".npz":
        return base + "_tmp.npz"
    return archive_path + "_tmp.npz"


def _scalar_to_python(value: Any) -> Any:
    array_value = np.asarray(value)
    if array_value.ndim == 0:
        return array_value.item()
    return array_value


def _coerce_visible_indices(visible_indices: Optional[np.ndarray]) -> np.ndarray:
    if visible_indices is None:
        return np.empty(0, dtype=np.int32)
    return np.asarray(visible_indices, dtype=np.int32).reshape(-1)


def save_index_map_archive(
    archive_path: str,
    index_map: np.ndarray,
    visible_indices: Optional[np.ndarray],
    *,
    element_type: str = "point",
    compress: bool = True,
    **extra_metadata,
) -> str:
    """Save a dense 2-D index map as a ``.npz`` archive.

    When ``compress`` is True (default) the archive is DEFLATE-compressed
    (``np.savez_compressed``); when False it is stored uncompressed
    (``np.savez``) — faster to write at the cost of disk size. ``load_index_map_archive``
    reads either transparently.

    Extra keyword metadata (e.g. ``scale_factor``) is stored verbatim and
    returned by :func:`load_index_map_archive`; ``None`` values are skipped.
    """
    archive_path = os.fspath(archive_path)
    temp_path = _npz_temp_path(archive_path)

    index_map_arr = np.asarray(index_map, dtype=np.int32)
    if index_map_arr.ndim != 2:
        raise ValueError("index_map must be a 2-D numpy array")

    payload: Dict[str, Any] = {
        "cache_format": np.asarray(INDEX_MAP_DENSE_FORMAT),
        "index_map": index_map_arr,
        "visible_indices": _coerce_visible_indices(visible_indices),
        "element_type": np.asarray(element_type),
    }

    for key, value in extra_metadata.items():
        if value is not None:
            payload[key] = np.asarray(value)

    try:
        saver = np.savez_compressed if compress else np.savez
        saver(temp_path, **payload)
        os.replace(temp_path, archive_path)
        return archive_path
    except Exception:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        raise


def load_index_map_archive(archive_path: str) -> Dict[str, Any]:
    """Load a dense index-map archive written by :func:`save_index_map_archive`.

    Raises ``ValueError`` for archives in any other (e.g. legacy) format, which
    callers treat as a cache miss and recompute.
    """
    archive_path = os.fspath(archive_path)
    with np.load(archive_path, allow_pickle=False) as data:
        cache_format = _scalar_to_python(data["cache_format"]) if "cache_format" in data else None
        if cache_format != INDEX_MAP_DENSE_FORMAT:
            raise ValueError(f"Unsupported cache format: {cache_format!r}")

        index_map = np.asarray(data["index_map"], dtype=np.int32)
        visible_indices = (
            _coerce_visible_indices(data["visible_indices"])
            if "visible_indices" in data else np.empty(0, dtype=np.int32)
        )
        element_type = _scalar_to_python(data["element_type"]) if "element_type" in data else "point"

        result: Dict[str, Any] = {
            "index_map": index_map,
            "visible_indices": visible_indices,
            "depth_map": None,
            "element_type": str(element_type),
            "cache_format": str(cache_format),
            "inverted_index": None,
        }

        known_keys = {
            "cache_format",
            "index_map",
            "visible_indices",
            "element_type",
        }
        for key in data.files:
            if key in known_keys:
                continue
            result[key] = _scalar_to_python(data[key])

        return result
