"""Shared RLE archive helpers for MVAT index maps."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np


INDEX_MAP_RLE_FORMAT = "index_map_rle_v1"


def encode_index_map_rle(index_map: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compress a dense 2-D index map into RLE value and length vectors."""
    index_map_arr = np.asarray(index_map, dtype=np.int32)
    if index_map_arr.ndim != 2:
        raise ValueError("index_map must be a 2-D numpy array")

    flat = index_map_arr.ravel()
    if flat.size == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.uint32),
            np.asarray(index_map_arr.shape, dtype=np.int32),
        )

    is_change = np.concatenate((np.array([True]), flat[1:] != flat[:-1]))
    change_indices = np.flatnonzero(is_change)
    values = flat[change_indices].astype(np.int32, copy=False)
    lengths = np.diff(np.append(change_indices, flat.size)).astype(np.uint32, copy=False)

    return values, lengths, np.asarray(index_map_arr.shape, dtype=np.int32)


def decode_index_map_rle(values: np.ndarray, lengths: np.ndarray, shape: np.ndarray | tuple) -> np.ndarray:
    """Inflate a run-length encoded index map back into a dense 2-D array."""
    values_arr = np.asarray(values, dtype=np.int32).ravel()
    lengths_arr = np.asarray(lengths, dtype=np.int64).ravel()
    shape_arr = np.asarray(shape, dtype=np.int64).ravel()

    if shape_arr.size != 2:
        raise ValueError("shape must describe a 2-D array")
    if np.any(shape_arr < 0):
        raise ValueError("shape dimensions must be non-negative")
    if values_arr.size != lengths_arr.size:
        raise ValueError("values and lengths must have the same length")
    if np.any(lengths_arr < 0):
        raise ValueError("lengths must be non-negative")

    height = int(shape_arr[0])
    width = int(shape_arr[1])
    if values_arr.size == 0:
        return np.empty((height, width), dtype=np.int32)

    flat = np.repeat(values_arr, lengths_arr)
    expected_size = height * width
    if flat.size != expected_size:
        raise ValueError(
            f"Decoded index map has {flat.size} elements but expected {expected_size}"
        )

    return flat.astype(np.int32, copy=False).reshape((height, width))


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
    depth_map: Optional[np.ndarray] = None,
    **extra_metadata,
) -> str:
    """Save a dense index map as an RLE-backed .npz archive."""
    archive_path = os.fspath(archive_path)
    temp_path = _npz_temp_path(archive_path)

    values, lengths, shape = encode_index_map_rle(index_map)
    payload: Dict[str, Any] = {
        "cache_format": np.asarray(INDEX_MAP_RLE_FORMAT),
        "index_map_rle_values": values,
        "index_map_rle_lengths": lengths,
        "index_map_rle_shape": shape,
        "visible_indices": _coerce_visible_indices(visible_indices),
        "element_type": np.asarray(element_type),
    }

    if depth_map is not None:
        payload["depth_map"] = np.asarray(depth_map).astype(np.float16, copy=False)

    for key, value in extra_metadata.items():
        if value is not None:
            payload[key] = np.asarray(value)

    try:
        np.savez(temp_path, **payload)
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
    """Load an RLE-backed index-map archive and return dense arrays."""
    archive_path = os.fspath(archive_path)
    with np.load(archive_path, allow_pickle=False) as data:
        cache_format = _scalar_to_python(data["cache_format"]) if "cache_format" in data else None
        if cache_format != INDEX_MAP_RLE_FORMAT:
            raise ValueError(f"Unsupported cache format: {cache_format!r}")

        index_map = decode_index_map_rle(
            data["index_map_rle_values"],
            data["index_map_rle_lengths"],
            data["index_map_rle_shape"],
        )

        visible_indices = _coerce_visible_indices(data["visible_indices"]) if "visible_indices" in data else np.empty(0, dtype=np.int32)

        depth_map = None
        if "depth_map" in data:
            depth_map = np.asarray(data["depth_map"]).astype(np.float16, copy=False)

        element_type = _scalar_to_python(data["element_type"]) if "element_type" in data else "point"

        result: Dict[str, Any] = {
            "index_map": index_map,
            "visible_indices": visible_indices,
            "depth_map": depth_map,
            "element_type": str(element_type),
            "cache_format": str(cache_format),
            "inverted_index": None,
        }

        known_keys = {
            "cache_format",
            "index_map_rle_values",
            "index_map_rle_lengths",
            "index_map_rle_shape",
            "visible_indices",
            "depth_map",
            "element_type",
        }
        for key in data.files:
            if key in known_keys:
                continue
            result[key] = _scalar_to_python(data[key])

        return result
