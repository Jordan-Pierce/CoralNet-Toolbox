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


def _palette_encode_rle_values(values: np.ndarray, visible_indices: np.ndarray) -> np.ndarray:
    """Re-encode RLE run values as 1-based palette indices into `visible_indices`,
    when that fits a smaller dtype than int32.

    A camera typically only sees a few thousand of a mesh's elements, so the
    palette (`visible_indices`, already stored alongside the map) is usually far
    smaller than the global element-ID range — letting per-run values shrink from
    int32 to uint8/uint16 regardless of total mesh size. The convention mirrors
    the GPU face-ID encoding already used elsewhere in MVAT: 0 = no content
    (-1), N = visible_indices[N-1].

    Falls back to returning `values` unchanged (raw int32 global IDs) whenever
    the palette doesn't fit uint8/uint16, or any non-background run value isn't
    present in it — keeping the format self-correcting rather than risking
    corruption if some future producer ever violates that invariant.
    """
    n_palette = visible_indices.size
    if n_palette == 0 or n_palette > 65535:
        return values

    dtype = np.uint8 if n_palette <= 255 else np.uint16

    mask = values >= 0
    if not np.any(mask):
        return np.zeros(values.shape, dtype=dtype)

    positions = np.clip(np.searchsorted(visible_indices, values), 0, n_palette - 1)
    if not np.array_equal(visible_indices[positions[mask]], values[mask]):
        return values

    encoded = np.zeros(values.shape, dtype=dtype)
    encoded[mask] = (positions[mask] + 1).astype(dtype)
    return encoded


def _palette_decode_rle_values(values: np.ndarray, visible_indices: np.ndarray) -> np.ndarray:
    """Inverse of `_palette_encode_rle_values`.

    Archives saved before palette encoding (or where it wasn't beneficial)
    store raw int32 global IDs and pass through unchanged — palette indices are
    only ever written as uint8/uint16, so the dtype alone identifies the format.
    """
    values = np.asarray(values)
    if values.dtype.kind != 'u' or values.dtype.itemsize > 2:
        return np.asarray(values, dtype=np.int32)

    decoded = np.full(values.shape, -1, dtype=np.int32)
    mask = values > 0
    if np.any(mask):
        decoded[mask] = visible_indices[values[mask].astype(np.int64) - 1]
    return decoded


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
    visible_indices_arr = _coerce_visible_indices(visible_indices)
    payload: Dict[str, Any] = {
        "cache_format": np.asarray(INDEX_MAP_RLE_FORMAT),
        "index_map_rle_values": _palette_encode_rle_values(values, visible_indices_arr),
        "index_map_rle_lengths": lengths,
        "index_map_rle_shape": shape,
        "visible_indices": visible_indices_arr,
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

        visible_indices = _coerce_visible_indices(data["visible_indices"]) if "visible_indices" in data else np.empty(0, dtype=np.int32)

        index_map = decode_index_map_rle(
            _palette_decode_rle_values(data["index_map_rle_values"], visible_indices),
            data["index_map_rle_lengths"],
            data["index_map_rle_shape"],
        )

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
