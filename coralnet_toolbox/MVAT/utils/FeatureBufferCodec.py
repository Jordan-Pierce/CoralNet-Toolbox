"""
Codec for persisting and loading MVAT Tier-2 feature buffers.

Mirrors IndexMapCodec: stores [N,D] features + metadata as `.npz` (compressed).
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

import numpy as np

from coralnet_toolbox.MVAT.core.FeatureBuffer import FeatureBuffer


FEATURE_BUFFER_FORMAT = "feature_buffer_v1"


def _scalar_to_python(value: Any) -> Any:
    """Convert numpy scalar to Python type."""
    array_value = np.asarray(value)
    if array_value.ndim == 0:
        return array_value.item()
    return array_value


def save_feature_buffer(
    archive_path: str,
    buffer: FeatureBuffer,
    *,
    compress: bool = True,
) -> str:
    """
    Save a FeatureBuffer to a compressed `.npz` archive.

    Args:
        archive_path: Path to write the `.npz` file.
        buffer: FeatureBuffer to save.
        compress: If True (default), use DEFLATE compression.

    Returns:
        Path to the saved archive.
    """
    archive_path = os.fspath(archive_path)
    os.makedirs(os.path.dirname(archive_path) or ".", exist_ok=True)

    # Enforce dtypes
    features = np.asarray(buffer.features, dtype=np.float16)
    coverage = np.asarray(buffer.coverage, dtype=np.float32)
    valid = np.asarray(buffer.valid, dtype=bool)
    pca_rgb = (
        np.asarray(buffer.pca_rgb, dtype=np.uint8)
        if buffer.pca_rgb is not None else None
    )

    payload: Dict[str, Any] = {
        "cache_format": np.asarray(FEATURE_BUFFER_FORMAT),
        "features": features,
        "coverage": coverage,
        "valid": valid,
    }

    if pca_rgb is not None:
        payload["pca_rgb"] = pca_rgb

    # Store compressor state and provenance as JSON strings
    if buffer.compressor_state:
        payload["compressor_state"] = np.asarray(
            json.dumps(buffer.compressor_state), dtype=object
        )
    if buffer.provenance:
        payload["provenance"] = np.asarray(
            json.dumps(buffer.provenance), dtype=object
        )

    try:
        saver = np.savez_compressed if compress else np.savez
        saver(archive_path, **payload)
        return archive_path
    except Exception as e:
        print(f"Warning: Failed to save feature buffer to {archive_path}: {e}")
        raise


def load_feature_buffer(archive_path: str) -> FeatureBuffer:
    """
    Load a FeatureBuffer from a saved `.npz` archive.

    Args:
        archive_path: Path to the `.npz` file.

    Returns:
        FeatureBuffer instance.

    Raises:
        ValueError: If the archive format is unsupported.
    """
    archive_path = os.fspath(archive_path)
    try:
        with np.load(archive_path, allow_pickle=True) as data:
            cache_format = (
                _scalar_to_python(data["cache_format"])
                if "cache_format" in data else None
            )
            if cache_format != FEATURE_BUFFER_FORMAT:
                raise ValueError(f"Unsupported cache format: {cache_format!r}")

            features = np.asarray(data["features"], dtype=np.float16)
            coverage = np.asarray(data["coverage"], dtype=np.float32)
            valid = np.asarray(data["valid"], dtype=bool)

            pca_rgb = (
                np.asarray(data["pca_rgb"], dtype=np.uint8)
                if "pca_rgb" in data else None
            )

            # Deserialize JSON metadata
            compressor_state = {}
            if "compressor_state" in data:
                try:
                    cs_str = _scalar_to_python(data["compressor_state"])
                    compressor_state = json.loads(cs_str)
                except Exception:
                    pass

            provenance = {}
            if "provenance" in data:
                try:
                    prov_str = _scalar_to_python(data["provenance"])
                    provenance = json.loads(prov_str)
                except Exception:
                    pass

            return FeatureBuffer(
                features=features,
                coverage=coverage,
                valid=valid,
                compressor_state=compressor_state,
                pca_rgb=pca_rgb,
                provenance=provenance,
            )
    except Exception as e:
        print(f"Warning: Failed to load feature buffer from {archive_path}: {e}")
        raise
