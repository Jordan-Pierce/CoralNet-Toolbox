from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2


# -------------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------------

VIDEO_FRAME_MARKER = "::frame_"
DEFAULT_VIDEO_EXPORT_EXTENSION = ".jpg"

# -------------------------------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------------------------------


def parse_frame_path(path: str) -> Tuple[str, Optional[int]]:
    """Split a static path or virtual video-frame path into source path and frame index."""
    if not isinstance(path, str) or VIDEO_FRAME_MARKER not in path:
        return path, None

    source_path, frame_text = path.rsplit(VIDEO_FRAME_MARKER, 1)
    try:
        return source_path, int(frame_text)
    except (TypeError, ValueError):
        return path, None


def normalize_source_path(path: str) -> str:
    """Return the underlying source path for a static path or virtual video-frame path."""
    source_path, _ = parse_frame_path(path)
    return source_path


def is_video_frame_path(path: str) -> bool:
    """Return True when the path points at a virtual video frame."""
    return frame_index_for_path(path) is not None


def frame_index_for_path(path: str) -> Optional[int]:
    """Return the frame index encoded in a virtual path, or None for static paths."""
    _, frame_idx = parse_frame_path(path)
    return frame_idx


def build_video_frame_path(video_path: str, frame_idx: int) -> str:
    """Build the canonical virtual path for a video frame."""
    return f"{video_path}{VIDEO_FRAME_MARKER}{int(frame_idx)}"


def build_video_frame_export_name(video_path: str, frame_idx: int, extension: str = DEFAULT_VIDEO_EXPORT_EXTENSION) -> str:
    """Build a stable exported filename for a video frame."""
    suffix = extension if extension.startswith(".") else f".{extension}"
    return f"{Path(video_path).stem}_frame_{int(frame_idx):06d}{suffix}"


def build_sample_export_name(sample_path: str, extension: Optional[str] = None) -> str:
    """Build a stable export filename for a static image or virtual video frame."""
    source_path, frame_idx = parse_frame_path(sample_path)

    if frame_idx is not None:
        suffix = extension or DEFAULT_VIDEO_EXPORT_EXTENSION
        return build_video_frame_export_name(source_path, frame_idx, suffix)

    if extension is None:
        return os.path.basename(source_path)

    suffix = extension if extension.startswith(".") else f".{extension}"
    return f"{Path(source_path).stem}{suffix}"


def group_annotations_by_source(annotations: Iterable) -> dict:
    """Group annotations by their effective source path.

    Virtual frame paths are grouped under the underlying video path so all
    annotations from the same video stay together.
    """
    grouped = {}
    for annotation in annotations:
        source_path = normalize_source_path(getattr(annotation, "image_path", ""))
        grouped.setdefault(source_path, []).append(annotation)
    return grouped


def frame_matches_stride(sample_path: str, frame_stride: int) -> bool:
    """Return True if the sample path should be kept for the current frame stride."""
    if frame_stride <= 1:
        return True

    frame_idx = frame_index_for_path(sample_path)
    return frame_idx is None or frame_idx % frame_stride == 0


def resolve_sample_source(sample_path: str, raster_manager=None):
    """Return the underlying source path, frame index, and raster object for a sample path."""
    source_path, frame_idx = parse_frame_path(sample_path)
    raster = None

    if raster_manager is not None:
        raster = raster_manager.get_raster(source_path)
        if frame_idx is not None and raster is not None and hasattr(raster, "update_shim_for_frame"):
            raster.update_shim_for_frame(frame_idx)

    return source_path, frame_idx, raster


def build_export_sample_paths(
    source_path: str,
    annotations: Iterable,
    raster_manager=None,
    frame_stride: int = 1,
    export_unlabeled_video_frames: bool = False,
) -> List[str]:
    """Expand a source into the concrete sample paths that should be exported."""
    source_path = normalize_source_path(source_path)
    annotations = list(annotations or [])
    raster = raster_manager.get_raster(source_path) if raster_manager is not None else None

    is_video_source = getattr(raster, "raster_type", "") == "VideoRaster"
    if not is_video_source:
        return [source_path]

    frame_stride = max(1, int(frame_stride or 1))
    frame_count = int(getattr(raster, "frame_count", 0) or 0)

    if export_unlabeled_video_frames:
        return [
            build_video_frame_path(source_path, frame_idx)
            for frame_idx in range(0, frame_count, frame_stride)
        ]

    frame_indices = []
    for annotation in annotations:
        frame_idx = frame_index_for_path(getattr(annotation, "image_path", ""))
        if frame_idx is not None and frame_idx % frame_stride == 0:
            frame_indices.append(frame_idx)

    return [build_video_frame_path(source_path, frame_idx) for frame_idx in sorted(set(frame_indices))]


def sample_dimensions(sample_path: str, raster_manager=None):
    """Return (height, width, raster) for a static image path or virtual video frame path."""
    source_path, frame_idx, raster = resolve_sample_source(sample_path, raster_manager)

    if raster is not None and getattr(raster, "height", 0) and getattr(raster, "width", 0):
        return int(raster.height), int(raster.width), raster

    if frame_idx is not None and raster is not None and hasattr(raster, "get_bgr_frame"):
        frame = raster.get_bgr_frame(frame_idx)
        if frame is not None:
            return int(frame.shape[0]), int(frame.shape[1]), raster

    from coralnet_toolbox.utilities import rasterio_open

    with rasterio_open(source_path) as src:
        return int(src.height), int(src.width), raster


def materialize_sample_image(sample_path: str, raster_manager, output_path: str) -> bool:
    """Write a sample image to disk, copying static images or saving a video frame."""
    source_path, frame_idx, raster = resolve_sample_source(sample_path, raster_manager)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if frame_idx is None:
        if os.path.exists(source_path):
            shutil.copy2(source_path, output_path)
            return True
        return False

    if raster is None:
        return False

    if hasattr(raster, "save_frame"):
        return bool(raster.save_frame(frame_idx, output_path))

    if hasattr(raster, "get_bgr_frame"):
        frame = raster.get_bgr_frame(frame_idx)
        if frame is None:
            return False
        return bool(cv2.imwrite(output_path, frame))

    return False