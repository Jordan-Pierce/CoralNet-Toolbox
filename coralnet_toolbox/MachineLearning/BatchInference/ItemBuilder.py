"""Build batch inference work items from selected rasters."""

from __future__ import annotations

from typing import Iterable

from .Contracts import InferenceItem, InferenceSourceKind


def make_video_frame_path(video_path: str, frame_idx: int) -> str:
    """Create the virtual path used throughout the app for a video frame."""
    return f"{video_path}::frame_{frame_idx}"


def _has_work_areas(raster) -> bool:
    has_work_areas = getattr(raster, "has_work_areas", None)
    if callable(has_work_areas):
        try:
            return bool(has_work_areas())
        except Exception:
            return False
    return bool(getattr(raster, "work_areas", None))


def _get_work_areas(raster) -> list:
    get_work_areas = getattr(raster, "get_work_areas", None)
    if callable(get_work_areas):
        try:
            return list(get_work_areas())
        except Exception:
            return []
    return list(getattr(raster, "work_areas", []) or [])


def build_inference_items(image_paths: Iterable[str], raster_manager, inference_type: str,
                          video_start=0, video_end=None, video_stride=1) -> list[InferenceItem]:
    """Build a flat list of worker items from selected image or video paths."""
    items: list[InferenceItem] = []
    if raster_manager is None:
        return items

    use_tiles = inference_type == "Tiled"

    for image_path in image_paths:
        try:
            raster = raster_manager.get_raster(image_path)
            if raster is None:
                continue

            is_video_raster = getattr(raster, "raster_type", "") == "VideoRaster"

            if is_video_raster:
                frame_count = int(getattr(raster, "frame_count", 1))
                start = max(0, int(video_start))
                end = min(
                    frame_count - 1,
                    int(video_end if video_end is not None else frame_count - 1),
                )
                stride = max(1, int(video_stride))

                for frame_idx in range(start, end + 1, stride):
                    virtual_path = make_video_frame_path(raster.image_path, frame_idx)
                    items.append(InferenceItem(
                        batch_key=virtual_path,
                        image_path=raster.image_path,
                        source=frame_idx,
                        work_area=None,
                        raster=raster,
                        is_video=True,
                        source_kind=InferenceSourceKind.VIDEO_FRAME,
                    ))

            elif use_tiles and _has_work_areas(raster):
                for work_area in _get_work_areas(raster):
                    items.append(InferenceItem(
                        batch_key=image_path,
                        image_path=image_path,
                        source=work_area,
                        work_area=work_area,
                        raster=raster,
                        is_video=False,
                        source_kind=InferenceSourceKind.WORK_AREA,
                    ))

            else:
                items.append(InferenceItem(
                    batch_key=image_path,
                    image_path=image_path,
                    source=image_path,
                    work_area=None,
                    raster=raster,
                    is_video=False,
                    source_kind=InferenceSourceKind.IMAGE_PATH,
                ))

        except Exception as e:
            print(f"build_inference_items: skipping {image_path}: {e}")

    return items