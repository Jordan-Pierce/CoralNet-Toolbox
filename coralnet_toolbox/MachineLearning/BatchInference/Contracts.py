"""Shared data contracts for batch inference workflows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class InferenceSourceKind(str, Enum):
    """Kinds of sources that can be decoded for model inference."""

    IMAGE_PATH = "image_path"
    VIDEO_FRAME = "video_frame"
    WORK_AREA = "work_area"


@dataclass(slots=True)
class InferenceThresholds:
    """Mutable threshold snapshot shared between the UI thread and worker."""

    conf: float = 0.25
    iou: float = 0.7
    max_det: int = 300
    boundary_tolerance: Any = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "InferenceThresholds":
        if values is None:
            return cls()
        if isinstance(values, cls):
            return values.snapshot()
        defaults = cls()
        return cls(
            conf=values.get("conf", defaults.conf),
            iou=values.get("iou", defaults.iou),
            max_det=values.get("max_det", defaults.max_det),
            boundary_tolerance=values.get("boundary_tolerance", defaults.boundary_tolerance),
        )

    def update(self, conf=None, iou=None, max_det=None, boundary_tolerance=None) -> None:
        if conf is not None:
            self.conf = conf
        if iou is not None:
            self.iou = iou
        if max_det is not None:
            self.max_det = max_det
        if boundary_tolerance is not None:
            self.boundary_tolerance = boundary_tolerance

    def snapshot(self) -> "InferenceThresholds":
        return InferenceThresholds(
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            boundary_tolerance=self.boundary_tolerance,
        )


@dataclass(slots=True)
class InferenceItem:
    """One unit of work for the batch inference worker."""

    batch_key: str
    image_path: str
    source: Any
    work_area: Any = None
    raster: Any = None
    is_video: bool = False
    source_kind: InferenceSourceKind = InferenceSourceKind.IMAGE_PATH

    def work_area_metadata(self) -> dict | None:
        if self.work_area is None:
            return None
        to_dict = getattr(self.work_area, "to_dict", None)
        if to_dict is None:
            return None
        try:
            return to_dict()
        except Exception:
            return None


@dataclass(slots=True)
class InferenceResult:
    """One model result emitted by the batch inference worker."""

    batch_key: str
    image_path: str
    yolo_result: Any
    work_area: Any = None
    is_video: bool = False
    q_image: Any = None


@dataclass(slots=True)
class SemanticOverlayRecord:
    """Cached semantic overlay for a video frame."""

    mask_qimage: Any
    mask_arr: Any
    opacity: float

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            "mask_qimage": self.mask_qimage,
            "mask_arr": self.mask_arr,
            "opacity": self.opacity,
        }