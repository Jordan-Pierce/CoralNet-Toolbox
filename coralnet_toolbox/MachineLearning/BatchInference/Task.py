"""Task objects that orchestrate batch inference runs."""

from __future__ import annotations

from abc import ABC
from typing import Any

from .Contracts import InferenceItem, InferenceThresholds


class BatchInferenceTask(ABC):
    """Base class for model-specific batch inference orchestration."""

    name = "Batch"
    uses_worker = False
    default_worker_task = "detect"
    progress_title = "Running Inference"

    def __init__(self, dialog: Any, model_dialog: Any, image_paths: list[str]):
        self.dialog = dialog
        self.model_dialog = model_dialog
        self.image_paths = list(image_paths or [])

    def selected_inference_type(self) -> str:
        combo = getattr(self.dialog, "inference_type_combo", None)
        if combo is None:
            return "Standard"
        try:
            return combo.currentText()
        except Exception:
            return "Standard"

    def video_options(self) -> tuple[int, int | None, int]:
        start_spin = getattr(self.dialog, "video_start_spin", None)
        end_spin = getattr(self.dialog, "video_end_spin", None)
        stride_spin = getattr(self.dialog, "video_stride_spin", None)
        try:
            start = int(start_spin.value()) if start_spin is not None else 0
        except Exception:
            start = 0
        try:
            end = int(end_spin.value()) if end_spin is not None else None
        except Exception:
            end = None
        try:
            stride = int(stride_spin.value()) if stride_spin is not None else 1
        except Exception:
            stride = 1
        return start, end, max(1, stride)

    def build_items(self) -> list[InferenceItem]:
        start, end, stride = self.video_options()
        return self.dialog._build_inference_items(
            self.image_paths,
            self.selected_inference_type(),
            video_start=start,
            video_end=end,
            video_stride=stride,
        )

    def initial_thresholds(self) -> InferenceThresholds:
        widget = getattr(self.dialog, "thresholds_widget", None)
        if widget is None:
            return InferenceThresholds()
        values = {}
        try:
            values["conf"] = widget.get_uncertainty_thresh()
        except Exception:
            pass
        try:
            values["iou"] = widget.get_iou_thresh()
        except Exception:
            pass
        try:
            values["max_det"] = widget.get_max_detections()
        except Exception:
            pass
        try:
            values["boundary_tolerance"] = widget.get_boundary_tolerance()
        except Exception:
            pass
        return InferenceThresholds.from_mapping(values)

    def prepare_before_worker(self, items: list[InferenceItem]) -> None:
        """Hook for task-specific state setup before the worker starts."""

    def worker_task(self) -> str:
        return getattr(self.model_dialog, "task", self.default_worker_task)

    def batch_size(self) -> int:
        spin = getattr(self.dialog, "batch_size_spin", None)
        try:
            return int(spin.value()) if spin is not None else 16
        except Exception:
            return 16

    def sam_enabled(self) -> bool:
        return False

    def live_preview(self) -> bool:
        checkbox = getattr(self.dialog, "live_preview_checkbox", None)
        try:
            return bool(checkbox is None or checkbox.isChecked())
        except Exception:
            return True

    def worker_kwargs(self, items: list[InferenceItem]) -> dict[str, Any]:
        return {
            "model": getattr(self.model_dialog, "loaded_model", None),
            "items": items,
            "initial_thresholds": self.initial_thresholds(),
            "device": getattr(getattr(self.dialog, "main_window", None), "device", None),
            "task": self.worker_task(),
            "batch_size": self.batch_size(),
            "imgsz": getattr(self.model_dialog, "imgsz", None),
            "is_semantic": False,
            "sam_enabled": self.sam_enabled(),
            "live_preview": self.live_preview(),
        }

    def create_worker(self, worker_cls: type, items: list[InferenceItem], parent: Any = None):
        kwargs = self.worker_kwargs(items)
        if parent is not None:
            kwargs["parent"] = parent
        return worker_cls(**kwargs)


class AsyncYoloBatchInferenceTask(BatchInferenceTask):
    """Base class for YOLO tasks that run through BatchInferenceWorker."""

    uses_worker = True

    def sam_enabled(self) -> bool:
        sam_dropdown = getattr(self.model_dialog, "use_sam_dropdown", None)
        sam_dialog = getattr(self.model_dialog, "sam_dialog", None)
        try:
            use_sam = sam_dropdown is not None and sam_dropdown.currentText() == "True"
        except Exception:
            use_sam = False
        return bool(
            use_sam
            and sam_dialog is not None
            and getattr(sam_dialog, "loaded_model", None) is not None
        )


class DetectBatchInferenceTask(AsyncYoloBatchInferenceTask):
    name = "Detect"
    default_worker_task = "detect"
    progress_title = "Running Inference"


class SegmentBatchInferenceTask(AsyncYoloBatchInferenceTask):
    name = "Segment"
    default_worker_task = "segment"
    progress_title = "Running Inference"


class SemanticBatchInferenceTask(AsyncYoloBatchInferenceTask):
    name = "Semantic"
    default_worker_task = "semantic"
    progress_title = "Running Semantic Inference"

    def worker_kwargs(self, items: list[InferenceItem]) -> dict[str, Any]:
        kwargs = super().worker_kwargs(items)
        kwargs["is_semantic"] = True
        kwargs["sam_enabled"] = False
        return kwargs

    def prepare_before_worker(self, items: list[InferenceItem]) -> None:
        dialog = self.dialog
        try:
            dialog._semantic_model_class_names = list(
                self.model_dialog.loaded_model.names.values())
        except Exception:
            dialog._semantic_model_class_names = []

        try:
            checkbox = getattr(self.model_dialog, "predict_background_checkbox", None)
            dialog._semantic_include_bg = bool(
                checkbox is not None and checkbox.isChecked())
        except Exception:
            dialog._semantic_include_bg = False

        dialog._semantic_processed_images = set()

        try:
            raster_manager = getattr(dialog.image_window, "raster_manager", None)
            images_to_clear = {item.image_path for item in items}
            for image_path in images_to_clear:
                try:
                    raster = raster_manager.get_raster(image_path) if raster_manager else None
                    if raster and raster.mask_annotation:
                        mask_annotation = raster.mask_annotation
                        mask_annotation.mask_data[:] = 0
                        mask_annotation.update_graphics_item()
                except Exception:
                    pass

            try:
                current = getattr(dialog.annotation_window, "current_image_path", None)
                if current and current in images_to_clear:
                    dialog.annotation_window.load_mask_annotation()
            except Exception:
                pass
        except Exception:
            pass


class ClassifyBatchInferenceTask(BatchInferenceTask):
    name = "Classify"
    progress_title = "Classifying"

    def _use_review_annotations(self) -> bool:
        checkbox = getattr(self.dialog, "review_checkbox", None)
        try:
            return bool(checkbox is not None and checkbox.isChecked())
        except Exception:
            return True

    def collect_image_annotations(self) -> dict[str, list[Any]]:
        annotation_window = self.dialog.annotation_window
        image_annotations = {}
        use_review = self._use_review_annotations()
        for path in self.image_paths:
            try:
                annotations = (
                    annotation_window.get_image_review_annotations(path)
                    if use_review
                    else annotation_window.get_image_annotations(path)
                )
            except Exception:
                annotations = []
            if annotations:
                image_annotations[path] = list(annotations)
        return image_annotations

    def run(self, progress_bar: Any) -> bool:
        image_annotations = self.collect_image_annotations()
        if not image_annotations:
            return False

        flat_annotations = [
            annotation
            for annotations in image_annotations.values()
            for annotation in annotations
        ]

        raster_manager = getattr(self.dialog.image_window, "raster_manager", None)
        uncropped = [
            annotation for annotation in flat_annotations
            if not getattr(annotation, "cropped_image", None)
        ]
        if uncropped:
            by_path: dict[str, list[Any]] = {}
            for annotation in uncropped:
                by_path.setdefault(annotation.image_path, []).append(annotation)

            progress_bar.set_title("Preparing Patches...")
            progress_bar.start_progress(len(uncropped))
            for path, annotations in by_path.items():
                raster = raster_manager.get_raster(path) if raster_manager else None
                rasterio_src = getattr(raster, "rasterio_src", None) if raster else None
                for annotation in annotations:
                    if rasterio_src is not None:
                        try:
                            annotation.create_cropped_image(rasterio_src)
                        except Exception:
                            pass
                    progress_bar.update_progress()

        progress_bar.set_title(
            f"Classifying {len(flat_annotations)} patches "
            f"across {len(image_annotations)} image(s)..."
        )
        self.model_dialog.predict(inputs=flat_annotations, progress_bar=progress_bar)

        try:
            from PyQt5.QtWidgets import QApplication
            annotation_window = self.dialog.annotation_window
            annotation_window.refresh_phantom_annotations()
            annotation_window.viewport().update()
            QApplication.processEvents()
        except Exception:
            pass

        return True


_TASKS = {
    "Detect": DetectBatchInferenceTask,
    "Segment": SegmentBatchInferenceTask,
    "Semantic": SemanticBatchInferenceTask,
    "Classify": ClassifyBatchInferenceTask,
}


def make_batch_inference_task(selected_model: str, dialog: Any, model_dialog: Any,
                              image_paths: list[str]) -> BatchInferenceTask:
    task_cls = _TASKS.get(selected_model, BatchInferenceTask)
    return task_cls(dialog, model_dialog, image_paths)