import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "coralnet_toolbox" / "MachineLearning" / "BatchInference"


def _ensure_package(name, path=None):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    if path is not None:
        module.__path__ = [str(path)]
    return module


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_batch_modules():
    _ensure_package("coralnet_toolbox", ROOT / "coralnet_toolbox")
    _ensure_package("coralnet_toolbox.MachineLearning", ROOT / "coralnet_toolbox" / "MachineLearning")
    _ensure_package("coralnet_toolbox.MachineLearning.BatchInference", BATCH_DIR)
    contracts = _load_module(
        "coralnet_toolbox.MachineLearning.BatchInference.Contracts",
        BATCH_DIR / "Contracts.py",
    )
    item_builder = _load_module(
        "coralnet_toolbox.MachineLearning.BatchInference.ItemBuilder",
        BATCH_DIR / "ItemBuilder.py",
    )
    timing = _load_module(
        "coralnet_toolbox.MachineLearning.BatchInference.Timing",
        BATCH_DIR / "Timing.py",
    )
    task = _load_module(
        "coralnet_toolbox.MachineLearning.BatchInference.Task",
        BATCH_DIR / "Task.py",
    )
    return contracts, item_builder, task, timing


class FakeWorkArea:
    def __init__(self, name):
        self.name = name

    def to_dict(self):
        return {"name": self.name}


class FakeRaster:
    def __init__(self, image_path, raster_type="Raster", frame_count=1, work_areas=None):
        self.image_path = image_path
        self.raster_type = raster_type
        self.frame_count = frame_count
        self._work_areas = list(work_areas or [])

    def has_work_areas(self):
        return bool(self._work_areas)

    def get_work_areas(self):
        return list(self._work_areas)


class FakeRasterManager:
    def __init__(self, rasters):
        self._rasters = dict(rasters)

    def get_raster(self, image_path):
        return self._rasters.get(image_path)


def test_build_standard_static_image_item():
    contracts, item_builder, _, _ = _load_batch_modules()
    raster = FakeRaster("image-a.tif")
    manager = FakeRasterManager({"image-a.tif": raster})

    items = item_builder.build_inference_items(["image-a.tif"], manager, "Standard")

    assert len(items) == 1
    assert items[0].batch_key == "image-a.tif"
    assert items[0].image_path == "image-a.tif"
    assert items[0].source == "image-a.tif"
    assert items[0].raster is raster
    assert items[0].is_video is False
    assert items[0].source_kind == contracts.InferenceSourceKind.IMAGE_PATH


def test_build_tiled_static_image_items_store_work_area_sources():
    contracts, item_builder, _, _ = _load_batch_modules()
    work_areas = [FakeWorkArea("tile-1"), FakeWorkArea("tile-2")]
    raster = FakeRaster("image-b.tif", work_areas=work_areas)
    manager = FakeRasterManager({"image-b.tif": raster})

    items = item_builder.build_inference_items(["image-b.tif"], manager, "Tiled")

    assert len(items) == 2
    assert [item.source for item in items] == work_areas
    assert [item.work_area_metadata() for item in items] == [
        {"name": "tile-1"},
        {"name": "tile-2"},
    ]
    assert all(item.batch_key == "image-b.tif" for item in items)
    assert all(item.source_kind == contracts.InferenceSourceKind.WORK_AREA for item in items)


def test_build_video_items_respects_start_end_and_stride():
    contracts, item_builder, _, _ = _load_batch_modules()
    raster = FakeRaster("clip.mp4", raster_type="VideoRaster", frame_count=10)
    manager = FakeRasterManager({"clip.mp4": raster})

    items = item_builder.build_inference_items(
        ["clip.mp4"],
        manager,
        "Standard",
        video_start=2,
        video_end=6,
        video_stride=2,
    )

    assert [item.source for item in items] == [2, 4, 6]
    assert [item.batch_key for item in items] == [
        "clip.mp4::frame_2",
        "clip.mp4::frame_4",
        "clip.mp4::frame_6",
    ]
    assert all(item.is_video for item in items)
    assert all(item.source_kind == contracts.InferenceSourceKind.VIDEO_FRAME for item in items)


def test_video_tiled_mode_falls_back_to_frame_items():
    contracts, item_builder, _, _ = _load_batch_modules()
    raster = FakeRaster(
        "clip.mp4",
        raster_type="VideoRaster",
        frame_count=3,
        work_areas=[FakeWorkArea("ignored")],
    )
    manager = FakeRasterManager({"clip.mp4": raster})

    items = item_builder.build_inference_items(["clip.mp4"], manager, "Tiled")


    assert [item.source for item in items] == [0, 1, 2]
    assert all(item.source_kind == contracts.InferenceSourceKind.VIDEO_FRAME for item in items)


def test_threshold_snapshot_is_isolated_from_later_updates():
    contracts, _, _, _ = _load_batch_modules()
    thresholds = contracts.InferenceThresholds.from_mapping({
        "conf": 0.5,
        "iou": 0.25,
        "max_det": 42,
        "boundary_tolerance": 3,
    })

    snapshot = thresholds.snapshot()
    thresholds.update(conf=0.9, max_det=7)

    assert snapshot.conf == 0.5
    assert snapshot.iou == 0.25
    assert snapshot.max_det == 42
    assert snapshot.boundary_tolerance == 3
    assert thresholds.conf == 0.9
    assert thresholds.max_det == 7


class FakeSpinBox:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class FakeComboBox:
    def __init__(self, text):
        self._text = text

    def currentText(self):
        return self._text


class FakeCheckBox:
    def __init__(self, checked):
        self._checked = checked

    def isChecked(self):
        return self._checked


class FakeThresholdsWidget:
    def get_uncertainty_thresh(self):
        return 0.42

    def get_iou_thresh(self):
        return 0.31

    def get_max_detections(self):
        return 12

    def get_boundary_tolerance(self):
        return 4


class FakeMainWindow:
    device = "cuda:0"


class FakeDialog:
    def __init__(self, items):
        self._items = items
        self.main_window = FakeMainWindow()
        self.thresholds_widget = FakeThresholdsWidget()
        self.inference_type_combo = FakeComboBox("Tiled")
        self.video_start_spin = FakeSpinBox(3)
        self.video_end_spin = FakeSpinBox(9)
        self.video_stride_spin = FakeSpinBox(2)
        self.batch_size_spin = FakeSpinBox(8)

    def _build_inference_items(self, image_paths, inference_type,
                               video_start=0, video_end=None, video_stride=1):
        self.build_args = (image_paths, inference_type, video_start, video_end, video_stride)
        return self._items


class FakeLoadedModel:
    names = {0: "background", 1: "coral"}


class FakeModelDialog:
    def __init__(self, task="detect", sam_loaded=False):
        self.loaded_model = FakeLoadedModel()
        self.task = task
        self.use_sam_dropdown = FakeComboBox("True" if sam_loaded else "False")
        self.sam_dialog = types.SimpleNamespace(loaded_model=object()) if sam_loaded else None


def test_detect_task_builds_items_and_worker_kwargs():
    contracts, _, task_module, _ = _load_batch_modules()
    item = contracts.InferenceItem("image.tif", "image.tif", "image.tif")
    dialog = FakeDialog([item])
    model_dialog = FakeModelDialog(task="detect", sam_loaded=True)

    task = task_module.make_batch_inference_task(
        "Detect", dialog, model_dialog, ["image.tif"])
    items = task.build_items()
    kwargs = task.worker_kwargs(items)

    assert items == [item]
    assert dialog.build_args == (["image.tif"], "Tiled", 3, 9, 2)
    assert kwargs["model"] is model_dialog.loaded_model
    assert kwargs["device"] == "cuda:0"
    assert kwargs["task"] == "detect"
    assert kwargs["batch_size"] == 8
    assert kwargs["sam_enabled"] is True
    assert kwargs["is_semantic"] is False
    assert kwargs["initial_thresholds"].conf == 0.42
    assert kwargs["initial_thresholds"].boundary_tolerance == 4


class FakeMaskAnnotation:
    def __init__(self):
        import numpy as np
        self.mask_data = np.ones((2, 2), dtype=np.uint8)
        self.updated = False

    def update_graphics_item(self):
        self.updated = True


class FakeImageWindow:
    def __init__(self, raster):
        self.raster_manager = FakeRasterManager({"image.tif": raster})


class FakeAnnotationWindow:
    current_image_path = "image.tif"

    def __init__(self):
        self.loaded_mask = False

    def load_mask_annotation(self):
        self.loaded_mask = True


def test_semantic_task_prepares_dialog_state_and_clears_masks():
    contracts, _, task_module, _ = _load_batch_modules()
    item = contracts.InferenceItem("image.tif", "image.tif", "image.tif")
    mask_annotation = FakeMaskAnnotation()
    raster = types.SimpleNamespace(mask_annotation=mask_annotation)
    dialog = FakeDialog([item])
    dialog.image_window = FakeImageWindow(raster)
    dialog.annotation_window = FakeAnnotationWindow()
    model_dialog = FakeModelDialog(task="semantic")
    model_dialog.predict_background_checkbox = FakeCheckBox(True)

    task = task_module.make_batch_inference_task(
        "Semantic", dialog, model_dialog, ["image.tif"])
    task.prepare_before_worker([item])
    kwargs = task.worker_kwargs([item])

    assert dialog._semantic_model_class_names == ["background", "coral"]
    assert dialog._semantic_include_bg is True
    assert dialog._semantic_processed_images == set()
    assert mask_annotation.mask_data.sum() == 0
    assert mask_annotation.updated is True
    assert dialog.annotation_window.loaded_mask is True
    assert kwargs["is_semantic"] is True
    assert kwargs["sam_enabled"] is False


def test_batch_timing_summary_aggregates_decode_model_and_gate_costs():
    _, _, _, timing = _load_batch_modules()
    aggregate = timing.BatchInferenceTiming()

    aggregate.add_record(timing.BatchTimingRecord(
        batch_size=4,
        is_video=False,
        decode_seconds=0.1,
        inference_seconds=0.4,
        postprocess_seconds=0.05,
        gate_wait_seconds=0.0,
    ))
    aggregate.add_record(timing.BatchTimingRecord(
        batch_size=1,
        is_video=True,
        decode_seconds=0.2,
        inference_seconds=0.3,
        postprocess_seconds=0.1,
        gate_wait_seconds=0.25,
    ))

    summary = aggregate.summary()
    message = timing.BatchInferenceTiming.human_summary(summary)

    assert summary["batches"] == 2
    assert summary["items"] == 5
    assert summary["image_items"] == 4
    assert summary["video_items"] == 1
    assert summary["avg_batch_size"] == 2.5
    assert round(summary["decode_seconds"], 6) == 0.3
    assert round(summary["inference_seconds"], 6) == 0.7
    assert round(summary["postprocess_seconds"], 6) == 0.15
    assert round(summary["gate_wait_seconds"], 6) == 0.25
    assert "5 items" in message
    assert "ui_gate=0.25s" in message