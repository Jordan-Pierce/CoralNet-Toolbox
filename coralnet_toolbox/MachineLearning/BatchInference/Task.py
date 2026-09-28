"""Task objects that orchestrate batch inference runs."""

from __future__ import annotations

from abc import ABC
from contextlib import contextmanager
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

    def keyframes_only(self) -> bool:
        combo = getattr(self.dialog, "video_keyframes_combo", None)
        if combo is None:
            return False
        try:
            return combo.currentText() == "True"
        except Exception:
            return False

    def build_items(self) -> list[InferenceItem]:
        start, end, stride = self.video_options()
        return self.dialog._build_inference_items(
            self.image_paths,
            self.selected_inference_type(),
            video_start=start,
            video_end=end,
            video_stride=stride,
            keyframes_only=self.keyframes_only(),
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
        combo = getattr(self.dialog, "live_preview_combo", None)
        try:
            return bool(combo is None or combo.currentText() == "True")
        except Exception:
            return True

    def model_call_overrides(self) -> dict[str, Any]:
        """Extra/override kwargs merged over the worker's default predict() call."""
        return {}

    def collapse_classes(self) -> bool:
        """Collapse every detected class id to 0 (single-class generators)."""
        return False

    def result_names(self) -> dict | None:
        """Names dict stamped onto each Results object after inference."""
        return None

    def positive_class_limit(self) -> int | None:
        """Class ids at or above this are dropped before anything else happens.

        See Anything prompts can carry decoy classes (negative examples) after the
        real ones. Their detections must go before `collapse_classes` sets every id
        to 0, after which a decoy's detection is indistinguishable from a real one.
        None drops nothing.
        """
        return None

    def supports_tensor_input(self) -> bool:
        """Whether the model accepts the preprocessed BCHW video tensor path."""
        return True

    def setup_model_dialog(self) -> bool:
        """Finalize model-dialog state (class_mapping, task, imgsz, prompts)
        before the dialog builds its ResultsProcessor and worker.

        Runs on the main thread.  Return False to abort the run cleanly.
        """
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
            "model_call_overrides": self.model_call_overrides(),
            "collapse_classes": self.collapse_classes(),
            "positive_class_limit": self.positive_class_limit(),
            "result_names": self.result_names(),
            "supports_tensor_input": self.supports_tensor_input(),
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


def _class_filter_overrides(model_dialog: Any) -> dict[str, Any]:
    """The deploy dialog's unticked classes, as Ultralytics' ``classes=`` filter.

    Read once when the worker is built, so a checkbox toggled mid-run cannot
    apply to half the batch. Empty when nothing is left out.
    """
    try:
        class_ids = model_dialog.allowed_class_ids()
    except Exception:
        class_ids = None
    return {} if class_ids is None else {"classes": class_ids}


class DetectBatchInferenceTask(AsyncYoloBatchInferenceTask):
    name = "Detect"
    default_worker_task = "detect"
    progress_title = "Running Inference"

    def model_call_overrides(self) -> dict[str, Any]:
        return _class_filter_overrides(self.model_dialog)


class SegmentBatchInferenceTask(AsyncYoloBatchInferenceTask):
    name = "Segment"
    default_worker_task = "segment"
    progress_title = "Running Inference"

    def model_call_overrides(self) -> dict[str, Any]:
        return _class_filter_overrides(self.model_dialog)


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

        # Classes unticked in the deploy dialog, snapshotted for the same
        # reason as the options below.
        dialog._semantic_excluded_class_names = set(
            getattr(self.model_dialog, "excluded_class_names", None) or ())

        # Whether the finished masks become polygons, and whether touching
        # objects are separated first, are the deploy dialog's settings -- there
        # is no second set of controls here. Snapshotted rather than read live
        # because the run is asynchronous: a toggle part-way through would
        # otherwise apply to some of the batch and not the rest.
        try:
            dialog._semantic_auto_vectorize = bool(
                self.model_dialog.should_auto_vectorize())
            dialog._semantic_split_touching = bool(
                self.model_dialog.should_split_touching())
        except Exception:
            dialog._semantic_auto_vectorize = False
            dialog._semantic_split_touching = False

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
        combo = getattr(self.dialog, "review_combo", None)
        try:
            return bool(combo is not None and combo.currentText() == "True")
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


class FeatureBatchInferenceTask(BatchInferenceTask):
    """Task for batch feature map extraction and storage."""

    name = "Feature"
    progress_title = "Extracting Features"

    def run(self, progress_bar: Any) -> bool:
        """Extract feature maps for highlighted images and save to rasters."""
        if not self.image_paths:
            return False

        feature_dialog = getattr(self.dialog.main_window, "feature_deploy_model_dialog", None)
        if feature_dialog is None or feature_dialog.loaded_model is None:
            return False

        extractor = feature_dialog.loaded_model

        # Only dense-capable backbones (ViT / ConvNext) can produce feature maps.
        if not getattr(extractor, "supports_dense", False):
            print(
                f"Feature extraction skipped: '{getattr(extractor, 'model_id', '?')}' "
                "is a pooled-only model and cannot produce dense feature maps."
            )
            return False

        raster_manager = getattr(self.dialog.image_window, "raster_manager", None)
        if raster_manager is None:
            return False

        store_pooled_combo = getattr(feature_dialog, "store_pooled_combo", None)
        try:
            store_pooled = (
                store_pooled_combo is None or store_pooled_combo.currentText() == "True"
            )
        except Exception:
            store_pooled = True

        normalize_combo = getattr(feature_dialog, "normalize_combo", None)
        try:
            normalized = (
                normalize_combo is None or normalize_combo.currentText() == "True"
            )
        except Exception:
            normalized = True

        progress_bar.set_title(f"Extracting features from {len(self.image_paths)} image(s)...")
        progress_bar.start_progress(len(self.image_paths))

        try:
            import os
            import numpy as np
            from coralnet_toolbox.Features.FeatureMapCodec import save_feature_map

            for image_path in self.image_paths:
                try:
                    raster = raster_manager.get_raster(image_path)
                    if raster is None:
                        progress_bar.update_progress()
                        continue

                    # Load image as RGB -- extract_dense takes `image_rgb`.
                    # pixmap_to_numpy used to return BGR here, so every cached
                    # feature map was baked from channel-swapped pixels.
                    from coralnet_toolbox.utilities import rasterio_to_qimage, pixmap_to_numpy
                    try:
                        qimage = rasterio_to_qimage(raster.rasterio_src)
                        if qimage is None:
                            progress_bar.update_progress()
                            continue
                        from PyQt5.QtGui import QPixmap
                        pixmap = QPixmap.fromImage(qimage)
                        img_rgb = pixmap_to_numpy(pixmap)
                    except Exception:
                        progress_bar.update_progress()
                        continue

                    if img_rgb is None or img_rgb.size == 0:
                        progress_bar.update_progress()
                        continue

                    # Extract dense features
                    feature_map_dense = extractor.extract_dense(img_rgb)
                    if feature_map_dense is None or feature_map_dense.size == 0:
                        progress_bar.update_progress()
                        continue

                    # Optionally extract pooled vector
                    feature_vector = None
                    if store_pooled:
                        try:
                            feature_vector = extractor.extract_pooled(img_rgb)
                        except Exception:
                            feature_vector = None

                    # Save feature map to a per-image-directory cache.
                    cache_dir = os.path.join(
                        os.path.dirname(image_path), ".cache", "features"
                    )
                    os.makedirs(cache_dir, exist_ok=True)

                    basename = os.path.splitext(os.path.basename(image_path))[0]
                    npy_path = os.path.join(cache_dir, f"{basename}_features.npy")

                    save_feature_map(
                        npy_path,
                        feature_map_dense,
                        model_id=extractor.model_id,
                        stride=extractor.patch_stride or 16,
                        dim=extractor.out_channels,
                        normalized=normalized,
                        feature_vector=feature_vector,
                        upsampler=getattr(extractor, "upsample_descriptor", None),
                    )

                    # Update raster with feature map metadata and path
                    raster.add_feature_map(
                        None,  # Lazy load from path
                        model_id=extractor.model_id,
                        stride=extractor.patch_stride or 16,
                        dim=extractor.out_channels,
                        path=npy_path,
                        normalized=normalized,
                        feature_vector=feature_vector,
                    )

                except Exception as e:
                    print(f"Feature extraction failed for {image_path}: {e}")

                finally:
                    progress_bar.update_progress()

            return True

        except Exception as e:
            print(f"Feature extraction error: {e}")
            return False
        finally:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


class FeatureDensifyBatchInferenceTask(BatchInferenceTask):
    """Densify each raster's existing labels into its mask over dense features.

    Regions are the whole image when Type is Standard, and only the raster's
    work areas when Type is Tiled (rasters without any are skipped).
    Orthomosaics need Tiled, since a whole mosaic squeezed into one model input
    is far too coarse. Per raster, in three passes:

      1. extract every region's features, accumulating image-wide channel
         statistics and the feature vectors under each seed;
      2. standardize with those statistics, score every region against the
         image-wide class prototypes, and decide which patches to replace;
      3. classify every region and paint the pixels it owns into the mask.

    A pixel in several regions belongs to the one it sits most centrally in
    (LabelPropagation.owned_mask), which also keeps a seed in an overlap from
    voting twice. Only the mask itself is ever held at image size. Batch edits
    are not undoable, so replacing patches is opt-in and confirmed by the dialog.
    """

    name = "Feature Densify"
    progress_title = "Densifying Patches"

    # Region edge (px) classified at before a nearest resize to full size;
    # bounds the [classes, H, W] float stack.
    MAX_CLASSIFY_EDGE = 2048
    # Region features kept in memory between passes 1 and 2; past this budget
    # regions are re-extracted instead (slower, but flat memory on orthomosaics).
    FEATURE_CACHE_BYTES = 1_500_000_000
    # Cells kept per seed per region, so a big polygon or mask class adds a
    # bounded number of vectors however many regions it spans.
    MAX_SEED_CELLS = 64

    def run(self, progress_bar: Any) -> bool:
        from collections import Counter

        extractor = getattr(self.model_dialog, "loaded_model", None)
        if extractor is None or not getattr(extractor, "supports_dense", False):
            return False
        raster_manager = getattr(self.dialog.image_window, "raster_manager", None)
        if raster_manager is None:
            return False

        options = self.dialog.get_densify_options()
        tiled = self.selected_inference_type() == "Tiled"
        annotation_window = self.dialog.annotation_window
        skipped = Counter()

        plans = []
        for image_path in self.image_paths:
            regions, reason = self._regions(raster_manager.get_raster(image_path), tiled)
            if regions is None:
                skipped[reason] += 1
            else:
                plans.append((image_path, regions))

        # Two feature passes per region; the third only upsamples stored scores.
        progress_bar.set_title(f"Densifying patches on {len(plans)} image(s)...")
        progress_bar.start_progress(max(1, 2 * sum(len(regions) for _, regions in plans)))
        densified, replaced, kept = [], [], 0
        try:
            for image_path, regions in plans:
                if progress_bar.wasCanceled():
                    break
                raster = raster_manager.get_raster(image_path)
                try:
                    result, reason = self._densify(raster, regions, extractor, options, progress_bar)
                except Exception as e:
                    print(f"Densify failed for {image_path}: {e}")
                    result, reason = None, "failed, see console"
                if result is None:
                    skipped[reason] += 1
                    continue
                densified.append(image_path)
                replaced.extend(result[0])
                kept += len(result[1])
        finally:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # Deleted once at the end: delete_annotations refreshes the UI per call.
        if replaced:
            annotation_window.delete_annotations(replaced, record_action=False)
        for image_path in densified:
            self.dialog.image_window.update_image_annotations(image_path)
        annotation_window.viewport().update()

        message = f"Densified {len(densified)} image(s)"
        if options["replace_patches"]:
            message += f", replaced {len(replaced)} patch(es), kept {kept} that disagree"
        if skipped:
            message += "; skipped: " + ", ".join(f"{reason} ({n})" for reason, n in skipped.items())
        self.dialog.main_window.status_bar.showMessage(message + ".", 15000)
        return True

    @staticmethod
    def _regions(raster, tiled):
        """``[(window, work_area)]`` to densify, or ``(None, skip reason)``.

        Windows are integer ``(x0, y0, x1, y1)`` pixel bounds; ``work_area`` is
        the WorkArea a window came from (None for the whole image).
        """
        kind = getattr(raster, "raster_type", "")
        if kind not in ("ImageRaster", "OrthoRaster"):
            return None, "not an image raster or orthomosaic"
        if not tiled:
            if kind == "OrthoRaster":
                return None, "orthomosaic with Type Standard (use Tiled)"
            return [((0, 0, raster.width, raster.height), None)], None
        # Tiled runs only on work areas, like every other batch task.
        regions = []
        for work_area in raster.get_work_areas():
            rect = work_area.rect
            x0, y0 = max(0, int(rect.left())), max(0, int(rect.top()))
            x1 = min(raster.width, int(round(rect.right())))
            y1 = min(raster.height, int(round(rect.bottom())))
            if x1 > x0 and y1 > y0:
                regions.append(((x0, y0, x1, y1), work_area))
        return (regions, None) if regions else (None, "no work areas")

    @staticmethod
    @contextmanager
    def _highlighted(work_area):
        """Show ``work_area`` highlighted while it is processed, as the Deploy dialogs do.

        Only work areas drawn on the current image are visible. Events are
        flushed first so the highlight paints before the heavy work starts.
        """
        if work_area is None:
            yield
            return
        from PyQt5.QtWidgets import QApplication

        work_area.highlight()
        QApplication.processEvents()
        try:
            yield
        finally:
            work_area.unhighlight()

    def _densify(self, raster, regions, extractor, options, progress_bar):
        """Densify one raster; ``((replaced, kept), None)`` or ``(None, skip reason)``."""
        import numpy as np
        from coralnet_toolbox.Features.QueryEngine import QueryEngine
        from coralnet_toolbox.Features.LabelPropagation import (
            ChannelStats, annotations_in, iter_owned_seeds, owned_mask, owner_of,
            partition_patches, patches_where, standardize_features, write_prediction,
        )

        windows = [window for window, _ in regions]
        work_areas = [work_area for _, work_area in regions]
        annotation_window = self.dialog.annotation_window
        annotations = list(annotation_window.get_image_annotations(raster.image_path))
        if not annotations and raster.mask_annotation is None:
            progress_bar.advance_progress(2 * len(windows))
            return None, "no annotations"
        local = [annotations_in(annotations, window) for window in windows]

        # Pass 1: features, image-wide channel statistics, seed vectors.
        stats, seeds, cache, grids, cached = ChannelStats(), {}, [], [], 0
        for t, window in enumerate(windows):
            if progress_bar.wasCanceled():
                return None, "canceled"
            with self._highlighted(work_areas[t]):
                features = self._extract(raster, window, extractor)
                grids.append(features.shape[:2])
                features = features.reshape(-1, features.shape[2])
                stats.add(features)
                for label, cells, _ in iter_owned_seeds(local[t], windows, t, grids[t],
                                                        raster.mask_annotation, self.MAX_SEED_CELLS):
                    seeds.setdefault(label.id, []).append(features[cells])
                keep = cached + features.nbytes <= self.FEATURE_CACHE_BYTES
                cache.append(features if keep else None)
                cached += features.nbytes if keep else 0
            progress_bar.update_progress()

        if len(seeds) < options["min_classes"]:
            progress_bar.advance_progress(len(windows))
            return None, f"fewer than {options['min_classes']} seeded classes"

        # Image-wide prototypes, in the shared standardized space.
        channel_stats = stats.result()
        centroids = QueryEngine.balance_vectors(
            {key: standardize_features(np.concatenate(v), channel_stats) for key, v in seeds.items()})
        mask_annotation = annotation_window._get_mask_annotation_for_bake(raster.image_path)
        if mask_annotation is None:
            progress_bar.advance_progress(len(windows))
            return None, "no mask available"
        label_to_class = mask_annotation.label_id_to_class_id_map
        # Score row k is centroids' k-th class; label maps are -1 when unlabeled,
        # so shift by one into a class-id lookup.
        lut = np.zeros(len(centroids) + 1, dtype=np.uint8)
        for k, key in enumerate(centroids):
            lut[k + 1] = label_to_class.get(key, 0)

        # Pass 2: score every region, and decide replacements before anything is
        # written, since a patch's footprint can reach into a neighbouring region.
        scores, replaced, kept = [], [], []
        for t, window in enumerate(windows):
            if progress_bar.wasCanceled():
                return None, "canceled"
            with self._highlighted(work_areas[t]):
                features = cache[t]
                if features is None:
                    features = self._extract(raster, window, extractor)
                    features = features.reshape(-1, features.shape[2])
                cache[t] = None
                engine = QueryEngine(standardize_features(features, channel_stats),
                                     np.ones(features.shape[0], dtype=bool))
                scores.append(engine.scores_from_centroids(centroids)[0].astype(np.float16))
                if options["replace_patches"]:
                    patches = patches_where(local[t], lambda x, y: owner_of(x, y, windows) == t)
                    if patches:
                        prediction = self._prediction(scores[t], grids[t], window, lut,
                                                      options["reject"])
                        agree, disagree = partition_patches(patches, prediction, label_to_class,
                                                            window[:2])
                        replaced += agree
                        kept += disagree
            progress_bar.update_progress()

        # Pass 3: paint the pixels each region owns.
        for t, window in enumerate(windows):
            with self._highlighted(work_areas[t]):
                x0, y0, x1, y1 = window
                prediction = self._prediction(scores[t], grids[t], window, lut, options["reject"])
                if options["fill_unlabeled_only"]:
                    current = mask_annotation.mask_data[y0:y1, x0:x1]
                    prediction[(current % mask_annotation.LOCK_BIT) != 0] = 0
                region = owned_mask(t, windows) if len(windows) > 1 else None
                write_prediction(mask_annotation, prediction, (x0, y0), local[t], replaced, region)
        return (replaced, kept), None

    @staticmethod
    def _extract(raster, window, extractor):
        """Dense [h, w, C] features for one window, read straight from the file."""
        from PyQt5.QtCore import QRectF
        from coralnet_toolbox.utilities import work_area_to_numpy

        x0, y0, x1, y1 = window
        image = work_area_to_numpy(raster.rasterio_src, QRectF(x0, y0, x1 - x0, y1 - y0))
        if image is None or image.size == 0:
            raise ValueError(f"could not read window {window}")
        return extractor.extract_dense(image)

    def _prediction(self, scores, grid_hw, window, lut, reject):
        """Full-size class-id prediction for ``window`` from its per-class scores."""
        import cv2
        from coralnet_toolbox.Features.LabelPropagation import label_map_from_scores

        x0, y0, x1, y1 = window
        h, w = y1 - y0, x1 - x0
        scale = min(1.0, self.MAX_CLASSIFY_EDGE / max(h, w))
        out_hw = (max(1, round(h * scale)), max(1, round(w * scale)))
        prediction = lut[label_map_from_scores(scores, grid_hw, out_hw, reject) + 1]
        if out_hw != (h, w):
            prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_NEAREST)
        return prediction


class SamBatchInferenceTask(AsyncYoloBatchInferenceTask):
    """SAM / FastSAM segment-everything generator routed through the worker.

    The loaded model is an ultralytics SAM/FastSAM model returning Results
    objects, so it shares the worker pipeline; only the predict() kwargs and
    a single-class collapse differ from plain YOLO.
    """

    name = "SAM"
    default_worker_task = "detect"
    progress_title = "Running SAM Generator"

    def setup_model_dialog(self) -> bool:
        md = self.model_dialog
        # Resolve the task from the dropdown (detect / segment).
        try:
            md.task = md.use_task_dropdown.currentText()
        except Exception:
            md.task = getattr(md, "task", "detect")
        # Sync imgsz from the spinbox so the worker uses the user's value.
        try:
            md.imgsz = md.get_imgsz()
        except Exception:
            pass
        # Ensure the single-class mapping is populated.
        try:
            md.update_class_mapping()
        except Exception:
            pass
        if not getattr(md, "class_mapping", None):
            try:
                label = self.dialog.main_window.label_window.labels[0]
                md.class_mapping = {0: label}
            except Exception:
                md.class_mapping = {}
        return bool(getattr(md, "class_mapping", None))

    # SAM is its own segmentation source; never run the separate SAM post-pass.
    def sam_enabled(self) -> bool:
        return False

    def worker_task(self) -> str:
        return getattr(self.model_dialog, "task", "detect")

    def collapse_classes(self) -> bool:
        return True

    def result_names(self) -> dict | None:
        mapping = getattr(self.model_dialog, "class_mapping", None)
        try:
            return {0: mapping[0].short_label_code}
        except Exception:
            return None

    def supports_tensor_input(self) -> bool:
        return False

    def model_call_overrides(self) -> dict[str, Any]:
        md = self.model_dialog
        task = getattr(md, "task", "detect")
        overrides: dict[str, Any] = {
            "agnostic_nms": False,
            "retina_masks": task == "segment",
        }
        # FP32 for MobileSAM (crashes in FP16) and on the CPU (FP16 there is
        # far slower for SAM); the dialog decides, as for its own calls.
        try:
            overrides["quantize"] = md.get_quantize()
        except Exception:
            pass
        return overrides


class SeeAnythingBatchInferenceTask(AsyncYoloBatchInferenceTask):
    """See Anything (YOLOE / VPE) generator routed through the worker.

    YOLOE is an ultralytics model returning Results objects.  The visual
    prompts are baked into the model up-front (set_classes), so each predict()
    call just passes an empty visual_prompts list like the interactive path.
    """

    name = "See Anything"
    default_worker_task = "detect"
    progress_title = "Running See Anything Generator"

    def setup_model_dialog(self) -> bool:
        from PyQt5.QtWidgets import QMessageBox

        md = self.model_dialog
        # Resolve the output label: what detections are saved as, which is no
        # longer the label of the annotations used as examples.
        label = getattr(md, "output_label", None)
        if label is None:
            try:
                label = md.output_label_combo.currentData()
                md.output_label = label
            except Exception:
                label = None
        if label is None:
            QMessageBox.warning(
                self.dialog, "See Anything",
                "Choose a label under 'Save detections as' in the See Anything Generator first.")
            return False
        md.class_mapping = {0: label}

        # Resolve the task, honouring the SAM-polygon override.
        try:
            md.task = md.use_task_dropdown.currentText()
        except Exception:
            md.task = getattr(md, "task", "detect")
        try:
            md.update_sam_task_state()
        except Exception:
            pass

        # If SAM polygons are requested, make sure the dialog's sam_dialog
        # points at the loaded SAM predictor so the worker's sam_enabled()
        # check and _apply_sam_to_cache can find it.
        try:
            if md.use_sam_dropdown.currentText() == "True":
                md.sam_dialog = getattr(
                    self.dialog.main_window, "sam_deploy_predictor_dialog", None)
        except Exception:
            pass

        # Sync imgsz from the spinbox, snapped to the stride exactly as the
        # Generator embeds at -- the worker predicts at this size.
        try:
            md.imgsz = md.get_imgsz()
        except Exception:
            try:
                md.imgsz = md.imgsz_spinbox.value()
            except Exception:
                pass

        # Configure the model with the prompt (heavy, main thread). Examples from
        # annotations embedded at another image size are embedded again at this
        # one first, before the worker starts.
        try:
            ok = md._setup_model_with_vpes()
        except Exception as e:
            QMessageBox.critical(
                self.dialog, "See Anything", f"Prompt setup failed: {e}")
            return False
        return bool(ok)

    def worker_task(self) -> str:
        return getattr(self.model_dialog, "task", "detect")

    def collapse_classes(self) -> bool:
        return True

    def positive_class_limit(self) -> int | None:
        # Set by the Generator's _setup_model_with_vpes when its prompt has decoys.
        return getattr(self.model_dialog, "n_positive_classes", None)

    def result_names(self) -> dict | None:
        label = getattr(self.model_dialog, "output_label", None)
        try:
            return {0: label.short_label_code}
        except Exception:
            return None

    def supports_tensor_input(self) -> bool:
        return False

    def sam_enabled(self) -> bool:
        md = self.model_dialog
        try:
            use_sam = md.use_sam_dropdown.currentText() == "True"
        except Exception:
            use_sam = False
        sam_dialog = getattr(md, "sam_dialog", None)
        return bool(
            use_sam
            and sam_dialog is not None
            and getattr(sam_dialog, "loaded_model", None) is not None
        )

    def model_call_overrides(self) -> dict[str, Any]:
        md = self.model_dialog
        task = getattr(md, "task", "detect")
        # YOLOE prompts are embedded in the model; pass an empty prompt list.
        # quantize=False matches the interactive path and avoids dtype mismatches
        # with the FP32 VPE prompt embeddings.
        return {
            "visual_prompts": [],
            "agnostic_nms": False,
            "quantize": 32,
            "retina_masks": task == "segment",
        }


_TASKS = {
    "Detect": DetectBatchInferenceTask,
    "Segment": SegmentBatchInferenceTask,
    "Semantic": SemanticBatchInferenceTask,
    "Classify": ClassifyBatchInferenceTask,
    "Feature": FeatureBatchInferenceTask,
    "SAM": SamBatchInferenceTask,
    "See Anything": SeeAnythingBatchInferenceTask,
}


def make_batch_inference_task(selected_model: str, dialog: Any, model_dialog: Any,
                              image_paths: list[str]) -> BatchInferenceTask:
    task_cls = _TASKS.get(selected_model, BatchInferenceTask)
    if task_cls is FeatureBatchInferenceTask and dialog.get_densify_options() is not None:
        task_cls = FeatureDensifyBatchInferenceTask
    return task_cls(dialog, model_dialog, image_paths)