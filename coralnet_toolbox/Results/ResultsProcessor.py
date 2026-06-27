from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QColor

import torch
from ultralytics.utils.nms import TorchNMS

from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ResultsProcessor:
    def __init__(self, main_window, class_mapping={}):
        self.main_window = main_window
        self.label_window = main_window.label_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window
        self.class_mapping = class_mapping
    
    def _get_uncertainty_thresh(self):
        """Get the current uncertainty threshold from main_window."""
        return self.main_window.get_uncertainty_thresh()
    
    def _get_iou_thresh(self):
        """Get the current IoU threshold from main_window."""
        return self.main_window.get_iou_thresh()
    
    def _get_area_thresh_min(self):
        """Get the current minimum area threshold from main_window."""
        return self.main_window.get_area_thresh_min()
    
    def _get_area_thresh_max(self):
        """Get the current maximum area threshold from main_window."""
        return self.main_window.get_area_thresh_max()
    
    def filter_by_uncertainty(self, results):
        """
        Filter the results based on the uncertainty threshold.
        """
        try:
            results = results[results.boxes.conf > self._get_uncertainty_thresh()]
        except Exception as e:
            print(f"Warning: Failed to filter results by uncertainty\n{e}")

        return results

    def filter_by_iou(self, results):
        """Filter the results based on the IoU threshold."""
        try:
            results = results[TorchNMS.fast_nms(results.boxes.xyxy, results.boxes.conf, self._get_iou_thresh())]
        except Exception as e:
            print(f"Warning: Failed to filter results by IoU\n{e}")

        return results

    def filter_by_area(self, results):
        """
        Filter the results based on the area threshold.
        """
        try:
            x_norm, y_norm, w_norm, h_norm = results.boxes.xywhn.T
            area_norm = w_norm * h_norm
            results = results[(area_norm >= self._get_area_thresh_min()) &
                              (area_norm <= self._get_area_thresh_max())]
        except Exception as e:
            print(f"Warning: Failed to filter results by area\n{e}")

        return results

    def apply_filters_to_results(self, results):
        """Check if the results passed all filters."""
        results = self.filter_by_uncertainty(results)
        results = self.filter_by_iou(results)
        results = self.filter_by_area(results)
        return results

    def indices_pass_uncertainty(self, results):
        """
        Get the indices of results that pass the uncertainty threshold.
        """
        try:
            mask = (results.boxes.conf > self._get_uncertainty_thresh()) & (results.boxes.conf < 1.0)
            indices = mask.nonzero().flatten().tolist()
        except Exception as e:
            print(f"Warning: Failed to get indices for uncertainty\n{e}")
            indices = []

        return indices

    def indices_pass_iou(self, results):
        """
        Get the indices of results that pass the IoU threshold.
        """
        try:
            indices = TorchNMS.fast_nms(results.boxes.xyxy, results.boxes.conf, self._get_iou_thresh()).tolist()
        except Exception as e:
            print(f"Warning: Failed to get indices for IoU\n{e}")
            indices = []

        return indices

    def indices_pass_area(self, results):
        """
        Get the indices of results that pass the area threshold.
        """
        try:
            x_norm, y_norm, w_norm, h_norm = results.boxes.xywhn.T
            area_norm = w_norm * h_norm
            mask = (area_norm >= self._get_area_thresh_min()) & (area_norm <= self._get_area_thresh_max())
            indices = mask.nonzero().flatten().tolist()
        except Exception as e:
            print(f"Warning: Failed to get indices for area\n{e}")
            indices = []

        return indices

    def indices_pass_filters(self, results):
        """
        Get the indices of results that pass all filters.
        """
        indices_uncertainty = set(self.indices_pass_uncertainty(results))
        indices_iou = set(self.indices_pass_iou(results))
        indices_area = set(self.indices_pass_area(results))

        # Get the indexes of results that pass all filters (intersection)
        indices = indices_uncertainty.intersection(indices_iou).intersection(indices_area)

        return list(indices)
    
    def _process_single_result_set(self, results, model_type, existing_annotations=None):
        """
        Processes a *single* Results object from a single work area.
        
        Optimized to eliminate per-item GPU-to-CPU syncs during loop processing.
        """
        annotations_to_add = []
        
        if not results or not results.boxes: 
            return annotations_to_add
            
        image_path = results.path.replace("\\", "/")
        
        # --- 1. Filter new results by uncertainty and area ---
        indices_uncertainty = set(self.indices_pass_uncertainty(results))
        indices_area = set(self.indices_pass_area(results))
        filtered_indices = list(indices_uncertainty.intersection(indices_area))
        
        if not filtered_indices:
            return annotations_to_add

        # --- 2. FAST EXTRACT: Move ALL necessary data to CPU once ---
        # By doing this here, we avoid calling .cpu() inside the for loop below
        new_bboxes = results.boxes.xyxy[filtered_indices].cpu()
        new_scores = results.boxes.conf[filtered_indices].cpu()
        new_classes = results.boxes.cls[filtered_indices].cpu().numpy().astype(int)
        
        # Pre-convert to numpy arrays for lightning-fast reading in the loop
        np_bboxes = new_bboxes.numpy()
        np_scores = new_scores.numpy()
        
        # --- 3. Fast-Extract Existing Annotations ---
        if existing_annotations is None:
            existing_annotations = self.annotation_window.get_image_annotations(image_path)
            existing_annotations = [a for a in existing_annotations if (isinstance(a, RectangleAnnotation) or
                                                                        isinstance(a, PolygonAnnotation))]

        if existing_annotations:
            ext_boxes = []
            for ann in existing_annotations:
                tl = ann.get_bounding_box_top_left()
                br = ann.get_bounding_box_bottom_right()
                ext_boxes.append([tl.x(), tl.y(), br.x(), br.y()])
            
            existing_bboxes = torch.tensor(ext_boxes, dtype=torch.float32)
            existing_scores = torch.ones(len(ext_boxes), dtype=torch.float32) 
            
            all_bboxes = torch.cat([existing_bboxes, new_bboxes])
            all_scores = torch.cat([existing_scores, new_scores])
            num_existing = len(existing_bboxes)
        else:
            all_bboxes = new_bboxes
            all_scores = new_scores
            num_existing = 0

        # --- 4. Run NMS (Class-Agnostic) ---
        try:
            keep_indices_tensor = TorchNMS.fast_nms(all_bboxes, all_scores, self._get_iou_thresh())
            keep_indices = keep_indices_tensor.tolist()  
        except Exception as e:
            print(f"Warning: Stage 2 NMS (TorchNMS) failed: {e}")
            keep_indices = list(range(len(all_bboxes)))
            
        # --- 5. Unpack Survivors and Create Annotations ---
        # Cache labels to avoid repeated dictionary lookups
        label_cache = {}
        
        for idx in keep_indices:
            if idx >= num_existing:
                # Calculate the relative index in our CPU pre-fetched arrays
                relative_idx = idx - num_existing
                original_yolo_idx = filtered_indices[relative_idx]
                
                try:
                    # Read instantly from our CPU numpy arrays
                    xyxy = np_bboxes[relative_idx]
                    conf = float(np_scores[relative_idx])
                    cls_id = new_classes[relative_idx]
                    cls_name = results.names[cls_id]
                    
                    # Use cache for label lookups
                    cache_key = (cls_name, conf <= self._get_uncertainty_thresh())
                    if cache_key not in label_cache:
                        short_label = self.get_mapped_short_label(cls_name, conf)
                        label_cache[cache_key] = self.label_window.get_label_by_short_code(short_label)
                    
                    label = label_cache[cache_key]
                    
                    annotation = None
                    if model_type == 'detection':
                        xmin, ymin, xmax, ymax = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                        annotation = self.create_rectangle_annotation(xmin, ymin, xmax, ymax, label, image_path)
                    
                    elif model_type == 'segmentation':
                        if results.masks and original_yolo_idx < len(results.masks.xy):
                            xy = results.masks.xy[original_yolo_idx]
                            points = [(float(x), float(y)) for x, y in xy]
                            annotation = self.create_polygon_annotation(points, label, image_path)

                    if annotation:
                        processed_annotation = self._post_process_new_annotation(annotation, cls_name, conf)
                        annotations_to_add.append(processed_annotation)
                
                except Exception as e:
                    print(f"Warning: Failed to create annotation for survivor: {e}")
        
        return annotations_to_add

    def _convert_annotations_to_nms_format(self, annotations):
        """Convert existing annotations to NMS format using their built-in method."""
        nms_detections = []
        
        for annotation in annotations:
            try:
                # Use the annotation's built-in NMS conversion method
                nms_detection = annotation.to_nms_detection()
                # Override confidence to 1.0 to ensure existing annotations
                # always win NMS against new predictions (which are <= 1.0).
                nms_detection['confidence'] = 1.0
                nms_detections.append(nms_detection)
            except Exception as e:
                print(f"Warning: Failed to convert annotation {annotation.id} to NMS format: {e}")
        
        return nms_detections
    
    def generate_fast_render_paths(self, results, model_type):
        """
        Ultra-fast method to generate Qt rendering paths directly from YOLO tensors.
        Used to draw boxes instantly without freezing the UI.
        """
        paths_data = []
        
        if not results or not results.boxes:
            return paths_data
            
        try:
            # Move data to CPU once
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            # Pre-fetch colors to avoid dictionary lookups in the loop
            color_cache = {}
            transparency = self.main_window.get_transparency_value()
            uncertainty_thresh = self._get_uncertainty_thresh()
            
            for i in range(len(confidences)):
                conf = confidences[i]
                cls_id = class_ids[i]
                
                if cls_id not in color_cache:
                    cls_name = results.names[cls_id]
                    short_label = self.get_mapped_short_label(cls_name, conf)
                    label = self.label_window.get_label_by_short_code(short_label)
                    color_cache[cls_id] = label.color if label else QColor(255, 255, 255)
                
                color = color_cache[cls_id]
                if conf < uncertainty_thresh:
                    review_label = self.label_window.get_label_by_id('-1')
                    if review_label:
                        color = review_label.color
                
                from PyQt5.QtGui import QPainterPath
                from PyQt5.QtCore import QPointF, QRectF
                from PyQt5.QtGui import QPolygonF
                path = QPainterPath()
                
                if model_type == 'segmentation' and results.masks:
                    if i < len(results.masks.xy):
                        xy_coords = results.masks.xy[i]
                        if len(xy_coords) > 2:
                            # Extremely fast polygon generation from array
                            qt_poly = QPolygonF([QPointF(xy_coords[j][0], xy_coords[j][1]) 
                                               for j in range(len(xy_coords))])
                            path.addPolygon(qt_poly)
                            path.closeSubpath()
                            paths_data.append((path, color, transparency))
                            
                elif model_type in ['detection', 'detect', 'segment']:
                    box = results.boxes.xyxy[i].cpu().numpy()
                    path.addRect(QRectF(box[0], box[1], box[2] - box[0], box[3] - box[1]))
                    paths_data.append((path, color, transparency))
                    
        except Exception as e:
            print(f"Warning: Fast path generation failed: {e}")
            
        return paths_data
        
    def get_mapped_short_label(self, cls_name, conf):
        """
        Get the short label for a detection result based on confidence and class mapping.

        :param cls_name: Class name
        :param conf: Confidence score
        :return: Short label as a string
        """
        if conf <= self._get_uncertainty_thresh():
            return 'Review'
        return self.class_mapping.get(cls_name, {}).get('short_label_code', 'Review')

    def create_rectangle_annotation(self, xmin, ymin, xmax, ymax, label, image_path):
        """
        Create a rectangle annotation for the given bounding box coordinates and label.

        :param xmin: Minimum x-coordinate
        :param ymin: Minimum y-coordinate
        :param xmax: Maximum x-coordinate
        :param ymax: Maximum y-coordinate
        :param label: Label object
        :param image_path: Path to the image
        :return: RectangleAnnotation object
        """
        try:
            top_left = QPointF(xmin, ymin)
            bottom_right = QPointF(xmax, ymax)
            annotation = RectangleAnnotation(
                top_left,
                bottom_right,
                label,
                image_path,
                transparency=self.main_window.get_transparency_value()
            )
        except Exception:
            annotation = None

        return annotation

    def create_polygon_annotation(self, points, label, image_path):
        """
        Create a polygon annotation for the given points and label.

        :param points: List of polygon points
        :param label: Label object
        :param image_path: Path to the image
        :return: PolygonAnnotation object
        """
        try:
            points = [QPointF(x, y) for x, y in points]
            annotation = PolygonAnnotation(
                points,
                label,
                image_path,
                transparency=self.main_window.get_transparency_value()
            )
        except Exception:
            annotation = None

        return annotation

    def extract_classification_result(self, result):
        """
        Extract relevant information from a classification result.
        """        
        predictions = {}

        try:
            image_path = result.path.replace("\\", "/")
            class_names = result.names
            top1 = result.probs.top1
            top1conf = result.probs.top1conf
            top1cls = class_names[top1]
            top5 = result.probs.top5
            top5conf = result.probs.top5conf
        except Exception as e:
            print(f"Warning: Failed to process classification result\n{e}")
            return None, None, None, {}

        for idx, conf in zip(top5, top5conf):
            class_name = class_names[idx]
            label = self.label_window.get_label_by_short_code(class_name)
            if label:
                predictions[label] = float(conf)

        return image_path, top1cls, top1conf, predictions

    def process_classification_results(self, results_list, annotations, progress_bar=None):
        """
        Process the classification results from the results generator.

        Args:
            results_list: List of classification results
            annotations: List of annotations to update
            progress_bar: Optional external progress bar. If None, creates its own.
        """
        # Track if we created the progress bar ourselves
        progress_bar_created_here = progress_bar is None

        if progress_bar is None:
            progress_bar = ProgressBar(self.annotation_window, title="Making Classification Predictions")
            progress_bar.show()

        progress_bar.start_progress(len(annotations))

        # Collect work so we can defer all UI side-effects until after the loop.
        # Key: set of image_paths that had at least one label change (for the
        # image-table count update), and a flag for whether the current image
        # needs its cropped-image cache refreshed.
        dirty_image_paths = set()
        label_changed_pairs = []      # [(annotation_id, label_id), ...]
        refresh_current_image = False  # need create_cropped_image on visible patches
        selected_annotation = None     # last selected annotation to show in confidence window

        try:
            for result, annotation in zip(results_list, annotations):
                if result:
                    try:
                        image_path, cls_name, conf, predictions = self.extract_classification_result(result)
                        if image_path is None:
                            continue

                        old_label_id = annotation.label.id
                        self._update_classification_data(annotation, cls_name, conf, predictions)

                        # Track what changed — actual UI calls happen after the loop
                        dirty_image_paths.add(annotation.image_path)

                        if old_label_id != annotation.label.id:
                            label_changed_pairs.append((annotation.id, annotation.label.id))

                        if annotation.image_path == self.annotation_window.current_image_path:
                            refresh_current_image = True
                            if annotation in self.annotation_window.selected_annotations:
                                selected_annotation = annotation

                    except Exception as e:
                        print(f"Warning: Failed to process classification result for annotation {annotation.id}\n{e}")

                if progress_bar:
                    progress_bar.update_progress()

        finally:
            # ── Deferred UI flush ──────────────────────────────────────────
            from PyQt5.QtWidgets import QApplication

            # 1. Emit label-change signals in one batch
            for ann_id, label_id in label_changed_pairs:
                try:
                    self.annotation_window.annotationLabelChanged.emit(ann_id, label_id)
                except Exception as e:
                    print(f"Warning: Failed to emit label change for {ann_id}: {e}")

            # 2. Refresh cropped-image cache for visible patches (one pass)
            if refresh_current_image:
                rasterio_src = getattr(self.annotation_window, 'rasterio_image', None)
                if rasterio_src:
                    current_path = self.annotation_window.current_image_path
                    for annotation in annotations:
                        if annotation.image_path == current_path:
                            try:
                                annotation.create_cropped_image(rasterio_src)
                            except Exception:
                                pass

            # 3. Update confidence window for the selected annotation (if any)
            if selected_annotation is not None:
                try:
                    self.main_window.confidence_window.display_cropped_image(selected_annotation)
                except Exception:
                    pass

            # 4. Update image-table annotation counts — one call per unique path
            for image_path in dirty_image_paths:
                try:
                    self.image_window.update_image_annotations(image_path)
                except Exception:
                    pass

            # 5. Close the progress bar if we own it
            if progress_bar_created_here and progress_bar:
                progress_bar.stop_progress()
                progress_bar.close()

            # 6. Rebuild the phantom layer once
            try:
                self.annotation_window.refresh_phantom_annotations()
                self.annotation_window.viewport().update()
                QApplication.processEvents()
            except Exception:
                pass
        
    def _update_classification_data(self, annotation, cls_name, conf, predictions):
        """
        Updates an annotation's data model with classification results.
        Pure data update — no UI side-effects.  UI refresh is done in a
        single deferred pass by the caller.

        :param annotation: Annotation object to update.
        :param cls_name: The top predicted class name.
        :param conf: The top confidence score.
        :param predictions: Dictionary of all class predictions.
        """
        # Update the machine confidence values with all predictions (top5)
        annotation.update_machine_confidence(predictions)

        current_uncertainty_thresh = self.main_window.get_uncertainty_thresh()

        if conf < current_uncertainty_thresh:
            final_label = self.label_window.get_label_by_id('-1')  # Review label
            annotation.label = final_label
            annotation.verified = False
            annotation.update_graphics_item()

    def _update_and_display_classification(self, annotation, cls_name, conf, predictions):
        """
        Updates an existing annotation with classification results and refreshes the UI if visible.
        Retained for external callers; internally prefer _update_classification_data + deferred flush.

        :param annotation: Annotation object to update.
        :param cls_name: The top predicted class name.
        :param conf: The top confidence score.
        :param predictions: Dictionary of all class predictions.
        """
        old_label_id = annotation.label.id
        self._update_classification_data(annotation, cls_name, conf, predictions)

        if old_label_id != annotation.label.id:
            try:
                self.annotation_window.annotationLabelChanged.emit(annotation.id, annotation.label.id)
            except Exception as e:
                print(f"Warning: Failed to emit label change for {annotation.id}: {e}")

        if annotation.image_path == self.annotation_window.current_image_path:
            annotation.create_cropped_image(self.annotation_window.rasterio_image)
            if annotation in self.annotation_window.selected_annotations:
                self.main_window.confidence_window.display_cropped_image(annotation)

        self.image_window.update_image_annotations(annotation.image_path)

    def extract_detection_result(self, result):
        """
        Extract relevant information from a detection result.

        :param result: Detection result
        :return: Tuple containing class, class name, confidence, and bounding box coordinates
        """
        # Class ID, class name, confidence, and bounding box coordinates
        image_path = result.path.replace("\\", "/")
        cls = int(result.boxes.cls.cpu().numpy()[0])
        cls_name = result.names[cls]
        conf = float(result.boxes.conf.cpu().numpy()[0])
        xmin, ymin, xmax, ymax = map(float, result.boxes.xyxy.cpu().numpy()[0])

        return image_path, cls, cls_name, conf, xmin, ymin, xmax, ymax
    
    def extract_segmentation_result(self, result):
        """
        Extract relevant information from a segmentation result.

        :param result: Segmentation result
        :return: Tuple containing class, class name, confidence, and polygon points
        """
        # Class ID, class name, confidence, and polygon points
        image_path = result.path.replace("\\", "/")
        cls = int(result.boxes.cls.cpu().numpy()[0])
        cls_name = result.names[cls]
        conf = float(result.boxes.conf.cpu().numpy()[0])

        return image_path, cls, cls_name, conf

    def build_segmentation_annotations(self, results_list):
        """Build segmentation annotations from results without saving them.

        Returns the list of new annotations so the caller can batch-commit
        multiple images in a single ``add_annotations`` call.
        """
        all_new_annos = []
        contributing_sources = 0

        existing_by_path = {}
        for results in results_list:
            if not results or not results.boxes:
                continue
            image_path = results.path.replace("\\", "/")
            if image_path not in existing_by_path:
                anns = self.annotation_window.get_image_annotations(image_path)
                existing_by_path[image_path] = [
                    a for a in anns
                    if isinstance(a, (RectangleAnnotation, PolygonAnnotation))
                ]
            new_annos = self._process_single_result_set(
                results, 'segmentation',
                existing_annotations=existing_by_path[image_path])
            if new_annos:
                all_new_annos.extend(new_annos)
                contributing_sources += 1

        if contributing_sources > 1 and len(all_new_annos) > 1:
            all_new_annos = self._dedupe_cross_work_area(all_new_annos)

        return all_new_annos

    def build_detection_annotations(self, results_list):
        """Build detection annotations from results without saving them.

        Returns the list of new annotations so the caller can batch-commit
        multiple images in a single ``add_annotations`` call.
        """
        all_new_annos = []
        contributing_sources = 0

        existing_by_path = {}
        for results in results_list:
            if not results or not results.boxes:
                continue
            image_path = results.path.replace("\\", "/")
            if image_path not in existing_by_path:
                anns = self.annotation_window.get_image_annotations(image_path)
                existing_by_path[image_path] = [
                    a for a in anns
                    if isinstance(a, (RectangleAnnotation, PolygonAnnotation))
                ]
            new_annos = self._process_single_result_set(
                results, 'detection',
                existing_annotations=existing_by_path[image_path])
            if new_annos:
                all_new_annos.extend(new_annos)
                contributing_sources += 1

        if contributing_sources > 1 and len(all_new_annos) > 1:
            all_new_annos = self._dedupe_cross_work_area(all_new_annos)

        return all_new_annos

    def process_segmentation_results(self, results_list):
        """
        Process segmentation results. Loops through the list and processes
        one Results object at a time.
        """
        image_path = next(
            (r.path.replace("\\", "/") for r in results_list if r and r.boxes),
            None,
        )
        all_new_annos = self.build_segmentation_annotations(results_list)

        if all_new_annos:
            self.annotation_window.add_annotations(all_new_annos)
            if self.annotation_window.current_image_path == image_path:
                self.annotation_window.load_annotations()

    def process_detection_results(self, results_list):
        """
        Process detection results. Loops through the list and processes
        one Results object at a time.
        """
        image_path = next(
            (r.path.replace("\\", "/") for r in results_list if r and r.boxes),
            None,
        )
        all_new_annos = self.build_detection_annotations(results_list)

        if all_new_annos:
            self.annotation_window.add_annotations(all_new_annos)
            if self.annotation_window.current_image_path == image_path:
                self.annotation_window.load_annotations()

    def _dedupe_cross_work_area(self, new_annotations):
        """Class-agnostic NMS across annotations produced by different work areas.

        Per-work-area NMS in ``_process_single_result_set`` only sees pre-existing
        annotations, not sibling new annotations from other work areas in the same
        pass. Overlapping work areas can therefore each emit a box for the same
        object. This pass reconciles the combined set, keeping the highest-confidence
        survivor per cluster.
        """
        try:
            bboxes = []
            scores = []
            for ann in new_annotations:
                det = ann.to_nms_detection()
                bboxes.append(det['bbox'])
                scores.append(det['confidence'])

            bbox_tensor = torch.tensor(bboxes, dtype=torch.float32)
            score_tensor = torch.tensor(scores, dtype=torch.float32)

            keep = TorchNMS.fast_nms(bbox_tensor, score_tensor, self._get_iou_thresh()).tolist()
            return [new_annotations[i] for i in keep]
        except Exception as e:
            print(f"Warning: cross-work-area NMS failed: {e}")
            return new_annotations
        
    def _post_process_new_annotation(self, annotation, cls_name, conf):
        """
        Applies final processing to a newly created annotation before batch addition.
        
        :param annotation: The annotation object to process.
        :param cls_name: The original class name from the model.
        :param conf: The confidence score from the model.
        :return: The processed annotation object.
        """
        # Update the confidence values for the prediction
        predictions = {self.label_window.get_label_by_short_code(cls_name): conf}
        annotation.update_machine_confidence(predictions)

        # If the confidence is below the threshold, update the label to 'Review'
        if conf < self._get_uncertainty_thresh():
            review_label = self.label_window.get_label_by_id('-1')
            annotation.update_label(review_label)
            annotation.set_verified(False)
        
        return annotation