from PyQt5.QtCore import QPointF

import torch
from ultralytics.utils.nms import TorchNMS

from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

from coralnet_toolbox.utilities import simplify_polygon

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
            mask = results.boxes.conf > self._get_uncertainty_thresh()
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
    
    def _process_single_result_set(self, results, model_type):
        """
        Processes a *single* Results object from a single work area.
        
        This function filters the results, runs NMS against existing annotations,
        and creates a list of new annotation objects to be added.
        """
        annotations_to_add = []
        new_detections = []
        
        if not results or not results.boxes: 
            return annotations_to_add
            
        image_path = results.path.replace("\\", "/")
        
        # --- 1. Filter new results by uncertainty and area ---
        indices_uncertainty = set(self.indices_pass_uncertainty(results))
        indices_area = set(self.indices_pass_area(results))
        filtered_indices = list(indices_uncertainty.intersection(indices_area))
        
        for idx in filtered_indices:
            try:
                box = results.boxes[idx]
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = results.names[cls_id]
                
                bbox_coords = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]

                detection_data = {
                    'bbox': bbox_coords,
                    'confidence': conf,
                    'class_name': cls_name,
                    'class_id': cls_id,
                    'is_existing': False,
                    'polygon_points': None,
                    'model_type': model_type
                }
                
                if model_type == 'segmentation':
                    if results.masks and idx < len(results.masks.xy):
                        xy = results.masks.xy[idx]
                        points = [(float(x), float(y)) for x, y in xy]
                        detection_data['polygon_points'] = points
                        new_detections.append(detection_data)
                else:  # 'detection'
                    new_detections.append(detection_data)
                    
            except Exception as e:
                print(f"Warning: Failed to extract result in first pass: {e}")
        
        if not new_detections:
            return annotations_to_add
            
        # --- 2. Run NMS vs. Existing Annotations (Class-Agnostic) ---
        
        existing_annotations = self.annotation_window.get_image_annotations(image_path)
        existing_annotations = [a for a in existing_annotations if (isinstance(a, RectangleAnnotation) or 
                                                                    isinstance(a, PolygonAnnotation))]
        existing_detections = self._convert_annotations_to_nms_format(existing_annotations)
        
        # Combine existing (with 1.0 conf) and new survivors
        all_detections_combined = existing_detections + new_detections
        
        if not all_detections_combined:
            return annotations_to_add
            
        # Convert to tensors for NMS
        bboxes = torch.tensor([det['bbox'] for det in all_detections_combined], dtype=torch.float32)
        scores = torch.tensor([det['confidence'] for det in all_detections_combined], dtype=torch.float32)
        
        try:
            # Apply NMS using TorchNMS. This is class-agnostic.
            keep_indices_tensor = TorchNMS.fast_nms(bboxes, scores, self._get_iou_thresh())
            keep_indices = keep_indices_tensor.tolist()  # Convert tensor to standard list of ints
        except Exception as e:
            print(f"Warning: Stage 2 NMS (TorchNMS) failed: {e}")
            keep_indices = list(range(len(all_detections_combined)))
            
        # --- 3. Create Annotations for final survivors ---
        
        for idx in keep_indices:
            # Check if the kept index is a NEW detection
            original_detection = all_detections_combined[idx]
            if not original_detection['is_existing']:
                try:
                    conf = original_detection['confidence']
                    cls_name = original_detection['class_name']
                    
                    short_label = self.get_mapped_short_label(cls_name, conf)
                    label = self.label_window.get_label_by_short_code(short_label)
                    
                    annotation = None
                    if original_detection['model_type'] == 'detection':
                        xmin, ymin, xmax, ymax = original_detection['bbox']
                        annotation = self.create_rectangle_annotation(xmin, ymin, xmax, ymax, label, image_path)
                    
                    elif original_detection['model_type'] == 'segmentation' and original_detection['polygon_points']:
                        points = simplify_polygon(original_detection['polygon_points'], 0.1) 
                        annotation = self.create_polygon_annotation(points, label, image_path)

                    if annotation:
                        processed_annotation = self._post_process_new_annotation(annotation, cls_name, conf)
                        annotations_to_add.append(processed_annotation)
                
                except Exception as e:
                    print(f"Warning: Failed to create annotation: {e}")
        
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
            annotation = RectangleAnnotation(top_left,
                                             bottom_right,
                                             label.short_label_code,
                                             label.long_label_code,
                                             label.color,
                                             image_path,
                                             label.id,
                                             self.main_window.get_transparency_value())
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
            annotation = PolygonAnnotation(points,
                                           label.short_label_code,
                                           label.long_label_code,
                                           label.color,
                                           image_path,
                                           label.id,
                                           self.main_window.get_transparency_value())
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

        try:
            for result, annotation in zip(results_list, annotations):
                if result:
                    try:
                        # Handle both Results objects (from stream) and pre-extracted tuples (from .engine)
                        image_path, cls_name, conf, predictions = self.extract_classification_result(result)
                        if image_path is None:
                            continue
                        self._update_and_display_classification(annotation, cls_name, conf, predictions)
                    except Exception as e:
                        print(f"Warning: Failed to process classification result for annotation {annotation.id}\n{e}")
                
                # Only update progress if we have a progress bar (external or internal)
                if progress_bar:
                    progress_bar.update_progress()

        finally:
            # Only close progress bar if we created it ourselves
            if progress_bar_created_here and progress_bar:
                progress_bar.stop_progress()
                progress_bar.close()
        
    def _update_and_display_classification(self, annotation, cls_name, conf, predictions):
        """
        Updates an existing annotation with classification results and refreshes the UI if visible.

        :param annotation: Annotation object to update.
        :param cls_name: The top predicted class name.
        :param conf: The top confidence score.
        :param predictions: Dictionary of all class predictions.
        """
        # Update the machine confidence values with all predictions (top5)
        annotation.update_machine_confidence(predictions)

        # Get the current uncertainty threshold from the UI (not the cached value)
        current_uncertainty_thresh = self.main_window.get_uncertainty_thresh()
        
        # Determine the final label based on the top1 prediction confidence
        if conf < current_uncertainty_thresh:
            # If top1 confidence is below threshold, set to 'Review' label
            # We do this AFTER update_machine_confidence to preserve all predictions
            final_label = self.label_window.get_label_by_id('-1')  # Review label
            annotation.label = final_label
            annotation.verified = False
            annotation.update_graphics_item()
        else:
            # Otherwise keep the label that was set by update_machine_confidence (top1 class)
            # Note: update_machine_confidence already set annotation.label to the top prediction
            pass

        # If the annotation's image is currently displayed, refresh its visual state
        if annotation.image_path == self.annotation_window.current_image_path:
            annotation.create_cropped_image(self.annotation_window.rasterio_image)
            # Update the confidence window if this annotation is selected
            if annotation in self.annotation_window.selected_annotations:
                self.main_window.confidence_window.display_cropped_image(annotation)
    
        # Update the annotation count in the image table
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

    def process_segmentation_results(self, results_list):
        """
        Process segmentation results. Loops through the list and processes
        one Results object at a time.
        """
        all_new_annos = []
        image_path = None
        
        for results in results_list:
            if not results or not results.boxes:
                continue
            if image_path is None:
                image_path = results.path.replace("\\", "/")
                
            new_annos = self._process_single_result_set(results, 'segmentation')
            all_new_annos.extend(new_annos)
            
        if all_new_annos:
            self.annotation_window.add_annotations(all_new_annos)
            if self.annotation_window.current_image_path == image_path:
                self.annotation_window.load_annotations()

    def process_detection_results(self, results_list):
        """
        Process detection results. Loops through the list and processes
        one Results object at a time.
        """
        all_new_annos = []
        image_path = None
        
        for results in results_list:
            if not results or not results.boxes:
                continue
            if image_path is None:
                image_path = results.path.replace("\\", "/")
                
            new_annos = self._process_single_result_set(results, 'detection')
            all_new_annos.extend(new_annos)
            
        if all_new_annos:
            self.annotation_window.add_annotations(all_new_annos)
            if self.annotation_window.current_image_path == image_path:
                self.annotation_window.load_annotations()
        
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