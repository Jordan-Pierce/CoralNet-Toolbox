from PyQt5.QtCore import QPointF

from ultralytics.utils.nms import TorchNMS

from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

from coralnet_toolbox.utilities import simplify_polygon

from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ResultsProcessor:
    def __init__(self,
                 main_window,
                 class_mapping={},
                 uncertainty_thresh=None,
                 iou_thresh=None,
                 min_area_thresh=None,
                 max_area_thresh=None):
        # Initialize the ResultsProcessor with the main window
        self.main_window = main_window
        self.label_window = main_window.label_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window

        self.class_mapping = class_mapping

        if uncertainty_thresh is None:
            uncertainty_thresh = self.main_window.get_uncertainty_thresh()

        if iou_thresh is None:
            iou_thresh = main_window.get_iou_thresh()

        if min_area_thresh is None:
            min_area_thresh = main_window.get_area_thresh_min()

        if max_area_thresh is None:
            max_area_thresh = main_window.get_area_thresh_max()

        self.uncertainty_thresh = uncertainty_thresh
        self.iou_thresh = iou_thresh
        self.min_area_thresh = min_area_thresh
        self.max_area_thresh = max_area_thresh

    def filter_by_uncertainty(self, results):
        """
        Filter the results based on the uncertainty threshold.
        """
        try:
            results = results[results.boxes.conf > self.uncertainty_thresh]
        except Exception as e:
            print(f"Warning: Failed to filter results by uncertainty\n{e}")

        return results

    def filter_by_iou(self, results):
        """Filter the results based on the IoU threshold."""
        try:
            results = results[TorchNMS.fast_nms(results.boxes.xyxy, results.boxes.conf, self.iou_thresh)]
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
            results = results[(area_norm >= self.min_area_thresh) & (area_norm <= self.max_area_thresh)]
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
            mask = results.boxes.conf > self.uncertainty_thresh
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
            indices = TorchNMS.fast_nms(results.boxes.xyxy, results.boxes.conf, self.iou_thresh).tolist()
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
            mask = (area_norm >= self.min_area_thresh) & (area_norm <= self.max_area_thresh)
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
    
    def _process_results_with_global_nms(self, results_list, model_type):
        """
        Internal method to process detection or segmentation results from a list 
        (e.g., from multiple work areas) using global NMS.
        """
        if not results_list:
            return

        if model_type == 'detection':
            progress_title = "Making Detection Predictions"
        else:
            progress_title = "Making Segmentation Predictions"
            
        progress_bar = ProgressBar(self.annotation_window, title=progress_title)
        progress_bar.show()
        progress_bar.start_progress(len(results_list))

        all_new_detections = []
        image_path = None

        # --- 1. Collect all potential detections from all work areas ---
        for results in results_list:
            # Both types have .boxes for cls, conf, etc.
            if not results or not results.boxes: 
                progress_bar.update_progress()
                continue
            
            if image_path is None:
                image_path = results.path.replace("\\", "/")
            
            # Get indices passing area and uncertainty filters
            # We explicitly DO NOT run the IOU (NMS) filter here.
            indices_uncertainty = set(self.indices_pass_uncertainty(results))
            indices_area = set(self.indices_pass_area(results))
            filtered_indices = list(indices_uncertainty.intersection(indices_area))
            
            for idx in filtered_indices:
                try:
                    box = results.boxes[idx]
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    cls_name = results.names[cls]
                    
                    # Create a standardized detection dictionary
                    detection_data = {
                        'bbox': [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                        'confidence': conf,
                        'class_name': cls_name,
                        'class_id': cls,
                        'is_existing': False,  # Flag as new detection
                        'area': float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])),
                        'polygon_points': None,  # Placeholder for segmentation
                        'model_type': model_type
                    }
                    
                    if model_type == 'segmentation':
                        if results.masks and idx < len(results.masks.xy):
                            xy = results.masks.xy[idx]
                            points = [(float(x), float(y)) for x, y in xy]
                            points = simplify_polygon(points, 0.1)
                            detection_data['polygon_points'] = points
                        else:
                            continue  # Skip if segmentation has no mask data

                    all_new_detections.append(detection_data)
                except Exception as e:
                    print(f"Warning: Failed to extract result in first pass: {e}")
            
            progress_bar.update_progress()
        
        # If no detections found, or no image path, stop.
        if not all_new_detections or not image_path:
            progress_bar.stop_progress()
            progress_bar.close()
            return
            
        progress_bar.set_title(f"Running Global NMS ({len(all_new_detections)} candidates)")
        progress_bar.start_progress(3) # 3 steps: NMS1, NMS2, Create

        # --- 2. Run Stage 1 NMS (Global NMS on all new detections) ---
        import torch
        stage1_bboxes = torch.tensor([det['bbox'] for det in all_new_detections], dtype=torch.float32)
        stage1_scores = torch.tensor([det['confidence'] for det in all_new_detections], dtype=torch.float32)
        
        try:
            # Run NMS on all new detections against each other
            surviving_stage1_indices = TorchNMS.fast_nms(stage1_bboxes, stage1_scores, self.iou_thresh).tolist()
            surviving_stage1_detections = [all_new_detections[i] for i in surviving_stage1_indices]
        except Exception as e:
            print(f"Warning: Stage 1 NMS failed: {e}")
            surviving_stage1_detections = all_new_detections  # Fallback: use all
            
        progress_bar.update_progress()

        # --- 3. Run Stage 2 NMS (vs. Existing Annotations) ---
        
        # Get existing annotations
        existing_annotations = self.annotation_window.get_image_annotations(image_path)
        existing_annotations = [a for a in existing_annotations if (isinstance(a, RectangleAnnotation) or 
                                                                    isinstance(a, PolygonAnnotation))]
        existing_detections = self._convert_annotations_to_nms_format(existing_annotations)
        
        # Combine existing (with 0.9 conf) and new survivors
        all_detections_combined = existing_detections + surviving_stage1_detections
        
        if not all_detections_combined:
            progress_bar.stop_progress()
            progress_bar.close()
            return
        
        # Apply NMS to the combined list
        surviving_final_indices = self._apply_nms_to_combined_detections(all_detections_combined)
        progress_bar.update_progress()

        # --- 4. Create Annotations for final survivors ---
        annotations_to_add = []
        num_existing = len(existing_detections)
        
        final_survivors = []
        # Filter out to get only the NEW detections that survived
        for idx in surviving_final_indices:
            if idx >= num_existing:  # This is a new detection
                detection = all_detections_combined[idx]
                final_survivors.append(detection)

        progress_bar.set_title(f"Creating {len(final_survivors)} Annotations")
        progress_bar.start_progress(len(final_survivors))

        for detection in final_survivors:
            try:
                conf = detection['confidence']
                cls_name = detection['class_name']
                
                short_label = self.get_mapped_short_label(cls_name, conf)
                label = self.label_window.get_label_by_short_code(short_label)
                
                annotation = None
                if detection['model_type'] == 'detection':
                    xmin, ymin, xmax, ymax = detection['bbox']
                    annotation = self.create_rectangle_annotation(xmin, ymin, xmax, ymax, label, image_path)
                
                elif detection['model_type'] == 'segmentation' and detection['polygon_points']:
                    points = detection['polygon_points']
                    annotation = self.create_polygon_annotation(points, label, image_path)

                if annotation:
                    # Apply final confidence/label processing
                    processed_annotation = self._post_process_new_annotation(annotation, cls_name, conf)
                    annotations_to_add.append(processed_annotation)
            
            except Exception as e:
                print(f"Warning: Failed to create annotation: {e}")
            
            progress_bar.update_progress()

        # --- 5. Add to window ---
        if annotations_to_add:
            self.annotation_window.add_annotations(annotations_to_add)
        
        # Refresh the current view if we added annotations to it
        if self.annotation_window.current_image_path == image_path:
            self.annotation_window.load_annotations()

        progress_bar.stop_progress()
        progress_bar.close()

    def _convert_annotations_to_nms_format(self, annotations):
        """Convert existing annotations to NMS format using their built-in method."""
        nms_detections = []
        
        for annotation in annotations:
            try:
                # Use the annotation's built-in NMS conversion method
                nms_detection = annotation.to_nms_detection()
                # Override confidence to 0.90 to ensure existing annotations
                # always win NMS against new predictions (which are <= 0.90).
                nms_detection['confidence'] = 0.90
                nms_detections.append(nms_detection)
            except Exception as e:
                print(f"Warning: Failed to convert annotation {annotation.id} to NMS format: {e}")
        
        return nms_detections

    def _apply_nms_to_combined_detections(self, combined_detections):
        """Apply NMS to combined detections and annotations using TorchNMS."""
        if not combined_detections:
            return []
        
        import torch
        
        # Convert to tensors for NMS
        bboxes = torch.tensor([det['bbox'] for det in combined_detections], dtype=torch.float32)
        scores = torch.tensor([det['confidence'] for det in combined_detections], dtype=torch.float32)
        
        # Apply NMS using TorchNMS
        try:
            keep_indices = TorchNMS.fast_nms(bboxes, scores, self.iou_thresh)
            return keep_indices.tolist()
        except Exception as e:
            print(f"Warning: NMS failed, returning all indices: {e}")
            return list(range(len(combined_detections)))
        
    def get_mapped_short_label(self, cls_name, conf):
        """
        Get the short label for a detection result based on confidence and class mapping.

        :param cls_name: Class name
        :param conf: Confidence score
        :return: Short label as a string
        """
        if conf <= self.uncertainty_thresh:
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
            return predictions

        for idx, conf in zip(top5, top5conf):
            class_name = class_names[idx]
            label = self.label_window.get_label_by_short_code(class_name)
            if label:
                predictions[label] = float(conf)

        return image_path, top1cls, top1conf, predictions

    def process_classification_results(self, results_generator, annotations):
        """
        Process the classification results from the results generator.
        """
        progress_bar = ProgressBar(self.annotation_window, title="Making Classification Predictions")
        progress_bar.show()
        progress_bar.start_progress(len(annotations))

        for result, annotation in zip(results_generator, annotations):
            if result:
                try:
                    # Extract results and pass them to the dedicated update/display method
                    _, cls_name, conf, predictions = self.extract_classification_result(result)
                    self._update_and_display_classification(annotation, cls_name, conf, predictions)
                except Exception as e:
                    print(f"Warning: Failed to process classification result for annotation {annotation.id}\n{e}")
            progress_bar.update_progress()

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
        # Update the machine confidence values with all predictions
        annotation.update_machine_confidence(predictions)

        # Determine the final label, setting it to 'Review' if confidence is too low
        final_label = self.label_window.get_label_by_short_code(cls_name)
        if conf < self.uncertainty_thresh:
            final_label = self.label_window.get_label_by_id('-1')
        
        # Update the annotation's label
        if final_label:
            annotation.update_label(final_label)

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
    
    def process_segmentation_results(self, results_list):
        """
        Process the segmentation results from the list of Results with global NMS
        across all results (e.g., from work areas) and vs. existing annotations.
        """
        self._process_results_with_global_nms(results_list, 'segmentation')

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

    def process_detection_results(self, results_list):
        """
        Process the detection results from the list of Results with global NMS
        across all results (e.g., from work areas) and vs. existing annotations.
        """
        self._process_results_with_global_nms(results_list, 'detection')
        
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
        if conf < self.uncertainty_thresh:
            review_label = self.label_window.get_label_by_id('-1')
            annotation.update_label(review_label)
            annotation.set_verified(False)
        
        return annotation
