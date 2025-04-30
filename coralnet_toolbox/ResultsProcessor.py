from PyQt5.QtCore import QPointF

import numpy as np

import torch
from torchvision.ops import nms

import supervision as sv

from ultralytics.engine.results import Results
from ultralytics.models.sam.amg import batched_mask_to_box
from ultralytics.utils import ops
from ultralytics.utils.ops import scale_masks

from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
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
            if isinstance(results, list):
                results = results[0]
            results = results[results.boxes.conf > self.uncertainty_thresh]
        except Exception as e:
            print(f"Warning: Failed to filter results by uncertainty\n{e}")
            
        return results
    
    def filter_by_iou(self, results):
        """Filter the results based on the IoU threshold."""
        try:
            if isinstance(results, list):
                results = results[0]
            results = results[nms(results.boxes.xyxy, results.boxes.conf, self.iou_thresh)]
        except Exception as e:  
            print(f"Warning: Failed to filter results by IoU\n{e}")
            
        return results
    
    def filter_by_area(self, results):
        """
        Filter the results based on the area threshold.
        """
        try:
            if isinstance(results, list):
                results = results[0]
            x_norm, y_norm, w_norm, h_norm = results.boxes.xywhn.T
            area_norm = w_norm * h_norm
            results = results[(area_norm >= self.min_area_thresh) & (area_norm <= self.max_area_thresh)]
        except Exception as e:
            print(f"Warning: Failed to filter results by area\n{e}")
            
        return results
    
    def apply_filters(self, results):
        """Check if the results passed all filters."""
        results = self.filter_by_uncertainty(results)
        results = self.filter_by_iou(results)
        results = self.filter_by_area(results)
        return results

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

    def process_single_classification_result(self, result, annotation):
        """
        Process a single classification result.
        """
        # Extract relevant information from the classification result
        image_path, cls_name, conf, predictions = self.extract_classification_result(result)
        # Store and display the annotation
        self.store_and_display_annotation(annotation, image_path, cls_name, conf, predictions)

    def process_classification_results(self, results_generator, annotations):
        """
        Process the classification results from the results generator.
        """
        progress_bar = ProgressBar(self.annotation_window, title="Making Classification Predictions")
        progress_bar.show()
        progress_bar.start_progress(len(annotations))

        for result, annotation in zip(results_generator, annotations):
            if result:
                self.process_single_classification_result(result, annotation)
            progress_bar.update_progress()

        progress_bar.stop_progress()
        progress_bar.close()

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
        x_min, y_min, x_max, y_max = map(float, result.boxes.xyxy.cpu().numpy()[0])

        return image_path, cls, cls_name, conf, x_min, y_min, x_max, y_max

    def process_single_detection_result(self, result):
        """
        Process a single detection result.
        """       
        # Get image path, class, class name, confidence, and bounding box coordinates
        image_path, cls, cls_name, conf, x_min, y_min, x_max, y_max = self.extract_detection_result(result)
        # Get the short label given the class name and confidence
        short_label = self.get_mapped_short_label(cls_name, conf)
        # Get the label object given the short label
        label = self.label_window.get_label_by_short_code(short_label)
        # Create the rectangle annotation
        annotation = self.create_rectangle_annotation(x_min, y_min, x_max, y_max, label, image_path)
        
        if annotation:
            # Store and display the annotation
            self.store_and_display_annotation(annotation, image_path, cls_name, conf)

    def process_detection_results(self, results_generator):
        """
        Process the detection results from the results generator.
        """
        progress_bar = ProgressBar(self.annotation_window, title="Making Detection Predictions")
        progress_bar.show()

        for results in results_generator:
            # Apply filtering to the results
            results = self.apply_filters(results)
            # Start the progress bar
            progress_bar.start_progress(len(results))
            # Loop through the results
            for result in results:
                try:
                    if result.boxes:
                        # Process a single detection result
                        self.process_single_detection_result(result)
                except Exception as e:
                    print(f"Warning: Failed to process detection result\n{e}")
                    
                # Update the progress bar
                progress_bar.update_progress()

        progress_bar.stop_progress()
        progress_bar.close()

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
        
        # Get the mask and convert to polygon points
        mask = result.masks.cpu().data.numpy().squeeze().astype(bool)
        
        # Convert to biggest polygon
        polygons = sv.detection.utils.mask_to_polygons(mask)
        
        if len(polygons) == 1:
            points = polygons[0]
        else:
            # Grab the index of the largest polygon
            points = max(polygons, key=lambda x: len(x))

        return image_path, cls, cls_name, conf, points

    def process_single_segmentation_result(self, result):
        """
        Process a single segmentation result.
        """
        # Get image path, class, class name, confidence, and polygon points
        image_path, cls, cls_name, conf, points = self.extract_segmentation_result(result)
        # Get the short label given the class name and confidence
        short_label = self.get_mapped_short_label(cls_name, conf)
        # Get the label object given the short label
        label = self.label_window.get_label_by_short_code(short_label)
        # Create the polygon annotation
        annotation = self.create_polygon_annotation(points, label, image_path)
        
        if annotation:
            # Store and display the annotation
            self.store_and_display_annotation(annotation, image_path, cls_name, conf)

    def process_segmentation_results(self, results_generator):
        """
        Process the segmentation results from the results generator.
        """
        progress_bar = ProgressBar(self.annotation_window, title="Making Segmentation Predictions")
        progress_bar.show()

        for results in results_generator:
            # Apply filtering to the results
            results = self.apply_filters(results)
            # Start the progress bar
            progress_bar.start_progress(len(results))
            # Loop through the results
            for result in results:
                try:
                    if result.boxes:
                        # Process a single segmentation result
                        self.process_single_segmentation_result(result)
                except Exception as e:
                    print(f"Warning: Failed to process segmentation result\n{e}")
                    
                # Update the progress bar
                progress_bar.update_progress()

        progress_bar.stop_progress()
        progress_bar.close()

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

    def create_rectangle_annotation(self, x_min, y_min, x_max, y_max, label, image_path):
        """
        Create a rectangle annotation for the given bounding box coordinates and label.

        :param x_min: Minimum x-coordinate
        :param y_min: Minimum y-coordinate
        :param x_max: Maximum x-coordinate
        :param y_max: Maximum y-coordinate
        :param label: Label object
        :param image_path: Path to the image
        :return: RectangleAnnotation object
        """
        try:
            top_left = QPointF(x_min, y_min)
            bottom_right = QPointF(x_max, y_max)
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

    def store_and_display_annotation(self, annotation, image_path, cls_name, conf, predictions=None):
        """
        Store and display the annotation in the annotation window and image window.

        :param annotation: Annotation object
        :param image_path: Path to the image
        :param cls_name: Class name
        :param conf: Confidence score
        :param predictions: Dictionary containing class predictions
        """
        # Add the annotation to the annotation window
        self.annotation_window.add_annotation_to_dict(annotation)

        # Connect signals
        annotation.selected.connect(self.annotation_window.select_annotation)
        annotation.annotationDeleted.connect(self.annotation_window.delete_annotation)
        annotation.annotationUpdated.connect(self.main_window.confidence_window.display_cropped_image)

        if not predictions:
            predictions = {self.label_window.get_label_by_short_code(cls_name): conf}

        # Update the confidence values for predictions
        annotation.update_machine_confidence(predictions)

        # If the confidence is below the threshold, set the label to review
        if conf < self.uncertainty_thresh:
            review_label = self.label_window.get_label_by_id('-1')
            annotation.update_label(review_label, set_review=True)

        # If the image is currently displayed in the annotation window, update the graphics item
        if image_path == self.annotation_window.current_image_path:
            annotation.create_graphics_item(self.annotation_window.scene)
            annotation.create_cropped_image(self.annotation_window.rasterio_image)
            self.main_window.confidence_window.display_cropped_image(annotation)

        # Update the image in the image window
        self.main_window.image_window.update_image_annotations(image_path)
        # Unselect all annotations
        self.annotation_window.unselect_annotations()
                
    def from_sam(self, masks, scores, image, image_path):
        """
        Converts SAM results to Ultralytics Results Object.

        Args:
            masks (torch.Tensor): Predicted masks with shape (N, 1, H, W).
            scores (torch.Tensor): Confidence scores for each mask with shape (N, 1).
            image (np.ndarray): The original, unprocessed image.
            image_path (str): Path to the image file.

        Returns:
            results (Results): Ultralytics Results object.
        """
        # Ensure the original image is in the correct format
        if not isinstance(image, np.ndarray):
            image = image.cpu().numpy()

        # Ensure masks have the correct shape (N, 1, H, W)
        if masks.ndim != 4 or masks.shape[1] != 1:
            raise ValueError(f"Expected masks to have shape (N, 1, H, W), but got {masks.shape}")

        # Scale masks to the original image size and remove extra dimensions
        scaled_masks = ops.scale_masks(masks.float(), image.shape[:2], padding=False)
        scaled_masks = scaled_masks > 0.5  # Apply threshold to masks

        # Ensure scaled_masks is 3D (N, H, W)
        if scaled_masks.ndim == 4:
            scaled_masks = scaled_masks.squeeze(1)

        # Generate bounding boxes from masks using batched_mask_to_box
        scaled_boxes = batched_mask_to_box(scaled_masks)

        # Ensure scores has shape (N,) by removing extra dimensions
        scores = scores.squeeze().cpu()
        if scores.ndim == 0:  # If only one score, make it a 1D tensor
            scores = scores.unsqueeze(0)

        # Generate class labels
        cls = torch.arange(len(masks), dtype=torch.int32).cpu()

        # Ensure all tensors are 2D before concatenating
        scaled_boxes = scaled_boxes.cpu()
        if scaled_boxes.ndim == 1:
            scaled_boxes = scaled_boxes.unsqueeze(0)
        scores = scores.view(-1, 1)  # Reshape to (N, 1)
        cls = cls.view(-1, 1)  # Reshape to (N, 1)

        # Combine bounding boxes, scores, and class labels
        scaled_boxes = torch.cat([scaled_boxes, scores, cls], dim=1)

        # Create names dictionary (placeholder for consistency)
        names = dict(enumerate(str(i) for i in range(len(masks))))

        # Create Results object
        results = Results(image, 
                          path=image_path, 
                          names=names, 
                          masks=scaled_masks, 
                          boxes=scaled_boxes)
        
        return results
    
    def from_supervision(self, detections, images, image_paths=None, names=None):
        """
        Convert Supervision Detections to Ultralytics Results format.
        Handles both single detection/image and lists of detections/images.

        Args:
            detections: Single Detections object or list of Detections objects
            images: Single image array or list of image arrays
            image_paths: Single image path or list of image paths (optional)
            names: Dictionary mapping class ids to class names (optional)

        Returns:
            generator: Yields Ultralytics Results objects
        """
        # Convert single inputs to lists
        if not isinstance(detections, list):
            detections = [detections]
        if not isinstance(images, list):
            images = [images]
        if image_paths and not isinstance(image_paths, list):
            image_paths = [image_paths]
        
        # Ensure image_paths exists
        if not image_paths:
            image_paths = [None] * len(images)

        for detection, image, path in zip(detections, images, image_paths):
            # Ensure image is numpy array
            if torch.is_tensor(image):
                image = image.cpu().numpy()

            # Create default names if not provided
            if names is None:
                names = {i: str(i) for i in range(len(detection))} if len(detection) > 0 else {}

            if len(detection) == 0:
                yield Results(orig_img=image, path=path, names=names)
                continue

            # Handle masks if present
            if hasattr(detection, 'mask') and detection.mask is not None:
                masks = torch.as_tensor(detection.mask, dtype=torch.float32)
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)
                scaled_masks = scale_masks(masks, image.shape[:2], padding=False)
                scaled_masks = scaled_masks > 0.5
                if scaled_masks.ndim == 4:
                    scaled_masks = scaled_masks.squeeze(1)
            else:
                scaled_masks = None

            # Convert boxes and scores
            scaled_boxes = torch.as_tensor(detection.xyxy, dtype=torch.float32)
            scores = torch.as_tensor(detection.confidence, dtype=torch.float32).view(-1, 1)
            cls = torch.as_tensor(detection.class_id, dtype=torch.int32).view(-1, 1)

            # Combine boxes, scores, and class IDs
            if scaled_boxes.ndim == 1:
                scaled_boxes = scaled_boxes.unsqueeze(0)
            scaled_boxes = torch.cat([scaled_boxes, scores, cls], dim=1)

            # Create and yield Results object
            yield Results(image,
                          path=path,
                          names=names,
                          boxes=scaled_boxes, 
                          masks=scaled_masks)
            
    def combine_results(self, results: list):
        """
        Combine multiple Results objects for the same image into a single Results object.
        
        This function combines detections from multiple Results objects that correspond to the same image,
        merging their boxes, masks, probs, keypoints, and/or OBB (Oriented Bounding Box) objects.
        
        Args:
            results (list): List of Results objects to combine. All Results should be for the same image.
            
        Returns:
            Results: A single combined Results object containing combined detections.
            
        Note:
            - If the input Results objects contain different types of data (e.g., some have boxes, some have masks),
              the combined Results object will contain all available data.
            - For classification results (probs), only the highest confidence classification is kept.
        """
        if not results:
            return None
        
        if any(r is None for r in results):
            raise ValueError("Cannot combine None results. Please provide valid Results objects.")
        
        if len(results) == 1:
            return results[0]
            
        # Ensure all results are for the same image
        first_path = results[0].path
        if not all(r.path == first_path for r in results):
            print("Warning: Attempting to combine Results from different images. Using the first image.")
        
        # Get the first result's data for base attributes
        combined_result = results[0].new()
        
        # combine boxes if any exist
        all_boxes = [r.boxes.data for r in results if r.boxes is not None]
        if all_boxes:
            # Concatenate all boxes
            if isinstance(all_boxes[0], torch.Tensor):
                combined_boxes = torch.cat(all_boxes, dim=0)
            else:
                combined_boxes = np.concatenate(all_boxes, axis=0)
            combined_result.update(boxes=combined_boxes)
        
        # combine masks if any exist
        all_masks = [r.masks.data for r in results if r.masks is not None]
        if all_masks:
            # Concatenate all masks
            if isinstance(all_masks[0], torch.Tensor):
                combined_masks = torch.cat(all_masks, dim=0)
            else:
                combined_masks = np.concatenate(all_masks, axis=0)
            combined_result.update(masks=combined_masks)
        
        # For classification results (probs), keep the one with highest confidence
        all_probs = [(r.probs, r.probs.top1conf) for r in results if r.probs is not None]
        if all_probs:
            # Get the probs object with highest top1 confidence
            best_probs = max(all_probs, key=lambda x: x[1])[0]
            combined_result.update(probs=best_probs.data)
        
        # combine keypoints if any exist
        all_keypoints = [r.keypoints.data for r in results if r.keypoints is not None]
        if all_keypoints:
            # Concatenate all keypoints
            if isinstance(all_keypoints[0], torch.Tensor):
                combined_keypoints = torch.cat(all_keypoints, dim=0)
            else:
                combined_keypoints = np.concatenate(all_keypoints, axis=0)
            combined_result.update(keypoints=combined_keypoints)
        
        # combine OBB (Oriented Bounding Boxes) if any exist
        all_obbs = [r.obb.data for r in results if r.obb is not None]
        if all_obbs:
            # Concatenate all OBBs
            if isinstance(all_obbs[0], torch.Tensor):
                combined_obbs = torch.cat(all_obbs, dim=0)
            else:
                combined_obbs = np.concatenate(all_obbs, axis=0)
            combined_result.update(obb=combined_obbs)
        
        # combine names dictionaries from all results
        combined_names = {}
        for r in results:
            if r.names:
                combined_names.update(r.names)
        combined_result.names = combined_names
        
        return combined_result
