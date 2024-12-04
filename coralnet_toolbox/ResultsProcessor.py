import numpy as np
import torch

from PyQt5.QtCore import QPointF

from torchvision.ops import nms
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
                 class_mapping, 
                 uncertainty_thresh=0.3, 
                 iou_thresh=0.2, 
                 min_area_thresh=0.00, 
                 max_area_thresh=0.40):
        self.main_window = main_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

        self.class_mapping = class_mapping
        
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
            for result in results:
                if result:
                    self.process_single_detection_result(result)
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
        points = result.masks.cpu().xy[0].astype(float)
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
            for result in results:
                if result:
                    self.process_single_segmentation_result(result)
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
        self.annotation_window.annotations_dict[annotation.id] = annotation

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
            annotation.update_label(review_label)

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
    
    def from_supervision(self, detections, image, image_path, names):
        """
        Convert Supervision Detections to Ultralytics Results format with proper mask handling.

        Args:
            detections (Detections): Supervision detection object
            image (np.ndarray): Original image array
            image_path (str, optional): Path to the image file
            names (dict, optional): Dictionary mapping class ids to class names

        Returns:
            results_generator (generator): A generator that yields Ultralytics Results.
        """
        # Ensure original image is numpy array
        if torch.is_tensor(image):
            image = image.cpu().numpy()

        # Create default names if not provided
        if names is None:
            names = {i: str(i) for i in range(len(detections))} if len(detections) > 0 else {}

        if len(detections) == 0:
            return Results(orig_img=image, path=image_path, names=names)

        # Handle masks if present
        if hasattr(detections, 'mask') and detections.mask is not None:
            # Convert masks to torch tensor if needed
            masks = torch.as_tensor(detections.mask, dtype=torch.float32)

            # Ensure masks have shape (N, 1, H, W)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            # Scale masks to match original image size
            scaled_masks = scale_masks(masks, image.shape[:2], padding=False)
            scaled_masks = scaled_masks > 0.5  # Apply threshold

            # Ensure scaled_masks is 3D (N, H, W)
            if scaled_masks.ndim == 4:
                scaled_masks = scaled_masks.squeeze(1)
        else:
            scaled_masks = None

        # Convert boxes and scores to torch tensors
        scaled_boxes = torch.as_tensor(detections.xyxy, dtype=torch.float32)
        scores = torch.as_tensor(detections.confidence, dtype=torch.float32).view(-1, 1)

        # Convert class IDs to torch tensor
        cls = torch.as_tensor(detections.class_id, dtype=torch.int32).view(-1, 1)

        # Combine boxes, scores, and class IDs
        if scaled_boxes.ndim == 1:
            scaled_boxes = scaled_boxes.unsqueeze(0)
        scaled_boxes = torch.cat([scaled_boxes, scores, cls], dim=1)

        # Create Results object
        results = Results(image,
                          path=image_path,
                          names=names,
                          boxes=scaled_boxes, 
                          masks=scaled_masks)

        yield results