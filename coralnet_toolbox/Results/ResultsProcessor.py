from PyQt5.QtCore import QPointF

from torchvision.ops import nms

from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

from coralnet_toolbox.utilities import clean_polygon

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
            results = results[nms(results.boxes.xyxy, results.boxes.conf, self.iou_thresh)]
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
            indices = nms(results.boxes.xyxy, results.boxes.conf, self.iou_thresh).tolist()
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

        # Get the indexes of results that pass all filters
        indices = set()
        indices.update(indices_uncertainty)
        indices.update(indices_iou)
        indices.update(indices_area)

        return list(indices)

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

    def process_detection_results(self, results_list):
        """
        Process the detection results from the list of Results.
        """
        progress_bar = ProgressBar(self.annotation_window, title="Making Detection Predictions")
        progress_bar.show()

        for results in results_list:
            # Find the indices of results that pass all filters
            indices = self.indices_pass_filters(results)
            # Start the progress bar
            progress_bar.start_progress(len(results))
            # Loop through the results
            for idx, result in enumerate(results):
                # Skip the result if it doesn't pass the filters
                if idx in indices:
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

        return image_path, cls, cls_name, conf

    def process_single_segmentation_result(self, result, xy):
        """
        Process a single segmentation result.
        """
        # Convert to list of tuples for consistency
        points = [(float(x), float(y)) for x, y in xy]
        # Filter out small disconnected polygons, keeping only the largest one
        points = clean_polygon(points)

        # Get image path, class, class name, confidence, and polygon points
        image_path, cls, cls_name, conf = self.extract_segmentation_result(result)
        # Get the short label given the class name and confidence
        short_label = self.get_mapped_short_label(cls_name, conf)
        # Get the label object given the short label
        label = self.label_window.get_label_by_short_code(short_label)
        # Create the polygon annotation
        annotation = self.create_polygon_annotation(points, label, image_path)

        if annotation:
            # Store and display the annotation
            self.store_and_display_annotation(annotation, image_path, cls_name, conf)

    def process_segmentation_results(self, results_list):
        """
        Process the segmentation results from the list of Results.
        """
        progress_bar = ProgressBar(self.annotation_window, title="Making Segmentation Predictions")
        progress_bar.show()

        for results in results_list:
            # Find the indices of results that pass all filters
            indices = self.indices_pass_filters(results)
            # Start the progress bar
            progress_bar.start_progress(len(results))
            # Loop through the results
            for idx, result in enumerate(results):
                # Skip the result if it doesn't pass the filters
                if idx in indices:
                    try:
                        # Extract the segmentation results
                        xy = results.masks.xy[idx]
                        # Process a single segmentation result
                        self.process_single_segmentation_result(result, xy)
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
        self.image_window.update_image_annotations(image_path)
        # Unselect all annotations
        self.annotation_window.unselect_annotations()
