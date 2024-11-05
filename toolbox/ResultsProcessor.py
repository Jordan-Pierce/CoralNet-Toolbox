from PyQt5.QtCore import QPointF

from toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

from toolbox.QtProgressBar import ProgressBar

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ResultsProcessor:
    def __init__(self, main_window, class_mapping, use_sam):
        self.main_window = main_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

        self.class_mapping = class_mapping
        self.use_sam = use_sam

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

    def process_single_classification_result(self, results, annotation):
        """
        Process a single classification result.
        """
        # Extract relevant information from the classification result
        image_path, cls_name, conf, predictions = self.extract_classification_result(results)
        # Store and display the annotation
        self.store_and_display_annotation(annotation, image_path, cls_name, conf, predictions)

    def process_classification_results(self, results_generator, annotations):
        """
        Process the classification results from the results generator.
        """
        progress_bar = ProgressBar(self.annotation_window, title="Making Classification Predictions")
        progress_bar.show()
        progress_bar.start_progress(len(annotations))

        for results, annotation in zip(results_generator, annotations):
            self.process_single_classification_result(results, annotation)
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

    def process_detection_results(self, results_generator):
        """
        Process the detection results from the results generator.
        """
        progress_bar = ProgressBar(self.annotation_window, title="Making Detection Predictions")
        progress_bar.show()

        for results in results_generator:
            for result in results:
                self.process_single_detection_result(result)
                progress_bar.update_progress()

        progress_bar.stop_progress()
        progress_bar.close()

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
        # Store and display the annotation
        self.store_and_display_annotation(annotation, image_path, cls_name, conf)

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
        # Store and display the annotation
        self.store_and_display_annotation(annotation, image_path, cls_name, conf)

    def process_segmentation_results(self, results_generator):
        """
        Process the segmentation results from the results generator.
        """
        progress_bar = ProgressBar(self.annotation_window, title=f"Making Segmentation Predictions")
        progress_bar.show()

        for results in results_generator:
            for result in results:
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
        if conf <= self.main_window.get_uncertainty_thresh():
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
        top_left = QPointF(x_min, y_min)
        bottom_right = QPointF(x_max, y_max)
        return RectangleAnnotation(top_left,
                                   bottom_right,
                                   label.short_label_code,
                                   label.long_label_code,
                                   label.color,
                                   image_path,
                                   label.id,
                                   self.main_window.get_transparency_value(),
                                   show_msg=True)

    def create_polygon_annotation(self, points, label, image_path):
        """
        Create a polygon annotation for the given points and label.

        :param points: List of polygon points
        :param label: Label object
        :param image_path: Path to the image
        :return: PolygonAnnotation object
        """
        points = [QPointF(x, y) for x, y in points]
        return PolygonAnnotation(points,
                                 label.short_label_code,
                                 label.long_label_code,
                                 label.color,
                                 image_path,
                                 label.id,
                                 self.main_window.get_transparency_value(),
                                 show_msg=True)

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
        if conf < self.main_window.get_uncertainty_thresh():
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