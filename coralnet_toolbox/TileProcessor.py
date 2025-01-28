import cv2
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter

from torchvision.ops import nms
from ultralytics.engine.results import Results
from ultralytics.models.sam.amg import batched_mask_to_box
from ultralytics.utils import ops
from ultralytics.utils.ops import scale_masks

from patched_yolo_infer.elements.CropElement import CropElement
from patched_yolo_infer.nodes.CombineDetections import CombineDetections

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.utilities import open_image


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class TileProcessor:
    def __init__(self, main_window):
        self.main_window = main_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

        self.tile_params = {}
        self.tile_inference_params = {}
        
        self.element_crops = None
        self.combined_detections = None
        
        self.image_path = None

    def params_set(self):
        return self.tile_params and self.tile_inference_params

    def set_tile_params(self, tile_params):
        self.tile_params = tile_params

    def set_tile_inference_params(self, tile_inference_params):
        self.tile_inference_params = tile_inference_params

    def set_params(self, tile_params, tile_inference_params):
        self.set_tile_params(tile_params)
        self.set_tile_inference_params(tile_inference_params)
        
    def make_crops(self, model, image, segment=False):
        
        if isinstance(image, list) and all(isinstance(x, str) for x in image):
            self.image_path = image[0]
            image = open_image(self.image_path)
            
        # Initialize 
        self.element_crops = MakeCropsDetectThem(
            imgsz=640,
            image=image,
            model=model,
            segment=segment,
            show_crops=False,
            shape_x=self.tile_params['shape_x'],
            shape_y=self.tile_params['shape_y'],
            overlap_x=self.tile_params['overlap_x'],
            overlap_y=self.tile_params['overlap_y'],
            conf=self.main_window.get_uncertainty_thresh(),
            iou=self.main_window.get_iou_thresh(),
            batch_inference=False,
            show_processing_status=True,
        )
        # Create crops
        self.element_crops.make_crops()
        
        return self.element_crops.get_crops()
    
    def detect_them(self, results):
        # Detect objects in crops
        self.element_crops.detect_them(results)
        
        # Combine them
        self.combined_detections = CombineDetections(self.element_crops, **self.tile_inference_params)
        
        # Convert to Ultralytics Results
        results = self.to_ultralytics()
        
        return results
    
    def to_ultralytics(self):
        """Convert detection results to Ultralytics Results format
        
        Returns:
            Results: Ultralytics Results object with boxes (N,6), masks (N,H,W)
        """
        # Convert to boxes tensor (N,6) [x1,y1,x2,y2,conf,cls]
        if len(self.combined_detections.filtered_boxes):
            boxes = torch.zeros((len(self.combined_detections.filtered_boxes), 6))
            boxes[:, :4] = torch.tensor(self.combined_detections.filtered_boxes)
            boxes[:, 4] = torch.tensor(self.combined_detections.filtered_confidences)
            boxes[:, 5] = torch.tensor(self.combined_detections.filtered_classes_id)
        else:
            boxes = None

        # Convert to masks tensor (N,H,W)
        if hasattr(self.combined_detections, 'segment') and self.combined_detections.segment:
            if self.combined_detections.filtered_polygons:
                H, W = self.combined_detections.image.shape[:2]
                masks = []
                for polygon in self.combined_detections.filtered_polygons:
                    mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)
                    masks.append(mask)
                masks = torch.tensor(np.array(masks))
            elif self.combined_detections.filtered_masks:
                masks = torch.tensor(np.array(self.combined_detections.filtered_masks))
            else:
                masks = None
        else:
            masks = None

        return Results(
            orig_img=self.combined_detections.image,
            path=self.image_path,
            names=self.combined_detections.class_names,
            boxes=boxes,
            masks=masks
        )
        

class MakeCropsDetectThem:
    """
    Class implementing cropping and passing crops through a neural network
    for detection/segmentation.

    Args:
        image (np.ndarray): Input image BGR.
        imgsz (int): Size of the input image for inference YOLO.
        conf (float): Confidence threshold for detections YOLO.
        iou (float): IoU threshold for non-maximum suppression YOLOv8 of single crop.
        segment (bool): Whether to perform segmentation (YOLO-seg).
        shape_x (int): Size of the crop in the x-coordinate.
        shape_y (int): Size of the crop in the y-coordinate.
        overlap_x (int): Percentage of overlap along the x-axis.
        overlap_y (int): Percentage of overlap along the y-axis.
        show_crops (bool): Whether to visualize the cropping.
        show_processing_status (bool): Whether to show the processing status using tqdm.
        resize_initial_size (bool): Whether to resize the results to the original image size (ps: slow operation).
        model: Pre-initialized model object.
        memory_optimize (bool): Memory optimization option for segmentation (less accurate results)
        batch_inference (bool): Batch inference of image crops through a neural network instead of 
                    sequential passes of crops (ps: Faster inference, higher memory use)
        progress_callback (function): Optional custom callback function, (task: str, current: int, total: int)
        inference_extra_args (dict): Dictionary with extra ultralytics inference parameters

    Attributes:
        model: YOLOv8 model loaded from the specified path.
        image (np.ndarray): Input image BGR.
        imgsz (int): Size of the input image for inference.
        conf (float): Confidence threshold for detections.
        iou (float): IoU threshold for non-maximum suppression.
        segment (bool): Whether to perform segmentation (YOLO-seg).
        shape_x (int): Size of the crop in the x-coordinate.
        shape_y (int): Size of the crop in the y-coordinate.
        overlap_x (int): Percentage of overlap along the x-axis.
        overlap_y (int): Percentage of overlap along the y-axis.
        crops (list): List to store the CropElement objects.
        show_crops (bool): Whether to visualize the cropping.
        show_processing_status (bool): Whether to show the processing status using tqdm.
        resize_initial_size (bool): Whether to resize the results to the original  
                                    image size (ps: slow operation).
        class_names_dict (dict): Dictionary containing class names of the YOLO model.
        memory_optimize (bool): Memory optimization option for segmentation (less accurate results)
        batch_inference (bool): Batch inference of image crops through a neural network instead of 
                                    sequential passes of crops (ps: Faster inference, higher memory use)
        progress_callback (function): Optional custom callback function, (task: str, current: int, total: int)
        inference_extra_args (dict): Dictionary with extra ultralytics inference parameters
    """

    def __init__(
        self,
        image: np.ndarray,
        imgsz=640,
        conf=0.25,
        iou=0.7,
        segment=False,
        shape_x=700,
        shape_y=600,
        overlap_x=25,
        overlap_y=25,
        show_crops=False,
        show_processing_status=True,
        resize_initial_size=True,
        model=None,
        memory_optimize=True,
        inference_extra_args=None,
        batch_inference=False,
        progress_callback=None,
    ) -> None:

        # Add show_process_status parameter and initialize progress bars dict
        self.show_process_status = show_processing_status
        self._progress_bars = {}

        # Set up the progress callback based on parameters
        if progress_callback is not None and show_processing_status:
            self.progress_callback = progress_callback
        elif show_processing_status:
            self.progress_callback = self._tqdm_callback
        else:
            self.progress_callback = None

        self.model = model
        
        # Input image
        self.image = image
        # Size of the input image for inference
        self.imgsz = imgsz  
        # Confidence threshold for detections
        self.conf = conf
        # IoU threshold for non-maximum suppression
        self.iou = iou
        # Classes to detect
        self.classes_list = None
        # Whether to perform segmentation
        self.segment = segment
        # Size of the crop in the x-coordinate
        self.shape_x = shape_x
        # Size of the crop in the y-coordinate  
        self.shape_y = shape_y
        # Percentage of overlap along the x-axis
        self.overlap_x = overlap_x
        # Percentage of overlap along the y-axis
        self.overlap_y = overlap_y
        # Whether to visualize the cropping
        self.show_crops = show_crops
        # slow operation !
        self.resize_initial_size = resize_initial_size
        # memory opimization option for segmentation
        self.memory_optimize = memory_optimize
        # dict with human-readable class names
        self.class_names_dict = self.model.names
        # dict with extra ultralytics inference parameters
        self.inference_extra_args = inference_extra_args
        # batch inference of image crops through a neural network
        self.batch_inference = batch_inference

        self.crops = None

    def make_crops(self):
        """Preprocessing of the image. Generating crops with overlapping."""
        cross_koef_x = 1 - (self.overlap_x / 100)
        cross_koef_y = 1 - (self.overlap_y / 100)

        data_all_crops = []

        y_steps = int((self.image.shape[0] - self.shape_y) / (self.shape_y * cross_koef_y)) + 1
        x_steps = int((self.image.shape[1] - self.shape_x) / (self.shape_x * cross_koef_x)) + 1

        y_new = round((y_steps - 1) * (self.shape_y * cross_koef_y) + self.shape_y)
        x_new = round((x_steps - 1) * (self.shape_x * cross_koef_x) + self.shape_x)
        
        image_initial = self.image.copy()
        image_full = cv2.resize(self.image, (x_new, y_new))
        batch_of_crops = []

        count = 0
        total_steps = y_steps * x_steps  # Total number of crops
        for i in range(y_steps):
            for j in range(x_steps):
                x_start = int(self.shape_x * j * cross_koef_x)
                y_start = int(self.shape_y * i * cross_koef_y)

                # Check for residuals
                if x_start + self.shape_x > image_full.shape[1]:
                    print('Error in generating crops along the x-axis')
                    continue
                if y_start + self.shape_y > image_full.shape[0]:
                    print('Error in generating crops along the y-axis')
                    continue

                im_temp = image_full[y_start:y_start + self.shape_y, x_start:x_start + self.shape_x]

                # Call the progress callback function if provided
                if self.progress_callback is not None:
                    self.progress_callback("Getting crops", count, total_steps)

                data_all_crops.append(CropElement(
                    source_image=image_initial,
                    source_image_resized=image_full,
                    crop=im_temp,
                    number_of_crop=count,
                    x_start=x_start,
                    y_start=y_start,
                ))
                if self.batch_inference:
                    batch_of_crops.append(im_temp)

        if self.batch_inference:
            self.crops = data_all_crops, batch_of_crops
        else:
            self.crops = data_all_crops

    def get_crops(self):
        """Get list of image arrays from all crops.

        Returns:
            List[np.ndarray]: List of image arrays from each crop
        """
        if self.crops is None:
            return []

        if self.batch_inference:
            crops, _ = self.crops
            return [crop.crop for crop in crops]

        return [crop.crop for crop in self.crops]

    def get_crops_batch(self):
        """Get batch array of all crops for batch inference.

        Returns:
            List[np.ndarray]: Batch array of all crops

        Raises:
            RuntimeError: If batch_inference=False
        """
        if not self.batch_inference:
            raise RuntimeError(
                "Batch crops only available when batch_inference=True")

        if self.crops is None:
            return []

        _, batch = self.crops
        return batch

    def detect_them(self, predictions=None):
        """Method to detect objects in each crop."""
        if self.batch_inference:
            self._detect_objects_batch(predictions)
        else:
            self._detect_objects()
            
    def _detect_objects(self):
        """
        Method to detect objects in each crop.

        This method iterates through each crop, performs inference using the YOLO model,
        calculates real values, and optionally resizes the results.

        Returns:
            None
        """
        total_crops = len(self.crops)  # Total number of crops
        for index, crop in enumerate(self.crops):
            crop.calculate_inference(
                self.model,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                segment=self.segment,
                classes_list=self.classes_list,
                memory_optimize=self.memory_optimize,
                extra_args=self.inference_extra_args
            )
            crop.calculate_real_values()
            if self.resize_initial_size:
                crop.resize_results()

            # Call the progress callback function if provided
            if self.progress_callback is not None:
                self.progress_callback(
                    "Detecting objects", (index + 1), total_crops)

    def _detect_objects_batch(self, predictions=None):
        """
        Method to detect objects in batch of image crops.

        Args:
            predictions: Optional pre-computed model predictions. If None, model.predict will be called.
        """
        crops, batch = self.crops
        self.crops = crops

        if self.progress_callback is not None:
            self.progress_callback("Detecting objects in batch", 0, 1)

        self._calculate_batch_inference(
            batch,
            self.crops,
            self.model,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            segment=self.segment,
            classes_list=self.classes_list,
            memory_optimize=self.memory_optimize,
            extra_args=self.inference_extra_args,
            predictions=predictions
        )
        for crop in self.crops:
            crop.calculate_real_values()
            if self.resize_initial_size:
                crop.resize_results()

        # Call the progress callback function if provided
        if self.progress_callback is not None:
            self.progress_callback("Detecting objects in batch", 1, 1)

    def _calculate_batch_inference(
        self,
        batch,
        crops,
        model,
        imgsz=640,
        conf=0.35,
        iou=0.7,
        segment=False,
        classes_list=None,
        memory_optimize=False,
        extra_args=None,
        predictions=None,
    ):
        """
        Method to calculate batch inference for a list of crops.

        Args:
            batch: List of image crops to pass through the neural network for inference.
            crops: List of CropElement objects corresponding to the image crops.
            model: YOLOv8 model loaded from the specified path.
            imgsz: Size of the input image for inference.
            conf: Confidence threshold for detections.
            iou: IoU threshold for non-maximum suppression.
            segment: Whether to perform segmentation.
            classes_list: List of classes to filter detections.
            memory_optimize: Memory optimization option for segmentation.
            extra_args: Dictionary with extra ultralytics inference parameters.
            predictions: Optional pre-computed model predictions. If None, model.predict will be called.
        """
        extra_args = {} if extra_args is None else extra_args

        # Use provided predictions or run model.predict
        if predictions is None:
            predictions = model.predict(
                batch,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                classes=classes_list,
                verbose=False,
                **extra_args
            )

        for pred, crop in zip(predictions, crops):

            # Get the bounding boxes and convert them to a list of lists
            crop.detected_xyxy = pred.boxes.xyxy.cpu().int().tolist()

            # Get the classes and convert them to a list
            crop.detected_cls = pred.boxes.cls.cpu().int().tolist()

            # Get the mask confidence scores
            crop.detected_conf = pred.boxes.conf.cpu().numpy()

            if segment and len(crop.detected_cls) != 0:
                if memory_optimize:
                    # Get the polygons
                    crop.polygons = [mask.astype(np.uint16) for mask in pred.masks.xy]
                else:
                    # Get the masks
                    crop.detected_masks = pred.masks.data.cpu().numpy()
                    
    def __str__(self):
        # Print info about patches amount
        return (
            f"{len(self.crops)} patches of size {self.crops[0].crop.shape} "
            f"were created from an image sized {self.image.shape}"
        )

    def patches_info(self):
        print(self)
        output = "\nDetailed information about the detections for each patch:"
        for i, patch in enumerate(self.crops):
            if len(patch.detected_cls) > 0:
                detected_cls_names_list = [
                    self.class_names_dict[value] for value in patch.detected_cls
                ]  # make str list

                # Count the occurrences of each class in the current patch
                class_counts = Counter(detected_cls_names_list)

                # Format the class counts into a readable string
                class_info = ", ".join(
                    [f"{count} {cls}" for cls, count in class_counts.items()])

                # Append the formatted string to the patch_info list
                output += f"\nOn patch № {i}, {class_info} were detected"
            else:
                # Append the formatted string to the patch_info list
                output += f"\nOn patch № {i}, nothing was detected"
        print(output)
                    
    def _tqdm_callback(self, task, current, total):
        """Internal callback function that uses tqdm for progress tracking

        Args:
            task (str): The name of the task being tracked
            current (int): The current progress value
            total (int): The total number of steps in the task

        """
        if task not in self._progress_bars:
            self._progress_bars[task] = tqdm(
                total=total,
                desc=task,
                unit='items'
            )

        # Update progress
        self._progress_bars[task].n = current
        self._progress_bars[task].refresh()

        # Close and cleanup if task is complete
        if current >= total:
            self._progress_bars[task].close()
            del self._progress_bars[task]

    def __del__(self):
        """Cleanup method to ensure all progress bars are closed"""
        for pbar in self._progress_bars.values():
            pbar.close()
        self._progress_bars.clear()

