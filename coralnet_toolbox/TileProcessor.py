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
            image=image,
            model=model,
            segment=segment,
            show_crops=False,
            shape_x=self.tile_params['shape_x'],
            shape_y=self.tile_params['shape_y'],
            overlap_x=self.tile_params['overlap_x'],
            overlap_y=self.tile_params['overlap_y'],
            conf=0.5,
            iou=0.7,
            batch_inference=True,
            show_processing_status=True,
        )
        # Create crops
        self.element_crops.make_crops()
        
        return self.element_crops.get_crops()
    
    def detect_them(self, results):
        # Detect objects in crops
        self.element_crops.detect_them(results)
        
        # Combine them
        self.combined_detections = CombineDetections(
            self.element_crops, 
            nms_threshold=0.05,
        )
        
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
        

# ----------------------------------------------------------------------------------------------------------------------
# Child Classes
# ----------------------------------------------------------------------------------------------------------------------


class MakeCropsDetectThem:
    def __init__(self, image, **kwargs):
        
        # Get parameters from kwargs with defaults
        self.show_process_status = kwargs.get('show_processing_status', False)
        self._progress_bars = {}

        progress_callback = kwargs.get('progress_callback', None)
        # Set up the progress callback based on parameters
        if progress_callback is not None and self.show_process_status:
            self.progress_callback = progress_callback
        elif self.show_process_status:
            self.progress_callback = self._tqdm_callback
        else:
            self.progress_callback = None

        # Input image
        self.image = image
        # Model 
        self.model = kwargs.get('model')
        
        # Size of the input image for inference
        # Size of the input image for inference
        self.imgsz = kwargs.get('imgsz', 640)  
        # Confidence threshold for detections
        self.conf = kwargs.get('conf', 0.35)  
        # IoU threshold for non-maximum suppression
        self.iou = kwargs.get('iou', 0.7)  
        # Classes to detect
        self.classes_list = kwargs.get('classes_list')  
        # Whether to perform segmentation
        self.segment = kwargs.get('segment', False)  
        # Size of the crop in the x-coordinate
        self.shape_x = kwargs.get('shape_x', 640)  
        # Size of the crop in the y-coordinate
        self.shape_y = kwargs.get('shape_y', 640)  
        # Percentage of overlap along the x-axis
        self.overlap_x = kwargs.get('overlap_x', 0)  
        # Percentage of overlap along the y-axis
        self.overlap_y = kwargs.get('overlap_y', 0)  
        # Whether to visualize the cropping
        self.show_crops = kwargs.get('show_crops', False)  
        # slow operation!
        self.resize_initial_size = kwargs.get('resize_initial_size', False)
        
        # memory optimization option for segmentation
        self.memory_optimize = kwargs.get('memory_optimize', False)
        # dict with human-readable class names
        self.class_names_dict = self.model.names if self.model else {}
        # dict with extra ultralytics inference parameters
        self.inference_extra_args = kwargs.get('inference_extra_args', {})
        # batch inference of image crops through a neural network
        self.batch_inference = kwargs.get('batch_inference', False)

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

