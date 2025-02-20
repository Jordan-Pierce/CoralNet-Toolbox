import cv2
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
import traceback

from torchvision.ops import nms
from ultralytics.engine.results import Results
from ultralytics.models.sam.amg import batched_mask_to_box
from ultralytics.utils import ops
from ultralytics.utils.ops import scale_masks

from PyQt5.Qt import Qt
from PyQt5.QtWidgets import QApplication as QtApplication

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
        
        self.progress_bar = ProgressBar(self.annotation_window, title="Tile Inference")

    def params_set(self):
        return self.tile_params and self.tile_inference_params

    def set_tile_params(self, tile_params):
        self.tile_params = tile_params

    def set_tile_inference_params(self, tile_inference_params):
        self.tile_inference_params = tile_inference_params

    def set_params(self, tile_params, tile_inference_params):
        self.set_tile_params(tile_params)
        self.set_tile_inference_params(tile_inference_params)
        
    def custom_progress_callback(self, task, current, total):
        """Progress callback function that uses QtProgressBar to show progress.
        
        Args:
            task (str): Description of current task
            current (int): Current progress value 
            total (int): Total steps for task
        """
        # Update progress bar
        self.progress_bar.setWindowTitle(task)
        self.progress_bar.show()
        progress_percentage = int((current / total) * 100)
        self.progress_bar.set_value(progress_percentage)
        self.progress_bar.update_progress()
        
        if self.progress_bar.wasCanceled():
            raise Exception("Tiling process was canceled by the user.")
    
    def make_crops(self, model, image_path):
        """Make crops from image and return them"""
        # Make cursor busy
        QtApplication.setOverrideCursor(Qt.WaitCursor)
        
        # Initialize 
        crops = []
        
        try:
            # Open image
            self.image_path = image_path
            image = open_image(self.image_path)
                
            # Initialize 
            self.element_crops = MakeCropsDetectThem(
                imgsz=self.tile_params['imgsz'],
                image=image,
                model=model,
                conf=self.main_window.get_uncertainty_thresh(),
                iou=self.main_window.get_iou_thresh(),
                show_crops=self.tile_params['show_crops'],
                shape_x=self.tile_params['shape_x'],
                shape_y=self.tile_params['shape_y'],
                overlap_x=self.tile_params['overlap_x'],
                overlap_y=self.tile_params['overlap_y'],
                marings=self.tile_params['margins'],
                include_residuals=self.tile_params['include_residuals'],
                show_processing_status=self.tile_params['show_processing_status'],
                progress_callback=self.custom_progress_callback 
            )
            # Create crops
            self.element_crops.make_crops()
            crops = self.element_crops.get_crops()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            # Make cursor not busy
            QtApplication.restoreOverrideCursor()
            self.progress_bar.stop_progress()
            self.progress_bar.close()
            
        # Return the crops list (numpy arrays)
        return crops
    
    def detect_them(self, predictions, segment=False):
        """Detect objects in image crops"""
        # Make cursor busy
        QtApplication.setOverrideCursor(Qt.WaitCursor)
        
        # Initialize
        results = []
        
        try:
            # Set segment flag
            self.element_crops.segment = segment
            
            # Predictions made, detect objects in crops
            self.element_crops.detect_them(predictions)
            
            # Combine them
            self.combined_detections = CombineDetections(
                element_crops=self.element_crops, 
                nms_threshold=self.tile_inference_params['nms_threshold'],
                match_metric=self.tile_inference_params['match_metric'],
                class_agnostic_nms=self.tile_inference_params['class_agnostic_nms'],
                intelligent_sorter=self.tile_inference_params['intelligent_sorter'],
                sorter_bins=self.tile_inference_params['sorter_bins'],                                  
            )
            
            # Convert to Ultralytics Results
            results = self.to_ultralytics()
            
        except Exception as e:
            print(f"Error: {e}")
        finally:
            # Make cursor not busy
            QtApplication.restoreOverrideCursor()
            self.progress_bar.stop_progress()
            self.progress_bar.close()
        
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

        yield Results(
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
        margins (tuple): Tuple of margins (left, top, right, bottom) to be applied to the image.
        show_crops (bool): Whether to visualize the cropping.
        show_processing_status (bool): Whether to show the processing status using tqdm.
        resize_initial_size (bool): Whether to resize the results to the original image size (ps: slow operation).
        model: Pre-initialized model object.
        memory_optimize (bool): Memory optimization option for segmentation (less accurate results)
        progress_callback (function): Optional custom callback function, (task: str, current: int, total: int)
        include_residuals (bool): Whether to include residuals in the crops.
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
        margins (tuple): Tuple of margins (left, top, right, bottom) to be applied to the image.
        crops (list): List to store the CropElement objects.
        show_crops (bool): Whether to visualize the cropping.
        show_processing_status (bool): Whether to show the processing status using tqdm.
        resize_initial_size (bool): Whether to resize the results to the original  
                                    image size (ps: slow operation).
        class_names_dict (dict): Dictionary containing class names of the YOLO model.
        memory_optimize (bool): Memory optimization option for segmentation (less accurate results)
        progress_callback (function): Optional custom callback function, (task: str, current: int, total: int)
        include_residuals (bool): Whether to include residuals in the crops.
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
        marings=(0, 0, 0, 0),
        show_crops=False,
        show_processing_status=True,
        resize_initial_size=True,
        model=None,
        memory_optimize=True,
        inference_extra_args=None,
        progress_callback=None,
        include_residuals=True,
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
        # Tuple of margins (left, top, right, bottom) to be applied to the image
        self.margins = marings
        # Whether to visualize the cropping
        self.show_crops = show_crops
        # slow operation !
        self.resize_initial_size = resize_initial_size
        # memory necessary option for segmentation
        self.memory_optimize = memory_optimize
        # dict with human-readable class names
        self.class_names_dict = self.model.names
        # dict with extra ultralytics inference parameters
        self.inference_extra_args = inference_extra_args
        # batch inference of image crops through a neural network
        # whether to include residuals in the crops
        self.include_residuals = include_residuals

        self.crops = None
    
    def make_crops(self):
        """
        Generates crops with overlapping using the same logic as update_tile_graphics.
        If include_residuals is False, only includes crops that match the specified shape.
        Returns a list of CropElement objects.
        """        
        # Get image dimensions
        image_full_height, image_full_width = self.image.shape[:2]

        # Calculate grid boundaries with margins
        x_start = self.margins[0]  # left margin
        y_start = self.margins[1]  # top margin
        x_end = image_full_width - self.margins[2]  # right margin
        y_end = image_full_height - self.margins[3]  # bottom margin

        # Calculate overlap coefficients
        if isinstance(self.overlap_x, float):
            cross_coef_x = 1 - self.overlap_x  # Float between 0-1
        else:
            cross_coef_x = 1 - (self.overlap_x / self.shape_x)  # Pixel value

        if isinstance(self.overlap_y, float):
            cross_coef_y = 1 - self.overlap_y  # Float between 0-1
        else:
            cross_coef_y = 1 - (self.overlap_y / self.shape_y)  # Pixel value

        # Calculate grid steps, adjusted to fit within margins
        x_steps = int((x_end - x_start - self.shape_x) / (self.shape_x * cross_coef_x)) + 1
        y_steps = int((y_end - y_start - self.shape_y) / (self.shape_y * cross_coef_y)) + 1

        data_all_crops = []
        batch_of_crops = []
        
        count = 0
        total_steps = (y_steps + 1) * (x_steps + 1)

        for i in range(y_steps + 1):
            for j in range(x_steps + 1):
                x = x_start + int(self.shape_x * j * cross_coef_x)
                y = y_start + int(self.shape_y * i * cross_coef_y)

                # Calculate actual crop dimensions (handling boundary cases)
                crop_width = min(self.shape_x, x_end - x)
                crop_height = min(self.shape_y, y_end - y)

                # Skip if crop is completely outside image
                if crop_width <= 0 or crop_height <= 0:
                    continue

                # Skip if not including residuals and dimensions don't match specified shape
                if not self.include_residuals:
                    if crop_width != self.shape_x or crop_height != self.shape_y:
                        continue

                # Extract the crop
                crop = self.image[y:y + crop_height, x:x + crop_width]
                
                count += 1
                if self.progress_callback is not None:
                    self.progress_callback("Getting Crops", count, total_steps)

                # Create crop element
                crop_element = CropElement(
                    source_image=self.image,
                    source_image_resized=self.image,
                    crop=crop,
                    number_of_crop=count,
                    x_start=x,
                    y_start=y,
                )
                
                data_all_crops.append(crop_element)
                
        self.crops = data_all_crops, batch_of_crops

        return self.crops

    def get_crops(self):
        """Get list of image arrays from all crops.

        Returns:
            List[np.ndarray]: List of image arrays from each crop
        """
        if self.crops is None:
            return []

        crops, _ = self.crops
        return [crop.crop for crop in crops]

    def detect_them(self, predictions=None):
        """
        Method to detect objects in batch of image crops.

        Args:
            predictions: Optional pre-computed model predictions. If None, model.predict will be called.
        """
        crops, batch = self.crops
        self.crops = crops

        self._calculate_inference(
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
        for idx, crop in enumerate(self.crops):
            
            try:
                crop.calculate_real_values()
                if self.resize_initial_size:
                    crop.resize_results()
            except Exception as e:
                print(f"Error: {e}")
                print(traceback.format_exc())
            
            if self.progress_callback is not None:
                self.progress_callback("Resizing Detections", idx + 1, len(self.crops))

    def _calculate_inference(
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

        for idx, (pred, crop) in enumerate(zip(predictions, crops)):

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

            if self.progress_callback is not None:
                self.progress_callback("Detecting Objects", idx + 1, len(crops))
                    
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

