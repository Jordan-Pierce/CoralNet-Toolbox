import os
import torch
import cv2
from ultralytics import YOLO
from collections import deque

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ObjectDetector:
    """
    Object detection using YOLOv11 from Ultralytics
    """
    def __init__(self, model_size='small', conf_thres=0.25, iou_thres=0.45, classes=None, device=None, path=None):
        """
        Initialize the object detector
        
        Args:
            model_size (str): Model size ('nano', 'small', 'medium', 'large', 'extra')
            conf_thres (float): Confidence threshold for detections
            iou_thres (float): IoU threshold for NMS
            classes (list): List of classes to detect (None for all classes)
            device (str): Device to run inference on ('cuda', 'cpu', 'mps')
            path (str): Custom path to model file (if None, uses default model based on model_size)
        """
        # Use provided device or default to CPU
        if device is None:
            device = 'cpu'
        
        self.device = device
        
        # Set MPS fallback for operations not supported on Apple Silicon
        if self.device == 'mps':
            print("Using MPS device with CPU fallback for unsupported operations")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        print(f"Using device: {self.device} for object detection")
        
        # Map model size to model name
        model_map = {
            'nano': 'yolo11n',
            'small': 'yolo11s',
            'medium': 'yolo11m',
            'large': 'yolo11l',
            'extra': 'yolo11x'
        }
        
        # Use custom path if provided, otherwise use default model name
        if path is not None:
            model_name = path
        else:
            model_name = model_map.get(model_size.lower(), model_map['small'])
        
        try:
            # Load model
            self.model = YOLO(model_name)
            
            try:
                self.imgsz = self.model.__dict__['overrides']['imgsz']
            except Exception:
                self.imgsz = 640
                
            print(f"Loaded YOLO model from {'custom path' if path else model_size} on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying to load with default settings...")
            self.model = YOLO(model_name)
        
        # Set model parameters
        self.model.overrides['conf'] = conf_thres
        self.model.overrides['iou'] = iou_thres
        self.model.overrides['agnostic_nms'] = False
        self.model.overrides['max_det'] = 1000
        self.model.overrides['imgsz'] = self.imgsz
        self.model.overrides['half'] = True
        
        if classes is not None:
            self.model.overrides['classes'] = classes
        
        # Initialize tracking trajectories
        self.tracking_trajectories = {}
    
    def detect(self, image, track=True):
        """
        Detect objects in an image
        
        Args:
            image (numpy.ndarray): Input image (BGR format)
            track (bool): Whether to track objects across frames
            
        Returns:
            tuple: (annotated_image, detections)
                - annotated_image (numpy.ndarray): Image with detections drawn
                - detections (list): List of detections [bbox, score, class_id, object_id]
        """
        detections = []
        
        # Make a copy of the image for annotation
        annotated_image = image.copy()
        
        try:
            if track:
                # Run inference with tracking
                results = self.model.track(image, imgsz=self.imgsz, verbose=False, device=self.device, persist=True)
            else:
                # Run inference without tracking
                results = self.model.predict(image, imgsz=self.imgsz, verbose=False, device=self.device)

        except RuntimeError as e:
            # Handle potential MPS errors
            if self.device == 'mps' and "not currently implemented for the MPS device" in str(e):
                print(f"MPS error during detection: {e}")
                print("Falling back to CPU for this frame")
                if track:
                    results = self.model.track(image, verbose=False, device='cpu', persist=True)
                else:
                    results = self.model.predict(image, verbose=False, device='cpu')
            else:
                # Re-raise the error if not MPS or not an implementation error
                raise
        
        if track:
            # Clean up trajectories for objects that are no longer tracked
            for id_ in list(self.tracking_trajectories.keys()):
                if id_ not in [int(bbox.id) for predictions in results if predictions is not None 
                               for bbox in predictions.boxes if bbox.id is not None]:
                    del self.tracking_trajectories[id_]
            
            # Process results
            for predictions in results:
                if predictions is None:
                    continue
                
                if predictions.boxes is None:
                    continue
                
                # Process boxes
                for bbox in predictions.boxes:
                    # Extract information
                    scores = bbox.conf
                    classes = bbox.cls
                    bbox_coords = bbox.xyxy
                    
                    # Check if tracking IDs are available
                    if hasattr(bbox, 'id') and bbox.id is not None:
                        ids = bbox.id
                    else:
                        ids = [None] * len(scores)
                    
                    # Process each detection
                    for score, class_id, bbox_coord, id_ in zip(scores, classes, bbox_coords, ids):
                        xmin, ymin, xmax, ymax = bbox_coord.cpu().numpy()
                        
                        # Add to detections list
                        detections.append([
                            [xmin, ymin, xmax, ymax],  # bbox
                            float(score),              # confidence score
                            int(class_id),             # class id
                            int(id_) if id_ is not None else None  # object id
                        ])
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_image, 
                                      (int(xmin), int(ymin)), 
                                      (int(xmax), int(ymax)), 
                                      (0, 0, 225), 2)
                        
                        # Add label
                        label_id = f"ID: {int(id_) if id_ is not None else 'N/A'}"
                        label = f"{label_id} {predictions.names[int(class_id)]} {float(score):.2f}"
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        dim, baseline = text_size[0], text_size[1]
                        cv2.rectangle(annotated_image, 
                                      (int(xmin), int(ymin)), 
                                      (int(xmin) + dim[0], int(ymin) - dim[1] - baseline), 
                                      (30, 30, 30), cv2.FILLED)
                        cv2.putText(annotated_image, label, 
                                    (int(xmin), int(ymin) - 7), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Update tracking trajectories
                        if id_ is not None:
                            centroid_x = (xmin + xmax) / 2
                            centroid_y = (ymin + ymax) / 2
                            
                            if int(id_) not in self.tracking_trajectories:
                                self.tracking_trajectories[int(id_)] = deque(maxlen=10)
                            
                            self.tracking_trajectories[int(id_)].append((centroid_x, centroid_y))
            
            # Draw trajectories
            for id_, trajectory in self.tracking_trajectories.items():
                for i in range(1, len(trajectory)):
                    thickness = int(2 * (i / len(trajectory)) + 1)
                    cv2.line(annotated_image, 
                             (int(trajectory[i - 1][0]), int(trajectory[i - 1][1])), 
                             (int(trajectory[i][0]), int(trajectory[i][1])), 
                             (255, 255, 255), thickness)
        
        else:
            # Process results for non-tracking mode
            for predictions in results:
                if predictions is None:
                    continue
                
                if predictions.boxes is None:
                    continue
                
                # Process boxes
                for bbox in predictions.boxes:
                    # Extract information
                    scores = bbox.conf
                    classes = bbox.cls
                    bbox_coords = bbox.xyxy
                    
                    # Process each detection
                    for score, class_id, bbox_coord in zip(scores, classes, bbox_coords):
                        xmin, ymin, xmax, ymax = bbox_coord.cpu().numpy()
                        
                        # Add to detections list
                        detections.append([
                            [xmin, ymin, xmax, ymax],  # bbox
                            float(score),              # confidence score
                            int(class_id),             # class id
                            None                       # object id (None for no tracking)
                        ])
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_image, 
                                      (int(xmin), int(ymin)), 
                                      (int(xmax), int(ymax)), 
                                      (0, 0, 225), 2)
                        
                        # Add label
                        label = f"{predictions.names[int(class_id)]} {float(score):.2f}"
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        dim, baseline = text_size[0], text_size[1]
                        cv2.rectangle(annotated_image, 
                                      (int(xmin), int(ymin)), 
                                      (int(xmin) + dim[0], int(ymin) - dim[1] - baseline), 
                                      (30, 30, 30), cv2.FILLED)
                        cv2.putText(annotated_image, label, 
                                    (int(xmin), int(ymin) - 7), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_image, detections
    
    def get_class_names(self):
        """
        Get the names of the classes that the model can detect
        
        Returns:
            list: List of class names
        """
        return self.model.names