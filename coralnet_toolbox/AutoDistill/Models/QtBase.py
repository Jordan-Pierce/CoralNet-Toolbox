from dataclasses import dataclass
from abc import ABC, abstractmethod

import cv2
import numpy as np

from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class QtBaseModel(DetectionBaseModel, ABC):
    """
    Base class for CoralNet foundation models that provides common functionality for 
    handling inputs, processing image data, and formatting detection results.
    """
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology, device: str = "cpu"):
        """
        Initialize the base model with ontology and device.
        
        Args:
            ontology: The CaptionOntology containing class labels
            device: The compute device (cpu, cuda, etc.)
        """
        self.ontology = ontology
        self.device = device
        self.processor = None
        self.model = None
        
    @abstractmethod
    def _process_predictions(self, image, texts, class_idx_mapper, confidence):
        """
        Process model predictions for a single image.
        
        Args:
            image: The input image
            texts: The text prompts from the ontology
            class_idx_mapper: Mapping from text labels to class indices
            confidence: Confidence threshold
            
        Returns:
            sv.Detections object or None if no detections
        """
        pass

    def predict(self, input, confidence=0.01):
        """
        Run inference on input images.
        
        Args:
            input: Can be an image path, a list of image paths, a numpy array, or a list of numpy arrays
            confidence: Detection confidence threshold
            
        Returns:
            Either a single sv.Detections object or a list of sv.Detections objects
        """
        # Normalize input into a list of CV2-format images
        images = []
        if isinstance(input, str):
            # Single image path
            images = [load_image(input, return_format="cv2")]
        elif isinstance(input, np.ndarray):
            # Single image numpy array or batch of images
            if input.ndim == 3:
                images = [cv2.cvtColor(input, cv2.COLOR_RGB2BGR)]
            elif input.ndim == 4:
                for img in input:
                    images.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            else:
                raise ValueError("Unsupported numpy array dimensions.")
        elif isinstance(input, list):
            if all(isinstance(i, str) for i in input):
                # List of image paths
                for path in input:
                    images.append(load_image(path, return_format="cv2"))
            elif all(isinstance(i, np.ndarray) for i in input):
                # List of image arrays
                for img in input:
                    images.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            else:
                raise ValueError("List must contain all image paths or all numpy arrays.")
        else:
            raise ValueError(
                "Input must be an image path, a list of image paths, a numpy array, or a list/array of numpy arrays."
            )

        detections_result = []
        
        # Get text prompts and create class index mapper
        texts = self.ontology.prompts()
        class_idx_mapper = {label: idx for idx, label in enumerate(texts)}
        
        # Loop through images
        for image in images:
            # Process predictions for this image
            detection = self._process_predictions(image, texts, class_idx_mapper, confidence)
            if detection is not None:
                detections_result.append(detection)

        # Return detections for a single image directly,
        # or a list of detections if multiple images were passed
        if len(detections_result) == 1:
            return detections_result[0]
        else:
            return detections_result