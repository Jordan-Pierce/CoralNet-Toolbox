from dataclasses import dataclass
from abc import ABC, abstractmethod

import cv2
import numpy as np

from ultralytics.engine.results import Results

from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image

from coralnet_toolbox.Results import CombineResults


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class QtBaseModel(DetectionBaseModel, ABC):
    """
    Base class for Transformer foundation models that provides common functionality for 
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
        
    def _normalize_input(self, input) -> list[np.ndarray]:
        """
        Normalizes various input types into a list of images in CV2 (BGR) format.

        Args:
            input: Can be an image path, a list of paths, a numpy array, or a list of numpy arrays.

        Returns:
            A list of images, each as a numpy array in CV2 (BGR) format.
        """
        images = []
        if isinstance(input, str):
            # Single image path
            images = [load_image(input, return_format="cv2")]
        elif isinstance(input, np.ndarray):
            # Single image numpy array (RGB) or a batch of images (NHWC, RGB)
            if input.ndim == 3:
                images = [cv2.cvtColor(input, cv2.COLOR_RGB2BGR)]
            elif input.ndim == 4:
                images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in input]
            else:
                raise ValueError(f"Unsupported numpy array dimensions: {input.ndim}")
        elif isinstance(input, list):
            if all(isinstance(i, str) for i in input):
                # List of image paths
                images = [load_image(path, return_format="cv2") for path in input]
            elif all(isinstance(i, np.ndarray) for i in input):
                # List of image arrays (RGB)
                images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in input]
            else:
                raise ValueError("A list input must contain either all image paths or all numpy arrays.")
        else:
            raise TypeError(f"Unsupported input type: {type(input)}")
        
        return images
        
    @abstractmethod
    def _process_predictions(self, image: np.ndarray, texts: list[str], confidence: float) -> Results:
        """
        Process model predictions for a single image.
        
        Args:
            image: The input image in CV2 (BGR) format.
            texts: The text prompts from the ontology.
            confidence: Confidence threshold.
            
        Returns:
            A single Ultralytics Results object, which may be empty if no detections are found.
        """
        pass

    def predict(self, inputs, confidence=0.01) -> list[Results]:
        """
        Run inference on input images.
        
        Args:
            inputs: Can be an image path, a list of image paths, a numpy array, or a list of numpy arrays.
            confidence: Detection confidence threshold.
            
        Returns:
            A list containing a single combined Ultralytics Results object with detections from all input images.
            Returns an empty list if no detections are found in any image.
        """
        # Step 1: Normalize the input into a consistent list of images
        normalized_inputs = self._normalize_input(inputs)

        # Step 2: Prepare for inference
        results = []
        texts = self.ontology.prompts()
        
        # Step 3: Loop through images and process predictions
        for normalized_input in normalized_inputs:
            result = self._process_predictions(normalized_input, texts, confidence)
            if result:
                results.append(result)
        
        if len(results):
            # Combine the results into one, then wrap in a list
            results = CombineResults().combine_results(results)

        return [results] if results else []