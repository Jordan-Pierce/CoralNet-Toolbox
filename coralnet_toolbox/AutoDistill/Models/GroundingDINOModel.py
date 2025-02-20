import os
import urllib.request
from dataclasses import dataclass

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import torch
import numpy as np

torch.use_deterministic_algorithms(False)

import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image
from groundingdino.util.inference import Model

from autodistill_grounding_dino.helpers import (combine_detections)

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def load_grounding_dino(model="SwinT"):
    """Load the grounding DINO model."""
    # Define the paths
    AUTODISTILL_CACHE_DIR = os.path.expanduser("~/.cache/autodistill")
    GROUDNING_DINO_CACHE_DIR = os.path.join(AUTODISTILL_CACHE_DIR, "groundingdino")

    if model == "SwinT":
        GROUNDING_DINO_CONFIG_PATH = os.path.join(GROUDNING_DINO_CACHE_DIR, "GroundingDINO_SwinT_OGC.py")
        GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GROUDNING_DINO_CACHE_DIR, "groundingdino_swint_ogc.pth")
    else:
        GROUNDING_DINO_CONFIG_PATH = os.path.join(GROUDNING_DINO_CACHE_DIR, "GroundingDINO_SwinB_OGC.py")
        GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GROUDNING_DINO_CACHE_DIR, "groundingdino_swinb_cogcoor.pth")
    
    try:
        print("trying to load grounding dino directly")
        grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            device=DEVICE,
        )
        return grounding_dino_model
    
    except Exception:
        print("downloading dino model weights")
        if not os.path.exists(GROUDNING_DINO_CACHE_DIR):
            os.makedirs(GROUDNING_DINO_CACHE_DIR)
            
        if model == "SwinT":
            if not os.path.exists(GROUNDING_DINO_CHECKPOINT_PATH):
                url = "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth"
                urllib.request.urlretrieve(url, GROUNDING_DINO_CHECKPOINT_PATH)

            if not os.path.exists(GROUNDING_DINO_CONFIG_PATH):
                url = "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.py"
                urllib.request.urlretrieve(url, GROUNDING_DINO_CONFIG_PATH)
        else:
            if not os.path.exists(GROUNDING_DINO_CHECKPOINT_PATH):
                url = "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth"
                urllib.request.urlretrieve(url, GROUNDING_DINO_CHECKPOINT_PATH)

            if not os.path.exists(GROUNDING_DINO_CONFIG_PATH):
                url = "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB_cfg.py"
                urllib.request.urlretrieve(url, GROUNDING_DINO_CONFIG_PATH)

        grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            device=DEVICE,
        )

        return grounding_dino_model
    

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class GroundingDINO(DetectionBaseModel):
    ontology: CaptionOntology
    grounding_dino_model: Model
    box_threshold: float
    text_threshold: float

    def __init__(self, ontology: CaptionOntology, box_threshold=0.35, text_threshold=0.25, model="SwinB"):
        self.ontology = ontology
        self.grounding_dino_model = load_grounding_dino(model)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def predict(self, input) -> sv.Detections:

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
        for image in images:
            detections_list = []
            for _, description in enumerate(self.ontology.prompts()):
                detections = self.grounding_dino_model.predict_with_classes(
                    image=image,
                    classes=[description],
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                )
                detections_list.append(detections)

            detections = combine_detections(
                detections_list, overwrite_class_ids=range(len(detections_list))
            )
            detections_result.append(detections)

        # Return detections for a single image directly,
        # or a list of detections if multiple images were passed
        if len(detections_result) == 1:
            return detections_result[0]
        else:
            return detections_result