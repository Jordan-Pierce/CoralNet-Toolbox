import os
import urllib.request
from dataclasses import dataclass

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

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

    def __init__(
        self, ontology: CaptionOntology, box_threshold=0.35, text_threshold=0.25, model="SwinB",
    ):
        self.ontology = ontology
        self.grounding_dino_model = load_grounding_dino(model)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def predict(self, input: str) -> sv.Detections:
        image = load_image(input, return_format="cv2")

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

        return detections