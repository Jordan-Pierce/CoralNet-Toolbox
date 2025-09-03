from dataclasses import dataclass

import torch

from ultralytics.engine.results import Results

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from autodistill.detection import CaptionOntology

from coralnet_toolbox.Transformers.Models.QtBase import QtBaseModel


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class GroundingDINOModel(QtBaseModel):
    def __init__(self, ontology: CaptionOntology, model="SwinB", device: str = "cpu"):
        super().__init__(ontology, device)
        
        if model == "SwinB":
            model_name = "IDEA-Research/grounding-dino-base"
        else:
            model_name = "IDEA-Research/grounding-dino-tiny"
            
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(self.device)

    def _process_predictions(self, image, texts, confidence):
        """Process model predictions for a single image."""
        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        results_processed = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=confidence,
            target_sizes=[image.shape[:2]],
        )[0]

        boxes = results_processed["boxes"]
        scores = results_processed["scores"]
        
        # If no objects are detected, return an empty list to match the original behavior.
        if scores.nelement() == 0:
            return []

        # Per original logic, assign all detections to class_id 0.
        # TODO: We are only supporting a single class right now
        class_ids = torch.zeros(scores.shape[0], 1, device=self.device)

        # Combine boxes, scores, and class_ids into the (N, 6) tensor format
        # required by the Results object: [x1, y1, x2, y2, confidence, class_id]
        combined_data = torch.cat([
            boxes,
            scores.unsqueeze(1),
            class_ids
        ], dim=1)

        # Create the dictionary mapping class indices to class names.
        names = {idx: text for idx, text in enumerate(self.ontology.classes())}
        
        # Create the Results object with a DETACHED tensor
        result = Results(orig_img=image, 
                         path=None, 
                         names=names, 
                         boxes=combined_data.detach().cpu())
        
        return result