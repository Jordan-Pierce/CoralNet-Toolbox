from dataclasses import dataclass

import cv2
import numpy as np

import supervision as sv

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from autodistill.detection import CaptionOntology
from autodistill.helpers import load_image

from coralnet_toolbox.AutoDistill.Models.QtBase import QtBaseModel


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

    def _process_predictions(self, image, texts, class_idx_mapper, confidence):
        """Process model predictions for a single image."""
        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=confidence,
            text_threshold=confidence,
            target_sizes=[image.shape[:2]],
        )[0]

        boxes, scores, labels = (
            results["boxes"],
            results["scores"],
            results["text_labels"],
        )

        final_boxes, final_scores, final_labels = [], [], []

        for box, score, label in zip(boxes, scores, labels):
            try:
                box = box.detach().cpu().numpy().astype(int).tolist()
                score = score.item()
                # Grounding Dino issues...
                label = class_idx_mapper[label.split(" ")[0]]
                
                # Amplify scores
                if score < confidence:
                    continue

                final_boxes.append(box)
                final_scores.append(score)
                final_labels.append(label)
            
            except Exception as e:
                print(f"Error: Issue converting predictions:\n{e}")
                continue

        if len(final_boxes) == 0:
            return None

        return sv.Detections(
            xyxy=np.array(final_boxes),
            class_id=np.array(final_labels),
            confidence=np.array(final_scores)
        )