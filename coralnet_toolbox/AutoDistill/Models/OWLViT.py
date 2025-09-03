from dataclasses import dataclass

import cv2
import numpy as np
import torch

import supervision as sv
from ultralytics.engine.results import Results

from transformers import OwlViTForObjectDetection, OwlViTProcessor

from autodistill.detection import CaptionOntology
from autodistill.helpers import load_image

from coralnet_toolbox.AutoDistill.Models.QtBase import QtBaseModel
from coralnet_toolbox.Results.ConvertResults import ConvertResults


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class OWLViTModel(QtBaseModel):
    def __init__(self, ontology: CaptionOntology, device: str = "cpu"):
        super().__init__(ontology, device)
        
        model_name = "google/owlvit-base-patch32"
        self.processor = OwlViTProcessor.from_pretrained(model_name, use_fast=True)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
        self.converter = ConvertResults()

    def _process_predictions(self, image, texts, class_idx_mapper, confidence):
        """Process model predictions for a single image."""
        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs,
            threshold=confidence,
            target_sizes=[image.shape[:2]]
        )[0]

        boxes, scores, labels = (
            results["boxes"],
            results["scores"],
            results["labels"],
        )

        final_boxes, final_scores, final_labels = [], [], []

        for box, score, label in zip(boxes, scores, labels):
            try:
                box = box.detach().cpu().numpy().astype(int).tolist()
                score = score.item()
                
                # TODO AutoDistill only supports a single class right now
                label = 0
                # label_index = label.item()
                # class_label = texts[label_index]
                # label = class_idx_mapper[class_label]
                
                # Filter by confidence
                if score < confidence:
                    continue

                final_boxes.append(box)
                final_scores.append(score)
                final_labels.append(label)
            
            except Exception as e:
                print(f"Error: Issue converting predictions:\n{e}")
                continue

        # Create supervision Detections object
        detections = sv.Detections(
            xyxy=np.array(final_boxes) if final_boxes else np.empty((0, 4), dtype=float),
            class_id=np.array(final_labels) if final_labels else np.empty(0, dtype=int),
            confidence=np.array(final_scores) if final_scores else np.empty(0, dtype=float)
        )
        
        # Convert to Ultralytics Results
        names = {idx: text for idx, text in enumerate(self.ontology.classes())}
        results = self.converter.from_supervision(detections, image, names=names)

        if results[0].boxes:
            return results
        else:
            return []