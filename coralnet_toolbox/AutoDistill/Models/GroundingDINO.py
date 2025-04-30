from dataclasses import dataclass

import cv2
import numpy as np

import supervision as sv

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

# TODO Resize images

@dataclass
class GroundingDINOModel(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology, model="SwinB", device: str = "cpu"):
        self.ontology = ontology
        self.device = device
        
        if model == "SwinB":
            model_name = "IDEA-Research/grounding-dino-base"
        else:
            model_name = "IDEA-Research/grounding-dino-tiny"
            
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(self.device)

    def predict(self, input, confidence=0.01):
        """"""
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
        
        # Loop through images
        for image in images:
            # Make predictions
            texts = self.ontology.prompts()
            class_idx_mapper = {label: idx for idx, label in enumerate(texts)}
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
                continue

            detection = sv.Detections(xyxy=np.array(final_boxes),
                                      class_id=np.array(final_labels),
                                      confidence=np.array(final_scores))
                
            detections_result.append(detection)

        # Return detections for a single image directly,
        # or a list of detections if multiple images were passed
        if len(detections_result) == 1:
            return detections_result[0]
        else:
            return detections_result
        