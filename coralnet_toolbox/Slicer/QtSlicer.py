import supervision as sv
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Slicer:
    def __init__(self, model, slice_size=(640, 640), overlap=0.2):
        self.model = model
        self.slice_size = slice_size
        self.overlap = overlap
        self.slicer = None
        
    def initialize_slicer(self):
        def inference_callback(image_slice: np.ndarray):
            results = self.model.predict(image_slice)[0]
            return sv.Detections.from_inference(results)
            
        self.slicer = sv.InferenceSlicer(
            callback=inference_callback,
            slice_size=self.slice_size,
            overlap=self.overlap
        )
        
    def process_image(self, image):
        if self.slicer is None:
            self.initialize_slicer()
        return self.slicer(image)