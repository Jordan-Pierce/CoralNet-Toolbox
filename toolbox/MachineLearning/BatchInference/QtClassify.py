from toolbox.MachineLearning.BatchInference.QtBase import Base

class Classify(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setup_generic_layout()

    def apply(self):
        """
        Apply batch inference for image classification.
        """
        # Get the Review Annotations
        if self.review_checkbox.isChecked():
            for image_path in self.get_selected_image_paths():
                self.annotations.extend(self.annotation_window.get_image_review_annotations(image_path))
        else:
            # Get all the annotations
            for image_path in self.get_selected_image_paths():
                self.annotations.extend(self.annotation_window.get_image_annotations(image_path))

        # Crop them, if not already cropped
        self.preprocess_patch_annotations()
        self.batch_inference('classify')
