import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby
from operator import attrgetter

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox, QCheckBox, QVBoxLayout,
                             QGroupBox, QButtonGroup)

from coralnet_toolbox.MachineLearning.BatchInference.QtBase import Base

from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Classify(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Classify Batch Inference")
        
        self.deploy_model_dialog = main_window.classify_deploy_model_dialog
        
    def setup_task_specific_layout(self):
        """
        Set up the layout with both generic and classification-specific options.
        """
        group_box = QGroupBox("Annotation Options")
        layout = QVBoxLayout()

        # Create a button group for the annotation checkboxes
        annotation_options_group = QButtonGroup(self)

        # Add the checkboxes to the button group
        self.review_checkbox = QCheckBox("Predict Review Annotation")
        self.all_checkbox = QCheckBox("Predict All Annotations")
        annotation_options_group.addButton(self.review_checkbox)
        annotation_options_group.addButton(self.all_checkbox)

        # Ensure only one checkbox can be checked at a time
        annotation_options_group.setExclusive(True)
        # Set the default checkbox
        self.review_checkbox.setChecked(True)

        # Build the annotation layout
        layout.addWidget(self.review_checkbox)
        layout.addWidget(self.all_checkbox)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def apply(self):
        """
        Apply batch inference for image classification.
        """
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
        
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
            self.batch_inference()
        
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            
        finally:
            # Restore the cursor
            QApplication.restoreOverrideCursor()
            self.annotations = []
            self.prepared_patches = []
            self.image_paths = []
        
        self.accept()
        
    def preprocess_patch_annotations(self):
        """
        Preprocess patch annotations by cropping the images based on the annotations.
        """
        # Get unique image paths
        self.image_paths = list(set(a.image_path for a in self.annotations))
        if not self.image_paths:
            return

        progress_bar = ProgressBar(self, title="Cropping Annotations")
        progress_bar.show()
        progress_bar.start_progress(len(self.image_paths))

        # Group annotations by image path
        grouped_annotations = groupby(sorted(self.annotations, key=attrgetter('image_path')), 
                                   key=attrgetter('image_path'))

        for image_path, group in grouped_annotations:
            try:
                # Process image annotations
                image_annotations = list(group)
                image_annotations = self.annotation_window.crop_these_image_annotations(image_path, image_annotations)
                self.prepared_patches.extend(image_annotations)

            except Exception as exc:
                print(f'{image_path} generated an exception: {exc}')
            finally:
                progress_bar.update_progress()

        progress_bar.stop_progress()
        progress_bar.close()
        
    def batch_inference(self):
        """
        Perform batch inference on the selected images and annotations.
        """
        self.loaded_model = self.deploy_model_dialog.loaded_model
        
        # Make predictions on each image's annotations
        progress_bar = ProgressBar(self, title="Batch Inference")
        progress_bar.show()
        progress_bar.start_progress(len(self.image_paths))

        if self.loaded_model is not None:
            # Group annotations by image path
            groups = groupby(sorted(self.prepared_patches, key=attrgetter('image_path')), key=attrgetter('image_path'))

            # Make predictions on each image's annotations
            for path, patches in groups:
                self.deploy_model_dialog.predict(inputs=list(patches))
                progress_bar.update_progress()

        progress_bar.stop_progress()
        progress_bar.close()
        
        # Clear the list of annotations
        self.annotations = []
        self.prepared_patches = []