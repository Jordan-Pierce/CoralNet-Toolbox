import warnings

import os
from itertools import groupby
from operator import attrgetter
from concurrent.futures import ThreadPoolExecutor, as_completed

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox, QCheckBox, QVBoxLayout,
                             QGroupBox, QButtonGroup, QDialogButtonBox)

from coralnet_toolbox.MachineLearning.BatchInference.QtBase import Base

from coralnet_toolbox.QtProgressBar import ProgressBar

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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

    def setup_buttons_layout(self):
        """
        Override the base class method to use the default OK button.
        """
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.apply)
        button_box.rejected.connect(self.reject)
        self.layout.addWidget(button_box)
    
    def preprocess_annotations(self):
        """
        Get annotations based on user selection and preprocess them.
        """
        # Get the annotations based on user selection
        if self.review_checkbox.isChecked():
            for image_path in self.image_paths:
                self.annotations.extend(self.annotation_window.get_image_review_annotations(image_path))
        else:
            for image_path in self.image_paths:
                self.annotations.extend(self.annotation_window.get_image_annotations(image_path))

        # Check if annotations need to be cropped
        annotations_to_crop = []
        for annotation in self.annotations:
            if hasattr(annotation, 'cropped_image') and annotation.cropped_image:
                # Annotation already has cropped image, add to prepared patches
                self.prepared_patches.append(annotation)
            else:
                # Annotation needs to be cropped
                annotations_to_crop.append(annotation)

        # Only crop annotations that need cropping
        if annotations_to_crop:
            self.bulk_preprocess_patch_annotations(annotations_to_crop)
    
    def apply(self):
        """
        Apply batch inference for image classification.
        """
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Get the selected image paths first
            self.image_paths = self.get_selected_image_paths()
            
            # Run preprocessing and inference
            self.preprocess_annotations()
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

    def bulk_preprocess_patch_annotations(self, annotations_to_crop=None):
        """
        Bulk preprocess patch annotations by cropping the images concurrently.
        
        Args:
            annotations_to_crop: List of annotations that need to be cropped.
                                If None, uses self.annotations.
        """
        if annotations_to_crop is None:
            annotations_to_crop = self.annotations
            
        if not annotations_to_crop:
            return

        # Get unique image paths for annotations that need cropping
        crop_image_paths = list(set(a.image_path for a in annotations_to_crop))
        
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Cropping Annotations")
        progress_bar.show()
        progress_bar.start_progress(len(crop_image_paths))

        # Group annotations by image path
        grouped_annotations = groupby(sorted(annotations_to_crop, key=attrgetter('image_path')),
                                      key=attrgetter('image_path'))

        try:
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
                # Dictionary to track futures and their corresponding image paths
                futures = {}
                
                # Process each group of annotations by image path
                for image_path, group in grouped_annotations:
                    # Convert group iterator to list for reuse
                    image_annotations = list(group)
                    
                    # Submit cropping task asynchronously for each image
                    # Returns a Future object representing pending execution
                    future = executor.submit(self.annotation_window.crop_annotations, 
                                             image_path, 
                                             image_annotations, 
                                             verbose=False)
                    
                    # Store image path for each future for error reporting
                    futures[future] = image_path

                # Process completed futures as they finish
                for future in as_completed(futures):
                    try:
                        # Get cropped patches from completed task
                        cropped = future.result()
                        # Add cropped patches to prepared patches list
                        self.prepared_patches.extend(cropped)
                    except Exception as exc:
                        print(f"{futures[future]} generated an exception: {exc}")
                    finally:
                        # Update progress bar after each image is processed
                        progress_bar.update_progress()

        except Exception as e:
            print(f"Error in bulk preprocessing: {e}")

        finally:
            # Restore the cursor
            QApplication.restoreOverrideCursor()
            progress_bar.finish_progress()
            progress_bar.stop_progress()
            progress_bar.close()

    def batch_inference(self):
        """
        Perform batch inference on the selected images.
        
        Slower doing a for-loop over the prepared patches, but it's safer and memory efficient.
        """
        if not self.prepared_patches:
            raise ValueError("No preprocessed annotations found. Please run preprocessing first.")
            
        self.loaded_model = self.deploy_model_dialog.loaded_model
        if self.loaded_model is None:
            raise ValueError("No model loaded. Please load a model first.")

        # Make predictions on each image's annotations
        progress_bar = ProgressBar(self.annotation_window, title="Batch Inference")
        progress_bar.show()

        # Group annotations by image path
        groups = groupby(sorted(self.prepared_patches, key=attrgetter('image_path')),
                         key=attrgetter('image_path'))
        
        # Count number of unique image paths
        num_paths = len(set(a.image_path for a in self.prepared_patches))

        # Make predictions on each image's annotations
        for idx, (path, patches) in enumerate(groups):
            try:
                progress_bar.set_title(f"Predicting: {idx + 1}/{num_paths} - {os.path.basename(path)}")
                self.deploy_model_dialog.predict(inputs=list(patches), progress_bar=progress_bar)
            except Exception as e:
                print(f"Failed to make predictions on {path}: {e}")
                continue

        QApplication.restoreOverrideCursor()
        progress_bar.finish_progress()
        progress_bar.stop_progress()
        progress_bar.close()