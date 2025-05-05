import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
from itertools import groupby
from operator import attrgetter
from concurrent.futures import ThreadPoolExecutor, as_completed

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox, QCheckBox, QVBoxLayout,
                             QGroupBox, QButtonGroup, QPushButton, QHBoxLayout,
                             QDialogButtonBox)

from coralnet_toolbox.MachineLearning.BatchInference.QtBase import Base

from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

# TODO Crash might be due to multithreading?
class Classify(Base):
    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self.setWindowTitle("Classify Batch Inference")

        self.deploy_model_dialog = main_window.classify_deploy_model_dialog
        
        # Add step-by-step buttons after the base class is initialized
        self.add_step_buttons()
        
    def add_step_buttons(self):
        """
        Add buttons for step-by-step processing (preprocess and inference).
        """
        # Create a group box for the separate action buttons
        step_group_box = QGroupBox("Step-by-Step Processing")
        step_layout = QHBoxLayout()
        
        # Add buttons for individual steps
        self.preprocess_button = QPushButton("1. Preprocess Annotations")
        self.infer_button = QPushButton("2. Run Inference")
        self.infer_button.setEnabled(False)  # Disable until preprocessing is done
        
        # Connect the buttons to their respective methods
        self.preprocess_button.clicked.connect(self.preprocess_only)
        self.infer_button.clicked.connect(self.infer_only)
        
        step_layout.addWidget(self.preprocess_button)
        step_layout.addWidget(self.infer_button)
        step_group_box.setLayout(step_layout)
        
        # Insert the group box before the last widget (which should be the button box)
        self.layout.insertWidget(self.layout.count() - 1, step_group_box)
        
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
        Override the base class method to use custom button labels.
        """
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok_button = button_box.button(QDialogButtonBox.Ok)
        ok_button.setText("Run All Steps")
        
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

        # Crop annotations
        self.bulk_preprocess_patch_annotations()
    
    def apply(self):
        """
        Apply batch inference for image classification.
        """
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
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

    def preprocess_only(self):
        """
        Perform only the preprocessing step.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Get the selected image paths
            self.image_paths = self.get_selected_image_paths()
            self.preprocess_annotations()
            self.infer_button.setEnabled(True)
            QMessageBox.information(self, 
                                    "Preprocessing Complete", 
                                    f"Successfully preprocessed {len(self.prepared_patches)} annotations.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to preprocess annotations: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()
    
    def infer_only(self):
        """
        Perform only the inference step.
        """
        if not self.prepared_patches:
            QMessageBox.warning(self, 
                                "Warning", 
                                "No preprocessed annotations found. Please run preprocessing first.")
            return
            
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.batch_inference()
            QMessageBox.information(self, 
                                    "Inference Complete", 
                                   "Batch inference completed successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to perform inference: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

    def preprocess_patch_annotations(self):
        """
        Preprocess patch annotations by cropping the images concurrently.
        
        Deprecated: Use bulk_preprocess_patch_annotations instead.
        """
        # Get unique image paths
        self.image_paths = list(set(a.image_path for a in self.annotations))
        if not self.image_paths:
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Cropping Annotations")
        progress_bar.show()
        progress_bar.start_progress(len(self.image_paths))

        # Group annotations by image path
        grouped_annotations = groupby(sorted(self.annotations, key=attrgetter('image_path')),
                                   key=attrgetter('image_path'))

        try:
            # Crop the annotations
            for idx, (image_path, group) in enumerate(grouped_annotations):
                # Process image annotations
                image_annotations = list(group)
                image_annotations = self.annotation_window.crop_annotations(image_path, 
                                                                            image_annotations, 
                                                                            verbose=False)
                # Add the cropped annotations to the list of prepared patches
                self.prepared_patches.extend(image_annotations)

                # Update the progress bar
                progress_bar.update_progress()

        except Exception as exc:
            print(f'{image_path} generated an exception: {exc}')

        finally:
            # Restore the cursor
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()
            
    def bulk_preprocess_patch_annotations(self):
        """
        Bulk preprocess patch annotations by cropping the images concurrently.
        """
        # Get unique image paths
        self.image_paths = list(set(a.image_path for a in self.annotations))
        if not self.image_paths:
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Cropping Annotations")
        progress_bar.show()
        progress_bar.start_progress(len(self.image_paths))

        # Group annotations by image path
        grouped_annotations = groupby(sorted(self.annotations, key=attrgetter('image_path')),
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
            print(f"{futures[future]} generated an exception: {e}")

        finally:
            # Restore the cursor
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()

    def batch_inference(self):
        """
        Perform batch inference on the selected images and annotations.
        
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
        progress_bar.start_progress(len(self.image_paths))

        # Group annotations by image path
        groups = groupby(sorted(self.prepared_patches, key=attrgetter('image_path')),
                        key=attrgetter('image_path'))

        # Make predictions on each image's annotations
        for path, patches in groups:
            try:
                print(f"\nMaking predictions on {path}")
                self.deploy_model_dialog.predict(inputs=list(patches))
                
            except Exception as e:
                print(f"Failed to make predictions on {path}: {e}")
                continue
            
            finally: 
                progress_bar.update_progress()

        QApplication.restoreOverrideCursor()
        progress_bar.finish_progress()
        progress_bar.stop_progress()
        progress_bar.close()
