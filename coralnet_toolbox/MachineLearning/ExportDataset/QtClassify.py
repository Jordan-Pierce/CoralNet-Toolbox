import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import gc
from itertools import groupby
from operator import attrgetter

from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QGroupBox, QVBoxLayout, QLabel, QMessageBox, QApplication)

from coralnet_toolbox.MachineLearning.ExportDataset.QtBase import Base
from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Classify(Base):
    def __init__(self, parent=None):
        super(Classify, self).__init__(parent)
        self.setWindowTitle("Export Classification Dataset")
        self.setWindowIcon(get_icon("coral"))

    def setup_info_layout(self):
        """Setup the info layout"""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_text = "Export Patches, Rectangles, and Polygons to create a YOLO-formatted Classification dataset."
        info_label = QLabel(info_text)

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def update_annotation_type_checkboxes(self):
        """
        Update the state of annotation type checkboxes based on the selected dataset type.
        """
        self.include_patches_checkbox.setChecked(True)
        self.include_patches_checkbox.setEnabled(False)
        self.include_rectangles_checkbox.setChecked(True)
        self.include_rectangles_checkbox.setEnabled(True)
        self.include_polygons_checkbox.setChecked(True)
        self.include_polygons_checkbox.setEnabled(True)

    def create_dataset(self, output_dir_path):
        """
        Create an image classification dataset.

        Args:
            output_dir_path (str): Path to the output directory.
        """
        # Create the train, val, and test directories
        train_dir = os.path.join(output_dir_path, 'train')
        val_dir = os.path.join(output_dir_path, 'val')
        test_dir = os.path.join(output_dir_path, 'test')

        # Create a blank sample in train folder if it's a test-only dataset
        # Ultralytics bug... it doesn't like empty directories (hacky)
        for label in self.selected_labels:
            label_folder = os.path.join(train_dir, label)
            os.makedirs(f"{train_dir}/{label}/", exist_ok=True)
            # Create blank RGB image array (224x224x3)
            blank_img = np.zeros((224, 224, 3), dtype=np.uint8)
            # Save as jpg using numpy
            cv2.imwrite(os.path.join(label_folder, 'NULL.jpg'), blank_img)

        self.process_annotations(self.train_annotations, train_dir, "Train")
        self.process_annotations(self.val_annotations, val_dir, "Validation")
        self.process_annotations(self.test_annotations, test_dir, "Test")

        # Output the annotations as CoralNet CSV file
        df = []

        for annotation in self.selected_annotations:
            df.append(annotation.to_coralnet())

        pd.DataFrame(df).to_csv(f"{output_dir_path}/dataset.csv", index=False)
        
    def process_annotations(self, annotations, split_dir, split):
        """Deprecated method for processing annotations without parallel execution"""
        
        # Get unique image paths
        image_paths = list(set(a.image_path for a in annotations))
        if not image_paths:
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title=f"Creating {split} Dataset")
        progress_bar.show()
        progress_bar.start_progress(len(image_paths))

        # Group annotations by image path
        grouped_annotations = groupby(sorted(annotations, key=attrgetter('image_path')),
                                   key=attrgetter('image_path'))

        for image_path, group in grouped_annotations:
            try:
                # Process image annotations
                image_annotations = list(group)
                image_annotations = self.annotation_window.crop_annotations(image_path,
                                                                            image_annotations,
                                                                            verbose=False)
            except Exception as e:
                print(f'{image_path} generated an exception: {e}')
                continue
            
            try:
                # Save each cropped annotation
                for annotation in image_annotations:
                    # If the annotation has no cropped image, skip it
                    if not annotation.cropped_image:
                        print(f"Skipping annotation {annotation.id} because it has no cropped image")
                        continue
                    
                    label_code = annotation.label.short_label_code
                    output_path = os.path.join(split_dir, label_code)
                    # Create a split / label directory if it does not exist
                    os.makedirs(output_path, exist_ok=True)
                    output_filename = f"{label_code}_{annotation.id}.jpg"
                    full_output_path = os.path.join(output_path, output_filename)

                    try:
                        annotation.cropped_image.save(full_output_path, "JPG", quality=100)
                        
                    except Exception as e:
                        print(f"ERROR: Issue saving image {full_output_path}: {e}")
                        # Optionally, save as PNG if JPG fails
                        png_path = full_output_path.replace(".jpg", ".png")
                        annotation.cropped_image.save(png_path, "PNG")

            except Exception as e:
                print(f'{image_path} generated an exception: {e}')
            finally:
                progress_bar.update_progress()

        # Make cursor normal
        QApplication.restoreOverrideCursor()
        progress_bar.stop_progress()
        progress_bar.close()
        progress_bar = None

    def process_annotations(self, annotations, split_dir, split):
        """
        Process annotations using parallel execution for cropping, then save them.
        """
        # Get unique image paths
        image_paths = list(set(a.image_path for a in annotations))
        if not image_paths:
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title=f"Creating {split} Dataset")
        progress_bar.show()
        progress_bar.start_progress(len(image_paths))

        # Group annotations by image path
        grouped_annotations = groupby(sorted(annotations, key=attrgetter('image_path')),
                                key=attrgetter('image_path'))
        
        cropped_annotations = []
        
        try:
            # Use ThreadPoolExecutor for parallel processing with worker threads
            with ThreadPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
                # Dictionary to track futures and their corresponding image paths
                futures = {}
                
                # Process each group of annotations by image path
                for image_path, group in grouped_annotations:
                    # Convert group iterator to list for reuse
                    image_annotations = list(group)
                    
                    # Submit cropping task asynchronously for each image
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
                        # Add cropped patches to our collection
                        cropped_annotations.extend(cropped)
                    except Exception as e:
                        print(f"{futures[future]} generated an exception: {e}")
                    finally:
                        # Update progress bar after each image is processed
                        progress_bar.update_progress()
                        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing annotations: {e}")
            # Make cursor normal
            QApplication.restoreOverrideCursor()
            progress_bar.finish_progress()
            progress_bar.stop_progress()
            progress_bar.close()
            progress_bar = None
            return
        
        finally:
            gc.collect()
        
        try:
            # Update progress bar for cropping annotations
            progress_bar.set_title(f"Saving {split} Dataset")
            progress_bar.start_progress(len(cropped_annotations))
            
            # Now save all cropped annotations
            for annotation in cropped_annotations:
                # If the annotation has no cropped image, skip it
                if not annotation.cropped_image:
                    print(f"Skipping annotation {annotation.id} because it has no cropped image")
                    continue
                
                label_code = annotation.label.short_label_code
                output_path = os.path.join(split_dir, label_code)
                # Create a split / label directory if it does not exist
                os.makedirs(output_path, exist_ok=True)
                output_filename = f"{label_code}_{annotation.id}.jpg"
                full_output_path = os.path.join(output_path, output_filename)

                try:
                    annotation.cropped_image.save(full_output_path, "JPG", quality=100)
                except Exception as e:
                    print(f"ERROR: Issue saving image {full_output_path}: {e}")
                    # Optionally, save as PNG if JPG fails
                    png_path = full_output_path.replace(".jpg", ".png")
                    annotation.cropped_image.save(png_path, "PNG")

        except Exception as exc:
            print(f'Error in parallel processing: {exc}')
        finally:
            # Make cursor normal
            QApplication.restoreOverrideCursor()
            progress_bar.finish_progress()
            progress_bar.stop_progress()
            progress_bar.close()
            progress_bar = None
            gc.collect()
