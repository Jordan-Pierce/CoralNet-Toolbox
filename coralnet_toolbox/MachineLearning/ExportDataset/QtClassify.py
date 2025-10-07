import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import gc
import json
import random 
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

        # Disable negative sample options for classification
        self.include_negatives_radio.setEnabled(False)
        self.exclude_negatives_radio.setEnabled(False)

    def determine_splits(self):
        """
        Determine the splits for train, validation, and test annotations.
        If all annotations come from a single image, split the annotations directly instead of by images.
        """
        unique_images = set(a.image_path for a in self.selected_annotations)
        if len(unique_images) == 1:
            # Split annotations directly
            random.shuffle(self.selected_annotations)
            train_split = int(len(self.selected_annotations) * self.train_ratio)
            val_split = int(len(self.selected_annotations) * (self.train_ratio + self.val_ratio))
            self.train_annotations = self.selected_annotations[:train_split]
            self.val_annotations = self.selected_annotations[train_split:val_split]
            self.test_annotations = self.selected_annotations[val_split:]
        else:
            # Original logic: split by images
            self.train_annotations = [a for a in self.selected_annotations if a.image_path in self.train_images]
            self.val_annotations = [a for a in self.selected_annotations if a.image_path in self.val_images]
            self.test_annotations = [a for a in self.selected_annotations if a.image_path in self.test_images]

    def accept(self):
        """
        Handle the OK button click event to create the dataset.
        """
        if not self.is_ready():
            return

        # Check for single image and warn about potential data contamination
        unique_images = set(a.image_path for a in self.selected_annotations)
        if len(unique_images) == 1:
            reply = QMessageBox.question(
                self,
                "Potential Data Contamination",
                "All annotations come from a single image. Splitting may lead to data contamination. "
                "Do you understand and want to proceed?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        # Create the output folder
        output_dir_path = os.path.join(self.output_dir, self.dataset_name)

        # Check if the output directory exists 
        if os.path.exists(output_dir_path):
            reply = QMessageBox.question(self,
                                        "Directory Exists",
                                        "The output directory already exists. Do you want to merge the datasets?",
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return

            # Read the existing class_mapping.json file if it exists
            class_mapping_path = os.path.join(output_dir_path, "class_mapping.json")
            if os.path.exists(class_mapping_path):
                with open(class_mapping_path, 'r') as json_file:
                    existing_class_mapping = json.load(json_file)
            else:
                existing_class_mapping = {}

            # Merge the new class mappings with the existing ones
            new_class_mapping = self.get_class_mapping()
            merged_class_mapping = self.merge_class_mappings(existing_class_mapping, new_class_mapping)
            self.save_class_mapping_json(merged_class_mapping, output_dir_path)
        else:
            # Save the class mapping JSON file
            os.makedirs(output_dir_path, exist_ok=True)
            class_mapping = self.get_class_mapping()
            self.save_class_mapping_json(class_mapping, output_dir_path)

        try:
            # Create the dataset
            self.create_dataset(output_dir_path)

            QMessageBox.information(self,
                                    "Dataset Created",
                                    "Dataset has been successfully created.")
        
        except Exception as e:
            QMessageBox.critical(self, "Failed to Create Dataset", f"{e}")
            
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
        
        # Create dummy data inside dataset folders if ratio is 0
        self.create_dummy_data(train_dir, self.train_annotations)
        self.create_dummy_data(val_dir, self.val_annotations)
        self.create_dummy_data(test_dir, self.test_annotations)
        
        # Crop the actual annotations
        self.process_annotations(self.train_annotations, train_dir, "Train")
        self.process_annotations(self.val_annotations, val_dir, "Validation")
        self.process_annotations(self.test_annotations, test_dir, "Test")

        # Output the annotations as CoralNet CSV file
        df = []

        for annotation in self.selected_annotations:
            df.append(annotation.to_coralnet())

        pd.DataFrame(df).to_csv(f"{output_dir_path}/dataset.csv", index=False)
        
    def create_dummy_data(self, dataset_dir, annotations):
        """
        Creates dummy data ('NULL' image) in each of the category folders with the dataset folder
        if there are not annotations for that category.
        
        Ultralytics requires there to be files in each of the train/valid/test dataset folders,
        even if they are not used.
        """
        # Get labels that have annotations in this split
        labels_with_annotations = set()
        for annotation in annotations:
            labels_with_annotations.add(annotation.label.short_label_code)
        
        # Loop through each of the selected labels
        for label in self.selected_labels:
            # Create the category folder within the dataset folder
            label_folder = f"{dataset_dir}/{label}"
            os.makedirs(label_folder, exist_ok=True)
            
            # Only create dummy data if there are no annotations for this label in this split
            if label not in labels_with_annotations:
                # Create blank RGB image array (224x224x3)
                blank_img = np.zeros((224, 224, 3), dtype=np.uint8)
                # Save as jpg using numpy
                blank_path = f"{label_folder}/NULL.jpg"
                if not os.path.exists(blank_path):
                    cv2.imwrite(blank_path, blank_img)

    def process_annotations(self, annotations, split_dir, split):
        """Process annotations using parallel execution for cropping, then save them."""
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
        sorted_annotations = sorted(annotations, key=attrgetter('image_path'))
        grouped_annotations = groupby(sorted_annotations, key=attrgetter('image_path'))
        
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