import warnings

import os
import gc
import random 
import shutil
import yaml

import numpy as np
from PIL import Image

from rasterio.features import rasterize
from shapely.geometry import Polygon, box

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QGroupBox, QVBoxLayout, QLabel, QApplication, QCheckBox, QTableWidgetItem)
from PyQt5.QtGui import QColor

from coralnet_toolbox.MachineLearning.ExportDataset.QtBase import Base
from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Semantic(Base):
    """
    Export semantic segmentation masks from MaskAnnotations.
    
    Unlike vector annotations (patches, rectangles, polygons), this class works with 
    mask annotations that contain pixel-level class information. The table shows 
    boolean presence or counts of class labels within mask annotations.
    """
    
    def __init__(self, main_window, parent=None):
        super(Semantic, self).__init__(main_window, parent)
        self.setWindowTitle("Export Semantic Segmentation Dataset")
        self.setWindowIcon(get_icon("mask.png"))

    def setup_info_layout(self):
        """Setup the info layout"""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text
        info_text = ("Export semantic segmentation masks in YOLO format. "
                     "Supports both MaskAnnotations and vector annotations (Patches, Rectangles, Polygons). "
                     "Vector annotations will be rasterized into semantic masks. "
                     "This exports pixel-level class labels for semantic segmentation training.")
        info_label = QLabel(info_text)

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def create_annotation_layout(self):
        """Creates the annotation type checkboxes layout group box - supports all annotation types for semantic."""
        group_box = QGroupBox("Annotation Types")
        layout = QVBoxLayout()

        # For semantic segmentation, we support both mask and vector annotations
        self.include_masks_checkbox = QCheckBox("Include Mask Annotations")
        self.include_masks_checkbox.setChecked(True)
        self.include_masks_checkbox.setEnabled(True)
        
        # Enable vector annotation types - these will be rasterized into semantic masks
        self.include_patches_checkbox = QCheckBox("Include Patch Annotations")
        self.include_patches_checkbox.setChecked(True)
        self.include_patches_checkbox.setEnabled(True)
        
        self.include_rectangles_checkbox = QCheckBox("Include Rectangle Annotations") 
        self.include_rectangles_checkbox.setChecked(True)
        self.include_rectangles_checkbox.setEnabled(True)
        
        self.include_polygons_checkbox = QCheckBox("Include Polygon Annotations")
        self.include_polygons_checkbox.setChecked(True)
        self.include_polygons_checkbox.setEnabled(True)

        layout.addWidget(self.include_masks_checkbox)
        layout.addWidget(self.include_patches_checkbox)
        layout.addWidget(self.include_rectangles_checkbox)
        layout.addWidget(self.include_polygons_checkbox)

        group_box.setLayout(layout)
        return group_box

    def update_annotation_type_checkboxes(self):
        """
        Update the state of annotation type checkboxes for semantic segmentation.
        """
        # Enable all annotation types for semantic segmentation
        self.include_masks_checkbox.setChecked(True)
        self.include_masks_checkbox.setEnabled(True)
        
        # Enable vector annotation types - they will be rasterized
        self.include_patches_checkbox.setChecked(True)
        self.include_patches_checkbox.setEnabled(True)
        self.include_rectangles_checkbox.setChecked(True)
        self.include_rectangles_checkbox.setEnabled(True)
        self.include_polygons_checkbox.setChecked(True)
        self.include_polygons_checkbox.setEnabled(True)

        # Enable negative sample options for semantic segmentation
        self.include_negatives_radio.setEnabled(True)
        self.exclude_negatives_radio.setEnabled(True)

    def get_mask_annotations(self):
        """
        Get all mask annotations from the current images.
        
        Returns:
            list: List of MaskAnnotation objects
        """
        mask_annotations = []
        
        # Get images based on selection
        if self.filtered_images_radio.isChecked():
            images = self.image_window.table_model.filtered_paths
        else:
            images = self.image_window.raster_manager.image_paths
            
        # Get mask annotation for each image
        for image_path in images:
            raster = self.image_window.raster_manager.get_raster(image_path)
            if raster and raster.mask_annotation:
                mask_annotations.append(raster.mask_annotation)
                    
        return mask_annotations

    def mask_contains_selected_labels(self, mask_annotation):
        """
        Check if a mask annotation contains any of the selected labels.
        
        Args:
            mask_annotation (MaskAnnotation): The mask annotation to check
            
        Returns:
            bool: True if mask contains selected labels, False otherwise
        """
        if not self.selected_labels:
            return False
            
        # Get class statistics for this mask
        class_stats = mask_annotation.get_class_statistics()
        
        # Check if any selected label is present in the mask
        for label_code in self.selected_labels:
            if label_code in class_stats and class_stats[label_code]['pixel_count'] > 0:
                return True
                
        return False

    def filter_annotations(self):
        """
        Filter both mask and vector annotations based on the selected types and labels.
        Override base class method to work with both mask and vector annotations.

        Returns:
            list: List of filtered annotations (both mask and vector types).
        """
        annotations = []
        
        # Get and filter MASK annotations if selected
        if self.include_masks_checkbox.isChecked():
            mask_annotations = self.get_mask_annotations()  # Gets ALL masks
            # FIX: This is the correct place to filter masks by selected labels
            for mask in mask_annotations:
                if self.mask_contains_selected_labels(mask):
                    annotations.append(mask)

        # Get and filter VECTOR annotations based on selected types
        # This re-implements the logic from the base class to work here
        vector_annotations = []
        all_vectors = list(self.annotation_window.annotations_dict.values())
        
        # Filter by annotation type
        if self.include_patches_checkbox.isChecked():
            vector_annotations.extend([a for a in all_vectors if a.__class__.__name__ == 'PatchAnnotation'])
        if self.include_rectangles_checkbox.isChecked():
            vector_annotations.extend([a for a in all_vectors if a.__class__.__name__ == 'RectangleAnnotation'])
        if self.include_polygons_checkbox.isChecked():
            vector_annotations.extend([a for a in all_vectors if a.__class__.__name__ == 'PolygonAnnotation'])
        
        # Filter vector annotations by selected labels
        filtered_vectors = [a for a in vector_annotations if a.label.short_label_code in self.selected_labels]

        # Filter by image source (all vs. filtered)
        if self.filtered_images_radio.isChecked():
            filtered_image_paths = set(self.image_window.table_model.filtered_paths)
            filtered_vectors = [a for a in filtered_vectors if a.image_path in filtered_image_paths]

        annotations.extend(filtered_vectors)
            
        # Return a list of unique annotations
        return list({anno.id: anno for anno in annotations}.values())

    def populate_class_filter_list(self):
        """
        Populate the class filter list with labels from both mask and vector annotations.
        Override base class method to work with both mask and vector annotations for semantic segmentation.
        """
        # Set the row count to 0
        self.label_counts_table.setRowCount(0)

        label_counts = {}  # Number of annotations/masks containing each label
        label_image_counts = {}  # Set of unique images containing each label

        # Get all possible annotations (vector and mask)
        all_annotations = list(self.annotation_window.annotations_dict.values())
        all_annotations.extend(self.get_mask_annotations())
        
        # Create a set of unique annotations to avoid double counting if a mask is in both lists
        unique_annotations = {anno.id: anno for anno in all_annotations}.values()

        for annotation in unique_annotations:
            image_path = annotation.image_path
            
            # Handle MaskAnnotation: iterate through its internal labels
            if annotation.__class__.__name__ == 'MaskAnnotation':
                class_stats = annotation.get_class_statistics()
                for label_code, stats in class_stats.items():
                    if stats.get('pixel_count', 0) > 0:
                        if label_code in label_counts:
                            label_counts[label_code] += 1
                            label_image_counts[label_code].add(image_path)
                        else:
                            label_counts[label_code] = 1
                            label_image_counts[label_code] = {image_path}
            
            # Handle Vector Annotations
            else:
                label_code = annotation.label.short_label_code
                if label_code != 'Review':
                    if label_code in label_counts:
                        label_counts[label_code] += 1
                        label_image_counts[label_code].add(image_path)
                    else:
                        label_counts[label_code] = 1
                        label_image_counts[label_code] = {image_path}

        # If no annotations are found, populate with all available project labels
        if not label_counts:
            for label in self.main_window.label_window.labels:
                if label.short_label_code != 'Review':
                    label_counts[label.short_label_code] = 0
                    label_image_counts[label.short_label_code] = set()

        # Sort and populate the table
        sorted_label_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)
        
        self.label_counts_table.setColumnCount(7)
        self.label_counts_table.setHorizontalHeaderLabels(["Include", 
                                                           "Label", 
                                                           "Annotations", 
                                                           "Train", 
                                                           "Val", 
                                                           "Test", 
                                                           "Images"])

        row = 0
        for label, count in sorted_label_counts:
            include_checkbox = QCheckBox()
            include_checkbox.setChecked(True)
            self.label_counts_table.insertRow(row)
            self.label_counts_table.setCellWidget(row, 0, include_checkbox)
            self.label_counts_table.setItem(row, 1, QTableWidgetItem(label))
            self.label_counts_table.setItem(row, 2, QTableWidgetItem(str(count)))
            self.label_counts_table.setItem(row, 3, QTableWidgetItem("0"))
            self.label_counts_table.setItem(row, 4, QTableWidgetItem("0"))
            self.label_counts_table.setItem(row, 5, QTableWidgetItem("0"))
            self.label_counts_table.setItem(row, 6, QTableWidgetItem(str(len(label_image_counts.get(label, set())))))
            row += 1

    def update_summary_statistics(self):
        """
        Update the summary statistics for the semantic segmentation dataset.
        This version uses the polymorphic get_class_statistics() method.
        """
        if self.updating_summary_statistics:
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self, "Updating Summary Statistics")
        progress_bar.show()
        progress_bar.start_progress(100)

        self.updating_summary_statistics = True

        # Selected labels based on user's selection
        self.selected_labels = []
        for row in range(self.label_counts_table.rowCount()):
            include_checkbox = self.label_counts_table.cellWidget(row, 0)
            if include_checkbox.isChecked():
                label = self.label_counts_table.item(row, 1).text()
                self.selected_labels.append(label)

        # Filter annotations based on the selected labels and image options
        self.selected_annotations = self.filter_annotations()
        
        # Split the data by images
        self.split_data()

        # Split the data by annotations
        self.determine_splits()

        # Update the label counts table
        for row in range(self.label_counts_table.rowCount()):
            include_checkbox = self.label_counts_table.cellWidget(row, 0)
            label = self.label_counts_table.item(row, 1).text()
            
            # Use the new polymorphic method directly
            total_count = sum(1 for anno in self.selected_annotations if label in anno.get_class_statistics())
            
            if include_checkbox.isChecked():
                train_count = sum(1 for anno in self.train_annotations if label in anno.get_class_statistics())
                val_count = sum(1 for anno in self.val_annotations if label in anno.get_class_statistics())
                test_count = sum(1 for anno in self.test_annotations if label in anno.get_class_statistics())
            else:
                train_count = 0
                val_count = 0
                test_count = 0

            self.label_counts_table.item(row, 2).setText(str(total_count))
            self.label_counts_table.item(row, 3).setText(str(train_count))
            self.label_counts_table.item(row, 4).setText(str(val_count))
            self.label_counts_table.item(row, 5).setText(str(test_count))

            # Set cell colors based on the counts and ratios
            red = QColor(255, 0, 0)
            green = QColor(0, 255, 0)

            if include_checkbox.isChecked():
                self.set_cell_color(row, 3, red if train_count == 0 and self.train_ratio > 0 else green)
                self.set_cell_color(row, 4, red if val_count == 0 and self.val_ratio > 0 else green)
                self.set_cell_color(row, 5, red if test_count == 0 and self.test_ratio > 0 else green)
            else:
                self.set_cell_color(row, 3, green)
                self.set_cell_color(row, 4, green)
                self.set_cell_color(row, 5, green)

        self.ready_status = self.check_label_distribution()
        self.split_status = abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-9
        self.ready_label.setText("✅ Ready" if (self.ready_status and self.split_status) else "❌ Not Ready")

        self.updating_summary_statistics = False

        # Restore the cursor to the default cursor
        QApplication.restoreOverrideCursor()
        progress_bar.finish_progress()
        progress_bar.close()
        progress_bar = None

    def check_label_distribution(self):
        """
        Check the label distribution in the splits for mask annotations.
        Override base class method to work with mask annotations.
        
        Returns:
            bool: True if all labels are present in all splits and split config is allowed, False otherwise.
        """
        # Get the ratios from the spinboxes
        train_ratio = self.train_ratio_spinbox.value()
        val_ratio = self.val_ratio_spinbox.value()
        test_ratio = self.test_ratio_spinbox.value()
    
        # Only allow these split combinations:
        # - Train only
        # - Test only
        # - Train/Val
        # - Train/Val/Test
        
        allowed = False

        # Train only
        if train_ratio == 1.0 and val_ratio == 0 and test_ratio == 0:
            allowed = True
            
        # Test only
        elif train_ratio == 0 and val_ratio == 0 and test_ratio == 1.0:
            allowed = True
            
        # Train/Val
        elif train_ratio > 0 and val_ratio > 0 and test_ratio == 0:
            if abs(train_ratio + val_ratio - 1.0) < 1e-9:
                allowed = True
                
        # Train/Val/Test
        elif train_ratio > 0 and val_ratio > 0 and test_ratio > 0:
            if abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9:
                allowed = True

        if not allowed:
            return False
    
        # Initialize dictionaries to store label counts for each split
        train_label_counts = {}
        val_label_counts = {}
        test_label_counts = {}
    
        # Count mask annotations containing each label in each split
        for annotation in self.train_annotations:
            class_stats = annotation.get_class_statistics()
            for label, stats in class_stats.items():
                if stats['pixel_count'] > 0:
                    train_label_counts[label] = train_label_counts.get(label, 0) + 1
    
        for annotation in self.val_annotations:
            class_stats = annotation.get_class_statistics()
            for label, stats in class_stats.items():
                if stats['pixel_count'] > 0:
                    val_label_counts[label] = val_label_counts.get(label, 0) + 1
    
        for annotation in self.test_annotations:
            class_stats = annotation.get_class_statistics()
            for label, stats in class_stats.items():
                if stats['pixel_count'] > 0:
                    test_label_counts[label] = test_label_counts.get(label, 0) + 1
    
        # Check the conditions for each split
        for label in self.selected_labels:
            if train_ratio > 0 and (label not in train_label_counts or train_label_counts[label] == 0):
                return False
            if val_ratio > 0 and (label not in val_label_counts or val_label_counts[label] == 0):
                return False
            if test_ratio > 0 and (label not in test_label_counts or test_label_counts[label] == 0):
                return False
    
        # Additional checks to ensure no empty splits
        if train_ratio > 0 and len(self.train_annotations) == 0:
            return False
        if val_ratio > 0 and len(self.val_annotations) == 0:
            return False
        if test_ratio > 0 and len(self.test_annotations) == 0:
            return False
    
        return True

    def create_dataset(self, output_dir_path):
        """
        Create the semantic segmentation dataset in YOLO format.
        
        Args:
            output_dir_path (str): Path to the output directory.
        """
        # Create the yaml file
        yaml_path = os.path.join(output_dir_path, 'data.yaml')

        # Create the train, val, and test directories (using 'valid' to match YOLO convention)
        train_dir = os.path.join(output_dir_path, 'train')
        val_dir = os.path.join(output_dir_path, 'valid')
        test_dir = os.path.join(output_dir_path, 'test')
        names = self.selected_labels
        num_classes = len(self.selected_labels)

        # Create dictionary of class names with numeric keys (0 is background, so classes start at 1)
        names_dict = {i + 1: name for i, name in enumerate(names)}
        # Add background class
        names_dict[0] = 'background'

        # Create colors dictionary mapping class indices to RGB values (no alpha)
        labels_dict = {label.short_label_code: label for label in self.main_window.label_window.labels}
        colors = {}
        for index, name in names_dict.items():
            if name == 'background':
                colors[index] = [0, 0, 0]  # Default black for background
            else:
                label = labels_dict.get(name)
                if label:
                    colors[index] = [label.color.red(), label.color.green(), label.color.blue()]

        # Define the data as a dictionary with absolute paths
        data = {
            'path': output_dir_path,
            'train': train_dir,
            'val': val_dir,
            'test': test_dir,
            'nc': num_classes + 1,  # +1 for background class
            'names': names_dict,  # Dictionary mapping from indices to class names
            # 'colors': colors  # Dictionary mapping from indices to RGB color lists
        }

        # Write the data to the YAML file
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        # Create the train, val, and test directories with images and labels subdirectories
        os.makedirs(f"{train_dir}/images", exist_ok=True)
        os.makedirs(f"{train_dir}/labels", exist_ok=True)
        os.makedirs(f"{val_dir}/images", exist_ok=True)
        os.makedirs(f"{val_dir}/labels", exist_ok=True)
        os.makedirs(f"{test_dir}/images", exist_ok=True)
        os.makedirs(f"{test_dir}/labels", exist_ok=True)

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self, "Creating Semantic Segmentation Dataset")
        progress_bar.show()

        try:
            # Process each split
            self.process_annotations(self.train_annotations, train_dir, "Training")
            self.process_annotations(self.val_annotations, val_dir, "Validation")
            self.process_annotations(self.test_annotations, test_dir, "Testing")

        finally:
            # Restore the cursor to the default cursor
            QApplication.restoreOverrideCursor()
            progress_bar.finish_progress()
            progress_bar.close()
            gc.collect()

    def process_mask_annotations(self, annotations, images_dir, labels_dir, progress_bar, image_paths):
        """
        Process both mask and vector annotations for a specific split in YOLO format.
        
        Args:
            annotations (list): List of annotation objects (both mask and vector types) for this split
            images_dir (str): Directory to save images
            labels_dir (str): Directory to save label masks
            progress_bar: Progress bar object
            image_paths (list): All image paths for this split (including negatives)
        """
        # Separate mask and vector annotations
        mask_annotations = [ann for ann in annotations if ann.__class__.__name__ == 'MaskAnnotation']
        vector_annotation_types = ['PatchAnnotation', 'RectangleAnnotation', 'PolygonAnnotation']
        vector_annotations = [ann for ann in annotations if ann.__class__.__name__ in vector_annotation_types]
        
        # Create mappings by image path
        image_to_mask = {ann.image_path: ann for ann in mask_annotations}
        
        # Group vector annotations by image path
        image_to_vectors = {}
        for ann in vector_annotations:
            if ann.image_path not in image_to_vectors:
                image_to_vectors[ann.image_path] = []
            image_to_vectors[ann.image_path].append(ann)
        
        for i, image_path in enumerate(image_paths):
            try:
                # Copy original image
                image_filename = os.path.basename(image_path)
                image_name, image_ext = os.path.splitext(image_filename)
                
                # Copy image to images directory
                image_output_path = os.path.join(images_dir, image_filename)
                if not os.path.exists(image_output_path):
                    shutil.copy2(image_path, image_output_path)
                
                # Create semantic segmentation mask
                mask_output_path = os.path.join(labels_dir, f"{image_name}.png")
                
                # Combine mask and vector annotations for this image
                self.create_combined_semantic_mask(
                    image_path,
                    image_to_mask.get(image_path),
                    image_to_vectors.get(image_path, []),
                    mask_output_path
                )
                
                # Update progress
                progress_bar.update_progress()
                
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")

    def create_semantic_mask(self, mask_annotation, output_path):
        """
        Create a semantic segmentation mask from a MaskAnnotation in YOLO format.
        Single channel PNG with uint8 values where 0 is background.
        
        Args:
            mask_annotation (MaskAnnotation): The mask annotation to process
            output_path (str): Path to save the mask
        """
        # Get the mask data
        mask_data = mask_annotation.mask_data.copy()
        
        # Create a mapping from class IDs to label indices
        label_to_index = {}
        for i, label in enumerate(self.selected_labels):
            # Find the class ID for this label in the mask annotation
            for class_id, label_obj in mask_annotation.class_id_to_label_map.items():
                if label_obj.short_label_code == label:
                    label_to_index[class_id] = i + 1  # +1 because 0 is background
                    break
        
        # Create output mask with background as 0
        output_mask = np.zeros_like(mask_data, dtype=np.uint8)
        
        # Map class IDs to label indices
        for class_id, label_index in label_to_index.items():
            # Handle both locked and unlocked pixels
            class_mask = (mask_data == class_id) | (mask_data == class_id + mask_annotation.LOCK_BIT)
            output_mask[class_mask] = label_index
        
        # Save as single channel PNG (uint8)
        mask_image = Image.fromarray(output_mask, mode='L')
        mask_image.save(output_path)

    def create_combined_semantic_mask(self, image_path, mask_annotation, vector_annotations, output_path):
        """
        Create a combined semantic segmentation mask from both mask and vector annotations.
        
        Args:
            image_path (str): Path to the source image
            mask_annotation (MaskAnnotation or None): Mask annotation for this image
            vector_annotations (list): List of vector annotations for this image
            output_path (str): Path to save the combined mask
        """
        try:
            # Get image dimensions
            from coralnet_toolbox.utilities import rasterio_open
            with rasterio_open(image_path) as src:
                height, width = src.shape
            
            # Start with empty mask (all background)
            combined_mask = np.zeros((height, width), dtype=np.uint8)
            
            # If we have a mask annotation, use it as the base
            if mask_annotation is not None:
                # Get the mask data and convert to semantic format
                mask_data = mask_annotation.mask_data.copy()
                
                # Create mapping from class IDs to label indices
                label_to_index = {}
                for i, label in enumerate(self.selected_labels):
                    # Find the class ID for this label in the mask annotation
                    for class_id, label_obj in mask_annotation.class_id_to_label_map.items():
                        if label_obj.short_label_code == label:
                            label_to_index[class_id] = i + 1  # +1 because 0 is background
                            break
                
                # Map class IDs to label indices
                for class_id, label_index in label_to_index.items():
                    # Handle both locked and unlocked pixels
                    class_mask = ((mask_data == class_id) |
                                  (mask_data == class_id + mask_annotation.LOCK_BIT))
                    combined_mask[class_mask] = label_index
            
            # If we have vector annotations, rasterize and overlay them
            if vector_annotations:
                vector_mask = self.rasterize_vector_annotations(vector_annotations, image_path)
                # Overlay vector mask onto combined mask (vector annotations take priority)
                combined_mask = np.where(vector_mask > 0, vector_mask, combined_mask)
            
            # Save as single channel PNG
            mask_image = Image.fromarray(combined_mask, mode='L')
            mask_image.save(output_path)
            
        except Exception as e:
            print(f"Error creating combined semantic mask for {image_path}: {e}")
            # Create empty mask as fallback
            self.create_empty_mask(image_path, output_path)
    
    def create_empty_mask(self, image_path, output_path):
        """
        Create an empty semantic segmentation mask (all background) for images without annotations.
        
        Args:
            image_path (str): Path to the original image to get dimensions
            output_path (str): Path to save the empty mask
        """
        try:
            # Get image dimensions
            from coralnet_toolbox.utilities import rasterio_open
            with rasterio_open(image_path) as src:
                height, width = src.shape
            
            # Create empty mask (all zeros = background)
            empty_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Save as single channel PNG
            mask_image = Image.fromarray(empty_mask, mode='L')
            mask_image.save(output_path)
            
        except Exception as e:
            print(f"Error creating empty mask for {image_path}: {e}")

    def rasterize_vector_annotations(self, vector_annotations, image_path):
        """
        Rasterize vector annotations (patches, rectangles, polygons) into a semantic mask.
        
        Args:
            vector_annotations (list): List of vector annotation objects
            image_path (str): Path to the image for getting dimensions
            
        Returns:
            np.ndarray: Semantic mask with rasterized annotations
        """
        try:
            # Get image dimensions
            from coralnet_toolbox.utilities import rasterio_open
            with rasterio_open(image_path) as src:
                height, width = src.shape
            
            # Create empty mask (all zeros = background)
            output_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Group annotations by label
            annotations_by_label = {}
            for annotation in vector_annotations:
                label_code = annotation.label.short_label_code
                if label_code not in annotations_by_label:
                    annotations_by_label[label_code] = []
                annotations_by_label[label_code].append(annotation)
            
            # Rasterize each label group
            for label_code, label_annotations in annotations_by_label.items():
                # Find the class index for this label (1-based, 0 is background)
                if label_code in self.selected_labels:
                    class_index = self.selected_labels.index(label_code) + 1
                    
                    # Convert annotations to geometries
                    geometries = []
                    for annotation in label_annotations:
                        geom = self._annotation_to_geometry(annotation)
                        if geom is not None:
                            geometries.append(geom)
                    
                    if geometries:
                        # Rasterize geometries for this class
                        class_mask = rasterize(
                            [(geom, class_index) for geom in geometries],
                            out_shape=(height, width),
                            fill=0,
                            dtype=np.uint8
                        )
                        
                        # Overlay this class mask onto output mask
                        output_mask = np.where(class_mask > 0, class_mask, output_mask)
            
            return output_mask
            
        except Exception as e:
            print(f"Error rasterizing vector annotations for {image_path}: {e}")
            # Return empty mask on error
            return np.zeros((height, width), dtype=np.uint8)

    def _annotation_to_geometry(self, annotation):
        """
        Convert an annotation object to a shapely geometry.
        
        Args:
            annotation: Annotation object (patch, rectangle, or polygon)
            
        Returns:
            shapely geometry or None
        """
        try:
            polygon = annotation.get_polygon()
            points = [(polygon.at(i).x(), polygon.at(i).y()) for i in range(polygon.count())]
            if len(points) >= 3:
                return Polygon(points)
        except Exception as e:
            print(f"Error converting annotation to geometry: {e}")
            
        return None

    def process_annotations(self, annotations, split_dir, split):
        """
        Process annotations for semantic segmentation export in YOLO format.
        
        Args:
            annotations (list): List of MaskAnnotation objects for this split
            split_dir (str): Directory for this split  
            split (str): Split name (e.g., "Training", "Validation", "Testing")
        """
        # Determine the full list of images for this split (including negatives)
        if split == "Training":
            image_paths = self.train_images
        elif split == "Validation":
            image_paths = self.val_images
        elif split == "Testing":
            image_paths = self.test_images
        else:
            image_paths = []

        if not image_paths:
            return

        # Set up progress bar
        progress_bar = ProgressBar(self, title=f"Creating {split} Dataset")
        progress_bar.show()
        progress_bar.start_progress(len(image_paths))

        try:
            # Create images and labels directories
            images_dir = os.path.join(split_dir, 'images')
            labels_dir = os.path.join(split_dir, 'labels')
            
            # Process all images in this split
            self.process_mask_annotations(annotations, images_dir, labels_dir, progress_bar, image_paths)
            
        finally:
            progress_bar.stop_progress()
            progress_bar.close()