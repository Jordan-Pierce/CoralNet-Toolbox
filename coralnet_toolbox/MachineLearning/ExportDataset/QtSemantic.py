import warnings
import os
import gc
import random 
import shutil
import numpy as np
from PIL import Image

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
        self.setWindowIcon(get_icon("coral"))

    def setup_info_layout(self):
        """Setup the info layout"""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text
        info_text = ("Export semantic segmentation masks from MaskAnnotations. "
                     "This exports pixel-level class labels for semantic segmentation training. "
                     "The table shows which class labels are present in each mask annotation.")
        info_label = QLabel(info_text)

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def create_annotation_layout(self):
        """Creates the annotation type checkboxes layout group box - overridden for semantic."""
        group_box = QGroupBox("Annotation Types")
        layout = QVBoxLayout()

        # For semantic segmentation, we only work with mask annotations
        self.include_masks_checkbox = QCheckBox("Include Mask Annotations")
        self.include_masks_checkbox.setChecked(True)
        self.include_masks_checkbox.setEnabled(True)
        
        # Add placeholders for compatibility with base class
        self.include_patches_checkbox = QCheckBox("Include Patch Annotations")
        self.include_patches_checkbox.setChecked(False)
        self.include_patches_checkbox.setEnabled(False)
        self.include_patches_checkbox.setVisible(False)
        
        self.include_rectangles_checkbox = QCheckBox("Include Rectangle Annotations") 
        self.include_rectangles_checkbox.setChecked(False)
        self.include_rectangles_checkbox.setEnabled(False)
        self.include_rectangles_checkbox.setVisible(False)
        
        self.include_polygons_checkbox = QCheckBox("Include Polygon Annotations")
        self.include_polygons_checkbox.setChecked(False)
        self.include_polygons_checkbox.setEnabled(False)
        self.include_polygons_checkbox.setVisible(False)

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
        # Only mask annotations are relevant for semantic segmentation
        self.include_masks_checkbox.setChecked(True)
        self.include_masks_checkbox.setEnabled(True)
        
        # Disable vector annotation types
        self.include_patches_checkbox.setChecked(False)
        self.include_patches_checkbox.setEnabled(False)
        self.include_rectangles_checkbox.setChecked(False)
        self.include_rectangles_checkbox.setEnabled(False)
        self.include_polygons_checkbox.setChecked(False)
        self.include_polygons_checkbox.setEnabled(False)

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
                mask_annotation = raster.mask_annotation
                
                # Check if this mask contains any of the selected labels
                if self.mask_contains_selected_labels(mask_annotation):
                    mask_annotations.append(mask_annotation)
                    
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
        Filter mask annotations based on the selected labels and image options.
        Override base class method to work with mask annotations.

        Returns:
            list: List of filtered MaskAnnotation objects.
        """
        if not self.include_masks_checkbox.isChecked():
            return []
            
        return self.get_mask_annotations()

    def populate_class_filter_list(self):
        """
        Populate the class filter list with labels and their mask presence/counts.
        Override base class method to work with mask annotations instead of vector annotations.
        """
        # Set the row count to 0
        self.label_counts_table.setRowCount(0)

        label_counts = {}  # Number of masks containing each label
        label_pixel_counts = {}  # Total pixel counts for each label across all masks
        label_image_counts = {}  # Set of images containing each label
        
        # Get all mask annotations
        all_mask_annotations = []
        for image_path in self.image_window.raster_manager.image_paths:
            raster = self.image_window.raster_manager.get_raster(image_path)
            if raster and raster.mask_annotation:
                all_mask_annotations.append(raster.mask_annotation)

        # Count occurrences of each label in mask annotations
        for mask_annotation in all_mask_annotations:
            class_stats = mask_annotation.get_class_statistics()
            image_path = mask_annotation.image_path
            
            for label_code, stats in class_stats.items():
                if label_code != 'Review' and stats['pixel_count'] > 0:
                    # Count masks containing this label
                    if label_code in label_counts:
                        label_counts[label_code] += 1
                        label_pixel_counts[label_code] += stats['pixel_count']
                        label_image_counts[label_code].add(image_path)
                    else:
                        label_counts[label_code] = 1
                        label_pixel_counts[label_code] = stats['pixel_count']
                        label_image_counts[label_code] = {image_path}

        # Sort the labels by their mask counts in descending order
        sorted_label_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)

        # Populate the label counts table with labels and their counts
        self.label_counts_table.setColumnCount(7)
        self.label_counts_table.setHorizontalHeaderLabels(["Include",
                                                           "Label", 
                                                           "Masks",  # Number of masks containing label
                                                           "Train",
                                                           "Val", 
                                                           "Test",
                                                           "Images"])

        # Populate the label counts table with labels and their counts
        row = 0
        for label, mask_count in sorted_label_counts:
            include_checkbox = QCheckBox()
            include_checkbox.setChecked(True)
            self.label_counts_table.insertRow(row)
            self.label_counts_table.setCellWidget(row, 0, include_checkbox)
            self.label_counts_table.setItem(row, 1, QTableWidgetItem(label))
            self.label_counts_table.setItem(row, 2, QTableWidgetItem(str(mask_count)))
            self.label_counts_table.setItem(row, 3, QTableWidgetItem("0"))
            self.label_counts_table.setItem(row, 4, QTableWidgetItem("0"))
            self.label_counts_table.setItem(row, 5, QTableWidgetItem("0"))
            self.label_counts_table.setItem(row, 6, QTableWidgetItem(str(len(label_image_counts[label]))))

            row += 1

    def split_data(self):
        """
        Split the data by images based on the specified ratios.
        Override base class method to work with mask annotations.
        """
        self.train_ratio = self.train_ratio_spinbox.value()
        self.val_ratio = self.val_ratio_spinbox.value()
        self.test_ratio = self.test_ratio_spinbox.value()

        # Get images, either filtered or all depending on radio button selection
        if self.filtered_images_radio.isChecked():
            images = self.image_window.table_model.filtered_paths
        else:
            images = self.image_window.raster_manager.image_paths

        # If "Exclude Negatives" is checked, only use images that have mask annotations with selected labels
        if self.exclude_negatives_radio.isChecked():
            images_with_selected_labels = set()
            for image_path in images:
                raster = self.image_window.raster_manager.get_raster(image_path)
                if raster and raster.mask_annotation:
                    if self.mask_contains_selected_labels(raster.mask_annotation):
                        images_with_selected_labels.add(image_path)
            images = [img for img in images if img in images_with_selected_labels]

        random.shuffle(images)

        train_split = int(len(images) * self.train_ratio)
        val_split = int(len(images) * (self.train_ratio + self.val_ratio))

        # Initialize splits
        self.train_images = []
        self.val_images = []
        self.test_images = []

        # Assign images to splits based on ratios
        if self.train_ratio > 0:
            self.train_images = images[:train_split]
        if self.val_ratio > 0:
            self.val_images = images[train_split:val_split]
        if self.test_ratio > 0:
            self.test_images = images[val_split:]

    def determine_splits(self):
        """
        Determine the splits for train, validation, and test mask annotations.
        Override base class method to work with mask annotations.
        """
        self.selected_annotations = self.get_mask_annotations()
        
        self.train_annotations = [a for a in self.selected_annotations if a.image_path in self.train_images]
        self.val_annotations = [a for a in self.selected_annotations if a.image_path in self.val_images]
        self.test_annotations = [a for a in self.selected_annotations if a.image_path in self.test_images]

    def update_summary_statistics(self):
        """
        Update the summary statistics for the semantic segmentation dataset.
        Override base class method to work with mask annotations.
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
            
            # Count masks containing this label
            def has_label_in_mask(annotation, label_name):
                stats = annotation.get_class_statistics()
                return label_name in stats and stats[label_name]['pixel_count'] > 0
                
            mask_count = sum(1 for a in self.selected_annotations if has_label_in_mask(a, label))
            
            if include_checkbox.isChecked():
                train_count = sum(1 for a in self.train_annotations if has_label_in_mask(a, label))
                val_count = sum(1 for a in self.val_annotations if has_label_in_mask(a, label))
                test_count = sum(1 for a in self.test_annotations if has_label_in_mask(a, label))
            else:
                train_count = 0
                val_count = 0
                test_count = 0

            self.label_counts_table.item(row, 2).setText(str(mask_count))
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
        Create the semantic segmentation dataset.
        
        Args:
            output_dir_path (str): Path to the output directory.
        """
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self, "Creating Semantic Segmentation Dataset")
        progress_bar.show()

        try:
            # Create split directories
            splits = []
            if self.train_ratio > 0:
                splits.append(('train', self.train_annotations, self.train_images))
            if self.val_ratio > 0:
                splits.append(('val', self.val_annotations, self.val_images))
            if self.test_ratio > 0:
                splits.append(('test', self.test_annotations, self.test_images))

            total_operations = sum(len(annotations) for _, annotations, _ in splits)
            progress_bar.start_progress(total_operations)
            
            operations_completed = 0

            for split_name, annotations, images in splits:
                split_dir = os.path.join(output_dir_path, split_name)
                images_dir = os.path.join(split_dir, 'images')
                masks_dir = os.path.join(split_dir, 'masks')
                
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(masks_dir, exist_ok=True)

                self.process_mask_annotations(annotations, images_dir, masks_dir, progress_bar, operations_completed)
                operations_completed += len(annotations)

        finally:
            # Restore the cursor to the default cursor
            QApplication.restoreOverrideCursor()
            progress_bar.finish_progress()
            progress_bar.close()
            gc.collect()

    def process_mask_annotations(self, annotations, images_dir, masks_dir, progress_bar, operations_completed):
        """
        Process mask annotations for a specific split.
        
        Args:
            annotations (list): List of MaskAnnotation objects
            images_dir (str): Directory to save images
            masks_dir (str): Directory to save masks
            progress_bar: Progress bar object
            operations_completed (int): Number of operations already completed
        """
        for i, mask_annotation in enumerate(annotations):
            try:
                # Copy original image
                image_filename = os.path.basename(mask_annotation.image_path)
                image_name, image_ext = os.path.splitext(image_filename)
                
                # Copy image to images directory
                image_output_path = os.path.join(images_dir, image_filename)
                if not os.path.exists(image_output_path):
                    shutil.copy2(mask_annotation.image_path, image_output_path)
                
                # Create semantic segmentation mask
                mask_output_path = os.path.join(masks_dir, f"{image_name}.png")
                self.create_semantic_mask(mask_annotation, mask_output_path)
                
                # Update progress
                progress_bar.update_progress(operations_completed + i + 1)
                
            except Exception as e:
                print(f"Error processing mask annotation {mask_annotation.image_path}: {e}")

    def create_semantic_mask(self, mask_annotation, output_path):
        """
        Create a semantic segmentation mask from a MaskAnnotation.
        
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
        
        # Save as PNG
        mask_image = Image.fromarray(output_mask, mode='L')
        mask_image.save(output_path)

    def process_annotations(self, annotations, split_dir, split):
        """
        Process annotations for semantic segmentation export.
        This method is required by the base class but redirects to process_mask_annotations.
        
        Args:
            annotations (list): List of MaskAnnotation objects
            split_dir (str): Directory for this split
            split (str): Split name ('train', 'val', 'test')
        """
        images_dir = os.path.join(split_dir, 'images')
        masks_dir = os.path.join(split_dir, 'masks')
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        
        # Create a dummy progress bar for this method
        progress_bar = ProgressBar(self, f"Processing {split} annotations")
        progress_bar.show()
        progress_bar.start_progress(len(annotations))
        
        try:
            self.process_mask_annotations(annotations, images_dir, masks_dir, progress_bar, 0)
        finally:
            progress_bar.finish_progress()
            progress_bar.close()