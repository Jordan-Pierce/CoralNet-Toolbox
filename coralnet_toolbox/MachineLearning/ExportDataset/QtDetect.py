import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import yaml
import shutil

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QGroupBox, QVBoxLayout, QLabel, QApplication)

from coralnet_toolbox.MachineLearning.ExportDataset.QtBase import Base
from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.utilities import rasterio_open
from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Detect(Base):
    def __init__(self, parent=None):
        super(Detect, self).__init__(parent)
        self.setWindowTitle("Export Detection Dataset")
        self.setWindowIcon(get_icon("coral"))

    def setup_info_layout(self):
        """Setup the info layout"""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_text = "Export Rectangles and Polygons to create a YOLO-formatted Detection dataset."
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
        self.include_patches_checkbox.setChecked(False)
        self.include_patches_checkbox.setEnabled(True)  # Enable patches for detection
        self.include_rectangles_checkbox.setChecked(True)
        self.include_rectangles_checkbox.setEnabled(True)  # Enable user to uncheck rectangles if desired
        self.include_polygons_checkbox.setChecked(True)
        self.include_polygons_checkbox.setEnabled(True)  # Already enabled
        
    def create_dataset(self, output_dir_path):
        """
        Create an object detection dataset.

        Args:
            output_dir_path (str): Path to the output directory.
        """
        # Create the yaml file
        yaml_path = os.path.join(output_dir_path, 'data.yaml')

        # Create the train, val, and test directories
        train_dir = os.path.join(output_dir_path, 'train')
        val_dir = os.path.join(output_dir_path, 'valid')
        test_dir = os.path.join(output_dir_path, 'test')
        names = self.selected_labels
        num_classes = len(self.selected_labels)

        # Create dictionary of class names with numeric keys
        names_dict = {i: name for i, name in enumerate(names)}

        # Define the data as a dictionary with absolute paths
        data = {
            'path': output_dir_path,
            'train': train_dir,
            'val': val_dir,
            'test': test_dir,
            'nc': num_classes,
            'names': list(range(num_classes)),  # List of numeric indices
            'names': names_dict  # Dictionary mapping from indices to class names
        }

        # Write the data to the YAML file
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        # Create the train, val, and test directories
        os.makedirs(f"{train_dir}/images", exist_ok=True)
        os.makedirs(f"{train_dir}/labels", exist_ok=True)
        os.makedirs(f"{val_dir}/images", exist_ok=True)
        os.makedirs(f"{val_dir}/labels", exist_ok=True)
        os.makedirs(f"{test_dir}/images", exist_ok=True)
        os.makedirs(f"{test_dir}/labels", exist_ok=True)

        self.process_annotations(self.train_annotations, train_dir, "Training")
        self.process_annotations(self.val_annotations, val_dir, "Validation")
        self.process_annotations(self.test_annotations, test_dir, "Testing")

    def process_annotations(self, annotations, split_dir, split):
        """
        Process and save detection annotations.

        Args:
            annotations (list): List of annotations.
            split_dir (str): Path to the split directory.
            split (str): Split name (e.g., "Training", "Validation", "Testing").
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

        for image_path in image_paths:
            yolo_annotations = []
            image_height, image_width = rasterio_open(image_path).shape
            image_annotations = [a for a in annotations if a.image_path == image_path]

            for image_annotation in image_annotations:
                class_label, annotation = image_annotation.to_yolo_detection(image_width, image_height)
                class_number = self.selected_labels.index(class_label)
                yolo_annotations.append(f"{class_number} {annotation}")

            # Save the annotations to a text file
            file_ext = image_path.split(".")[-1]
            text_file = os.path.basename(image_path).replace(f".{file_ext}", ".txt")
            text_path = os.path.join(f"{split_dir}/labels", text_file)

            # Write the annotations to the text file
            with open(text_path, 'w') as f:
                for annotation in yolo_annotations:
                    f.write(annotation + '\n')

            # Copy the image to the split directory
            shutil.copy(image_path, f"{split_dir}/images/{os.path.basename(image_path)}")

            progress_bar.update_progress()

        # Reset cursor
        QApplication.restoreOverrideCursor()
        progress_bar.stop_progress()
        progress_bar.close()
