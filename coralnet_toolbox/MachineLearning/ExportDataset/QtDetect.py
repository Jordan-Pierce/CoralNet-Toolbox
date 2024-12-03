import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import yaml
import shutil

from coralnet_toolbox.MachineLearning.ExportDataset.QtBase import Base
from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class Detect(Base):
    def __init__(self, parent=None):
        super(Detect, self).__init__(parent)
        self.setWindowTitle("Export Detection Dataset")
        self.setWindowIcon(get_icon("coral"))
        
    def update_annotation_type_checkboxes(self):
        """
        Update the state of annotation type checkboxes based on the selected dataset type.
        """
        self.include_patches_checkbox.setChecked(False)
        self.include_patches_checkbox.setEnabled(False)
        self.include_rectangles_checkbox.setChecked(True)
        self.include_rectangles_checkbox.setEnabled(False)
        self.include_polygons_checkbox.setChecked(True)
        self.include_polygons_checkbox.setEnabled(True)

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

        # Define the data as a dictionary
        data = {
            'train': '../train/images',
            'val': '../valid/images',
            'test': '../test/images',
            'nc': num_classes,  
            'names': names  
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

        progress_bar = ProgressBar(self, title=f"Creating {split} Dataset")
        progress_bar.show()
        progress_bar.start_progress(len(image_paths))

        for image_path in image_paths:
            yolo_annotations = []
            image_height, image_width = self.image_window.rasterio_open(image_path).shape
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

        progress_bar.stop_progress()
        progress_bar.close()