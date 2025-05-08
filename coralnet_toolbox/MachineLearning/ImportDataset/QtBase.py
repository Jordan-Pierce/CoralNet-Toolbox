import warnings

import os
import uuid
import yaml
import glob
import random
import shutil
import ujson as json

from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QVBoxLayout, QGroupBox,
                             QLabel, QLineEdit, QDialog, QPushButton, QDialogButtonBox,
                             QGridLayout)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.utilities import rasterio_open
from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Base(QDialog):
    """
    Dialog for importing datasets for object detection and instance segmentation.

    :param main_window: MainWindow object
    :param parent: Parent widget
    """
    def __init__(self, main_window, parent=None):
        super(Base, self).__init__(parent)
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window
        self.image_window = main_window.image_window

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Import Dataset")
        self.resize(500, 300)

        self.layout = QVBoxLayout(self)

        # Setup the info layout
        self.setup_info_layout()
        # Setup the YAML layout
        self.setup_yaml_layout()
        # Self the output layout
        self.setup_output_layout()
        # Setup the buttons layout
        self.setup_buttons_layout()

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        raise NotImplementedError("Subclasses must implement method.")

    def setup_yaml_layout(self):
        """
        Initialize the user interface for the ImportDatasetDialog using a form layout.
        """
        group_box = QGroupBox("Data YAML File")
        layout = QGridLayout()

        # YAML file selection row
        layout.addWidget(QLabel("File:"), 0, 0)
        self.yaml_path_label = QLineEdit()
        self.yaml_path_label.setReadOnly(True)
        self.yaml_path_label.setPlaceholderText("Select data.yaml file...")
        layout.addWidget(self.yaml_path_label, 0, 1)

        self.browse_yaml_button = QPushButton("Browse")
        self.browse_yaml_button.clicked.connect(self.browse_data_yaml)
        layout.addWidget(self.browse_yaml_button, 0, 2)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_output_layout(self):
        """
        """
        # Group for output directory selection
        group_box = QGroupBox("Output Settings")
        layout = QGridLayout()

        # Directory selection row
        layout.addWidget(QLabel("Directory:"), 0, 0)
        self.output_dir_label = QLineEdit()
        self.output_dir_label.setPlaceholderText("Select output directory...")
        layout.addWidget(self.output_dir_label, 0, 1)
        self.browse_output_button = QPushButton("Browse")
        self.browse_output_button.clicked.connect(self.browse_output_dir)
        layout.addWidget(self.browse_output_button, 0, 2)

        # Folder name row
        layout.addWidget(QLabel("Folder Name:"), 1, 0)
        self.output_folder_name = QLineEdit("")
        self.output_folder_name.setPlaceholderText("data")
        layout.addWidget(self.output_folder_name, 1, 1, 1, 2)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_buttons_layout(self):
        """
        """
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def browse_data_yaml(self):
        """
        Browse and select a data.yaml file.
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select data.yaml", "", "YAML Files (*.yaml);;All Files (*)", options=options
        )
        if file_path:
            self.yaml_path_label.setText(file_path)

    def browse_output_dir(self):
        """
        Browse and select an output directory.
        """
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_label.setText(dir_path)

    def accept(self):
        """
        Handle the OK button click event to validate and process the dataset.
        """
        # Perform validation and processing here
        if not self.yaml_path_label.text():
            QMessageBox.warning(self, "Error", "Please select a data.yaml file.")
            return
        if not self.output_dir_label.text():
            QMessageBox.warning(self, "Error", "Please select an output directory.")
            return
        if not self.output_folder_name.text():
            QMessageBox.warning(self, "Error", "Please enter an output folder name.")
            return

        # Call the process_dataset method
        self.process_dataset()

        # If validation passes, call the base class accept method
        super().accept()

    def reject(self):
        """
        Handle the cancel action.
        """
        super().reject()

    def process_dataset(self):
        """
        Process the dataset based on the selected data.yaml file and output directory.
        """
        if not self.yaml_path_label.text():
            QMessageBox.warning(self,
                                "No File Selected",
                                "Please select a data.yaml file.")
            return

        output_folder = os.path.join(self.output_dir_label.text(), self.output_folder_name.text())
        os.makedirs(f"{output_folder}/images", exist_ok=True)

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        with open(self.yaml_path_label.text(), 'r') as file:
            data = yaml.safe_load(file)

        # Get the paths for train, valid, and test images
        dir_path = os.path.dirname(self.yaml_path_label.text())
        class_names = data.get('names', [])

        # Collect all images from the train, valid, and test folders
        image_paths = glob.glob(f"{dir_path}/**/images/*.*", recursive=True)
        label_paths = glob.glob(f"{dir_path}/**/labels/*.txt", recursive=True)

        if not image_paths or not label_paths:
            QMessageBox.warning(self,
                                "No Images or Labels Found",
                                "No images or labels were found in the specified directories.")
            return

        # Check that each label file has a corresponding image file
        image_label_paths = {}
        added_paths = []

        for label_path in label_paths:
            # Create the image path from the label path
            src_image_path = label_path.replace('labels', 'images').replace('.txt', '.jpg')
            if src_image_path in image_paths:
                # Copy the image to the output folder
                image_path = f"{output_folder}/images/{os.path.basename(src_image_path)}"
                shutil.copy(src_image_path, image_path)
                # Reformat paths
                label_path = label_path.replace("\\", "/")
                image_path = image_path.replace("\\", "/")
                # Add to the dict, and add the image to the image window
                image_label_paths[image_path] = label_path
                if self.image_window.add_image(image_path):
                    added_paths.append(image_path)

        # Update filtered images
        self.image_window.filter_images()

        # Set the last added image as the current image if we have any
        if added_paths:
            self.image_window.load_image_by_path(added_paths[-1])

        # Determine the annotation type based on selected radio button
        if self.__class__.__name__ == "Detect":
            annotation_type = 'RectangleAnnotation'
        elif self.__class__.__name__ == "Segment":
            annotation_type = 'PolygonAnnotation'
        else:
            raise ValueError("No annotation type selected")

        # Process the annotations based on the selected type
        progress_bar = ProgressBar(self, title="Importing YOLO Dataset")
        progress_bar.show()
        progress_bar.start_progress(len(image_label_paths))

        try:
            annotations = []

            for image_path, label_path in image_label_paths.items():

                # Read the label file
                image_height, image_width = rasterio_open(image_path).shape

                with open(label_path, 'r') as file:
                    lines = file.readlines()

                for line in lines:
                    if annotation_type == 'RectangleAnnotation':
                        class_id, x_center, y_center, width, height = map(float, line.split())
                        x_center, y_center, width, height = (x_center * image_width,
                                                             y_center * image_height,
                                                             width * image_width,
                                                             height * image_height)

                        top_left = QPointF(x_center - width / 2, y_center - height / 2)
                        bottom_right = QPointF(x_center + width / 2, y_center + height / 2)

                        class_name = class_names[int(class_id)]
                        short_label_code = long_label_code = class_name
                        existing_label = self.main_window.label_window.get_label_by_short_code(short_label_code)

                        if existing_label:
                            color = existing_label.color
                            label_id = existing_label.id
                        else:
                            label_id = str(uuid.uuid4())
                            color = QColor(random.randint(0, 255),
                                           random.randint(0, 255),
                                           random.randint(0, 255))

                            self.main_window.label_window.add_label_if_not_exists(short_label_code,
                                                                                  long_label_code,
                                                                                  color,
                                                                                  label_id)

                        annotation = RectangleAnnotation(top_left,
                                                         bottom_right,
                                                         short_label_code,
                                                         long_label_code,
                                                         color,
                                                         image_path,
                                                         label_id,
                                                         self.main_window.get_transparency_value())

                    else:
                        class_id, *points = map(float, line.split())
                        points = [QPointF(x * image_width, y * image_height) for x, y in zip(points[::2], points[1::2])]

                        class_name = class_names[int(class_id)]
                        short_label_code = long_label_code = class_name
                        existing_label = self.main_window.label_window.get_label_by_short_code(short_label_code)

                        if existing_label:
                            color = existing_label.color
                            label_id = existing_label.id
                        else:
                            label_id = str(uuid.uuid4())
                            color = QColor(random.randint(0, 255),
                                           random.randint(0, 255),
                                           random.randint(0, 255))

                            self.main_window.label_window.add_label_if_not_exists(short_label_code,
                                                                                  long_label_code,
                                                                                  color,
                                                                                  label_id)

                        annotation = PolygonAnnotation(points,
                                                       short_label_code,
                                                       long_label_code,
                                                       color,
                                                       image_path,
                                                       label_id,
                                                       self.main_window.get_transparency_value())

                    # Add the annotation to the list for export
                    annotations.append(annotation)

                    # Add annotation to the dict
                    self.annotation_window.add_annotation_to_dict(annotation)

                    # Update the progress bar
                    progress_bar.update_progress()

                # Update the image window's image annotations
                self.image_window.update_image_annotations(image_path)

            # Load the annotations for current image
            self.annotation_window.load_annotations()

            # Export annotations as JSON in output
            self.export_annotations(annotations, output_folder)

        except Exception as e:
            QMessageBox.warning(self,
                                "Error Importing Dataset",
                                f"An error occurred while importing the dataset: {str(e)}")

        # Stop the progress bar
        progress_bar.stop_progress()
        progress_bar.close()

        # Make cursor normal
        QApplication.restoreOverrideCursor()

        QMessageBox.information(self,
                                "Dataset Imported",
                                "Dataset has been successfully imported.")

    def export_annotations(self, annotations, output_dir):
        """
        Export the annotations as a JSON file in the specified output directory.

        :param annotations: List of annotations to export
        :param output_dir: Path to the output directory
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)

        progress_bar = ProgressBar(self.annotation_window, title="Exporting Annotations")
        progress_bar.show()
        progress_bar.start_progress(len(annotations))

        export_dict = {}

        try:
            for annotation in annotations:
                image_path = annotation.image_path
                if image_path not in export_dict:
                    export_dict[image_path] = []

                # Convert annotation to dictionary based on its type
                if isinstance(annotation, PatchAnnotation):
                    annotation_dict = {
                        'type': 'PatchAnnotation',
                        **annotation.to_dict()
                    }
                elif isinstance(annotation, PolygonAnnotation):
                    annotation_dict = {
                        'type': 'PolygonAnnotation',
                        **annotation.to_dict()
                    }
                elif isinstance(annotation, RectangleAnnotation):
                    annotation_dict = {
                        'type': 'RectangleAnnotation',
                        **annotation.to_dict()
                    }
                else:
                    raise ValueError(f"Unknown annotation type: {type(annotation)}")

                export_dict[image_path].append(annotation_dict)
                progress_bar.update_progress()

            with open(f"{output_dir}/annotations.json", 'w') as file:
                json.dump(export_dict, file, indent=4)
                file.flush()

        except Exception as e:
            QMessageBox.warning(self,
                                "Error Exporting Annotations",
                                f"An error occurred while exporting the annotations: {str(e)}")

        progress_bar.stop_progress()
        progress_bar.close()

        # Make the cursor normal again
        QApplication.restoreOverrideCursor()
