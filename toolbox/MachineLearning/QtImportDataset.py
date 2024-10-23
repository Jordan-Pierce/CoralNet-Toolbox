import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import uuid
import yaml
import glob
import json
import os
import random
import shutil

from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QVBoxLayout,
                             QLabel, QLineEdit, QDialog, QHBoxLayout, QPushButton, QDialogButtonBox, QGroupBox,
                             QButtonGroup, QRadioButton, QGridLayout)

from toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

from toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ImportDatasetDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super(ImportDatasetDialog, self).__init__(parent)
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window

        self.setWindowTitle("Import Dataset")
        self.resize(500, 300)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Radio buttons for Object Detection and Instance Segmentation
        detection_type_group = QGroupBox("Detection Type")
        detection_type_layout = QHBoxLayout()
        self.object_detection_radio = QRadioButton("Object Detection")
        self.instance_segmentation_radio = QRadioButton("Instance Segmentation")
        self.detection_type_group = QButtonGroup()
        self.detection_type_group.addButton(self.object_detection_radio)
        self.detection_type_group.addButton(self.instance_segmentation_radio)
        self.object_detection_radio.setChecked(True)  # Set default selection

        detection_type_layout.addWidget(self.object_detection_radio)
        detection_type_layout.addWidget(self.instance_segmentation_radio)
        detection_type_group.setLayout(detection_type_layout)
        main_layout.addWidget(detection_type_group)

        # Group for data.yaml file selection
        yaml_group = QGroupBox("Data YAML File")
        yaml_layout = QGridLayout()
        yaml_group.setLayout(yaml_layout)

        self.yaml_path_label = QLineEdit()
        self.yaml_path_label.setReadOnly(True)
        self.yaml_path_label.setPlaceholderText("Select data.yaml file...")
        self.browse_yaml_button = QPushButton("Browse")
        self.browse_yaml_button.clicked.connect(self.browse_data_yaml)

        yaml_layout.addWidget(QLabel("Path:"), 0, 0)
        yaml_layout.addWidget(self.yaml_path_label, 0, 1)
        yaml_layout.addWidget(self.browse_yaml_button, 0, 2)

        main_layout.addWidget(yaml_group)

        # Group for output directory selection
        output_group = QGroupBox("Output Settings")
        output_layout = QGridLayout()
        output_group.setLayout(output_layout)

        self.output_dir_label = QLineEdit()
        self.output_dir_label.setPlaceholderText("Select output directory...")
        self.browse_output_button = QPushButton("Browse")
        self.browse_output_button.clicked.connect(self.browse_output_dir)

        self.output_folder_name = QLineEdit("")
        self.output_folder_name.setPlaceholderText("data")

        output_layout.addWidget(QLabel("Directory:"), 0, 0)
        output_layout.addWidget(self.output_dir_label, 0, 1)
        output_layout.addWidget(self.browse_output_button, 0, 2)
        output_layout.addWidget(QLabel("Folder Name:"), 1, 0)
        output_layout.addWidget(self.output_folder_name, 1, 1, 1, 2)

        main_layout.addWidget(output_group)

        # Accept and Cancel buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)

        self.setLayout(main_layout)

    def browse_data_yaml(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select data.yaml", "", "YAML Files (*.yaml);;All Files (*)", options=options
        )
        if file_path:
            self.yaml_path_label.setText(file_path)

    def browse_output_dir(self):
        options = QFileDialog.Options()
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_label.setText(dir_path)

    def accept(self):
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
        # Handle cancel action if needed
        super().reject()

    def process_dataset(self):
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
                self.main_window.image_window.add_image(image_path)

        # Update filtered images
        self.main_window.image_window.filter_images()
        # Set the last image as the current image
        current_image = self.main_window.image_window.image_paths[-1]
        self.main_window.image_window.load_image_by_path(current_image)

        # Determine the annotation type based on selected radio button
        if self.object_detection_radio.isChecked():
            annotation_type = 'RectangleAnnotation'
        elif self.instance_segmentation_radio.isChecked():
            annotation_type = 'PolygonAnnotation'
        else:
            raise ValueError("No annotation type selected")

        # Process the annotations based on the selected type
        progress_bar = ProgressBar(self, title=f"Importing YOLO Dataset")
        progress_bar.show()
        progress_bar.start_progress(len(image_label_paths))

        try:
            annotations = []

            for image_path, label_path in image_label_paths.items():

                # Read the label file
                image_height, image_width = self.main_window.image_window.rasterio_open(image_path).shape

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
                                                         self.main_window.get_transparency_value(),
                                                         show_msg=False)

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
                                                       self.main_window.get_transparency_value(),
                                                       show_msg=False)

                    # Add the annotation to the list for export
                    annotations.append(annotation)

                    # Add annotation to the dict
                    self.annotation_window.annotations_dict[annotation.id] = annotation
                    progress_bar.update_progress()

                # Update the image window's image dict
                self.main_window.image_window.update_image_annotations(image_path)

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