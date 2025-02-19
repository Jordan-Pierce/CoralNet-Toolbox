import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import json

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QDialog, QFileDialog, QVBoxLayout, QPushButton, QLabel, QMessageBox, 
                             QApplication)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class OpenProject(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.label_window = main_window.label_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window

        self.setWindowTitle("Open Project")
        self.resize(400, 100)

        # Setup the open file layout
        self.setup_open_layout()
        # Setup the buttons layout
        self.setup_buttons_layout()

    def setup_open_layout(self):
        layout = QVBoxLayout()
        self.label = QLabel("Select a project JSON file to open:")
        layout.addWidget(self.label)
        self.setLayout(layout)

    def setup_buttons_layout(self):
        layout = self.layout()
        self.open_button = QPushButton("Open")
        self.open_button.clicked.connect(self.open_project)
        layout.addWidget(self.open_button)

    def open_project(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Project JSON", "", "JSON Files (*.json);;All Files (*)", options=options)
        if file_path:
            self.load_project(file_path)

    def load_project(self, file_path):
        try:
            # Make the cursor busy
            QApplication.setOverrideCursor(Qt.WaitCursor)

            with open(file_path, 'r') as file:
                project_data = json.load(file)

            # Update main window with loaded project data
            self.import_labels.load_labels(project_data['labels'])
            self.import_images(project_data['image_paths'])
            self.import_annotations(project_data['annotations'])

            # Update current project label in status bar
            self.main_window.current_project_label.setText(f"{os.path.basename(file_path)}")

            QMessageBox.information(self.annodation_window, "Project Loaded", "Project has been successfully loaded.")

        except Exception as e:
            QMessageBox.warning(self.annotation_window, "Error Loading Project", f"An error occurred while loading the project: {str(e)}")

        finally:
            # Restore the cursor to the default cursor
            QApplication.restoreOverrideCursor()
        
        # Exit
        self.accept()

    def import_images(self, image_paths):
        # Start progress bar
        total_images = len(image_paths)
        progress_bar = ProgressBar(self.image_window, title="Importing Images")
        progress_bar.show()
        progress_bar.start_progress(total_images)

        try:
            # Add images to the image window
            for i, image_path in enumerate(image_paths):
                if image_path not in set(self.image_window.image_paths):
                    try:
                        self.image_window.add_image(image_path)
                    except Exception as e:
                        print(f"Warning: Could not import image {image_path}. Error: {e}")

                # Update the progress bar
                progress_bar.update_progress()

            # Update filtered images
            self.image_window.filter_images()
            # Show the last image
            self.image_window.load_image_by_path(self.image_window.image_paths[-1])

            QMessageBox.information(self.annoation_window,
                                    "Image(s) Imported",
                                    "Image(s) have been successfully imported.")
        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Importing Image(s)",
                                f"An error occurred while importing image(s): {str(e)}")
        finally:
            # Close progress bar
            progress_bar.stop_progress()
            progress_bar.close()

    def import_labels(self, labels):
        # Create a progress bar
        total_labels = len(labels)
        progress_bar = ProgressBar(self.annoation_window, "Importing Labels")
        progress_bar.show()
        progress_bar.start_progress(total_labels)

        try:
            # Import the labels
            for label in labels:
                label = Label.from_dict(label)
                # Create a new label if it doesn't already exist
                if not self.label_window.label_exists(label.short_label_code, label.long_label_code):
                    self.label_window.add_label(label.short_label_code,
                                                label.long_label_code,
                                                label.color,
                                                label.id)
                # Update the progress bar
                progress_bar.update_progress()

            # Set the Review label as active
            self.label_window.set_active_label(self.label_window.get_label_by_long_code("Review"))

            QMessageBox.information(self.annotation_window,
                                    "Labels Imported",
                                    "Annotations have been successfully imported.")

        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Importing Labels",
                                f"An error occurred while importing Labels: {str(e)}")

        finally:
            # Close progress bar
            progress_bar.stop_progress()
            progress_bar.close()

    def import_annotations(self, annotations):
        # Start the progress bar
        total_annotations = len(annotations)
        progress_bar = ProgressBar(self.annotation_window, title="Importing Annotations")
        progress_bar.show()
        progress_bar.start_progress(total_annotations)

        # Required attributes of an annotation
        keys = ['label_short_code', 'label_long_code', 'annotation_color', 'image_path', 'label_id']

        try:
            # Loop through the annotations
            for annotation_data in annotations:
                # Check it has needed attributes
                if not all(key in annotation_data for key in keys):
                    continue

                # Get the annotation type
                annotation_type = annotation_data.get('type')
                if annotation_type == 'PatchAnnotation':
                    annotation = PatchAnnotation.from_dict(annotation_data, self.label_window)
                elif annotation_type == 'PolygonAnnotation':
                    annotation = PolygonAnnotation.from_dict(annotation_data, self.label_window)
                elif annotation_type == 'RectangleAnnotation':
                    annotation = RectangleAnnotation.from_dict(annotation_data, self.label_window)
                else:
                    raise ValueError(f"Unknown annotation type: {annotation_type}")

                # Add annotation to the dict
                self.annotation_window.annotations_dict[annotation.id] = annotation
                progress_bar.update_progress()

            # Update the image window's image dict
            self.image_window.update_image_annotations(self.image_window.image_paths[-1])

            # Load the annotations for current image
            self.annotation_window.load_annotations()

            QMessageBox.information(self.annotation_window,
                                    "Annotations Imported",
                                    "Annotations have been successfully imported.")

        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Importing Annotations",
                                f"An error occurred while importing annotations: {str(e)}")

        finally:
            # Close progress bar
            progress_bar.stop_progress()
            progress_bar.close()
