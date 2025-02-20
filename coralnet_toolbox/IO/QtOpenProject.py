import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import json

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QDialog, QFileDialog, QVBoxLayout, QPushButton, QLabel,
                             QMessageBox, QApplication, QGroupBox, QHBoxLayout, QFormLayout, 
                             QLineEdit)

from coralnet_toolbox.QtLabelWindow import Label
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

        self.current_project_path = self.main_window.current_project_path

        self.setWindowTitle("Open Project")
        self.resize(600, 100)

        # Setup the open file layout
        self.setup_open_layout()
        # Setup the buttons layout
        self.setup_buttons_layout()
        
    def showEvent(self, event):
        super().showEvent(event)
        self.file_path_edit.setText(self.current_project_path)

    def setup_open_layout(self):
        # Create main layout
        layout = QVBoxLayout()
        group_box = QGroupBox("Open Project")
        form_layout = QFormLayout()
        
        # Create horizontal layout for path and browse button
        path_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        path_layout.addWidget(self.file_path_edit)
        
        # Add browse button
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_file)
        path_layout.addWidget(self.browse_button)
        
        # Add to form layout
        form_layout.addRow("Project File:", path_layout)
        
        # Set group box layout
        group_box.setLayout(form_layout)
        
        # Add group box to main layout
        layout.addWidget(group_box)
        self.setLayout(layout)

    def setup_buttons_layout(self):
        layout = self.layout()
        
        # Create horizontal layout for buttons
        button_layout = QHBoxLayout()
        
        # Add open button
        self.open_button = QPushButton("Open")
        self.open_button.clicked.connect(self.load_selected_project)
        button_layout.addWidget(self.open_button)
        
        # Add cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)

    def browse_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 
                                                   "Open Project JSON", 
                                                   "", 
                                                   "JSON Files (*.json);;All Files (*)", 
                                                   options=options)
        if file_path:
            self.file_path_edit.setText(file_path)
            
    def load_selected_project(self):
        file_path = self.file_path_edit.text()
        if file_path:
            self.load_project(file_path)
        else:
            QMessageBox.warning(self, 
                                "Error", 
                                "Please select a project file first.")

    def open_project(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 
                                                   "Open Project JSON", 
                                                   "", 
                                                   "JSON Files (*.json);;All Files (*)", 
                                                   options=options)
        if file_path:
            self.load_project(file_path)

    def load_project(self, file_path):
        try:
            # Make the cursor busy
            QApplication.setOverrideCursor(Qt.WaitCursor)

            with open(file_path, 'r') as file:
                project_data = json.load(file)

            # Update main window with loaded project data
            self.import_images(project_data['image_paths'])
            self.import_labels(project_data['labels'])
            self.import_annotations(project_data['annotations'])

            # Update current project path
            self.current_project_path = file_path

            QMessageBox.information(self.annotation_window, 
                                    "Project Loaded",
                                    "Project has been successfully loaded.")

        except Exception as e:
            QMessageBox.warning(self.annotation_window, 
                                "Error Loading Project", 
                                f"An error occurred while loading the project: {str(e)}")

        finally:
            # Restore the cursor to the default cursor
            QApplication.restoreOverrideCursor()
        
        # Exit
        self.accept()

    def import_images(self, image_paths):
        if not image_paths:
            return
        
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

        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Importing Image(s)",
                                f"An error occurred while importing image(s): {str(e)}")
        finally:
            # Close progress bar
            progress_bar.stop_progress()
            progress_bar.close()

    def import_labels(self, labels):
        if not labels:
            return
        
        # Create a progress bar
        total_labels = len(labels)
        progress_bar = ProgressBar(self.annotation_window, "Importing Labels")
        progress_bar.show()
        progress_bar.start_progress(total_labels)

        try:
            # Import the labels
            for label in labels:
                # Create a new label object
                label = Label.from_dict(label)
                
                # Create a new label if it doesn't already exist
                self.label_window.add_label_if_not_exists(label.short_label_code,
                                                          label.long_label_code,
                                                          label.color,
                                                          label.id)
                # Update the progress bar
                progress_bar.update_progress()
                
        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Importing Labels",
                                f"An error occurred while importing Labels: {str(e)}")

        finally:
            # Close progress bar
            progress_bar.stop_progress()
            progress_bar.close()

    def import_annotations(self, annotations):
        if not annotations:
            return
        
        # Start the progress bar
        total_annotations = sum(len(image_annotations) for image_annotations in annotations.values())
        progress_bar = ProgressBar(self.annotation_window, title="Importing Annotations")
        progress_bar.show()
        progress_bar.start_progress(total_annotations)

        # Required attributes of an annotation
        keys = ['label_short_code', 'label_long_code', 'annotation_color', 'image_path', 'label_id']

        try:
            # Loop through the annotations
            for image_path, image_annotations in annotations.items():
                for annotation in image_annotations:
                    # Check if all required keys are present
                    if not all(key in annotation for key in keys):
                        print(f"Warning: Missing required keys in annotation: {annotation}")
                        continue
                    
                    # Get the annotation type
                    annotation_type = annotation.get('type')
                    if annotation_type == 'PatchAnnotation':
                        annotation = PatchAnnotation.from_dict(annotation, self.label_window)
                    elif annotation_type == 'PolygonAnnotation':
                        annotation = PolygonAnnotation.from_dict(annotation, self.label_window)
                    elif annotation_type == 'RectangleAnnotation':
                        annotation = RectangleAnnotation.from_dict(annotation, self.label_window)
                    else:
                        raise ValueError(f"Unknown annotation type: {annotation_type}")

                    # Add annotation to the dict
                    self.annotation_window.add_annotation_to_dict(annotation)
                    
                    # Update the progress bar
                    progress_bar.update_progress()
                    
                # Update the image window's image dict
                self.image_window.update_image_annotations(image_path)

            # Load the annotations for current image
            self.annotation_window.load_annotations()

        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Importing Annotations",
                                f"An error occurred while importing annotations: {str(e)}")

        finally:
            # Close progress bar
            progress_bar.stop_progress()
            progress_bar.close()

    def get_project_path(self):
        return self.current_project_path
