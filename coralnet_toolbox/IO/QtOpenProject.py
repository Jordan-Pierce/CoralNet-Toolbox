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

from coralnet_toolbox.QtWorkArea import WorkArea

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation

from coralnet_toolbox.Common.QtUpdateImagePaths import UpdateImagePaths

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
        
        self.updated_paths = {}

        self.setWindowTitle("Open Project")
        self.resize(600, 100)

        # Setup the open file layout
        self.setup_open_layout()
        # Setup the buttons layout
        self.setup_buttons_layout()
        
    def showEvent(self, event):
        """Override showEvent to set the current project path in the file path edit."""
        super().showEvent(event)
        self.file_path_edit.setText(self.current_project_path)

    def setup_open_layout(self):
        """Setup the layout for opening a project."""
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
        """Setup the layout for the buttons."""
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
        """Open a file dialog to select a project JSON file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 
                                                   "Open Project JSON", 
                                                   "", 
                                                   "JSON Files (*.json);;All Files (*)", 
                                                   options=options)
        if file_path:
            self.file_path_edit.setText(file_path)
            
    def load_selected_project(self):
        """Load the selected project file."""
        file_path = self.file_path_edit.text()
        if file_path:
            self.load_project(file_path)
        else:
            QMessageBox.warning(self, 
                                "Error", 
                                "Please select a project file first.")

    def open_project(self):
        """Open a project from a JSON file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 
                                                   "Open Project JSON", 
                                                   "", 
                                                   "JSON Files (*.json);;All Files (*)", 
                                                   options=options)
        if file_path:
            self.load_project(file_path)

    def load_project(self, file_path):
        """Load a project from a JSON file."""
        try:
            # Make the cursor busy
            QApplication.setOverrideCursor(Qt.WaitCursor)

            with open(file_path, 'r') as file:
                project_data = json.load(file)

            # Update main window with loaded project data
            self.import_images(project_data.get('image_paths'))
            self.import_workareas(project_data.get('workareas'))
            self.import_labels(project_data.get('labels'))
            self.import_annotations(project_data.get('annotations'))
            
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
        """Import images from the given paths."""
        if not image_paths:
            return
        
        if not all([os.path.exists(path) for path in image_paths]):
            image_paths, self.updated_paths = UpdateImagePaths.update_paths(image_paths)
        
        # Start progress bar
        total_images = len(image_paths)
        progress_bar = ProgressBar(self.image_window, title="Importing Images")
        progress_bar.show()
        progress_bar.start_progress(total_images)

        try:
            # Add images to the image window's raster manager one by one
            for path in image_paths:
                # Use the improved add_image method which handles both
                # adding to raster_manager and updating filtered_paths
                self.image_window.add_image(path)
                
                # Update the progress bar
                progress_bar.update_progress()
            
            # Show the last image if any were imported
            if self.image_window.raster_manager.image_paths:
                self.image_window.load_image_by_path(self.image_window.raster_manager.image_paths[-1])

        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Importing Image(s)",
                                f"An error occurred while importing image(s): {str(e)}")
        finally:
            # Close progress bar
            progress_bar.stop_progress()
            progress_bar.close()
            
    def import_workareas(self, workareas):
        """Import work areas for each image."""
        if not workareas:
            return
        
        # Start the progress bar
        total_images = len(workareas)
        progress_bar = ProgressBar(self.annotation_window, title="Importing Work Areas")
        progress_bar.show()
        progress_bar.start_progress(total_images)

        try:
            # Loop through each image's work areas
            for image_path, work_areas_list in workareas.items():
                
                # Check if the image path was updated (moved)
                updated_path = False
                
                if image_path not in self.image_window.raster_manager.image_paths:
                    # Check if the path was updated
                    if image_path in self.updated_paths:
                        image_path = self.updated_paths[image_path]
                        updated_path = True
                    else:
                        print(f"Warning: Image not found for work areas: {image_path}")
                        continue
                
                # Get the raster for this image
                raster = self.image_window.raster_manager.get_raster(image_path)
                if not raster:
                    print(f"Warning: Could not get raster for image: {image_path}")
                    continue
                
                # Import each work area for this image
                for work_area_data in work_areas_list:
                    try:
                        # Update image path if it was changed
                        if updated_path:
                            work_area_data['image_path'] = image_path
                        
                        # Create WorkArea from dictionary
                        work_area = WorkArea.from_dict(work_area_data, image_path)
                        
                        # Add work area to the raster
                        raster.add_work_area(work_area)
                        
                    except Exception as e:
                        print(f"Warning: Could not import work area {work_area_data}: {str(e)}")
                        continue
                
                # Update the progress bar
                progress_bar.update_progress()

        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Importing Work Areas",
                                f"An error occurred while importing work areas: {str(e)}")

        finally:
            # Close progress bar
            progress_bar.stop_progress()
            progress_bar.close()

    def import_labels(self, labels):
        """Import labels from the given list."""
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
                label = self.label_window.add_label_if_not_exists(label.short_label_code,
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
        """Import annotations from the given dictionary."""
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
                
                # Checking if the image path is updated (moved)
                updated_path = False
                
                if image_path not in self.image_window.raster_manager.image_paths:
                    # Check if the path was updated
                    if image_path in self.updated_paths:
                        image_path = self.updated_paths[image_path]
                        updated_path = True
                    else:
                        print(f"Warning: Image not found: {image_path}")
                        continue
                
                for annotation in image_annotations:
                    # Check if all required keys are present
                    if not all(key in annotation for key in keys):
                        print(f"Warning: Missing required keys in annotation: {annotation}")
                        continue
                    
                    # Check if the image path was updated
                    if updated_path:
                        annotation['image_path'] = image_path
                    
                    # Get the annotation type
                    annotation_type = annotation.get('type')
                    if annotation_type == 'PatchAnnotation':
                        annotation = PatchAnnotation.from_dict(annotation, self.label_window)
                    elif annotation_type == 'PolygonAnnotation':
                        annotation = PolygonAnnotation.from_dict(annotation, self.label_window)
                    elif annotation_type == 'RectangleAnnotation':
                        annotation = RectangleAnnotation.from_dict(annotation, self.label_window)
                    elif annotation_type == 'MultiPolygonAnnotation':
                        annotation = MultiPolygonAnnotation.from_dict(annotation, self.label_window)
                    else:
                        raise ValueError(f"Unknown annotation type: {annotation_type}")

                    # Add annotation to the dict
                    self.annotation_window.add_annotation_to_dict(annotation)
                    
                    # Update the progress bar
                    progress_bar.update_progress()
                    
                # Update the image window's image annotations
                self.image_window.update_image_annotations(image_path)

            # Load the annotations for current image and update counts
            self.annotation_window.load_annotations()
            self.label_window.update_annotation_count()

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