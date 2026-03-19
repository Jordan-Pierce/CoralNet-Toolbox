import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import json
import pickle
import time

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QFileDialog, QVBoxLayout, QPushButton,
                             QMessageBox, QApplication, QGroupBox, QHBoxLayout, 
                             QFormLayout, QLineEdit)

from coralnet_toolbox.QtLabelWindow import Label

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation
from coralnet_toolbox.Annotations.QtMaskAnnotation import MaskAnnotation

from coralnet_toolbox.WorkArea import WorkArea

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
                                                   "Open Project", 
                                                   "", 
                                                   "Binary Files (*.bin);;JSON Files (*.json);;All Files (*)", 
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
                                                   "Open Project", 
                                                   "", 
                                                   "Binary Files (*.bin);;JSON Files (*.json);;All Files (*)", 
                                                   options=options)
        if file_path:
            self.load_project(file_path)

    def load_project(self, file_path):
        """Load a project from a JSON file."""
        try:
            # Make the cursor busy
            QApplication.setOverrideCursor(Qt.WaitCursor)
            start_time = time.perf_counter()

            # Decide how to load based on extension. Support .bin (pickle) and .json.
            _, ext = os.path.splitext(file_path)
            if ext.lower() == '.bin':
                with open(file_path, 'rb') as file:
                    project_data = pickle.load(file)
            elif ext.lower() == '.json':
                with open(file_path, 'r') as file:
                    project_data = json.load(file)
            else:
                # Try JSON first, then pickle as fallback
                try:
                    with open(file_path, 'r') as file:
                        project_data = json.load(file)
                except Exception:
                    with open(file_path, 'rb') as file:
                        project_data = pickle.load(file)

            # Handle both new and old project formats for images and work areas
            images_data = project_data.get('images', project_data.get('image_paths'))
            legacy_workareas = project_data.get('workareas')  # For backward compatibility

            # Update main window with loaded project data
            self.import_images(images_data, legacy_workareas)
            self.import_labels(project_data.get('labels'))
            self.import_annotations(project_data.get('annotations'))
                        
            # Update current project path
            self.current_project_path = file_path
            elapsed = time.perf_counter() - start_time
            elapsed_msg = f"Import complete ({elapsed:.2f}s)."
            try:
                self.main_window.status_bar.showMessage(elapsed_msg, 5000)
            except Exception:
                pass

        except Exception as e:
            QMessageBox.warning(self.annotation_window, 
                                "Error Loading Project", 
                                f"An error occurred while loading the project: {str(e)}")

        finally:
            # Restore the cursor to the default cursor
            QApplication.restoreOverrideCursor()
        
        # Exit
        self.accept()

    def import_images(self, images_data, legacy_workareas=None):
        """Import images, states, and work areas from the given data."""
        try:
            self.main_window.status_bar.showMessage("Importing images...", 0)
        except Exception:
            pass
        if not images_data:
            return

        is_new_format = isinstance(images_data[0], dict)
        image_paths = [img['path'] for img in images_data] if is_new_format else images_data
        
        if not all([os.path.exists(path) for path in image_paths]):
            # Ask user to update missing paths. If they cancel, abort the import.
            new_paths, self.updated_paths = UpdateImagePaths.update_paths(image_paths)
            if new_paths is None:
                # User cancelled path update — stop importing images
                return
            
            image_paths = new_paths
        
        total_images = len(image_paths)
        progress_bar = ProgressBar(self.image_window, title="Importing Images")
        progress_bar.show()
        progress_bar.start_progress(total_images)

        try:
            image_data_map = {img['path']: img for img in images_data} if is_new_format else {}

            # Add images directly to the manager without emitting signals
            for path in image_paths:
                # Call the manager directly to add the raster silently
                self.image_window.raster_manager.add_raster(path, emit_signal=False) 
                
                raster = self.image_window.raster_manager.get_raster(path)
                if not raster:
                    progress_bar.update_progress()
                    continue

                if is_new_format and path in image_data_map:
                    data = image_data_map[path]
                    # Use the raster's update_from_dict method
                    raster.update_from_dict(data)
                
                progress_bar.update_progress()

            if legacy_workareas:
                for image_path, work_areas_list in legacy_workareas.items():
                    current_path = self.updated_paths.get(image_path, image_path)
                    raster = self.image_window.raster_manager.get_raster(current_path)
                    if raster:
                        for work_area_data in work_areas_list:
                            work_area = WorkArea.from_dict(work_area_data, current_path)
                            raster.add_work_area(work_area)

        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Importing Image(s)",
                                f"An error occurred while importing image(s): {str(e)}")
        finally:
            # Manually perform the UI updates ONCE for all imported images
            self.image_window.update_search_bars()
            self.image_window.filter_images()

            if self.image_window.raster_manager.image_paths:
                self.image_window.load_image_by_path(self.image_window.raster_manager.image_paths[0])
            
            progress_bar.stop_progress()
            progress_bar.close()
            try:
                self.main_window.status_bar.showMessage("Import complete.", 3000)
            except Exception:
                pass

    def import_labels(self, labels):
        """Import labels from the given list."""
        try:
            self.main_window.status_bar.showMessage("Importing labels...", 0)
        except Exception:
            pass
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
            try:
                self.main_window.status_bar.showMessage("Import complete.", 3000)
            except Exception:
                pass

    def import_annotations(self, annotations):
        """Import annotations from the given dictionary."""
        try:
            self.main_window.status_bar.showMessage("Importing annotations...", 0)
        except Exception:
            pass
        if not annotations:
            return

        # Start the progress bar
        total_annotations = sum(len(image_annotations) for image_annotations in annotations.values())
        progress_bar = ProgressBar(self.annotation_window, title="Importing Annotations")
        progress_bar.show()
        progress_bar.start_progress(total_annotations)

        # Required attributes of an annotation
        keys = ['label_short_code', 'label_long_code', 'annotation_color', 'image_path', 'label_id']

        skipped_count = 0
        duplicate_count = 0

        # OPTIMIZATION: Create lists to hold objects for batch processing
        all_new_annotations = []
        images_to_update = set()
        mask_annotations_to_set = []  # collect (image_path, MaskAnnotation) to assign after loop

        try:
            # Local references for speed
            ann_window = self.annotation_window
            img_window = self.image_window
            lbl_window = self.label_window
            raster_manager = img_window.raster_manager

            # Pre-cache existing annotation IDs and label map for fast lookup
            existing_ids = set(ann_window.annotations_dict.keys())

            # Batched progress updates: reduce frequent UI calls
            progress_batch = 0
            PROGRESS_BATCH_SIZE = 64

            # Loop through the annotations to create objects
            for image_path, image_annotations in annotations.items():
                # Resolve updated path mapping quickly
                updated_path = False
                if image_path not in raster_manager.image_paths:
                    if image_path in self.updated_paths:
                        image_path = self.updated_paths[image_path]
                        updated_path = True
                    else:
                        print(f"Warning: Image not found: {image_path}")
                        skipped_count += len(image_annotations)
                        progress_batch += len(image_annotations)
                        # flush progress batch if needed
                        if progress_batch >= PROGRESS_BATCH_SIZE:
                            for _ in range(progress_batch):
                                progress_bar.update_progress()
                            progress_batch = 0
                        continue

                per_image_created = 0
                per_image_skipped = 0

                for annotation_dict in image_annotations:
                    # Quick key check
                    if not all(key in annotation_dict for key in keys):
                        skipped_count += 1
                        per_image_skipped += 1
                        progress_batch += 1
                        continue

                    # Duplicate check fast-path
                    ann_id = annotation_dict.get('id')
                    if ann_id and ann_id in existing_ids:
                        duplicate_count += 1
                        per_image_skipped += 1
                        progress_batch += 1
                        continue

                    # If the image path was updated, fix it in-place
                    if updated_path:
                        annotation_dict['image_path'] = image_path

                    annotation_type = annotation_dict.get('type')
                    annotation = None

                    # Create annotation objects without adding to the window yet
                    try:
                        if annotation_type == 'MaskAnnotation':
                            # Delay attaching mask annotations to avoid rasterio/scene ops in the hot loop
                            mask_ann = MaskAnnotation.from_dict(annotation_dict, lbl_window)
                            mask_annotations_to_set.append((image_path, mask_ann))
                        elif annotation_type == 'PatchAnnotation':
                            annotation = PatchAnnotation.from_dict(annotation_dict, lbl_window)
                        elif annotation_type == 'PolygonAnnotation':
                            annotation = PolygonAnnotation.from_dict(annotation_dict, lbl_window)
                        elif annotation_type == 'RectangleAnnotation':
                            annotation = RectangleAnnotation.from_dict(annotation_dict, lbl_window)
                        elif annotation_type == 'MultiPolygonAnnotation':
                            annotation = MultiPolygonAnnotation.from_dict(annotation_dict, lbl_window)
                        else:
                            skipped_count += 1
                            per_image_skipped += 1
                            progress_batch += 1
                            continue

                        if annotation:
                            all_new_annotations.append(annotation)
                            images_to_update.add(image_path)
                            per_image_created += 1

                        # Mark new ID in the temporary set to avoid duplicates in same import
                        if ann_id:
                            existing_ids.add(ann_id)

                    except Exception as inner_e:
                        print(f"Warning: failed creating annotation for {image_path}: {inner_e}")
                        skipped_count += 1
                        per_image_skipped += 1

                    # Batched progress increment (reduce UI overhead)
                    progress_batch += 1
                    if progress_batch >= PROGRESS_BATCH_SIZE:
                        for _ in range(progress_batch):
                            progress_bar.update_progress()
                        progress_batch = 0

                # flush per-image small remainder to progress
                if progress_batch > 0:
                    for _ in range(progress_batch):
                        progress_bar.update_progress()
                    progress_batch = 0

            # Add all vector annotations in a single batch operation
            if all_new_annotations:
                self.annotation_window.add_annotations(all_new_annotations)

            # Attach mask annotations (done after batch creation to avoid scene thrash)
            for image_path, mask_ann in mask_annotations_to_set:
                raster = raster_manager.get_raster(image_path)
                if raster:
                    raster.mask_annotation = mask_ann

            # Update UI and counts only ONCE after the batch is added
            for path in images_to_update:
                img_window.update_image_annotations(path)

            # Load the annotations for current image and update counts
            self.annotation_window.load_annotations()
            self.label_window.update_annotation_count()

        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Importing Annotations",
                                f"An error occurred while importing annotations: {str(e)}")

        finally:
            if skipped_count > 0:
                print(f"Warning: Skipped {skipped_count} annotations due to missing keys or other issues.")
            if duplicate_count > 0:
                print(f"Warning: Skipped {duplicate_count} duplicate annotations based on ID.")

            # Ensure progress bar completes
            progress_bar.stop_progress()
            progress_bar.close()
            try:
                self.main_window.status_bar.showMessage("Import complete.", 3000)
            except Exception:
                pass

    def get_project_path(self):
        return self.current_project_path