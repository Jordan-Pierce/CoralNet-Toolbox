import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import ujson as json

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ImportAnnotations:
    def __init__(self, main_window):
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

    def import_annotations(self):
        self.main_window.untoggle_all_tools()

        if not self.annotation_window.active_image:
            QMessageBox.warning(self.annotation_window,
                                "No Images Loaded",
                                "Please load images first before importing annotations.")
            return

        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(self.annotation_window,
                                                     "Load Annotations",
                                                     "",
                                                     "JSON Files (*.json);;All Files (*)",
                                                     options=options)
        if not file_paths:
            return

        try:
            # Make cursor busy
            QApplication.setOverrideCursor(Qt.WaitCursor)
            progress_bar = ProgressBar(self.annotation_window, title="Reading Annotation File(s)")
            progress_bar.show()

            # Start the progress bar
            progress_bar.start_progress(len(file_paths))

            # Read the annotations
            all_data = {}
            for file_path in file_paths:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    all_data.update(data)

                # Update the progress bar
                progress_bar.update_progress()

        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Loading Annotations",
                                f"An error occurred while loading annotations: {str(e)}")
        finally:
            # Restore the cursor
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()

        if not all_data:
            QMessageBox.warning(self.annotation_window,
                                "No Annotations Found",
                                "No annotations were found in the selected files.")
            return

        # Check if the annotations are in the correct format
        keys = ['label_short_code', 'label_long_code', 'annotation_color', 'image_path', 'label_id']

        # Reformat the data
        filtered_annotations = {p: a for p, a in all_data.items() if p in self.image_window.raster_manager.image_paths}
        total_annotations = sum(len(annotations) for annotations in filtered_annotations.values())

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Importing Annotations")
        progress_bar.show()

        # Start the progress bar
        progress_bar.start_progress(total_annotations)

        updated_annotations = False

        try:
            # Load the labels
            for image_path, image_annotations in filtered_annotations.items():

                for annotation in image_annotations:
                    # Skip if missing required keys
                    if not all(key in annotation for key in keys):
                        continue

                    # Extract label data
                    short_label = annotation['label_short_code']
                    long_label = annotation['label_long_code']
                    color = QColor(*annotation['annotation_color'])
                    label_id = annotation['label_id']

                    # Add label if it doesn't exist
                    label = self.label_window.add_label_if_not_exists(short_label, 
                                                                      long_label, 
                                                                      color,
                                                                      label_id)
                    if label.color != color:
                        annotation['annotation_color'] = label.color.getRgb()
                        updated_annotations = True

                    # Update progress
                    progress_bar.update_progress()

        except Exception as e:
            print(f"Error loading label: {str(e)}")

        finally:
            # Restore the cursor
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()

        if updated_annotations:
            QMessageBox.information(self.annotation_window,
                                    "Annotations Updated",
                                    "Some annotations have been updated to match the "
                                    "color of the labels already in the project.")

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Importing Annotations")
        progress_bar.show()

        # Start the progress bar
        progress_bar.start_progress(total_annotations)

        try:
            # Load the annotations
            for image_path, image_annotations in filtered_annotations.items():
                for annotation in image_annotations:
                    if not all(key in annotation for key in keys):
                        continue

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

                # Update the image window's image dict
                self.image_window.update_image_annotations(image_path)

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
            # Restore the cursor
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()
