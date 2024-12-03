import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import json

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
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
        file_path, _ = QFileDialog.getOpenFileName(self.annotation_window,
                                                   "Load Annotations",
                                                   "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)

                with open(file_path, 'r') as file:
                    data = json.load(file)

                keys = ['label_short_code', 'label_long_code', 'annotation_color', 'image_path', 'label_id']

                filtered_annotations = {p: a for p, a in data.items() if p in self.image_window.image_paths}
                total_annotations = sum(len(annotations) for annotations in filtered_annotations.values())

                progress_bar = ProgressBar(self.annotation_window, title="Importing Annotations")
                progress_bar.show()
                progress_bar.start_progress(total_annotations)

                updated_annotations = False

                for image_path, annotations in filtered_annotations.items():
                    for annotation_data in annotations:
                        if not all(key in annotation_data for key in keys):
                            continue

                        short_label = annotation_data['label_short_code']
                        long_label = annotation_data['label_long_code']
                        color = QColor(*annotation_data['annotation_color'])

                        label_id = annotation_data['label_id']
                        self.label_window.add_label_if_not_exists(short_label, long_label, color, label_id)

                        existing_color = self.label_window.get_label_color(short_label, long_label)

                        if existing_color != color:
                            annotation_data['annotation_color'] = existing_color.getRgb()
                            updated_annotations = True

                        progress_bar.update_progress()

                if updated_annotations:
                    QMessageBox.information(self.annotation_window,
                                            "Annotations Updated",
                                            "Some annotations have been updated to match the "
                                            "color of the labels already in the project.")

                for image_path, annotations in filtered_annotations.items():
                    for annotation_data in annotations:
                        if not all(key in annotation_data for key in keys):
                            continue

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
                    self.image_window.update_image_annotations(image_path)

                # Load the annotations for current image
                self.annotation_window.load_annotations()

                # Stop the progress bar
                progress_bar.stop_progress()
                progress_bar.close()

                QMessageBox.information(self.annotation_window,
                                        "Annotations Imported",
                                        "Annotations have been successfully imported.")

            except Exception as e:
                QMessageBox.warning(self.annotation_window,
                                    "Error Importing Annotations",
                                    f"An error occurred while importing annotations: {str(e)}")

            QApplication.restoreOverrideCursor()