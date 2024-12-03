import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import json
import uuid

import numpy as np
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QInputDialog)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ImportTagLabAnnotations:
    def __init__(self, main_window):
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

    def taglabToPoints(self, c):
        d = (c * 10).astype(int)
        d = np.diff(d, axis=0, prepend=[[0, 0]])
        d = np.reshape(d, -1)
        d = np.char.mod('%d', d)
        d = " ".join(d)
        return d

    def taglabToContour(self, p):
        if type(p) is str:
            p = map(int, p.split(' '))
            c = np.fromiter(p, dtype=int)
        else:
            c = np.asarray(p)

        if len(c.shape) == 2:
            return c

        c = np.reshape(c, (-1, 2))
        c = np.cumsum(c, axis=0)
        c = c / 10.0
        return c

    def parse_contour(self, contour_str):
        """Parse the contour string into a list of QPointF objects."""
        points = self.taglabToContour(contour_str)
        return [QPointF(x, y) for x, y in points]

    def import_annotations(self):
        self.main_window.untoggle_all_tools()

        if not self.annotation_window.active_image:
            QMessageBox.warning(self.annotation_window,
                                "No Images Loaded",
                                "Please load images first before importing annotations.")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self.annotation_window,
                                                   "Import TagLab Annotations",
                                                   "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)

        if not file_path:
            return

        annotation_size, ok = QInputDialog.getInt(self.annotation_window,
                                                  "Patch Annotation Size",
                                                  "Enter the default patch annotation size for imported annotations:",
                                                  224, 1, 10000, 1)
        if not ok:
            return

        try:
            with open(file_path, 'r') as file:
                taglab_data = json.load(file)

            required_keys = ['labels', 'images']
            if not all(key in taglab_data for key in required_keys):
                QMessageBox.warning(self.annotation_window,
                                    "Invalid JSON Format",
                                    "The selected JSON file does not match the expected TagLab format.")
                return

            # Map image names to image paths
            image_path_map = {os.path.basename(path): path for path in self.image_window.image_paths}

            progress_bar = ProgressBar(self.annotation_window, title="Importing TagLab Annotations")
            progress_bar.show()
            progress_bar.start_progress(len(taglab_data['images']))

            QApplication.setOverrideCursor(Qt.WaitCursor)

            for image_data in taglab_data['images']:
                image_basename = os.path.basename(image_data['channels'][0]['filename'])
                image_full_path = image_path_map[image_basename]

                if not image_full_path:
                    QMessageBox.warning(self.annotation_window,
                                        "Image Not Found",
                                        f"The image '{image_basename}' "
                                        f"from the TagLab project was not found in this project.")
                    continue

                # Inner progress bar
                num_polygons = len(image_data['annotations']['regions'])
                num_points = len(image_data['annotations']['points'])
                num_annotations = num_polygons + num_points
                inner_progress_bar = ProgressBar(self.annotation_window, title="Importing Annotations")
                inner_progress_bar.show()
                inner_progress_bar.start_progress(num_annotations)

                # Loop through all the polygon annotations for this image
                for annotation in list(image_data['annotations']['regions']):
                    # Get the information for the label for this annotation
                    label_id = annotation['class name']
                    label_info = taglab_data['labels'][label_id]
                    short_label_code = label_info['name']
                    long_label_code = label_info['name']
                    color = QColor(*label_info['fill'])

                    # Unpack the annotation data
                    bbox = annotation['bbox']
                    centroid = annotation['centroid']
                    area = annotation['area']
                    perimeter = annotation['perimeter']
                    contour = annotation['contour']
                    inner_contours = annotation['inner contours']
                    class_name = annotation['class name']
                    instance_name = annotation['instance name']
                    blob_name = annotation['blob name']
                    idx = annotation['id']
                    note = annotation['note']
                    data = annotation['data']

                    # Convert contour string to points
                    points = self.parse_contour(annotation['contour'])

                    existing_label = self.label_window.get_label_by_codes(short_label_code, long_label_code)

                    if existing_label:
                        label_id = existing_label.id
                    else:
                        label_id = str(uuid.uuid4())
                        self.label_window.add_label_if_not_exists(short_label_code, long_label_code, color, label_id)

                    polygon_annotation = PolygonAnnotation(
                        points=points,
                        short_label_code=short_label_code,
                        long_label_code=long_label_code,
                        color=color,
                        image_path=image_full_path,
                        label_id=label_id
                    )

                    # Add annotation to the dict
                    self.annotation_window.annotations_dict[polygon_annotation.id] = polygon_annotation

                    # Update the inner progress bar
                    inner_progress_bar.update_progress()

                # Loop through all the point annotations for this image
                for annotation in list(image_data['annotations']['points']):
                    # Get the information for the label for this annotation
                    label_id = annotation['Class']  # Inconsistent
                    label_info = taglab_data['labels'][label_id]
                    short_label_code = label_info['name']
                    long_label_code = label_info['name']
                    color = QColor(*label_info['fill'])

                    # Unpack the annotation data
                    class_name = annotation['Class']
                    x = annotation['X']
                    y = annotation['Y']
                    idx = annotation['Id']
                    note = annotation['Note']
                    data = annotation['Data']

                    existing_label = self.label_window.get_label_by_codes(short_label_code, long_label_code)

                    if existing_label:
                        label_id = existing_label.id
                    else:
                        label_id = str(uuid.uuid4())
                        self.label_window.add_label_if_not_exists(short_label_code, long_label_code, color, label_id)

                    patch_annotation = PatchAnnotation(
                        center_xy=QPointF(x, y),
                        annotation_size=annotation_size,
                        short_label_code=short_label_code,
                        long_label_code=long_label_code,
                        color=color,
                        image_path=image_full_path,
                        label_id=label_id
                    )

                    # Add annotation to the dict
                    self.annotation_window.annotations_dict[patch_annotation.id] = patch_annotation
                    # Update the inner progress bar
                    inner_progress_bar.update_progress()

                # Close the inner progress bar
                inner_progress_bar.stop_progress()
                inner_progress_bar.close()

                # Update the image window's image dict
                self.image_window.update_image_annotations(image_full_path)
                # Update progress bar
                progress_bar.update_progress()

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