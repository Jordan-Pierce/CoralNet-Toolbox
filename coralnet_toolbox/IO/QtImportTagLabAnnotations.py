import warnings

import os
import uuid
import traceback
import ujson as json

import numpy as np

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QInputDialog)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.QtProgressBar import ProgressBar

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------
# Note: Field names are case-sensitive and should match the CoralNet export exactly.


class ImportTagLabAnnotations:
    def __init__(self, main_window):
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

    def taglabToPoints(self, c):
        """Convert a TagLab contour string to a list of points."""
        d = (c * 10).astype(int)
        d = np.diff(d, axis=0, prepend=[[0, 0]])
        d = np.reshape(d, -1)
        d = np.char.mod('%d', d)
        d = " ".join(d)
        return d

    def taglabToContour(self, p):
        """Convert a TagLab contour string to a list of readable points."""
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

    def standardize_data(self, image_data):
        """Standardize the data format for TagLab annotations."""
        # Deals with the fact that TagLab JSON files can have different structures
        # before and after point annotations were introduced in version v2024.10.29
        for image in image_data:
            # Older versions do not have regions and points, annotations is a list not a dict
            if isinstance(image['annotations'], list):
                image['annotations'] = {'regions': image['annotations'], 'points': []}

        return image_data

    def import_annotations(self):
        """Import annotations from TagLab JSON files."""
        self.main_window.untoggle_all_tools()

        if not self.annotation_window.active_image:
            QMessageBox.warning(self.annotation_window,
                                "No Images Loaded",
                                "Please load images first before importing annotations.")
            return

        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(self.annotation_window,
                                                     "Import TagLab Annotations",
                                                     "",
                                                     "JSON Files (*.json);;All Files (*)",
                                                     options=options)

        if not file_paths:
            return

        annotation_size, ok = QInputDialog.getInt(self.annotation_window,
                                                  "Patch Annotation Size",
                                                  "Enter the default patch annotation size for imported annotations:",
                                                  224, 1, 10000, 1)
        if not ok:
            return

        try:
            all_data = []
            for file_path in file_paths:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    all_data.append(data)

            merged_data = {
                "labels": {},
                "images": []
            }

            for data in all_data:
                merged_data["labels"].update(data["labels"])
                merged_data["images"].extend(data["images"])

            if not all(key in merged_data for key in ['labels', 'images']):
                raise Exception("The selected JSON files do not match the expected TagLab format.")

            # Standardize the data (deals with different TagLab JSON structures)
            merged_data["images"] = self.standardize_data(merged_data["images"])

            # Map image names to image paths
            image_path_map = {os.path.basename(path): path for path in self.image_window.raster_manager.image_paths}

            num_regions = sum(len(image_data['annotations']['regions']) for image_data in merged_data['images'])
            num_points = sum(len(image_data['annotations']['points']) for image_data in merged_data['images'])
            total_annotations = num_regions + num_points

        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Loading Annotations",
                                f"An error occurred while loading annotations: {str(e)}")
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Importing TagLab Annotations")
        progress_bar.show()
        progress_bar.start_progress(total_annotations)

        try:
            # Import the annotations
            for image_data in merged_data['images']:
                # Get the basename from the TagLab project file (not path)
                image_basename = os.path.basename(image_data['channels'][0]['filename'])
                # Check to see if there is a matching image (using basename) in the current project
                if image_basename not in image_path_map:
                    continue  # Skip this image

                # Get the full path to the image
                image_full_path = image_path_map[image_basename]

                # Loop through all the polygon annotations for this image
                for annotation in list(image_data['annotations']['regions']):
                    try:
                        # Get the information for the label for this annotation
                        label_id = annotation['class name']
                        label_info = merged_data['labels'][label_id]
                        short_label_code = label_info['name'].strip()
                        long_label_code = label_info['name'].strip()
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
                            self.label_window.add_label_if_not_exists(short_label_code,
                                                                      long_label_code,
                                                                      color,
                                                                      label_id)
                        # Create the polygon annotation
                        polygon_annotation = PolygonAnnotation(
                            points=points,
                            short_label_code=short_label_code,
                            long_label_code=long_label_code,
                            color=color,
                            image_path=image_full_path,
                            label_id=label_id
                        )
                        # Add annotation to the dict
                        self.annotation_window.add_annotation_to_dict(polygon_annotation)

                    except Exception as e:
                        print(f"Error importing annotation: {str(e)}\n{traceback.print_exc()}")
                    finally:
                        # Update the progress bar
                        progress_bar.update_progress()

                # Loop through all the point annotations for this image
                for annotation in list(image_data['annotations']['points']):
                    try:
                        # Get the information for the label for this annotation
                        label_id = annotation['Class']  # Inconsistent
                        label_info = merged_data['labels'][label_id]
                        short_label_code = label_info['name'].strip()
                        long_label_code = label_info['name'].strip()
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
                            self.label_window.add_label_if_not_exists(short_label_code,
                                                                      long_label_code,
                                                                      color,
                                                                      label_id)
                        # Create the patch annotation
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
                        self.annotation_window.add_annotation_to_dict(patch_annotation)

                    except Exception as e:
                        print(f"Error importing annotation: {str(e)}\n{traceback.print_exc()}")
                    finally:
                        # Update the progress bar
                        progress_bar.update_progress()

                # Update the image window's image dict
                self.image_window.update_image_annotations(image_full_path)

            # Load the annotations for current image
            self.annotation_window.load_annotations()

            QMessageBox.information(self.annotation_window,
                                    "Annotations Imported",
                                    "Annotations have been successfully imported.")

        except Exception as e:
            print(f"Error importing annotation: {str(e)}\n{traceback.print_exc()}")

            QMessageBox.warning(self.annotation_window,
                                "Error Importing Annotations",
                                f"An error occurred while importing annotations:\n\n{str(e)}\
                                    \n\nPlease check the console for more details.")

        finally:
            # Stop the progress bar
            progress_bar.stop_progress()
            progress_bar.close()
            QApplication.restoreOverrideCursor()
