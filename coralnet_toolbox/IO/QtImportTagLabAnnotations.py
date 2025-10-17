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
            # OPTIMIZATION: Create lists to hold objects for batch processing
            all_polygon_annotations = []
            all_patch_annotations = []
            images_to_update = set()
            
            # 1. Create all annotation objects in memory first
            for image_data in merged_data['images']:
                image_basename = os.path.basename(image_data['channels'][0]['filename'])
                if image_basename not in image_path_map:
                    progress_bar.update_progress(len(image_data['annotations']['regions']) + 
                                                 len(image_data['annotations']['points']))
                    continue

                image_full_path = image_path_map[image_basename]
                images_to_update.add(image_full_path)

                # Process Polygon annotations (regions)
                for annotation in image_data['annotations']['regions']:
                    try:
                        label_id = annotation['class name']
                        label_info = merged_data['labels'][label_id]
                        short_label_code = label_info['name'].strip()
                        color = QColor(*label_info['fill'])
                        
                        self.label_window.add_label_if_not_exists(short_label_code, short_label_code, color, label_id)
                        
                        points = self.parse_contour(annotation['contour'])
                        holes = [self.parse_contour(inner) for inner in annotation.get('inner contours', [])]

                        polygon_annotation = PolygonAnnotation(
                            points=points,
                            short_label_code=short_label_code,
                            long_label_code=short_label_code,
                            color=color,
                            image_path=image_full_path,
                            label_id=label_id,
                            holes=holes,
                        )
                        polygon_annotation.data = {k: annotation.get(k) for k in ['bbox', 
                                                                                  'centroid', 
                                                                                  'area', 
                                                                                  'perimeter', 
                                                                                  'class name', 
                                                                                  'instance_name', 
                                                                                  'blob_name', 
                                                                                  'id', 
                                                                                  'note', 
                                                                                  'data']}
                        all_polygon_annotations.append(polygon_annotation)

                    except Exception as e:
                        print(f"Error importing region annotation: {str(e)}\n{traceback.print_exc()}")
                    finally:
                        progress_bar.update_progress()

                # Process Patch annotations (points)
                for annotation in image_data['annotations']['points']:
                    try:
                        label_id = annotation['Class']
                        label_info = merged_data['labels'][label_id]
                        short_label_code = label_info['name'].strip()
                        color = QColor(*label_info['fill'])

                        self.label_window.add_label_if_not_exists(short_label_code, short_label_code, color, label_id)
                        
                        patch_annotation = PatchAnnotation(
                            center_xy=QPointF(annotation['X'], annotation['Y']),
                            annotation_size=annotation_size,
                            short_label_code=short_label_code,
                            long_label_code=short_label_code,
                            color=color,
                            image_path=image_full_path,
                            label_id=label_id
                        )
                        all_patch_annotations.append(patch_annotation)
                    except Exception as e:
                        print(f"Error importing point annotation: {str(e)}\n{traceback.print_exc()}")
                    finally:
                        progress_bar.update_progress()

            # 2. Add all created annotations in efficient batch operations
            if all_polygon_annotations:
                self.annotation_window.add_annotations(all_polygon_annotations)
            if all_patch_annotations:
                self.annotation_window.add_annotations(all_patch_annotations)
                
            # 3. Update UI counts for each affected image only ONCE
            for path in images_to_update:
                self.image_window.update_image_annotations(path)
                
            # Load the annotations for the currently visible image
            self.annotation_window.load_annotations()

            QMessageBox.information(self.annotation_window,
                                    "Annotations Imported",
                                    "Annotations have been successfully imported.")

        except Exception as e:
            print(f"Error importing annotations: {str(e)}\n{traceback.print_exc()}")
            QMessageBox.warning(self.annotation_window,
                                "Error Importing Annotations",
                                f"An error occurred while importing annotations:\n\n{str(e)}\n\n"
                                "Please check the console for more details.")
        finally:
            progress_bar.stop_progress()
            progress_bar.close()
            QApplication.restoreOverrideCursor()
