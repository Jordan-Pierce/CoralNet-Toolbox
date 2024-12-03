import json
import os
import warnings

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.QtProgressBar import ProgressBar

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------
# TODO export points to TagLab annotations

class ExportTagLabAnnotations:
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

    def export_annotations(self):
        self.main_window.untoggle_all_tools()

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self.annotation_window,
                                                   "Export TagLab Annotations",
                                                   "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)

                total_annotations = len(list(self.annotation_window.annotations_dict.values()))
                progress_bar = ProgressBar(self.annotation_window, title="Exporting TagLab Annotations")
                progress_bar.show()
                progress_bar.start_progress(total_annotations)

                taglab_data = {
                    "filename": file_path,
                    "working_area": None,
                    "dictionary_name": "scripps",
                    "dictionary_description": "These color codes are the ones typically used by the "
                                              "Scripps Institution of Oceanography (UCSD)",
                    "labels": {},
                    "images": []
                }

                # Collect all labels
                for annotation in self.annotation_window.annotations_dict.values():
                    label_id = annotation.label.short_label_code
                    if label_id not in taglab_data["labels"]:
                        label_info = {
                            "id": label_id,
                            "name": label_id,
                            "description": None,
                            "fill": annotation.label.color.getRgb()[:3],
                            "border": [200, 200, 200],
                            "visible": True
                        }
                        taglab_data["labels"][label_id] = label_info

                # Collect all images and their annotations
                image_annotations = {}
                for idx, annotation in enumerate(self.annotation_window.annotations_dict.values()):

                    # Get the image once, create a dict entry
                    image_path = annotation.image_path
                    if image_path not in image_annotations:
                        image_annotations[image_path] = {
                            "rect": [0.0, 0.0, 0.0, 0.0],
                            "map_px_to_mm_factor": "1",
                            "width": 0,
                            "height": 0,
                            "annotations": {
                                "regions": [],
                                "points": []
                            },
                            "layers": [],
                            "channels": [
                                {
                                    "filename": image_path,
                                    "type": "RGB"
                                }
                            ],
                            "id": os.path.basename(image_path),
                            "name": os.path.basename(image_path),
                            "workspace": [],
                            "export_dataset_area": [],
                            "acquisition_date": "2000-01-01",
                            "georef_filename": "",
                            "metadata": {},
                            "grid": None
                        }

                    # Create a polygon annotation
                    if isinstance(annotation, PolygonAnnotation):
                        # Calculate bounding box, centroid, area, perimeter, and contour
                        points = annotation.points
                        min_x = min(point.x() for point in points)
                        min_y = min(point.y() for point in points)
                        max_x = max(point.x() for point in points)
                        max_y = max(point.y() for point in points)
                        centroid_x = sum(point.x() for point in points) / len(points)
                        centroid_y = sum(point.y() for point in points) / len(points)
                        area = annotation.calculate_area()
                        perimeter = annotation.calculate_perimeter()
                        contour = self.taglabToPoints(np.array([[point.x(), point.y()] for point in points]))

                        annotation_dict = {
                            "bbox": [min_x, min_y, max_x, max_y],
                            "centroid": [centroid_x, centroid_y],
                            "area": area,
                            "perimeter": perimeter,
                            "contour": contour,
                            "inner contours": [],
                            "class name": annotation.label.short_label_code,
                            "instance name": "coral0",  # Placeholder, update as needed
                            "blob name": f"c-0-{centroid_x}x-{centroid_y}y",
                            "id": idx,
                            "note": "",
                            "data": {}
                        }
                        image_annotations[image_path]["annotations"]["regions"].append(annotation_dict)

                    # Create a point annotation
                    if isinstance(annotation, PatchAnnotation):
                        # Calculate the XY
                        x = annotation.center_xy.x()
                        y = annotation.center_xy.y()

                        annotation_dict = {
                            "X": x,
                            "Y": y,
                            "Class": annotation.label.short_label_code,
                            "Id": idx,
                            "Note": "",
                            "Data": {}
                        }
                        image_annotations[image_path]["annotations"]["points"].append(annotation_dict)

                    # Update the progress bar
                    progress_bar.update_progress()

                # Add images to the main data structure
                taglab_data["images"] = list(image_annotations.values())

                # Save the JSON data to the selected file
                with open(file_path, 'w') as file:
                    json.dump(taglab_data, file, indent=4)
                    file.flush()

                progress_bar.stop_progress()
                progress_bar.close()

                QMessageBox.information(self.annotation_window,
                                        "Annotations Exported",
                                        "Annotations have been successfully exported.")

            except Exception as e:
                QMessageBox.warning(self.annotation_window,
                                    "Error Exporting Annotations",
                                    f"An error occurred while exporting annotations: {str(e)}")

            QApplication.restoreOverrideCursor()