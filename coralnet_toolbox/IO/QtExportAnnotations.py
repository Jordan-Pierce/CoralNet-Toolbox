import warnings

import ujson as json

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation
from coralnet_toolbox.Annotations.QtMaskAnnotation import MaskAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ExportAnnotations:
    def __init__(self, main_window):
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

    def export_annotations(self):
        self.main_window.untoggle_all_tools()

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self.annotation_window,
                                                   "Save Annotations",
                                                   "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if not file_path:
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Combine vector annotations with any existing mask annotations
        all_annotations = list(self.annotation_window.annotations_dict.values())
        for image_path in self.image_window.raster_manager.image_paths:
            raster = self.image_window.raster_manager.get_raster(image_path)
            if raster and raster.mask_annotation:
                all_annotations.append(raster.mask_annotation)

        total_annotations = len(all_annotations)
        progress_bar = ProgressBar(self.annotation_window, title="Exporting Annotations")
        progress_bar.show()
        progress_bar.start_progress(total_annotations)

        try:
            export_dict = {}
            for annotation in all_annotations:
                image_path = annotation.image_path

                if image_path not in export_dict:
                    export_dict[image_path] = []

                # Convert annotation to dictionary based on its type
                if isinstance(annotation, PatchAnnotation):
                    annotation_type = 'PatchAnnotation'
                elif isinstance(annotation, PolygonAnnotation):
                    annotation_type = 'PolygonAnnotation'
                elif isinstance(annotation, RectangleAnnotation):
                    annotation_type = 'RectangleAnnotation'
                elif isinstance(annotation, MultiPolygonAnnotation):
                    annotation_type = 'MultiPolygonAnnotation'
                elif isinstance(annotation, MaskAnnotation):
                    annotation_type = 'MaskAnnotation'
                else:
                    raise ValueError(f"Unknown annotation type: {type(annotation)}")

                annotation_dict = {
                    'type': annotation_type,
                    **annotation.to_dict()
                }

                export_dict[image_path].append(annotation_dict)
                progress_bar.update_progress()

            with open(file_path, 'w') as file:
                json.dump(export_dict, file, indent=4)
                file.flush()

            QMessageBox.information(self.annotation_window,
                                    "Annotations Exported",
                                    "Annotations have been successfully exported.")

        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Exporting Annotations",
                                f"An error occurred while exporting annotations: {str(e)}")

        finally:
            # Restore the cursor
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()
