import warnings

import ujson as json

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation

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

        total_annotations = len(list(self.annotation_window.annotations_dict.values()))
        progress_bar = ProgressBar(self.annotation_window, title="Exporting Annotations")
        progress_bar.show()
        progress_bar.start_progress(total_annotations)

        try:
            export_dict = {}
            for annotation in self.annotation_window.annotations_dict.values():
                image_path = annotation.image_path

                # Verify the image path exists in the raster manager
                if image_path not in self.image_window.raster_manager.image_paths:
                    # Skip annotations for images that are not in the raster manager
                    continue

                if image_path not in export_dict:
                    export_dict[image_path] = []

                # Convert annotation to dictionary based on its type
                if isinstance(annotation, PatchAnnotation):
                    annotation_dict = {
                        'type': 'PatchAnnotation',
                        **annotation.to_dict()
                    }
                elif isinstance(annotation, PolygonAnnotation):
                    annotation_dict = {
                        'type': 'PolygonAnnotation',
                        **annotation.to_dict()
                    }
                elif isinstance(annotation, RectangleAnnotation):
                    annotation_dict = {
                        'type': 'RectangleAnnotation',
                        **annotation.to_dict()
                    }
                elif isinstance(annotation, MultiPolygonAnnotation):
                    annotation_dict = {
                        'type': 'MultiPolygonAnnotation',
                        **annotation.to_dict()
                    }
                else:
                    raise ValueError(f"Unknown annotation type: {type(annotation)}")

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
