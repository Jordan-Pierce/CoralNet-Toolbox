import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ExportViscoreAnnotations:
    def __init__(self, main_window):
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

    def export_annotations(self):
        self.main_window.untoggle_all_tools()

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self.annotation_window,
                                                   "Export Viscore Annotations",
                                                   "",
                                                   "CSV Files (*.csv);;All Files (*)",
                                                   options=options)
        if file_path:

            QApplication.setOverrideCursor(Qt.WaitCursor)
            progress_bar = ProgressBar(self.annotation_window, title="Exporting Viscore Annotations")
            progress_bar.show()
            progress_bar.start_progress(len(self.annotation_window.annotations_dict))

            try:
                df = []

                for annotation in self.annotation_window.annotations_dict.values():
                    if isinstance(annotation, PatchAnnotation):
                        if 'Dot' in annotation.data:
                            df.append(annotation.to_coralnet())
                            progress_bar.update_progress()

                df = pd.DataFrame(df)
                df.to_csv(file_path, index=False)

                QMessageBox.information(self.annotation_window,
                                        "Viscore Annotations Exported",
                                        "Annotations have been successfully exported.")

            except Exception as e:
                QMessageBox.warning(self.annotation_window,
                                    "Error Exporting Viscore Annotations",
                                    f"An error occurred while exporting annotations: {str(e)}")

            progress_bar.stop_progress()
            progress_bar.close()
            QApplication.restoreOverrideCursor()