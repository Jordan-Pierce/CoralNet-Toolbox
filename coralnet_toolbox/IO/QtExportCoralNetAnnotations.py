import os

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox)

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.utilities import STATUS_SUMMARY_MS, plural, show_status


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------
# Note: Field names are case-sensitive and should match the CoralNet export exactly.


class ExportCoralNetAnnotations:
    def __init__(self, main_window):
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

    def export_annotations(self):
        self.main_window.untoggle_all_tools()

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self.annotation_window,
                                                   "Export CoralNet Annotations",
                                                   "",
                                                   "CSV Files (*.csv);;All Files (*)",
                                                   options=options)
        if file_path:
            
            # Make cursor busy
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            total_annotations = len(self.annotation_window.annotations_dict)
            show_status(self.main_window, f"Exporting {plural(total_annotations, 'CoralNet annotation')}...")
            progress_bar = ProgressBar(self.annotation_window, title="Exporting CoralNet Annotations")
            progress_bar.show()
            progress_bar.start_progress(total_annotations)

            try:
                df = []

                for annotation in self.annotation_window.annotations_dict.values():
                    df.append(annotation.to_coralnet())
                    progress_bar.update_progress()

                df = pd.DataFrame(df)
                df.to_csv(file_path, index=False)

                show_status(self.main_window,
                            f"Exported {plural(len(df), 'CoralNet annotation')} to {os.path.basename(file_path)}.",
                            STATUS_SUMMARY_MS)

                QMessageBox.information(self.annotation_window,
                                        "Annotations Exported",
                                        "Annotations have been successfully exported.")

            except Exception as e:
                show_status(self.main_window, f"CoralNet export failed: {e}", STATUS_SUMMARY_MS)
                QMessageBox.warning(self.annotation_window,
                                    "Error Exporting Annotations",
                                    f"An error occurred while exporting annotations: {str(e)}")
                
            finally:
                # Restore the cursor
                QApplication.restoreOverrideCursor()
                progress_bar.stop_progress()
                progress_bar.close()