import warnings

import ujson as json

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QMessageBox, QApplication)

from coralnet_toolbox.QtProgressBar import ProgressBar

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ExportLabels:
    def __init__(self, main_window):
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

    def export_labels(self):
        self.main_window.untoggle_all_tools()

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self.label_window,
                                                   "Export Labels",
                                                   "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_path:
            # Make cursor busy
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # Create a progress bar
            total_labels = len(self.label_window.labels)
            progress_bar = ProgressBar(self.label_window, "Exporting Labels")
            progress_bar.show()
            progress_bar.start_progress(total_labels)

            try:
                labels_data = []
                for i, label in enumerate(self.label_window.labels):
                    labels_data.append(label.to_dict())
                    progress_bar.update_progress()

                with open(file_path, 'w') as file:
                    json.dump(labels_data, file, indent=4)

                QMessageBox.information(self.label_window,
                                        "Labels Exported",
                                        "Labels have been successfully exported.")

            except Exception as e:
                QMessageBox.warning(self.label_window,
                                    "Error Importing Labels",
                                    f"An error occurred while importing labels: {str(e)}")
            finally:
                # Reset the cursor
                QApplication.restoreOverrideCursor()
                progress_bar.stop_progress()
                progress_bar.close()
