import json
import warnings

from PyQt5.QtWidgets import (QFileDialog, QMessageBox)

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
            try:
                labels_data = [label.to_dict() for label in self.label_window.labels]
                with open(file_path, 'w') as file:
                    json.dump(labels_data, file, indent=4)

                QMessageBox.information(self.label_window,
                                        "Labels Exported",
                                        "Labels have been successfully exported.")

            except Exception as e:
                QMessageBox.warning(self.label_window,
                                    "Error Importing Labels",
                                    f"An error occurred while importing labels: {str(e)}")