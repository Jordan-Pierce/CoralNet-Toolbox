import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import ujson as json

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QApplication

from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------
# Note: Field names are case-sensitive and should match the CoralNet export exactly.


class ExportTagLabLabels:
    def __init__(self, main_window):
        self.main_window = main_window
        self.label_window = main_window.label_window

    def export_taglab_labels(self):
        self.main_window.untoggle_all_tools()

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self.label_window,
                                                   "Export TagLab Labels",
                                                   "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_path:
            # Make cursor busy
            QApplication.setOverrideCursor(Qt.WaitCursor)
            total_labels = len(self.label_window.labels)
            progress_bar = ProgressBar(self.label_window, "Exporting TagLab Labels")
            progress_bar.show()
            progress_bar.start_progress(total_labels)

            try:
                # Convert labels to list format
                labels_list = []
                for label in self.label_window.labels:
                    label_entry = {
                        'id': f"{label.short_label_code}",
                        'name': f"{label.short_label_code}",
                        'description': None,
                        'fill': label.color.getRgb()[:3],
                        'border': [200, 200, 200],
                        'visible': True
                    }
                    labels_list.append(label_entry)
                    progress_bar.update_progress()

                taglab_data = {
                    'Name': 'custom_dictionary',
                    'Description': 'This label dictionary was exported from CoralNet-Toolbox.',
                    'Labels': labels_list
                }

                with open(file_path, 'w') as file:
                    json.dump(taglab_data, file, indent=4)

                QMessageBox.information(self.label_window,
                                        "Labels Exported",
                                        "TagLab labels have been successfully exported.")

            except Exception as e:
                QMessageBox.warning(self.label_window,
                                    "Error Exporting Labels",
                                    f"An error occurred while exporting TagLab labels: {str(e)}")

            finally:
                QApplication.restoreOverrideCursor()
                progress_bar.stop_progress()
                progress_bar.close()
