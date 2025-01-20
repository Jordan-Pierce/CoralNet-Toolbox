import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import json

from PyQt5.QtWidgets import (QApplication, QFileDialog, QMessageBox)

from coralnet_toolbox.QtLabelWindow import Label

from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ImportLabels:
    def __init__(self, main_window):
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

    def import_labels(self):
        self.main_window.untoggle_all_tools()

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self.label_window,
                                                   "Import Labels",
                                                   "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                # Open the file
                with open(file_path, 'r') as file:
                    labels_data = json.load(file)

            except Exception as e:
                QMessageBox.warning(self.label_window,
                                    "Error Loading Labels",
                                    f"An error occurred while loading Labels: {str(e)}")
                return

            # Create a progress bar
            total_labels = len(labels_data)
            progress_bar = ProgressBar(self.label_window, "Importing Labels")
            progress_bar.show()
            progress_bar.start_progress(total_labels)

            try:
                # Import the labels
                for label_data in labels_data:
                    label = Label.from_dict(label_data)
                    if not self.label_window.label_exists(label.short_label_code, label.long_label_code):
                        self.label_window.add_label(label.short_label_code,
                                                    label.long_label_code,
                                                    label.color,
                                                    label.id)
                    # Update the progress bar
                    progress_bar.update_progress()

                # Set the Review label as active
                self.label_window.set_active_label(self.label_window.get_label_by_long_code("Review"))

                QMessageBox.information(self.label_window,
                                        "Labels Imported",
                                        "Annotations have been successfully imported.")

            except Exception as e:
                QMessageBox.warning(self.label_window,
                                    "Error Importing Labels",
                                    f"An error occurred while importing Labels: {str(e)}")

            finally:
                # Stop the progress bar
                progress_bar.stop_progress()
                progress_bar.close()
                QApplication.restoreOverrideCursor()
