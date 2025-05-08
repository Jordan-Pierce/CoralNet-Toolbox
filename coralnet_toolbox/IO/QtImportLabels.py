import warnings

import ujson as json

from PyQt5.QtWidgets import (QApplication, QFileDialog, QMessageBox)

from coralnet_toolbox.QtLabelWindow import Label

from coralnet_toolbox.QtProgressBar import ProgressBar

warnings.filterwarnings("ignore", category=DeprecationWarning)


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
                    labels = json.load(file)

            except Exception as e:
                QMessageBox.warning(self.label_window,
                                    "Error Loading Labels",
                                    f"An error occurred while loading Labels: {str(e)}")
                return

            # Create a progress bar
            total_labels = len(labels)
            progress_bar = ProgressBar(self.label_window, "Importing Labels")
            progress_bar.show()
            progress_bar.start_progress(total_labels)

            try:
                # Import the labels
                for label in labels:
                    # Create a new label object
                    label = Label.from_dict(label)

                    # Create a new label if it doesn't already exist
                    self.label_window.add_label_if_not_exists(label.short_label_code,
                                                              label.long_label_code,
                                                              label.color,
                                                              label.id)
                    # Update the progress bar
                    progress_bar.update_progress()

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
