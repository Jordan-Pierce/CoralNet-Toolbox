import warnings

import ujson as json

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QApplication

from coralnet_toolbox.QtProgressBar import ProgressBar

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------
# Note: Field names are case-sensitive and should match the CoralNet export exactly.


class ImportTagLabLabels:
    def __init__(self, main_window):
        self.main_window = main_window
        self.label_window = main_window.label_window

    def import_taglab_labels(self):
        self.main_window.untoggle_all_tools()

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self.label_window,
                                                   "Import TagLab Labels",
                                                   "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)

                if 'Labels' in data:
                    data['labels'] = data.pop('Labels')
                elif 'labels' in data:
                    pass
                else:
                    raise Exception("The selected JSON file does not contain 'Labels' or 'labels' key.")

            except Exception as e:
                QMessageBox.warning(self.label_window,
                                    "Error Loading Labels",
                                    f"An error occurred while loading Labels: {str(e)}")
                return

            # Make cursor busy
            QApplication.setOverrideCursor(Qt.WaitCursor)
            total_labels = len(data['labels'])
            progress_bar = ProgressBar(self.label_window, "Importing TagLab Labels")
            progress_bar.show()
            progress_bar.start_progress(total_labels)

            try:
                # Import the labels
                for label_info in data['labels']:
                    try:
                        short_label_code = label_info['name'].strip()
                        long_label_code = label_info['name'].strip()
                        color = QColor(label_info['fill'][0], 
                                       label_info['fill'][1], 
                                       label_info['fill'][2])

                        # Add label if it does not exist
                        label = self.label_window.add_label_if_not_exists(short_label_code, 
                                                                          long_label_code, 
                                                                          color)

                    except Exception as e:
                        print(f"Warning: Could not import label {label_info['name']}: {str(e)}")

                    # Update the progress bar
                    progress_bar.update_progress()

                QMessageBox.information(self.label_window,
                                        "Labels Imported",
                                        "TagLab labels have been successfully imported.")

            except Exception as e:
                QMessageBox.warning(self.label_window,
                                    "Error Importing Labels",
                                    f"An error occurred while importing TagLab labels: {str(e)}")

            finally:
                # Stop the progress bar
                progress_bar.stop_progress()
                progress_bar.close()
                QApplication.restoreOverrideCursor()
