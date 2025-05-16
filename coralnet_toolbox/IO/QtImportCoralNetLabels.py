import warnings

import uuid
import random
import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QApplication

from coralnet_toolbox.QtProgressBar import ProgressBar

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------
# Note: Field names are case-sensitive and should match the CoralNet export exactly.


class ImportCoralNetLabels:
    def __init__(self, main_window):
        self.main_window = main_window
        self.label_window = main_window.label_window

    def import_coralnet_labels(self):
        self.main_window.untoggle_all_tools()

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self.label_window,
                                                   "Import CoralNet Labels",
                                                   "",
                                                   "CSV Files (*.csv);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                data = pd.read_csv(file_path)

                if 'Label ID' not in data.columns:
                    raise Exception("The selected CSV file does not contain 'Label ID' column.")
                if 'Name' not in data.columns:
                    raise Exception("The selected CSV file does not contain 'Name' column.")
                if 'Short Code' not in data.columns:
                    raise Exception("The selected CSV file does not contain 'Short Code' column.")

            except Exception as e:
                QMessageBox.warning(self.label_window,
                                    "Error Loading Labels",
                                    f"An error occurred while loading Labels: {str(e)}")
                return

            # Make cursor busy
            QApplication.setOverrideCursor(Qt.WaitCursor)
            total_labels = len(data)
            progress_bar = ProgressBar(self.label_window, "Importing CoralNet Labels")
            progress_bar.show()
            progress_bar.start_progress(total_labels)

            try:
                # Import the labels
                for i, r in data.iterrows():
                    try:
                        # CoralNet is inconsistent with label names in Labeset vs Annotations; drop Name
                        short_label_code = r['Short Code'].strip()
                        
                        # Create a QtColor object from the color string
                        label = self.label_window.add_label_if_not_exists(short_label_code,
                                                                          long_label_code=None,
                                                                          color=None,
                                                                          label_id=None)

                    except Exception as e:
                        print(f"Warning: Could not import label {r}: {str(e)}")

                    # Update the progress bar
                    progress_bar.update_progress()

                QMessageBox.information(self.label_window,
                                        "Labels Imported",
                                        "CoralNet labels have been successfully imported.")

            except Exception as e:
                QMessageBox.warning(self.label_window,
                                    "Error Importing Labels",
                                    f"An error occurred while importing CoralNet labels: {str(e)}")

            finally:
                # Stop the progress bar
                progress_bar.stop_progress()
                progress_bar.close()
                QApplication.restoreOverrideCursor()
