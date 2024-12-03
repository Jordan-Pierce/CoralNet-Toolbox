import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import random
import uuid

import pandas as pd
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QInputDialog)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ImportCoralNetAnnotations:
    def __init__(self, main_window):
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

    def import_annotations(self):
        self.main_window.untoggle_all_tools()

        if not self.annotation_window.active_image:
            QMessageBox.warning(self.annotation_window,
                                "No Images Loaded",
                                "Please load images first before importing annotations.")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self.annotation_window,
                                                   "Import CoralNet Annotations",
                                                   "",
                                                   "CSV Files (*.csv);;All Files (*)",
                                                   options=options)

        if not file_path:
            return

        annotation_size, ok = QInputDialog.getInt(self.annotation_window,
                                                  "Patch Annotation Size",
                                                  "Enter the default patch annotation size for imported annotations:",
                                                  224, 1, 10000, 1)
        if not ok:
            return

        try:
            progress_bar = ProgressBar(self.annotation_window, title="Importing Annotations")
            progress_bar.show()
            df = pd.read_csv(file_path)
            progress_bar.close()

            required_columns = ['Name', 'Row', 'Column', 'Label']
            if not all(col in df.columns for col in required_columns):
                QMessageBox.warning(self.annotation_window,
                                    "Invalid CSV Format",
                                    "The selected CSV file does not match the expected CoralNet format.")
                return

            image_path_map = {os.path.basename(path): path for path in self.image_window.image_paths}
            df['Name'] = df['Name'].apply(lambda x: os.path.basename(x))
            df = df[df['Name'].isin(image_path_map.keys())]
            df = df.dropna(how='any', subset=['Row', 'Column', 'Label'])
            df = df.assign(Row=df['Row'].astype(int))
            df = df.assign(Column=df['Column'].astype(int))

            if df.empty:
                raise Exception("No annotations found for loaded images.")

            # Start the import process
            progress_bar = ProgressBar(self.annotation_window, title="Importing CoralNet Annotations")
            progress_bar.show()
            progress_bar.start_progress(len(df))

            QApplication.setOverrideCursor(Qt.WaitCursor)

            for image_name, group in df.groupby('Name'):
                image_path = image_path_map.get(image_name)
                if not image_path:
                    continue

                for index, row in group.iterrows():
                    row_coord = row['Row']
                    col_coord = row['Column']
                    label_code = row['Label']

                    short_label_code = label_code
                    long_label_code = row['Long Label'] if 'Long Label' in row else label_code

                    existing_label = self.label_window.get_label_by_codes(short_label_code, long_label_code)

                    if existing_label:
                        color = existing_label.color
                        label_id = existing_label.id
                    else:
                        label_id = str(uuid.uuid4())
                        color = QColor(random.randint(0, 255),
                                       random.randint(0, 255),
                                       random.randint(0, 255))

                        self.label_window.add_label_if_not_exists(short_label_code,
                                                                  long_label_code,
                                                                  color,
                                                                  label_id)

                    annotation = PatchAnnotation(QPointF(col_coord, row_coord),
                                                 row['Patch Size'] if "Patch Size" in row else annotation_size,
                                                 short_label_code,
                                                 long_label_code,
                                                 color,
                                                 image_path,
                                                 label_id)

                    machine_confidence = {}

                    for i in range(1, 6):
                        confidence_col = f'Machine confidence {i}'
                        suggestion_col = f'Machine suggestion {i}'
                        if confidence_col in row and suggestion_col in row:
                            if pd.isna(row[confidence_col]) or pd.isna(row[suggestion_col]):
                                continue

                            confidence = float(row[confidence_col])
                            suggestion = str(row[suggestion_col])

                            suggested_label = self.label_window.get_label_by_short_code(suggestion)

                            if not suggested_label:
                                color = QColor(random.randint(0, 255),
                                               random.randint(0, 255),
                                               random.randint(0, 255))

                                self.label_window.add_label_if_not_exists(suggestion, suggestion, color)

                            suggested_label = self.label_window.get_label_by_short_code(suggestion)
                            machine_confidence[suggested_label] = confidence

                    # Update the machine confidence
                    annotation.update_machine_confidence(machine_confidence)

                    # Add annotation to the dict
                    self.annotation_window.annotations_dict[annotation.id] = annotation
                    progress_bar.update_progress()

                # Update the image window's image dict
                self.image_window.update_image_annotations(image_path)

            # Load the annotations for current image
            self.annotation_window.load_annotations()

            # Stop the progress bar
            progress_bar.stop_progress()
            progress_bar.close()

            QMessageBox.information(self.annotation_window,
                                    "Annotations Imported",
                                    "Annotations have been successfully imported.")

        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Importing Annotations",
                                f"An error occurred while importing annotations: {str(e)}")

        QApplication.restoreOverrideCursor()