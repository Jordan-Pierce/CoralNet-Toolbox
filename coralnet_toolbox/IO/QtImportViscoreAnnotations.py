import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import random
import uuid

import pandas as pd
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QInputDialog, QLineEdit, QDialog, QVBoxLayout,
                             QLabel, QHBoxLayout, QPushButton, QDialogButtonBox)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.QtProgressBar import ProgressBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ImportViscoreAnnotations:
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

        def browse_csv_file(file_path_input):
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(self.annotation_window,
                                                       "Import Viscore Annotations",
                                                       "",
                                                       "CSV Files (*.csv);;All Files (*)",
                                                       options=options)
            if file_path:
                file_path_input.setText(file_path)

        dialog = QDialog(self.annotation_window)
        dialog.setWindowTitle("Import Viscore Annotations")
        dialog.resize(500, 200)

        layout = QVBoxLayout(dialog)

        file_path_label = QLabel("CSV File Path:")
        file_path_input = QLineEdit()
        file_path_button = QPushButton("Browse")
        file_path_button.clicked.connect(lambda: browse_csv_file(file_path_input))
        file_path_layout = QHBoxLayout()
        file_path_layout.addWidget(file_path_input)
        file_path_layout.addWidget(file_path_button)
        layout.addWidget(file_path_label)
        layout.addLayout(file_path_layout)

        reprojection_error_label = QLabel("ReprojectionError (Default: 0.01, float):")
        reprojection_error_input = QLineEdit()
        reprojection_error_input.setPlaceholderText("Error between an image point, reprojected to its 3D dot location")
        layout.addWidget(reprojection_error_label)
        layout.addWidget(reprojection_error_input)

        view_index_label = QLabel("ViewIndex (Default: 10, int):")
        view_index_input = QLineEdit()
        view_index_input.setPlaceholderText("The image's index in VPI view (includes a form pre-filtering)")
        layout.addWidget(view_index_label)
        layout.addWidget(view_index_input)

        view_count_label = QLabel("ViewCount (Default: 5, int):")
        view_count_input = QLineEdit()
        view_count_input.setPlaceholderText("The number of images the dot has been seen in")
        layout.addWidget(view_count_label)
        layout.addWidget(view_count_input)

        rand_sub_ceil_label = QLabel("RandSubCeil (Default: 1.0, float, [0-1]):")
        rand_sub_ceil_input = QLineEdit()
        rand_sub_ceil_input.setPlaceholderText("Randomly sample a subset of the data")
        layout.addWidget(rand_sub_ceil_label)
        layout.addWidget(rand_sub_ceil_input)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec_() == QDialog.Accepted:
            file_path = file_path_input.text()

            reprojection_error = reprojection_error_input.text()
            if not reprojection_error:
                reprojection_error = "0.01"
            try:
                reprojection_error = float(reprojection_error)
                if reprojection_error < 0:
                    raise ValueError("ReprojectionError must be a non-negative float.")
            except ValueError as e:
                QMessageBox.warning(self.annotation_window, "Invalid Input", f"Invalid ReprojectionError: {e}")
                return

            view_index = view_index_input.text()
            if not view_index:
                view_index = "10"
            try:
                view_index = int(view_index)
                if view_index < 0:
                    raise ValueError("ViewIndex must be a non-negative integer.")
            except ValueError as e:
                QMessageBox.warning(self.annotation_window, "Invalid Input", f"Invalid ViewIndex: {e}")
                return

            view_count = view_count_input.text()
            if not view_count:
                view_count = "5"
            try:
                view_count = int(view_count)
                if view_count < 0:
                    raise ValueError("ViewCount must be a non-negative integer.")
            except ValueError as e:
                QMessageBox.warning(self.annotation_window, "Invalid Input", f"Invalid ViewCount: {e}")
                return

            rand_sub_ceil = rand_sub_ceil_input.text()
            if not rand_sub_ceil:
                rand_sub_ceil = "1.0"
            try:
                rand_sub_ceil = float(rand_sub_ceil)
                if not (0 <= rand_sub_ceil <= 1):
                    raise ValueError("RandSubCeil must be a float between 0 and 1.")
            except ValueError as e:
                QMessageBox.warning(self.annotation_window, "Invalid Input", f"Invalid RandSubCeil: {e}")
                return

            try:
                progress_bar = ProgressBar(self.annotation_window, title="Reading CSV File")
                progress_bar.show()
                df = pd.read_csv(file_path, index_col=False)
                progress_bar.close()

                if df.empty:
                    QMessageBox.warning(self.annotation_window, "Empty CSV", "The CSV file is empty.")
                    return

                required_columns = ['Name',
                                    'Row',
                                    'Column',
                                    'Label',
                                    'Dot']

                if not all(col in df.columns for col in required_columns):
                    QMessageBox.warning(self.annotation_window,
                                        "Invalid CSV Format",
                                        "The selected CSV file does not match the expected Viscore format.")
                    return

                df = df[required_columns]
                df = df.dropna(how='any')
                df = df.assign(Row=df['Row'].astype(int))
                df = df.assign(Column=df['Column'].astype(int))

                image_paths = df['Name'].unique()
                image_paths = [path for path in image_paths if os.path.exists(path)]

                if not image_paths and not self.annotation_window.active_image:
                    QMessageBox.warning(self.annotation_window,
                                        "No Images Found",
                                        "Please load an image before importing annotations.")
                    return

                # Perform the filtering
                if 'RandSubCeil' in df.columns:
                    df = df[df['RandSubCeil'] <= rand_sub_ceil]
                if 'ReprojectionError' in df.columns:
                    df = df[df['ReprojectionError'] <= reprojection_error]
                if 'ViewIndex' in df.columns:
                    df = df[df['ViewIndex'] <= view_index]
                if 'ViewCount' in df.columns:
                    df = df[df['ViewCount'] >= view_count]

                num_images = df['Name'].nunique()
                num_annotations = len(df)

                msg_box = QMessageBox(self.annotation_window)
                msg_box.setWindowTitle("Filtered Data Summary")
                msg_box.setText(f"Number of Images: {num_images}\nNumber of Annotations: {num_annotations}")
                msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                msg_box.setDefaultButton(QMessageBox.Ok)

                result = msg_box.exec_()

                if result == QMessageBox.Cancel:
                    self.import_viscore_annotations()
                    return

                annotation_size, ok = QInputDialog.getInt(self.annotation_window,
                                                          "Annotation Size",
                                                          "Enter the annotation size for all imported annotations:",
                                                          224, 1, 10000, 1)
                if not ok:
                    return

                # Start the import process
                QApplication.setOverrideCursor(Qt.WaitCursor)
                progress_bar = ProgressBar(self.annotation_window, title="Importing Viscore Annotations")
                progress_bar.show()
                progress_bar.start_progress(len(df))

                # Map image names to image paths
                image_path_map = {os.path.basename(path): path for path in self.image_window.image_paths}

                for image_name, group in df.groupby('Name'):
                    image_path = image_path_map.get(image_name)
                    if not image_path:
                        continue

                    for index, row in group.iterrows():
                        row_coord = row['Row']
                        col_coord = row['Column']
                        label_code = row['Label']

                        short_label_code = long_label_code = label_code
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
                                                     annotation_size,
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

                        if machine_confidence:
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
                QMessageBox.critical(self.annotation_window, "Critical Error", f"Failed to import annotations: {e}")

        # Make the cursor active
        QApplication.restoreOverrideCursor()