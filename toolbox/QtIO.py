import json
import os
import random
import uuid
import warnings

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QInputDialog, QLineEdit, QDialog, QVBoxLayout,
                             QLabel, QHBoxLayout, QPushButton, QDialogButtonBox)

from toolbox.QtProgressBar import ProgressBar

from toolbox.QtLabelWindow import Label
from toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation

warnings.filterwarnings("ignore", category=DeprecationWarning)


class IODialog:
    def __init__(self, main_window):
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

    def import_images(self):
        file_names, _ = QFileDialog.getOpenFileNames(self.image_window,
                                                     "Open Image Files",
                                                     "",
                                                     "Image Files (*.png *.jpg *.jpeg *.tif* *.bmp)")
        if file_names:
            # Set the cursor to waiting (busy) cursor
            QApplication.setOverrideCursor(Qt.WaitCursor)

            progress_bar = ProgressBar(self.image_window, title="Importing Images")
            progress_bar.show()
            progress_bar.start_progress(len(file_names))
            progress_bar.set_value(1)

            for i, file_name in enumerate(file_names):
                if file_name not in set(self.image_window.image_paths):
                    self.image_window.add_image(file_name)
                progress_bar.update_progress()

            progress_bar.stop_progress()
            progress_bar.close()

            # Update filtered images
            self.image_window.filter_images()
            # Show the last image
            self.image_window.load_image_by_path(self.image_window.image_paths[-1])

            # Restore the cursor to the default cursor
            QApplication.restoreOverrideCursor()

            QMessageBox.information(self.image_window,
                                    "Image(s) Imported",
                                    "Image(s) have been successfully imported.")

    def export_labels(self):
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
                with open(file_path, 'r') as file:
                    labels_data = json.load(file)

                for label_data in labels_data:
                    label = Label.from_dict(label_data)
                    if not self.label_window.label_exists(label.short_label_code, label.long_label_code):
                        self.label_window.add_label(label.short_label_code,
                                                    label.long_label_code,
                                                    label.color,
                                                    label.id)

                # Set the Review label as active
                self.label_window.set_active_label(self.label_window.get_label_by_long_code("Review"))

                QMessageBox.information(self.label_window,
                                        "Labels Imported",
                                        "Annotations have been successfully imported.")

            except Exception as e:
                QMessageBox.warning(self.label_window,
                                    "Error Importing Labels",
                                    f"An error occurred while importing Labels: {str(e)}")

    def export_annotations(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self.annotation_window,
                                                   "Save Annotations",
                                                   "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)

                total_annotations = len(list(self.annotation_window.annotations_dict.values()))
                progress_bar = ProgressBar(self.annotation_window, title="Exporting Annotations")
                progress_bar.show()
                progress_bar.start_progress(total_annotations)

                export_dict = {}
                for annotation in self.annotation_window.annotations_dict.values():
                    image_path = annotation.image_path
                    if image_path not in export_dict:
                        export_dict[image_path] = []

                    # Convert annotation to dictionary based on its type
                    if isinstance(annotation, PatchAnnotation):
                        annotation_dict = {
                            'type': 'PatchAnnotation',
                            **annotation.to_dict()
                        }
                    elif isinstance(annotation, PolygonAnnotation):
                        annotation_dict = {
                            'type': 'PolygonAnnotation',
                            **annotation.to_dict()
                        }
                    elif isinstance(annotation, RectangleAnnotation):
                        annotation_dict = {
                            'type': 'RectangleAnnotation',
                            **annotation.to_dict()
                        }
                    else:
                        raise ValueError(f"Unknown annotation type: {type(annotation)}")

                    export_dict[image_path].append(annotation_dict)
                    progress_bar.update_progress()

                with open(file_path, 'w') as file:
                    json.dump(export_dict, file, indent=4)
                    file.flush()

                progress_bar.stop_progress()
                progress_bar.close()

                QMessageBox.information(self.annotation_window,
                                        "Annotations Exported",
                                        "Annotations have been successfully exported.")

            except Exception as e:
                QMessageBox.warning(self.annotation_window,
                                    "Error Exporting Annotations",
                                    f"An error occurred while exporting annotations: {str(e)}")

            QApplication.restoreOverrideCursor()

    def import_annotations(self):
        self.main_window.untoggle_all_tools()

        if not self.annotation_window.active_image:
            QMessageBox.warning(self.annotation_window,
                                "No Images Loaded",
                                "Please load images first before importing annotations.")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self.annotation_window,
                                                   "Load Annotations",
                                                   "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)

                with open(file_path, 'r') as file:
                    data = json.load(file)

                keys = ['label_short_code', 'label_long_code', 'annotation_color', 'image_path', 'label_id']

                filtered_annotations = {p: a for p, a in data.items() if p in self.image_window.image_paths}
                total_annotations = sum(len(annotations) for annotations in filtered_annotations.values())

                progress_bar = ProgressBar(self.annotation_window, title="Importing Annotations")
                progress_bar.show()
                progress_bar.start_progress(total_annotations)

                updated_annotations = False

                for image_path, annotations in filtered_annotations.items():
                    for annotation_data in annotations:
                        if not all(key in annotation_data for key in keys):
                            continue

                        short_label = annotation_data['label_short_code']
                        long_label = annotation_data['label_long_code']
                        color = QColor(*annotation_data['annotation_color'])

                        label_id = annotation_data['label_id']
                        self.label_window.add_label_if_not_exists(short_label, long_label, color, label_id)

                        existing_color = self.label_window.get_label_color(short_label, long_label)

                        if existing_color != color:
                            annotation_data['annotation_color'] = existing_color.getRgb()
                            updated_annotations = True

                        progress_bar.update_progress()

                if updated_annotations:
                    QMessageBox.information(self.annotation_window,
                                            "Annotations Updated",
                                            "Some annotations have been updated to match the "
                                            "color of the labels already in the project.")

                for image_path, annotations in filtered_annotations.items():
                    for annotation_data in annotations:
                        if not all(key in annotation_data for key in keys):
                            continue

                        annotation_type = annotation_data.get('type')
                        if annotation_type == 'PatchAnnotation':
                            annotation = PatchAnnotation.from_dict(annotation_data, self.label_window)
                        elif annotation_type == 'PolygonAnnotation':
                            annotation = PolygonAnnotation.from_dict(annotation_data, self.label_window)
                        elif annotation_type == 'RectangleAnnotation':
                            annotation = RectangleAnnotation.from_dict(annotation_data, self.label_window)
                        else:
                            raise ValueError(f"Unknown annotation type: {annotation_type}")

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

    def export_coralnet_annotations(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self.annotation_window,
                                                   "Export CoralNet Annotations",
                                                   "",
                                                   "CSV Files (*.csv);;All Files (*)",
                                                   options=options)
        if file_path:

            QApplication.setOverrideCursor(Qt.WaitCursor)
            progress_bar = ProgressBar(self.annotation_window, title="Exporting CoralNet Annotations")
            progress_bar.show()
            progress_bar.start_progress(len(self.annotation_window.annotations_dict))

            try:
                df = []

                for annotation in self.annotation_window.annotations_dict.values():
                    df.append(annotation.to_coralnet())
                    progress_bar.update_progress()

                df = pd.DataFrame(df)
                df.to_csv(file_path, index=False)

                QMessageBox.information(self.annotation_window,
                                        "Annotations Exported",
                                        "Annotations have been successfully exported.")

            except Exception as e:
                QMessageBox.warning(self.annotation_window,
                                    "Error Exporting Annotations",
                                    f"An error occurred while exporting annotations: {str(e)}")

            progress_bar.stop_progress()
            progress_bar.close()
            QApplication.restoreOverrideCursor()

    def import_coralnet_annotations(self):
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

            annotation_size, ok = QInputDialog.getInt(self.annotation_window,
                                                      "Annotation Size",
                                                      "Enter the default annotation size for imported annotations:",
                                                      224, 1, 10000, 1)
            if not ok:
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
            QMessageBox.warning(self.annotation_window,
                                "Error Importing Annotations",
                                f"An error occurred while importing annotations: {str(e)}")

        QApplication.restoreOverrideCursor()

    def export_viscore_annotations(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self.annotation_window,
                                                   "Export Viscore Annotations",
                                                   "",
                                                   "CSV Files (*.csv);;All Files (*)",
                                                   options=options)
        if file_path:

            QApplication.setOverrideCursor(Qt.WaitCursor)
            progress_bar = ProgressBar(self.annotation_window, title="Exporting Viscore Annotations")
            progress_bar.show()
            progress_bar.start_progress(len(self.annotation_window.annotations_dict))

            try:
                df = []

                for annotation in self.annotation_window.annotations_dict.values():
                    if isinstance(annotation, PatchAnnotation):
                        if 'Dot' in annotation.data:
                            df.append(annotation.to_coralnet())
                            progress_bar.update_progress()

                df = pd.DataFrame(df)
                df.to_csv(file_path, index=False)

                QMessageBox.information(self.annotation_window,
                                        "Viscore Annotations Exported",
                                        "Annotations have been successfully exported.")

            except Exception as e:
                QMessageBox.warning(self.annotation_window,
                                    "Error Exporting Viscore Annotations",
                                    f"An error occurred while exporting annotations: {str(e)}")

            progress_bar.stop_progress()
            progress_bar.close()
            QApplication.restoreOverrideCursor()

    def import_viscore_annotations(self):
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

    def taglabToPoints(self, c):
        d = (c * 10).astype(int)
        d = np.diff(d, axis=0, prepend=[[0, 0]])
        d = np.reshape(d, -1)
        d = np.char.mod('%d', d)
        d = " ".join(d)
        return d

    def taglabToContour(self, p):
        if type(p) is str:
            p = map(int, p.split(' '))
            c = np.fromiter(p, dtype=int)
        else:
            c = np.asarray(p)

        if len(c.shape) == 2:
            return c

        c = np.reshape(c, (-1, 2))
        c = np.cumsum(c, axis=0)
        c = c / 10.0
        return c

    def export_taglab_annotations(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self.annotation_window,
                                                   "Export TagLab Annotations",
                                                   "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)

                total_annotations = len(list(self.annotation_window.annotations_dict.values()))
                progress_bar = ProgressBar(self.annotation_window, title="Exporting TagLab Annotations")
                progress_bar.show()
                progress_bar.start_progress(total_annotations)

                taglab_data = {
                    "filename": file_path,
                    "working_area": None,
                    "dictionary_name": "scripps",
                    "dictionary_description": "These color codes are the ones typically used by the "
                                              "Scripps Institution of Oceanography (UCSD)",
                    "labels": {},
                    "images": []
                }

                # Collect all labels
                for annotation in self.annotation_window.annotations_dict.values():
                    label_id = annotation.label.short_label_code
                    if label_id not in taglab_data["labels"]:
                        label_info = {
                            "id": label_id,
                            "name": label_id,
                            "description": None,
                            "fill": annotation.label.color.getRgb()[:3],
                            "border": [200, 200, 200],
                            "visible": True
                        }
                        taglab_data["labels"][label_id] = label_info

                # Collect all images and their annotations
                image_annotations = {}
                for idx, annotation in enumerate(self.annotation_window.annotations_dict.values()):
                    if not isinstance(annotation, PolygonAnnotation):
                        continue

                    image_path = annotation.image_path
                    if image_path not in image_annotations:
                        image_annotations[image_path] = {
                            "rect": [0.0, 0.0, 0.0, 0.0],
                            "map_px_to_mm_factor": "1",
                            "width": 0,
                            "height": 0,
                            "annotations": [],
                            "layers": [],
                            "channels": [
                                {"filename": image_path, "type": "RGB"}
                            ],
                            "id": os.path.basename(image_path),
                            "name": os.path.basename(image_path),
                            "workspace": [],
                            "export_dataset_area": [],
                            "acquisition_date": "2000-01-01",
                            "georef_filename": "",
                            "metadata": {},
                            "grid": None
                        }

                    # Calculate bounding box, centroid, area, perimeter, and contour
                    points = annotation.points
                    min_x = min(point.x() for point in points)
                    min_y = min(point.y() for point in points)
                    max_x = max(point.x() for point in points)
                    max_y = max(point.y() for point in points)
                    centroid_x = sum(point.x() for point in points) / len(points)
                    centroid_y = sum(point.y() for point in points) / len(points)
                    area = annotation.calculate_polygon_area()
                    perimeter = annotation.calculate_polygon_perimeter()
                    contour = self.taglabToPoints(np.array([[point.x(), point.y()] for point in points]))

                    annotation_dict = {
                        "bbox": [min_x, min_y, max_x, max_y],
                        "centroid": [centroid_x, centroid_y],
                        "area": area,
                        "perimeter": perimeter,
                        "contour": contour,
                        "inner contours": [],
                        "class name": annotation.label.short_label_code,
                        "instance name": "coral0",  # Placeholder, update as needed
                        "blob name": f"c-0-{centroid_x}x-{centroid_y}y",
                        # Placeholder, update as needed
                        "id": idx,
                        "note": "",  # Placeholder, update as needed
                        "data": {}  # Placeholder, update as needed
                    }
                    image_annotations[image_path]["annotations"].append(annotation_dict)
                    progress_bar.update_progress()

                # Add images to the main data structure
                taglab_data["images"] = list(image_annotations.values())

                # Save the JSON data to the selected file
                with open(file_path, 'w') as file:
                    json.dump(taglab_data, file, indent=4)
                    file.flush()

                progress_bar.stop_progress()
                progress_bar.close()

                QMessageBox.information(self.annotation_window,
                                        "Annotations Exported",
                                        "Annotations have been successfully exported.")

            except Exception as e:
                QMessageBox.warning(self.annotation_window,
                                    "Error Exporting Annotations",
                                    f"An error occurred while exporting annotations: {str(e)}")

            QApplication.restoreOverrideCursor()

    def import_taglab_annotations(self):

        def parse_contour(contour_str):
            """Parse the contour string into a list of QPointF objects."""
            points = self.taglabToContour(contour_str)
            return [QPointF(x, y) for x, y in points]

        self.main_window.untoggle_all_tools()

        if not self.annotation_window.active_image:
            QMessageBox.warning(self.annotation_window,
                                "No Images Loaded",
                                "Please load images first before importing annotations.")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self.annotation_window,
                                                   "Import TagLab Annotations",
                                                   "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)

        if not file_path:
            return

        try:
            with open(file_path, 'r') as file:
                taglab_data = json.load(file)

            required_keys = ['labels', 'images']
            if not all(key in taglab_data for key in required_keys):
                QMessageBox.warning(self.annotation_window,
                                    "Invalid JSON Format",
                                    "The selected JSON file does not match the expected TagLab format.")
                return

            # Map image names to image paths
            image_path_map = {os.path.basename(path): path for path in self.image_window.image_paths}

            progress_bar = ProgressBar(self.annotation_window, title="Importing TagLab Annotations")
            progress_bar.show()
            progress_bar.start_progress(len(taglab_data['images']))

            QApplication.setOverrideCursor(Qt.WaitCursor)

            for image in taglab_data['images']:
                image_basename = os.path.basename(image['channels'][0]['filename'])
                image_full_path = image_path_map[image_basename]

                if not image_full_path:
                    QMessageBox.warning(self.annotation_window,
                                        "Image Not Found",
                                        f"The image '{image_basename}' "
                                        f"from the TagLab annotations was not found in the project.")
                    continue

                for annotation in list(image['annotations']['regions']):
                    label_id = annotation['class name']
                    label_info = taglab_data['labels'][label_id]
                    short_label_code = label_info['name']
                    long_label_code = label_info['name']
                    color = QColor(*label_info['fill'])

                    # Convert contour string to points
                    points = parse_contour(annotation['contour'])

                    existing_label = self.label_window.get_label_by_codes(short_label_code, long_label_code)

                    if existing_label:
                        label_id = existing_label.id
                    else:
                        label_id = str(uuid.uuid4())
                        self.label_window.add_label_if_not_exists(short_label_code, long_label_code, color, label_id)

                    polygon_annotation = PolygonAnnotation(
                        points=points,
                        short_label_code=short_label_code,
                        long_label_code=long_label_code,
                        color=color,
                        image_path=image_full_path,
                        label_id=label_id
                    )

                    # Add annotation to the dict
                    self.annotation_window.annotations_dict[polygon_annotation.id] = polygon_annotation
                    progress_bar.update_progress()

                # Update the image window's image dict
                self.image_window.update_image_annotations(image_full_path)

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