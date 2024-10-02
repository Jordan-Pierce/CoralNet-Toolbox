import json
import os
import random
import uuid
import warnings

import pandas as pd
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QInputDialog, QLineEdit, QDialog, QVBoxLayout,
                             QLabel, QHBoxLayout, QPushButton, QDialogButtonBox)

from toolbox.QtProgressBar import ProgressBar

from toolbox.QtLabelWindow import Label
from toolbox.QtPatchAnnotation import PatchAnnotation

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
                    export_dict[image_path].append(annotation.to_dict())

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
                        self.label_window.add_label_if_not_exists(short_label,
                                                                  long_label,
                                                                  color,
                                                                  label_id)

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
                        annotation = PatchAnnotation.from_dict(annotation_data)
                        self.annotation_window.annotations_dict[annotation.id] = annotation
                        progress_bar.update_progress()

                    self.image_window.update_image_annotations(image_path)

                progress_bar.stop_progress()
                progress_bar.close()

                self.annotation_window.load_annotations()

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
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)

                data = []
                total_annotations = len(self.annotation_window.annotations_dict)

                progress_bar = ProgressBar(self.annotation_window, title="Exporting CoralNet Annotations")
                progress_bar.show()
                progress_bar.start_progress(total_annotations)

                for annotation in self.annotation_window.annotations_dict.values():
                    data.append(annotation.to_coralnet_format())
                    progress_bar.update_progress()

                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False)

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
                                                      "Enter the annotation size for all imported annotations:",
                                                      224, 1, 10000, 1)
            if not ok:
                return

            image_path_map = {os.path.basename(path): path for path in self.image_window.image_paths}
            df = df[df['Name'].isin(image_path_map.keys())]
            df = df.dropna(how='any', subset=['Row', 'Column', 'Label'])
            df = df.assign(Row=df['Row'].astype(int))
            df = df.assign(Column=df['Column'].astype(int))

            if df.empty:
                raise Exception("No annotations found for loaded images.")

            total_annotations = len(df)

            progress_bar = ProgressBar(self.annotation_window, title="Importing CoralNet Annotations")
            progress_bar.show()
            progress_bar.start_progress(total_annotations)

            QApplication.setOverrideCursor(Qt.WaitCursor)

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

                    self.annotation_window.annotations_dict[annotation.id] = annotation
                    progress_bar.update_progress()

                self.image_window.update_image_annotations(image_path)

            progress_bar.stop_progress()
            progress_bar.close()

            self.annotation_window.load_annotations()

            QMessageBox.information(self.annotation_window,
                                    "Annotations Imported",
                                    "Annotations have been successfully imported.")

        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Importing Annotations",
                                f"An error occurred while importing annotations: {str(e)}")

        QApplication.restoreOverrideCursor()

    def export_viscore_annotations(self):

        def get_qclass_mapping(qclasses_data, use_short_code=True):
            qclass_mapping = {}
            for item in qclasses_data['classlist']:
                id_number, short_code, long_code = item
                if use_short_code:
                    qclass_mapping[short_code] = id_number
                else:
                    qclass_mapping[long_code] = id_number
            return qclass_mapping

        def browse_user_file(user_file_input):
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(self.annotation_window,
                                                       "Select User File",
                                                       "",
                                                       "JSON Files (*.json);;All Files (*)",
                                                       options=options)
            if file_path:
                user_file_input.setText(file_path)

        def browse_qclasses_file(qclasses_file_input):
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(self.annotation_window,
                                                       "Select QClasses File",
                                                       "",
                                                       "JSON Files (*.json);;All Files (*)",
                                                       options=options)
            if file_path:
                qclasses_file_input.setText(file_path)

        def browse_output_directory(output_directory_input):
            options = QFileDialog.Options()
            directory = QFileDialog.getExistingDirectory(self.annotation_window,
                                                         "Select Output Directory",
                                                         "",
                                                         options=options)
            if directory:
                output_directory_input.setText(directory)

        dialog = QDialog(self.annotation_window)
        dialog.setWindowTitle("Export Viscore Annotations")
        dialog.resize(400, 300)

        layout = QVBoxLayout(dialog)

        user_file_label = QLabel("User File (JSON):")
        user_file_input = QLineEdit()
        user_file_button = QPushButton("Browse")
        user_file_button.clicked.connect(lambda: browse_user_file(user_file_input))
        user_file_layout = QHBoxLayout()
        user_file_layout.addWidget(user_file_input)
        user_file_layout.addWidget(user_file_button)
        layout.addWidget(user_file_label)
        layout.addLayout(user_file_layout)

        qclasses_file_label = QLabel("QClasses File (JSON):")
        qclasses_file_input = QLineEdit()
        qclasses_file_button = QPushButton("Browse")
        qclasses_file_button.clicked.connect(lambda: browse_qclasses_file(qclasses_file_input))
        qclasses_file_layout = QHBoxLayout()
        qclasses_file_layout.addWidget(qclasses_file_input)
        qclasses_file_layout.addWidget(qclasses_file_button)
        layout.addWidget(qclasses_file_label)
        layout.addLayout(qclasses_file_layout)

        username_label = QLabel("Username:")
        username_input = QLineEdit()
        username_input.setPlaceholderText("robot")
        layout.addWidget(username_label)
        layout.addWidget(username_input)

        output_directory_label = QLabel("Output Directory:")
        output_directory_input = QLineEdit()
        output_directory_button = QPushButton("Browse")
        output_directory_button.clicked.connect(lambda: browse_output_directory(output_directory_input))
        output_directory_layout = QHBoxLayout()
        output_directory_layout.addWidget(output_directory_input)
        output_directory_layout.addWidget(output_directory_button)
        layout.addWidget(output_directory_label)
        layout.addLayout(output_directory_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec_() == QDialog.Accepted:
            user_file_path = user_file_input.text()
            qclasses_file_path = qclasses_file_input.text()
            username = username_input.text()
            output_directory = output_directory_input.text()

            if not username:
                username = "robot"

            if not os.path.exists(user_file_path):
                QMessageBox.warning(self.annotation_window,
                                    "File Not Found",
                                    f"User file not found: {user_file_path}")
                return

            if not os.path.exists(qclasses_file_path):
                QMessageBox.warning(self.annotation_window,
                                    "File Not Found",
                                    f"QClasses file not found: {qclasses_file_path}")
                return

            if not os.path.exists(output_directory):
                QMessageBox.warning(self.annotation_window,
                                    "Directory Not Found",
                                    f"Output directory not found: {output_directory}")
                return

            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                progress_bar = ProgressBar(self.annotation_window, title="Exporting Viscore Annotations")
                progress_bar.show()

                with open(user_file_path, 'r') as user_file:
                    user_data = json.load(user_file)

                with open(qclasses_file_path, 'r') as qclasses_file:
                    qclasses_data = json.load(qclasses_file)

                qclasses_mapping_short = get_qclass_mapping(qclasses_data, use_short_code=True)
                qclasses_mapping_long = get_qclass_mapping(qclasses_data, use_short_code=False)

                annotations = [a for a in self.annotation_window.annotations_dict.values() if "Dot" in a.data]

                dot_annotations = {}
                for annotation in annotations:
                    dot_id = annotation.data["Dot"]
                    dot_annotations.setdefault(dot_id, []).append(annotation)

                def get_mode_label_id(annotations):
                    labels = [a.label.id for a in annotations]
                    return max(set(labels), key=labels.count)

                dot_labels = {d_id: get_mode_label_id(anns) for d_id, anns in dot_annotations.items()}

                for index in range(len(user_data['cl'])):
                    label_id = dot_labels.get(index)
                    if label_id is not None:
                        label = self.label_window.get_label_by_id(label_id)
                        updated_label = qclasses_mapping_long.get(label.long_label_code)
                        if updated_label is None:
                            updated_label = qclasses_mapping_short.get(label.short_label_code)
                        if updated_label is None:
                            updated_label = -1
                        user_data['cl'][index] = updated_label

                output_file_path = os.path.join(output_directory, f"samples.cl.user.{username}.json")

                with open(output_file_path, 'w') as output_file:
                    json.dump(user_data, output_file, indent=4)

                progress_bar.stop_progress()
                progress_bar.close()

                QMessageBox.information(self.annotation_window,
                                        "Export Successful",
                                        f"Annotations have been successfully exported.")

            except Exception as e:
                QMessageBox.critical(self.annotation_window,
                                     "Export Failed",
                                     f"An error occurred while exporting annotations: {e}")

            QApplication.restoreOverrideCursor()

    def import_viscore_annotations(self):
        self.main_window.untoggle_all_tools()

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
                df = pd.read_csv(file_path)
                progress_bar.close()

                if df.empty:
                    QMessageBox.warning(self.annotation_window, "Empty CSV", "The CSV file is empty.")
                    return

                required_columns = ['Name',
                                    'Row',
                                    'Column',
                                    'Label',
                                    'Dot',
                                    'RandSubCeil',
                                    'ReprojectionError',
                                    'ViewIndex',
                                    'ViewCount']

                if not all(col in df.columns for col in required_columns):
                    QMessageBox.warning(self.annotation_window,
                                        "Invalid CSV Format",
                                        "The selected CSV file does not match the expected Viscore format.")
                    return

                df = df[required_columns]
                df_filtered = df.dropna(how='any')
                df_filtered = df_filtered.assign(Row=df_filtered['Row'].astype(int))
                df_filtered = df_filtered.assign(Column=df_filtered['Column'].astype(int))

                image_paths = df_filtered['Name'].unique()
                image_paths = [path for path in image_paths if os.path.exists(path)]

                if not image_paths and not self.annotation_window.active_image:
                    QMessageBox.warning(self.annotation_window,
                                        "No Images Found",
                                        "Please load an image before importing annotations.")
                    return

                mask = (
                        (df_filtered['RandSubCeil'] <= rand_sub_ceil) &
                        (df_filtered['ReprojectionError'] <= reprojection_error) &
                        (df_filtered['ViewIndex'] <= view_index) &
                        (df_filtered['ViewCount'] >= view_count)
                )
                filtered_df = df_filtered[mask]

                num_images = filtered_df['Name'].nunique()
                num_annotations = len(filtered_df)

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

                if image_paths:
                    progress_bar = ProgressBar(self.annotation_window, title="Importing Images")
                    progress_bar.show()
                    progress_bar.start_progress(len(image_paths))

                    for i, image_path in enumerate(image_paths):
                        if image_path not in set(self.image_window.image_paths):
                            self.image_window.add_image(image_path)
                            progress_bar.update_progress()

                    progress_bar.stop_progress()
                    progress_bar.close()

                    self.image_window.load_image_by_path(image_paths[-1])
                else:
                    loaded_images = {os.path.basename(path) for path in self.image_window.image_paths}
                    filtered_df.loc[:, 'Name'] = filtered_df['Name'].apply(os.path.basename)
                    filtered_df = filtered_df[filtered_df['Name'].isin(loaded_images)]

                    if filtered_df.empty:
                        QMessageBox.warning(self.annotation_window,
                                            "No Matching Images",
                                            "None of the image names in the CSV match loaded images.")
                        return

                QApplication.setOverrideCursor(Qt.WaitCursor)

                image_path_map = {os.path.basename(path): path for path in self.image_window.image_paths}

                progress_bar = ProgressBar(self.annotation_window, title="Importing Viscore Annotations")
                progress_bar.show()
                progress_bar.start_progress(len(filtered_df))

                for image_name, group in filtered_df.groupby('Name'):
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

                        annotation.data['Dot'] = row['Dot']

                        self.annotation_window.annotations_dict[annotation.id] = annotation
                        progress_bar.update_progress()

                    self.image_window.update_image_annotations(image_path)

                progress_bar.stop_progress()
                progress_bar.close()

                self.annotation_window.load_annotations()

                QMessageBox.information(self.annotation_window,
                                        "Annotations Imported",
                                        "Annotations have been successfully imported.")

            except Exception as e:
                QMessageBox.critical(self.annotation_window, "Critical Error", f"Failed to import annotations: {e}")