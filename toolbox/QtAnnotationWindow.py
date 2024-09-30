import json
import os
import random
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from PyQt5.QtCore import Qt, pyqtSignal, QPointF
from PyQt5.QtGui import QMouseEvent, QPixmap, QColor
from PyQt5.QtWidgets import (QFileDialog, QApplication, QGraphicsView, QGraphicsScene, QMessageBox, QVBoxLayout, QLabel,
                             QDialog, QHBoxLayout, QPushButton, QGraphicsPixmapItem, QGraphicsRectItem, QInputDialog,
                             QLineEdit, QDialogButtonBox)

from toolbox.QtProgressBar import ProgressBar
from toolbox.QtPatchAnnotation import PatchAnnotation

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class AnnotationWindow(QGraphicsView):
    imageLoaded = pyqtSignal(int, int)  # Signal to emit when image is loaded
    mouseMoved = pyqtSignal(int, int)  # Signal to emit when mouse is moved
    toolChanged = pyqtSignal(str)  # Signal to emit when the tool changes
    labelSelected = pyqtSignal(str)  # Signal to emit when the label changes
    annotationSizeChanged = pyqtSignal(int)  # Signal to emit when annotation size changes
    annotationSelected = pyqtSignal(int)  # Signal to emit when annotation is selected
    transparencyChanged = pyqtSignal(int)  # Signal to emit when transparency changes

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.annotation_size = 224
        self.annotation_color = None
        self.transparency = 128

        self.zoom_factor = 1.0
        self.pan_active = False
        self.pan_start = None
        self.drag_start_pos = None
        self.cursor_annotation = None

        self.annotations_dict = {}  # Dictionary to store annotations by UUID

        self.selected_annotation = None  # Stores the selected annotation
        self.selected_label = None  # Flag to check if an active label is set
        self.selected_tool = None  # Store the current tool state

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.NoDrag)  # Disable default drag mode

        self.image_pixmap = None
        self.rasterio_image = None
        self.active_image = False  # Flag to check if the image has been set
        self.current_image_path = None

        self.toolChanged.connect(self.set_selected_tool)

    def export_annotations(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self,
                                                   "Save Annotations",
                                                   "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                # Set the cursor to waiting (busy) cursor
                QApplication.setOverrideCursor(Qt.WaitCursor)

                # Get the total number of annotations
                total_annotations = len(list(self.annotations_dict.values()))
                # Show a progress bar
                progress_bar = ProgressBar(self, title="Exporting Annotations")
                progress_bar.show()
                progress_bar.start_progress(total_annotations)

                export_dict = {}
                for annotation in self.annotations_dict.values():
                    image_path = annotation.image_path
                    if image_path not in export_dict:
                        export_dict[image_path] = []
                    export_dict[image_path].append(annotation.to_dict())

                    progress_bar.update_progress()

                with open(file_path, 'w') as file:
                    json.dump(export_dict, file, indent=4)
                    file.flush()  # Ensure the data is written to the file

                progress_bar.stop_progress()
                progress_bar.close()

                QMessageBox.information(self,
                                        "Annotations Exported",
                                        "Annotations have been successfully exported.")

            except Exception as e:
                QMessageBox.warning(self,
                                    "Error Exporting Annotations",
                                    f"An error occurred while exporting annotations: {str(e)}")

        # Restore the cursor to the default cursor
        QApplication.restoreOverrideCursor()

    def import_annotations(self):
        self.set_selected_tool(None)
        self.toolChanged.emit(None)

        if not self.active_image:
            QMessageBox.warning(self,
                                "No Images Loaded",
                                "Please load images first before importing annotations.")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Load Annotations",
                                                   "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                # Set the cursor to waiting (busy) cursor
                QApplication.setOverrideCursor(Qt.WaitCursor)

                # Load the JSON file
                with open(file_path, 'r') as file:
                    data = json.load(file)

                # Needed keys for each annotation
                keys = ['label_short_code', 'label_long_code', 'annotation_color', 'image_path', 'label_id']

                # Filter out annotations that are not associated with any loaded images
                filtered_annotations = {p: a for p, a in data.items() if p in self.main_window.image_window.image_paths}
                total_annotations = sum(len(annotations) for annotations in filtered_annotations.values())

                progress_bar = ProgressBar(self, title="Importing Annotations")
                progress_bar.show()
                progress_bar.start_progress(total_annotations)

                # Check to see if any imported annotations have a label that matches an existing label
                updated_annotations = False

                for image_path, annotations in filtered_annotations.items():
                    for annotation_data in annotations:
                        # Check if the annotation data contains the required keys
                        if not all(key in annotation_data for key in keys):
                            continue

                        short_label = annotation_data['label_short_code']
                        long_label = annotation_data['label_long_code']
                        color = QColor(*annotation_data['annotation_color'])
                        label_id = annotation_data['label_id']
                        self.main_window.label_window.add_label_if_not_exists(short_label,
                                                                              long_label,
                                                                              color,
                                                                              label_id)

                        # Check if the imported annotation has a label color that matches an existing label
                        existing_color = self.main_window.label_window.get_label_color(short_label, long_label)
                        if existing_color != color:
                            annotation_data['annotation_color'] = existing_color.getRgb()
                            updated_annotations = True

                        progress_bar.update_progress()

                if updated_annotations:
                    # Inform the user that some annotations have been updated
                    QMessageBox.information(self,
                                            "Annotations Updated",
                                            "Some annotations have been updated to match the "
                                            "color of the labels already in the project.")

                # Add annotations to the AnnotationWindow dict
                for image_path, annotations in filtered_annotations.items():
                    for annotation_data in annotations:
                        # Check if the annotation data contains the required keys
                        if not all(key in annotation_data for key in keys):
                            continue
                        annotation = PatchAnnotation.from_dict(annotation_data)
                        self.annotations_dict[annotation.id] = annotation
                        progress_bar.update_progress()

                    # Update the image window's image dict
                    self.main_window.image_window.update_image_annotations(image_path)

                progress_bar.stop_progress()
                progress_bar.close()

                self.load_annotations()

                QMessageBox.information(self,
                                        "Annotations Imported",
                                        "Annotations have been successfully imported.")

            except Exception as e:
                QMessageBox.warning(self,
                                    "Error Importing Annotations",
                                    f"An error occurred while importing annotations: {str(e)}")

        # Restore the cursor to the default cursor
        QApplication.restoreOverrideCursor()

    def export_coralnet_annotations(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self,
                                                   "Export CoralNet Annotations",
                                                   "",
                                                   "CSV Files (*.csv);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                # Set the cursor to waiting (busy) cursor
                QApplication.setOverrideCursor(Qt.WaitCursor)

                data = []
                total_annotations = len(self.annotations_dict)

                progress_bar = ProgressBar(self, title="Exporting CoralNet Annotations")
                progress_bar.show()
                progress_bar.start_progress(total_annotations)

                for annotation in self.annotations_dict.values():
                    data.append(annotation.to_coralnet_format())
                    progress_bar.update_progress()

                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False)

                progress_bar.stop_progress()
                progress_bar.close()

                QMessageBox.information(self,
                                        "Annotations Exported",
                                        "Annotations have been successfully exported.")

            except Exception as e:
                QMessageBox.warning(self,
                                    "Error Exporting Annotations",
                                    f"An error occurred while exporting annotations: {str(e)}")

        # Restore the cursor to the default cursor
        QApplication.restoreOverrideCursor()

    def import_coralnet_annotations(self):
        self.set_selected_tool(None)
        self.toolChanged.emit(None)

        if not self.active_image:
            QMessageBox.warning(self,
                                "No Images Loaded",
                                "Please load images first before importing annotations.")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Import CoralNet Annotations",
                                                   "",
                                                   "CSV Files (*.csv);;All Files (*)",
                                                   options=options)

        if not file_path:
            return

        try:
            # Show a progress bar
            progress_bar = ProgressBar(self, title="Importing Annotations")
            progress_bar.show()
            # Read the CSV file using pandas
            df = pd.read_csv(file_path)
            # Close the progress bar
            progress_bar.close()

            required_columns = ['Name', 'Row', 'Column', 'Label']
            if not all(col in df.columns for col in required_columns):
                QMessageBox.warning(self,
                                    "Invalid CSV Format",
                                    "The selected CSV file does not match the expected CoralNet format.")
                return

            annotation_size, ok = QInputDialog.getInt(self,
                                                      "Annotation Size",
                                                      "Enter the annotation size for all imported annotations:",
                                                      224, 1, 10000, 1)
            if not ok:
                return

            # Create a dictionary mapping image basenames to their full paths
            image_path_map = {os.path.basename(path): path for path in self.main_window.image_window.image_paths}

            # Filter out annotations that are not associated with any loaded images
            df = df[df['Name'].isin(image_path_map.keys())]
            df = df.dropna(how='any', subset=['Row', 'Column', 'Label'])
            df = df.assign(Row=df['Row'].astype(int))
            df = df.assign(Column=df['Column'].astype(int))

            if df.empty:
                raise Exception("No annotations found for loaded images.")

            # Get the total number of annotations
            total_annotations = len(df)

            progress_bar = ProgressBar(self, title="Importing CoralNet Annotations")
            progress_bar.show()
            progress_bar.start_progress(total_annotations)

            # Set the cursor to waiting (busy) cursor
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # Iterate through the DataFrame and create annotations, group by image name
            for image_name, group in df.groupby('Name'):
                image_path = image_path_map.get(image_name)
                if not image_path:
                    continue

                for index, row in group.iterrows():
                    row_coord = row['Row']
                    col_coord = row['Column']
                    label_code = row['Label']

                    short_label_code = long_label_code = label_code
                    existing_label = self.main_window.label_window.get_label_by_codes(short_label_code,
                                                                                      long_label_code)

                    if existing_label:
                        # Use the existing label if it exists
                        color = existing_label.color
                        label_id = existing_label.id
                    else:
                        # Create a new label if it doesn't exist
                        label_id = str(uuid.uuid4())
                        color = QColor(random.randint(0, 255),
                                       random.randint(0, 255),
                                       random.randint(0, 255))

                        # Add the new label to the LabelWindow
                        self.main_window.label_window.add_label_if_not_exists(short_label_code,
                                                                              long_label_code,
                                                                              color,
                                                                              label_id)
                    # Create the annotation
                    annotation = PatchAnnotation(QPointF(col_coord, row_coord),
                                                 annotation_size,
                                                 short_label_code,
                                                 long_label_code,
                                                 color,
                                                 image_path,
                                                 label_id)

                    # Add machine confidence and suggestions if they exist
                    machine_confidence = {}

                    for i in range(1, 6):
                        confidence_col = f'Machine confidence {i}'
                        suggestion_col = f'Machine suggestion {i}'
                        if confidence_col in row and suggestion_col in row:
                            if pd.isna(row[confidence_col]) or pd.isna(row[suggestion_col]):
                                continue

                            confidence = float(row[confidence_col])
                            suggestion = str(row[suggestion_col])

                            # Ensure the suggestion is an existing label
                            suggested_label = self.main_window.label_window.get_label_by_short_code(suggestion)

                            # If it doesn't exist, add it to the LabelWindow
                            if not suggested_label:
                                color = QColor(random.randint(0, 255),
                                               random.randint(0, 255),
                                               random.randint(0, 255))

                                # Using both the short and long code as the same value
                                self.main_window.label_window.add_label_if_not_exists(suggestion, suggestion, color)

                            suggested_label = self.main_window.label_window.get_label_by_short_code(suggestion)
                            machine_confidence[suggested_label] = confidence

                    if machine_confidence:
                        annotation.update_machine_confidence(machine_confidence)

                    # Add to the AnnotationWindow dictionary
                    self.annotations_dict[annotation.id] = annotation
                    progress_bar.update_progress()

                # Update the image window's image dict
                self.main_window.image_window.update_image_annotations(image_path)

            progress_bar.stop_progress()
            progress_bar.close()

            self.load_annotations()

            QMessageBox.information(self,
                                    "Annotations Imported",
                                    "Annotations have been successfully imported.")

        except Exception as e:
            QMessageBox.warning(self,
                                "Error Importing Annotations",
                                f"An error occurred while importing annotations: {str(e)}")

        # Restore the cursor to the default cursor
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
            file_path, _ = QFileDialog.getOpenFileName(self,
                                                       "Select User File",
                                                       "",
                                                       "JSON Files (*.json);;All Files (*)",
                                                       options=options)
            if file_path:
                user_file_input.setText(file_path)

        def browse_qclasses_file(qclasses_file_input):
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(self,
                                                       "Select QClasses File",
                                                       "",
                                                       "JSON Files (*.json);;All Files (*)",
                                                       options=options)
            if file_path:
                qclasses_file_input.setText(file_path)

        def browse_output_directory(output_directory_input):
            options = QFileDialog.Options()
            directory = QFileDialog.getExistingDirectory(self,
                                                         "Select Output Directory",
                                                         "",
                                                         options=options)
            if directory:
                output_directory_input.setText(directory)

        # Create a dialog to get the required inputs
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Viscore Annotations")
        dialog.resize(400, 300)

        layout = QVBoxLayout(dialog)

        # User File (JSON)
        user_file_label = QLabel("User File (JSON):")
        user_file_input = QLineEdit()
        user_file_button = QPushButton("Browse")
        user_file_button.clicked.connect(lambda: browse_user_file(user_file_input))
        user_file_layout = QHBoxLayout()
        user_file_layout.addWidget(user_file_input)
        user_file_layout.addWidget(user_file_button)
        layout.addWidget(user_file_label)
        layout.addLayout(user_file_layout)

        # QClasses File (JSON)
        qclasses_file_label = QLabel("QClasses File (JSON):")
        qclasses_file_input = QLineEdit()
        qclasses_file_button = QPushButton("Browse")
        qclasses_file_button.clicked.connect(lambda: browse_qclasses_file(qclasses_file_input))
        qclasses_file_layout = QHBoxLayout()
        qclasses_file_layout.addWidget(qclasses_file_input)
        qclasses_file_layout.addWidget(qclasses_file_button)
        layout.addWidget(qclasses_file_label)
        layout.addLayout(qclasses_file_layout)

        # Username
        username_label = QLabel("Username:")
        username_input = QLineEdit()
        username_input.setPlaceholderText("robot")
        layout.addWidget(username_label)
        layout.addWidget(username_input)

        # Output Directory
        output_directory_label = QLabel("Output Directory:")
        output_directory_input = QLineEdit()
        output_directory_button = QPushButton("Browse")
        output_directory_button.clicked.connect(lambda: browse_output_directory(output_directory_input))
        output_directory_layout = QHBoxLayout()
        output_directory_layout.addWidget(output_directory_input)
        output_directory_layout.addWidget(output_directory_button)
        layout.addWidget(output_directory_label)
        layout.addLayout(output_directory_layout)

        # OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec_() == QDialog.Accepted:
            user_file_path = user_file_input.text()
            qclasses_file_path = qclasses_file_input.text()
            username = username_input.text()
            output_directory = output_directory_input.text()

            # Set default username if empty
            if not username:
                username = "robot"

            # Check if the files exist
            if not os.path.exists(user_file_path):
                QMessageBox.warning(self, "File Not Found", f"User file not found: {user_file_path}")
                return

            if not os.path.exists(qclasses_file_path):
                QMessageBox.warning(self, "File Not Found", f"QClasses file not found: {qclasses_file_path}")
                return

            if not os.path.exists(output_directory):
                QMessageBox.warning(self, "Directory Not Found", f"Output directory not found: {output_directory}")
                return

            try:
                # Set the cursor to waiting (busy) cursor
                QApplication.setOverrideCursor(Qt.WaitCursor)
                # Show a progress bar
                progress_bar = ProgressBar(self, title="Exporting Viscore Annotations")
                progress_bar.show()

                # Open and load the JSON files
                with open(user_file_path, 'r') as user_file:
                    user_data = json.load(user_file)

                with open(qclasses_file_path, 'r') as qclasses_file:
                    qclasses_data = json.load(qclasses_file)

                # Get the QClasses mapping
                qclasses_mapping_short = get_qclass_mapping(qclasses_data, use_short_code=True)
                qclasses_mapping_long = get_qclass_mapping(qclasses_data, use_short_code=False)

                # Get all annotations with Dot data
                annotations = [a for a in self.annotations_dict.values() if "Dot" in a.data]

                # Group annotations by Dot ID
                dot_annotations = {}
                for annotation in annotations:
                    dot_id = annotation.data["Dot"]
                    dot_annotations.setdefault(dot_id, []).append(annotation)

                # Function to get the mode label ID from annotations
                def get_mode_label_id(annotations):
                    labels = [a.label.id for a in annotations]
                    return max(set(labels), key=labels.count)

                # Map Dot IDs to their mode label IDs
                dot_labels = {d_id: get_mode_label_id(anns) for d_id, anns in dot_annotations.items()}

                # Update user_data with the mode label codes
                for index in range(len(user_data['cl'])):
                    # Get the Label id from the dot_labels
                    label_id = dot_labels.get(index)
                    # If it doesn't exist, then the filtering process has removed all views; skip
                    if label_id is not None:
                        # Get the label from the LabelWindow
                        label = self.main_window.label_window.get_label_by_id(label_id)
                        # Try to map the long code
                        updated_label = qclasses_mapping_long.get(label.long_label_code)
                        # If long code is not found, try mapping the short code
                        if updated_label is None:
                            updated_label = qclasses_mapping_short.get(label.short_label_code)
                        # If neither long nor short code is found, set it to -1
                        if updated_label is None:
                            updated_label = -1
                        # Update the label in the user_data
                        user_data['cl'][index] = updated_label

                # Create the output file path
                output_file_path = os.path.join(output_directory, f"samples.cl.user.{username}.json")

                # Write the output data to the file
                with open(output_file_path, 'w') as output_file:
                    json.dump(user_data, output_file, indent=4)

                # Close progress bar
                progress_bar.stop_progress()
                progress_bar.close()

                QMessageBox.information(self,
                                        "Export Successful",
                                        f"Annotations have been successfully exported.")

            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"An error occurred while exporting annotations: {e}")

        # Restore the cursor to the default cursor
        QApplication.restoreOverrideCursor()

    def import_viscore_annotations(self):

        def browse_csv_file(file_path_input):
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(self,
                                                       "Import Viscore Annotations",
                                                       "",
                                                       "CSV Files (*.csv);;All Files (*)",
                                                       options=options)
            if file_path:
                file_path_input.setText(file_path)

        self.set_selected_tool(None)
        self.toolChanged.emit(None)

        # Create a dialog to get the CSV file path and additional values
        dialog = QDialog(self)
        dialog.setWindowTitle("Import Viscore Annotations")
        dialog.resize(500, 200)

        layout = QVBoxLayout(dialog)

        # CSV file path
        file_path_label = QLabel("CSV File Path:")
        file_path_input = QLineEdit()
        file_path_button = QPushButton("Browse")
        file_path_button.clicked.connect(lambda: browse_csv_file(file_path_input))
        file_path_layout = QHBoxLayout()
        file_path_layout.addWidget(file_path_input)
        file_path_layout.addWidget(file_path_button)
        layout.addWidget(file_path_label)
        layout.addLayout(file_path_layout)

        # ReprojectionError
        reprojection_error_label = QLabel("ReprojectionError (Default: 0.01, float):")
        reprojection_error_input = QLineEdit()
        reprojection_error_input.setPlaceholderText("Error between an image point, reprojected to its 3D dot location")
        layout.addWidget(reprojection_error_label)
        layout.addWidget(reprojection_error_input)

        # ViewIndex
        view_index_label = QLabel("ViewIndex (Default: 10, int):")
        view_index_input = QLineEdit()
        view_index_input.setPlaceholderText("The image's index in VPI view (includes a form pre-filtering)")
        layout.addWidget(view_index_label)
        layout.addWidget(view_index_input)

        # ViewCount
        view_count_label = QLabel("ViewCount (Default: 5, int):")
        view_count_input = QLineEdit()
        view_count_input.setPlaceholderText("The number of images the dot has been seen in")
        layout.addWidget(view_count_label)
        layout.addWidget(view_count_input)

        # RandSubCeil
        rand_sub_ceil_label = QLabel("RandSubCeil (Default: 1.0, float, [0-1]):")
        rand_sub_ceil_input = QLineEdit()
        rand_sub_ceil_input.setPlaceholderText("Randomly sample a subset of the data")
        layout.addWidget(rand_sub_ceil_label)
        layout.addWidget(rand_sub_ceil_input)

        # OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec_() == QDialog.Accepted:
            file_path = file_path_input.text()

            # Set default values if the input fields are empty
            reprojection_error = reprojection_error_input.text()
            if not reprojection_error:
                reprojection_error = "0.01"
            try:
                reprojection_error = float(reprojection_error)
                if reprojection_error < 0:
                    raise ValueError("ReprojectionError must be a non-negative float.")
            except ValueError as e:
                QMessageBox.warning(self, "Invalid Input", f"Invalid ReprojectionError: {e}")
                return

            view_index = view_index_input.text()
            if not view_index:
                view_index = "10"
            try:
                view_index = int(view_index)
                if view_index < 0:
                    raise ValueError("ViewIndex must be a non-negative integer.")
            except ValueError as e:
                QMessageBox.warning(self, "Invalid Input", f"Invalid ViewIndex: {e}")
                return

            view_count = view_count_input.text()
            if not view_count:
                view_count = "5"
            try:
                view_count = int(view_count)
                if view_count < 0:
                    raise ValueError("ViewCount must be a non-negative integer.")
            except ValueError as e:
                QMessageBox.warning(self, "Invalid Input", f"Invalid ViewCount: {e}")
                return

            rand_sub_ceil = rand_sub_ceil_input.text()
            if not rand_sub_ceil:
                rand_sub_ceil = "1.0"
            try:
                rand_sub_ceil = float(rand_sub_ceil)
                if not (0 <= rand_sub_ceil <= 1):
                    raise ValueError("RandSubCeil must be a float between 0 and 1.")
            except ValueError as e:
                QMessageBox.warning(self, "Invalid Input", f"Invalid RandSubCeil: {e}")
                return

            try:
                # Show a progress bar
                progress_bar = ProgressBar(self, title="Reading CSV File")
                progress_bar.show()
                # Read the CSV file using pandas
                df = pd.read_csv(file_path)
                # Close the progress bar
                progress_bar.close()

                if df.empty:
                    QMessageBox.warning(self, "Empty CSV", "The CSV file is empty.")
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
                    QMessageBox.warning(self,
                                        "Invalid CSV Format",
                                        "The selected CSV file does not match the expected Viscore format.")
                    return

                # Show a progress bar
                progress_bar = ProgressBar(self, title="Filtering CSV File")
                progress_bar.show()
                progress_bar.set_value(1)

                # Apply filters more efficiently
                df = df[required_columns]
                df_filtered = df.dropna(how='any')
                df_filtered = df_filtered.assign(Row=df_filtered['Row'].astype(int))
                df_filtered = df_filtered.assign(Column=df_filtered['Column'].astype(int))

                progress_bar.set_value(25)

                # Check if 'Name' exists as a path (or just basename) and create a unique list
                image_paths = df_filtered['Name'].unique()
                image_paths = [path for path in image_paths if os.path.exists(path)]

                if not image_paths and not self.active_image:
                    QMessageBox.warning(self, "No Images Found", "Please load an image before importing annotations.")
                    progress_bar.close()
                    return

                progress_bar.set_value(50)

                # Filter the DataFrame based on the input values
                mask = (
                        (df_filtered['RandSubCeil'] <= rand_sub_ceil) &
                        (df_filtered['ReprojectionError'] <= reprojection_error) &
                        (df_filtered['ViewIndex'] <= view_index) &
                        (df_filtered['ViewCount'] >= view_count)
                )
                filtered_df = df_filtered[mask]

                progress_bar.set_value(100)
                progress_bar.close()

                # Calculate the number of unique images and annotations
                num_images = filtered_df['Name'].nunique()
                num_annotations = len(filtered_df)

                # Display the number of images and annotations
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Filtered Data Summary")
                msg_box.setText(f"Number of Images: {num_images}\nNumber of Annotations: {num_annotations}")
                msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                msg_box.setDefaultButton(QMessageBox.Ok)

                result = msg_box.exec_()

                if result == QMessageBox.Cancel:
                    self.import_viscore_annotations()
                    return

                annotation_size, ok = QInputDialog.getInt(self,
                                                          "Annotation Size",
                                                          "Enter the annotation size for all imported annotations:",
                                                          224, 1, 10000, 1)
                if not ok:
                    return

                if image_paths:
                    # Import images to project
                    progress_bar = ProgressBar(self, title="Importing Images")
                    progress_bar.show()
                    progress_bar.start_progress(len(image_paths))

                    for i, image_path in enumerate(image_paths):
                        if image_path not in set(self.main_window.image_window.image_paths):
                            self.main_window.image_window.add_image(image_path)
                            progress_bar.update_progress()

                    progress_bar.stop_progress()
                    progress_bar.close()

                    # Load the last image
                    self.main_window.image_window.load_image_by_path(image_paths[-1])
                else:
                    # Update the DataFrame to only include annotations for loaded images
                    loaded_images = {os.path.basename(path) for path in self.main_window.image_window.image_paths}
                    filtered_df.loc[:, 'Name'] = filtered_df['Name'].apply(os.path.basename)
                    filtered_df = filtered_df[filtered_df['Name'].isin(loaded_images)]

                    if filtered_df.empty:
                        QMessageBox.warning(self,
                                            "No Matching Images",
                                            "None of the image names in the CSV match loaded images.")
                        return

                # Set the cursor to waiting (busy) cursor
                QApplication.setOverrideCursor(Qt.WaitCursor)

                # Create a dictionary mapping basenames to full paths
                image_path_map = {os.path.basename(path): path for path in self.main_window.image_window.image_paths}

                progress_bar = ProgressBar(self, title="Importing Viscore Annotations")
                progress_bar.show()
                progress_bar.start_progress(len(filtered_df))

                # Process the filtered CSV data and import the annotations
                for image_name, group in filtered_df.groupby('Name'):
                    image_path = image_path_map.get(image_name)
                    if not image_path:
                        continue

                    for index, row in group.iterrows():
                        row_coord = row['Row']
                        col_coord = row['Column']
                        label_code = row['Label']

                        # Check if the label exists in the LabelWindow
                        short_label_code = long_label_code = label_code
                        existing_label = self.main_window.label_window.get_label_by_codes(short_label_code,
                                                                                          long_label_code)

                        if existing_label:
                            # Use the existing label if it exists
                            color = existing_label.color
                            label_id = existing_label.id
                        else:
                            # Create a new label if it doesn't exist
                            label_id = str(uuid.uuid4())
                            color = QColor(random.randint(0, 255),
                                           random.randint(0, 255),
                                           random.randint(0, 255))

                            # Add the new label to the LabelWindow
                            self.main_window.label_window.add_label_if_not_exists(short_label_code,
                                                                                  long_label_code,
                                                                                  color,
                                                                                  label_id)
                        # Create the annotation
                        annotation = PatchAnnotation(QPointF(col_coord, row_coord),
                                                     annotation_size,
                                                     short_label_code,
                                                     long_label_code,
                                                     color,
                                                     image_path,
                                                     label_id)

                        # Add additional data to the annotation
                        annotation.data['Dot'] = row['Dot']

                        # Add to the AnnotationWindow dictionary
                        self.annotations_dict[annotation.id] = annotation
                        progress_bar.update_progress()

                    # Update the image window's image dict
                    self.main_window.image_window.update_image_annotations(image_path)

                progress_bar.stop_progress()
                progress_bar.close()

                # Load annotations for current image
                self.load_annotations()

                QMessageBox.information(self,
                                        "Annotations Imported",
                                        "Annotations have been successfully imported.")

            except Exception as e:
                QMessageBox.critical(self, "Critical Error", f"Failed to import annotations: {e}")

        # Restore the cursor to the default cursor
        QApplication.restoreOverrideCursor()

    def set_selected_tool(self, tool):
        self.selected_tool = tool
        if self.selected_tool == "select":
            self.setCursor(Qt.PointingHandCursor)
        elif self.selected_tool == "patch":
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        self.unselect_annotation()

    def set_selected_label(self, label):
        self.selected_label = label
        self.annotation_color = label.color

        if self.selected_annotation:
            if self.selected_annotation.label.id != label.id:
                # self.selected_annotation.update_label(self.selected_label)
                self.selected_annotation.update_user_confidence(self.selected_label)
                self.selected_annotation.create_cropped_image(self.rasterio_image)
                self.main_window.confidence_window.display_cropped_image(self.selected_annotation)

        if self.cursor_annotation:
            if self.cursor_annotation.label.id != label.id:
                self.toggle_cursor_annotation()

    def set_annotation_size(self, size=None, delta=0):
        if size is not None:
            self.annotation_size = size
        else:
            self.annotation_size += delta
            self.annotation_size = max(1, self.annotation_size)

        if self.selected_annotation:
            self.selected_annotation.update_annotation_size(self.annotation_size)
            self.selected_annotation.create_cropped_image(self.rasterio_image)
            # Notify ConfidenceWindow the selected annotation has changed
            self.main_window.confidence_window.display_cropped_image(self.selected_annotation)

        if self.cursor_annotation:
            self.cursor_annotation.update_annotation_size(self.annotation_size)

        # Emit that the annotation size has changed
        self.annotationSizeChanged.emit(self.annotation_size)

    def set_annotation_location(self, annotation_id, new_center_xy: QPointF):
        if annotation_id in self.annotations_dict:
            annotation = self.annotations_dict[annotation_id]
            old_center_xy = annotation.center_xy
            annotation.update_location(new_center_xy)

    def toggle_cursor_annotation(self, scene_pos: QPointF = None):
        if scene_pos:
            if not self.selected_label or not self.annotation_color:
                return

            if not self.cursor_annotation:
                self.cursor_annotation = PatchAnnotation(scene_pos,
                                                         self.annotation_size,
                                                         self.selected_label.short_label_code,
                                                         self.selected_label.long_label_code,
                                                         self.selected_label.color,
                                                         self.current_image_path,
                                                         self.selected_label.id,
                                                         transparency=128,
                                                         show_msg=False)

                self.cursor_annotation.create_graphics_item(self.scene)
            else:
                self.cursor_annotation.update_location(scene_pos)
                self.cursor_annotation.update_graphics_item()
                self.cursor_annotation.update_transparency(128)
        else:
            if self.cursor_annotation:
                self.cursor_annotation.delete()
                self.cursor_annotation = None

    def display_image_item(self, image_item):
        # Clean up
        self.clear_scene()

        # Display NaN values the image dimensions in status bar
        self.imageLoaded.emit(-0, -0)

        # Set the image representations
        self.image_pixmap = QPixmap(image_item)
        self.scene.addItem(QGraphicsPixmapItem(self.image_pixmap))
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        # Clear the confidence window
        self.main_window.confidence_window.clear_display()
        QApplication.processEvents()

    def set_image(self, image_path):

        # Clean up
        self.clear_scene()

        # Set the image representations
        self.image_pixmap = QPixmap(self.main_window.image_window.images[image_path])
        self.rasterio_image = self.main_window.image_window.rasterio_images[image_path]

        self.current_image_path = image_path
        self.active_image = True

        # Set the image dimensions in status bar
        self.imageLoaded.emit(self.image_pixmap.width(), self.image_pixmap.height())

        self.scene.addItem(QGraphicsPixmapItem(self.image_pixmap))
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.toggle_cursor_annotation()

        # Load all associated annotations in parallel
        self.load_annotations_parallel()
        # Update the image window's image dict
        self.main_window.image_window.update_image_annotations(image_path)

        # Clear the confidence window
        self.main_window.confidence_window.clear_display()
        QApplication.processEvents()

    def wheelEvent(self, event: QMouseEvent):
        if event.angleDelta().y() > 0:
            factor = 1.1
        else:
            factor = 0.9

        self.zoom_factor *= factor
        self.scale(factor, factor)

        if self.selected_tool == "select":
            self.setCursor(Qt.PointingHandCursor)
        elif self.selected_tool == "patch":
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def mousePressEvent(self, event: QMouseEvent):
        if self.active_image:

            if event.button() == Qt.RightButton:
                self.pan_active = True
                self.pan_start = event.pos()
                self.setCursor(Qt.ClosedHandCursor)  # Change cursor to indicate panning

            if event.button() == Qt.LeftButton and self.selected_tool == "select":
                position = self.mapToScene(event.pos())
                items = self.scene.items(position)

                rect_items = [item for item in items if isinstance(item, QGraphicsRectItem)]
                rect_items.sort(key=lambda item: item.zValue(), reverse=True)

                for rect_item in rect_items:
                    annotation_id = rect_item.data(0)  # Retrieve the UUID from the graphics item's data
                    annotation = self.annotations_dict.get(annotation_id)
                    if annotation.contains_point(position):
                        self.select_annotation(annotation)
                        self.drag_start_pos = position  # Store the start position for dragging
                        break

            elif self.selected_tool == "patch" and event.button() == Qt.LeftButton:
                # Annotation cannot be selected in annotate mode
                self.unselect_annotation()
                # Add annotation to the scene
                self.add_annotation(self.mapToScene(event.pos()))

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.pan_active:
            self.pan(event.pos())
        elif self.selected_tool == "select" and self.selected_annotation:
            # Check if the left mouse button is pressed, then drag the annotation
            if event.buttons() & Qt.LeftButton:
                # Get the current position of the mouse in the scene
                current_pos = self.mapToScene(event.pos())
                # Check that it's not the first time dragging
                if hasattr(self, 'drag_start_pos'):
                    if not self.drag_start_pos:
                        # Start the dragging
                        self.drag_start_pos = current_pos
                    # Continue the dragging
                    delta = current_pos - self.drag_start_pos
                    new_center = self.selected_annotation.center_xy + delta
                    # Check if the new center is within the image bounds using cursorInWindow
                    if self.cursorInWindow(current_pos, mapped=True) and self.selected_annotation:
                        self.set_annotation_location(self.selected_annotation.id, new_center)
                        self.selected_annotation.create_cropped_image(self.rasterio_image)
                        self.main_window.confidence_window.display_cropped_image(self.selected_annotation)
                        self.drag_start_pos = current_pos  # Update the start position for smooth dragging

        # Normal movement with annotation tool selected
        elif (self.selected_tool == "patch" and
              self.active_image and self.image_pixmap and
              self.cursorInWindow(event.pos())):
            self.toggle_cursor_annotation(self.mapToScene(event.pos()))
        else:
            self.toggle_cursor_annotation()

        scene_pos = self.mapToScene(event.pos())
        self.mouseMoved.emit(int(scene_pos.x()), int(scene_pos.y()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.RightButton:
            self.pan_active = False
            self.setCursor(Qt.ArrowCursor)
        self.toggle_cursor_annotation()
        if hasattr(self, 'drag_start_pos'):
            # Clean up the drag start position
            del self.drag_start_pos
        super().mouseReleaseEvent(event)

    def pan(self, pos):
        delta = pos - self.pan_start
        self.pan_start = pos
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())

    def cursorInWindow(self, pos, mapped=False):
        if self.image_pixmap:
            image_rect = QGraphicsPixmapItem(self.image_pixmap).boundingRect()
            if not mapped:
                pos = self.mapToScene(pos)
            return image_rect.contains(pos)
        return False

    def cycle_annotations(self, direction):
        if self.selected_tool == "select" and self.active_image:
            annotations = self.get_image_annotations()
            if annotations:
                if self.selected_annotation:
                    current_index = annotations.index(self.selected_annotation)
                    new_index = (current_index + direction) % len(annotations)
                else:
                    new_index = 0
                # Select the new annotation
                self.select_annotation(annotations[new_index])
                # Center the view on the new annotation
                self.center_on_annotation(annotations[new_index])

    def center_on_annotation(self, annotation):
        # Get the bounding rect of the annotation in scene coordinates
        annotation_rect = annotation.graphics_item.boundingRect()
        annotation_center = annotation_rect.center()

        # Center the view on the annotation's center
        self.centerOn(annotation_center)

    def update_current_image_path(self, image_path):
        self.current_image_path = image_path

    def select_annotation(self, annotation):
        if self.selected_annotation != annotation:
            if self.selected_annotation:
                self.unselect_annotation()
            # Select the annotation
            self.selected_annotation = annotation
            self.selected_annotation.select()
            # Set the label and color for selected annotation
            self.selected_label = self.selected_annotation.label
            self.annotation_color = self.selected_annotation.label.color
            # Emit a signal indicating the selected annotations label to LabelWindow
            self.labelSelected.emit(annotation.label.id)
            # Emit a signal indicating the selected annotation's transparency to MainWindow
            self.transparencyChanged.emit(annotation.transparency)
            # Crop the image from annotation using current image item
            if not self.selected_annotation.cropped_image:
                self.selected_annotation.create_cropped_image(self.rasterio_image)
            # Display the selected annotation in confidence window
            self.main_window.confidence_window.display_cropped_image(self.selected_annotation)

    def unselect_annotation(self):
        if self.selected_annotation:
            self.selected_annotation.deselect()
            self.selected_annotation = None

        # Clear the confidence window
        self.main_window.confidence_window.clear_display()

    def update_annotation_transparency(self, transparency):
        if self.selected_annotation:
            # Update the label's transparency in the LabelWindow
            self.main_window.label_window.update_label_transparency(transparency)
            label = self.selected_annotation.label
            for annotation in self.annotations_dict.values():
                if annotation.label.id == label.id:
                    annotation.update_transparency(transparency)

    def load_annotation(self, annotation):
        # Create the graphics item (scene previously cleared)
        annotation.create_graphics_item(self.scene)
        # Connect update signals
        annotation.selected.connect(self.select_annotation)
        annotation.annotation_deleted.connect(self.delete_annotation)
        annotation.annotation_updated.connect(self.main_window.confidence_window.display_cropped_image)

    def load_annotations(self):
        # Crop all the annotations for current image (if not already cropped)
        annotations = self.crop_image_annotations(return_annotations=True)

        # Connect update signals for all the annotations
        for annotation in annotations:
            self.load_annotation(annotation)

    def load_annotations_parallel(self):
        # Crop all the annotations for current image (if not already cropped)
        annotations = self.crop_image_annotations(return_annotations=True)

        # Use ThreadPoolExecutor to process annotations in parallel
        with ThreadPoolExecutor() as executor:
            for annotation in annotations:
                executor.submit(self.load_annotation, annotation)

    def get_image_annotations(self, image_path=None):
        if not image_path:
            image_path = self.current_image_path

        annotations = []
        for annotation_id, annotation in self.annotations_dict.items():
            if annotation.image_path == image_path:
                annotations.append(annotation)

        return annotations

    def get_image_review_annotations(self, image_path=None):
        if not image_path:
            image_path = self.current_image_path

        annotations = []
        for annotation_id, annotation in self.annotations_dict.items():
            if annotation.image_path == image_path and annotation.label.id == '-1':
                annotations.append(annotation)

        return annotations

    def crop_image_annotations(self, image_path=None, return_annotations=False):
        if not image_path:
            # Set the image path if not provided
            image_path = self.current_image_path
        # Get the annotations for the image
        annotations = self.get_image_annotations(image_path)
        self._crop_annotations_batch(image_path, annotations)
        # Return the annotations if flag is set
        if return_annotations:
            return annotations

    def crop_these_image_annotations(self, image_path, annotations):
        # Crop these annotations for this image
        self._crop_annotations_batch(image_path, annotations)
        return annotations

    def _crop_annotations_batch(self, image_path, annotations):
        # Get the rasterio representation
        rasterio_image = self.main_window.image_window.rasterio_open(image_path)
        # Loop through the annotations, crop the image if not already cropped
        for annotation in annotations:
            if not annotation.cropped_image:
                annotation.create_cropped_image(rasterio_image)

    def add_annotation(self, scene_pos: QPointF, annotation=None):
        if not self.selected_label:
            QMessageBox.warning(self, "No Label Selected", "A label must be selected before adding an annotation.")
            return

        if not self.active_image or not self.image_pixmap or not self.cursorInWindow(scene_pos, mapped=True):
            return

        if annotation is None:
            annotation = PatchAnnotation(scene_pos,
                                         self.annotation_size,
                                         self.selected_label.short_label_code,
                                         self.selected_label.long_label_code,
                                         self.selected_label.color,
                                         self.current_image_path,
                                         self.selected_label.id,
                                         transparency=self.main_window.label_window.active_label.transparency)

        annotation.create_graphics_item(self.scene)
        annotation.create_cropped_image(self.rasterio_image)

        # Connect update signals
        annotation.selected.connect(self.select_annotation)
        annotation.annotation_deleted.connect(self.delete_annotation)
        annotation.annotation_updated.connect(self.main_window.confidence_window.display_cropped_image)

        self.annotations_dict[annotation.id] = annotation

        self.main_window.confidence_window.display_cropped_image(annotation)

    def delete_annotation(self, annotation_id):
        if annotation_id in self.annotations_dict:
            annotation = self.annotations_dict[annotation_id]
            annotation.delete()
            del self.annotations_dict[annotation_id]

    def delete_selected_annotation(self):
        if self.selected_annotation:
            self.delete_annotation(self.selected_annotation.id)
            self.selected_annotation = None
            # Clear the confidence window
            self.main_window.confidence_window.clear_display()

    def delete_annotations(self, annotations):
        for annotation in annotations:
            self.delete_annotation(annotation.id)

    def delete_label_annotations(self, label):
        for annotation in list(self.annotations_dict.values()):
            if annotation.label.id == label.id:
                annotation.delete()
                del self.annotations_dict[annotation.id]

    def delete_image_annotations(self, image_path):
        annotations = self.get_image_annotations(image_path)
        self.delete_annotations(annotations)

    def delete_image(self, image_path):
        # Delete all annotations associated with image path
        self.delete_annotations(self.get_image_annotations(image_path))
        # Delete the image
        if self.current_image_path == image_path:
            self.scene.clear()
            self.current_image_path = None
            self.image_pixmap = None
            self.rasterio_image = None
            self.active_image = False  # Reset image_set flag

    def clear_scene(self):
        # Clean up
        self.unselect_annotation()

        # Clear the previous scene and delete its items
        if self.scene:
            for item in self.scene.items():
                self.scene.removeItem(item)
                del item
            self.scene.deleteLater()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)