import json
import os
import random
import uuid
import warnings

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, pyqtSignal, QObject, QPointF, QRectF
from PyQt5.QtGui import QMouseEvent, QImage, QPixmap, QColor, QPen, QBrush
from PyQt5.QtWidgets import (QFileDialog, QApplication, QGraphicsView, QGraphicsScene, QMessageBox, QCheckBox,
                             QVBoxLayout, QLabel, QDialog, QHBoxLayout, QPushButton, QComboBox, QSpinBox,
                             QGraphicsPixmapItem, QGraphicsRectItem, QFormLayout, QInputDialog, QLineEdit,
                             QDialogButtonBox)

from rasterio.windows import Window

from toolbox.QtLabel import Label
from toolbox.QtProgressBar import ProgressBar

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Annotation(QObject):
    color_changed = pyqtSignal(QColor)
    selected = pyqtSignal(object)
    annotation_deleted = pyqtSignal(object)
    annotation_updated = pyqtSignal(object)

    def __init__(self, center_xy: QPointF,
                 annotation_size: int,
                 short_label_code: str,
                 long_label_code: str,
                 color: QColor,
                 image_path: str,
                 label_id: str,
                 transparency: int = 128,
                 show_msg=True):
        super().__init__()
        self.id = str(uuid.uuid4())
        self.center_xy = center_xy
        self.annotation_size = annotation_size
        self.label = Label(short_label_code, long_label_code, color, label_id)
        self.image_path = image_path
        self.is_selected = False
        self.graphics_item = None
        self.transparency = transparency
        self.user_confidence = {self.label: 1.0}
        self.machine_confidence = {}
        self.data = {}
        self.cropped_image = None

        self.show_message = show_msg

    def show_warning_message(self):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("Warning")
        msg_box.setText("Altering an annotation with predictions will remove the machine suggestions.")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

        # Only show once
        self.show_message = False

    def contains_point(self, point: QPointF) -> bool:
        half_size = self.annotation_size / 2
        rect = QRectF(self.center_xy.x() - half_size,
                      self.center_xy.y() - half_size,
                      self.annotation_size,
                      self.annotation_size)
        return rect.contains(point)

    def select(self):
        self.is_selected = True
        self.update_graphics_item()

    def deselect(self):
        self.is_selected = False
        self.update_graphics_item()

    def delete(self):
        self.annotation_deleted.emit(self)
        if self.graphics_item and self.graphics_item.scene():
            self.graphics_item.scene().removeItem(self.graphics_item)
            self.graphics_item = None

    def create_cropped_image(self, rasterio_src):
        half_size = self.annotation_size / 2

        # Convert center coordinates to pixel coordinates
        pixel_x = int(self.center_xy.x())
        pixel_y = int(self.center_xy.y())

        # Calculate the window for rasterio
        window = Window(
            col_off=max(0, pixel_x - half_size),
            row_off=max(0, pixel_y - half_size),
            width=min(rasterio_src.width - (pixel_x - half_size), self.annotation_size),
            height=min(rasterio_src.height - (pixel_y - half_size), self.annotation_size)
        )

        # Read the data from rasterio
        data = rasterio_src.read(window=window)

        # Ensure the data is in the correct format for QImage
        if data.shape[0] == 3:  # RGB image
            data = np.transpose(data, (1, 2, 0))
        elif data.shape[0] == 1:  # Grayscale image
            data = np.squeeze(data)

        # Normalize data to 0-255 range if it's not already
        if data.dtype != np.uint8:
            data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)

        # Convert numpy array to QImage
        height, width = data.shape[:2]
        bytes_per_line = 3 * width if len(data.shape) == 3 else width
        image_format = QImage.Format_RGB888 if len(data.shape) == 3 else QImage.Format_Grayscale8

        # Convert numpy array to bytes
        if len(data.shape) == 3:
            data = data.tobytes()
        else:
            data = np.expand_dims(data, -1).tobytes()

        q_image = QImage(data, width, height, bytes_per_line, image_format)

        # Convert QImage to QPixmap
        self.cropped_image = QPixmap.fromImage(q_image)

        self.annotation_updated.emit(self)  # Notify update

    def create_graphics_item(self, scene: QGraphicsScene):
        half_size = self.annotation_size / 2
        self.graphics_item = QGraphicsRectItem(self.center_xy.x() - half_size,
                                               self.center_xy.y() - half_size,
                                               self.annotation_size,
                                               self.annotation_size)
        self.update_graphics_item()
        self.graphics_item.setData(0, self.id)
        scene.addItem(self.graphics_item)

    def update_machine_confidence(self, prediction: dict):
        # Set user confidence to None
        self.user_confidence = {}
        # Update machine confidence
        self.machine_confidence = prediction
        # Pass the label with the largest confidence as the label
        self.label = max(prediction, key=prediction.get)
        # Create the graphic
        self.update_graphics_item()

    def update_user_confidence(self, new_label: 'Label'):
        # Set machine confidence to None
        self.machine_confidence = {}
        # Update user confidence
        self.user_confidence = {new_label: 1.0}
        # Pass the label with the largest confidence as the label
        self.label = new_label
        # Create the graphic
        self.update_graphics_item()

    def update_label(self, new_label: 'Label'):
        self.label = new_label
        self.update_graphics_item()

    def update_location(self, new_center_xy: QPointF):
        if self.machine_confidence and self.show_message:
            self.show_warning_message()
            return

        # Clear the machine confidence
        self.update_user_confidence(self.label)
        # Update the location, graphic
        self.center_xy = new_center_xy
        self.update_graphics_item()
        self.annotation_updated.emit(self)  # Notify update

    def update_annotation_size(self, size):
        if self.machine_confidence and self.show_message:
            self.show_warning_message()
            return

        # Clear the machine confidence
        self.update_user_confidence(self.label)
        # Update the size, graphic
        self.annotation_size = size
        self.update_graphics_item()
        self.annotation_updated.emit(self)  # Notify update

    def update_transparency(self, transparency: int):
        self.transparency = transparency
        self.update_graphics_item()

    def update_graphics_item(self):
        if self.graphics_item:
            half_size = self.annotation_size / 2
            self.graphics_item.setRect(self.center_xy.x() - half_size,
                                       self.center_xy.y() - half_size,
                                       self.annotation_size,
                                       self.annotation_size)
            color = QColor(self.label.color)
            color.setAlpha(self.transparency)

            if self.is_selected:
                inverse_color = QColor(255 - color.red(), 255 - color.green(), 255 - color.blue())
                pen = QPen(inverse_color, 4, Qt.DotLine)  # Inverse color, thicker border, and dotted line
            else:
                pen = QPen(color, 2, Qt.SolidLine)  # Default border color and thickness

            self.graphics_item.setPen(pen)
            brush = QBrush(color)
            self.graphics_item.setBrush(brush)
            self.graphics_item.update()

    def to_coralnet_format(self):
        return [os.path.basename(self.image_path),
                int(self.center_xy.y()),
                int(self.center_xy.x()),
                self.label.short_label_code,
                self.label.long_label_code,
                self.annotation_size]

    def to_dict(self):
        return {
            'id': self.id,
            'center_xy': (self.center_xy.x(), self.center_xy.y()),
            'annotation_size': self.annotation_size,
            'label_short_code': self.label.short_label_code,
            'label_long_code': self.label.long_label_code,
            'annotation_color': self.label.color.getRgb(),
            'image_path': self.image_path,
            'label_id': self.label.id,
            'data': self.data
        }

    @classmethod
    def from_dict(cls, data):
        annotation = cls(QPointF(*data['center_xy']),
                         data['annotation_size'],
                         data['label_short_code'],
                         data['label_long_code'],
                         QColor(*data['annotation_color']),
                         data['image_path'],
                         data['label_id'])
        annotation.data = data.get('data', {})
        return annotation

    def __repr__(self):
        return (f"Annotation(id={self.id}, center_xy={self.center_xy}, "
                f"annotation_size={self.annotation_size}, "
                f"annotation_color={self.label.color.name()}, "
                f"image_path={self.image_path}, "
                f"label={self.label.short_label_code}, "
                f"data={self.data})")


class AnnotationWindow(QGraphicsView):
    imageLoaded = pyqtSignal(int, int)  # Signal to emit when image is loaded
    mouseMoved = pyqtSignal(int, int)  # Signal to emit when mouse is moved
    toolChanged = pyqtSignal(str)  # Signal to emit when the tool changes
    labelSelected = pyqtSignal(str)  # Signal to emit when the label changes
    annotationSizeChanged = pyqtSignal(int)  # Signal to emit when annotation size changes

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
                    QApplication.processEvents()  # Update GUI

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
                        QApplication.processEvents()  # Update GUI

                if updated_annotations:
                    # Inform the user that some annotations have been updated
                    QMessageBox.information(self,
                                            "Annotations Updated",
                                            "Some annotations have been updated to match the "
                                            "color of the labels already in the project.")

                # Add annotations to the AnnotationWindow dict
                for image_path, annotations in filtered_annotations.items():
                    for annotation_data in annotations:
                        annotation = Annotation.from_dict(annotation_data)
                        self.annotations_dict[annotation.id] = annotation

                        progress_bar.update_progress()
                        QApplication.processEvents()

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
                    QApplication.processEvents()  # Update GUI

                df = pd.DataFrame(data, columns=['Name', 'Row', 'Column', 'Label', 'Long Label', 'Patch Size'])
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

            # Drop everything else
            df = df[required_columns]
            df = df.dropna(how='any')
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
                    annotation = Annotation(QPointF(col_coord, row_coord),
                                            annotation_size,
                                            short_label_code,
                                            long_label_code,
                                            color,
                                            image_path,
                                            label_id)

                    # Add to the AnnotationWindow dictionary
                    self.annotations_dict[annotation.id] = annotation

                    progress_bar.update_progress()
                    QApplication.processEvents()

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
                            QApplication.processEvents()  # Update GUI

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
                        annotation = Annotation(QPointF(col_coord, row_coord),
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
                        QApplication.processEvents()

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
        elif self.selected_tool == "annotate":
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        self.unselect_annotation()

    def set_selected_label(self, label):
        self.selected_label = label
        self.annotation_color = label.color

        if self.selected_annotation:
            if self.selected_annotation.label.id != label.id:
                self.selected_annotation.update_label(self.selected_label)
                self.selected_annotation.create_cropped_image(self.rasterio_image)
                # Notify ConfidenceWindow the selected annotation has changed
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

    def set_transparency(self, transparency: int):
        self.transparency = transparency

    def toggle_cursor_annotation(self, scene_pos: QPointF = None):
        if scene_pos:
            if not self.selected_label or not self.annotation_color:
                return

            if not self.cursor_annotation:
                self.cursor_annotation = Annotation(scene_pos,
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

        # Set the image representations
        self.image_pixmap = QPixmap(image_item)
        self.scene.addItem(QGraphicsPixmapItem(self.image_pixmap))
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        # Clear the confidence window
        self.main_window.confidence_window.clear_display()

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

        # Load all associated annotations
        self.load_annotations()
        # Update the image window's image dict
        self.main_window.image_window.update_image_annotations(image_path)

        # Clear the confidence window
        self.main_window.confidence_window.clear_display()

    def wheelEvent(self, event: QMouseEvent):
        if event.angleDelta().y() > 0:
            factor = 1.1
        else:
            factor = 0.9

        self.zoom_factor *= factor
        self.scale(factor, factor)

        if self.selected_tool == "select":
            self.setCursor(Qt.PointingHandCursor)
        elif self.selected_tool == "annotate":
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

            elif self.selected_tool == "annotate" and event.button() == Qt.LeftButton:
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
        elif (self.selected_tool == "annotate" and
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

    def update_annotations_transparency(self, label, transparency):
        self.set_transparency(transparency)
        for annotation in self.annotations_dict.values():
            if annotation.label.id == label.id:
                annotation.update_transparency(transparency)

    def load_annotations(self):
        # Crop all the annotations for current image (if not already cropped)
        annotations = self.crop_image_annotations(return_annotations=True)

        # Connect update signals for all the annotations
        for annotation in annotations:
            # Create the graphics item (scene previously cleared)
            annotation.create_graphics_item(self.scene)

            # Connect update signals
            annotation.selected.connect(self.select_annotation)
            annotation.annotation_deleted.connect(self.delete_annotation)
            annotation.annotation_updated.connect(self.main_window.confidence_window.display_cropped_image)

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
            annotation = Annotation(scene_pos,
                                    self.annotation_size,
                                    self.selected_label.short_label_code,
                                    self.selected_label.long_label_code,
                                    self.selected_label.color,
                                    self.current_image_path,
                                    self.selected_label.id,
                                    transparency=self.transparency)

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


class AnnotationSamplingDialog(QDialog):
    annotationsSampled = pyqtSignal(list, bool)  # Signal to emit the sampled annotations and apply to all flag

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.annotation_window = main_window.annotation_window
        self.label_window = main_window.label_window
        self.image_window = main_window.image_window
        self.deploy_model_dialog = main_window.deploy_model_dialog

        self.setWindowTitle("Sample Annotations")
        self.setWindowState(Qt.WindowMaximized)  # Ensure the dialog is maximized

        self.layout = QVBoxLayout(self)

        # Sampling Method
        self.method_label = QLabel("Sampling Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Random", "Stratified Random", "Uniform"])
        self.layout.addWidget(self.method_label)
        self.layout.addWidget(self.method_combo)

        # Number of Annotations
        self.num_annotations_label = QLabel("Number of Annotations:")
        self.num_annotations_spinbox = QSpinBox()
        self.num_annotations_spinbox.setMinimum(0)
        self.num_annotations_spinbox.setMaximum(10000)
        self.layout.addWidget(self.num_annotations_label)
        self.layout.addWidget(self.num_annotations_spinbox)

        # Annotation Size
        self.annotation_size_label = QLabel("Annotation Size:")
        self.annotation_size_spinbox = QSpinBox()
        self.annotation_size_spinbox.setMinimum(0)
        self.annotation_size_spinbox.setMaximum(10000)  # Arbitrary large number for "infinite"
        self.annotation_size_spinbox.setValue(self.annotation_window.annotation_size)
        self.layout.addWidget(self.annotation_size_label)
        self.layout.addWidget(self.annotation_size_spinbox)

        # Margin Offsets using QFormLayout
        self.margin_form_layout = QFormLayout()
        self.margin_x_min_spinbox = self.create_margin_spinbox("X Min", self.margin_form_layout)
        self.margin_y_min_spinbox = self.create_margin_spinbox("Y Min", self.margin_form_layout)
        self.margin_x_max_spinbox = self.create_margin_spinbox("X Max", self.margin_form_layout)
        self.margin_y_max_spinbox = self.create_margin_spinbox("Y Max", self.margin_form_layout)
        self.layout.addLayout(self.margin_form_layout)

        # Apply to Filtered Images Checkbox
        self.apply_filtered_checkbox = QCheckBox("Apply to filtered images")
        self.layout.addWidget(self.apply_filtered_checkbox)
        # Apply to Previous Images Checkbox
        self.apply_prev_checkbox = QCheckBox("Apply to previous images")
        self.layout.addWidget(self.apply_prev_checkbox)
        # Apply to Next Images Checkbox
        self.apply_next_checkbox = QCheckBox("Apply to next images")
        self.layout.addWidget(self.apply_next_checkbox)
        # Apply to All Images Checkbox
        self.apply_all_checkbox = QCheckBox("Apply to all images")
        self.layout.addWidget(self.apply_all_checkbox)

        # Ensure only one of the apply checkboxes can be selected at a time
        self.apply_filtered_checkbox.stateChanged.connect(self.update_apply_filtered_checkboxes)
        self.apply_prev_checkbox.stateChanged.connect(self.update_apply_prev_checkboxes)
        self.apply_next_checkbox.stateChanged.connect(self.update_apply_next_checkboxes)
        self.apply_all_checkbox.stateChanged.connect(self.update_apply_all_checkboxes)

        # Preview Button
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.preview_annotations)
        self.layout.addWidget(self.preview_button)

        # Preview Area
        self.preview_view = QGraphicsView(self)
        self.preview_scene = QGraphicsScene(self)
        self.preview_view.setScene(self.preview_scene)
        self.preview_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.preview_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.layout.addWidget(self.preview_view)

        # Accept/Cancel Buttons
        self.button_box = QHBoxLayout()
        self.accept_button = QPushButton("Accept")
        self.accept_button.clicked.connect(self.accept_annotations)
        self.button_box.addWidget(self.accept_button)

        self.layout.addLayout(self.button_box)

        self.sampled_annotations = []

    def showEvent(self, event):
        super().showEvent(event)
        self.reset_defaults()

    def reset_defaults(self):
        self.preview_scene.clear()
        self.method_combo.setCurrentIndex(0)
        self.num_annotations_spinbox.setValue(1)
        self.annotation_size_spinbox.setValue(self.annotation_window.annotation_size)
        self.margin_x_min_spinbox.setValue(0)
        self.margin_y_min_spinbox.setValue(0)
        self.margin_x_max_spinbox.setValue(0)
        self.margin_y_max_spinbox.setValue(0)
        self.apply_prev_checkbox.setChecked(False)
        self.apply_next_checkbox.setChecked(False)
        self.apply_all_checkbox.setChecked(False)

    def create_margin_spinbox(self, label_text, layout):
        label = QLabel(label_text + ":")
        spinbox = QSpinBox()
        spinbox.setMinimum(0)
        spinbox.setMaximum(1000)
        layout.addRow(label, spinbox)
        return spinbox

    def update_apply_filtered_checkboxes(self):
        if self.apply_filtered_checkbox.isChecked():
            self.apply_filtered_checkbox.setChecked(True)
            self.apply_prev_checkbox.setChecked(False)
            self.apply_next_checkbox.setChecked(False)
            self.apply_all_checkbox.setChecked(False)
            return

        if not self.apply_filtered_checkbox.isChecked():
            self.apply_filtered_checkbox.setChecked(False)
            return

    def update_apply_prev_checkboxes(self):
        if self.apply_prev_checkbox.isChecked():
            self.apply_prev_checkbox.setChecked(True)
            self.apply_filtered_checkbox.setChecked(False)
            self.apply_next_checkbox.setChecked(False)
            self.apply_all_checkbox.setChecked(False)
            return

        if not self.apply_prev_checkbox.isChecked():
            self.apply_prev_checkbox.setChecked(False)
            return

    def update_apply_next_checkboxes(self):
        if self.apply_next_checkbox.isChecked():
            self.apply_next_checkbox.setChecked(True)
            self.apply_filtered_checkbox.setChecked(False)
            self.apply_prev_checkbox.setChecked(False)
            self.apply_all_checkbox.setChecked(False)
            return

        if not self.apply_next_checkbox.isChecked():
            self.apply_next_checkbox.setChecked(False)
            return

    def update_apply_all_checkboxes(self):
        if self.apply_all_checkbox.isChecked():
            self.apply_all_checkbox.setChecked(True)
            self.apply_filtered_checkbox.setChecked(False)
            self.apply_prev_checkbox.setChecked(False)
            self.apply_next_checkbox.setChecked(False)
            return

        if not self.apply_all_checkbox.isChecked():
            self.apply_all_checkbox.setChecked(False)
            return

    def sample_annotations(self, method, num_annotations, annotation_size, margins, image_width, image_height):
        # Extract the margins
        margin_x_min, margin_y_min, margin_x_max, margin_y_max = margins

        annotations = []

        if method == "Random":
            for _ in range(num_annotations):
                x = random.randint(margin_x_min, image_width - annotation_size - margin_x_max)
                y = random.randint(margin_y_min, image_height - annotation_size - margin_y_max)
                annotations.append((x, y, annotation_size))

        if method == "Uniform":
            grid_size = int(num_annotations ** 0.5)
            x_step = (image_width - margin_x_min - margin_x_max - annotation_size) / grid_size
            y_step = (image_height - margin_y_min - margin_y_max - annotation_size) / grid_size
            for i in range(grid_size):
                for j in range(grid_size):
                    x = margin_x_min + int(i * x_step + annotation_size / 2)
                    y = margin_y_min + int(j * y_step + annotation_size / 2)
                    annotations.append((x, y, annotation_size))

        if method == "Stratified Random":
            grid_size = int(num_annotations ** 0.5)
            x_step = (image_width - margin_x_min - margin_x_max - annotation_size) / grid_size
            y_step = (image_height - margin_y_min - margin_y_max - annotation_size) / grid_size
            for i in range(grid_size):
                for j in range(grid_size):
                    x = margin_x_min + int(
                        i * x_step + random.uniform(annotation_size / 2, x_step - annotation_size / 2))
                    y = margin_y_min + int(
                        j * y_step + random.uniform(annotation_size / 2, y_step - annotation_size / 2))
                    annotations.append((x, y, annotation_size))

        return annotations

    def preview_annotations(self):
        method = self.method_combo.currentText()
        num_annotations = self.num_annotations_spinbox.value()
        annotation_size = self.annotation_size_spinbox.value()
        margin_x_min = self.margin_x_min_spinbox.value()
        margin_y_min = self.margin_y_min_spinbox.value()
        margin_x_max = self.margin_x_max_spinbox.value()
        margin_y_max = self.margin_y_max_spinbox.value()

        margins = margin_x_min, margin_y_min, margin_x_max, margin_y_max

        self.sampled_annotations = self.sample_annotations(method,
                                                           num_annotations,
                                                           annotation_size,
                                                           margins,
                                                           self.annotation_window.image_pixmap.width(),
                                                           self.annotation_window.image_pixmap.height())

        self.draw_annotation_previews(margins)

    def draw_annotation_previews(self, margins):

        margin_x_min, margin_y_min, margin_x_max, margin_y_max = margins

        self.preview_scene.clear()
        pixmap = self.annotation_window.image_pixmap
        if pixmap:
            # Add the image to the scene
            self.preview_scene.addItem(QGraphicsPixmapItem(pixmap))

            # Draw annotations
            for annotation in self.sampled_annotations:
                x, y, size = annotation
                rect_item = QGraphicsRectItem(x, y, size, size)
                rect_item.setPen(QPen(Qt.white, 4))
                brush = QBrush(Qt.white)
                brush.setStyle(Qt.SolidPattern)
                color = brush.color()
                color.setAlpha(75)
                brush.setColor(color)
                rect_item.setBrush(brush)
                self.preview_scene.addItem(rect_item)

            # Draw margin lines
            pen = QPen(QColor("red"), 5)
            pen.setStyle(Qt.DotLine)
            image_width = pixmap.width()
            image_height = pixmap.height()

            self.preview_scene.addLine(margin_x_min, 0, margin_x_min, image_height, pen)
            self.preview_scene.addLine(image_width - margin_x_max, 0, image_width - margin_x_max, image_height, pen)
            self.preview_scene.addLine(0, margin_y_min, image_width, margin_y_min, pen)
            self.preview_scene.addLine(0, image_height - margin_y_max, image_width, image_height - margin_y_max, pen)

            # Apply dark transparency outside the margins
            overlay_color = QColor(0, 0, 0, 150)  # Black with transparency

            # Left overlay
            left_overlay = QGraphicsRectItem(0, 0, margin_x_min, image_height)
            left_overlay.setBrush(QBrush(overlay_color))
            left_overlay.setPen(QPen(Qt.NoPen))
            self.preview_scene.addItem(left_overlay)

            # Right overlay
            right_overlay = QGraphicsRectItem(image_width - margin_x_max, 0, margin_x_max, image_height)
            right_overlay.setBrush(QBrush(overlay_color))
            right_overlay.setPen(QPen(Qt.NoPen))
            self.preview_scene.addItem(right_overlay)

            # Top overlay
            top_overlay = QGraphicsRectItem(margin_x_min, 0, image_width - margin_x_min - margin_x_max, margin_y_min)
            top_overlay.setBrush(QBrush(overlay_color))
            top_overlay.setPen(QPen(Qt.NoPen))
            self.preview_scene.addItem(top_overlay)

            # Bottom overlay
            bottom_overlay = QGraphicsRectItem(margin_x_min,
                                               image_height - margin_y_max,
                                               image_width - margin_x_min - margin_x_max,
                                               margin_y_max)

            bottom_overlay.setBrush(QBrush(overlay_color))
            bottom_overlay.setPen(QPen(Qt.NoPen))
            self.preview_scene.addItem(bottom_overlay)

            self.preview_view.fitInView(self.preview_scene.sceneRect(), Qt.KeepAspectRatio)

    def accept_annotations(self):

        margins = (self.margin_x_min_spinbox.value(),
                   self.margin_y_min_spinbox.value(),
                   self.margin_x_max_spinbox.value(),
                   self.margin_y_max_spinbox.value())

        self.add_sampled_annotations(self.method_combo.currentText(),
                                     self.num_annotations_spinbox.value(),
                                     self.annotation_size_spinbox.value(),
                                     margins)
        self.accept()

    def add_sampled_annotations(self, method, num_annotations, annotation_size, margins):

        # Set the cursor to waiting (busy) cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        self.apply_to_filtered = self.apply_filtered_checkbox.isChecked()
        self.apply_to_prev = self.apply_prev_checkbox.isChecked()
        self.apply_to_next = self.apply_next_checkbox.isChecked()
        self.apply_to_all = self.apply_all_checkbox.isChecked()

        # Sets the LabelWindow and AnnotationWindow to Review
        self.label_window.set_selected_label("-1")
        review_label = self.annotation_window.selected_label

        # Current image path showing
        current_image_path = self.annotation_window.current_image_path

        if self.apply_to_filtered:
            image_paths = self.image_window.filtered_image_paths
        elif self.apply_to_prev:
            current_image_index = self.image_window.image_paths.index(current_image_path)
            image_paths = self.image_window.image_paths[:current_image_index + 1]
        elif self.apply_to_next:
            current_image_index = self.image_window.image_paths.index(current_image_path)
            image_paths = self.image_window.image_paths[current_image_index:]
        elif self.apply_to_all:
            image_paths = self.image_window.image_paths
        else:
            # Only apply to the current image
            image_paths = [current_image_path]

        # Create and show the progress bar
        progress_bar = ProgressBar(self, title="Sampling Annotations")
        progress_bar.show()
        progress_bar.start_progress(len(image_paths) * num_annotations)

        for image_path in image_paths:
            # Load the rasterio representation
            rasterio_image = self.image_window.rasterio_open(image_path)
            height, width = rasterio_image.shape[0:2]

            # Sample the annotation, given params
            annotations = self.sample_annotations(method,
                                                  num_annotations,
                                                  annotation_size,
                                                  margins,
                                                  width,
                                                  height)

            for annotation in annotations:
                x, y, size = annotation
                new_annotation = Annotation(QPointF(x + size // 2, y + size // 2),
                                            size,
                                            review_label.short_label_code,
                                            review_label.long_label_code,
                                            review_label.color,
                                            image_path,
                                            review_label.id,
                                            transparency=self.annotation_window.transparency)

                # Add annotation to the dict
                self.annotation_window.annotations_dict[new_annotation.id] = new_annotation

                # Update the progress bar
                progress_bar.update_progress()
                QApplication.processEvents()  # Update GUI

            # Update the image window's image dict
            self.image_window.update_image_annotations(image_path)

        # Stop the progress bar
        progress_bar.stop_progress()
        progress_bar.close()

        # # Set / load the image / annotations of the last image
        self.annotation_window.set_image(current_image_path)
        # Reset dialog for next time
        self.reset_defaults()

        # Restore the cursor to the default cursor
        QApplication.restoreOverrideCursor()