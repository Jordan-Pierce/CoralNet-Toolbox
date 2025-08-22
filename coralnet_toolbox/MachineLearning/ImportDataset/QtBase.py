import warnings

import os
import uuid
import yaml
import glob
import shutil
import ujson as json

from PyQt5.QtCore import Qt, QPointF, QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QVBoxLayout, QGroupBox,
                             QLabel, QLineEdit, QDialog, QPushButton, QDialogButtonBox,
                             QGridLayout, QScrollArea, QFrame, QCheckBox, QRadioButton,
                             QToolButton)

from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.utilities import rasterio_open
from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Worker Class for Threading
# ----------------------------------------------------------------------------------------------------------------------


class DatasetProcessor(QObject):
    """
    Worker object to process a dataset in a separate thread.
    It is completely decoupled from the GUI.
    """
    status_changed = pyqtSignal(str, int)
    progress_updated = pyqtSignal(int)
    processing_complete = pyqtSignal(list, list, list)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, yaml_path, output_folder, task, import_as, rename_on_conflict=False,
                 excluded_classes=None, image_import_policy='annotated_only', parent=None):
        super().__init__(parent)
        self.yaml_path = yaml_path
        self.output_folder = output_folder
        self.task = task  # 'detect' or 'segment' (source format)
        self.import_as = import_as  # 'rectangle' or 'polygon' (target format)
        self.rename_on_conflict = rename_on_conflict
        self.excluded_classes = excluded_classes if excluded_classes is not None else set()
        self.image_import_policy = image_import_policy
        self.is_running = True
        self.parsing_errors = []  # To collect errors instead of printing

    def stop(self):
        self.is_running = False

    # In class DatasetProcessor, update this method:

    def run(self):
        """Main processing method executed in the thread."""
        try:
            # Step 1: Read YAML and discover files
            with open(self.yaml_path, 'r') as file:
                data = yaml.safe_load(file)
            class_names = data.get('names', [])

            source_image_label_map = self._find_source_files()

            if not source_image_label_map:
                self.error.emit("No valid image/label pairs found in the dataset.")
                self.finished.emit()
                return

            # --- Step 2: Copy files with progress reporting ---
            self.status_changed.emit("Copying image files...", len(source_image_label_map))
            image_label_paths = self._copy_files_with_progress(source_image_label_map)

            if not self.is_running:
                self.finished.emit()
                return

            # Step 3: Parse label files and create annotation data
            self.status_changed.emit("Importing annotations...", len(image_label_paths))
            raw_annotations = self._create_raw_annotations(image_label_paths, class_names)

            if not self.is_running:
                self.finished.emit()
                return

            # Step 4 (REMOVED): The JSON export is no longer done here.

            # Step 5: Emit results for GUI to consume
            image_paths = list(image_label_paths.keys())
            self.processing_complete.emit(raw_annotations, image_paths, self.parsing_errors)

        except Exception as e:
            self.error.emit(f"An error occurred during processing: {str(e)}")
        finally:
            self.finished.emit()

    def _find_source_files(self):
        """Finds all source image and label paths based on the import policy."""
        dir_path = os.path.dirname(self.yaml_path)
        image_paths = glob.glob(f"{dir_path}/**/images/*.*", recursive=True)
        label_paths = glob.glob(f"{dir_path}/**/labels/*.txt", recursive=True)

        source_map = {}
        if self.image_import_policy == 'all':
            # Policy: Find all images, and match labels to them if they exist
            label_basenames_map = {os.path.splitext(os.path.basename(p))[0]: p for p in label_paths}
            for img_path in image_paths:
                img_basename = os.path.splitext(os.path.basename(img_path))[0]
                label_path = label_basenames_map.get(img_basename, None)  # Label can be None
                source_map[img_path] = label_path
        else:
            # Policy: Only find images that have a corresponding label file
            image_basenames_map = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}
            for label_path in label_paths:
                label_basename_no_ext = os.path.splitext(os.path.basename(label_path))[0]
                if label_basename_no_ext in image_basenames_map:
                    src_image_path = image_basenames_map[label_basename_no_ext]
                    source_map[src_image_path] = label_path
        return source_map

    def _copy_files_with_progress(self, source_image_label_map):
        """Copies files and reports progress for each file."""
        img_out_dir = os.path.join(self.output_folder, "images")
        os.makedirs(img_out_dir, exist_ok=True)

        dest_label_map = {}
        for i, (src_image_path, label_path) in enumerate(source_image_label_map.items()):
            if not self.is_running:
                break

            original_img_basename = os.path.basename(src_image_path)

            if self.rename_on_conflict:
                base, ext = os.path.splitext(original_img_basename)
                unique_id = str(uuid.uuid4())[:8]
                new_img_basename = f"{base}_{unique_id}{ext}"
            else:
                new_img_basename = original_img_basename

            dest_image_path = os.path.join(img_out_dir, new_img_basename)
            shutil.copy(src_image_path, dest_image_path)

            dest_label_map[dest_image_path.replace("\\", "/")] = label_path
            self.progress_updated.emit(i + 1)

        return dest_label_map

    def _create_raw_annotations(self, image_label_paths, class_names):
        """
        Parses label files, converts format if needed, and creates raw annotation data.
        Returns a list of annotation dictionaries.
        """
        all_raw_annotations = []
        for i, (image_path, label_path) in enumerate(image_label_paths.items()):
            if not self.is_running:
                break

            # If there's no label file for this image (e.g., 'import all' policy), skip to progress update
            if not label_path:
                self.progress_updated.emit(i + 1)
                continue

            image_height, image_width = rasterio_open(image_path).shape
            with open(label_path, 'r') as file:
                lines = file.readlines()

            for line_num, line in enumerate(lines):
                try:
                    parts = list(map(float, line.split()))
                    class_id = int(parts[0])

                    class_name = class_names[class_id]
                    if class_name in self.excluded_classes:
                        continue  # Skip this annotation if its class was unchecked

                    raw_ann_data = {"image_path": image_path, "class_name": class_name}

                    parsed_data = {}
                    if self.task == 'detect':  # Source is bbox: class, x_c, y_c, w, h
                        _, x_c, y_c, w, h = parts
                        x = x_c * image_width
                        y = y_c * image_height
                        width = w * image_width
                        height = h * image_height
                        parsed_data['top_left'] = (x - width / 2, y - height / 2)
                        parsed_data['bottom_right'] = (x + width / 2, y + height / 2)
                    else:  # Source is polygon: class, x1, y1, x2, y2, ...
                        points_norm = parts[1:]
                        # Convert normalized coordinates to pixel coordinates
                        # Extract x and y coordinates from the flattened list
                        x_coords = points_norm[::2]  # Every even index (0, 2, 4...)
                        y_coords = points_norm[1::2]  # Every odd index (1, 3, 5...)
                        
                        # Scale coordinates by image dimensions
                        points = []
                        for x, y in zip(x_coords, y_coords):
                            pixel_x = x * image_width
                            pixel_y = y * image_height
                            points.append((pixel_x, pixel_y))
                        parsed_data['points'] = points

                    if self.import_as == 'rectangle':
                        raw_ann_data["type"] = "RectangleAnnotation"
                        if 'top_left' in parsed_data:
                            raw_ann_data.update(parsed_data)
                        else:
                            points = parsed_data['points']
                            x_coords = [p[0] for p in points]
                            y_coords = [p[1] for p in points]
                            raw_ann_data["top_left"] = (min(x_coords), min(y_coords))
                            raw_ann_data["bottom_right"] = (max(x_coords), max(y_coords))
                    elif self.import_as == 'polygon':
                        raw_ann_data["type"] = "PolygonAnnotation"
                        if 'points' in parsed_data:
                            raw_ann_data.update(parsed_data)
                        else:
                            tl, br = parsed_data['top_left'], parsed_data['bottom_right']
                            raw_ann_data["points"] = [(tl[0], tl[1]), (br[0], tl[1]), (br[0], br[1]), (tl[0], br[1])]
                    all_raw_annotations.append(raw_ann_data)
                except (ValueError, IndexError) as e:
                    error_msg = (f"In file '{os.path.basename(label_path)}' on line {line_num + 1}:\n"
                                 f"Skipped malformed content: '{line.strip()}'\nReason: {e}\n")
                    self.parsing_errors.append(error_msg)

            self.progress_updated.emit(i + 1)
        return all_raw_annotations


# ----------------------------------------------------------------------------------------------------------------------
# Dialog Classes
# ----------------------------------------------------------------------------------------------------------------------


class Base(QDialog):
    def __init__(self, main_window, parent=None):
        super(Base, self).__init__(parent)
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window
        self.image_window = main_window.image_window

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Import Dataset")
        self.resize(500, 350)

        self.task = None
        self.progress_bar = None
        self.thread = None
        self.worker = None
        self.output_folder = None
        self.class_checkboxes = []

        self.layout = QVBoxLayout(self)
        self.setup_info_layout()
        self.setup_yaml_layout()
        self.setup_output_layout()
        self.setup_buttons_layout()

        self.advanced_options_toggle.setEnabled(False)
        self.advanced_options_frame.setVisible(False)

    def setup_info_layout(self):
        raise NotImplementedError("Subclasses must implement method.")

    def setup_yaml_layout(self):
        """Set up the layout for selecting the data YAML file."""
        group_box = QGroupBox("Data YAML File")
        layout = QGridLayout()
        layout.addWidget(QLabel("File:"), 0, 0)
        self.yaml_path_label = QLineEdit()
        self.yaml_path_label.setReadOnly(True)
        self.yaml_path_label.setPlaceholderText("Select Data YAML file...")
        layout.addWidget(self.yaml_path_label, 0, 1)
        self.browse_yaml_button = QPushButton("Browse")
        self.browse_yaml_button.clicked.connect(self.browse_data_yaml)
        layout.addWidget(self.browse_yaml_button, 0, 2)
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_output_layout(self):
        """Set up the layout for output directory and the advanced options accordion."""
        group_box = QGroupBox("Output Settings")
        layout = QGridLayout()
        layout.addWidget(QLabel("Directory:"), 0, 0)
        self.output_dir_label = QLineEdit()
        self.output_dir_label.setPlaceholderText("Select output directory...")
        layout.addWidget(self.output_dir_label, 0, 1)
        self.browse_output_button = QPushButton("Browse")
        self.browse_output_button.clicked.connect(self.browse_output_dir)
        layout.addWidget(self.browse_output_button, 0, 2)
        layout.addWidget(QLabel("Folder Name:"), 1, 0)
        self.output_folder_name = QLineEdit("data")
        self.output_folder_name.setPlaceholderText("data")
        layout.addWidget(self.output_folder_name, 1, 1, 1, 2)
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

        self.advanced_options_toggle = QToolButton()
        self.advanced_options_toggle.setText("Advanced Options")
        self.advanced_options_toggle.setCheckable(True)
        self.advanced_options_toggle.setStyleSheet("QToolButton { border: none; }")
        self.advanced_options_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_options_toggle.setArrowType(Qt.RightArrow)
        self.advanced_options_toggle.toggled.connect(self.toggle_advanced_options)
        self.layout.addWidget(self.advanced_options_toggle)

        self.advanced_options_frame = QFrame()
        self.advanced_options_frame.setFrameShape(QFrame.StyledPanel)
        advanced_layout = QVBoxLayout(self.advanced_options_frame)

        image_rule_box = QGroupBox("Image Import Rule")
        image_rule_layout = QVBoxLayout()
        self.import_annotated_images_radio = QRadioButton("Import only images with annotations")
        self.import_all_images_radio = QRadioButton("Import all images found in dataset")
        self.import_annotated_images_radio.setChecked(True)
        image_rule_layout.addWidget(self.import_annotated_images_radio)
        image_rule_layout.addWidget(self.import_all_images_radio)
        image_rule_box.setLayout(image_rule_layout)
        advanced_layout.addWidget(image_rule_box)

        class_filter_box = QGroupBox("Classes to Import")
        class_filter_layout = QVBoxLayout()
        self.class_scroll_area = QScrollArea()
        self.class_scroll_area.setWidgetResizable(True)
        self.class_widget = QFrame()
        self.class_layout = QVBoxLayout(self.class_widget)
        self.class_scroll_area.setWidget(self.class_widget)
        class_filter_layout.addWidget(self.class_scroll_area)
        class_filter_box.setLayout(class_filter_layout)
        advanced_layout.addWidget(class_filter_box)
        self.layout.addWidget(self.advanced_options_frame)

    def toggle_advanced_options(self, checked):
        self.advanced_options_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.advanced_options_frame.setVisible(checked)

    def setup_buttons_layout(self):
        """Set up the OK/Cancel button box."""
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.start_processing)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def browse_data_yaml(self):
        """Open a file dialog to select the data YAML file and populate advanced options."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Data YAML", "", "YAML Files (*.yaml);;All Files (*)", options=options
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
            class_names = data.get('names', [])
            if not class_names:
                QMessageBox.warning(self, "Warning", "Could not find a 'names' list in the selected YAML file.")
                return

            self.yaml_path_label.setText(file_path)
            yaml_dir = os.path.dirname(file_path)
            self.output_dir_label.setText(yaml_dir)
            self.output_folder_name.setText("data")

            for checkbox in self.class_checkboxes:
                self.class_layout.removeWidget(checkbox)
                checkbox.deleteLater()
            self.class_checkboxes.clear()

            for name in class_names:
                checkbox = QCheckBox(name)
                checkbox.setChecked(True)
                self.class_layout.addWidget(checkbox)
                self.class_checkboxes.append(checkbox)

            self.advanced_options_toggle.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read or parse YAML file:\n{e}")
            self.advanced_options_toggle.setEnabled(False)

    def browse_output_dir(self):
        """Open a dialog to select the output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_label.setText(dir_path)

    def start_processing(self):
        """Validate inputs, check for duplicates, and start the worker thread."""
        if not all([self.yaml_path_label.text(), self.output_dir_label.text(), self.output_folder_name.text()]):
            QMessageBox.warning(self, "Error", "Please fill in all fields.")
            return

        self.output_folder = os.path.join(self.output_dir_label.text(), self.output_folder_name.text())
        if os.path.exists(self.output_folder) and os.listdir(self.output_folder):
            reply = QMessageBox.question(self, 
                                         'Directory Not Empty', 
                                         f"The directory '{self.output_folder}' is not empty. Continue?", 
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No: return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            image_paths = glob.glob(f"{os.path.dirname(self.yaml_path_label.text())}/**/images/*.*", recursive=True)
        finally:
            QApplication.restoreOverrideCursor()

        basenames, duplicates_exist = set(), False
        for path in image_paths:
            basename_no_ext = os.path.splitext(os.path.basename(path))[0]
            if basename_no_ext in basenames:
                duplicates_exist = True
                break
            basenames.add(basename_no_ext)

        rename_files = False
        if duplicates_exist:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle('Duplicate Filenames Found')
            msg_box.setText(
                "Images with the same base name exist in different subdirectories.\n"
                "This can cause files to be overwritten in the output directory."
            )
            msg_box.setInformativeText("How would you like to handle these conflicts?")

            # Add buttons with proper line breaks
            rename_button = msg_box.addButton(
                "Rename Files (Safe)", 
                QMessageBox.AcceptRole
            )
            overwrite_button = msg_box.addButton(
                "Overwrite", 
                QMessageBox.DestructiveRole
            )
            cancel_button = msg_box.addButton(
                "Cancel", 
                QMessageBox.RejectRole
            )

            msg_box.setDefaultButton(rename_button)
            msg_box.exec_()

            clicked_button = msg_box.clickedButton()
            if clicked_button == cancel_button:
                return
            elif clicked_button == rename_button:
                rename_files = True
            elif clicked_button == overwrite_button:
                rename_files = False
            else:
                return

        excluded_classes = set()
        if self.advanced_options_toggle.isEnabled():
            for checkbox in self.class_checkboxes:
                if not checkbox.isChecked():
                    excluded_classes.add(checkbox.text())
        image_import_policy = 'all' if self.import_all_images_radio.isChecked() else 'annotated_only'

        self.button_box.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.progress_bar = ProgressBar(self, title="Preparing to Import...")
        self.progress_bar.show()

        import_as = 'polygon' if 'Polygon' in self.import_as_combo.currentText() else 'rectangle'

        self.thread = QThread()
        self.worker = DatasetProcessor(
            yaml_path=self.yaml_path_label.text(),
            output_folder=self.output_folder,
            task=self.task,
            import_as=import_as,
            rename_on_conflict=rename_files,
            excluded_classes=excluded_classes,
            image_import_policy=image_import_policy
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error.connect(self.on_error)
        self.worker.status_changed.connect(self.on_status_changed)
        self.worker.progress_updated.connect(self.on_progress_update)
        self.worker.processing_complete.connect(self.on_processing_complete)
        self.thread.start()

    def on_status_changed(self, title, total):
        self.progress_bar.set_title(title)
        self.progress_bar.start_progress(total)

    def on_progress_update(self, value):
        self.progress_bar.set_value(value)

    def on_processing_complete(self, raw_annotations, image_paths, parsing_errors):
        progress_bar = ProgressBar(self, title="Adding Data to Project...")
        progress_bar.show()

        added_paths = []
        progress_bar.set_title(f"Adding {len(image_paths)} images...")
        progress_bar.start_progress(len(image_paths))
        for path in image_paths:
            if self.image_window.add_image(path):
                added_paths.append(path)
            progress_bar.update_progress()

        newly_created_annotations = []
        progress_bar.set_title(f"Adding {len(raw_annotations)} annotations...")
        progress_bar.start_progress(len(raw_annotations))
        for raw_ann in raw_annotations:
            label = self.main_window.label_window.add_label_if_not_exists(raw_ann["class_name"])
            if raw_ann["type"] == "RectangleAnnotation":
                tl, br = raw_ann["top_left"], raw_ann["bottom_right"]
                annotation = RectangleAnnotation(QPointF(tl[0], tl[1]), 
                                                 QPointF(br[0], br[1]), 
                                                 label.short_label_code, 
                                                 label.long_label_code, 
                                                 label.color, 
                                                 raw_ann["image_path"], 
                                                 label.id, 
                                                 self.main_window.get_transparency_value())
            else:
                points = [QPointF(p[0], p[1]) for p in raw_ann["points"]]
                annotation = PolygonAnnotation(points, 
                                               label.short_label_code, 
                                               label.long_label_code,
                                               label.color, 
                                               raw_ann["image_path"], 
                                               label.id, 
                                               self.main_window.get_transparency_value())

            self.annotation_window.add_annotation_to_dict(annotation)
            newly_created_annotations.append(annotation)  # Collect created objects
            
            progress_bar.update_progress()

        # --- Call the restored, correct export function ---
        progress_bar.set_title("Exporting annotations.json...")
        self._export_annotations_to_json(newly_created_annotations, self.output_folder)
        
        progress_bar.finish_progress()
        progress_bar.stop_progress()
        progress_bar.close()

        self.image_window.filter_images()
        if added_paths:
            self.image_window.load_image_by_path(added_paths[-1])
            self.image_window.update_image_annotations(added_paths[-1])
            self.annotation_window.load_annotations()

        summary_message = "Dataset has been successfully imported."
        if parsing_errors:
            QMessageBox.warning(self, 
                                "Import Complete with Warnings", 
                                f"{summary_message}\n\nHowever, {len(parsing_errors)} issue(s) were found. "
                                "Please review them below.", 
                                details='\n'.join(parsing_errors))
        else:
            QMessageBox.information(self, "Dataset Imported", summary_message)
            
    def _export_annotations_to_json(self, annotations_list, output_dir):
        """
        Merges the list of annotation objects into an existing annotations.json file,
        or creates a new one if it doesn't exist.
        The output is a dictionary mapping image paths to lists of annotation dicts.
        """
        export_dict = {}
        json_path = os.path.join(output_dir, "annotations.json")

        # Step 1: Check for the existing file and load it if present.
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as file:
                    export_dict = json.load(file)
                # Ensure the loaded data is a dictionary
                if not isinstance(export_dict, dict):
                    raise TypeError("annotations.json is not in the expected format (dict).")
            except (json.JSONDecodeError, TypeError, IOError) as e:
                # If file is corrupt, unreadable, or has wrong format, warn the user and start fresh.
                QMessageBox.warning(self, 
                                    "Read Error",
                                    f"Could not read or parse existing annotations.json:\n{e}\n\n"
                                    "A new file will be created, overwriting the old one.")
                export_dict = {}  # Reset to be safe

        # Step 2: Iterate through new annotations and merge them into the dictionary.
        for annotation in annotations_list:
            image_path = annotation.image_path
            
            # Use setdefault to initialize a list for a new image path or get the existing one.
            export_dict.setdefault(image_path, [])
            
            # Create the dictionary for the annotation using its own method
            if isinstance(annotation, RectangleAnnotation):
                annotation_dict = {
                    'type': 'RectangleAnnotation',
                    **annotation.to_dict()
                }
            elif isinstance(annotation, PolygonAnnotation):
                annotation_dict = {
                    'type': 'PolygonAnnotation',
                    **annotation.to_dict()
                }
            else:
                warnings.warn(f"Unknown annotation type skipped during export: {type(annotation)}")
                continue

            export_dict[image_path].append(annotation_dict)

        # Step 3: Write the final, merged dictionary back to the JSON file.
        try:
            with open(json_path, 'w') as file:
                json.dump(export_dict, file, indent=4)
                file.flush()
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to write annotations.json:\n{e}")

    def on_error(self, message):
        QMessageBox.warning(self, "Error", message)

    def on_worker_finished(self):
        if self.progress_bar:
            self.progress_bar.stop_progress()
            self.progress_bar.close()
        self.thread.quit()
        self.thread.wait()
        self.worker.deleteLater()
        self.thread.deleteLater()
        self.worker = None
        self.thread = None
        QApplication.restoreOverrideCursor()
        self.button_box.setEnabled(True)
        self.accept()

    def reject(self):
        if self.thread and self.thread.isRunning():
            self.worker.stop()
            self.button_box.setEnabled(False)
        else:
            super().reject()

    def closeEvent(self, event):
        self.reject()
        super().closeEvent(event)