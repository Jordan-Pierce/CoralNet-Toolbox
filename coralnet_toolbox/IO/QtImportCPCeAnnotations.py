import os
import csv
import ntpath

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QDialog, QInputDialog,
                             QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QLineEdit,
                             QGroupBox, QFormLayout)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_window_icon

from coralnet_toolbox.utilities import STATUS_SUMMARY_MS, plural, show_status


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------
# Note: CPCe (Coral Point Count with Excel extensions) is a Windows program. Its files are ANSI text, and the
# paths stored inside them are absolute Windows paths from the machine that wrote them, so only file names are used.


class ImportCPCeAnnotations(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_window_icon("coralnet.svg"))
        self.setWindowTitle("Import CPCe Annotations")
        self.resize(500, 200)

        # Create the main layout
        self.layout = QVBoxLayout(self)

        # Setup UI components
        self.setup_info_layout()
        self.setup_input_layout()
        self.setup_buttons_layout()

    def setup_info_layout(self):
        """Set up the information section layout."""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        info_label = QLabel(
            "Import point annotations from CPCe (.cpc) files. Each file is matched to a loaded image "
            "by the image name stored inside it. The code file (.txt) is optional. When provided, it "
            "supplies the full name for each label code."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_input_layout(self):
        """Set up the input file selection layout."""
        group_box = QGroupBox("Input")
        layout = QFormLayout()

        # CPC file selection
        self.cpc_files_edit = QLineEdit()
        self.cpc_files_edit.setToolTip("Paths to the CPCe .cpc files to import, separated by semicolons.")
        self.cpc_files_button = QPushButton("Browse...")
        self.cpc_files_button.clicked.connect(self.browse_cpc_files)
        self.cpc_files_button.setToolTip("Browse for one or more CPCe .cpc files.")
        cpc_layout = QHBoxLayout()
        cpc_layout.addWidget(self.cpc_files_edit)
        cpc_layout.addWidget(self.cpc_files_button)
        layout.addRow("CPC Files:", cpc_layout)

        # Code file selection (optional)
        self.code_file_edit = QLineEdit()
        self.code_file_edit.setPlaceholderText("Optional")
        self.code_file_edit.setToolTip("Path to the CPCe code file (.txt) used to make the .cpc files.\n"
                                       "Without it, labels use the short code as their name.")
        self.code_file_button = QPushButton("Browse...")
        self.code_file_button.clicked.connect(self.browse_code_file)
        self.code_file_button.setToolTip("Browse for a CPCe code file.")
        code_layout = QHBoxLayout()
        code_layout.addWidget(self.code_file_edit)
        code_layout.addWidget(self.code_file_button)
        layout.addRow("Code File:", code_layout)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_buttons_layout(self):
        """Set up the bottom buttons layout."""
        buttons_layout = QHBoxLayout()

        self.import_button = QPushButton("Import")
        self.import_button.setToolTip("Import annotations from the selected .cpc files.")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setToolTip("Close this dialog without importing.")

        # Lambda so the clicked(bool) argument is not passed in as file_paths
        self.import_button.clicked.connect(lambda: self.import_annotations())
        self.cancel_button.clicked.connect(self.reject)

        buttons_layout.addWidget(self.import_button)
        buttons_layout.addWidget(self.cancel_button)

        self.layout.addLayout(buttons_layout)

    def browse_cpc_files(self):
        """Open file dialog to select .cpc files."""
        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import CPCe Annotations",
            "",
            "CPCe Files (*.cpc);;All Files (*)",
            options=options
        )
        if file_paths:
            self.cpc_files_edit.setText(";".join(file_paths))

            # Fill in the code file if the one named in the .cpc header sits nearby
            if not self.code_file_edit.text().strip():
                code_file = self.find_code_file(file_paths[0])
                if code_file:
                    self.code_file_edit.setText(code_file)

    def browse_code_file(self):
        """Open file dialog to select the code file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CPCe Code File",
            "",
            "Text Files (*.txt);;All Files (*)",
            options=options
        )
        if file_path:
            self.code_file_edit.setText(file_path)

    def find_code_file(self, cpc_path):
        """Look for the code file named in a .cpc header, next to the .cpc file or one folder up."""
        try:
            with open(cpc_path, 'r', encoding='latin-1') as f:
                header = next(csv.reader([f.readline()]))
            code_name = ntpath.basename(header[0].strip())
        except Exception:
            return None

        if not code_name:
            return None

        cpc_folder = os.path.dirname(cpc_path)
        for folder in (cpc_folder, os.path.dirname(cpc_folder)):
            candidate = os.path.join(folder, code_name)
            if os.path.isfile(candidate):
                return candidate.replace("\\", "/")
        return None

    def import_annotations(self, file_paths=None, code_file=None, annotation_size=None):
        """Import annotations from CPCe .cpc files into the current project."""
        if not self.annotation_window.active_image:
            QMessageBox.warning(self,
                                "No Images Loaded",
                                "Please load images first before importing annotations.")
            return

        if file_paths is None:
            file_paths = [p.strip() for p in self.cpc_files_edit.text().split(";") if p.strip()]

        if code_file is None:
            code_file = self.code_file_edit.text().strip()

        if not file_paths or not all(os.path.isfile(p) for p in file_paths):
            QMessageBox.warning(self,
                                "Invalid Files",
                                "Please select one or more valid .cpc files.")
            return

        if code_file and not os.path.isfile(code_file):
            QMessageBox.warning(self,
                                "Invalid File",
                                "The code file does not exist. Clear the field to import without it.")
            return

        if annotation_size is None:
            annotation_size, ok = QInputDialog.getInt(self,
                                                      "Patch Annotation Size",
                                                      "Enter the default patch annotation size for imported annotations:",
                                                      224, 1, 10000, 1)
            if not ok:
                return

        image_path_map = {os.path.basename(path): path for path in self.image_window.raster_manager.image_paths}
        image_stem_map = {os.path.splitext(name)[0]: path for name, path in image_path_map.items()}

        show_status(self.main_window, f"Reading {plural(len(file_paths), 'CPC file')}...")

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self, title="Reading CPC Files")
        progress_bar.start_progress(len(file_paths))
        progress_bar.show()

        error = None
        records = []         # (image_path, column, row, label code)
        unmatched_files = []
        skipped_points = 0

        try:
            long_names = self.read_code_file(code_file) if code_file else {}

            for file_path in file_paths:
                cpc = self.read_cpc_file(file_path)

                # Match on the image name in the header, then fall back to the .cpc file's own name
                image_path = image_path_map.get(cpc['image_name'])
                if not image_path:
                    image_path = image_stem_map.get(os.path.splitext(os.path.basename(file_path))[0])

                raster = self.image_window.raster_manager.get_raster(image_path) if image_path else None
                if raster is None or not raster.width or not raster.height:
                    unmatched_files.append(os.path.basename(file_path))
                    progress_bar.update_progress()
                    continue

                # CPCe stores positions in sub-pixel units (usually 1/15 of a pixel, 1/12 on some older
                # installs), so derive the scale from the header size rather than assuming 15.
                scale_x = cpc['width'] / raster.width
                scale_y = cpc['height'] / raster.height

                # Unequal scales mean the .cpc file was made on a different image
                if scale_x <= 0 or abs(scale_x - scale_y) / scale_x > 0.01:
                    unmatched_files.append(os.path.basename(file_path))
                    progress_bar.update_progress()
                    continue

                for x, y, code in cpc['points']:
                    column = int(round(x / scale_x))
                    row = int(round(y / scale_y))
                    if not code or not (0 <= column < raster.width and 0 <= row < raster.height):
                        skipped_points += 1
                        continue
                    records.append((image_path, column, row, code))

                progress_bar.update_progress()

        except Exception as e:
            error = e

        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()

        if error is not None:
            show_status(self.main_window, f"CPCe import failed: {error}", STATUS_SUMMARY_MS)
            QMessageBox.warning(self,
                                "Error Importing Annotations",
                                f"An error occurred while reading the CPCe files: {str(error)}")
            return

        if not records:
            show_status(self.main_window, "CPCe import: no annotations found for loaded images.", STATUS_SUMMARY_MS)
            QMessageBox.warning(self,
                                "Error Importing Annotations",
                                "No annotations found for loaded images.")
            return

        show_status(self.main_window, f"Importing {plural(len(records), 'CPCe annotation')}...")

        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self, title="Importing CPCe Annotations")
        progress_bar.start_progress(len(records))  # Progress per annotation
        progress_bar.show()

        error = None
        annotations_to_add = []
        images_to_update = set()

        try:
            # Pre-process labels to avoid repeated lookups
            label_cache = {}
            for code in sorted({record[3] for record in records}):
                # Look up by short code alone. add_label_if_not_exists matches short and long together,
                # so a label made by an import without the code file (long == short) would otherwise
                # be duplicated once the code file supplies a long name.
                label = self.label_window.get_label_by_short_code(code, return_review=False)
                if label is None:
                    # Defer UI refresh; batch-refresh once after the loop to avoid flashing.
                    label = self.label_window.add_label_if_not_exists(code,
                                                                      long_names.get(code.lower(), code),
                                                                      refresh_ui=False)
                label_cache[code] = label
            # Single UI refresh for the whole batch (avoids per-label flashing).
            self.label_window.refresh_after_batch_add()

            # 1. Create all annotation objects in memory first
            for image_path, column, row, code in records:
                annotation = PatchAnnotation(
                    QPointF(column, row),
                    annotation_size,
                    label_cache[code],
                    image_path,
                )
                annotations_to_add.append(annotation)
                images_to_update.add(image_path)
                progress_bar.update_progress()

            # 2. Add all created annotations in a single, efficient batch operation
            if annotations_to_add:
                self.annotation_window.add_annotations(annotations_to_add)

            # 3. Update UI counts for each affected image only ONCE
            for image_path in images_to_update:
                self.image_window.update_image_annotations(image_path)

            # Load the annotations for the currently visible image
            self.annotation_window.load_annotations()

        except Exception as e:
            error = e

        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()

        if error is not None:
            show_status(self.main_window, f"CPCe import failed: {error}", STATUS_SUMMARY_MS)
            QMessageBox.warning(self,
                                "Error Importing Annotations",
                                f"An error occurred while importing annotations: {str(error)}")
            return

        message = (f"Imported {plural(len(annotations_to_add), 'annotation')} "
                   f"into {plural(len(images_to_update), 'image')}.")
        status = f"CPCe: {message}"
        if unmatched_files:
            status += f" {plural(len(unmatched_files), 'file')} had no matching image."
        show_status(self.main_window, status, STATUS_SUMMARY_MS)
        if unmatched_files:
            shown = ", ".join(unmatched_files[:10])
            more = f" and {len(unmatched_files) - 10} more" if len(unmatched_files) > 10 else ""
            message += f"\n\n{len(unmatched_files)} files had no matching image loaded: {shown}{more}."
        if skipped_points:
            message += f"\n\n{skipped_points} points were skipped (no label code, or outside the image)."
        if long_names:
            missing_codes = sorted({record[3] for record in records if record[3].lower() not in long_names})
            if missing_codes:
                message += f"\n\nCodes not found in the code file: {', '.join(missing_codes)}."

        QMessageBox.information(self, "Annotations Imported", message)
        self.accept()

    @staticmethod
    def read_cpc_file(file_path):
        """Parse a CPCe .cpc file.

        Layout (one image per file):
            line 1      "code file path","image path",width,height,display width,display height
            lines 2-5   the four corners of the frame border
            line 6      number of points N
            next N      x,y of each point, in CPCe units
            next N      "point id","label code","notes code","notes text"
            remainder   image header fields (Institution, Project, Station, ...), not used here

        Returns:
            dict: image_name, width and height in CPCe units, and points as (x, y, label code)
        """
        name = os.path.basename(file_path)

        with open(file_path, 'r', encoding='latin-1') as f:
            lines = f.read().splitlines()

        if len(lines) < 6:
            raise ValueError(f"{name} is too short to be a .cpc file.")

        header = next(csv.reader([lines[0]]))
        if len(header) < 4:
            raise ValueError(f"{name} has an unexpected header line.")

        # ntpath splits on both separators, so a Windows path is handled on any platform
        image_name = ntpath.basename(header[1].strip())
        width = float(header[2])
        height = float(header[3])

        # Lines 2-5 are the frame border CPCe placed its random points inside. A frame border is
        # not the same thing as a work area, so it is ignored for now.

        num_points = int(lines[5].strip())
        coord_lines = lines[6:6 + num_points]
        label_lines = lines[6 + num_points:6 + 2 * num_points]

        if len(label_lines) < num_points:
            raise ValueError(f"{name} lists {num_points} points but is missing some of them.")

        points = []
        for coord_line, label_line in zip(coord_lines, label_lines):
            coords = next(csv.reader([coord_line]))
            fields = next(csv.reader([label_line]))
            code = fields[1].strip() if len(fields) > 1 else ""
            points.append((float(coords[0]), float(coords[1]), code))

        return {'image_name': image_name, 'width': width, 'height': height, 'points': points}

    @staticmethod
    def read_code_file(file_path):
        """Parse a CPCe code file into a map of short code (lowercase) to long name."""
        return {code.lower(): name for code, name in read_cpce_code_file(file_path)}


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def read_cpce_code_file(file_path):
    """Parse a CPCe code file into (short code, long name) pairs, in file order.

    Layout:
        leading lines   colors and the number of major categories
        category rows   "category code","category name",color
        label rows      "code","name","category code"
        NOTES,NOTES,NOTES starts the notes codes section, which is not used here

    Label rows are the ones whose third field is a category code. Category colors are not used:
    they belong to the category, not the label.
    """
    with open(file_path, 'r', encoding='latin-1') as f:
        rows = list(csv.reader(f, skipinitialspace=True))

    code_rows = []
    for row in rows:
        row = [field.strip() for field in row]
        if row and row[0].upper() == "NOTES":
            break
        if len(row) >= 3 and row[0]:
            code_rows.append(row)

    category_codes = {row[0] for row in code_rows}
    labels = [(row[0], row[1] or row[0]) for row in code_rows if row[2] in category_codes]

    if not labels:
        raise ValueError(f"No label codes found in {os.path.basename(file_path)}.")

    return labels
