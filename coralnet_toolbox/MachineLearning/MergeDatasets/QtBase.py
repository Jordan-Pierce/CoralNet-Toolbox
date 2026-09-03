import warnings

import os
import shutil
import ujson as json
from concurrent.futures import ThreadPoolExecutor, as_completed

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QWidget, QVBoxLayout,
                             QLabel, QLineEdit, QDialog, QHBoxLayout, QPushButton,
                             QDialogButtonBox, QFormLayout, QGroupBox, QScrollArea,
                             QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
                             QSizePolicy)

from coralnet_toolbox.MachineLearning.MergeDatasets.merge_dataset_utils import (
    MAX_SEMANTIC_CLASSES,
    OUTPUT_SPLITS,
    build_mask_lut,
    build_union_vocabulary,
    copy_image,
    discover_dataset_splits,
    merge_class_mappings,
    read_class_mapping,
    read_dataset_names,
    remap_label_file,
    remap_mask_file,
    sidecar_spec,
    unique_stem,
    write_data_yaml,
)

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_window_icon

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def classify_class_names(dir_path):
    """Return a classification dataset's classes as {index: name}.

    A classification dataset carries no data.yaml: each split holds one folder
    per class, and the folder name is the class. Every split is scanned because
    a class present only in validation still has to appear in the merged
    vocabulary.
    """
    ordered = []
    seen = set()

    for split in ('train', 'val', 'valid', 'test'):
        split_dir = os.path.join(dir_path, split)
        if not os.path.isdir(split_dir):
            continue
        for name in sorted(os.listdir(split_dir)):
            key = name.strip().lower()
            if key not in seen and os.path.isdir(os.path.join(split_dir, name)):
                seen.add(key)
                ordered.append(name)

    return {index: name for index, name in enumerate(ordered)}


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DatasetEntry(QGroupBox):
    """One source dataset in the merge dialog.

    Each entry owns its own widgets and the classes read off disk, so the dialog
    never has to keep a parallel list in sync with the widget tree.
    """
    removed = pyqtSignal(object)
    # Emitted only on user edits, never from validate() itself, which the dialog
    # calls on every entry while rebuilding the preview this signal triggers.
    changed = pyqtSignal()

    def __init__(self, index, task, parent=None):
        super().__init__(f"Dataset {index}", parent)

        self.task = task
        self.names = {}
        self.class_mapping = {}

        self.status_label = QLabel()
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)

        form_layout = QFormLayout()
        self.setLayout(form_layout)

        # A classification dataset is identified by its folder, since it has no
        # data.yaml; every other task is identified by the data.yaml itself,
        # which is the only thing that says what its class indices mean.
        self.path_edit = QLineEdit()
        path_button = QPushButton("Browse Directory..." if self.is_classify else "Browse data.yaml...")
        path_button.clicked.connect(self.browse_source)
        self.path_edit.editingFinished.connect(self.on_user_edit)

        if self.is_classify:
            self.path_edit.setToolTip("Path to a classification dataset to include in the merge.")
            path_button.setToolTip("Browse for a dataset directory.")
            path_row_label = "Directory:"
        else:
            self.path_edit.setToolTip("Path to the data.yaml of a dataset to include in the merge.\n"
                                      "Its class names are read from here and merged with the others.")
            path_button.setToolTip("Browse for a dataset data.yaml file.")
            path_row_label = "Data YAML:"

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(path_button)
        form_layout.addRow(path_row_label, path_layout)

        # Class mapping input row
        self.mapping_edit = QLineEdit()
        mapping_button = QPushButton("Select Class Mapping")
        mapping_button.clicked.connect(self.browse_class_mapping)
        self.mapping_edit.editingFinished.connect(self.on_user_edit)
        self.mapping_edit.setToolTip("Optional class_mapping.json carrying this dataset's label colors and\n"
                                     "long codes. Classes are matched between datasets by name, so this file\n"
                                     "is not needed to resolve class ID conflicts.")
        mapping_button.setToolTip("Browse for a class mapping JSON file.")
        mapping_layout = QHBoxLayout()
        mapping_layout.addWidget(self.mapping_edit)
        mapping_layout.addWidget(mapping_button)
        form_layout.addRow("Class Mapping:", mapping_layout)

        # Detected classes row
        form_layout.addRow("Classes:", self.summary_label)

        # Remove button row
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(lambda: self.removed.emit(self))
        remove_button.setToolTip("Remove this dataset from the merge.")
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.status_label)
        button_layout.addWidget(remove_button)
        button_layout.setAlignment(Qt.AlignRight)
        form_layout.addRow("", button_layout)

        self.validate()

    @property
    def is_classify(self):
        """True when this entry describes a classification dataset."""
        return self.task == 'classify'

    def source_path(self):
        """The dataset directory (classify) or data.yaml path (every other task)."""
        return self.path_edit.text().strip()

    def class_mapping_path(self):
        """The optional class_mapping.json path."""
        return self.mapping_edit.text().strip()

    def dataset_dir(self):
        """The folder holding the dataset, whichever way it was identified."""
        path = self.source_path()
        if not path:
            return ""
        return path if self.is_classify else os.path.dirname(os.path.abspath(path))

    def is_valid(self):
        """True when a source has been chosen and classes were read from it."""
        return bool(self.source_path()) and bool(self.names)

    def on_user_edit(self):
        """Re-read this entry after the user changed it, and say so."""
        self.validate()
        self.changed.emit()

    def browse_source(self):
        """Choose this entry's dataset, then read its classes."""
        if self.is_classify:
            path = QFileDialog.getExistingDirectory(self, "Select Existing Dataset Directory")
        else:
            path, _ = QFileDialog.getOpenFileName(self,
                                                  "Select data.yaml",
                                                  "",
                                                  "YAML Files (*.yaml *.yml);;All Files (*)")
        if not path:
            return

        self.path_edit.setText(path)
        # validate() picks up the dataset's own class_mapping.json from here.
        self.on_user_edit()

    def browse_class_mapping(self):
        """Choose a class_mapping.json for this entry."""
        path, _ = QFileDialog.getOpenFileName(self,
                                              "Select class_mapping.json",
                                              "",
                                              "JSON Files (*.json);;All Files (*)")
        if path:
            self.mapping_edit.setText(path)
            self.on_user_edit()

    def validate(self):
        """Re-read this entry's classes from disk and refresh its status."""
        self.names = {}
        self.class_mapping = {}

        path = self.source_path()
        if not path:
            self.status_label.setText("")
            self.summary_label.setText("No dataset selected.")
            return False

        if self.is_classify:
            if os.path.isdir(path):
                self.names = classify_class_names(path)
        elif os.path.isfile(path):
            self.names = read_dataset_names(path)

        # Fall back to the class_mapping.json the dataset ships with. This lives
        # here rather than in browse_source so a path that was typed, pasted or
        # restored is picked up the same way a browsed one is -- otherwise an
        # entry silently contributes no label colors to the merged mapping.
        mapping_path = self.class_mapping_path()
        if not mapping_path:
            default_mapping = os.path.join(self.dataset_dir(), "class_mapping.json")
            if os.path.isfile(default_mapping):
                mapping_path = default_mapping
                self.mapping_edit.setText(default_mapping)

        self.class_mapping = read_class_mapping(mapping_path)

        if self.names:
            self.status_label.setText("✅")
            listed = ", ".join(self.names[index] for index in sorted(self.names))
            self.summary_label.setText(f"{len(self.names)} found: {listed}")
        else:
            self.status_label.setText("❌")
            if self.is_classify:
                reason = "No class folders found under train/val/test."
            elif os.path.isfile(path):
                reason = "No 'names' entry found in this data.yaml."
            else:
                reason = "File not found."
            self.summary_label.setText(reason)

        return bool(self.names)


class Base(QDialog):
    """
    Dialog for merging multiple datasets into a single dataset for machine learning
    tasks such as image classification, object detection, instance segmentation and
    semantic segmentation.
    """
    # Set by the task-specific subclasses.
    task = None

    def __init__(self, parent=None):
        """
        Initializes the MergeDatasetsDialog.

        :param parent: Parent widget, default is None.
        """
        super().__init__(parent)

        self.setWindowIcon(get_window_icon("coralnet.svg"))
        self.setWindowTitle("Merge Datasets")
        self.resize(700, 860)

        self.entries = []
        self.dataset_count = 0

        self.layout = QVBoxLayout(self)

        # Setup the info layout
        self.setup_info_layout()
        # Setup the outputs_layout
        self.setup_outputs_layout()
        # Setup the datasets layout
        self.setup_datasets_layout()
        # Setup the class preview layout
        self.setup_preview_layout()
        # Setup the buttons layout
        self.setup_buttons_layout()

        # The datasets list is the only section that should grow. Without pinning
        # the others to their preferred height they each absorb a share of the
        # extra space, leaving the list barely taller than a single entry.
        for group_box in (self.info_group, self.outputs_group, self.preview_group):
            group_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.datasets_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.layout.setStretchFactor(self.datasets_group, 1)

        self.fit_to_screen()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def fit_to_screen(self):
        """Shrink the dialog if its preferred size does not fit the display.

        The default size is chosen so the first dataset entry is fully visible
        rather than half-hidden behind the scroll area's edge, which is taller
        than a small laptop screen can show.
        """
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        self.resize(min(self.width(), int(available.width() * 0.9)),
                    min(self.height(), int(available.height() * 0.9)))

    def task_description(self):
        """Return the human-readable name of the task being merged."""
        return {
            'classify': "Classification",
            'detect': "Detection",
            'segment': "Instance Segmentation",
            'semantic': "Semantic Segmentation",
        }.get(self.task, "")

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        if self.task == 'classify':
            info_text = ("Select multiple Classification datasets to merge into a single combined dataset.\n"
                         "Classes are matched by folder name, so datasets sharing a class are combined into it.")
        else:
            info_text = (f"Select multiple {self.task_description()} datasets to merge into a single combined "
                         "dataset.\nClasses are matched by name across datasets, and each dataset's class IDs "
                         "are rewritten to the merged numbering.")

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel(info_text)

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        info_label.setToolTip("Merge multiple datasets together, resolving class ID conflicts automatically.\n"
                              "Useful for combining data from different sources or annotation sessions.")
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.info_group = group_box
        self.layout.addWidget(group_box)

    def setup_outputs_layout(self):
        """Setup the outputs layout."""
        group_box = QGroupBox("Output Dataset")
        layout = QFormLayout()

        # Dataset Name Input
        self.dataset_name_edit = QLineEdit()
        self.dataset_name_edit.setToolTip("Name for the merged output dataset.\nWill be created in the output directory.")
        layout.addRow("Dataset Name:", self.dataset_name_edit)

        # Output Directory Chooser
        self.output_dir_edit = QLineEdit()
        output_dir_button = QPushButton("Browse...")
        output_dir_button.clicked.connect(lambda: self.browse_output_directory(self.output_dir_edit))
        self.output_dir_edit.setToolTip("Directory where the merged dataset will be saved.\nDataset Name subdirectory will be created here.")
        output_dir_button.setToolTip("Browse for an output directory.")
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.output_dir_edit)
        dir_layout.addWidget(output_dir_button)
        layout.addRow("Output Directory:", dir_layout)

        group_box.setLayout(layout)
        self.outputs_group = group_box
        self.layout.addWidget(group_box)

    def setup_datasets_layout(self):
        """Setup the datasets layout."""
        group_box = QGroupBox("Datasets")
        layout = QVBoxLayout()

        # Create a scroll area, tall enough that a freshly added entry is shown
        # whole rather than clipped at the bottom edge.
        self.datasets_scroll = QScrollArea()
        self.datasets_scroll.setWidgetResizable(True)
        self.datasets_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Floor is one entry plus a little slack, so a small screen can still
        # shrink the dialog; the stretch factor is what makes it open tall.
        self.datasets_scroll.setMinimumHeight(190)

        # Create a widget to hold the dataset entries
        self.datasets_widget = QWidget()
        self.datasets_layout = QVBoxLayout(self.datasets_widget)
        # Entries keep their natural height and stack from the top instead of
        # stretching to share whatever space the scroll area has.
        self.datasets_layout.addStretch(1)

        # Add the widget to the scroll area
        self.datasets_scroll.setWidget(self.datasets_widget)

        # Add button to add new dataset
        add_dataset_button = QPushButton("Add Dataset")
        add_dataset_button.clicked.connect(self.add_dataset_entry)
        add_dataset_button.setToolTip("Add another dataset to merge.\nYou can add multiple datasets; their classes are combined by name.")

        layout.addWidget(self.datasets_scroll)
        layout.addWidget(add_dataset_button)

        group_box.setLayout(layout)
        self.datasets_group = group_box
        self.layout.addWidget(group_box)

    def setup_preview_layout(self):
        """Setup the merged class preview layout."""
        group_box = QGroupBox("Merged Classes")
        layout = QVBoxLayout()

        self.preview_table = QTableWidget()
        self.preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.preview_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.preview_table.setMaximumHeight(180)
        self.preview_table.setToolTip("The classes the merged dataset will contain, and the ID each source\n"
                                      "dataset used for them. A dash means that dataset has no such class.")

        refresh_button = QPushButton("Refresh Preview")
        refresh_button.clicked.connect(self.refresh_class_preview)
        refresh_button.setToolTip("Re-read every selected dataset and rebuild the merged class list.")

        layout.addWidget(self.preview_table)
        layout.addWidget(refresh_button)

        group_box.setLayout(layout)
        self.preview_group = group_box
        self.layout.addWidget(group_box)

    def setup_buttons_layout(self):
        """Setup the buttons layout."""
        # OK and Cancel Buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

    # ------------------------------------------------------------------
    # Entries
    # ------------------------------------------------------------------

    def browse_output_directory(self, output_dir_edit):
        """
        Opens a dialog to select the output directory and sets the selected path."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            output_dir_edit.setText(dir_path)

    def add_dataset_entry(self):
        """Add a new dataset entry row."""
        self.dataset_count += 1

        entry = DatasetEntry(self.dataset_count, self.task, self)
        entry.removed.connect(self.remove_dataset_entry)
        entry.changed.connect(self.refresh_class_preview)

        self.entries.append(entry)
        # Insert ahead of the trailing stretch, which would otherwise push every
        # entry added after the first one below it.
        self.datasets_layout.insertWidget(self.datasets_layout.count() - 1, entry)
        self.refresh_class_preview()

        # Bring the new entry into view, since it is added at the bottom of a
        # list that may already be scrolled.
        QApplication.processEvents()
        self.datasets_scroll.ensureWidgetVisible(entry)

        return entry

    def remove_dataset_entry(self, entry):
        """Remove a dataset entry."""
        if entry in self.entries:
            self.entries.remove(entry)
        entry.setParent(None)
        entry.deleteLater()
        self.refresh_class_preview()

    def valid_entries(self):
        """Return the entries that name a dataset whose classes could be read."""
        return [entry for entry in self.entries if entry.is_valid()]

    # ------------------------------------------------------------------
    # Class preview
    # ------------------------------------------------------------------

    def refresh_class_preview(self):
        """Rebuild the table showing the merged vocabulary and its sources."""
        entries = []
        for entry in self.entries:
            entry.validate()
            if entry.is_valid():
                entries.append(entry)

        names, luts = build_union_vocabulary([entry.names for entry in entries],
                                             background_first=(self.task == 'semantic'))

        # The merged ID column is meaningless for classification, where a class
        # is a folder name and never a number.
        show_ids = self.task != 'classify'
        headers = (["Class"] + (["Merged ID"] if show_ids else []) +
                   [f"Dataset {index + 1}" for index in range(len(entries))])

        self.preview_table.clear()
        self.preview_table.setColumnCount(len(headers))
        self.preview_table.setHorizontalHeaderLabels(headers)
        self.preview_table.setRowCount(len(names))

        # Invert each dataset's lookup so a merged class can name the ID it came from.
        reverse_luts = []
        for lut in luts:
            reverse = {}
            for old_index, new_index in lut.items():
                reverse.setdefault(new_index, old_index)
            reverse_luts.append(reverse)

        for row, name in enumerate(names):
            column = 0
            self.preview_table.setItem(row, column, QTableWidgetItem(name))
            column += 1
            if show_ids:
                self.preview_table.setItem(row, column, QTableWidgetItem(str(row)))
                column += 1
            for reverse in reverse_luts:
                old_index = reverse.get(row)
                text = str(old_index) if old_index is not None else "—"
                if not show_ids and old_index is not None:
                    text = "✓"
                self.preview_table.setItem(row, column, QTableWidgetItem(text))
                column += 1

        header = self.preview_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for index in range(1, len(headers)):
            header.setSectionResizeMode(index, QHeaderView.ResizeToContents)

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def validate_output(self):
        """Check the output fields and the chosen datasets, warning if unusable.

        Returns:
            tuple: (output path, valid entries), or (None, None) if unusable.
        """
        output_dir = self.output_dir_edit.text().strip()
        dataset_name = self.dataset_name_edit.text().strip()

        if not output_dir:
            QMessageBox.warning(self, "Input Error", "Output directory must be specified.")
            return None, None

        if not dataset_name:
            QMessageBox.warning(self, "Input Error", "Dataset name must be specified.")
            return None, None

        entries = self.valid_entries()
        if len(entries) < 2:
            QMessageBox.warning(self,
                                "Input Error",
                                "At least two valid datasets must be selected to merge.")
            return None, None

        output_dir_path = os.path.join(output_dir, dataset_name)

        # Merging into a folder that is itself one of the sources would have the
        # merge read files it is in the middle of writing.
        output_resolved = os.path.normcase(os.path.abspath(output_dir_path))
        for entry in entries:
            source_resolved = os.path.normcase(os.path.abspath(entry.dataset_dir()))
            if output_resolved == source_resolved or output_resolved.startswith(source_resolved + os.sep):
                QMessageBox.warning(self,
                                    "Input Error",
                                    "The output dataset cannot be inside one of the source datasets.")
                return None, None

        return output_dir_path, entries

    def merge_datasets(self):
        """
        Merges the selected datasets into a single output directory.

        Returns:
            bool: True if the merge ran to completion.
        """
        output_dir_path, entries = self.validate_output()
        if not output_dir_path:
            return False

        os.makedirs(output_dir_path, exist_ok=True)

        # Make cursor busy to indicate processing
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if self.task == 'classify':
                return self.merge_classify_datasets(output_dir_path, entries)
            return self.merge_yolo_datasets(output_dir_path, entries)
        finally:
            # Restore the cursor to default
            QApplication.restoreOverrideCursor()

    def merge_classify_datasets(self, output_dir_path, entries):
        """Merge classification datasets by folding their class folders together.

        A classification dataset stores class identity as the name of the folder
        an image sits in, so merging is a directory union: the same class name in
        two datasets is already the same folder, and no file contents change.
        """
        merged_class_mapping = {}
        copy_jobs = []

        for entry in entries:
            merged_class_mapping.update(entry.class_mapping)

            # Classification exports use 'val'; accept 'valid' too, since a
            # dataset may have come from tooling that writes it the other way.
            for split in ('train', 'val', 'valid', 'test'):
                src_split_dir = os.path.join(entry.source_path(), split)
                if not os.path.isdir(src_split_dir):
                    continue
                dest_split = 'val' if split == 'valid' else split
                dest_split_dir = os.path.join(output_dir_path, dest_split)
                copy_jobs.append((src_split_dir, dest_split_dir))

        if not copy_jobs:
            QMessageBox.warning(self, "Warning", "No train/val/test splits were found in the selected datasets.")
            return False

        errors = []
        progress_bar = ProgressBar(self, title="Merging Datasets")
        progress_bar.show()
        progress_bar.start_progress(len(copy_jobs))

        try:
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(shutil.copytree, src, dest, dirs_exist_ok=True): src
                    for src, dest in copy_jobs
                }
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exception:
                        errors.append(f"{futures[future]}: {exception}")
                    progress_bar.update_progress()
        finally:
            progress_bar.stop_progress()
            progress_bar.close()

        # Save the merged class mapping if available
        if merged_class_mapping:
            merged_class_mapping_path = os.path.join(output_dir_path, "class_mapping.json")
            with open(merged_class_mapping_path, 'w') as json_file:
                json.dump(merged_class_mapping, json_file, indent=4)

        self.report_result(output_dir_path, len(copy_jobs), errors, unit="splits")
        return True

    def merge_yolo_datasets(self, output_dir_path, entries):
        """Merge detection, instance or semantic segmentation datasets.

        Unlike classification, class identity here is a bare integer whose
        meaning lives in each dataset's own data.yaml, so every annotation has to
        be rewritten: label rows get a new leading class index, and mask pixels
        get remapped through a lookup table. Images land in one flat folder per
        split, so filenames are made unique in lockstep with their sidecars.
        """
        names, luts = build_union_vocabulary([entry.names for entry in entries],
                                             background_first=(self.task == 'semantic'))

        if not names:
            QMessageBox.warning(self, "Warning", "No classes were found in the selected datasets.")
            return False

        if self.task == 'semantic' and len(names) > MAX_SEMANTIC_CLASSES:
            QMessageBox.warning(self,
                                "Warning",
                                f"The merged dataset has {len(names)} classes, but a semantic mask can carry "
                                f"at most {MAX_SEMANTIC_CLASSES} (255 is reserved for ignore).")
            return False

        sidecar_name, _ = sidecar_spec(self.task)

        # Every split is created even when empty, since the trainers expect the
        # folders named in data.yaml to exist.
        for split in OUTPUT_SPLITS:
            os.makedirs(os.path.join(output_dir_path, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir_path, split, sidecar_name), exist_ok=True)

        # Plan every copy up front. Destination names are assigned serially here
        # so the uniqueness check cannot race once the pool starts.
        jobs = []
        used_stems = {split: set() for split in OUTPUT_SPLITS}
        missing_sidecars = 0

        for entry, lut in zip(entries, luts):
            mask_table = build_mask_lut(lut) if self.task == 'semantic' else None
            splits = discover_dataset_splits(entry.source_path(), self.task)

            for split in OUTPUT_SPLITS:
                for image_path in sorted(splits.get(split, {})):
                    sidecar_path = splits[split][image_path]
                    if sidecar_path is None:
                        missing_sidecars += 1

                    source_stem, extension = os.path.splitext(os.path.basename(image_path))
                    stem = unique_stem(used_stems[split], source_stem)

                    image_dst = os.path.join(output_dir_path, split, 'images', stem + extension)
                    sidecar_dst = None
                    if sidecar_path is not None:
                        sidecar_ext = '.png' if self.task == 'semantic' else '.txt'
                        sidecar_dst = os.path.join(output_dir_path, split, sidecar_name, stem + sidecar_ext)

                    jobs.append((image_path, image_dst, sidecar_path, sidecar_dst, lut, mask_table))

        if not jobs:
            QMessageBox.warning(self,
                                "Warning",
                                "No images were found in the selected datasets. Check that each data.yaml "
                                "points at its train/val/test folders.")
            return False

        errors = self.run_jobs(jobs)

        # Write the merged vocabulary alongside the data, in both the form the
        # trainers read and the form this toolbox reads back on import.
        write_data_yaml(output_dir_path, names, self.task)

        merged_class_mapping = merge_class_mappings([entry.class_mapping for entry in entries], names)
        if merged_class_mapping:
            merged_class_mapping_path = os.path.join(output_dir_path, "class_mapping.json")
            with open(merged_class_mapping_path, 'w') as json_file:
                json.dump(merged_class_mapping, json_file, indent=4)

        extra = []
        extra.append(f"Merged classes: {len(names)}")
        if missing_sidecars:
            extra.append(f"Images with no annotation file: {missing_sidecars}")

        self.report_result(output_dir_path, len(jobs), errors, unit="images", extra=extra)
        return True

    def run_jobs(self, jobs):
        """Run the planned copy/remap work, returning the errors it hit.

        One failed file does not abandon the merge; the failures are counted and
        reported at the end so a single unreadable mask cannot cost the user the
        whole run.
        """
        errors = []

        progress_bar = ProgressBar(self, title="Merging Datasets")
        progress_bar.show()
        progress_bar.start_progress(len(jobs))

        try:
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(self.run_job, job): job for job in jobs}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exception:
                        errors.append(f"{futures[future][0]}: {exception}")
                    progress_bar.update_progress()
        finally:
            progress_bar.stop_progress()
            progress_bar.close()

        return errors

    def run_job(self, job):
        """Copy one image and rewrite the annotation that goes with it."""
        image_src, image_dst, sidecar_src, sidecar_dst, lut, mask_table = job

        copy_image(image_src, image_dst)

        if sidecar_src is None:
            return

        if mask_table is not None:
            remap_mask_file(sidecar_src, sidecar_dst, mask_table)
        else:
            remap_label_file(sidecar_src, sidecar_dst, lut)

    def report_result(self, output_dir_path, total, errors, unit="images", extra=None):
        """Tell the user what the merge produced, including anything that failed."""
        lines = [f"Datasets merged into:\n{output_dir_path}", ""]
        lines.append(f"{unit.capitalize()} processed: {total - len(errors)} of {total}")
        lines.extend(extra or [])

        if errors:
            lines.append("")
            lines.append(f"Failed: {len(errors)}")
            lines.extend(errors[:5])
            if len(errors) > 5:
                lines.append(f"...and {len(errors) - 5} more.")
            QMessageBox.warning(self, "Merged with Errors", "\n".join(lines))
        else:
            QMessageBox.information(self, "Success", "\n".join(lines))

    def accept(self):
        """
        Overrides the accept method to perform dataset merging before closing the dialog.
        """
        if self.merge_datasets():
            super().accept()
