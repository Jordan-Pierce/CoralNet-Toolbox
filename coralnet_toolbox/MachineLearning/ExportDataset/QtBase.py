import os
import random
import ujson as json

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (QFileDialog, QMessageBox, QCheckBox,
                             QVBoxLayout, QLabel, QLineEdit, QDialog, QHBoxLayout,
                             QPushButton, QFormLayout, QDialogButtonBox, QDoubleSpinBox,
                             QGroupBox, QTableWidget, QTableWidgetItem, QButtonGroup, QRadioButton,
                             QSpinBox, QHeaderView, QScrollArea, QFrame,
                             QWidget, QApplication)

from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon, get_window_icon
from coralnet_toolbox.MachineLearning.ExportDataset.export_dataset_utils import (
    REFRESH_SLOW_MS,
    RefreshTimings,
    format_refresh_duration,
    build_export_sample_paths,
    build_video_frame_path,
    busy_cursor,
    closing_progress_bar,
    frame_matches_stride,
    group_annotations_by_source,
    locked_window,
    normalize_source_path,
    parse_frame_path,
)
from coralnet_toolbox.MachineLearning.TrainModel.QtBase import (
    open_train_model_dialog_later,
    prompt_train_model,
)
from coralnet_toolbox.Rasters.extracted_images import has_active_set


# How long the refresh summary stays in the status bar
REFRESH_STATUS_TIMEOUT_MS = 6000

# Used when nothing more specific set a trigger: a label row checkbox is the
# only control wired straight to update_summary_statistics.
DEFAULT_REFRESH_TRIGGER = "label checkbox"


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Base(QDialog):
    supports_unlabeled_video_frames = False
    # Task the exported dataset trains; set by the subclasses
    task = None
    # Every dialog here is on the index now. A subclass that cannot be set
    # this False and keeps the per-annotation rescan in its own override.
    uses_annotation_index = True

    def __init__(self, main_window, parent=None):
        """
        Initialize the ExportDatasetDialog class.

        Args:
            main_window: The main window object.
            parent: The parent widget.
        """
        super().__init__(parent)
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window
        self.image_window = main_window.image_window

        # Height is set in fit_to_options, once the options column is built
        self.resize(1400, 700)
        self.setWindowTitle("Export Dataset")
        self.setWindowIcon(get_window_icon("coralnet.svg"))

        self.selected_labels = []
        self.selected_annotations = []
        self.updating_summary_statistics = False

        # Annotation index. One pass over the annotations groups them by label
        # and image path; after that a label checkbox costs a walk of that one
        # label's paths, not a rescan of every annotation. Rebuilt only when a
        # structural option changes - annotation types, image source, stride.
        self._index_dirty = True
        self._label_paths = {}         # label -> {image_path: [annotations]}
        self._label_totals = {}        # label -> annotation count
        self._path_counts = {}         # image_path -> {label: count}
        self._path_source = {}         # image_path -> source path
        self._path_frame = {}          # image_path -> frame index, or None
        self._paths_by_source = {}     # source path -> [image_path]
        self._video_frame_counts = {}  # source path -> frame count (video only)
        self._index_sources = []
        # Running selection totals, folded forward by _apply_label_selection
        self._path_selected = {}
        self._source_selected = {}
        self._applied_labels = set()
        # (row, label, checkbox) per table row, so the hot loop skips findChild
        self._label_rows = []
        self._split_dirty = True
        self._split_label_counts = ({}, {}, {})
        self._split_totals = (0, 0, 0)
        # What caused the refresh that is about to run, for the readout
        self._refresh_trigger = DEFAULT_REFRESH_TRIGGER
        # A record opened before update_summary_statistics was called, so the
        # phases that ran first are part of the same total
        self._pending_timings = None

        self.output_dir = None
        self.dataset_name = None
        self.train_ratio = 0.7
        self.val_ratio = 0.2
        self.test_ratio = 0.1

        # Options scroll on the left, the summary table sits on the right, and
        # the buttons span the bottom. self.layout is the left column, so the
        # subclasses' setup_* methods keep adding their groups there.
        self.root_layout = QVBoxLayout(self)
        columns_layout = QHBoxLayout()
        self.root_layout.addLayout(columns_layout, 1)

        options_widget = QWidget()
        self.layout = QVBoxLayout(options_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.options_scroll = QScrollArea()
        self.options_scroll.setWidgetResizable(True)
        self.options_scroll.setFrameShape(QFrame.NoFrame)
        self.options_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.options_scroll.setWidget(options_widget)
        columns_layout.addWidget(self.options_scroll)

        self.summary_layout = QVBoxLayout()
        columns_layout.addLayout(self.summary_layout, 1)

        # Setup the layout
        self.setup_info_layout()
        # Setup the output layout
        self.setup_output_layout()
        # Setup the ratio layout
        self.setup_ratio_layout()
        # Setup the data selection layout
        self.setup_data_selection_layout()
        # Setup the unlabeled handling layout (for semantic segmentation)
        self.setup_unlabeled_handling_layout()
        # Setup the table layout
        self.setup_table_layout()
        # Setup the status layout
        self.setup_status_layout()
        # Setup the button layout
        self.setup_button_layout()

        # Options pack to the top; any spare height stays below them
        self.layout.addStretch(1)

    def fit_to_options(self):
        """
        Size the dialog around the options column.

        The column shows in full, so it sets the dialog's height, and the
        summary table fills that height and scrolls its own rows. Only on a
        screen too short for the options does the column itself scroll.
        """
        options_hint = self.options_scroll.widget().sizeHint()
        scrollbar_width = self.options_scroll.verticalScrollBar().sizeHint().width()

        # Fit the column to its widest group so it never needs to scroll sideways
        self.options_scroll.setMinimumWidth(options_hint.width() + scrollbar_width)

        # Leave room for the title bar, buttons, and taskbar on short screens
        screen = self.screen() or QApplication.primaryScreen()
        max_height = int(screen.availableGeometry().height() * 0.8)
        self.options_scroll.setMinimumHeight(min(options_hint.height(), max_height))

        # The minimum hint now tracks the options column, not the table
        self.resize(self.width(), self.minimumSizeHint().height())

    def showEvent(self, event):
        """
        Handle the show event to update annotation type checkboxes, populate class filter list,
        and update summary statistics.

        Args:
            event: The show event.
        """
        super().showEvent(event)
        # Opening a dialog does its heaviest work in populate_class_filter_list,
        # which runs before update_summary_statistics and used to sit outside
        # the timing record - so the readout claimed a fraction of the real
        # wait. One record spans both, and the refresh appends to it.
        timings = RefreshTimings(type(self).__name__, "dialog open")

        # Setting these option widgets fires their signals, and those now
        # invalidate the index. Hold the summary off until the rows exist, or
        # every setChecked() here rebuilds the index for nothing.
        self.updating_summary_statistics = True
        try:
            self.update_annotation_type_checkboxes()
            self.update_video_options()
            self.update_extracted_note()
            # After update_video_options: the Video Frames group changes the height
            self.fit_to_options()
            timings.mark("options and layout")
            self.populate_class_filter_list()
            timings.mark("populate class list")
        finally:
            self.updating_summary_statistics = False
        self._refresh_trigger = "dialog open"
        self._pending_timings = timings
        self.update_summary_statistics()

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def setup_output_layout(self):
        """Setup output directory layout."""
        group_box = QGroupBox("Output Parameters")
        layout = QFormLayout()

        # Output Directory with Browse button on same line
        output_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_button = QPushButton("Browse...")
        self.output_dir_button.clicked.connect(self.browse_output_dir)
        self.output_dir_edit.setToolTip("Directory where the exported dataset will be saved.\nWill create subdirectories: images/, labels/, dataset.yaml")
        self.output_dir_button.setToolTip("Browse for an output directory.")
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(self.output_dir_button)
        layout.addRow("Output Directory:", output_layout)

        # Dataset Name
        self.dataset_name_edit = QLineEdit()
        self.dataset_name_edit.setToolTip("Name for the exported dataset.\nUsed for subdirectory naming and in the dataset.yaml file.")
        layout.addRow("Dataset Name:", self.dataset_name_edit)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_ratio_layout(self):
        """Setup the train, validation, and test ratio layout."""
        group_box = QGroupBox("Split Ratios")
        layout = QHBoxLayout()

        # Split Ratios
        self.train_ratio_spinbox = QDoubleSpinBox()
        self.train_ratio_spinbox.setRange(0.0, 1.0)
        self.train_ratio_spinbox.setSingleStep(0.1)
        self.train_ratio_spinbox.setValue(0.7)
        self.train_ratio_spinbox.setToolTip("Fraction of data for training set (0.0 to 1.0).\nStandard: 0.7 (70%). Used to train the model.")

        self.val_ratio_spinbox = QDoubleSpinBox()
        self.val_ratio_spinbox.setRange(0.0, 1.0)
        self.val_ratio_spinbox.setSingleStep(0.1)
        self.val_ratio_spinbox.setValue(0.2)
        self.val_ratio_spinbox.setToolTip("Fraction of data for validation set (0.0 to 1.0).\nStandard: 0.2 (20%). Used to tune hyperparameters during training.")

        self.test_ratio_spinbox = QDoubleSpinBox()
        self.test_ratio_spinbox.setRange(0.0, 1.0)
        self.test_ratio_spinbox.setSingleStep(0.1)
        self.test_ratio_spinbox.setValue(0.1)
        self.test_ratio_spinbox.setToolTip("Fraction of data for test set (0.0 to 1.0).\nStandard: 0.1 (10%). Used to evaluate final model performance.\nNote: ratios should sum to 1.0")

        # A ratio change re-splits. Without this the table - and the export
        # that reads train_images - kept the split built from the ratios the
        # dialog opened with.
        for spinbox in (self.train_ratio_spinbox, self.val_ratio_spinbox, self.test_ratio_spinbox):
            spinbox.valueChanged.connect(self.invalidate_split)

        layout.addWidget(QLabel("Train Ratio:"))
        layout.addWidget(self.train_ratio_spinbox)
        layout.addWidget(QLabel("Validation Ratio:"))
        layout.addWidget(self.val_ratio_spinbox)
        layout.addWidget(QLabel("Test Ratio:"))
        layout.addWidget(self.test_ratio_spinbox)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_data_selection_layout(self):
        """Setup the layout for data selection options in a horizontal arrangement."""
        options_layout = QHBoxLayout()

        # Create and add the group boxes
        annotation_types_group = self.create_annotation_layout()
        image_options_group = self.create_image_source_layout()
        negative_samples_group = self.create_negative_samples_layout()

        options_layout.addWidget(annotation_types_group)
        options_layout.addWidget(image_options_group)
        options_layout.addWidget(negative_samples_group)

        self.layout.addLayout(options_layout)

        self.video_options_group = self.create_video_options_layout()
        self.layout.addWidget(self.video_options_group)

    def create_annotation_layout(self):
        """Creates the annotation type checkboxes layout group box."""
        group_box = QGroupBox("Annotation Types")
        layout = QVBoxLayout()

        self.include_patches_checkbox = QCheckBox("Include Patch Annotations")
        self.include_patches_checkbox.setToolTip("Include point-based patch annotations in the export.\nPatches are converted to label files for training.")
        self.include_rectangles_checkbox = QCheckBox("Include Rectangle Annotations")
        self.include_rectangles_checkbox.setToolTip("Include bounding box rectangle annotations in the export.\nRectangles are converted to normalized YOLO format.")
        self.include_polygons_checkbox = QCheckBox("Include Polygon Annotations")
        self.include_polygons_checkbox.setToolTip("Include polygon annotations in the export.\nPolygons are converted to normalized coordinate format for segmentation tasks.")

        # These are user-editable, so they have to invalidate the index; before
        # this the table ignored them until Refresh was pressed.
        for checkbox in (self.include_patches_checkbox,
                         self.include_rectangles_checkbox,
                         self.include_polygons_checkbox):
            checkbox.stateChanged.connect(self.refresh_structure)

        layout.addWidget(self.include_patches_checkbox)
        layout.addWidget(self.include_rectangles_checkbox)
        layout.addWidget(self.include_polygons_checkbox)

        group_box.setLayout(layout)
        return group_box

    def create_image_source_layout(self):
        """Creates the image source options layout group box."""
        group_box = QGroupBox("Image Source")
        layout = QVBoxLayout()

        self.image_options_group = QButtonGroup(self)

        self.all_images_radio = QRadioButton("All Images")
        self.all_images_radio.setToolTip("Export all images in the project.\nIncludes all images regardless of annotations or filtering.")
        self.filtered_images_radio = QRadioButton("Filtered Images")
        self.filtered_images_radio.setToolTip("Export only images that match the current table filters.\nUse table filters to select a subset of images before export.")

        self.image_options_group.addButton(self.all_images_radio)
        self.image_options_group.addButton(self.filtered_images_radio)
        self.image_options_group.setExclusive(True)

        # Default to the filtered set: it matches the table the user is looking
        # at, and equals every image when no filter is active.
        self.filtered_images_radio.setChecked(True)

        # One connection for the pair: a click toggles both buttons, so wiring
        # both ran the whole update twice per click.
        self.filtered_images_radio.toggled.connect(self.update_image_selection)

        layout.addWidget(self.all_images_radio)
        layout.addWidget(self.filtered_images_radio)

        group_box.setLayout(layout)
        return group_box

    def create_negative_samples_layout(self):
        """Creates the negative sample options layout group box."""
        group_box = QGroupBox("Negative Samples")
        layout = QVBoxLayout()

        self.negative_samples_group = QButtonGroup(self)

        self.include_negatives_radio = QRadioButton("Include Negatives")
        self.include_negatives_radio.setToolTip("Include images with NO annotations.\nUseful for training models to recognize negative examples (background/non-objects).")
        self.exclude_negatives_radio = QRadioButton("Exclude Negatives")
        self.exclude_negatives_radio.setToolTip("Exclude images with NO annotations.\nOnly export images that have at least one annotation.")

        self.negative_samples_group.addButton(self.include_negatives_radio)
        self.negative_samples_group.addButton(self.exclude_negatives_radio)
        self.negative_samples_group.setExclusive(True)

        self.exclude_negatives_radio.setChecked(True)

        # Connect to update stats when changed. Only one needed for the group.
        # Negatives decide which sources yield sample paths, so the split goes too.
        self.include_negatives_radio.toggled.connect(self.refresh_structure)

        layout.addWidget(self.include_negatives_radio)
        layout.addWidget(self.exclude_negatives_radio)

        group_box.setLayout(layout)
        return group_box

    def create_video_options_layout(self):
        """Create the optional video export controls."""
        group_box = QGroupBox("Video Frames")
        layout = QHBoxLayout()

        layout.addStretch(1)

        self.split_by_source_checkbox = QCheckBox("Split by source video")
        self.split_by_source_checkbox.setChecked(True)
        self.split_by_source_checkbox.setToolTip(
            "Keep frames from the same source video together when splitting train, val, and test.\n"
            "Prevents the model from seeing different frames from the same video across splits."
        )
        self.split_by_source_checkbox.stateChanged.connect(self.invalidate_split)
        layout.addWidget(self.split_by_source_checkbox)

        layout.addStretch(1)

        stride_widget = QWidget()
        stride_layout = QHBoxLayout(stride_widget)
        stride_layout.setContentsMargins(0, 0, 0, 0)
        stride_layout.setSpacing(6)

        stride_label = QLabel("Frame stride:")
        stride_label.setToolTip("Only export every Nth frame from a video source.\nReduces dataset size by skipping frames.\nExample: 2 = every other frame, 10 = every 10th frame.")
        stride_layout.addWidget(stride_label)

        self.frame_stride_spinbox = QSpinBox()
        self.frame_stride_spinbox.setRange(1, 999999)
        self.frame_stride_spinbox.setValue(1)
        self.frame_stride_spinbox.setToolTip("Extract every Nth frame (stride=1 means all frames).\nUseful to reduce redundancy in video-based datasets.")
        self.frame_stride_spinbox.valueChanged.connect(self.refresh_structure)
        stride_layout.addWidget(self.frame_stride_spinbox)

        layout.addWidget(stride_widget)

        layout.addStretch(1)

        self.export_unlabeled_video_frames_checkbox = QCheckBox("Export unlabeled video frames")
        self.export_unlabeled_video_frames_checkbox.setChecked(False)
        self.export_unlabeled_video_frames_checkbox.setToolTip(
            "Also export video frames without annotations in the export.\n"
            "Useful for negative samples. Disabled for classification exports."
        )
        self.export_unlabeled_video_frames_checkbox.stateChanged.connect(self.invalidate_split)
        layout.addWidget(self.export_unlabeled_video_frames_checkbox)

        layout.addStretch(1)

        group_box.setLayout(layout)
        group_box.setVisible(self.has_video_rasters())
        return group_box

    def has_video_rasters(self):
        """Return True when at least one loaded raster is a VideoRaster."""
        for image_path in self.image_window.raster_manager.image_paths:
            raster = self.image_window.raster_manager.get_raster(image_path)
            if raster is not None and getattr(raster, 'raster_type', '') == 'VideoRaster':
                return True
        return False

    def update_video_options(self):
        """Refresh video option availability for the current project state."""
        if not hasattr(self, 'video_options_group'):
            return

        has_video = self.has_video_rasters()
        self.video_options_group.setVisible(has_video)
        self.split_by_source_checkbox.setEnabled(has_video)
        self.frame_stride_spinbox.setEnabled(has_video)
        self.export_unlabeled_video_frames_checkbox.setEnabled(
            has_video and self.supports_unlabeled_video_frames
        )

        if not self.supports_unlabeled_video_frames:
            self.export_unlabeled_video_frames_checkbox.setChecked(False)

    def get_selected_image_paths(self):
        """Return the currently selected image paths from the project or the filtered table.

        A raster whose work areas are extracted is left out: its extracted
        images carry the same pixels, and exporting both counts them twice.
        """
        paths, _ = self._split_extracted_parents(self._candidate_image_paths())
        return paths

    def _candidate_image_paths(self):
        if self.filtered_images_radio.isChecked():
            return list(self.image_window.table_model.filtered_paths)
        return list(self.image_window.raster_manager.image_paths)

    def _split_extracted_parents(self, paths):
        """(paths to export, paths left out because their work areas are extracted)."""
        raster_manager = self.image_window.raster_manager
        kept, left_out = [], []
        for path in paths:
            raster = raster_manager.get_raster(path)
            (left_out if raster is not None and has_active_set(raster) else kept).append(path)
        return kept, left_out

    def update_extracted_note(self):
        """Say how many rasters are left out because their work areas are extracted."""
        _, left_out = self._split_extracted_parents(self._candidate_image_paths())
        if left_out:
            count = len(left_out)
            self.extracted_note_label.setText(
                f"{count} raster{'s' if count != 1 else ''} left out: "
                f"{'its' if count == 1 else 'their'} work areas are extracted.")
        self.extracted_note_label.setVisible(bool(left_out))

    def get_selected_source_paths(self):
        """Return the selected paths normalized to their underlying source path."""
        return list(dict.fromkeys(normalize_source_path(path) for path in self.get_selected_image_paths()))

    def allows_unlabeled_video_export(self):
        """Return True when unlabeled video-frame export is enabled for this dialog."""
        return bool(
            hasattr(self, 'export_unlabeled_video_frames_checkbox')
            and self.export_unlabeled_video_frames_checkbox.isEnabled()
            and self.export_unlabeled_video_frames_checkbox.isChecked()
        )

    def _frame_stride(self):
        """Return the current video frame stride."""
        if hasattr(self, 'frame_stride_spinbox'):
            return max(1, int(self.frame_stride_spinbox.value()))
        return 1
    
    def setup_unlabeled_handling_layout(self):
        """Setup the unlabeled handling options layout group box (for semantic segmentation)."""
        raise NotImplementedError("Method must be implemented in the subclass.")

    def setup_table_layout(self):
        """Setup the label counts table layout."""
        group_box = QGroupBox("Annotation Table")
        layout = QVBoxLayout()

        # Label Counts Table
        self.label_counts_table = QTableWidget(0, 7)
        self.label_counts_table.setHorizontalHeaderLabels(["Include",
                                                           "Label",
                                                           "Annotations",
                                                           "Train",
                                                           "Val",
                                                           "Test",
                                                           "Images"])
        header = self.label_counts_table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignCenter)
        # The table widget always widened with the dialog, but its columns kept
        # their fixed default widths, so the extra width sat empty to the right.
        # Share it out instead; the checkbox column keeps only what it needs.
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        # Note: No delegate needed - using widget-based checkboxes via setCellWidget
        layout.addWidget(self.label_counts_table)

        group_box.setLayout(layout)
        # Stretch 1 so the table takes the right column's full height
        self.summary_layout.addWidget(group_box, 1)

    def setup_status_layout(self):
        """Setup the ready status layout."""
        group_box = QGroupBox("Status")
        layout = QHBoxLayout()

        # Label for Ready Status
        self.ready_label = QLabel("❌ Not Ready")
        layout.addWidget(self.ready_label)

        # Rasters left out because their work areas are extracted
        self.extracted_note_label = QLabel()
        self.extracted_note_label.setVisible(False)
        layout.addWidget(self.extracted_note_label)

        # Add a spacer to push image counts to the right
        layout.addStretch() 

        # Label for Total Images
        self.total_images_label = QLabel("Total Images: 0")
        self.total_images_label.setAlignment(Qt.AlignRight)
        layout.addWidget(self.total_images_label)

        # Label for Split Counts
        self.split_summary_label = QLabel("(Train: 0, Val: 0, Test: 0)")
        self.split_summary_label.setAlignment(Qt.AlignRight)
        layout.addWidget(self.split_summary_label)

        group_box.setLayout(layout)
        self.summary_layout.addWidget(group_box)

    def setup_button_layout(self):
        """Setup the button layout."""
        button_layout = QHBoxLayout()

        # Add Refresh button
        self.refresh_button = QPushButton("Refresh | Shuffle")
        self.refresh_button.setToolTip("Recalculate stats and re-shuffle train/val/test splits")
        self.refresh_button.clicked.connect(self.refresh_all)
        button_layout.addWidget(self.refresh_button)

        # Add spacer to push OK/Cancel to right
        button_layout.addStretch()

        # Add OK and Cancel buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        button_layout.addWidget(self.buttons)

        self.root_layout.addLayout(button_layout)

    def update_annotation_type_checkboxes(self):
        raise NotImplementedError("Method must be implemented in the subclass.")

    def set_cell_color(self, row, column, color):
        """
        Set the background color of a cell in the label counts table.

        Args:
            row: The row index of the cell.
            column: The column index of the cell.
            color: The color to set as the background.
        """
        item = self.label_counts_table.item(row, column)
        if item is not None:
            background = QColor(color)
            item.setBackground(QBrush(background))
            item.setForeground(QBrush(self._foreground_for_background(background)))

    @staticmethod
    def _foreground_for_background(background):
        """Return a readable text color for the given background."""
        red, green, blue, _ = QColor(background).getRgb()
        luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255.0
        return QColor(0, 0, 0) if luminance > 0.5 else QColor(255, 255, 255)

    def browse_output_dir(self):
        """
        Browse and select an output directory.
        """
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory")
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def get_class_mapping(self):
        """
        Get the class mapping for the selected labels.

        Returns:
            dict: Dictionary containing class mappings.
        """
        # Get the label objects for the selected labels
        class_mapping = {}

        for label in self.main_window.label_window.labels:
            if label.short_label_code in self.selected_labels:
                class_mapping[label.short_label_code] = label.to_dict()

        return class_mapping

    @staticmethod
    def create_centered_item(text):
        """
        Create a QTableWidgetItem with centered text alignment.

        Args:
            text (str): The text to display in the item.

        Returns:
            QTableWidgetItem: A table item with centered alignment.
        """
        item = QTableWidgetItem(str(text))
        item.setTextAlignment(Qt.AlignCenter)
        return item

    @staticmethod
    def save_class_mapping_json(class_mapping, output_dir_path):
        """
        Save the class mapping dictionary as a JSON file.

        Args:
            class_mapping (dict): Dictionary containing class mappings.
            output_dir_path (str): Path to the output directory.
        """
        # Save the class_mapping dictionary as a JSON file
        class_mapping_path = os.path.join(output_dir_path, "class_mapping.json")
        with open(class_mapping_path, 'w') as json_file:
            json.dump(class_mapping, json_file, indent=4)

    @staticmethod
    def merge_class_mappings(existing_mapping, new_mapping):
        """
        Merge the new class mappings with the existing ones without duplicates.

        Args:
            existing_mapping (dict): Existing class mappings.
            new_mapping (dict): New class mappings.

        Returns:
            dict: Merged class mappings.
        """
        # Merge the new class mappings with the existing ones without duplicates
        merged_mapping = existing_mapping.copy()
        for key, value in new_mapping.items():
            if key not in merged_mapping:
                merged_mapping[key] = value

        return merged_mapping

    def filter_annotations(self):
        """
        Filter annotations based on the selected annotation types and current tab.

        Returns:
            list: List of filtered annotations.
        """
        allowed_types = set()
        if self.include_patches_checkbox.isChecked():
            allowed_types.add(PatchAnnotation)
        if self.include_rectangles_checkbox.isChecked():
            allowed_types.add(RectangleAnnotation)
        if self.include_polygons_checkbox.isChecked():
            allowed_types.add(PolygonAnnotation)

        selected_set = set(self.selected_labels)
        selected_sources = set(self.get_selected_source_paths())
        frame_stride = self.frame_stride_spinbox.value() if hasattr(self, 'frame_stride_spinbox') else 1

        return [
            annotation for annotation in self.annotation_window.annotations_dict.values()
            if type(annotation) in allowed_types
            and annotation.label.short_label_code in selected_set
            and normalize_source_path(annotation.image_path) in selected_sources
            and frame_matches_stride(annotation.image_path, frame_stride)
        ]

    def get_hidden_label_codes(self):
        """
        Return the label codes the user has hidden in the Label Window.

        Returns an empty set when every label is hidden: hiding everything is a
        view gesture, not an export filter, and an all-unchecked table would
        export nothing.

        Returns:
            set: Short label codes currently hidden in the Label Window.
        """
        labels = self.main_window.label_window.labels
        hidden = {label.short_label_code for label in labels if not label.is_visible}
        return set() if len(hidden) == len(labels) else hidden

    def create_include_checkbox_cell(self, label_code, hidden_codes):
        """
        Build the centered "Include" checkbox cell for one label row.

        Label visibility is read here and never written back - nothing in this
        dialog touches the Label Window - so re-checking a hidden label exports
        it without making it visible again. Keep it one-way.

        Args:
            label_code (str): Short label code for this row.
            hidden_codes (set): Label codes hidden in the Label Window.

        Returns:
            QWidget: Container widget holding the checkbox.
        """
        is_hidden = label_code in hidden_codes

        include_checkbox = QCheckBox()
        include_checkbox.setChecked(not is_hidden)
        if is_hidden:
            include_checkbox.setToolTip("Hidden in the Label Window, so unchecked by default.\n"
                                        "Re-check to export it; that does not unhide it.")
        include_checkbox.stateChanged.connect(self.update_summary_statistics)

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        layout.addWidget(include_checkbox)
        layout.addStretch()
        return container

    def populate_class_filter_list(self):
        """
        Populate the class filter list with labels and their counts.
        """
        # Snapshot before the loop below: update_progress() pumps the event
        # loop, so the live dict can change size mid-iteration.
        all_annotations = list(self.annotation_window.annotations_dict.values())

        with busy_cursor():
            # Set the row count to 0
            self.label_counts_table.setRowCount(0)

            # Create a progress bar
            progress_bar = ProgressBar(self, "Populating Class Lists")
            progress_bar.show()
            progress_bar.start_progress(len(all_annotations))

            with closing_progress_bar(progress_bar):
                label_counts = {}
                label_image_counts = {}
                # Count the occurrences of each label and unique images per label
                for annotation in all_annotations:
                    label = annotation.label.short_label_code
                    image_path = annotation.image_path
                    if label != 'Review':
                        if label in label_counts:
                            label_counts[label] += 1
                            label_image_counts[label].add(image_path)
                        else:
                            label_counts[label] = 1
                            label_image_counts[label] = {image_path}

                    progress_bar.update_progress()

                # Sort the labels by their counts in descending order
                sorted_label_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)

                # Populate the label counts table with labels and their counts
                self.label_counts_table.setColumnCount(7)
                self.label_counts_table.setHorizontalHeaderLabels(["Include",
                                                                   "Label",
                                                                   "Annotations",
                                                                   "Train",
                                                                   "Val",
                                                                   "Test",
                                                                   "Images"])

                # Populate the label counts table with labels and their counts
                # Labels hidden in the Label Window start unchecked, the same way
                # the Image Source defaults to the filtered table.
                hidden_codes = self.get_hidden_label_codes()

                label_rows = []
                self.label_counts_table.setUpdatesEnabled(False)
                row = 0
                for label, count in sorted_label_counts:
                    container = self.create_include_checkbox_cell(label, hidden_codes)

                    # Create centered table items using helper function
                    label_item = self.create_centered_item(label)
                    anno_count = self.create_centered_item(count)
                    train_item = self.create_centered_item("0")
                    val_item = self.create_centered_item("0")
                    test_item = self.create_centered_item("0")
                    images_item = self.create_centered_item(len(label_image_counts[label]))

                    self.label_counts_table.insertRow(row)
                    self.label_counts_table.setCellWidget(row, 0, container)
                    self.label_counts_table.setItem(row, 1, label_item)
                    self.label_counts_table.setItem(row, 2, anno_count)
                    self.label_counts_table.setItem(row, 3, train_item)
                    self.label_counts_table.setItem(row, 4, val_item)
                    self.label_counts_table.setItem(row, 5, test_item)
                    self.label_counts_table.setItem(row, 6, images_item)

                    label_rows.append((row, label, container.findChild(QCheckBox)))
                    row += 1
                self.label_counts_table.setUpdatesEnabled(True)
                progress_bar.finish_progress()

        # The rows hold new checkboxes, and the annotations may have changed
        # since this dialog last opened, so both caches start over.
        self._label_rows = label_rows
        self._index_dirty = True
        self._reset_selection_state()

    def split_data(self):
        """
        Split the data by images based on the specified ratios.
        """
        self.train_ratio = self.train_ratio_spinbox.value()
        self.val_ratio = self.val_ratio_spinbox.value()
        self.test_ratio = self.test_ratio_spinbox.value()

        selected_source_paths = self.get_selected_source_paths()
        annotations_by_source = group_annotations_by_source(self.selected_annotations)
        frame_stride = self.frame_stride_spinbox.value() if hasattr(self, 'frame_stride_spinbox') else 1
        split_by_source = not hasattr(self, 'split_by_source_checkbox') or self.split_by_source_checkbox.isChecked()
        export_unlabeled_video_frames = self.allows_unlabeled_video_export() and self.include_negatives_radio.isChecked()

        source_entries = []
        for source_path in selected_source_paths:
            source_annotations = annotations_by_source.get(source_path, [])

            if self.exclude_negatives_radio.isChecked() and not source_annotations:
                continue

            sample_paths = build_export_sample_paths(
                source_path,
                source_annotations,
                self.image_window.raster_manager,
                frame_stride=frame_stride,
                export_unlabeled_video_frames=export_unlabeled_video_frames,
            )
            if not sample_paths:
                continue

            source_entries.append((source_path, sample_paths, source_annotations))

        self.train_images = []
        self.val_images = []
        self.test_images = []

        if not source_entries:
            return

        if split_by_source:
            random.shuffle(source_entries)
            train_split = int(len(source_entries) * self.train_ratio)
            val_split = int(len(source_entries) * (self.train_ratio + self.val_ratio))

            train_entries = source_entries[:train_split] if self.train_ratio > 0 else []
            val_entries = source_entries[train_split:val_split] if self.val_ratio > 0 else []
            test_entries = source_entries[val_split:] if self.test_ratio > 0 else []

            self.train_images = [sample_path for _, sample_paths, _ in train_entries for sample_path in sample_paths]
            self.val_images = [sample_path for _, sample_paths, _ in val_entries for sample_path in sample_paths]
            self.test_images = [sample_path for _, sample_paths, _ in test_entries for sample_path in sample_paths]
            return

        sample_paths = [sample_path for _, sample_paths, _ in source_entries for sample_path in sample_paths]
        random.shuffle(sample_paths)

        train_split = int(len(sample_paths) * self.train_ratio)
        val_split = int(len(sample_paths) * (self.train_ratio + self.val_ratio))

        if self.train_ratio > 0:
            self.train_images = sample_paths[:train_split]
        if self.val_ratio > 0:
            self.val_images = sample_paths[train_split:val_split]
        if self.test_ratio > 0:
            self.test_images = sample_paths[val_split:]

    def determine_splits(self):
        """
        Determine the splits for train, validation, and test annotations.
        """
        train_set = set(self.train_images)
        val_set = set(self.val_images)
        test_set = set(self.test_images)
        self.train_annotations = [a for a in self.selected_annotations if a.image_path in train_set]
        self.val_annotations = [a for a in self.selected_annotations if a.image_path in val_set]
        self.test_annotations = [a for a in self.selected_annotations if a.image_path in test_set]

    def check_label_distribution(self):
        """
        Check the label distribution in the splits to ensure all labels are present,
        and only allow specific split combinations.
    
        Returns:
            bool: True if all labels are present in all splits and split config is allowed, False otherwise.
        """
        # Get the ratios from the spinboxes
        train_ratio = self.train_ratio_spinbox.value()
        val_ratio = self.val_ratio_spinbox.value()
        test_ratio = self.test_ratio_spinbox.value()
    
        # Only allow these split combinations:
        # - Train only
        # - Test only
        # - Train/Val
        # - Train/Val/Test
        
        allowed = False

        # Train only
        if train_ratio == 1.0 and val_ratio == 0 and test_ratio == 0:
            allowed = True
            
        # Test only
        elif train_ratio == 0 and val_ratio == 0 and test_ratio == 1.0:
            allowed = True
            
        # Train/Val
        elif train_ratio > 0 and val_ratio > 0 and test_ratio == 0:
            if abs(train_ratio + val_ratio - 1.0) < 1e-9:
                allowed = True
                
        # Train/Val/Test
        elif train_ratio > 0 and val_ratio > 0 and test_ratio > 0:
            if abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9:
                allowed = True

        if not allowed:
            return False

        if self.allows_unlabeled_video_export() and self.include_negatives_radio.isChecked():
            if train_ratio > 0 and len(self.train_images) == 0:
                return False
            if val_ratio > 0 and len(self.val_images) == 0:
                return False
            if test_ratio > 0 and len(self.test_images) == 0:
                return False
            return True
    
        # These come from the index, already summed for the table. The old
        # Counters here were a second pass over the same three split lists.
        train_label_counts, val_label_counts, test_label_counts = self._split_label_counts
        train_total, val_total, test_total = self._split_totals

        # Check the conditions for each split
        for label in self.selected_labels:
            if train_ratio > 0 and train_label_counts.get(label, 0) == 0:
                return False
            if val_ratio > 0 and val_label_counts.get(label, 0) == 0:
                return False
            if test_ratio > 0 and test_label_counts.get(label, 0) == 0:
                return False
    
        # Additional checks to ensure no empty splits
        if train_ratio > 0 and train_total == 0:
            return False
        if val_ratio > 0 and val_total == 0:
            return False
        if test_ratio > 0 and test_total == 0:
            return False
    
        return True

    def _begin_timings(self):
        """Open a timing record for the refresh that is starting.

        A record left in _pending_timings by a caller that began earlier is
        continued instead, so its phases and the refresh's add up to one total.
        """
        trigger = self._refresh_trigger
        self._refresh_trigger = DEFAULT_REFRESH_TRIGGER

        pending, self._pending_timings = self._pending_timings, None
        return pending if pending is not None else RefreshTimings(type(self).__name__, trigger)

    def _refresh_scope(self):
        """(labels, images, annotations) currently selected, for the readout.

        An annotation carrying several selected labels counts once per label, so
        for a semantic project the annotation figure reads high. It is a status
        line, not a total an export depends on.
        """
        images = len(self.train_images) + len(self.val_images) + len(self.test_images)
        annotations = sum(self._label_totals.get(label, 0) for label in self.selected_labels)
        return len(self.selected_labels), images, annotations

    def _report_timings(self, timings):
        """Summarise the refresh in the status bar.

        A refresh that takes a noticeable moment should say what it covered and
        how long it took, rather than leaving the dialog looking stuck. Past
        REFRESH_SLOW_MS it also names the phase that accounted for most of it,
        which is the first thing worth knowing when one is slow.
        """
        try:
            labels, images, annotations = self._refresh_scope()
        except Exception:
            # A readout must never be the reason a refresh fails
            return

        duration = format_refresh_duration(timings.total_ms)
        if timings.total_ms >= REFRESH_SLOW_MS:
            slowest = timings.slowest()
            if slowest:
                duration += f", most of it {slowest[0]}"

        message = (f"Export summary: {labels:,} labels, {images:,} images, "
                   f"{annotations:,} annotations ({duration})")

        try:
            status_bar = getattr(self.main_window, 'status_bar', None)
            if status_bar is not None:
                status_bar.showMessage(message, REFRESH_STATUS_TIMEOUT_MS)
        except Exception:
            pass

    def refresh_all(self):
        """Rebuild the index from scratch and re-shuffle the splits."""
        self._refresh_trigger = "Refresh button"
        self._index_dirty = True
        self._split_dirty = True
        self.update_summary_statistics()

    def refresh_structure(self):
        """Mark the index stale - an option changed which annotations qualify."""
        self._refresh_trigger = "structural option"
        self._index_dirty = True
        self.update_summary_statistics()

    def invalidate_split(self):
        """Mark the split stale - an option changed how samples are divided."""
        self._refresh_trigger = "split option"
        self._split_dirty = True
        self.update_summary_statistics()

    def _allowed_annotation_types(self):
        """Return the annotation classes the type checkboxes currently allow."""
        allowed = set()
        if self.include_patches_checkbox.isChecked():
            allowed.add(PatchAnnotation)
        if self.include_rectangles_checkbox.isChecked():
            allowed.add(RectangleAnnotation)
        if self.include_polygons_checkbox.isChecked():
            allowed.add(PolygonAnnotation)
        return allowed

    def _indexable_annotations(self):
        """The annotations the index should consider, before path filtering.

        The type checkboxes are applied here, so a subclass that indexes
        something else - masks, which are not in annotations_dict - overrides
        this rather than the index itself.
        """
        allowed_types = self._allowed_annotation_types()
        return [annotation for annotation in self.annotation_window.annotations_dict.values()
                if type(annotation) in allowed_types]

    def _annotation_labels(self, annotation):
        """The label codes this annotation counts towards.

        One for a vector annotation. A subclass whose annotations carry several
        - a semantic mask holds a label per painted class - returns all of them,
        and the index then lists that annotation under each.
        """
        return (annotation.label.short_label_code,)

    def rebuild_annotation_index(self):
        """Group every exportable annotation by label and by image path.

        This is the only pass over the annotations. It applies the filters that
        do not depend on the label checkboxes - annotation type, image source,
        frame stride - and parses each distinct image path once rather than
        twice per annotation, which is what the old per-click rescan did.
        """
        sources = self.get_selected_source_paths()
        source_set = set(sources)
        frame_stride = self._frame_stride()

        label_paths = {}
        label_totals = {}
        path_counts = {}
        path_source = {}
        path_frame = {}
        excluded_paths = set()

        for annotation in self._indexable_annotations():
            image_path = annotation.image_path
            source = path_source.get(image_path)
            if source is None:
                if image_path in excluded_paths:
                    continue
                # Parse once per distinct path, not once per annotation
                source, frame_idx = parse_frame_path(image_path)
                if source not in source_set or not frame_matches_stride(image_path, frame_stride):
                    excluded_paths.add(image_path)
                    continue
                path_source[image_path] = source
                path_frame[image_path] = frame_idx
                path_counts[image_path] = {}

            counts = path_counts[image_path]
            for label in self._annotation_labels(annotation):
                label_paths.setdefault(label, {}).setdefault(image_path, []).append(annotation)
                label_totals[label] = label_totals.get(label, 0) + 1
                counts[label] = counts.get(label, 0) + 1

        paths_by_source = {}
        for image_path, source in path_source.items():
            paths_by_source.setdefault(source, []).append(image_path)

        # Frame counts for the video sources, so the split never calls
        # get_raster() again while the user is clicking labels.
        raster_manager = self.image_window.raster_manager
        video_frame_counts = {}
        for source in sources:
            raster = raster_manager.get_raster(source)
            if getattr(raster, 'raster_type', '') == 'VideoRaster':
                video_frame_counts[source] = int(getattr(raster, 'frame_count', 0) or 0)

        self._label_paths = label_paths
        self._label_totals = label_totals
        self._path_counts = path_counts
        self._path_source = path_source
        self._path_frame = path_frame
        self._paths_by_source = paths_by_source
        self._video_frame_counts = video_frame_counts
        self._index_sources = sources
        self._index_dirty = False

        # The running totals belong to the old index, so start them over
        self._reset_selection_state()

    def _reset_selection_state(self):
        """Drop the running selection totals and force a re-split."""
        self._path_selected = {}
        self._source_selected = {}
        self._applied_labels = set()
        self._split_dirty = True

    def _apply_label_selection(self, selected_labels):
        """Fold the checkbox changes into the running per-path totals.

        Only the labels that changed are walked, so one checkbox costs that
        label's paths and nothing else.

        Args:
            selected_labels: Short label codes currently checked.

        Returns:
            tuple: (paths, sources) whose selected count crossed zero.
        """
        selected = set(selected_labels)
        path_selected = self._path_selected
        source_selected = self._source_selected
        path_source = self._path_source

        crossed_paths = set()
        crossed_sources = set()

        for labels, sign in ((selected - self._applied_labels, 1),
                             (self._applied_labels - selected, -1)):
            for label in labels:
                for image_path, annotations in self._label_paths.get(label, {}).items():
                    delta = sign * len(annotations)

                    before = path_selected.get(image_path, 0)
                    after = before + delta
                    path_selected[image_path] = after
                    if (before == 0) != (after == 0):
                        crossed_paths.add(image_path)

                    source = path_source[image_path]
                    source_before = source_selected.get(source, 0)
                    source_after = source_before + delta
                    source_selected[source] = source_after
                    if (source_before == 0) != (source_after == 0):
                        crossed_sources.add(source)

        self._applied_labels = selected
        return crossed_paths, crossed_sources

    def _selection_changes_splits(self, crossed_paths, crossed_sources):
        """Return True when the label change alters which samples get split.

        Most label clicks do not: a static source contributes itself whatever
        its labels are. Only a source appearing or disappearing, or a video
        frame gaining or losing its last annotation, moves the split.
        """
        if self.exclude_negatives_radio.isChecked() and crossed_sources:
            return True

        if self.allows_unlabeled_video_export() and self.include_negatives_radio.isChecked():
            # Every frame by stride, annotated or not, so labels do not matter
            return False

        for image_path in crossed_paths:
            if self._path_source[image_path] in self._video_frame_counts:
                return True
        return False

    def _annotated_frame_paths(self, source_path):
        """Return this video source's frame paths that still hold a selection."""
        path_frame = self._path_frame
        paths = [path for path in self._paths_by_source.get(source_path, ())
                 if self._path_selected.get(path, 0) > 0 and path_frame[path] is not None]
        paths.sort(key=lambda path: path_frame[path])
        return paths

    def split_data_from_index(self):
        """Split the data by images, reading the index instead of annotations.

        Mirrors split_data(), which still serves the subclasses that do not use
        the index, but builds each source's sample paths from the running
        totals rather than from a fresh pass over the annotation objects.
        """
        self.train_ratio = self.train_ratio_spinbox.value()
        self.val_ratio = self.val_ratio_spinbox.value()
        self.test_ratio = self.test_ratio_spinbox.value()

        frame_stride = self._frame_stride()
        split_by_source = not hasattr(self, 'split_by_source_checkbox') or self.split_by_source_checkbox.isChecked()
        export_unlabeled_video_frames = self.allows_unlabeled_video_export() and self.include_negatives_radio.isChecked()
        exclude_negatives = self.exclude_negatives_radio.isChecked()

        source_entries = []
        for source_path in self._index_sources:
            if exclude_negatives and self._source_selected.get(source_path, 0) == 0:
                continue

            frame_count = self._video_frame_counts.get(source_path)
            if frame_count is None:
                sample_paths = [source_path]
            elif export_unlabeled_video_frames:
                sample_paths = [build_video_frame_path(source_path, frame_idx)
                                for frame_idx in range(0, frame_count, frame_stride)]
            else:
                sample_paths = self._annotated_frame_paths(source_path)

            if not sample_paths:
                continue

            source_entries.append(sample_paths)

        self.train_images = []
        self.val_images = []
        self.test_images = []

        if not source_entries:
            return

        if split_by_source:
            random.shuffle(source_entries)
            train_split = int(len(source_entries) * self.train_ratio)
            val_split = int(len(source_entries) * (self.train_ratio + self.val_ratio))

            train_entries = source_entries[:train_split] if self.train_ratio > 0 else []
            val_entries = source_entries[train_split:val_split] if self.val_ratio > 0 else []
            test_entries = source_entries[val_split:] if self.test_ratio > 0 else []

            self.train_images = [path for paths in train_entries for path in paths]
            self.val_images = [path for paths in val_entries for path in paths]
            self.test_images = [path for paths in test_entries for path in paths]
            return

        sample_paths = [path for paths in source_entries for path in paths]
        random.shuffle(sample_paths)

        train_split = int(len(sample_paths) * self.train_ratio)
        val_split = int(len(sample_paths) * (self.train_ratio + self.val_ratio))

        if self.train_ratio > 0:
            self.train_images = sample_paths[:train_split]
        if self.val_ratio > 0:
            self.val_images = sample_paths[train_split:val_split]
        if self.test_ratio > 0:
            self.test_images = sample_paths[val_split:]

    def compute_split_label_counts(self):
        """Sum each split's per-label counts straight out of the index.

        Sets self._split_label_counts and self._split_totals, which stand in
        for the train/val/test annotation lists the table used to need.
        """
        selected = self._applied_labels
        path_counts = self._path_counts

        split_counts = []
        split_totals = []
        for image_paths in (self.train_images, self.val_images, self.test_images):
            counts = {}
            total = 0
            for image_path in image_paths:
                for label, count in path_counts.get(image_path, {}).items():
                    if label in selected:
                        counts[label] = counts.get(label, 0) + count
                        total += count
            split_counts.append(counts)
            split_totals.append(total)

        self._split_label_counts = tuple(split_counts)
        self._split_totals = tuple(split_totals)

    def _selected_annotations_from_index(self):
        """Return the selected annotations - the set filter_annotations gives.

        Deduplicated by id: an annotation carrying more than one selected label
        is listed under each of them, and an export must see it once.
        """
        unique = {}
        for label in self._applied_labels:
            for annotations in self._label_paths.get(label, {}).values():
                for annotation in annotations:
                    unique[annotation.id] = annotation
        return list(unique.values())

    def materialize_selection(self):
        """Build the concrete annotation lists an export needs.

        The table runs off the index, which counts without ever holding the
        split lists, so they are built here - once, on the way into an export -
        instead of on every checkbox click.
        """
        if not self.uses_annotation_index:
            return

        if self._index_dirty:
            self.rebuild_annotation_index()
            self._apply_label_selection(self.selected_labels)
        if self._split_dirty:
            self.split_data_from_index()
            self._split_dirty = False

        self.selected_annotations = self._selected_annotations_from_index()
        self.determine_splits()

    def update_image_selection(self):
        """
        Update the table based on the selected image option.
        """
        self.update_extracted_note()
        # The index is keyed on the selected sources, so it has to go. The old
        # body also filtered the annotations here and again inside the update.
        self.refresh_structure()

    def update_summary_statistics(self):
        """
        Update the summary statistics for the dataset creation.

        Everything expensive lives in the annotation index. A label checkbox
        folds its own paths into the running totals and re-sums the split
        columns; it never touches an annotation object.
        """
        if self.updating_summary_statistics:
            return

        with busy_cursor():
            self.updating_summary_statistics = True
            timings = self._begin_timings()
            try:
                if self._index_dirty:
                    self.rebuild_annotation_index()
                timings.mark("rebuild index")

                # Selected labels based on user's selection
                self.selected_labels = [label for _, label, checkbox in self._label_rows
                                        if checkbox.isChecked()]
                timings.mark("read checkboxes")

                # Fold the change into the running totals, then re-split only
                # if the change moved which samples there are to split
                crossed = self._apply_label_selection(self.selected_labels)
                if self._split_dirty or self._selection_changes_splits(*crossed):
                    self.split_data_from_index()
                    self._split_dirty = False
                timings.mark("selection and split")

                self.compute_split_label_counts()
                timings.mark("split counts")
                train_counts, val_counts, test_counts = self._split_label_counts

                unlabeled_video_export = self.allows_unlabeled_video_export() and self.include_negatives_radio.isChecked()

                red = QColor(255, 220, 220)
                green = QColor(220, 255, 220)

                # Whole-split emptiness, read once instead of once per row
                train_empty = self.train_ratio > 0 and len(self.train_images) == 0
                val_empty = self.val_ratio > 0 and len(self.val_images) == 0
                test_empty = self.test_ratio > 0 and len(self.test_images) == 0

                # Update the label counts table
                self.label_counts_table.setUpdatesEnabled(False)
                for row, label, include_checkbox in self._label_rows:
                    checked = include_checkbox.isChecked()
                    # An unchecked label reads zero, the same as when its
                    # annotations were filtered out of the old selection pass
                    anno_count = self._label_totals.get(label, 0) if checked else 0
                    image_count = len(self._label_paths.get(label, ())) if checked else 0
                    train_count = train_counts.get(label, 0) if checked else 0
                    val_count = val_counts.get(label, 0) if checked else 0
                    test_count = test_counts.get(label, 0) if checked else 0

                    self.label_counts_table.item(row, 2).setText(str(anno_count))
                    self.label_counts_table.item(row, 3).setText(str(train_count))
                    self.label_counts_table.item(row, 4).setText(str(val_count))
                    self.label_counts_table.item(row, 5).setText(str(test_count))
                    self.label_counts_table.item(row, 6).setText(str(image_count))

                    if checked:
                        if unlabeled_video_export:
                            self.set_cell_color(row, 3, red if train_empty else green)
                            self.set_cell_color(row, 4, red if val_empty else green)
                            self.set_cell_color(row, 5, red if test_empty else green)
                        else:
                            self.set_cell_color(row, 3, red if train_count == 0 and self.train_ratio > 0 else green)
                            self.set_cell_color(row, 4, red if val_count == 0 and self.val_ratio > 0 else green)
                            self.set_cell_color(row, 5, red if test_count == 0 and self.test_ratio > 0 else green)
                    else:
                        self.set_cell_color(row, 3, green)
                        self.set_cell_color(row, 4, green)
                        self.set_cell_color(row, 5, green)
                self.label_counts_table.setUpdatesEnabled(True)
                timings.mark("table")

                self.ready_status = self.check_label_distribution()
                self.split_status = abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-9
                self.ready_label.setText("✅ Ready" if (self.ready_status and self.split_status) else "❌ Not Ready")

                # Get counts directly from the image split lists
                train_count = len(self.train_images)
                val_count = len(self.val_images)
                test_count = len(self.test_images)
                total_count = train_count + val_count + test_count

                # Update the new labels
                self.total_images_label.setText(f"Total Images: {total_count}")
                self.split_summary_label.setText(f"(Train: {train_count}, Val: {val_count}, Test: {test_count})")
                timings.mark("status")
            finally:
                # An exception here must not leave the guard set - every later
                # refresh would return at the top without updating anything.
                self.updating_summary_statistics = False
                self._report_timings(timings)

    def is_ready(self):
        """Check if the dataset is ready to be created."""
        # Extract the input values, store them in the class variables
        self.dataset_name = self.dataset_name_edit.text()
        self.output_dir = self.output_dir_edit.text()
        self.train_ratio = self.train_ratio_spinbox.value()
        self.val_ratio = self.val_ratio_spinbox.value()
        self.test_ratio = self.test_ratio_spinbox.value()
        
        # Check that all fields are filled
        if not self.dataset_name or not self.output_dir:
            QMessageBox.warning(self,
                                "Input Error",
                                "All fields must be filled.")
            return False
        
        # Check that the ratios sum to 1.0
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-9:
            QMessageBox.warning(self,
                                "Input Error",
                                "Train, Validation, and Test ratios must sum to 1.0")
            return False

        # The table runs off the index, which counts without holding the split
        # lists, so build them here - once, on the way into an export.
        self.materialize_selection()

        if not self.ready_status:
            reply = QMessageBox.question(
                self,
                "Dataset Not Ready",
                "Not all selected labels are present in all sets.\n"
                "Are you sure you want to proceed?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                return True
            else:
                return False

        return True

    def accept(self):
        """
        Handle the OK button click event to create the dataset.
        """
        if not self.is_ready():
            return

        # Create the output folder
        output_dir_path = os.path.join(self.output_dir, self.dataset_name)

        # Check if the output directory exists. Ask before the busy cursor goes
        # up, so the prompt is not sitting under an hourglass.
        merge_existing = os.path.exists(output_dir_path)
        if merge_existing:
            reply = QMessageBox.question(self,
                                         "Directory Exists",
                                         "The output directory already exists. Do you want to merge the datasets?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return

        train_requested = False

        # create_dataset() runs on the GUI thread and pumps events through its
        # progress bars, so the project stays locked for the whole write -
        # otherwise annotations can be edited or deleted out from under it.
        with busy_cursor(), locked_window(self.main_window):
            if merge_existing:
                # Read the existing class_mapping.json file if it exists
                class_mapping_path = os.path.join(output_dir_path, "class_mapping.json")
                if os.path.exists(class_mapping_path):
                    with open(class_mapping_path, 'r') as json_file:
                        existing_class_mapping = json.load(json_file)
                else:
                    existing_class_mapping = {}

                # Merge the new class mappings with the existing ones
                new_class_mapping = self.get_class_mapping()
                merged_class_mapping = self.merge_class_mappings(existing_class_mapping, new_class_mapping)
                self.save_class_mapping_json(merged_class_mapping, output_dir_path)
            else:
                # Save the class mapping JSON file
                os.makedirs(output_dir_path, exist_ok=True)
                class_mapping = self.get_class_mapping()
                self.save_class_mapping_json(class_mapping, output_dir_path)

            try:
                # Create the dataset
                self.create_dataset(output_dir_path)

                train_requested = prompt_train_model(self,
                                                     "Dataset Created",
                                                     "Dataset has been successfully created.")

            except Exception as e:
                QMessageBox.critical(self, "Failed to Create Dataset", f"{e}")

            finally:
                super().accept()

        # Hand off to the Train Model dialog once this dialog has closed
        if train_requested:
            open_train_model_dialog_later(self.main_window, self.task, output_dir_path)

    def create_dataset(self, output_dir_path):
        raise NotImplementedError("Method must be implemented in the subclass.")

    def process_annotations(self, annotations, split_dir, split):
        raise NotImplementedError("Method must be implemented in the subclass.")
