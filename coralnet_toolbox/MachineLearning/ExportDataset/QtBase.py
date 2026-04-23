import warnings

import os
import random
from collections import Counter
import ujson as json

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QCheckBox,
                             QVBoxLayout, QLabel, QLineEdit, QDialog, QHBoxLayout,
                             QPushButton, QFormLayout, QDialogButtonBox, QDoubleSpinBox,
                             QGroupBox, QTableWidget, QTableWidgetItem, QButtonGroup, QRadioButton,
                             QSpinBox,
                             QWidget)

from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon, get_window_icon
from coralnet_toolbox.MachineLearning.ExportDataset.export_dataset_utils import (
    build_export_sample_paths,
    frame_matches_stride,
    group_annotations_by_source,
    normalize_source_path,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Base(QDialog):
    supports_unlabeled_video_frames = False

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

        self.resize(800, 800)
        self.setWindowTitle("Export Dataset")
        self.setWindowIcon(get_window_icon("coralnet.svg"))

        self.selected_labels = []
        self.selected_annotations = []
        self.updating_summary_statistics = False

        self.output_dir = None
        self.dataset_name = None
        self.train_ratio = 0.7
        self.val_ratio = 0.2
        self.test_ratio = 0.1

        self.layout = QVBoxLayout(self)

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

    def showEvent(self, event):
        """
        Handle the show event to update annotation type checkboxes, populate class filter list,
        and update summary statistics.

        Args:
            event: The show event.
        """
        super().showEvent(event)
        self.update_annotation_type_checkboxes()
        self.update_video_options()
        self.populate_class_filter_list()
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
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(self.output_dir_button)
        layout.addRow("Output Directory:", output_layout)
        
        # Dataset Name
        self.dataset_name_edit = QLineEdit()
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

        self.val_ratio_spinbox = QDoubleSpinBox()
        self.val_ratio_spinbox.setRange(0.0, 1.0)
        self.val_ratio_spinbox.setSingleStep(0.1)
        self.val_ratio_spinbox.setValue(0.2)

        self.test_ratio_spinbox = QDoubleSpinBox()
        self.test_ratio_spinbox.setRange(0.0, 1.0)
        self.test_ratio_spinbox.setSingleStep(0.1)
        self.test_ratio_spinbox.setValue(0.1)

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
        self.include_rectangles_checkbox = QCheckBox("Include Rectangle Annotations")
        self.include_polygons_checkbox = QCheckBox("Include Polygon Annotations")

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
        self.filtered_images_radio = QRadioButton("Filtered Images")

        self.image_options_group.addButton(self.all_images_radio)
        self.image_options_group.addButton(self.filtered_images_radio)
        self.image_options_group.setExclusive(True)

        self.all_images_radio.setChecked(True)

        self.all_images_radio.toggled.connect(self.update_image_selection)
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
        self.exclude_negatives_radio = QRadioButton("Exclude Negatives")

        self.negative_samples_group.addButton(self.include_negatives_radio)
        self.negative_samples_group.addButton(self.exclude_negatives_radio)
        self.negative_samples_group.setExclusive(True)

        self.exclude_negatives_radio.setChecked(True)

        # Connect to update stats when changed. Only one needed for the group.
        self.include_negatives_radio.toggled.connect(self.update_summary_statistics)

        layout.addWidget(self.include_negatives_radio)
        layout.addWidget(self.exclude_negatives_radio)

        group_box.setLayout(layout)
        return group_box

    def create_video_options_layout(self):
        """Create the optional video export controls."""
        group_box = QGroupBox("Video Frames")
        layout = QHBoxLayout()

        self.split_by_source_checkbox = QCheckBox("Split by source video")
        self.split_by_source_checkbox.setChecked(True)
        self.split_by_source_checkbox.setToolTip(
            "Keep frames from the same source video together when splitting train, val, and test."
        )
        self.split_by_source_checkbox.stateChanged.connect(self.update_summary_statistics)
        layout.addWidget(self.split_by_source_checkbox)

        stride_label = QLabel("Frame stride:")
        stride_label.setToolTip("Only export every Nth frame from a video source.")
        layout.addWidget(stride_label)

        self.frame_stride_spinbox = QSpinBox()
        self.frame_stride_spinbox.setRange(1, 999999)
        self.frame_stride_spinbox.setValue(1)
        self.frame_stride_spinbox.setToolTip("Only export every Nth frame from a video source.")
        self.frame_stride_spinbox.valueChanged.connect(self.update_summary_statistics)
        layout.addWidget(self.frame_stride_spinbox)

        self.export_unlabeled_video_frames_checkbox = QCheckBox("Export unlabeled video frames")
        self.export_unlabeled_video_frames_checkbox.setChecked(False)
        self.export_unlabeled_video_frames_checkbox.setToolTip(
            "Also export video frames without annotations. Disabled for classification exports."
        )
        self.export_unlabeled_video_frames_checkbox.stateChanged.connect(self.update_summary_statistics)
        layout.addWidget(self.export_unlabeled_video_frames_checkbox)

        layout.addStretch()

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
        """Return the currently selected image paths from the project or the filtered table."""
        if self.filtered_images_radio.isChecked():
            return list(self.image_window.table_model.filtered_paths)
        return list(self.image_window.raster_manager.image_paths)

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
        self.label_counts_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        # Note: No delegate needed - using widget-based checkboxes via setCellWidget
        layout.addWidget(self.label_counts_table)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_status_layout(self):
        """Setup the ready status layout."""
        group_box = QGroupBox("Status")
        layout = QHBoxLayout()

        # Label for Ready Status
        self.ready_label = QLabel("❌ Not Ready")
        layout.addWidget(self.ready_label)

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
        self.layout.addWidget(group_box)

    def setup_button_layout(self):
        """Setup the button layout."""
        button_layout = QHBoxLayout()

        # Add Refresh button
        self.refresh_button = QPushButton("Refresh | Shuffle")
        self.refresh_button.setToolTip("Recalculate stats and re-shuffle train/val/test splits")
        self.refresh_button.clicked.connect(self.update_summary_statistics)
        button_layout.addWidget(self.refresh_button)

        # Add spacer to push OK/Cancel to right
        button_layout.addStretch()

        # Add OK and Cancel buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        button_layout.addWidget(self.buttons)

        self.layout.addLayout(button_layout)

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

    def populate_class_filter_list(self):
        """
        Populate the class filter list with labels and their counts.
        """
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        # Set the row count to 0
        self.label_counts_table.setRowCount(0)

        # Create a progress bar
        progress_bar = ProgressBar(self, "Populating Class Lists")
        progress_bar.show()
        progress_bar.start_progress(len(self.annotation_window.annotations_dict))

        label_counts = {}
        label_image_counts = {}
        # Count the occurrences of each label and unique images per label
        for annotation in self.annotation_window.annotations_dict.values():
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
        self.label_counts_table.setUpdatesEnabled(False)
        row = 0
        for label, count in sorted_label_counts:
            include_checkbox = QCheckBox()
            include_checkbox.setChecked(True)
            include_checkbox.stateChanged.connect(self.update_summary_statistics)
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addStretch()
            layout.addWidget(include_checkbox)
            layout.addStretch()

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

            row += 1
        self.label_counts_table.setUpdatesEnabled(True)
            
        # Restore the cursor to the default cursor
        QApplication.restoreOverrideCursor()
        progress_bar.finish_progress()
        progress_bar.stop_progress()
        progress_bar.close()

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
    
        train_label_counts = Counter(a.label.short_label_code for a in self.train_annotations)
        val_label_counts = Counter(a.label.short_label_code for a in self.val_annotations)
        test_label_counts = Counter(a.label.short_label_code for a in self.test_annotations)

        # Check the conditions for each split
        for label in self.selected_labels:
            if train_ratio > 0 and train_label_counts[label] == 0:
                return False
            if val_ratio > 0 and val_label_counts[label] == 0:
                return False
            if test_ratio > 0 and test_label_counts[label] == 0:
                return False
    
        # Additional checks to ensure no empty splits
        if train_ratio > 0 and len(self.train_annotations) == 0:
            return False
        if val_ratio > 0 and len(self.val_annotations) == 0:
            return False
        if test_ratio > 0 and len(self.test_annotations) == 0:
            return False
    
        return True

    def update_image_selection(self):
        """
        Update the table based on the selected image option.
        """
        self.selected_annotations = self.filter_annotations()
        self.update_summary_statistics()

    def update_summary_statistics(self):
        """
        Update the summary statistics for the dataset creation.
        """
        if self.updating_summary_statistics:
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        self.updating_summary_statistics = True

        # Selected labels based on user's selection
        self.selected_labels = []
        for row in range(self.label_counts_table.rowCount()):
            container = self.label_counts_table.cellWidget(row, 0)
            include_checkbox = container.findChild(QCheckBox)
            if include_checkbox.isChecked():
                label = self.label_counts_table.item(row, 1).text()
                self.selected_labels.append(label)

        # Filter annotations based on the selected annotation types and current tab
        self.selected_annotations = self.filter_annotations()
        
        # Split the data by images
        self.split_data()

        # Split the data by annotations
        self.determine_splits()

        # Precompute counts in a single pass each — O(n) instead of O(labels × n)
        selected_counts = Counter(a.label.short_label_code for a in self.selected_annotations)
        train_counts = Counter(a.label.short_label_code for a in self.train_annotations)
        val_counts = Counter(a.label.short_label_code for a in self.val_annotations)
        test_counts = Counter(a.label.short_label_code for a in self.test_annotations)
        unlabeled_video_export = self.allows_unlabeled_video_export() and self.include_negatives_radio.isChecked()

        red = QColor(255, 220, 220)
        green = QColor(220, 255, 220)

        # Update the label counts table
        self.label_counts_table.setUpdatesEnabled(False)
        for row in range(self.label_counts_table.rowCount()):
            container = self.label_counts_table.cellWidget(row, 0)
            include_checkbox = container.findChild(QCheckBox)
            label = self.label_counts_table.item(row, 1).text()
            anno_count = selected_counts.get(label, 0)
            if include_checkbox.isChecked():
                train_count = train_counts.get(label, 0)
                val_count = val_counts.get(label, 0)
                test_count = test_counts.get(label, 0)
            else:
                train_count = 0
                val_count = 0
                test_count = 0

            self.label_counts_table.item(row, 2).setText(str(anno_count))
            self.label_counts_table.item(row, 3).setText(str(train_count))
            self.label_counts_table.item(row, 4).setText(str(val_count))
            self.label_counts_table.item(row, 5).setText(str(test_count))

            if include_checkbox.isChecked():
                if unlabeled_video_export:
                    self.set_cell_color(row, 3, red if (self.train_ratio > 0 and len(self.train_images) == 0) else green)
                    self.set_cell_color(row, 4, red if (self.val_ratio > 0 and len(self.val_images) == 0) else green)
                    self.set_cell_color(row, 5, red if (self.test_ratio > 0 and len(self.test_images) == 0) else green)
                else:
                    self.set_cell_color(row, 3, red if train_count == 0 and self.train_ratio > 0 else green)
                    self.set_cell_color(row, 4, red if val_count == 0 and self.val_ratio > 0 else green)
                    self.set_cell_color(row, 5, red if test_count == 0 and self.test_ratio > 0 else green)
            else:
                self.set_cell_color(row, 3, green)
                self.set_cell_color(row, 4, green)
                self.set_cell_color(row, 5, green)
        self.label_counts_table.setUpdatesEnabled(True)

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
        
        self.updating_summary_statistics = False

        # Restore the cursor to the default cursor
        QApplication.restoreOverrideCursor()

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
        
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Create the output folder
        output_dir_path = os.path.join(self.output_dir, self.dataset_name)

        # Check if the output directory exists
        if os.path.exists(output_dir_path):
            reply = QMessageBox.question(self,
                                         "Directory Exists",
                                         "The output directory already exists. Do you want to merge the datasets?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                QApplication.restoreOverrideCursor()
                return

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

            QMessageBox.information(self,
                                    "Dataset Created",
                                    "Dataset has been successfully created.")
        
        except Exception as e:
            QMessageBox.critical(self, "Failed to Create Dataset", f"{e}")
            
        finally:
            # Restore the cursor to the default cursor
            QApplication.restoreOverrideCursor()
            super().accept()

    def create_dataset(self, output_dir_path):
        raise NotImplementedError("Method must be implemented in the subclass.")

    def process_annotations(self, annotations, split_dir, split):
        raise NotImplementedError("Method must be implemented in the subclass.")
