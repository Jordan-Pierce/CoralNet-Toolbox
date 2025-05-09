import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import shutil

from yolo_tiler import YoloTiler, TileConfig, TileProgress

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox, QVBoxLayout, QLabel, QDialog,
                             QDialogButtonBox, QGroupBox, QFormLayout, QLineEdit,
                             QDoubleSpinBox, QComboBox, QPushButton, QFileDialog, QSpinBox,
                             QHBoxLayout, QWidget)

from coralnet_toolbox.Common.QtTileSizeInput import TileSizeInput
from coralnet_toolbox.Common.QtOverlapInput import OverlapInput
from coralnet_toolbox.Common.QtMarginInput import MarginInput

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Base(QDialog):
    """
    Base class for tiling object detection, and instance segmentation datasets using yolo-tiling.

    :param main_window: MainWindow object
    :param parent: Parent widget
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Tile Dataset")
        self.resize(600, 550)

        # Main vertical layout
        main_layout = QVBoxLayout(self)

        # Create group boxes
        self.info_group = QGroupBox()
        self.dataset_group = QGroupBox()
        self.tile_config_group = QGroupBox()
        self.dataset_config_group = QGroupBox()

        # Setup layouts
        self.setup_info_layout()
        self.setup_dataset_layout()
        self.setup_tile_config_layout()
        self.setup_dataset_config_layout()

        # Add info group at the top
        main_layout.addWidget(self.info_group)
        
        # Add dataset group below info
        main_layout.addWidget(self.dataset_group)
        
        # Create bottom row with tile config and dataset config side by side
        bottom_row_layout = QHBoxLayout()
        bottom_row_layout.addWidget(self.tile_config_group)
        bottom_row_layout.addWidget(self.dataset_config_group)
        main_layout.addLayout(bottom_row_layout)

        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.apply)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

    def setup_info_layout(self):
        """Set up the info layout."""
        self.info_group.setTitle("Information")
        layout = QVBoxLayout()

        info_label = QLabel("Tile an existing YOLO dataset into smaller non / overlapping images.")
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self.info_group.setLayout(layout)

    def setup_tile_config_layout(self):
        """Set up tile config layout."""
        self.tile_config_group.setTitle("Tile Configuration Parameters")
        layout = QFormLayout()

        self.tile_size_input = TileSizeInput()
        layout.addRow(self.tile_size_input)

        self.overlap_input = OverlapInput()
        layout.addRow(self.overlap_input)

        self.margins_input = MarginInput()
        layout.addRow(self.margins_input)

        self.tile_config_group.setLayout(layout)

    def setup_dataset_layout(self):
        """Set up dataset layout."""
        self.dataset_group.setTitle("Dataset Parameters")
        layout = QFormLayout()

        # Source Directory
        self.src_edit = QLineEdit()
        self.src_button = QPushButton("Browse...")
        self.src_button.clicked.connect(self.browse_src_dir)
        src_layout = QHBoxLayout()
        src_layout.addWidget(self.src_edit)
        src_layout.addWidget(self.src_button)
        layout.addRow("Source Directory:", src_layout)

        # Name of Destination Dataset
        self.dst_name_edit = QLineEdit()
        layout.addRow("Destination Dataset Name:", self.dst_name_edit)

        # Destination Directory
        self.dst_edit = QLineEdit()
        self.dst_button = QPushButton("Browse...")
        self.dst_button.clicked.connect(self.browse_dst_dir)
        dst_layout = QHBoxLayout()
        dst_layout.addWidget(self.dst_edit)
        dst_layout.addWidget(self.dst_button)
        layout.addRow("Destination Directory:", dst_layout)

        self.dataset_group.setLayout(layout)

    def setup_dataset_config_layout(self):
        """Set up dataset config layout."""
        self.dataset_config_group.setTitle("Dataset Configuration Parameters")
        layout = QFormLayout()

        # Image Extensions 
        ext_group = QGroupBox("Image Extensions")
        ext_form = QFormLayout(ext_group)
        
        self.input_ext_combo = QComboBox()
        self.input_ext_combo.addItems([".png", ".tif", ".jpeg", ".jpg"])
        self.input_ext_combo.setEditable(True)
        
        self.output_ext_combo = QComboBox()
        self.output_ext_combo.addItems([".png", ".tif", ".jpeg", ".jpg"])
        self.output_ext_combo.setEditable(True)
        
        # Add compression spinbox
        self.compression_spinbox = QSpinBox()
        self.compression_spinbox.setRange(0, 100)
        self.compression_spinbox.setValue(90)
        self.compression_spinbox.setSingleStep(5)
        
        ext_form.addRow("Input Extension:", self.input_ext_combo)
        ext_form.addRow("Output Extension:", self.output_ext_combo)
        ext_form.addRow("Compression (0-100):", self.compression_spinbox)
        
        layout.addRow(ext_group)

        # Train, Validation, and Test Ratios 
        ratios_group = QGroupBox("Dataset Split Ratios")
        ratios_form = QFormLayout(ratios_group)

        self.train_ratio_spinbox = QDoubleSpinBox()
        self.train_ratio_spinbox.setRange(0.0, 1.0)
        self.train_ratio_spinbox.setSingleStep(0.1)
        self.train_ratio_spinbox.setValue(0.7)

        self.valid_ratio_spinbox = QDoubleSpinBox()
        self.valid_ratio_spinbox.setRange(0.0, 1.0)
        self.valid_ratio_spinbox.setSingleStep(0.1)
        self.valid_ratio_spinbox.setValue(0.2)

        self.test_ratio_spinbox = QDoubleSpinBox()
        self.test_ratio_spinbox.setRange(0.0, 1.0)
        self.test_ratio_spinbox.setSingleStep(0.1)
        self.test_ratio_spinbox.setValue(0.1)

        ratios_form.addRow("Train Ratio:", self.train_ratio_spinbox)
        ratios_form.addRow("Validation Ratio:", self.valid_ratio_spinbox)
        ratios_form.addRow("Test Ratio:", self.test_ratio_spinbox)

        layout.addRow(ratios_group)

        # Advanced options 
        advanced_group = QGroupBox("Advanced Parameters")
        advanced_form = QFormLayout(advanced_group)
        
        self.densify_factor_spinbox = QDoubleSpinBox()
        self.densify_factor_spinbox.setRange(0.0, 1.0)
        self.densify_factor_spinbox.setSingleStep(0.1)
        self.densify_factor_spinbox.setValue(0.5)
        
        self.smoothing_tolerance_spinbox = QDoubleSpinBox()
        self.smoothing_tolerance_spinbox.setRange(0.0, 1.0)
        self.smoothing_tolerance_spinbox.setSingleStep(0.1)
        self.smoothing_tolerance_spinbox.setValue(0.1)
        
        advanced_form.addRow("Densify Factor:", self.densify_factor_spinbox)
        advanced_form.addRow("Smoothing Tolerance:", self.smoothing_tolerance_spinbox)
        
        layout.addRow(advanced_group)
        
        # Misc. options
        misc_group = QGroupBox("Misc.")
        misc_form = QFormLayout(misc_group)
        
        # Include negative samples
        self.include_negatives_combo = QComboBox()
        self.include_negatives_combo.addItems(["True", "False"])
        self.include_negatives_combo.setEditable(False)
        self.include_negatives_combo.setCurrentIndex(0)
        misc_form.addRow("Include Negative Samples:", self.include_negatives_combo)
        
        # Copy source data
        self.copy_source_data_combo = QComboBox()
        self.copy_source_data_combo.addItems(["True", "False"])
        self.copy_source_data_combo.setEditable(False)
        self.copy_source_data_combo.setCurrentIndex(1)
        misc_form.addRow("Copy Source Data:", self.copy_source_data_combo)
        
        # Number of Visualization Samples
        self.num_viz_sample_spinbox = QSpinBox()
        self.num_viz_sample_spinbox.setRange(1, 1000)
        self.num_viz_sample_spinbox.setValue(3)
        misc_form.addRow("# Visualization Samples:", self.num_viz_sample_spinbox)
        
        layout.addRow(misc_group)

        self.dataset_config_group.setLayout(layout)

    def setup_buttons_layout(self, layout):
        """Set up buttons layout."""
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.apply)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def browse_src_dir(self):
        """
        Browse and select a source directory.
        """
        dir_path = QFileDialog.getExistingDirectory(self, "Select Source Directory")
        if dir_path:
            self.src_edit.setText(dir_path)

    def browse_dst_dir(self):
        """
        Browse and select a destination directory.
        """
        dir_path = QFileDialog.getExistingDirectory(self, "Select Destination Directory")
        if dir_path:
            self.dst_edit.setText(dir_path)

    def validate_source_directory(self, src):
        """
        Validate the source directory to ensure it contains a 'train' sub-folder.

        :param src: Source directory path
        :return: True if valid, False otherwise
        """
        if not os.path.isdir(os.path.join(src, 'train')):
            QMessageBox.warning(self,
                                "Invalid Source Directory",
                                "The source directory must contain a 'train' sub-folder.")
            return False
        return True

    def validate_ext(self, ext):
        """
        Validate the ext parameter to ensure it is a string and starts with a dot.

        :param ext: Image file extension
        :return: True if valid, False otherwise
        """
        if not isinstance(ext, str) or not ext.startswith('.'):
            QMessageBox.warning(self,
                                "Invalid Image Extension",
                                "The image extension must be a string starting with a dot.")
            return False
        return True

    def validate_densify_factor(self, densify_factor):
        """
        Validate the densify_factor parameter to ensure it is a float between 0 and 1.

        :param densify_factor: Densify factor
        :return: True if valid, False otherwise
        """
        if not isinstance(densify_factor, float) or not (0.0 <= densify_factor <= 1.0):
            QMessageBox.warning(self,
                                "Invalid Densify Factor",
                                "The densify factor must be a float between 0 and 1.")
            return False
        return True

    def validate_smoothing_tolerance(self, smoothing_tolerance):
        """
        Validate the smoothing_tolerance parameter to ensure it is a float between 0 and 1.

        :param smoothing_tolerance: Smoothing tolerance
        :return: True if valid, False otherwise
        """
        if not isinstance(smoothing_tolerance, float) or not (0.0 <= smoothing_tolerance <= 1.0):
            QMessageBox.warning(self,
                                "Invalid Smoothing Tolerance",
                                "The smoothing tolerance must be a float between 0 and 1.")
            return False
        return True

    def validate_ratios(self, train_ratio, valid_ratio, test_ratio):
        """
        Validate the train_ratio, valid_ratio, and test_ratio parameters to ensure they are
        floats between 0 and 1 and sum to 1.

        :param train_ratio: Train ratio
        :param valid_ratio: Validation ratio
        :param test_ratio: Test ratio
        :return: True if valid, False otherwise
        """
        ratios = [train_ratio, valid_ratio, test_ratio]
        if not all(isinstance(ratio, float) and 0.0 <= ratio <= 1.0 for ratio in ratios):
            QMessageBox.warning(self,
                                "Invalid Ratios",
                                "The train, validation, and test ratios must be floats between 0 and 1.")
            return False
        if not abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-9:
            QMessageBox.warning(self,
                                "Invalid Ratios",
                                "The train, validation, and test ratios must sum to 1.")
            return False
        return True

    def copy_class_mapping(self):
        """Checks to see if a class_mapping.json file exists in the source directory
        and copies it to the destination directory."""
        src = self.src_edit.text()
        dst = os.path.join(self.dst_edit.text(), self.dst_name_edit.text())

        src_class_mapping = os.path.join(src, 'class_mapping.json')
        dst_class_mapping = os.path.join(dst, 'class_mapping.json')

        if os.path.exists(src_class_mapping):
            shutil.copy(src_class_mapping, dst_class_mapping)

    def apply(self):
        """
        Apply the tile dataset options.
        """
        # Pause the cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # Tile the dataset
            self.tile_dataset()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to tile dataset: {str(e)}")
        finally:
            # Resume the cursor
            QApplication.restoreOverrideCursor()

        self.accept()

    def tile_dataset(self):
        """
        Use yolo-tiling to tile the dataset.
        """
        src = self.src_edit.text()
        dst = os.path.join(self.dst_edit.text(), self.dst_name_edit.text())

        # Extract values
        margins = self.margins_input.get_margins(validate=False)
        slice_wh = self.tile_size_input.get_sizes(validate=False)
        overlap_wh = self.overlap_input.get_overlap(validate=False)

        input_ext = self.input_ext_combo.currentText()
        output_ext = self.output_ext_combo.currentText()
        compression = self.compression_spinbox.value()
        densify_factor = self.densify_factor_spinbox.value()
        smoothing_tolerance = self.smoothing_tolerance_spinbox.value()
        train_ratio = self.train_ratio_spinbox.value()
        valid_ratio = self.valid_ratio_spinbox.value()
        test_ratio = self.test_ratio_spinbox.value()
        include_negatives = self.include_negatives_combo.currentText()
        include_negatives = True if include_negatives == "True" else False
        copy_source_data = self.copy_source_data_combo.currentText()
        copy_source_data = True if copy_source_data == "True" else False
        num_viz_samples = self.num_viz_sample_spinbox.value()

        # Perform all validation checks
        validation_checks = [
            (self.validate_source_directory(src), "Source directory validation failed"),
            (self.validate_ext(input_ext), "Input extension validation failed"),
            (self.validate_ext(output_ext), "Output extension validation failed"),
            (self.validate_densify_factor(densify_factor), "Densify factor validation failed"),
            (self.validate_smoothing_tolerance(smoothing_tolerance), "Smoothing tolerance validation failed"),
            (self.validate_ratios(train_ratio, valid_ratio, test_ratio), "Dataset split ratios validation failed"),
        ]

        # Check if any validation failed
        for is_valid, error_msg in validation_checks:
            if not is_valid:
                return

        # Pause the cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Create and show the progress bar
        self.progress_bar = ProgressBar(self, title="Tiling Progress")
        self.progress_bar.show()

        def progress_callback(progress: TileProgress):
            title = f"Processing {progress.current_set_name.capitalize()} Set"

            if progress.total_tiles:
                progress_percentage = int((progress.current_tile_idx / progress.total_tiles) * 100)
                title += f": {int(progress.current_image_idx/progress.total_images*100)}%"
            else:
                progress_percentage = int((progress.current_image_idx / progress.total_images) * 100)

            self.progress_bar.setWindowTitle(title)
            self.progress_bar.set_value(progress_percentage)
            self.progress_bar.update_progress()

            if self.progress_bar.wasCanceled():
                raise Exception("Tiling process was canceled by the user.")

        config = TileConfig(
            slice_wh=slice_wh,
            overlap_wh=overlap_wh,
            input_ext=input_ext,
            output_ext=output_ext,
            annotation_type=self.annotation_type,
            densify_factor=densify_factor,
            smoothing_tolerance=smoothing_tolerance,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            margins=margins,
            include_negative_samples=include_negatives,
            copy_source_data=copy_source_data,
            compression=compression,
        )

        tiler = YoloTiler(
            source=src,
            target=dst,
            config=config,
            progress_callback=progress_callback,
            num_viz_samples=num_viz_samples,        # Number of samples to visualize
        )

        # Copy the class_mapping.json file if it exists
        self.copy_class_mapping()

        try:
            tiler.run()
            QMessageBox.information(self,
                                    "Tiling Complete",
                                    "The dataset has been tiled successfully.")

        except Exception as e:
            QMessageBox.critical(self,
                                 "Error",
                                 f"Failed to tile dataset: {str(e)}")
        finally:
            self.progress_bar.stop_progress()
            self.progress_bar.close()
            QApplication.restoreOverrideCursor()
