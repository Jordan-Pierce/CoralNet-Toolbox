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
                             QHBoxLayout)

from coralnet_toolbox.Tile.QtCommon import TileSizeInput, OverlapInput, MarginInput

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
        self.resize(600, 100)

        # Object Detection / Instance Segmentation
        self.annotation_type = None

        self.layout = QVBoxLayout(self)

        # Setup the info layout
        self.setup_info_layout()
        # Setup the dataset layout
        self.setup_dataset_layout()
        # Setup the tile config layout
        self.setup_tile_config_layout()
        # Setup the dataset config layout
        self.setup_dataset_config_layout()
        # Setup the buttons layout
        self.setup_buttons_layout()

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Tile an existing YOLO dataset into smaller non / overlapping images.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_dataset_layout(self):
        """
        Set up the dataset layout.
        """
        group_box = QGroupBox("Dataset Parameters")
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

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_tile_config_layout(self):
        """
        Set up the tile config parameters layout.
        """
        group_box = QGroupBox("Tile Configuration Parameters")
        layout = QFormLayout()

        # Tile Size
        self.tile_size_input = TileSizeInput()
        layout.addRow(self.tile_size_input)

        # Overlap
        self.overlap_input = OverlapInput()
        layout.addRow(self.overlap_input)

        # Margins
        self.margins_input = MarginInput()
        layout.addRow(self.margins_input)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
    def setup_dataset_config_layout(self):
        """
        Set up the dataset configuration parameters layout.
        """
        group_box = QGroupBox("Dataset Configuration Parameters")
        layout = QFormLayout()

        # Image Extensions
        ext_layout = QHBoxLayout()
        self.input_ext_combo = QComboBox()
        self.input_ext_combo.addItems([".png", ".tif", ".jpeg", ".jpg"])
        self.input_ext_combo.setEditable(True)
        self.output_ext_combo = QComboBox()
        self.output_ext_combo.addItems([".png", ".tif", ".jpeg", ".jpg"])
        self.output_ext_combo.setEditable(True)
        ext_layout.addWidget(QLabel("Input Ext:"))
        ext_layout.addWidget(self.input_ext_combo)
        ext_layout.addWidget(QLabel("Output Ext:"))
        ext_layout.addWidget(self.output_ext_combo)
        layout.addRow("Image Extensions:", ext_layout)

        # Include negative samples
        self.include_negatives_combo = QComboBox()
        self.include_negatives_combo.addItems(["True", "False"])
        self.include_negatives_combo.setEditable(False)
        self.include_negatives_combo.setCurrentIndex(0)
        layout.addRow("Include Negative Samples:", self.include_negatives_combo)

        # Train, Validation, and Test Ratios
        ratios_layout = QHBoxLayout()

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

        ratios_layout.addWidget(QLabel("Train:"))
        ratios_layout.addWidget(self.train_ratio_spinbox)
        ratios_layout.addWidget(QLabel("Valid:"))
        ratios_layout.addWidget(self.valid_ratio_spinbox)
        ratios_layout.addWidget(QLabel("Test:"))
        ratios_layout.addWidget(self.test_ratio_spinbox)

        layout.addRow("Dataset Split Ratios:", ratios_layout)

        # Number of Visualization Samples
        self.num_viz_sample_spinbox = QSpinBox()
        self.num_viz_sample_spinbox.setRange(1, 1000)
        self.num_viz_sample_spinbox.setValue(25)
        layout.addRow("# Visualization Samples:", self.num_viz_sample_spinbox)

        # Advanced options (densify factor and smoothing tolerance)
        advanced_layout = QHBoxLayout()
        self.densify_factor_spinbox = QDoubleSpinBox()
        self.densify_factor_spinbox.setRange(0.0, 1.0)
        self.densify_factor_spinbox.setSingleStep(0.1)
        self.densify_factor_spinbox.setValue(0.5)
        self.smoothing_tolerance_spinbox = QDoubleSpinBox()
        self.smoothing_tolerance_spinbox.setRange(0.0, 1.0)
        self.smoothing_tolerance_spinbox.setSingleStep(0.1)
        self.smoothing_tolerance_spinbox.setValue(0.1)
        advanced_layout.addWidget(QLabel("Densify:"))
        advanced_layout.addWidget(self.densify_factor_spinbox)
        advanced_layout.addWidget(QLabel("Smoothing:"))
        advanced_layout.addWidget(self.smoothing_tolerance_spinbox)
        layout.addRow("Advanced Parameters:", advanced_layout)
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_buttons_layout(self):
        """
        Set up the layout with buttons.
        """
        # Create a button box for the buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.apply)
        button_box.rejected.connect(self.reject)

        self.layout.addWidget(button_box)

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

    def validate_slice_wh(self, slice_wh):
        """
        Validate the slice_wh parameter to ensure it is a tuple of two integers.

        :param slice_wh: Slice width and height
        :return: True if valid, False otherwise
        """
        if not isinstance(slice_wh, tuple) or len(slice_wh) != 2 or not all(isinstance(i, int) for i in slice_wh):
            QMessageBox.warning(self, "Invalid Tile Size", "The tile size must be a tuple of two integers.")
            return False
        return True

    def validate_overlap_wh(self, overlap_wh):
        """
        Validate the overlap_wh parameter to ensure it is a tuple of two floats.

        :param overlap_wh: Overlap width and height
        :return: True if valid, False otherwise
        """
        correct_type = all(isinstance(i, (int, float)) for i in overlap_wh)
        if not isinstance(overlap_wh, tuple) or len(overlap_wh) != 2 or not correct_type:
            QMessageBox.warning(self, "Invalid Overlap", "The overlap must be a tuple of two floats.")
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

    def validate_margins(self, margins):
        """
        Validate the margins parameter to ensure it is a valid type and value.

        :param margins: Margins
        :return: True if valid, False otherwise
        """
        if isinstance(margins, (int, float)):
            if isinstance(margins, float) and not (0.0 <= margins <= 1.0):
                QMessageBox.warning(self,
                                    "Invalid Margins",
                                    "The margin percentage must be between 0 and 1.")
                return False
            return True
        elif isinstance(margins, tuple) and len(margins) == 4:
            if all(isinstance(i, (int, float)) for i in margins):
                if all(isinstance(i, float) for i in margins) and not all(0.0 <= i <= 1.0 for i in margins):
                    QMessageBox.warning(self,
                                        "Invalid Margins",
                                        "All margin percentages must be between 0 and 1.")
                    return False
                return True
        QMessageBox.warning(self,
                            "Invalid Margins",
                            "The margins must be a single integer, float, or a tuple of four integers/floats.")
        return False

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
        margins = self.margins_input.get_value()
        slice_wh = self.tile_size_input.get_value()
        overlap_wh = self.overlap_input.get_value(slice_wh[0], slice_wh[1])
        
        input_ext = self.input_ext_combo.currentText()
        output_ext = self.output_ext_combo.currentText()
        densify_factor = self.densify_factor_spinbox.value()
        smoothing_tolerance = self.smoothing_tolerance_spinbox.value()
        train_ratio = self.train_ratio_spinbox.value()
        valid_ratio = self.valid_ratio_spinbox.value()
        test_ratio = self.test_ratio_spinbox.value()
        include_negatives = self.include_negatives_combo.currentText()
        include_negatives = True if include_negatives == "True" else False
        num_viz_samples = self.num_viz_sample_spinbox.value()

        # Perform all validation checks
        validation_checks = [
            (self.validate_source_directory(src), "Source directory validation failed"),
            (self.validate_slice_wh(slice_wh), "Slice width/height validation failed"),
            (self.validate_overlap_wh(overlap_wh), "Overlap width/height validation failed"),
            (self.validate_ext(input_ext), "Input extension validation failed"),
            (self.validate_ext(output_ext), "Output extension validation failed"),
            (self.validate_densify_factor(densify_factor), "Densify factor validation failed"),
            (self.validate_smoothing_tolerance(smoothing_tolerance), "Smoothing tolerance validation failed"),
            (self.validate_ratios(train_ratio, valid_ratio, test_ratio), "Dataset split ratios validation failed"),
            (self.validate_margins(margins), "Margins validation failed")
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
            include_negative_samples=include_negatives
        )

        tiler = YoloTiler(
            source=src,
            target=dst,
            config=config,
            progress_callback=progress_callback,
            num_viz_samples=num_viz_samples,  # Number of samples to visualize
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
