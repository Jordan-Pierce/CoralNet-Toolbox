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
                             QHBoxLayout, QWidget, QStackedWidget, QGridLayout)

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class OverlapInput(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Overlap", parent)
        layout = QFormLayout(self)

        # Unit selection
        self.value_type = QComboBox()
        self.value_type.addItems(["Pixels", "Percentage"])
        layout.addRow("Unit:", self.value_type)

        # Width and height inputs
        self.width_spin = QSpinBox()
        self.width_spin.setRange(0, 9999)
        self.width_spin.setValue(0)
        self.width_double = QDoubleSpinBox()
        self.width_double.setRange(0, 1)
        self.width_double.setValue(0)
        self.width_double.setSingleStep(0.1)
        self.width_double.setDecimals(2)
        self.width_double.hide()

        self.height_spin = QSpinBox()
        self.height_spin.setRange(0, 9999)
        self.height_spin.setValue(0)
        self.height_double = QDoubleSpinBox()
        self.height_double.setRange(0, 1)
        self.height_double.setValue(0)
        self.height_double.setSingleStep(0.1)
        self.height_double.setDecimals(2)
        self.height_double.hide()

        layout.addRow("Width:", self.width_spin)
        layout.addRow("", self.width_double)
        layout.addRow("Height:", self.height_spin)
        layout.addRow("", self.height_double)

        self.value_type.currentIndexChanged.connect(self.update_input_mode)

    def update_input_mode(self, index):
        is_percentage = index == 1
        self.width_spin.setVisible(not is_percentage)
        self.width_double.setVisible(is_percentage)
        self.height_spin.setVisible(not is_percentage)
        self.height_double.setVisible(is_percentage)

    def get_value(self):
        is_percentage = self.value_type.currentIndex() == 1
        if is_percentage:
            return self.width_double.value(), self.height_double.value()
        return self.width_spin.value(), self.height_spin.value()


class MarginInput(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Margins", parent)
        layout = QVBoxLayout(self)

        # Input type selection
        type_layout = QHBoxLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Single Value", "Multiple Values"])
        self.value_type = QComboBox()
        self.value_type.addItems(["Pixels", "Percentage"])

        type_layout.addWidget(QLabel("Type:"))
        type_layout.addWidget(self.type_combo)
        type_layout.addWidget(QLabel("Unit:"))
        type_layout.addWidget(self.value_type)
        layout.addLayout(type_layout)

        # Stacked widget for different input types
        self.stack = QStackedWidget()

        # Single value widgets
        single_widget = QWidget()
        single_layout = QHBoxLayout(single_widget)
        self.single_spin = QSpinBox()
        self.single_spin.setRange(0, 9999)
        self.single_double = QDoubleSpinBox()
        self.single_double.setRange(0, 1)
        self.single_double.setSingleStep(0.1)
        self.single_double.setDecimals(2)
        single_layout.addWidget(self.single_spin)
        single_layout.addWidget(self.single_double)
        self.single_double.hide()

        # Multiple values widgets
        multi_widget = QWidget()
        multi_layout = QGridLayout(multi_widget)
        self.margin_spins = []
        self.margin_doubles = []
        positions = [("Top", 0, 1),
                     ("Right", 1, 2),
                     ("Bottom", 2, 1),
                     ("Left", 1, 0)]

        for label, row, col in positions:
            spin = QSpinBox()
            spin.setRange(0, 9999)
            double = QDoubleSpinBox()
            double.setRange(0, 1)
            double.setSingleStep(0.1)
            double.setDecimals(2)
            double.hide()

            self.margin_spins.append(spin)
            self.margin_doubles.append(double)
            multi_layout.addWidget(QLabel(label), row, col)
            multi_layout.addWidget(spin, row + 1, col)
            multi_layout.addWidget(double, row + 1, col)

        self.stack.addWidget(single_widget)
        self.stack.addWidget(multi_widget)
        layout.addWidget(self.stack)

        # Connect signals
        self.type_combo.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.value_type.currentIndexChanged.connect(self.update_input_mode)

    def update_input_mode(self, index):
        is_percentage = index == 1
        if is_percentage:
            self.single_spin.hide()
            self.single_double.show()
            for spin, double in zip(self.margin_spins, self.margin_doubles):
                spin.hide()
                double.show()
        else:
            self.single_double.hide()
            self.single_spin.show()
            for spin, double in zip(self.margin_spins, self.margin_doubles):
                double.hide()
                spin.show()

    def get_value(self):
        is_percentage = self.value_type.currentIndex() == 1
        if self.type_combo.currentIndex() == 0:
            return self.single_double.value() if is_percentage else self.single_spin.value()
        else:
            widgets = self.margin_doubles if is_percentage else self.margin_spins
            return tuple(w.value() for w in widgets)


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
        # Setup the config layout
        self.setup_config_layout()
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
        layout.addRow("Name of Destination Dataset:", self.dst_name_edit)

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

    def setup_config_layout(self):
        """
        Set up the TileConfig parameters layout.
        """
        group_box = QGroupBox("Configuration Parameters")
        layout = QFormLayout()

        # Overlap
        self.overlap_input = OverlapInput()
        layout.addRow(self.overlap_input)

        # Margins
        self.margins_input = MarginInput()
        layout.addRow(self.margins_input)

        # Tile Size
        tile_layout = QHBoxLayout()
        self.slice_width = QSpinBox()
        self.slice_width.setRange(1, 9999)
        self.slice_width.setValue(640)
        self.slice_height = QSpinBox()
        self.slice_height.setRange(1, 9999)
        self.slice_height.setValue(480)
        tile_layout.addWidget(QLabel("Width (px):"))
        tile_layout.addWidget(self.slice_width)
        tile_layout.addWidget(QLabel("Height (px):"))
        tile_layout.addWidget(self.slice_height)
        layout.addRow("Tile Size:", tile_layout)

        # Image Extension
        self.ext_combo = QComboBox()
        self.ext_combo.addItems([".png", ".tif", ".jpeg", ".jpg"])
        self.ext_combo.setEditable(True)
        layout.addRow("Image Extension:", self.ext_combo)

        # Include negative samples
        self.include_negatives_combo = QComboBox()
        self.include_negatives_combo.addItems(["True", "False"])
        self.include_negatives_combo.setEditable(False)
        self.include_negatives_combo.setCurrentIndex(0)
        layout.addRow("Include Negative Samples:", self.include_negatives_combo)

        # Densify Factor
        self.densify_factor_spinbox = QDoubleSpinBox()
        self.densify_factor_spinbox.setRange(0.0, 1.0)
        self.densify_factor_spinbox.setSingleStep(0.1)
        self.densify_factor_spinbox.setValue(0.5)
        layout.addRow("Densify Factor:", self.densify_factor_spinbox)

        # Smoothing Tolerance
        self.smoothing_tolerance_spinbox = QDoubleSpinBox()
        self.smoothing_tolerance_spinbox.setRange(0.0, 1.0)
        self.smoothing_tolerance_spinbox.setSingleStep(0.1)
        self.smoothing_tolerance_spinbox.setValue(0.1)
        layout.addRow("Smoothing Tolerance:", self.smoothing_tolerance_spinbox)

        # Train, Validation, and Test Ratios
        self.train_ratio_spinbox = QDoubleSpinBox()
        self.train_ratio_spinbox.setRange(0.0, 1.0)
        self.train_ratio_spinbox.setSingleStep(0.1)
        self.train_ratio_spinbox.setValue(0.7)
        layout.addRow("Train Ratio:", self.train_ratio_spinbox)

        self.valid_ratio_spinbox = QDoubleSpinBox()
        self.valid_ratio_spinbox.setRange(0.0, 1.0)
        self.valid_ratio_spinbox.setSingleStep(0.1)
        self.valid_ratio_spinbox.setValue(0.2)
        layout.addRow("Validation Ratio:", self.valid_ratio_spinbox)

        self.test_ratio_spinbox = QDoubleSpinBox()
        self.test_ratio_spinbox.setRange(0.0, 1.0)
        self.test_ratio_spinbox.setSingleStep(0.1)
        self.test_ratio_spinbox.setValue(0.1)
        layout.addRow("Test Ratio:", self.test_ratio_spinbox)

        # Number of Visualization Samples
        self.num_viz_sample_spinbox = QSpinBox()
        self.num_viz_sample_spinbox.setRange(1, 100)
        self.num_viz_sample_spinbox.setValue(25)
        layout.addRow("Number of Visualization Samples:", self.num_viz_sample_spinbox)

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
        Apply the selected batch inference options.
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

        # -------------------------
        # Perform validation checks

        # Check the directory has a 'train' folder
        if not self.validate_source_directory(src):
            return

        # Validate the slice_wh parameter
        slice_wh = (self.slice_width.value(), self.slice_height.value())
        if not self.validate_slice_wh(slice_wh):
            return

        # Validate the overlap_wh parameter
        overlap_wh = self.overlap_input.get_value()
        if not self.validate_overlap_wh(overlap_wh):
            return

        # Validate the ext parameter
        ext = self.ext_combo.currentText()
        if not self.validate_ext(ext):
            return

        # Validate the densify_factor parameter
        densify_factor = self.densify_factor_spinbox.value()
        if not self.validate_densify_factor(densify_factor):
            return

        # Validate the smoothing_tolerance parameter
        smoothing_tolerance = self.smoothing_tolerance_spinbox.value()
        if not self.validate_smoothing_tolerance(smoothing_tolerance):
            return

        # Validate the train_ratio, valid_ratio, and test_ratio parameters
        train_ratio = self.train_ratio_spinbox.value()
        valid_ratio = self.valid_ratio_spinbox.value()
        test_ratio = self.test_ratio_spinbox.value()
        if not self.validate_ratios(train_ratio, valid_ratio, test_ratio):
            return

        # Validate the margins parameter
        margins = self.margins_input.get_value()
        if not self.validate_margins(margins):
            return

        # Include negative samples
        include_negatives = self.include_negatives_combo.currentText()
        include_negatives = True if include_negatives == "True" else False

        # Get the number of visualization samples
        num_viz_samples = self.num_viz_sample_spinbox.value()

        # Pause the cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Create and show the progress bar
        self.progress_bar = ProgressBar(self, title="Tiling Progress")
        self.progress_bar.show()

        def progress_callback(progress: TileProgress):
            self.progress_bar.setWindowTitle(f"Processing {progress.current_set.capitalize()} Set")
            progress_percentage = int((progress.current_tile / progress.total_tiles) * 100)
            self.progress_bar.set_value(progress_percentage)
            self.progress_bar.update_progress()
            if self.progress_bar.wasCanceled():
                raise Exception("Tiling process was canceled by the user.")

        config = TileConfig(
            slice_wh=slice_wh,
            overlap_wh=overlap_wh,
            ext=ext,
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
            callback=progress_callback,
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
