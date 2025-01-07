import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
from yolo_tiler import YoloTiler, TileConfig

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox, QCheckBox, QVBoxLayout,
                             QLabel, QDialog, QDialogButtonBox, QGroupBox, QButtonGroup,
                             QFormLayout, QLineEdit, QDoubleSpinBox, QComboBox, QPushButton, QFileDialog, QSpinBox)

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
        self.resize(400, 100)

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
        src_layout = QVBoxLayout()
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
        dst_layout = QVBoxLayout()
        dst_layout.addWidget(self.dst_edit)
        dst_layout.addWidget(self.dst_button)
        layout.addRow("Destination Directory:", dst_layout)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
    def setup_config_layout(self):
        """
        Set up the TileConfig parameters layout.
        """
        group_box = QGroupBox("Configuation Parameters")
        layout = QFormLayout()

        # Tile Size
        self.slice_wh_edit = QLineEdit()
        layout.addRow("Tile Size (width, height):", self.slice_wh_edit)

        # Overlap
        self.overlap_wh_edit = QLineEdit()
        layout.addRow("Overlap (width, height):", self.overlap_wh_edit)

        # Image Extension
        self.ext_combo = QComboBox()
        self.ext_combo.addItems([".png", ".tif", ".jpeg", ".jpg"])
        self.ext_combo.setEditable(True)
        layout.addRow("Image Extension:", self.ext_combo)

        # Densify Factor
        self.densify_factor_spinbox = QDoubleSpinBox()
        self.densify_factor_spinbox.setRange(0.0, 1.0)
        self.densify_factor_spinbox.setSingleStep(0.1)
        layout.addRow("Densify Factor:", self.densify_factor_spinbox)

        # Smoothing Tolerance
        self.smoothing_tolerance_spinbox = QDoubleSpinBox()
        self.smoothing_tolerance_spinbox.setRange(0.0, 1.0)
        self.smoothing_tolerance_spinbox.setSingleStep(0.1)
        layout.addRow("Smoothing Tolerance:", self.smoothing_tolerance_spinbox)

        # Train, Validation, and Test Ratios
        self.train_ratio_spinbox = QDoubleSpinBox()
        self.train_ratio_spinbox.setRange(0.0, 1.0)
        self.train_ratio_spinbox.setSingleStep(0.1)
        layout.addRow("Train Ratio:", self.train_ratio_spinbox)

        self.valid_ratio_spinbox = QDoubleSpinBox()
        self.valid_ratio_spinbox.setRange(0.0, 1.0)
        self.valid_ratio_spinbox.setSingleStep(0.1)
        layout.addRow("Validation Ratio:", self.valid_ratio_spinbox)

        self.test_ratio_spinbox = QDoubleSpinBox()
        self.test_ratio_spinbox.setRange(0.0, 1.0)
        self.test_ratio_spinbox.setSingleStep(0.1)
        layout.addRow("Test Ratio:", self.test_ratio_spinbox)

        # Margins
        self.margins_edit = QLineEdit()
        layout.addRow("Margins:", self.margins_edit)

        # Number of Visualization Samples
        self.num_viz_sample_spinbox = QSpinBox()
        self.num_viz_sample_spinbox.setRange(1, 100)
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
            QMessageBox.warning(self, "Invalid Source Directory", "The source directory must contain a 'train' sub-folder.")
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
        if not isinstance(overlap_wh, tuple) or len(overlap_wh) != 2 or not all(isinstance(i, (int, float)) for i in overlap_wh):
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
            QMessageBox.warning(self, "Invalid Image Extension", "The image extension must be a string starting with a dot.")
            return False
        return True

    def validate_densify_factor(self, densify_factor):
        """
        Validate the densify_factor parameter to ensure it is a float between 0 and 1.
        
        :param densify_factor: Densify factor
        :return: True if valid, False otherwise
        """
        if not isinstance(densify_factor, float) or not (0.0 <= densify_factor <= 1.0):
            QMessageBox.warning(self, "Invalid Densify Factor", "The densify factor must be a float between 0 and 1.")
            return False
        return True

    def validate_smoothing_tolerance(self, smoothing_tolerance):
        """
        Validate the smoothing_tolerance parameter to ensure it is a float between 0 and 1.
        
        :param smoothing_tolerance: Smoothing tolerance
        :return: True if valid, False otherwise
        """
        if not isinstance(smoothing_tolerance, float) or not (0.0 <= smoothing_tolerance <= 1.0):
            QMessageBox.warning(self, "Invalid Smoothing Tolerance", "The smoothing tolerance must be a float between 0 and 1.")
            return False
        return True

    def validate_ratios(self, train_ratio, valid_ratio, test_ratio):
        """
        Validate the train_ratio, valid_ratio, and test_ratio parameters to ensure they are floats between 0 and 1 and sum to 1.
        
        :param train_ratio: Train ratio
        :param valid_ratio: Validation ratio
        :param test_ratio: Test ratio
        :return: True if valid, False otherwise
        """
        if not all(isinstance(ratio, float) and 0.0 <= ratio <= 1.0 for ratio in [train_ratio, valid_ratio, test_ratio]):
            QMessageBox.warning(self, "Invalid Ratios", "The train, validation, and test ratios must be floats between 0 and 1.")
            return False
        if not abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-9:
            QMessageBox.warning(self, "Invalid Ratios", "The train, validation, and test ratios must sum to 1.")
            return False
        return True

    def validate_margins(self, margins):
        """
        Validate the margins parameter to ensure it is a tuple of four integers.
        
        :param margins: Margins
        :return: True if valid, False otherwise
        """
        if not isinstance(margins, tuple) or len(margins) != 4 or not all(isinstance(i, (int, float)) for i in margins):
            QMessageBox.warning(self, "Invalid Margins", "The margins must be a tuple of four integers.")
            return False
        return True
    
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
        slice_wh = eval(self.slice_wh_edit.text())
        if not self.validate_slice_wh(slice_wh):
            return

        # Validate the overlap_wh parameter
        overlap_wh = eval(self.overlap_wh_edit.text())
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
        margins = eval(self.margins_edit.text())
        if not self.validate_margins(margins):
            return

        # Get the number of visualization samples
        num_viz_samples = self.num_viz_sample_spinbox.value()

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
            margins=margins
        )

        tiler = YoloTiler(
            source=src,
            target=dst,
            config=config,
            num_viz_samples=num_viz_samples,  # Number of samples to visualize
        )

        tiler.run()
