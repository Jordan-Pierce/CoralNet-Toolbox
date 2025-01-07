import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
from yolo_tiler import YoloTiler, TileConfig

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox, QCheckBox, QVBoxLayout,
                             QLabel, QDialog, QDialogButtonBox, QGroupBox, QButtonGroup,
                             QFormLayout, QLineEdit, QDoubleSpinBox, QComboBox, QPushButton, QFileDialog)

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

        # TODO Name of Destination Dataset edit text

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

        # Image Extension TODO Convert to editable dropdown with pre-filled (.png, .tif, .jpeg, .jpg)
        self.ext_edit = QLineEdit()
        layout.addRow("Image Extension:", self.ext_edit)

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
        dst = self.dst_edit.text()
        # TODO dst = dst + Name of Destination Dataset

        # -------------------------
        # Perform validation checks

        # Check the directory has a 'train' folder
        if not self.validate_source_directory(src):
            return
        
        # TODO Perform the other checks here

        config = TileConfig(
            slice_wh=eval(self.slice_wh_edit.text()),
            overlap_wh=eval(self.overlap_wh_edit.text()),
            ext=self.ext_edit.text(),
            annotation_type=self.annotation_type,
            densify_factor=self.densify_factor_spinbox.value(),
            smoothing_tolerance=self.smoothing_tolerance_spinbox.value(),
            train_ratio=self.train_ratio_spinbox.value(),
            valid_ratio=self.valid_ratio_spinbox.value(),
            test_ratio=self.test_ratio_spinbox.value(),
            margins=eval(self.margins_edit.text())
        )

        tiler = YoloTiler(
            source=src,
            target=dst,
            config=config,
            num_viz_samples=int(self.num_viz_samples.value()), # TODO
        )

        tiler.run()

