import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMessageBox, QCheckBox, QVBoxLayout,
                             QLabel, QDialog, QDialogButtonBox, QGroupBox, QButtonGroup,
                             QFormLayout, QLineEdit, QDoubleSpinBox, QComboBox, QPushButton, QFileDialog)

from coralnet_toolbox.Icons import get_icon
from yolo_tiler import YoloTiler, TileConfig
import os


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
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window
        
        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Tile Dataset")
        self.resize(400, 100)

        self.deploy_model_dialog = None
        self.loaded_model = None
        self.annotations = []
        self.prepared_patches = []
        self.image_paths = []

        self.layout = QVBoxLayout(self)

        # Setup the info layout
        self.setup_info_layout()
        # Setup the image options layout
        self.setup_options_layout()
        # Setup the task specific layout
        self.setup_task_specific_layout()
        # Setup the buttons layout
        self.setup_buttons_layout()
        
    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()
        
        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Tile an existing YOLO dataset.")
        
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_options_layout(self):
        """
        Set up the layout with image options.
        """
        # Create a group box for image options
        group_box = QGroupBox("Image Options")
        layout = QVBoxLayout()

        # Create a button group for the image checkboxes
        image_options_group = QButtonGroup(self)

        self.apply_filtered_checkbox = QCheckBox("Apply to filtered images")
        self.apply_prev_checkbox = QCheckBox("Apply to previous images")
        self.apply_next_checkbox = QCheckBox("Apply to next images")
        self.apply_all_checkbox = QCheckBox("Apply to all images")

        # Add the checkboxes to the button group
        image_options_group.addButton(self.apply_filtered_checkbox)
        image_options_group.addButton(self.apply_prev_checkbox)
        image_options_group.addButton(self.apply_next_checkbox)
        image_options_group.addButton(self.apply_all_checkbox)

        # Ensure only one checkbox can be checked at a time
        image_options_group.setExclusive(True)

        # Set the default checkbox
        self.apply_all_checkbox.setChecked(True)

        layout.addWidget(self.apply_filtered_checkbox)
        layout.addWidget(self.apply_prev_checkbox)
        layout.addWidget(self.apply_next_checkbox)
        layout.addWidget(self.apply_all_checkbox)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
    def setup_task_specific_layout(self):
        """
        Set up the layout with input fields for YoloTiler and TileConfig parameters.
        """
        group_box = QGroupBox("Tiling Parameters")
        layout = QFormLayout()

        # Source and Destination Directories
        self.src_edit = QLineEdit()
        self.src_button = QPushButton("Browse...")
        self.src_button.clicked.connect(self.browse_src_dir)
        src_layout = QVBoxLayout()
        src_layout.addWidget(self.src_edit)
        src_layout.addWidget(self.src_button)
        layout.addRow("Source Directory:", src_layout)

        self.dst_edit = QLineEdit()
        self.dst_button = QPushButton("Browse...")
        self.dst_button.clicked.connect(self.browse_dst_dir)
        dst_layout = QVBoxLayout()
        dst_layout.addWidget(self.dst_edit)
        dst_layout.addWidget(self.dst_button)
        layout.addRow("Destination Directory:", dst_layout)

        # Tile Size
        self.slice_wh_edit = QLineEdit()
        layout.addRow("Tile Size (width, height):", self.slice_wh_edit)

        # Overlap
        self.overlap_wh_edit = QLineEdit()
        layout.addRow("Overlap (width, height):", self.overlap_wh_edit)

        # Image Extension
        self.ext_edit = QLineEdit()
        layout.addRow("Image Extension:", self.ext_edit)

        # Annotation Type
        self.annotation_type_combo = QComboBox()
        self.annotation_type_combo.addItems(["object_detection", "instance_segmentation"])
        layout.addRow("Annotation Type:", self.annotation_type_combo)

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
    
    def update_uncertainty_label(self):
        """
        Update the uncertainty threshold label based on the slider value.
        """
        # Convert the slider value to a ratio (0-1)
        value = self.uncertainty_threshold_slider.value() / 100.0
        self.main_window.update_uncertainty_thresh(value)

    def on_uncertainty_changed(self, value):
        """
        Update the slider and label when the shared data changes.
        
        :param value: New uncertainty threshold value
        """
        self.uncertainty_threshold_slider.setValue(int(value * 100))
        self.uncertainty_threshold_label.setText(f"{value:.2f}")
        
    def get_selected_image_paths(self):
        """
        Get the selected image paths based on the options.
        
        :return: List of selected image paths
        """
        if self.apply_filtered_checkbox.isChecked():
            return self.image_window.filtered_image_paths
        elif self.apply_prev_checkbox.isChecked():
            current_image_index = self.image_window.image_paths.index(self.annotation_window.current_image_path)
            return self.image_window.image_paths[:current_image_index + 1]
        elif self.apply_next_checkbox.isChecked():
            current_image_index = self.image_window.image_paths.index(self.annotation_window.current_image_path)
            return self.image_window.image_paths[current_image_index:]
        else:
            return self.image_window.image_paths

    def apply(self):
        """
        Apply the selected batch inference options.
        """
        # Pause the cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # Get the selected image paths
            self.image_paths = self.get_selected_image_paths()
            self.batch_inference()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to make predictions: {str(e)}")
        finally:
            # Resume the cursor
            QApplication.restoreOverrideCursor()
        
        self.accept()

    def batch_inference(self):
        """
        Perform batch inference on the selected images and annotations.
        """
        src = self.src_edit.text()
        dst = self.dst_edit.text()

        if not self.validate_source_directory(src):
            return

        config = TileConfig(
            slice_wh=eval(self.slice_wh_edit.text()),
            overlap_wh=eval(self.overlap_wh_edit.text()),
            ext=self.ext_edit.text(),
            annotation_type=self.annotation_type_combo.currentText(),
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
            num_viz_samples=15,
        )

        tiler.run()

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
