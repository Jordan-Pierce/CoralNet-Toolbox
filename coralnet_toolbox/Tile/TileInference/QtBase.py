import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPen, QBrush
from PyQt5.QtWidgets import (QMessageBox, QVBoxLayout, QLabel, QDialog, QDialogButtonBox, 
                             QGroupBox, QFormLayout, QComboBox, QPushButton, QSpinBox,
                             QHBoxLayout, QWidget, QGraphicsRectItem, QDoubleSpinBox)

from coralnet_toolbox.QtCommon import TileSizeInput, OverlapInput, MarginInput

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Base(QDialog):
    """
    Base class for performing tiled inference on images using object detection, and instance segmentation
    datasets using YOLO-Patch-Based-Inference.

    :param main_window: MainWindow object
    :param parent: Parent widget
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Tile Inference")
        self.resize(400, 600)

        # Initialize debounce timer
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_tile_graphics)

        # Tile parameters
        self.tile_params = {}
        self.tile_inference_params = {}

        self.shape_x = None
        self.shape_y = None
        self.overlap_x = None
        self.overlap_y = None
        self.margins = None
        self.imgsz = None
        self.batch_inference = None
        self.include_residuals = None

        self.nms_threshold = None
        self.match_metric = None
        self.class_agnostic_nms = None
        self.intelligent_sorter = None
        self.sorter_bins = None
        self.memory_optimize = None

        self.image = None
        self.tile_graphics = []

        self.layout = QVBoxLayout(self)

        # Info layout at top
        self.setup_info_layout()

        # Create horizontal layout for configs
        config_layout = QHBoxLayout()

        # Left side - Tile Config
        self.tile_config_widget = QWidget()
        tile_layout = QVBoxLayout(self.tile_config_widget)
        self.setup_tile_config_layout(tile_layout)
        config_layout.addWidget(self.tile_config_widget)

        # Right side - Inference Config
        self.inference_config_widget = QWidget()
        inference_layout = QVBoxLayout(self.inference_config_widget)
        self.setup_inference_config_layout(inference_layout)
        config_layout.addWidget(self.inference_config_widget)

        # Add the horizontal config layout to main layout
        self.layout.addLayout(config_layout)

        # Buttons at bottom
        self.setup_buttons_layout()

    def showEvent(self, event):
        # Get the image pixmap from the annotation window
        self.image = self.annotation_window.image_pixmap
        self.update_tile_graphics()

    def closeEvent(self, event):
        self.clear_tile_graphics()

    def debounce_update(self):
        """Debounce the update_tile_graphics call by Nms"""
        self.update_timer.start(1000)

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Tile an image into smaller non / overlapping images, performing inference on each.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_tile_config_layout(self, parent_layout):
        """
        Set up the tile config parameters layout.
        """
        group_box = QGroupBox("Tile Configuration Parameters")
        layout = QFormLayout()

        # Tile Size
        self.tile_size_input = TileSizeInput()

        # Connect width and height spinboxes with debounce
        self.tile_size_input.width_spin.valueChanged.connect(self.debounce_update)
        self.tile_size_input.height_spin.valueChanged.connect(self.debounce_update)
        layout.addRow(self.tile_size_input)

        # Overlap
        self.overlap_input = OverlapInput()

        # Connect all spinboxes/doublespinboxes with debounce
        self.overlap_input.width_spin.valueChanged.connect(self.debounce_update)
        self.overlap_input.height_spin.valueChanged.connect(self.debounce_update)
        self.overlap_input.width_double.valueChanged.connect(self.debounce_update)
        self.overlap_input.height_double.valueChanged.connect(self.debounce_update)
        layout.addRow(self.overlap_input)

        # Margins
        self.margins_input = MarginInput()

        # Connect single value inputs with debounce
        self.margins_input.single_spin.valueChanged.connect(self.debounce_update)
        self.margins_input.single_double.valueChanged.connect(self.debounce_update)

        # Connect all margin spinboxes with debounce
        for spin in self.margins_input.margin_spins:
            spin.valueChanged.connect(self.debounce_update)

        # Connect all margin doublespinboxes with debounce
        for double in self.margins_input.margin_doubles:
            double.valueChanged.connect(self.debounce_update)

        layout.addRow(self.margins_input)

        group_box.setLayout(layout)
        parent_layout.addWidget(group_box)

    def setup_inference_config_layout(self, parent_layout):
        """
        Set up the inference configuration parameters layout.
        """
        group_box = QGroupBox("Inference Configuration Parameters")
        layout = QFormLayout()
        
        # Image size
        self.imgsz_input = QSpinBox()
        self.imgsz_input.setRange(1, 4096)
        self.imgsz_input.setValue(1024)
        layout.addRow("Image Size:", self.imgsz_input)
        
        # NMS threshold
        self.nms_threshold_input = QDoubleSpinBox()
        self.nms_threshold_input.setRange(0, 1)
        self.nms_threshold_input.setSingleStep(0.01)
        self.nms_threshold_input.setValue(0.5)
        layout.addRow("NMS Threshold:", self.nms_threshold_input)

        # Match metric
        self.match_metric_input = QComboBox()
        self.match_metric_input.addItems(["IOU", "IOS"])
        self.match_metric_input.setCurrentText("IOU")
        layout.addRow("Match Metric:", self.match_metric_input)

        # Class agnostic NMS
        self.class_agnostic_nms_input = QComboBox()
        self.class_agnostic_nms_input.addItems(["True", "False"])
        layout.addRow("Class Agnostic NMS:", self.class_agnostic_nms_input)

        # Intelligent sorter
        self.intelligent_sorter_input = QComboBox()
        self.intelligent_sorter_input.addItems(["True", "False"])
        layout.addRow("Intelligent Sorter:", self.intelligent_sorter_input)

        # Sorter bins
        self.sorter_bins_input = QSpinBox()
        self.sorter_bins_input.setRange(1, 10)
        self.sorter_bins_input.setValue(7)
        layout.addRow("Sorter Bins:", self.sorter_bins_input)

        # Memory optimization
        self.memory_input = QComboBox()
        self.memory_input.addItems(["True", "False"])
        layout.addRow("Memory Optimization:", self.memory_input)
        
        # Batch inference
        self.batch_inference_input = QComboBox()
        self.batch_inference_input.addItems(["True", "False"])
        layout.addRow("Batch Inference:", self.batch_inference_input)
        
        # Include residuals (dropbox True or False)
        self.include_residuals_input = QComboBox()
        self.include_residuals_input.addItems(["True", "False"])
        layout.addRow("<b>Include Residuals:</b>", self.include_residuals_input)

        group_box.setLayout(layout)
        parent_layout.addWidget(group_box)

    def setup_buttons_layout(self):
        """
        Set up the layout with buttons.
        """
        # Create a button box with custom buttons
        button_box = QDialogButtonBox()
        apply_button = QPushButton("Apply")
        unapply_button = QPushButton("Unapply")
        cancel_button = QPushButton("Cancel")

        button_box.addButton(apply_button, QDialogButtonBox.AcceptRole)
        button_box.addButton(unapply_button, QDialogButtonBox.ActionRole)
        button_box.addButton(cancel_button, QDialogButtonBox.RejectRole)

        button_box.accepted.connect(self.apply)
        unapply_button.clicked.connect(self.unapply)
        button_box.rejected.connect(self.reject)

        self.layout.addWidget(button_box)
        
    def get_tile_params(self):
        """
        Get the tile parameters.

        :return: Tile parameters
        """
        return self.tile_params

    def get_tile_inference_params(self):
        """
        Get the tile inference parameters.

        :return: Tile inference parameters
        """
        return self.tile_inference_params

    def get_params(self):
        """
        Get both the tile and tile inference parameters.

        :return: Tuple of tile and tile inference parameters
        """
        return self.get_tile_params(), self.get_tile_inference_params()

    def update_params(self):
        """
        Update the tile inference and inference parameters.
        """
        # Get image dimensions
        image_width = self.annotation_window.image_pixmap.width()
        image_height = self.annotation_window.image_pixmap.height()
        
        # Get the shape of the tiles in pixels
        self.shape_x, self.shape_y = self.tile_size_input.get_sizes(image_width, image_height)
        
        # Get the overlap in pixels
        self.overlap_x, self.overlap_y = self.overlap_input.get_overlap(image_width, 
                                                                        image_height, 
                                                                        return_pixels=False)
        
        # Get the margins in pixels
        self.margins = self.margins_input.get_margins(image_width, image_height)

        # Get the tile inference parameters
        self.imgsz = self.imgsz_input.value()
        self.nms_threshold = self.nms_threshold_input.value()
        self.match_metric = self.match_metric_input.currentText()
        self.sorter_bins = self.sorter_bins_input.value()
        self.class_agnostic_nms = self.class_agnostic_nms_input.currentText() == "True"
        self.intelligent_sorter = self.intelligent_sorter_input.currentText() == "True"
        self.memory_optimize = self.memory_input.currentText() == "True"
        self.batch_inference = self.batch_inference_input.currentText() == "True"
        self.include_residuals = self.include_residuals_input.currentText() == "True"
        
    def update_tile_graphics(self):
        """
        Uses class tile parameters to create a grid of tiles on the annotation window image.
        """
        # Clear existing tile graphics
        self.clear_tile_graphics()

        # Update and validate all parameters
        self.update_params()

        if not self.annotation_window.image_pixmap:
            return

        # Get image dimensions
        image_full_width = self.annotation_window.image_pixmap.width()
        image_full_height = self.annotation_window.image_pixmap.height()

        # Calculate grid boundaries
        x_start = self.margins[0]  # left margin
        y_start = self.margins[1]  # top margin
        x_end = image_full_width - self.margins[2]  # right margin
        y_end = image_full_height - self.margins[3]  # bottom margin
        
        # Calculate overlap coefficients
        if isinstance(self.overlap_x, float):
            cross_coef_x = 1 - self.overlap_x  # Float between 0-1
        else:
            cross_coef_x = 1 - (self.overlap_x / self.shape_x)  # Pixel value

        if isinstance(self.overlap_y, float):
            cross_coef_y = 1 - self.overlap_y  # Float between 0-1
        else:
            cross_coef_y = 1 - (self.overlap_y / self.shape_y)  # Pixel value

        # Calculate grid steps, adjusted to fit within margins
        x_steps = int((x_end - x_start - self.shape_x) / (self.shape_x * cross_coef_x)) + 1
        y_steps = int((y_end - y_start - self.shape_y) / (self.shape_y * cross_coef_y)) + 1

        # Calculate line thickness based on resolution
        line_thickness = max(10, min(20, max(image_full_width, image_full_height) // 1000))

        # Draw tiles
        for i in range(y_steps + 1):
            for j in range(x_steps + 1):
                x = x_start + int(self.shape_x * j * cross_coef_x)
                y = y_start + int(self.shape_y * i * cross_coef_y)

                # Truncate tile width and height if extending beyond boundaries
                tile_width = min(self.shape_x, x_end - x)
                tile_height = min(self.shape_y, y_end - y)

                # Determine tile color and transparency
                if tile_width == self.shape_x and tile_height == self.shape_y:
                    # Full tiles within image boundaries
                    color = QColor(0, 0, 0) if (i * (x_steps + 1) + j) % 2 == 0 else QColor(255, 255, 255)
                    opacity = 0.5
                    line_style = Qt.DotLine
                    brush = QBrush(color)
                else:
                    # Tiles outside/partially outside image boundaries
                    color = QColor(255, 0, 0)  # Red color
                    opacity = 1.0
                    line_style = Qt.SolidLine
                    brush = QBrush(Qt.NoBrush)  # No fill for boundary tiles

                # Skip if tile is completely outside image
                if tile_width > 0 and tile_height > 0:
                    tile = QGraphicsRectItem(x, y, tile_width, tile_height)
                    tile.setPen(QPen(color, line_thickness, line_style))
                    tile.setBrush(brush)
                    tile.setOpacity(opacity)

                    self.annotation_window.scene.addItem(tile)
                    self.tile_graphics.append(tile)

    def apply(self):
        """
        Apply the tile inference options.
        """
        try:
            # Update and validate all parameters
            self.update_params()
            
            # Create tiling parameters dictionary
            self.tile_params = {
                "imgsz": self.imgsz,
                "shape_x": self.shape_x,
                "shape_y": self.shape_y,
                "overlap_x": self.overlap_x,
                "overlap_y": self.overlap_y,
                "margins": self.margins,
                "batch_inference": self.batch_inference,
                "memory_optimize": self.memory_optimize,
                "include_residuals": self.include_residuals,
                "show_processing_status": True,
                "show_crops": False,
            }

            # Create inference parameters dictionary
            self.tile_inference_params = {
                "nms_threshold": self.nms_threshold,
                "match_metric": self.match_metric,
                "class_agnostic_nms": self.class_agnostic_nms,
                "intelligent_sorter": self.intelligent_sorter,
                "sorter_bins": self.sorter_bins,
            }

            QMessageBox.information(self,
                                    "Success",
                                    "Tile Inference parameters set successfully.")

        except Exception as e:
            QMessageBox.critical(self,
                                 "Error",
                                 f"Failed to set Tile Inference parameters: {str(e)}")
        finally:
            self.clear_tile_graphics()

        self.accept()

    def unapply(self):
        """
        Reset tile inference configurations.
        """
        try:
            # Reset tile inference parameters
            self.tile_params = {}
            self.tile_inference_params = {}
            # Update and validate all parameters
            self.update_params()
    
            QMessageBox.information(self,
                                    "Success",
                                    "Tile inference parameters reset successfully.")

        except Exception as e:
            QMessageBox.critical(self,
                                 "Error",
                                 f"Failed to reset Tile Inference parameters: {str(e)}")
        finally:
            self.clear_tile_graphics()

        self.accept()

    def clear_tile_graphics(self):
        """
        Clear the tile graphics from the annotation window.
        """
        # Remove all tile graphics from the scene
        for tile_graphic in self.tile_graphics:
            tile_graphic.scene().removeItem(tile_graphic)
        # Update the viewport to remove the tiles
        self.annotation_window.viewport().update()
        self.annotation_window.fitInView(self.annotation_window.scene.sceneRect(), Qt.KeepAspectRatio)
        # Clear the tile graphics list
        self.tile_graphics = []
