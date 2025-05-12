import warnings

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPen, QBrush
from PyQt5.QtWidgets import (QMessageBox, QVBoxLayout, QLabel, QDialog, QDialogButtonBox, 
                             QGroupBox, QFormLayout, QComboBox, QPushButton, QSpinBox,
                             QHBoxLayout, QWidget, QGraphicsRectItem, QDoubleSpinBox, QCheckBox)

from coralnet_toolbox.QtWorkArea import WorkArea

from coralnet_toolbox.Common.QtTileSizeInput import TileSizeInput
from coralnet_toolbox.Common.QtOverlapInput import OverlapInput
from coralnet_toolbox.Common.QtMarginInput import MarginInput

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class TileInference(QDialog):
    """
    Base class for performing tiled inference on images using object detection, and instance segmentation.

    :param main_window: MainWindow object
    :param parent: Parent widget
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window
        self.graphics_utility = main_window.annotation_window.graphics_utility

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Tile Inference")
        self.resize(400, 600)
        
        # Initialize graphics tracking lists and objects
        self.margin_work_area = None
        self.margin_graphics = []
        self.tile_work_areas = []
        self.all_graphics = []

        self.layout = QVBoxLayout(self)

        # Setup the info layout
        self.setup_info_layout()
        # Setup the options layout
        self.setup_options_layout()
        # Setup the tile configuration layout
        self.setup_tile_config_layout()
        # Buttons at bottom
        self.setup_buttons_layout()

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
        
    def setup_options_layout(self):
        """Set up additional options layout."""
        group_box = QGroupBox("Tiling Options")
        layout = QVBoxLayout()

        # Option to ensure full coverage (enabled by default)
        self.ensure_coverage_checkbox = QCheckBox("Ensure full coverage of usable area")
        self.ensure_coverage_checkbox.setChecked(True)
        self.ensure_coverage_checkbox.setToolTip(
            "When enabled, adds additional tiles at the right and bottom edges to ensure the entire area is covered"
        )
        layout.addWidget(self.ensure_coverage_checkbox)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_tile_config_layout(self):
        """Set up tile config layout."""
        group_box = QGroupBox("Tile Configuration Parameters")
        layout = QFormLayout()

        self.tile_size_input = TileSizeInput()
        layout.addRow(self.tile_size_input)

        self.overlap_input = OverlapInput()
        layout.addRow(self.overlap_input)

        self.margins_input = MarginInput()
        layout.addRow(self.margins_input)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_buttons_layout(self):
        """
        Set up the layout with buttons.
        """
        buttons_layout = QHBoxLayout()
        
        # Preview button
        self.preview_button = QPushButton("Preview Tiles")
        self.preview_button.clicked.connect(self.preview_tiles)
        buttons_layout.addWidget(self.preview_button)
        
        # Clear button
        self.clear_button = QPushButton("Clear Tiles")
        self.clear_button.clicked.connect(self.clear_tiles)
        buttons_layout.addWidget(self.clear_button)
        
        # Create a button box with custom buttons
        button_box = QDialogButtonBox()
        apply_button = QPushButton("Apply")
        cancel_button = QPushButton("Cancel")

        button_box.addButton(apply_button, QDialogButtonBox.AcceptRole)
        button_box.addButton(cancel_button, QDialogButtonBox.RejectRole)

        button_box.accepted.connect(self.apply)
        button_box.rejected.connect(self.reject)

        # Add button layout first, then the standard button box
        self.layout.addLayout(buttons_layout)
        self.layout.addWidget(button_box)

    def showEvent(self, event):
        """Handle dialog show event."""
        super().showEvent(event)
        # Ensure graphics are cleared when dialog is shown
        self.clear_tiles()

    def closeEvent(self, event):
        """Handle dialog close event."""
        self.clear_tiles()
        event.accept()

    def reject(self):
        """Handle dialog rejection."""
        self.clear_tiles()
        super().reject()

    def preview_tiles(self):
        """
        Preview tile grid based on current settings.
        Creates work areas for each tile.
        """
        # Clear any existing tiles first
        self.clear_tiles()
        
        # Get current image
        if not hasattr(self.annotation_window, 'current_image_path') or not self.annotation_window.current_image_path:
            QMessageBox.warning(self, 
                                "No Image", 
                                "No image is currently loaded.")
            return
            
        # Get image dimensions
        image_width = self.annotation_window.pixmap_image.width()
        image_height = self.annotation_window.pixmap_image.height()
            
        # Get parameters from inputs
        try:
            # Get tile size as a tuple (width, height)
            tile_width, tile_height = self.tile_size_input.get_value()
            
            # Get overlap as a tuple of percentages (width_overlap, height_overlap)
            overlap_width_pct, overlap_height_pct = self.overlap_input.get_overlap(image_width, image_height)
            
            # Convert overlap percentages to pixel values
            if self.overlap_input.value_type.currentIndex() == 1:  # Percentage
                overlap_width = int(overlap_width_pct * tile_width)
                overlap_height = int(overlap_height_pct * tile_height)
            else:  # Pixels
                overlap_width = int(self.overlap_input.width_spin.value())
                overlap_height = int(self.overlap_input.height_spin.value())
            
            # Get margins
            margins = self.margins_input.get_margins(image_width, image_height)
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Parameters", str(e))
            return
        
        # Calculate area inside margins
        left, top, right, bottom = margins
        usable_width = image_width - left - right
        usable_height = image_height - top - bottom
        
        # Calculate effective tile size (adjusted for overlap)
        effective_width = tile_width - overlap_width
        effective_height = tile_height - overlap_height
        
        if effective_width <= 0 or effective_height <= 0:
            QMessageBox.warning(self, 
                                "Invalid Parameters", 
                                f"Effective tile size must be positive. Reduce overlap.\n"
                                f"Current overlap: {overlap_width}×{overlap_height} pixels\n"
                                f"Tile size: {tile_width}×{tile_height} pixels")
            return
        
        # Get the thickness of the work areas
        thickness = self.graphics_utility.get_workarea_thickness(self.annotation_window)
        
        # Create a margin work area to show the boundary with shadow
        self.margin_work_area = WorkArea(
            left, top, usable_width, usable_height,
            self.annotation_window.current_image_path
        )
        
        # Set a different color for the margin boundary
        self.margin_work_area.work_area_pen = QPen(QColor(0, 0, 255), 2, Qt.DashLine)
        
        # Create graphics with shadow to highlight the working area
        margin_graphics = self.margin_work_area.create_graphics(
            self.annotation_window.scene, 
            pen_width=thickness + 3,
            include_shadow=True  # Add shadow to highlight the usable area
        )
        
        # Add the margin rectangle to our graphics list
        self.all_graphics.append(margin_graphics)
        
        # Also track the shadow item for proper cleanup
        if self.margin_work_area.shadow_area:
            self.all_graphics.append(self.margin_work_area.shadow_area)
        
        # Calculate tile positions using the ensure full coverage option
        ensure_coverage = self.ensure_coverage_checkbox.isChecked()
        if ensure_coverage:
            # Calculate number of tiles in each dimension for the regular grid
            num_tiles_x = max(1, int(usable_width / effective_width))
            num_tiles_y = max(1, int(usable_height / effective_height))
            
            # Calculate the exact covered area with the regular grid
            covered_width = num_tiles_x * effective_width + overlap_width
            covered_height = num_tiles_y * effective_height + overlap_height
            
            # Create work areas for the regular grid
            for i in range(num_tiles_y):
                for j in range(num_tiles_x):
                    # Calculate tile coordinates
                    x = left + j * effective_width
                    y = top + i * effective_height
                    
                    # Create work area for this tile
                    tile_work_area = WorkArea(
                        x, y, tile_width, tile_height,
                        self.annotation_window.current_image_path
                    )
                    
                    # Add to scene with thinner line and store the graphics
                    tile_graphics = tile_work_area.create_graphics(
                        self.annotation_window.scene, pen_width=thickness,
                    )
                    
                    # Store the graphics for later removal
                    self.all_graphics.append(tile_graphics)
                    
                    # Store the work area
                    self.tile_work_areas.append(tile_work_area)
            
            # Check if we need extra columns on the right
            if covered_width < usable_width:
                # Create extra column of tiles aligned to the right edge
                right_edge = left + usable_width - tile_width
                for i in range(num_tiles_y):
                    y = top + i * effective_height
                    
                    # Create work area aligned to right edge
                    tile_work_area = WorkArea(
                        right_edge, y, tile_width, tile_height,
                        self.annotation_window.current_image_path
                    )
                    
                    # Add to scene with thinner line and store the graphics
                    tile_graphics = tile_work_area.create_graphics(
                        self.annotation_window.scene, pen_width=thickness,
                    )
                    
                    # Store the graphics for later removal
                    self.all_graphics.append(tile_graphics)
                    
                    # Store the work area
                    self.tile_work_areas.append(tile_work_area)
            
            # Check if we need extra rows at the bottom
            if covered_height < usable_height:
                # Create extra row of tiles aligned to the bottom edge
                bottom_edge = top + usable_height - tile_height
                for j in range(num_tiles_x):
                    x = left + j * effective_width
                    
                    # Create work area aligned to bottom edge
                    tile_work_area = WorkArea(
                        x, bottom_edge, tile_width, tile_height,
                        self.annotation_window.current_image_path
                    )
                    
                    # Add to scene with thinner line and store the graphics
                    tile_graphics = tile_work_area.create_graphics(
                        self.annotation_window.scene, pen_width=thickness,
                    )
                    
                    # Store the graphics for later removal
                    self.all_graphics.append(tile_graphics)
                    
                    # Store the work area
                    self.tile_work_areas.append(tile_work_area)
                
                # Check if we need a corner tile (if both right and bottom need coverage)
                if covered_width < usable_width:
                    # Create the bottom right corner tile
                    tile_work_area = WorkArea(
                        right_edge, bottom_edge, tile_width, tile_height,
                        self.annotation_window.current_image_path
                    )
                    
                    # Add to scene with thinner line and store the graphics
                    tile_graphics = tile_work_area.create_graphics(
                        self.annotation_window.scene, pen_width=thickness,
                    )
                    
                    # Store the graphics for later removal
                    self.all_graphics.append(tile_graphics)
                    
                    # Store the work area
                    self.tile_work_areas.append(tile_work_area)
        else:
            # Original logic when ensure_coverage is disabled
            num_tiles_x = max(1, int((usable_width - overlap_width) / effective_width) + 1)
            num_tiles_y = max(1, int((usable_height - overlap_height) / effective_height) + 1)
            
            # Create work areas for each tile
            for i in range(num_tiles_y):
                for j in range(num_tiles_x):
                    # Calculate tile coordinates
                    x = left + j * effective_width
                    y = top + i * effective_height
                    
                    # Ensure the tile doesn't go beyond the usable area
                    if x + tile_width > left + usable_width or y + tile_height > top + usable_height:
                        continue
                    
                    # Create work area for this tile
                    tile_work_area = WorkArea(
                        x, y, tile_width, tile_height,
                        self.annotation_window.current_image_path
                    )
                    
                    # Add to scene with thinner line and store the graphics
                    tile_graphics = tile_work_area.create_graphics(
                        self.annotation_window.scene, pen_width=thickness,
                    )
                    
                    # Store the graphics for later removal
                    self.all_graphics.append(tile_graphics)
                    
                    # Store the work area
                    self.tile_work_areas.append(tile_work_area)
        
        # Count tiles
        total_tiles = len(self.tile_work_areas)
        
        # Show tile count in a message
        coverage_status = "with full coverage" if ensure_coverage else "with standard grid"
        QMessageBox.information(
            self, 
            "Tile Preview", 
            f"Created {total_tiles} tiles {coverage_status}:\n"
            f"• Tile size: {tile_width}×{tile_height} pixels\n"
            f"• Overlap: {overlap_width}×{overlap_height} pixels\n"
            f"• Effective step: {effective_width}×{effective_height} pixels"
        )

    def clear_tiles(self):
        """Remove all tile work areas and their graphics from the scene."""
        # Use margin_work_area's remove_from_scene method to properly clean up shadow
        if self.margin_work_area:
            self.margin_work_area.remove_from_scene()
            self.margin_work_area = None
            
        # Remove all other graphics items from the scene
        for item in self.all_graphics:
            if item in self.annotation_window.scene.items():
                self.annotation_window.scene.removeItem(item)
        
        # Clear all tracking lists
        self.all_graphics = []
        self.margin_graphics = []
            
        # Remove all tile work areas
        for work_area in self.tile_work_areas:
            if hasattr(work_area, 'remove_from_scene'):
                work_area.remove_from_scene()
        
        self.tile_work_areas = []
        
        # Update the view
        if hasattr(self.annotation_window, 'viewport'):
            self.annotation_window.viewport().update()

    def apply(self):
        """
        Method called when the Apply button is clicked.
        Creates work areas for each tile and adds them to the current raster.
        """
        # Get current image dimensions
        if not hasattr(self.annotation_window, 'current_image_path') or not self.annotation_window.current_image_path:
            QMessageBox.warning(self, "No Image", "No image is currently loaded.")
            return
        
        # First preview the tiles (this will also validate parameters)
        self.preview_tiles()
        
        # Check if we have valid tiles
        if not self.tile_work_areas:
            return
        
        # Get the current raster
        image_path = self.annotation_window.current_image_path
        raster = self.main_window.image_window.raster_manager.get_raster(image_path)
        
        if not raster:
            QMessageBox.warning(self, "Error", "Could not get current raster.")
            return
        
        # Add each work area to the raster
        for work_area in self.tile_work_areas:
            raster.add_work_area(work_area)
        
        # Inform the user
        num_tiles = len(self.tile_work_areas)
        QMessageBox.information(
            self, 
            "Tiles Added", 
            f"Added {num_tiles} tiles to the current image.\n"
            f"You can now process these work areas individually."
        )
        
        # Clear the preview graphics but keep the work areas added to the raster
        self.clear_tiles()
        
        # Close the dialog
        self.accept()
