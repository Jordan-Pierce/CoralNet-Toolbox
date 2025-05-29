import warnings

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPen, QBrush
from PyQt5.QtWidgets import (QMessageBox, QVBoxLayout, QLabel, QDialog, QDialogButtonBox, 
                             QGroupBox, QFormLayout, QComboBox, QPushButton, QSpinBox,
                             QHBoxLayout, QWidget, QGraphicsRectItem, QDoubleSpinBox, QCheckBox,
                             QButtonGroup, QListWidget, QTableWidget, QTableWidgetItem, QHeaderView)

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


class TileCreation(QDialog):
    """
    Base class for performing tiled creation on images using object detection, and instance segmentation.

    :param main_window: MainWindow object
    :param parent: Parent widget
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window
        self.graphics_utility = main_window.annotation_window.graphics_utility

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Tile Creation")
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
        # Set up apply to options layout
        self.setup_apply_options_layout()
        # Buttons at bottom
        self.setup_buttons_layout()
        
    def showEvent(self, event):
        """Handle dialog show event."""
        super().showEvent(event)
        self.update_tile_size_limits()
        self.clear_tiles()
        self.clear_checkboxes()

    def closeEvent(self, event):
        """Handle dialog close event."""
        self.clear_tiles()
        self.clear_checkboxes()
        event.accept()

    def reject(self):
        """Handle dialog rejection."""
        self.clear_tiles()
        self.clear_checkboxes()
        super().reject()

    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Tile images into smaller non / overlapping images for performing inference on.")

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
        
    def setup_apply_options_layout(self):
        """Set up the application scope options."""
        group_box = QGroupBox("Apply To")
        layout = QVBoxLayout()

        self.apply_filtered_checkbox = QCheckBox("▼ Apply to filtered images")
        self.apply_prev_checkbox = QCheckBox("↑ Apply to previous images")
        self.apply_next_checkbox = QCheckBox("↓ Apply to next images")
        self.apply_all_checkbox = QCheckBox("↕ Apply to all images")

        layout.addWidget(self.apply_filtered_checkbox)
        layout.addWidget(self.apply_prev_checkbox)
        layout.addWidget(self.apply_next_checkbox)
        layout.addWidget(self.apply_all_checkbox)

        self.apply_group = QButtonGroup(self)
        self.apply_group.addButton(self.apply_filtered_checkbox)
        self.apply_group.addButton(self.apply_prev_checkbox)
        self.apply_group.addButton(self.apply_next_checkbox)
        self.apply_group.addButton(self.apply_all_checkbox)
        self.apply_group.setExclusive(True)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
    def clear_checkboxes(self):
        """Clear all apply checkboxes."""
        # Temporarily disable exclusivity to allow clearing all
        self.apply_group.setExclusive(False)
        
        self.apply_filtered_checkbox.setChecked(False)
        self.apply_prev_checkbox.setChecked(False)
        self.apply_next_checkbox.setChecked(False)
        self.apply_all_checkbox.setChecked(False)
        
        # Re-enable exclusivity
        self.apply_group.setExclusive(True)
        
    def show_skipped_images_dialog(self, errors):
        """
        Show a dialog with a QTableWidget listing skipped images and reasons.
        """    
        dialog = QDialog(self)
        dialog.setWindowTitle("Some Images Skipped")
        dialog.resize(700, 350)
        layout = QVBoxLayout(dialog)
        label = QLabel("The following images were skipped:")
        layout.addWidget(label)
        table_widget = QTableWidget()
        table_widget.setColumnCount(2)
        table_widget.setHorizontalHeaderLabels(["Image Path", "Reason"])
        table_widget.setRowCount(len(errors))
        for row, error in enumerate(errors):
            if ": " in error:
                image_path, reason = error.split(": ", 1)
            else:
                image_path = error
                reason = ""
            table_widget.setItem(row, 0, QTableWidgetItem(image_path))
            table_widget.setItem(row, 1, QTableWidgetItem(reason))
        table_widget.resizeColumnsToContents()
        layout.addWidget(table_widget)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(dialog.accept)
        layout.addWidget(button_box)
        dialog.exec_()

    def update_tile_size_limits(self):
        """Set tile size spin box maximums to current image dimensions."""
        if self.annotation_window.current_image_path:
            image_width = self.annotation_window.pixmap_image.width()
            image_height = self.annotation_window.pixmap_image.height()
            self.tile_size_input.width_spin.setMaximum(image_width)
            self.tile_size_input.height_spin.setMaximum(image_height)
        
    def get_selected_image_paths(self):
        """
        Get the selected image paths based on the options.

        :return: List of selected image paths
        """
        # Current image path showing
        current_image_path = self.annotation_window.current_image_path
        if not current_image_path:
            return []

        # Determine which images to export annotations for
        if self.apply_filtered_checkbox.isChecked():
            return self.image_window.table_model.filtered_paths
        elif self.apply_prev_checkbox.isChecked():
            if current_image_path in self.image_window.table_model.filtered_paths:
                current_index = self.image_window.table_model.get_row_for_path(current_image_path)
                return self.image_window.table_model.filtered_paths[:current_index + 1]
            else:
                return [current_image_path]
        elif self.apply_next_checkbox.isChecked():
            if current_image_path in self.image_window.table_model.filtered_paths:
                current_index = self.image_window.table_model.get_row_for_path(current_image_path)
                return self.image_window.table_model.filtered_paths[current_index:]
            else:
                return [current_image_path]
        elif self.apply_all_checkbox.isChecked():
            return self.image_window.raster_manager.image_paths
        else:
            # Only apply to the current image
            return [current_image_path]

    def validate_parameters(self, image_path):
        """
        Validate current tile, overlap, and margin parameters for the given image.
        Returns (is_valid, error_message, params_dict).
        params_dict contains: tile_width, tile_height, overlap_width, overlap_height, margins, image_width, image_height
        """
        if not image_path:
            return False, "No image is currently loaded.", None
    
        image_width = self.annotation_window.pixmap_image.width()
        image_height = self.annotation_window.pixmap_image.height()
    
        try:
            tile_width, tile_height = self.tile_size_input.get_value()
            if tile_width > image_width or tile_height > image_height:
                return (
                    False, 
                    f"Tile size ({tile_width}×{tile_height}) exceeds image size ({image_width}×{image_height}).", 
                    None
                )
    
            overlap_width_pct, overlap_height_pct = self.overlap_input.get_overlap(image_width, image_height)
            if self.overlap_input.value_type.currentIndex() == 1:  # Percentage
                overlap_width = int(overlap_width_pct * tile_width)
                overlap_height = int(overlap_height_pct * tile_height)
            else:
                overlap_width = int(self.overlap_input.width_spin.value())
                overlap_height = int(self.overlap_input.height_spin.value())
    
            margins = self.margins_input.get_margins(image_width, image_height)
    
            effective_width = tile_width - overlap_width
            effective_height = tile_height - overlap_height
            if effective_width <= 0 or effective_height <= 0:
                return False, (
                    f"Effective tile size must be positive. Reduce overlap.\n"
                    f"Current overlap: {overlap_width}×{overlap_height} pixels\n"
                    f"Tile size: {tile_width}×{tile_height} pixels"
                ), None
    
            left, top, right, bottom = margins
            if left + right >= image_width or top + bottom >= image_height:
                return False, "Margins are too large for the image size.", None
    
            params = {
                "tile_width": tile_width,
                "tile_height": tile_height,
                "overlap_width": overlap_width,
                "overlap_height": overlap_height,
                "margins": margins,
                "image_width": image_width,
                "image_height": image_height,
            }
            return True, "", params
    
        except Exception as e:
            return False, f"Invalid parameters: {e}", None
    
    def preview_tiles(self):
        """
        Preview tile grid based on current settings.
        Creates work areas for each tile.
        """
        self.update_tile_size_limits()
        self.clear_tiles()

        # Validate and extract parameters
        is_valid, error_message, params = self.validate_parameters(self.annotation_window.current_image_path)
        if not is_valid:
            QMessageBox.warning(self, "Invalid Parameters", error_message)
            return

        # Unpack validated parameters
        tile_width = params["tile_width"]
        tile_height = params["tile_height"]
        overlap_width = params["overlap_width"]
        overlap_height = params["overlap_height"]
        margins = params["margins"]
        image_width = params["image_width"]
        image_height = params["image_height"]
            
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
                    
                    # Ensure tile doesn't exceed image boundaries
                    if x + tile_width > image_width or y + tile_height > image_height:
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
            
            # Check if we need extra columns on the right
            if covered_width < usable_width:
                # Calculate right edge position ensuring tile fits within image
                right_edge = min(left + usable_width - tile_width, image_width - tile_width)
                
                # Only add right edge tiles if they would be valid
                if right_edge >= left and right_edge + tile_width <= image_width:
                    for i in range(num_tiles_y):
                        y = top + i * effective_height
                        
                        # Ensure tile doesn't exceed image boundaries
                        if y + tile_height > image_height:
                            continue
                        
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
                # Calculate bottom edge position ensuring tile fits within image
                bottom_edge = min(top + usable_height - tile_height, image_height - tile_height)
                
                # Only add bottom edge tiles if they would be valid
                if bottom_edge >= top and bottom_edge + tile_height <= image_height:
                    for j in range(num_tiles_x):
                        x = left + j * effective_width
                        
                        # Ensure tile doesn't exceed image boundaries
                        if x + tile_width > image_width:
                            continue
                        
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
                        # Calculate corner position ensuring tile fits within image
                        right_edge = min(left + usable_width - tile_width, image_width - tile_width)
                        
                        # Only add corner tile if it would be valid
                        if (right_edge >= left and right_edge + tile_width <= image_width and 
                            bottom_edge >= top and bottom_edge + tile_height <= image_height):
                            
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
                    
                    # Ensure the tile doesn't go beyond the usable area OR image boundaries
                    if (x + tile_width > left + usable_width or y + tile_height > top + usable_height or
                        x + tile_width > image_width or y + tile_height > image_height):
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
            work_area.remove_from_scene()
        
        self.tile_work_areas = []
        
        # Update the view
        self.annotation_window.viewport().update()

    def generate_tile_work_areas(self, params, image_path):
        """
        Generate tile work areas for a given image and parameters.
        Returns a list of WorkArea objects.
        """
        # Extract tiling parameters
        tile_width = params["tile_width"]
        tile_height = params["tile_height"]
        overlap_width = params["overlap_width"]
        overlap_height = params["overlap_height"]
        margins = params["margins"]
        image_width = params["image_width"]
        image_height = params["image_height"]
        ensure_coverage = self.ensure_coverage_checkbox.isChecked()

        # Calculate usable area inside margins
        left, top, right, bottom = margins
        usable_width = image_width - left - right
        usable_height = image_height - top - bottom
        effective_width = tile_width - overlap_width
        effective_height = tile_height - overlap_height

        tile_work_areas = []
        if ensure_coverage:
            # Calculate number of tiles for regular grid
            num_tiles_x = max(1, int(usable_width / effective_width))
            num_tiles_y = max(1, int(usable_height / effective_height))
            
            # Calculate covered area by the grid
            covered_width = num_tiles_x * effective_width + overlap_width
            covered_height = num_tiles_y * effective_height + overlap_height
            
            # Create tiles for the regular grid
            for i in range(num_tiles_y):
                for j in range(num_tiles_x):
                    x = left + j * effective_width
                    y = top + i * effective_height
                    
                    # Ensure tile doesn't exceed image boundaries
                    if x + tile_width > image_width or y + tile_height > image_height:
                        continue
                        
                    tile_work_area = WorkArea(
                        x, y, tile_width, tile_height, image_path
                    )
                    tile_work_areas.append(tile_work_area)
                    
            # Add extra column of tiles at the right edge if needed
            if covered_width < usable_width:
                # Calculate right edge position ensuring tile fits within image
                right_edge = min(left + usable_width - tile_width, image_width - tile_width)
                
                # Only add right edge tiles if they would be valid
                if right_edge >= left and right_edge + tile_width <= image_width:
                    for i in range(num_tiles_y):
                        y = top + i * effective_height
                        
                        # Ensure tile doesn't exceed image boundaries
                        if y + tile_height > image_height:
                            continue
                            
                        tile_work_area = WorkArea(
                            right_edge, y, tile_width, tile_height, image_path
                        )
                        tile_work_areas.append(tile_work_area)
                    
            # Add extra row of tiles at the bottom edge if needed
            if covered_height < usable_height:
                # Calculate bottom edge position ensuring tile fits within image
                bottom_edge = min(top + usable_height - tile_height, image_height - tile_height)
                
                # Only add bottom edge tiles if they would be valid
                if bottom_edge >= top and bottom_edge + tile_height <= image_height:
                    for j in range(num_tiles_x):
                        x = left + j * effective_width
                        
                        # Ensure tile doesn't exceed image boundaries
                        if x + tile_width > image_width:
                            continue
                            
                        tile_work_area = WorkArea(
                            x, bottom_edge, tile_width, tile_height, image_path
                        )
                        tile_work_areas.append(tile_work_area)
                        
                    # Add bottom-right corner tile if both right and bottom need coverage
                    if covered_width < usable_width:
                        # Calculate corner position ensuring tile fits within image
                        right_edge = min(left + usable_width - tile_width, image_width - tile_width)
                        
                        # Only add corner tile if it would be valid
                        if (right_edge >= left and right_edge + tile_width <= image_width and 
                            bottom_edge >= top and bottom_edge + tile_height <= image_height):
                            
                            tile_work_area = WorkArea(
                                right_edge, bottom_edge, tile_width, tile_height, image_path
                            )
                            tile_work_areas.append(tile_work_area)
        else:
            # Standard grid logic (no extra coverage)
            num_tiles_x = max(1, int((usable_width - overlap_width) / effective_width) + 1)
            num_tiles_y = max(1, int((usable_height - overlap_height) / effective_height) + 1)
            for i in range(num_tiles_y):
                for j in range(num_tiles_x):
                    x = left + j * effective_width
                    y = top + i * effective_height
                    
                    # Skip tiles that would exceed the usable area OR image boundaries
                    if (x + tile_width > left + usable_width or y + tile_height > top + usable_height or
                        x + tile_width > image_width or y + tile_height > image_height):
                        continue
                        
                    tile_work_area = WorkArea(
                        x, y, tile_width, tile_height, image_path
                    )
                    tile_work_areas.append(tile_work_area)
                    
        return tile_work_areas

    def apply(self):
        """
        Method called when the Apply button is clicked.
        Applies tiling to all selected images.
        """
        # Get the list of image paths to apply tiling to
        image_paths = self.get_selected_image_paths()
        if not image_paths:
            QMessageBox.warning(self, "No Images", "No images are currently selected.")
            return
    
        total_tiles = 0
        errors = []
    
        # Calculate margin/overlap percentages from current image if needed
        current_image_display_width = self.annotation_window.pixmap_image.width()
        current_image_display_height = self.annotation_window.pixmap_image.height()
    
        # Get the tile size from the input fields, as this is constant across images
        input_tile_width, input_tile_height = self.tile_size_input.get_value()
    
        # Overlap
        if self.overlap_input.value_type.currentIndex() == 0:  # Pixels
            overlap_width_px = self.overlap_input.width_spin.value()
            overlap_height_px = self.overlap_input.height_spin.value()
            # Convert pixel overlap to percentage of INPUT TILE SIZE
            overlap_width_pct = overlap_width_px / input_tile_width if input_tile_width else 0
            overlap_height_pct = overlap_height_px / input_tile_height if input_tile_height else 0
        else:  # Percentage (already a percentage of tile size)
            overlap_width_pct = self.overlap_input.width_double.value()
            overlap_height_pct = self.overlap_input.height_double.value()
    
        # Margin
        if self.margins_input.value_type.currentIndex() == 0:  # Pixels
            margins_px = self.margins_input.get_margins(current_image_display_width, 
                                                        current_image_display_height, 
                                                        validate=False)
            
            if isinstance(margins_px, tuple) and len(margins_px) == 4:
                left_pct = margins_px[0] / current_image_display_width if current_image_display_width else 0
                top_pct = margins_px[1] / current_image_display_height if current_image_display_height else 0
                right_pct = margins_px[2] / current_image_display_width if current_image_display_width else 0
                bottom_pct = margins_px[3] / current_image_display_height if current_image_display_height else 0
            else:
                if current_image_display_width:
                    left_pct = margins_px[0] / current_image_display_width
                    right_pct = margins_px[2] / current_image_display_width
                else:
                    left_pct = right_pct = 0
                if current_image_display_height:
                    top_pct = margins_px[1] / current_image_display_height
                    bottom_pct = margins_px[3] / current_image_display_height
                else:
                    top_pct = bottom_pct = 0
                
        else:  # Percentage
            margins = self.margins_input.get_margins(current_image_display_width, 
                                                     current_image_display_height, 
                                                     validate=False)
            if isinstance(margins, tuple) and len(margins) == 4:
                left_pct, top_pct, right_pct, bottom_pct = margins
            else:
                left_pct = top_pct = right_pct = bottom_pct = margins[0]
    
        for image_path in image_paths:
            # For the current image, use the annotation window for validation
            if image_path == self.annotation_window.current_image_path:
                is_valid, error_message, params = self.validate_parameters(image_path)
            else:
                # For other images, get dimensions from the raster manager
                raster = self.main_window.image_window.raster_manager.get_raster(image_path)
                if not raster or not hasattr(raster, 'width') or not hasattr(raster, 'height'):
                    errors.append(f"{image_path}: Could not get image dimensions.")
                    continue
    
                # Extract image dimensions
                image_width = raster.width
                image_height = raster.height
                try:
                    # Get tile size and check if it fits in the image
                    tile_width, tile_height = self.tile_size_input.get_value()
                    if tile_width > image_width or tile_height > image_height:
                        errors.append(f"{image_path}: Tile size exceeds image size ({image_width}×{image_height}).")
                        continue
    
                    # Always use percentage-based overlap/margins for other images
                    # Now correctly calculate overlap based on tile size, not image size
                    overlap_width = int(overlap_width_pct * tile_width)
                    overlap_height = int(overlap_height_pct * tile_height)
                    margins = (
                        int(left_pct * image_width),
                        int(top_pct * image_height),
                        int(right_pct * image_width),
                        int(bottom_pct * image_height),
                    )
    
                    # Calculate effective tile size
                    effective_width = tile_width - overlap_width
                    effective_height = tile_height - overlap_height
                    if effective_width <= 0 or effective_height <= 0:
                        errors.append(f"{image_path}: Effective tile size must be positive. Reduce overlap.")
                        continue
    
                    left, top, right, bottom = margins
                    if left + right >= image_width or top + bottom >= image_height:
                        errors.append(f"{image_path}: Margins are too large for the image size.")
                        continue
    
                    # Prepare parameters dict for tiling
                    params = {
                        "tile_width": tile_width,
                        "tile_height": tile_height,
                        "overlap_width": overlap_width,
                        "overlap_height": overlap_height,
                        "margins": margins,
                        "image_width": image_width,
                        "image_height": image_height,
                    }
                    is_valid = True
                    error_message = ""
    
                except Exception as e:
                    errors.append(f"{image_path}: Invalid parameters: {e}")
                    continue
    
            # If parameters are not valid, record the error and skip
            if not is_valid:
                errors.append(f"{image_path}: {error_message}")
                continue
    
            # Generate tile work areas for this image
            tile_work_areas = self.generate_tile_work_areas(params, image_path)
            raster = self.main_window.image_window.raster_manager.get_raster(image_path)
    
            if not raster:
                errors.append(f"{image_path}: Could not get raster.")
                continue
    
            # Add each tile work area to the raster
            for work_area in tile_work_areas:
                raster.add_work_area(work_area)
            total_tiles += len(tile_work_areas)
    
        # Show summary message
        QMessageBox.information(self, "Tiles Added", f"Added {total_tiles} tiles to selected images.")
    
        # Show dialog if any images were skipped due to errors
        if errors:
            self.show_skipped_images_dialog(errors)
    
        # Clear any previewed tiles and close the dialog
        self.clear_tiles()
        self.accept()
