import warnings

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtWidgets import (QMessageBox, QVBoxLayout, QLabel, QDialog, QDialogButtonBox, 
                             QGroupBox, QPushButton, QHBoxLayout, QCheckBox,
                             QTableWidget, QTableWidgetItem, QApplication)

from coralnet_toolbox.QtWorkArea import WorkArea

from coralnet_toolbox.Common.QtTileSizeInput import TileSizeInput
from coralnet_toolbox.Common.QtOverlapInput import OverlapInput
from coralnet_toolbox.Common.QtMarginInput import MarginInput

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class TileManager(QDialog):
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
        
        self.animation_manager = main_window.animation_manager

        self.setWindowIcon(get_icon("tile.png"))
        self.setWindowTitle("Tile Manager")
        self.resize(400, 600)
        
        # Keep dialog on top while user is working
        from PyQt5.QtCore import Qt
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        
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
        
        # Initialize status labels (will be added after buttons in setup_buttons_layout)
        self.status_label = QLabel("No images highlighted")
        self.status_label.setAlignment(Qt.AlignLeft)
        
        self.tiles_status_label = QLabel("No tiles previewed")
        self.tiles_status_label.setAlignment(Qt.AlignLeft)
        
        # Buttons at bottom
        self.setup_buttons_layout()
        
        # Connect to table model signals to update highlighted count
        self.image_window.table_model.rowsChanged.connect(self.update_status_label)
        
        # Connect to image selection changes to update preview when user switches images
        self.image_window.imageSelected.connect(self.on_image_changed)
        
    def showEvent(self, event):
        """Handle dialog show event."""
        super().showEvent(event)
        
        # Automatically highlight the current image if one is loaded
        current_image_path = self.annotation_window.current_image_path
        if current_image_path:
            # Check if current image is already highlighted
            highlighted_paths = self.image_window.table_model.get_highlighted_paths()
            if current_image_path not in highlighted_paths:
                # Highlight only the current image
                self.image_window.table_model.set_highlighted_paths([current_image_path])
        
        self.update_tile_size_limits()
        self.update_status_label()
        self.clear_tiles()
        self.preview_tiles()

    def closeEvent(self, event):
        """Handle dialog close event."""
        self.clear_tiles()
        event.accept()

    def reject(self):
        """Handle dialog rejection."""
        self.clear_tiles()
        super().reject()

    def update_status_label(self):
        """Update the status label to show the number of images highlighted."""
        highlighted_paths = self.image_window.table_model.get_highlighted_paths()
        count = len(highlighted_paths)
        if count == 0:
            self.status_label.setText("No images highlighted")
        elif count == 1:
            self.status_label.setText("1 image highlighted")
        else:
            self.status_label.setText(f"{count} images highlighted")

    def on_image_changed(self, image_path):
        """Handle when the user changes the selected image in the ImageWindow.
        
        When annotation_window.set_image() is called, it clears the scene, which removes
        all graphics. We need to clear our stale references and then regenerate the preview
        if one was being shown.
        """
        # Store whether we had preview graphics before the image changed
        had_preview = len(self.all_graphics) > 0
        
        # Clear stale graphics references since annotation window just cleared its scene
        self.all_graphics = []
        self.margin_work_area = None
        
        # Only regenerate preview if we had one shown and new image is valid
        if had_preview and image_path and self.annotation_window.active_image:
            self.preview_tiles()

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
        """Set up tile config layout without groupbox."""
        # Tile Size Input
        self.tile_size_input = TileSizeInput()
        self.layout.addWidget(self.tile_size_input)

        # Overlap Input
        self.overlap_input = OverlapInput()
        self.layout.addWidget(self.overlap_input)

        # Margins Input
        self.margins_input = MarginInput()
        self.layout.addWidget(self.margins_input)

    def setup_buttons_layout(self):
        """
        Set up the layout with buttons and status label.
        """
        buttons_layout = QHBoxLayout()
        
        # Preview button
        self.preview_button = QPushButton("Update Preview")
        self.preview_button.clicked.connect(self.preview_tiles)
        buttons_layout.addWidget(self.preview_button)
        
        # Clear button
        self.clear_button = QPushButton("Clear Preview")
        self.clear_button.clicked.connect(self.clear_tiles)
        buttons_layout.addWidget(self.clear_button)
        
        # Add button layout first
        self.layout.addLayout(buttons_layout)
        
        # Create a second row for delete buttons
        delete_buttons_layout = QHBoxLayout()
        
        # Delete tiles for current image button
        self.delete_current_button = QPushButton("Delete Tiles (Current Image)")
        self.delete_current_button.clicked.connect(self.delete_tiles_current_image)
        # Red text to indicate destructive action
        self.delete_current_button.setStyleSheet("QPushButton { color: #d32f2f; }")
        delete_buttons_layout.addWidget(self.delete_current_button)
        
        # Delete tiles for all images button
        self.delete_all_button = QPushButton("Delete Tiles (All Images)")
        self.delete_all_button.clicked.connect(self.delete_tiles_all_images)
        # Red text to indicate destructive action
        self.delete_all_button.setStyleSheet("QPushButton { color: #d32f2f; }")
        delete_buttons_layout.addWidget(self.delete_all_button)
        
        self.layout.addLayout(delete_buttons_layout)
        
        # Add status labels below action buttons, above apply/cancel
        # Tiles status on top, highlighted images count on bottom
        self.layout.addWidget(self.tiles_status_label)
        self.layout.addWidget(self.status_label)
        
        # Create a button box with custom buttons
        button_box = QDialogButtonBox()
        apply_button = QPushButton("Apply")
        cancel_button = QPushButton("Cancel")

        button_box.addButton(apply_button, QDialogButtonBox.AcceptRole)
        button_box.addButton(cancel_button, QDialogButtonBox.RejectRole)

        button_box.accepted.connect(self.apply)
        button_box.rejected.connect(self.reject)

        # Add the standard button box
        self.layout.addWidget(button_box)
        
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
        Get the selected image paths - only highlighted rows.

        :return: List of highlighted image paths
        """
        # Get highlighted image paths from the table model
        return self.image_window.table_model.get_highlighted_paths()

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
        
        # Create a margin work area to show the boundary with shadow
        self.margin_work_area = WorkArea(
            left, top, usable_width, usable_height,
            self.annotation_window.current_image_path
        )
        
        self.margin_work_area.set_animation_manager(self.animation_manager)
        
        # Set a different color for the margin boundary
        self.margin_work_area.work_area_pen = QPen(QColor(230, 62, 0), 2, Qt.DashLine)  # blood red
        
        # Create graphics with shadow to highlight the working area
        margin_graphics = self.margin_work_area.create_graphics(
            self.annotation_window.scene, 
            include_shadow=True  # Add shadow to highlight the usable area
        )
        
        self.all_graphics.append(margin_graphics)
        
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
                    
                    tile_work_area.set_animation_manager(self.animation_manager)
                    
                    # Add to scene with thinner line and store the graphics
                    tile_graphics = tile_work_area.create_graphics(
                        self.annotation_window.scene, 
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
                        
                        tile_work_area.set_animation_manager(self.animation_manager)
                        
                        # Add to scene with thinner line and store the graphics
                        tile_graphics = tile_work_area.create_graphics(
                            self.annotation_window.scene, 
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
                        
                        tile_work_area.set_animation_manager(self.animation_manager)
                        
                        # Add to scene with thinner line and store the graphics
                        tile_graphics = tile_work_area.create_graphics(
                            self.annotation_window.scene, 
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
                            
                            tile_work_area.set_animation_manager(self.animation_manager)
                            
                            # Add to scene with thinner line and store the graphics
                            tile_graphics = tile_work_area.create_graphics(
                                self.annotation_window.scene, 
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
                    
                    tile_work_area.set_animation_manager(self.animation_manager)
                    
                    # Add to scene with thinner line and store the graphics
                    tile_graphics = tile_work_area.create_graphics(
                        self.annotation_window.scene, 
                    )
                    
                    # Store the graphics for later removal
                    self.all_graphics.append(tile_graphics)
                    
                    # Store the work area
                    self.tile_work_areas.append(tile_work_area)
        
        # Count tiles
        total_tiles = len(self.tile_work_areas)
        
        # Update tile status label
        coverage_status = "with full coverage" if ensure_coverage else "with standard grid"
        self.tiles_status_label.setText(
            f"Tiles: {total_tiles} ({num_tiles_x}×{num_tiles_y} grid, {coverage_status}) | "
            f"Size: {tile_width}×{tile_height} | Overlap: {overlap_width}×{overlap_height}"
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
        
        # Reset tile status
        self.tiles_status_label.setText("No tiles previewed")

    def delete_tiles_current_image(self):
        """Delete all existing tile work areas for the current image."""
        # Get the current image path
        current_image_path = self.annotation_window.current_image_path
        if not current_image_path:
            QMessageBox.warning(self, "No Image", "No image is currently loaded.")
            return
        
        # Get the raster for the current image
        raster = self.main_window.image_window.raster_manager.get_raster(current_image_path)
        if not raster:
            QMessageBox.warning(self, "Error", "Could not access image data.")
            return
        
        # Check if there are any work areas to delete
        if not raster.has_work_areas():
            QMessageBox.information(self, 
                                    "No Tiles", 
                                    "There are no tiles to delete for the current image.")
            return
        
        # Confirm deletion
        tile_count = len(raster.get_work_areas())
        reply = QMessageBox.question(
            self, "Confirm Deletion", 
            f"Are you sure you want to delete {tile_count} tile(s) from the current image?\n\n"
            f"This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Clear all work areas from the raster
            raster.clear_work_areas()
            
            # Clear any preview tiles from the scene as well
            self.clear_tiles()
            
            QMessageBox.information(self, 
                                    "Tiles Deleted", 
                                    f"Successfully deleted {tile_count} tile(s) from the current image.")
    
    def delete_tiles_all_images(self):
        """Delete all existing tile work areas for all images in the project."""
        # Get all rasters from the raster manager
        raster_manager = self.main_window.image_window.raster_manager
        all_rasters = [raster_manager.get_raster(path) for path in raster_manager.image_paths]
        
        # Count total tiles across all images
        total_tiles = 0
        images_with_tiles = 0
        for raster in all_rasters:
            if raster and raster.has_work_areas():
                total_tiles += len(raster.get_work_areas())
                images_with_tiles += 1
        
        # Check if there are any tiles to delete
        if total_tiles == 0:
            QMessageBox.information(self, 
                                    "No Tiles", 
                                    "There are no tiles to delete in any images.")
            return
        
        # Confirm deletion
        reply = QMessageBox.question(
            self, "Confirm Deletion", 
            f"Are you sure you want to delete {total_tiles} tile(s) from "
            f"{images_with_tiles} image(s)?\n\n"
            f"This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Clear work areas from all rasters
            deleted_count = 0
            for raster in all_rasters:
                if raster and raster.has_work_areas():
                    deleted_count += len(raster.get_work_areas())
                    raster.clear_work_areas()
            
            # Clear any preview tiles from the scene as well
            self.clear_tiles()
            
            QMessageBox.information(self, 
                                    "Tiles Deleted", 
                                    f"Successfully deleted {deleted_count} tile(s) from "
                                    f"{images_with_tiles} image(s).")

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
        Applies tiling to all highlighted images.
        """
        # Get the list of image paths to apply tiling to (highlighted rows only)
        image_paths = self.get_selected_image_paths()
        if not image_paths:
            msg = "Please highlight at least one image row to apply tiles to highlighted images."
            QMessageBox.warning(self, "No Selection", msg)
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
                
        # Create a progress bar
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Applying Tile Work Areas")
        progress_bar.show()
        progress_bar.start_progress(len(image_paths))
    
        for image_path in image_paths:
            # Update progress bar first
            progress_bar.update_progress()
            
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
            
        # Restore cursor
        QApplication.restoreOverrideCursor()
        progress_bar.finish_progress()
        progress_bar.stop_progress()
        progress_bar.close()
    
        # Show dialog if any images were skipped due to errors
        if errors:
            self.show_skipped_images_dialog(errors)
    
        # Clear any previewed tiles and close the dialog
        self.clear_tiles()
        self.accept()
