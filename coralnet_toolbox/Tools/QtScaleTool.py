import warnings

from PyQt5.QtCore import Qt, QLineF
from PyQt5.QtGui import QMouseEvent, QPen, QColor
from PyQt5.QtWidgets import (QDialog, QWidget, QVBoxLayout, QFormLayout, 
                             QDoubleSpinBox, QComboBox, QLabel,
                             QDialogButtonBox, QMessageBox, QGraphicsLineItem,
                             QGroupBox, QPushButton, QTabWidget)

from coralnet_toolbox.Tools.QtTool import Tool
from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# ScaleToolDialog Class
# ----------------------------------------------------------------------------------------------------------------------


class ScaleToolDialog(QDialog):
    """
    A modeless dialog for the ScaleTool, allowing user input for scale calculation
    and propagation, including Z-channel calibration.
    """
    def __init__(self, tool, parent=None):
        super().__init__(parent)
        # Get references from the tool
        self.tool = tool
        self.annotation_window = self.tool.annotation_window
        self.main_window = self.annotation_window.main_window

        self.setWindowTitle("Scale Tool")
        self.setWindowIcon(get_icon("scale.png"))
        self.setMinimumWidth(450)
        
        # This dialog is modeless
        self.setModal(False) 

        self.main_layout = QVBoxLayout(self)

        # --- Create Tab Widget ---
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.scale_tab = self.create_scale_tab()
        self.z_scale_tab = self.create_z_scale_tab()
        self.z_anchor_tab = self.create_z_anchor_tab()
        
        # Add tabs to tab widget
        self.tab_widget.addTab(self.scale_tab, "Set Scale")
        self.tab_widget.addTab(self.z_scale_tab, "Z-Scale")
        self.tab_widget.addTab(self.z_anchor_tab, "Z-Anchor")
        
        # Connect tab change signal
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        self.main_layout.addWidget(self.tab_widget)

        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Close)
        
        # Rename "Apply" to "Set Scale" for clarity
        self.apply_button = self.button_box.button(QDialogButtonBox.Apply)
        self.apply_button.setText("Apply")
        
        self.main_layout.addWidget(self.button_box)
        
        # --- Status Label (at bottom of dialog) ---
        self.status_label = QLabel("No images highlighted")
        self.status_label.setAlignment(Qt.AlignLeft)
        self.main_layout.addWidget(self.status_label)
        
        # Signal connection will be made in activate() when image_window is guaranteed to exist
        self._signal_connected = False
        
        # Track current interaction mode
        self.current_mode = None  # 'xy_scale', 'z_scale', or 'z_anchor'

    def create_scale_tab(self):
        """Create the 'Set Scale' tab for XY calibration."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add Information groupbox at top
        info_groupbox = QGroupBox("Information")
        info_layout = QVBoxLayout()
        instruction_label = QLabel(
            "Draw a line across a known distance to calibrate image scale.\n\n"
            "Instructions:\n"
            "  1. Select the measurement unit (mm, cm, m, etc.)\n"
            "  2. Enter the known real-world length of your reference object\n"
            "  3. Click once on the image to start drawing a line\n"
            "  4. Click again to finish the line across the reference object\n"
            "  5. Click 'Apply' to calibrate the highlighted images\n\n"
            "Tips: Use scale bars or known objects for best accuracy."
        )
        instruction_label.setWordWrap(True)
        instruction_label.setStyleSheet("color: #333; font-size: 10pt;")
        info_layout.addWidget(instruction_label)
        info_groupbox.setLayout(info_layout)
        layout.addWidget(info_groupbox)
        
        scale_layout = QFormLayout()
        
        # Units at top
        self.units_combo = QComboBox()
        self.units_combo.addItems(["mm", "cm", "m", "km", "in", "ft", "yd", "mi"])
        self.units_combo.setCurrentText("m")
        scale_layout.addRow("Units:", self.units_combo)
        
        # Known length
        self.known_length_input = QDoubleSpinBox()
        self.known_length_input.setRange(0.001, 1000000.0)
        self.known_length_input.setValue(1.0)
        self.known_length_input.setDecimals(3)
        scale_layout.addRow("Known Length:", self.known_length_input)

        # Pixel length (measured from drawn line)
        self.pixel_length_label = QLabel("Draw a line on the image")
        scale_layout.addRow("Pixel Length:", self.pixel_length_label)
        
        # Calculated scale
        self.calculated_scale_label = QLabel("N/A")
        scale_layout.addRow("Calculated Scale:", self.calculated_scale_label)
        
        layout.addLayout(scale_layout)
        
        # Clear Line button
        self.clear_line_button = QPushButton("Clear Line")
        self.clear_line_button.clicked.connect(self.clear_scale_line)
        layout.addWidget(self.clear_line_button)
        
        # --- Danger Zone (Collapsible) ---
        self.danger_zone_group_box = QGroupBox("Danger Zone")
        self.danger_zone_group_box.setCheckable(True)
        self.danger_zone_group_box.setChecked(False)

        danger_zone_container = QWidget()
        danger_zone_layout = QVBoxLayout(danger_zone_container)
        danger_zone_layout.setContentsMargins(0, 0, 0, 0)

        self.remove_highlighted_button = QPushButton("Remove Scale from Highlighted Images")
        self.remove_highlighted_button.setStyleSheet(
            "background-color: #D9534F; color: white; font-weight: bold;"
        )
        danger_zone_layout.addWidget(self.remove_highlighted_button)

        group_layout = QVBoxLayout()
        group_layout.addWidget(danger_zone_container)
        self.danger_zone_group_box.setLayout(group_layout)

        self.danger_zone_group_box.toggled.connect(danger_zone_container.setVisible)
        danger_zone_container.setVisible(False)

        layout.addWidget(self.danger_zone_group_box)
        layout.addStretch()
        
        return tab

    def create_z_scale_tab(self):
        """Create the 'Z-Scale' tab for vertical scale calibration."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add Information groupbox at top
        info_groupbox = QGroupBox("Information")
        info_layout = QVBoxLayout()
        instruction_label = QLabel(
            "Adjust vertical scale by drawing a line across a feature with known height/depth difference.\n\n"
            "Instructions:\n"
            "  1. Select the measurement unit for vertical measurements\n"
            "  2. Enter the known real-world vertical difference (height or depth)\n"
            "  3. Draw a line from bottom to top of a feature with known height\n"
            "  4. The tool will calculate a scalar multiplier for all Z-values\n"
            "  5. Click 'Apply' to calibrate the highlighted images\n\n"
            "Tips: Draw vertical lines (<45° from vertical) for best results.\n"
            "This adjusts RELATIVE accuracy (shape is correct, size is wrong)."
        )
        instruction_label.setWordWrap(True)
        instruction_label.setStyleSheet("color: #333; font-size: 10pt;")
        info_layout.addWidget(instruction_label)
        info_groupbox.setLayout(info_layout)
        layout.addWidget(info_groupbox)
        
        scale_layout = QFormLayout()
        
        # Units at top
        self.z_units_combo = QComboBox()
        self.z_units_combo.addItems(["mm", "cm", "m", "km", "in", "ft", "yd", "mi"])
        self.z_units_combo.setCurrentText("m")
        scale_layout.addRow("Units:", self.z_units_combo)
        
        # Known Z-difference
        self.z_known_difference_input = QDoubleSpinBox()
        self.z_known_difference_input.setRange(0.001, 1000000.0)
        self.z_known_difference_input.setValue(1.0)
        self.z_known_difference_input.setDecimals(3)
        scale_layout.addRow("Known Z-Difference:", self.z_known_difference_input)

        # Raw Z-difference (measured from line)
        self.z_raw_difference_label = QLabel("Draw a line on the image")
        scale_layout.addRow("Raw Z-Difference:", self.z_raw_difference_label)
        
        # Calculated scalar
        self.z_calculated_scalar_label = QLabel("N/A")
        scale_layout.addRow("Calculated Scalar:", self.z_calculated_scalar_label)
        
        layout.addLayout(scale_layout)
        
        # Clear Line button
        self.clear_z_line_button = QPushButton("Clear Line")
        self.clear_z_line_button.clicked.connect(self.clear_z_scale_line)
        layout.addWidget(self.clear_z_line_button)
        
        # --- Danger Zone (Collapsible) ---
        self.z_scale_danger_zone = QGroupBox("Danger Zone")
        self.z_scale_danger_zone.setCheckable(True)
        self.z_scale_danger_zone.setChecked(False)

        danger_zone_container = QWidget()
        danger_zone_layout = QVBoxLayout(danger_zone_container)
        danger_zone_layout.setContentsMargins(0, 0, 0, 0)

        self.reset_z_scalar_button = QPushButton("Reset Z-Scalar to Default (1.0)")
        self.reset_z_scalar_button.setStyleSheet(
            "background-color: #D9534F; color: white; font-weight: bold;"
        )
        danger_zone_layout.addWidget(self.reset_z_scalar_button)

        group_layout = QVBoxLayout()
        group_layout.addWidget(danger_zone_container)
        self.z_scale_danger_zone.setLayout(group_layout)

        self.z_scale_danger_zone.toggled.connect(danger_zone_container.setVisible)
        danger_zone_container.setVisible(False)

        layout.addWidget(self.z_scale_danger_zone)
        layout.addStretch()
        
        return tab

    def create_z_anchor_tab(self):
        """Create the 'Z-Anchor' tab for datum offset calibration."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add Information groupbox at top
        info_groupbox = QGroupBox("Information")
        info_layout = QVBoxLayout()
        instruction_label = QLabel(
            "Set the absolute depth/elevation by clicking a reference point with known Z-value.\n\n"
            "Instructions:\n"
            "  1. Enter the known Z-value at your reference point (e.g., seafloor depth)\n"
            "  2. Click 'Tare to Zero' for quick zero-referencing (optional)\n"
            "  3. Click anywhere on the image to set an anchor point\n"
            "  4. The tool will calculate an offset to shift all Z-values\n"
            "  5. Select the Z-direction convention (depth vs elevation)\n"
            "  6. Click 'Apply' to calibrate the highlighted images\n\n"
            "Tips: Use known reference points like seafloor depth or surface elevation.\n"
            "This adjusts ABSOLUTE accuracy (sets the zero point or datum)."
        )
        instruction_label.setWordWrap(True)
        instruction_label.setStyleSheet("color: #333; font-size: 10pt;")
        info_layout.addWidget(instruction_label)
        info_groupbox.setLayout(info_layout)
        layout.addWidget(info_groupbox)
        
        anchor_layout = QFormLayout()
        
        # Current Z-value (at clicked point)
        self.z_current_value_label = QLabel("Click on the image")
        anchor_layout.addRow("Current Z-Value:", self.z_current_value_label)
        
        # Target Z-value input
        self.z_target_value_input = QDoubleSpinBox()
        self.z_target_value_input.setRange(-100000.0, 100000.0)
        self.z_target_value_input.setValue(0.0)
        self.z_target_value_input.setDecimals(3)
        anchor_layout.addRow("Target Z-Value:", self.z_target_value_input)
        
        # Tare button
        self.z_tare_button = QPushButton("Tare to Zero")
        self.z_tare_button.clicked.connect(self.tare_to_zero)
        anchor_layout.addRow("", self.z_tare_button)
        
        # Calculated offset
        self.z_calculated_offset_label = QLabel("N/A")
        anchor_layout.addRow("Calculated Offset:", self.z_calculated_offset_label)
        
        # Direction toggle
        self.z_direction_combo = QComboBox()
        self.z_direction_combo.addItem("Depth (positive down)", 1)
        self.z_direction_combo.addItem("Elevation (positive up)", -1)
        self.z_direction_combo.setCurrentIndex(0)
        anchor_layout.addRow("Direction:", self.z_direction_combo)
        
        # Invert Direction button
        self.z_invert_direction_button = QPushButton("Invert Direction")
        self.z_invert_direction_button.clicked.connect(self.invert_z_direction)
        anchor_layout.addRow("", self.z_invert_direction_button)
        
        layout.addLayout(anchor_layout)
        
        # --- Danger Zone (Collapsible) ---
        self.z_anchor_danger_zone = QGroupBox("Danger Zone")
        self.z_anchor_danger_zone.setCheckable(True)
        self.z_anchor_danger_zone.setChecked(False)

        danger_zone_container = QWidget()
        danger_zone_layout = QVBoxLayout(danger_zone_container)
        danger_zone_layout.setContentsMargins(0, 0, 0, 0)

        self.reset_z_offset_button = QPushButton("Reset Z-Offset to Default (0.0)")
        self.reset_z_offset_button.setStyleSheet(
            "background-color: #D9534F; color: white; font-weight: bold;"
        )
        danger_zone_layout.addWidget(self.reset_z_offset_button)

        group_layout = QVBoxLayout()
        group_layout.addWidget(danger_zone_container)
        self.z_anchor_danger_zone.setLayout(group_layout)

        self.z_anchor_danger_zone.toggled.connect(danger_zone_container.setVisible)
        danger_zone_container.setVisible(False)

        layout.addWidget(self.z_anchor_danger_zone)
        layout.addStretch()
        
        return tab

    def on_tab_changed(self, index):
        """Handle tab changes to update interaction mode."""
        # Clear any current drawings when switching tabs
        if hasattr(self.tool, 'stop_current_drawing'):
            self.tool.stop_current_drawing()
        
        # Reset fields for the tab being left
        self.reset_fields()
        
        # Update mode based on tab
        if index == 0:
            self.current_mode = 'xy_scale'
            self.apply_button.setText("Apply")
        elif index == 1:
            self.current_mode = 'z_scale'
            self.apply_button.setText("Apply")
        elif index == 2:
            self.current_mode = 'z_anchor'
            self.apply_button.setText("Apply")
        
        # Update tab states
        self.update_z_tab_states()
    
    def update_z_tab_states(self):
        """Enable/disable Z tabs based on whether current image has Z-channel."""
        current_raster = self.main_window.image_window.current_raster
        has_z_channel = current_raster is not None and current_raster.z_channel is not None
        
        # Enable/disable Z tabs
        self.tab_widget.setTabEnabled(1, has_z_channel)  # Z-Scale tab
        self.tab_widget.setTabEnabled(2, has_z_channel)  # Z-Anchor tab
        
        # If current tab is disabled, switch to first tab
        if not has_z_channel and self.tab_widget.currentIndex() > 0:
            self.tab_widget.setCurrentIndex(0)
    
    def tare_to_zero(self):
        """Set target value to 0.0 (tare button)."""
        self.z_target_value_input.setValue(0.0)
    
    def invert_z_direction(self):
        """Invert the Z-direction for all highlighted images."""
        highlighted_paths = self.get_selected_image_paths()
        
        if not highlighted_paths:
            QMessageBox.warning(self, "No Images Selected",
                                "Please highlight at least one image to invert direction.")
            return
        
        # Confirmation dialog
        reply = QMessageBox.question(
            self,
            "Confirm Direction Inversion",
            f"Invert Z-direction (Depth ↔ Elevation) for {len(highlighted_paths)} image(s)?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Invert direction for each highlighted image
        raster_manager = self.main_window.image_window.raster_manager
        
        for image_path in highlighted_paths:
            raster = raster_manager.get_raster(image_path)
            if raster and raster.z_channel is not None:
                # Invert the direction: 1 -> -1, -1 -> 1
                current_direction = raster.z_settings.get('direction', 1)
                raster.z_settings['direction'] = -current_direction
                raster_manager.rasterUpdated.emit(image_path)
        
        # Refresh Z-channel visualization if current image was affected
        current_image_path = self.tool.annotation_window.current_image_path
        if current_image_path in highlighted_paths:
            self.tool.annotation_window.refresh_z_channel_visualization()
        
        QMessageBox.information(self, "Direction Inverted",
                              f"Z-direction inverted for {len(highlighted_paths)} image(s).")
    
    def clear_scale_line(self):
        """Clear the XY scale line and reset measurements."""
        if hasattr(self.tool, 'stop_current_drawing'):
            self.tool.stop_current_drawing()
        self.pixel_length_label.setText("Draw a line on the image")
        # Reload existing scale instead of showing N/A
        if hasattr(self.tool, 'load_existing_scale'):
            self.tool.load_existing_scale()
        else:
            self.calculated_scale_label.setText("N/A")
    
    def clear_z_scale_line(self):
        """Clear the Z-scale line and reset measurements."""
        if hasattr(self.tool, 'stop_current_drawing'):
            self.tool.stop_current_drawing()
        self.z_raw_difference_label.setText("Draw a line on the image")
        self.z_calculated_scalar_label.setText("N/A")

    def get_selected_image_paths(self):
        """
        Get the selected image paths - only highlighted rows.
        
        :return: List of highlighted image paths
        """
        # Get highlighted image paths from the table model
        return self.main_window.image_window.table_model.get_highlighted_paths()

    def reset_fields(self):
        """Resets the dialog fields to their default state based on current tab."""
        # XY Scale tab
        self.pixel_length_label.setText("Draw a line on the image")
        self.calculated_scale_label.setText("N/A")
        
        # Z-Scale tab
        self.z_raw_difference_label.setText("Draw a line on the image")
        self.z_calculated_scalar_label.setText("N/A")
        
        # Z-Anchor tab
        self.z_current_value_label.setText("Click on the image")
        self.z_calculated_offset_label.setText("N/A")

    def update_status_label(self):
        """Update the status label to show the number of images highlighted."""
        highlighted_paths = self.main_window.image_window.table_model.get_highlighted_paths()
        count = len(highlighted_paths)
        if count == 0:
            self.status_label.setText("No images highlighted")
        elif count == 1:
            self.status_label.setText("1 image highlighted")
        else:
            self.status_label.setText(f"{count} images highlighted")

    def closeEvent(self, event):
        """
        Handle the dialog close event (e.g., user clicks 'X').
        This clears any drawings and deactivates the tool.
        """
        # Clear any current drawings
        if hasattr(self.tool, 'stop_current_drawing'):
            self.tool.stop_current_drawing()
        
        # Deactivate the tool
        self.tool.deactivate()
        event.accept()


# ----------------------------------------------------------------------------------------------------------------------
# ScaleTool Class
# ----------------------------------------------------------------------------------------------------------------------


class ScaleTool(Tool):
    """
    Tool for setting image scale and Z-channel calibration.
    """
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.cursor = Qt.CrossCursor
        self.dialog = ScaleToolDialog(self, self.annotation_window)
        
        # --- Button Connections ---
        apply_btn = self.dialog.button_box.button(QDialogButtonBox.Apply)
        apply_btn.clicked.connect(self.handle_apply)
        
        close_btn = self.dialog.button_box.button(QDialogButtonBox.Close)
        close_btn.clicked.connect(self.deactivate)

        # XY Scale tab connections
        self.dialog.remove_highlighted_button.clicked.connect(self.remove_scale_highlighted)
        
        # Z-Scale tab connections
        self.dialog.reset_z_scalar_button.clicked.connect(self.reset_z_scalar)
        
        # Z-Anchor tab connections
        self.dialog.reset_z_offset_button.clicked.connect(self.reset_z_offset)

        # --- Drawing State ---
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        self.pixel_length = 0.0
        
        # Z-Anchor state
        self.z_anchor_point = None

        # --- Graphics Items ---
        # Line for scale setting and Z-scale
        self.preview_line = QGraphicsLineItem()
        pen = QPen(QColor(230, 62, 0), 3, Qt.DashLine)  # Blood red dashed line
        pen.setCosmetic(True)  # Make pen width independent of zoom level
        self.preview_line.setPen(pen)
        self.preview_line.setZValue(100)

        self.show_crosshair = True  # Enable crosshair for precise measurements

    def load_existing_scale(self):
        """Loads and displays existing scale data for the current image if available."""
        current_path = self.annotation_window.current_image_path
        if not current_path:
            # No image loaded, show N/A
            self.dialog.calculated_scale_label.setText("N/A")
            return
        
        raster = self.main_window.image_window.raster_manager.get_raster(current_path)
        if not raster or raster.scale_x is None:
            # No scale data available
            self.dialog.calculated_scale_label.setText("N/A")
            return
        
        # Display the existing scale
        scale_value = raster.scale_x  # Assuming square pixels
        units = raster.scale_units if raster.scale_units else "metre"
        
        # Convert full unit names to abbreviations for display
        unit_reverse_mapping = {
            'millimetre': 'mm',
            'centimetre': 'cm',
            'metre': 'm',
            'kilometre': 'km',
            'inch': 'in',
            'foot': 'ft',
            'yard': 'yd',
            'mile': 'mi'
        }
        
        # Standardize unit display
        units = unit_reverse_mapping.get(units, units)
        
        # Format the scale text (just the value, no "Scale:" prefix)
        scale_text = f"{scale_value:.6f} {units}/pixel"
        self.dialog.calculated_scale_label.setText(scale_text)

    def activate(self):
        super().activate()
        
        # Add preview line to scene
        if not self.preview_line.scene():
            self.annotation_window.scene.addItem(self.preview_line)
        
        self.stop_current_drawing()
        self.dialog.reset_fields()
        
        # Connect signal to update highlighted count (only once)
        if not self.dialog._signal_connected:
            self.main_window.image_window.table_model.rowsChanged.connect(self.dialog.update_status_label)
            self.main_window.image_window.imageChanged.connect(self.on_image_changed)
            self.dialog._signal_connected = True
        
        # Automatically highlight the current image if one is loaded
        current_image_path = self.annotation_window.current_image_path
        if current_image_path:
            # Check if current image is already highlighted
            highlighted_paths = self.main_window.image_window.table_model.get_highlighted_paths()
            if current_image_path not in highlighted_paths:
                # Highlight only the current image
                self.main_window.image_window.table_model.set_highlighted_paths([current_image_path])
        
        # Update status label with highlighted count
        self.dialog.update_status_label()

        # Load and display existing scale if present
        self.load_existing_scale()
        
        # Update Z-tab states
        self.dialog.update_z_tab_states()

        self.dialog.show()
        self.dialog.activateWindow()

    def deactivate(self):
        # This function is called when another tool is selected
        # or when the dialog's "Close" button is clicked.
        if not self.active:
            return
            
        super().deactivate()
        self.dialog.hide()
        self.preview_line.hide()
        
        self.is_drawing = False
        
        # Untoggle all tools when closing the scale tool
        self.main_window.untoggle_all_tools()

    def stop_current_drawing(self):
        """Stop any active drawing and clear graphics."""
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        self.pixel_length = 0.0
        
        # Remove graphics
        if self.preview_line.scene():
            self.preview_line.hide()

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for starting scale line or Z-anchor point."""
        if event.button() == Qt.LeftButton:
            scene_pos = self.annotation_window.mapToScene(event.pos())
            
            # Handle based on current mode
            if self.dialog.current_mode == 'xy_scale' or self.dialog.current_mode == 'z_scale':
                if not self.is_drawing:
                    # Start new line
                    self.start_point = scene_pos
                    self.end_point = scene_pos
                    self.is_drawing = True
                else:
                    # Finish line
                    self.end_point = scene_pos
                    self.is_drawing = False
                    if self.dialog.current_mode == 'xy_scale':
                        self.calculate_scale()
                    else:
                        self.calculate_z_scale()
                        
            elif self.dialog.current_mode == 'z_anchor':
                # Click to set anchor point
                self.set_z_anchor_point(scene_pos)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for drawing scale line."""
        if self.is_drawing and self.start_point:
            scene_pos = self.annotation_window.mapToScene(event.pos())
            self.end_point = scene_pos
            
            # Update graphics
            line = QLineF(self.start_point, self.end_point)
            self.preview_line.setLine(line)
            self.preview_line.show()
            
            # Update pixel length label
            pixel_length = line.length()
            self.dialog.pixel_length_label.setText(f"{pixel_length:.2f} pixels")

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        pass

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Backspace:
            self.stop_current_drawing()
            self.dialog.pixel_length_label.setText("Draw a line on the image")
            self.dialog.calculated_scale_label.setText("N/A")

    def calculate_scale(self):
        """Calculate the scale based on the drawn line."""
        if not self.start_point or not self.end_point:
            return
        
        line = QLineF(self.start_point, self.end_point)
        pixel_length = line.length()
        
        if pixel_length == 0:
            QMessageBox.warning(self.dialog, 
                                "Invalid Line", 
                                "Please draw a line with non-zero length.")
            self.stop_current_drawing()
            return
        
        # Get known length and units from dialog
        known_length = self.dialog.known_length_input.value()
        units = self.dialog.units_combo.currentText()
        
        # Calculate scale (real-world units per pixel)
        scale = known_length / pixel_length
        
        # Update label (just the value, no "Scale:" prefix)
        scale_text = f"{scale:.6f} {units}/pixel"
        self.dialog.calculated_scale_label.setText(scale_text)
        self.dialog.pixel_length_label.setText(f"{pixel_length:.2f} pixels")

    def apply_scale(self):
        """Apply the calculated scale to highlighted images."""
        # Get highlighted image paths
        highlighted_paths = self.dialog.get_selected_image_paths()
        
        if not highlighted_paths:
            QMessageBox.warning(self.dialog, "No Images Selected",
                              "Please highlight at least one image to apply the scale.")
            return
        
        # Get calculated scale
        scale_text = self.dialog.calculated_scale_label.text()
        if "N/A" in scale_text:
            QMessageBox.warning(self.dialog, "No Scale Set",
                              "Please draw a line and calculate the scale before applying it.")
            return
        
        # Parse scale from label
        parts = scale_text.split()
        if len(parts) < 2:
            QMessageBox.warning(self.dialog, "Invalid Scale",
                              "Could not parse the scale value.")
            return
        
        try:
            scale_value = float(parts[1])
            units_and_ratio = parts[2]
            units = units_and_ratio.split('/')[0]  # Extract unit before '/pixel'
        except (ValueError, IndexError):
            QMessageBox.warning(self.dialog, "Invalid Scale",
                              "Could not parse the scale value.")
            return
        
        # Standardize units to 'metre'
        unit_mapping = {
            'mm': 'millimetre',
            'cm': 'centimetre',
            'm': 'metre',
            'km': 'kilometre',
            'in': 'inch',
            'ft': 'foot',
            'yd': 'yard',
            'mi': 'mile'
        }
        
        scale_units = unit_mapping.get(units, 'metre')
        
        # Apply scale to each highlighted image
        raster_manager = self.main_window.image_window.raster_manager
        current_image_path = self.annotation_window.current_image_path
        current_image_affected = False
        
        for image_path in highlighted_paths:
            raster = raster_manager.get_raster(image_path)
            if raster:
                # Use the proper update_scale method instead of directly setting properties
                raster.update_scale(scale_value, scale_value, scale_units)
                # Emit signal to notify the UI that this raster was updated
                raster_manager.rasterUpdated.emit(image_path)
                
                # Check if the current image is being updated
                if image_path == current_image_path:
                    current_image_affected = True
        
        # If the currently displayed image was updated, refresh the view to show new scale
        if current_image_affected:
            width, height = self.annotation_window.get_image_dimensions()
            if width and height:
                self.main_window.update_view_dimensions(width, height)
        
        QMessageBox.information(self.dialog, "Scale Applied",
                              f"Scale applied to {len(highlighted_paths)} image(s).")
        
        # Reload the scale display to show the newly applied scale
        self.load_existing_scale()
        
        # Clear drawing after applying scale
        self.stop_current_drawing()
        self.dialog.reset_fields()

    def remove_scale_highlighted(self):
        """Remove scale data from highlighted images."""
        highlighted_paths = self.dialog.get_selected_image_paths()
        
        if not highlighted_paths:
            QMessageBox.warning(self.dialog, "No Images Selected",
                              "Please highlight at least one image to remove the scale.")
            return
        
        reply = QMessageBox.question(self.dialog, 
                                     "Confirm Removal",
                                     f"Remove scale from {len(highlighted_paths)} image(s)?",
                                     QMessageBox.Yes | QMessageBox.No)
        
        if reply != QMessageBox.Yes:
            return
        
        # Remove scale from each highlighted image
        raster_manager = self.main_window.image_window.raster_manager
        current_image_path = self.annotation_window.current_image_path
        current_image_affected = False
        
        for image_path in highlighted_paths:
            raster = raster_manager.get_raster(image_path)
            if raster:
                # Use the proper remove_scale method instead of directly setting properties
                raster.remove_scale()
                # Emit signal to notify the UI that this raster was updated
                raster_manager.rasterUpdated.emit(image_path)
                
                # Check if the current image is being updated
                if image_path == current_image_path:
                    current_image_affected = True
        
        # If the currently displayed image was updated, refresh the view to hide scale
        if current_image_affected:
            width, height = self.annotation_window.get_image_dimensions()
            if width and height:
                self.main_window.update_view_dimensions(width, height)
        
        QMessageBox.information(self.dialog, "Scale Removed",
                              f"Scale removed from {len(highlighted_paths)} image(s).")
        
        # Reload the scale display to reflect the removal
        self.load_existing_scale()

    def handle_apply(self):
        """Handle the Apply button click based on current tab/mode."""
        if self.dialog.current_mode == 'xy_scale':
            self.apply_scale()
        elif self.dialog.current_mode == 'z_scale':
            self.apply_z_scale()
        elif self.dialog.current_mode == 'z_anchor':
            self.apply_z_anchor()
    
    def on_image_changed(self):
        """Handle image change to update Z-tab states and load existing data."""
        self.dialog.update_z_tab_states()
        self.load_existing_scale()
        self.stop_current_drawing()
    
    def calculate_z_scale(self):
        """Calculate the Z-scale scalar based on the drawn line."""
        from coralnet_toolbox.utilities import calculate_z_scalar, validate_line_angle
        
        if not self.start_point or not self.end_point:
            return
        
        # Validate line angle
        is_valid, angle_deg, vertical_component, warning_message = validate_line_angle(
            self.start_point, self.end_point, warn_threshold=45.0
        )
        
        if not is_valid:
            QMessageBox.warning(self.dialog, "Invalid Line", warning_message)
            self.stop_current_drawing()
            return
        
        # Get Z-values at start and end points
        current_raster = self.main_window.image_window.current_raster
        if not current_raster or current_raster.z_channel is None:
            QMessageBox.warning(self.dialog, "No Z-Channel",
                              "Current image does not have a Z-channel.")
            self.stop_current_drawing()
            return
        
        # Get raw Z-values
        start_x, start_y = int(self.start_point.x()), int(self.start_point.y())
        end_x, end_y = int(self.end_point.x()), int(self.end_point.y())
        
        start_z = current_raster.get_z_value(start_x, start_y)
        end_z = current_raster.get_z_value(end_x, end_y)
        
        if start_z is None or end_z is None:
            QMessageBox.warning(self.dialog, "Invalid Z-Values",
                              "Could not read Z-values at the selected points.")
            self.stop_current_drawing()
            return
        
        raw_difference = abs(end_z - start_z)
        
        # Get known difference from dialog
        known_difference = self.dialog.z_known_difference_input.value()
        units = self.dialog.z_units_combo.currentText()
        
        try:
            scalar = calculate_z_scalar(raw_difference, known_difference)
            
            # Update labels
            self.dialog.z_raw_difference_label.setText(f"{raw_difference:.4f} units")
            self.dialog.z_calculated_scalar_label.setText(f"{scalar:.4f}")
            
        except ValueError as e:
            QMessageBox.warning(self.dialog, "Calculation Error", str(e))
            self.stop_current_drawing()
    
    def apply_z_scale(self):
        """Apply the calculated Z-scalar to highlighted images."""
        highlighted_paths = self.dialog.get_selected_image_paths()
        
        if not highlighted_paths:
            QMessageBox.warning(self.dialog, "No Images Selected",
                              "Please highlight at least one image to apply the Z-scale.")
            return
        
        # Get calculated scalar
        scalar_text = self.dialog.z_calculated_scalar_label.text()
        if "N/A" in scalar_text:
            QMessageBox.warning(self.dialog, "No Scalar Calculated",
                              "Please draw a line and calculate the scalar before applying it.")
            return
        
        try:
            scalar = float(scalar_text)
        except ValueError:
            QMessageBox.warning(self.dialog, "Invalid Scalar",
                              "Could not parse the scalar value.")
            return
        
        # Confirmation dialog
        reply = QMessageBox.question(
            self.dialog,
            "Confirm Z-Scale",
            f"Apply scalar {scalar:.4f} to {len(highlighted_paths)} image(s)?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Apply scalar to each highlighted image
        raster_manager = self.main_window.image_window.raster_manager
        
        for image_path in highlighted_paths:
            raster = raster_manager.get_raster(image_path)
            if raster and raster.z_channel is not None:
                # Update the scalar in z_settings
                raster.z_settings['scalar'] = scalar
                raster_manager.rasterUpdated.emit(image_path)
        
        # Refresh Z-channel visualization if current image was affected
        if self.annotation_window.current_image_path in highlighted_paths:
            self.annotation_window.refresh_z_channel_visualization()
        
        QMessageBox.information(self.dialog, "Z-Scale Applied",
                              f"Z-scalar applied to {len(highlighted_paths)} image(s).")
        
        # Clear drawing
        self.stop_current_drawing()
        self.dialog.z_raw_difference_label.setText("Draw a line on the image")
        self.dialog.z_calculated_scalar_label.setText("N/A")
    
    def set_z_anchor_point(self, scene_pos):
        """Set the Z-anchor point and calculate the current Z-value."""
        current_raster = self.main_window.image_window.current_raster
        if not current_raster or current_raster.z_channel is None:
            QMessageBox.warning(self.dialog, "No Z-Channel",
                              "Current image does not have a Z-channel.")
            return
        
        # Get Z-value at clicked point
        x, y = int(scene_pos.x()), int(scene_pos.y())
        current_z = current_raster.get_z_value(x, y)
        
        if current_z is None:
            QMessageBox.warning(self.dialog, "Invalid Point",
                              "Could not read Z-value at the selected point.")
            return
        
        # Store anchor point
        self.z_anchor_point = scene_pos
        
        # Update current value label
        z_unit = current_raster.z_unit if current_raster.z_unit else "units"
        self.dialog.z_current_value_label.setText(f"{current_z:.4f} {z_unit}")
        
        # Calculate and display offset
        target_value = self.dialog.z_target_value_input.value()
        from coralnet_toolbox.utilities import calculate_z_offset
        offset = calculate_z_offset(current_z, target_value)
        self.dialog.z_calculated_offset_label.setText(f"{offset:.4f} {z_unit}")
    
    def apply_z_anchor(self):
        """Apply the calculated Z-offset to highlighted images."""
        highlighted_paths = self.dialog.get_selected_image_paths()
        
        if not highlighted_paths:
            QMessageBox.warning(self.dialog, "No Images Selected",
                              "Please highlight at least one image to apply the Z-anchor.")
            return
        
        # Get calculated offset
        offset_text = self.dialog.z_calculated_offset_label.text()
        if "N/A" in offset_text:
            QMessageBox.warning(self.dialog, "No Offset Calculated",
                              "Please click on the image to set an anchor point before applying.")
            return
        
        try:
            # Parse offset value (strip units)
            offset = float(offset_text.split()[0])
        except (ValueError, IndexError):
            QMessageBox.warning(self.dialog, "Invalid Offset",
                              "Could not parse the offset value.")
            return
        
        # Get direction
        direction = self.dialog.z_direction_combo.currentData()
        
        # Confirmation dialog
        reply = QMessageBox.question(
            self.dialog,
            "Confirm Z-Anchor",
            f"Apply offset {offset:.4f} and direction {direction} to {len(highlighted_paths)} image(s)?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Apply offset and direction to each highlighted image
        raster_manager = self.main_window.image_window.raster_manager
        
        for image_path in highlighted_paths:
            raster = raster_manager.get_raster(image_path)
            if raster and raster.z_channel is not None:
                # Update the offset and direction in z_settings
                raster.z_settings['offset'] = offset
                raster.z_settings['direction'] = direction
                raster_manager.rasterUpdated.emit(image_path)
        
        # Refresh Z-channel visualization if current image was affected
        if self.annotation_window.current_image_path in highlighted_paths:
            self.annotation_window.refresh_z_channel_visualization()
        
        QMessageBox.information(self.dialog, "Z-Anchor Applied",
                              f"Z-anchor applied to {len(highlighted_paths)} image(s).")
        
        # Clear state
        self.z_anchor_point = None
        self.dialog.z_current_value_label.setText("Click on the image")
        self.dialog.z_calculated_offset_label.setText("N/A")
    
    def reset_z_scalar(self):
        """Reset Z-scalar to default (1.0) for highlighted images."""
        highlighted_paths = self.dialog.get_selected_image_paths()
        
        if not highlighted_paths:
            QMessageBox.warning(self.dialog, "No Images Selected",
                              "Please highlight at least one image to reset the Z-scalar.")
            return
        
        # Confirmation dialog
        reply = QMessageBox.question(
            self.dialog,
            "Confirm Reset",
            f"Reset Z-scalar to 1.0 for {len(highlighted_paths)} image(s)?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Reset scalar for each highlighted image
        raster_manager = self.main_window.image_window.raster_manager
        
        for image_path in highlighted_paths:
            raster = raster_manager.get_raster(image_path)
            if raster and raster.z_channel is not None:
                raster.z_settings['scalar'] = 1.0
                raster_manager.rasterUpdated.emit(image_path)
        
        # Refresh Z-channel visualization if current image was affected
        if self.annotation_window.current_image_path in highlighted_paths:
            self.annotation_window.refresh_z_channel_visualization()
        
        QMessageBox.information(self.dialog, "Z-Scalar Reset",
                              f"Z-scalar reset for {len(highlighted_paths)} image(s).")
        
        # Clear UI
        self.dialog.z_calculated_scalar_label.setText("N/A")
    
    def reset_z_offset(self):
        """Reset Z-offset to default (0.0) for highlighted images."""
        highlighted_paths = self.dialog.get_selected_image_paths()
        
        if not highlighted_paths:
            QMessageBox.warning(self.dialog, "No Images Selected",
                              "Please highlight at least one image to reset the Z-offset.")
            return
        
        # Confirmation dialog
        reply = QMessageBox.question(
            self.dialog,
            "Confirm Reset",
            f"Reset Z-offset to 0.0 for {len(highlighted_paths)} image(s)?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Reset offset for each highlighted image
        raster_manager = self.main_window.image_window.raster_manager
        
        for image_path in highlighted_paths:
            raster = raster_manager.get_raster(image_path)
            if raster and raster.z_channel is not None:
                raster.z_settings['offset'] = 0.0
                raster_manager.rasterUpdated.emit(image_path)
        
        # Refresh Z-channel visualization if current image was affected
        if self.annotation_window.current_image_path in highlighted_paths:
            self.annotation_window.refresh_z_channel_visualization()
        
        QMessageBox.information(self.dialog, "Z-Offset Reset",
                              f"Z-offset reset for {len(highlighted_paths)} image(s).")
        
        # Clear UI
        self.dialog.z_calculated_offset_label.setText("N/A")
