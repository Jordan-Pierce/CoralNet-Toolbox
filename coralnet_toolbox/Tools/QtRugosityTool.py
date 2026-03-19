import warnings

import os
import csv

import math
import numpy as np
from random import randint

import pyqtgraph as pg

from PyQt5.QtCore import Qt, QLineF, QRectF, QPointF, QEvent
from PyQt5.QtGui import QMouseEvent, QPen, QColor, QBrush
from PyQt5.QtWidgets import (QApplication, QDialog, QWidget, QVBoxLayout,
                             QFormLayout, QDoubleSpinBox, QComboBox, 
                             QDialogButtonBox, QMessageBox, QLabel,
                             QGroupBox, QPushButton, QSpacerItem,
                             QSizePolicy, QScrollArea, QFrame, QSpinBox, QHBoxLayout,
                             QGraphicsLineItem, QFileDialog)

from coralnet_toolbox.Tools.QtTool import Tool
from coralnet_toolbox.WorkArea import WorkArea
from coralnet_toolbox.Common.QtMarginInput import MarginInput

from coralnet_toolbox.Icons import get_icon
from coralnet_toolbox.utilities import convert_scale_units

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class RugosityDialog(QDialog):
    """
    A modeless dialog for Rugosity, allowing measurements and spatial analysis.
    """
    def __init__(self, tool, parent=None):
        super().__init__(parent)
        # Get references from the tool
        self.tool = tool
        
        self.annotation_window = self.tool.annotation_window
        self.main_window = self.annotation_window.main_window
        
        self.animation_manager = self.annotation_window.animation_manager
        
        # Drawing state
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        
        # Graphics items
        self.line_graphic = None
        self.wireframe_graphic = None  # For 3D visualization
        self.endpoint_dots = []  # Store endpoint circles (white and black dots)
        
        # Recorded measurements with color info
        self.recorded_line_measurements = []  # List of dicts with line data and color
        
        # Grid settings
        self.working_area = None  # Visual representation of working area
        self.grid_lines = []  # List of grid line measurements
        self.grid_enabled = False
        
        # Profile data for plotting
        self.current_profiles = []
        self.profile_dialog = None
        
        # Last calculated color for measurements
        self.last_calculated_color = (128, 128, 128)  # Default to gray for grid lines

        self.setWindowTitle("Rugosity")
        self.setWindowIcon(get_icon("spatial.svg"))  # use the spatial icon
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.resize(450, 300)
        self.setModal(False)

        self.layout = QVBoxLayout(self)

        # Set up the information layout
        self.setup_info_layout()
        # Set up the rugosity parameters layout
        self.setup_rugosity_params_layout()
        # Set up the 2D measurements layout
        self.setup_2d_measurements_layout()
        # Set up the grid settings layout
        self.setup_grid_settings_layout()
        # Set up the 3D Z-metrics layout
        self.setup_3d_zmetrics_layout()
        # Set up the status label
        self.setup_status_label()
        # Set up the buttons layout
        self.setup_buttons_layout()
        
        self.layout.addStretch()

    def setup_info_layout(self):
        """Set up the information layout with explanatory text."""
        groupbox = QGroupBox("Information")
        layout = QVBoxLayout()
        
        instructions = QLabel(
            "Draw lines to measure rugosity and surface complexity. "
            "Use the grid automatically to sample specified area. "
            "Color corresponds to rugosity (red: high, blue: low)."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)

    def setup_rugosity_params_layout(self):
        """Set up the rugosity parameters layout."""
        groupbox = QGroupBox("Rugosity Parameters")
        layout = QFormLayout()
        
        chain_layout = QHBoxLayout()
        self.chain_length_spin = QDoubleSpinBox()
        self.chain_length_spin.setRange(0.001, 10000.0)
        self.chain_length_spin.setValue(1.0)
        self.chain_length_spin.setSingleStep(0.1)
        self.chain_length_spin.setToolTip(
            "The step size used to sample the 3D surface (simulates a physical chain link)."
        )
        self.chain_unit_combo = QComboBox()
        self.chain_unit_combo.addItems(['mm', 'cm', 'm', 'in', 'ft', 'yd'])
        self.chain_unit_combo.setCurrentText('cm')
        self.chain_unit_combo.setToolTip("The unit of measurement for the chain length.")
        chain_layout.addWidget(self.chain_length_spin)
        chain_layout.addWidget(self.chain_unit_combo)
        layout.addRow("Chain Length:", chain_layout)
        
        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)

    def setup_2d_measurements_layout(self):
        """Set up the 2D measurements layout."""
        groupbox = QGroupBox("2D Measurements")
        layout = QFormLayout()
        
        self.line_length_2d_label = QLabel("---")
        layout.addRow("Distance:", self.line_length_2d_label)
        self.line_units_combo = QComboBox()
        self.line_units_combo.addItems(['mm', 'cm', 'm', 'km', 'in', 'ft', 'yd', 'mi'])
        self.line_units_combo.setCurrentText('m')
        self.line_units_combo.setToolTip("The unit of measurement for the 2D distance.")
        
        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)

    def setup_grid_settings_layout(self):
        """Set up the grid settings layout."""
        groupbox = QGroupBox("Grid Settings")
        layout = QVBoxLayout()
        
        self.margin_input = MarginInput()
        layout.addWidget(self.margin_input)
        
        grid_params_layout = QFormLayout()
        rowcol_layout = QHBoxLayout()
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(0, 100)
        self.rows_spin.setValue(10)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(0, 100)
        self.cols_spin.setValue(10)
        rowcol_layout.addWidget(QLabel("Rows:"))
        rowcol_layout.addWidget(self.rows_spin)
        rowcol_layout.addSpacing(20)
        rowcol_layout.addWidget(QLabel("Columns:"))
        rowcol_layout.addWidget(self.cols_spin)
        grid_params_layout.addRow(rowcol_layout)
        layout.addLayout(grid_params_layout)
        
        grid_btn_layout = QHBoxLayout()
        self.generate_grid_button = QPushButton("Preview Grid")
        self.generate_grid_button.clicked.connect(self.generate_grid)
        self.clear_grid_button = QPushButton("Clear Grid")
        self.clear_grid_button.clicked.connect(self.clear_grid)
        grid_btn_layout.addWidget(self.generate_grid_button)
        grid_btn_layout.addWidget(self.clear_grid_button)
        layout.addLayout(grid_btn_layout)
        
        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)

    def setup_3d_zmetrics_layout(self):
        """Set up the 3D Z-metrics layout."""
        groupbox = QGroupBox("3D Z-Metrics")
        groupbox.setEnabled(False)
        layout = QFormLayout()
        
        self.line_length_3d_label = QLabel("---")
        self.line_delta_z_label = QLabel("---")
        self.line_slope_label = QLabel("---")
        self.line_rugosity_label = QLabel("---")
        self.line_rugosity_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addRow("3D Length:", self.line_length_3d_label)
        layout.addRow("ΔZ:", self.line_delta_z_label)
        layout.addRow("Slope / Grade:", self.line_slope_label)
        layout.addRow("Rugosity:", self.line_rugosity_label)
        self.line_rugosity_label.setToolTip("The rugosity value calculated from the 3D line.")
        
        self.line_profile_button = QPushButton("Show Elevation Profile")
        self.line_profile_button.clicked.connect(self._show_elevation_profile)
        self.line_profile_button.setEnabled(False)
        layout.addRow("", self.line_profile_button)
        self.line_profile_button.setToolTip("Show the elevation profile for the 3D line.")
        
        groupbox.setLayout(layout)
        self.line_3d_group = groupbox
        self.layout.addWidget(groupbox)

    def setup_status_label(self):
        """Set up the status label."""
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.status_label)

    def setup_buttons_layout(self):
        """Set up the buttons layout."""
        button_layout = QHBoxLayout()
        
        self.clear_button = QPushButton("Clear All Measurements")
        self.clear_button.clicked.connect(self.clear_all_measurements)
        button_layout.addWidget(self.clear_button)
        
        button_layout.addStretch()
        
        self.export_button = QPushButton("Export Measurements")
        self.export_button.clicked.connect(self.export_measurements_csv)
        button_layout.addWidget(self.export_button)
        
        self.layout.addLayout(button_layout)

    def set_status(self, text):
        """Set the status label text."""
        self.status_label.setText(text)

    def closeEvent(self, event):
        """Handle dialog close - deactivate tool"""
        self.cleanup()
        event.accept()
    
    def reject(self):
        """Handle dialog rejection."""
        self.cleanup()
        super().reject()
        
    def cleanup(self):
        """Clean up temporary graphics, reset UI to defaults, and deactivate tool."""
        # Clear temporary graphics and measurements, reset UI
        self.clear_all_measurements()
        
        # Reset grid settings to defaults
        self.rows_spin.setValue(10)
        self.cols_spin.setValue(10)
        
        # Reset additional state variables
        self.recorded_line_measurements = []
        self.grid_lines = []
        self.grid_enabled = False
        self.current_profiles = []
        if self.profile_dialog:
            self.profile_dialog.close()
            self.profile_dialog = None
        self.last_calculated_color = (128, 128, 128)
        
        # Deactivate the tool
        self.tool.deactivate()
        
        # Untoggle all tools in the main window
        self.main_window.untoggle_all_tools()
    
    # --- Helper Methods ---
    
    def get_current_scale(self):
        """Get current scale information from raster"""
        image_path = self.annotation_window.current_image_path
        if not image_path:
            return None, None, None
            
        raster_manager = self.annotation_window.main_window.image_window.raster_manager
        raster = raster_manager.get_raster(image_path)
        
        if raster and raster.scale_x and raster.scale_y and raster.scale_units:
            return raster.scale_x, raster.scale_y, raster.scale_units
        return None, None, None

    def get_current_z_data(self):
        """Get current Z-channel data from raster (raw values)"""
        image_path = self.annotation_window.current_image_path
        if not image_path:
            return None, None
            
        raster_manager = self.annotation_window.main_window.image_window.raster_manager
        raster = raster_manager.get_raster(image_path)
        
        # Check if scale exists (required for measurements)
        scale_x, scale_y, scale_units = self.get_current_scale()
        if scale_x is None:
            return None, None
        
        # Get Z-channel data using lazy loading
        if raster and raster.z_channel_lazy is not None:
            z_data = raster.z_channel_lazy
            z_unit = raster.z_unit or 'px'
            
            return z_data, z_unit
        
        return None, None

    def _generate_random_color(self):
        """Generate a random RGB color tuple"""
        return (randint(0, 255), randint(0, 255), randint(0, 255))
    
    def _create_colored_line(self, start, end, color_rgb):
        """Create a solid colored line with endpoint dots"""
        scene = self.annotation_window.scene
        
        # Create solid line in the random color
        pen = QPen(QColor(*color_rgb), 3, Qt.SolidLine)
        pen.setCosmetic(True)
        line = QLineF(start, end)
        line_graphic = scene.addLine(line, pen)
        line_graphic.setZValue(1000)
        
        # Create white dot at start point
        start_dot = scene.addEllipse(
            start.x() - 3, start.y() - 3, 6, 6,
            QPen(QColor(255, 255, 255), 3),
            QBrush(QColor(255, 255, 255))
        )
        start_dot.setZValue(1001)
        
        # Create black dot at end point
        end_dot = scene.addEllipse(
            end.x() - 3, end.y() - 3, 6, 6,
            QPen(QColor(0, 0, 0), 3),
            QBrush(QColor(0, 0, 0))
        )
        end_dot.setZValue(1001)
        
        return {
            'line': line_graphic,
            'start_dot': start_dot,
            'end_dot': end_dot,
            'start_point': QPointF(start),
            'end_point': QPointF(end),
            'color': color_rgb
        }
        
    def _get_rugosity_color(self, rugosity):
        """
        Map a rugosity value to an RGB color based on SESSION MAXIMUM.
        1.0 -> Grey (Flat)
        Higher values map through Blue -> Green -> Orange -> Red (Scaled to session max)
        """
        # Strictly Grey for flat surfaces (allowing for tiny floating point noise)
        if rugosity <= 1.001: 
            return (128, 128, 128)
            
        # 1. Determine the Max Rugosity for Scaling (Auto-Scaling)
        # Start with a sensible default ceiling (e.g. 1.2) so slight bumps don't turn bright red immediately
        current_max = 1.2 
        
        # Check all existing profiles to find the true session max
        for profile in self.current_profiles:
            stats = profile.get('stats', {})
            r = stats.get('rugosity', 1.0)
            if r > current_max:
                current_max = r
        
        # Also check the current value being calculated
        if rugosity > current_max:
            current_max = rugosity
            
        # 2. Normalize: 0.0 at R=1, 1.0 at R=current_max
        norm = (rugosity - 1.0) / (current_max - 1.0)
        norm = max(0.0, min(1.0, norm))
        
        # 3. Heatmap Gradient Logic (Jet-like: Blue -> Green -> Orange -> Red)
        if norm < 0.25:
            # Blue Range (Low complexity)
            # 0.0: Blue (0,0,255) -> 0.25: Cyan (0,255,255)
            local_t = norm / 0.25
            r = 0
            g = int(0 * (1 - local_t) + 255 * local_t)
            b = 255
            return (r, g, b)
        
        elif norm < 0.5:
            # Cyan -> Green
            local_t = (norm - 0.25) / 0.25
            r = 0
            g = 255
            b = int(255 * (1 - local_t) + 0 * local_t)
            return (r, g, b)
        
        elif norm < 0.75:
            # Green -> Yellow/Orange
            local_t = (norm - 0.5) / 0.25
            r = int(0 * (1 - local_t) + 255 * local_t)
            g = 255
            b = 0
            return (r, g, b)
        
        else:
            # Yellow -> Red
            local_t = (norm - 0.75) / 0.25
            r = 255
            g = int(255 * (1 - local_t) + 0 * local_t)
            b = 0
            return (r, g, b)

    def stop_current_drawing(self):
        """Stop any active drawing and clear graphics"""
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        
        # Remove graphics
        scene = self.annotation_window.scene
        if self.line_graphic and self.line_graphic.scene():
            scene.removeItem(self.line_graphic)
            self.line_graphic = None
        if self.wireframe_graphic and self.wireframe_graphic.scene():
            scene.removeItem(self.wireframe_graphic)
            self.wireframe_graphic = None

    def clear_all_graphics(self):
        """Clear all graphics including recorded measurements from the scene"""
        scene = self.annotation_window.scene
        
        # Clear current drawing graphics
        self.stop_current_drawing()
        
        # Clear all recorded line measurements
        for measurement in self.recorded_line_measurements:
            for key in ['line', 'start_dot', 'end_dot']:
                if key in measurement and measurement[key]:
                    if measurement[key].scene() == scene:
                        scene.removeItem(measurement[key])
        self.recorded_line_measurements.clear()
        
        # Clear grid
        self.clear_grid()
        
        # Close profile dialog if open
        if self.profile_dialog:
            self.profile_dialog.close()
            self.profile_dialog = None
        
        # Clear profile data
        self.current_profiles.clear()

    def clear_all_measurements(self):
        """Clear all measurements and reset UI"""
        self.clear_all_graphics()
        self.line_length_2d_label.setText("---")
        self._reset_line_3d_labels()
        self.line_profile_button.setEnabled(False)
        # Reset chain length and unit
        self.chain_length_spin.setValue(1.0)
        self.chain_unit_combo.setCurrentText('cm')
        # Reset margin input to defaults
        self.margin_input.type_combo.setCurrentIndex(1)  # Multiple Values
        self.margin_input.value_type.setCurrentIndex(0)  # Pixels
        for spin in self.margin_input.margin_spins:
            spin.setValue(0)
        self.margin_input.single_spin.setValue(0)
        self.margin_input.single_double.setValue(0.0)
        self.margin_input.update_input_mode(0)
        # Recalculate grid spinbox values to auto-calculated defaults
        self.calculate_spacing()

    def export_measurements_csv(self):
        """Export all current rugosity measurements to a CSV file."""
        if not self.current_profiles:
            QMessageBox.warning(self, "No Data", "No measurements to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return
        
        image_path = self.annotation_window.current_image_path or ""
        image_name = os.path.basename(image_path) if image_path else ""
        display_units = self.line_units_combo.currentText()
        chain_length = self.chain_length_spin.value()
        
        try:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    "Image Name", "Image Path", "Line Name", "Start X", "Start Y", "End X", "End Y",
                    "Start Z", "End Z", "Min Z", "Max Z", "2D Length", "3D Length", "Delta Z",
                    "Rugosity", "Slope", "Color (RGB)", "Units", "Chain Length"
                ])
                
                for profile in self.current_profiles:
                    stats = profile.get('stats', {})
                    start_point = profile.get('start_point', QPointF(0, 0))
                    end_point = profile.get('end_point', QPointF(0, 0))
                    color = profile.get('color', (0, 0, 0))
                    color_str = f"{color[0]},{color[1]},{color[2]}"
                    
                    writer.writerow([
                        image_name,
                        image_path,
                        profile.get('name', ''),
                        f"{start_point.x():.3f}",
                        f"{start_point.y():.3f}",
                        f"{end_point.x():.3f}",
                        f"{end_point.y():.3f}",
                        f"{stats.get('start_z', 0):.3f}",
                        f"{stats.get('end_z', 0):.3f}",
                        f"{stats.get('min_z', 0):.3f}",
                        f"{stats.get('max_z', 0):.3f}",
                        f"{stats.get('length_2d', 0):.3f} {display_units}",
                        f"{stats.get('length_3d', 0):.3f} {display_units}",
                        f"{stats.get('delta_z', 0):.3f}",
                        f"{stats.get('rugosity', 0):.3f}",
                        f"{stats.get('slope', 0):.2f}",
                        color_str,
                        display_units,
                        f"{chain_length} {self.chain_unit_combo.currentText()}"
                    ])
            
            QMessageBox.information(self, "Export Complete", f"Measurements exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export CSV: {str(e)}")

    def calculate_spacing(self):
        """Calculate and store row and column spacing as attributes based on image dimensions."""
        if not self.annotation_window.current_image_path:
            self.row_spacing = None
            self.col_spacing = None
            return
        image_width, image_height = self.annotation_window.get_image_dimensions()
        if not image_width or not image_height:
            self.row_spacing = None
            self.col_spacing = None
            return
        num_rows = self.rows_spin.value()
        num_cols = self.cols_spin.value()
        self.row_spacing = image_height / (num_rows - 1) if num_rows > 1 else image_height
        self.col_spacing = image_width / (num_cols - 1) if num_cols > 1 else image_width

    def generate_grid(self):
        """Generate grid lines based on margin and spacing settings with Rugosity Coloring"""
        # Get image dimensions
        if not self.annotation_window.current_image_path:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        image_width, image_height = self.annotation_window.get_image_dimensions()
        if not image_width or not image_height:
            return

        # Get margins in pixels
        margins = self.margin_input.get_margins(image_width, image_height, validate=True)
        if margins is None:
            return

        left, top, right, bottom = margins

        # Calculate ROI bounds
        roi_left = left
        roi_top = top
        roi_right = image_width - right
        roi_bottom = image_height - bottom

        if roi_right <= roi_left or roi_bottom <= roi_top:
            QMessageBox.warning(self, "Invalid Working Area", "Margins result in invalid working area.")
            return

        # Calculate and update spacing attributes
        self.calculate_spacing()
        row_spacing = self.row_spacing
        col_spacing = self.col_spacing
        num_rows = self.rows_spin.value()
        num_cols = self.cols_spin.value()

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            bounds = QRectF(roi_left, roi_top, roi_right - roi_left, roi_bottom - roi_top)
            self.grid_enabled = True

            # Draw working area rectangle
            self._draw_working_area_graphic(bounds)

            # Clear existing grid lines first
            self.clear_grid()

            # --- Helper to process a grid line ---
            def process_grid_line(p1, p2):
                # 1. Set points for calculation
                self.start_point = p1
                self.end_point = p2

                # 2. Calculate measurement (color=None enables auto-coloring)
                self.calculate_line_measurement(final_calc=True, color=None)

                # 3. Tag the last added profile as a "grid" line so we can clear it later
                if self.current_profiles:
                    self.current_profiles[-1]['is_grid'] = True
                    self.current_profiles[-1]['name'] = f"Grid Line {len(self.current_profiles)}"

                # 4. Retrieve the calculated color
                final_color = self.last_calculated_color

                # 5. Create the graphic with this specific color
                measurement = self._create_colored_line(p1, p2, final_color)
                self.grid_lines.append(measurement)

                # Reset points
                self.start_point = None
                self.end_point = None

            # --- Generate Rows ---
            current_y = roi_top
            row_count = 0
            while current_y <= roi_bottom and (num_rows == 0 or row_count < num_rows):
                if current_y >= roi_top:
                    start = QPointF(roi_left, current_y)
                    end = QPointF(roi_right, current_y)
                    process_grid_line(start, end)
                current_y += row_spacing
                row_count += 1

            # --- Generate Columns ---
            current_x = roi_left
            col_count = 0
            while current_x <= roi_right and (num_cols == 0 or col_count < num_cols):
                if current_x >= roi_left:
                    start = QPointF(current_x, roi_top)
                    end = QPointF(current_x, roi_bottom)
                    process_grid_line(start, end)
                current_x += col_spacing
                col_count += 1

            # Update profile button state
            self.update_profile_button_state()

            # Update profile dialog if it's already open
            if self.profile_dialog and self.profile_dialog.isVisible():
                self.profile_dialog.update_plot(self.current_profiles)
        finally:
            QApplication.restoreOverrideCursor()

    def clear_grid(self):
        """Clear all grid lines, ROI, and remove grid profiles from plot"""
        scene = self.annotation_window.scene
        
        # Remove all grid line graphics
        for measurement in self.grid_lines:
            for key in ['line', 'start_dot', 'end_dot']:
                if key in measurement and measurement[key]:
                    if measurement[key].scene() == scene:
                        scene.removeItem(measurement[key])
        self.grid_lines.clear()
        
        # Remove working area
        if self.working_area:
            self.working_area.remove_from_scene()
            self.working_area = None
            
        self.grid_enabled = False
        
        # Clear grid profiles from current_profiles
        self.current_profiles = [p for p in self.current_profiles if not p.get('is_grid', False)]
        
        # Update profile button and dialog
        self.update_profile_button_state()
        if self.profile_dialog and self.profile_dialog.isVisible():
            self.profile_dialog.update_plot(self.current_profiles)

    def _draw_working_area_graphic(self, bounds):
        """Create working area on the scene with shadow"""
        if not bounds:
            return
            
        # Remove old working area if exists
        if self.working_area:
            self.working_area.remove_from_scene()
            self.working_area = None
            
        # Create WorkArea for working area
        image_path = self.annotation_window.current_image_path
        self.working_area = WorkArea(
            bounds.x(), 
            bounds.y(), 
            bounds.width(), 
            bounds.height(), 
            image_path
        )
        
        # Set the animation manager
        self.working_area.set_animation_manager(self.animation_manager)
        
        # Create graphics with shadow and animation
        self.working_area.create_graphics(
            self.annotation_window.scene, 
            include_shadow=True, 
            animate=True,
            image_rect=self.annotation_window.get_image_rect()
        )
        
        # Set ZValues to match original (below measurements but above image)
        if self.working_area.graphics_item:
            self.working_area.graphics_item.setZValue(999)
        if self.working_area.shadow_area:
            self.working_area.shadow_area.setZValue(998)
    
        # Force scene to update and render the new graphics
        self.annotation_window.scene.update()

    def _is_point_in_working_area(self, point):
        """Check if a point is within the working area"""
        if not self.grid_enabled or not self.working_area:
            return True
        rect = QRectF(self.working_area.x(), 
                      self.working_area.y(), 
                      self.working_area.width(), 
                      self.working_area.height())
        return rect.contains(point)

    def _clamp_point_to_working_area(self, point):
        """Clamp a point to working area bounds"""
        if not self.grid_enabled or not self.working_area:
            return point
            
        rect = QRectF(self.working_area.x(), 
                      self.working_area.y(),
                      self.working_area.width(), 
                      self.working_area.height())
        x = max(rect.left(), min(point.x(), rect.right()))
        y = max(rect.top(), min(point.y(), rect.bottom()))
        return QPointF(x, y)

    def update_z_controls(self):
        """Enable/disable Z-controls based on data availability"""
        z_data, z_unit = self.get_current_z_data()
        has_z = z_data is not None
        
        self.line_3d_group.setEnabled(has_z)
        self.line_profile_button.setEnabled(False)  # Enable after line drawn

    def handle_mouse_press(self, event: QMouseEvent):
        """Handle mouse press for starting measurements"""
        if event.button() == Qt.LeftButton:
            scene_pos = self.annotation_window.mapToScene(event.pos())
            
            # Check if point is within working area when grid is enabled
            if self.grid_enabled and not self._is_point_in_working_area(scene_pos):
                return
            
            # Only start new drawing if not already drawing
            if not self.is_drawing:
                # Clear previous graphics
                self.stop_current_drawing()
                
                # Start new measurement
                self.start_point = scene_pos
                self.end_point = scene_pos
                self.is_drawing = True
            else:
                # Second click finishes the measurement
                self.end_point = self._clamp_point_to_working_area(scene_pos)
                self.is_drawing = False
                
                # 1. Final Calculation 
                self.calculate_line_measurement(final_calc=True, color=None)
                
                # 2. Get the calculated color
                final_color = self.last_calculated_color
                
                # 3. Create and record the permanent graphic with this color
                measurement = self._create_colored_line(self.start_point, self.end_point, final_color)
                self.recorded_line_measurements.append(measurement)
                
                # 4. Updates
                self._update_wireframe_graphic()
                self.update_profile_button_state()

    def handle_mouse_move(self, event: QMouseEvent):
        """Handle mouse move for drawing measurements"""
        scene_pos = self.annotation_window.mapToScene(event.pos())
        if self.is_drawing and self.start_point:
            self.end_point = self._clamp_point_to_working_area(scene_pos)
            
            # 1. Update the visual line first
            self._update_line_graphic()
            
            # 2. Force the app to process the paint event immediately
            QApplication.processEvents() 
            
            # 3. Perform the calculation
            self.calculate_line_measurement(final_calc=False)

    def handle_mouse_release(self, event: QMouseEvent):
        """Handle mouse release - measurement is finalized on second click"""
        pass
    
    def _update_line_graphic(self):
        """Update or create line graphic with cosmetic pen (visible at all zoom levels)"""
        if not self.start_point or not self.end_point:
            return
            
        scene = self.annotation_window.scene
        
        # Determine the color and pen style
        if self.is_drawing:
            color = QColor(*self.last_calculated_color)
        else:
            color = QColor(230, 62, 0)
            
        pen = QPen(color, 3, Qt.DashLine)
        pen.setCosmetic(True)
        line = QLineF(self.start_point, self.end_point)
        
        # Update existing item or create new one
        if self.line_graphic and self.line_graphic.scene() == scene:
            self.line_graphic.setLine(line)
            self.line_graphic.setPen(pen)
        else:
            if self.line_graphic and self.line_graphic.scene():
                self.line_graphic.scene().removeItem(self.line_graphic)

            self.line_graphic = scene.addLine(line, pen)
            self.line_graphic.setZValue(1000)
        
        # Force scene update for real-time visibility
        scene.update()

    def _update_wireframe_graphic(self):
        """Create 3D wireframe visualization for line measurement"""
        # Placeholder for future implementation
        pass

    def _show_elevation_profile(self):
        """Show elevation profile plot dialog"""
        if not self.current_profiles:
            QMessageBox.warning(self, "No Data", "No profile data available to display.")
            return
        
        if self.profile_dialog is None or not self.profile_dialog.isVisible():
            self.profile_dialog = ProfilePlotDialog(self.current_profiles, self)
        else:
            self.profile_dialog.update_plot(self.current_profiles)
        
        self.profile_dialog.show()
        self.profile_dialog.raise_()

    def update_profile_button_state(self):
        """Enable/disable profile button based on data availability"""
        has_data = len(self.current_profiles) > 0
        self.line_profile_button.setEnabled(has_data)

    def calculate_line_measurement(self, final_calc=False, color=None):
        """Calculate line measurements (2D and 3D) using transform pipeline"""
        if not self.start_point or not self.end_point:
            return
            
        # Initialize default color (Grey) in case of early exit
        self.last_calculated_color = (128, 128, 128)
            
        # --- 1. Get Data ---
        scale_x, scale_y, scale_units = self.get_current_scale()
        
        # Check for invalid scale
        if scale_x is None or scale_y is None:
            self.line_length_2d_label.setText("Scale Not Set")
            self._reset_line_3d_labels()
            return

        display_units = self.line_units_combo.currentText()
        
        # --- 2. Calculate 2D Length ---
        line = QLineF(self.start_point, self.end_point)
        pixel_length = line.length()
        length_2d_meters = pixel_length * scale_x
        
        # Convert to display units
        if display_units != "m":
            length_2d_display = convert_scale_units(length_2d_meters, 'metre', display_units)
        else:
            length_2d_display = length_2d_meters
        
        self.line_length_2d_label.setText(f"{length_2d_display:.3f} {display_units}")

        # --- 3. 3D Calculations (if Z-data available) ---
        z_data, z_unit = self.get_current_z_data()
        
        if z_data is None:
            self._reset_line_3d_labels()
            return
        
        try:
            image_path = self.annotation_window.current_image_path
            raster_manager = self.annotation_window.main_window.image_window.raster_manager
            raster = raster_manager.get_raster(image_path)
            
            h, w = z_data.shape
            z_unit_str = z_unit if z_unit else 'px'
            
            # Convert z_unit to meters for 3D calculations
            z_to_meters_factor = convert_scale_units(1.0, z_unit, 'metre') if z_unit else 1.0

            # Get raw Z values at start/end
            p1_x = int(max(0, min(self.start_point.x(), w - 1)))
            p1_y = int(max(0, min(self.start_point.y(), h - 1)))
            p2_x = int(max(0, min(self.end_point.x(), w - 1)))
            p2_y = int(max(0, min(self.end_point.y(), h - 1)))
            
            z_start = raster.get_z_value(p1_x, p1_y)
            z_end = raster.get_z_value(p2_x, p2_y)
            
            if z_start is None or z_end is None:
                self._reset_line_3d_labels()
                return
            
            delta_z = z_end - z_start
            self.line_delta_z_label.setText(f"{delta_z:.3f} {z_unit_str}")
            
            # Calculate Slope
            slope = 0.0
            if length_2d_meters > 0:
                delta_z_meters = delta_z * z_to_meters_factor
                slope = (delta_z_meters / length_2d_meters) * 100.0
                self.line_slope_label.setText(f"{slope:.2f} %")
            else:
                self.line_slope_label.setText("N/A")

            # --- Calculate 3D Length with Chain Resolution ---
            chain_len_val = self.chain_length_spin.value()
            chain_len_unit = self.chain_unit_combo.currentText()
            chain_len_meters = convert_scale_units(chain_len_val, chain_len_unit, 'metre')
            
            if scale_x > 0:
                step_pixels = chain_len_meters / scale_x
            else:
                step_pixels = 1.0
                
            step_pixels = max(1.0, step_pixels)
            
            num_steps = int(pixel_length / step_pixels)
            num_samples = max(2, num_steps + 1)

            x_samples = np.linspace(self.start_point.x(), self.end_point.x(), num_samples)
            y_samples = np.linspace(self.start_point.y(), self.end_point.y(), num_samples)
            
            # Get raw Z-values along the line
            z_samples = []
            for i in range(num_samples):
                x_idx = int(max(0, min(x_samples[i], w - 1)))
                y_idx = int(max(0, min(y_samples[i], h - 1)))
                z_samples.append(z_data[y_idx, x_idx])
            
            z_array = np.array(z_samples)
            
            # Calculate 3D length
            profile_data_x = []
            profile_data_y = []
            total_3d_length = 0.0
            dist_2d_so_far = 0.0
            
            profile_data_x.append(0.0)
            profile_data_y.append(z_array[0])
            
            for i in range(num_samples - 1):
                x_a, y_a = x_samples[i], y_samples[i]
                x_b, y_b = x_samples[i + 1], y_samples[i + 1]
                z_a = z_array[i]
                z_b = z_array[i + 1]
                
                dx_m = (x_b - x_a) * scale_x
                dy_m = (y_b - y_a) * scale_y
                dz_meters = (z_b - z_a) * z_to_meters_factor
                
                total_3d_length += math.sqrt(dx_m**2 + dy_m**2 + dz_meters**2)
                
                dist_2d_so_far += math.sqrt(dx_m**2 + dy_m**2)
                profile_data_x.append(dist_2d_so_far)
                profile_data_y.append(z_b)

            # Convert 3D length to display units
            if display_units != "m":
                length_3d_display = convert_scale_units(total_3d_length, 'metre', display_units)
                conv_factor = convert_scale_units(1.0, 'metre', display_units)
                plot_x_data = [x * conv_factor for x in profile_data_x]
            else:
                length_3d_display = total_3d_length
                plot_x_data = profile_data_x
            
            self.line_length_3d_label.setText(f"{length_3d_display:.3f} {display_units}")

            # --- Calculate Linear Rugosity & Color ---
            linear_rugosity = 1.0
            if length_2d_meters > 0:
                linear_rugosity = total_3d_length / length_2d_meters
                self.line_rugosity_label.setText(f"{linear_rugosity:.3f}")
            else:
                self.line_rugosity_label.setText("N/A")
            
            # Determine Color
            rugosity_color = self._get_rugosity_color(linear_rugosity)
            final_color_to_save = color if color else rugosity_color
            
            # Store calculated color
            self.last_calculated_color = final_color_to_save

            # Store profile data
            if final_calc:
                profile_data = {
                    'name': f'Line {len(self.current_profiles) + 1}',
                    'color': final_color_to_save, 
                    'x_data': plot_x_data,
                    'y_data': profile_data_y,
                    'start_point': self.start_point,
                    'end_point': self.end_point,
                    'stats': {
                        'length_2d': length_2d_display,
                        'length_3d': length_3d_display,
                        'delta_z': delta_z,
                        'rugosity': linear_rugosity if length_2d_meters > 0 else 0,
                        'slope': slope if length_2d_meters > 0 else 0,
                        'start_z': z_start,
                        'end_z': z_end,
                        'min_z': np.min(z_array),
                        'max_z': np.max(z_array)
                    }
                }
                
                self.current_profiles.append(profile_data)
                self.update_profile_button_state()
                
                if self.profile_dialog and self.profile_dialog.isVisible():
                    self.profile_dialog.update_plot(self.current_profiles)
                
        except Exception as e:
            print(f"Error in 3D line calculation: {e}")
            import traceback
            traceback.print_exc()
            self._reset_line_3d_labels()

    def _reset_line_3d_labels(self):
        """Reset all 3D line labels"""
        self.line_length_3d_label.setText("---")
        self.line_delta_z_label.setText("---")
        self.line_slope_label.setText("---")
        self.line_rugosity_label.setText("---")


class ProfilePlotDialog(QDialog):
    """
    A pop-up dialog to display a scrollable list of elevation profile plots
    using pyqtgraph.
    """
    def __init__(self, profiles_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Elevation Profile")
        self.setWindowIcon(get_icon("spatial.svg"))
        self.setMinimumSize(800, 700)

        # Set a white background for plots
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.main_layout = QVBoxLayout(self)
        
        # --- Scroll Area Setup ---
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        self.plot_container_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container_widget)
        
        self.scroll_area.setWidget(self.plot_container_widget)
        self.main_layout.addWidget(self.scroll_area)
        
        # --- Plot Items Storage ---
        self.plot_widgets = []

        # Initial build
        self.rebuild_plots(profiles_list)

        # --- Close Button ---
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        self.main_layout.addWidget(button_box)
        
    def rebuild_plots(self, profiles_list):
        """Clears and rebuilds all plots in the scroll area."""
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Clear all old plot widgets
            for widget in self.plot_widgets:
                widget.deleteLater()
            self.plot_widgets = []
            
            if not profiles_list:
                no_data_label = QLabel("No profile data available.")
                no_data_label.setAlignment(Qt.AlignCenter)
                self.plot_layout.addWidget(no_data_label)
                self.plot_widgets.append(no_data_label)
                return

            # Plot 1: Combined (Non-Normalized)
            if len(profiles_list) > 0:
                combined_plot_widget = pg.PlotWidget(title="All Profiles (Combined)")
                combined_plot_widget.setMinimumHeight(300)
                combined_plot = combined_plot_widget.getPlotItem()
                combined_plot.setLabel('left', 'Z')
                combined_plot.showGrid(x=True, y=True, alpha=0.3)

                for profile in profiles_list:
                    x_data = profile.get('x_data', [])
                    y_data = profile.get('y_data', [])
                    name = profile.get('name', 'Profile')
                    color = profile.get('color', (0, 0, 255))

                    pen = pg.mkPen(color=color, width=2)
                    combined_plot.plot(x_data, y_data, pen=pen, name=name)
                    
                    # Add start/end markers
                    if x_data and y_data:
                        start_scatter = pg.ScatterPlotItem(
                            [x_data[0]], [y_data[0]],
                            size=10, pen=pg.mkPen('k', width=1), brush=pg.mkBrush('w')
                        )
                        combined_plot.addItem(start_scatter)
                        
                        end_scatter = pg.ScatterPlotItem(
                            [x_data[-1]], [y_data[-1]],
                            size=10, pen=pg.mkPen('k', width=1), brush=pg.mkBrush('k')
                        )
                        combined_plot.addItem(end_scatter)
                
                self.plot_layout.addWidget(combined_plot_widget)
                self.plot_widgets.append(combined_plot_widget)

            # Separator
            separator_label = QLabel("Individual Profiles")
            separator_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
            separator_label.setAlignment(Qt.AlignCenter)
            self.plot_layout.addWidget(separator_label)
            self.plot_widgets.append(separator_label)
            
            separator_line = QFrame()
            separator_line.setFrameShape(QFrame.HLine)
            separator_line.setFrameShadow(QFrame.Sunken)
            self.plot_layout.addWidget(separator_line)
            self.plot_widgets.append(separator_line)

            # Individual Plots
            for profile in profiles_list:
                x_data = profile.get('x_data', [])
                y_data = profile.get('y_data', [])
                name = profile.get('name', 'Profile')
                color = profile.get('color', (0, 0, 255))
                stats = profile.get('stats', {})

                plot_widget = pg.PlotWidget(title=name)
                plot_widget.setMinimumHeight(250)
                plot = plot_widget.getPlotItem()
                plot.setLabel('left', 'Z')
                plot.showGrid(x=True, y=True, alpha=0.3)

                pen = pg.mkPen(color=color, width=2)
                plot.plot(x_data, y_data, pen=pen)
                
                # Add markers
                if x_data and y_data:
                    start_scatter = pg.ScatterPlotItem(
                        [x_data[0]], [y_data[0]],
                        size=10, pen=pg.mkPen('k', width=1), brush=pg.mkBrush('w')
                    )
                    plot.addItem(start_scatter)
                    
                    end_scatter = pg.ScatterPlotItem(
                        [x_data[-1]], [y_data[-1]],
                        size=10, pen=pg.mkPen('k', width=1), brush=pg.mkBrush('k')
                    )
                    plot.addItem(end_scatter)

                # Add statistics
                if stats:
                    length_3d = stats.get('length_3d', 0)
                    delta_z = stats.get('delta_z', 0)
                    rugosity = stats.get('rugosity', 0)
                    
                    stats_text = (
                        f"3D-Length: {length_3d:.3f}\n"
                        f"ΔZ: {delta_z:.3f}\n"
                        f"Rugosity: {rugosity:.3f}"
                    )
                    text_item = pg.TextItem(text=stats_text, anchor=(1, 0), color=(0, 0, 0))
                    text_item.setPos(max(x_data) if x_data else 0, max(y_data) if y_data else 0)
                    plot.addItem(text_item)

                self.plot_layout.addWidget(plot_widget)
                self.plot_widgets.append(plot_widget)
                    
            # Add spacer
            spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
            self.plot_layout.addSpacerItem(spacer)
        finally:
            QApplication.restoreOverrideCursor()

    def update_plot(self, profiles_list):
        """Clears and rebuilds all plots with new data."""
        try:
            self.rebuild_plots(profiles_list)
        except Exception as e:
            print(f"Error updating profile plot: {e}")


class RugosityTool(Tool):
    """
    Tool for measuring rugosity and surface complexity on images with Z-channel data.
    """
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.cursor = Qt.CrossCursor
        self.show_crosshair = True
        
        # Create the dialog (owned by the tool)
        self.dialog = RugosityDialog(self, annotation_window)

    def activate(self):
        """Activate the rugosity measurement tool"""
        super().activate()
        
        # Update UI based on available data
        self.dialog.update_z_controls()
        
        # Show the dialog
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def deactivate(self):
        """Deactivate the rugosity measurement tool"""
        if not self.active:
            return
            
        super().deactivate()
        
        # Hide the dialog
        self.dialog.hide()
        
        # Clear all graphics
        self.dialog.clear_all_graphics()

    def stop_current_drawing(self):
        """Stop any active drawing and clear graphics"""
        if self.dialog:
            self.dialog.stop_current_drawing()

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press - delegate to dialog"""
        # Return early if tool is not active or dialog is not visible
        if not self.active or not self.dialog.isVisible():
            return
            
        if self.dialog:
            self.dialog.handle_mouse_press(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move - delegate to dialog and update crosshair"""
        # Return early if tool is not active or dialog is not visible
        if not self.active or not self.dialog.isVisible():
            return
            
        # Call parent for crosshair handling
        super().mouseMoveEvent(event)
        
        # Delegate to dialog for tool-specific behavior
        if self.dialog:
            self.dialog.handle_mouse_move(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release - delegate to dialog"""
        if self.dialog:
            self.dialog.handle_mouse_release(event)
