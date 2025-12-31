import warnings

import traceback

import math
import numpy as np
from random import randint

import pyqtgraph as pg

from PyQt5.QtCore import Qt, QLineF, QRectF, QPoint, QPointF
from PyQt5.QtGui import QMouseEvent, QPen, QColor, QPixmap, QPainter, QBrush, QFontMetrics, QPolygonF
from PyQt5.QtWidgets import (QApplication, QDialog, QWidget, QVBoxLayout, QTabWidget,
                             QFormLayout, QDoubleSpinBox, QComboBox, QLabel,
                             QDialogButtonBox, QMessageBox, QGraphicsLineItem,
                             QGroupBox, QCheckBox, QButtonGroup, QPushButton,
                             QGraphicsRectItem, QGraphicsItemGroup, QGraphicsEllipseItem, QSpacerItem,
                             QSizePolicy, QScrollArea, QFrame)

from coralnet_toolbox.Tools.QtTool import Tool
from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_icon
from coralnet_toolbox.utilities import convert_scale_units

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Helper Class for Elevation Profile Plot
# ----------------------------------------------------------------------------------------------------------------------


class ProfilePlotDialog(QDialog):
    """
    A pop-up dialog to display a scrollable list of elevation profile plots
    using pyqtgraph.
    """
    def __init__(self, profiles_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Elevation Profile")
        self.setWindowIcon(get_icon("spatial.png"))
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
        
        # 1. Clear all old plot widgets
        for widget in self.plot_widgets:
            widget.deleteLater()
        self.plot_widgets = []
        
        if not profiles_list:
            no_data_label = QLabel("No profile data available.")
            no_data_label.setAlignment(Qt.AlignCenter)
            self.plot_layout.addWidget(no_data_label)
            self.plot_widgets.append(no_data_label)
            return

        # 2. --- Plot 1: Combined (Non-Normalized) ---
        if len(profiles_list) > 0:
            combined_plot_widget = pg.PlotWidget(title="All Profiles (Combined)")
            combined_plot = combined_plot_widget.getPlotItem()
            combined_plot.setLabel('left', 'Z')
            combined_plot.setLabel('bottom', 'Distance')
            combined_plot.addLegend()
            combined_plot.showGrid(x=True, y=True, alpha=0.3)

            for profile in profiles_list:
                x_data = profile.get('x_data', [])
                y_data = profile.get('y_data', [])
                name = profile.get('name', 'Profile')
                color = profile.get('color', (0, 0, 255))  # Default blue

                pen = pg.mkPen(color=color, width=2)
                combined_plot.plot(
                    x_data, y_data, pen=pen, name=name
                )
            
            self.plot_layout.addWidget(combined_plot_widget)
            self.plot_widgets.append(combined_plot_widget)

        # 3. --- Separator ---
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

        # 4. --- Plots 2...N: Individual Plots ---
        for profile in profiles_list:
            x_data = profile.get('x_data', [])
            y_data = profile.get('y_data', [])
            name = profile.get('name', 'Profile')
            color = profile.get('color', (0, 0, 255))
            stats = profile.get('stats', {})

            # Create a plot widget for this profile
            plot_widget = pg.PlotWidget(title=name)
            plot = plot_widget.getPlotItem()
            plot.setLabel('left', 'Z')
            plot.setLabel('bottom', 'Distance')
            plot.showGrid(x=True, y=True, alpha=0.3)

            # Plot the data
            pen = pg.mkPen(color=color, width=2)
            plot.plot(x_data, y_data, pen=pen)

            # Add statistics as text if available
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
                
        # Add a spacer at the bottom
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.plot_layout.addSpacerItem(spacer)

    def update_plot(self, profiles_list):
        """Clears and rebuilds all plots with new data."""
        try:
            self.rebuild_plots(profiles_list)
        except Exception as e:
            print(f"Error updating profile plot: {e}")


# ----------------------------------------------------------------------------------------------------------------------
# SpatialToolDialog Class
# ----------------------------------------------------------------------------------------------------------------------


class SpatialToolDialog(QDialog):
    """
    A modeless dialog for the SpatialTool, allowing measurements and spatial analysis.
    """
    def __init__(self, tool, parent=None):
        super().__init__(parent)
        # Get references from the tool
        self.tool = tool
        self.annotation_window = self.tool.annotation_window
        self.main_window = self.annotation_window.main_window

        self.setWindowTitle("Spatial Measurement Tool")
        self.setWindowIcon(get_icon("spatial.png"))
        self.setMinimumWidth(400)
        
        # This dialog is modeless
        self.setModal(False) 

        self.main_layout = QVBoxLayout(self)

        # --- Tab Widget ---
        self.tab_widget = QTabWidget()
        
        # --- Tab 1: Measure Line ---
        self.line_tab = QWidget()
        self.setup_line_tab(self.line_tab)
        self.tab_widget.addTab(self.line_tab, "Measure Line")

        # --- Tab 2: Measure Rectangle ---
        self.rect_tab = QWidget()
        self.setup_rect_tab(self.rect_tab)
        self.tab_widget.addTab(self.rect_tab, "Measure Rectangle")

        self.main_layout.addWidget(self.tab_widget)

        # Connect tab changes
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def setup_line_tab(self, tab_widget):
        """Setup the Measure Line tab"""
        layout = QVBoxLayout(tab_widget)
        
        # Instructions
        instructions = QLabel("Draw a line to measure 2D/3D distances")
        instructions.setStyleSheet("font-style: italic;")
        layout.addWidget(instructions)
        
        # --- 2D Measurements Group ---
        group_2d = QGroupBox("2D Measurements")
        form_2d = QFormLayout()
        
        self.line_length_2d_label = QLabel("---")
        self.line_total_2d_label = QLabel("---")
        
        form_2d.addRow("Distance:", self.line_length_2d_label)
        form_2d.addRow("Total:", self.line_total_2d_label)
        
        # Buttons for total management
        btn_layout = QVBoxLayout()
        self.line_add_button = QPushButton("Add to Total")
        self.line_add_button.clicked.connect(self.tool.add_line_to_total)
        self.line_clear_button = QPushButton("Clear Total")
        self.line_clear_button.clicked.connect(self.tool.clear_line_total)
        btn_layout.addWidget(self.line_add_button)
        btn_layout.addWidget(self.line_clear_button)
        form_2d.addRow("", btn_layout)
        
        # Units dropdown
        self.line_units_combo = QComboBox()
        self.line_units_combo.addItems(['m', 'cm', 'mm', 'km', 'ft', 'in', 'yd', 'mi'])
        self.line_units_combo.setCurrentText('m')
        form_2d.addRow("Units:", self.line_units_combo)
        
        group_2d.setLayout(form_2d)
        layout.addWidget(group_2d)
        
        # --- 3D Z-Metrics Group ---
        self.line_3d_group = QGroupBox("3D Z-Metrics")
        self.line_3d_group.setEnabled(False)  # Disabled until Z-data available
        
        form_3d = QFormLayout()
        
        self.line_length_3d_label = QLabel("---")
        self.line_delta_z_label = QLabel("---")
        self.line_slope_label = QLabel("---")
        self.line_rugosity_label = QLabel("---")
        
        form_3d.addRow("3D Length:", self.line_length_3d_label)
        form_3d.addRow("ΔZ:", self.line_delta_z_label)
        form_3d.addRow("Slope/Grade:", self.line_slope_label)
        form_3d.addRow("Rugosity:", self.line_rugosity_label)
        
        # Profile button
        self.line_profile_button = QPushButton("Show Elevation Profile")
        self.line_profile_button.clicked.connect(self.tool._show_elevation_profile)
        self.line_profile_button.setEnabled(False)
        form_3d.addRow("", self.line_profile_button)
        
        self.line_3d_group.setLayout(form_3d)
        layout.addWidget(self.line_3d_group)
        
        layout.addStretch()

    def setup_rect_tab(self, tab_widget):
        """Setup the Measure Rectangle tab"""
        layout = QVBoxLayout(tab_widget)
        
        # Instructions
        instructions = QLabel("Draw a rectangle to measure 2D/3D areas")
        instructions.setStyleSheet("font-style: italic;")
        layout.addWidget(instructions)
        
        # --- 2D Measurements Group ---
        group_2d = QGroupBox("2D Measurements")
        form_2d = QFormLayout()
        
        self.rect_perimeter_label = QLabel("---")
        self.rect_area_2d_label = QLabel("---")
        self.rect_total_area_label = QLabel("---")
        
        form_2d.addRow("Perimeter:", self.rect_perimeter_label)
        form_2d.addRow("Area:", self.rect_area_2d_label)
        form_2d.addRow("Total:", self.rect_total_area_label)
        
        # Buttons for total management
        btn_layout = QVBoxLayout()
        self.rect_add_button = QPushButton("Add to Total")
        self.rect_add_button.clicked.connect(self.tool.add_rect_to_total)
        self.rect_clear_button = QPushButton("Clear Total")
        self.rect_clear_button.clicked.connect(self.tool.clear_rect_total)
        btn_layout.addWidget(self.rect_add_button)
        btn_layout.addWidget(self.rect_clear_button)
        form_2d.addRow("", btn_layout)
        
        # Units dropdown
        self.rect_units_combo = QComboBox()
        self.rect_units_combo.addItems(['m', 'cm', 'mm', 'km', 'ft', 'in', 'yd', 'mi'])
        self.rect_units_combo.setCurrentText('m')
        form_2d.addRow("Units:", self.rect_units_combo)
        
        group_2d.setLayout(form_2d)
        layout.addWidget(group_2d)
        
        # --- 3D Z-Metrics Group ---
        self.rect_3d_group = QGroupBox("3D Z-Metrics")
        self.rect_3d_group.setEnabled(False)  # Disabled until Z-data available
        
        form_3d = QFormLayout()
        
        self.rect_z_stats_label = QLabel("---")
        self.rect_area_3d_label = QLabel("---")
        self.rect_volume_label = QLabel("---")
        self.rect_rugosity_label = QLabel("---")
        
        form_3d.addRow("Z (min/max/mean):", self.rect_z_stats_label)
        form_3d.addRow("3D Surface Area:", self.rect_area_3d_label)
        form_3d.addRow("Volume:", self.rect_volume_label)
        form_3d.addRow("Rugosity:", self.rect_rugosity_label)
        
        self.rect_3d_group.setLayout(form_3d)
        layout.addWidget(self.rect_3d_group)
        
        layout.addStretch()

    def on_tab_changed(self, index):
        """Handle tab change"""
        self.tool.on_tab_changed(index)

    def closeEvent(self, event):
        """Handle dialog close"""
        self.tool.deactivate()
        event.accept()


# ----------------------------------------------------------------------------------------------------------------------
# SpatialTool Class  
# ----------------------------------------------------------------------------------------------------------------------

class SpatialTool(Tool):
    """
    Tool for measuring distances, areas, perimeters, and performing spatial analysis.
    """
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.name = "spatial"
        self.cursor = Qt.CrossCursor  # Show crosshair cursor like scale tool
        
        # Drawing state
        self.current_mode = None  # 'line' or 'rect'
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        
        # Graphics items
        self.line_graphic = None
        self.rect_graphic = None
        self.wireframe_graphic = None  # For 3D visualization
        self.endpoint_dots = []  # Store endpoint circles (white and black dots)
        
        # Recorded measurements (persistent across tab switches) with color info
        self.recorded_line_measurements = []  # List of dicts with line data and color
        self.recorded_rect_measurements = []  # List of dicts with rect data and color
        
        # Measurement accumulators
        self.line_total_distance = 0.0
        self.rect_total_area = 0.0
        
        # Profile data for plotting
        self.current_profiles = []
        self.profile_dialog = None
        
        # Dialog
        self.dialog = None
        
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
        """Get current Z-channel data and settings from raster using transform pipeline"""
        image_path = self.annotation_window.current_image_path
        if not image_path:
            return None, None, None, None, None
            
        raster_manager = self.annotation_window.main_window.image_window.raster_manager
        raster = raster_manager.get_raster(image_path)
        
        # Check if scale exists (required for measurements)
        scale_x, scale_y, scale_units = self.get_current_scale()
        if scale_x is None:
            return None, None, None, None, None
        
        # Get Z-channel data using lazy loading
        if raster and raster.z_channel_lazy is not None:
            z_data = raster.z_channel_lazy
            z_unit = raster.z_unit or 'px'
            
            # Get transform settings
            z_settings = raster.z_settings
            scalar = z_settings.get('scalar', 1.0)
            offset = z_settings.get('offset', 0.0)
            direction = z_settings.get('direction', 1)
            
            return z_data, z_unit, scalar, offset, direction
        
        return None, None, None, None, None

    def _generate_random_color(self):
        """Generate a random RGB color tuple"""
        return (randint(0, 255), randint(0, 255), randint(0, 255))
    
    def _create_colored_line(self, start, end, color_rgb):
        """Create a solid colored line with endpoint dots"""
        scene = self.annotation_window.scene
        
        # Create solid line in the random color
        pen = QPen(QColor(*color_rgb), 2, Qt.SolidLine)
        pen.setCosmetic(True)
        line = QLineF(start, end)
        line_graphic = scene.addLine(line, pen)
        line_graphic.setZValue(1000)
        
        # Create white dot at start point
        start_dot = scene.addEllipse(
            start.x() - 3, start.y() - 3, 6, 6,
            QPen(QColor(255, 255, 255), 1),
            QBrush(QColor(255, 255, 255))
        )
        start_dot.setZValue(1001)
        
        # Create black dot at end point
        end_dot = scene.addEllipse(
            end.x() - 3, end.y() - 3, 6, 6,
            QPen(QColor(0, 0, 0), 1),
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
    
    def _create_colored_rect(self, rect, color_rgb):
        """Create a solid colored rectangle with endpoint dots"""
        scene = self.annotation_window.scene
        normalized_rect = rect.normalized()
        
        # Create solid rectangle outline in the random color
        pen = QPen(QColor(*color_rgb), 2, Qt.SolidLine)
        pen.setCosmetic(True)
        brush = QBrush(QColor(*color_rgb, 30))  # Semi-transparent fill
        rect_graphic = scene.addRect(normalized_rect, pen, brush)
        rect_graphic.setZValue(1000)
        
        # Create white dot at top-left
        tl_dot = scene.addEllipse(
            normalized_rect.left() - 3, normalized_rect.top() - 3, 6, 6,
            QPen(QColor(255, 255, 255), 1),
            QBrush(QColor(255, 255, 255))
        )
        tl_dot.setZValue(1001)
        
        # Create black dot at bottom-right
        br_dot = scene.addEllipse(
            normalized_rect.right() - 3, normalized_rect.bottom() - 3, 6, 6,
            QPen(QColor(0, 0, 0), 1),
            QBrush(QColor(0, 0, 0))
        )
        br_dot.setZValue(1001)
        
        return {
            'rect': rect_graphic,
            'tl_dot': tl_dot,
            'br_dot': br_dot,
            'bounds': normalized_rect,
            'color': color_rgb
        }

    def activate(self):
        """Activate the spatial measurement tool"""
        super().activate()
        
        # Create dialog if it doesn't exist
        if self.dialog is None:
            self.dialog = SpatialToolDialog(self, self.annotation_window.main_window)
        
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
        
        # Update UI based on available data
        self.update_z_controls()
        
        # Set initial mode based on current tab
        current_tab = self.dialog.tab_widget.currentIndex()
        self.on_tab_changed(current_tab)

    def deactivate(self):
        """Deactivate the spatial measurement tool"""
        self.stop_current_drawing()
        if self.dialog:
            self.dialog.close()
        super().deactivate()

    def on_tab_changed(self, index):
        """Handle tab changes - save current measurements and restore previous ones"""
        # Save current mode's measurements before switching
        if self.current_mode == 'line':
            self._save_line_measurements()
        elif self.current_mode == 'rect':
            self._save_rect_measurements()
        
        # Stop current drawing
        self.stop_current_drawing()
        
        # Set new mode
        if index == 0:  # Measure Line
            self.current_mode = 'line'
            self._restore_line_measurements()
        elif index == 1:  # Measure Rectangle
            self.current_mode = 'rect'
            self._restore_rect_measurements()

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
        if self.rect_graphic and self.rect_graphic.scene():
            scene.removeItem(self.rect_graphic)
            self.rect_graphic = None
        if self.wireframe_graphic and self.wireframe_graphic.scene():
            scene.removeItem(self.wireframe_graphic)
            self.wireframe_graphic = None

    def update_z_controls(self):
        """Enable/disable Z-controls based on data availability"""
        z_data, z_unit, scalar, offset, direction = self.get_current_z_data()
        has_z = z_data is not None
        
        if self.dialog:
            self.dialog.line_3d_group.setEnabled(has_z)
            self.dialog.rect_3d_group.setEnabled(has_z)
            self.dialog.line_profile_button.setEnabled(False)  # Enable after line drawn

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for starting measurements"""
        if event.button() == Qt.LeftButton:
            scene_pos = self.annotation_window.mapToScene(event.pos())
            
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
                self.end_point = scene_pos
                self.is_drawing = False
                
                # Generate random color for this measurement
                color = self._generate_random_color()
                
                # Record the measurement with color
                if self.current_mode == 'line':
                    measurement = self._create_colored_line(self.start_point, self.end_point, color)
                    self.recorded_line_measurements.append(measurement)
                    
                    # Final calculation
                    self.calculate_line_measurement(final_calc=True, color=color)
                    self._update_wireframe_graphic()  # Add 3D wireframe if available
                    self.update_profile_button_state()
                elif self.current_mode == 'rect':
                    rect = QRectF(self.start_point, self.end_point)
                    measurement = self._create_colored_rect(rect, color)
                    self.recorded_rect_measurements.append(measurement)
                    
                    # Final calculation
                    self.calculate_rect_measurement(final_calc=True)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for drawing measurements"""
        if self.is_drawing and self.start_point and self.current_mode:
            scene_pos = self.annotation_window.mapToScene(event.pos())
            self.end_point = scene_pos
            
            # Update graphics and measurements based on mode
            if self.current_mode == 'line':
                self._update_line_graphic()
                self.calculate_line_measurement(final_calc=False)
            elif self.current_mode == 'rect':
                self._update_rect_graphic()
                self.calculate_rect_measurement(final_calc=False)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release - measurement is finalized on second click"""
        pass

    def _save_line_measurements(self):
        """Save current line graphics to recorded list - already done in mousePressEvent"""
        # Measurements are now recorded immediately when finalized
        pass
    
    def _restore_line_measurements(self):
        """Restore previously recorded line graphics"""
        scene = self.annotation_window.scene
        for measurement in self.recorded_line_measurements:
            # Re-add all graphics if they're not in the scene
            for key in ['line', 'start_dot', 'end_dot']:
                if key in measurement and measurement[key]:
                    if measurement[key].scene() != scene:
                        scene.addItem(measurement[key])
    
    def _save_rect_measurements(self):
        """Save current rect graphics to recorded list - already done in mousePressEvent"""
        # Measurements are now recorded immediately when finalized
        pass
    
    def _restore_rect_measurements(self):
        """Restore previously recorded rect graphics"""
        scene = self.annotation_window.scene
        for measurement in self.recorded_rect_measurements:
            # Re-add all graphics if they're not in the scene
            for key in ['rect', 'tl_dot', 'br_dot']:
                if key in measurement and measurement[key]:
                    if measurement[key].scene() != scene:
                        scene.addItem(measurement[key])

    def _update_line_graphic(self):
        """Update or create line graphic with cosmetic pen (visible at all zoom levels)"""
        if not self.start_point or not self.end_point:
            return
            
        scene = self.annotation_window.scene
        
        # Remove old graphic
        if self.line_graphic and self.line_graphic.scene():
            scene.removeItem(self.line_graphic)
        
        # Create new line with cosmetic pen (doesn't scale with zoom)
        pen = QPen(QColor(255, 0, 0), 2, Qt.SolidLine)
        pen.setCosmetic(True)  # Make pen width independent of zoom level
        line = QLineF(self.start_point, self.end_point)
        self.line_graphic = scene.addLine(line, pen)
        self.line_graphic.setZValue(1000)  # Draw on top

    def _update_rect_graphic(self):
        """Update or create rectangle graphic with cosmetic pen (visible at all zoom levels)"""
        if not self.start_point or not self.end_point:
            return
            
        scene = self.annotation_window.scene
        
        # Remove old graphic
        if self.rect_graphic and self.rect_graphic.scene():
            scene.removeItem(self.rect_graphic)
        
        # Create rectangle with cosmetic pen (doesn't scale with zoom)
        rect = QRectF(self.start_point, self.end_point).normalized()
        pen = QPen(QColor(255, 0, 0), 2, Qt.SolidLine)
        pen.setCosmetic(True)  # Make pen width independent of zoom level
        brush = QBrush(QColor(255, 0, 0, 30))  # Semi-transparent fill
        self.rect_graphic = scene.addRect(rect, pen, brush)
        self.rect_graphic.setZValue(1000)

    def _update_wireframe_graphic(self):
        """Create 3D wireframe visualization for line measurement"""
        # This will be implemented in the next part
        # For now, just a placeholder
        pass

    def _show_elevation_profile(self):
        """Show elevation profile plot dialog"""
        if not self.current_profiles:
            QMessageBox.warning(self.dialog, "No Data", "No profile data available to display.")
            return
        
        if self.profile_dialog is None or not self.profile_dialog.isVisible():
            self.profile_dialog = ProfilePlotDialog(self.current_profiles, self.dialog)
        else:
            self.profile_dialog.update_plot(self.current_profiles)
        
        self.profile_dialog.show()
        self.profile_dialog.raise_()

    def update_profile_button_state(self):
        """Enable/disable profile button based on data availability"""
        has_data = len(self.current_profiles) > 0
        if self.dialog:
            self.dialog.line_profile_button.setEnabled(has_data)

    # Calculation methods - adapted to use transform pipeline
    
    def calculate_line_measurement(self, final_calc=False, color=None):
        """Calculate line measurements (2D and 3D) using transform pipeline"""
        if not self.start_point or not self.end_point:
            return
            
        # --- 1. Get Data ---
        scale_x, scale_y, scale_units = self.get_current_scale()
        
        # Check for invalid scale
        if scale_x is None or scale_y is None:
            if self.dialog:
                self.dialog.line_length_2d_label.setText("Scale Not Set")
                self._reset_line_3d_labels()
            return

        display_units = self.dialog.line_units_combo.currentText() if self.dialog else 'm'
        
        # --- 2. Calculate 2D Length ---
        line = QLineF(self.start_point, self.end_point)
        pixel_length = line.length()
        length_2d_meters = pixel_length * scale_x  # Assume square pixels
        
        # Convert to display units
        if display_units != "m":
            length_2d_display = convert_scale_units(length_2d_meters, 'metre', display_units)
        else:
            length_2d_display = length_2d_meters
        
        if self.dialog:
            self.dialog.line_length_2d_label.setText(f"{length_2d_display:.3f} {display_units}")

        # --- 3. 3D Calculations (if Z-data available) ---
        z_data, z_unit, scalar, offset, direction = self.get_current_z_data()
        
        if z_data is None:
            if self.dialog:
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

            # Get transformed Z values at start/end using get_z_value
            p1_x = int(max(0, min(self.start_point.x(), w - 1)))
            p1_y = int(max(0, min(self.start_point.y(), h - 1)))
            p2_x = int(max(0, min(self.end_point.x(), w - 1)))
            p2_y = int(max(0, min(self.end_point.y(), h - 1)))
            
            z_start = raster.get_z_value(p1_x, p1_y)
            z_end = raster.get_z_value(p2_x, p2_y)
            
            if z_start is None or z_end is None:
                if self.dialog:
                    self._reset_line_3d_labels()
                return
            
            delta_z = z_end - z_start
            if self.dialog:
                self.dialog.line_delta_z_label.setText(f"{delta_z:.3f} {z_unit_str}")
            
            # Calculate Slope
            if length_2d_meters > 0:
                delta_z_meters = delta_z * z_to_meters_factor
                slope = (delta_z_meters / length_2d_meters) * 100.0
                if self.dialog:
                    self.dialog.line_slope_label.setText(f"{slope:.2f} %")
            else:
                if self.dialog:
                    self.dialog.line_slope_label.setText("N/A")

            # --- Calculate 3D Length & Linear Rugosity ---
            # Sample points along the line
            num_samples = max(2, int(pixel_length / 2))  # Sample every 2 pixels
            x_samples = np.linspace(self.start_point.x(), self.end_point.x(), num_samples)
            y_samples = np.linspace(self.start_point.y(), self.end_point.y(), num_samples)
            
            # Get RAW Z-values along the line
            raw_z_samples = []
            for i in range(num_samples):
                x_idx = int(max(0, min(x_samples[i], w - 1)))
                y_idx = int(max(0, min(y_samples[i], h - 1)))
                raw_z_samples.append(z_data[y_idx, x_idx])
            
            raw_z_array = np.array(raw_z_samples)
            
            # Apply transform vectorially: Z_transformed = direction * (Raw * scalar) + offset
            z_transformed_array = (raw_z_array * scalar * direction) + offset
            
            # Calculate 3D length using transformed Z-values
            profile_data_x = []  # For plot
            profile_data_y = []  # For plot
            total_3d_length = 0.0
            dist_2d_so_far = 0.0
            
            profile_data_x.append(0.0)
            profile_data_y.append(z_transformed_array[0])
            
            for i in range(num_samples - 1):
                # Get segment start/end points (pixel coords)
                x_a, y_a = x_samples[i], y_samples[i]
                x_b, y_b = x_samples[i + 1], y_samples[i + 1]
                
                # Get transformed Z values
                z_a = z_transformed_array[i]
                z_b = z_transformed_array[i + 1]
                
                # Get segment components in meters
                dx_m = (x_b - x_a) * scale_x
                dy_m = (y_b - y_a) * scale_y
                dz_meters = (z_b - z_a) * z_to_meters_factor
                
                # Add 3D segment length
                total_3d_length += math.sqrt(dx_m**2 + dy_m**2 + dz_meters**2)
                
                # Add data for plot
                dist_2d_so_far += math.sqrt(dx_m**2 + dy_m**2)
                profile_data_x.append(dist_2d_so_far)
                profile_data_y.append(z_b)

            # Convert 3D length to display units
            if display_units != "m":
                length_3d_display = convert_scale_units(total_3d_length, 'metre', display_units)
                # Also convert plot x-axis
                conv_factor = convert_scale_units(1.0, 'metre', display_units)
                plot_x_data = [x * conv_factor for x in profile_data_x]
                plot_x_label = f"Distance ({display_units})"
            else:
                length_3d_display = total_3d_length
                plot_x_data = profile_data_x
                plot_x_label = "Distance (m)"
            
            if self.dialog:
                self.dialog.line_length_3d_label.setText(f"{length_3d_display:.3f} {display_units}")

            # Calculate Linear Rugosity
            if length_2d_meters > 0:
                linear_rugosity = total_3d_length / length_2d_meters
                if self.dialog:
                    self.dialog.line_rugosity_label.setText(f"{linear_rugosity:.3f}")
            else:
                if self.dialog:
                    self.dialog.line_rugosity_label.setText("N/A")
                    
            # Store profile data
            if final_calc:
                plot_y_label = f"Z ({z_unit_str})" if direction == 1 else f"Elevation ({z_unit_str})"
                
                profile_data = {
                    'name': 'Current Line',
                    'color': color if color else (255, 128, 0),  # Use measurement color or default orange
                    'x_data': plot_x_data,
                    'y_data': profile_data_y,
                    'start_point': self.start_point,
                    'end_point': self.end_point,
                    'stats': {
                        'length_3d': length_3d_display,
                        'delta_z': delta_z,
                        'rugosity': linear_rugosity if length_2d_meters > 0 else 0
                    }
                }
                
                self.current_profiles = [profile_data]
                self.update_profile_button_state()
                
        except Exception as e:
            print(f"Error in 3D line calculation: {e}")
            import traceback
            traceback.print_exc()
            if self.dialog:
                self._reset_line_3d_labels()

    def calculate_rect_measurement(self, final_calc=False):
        """Calculate rectangle measurements (2D and 3D) using transform pipeline"""
        if not self.start_point or not self.end_point:
            return
            
        # --- 1. Get Data ---
        scale_x, scale_y, scale_units = self.get_current_scale()

        # Check for invalid scale
        if scale_x is None or scale_y is None:
            if self.dialog:
                self.dialog.rect_perimeter_label.setText("Scale Not Set")
                self.dialog.rect_area_2d_label.setText("Scale Not Set")
                self._reset_rect_3d_labels()
            return
            
        display_units = self.dialog.rect_units_combo.currentText() if self.dialog else 'm'
        area_units = f"{display_units}²" if display_units != "px" else "px²"

        rect = QRectF(self.start_point, self.end_point).normalized()
        pixel_width = rect.width()
        pixel_height = rect.height()
        
        # Only reject tiny rectangles on final calculation
        if final_calc and (pixel_width < 1 or pixel_height < 1):
            if self.dialog:
                self.dialog.rect_perimeter_label.setText("Too Small")
                self.dialog.rect_area_2d_label.setText("Too Small")
                self._reset_rect_3d_labels()
            return
        elif not final_calc and (pixel_width < 1 or pixel_height < 1):
            return

        # --- 2. 2D Calculations ---
        real_width_m = pixel_width * scale_x
        real_height_m = pixel_height * scale_y
        area_2d_meters = real_width_m * real_height_m
        
        # Convert to display units
        if display_units != "m":
            real_width_display = convert_scale_units(real_width_m, 'metre', display_units)
            real_height_display = convert_scale_units(real_height_m, 'metre', display_units)
        else:
            real_width_display = real_width_m
            real_height_display = real_height_m
        
        perimeter_display = 2 * (real_width_display + real_height_display)
        area_2d_display = real_width_display * real_height_display
        
        if self.dialog:
            self.dialog.rect_perimeter_label.setText(f"{perimeter_display:.3f} {display_units}")
            self.dialog.rect_area_2d_label.setText(f"{area_2d_display:.3f} {area_units}")

        # --- 3. 3D Calculations (if Z-data available) ---
        z_data, z_unit, scalar, offset, direction = self.get_current_z_data()
        
        if z_data is None:
            if self.dialog:
                self._reset_rect_3d_labels()
            return

        try:
            h, w = z_data.shape
            z_unit_str = z_unit if z_unit else 'px'
            
            # Convert z_unit to meters for 3D calculations
            z_to_meters_factor = convert_scale_units(1.0, z_unit, 'metre') if z_unit else 1.0
            
            # Get integer bounds for slicing, clamped to raster dims
            x1 = max(0, int(math.floor(rect.left())))
            y1 = max(0, int(math.floor(rect.top())))
            x2 = min(w, int(math.ceil(rect.right())))
            y2 = min(h, int(math.ceil(rect.bottom())))
            
            if x1 >= x2 or y1 >= y2:
                if self.dialog:
                    self._reset_rect_3d_labels()
                return

            # Get RAW Z-Slice
            raw_z_slice = z_data[y1:y2, x1:x2]
            if raw_z_slice.size == 0:
                if self.dialog:
                    self._reset_rect_3d_labels()
                return

            # Apply transform vectorially: Z_transformed = direction * (Raw * scalar) + offset
            z_slice_transformed = (raw_z_slice * scalar * direction) + offset

            # Calculate Z-Stats (using transformed values)
            z_min = np.min(z_slice_transformed)
            z_max = np.max(z_slice_transformed)
            z_mean = np.mean(z_slice_transformed)
            
            if self.dialog:
                self.dialog.rect_z_stats_label.setText(
                    f"Min: {z_min:.2f} | Max: {z_max:.2f} | Mean: {z_mean:.2f} ({z_unit_str})"
                )

            # Calculate Prismatic Volume (using transformed Z-values)
            pixel_area_2d = scale_x * scale_y
            volume = np.sum(z_slice_transformed) * pixel_area_2d
            vol_units = f"{display_units}² · {z_unit_str}"
            
            if self.dialog:
                self.dialog.rect_volume_label.setText(f"{volume:.3f} {vol_units}")

            # Calculate 3D Surface Area
            # Convert transformed z_slice to meters
            z_slice_meters = z_slice_transformed * z_to_meters_factor
            
            # Calculate gradients with proper spacing (all in meters)
            dz_dy, dz_dx = np.gradient(z_slice_meters, scale_y, scale_x)
            multiplier = np.sqrt(1.0 + dz_dx**2 + dz_dy**2)
            pixel_areas_3d = pixel_area_2d * multiplier
            surface_area_3d_meters = np.sum(pixel_areas_3d)

            # Convert to display units
            if display_units != "m":
                conv_factor = convert_scale_units(1.0, 'metre', display_units)
                area_conv_factor = conv_factor * conv_factor
                surface_area_3d_display = surface_area_3d_meters * area_conv_factor
            else:
                surface_area_3d_display = surface_area_3d_meters
            
            if self.dialog:
                self.dialog.rect_area_3d_label.setText(f"{surface_area_3d_display:.3f} {area_units}")

            # Calculate Areal Rugosity
            if area_2d_meters > 0:
                areal_rugosity = surface_area_3d_meters / area_2d_meters
                if self.dialog:
                    self.dialog.rect_rugosity_label.setText(f"{areal_rugosity:.3f}")
            else:
                if self.dialog:
                    self.dialog.rect_rugosity_label.setText("N/A")
                
        except Exception as e:
            print(f"Error in 3D rect calculation: {e}")
            import traceback
            traceback.print_exc()
            if self.dialog:
                self._reset_rect_3d_labels()

    def _reset_line_3d_labels(self):
        """Reset all 3D line labels"""
        if self.dialog:
            self.dialog.line_length_3d_label.setText("---")
            self.dialog.line_delta_z_label.setText("---")
            self.dialog.line_slope_label.setText("---")
            self.dialog.line_rugosity_label.setText("---")

    def _reset_rect_3d_labels(self):
        """Reset all 3D rectangle labels"""
        if self.dialog:
            self.dialog.rect_z_stats_label.setText("---")
            self.dialog.rect_area_3d_label.setText("---")
            self.dialog.rect_volume_label.setText("---")
            self.dialog.rect_rugosity_label.setText("---")

    def add_line_to_total(self):
        """Add current line measurement to total"""
        if not self.start_point or not self.end_point:
            QMessageBox.warning(self.dialog, "Warning", "No line measurement to add")
            return
            
        # Recalculate current measurement in meters
        scale_x, scale_y, scale_units = self.get_current_scale()
        if scale_x is None or scale_y is None:
            QMessageBox.warning(self.dialog, "Warning", "Scale not set - cannot add to total")
            return
            
        line = QLineF(self.start_point, self.end_point)
        pixel_length = line.length()
        length_meters = pixel_length * scale_x  # Assume square pixels
        
        # Add to total
        self.line_total_distance += length_meters
        
        # Update display
        self._update_line_total_display()
        
    def clear_line_total(self):
        """Clear line total"""
        self.line_total_distance = 0.0
        self._update_line_total_display()
        
    def add_rect_to_total(self):
        """Add current rect measurement to total"""
        if not self.start_point or not self.end_point:
            QMessageBox.warning(self.dialog, "Warning", "No rectangle measurement to add")
            return
            
        # Recalculate current measurement in meters
        scale_x, scale_y, scale_units = self.get_current_scale()
        if scale_x is None or scale_y is None:
            QMessageBox.warning(self.dialog, "Warning", "Scale not set - cannot add to total")
            return
            
        rect = QRectF(self.start_point, self.end_point).normalized()
        pixel_width = rect.width()
        pixel_height = rect.height()
        
        if pixel_width < 1 or pixel_height < 1:
            QMessageBox.warning(self.dialog, "Warning", "Rectangle too small to add to total")
            return
            
        real_width_m = pixel_width * scale_x
        real_height_m = pixel_height * scale_y
        area_meters = real_width_m * real_height_m
        
        # Add to total
        self.rect_total_area += area_meters
        
        # Update display
        self._update_rect_total_display()
        
    def clear_rect_total(self):
        """Clear rect total"""
        self.rect_total_area = 0.0
        self._update_rect_total_display()
        
    def _update_line_total_display(self):
        """Update the line total display label"""
        if not self.dialog:
            return
            
        display_units = self.dialog.line_units_combo.currentText()
        
        # Convert total from meters to display units
        if display_units != "m":
            total_display = convert_scale_units(self.line_total_distance, 'metre', display_units)
        else:
            total_display = self.line_total_distance
            
        self.dialog.line_total_2d_label.setText(f"{total_display:.3f} {display_units}")
        
    def _update_rect_total_display(self):
        """Update the rect total display label"""
        if not self.dialog:
            return
            
        display_units = self.dialog.rect_units_combo.currentText()
        area_units = f"{display_units}²" if display_units != "px" else "px²"
        
        # Convert total area from square meters to display units
        if display_units != "m":
            # Convert linear dimension first, then square
            linear_total = convert_scale_units(self.rect_total_area ** 0.5, 'metre', display_units)
            total_display = linear_total ** 2
        else:
            total_display = self.rect_total_area
            
        self.dialog.rect_total_area_label.setText(f"{total_display:.3f} {area_units}")
