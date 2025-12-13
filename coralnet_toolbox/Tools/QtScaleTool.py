import warnings
import math
import numpy as np

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
        self.setWindowIcon(get_icon("scale.png"))
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
            label = QLabel("No profiles to display. Draw a line and 'Add to Total'.")
            label.setAlignment(Qt.AlignCenter)
            self.plot_layout.addWidget(label)
            self.plot_widgets.append(label)
            return

        # 2. --- Plot 1: Combined (Non-Normalized) ---
        if len(profiles_list) > 0:
            combined_plot_widget = pg.PlotWidget()
            combined_plot_widget.setMinimumHeight(350)
            combined_plot_widget.setMinimumWidth(600)
            combined_plot_item = combined_plot_widget.getPlotItem()
            combined_plot_item.setTitle("Combined Profiles")
            
            combined_plot_item.setLabel('bottom', '')
            combined_plot_item.setLabel('left', "Elevation / Z-Value")
            combined_plot_item.showGrid(x=True, y=True, alpha=0.3)
            combined_plot_item.addLegend()
            
            # Enable antialiasing for better cross-platform rendering
            combined_plot_widget.setAntialiasing(True)
            
            for profile in profiles_list:
                try:
                    x_data = np.array(profile["x_data"])
                    if x_data.size > 0:
                        combined_plot_item.plot(
                            profile["x_data"],  # Use original x_data
                            profile["y_data"],
                            pen=profile["color"],
                            name=profile["name"]
                        )
                        # Add start and end points
                        combined_plot_item.plot([profile["x_data"][0]], 
                                                [profile["y_data"][0]], 
                                                pen=None, symbol='o', symbolBrush='w', symbolPen='k', symbolSize=8)
                        
                        combined_plot_item.plot([profile["x_data"][-1]], 
                                                [profile["y_data"][-1]], 
                                                pen=None, symbol='o', symbolBrush='k', symbolPen='k', symbolSize=8)
                except Exception as e:
                    print(f"Error plotting combined profile '{profile['name']}': {e}")

            combined_plot_item.enableAutoRange()
            
            # Zoom out a bit more
            x_range = combined_plot_item.getAxis('bottom').range
            y_range = combined_plot_item.getAxis('left').range
            x_center = (x_range[0] + x_range[1]) / 2
            x_span = x_range[1] - x_range[0]
            y_center = (y_range[0] + y_range[1]) / 2
            y_span = y_range[1] - y_range[0]
            zoom_factor = 1.2  # Zoom out by 20%
            combined_plot_item.setXRange(x_center - x_span * zoom_factor / 2, x_center + x_span * zoom_factor / 2)
            combined_plot_item.setYRange(y_center - y_span * zoom_factor / 2, y_center + y_span * zoom_factor / 2)

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
            try:
                ind_plot_widget = pg.PlotWidget()
                ind_plot_widget.setMinimumHeight(300)
                ind_plot_widget.setMinimumWidth(600)
                ind_plot_item = ind_plot_widget.getPlotItem()
                
                title = profile["name"]
                stats_str = profile.get("stats_str")  # Get the pre-formatted string
                if stats_str:
                    title += f"  ({stats_str})"
                ind_plot_item.setTitle(title)
                    
                ind_plot_item.setLabel('bottom', '')
                ind_plot_item.setLabel('left', profile["y_label"])
                ind_plot_item.showGrid(x=True, y=True, alpha=0.3)
                
                # Enable antialiasing for better cross-platform rendering
                ind_plot_widget.setAntialiasing(True)
                
                ind_plot_item.plot(
                    profile["x_data"],
                    profile["y_data"],
                    pen=profile["color"]
                )
                # Add start and end points
                ind_plot_item.plot([profile["x_data"][0]], 
                                   [profile["y_data"][0]], 
                                   pen=None, symbol='o', symbolBrush='w', symbolPen='k', symbolSize=8)
                
                ind_plot_item.plot([profile["x_data"][-1]], 
                                   [profile["y_data"][-1]], 
                                   pen=None, symbol='o', symbolBrush='k', symbolPen='k', symbolSize=8)
                
                ind_plot_item.enableAutoRange()
                
                # Zoom out a bit more
                x_range = ind_plot_item.getAxis('bottom').range
                y_range = ind_plot_item.getAxis('left').range
                x_center = (x_range[0] + x_range[1]) / 2
                x_span = x_range[1] - x_range[0]
                y_center = (y_range[0] + y_range[1]) / 2
                y_span = y_range[1] - y_range[0]
                zoom_factor = 1.2  # Zoom out by 20%
                ind_plot_item.setXRange(x_center - x_span * zoom_factor / 2, x_center + x_span * zoom_factor / 2)
                ind_plot_item.setYRange(y_center - y_span * zoom_factor / 2, y_center + y_span * zoom_factor / 2)
                
                self.plot_layout.addWidget(ind_plot_widget)
                self.plot_widgets.append(ind_plot_widget)
            except Exception as e:
                print(f"Error plotting individual profile '{profile['name']}': {e}")
                
        # Add a spacer at the bottom
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.plot_layout.addSpacerItem(spacer)

    def update_plot(self, profiles_list):
        """Clears and rebuilds all plots with new data."""
        try:
            self.rebuild_plots(profiles_list)
        except Exception as e:
            print(f"Error updating plot: {e}")


# ----------------------------------------------------------------------------------------------------------------------
# ScaleToolDialog Class
# ----------------------------------------------------------------------------------------------------------------------


class ScaleToolDialog(QDialog):
    """
    A modeless dialog for the ScaleTool, allowing user input for scale calculation
    and propagation, as well as measurement tools.
    """
    def __init__(self, tool, parent=None):
        super().__init__(parent)
        # Get references from the tool
        self.tool = tool
        self.annotation_window = self.tool.annotation_window
        self.main_window = self.annotation_window.main_window

        self.setWindowTitle("Scale Tool")
        self.setWindowIcon(get_icon("scale.png"))
        self.setMinimumWidth(400)
        
        # This dialog is modeless
        self.setModal(False) 

        self.main_layout = QVBoxLayout(self)

        # --- Tab Widget ---
        self.tab_widget = QTabWidget()
        
        # --- Tab 1: Set Scale ---
        self.scale_tab = QWidget()
        self.setup_scale_tab(self.scale_tab)
        self.tab_widget.addTab(self.scale_tab, "Set Scale")

        # --- Tab 2: Measure Line ---
        self.line_tab = QWidget()
        self.setup_line_tab(self.line_tab)
        self.tab_widget.addTab(self.line_tab, "Measure Line")

        # --- Tab 3: Measure Rectangle ---
        self.rect_tab = QWidget()
        self.setup_rect_tab(self.rect_tab)
        self.tab_widget.addTab(self.rect_tab, "Measure Rectangle")

        self.main_layout.addWidget(self.tab_widget)  # Add tab widget FIRST

        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Close)
        
        # Rename "Apply" to "Set Scale" for clarity
        self.set_scale_button = self.button_box.button(QDialogButtonBox.Apply)
        self.set_scale_button.setText("Set Scale")
        
        self.main_layout.addWidget(self.button_box)
        
        # --- Status Label (at bottom of dialog) ---
        self.status_label = QLabel("No images highlighted")
        self.status_label.setAlignment(Qt.AlignLeft)
        self.main_layout.addWidget(self.status_label)
        
        # Signal connection will be made in activate() when image_window is guaranteed to exist
        self._signal_connected = False

    def setup_scale_tab(self, tab_widget):
        """Populates the 'Set Scale' tab."""
        self.scale_layout = QFormLayout(tab_widget)
        
        self.known_length_input = QDoubleSpinBox()
        self.known_length_input.setRange(0.001, 1000000.0)
        self.known_length_input.setValue(1.0)
        self.known_length_input.setDecimals(3)
        self.known_length_input.setToolTip("Enter the real-world length of the line you will draw.")
        
        self.units_combo = QComboBox()
        self.units_combo.addItems(["mm", "cm", "m", "km", "in", "ft", "yd", "mi"])
        self.units_combo.setCurrentText("m")
        self.units_combo.setToolTip("Select the units for the known length.")

        self.pixel_length_label = QLabel("Draw a line on the image")
        self.pixel_length_label.setToolTip("The length of the drawn line in pixels.")
        
        self.calculated_scale_label = QLabel("Scale: N/A")
        self.calculated_scale_label.setToolTip("The resulting scale in meters per pixel.")

        self.scale_layout.addRow("Known Length:", self.known_length_input)
        self.scale_layout.addRow("Units:", self.units_combo)
        self.scale_layout.addRow("Pixel Length:", self.pixel_length_label)
        self.scale_layout.addRow("Scale:", self.calculated_scale_label)
        
        # --- Danger Zone (Collapsible) ---
        self.danger_zone_group_box = QGroupBox("Danger Zone")
        self.danger_zone_group_box.setCheckable(True)
        self.danger_zone_group_box.setChecked(False)  # Collapsed by default

        # Create a container widget to hold the buttons
        danger_zone_container = QWidget()
        danger_zone_layout = QVBoxLayout(danger_zone_container)
        danger_zone_layout.setContentsMargins(0, 0, 0, 0)

        # Button for removing scale from highlighted images (styled in red)
        self.remove_highlighted_button = QPushButton("Remove Scale from Highlighted Images")
        self.remove_highlighted_button.setToolTip("Removes the scale data from all highlighted images.")
        self.remove_highlighted_button.setStyleSheet(
            "background-color: #D9534F; color: white; font-weight: bold;"
        )
        danger_zone_layout.addWidget(self.remove_highlighted_button)

        # Set the container as the group box layout
        group_layout = QVBoxLayout()
        group_layout.addWidget(danger_zone_container)
        self.danger_zone_group_box.setLayout(group_layout)

        # Connect the toggled signal to show/hide the container
        self.danger_zone_group_box.toggled.connect(danger_zone_container.setVisible)
        danger_zone_container.setVisible(False)  # Start hidden

        self.scale_layout.addRow(self.danger_zone_group_box)

    def setup_line_tab(self, tab_widget):
        """Populates the 'Measure Line' tab."""
        layout = QFormLayout(tab_widget)
        
        self.line_units_combo = QComboBox()
        self.line_units_combo.addItems(["mm", "cm", "m", "km", "in", "ft", "yd", "mi"])
        self.line_units_combo.setCurrentText("m")
        self.line_units_combo.setToolTip("Select the units for displaying measurements.")
        
        self.line_length_label = QLabel("N/A")
        self.line_total_length_label = QLabel("0.0")

        self.line_add_button = QPushButton("Add to Total")
        self.line_clear_button = QPushButton("Clear Total")
        self.line_add_button.setEnabled(False)

        layout.addRow("Display Units:", self.line_units_combo)
        layout.addRow("2D Length:", self.line_length_label)
        layout.addRow("Total 2D Length:", self.line_total_length_label)
        layout.addRow(self.line_add_button)
        layout.addRow(self.line_clear_button)

        # --- 3D Metrics ---
        self.line_3d_group = QGroupBox("3D Z-Metrics")
        self.line_3d_group.setToolTip("Requires a loaded Z-Channel to activate.")
        self.line_3d_layout = QFormLayout()
        self.line_3d_group.setLayout(self.line_3d_layout)

        self.line_3d_length_label = QLabel("N/A")
        self.line_delta_z_label = QLabel("N/A")
        self.line_slope_label = QLabel("N/A")
        self.line_rugosity_label = QLabel("N/A")
        
        self.line_3d_layout.addRow("3D Surface Length:", self.line_3d_length_label)
        self.line_3d_layout.addRow("ΔZ (Elevation Change):", self.line_delta_z_label)
        self.line_3d_layout.addRow("Slope / Grade:", self.line_slope_label)
        self.line_3d_layout.addRow("Linear Rugosity:", self.line_rugosity_label)
        
        # --- New Visual Elements ---
        self.line_profile_button = QPushButton("Show Elevation Profile")
        self.line_profile_button.setEnabled(False)
        self.line_3d_layout.addRow(self.line_profile_button)
        # --- End New ---
        
        self.line_3d_group.setEnabled(False)  # Disabled by default
        layout.addRow(self.line_3d_group)

    def setup_rect_tab(self, tab_widget):
        """Populates the 'Measure Rectangle' tab."""
        layout = QFormLayout(tab_widget)
        
        self.rect_units_combo = QComboBox()
        self.rect_units_combo.addItems(["mm", "cm", "m", "km", "in", "ft", "yd", "mi"])
        self.rect_units_combo.setCurrentText("m")
        self.rect_units_combo.setToolTip("Select the units for displaying measurements.")
        
        self.rect_perimeter_label = QLabel("N/A")
        self.rect_area_label = QLabel("N/A")
        self.rect_total_area_label = QLabel("0.0")
        
        self.rect_add_button = QPushButton("Add Area to Total")
        self.rect_clear_button = QPushButton("Clear Total")
        self.rect_add_button.setEnabled(False)

        layout.addRow("Display Units:", self.rect_units_combo)
        layout.addRow("2D Perimeter:", self.rect_perimeter_label)
        layout.addRow("2D Area:", self.rect_area_label)
        layout.addRow("Total 2D Area:", self.rect_total_area_label)
        layout.addRow(self.rect_add_button)
        layout.addRow(self.rect_clear_button)
        
        # --- 3D Metrics ---
        self.rect_3d_group = QGroupBox("3D Z-Metrics")
        self.rect_3d_group.setToolTip("Requires a loaded Z-Channel to activate.")
        self.rect_3d_layout = QFormLayout()
        self.rect_3d_group.setLayout(self.rect_3d_layout)
        
        self.rect_z_stats_label = QLabel("N/A")
        self.rect_3d_surface_area_label = QLabel("N/A")
        self.rect_volume_label = QLabel("N/A")
        self.rect_rugosity_label = QLabel("N/A")
        
        self.rect_3d_layout.addRow("Z-Stats (Min/Max/Mean):", self.rect_z_stats_label)
        self.rect_3d_layout.addRow("3D Surface Area:", self.rect_3d_surface_area_label)
        self.rect_3d_layout.addRow("Prismatic Volume:", self.rect_volume_label)
        self.rect_3d_layout.addRow("Areal Rugosity:", self.rect_rugosity_label)
        
        self.rect_3d_group.setEnabled(False)  # Disabled by default
        layout.addRow(self.rect_3d_group)

    def get_selected_image_paths(self):
        """
        Get the selected image paths - only highlighted rows.
        
        :return: List of highlighted image paths
        """
        # Get highlighted image paths from the table model
        return self.main_window.image_window.table_model.get_highlighted_paths()

    def reset_fields(self):
        """Resets the dialog fields to their default state."""
        # Reset Set Scale Tab
        self.pixel_length_label.setText("Draw a line on the image")
        self.calculated_scale_label.setText("Scale: N/A")
        
        # Reset Line Tab
        self.line_length_label.setText("N/A")
        self.line_total_length_label.setText("0.0")
        self.line_add_button.setEnabled(False)
        self.line_units_combo.setCurrentText("m")  # Reset to meters
        self.line_3d_group.setEnabled(False)
        self.line_3d_length_label.setText("N/A")
        self.line_delta_z_label.setText("N/A")
        self.line_slope_label.setText("N/A")
        self.line_rugosity_label.setText("N/A")
        self.line_profile_button.setEnabled(False)
        
        # Reset Rect Tab
        self.rect_perimeter_label.setText("N/A")
        self.rect_area_label.setText("N/A")
        self.rect_total_area_label.setText("0.0")
        self.rect_add_button.setEnabled(False)
        self.rect_units_combo.setCurrentText("m")  # Reset to meters
        self.rect_3d_group.setEnabled(False)
        self.rect_z_stats_label.setText("N/A")
        self.rect_3d_surface_area_label.setText("N/A")
        self.rect_volume_label.setText("N/A")
        self.rect_rugosity_label.setText("N/A")

        # Reset locked flags and enable combos
        self.line_total_locked = False
        self.rect_total_locked = False
        self.line_units_combo.setEnabled(True)
        self.rect_units_combo.setEnabled(True)
    
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
        This should deactivate the tool.
        """
        # Get the tool and call its deactivate method
        # self.tool is ScaleTool
        self.tool.deactivate()
        event.accept()

# ----------------------------------------------------------------------------------------------------------------------
# ScaleTool Class
# ----------------------------------------------------------------------------------------------------------------------

class ScaleTool(Tool):
    """
    Tool for setting image scale and measuring distances, areas, and perimeters.
    """
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.cursor = Qt.CrossCursor
        self.dialog = ScaleToolDialog(self, self.annotation_window)  # tool, parent
        
        # --- Mode Management ---
        # 0: Set Scale, 1: Measure Line, 2: Measure Rect
        self.current_mode = 0 
        self.dialog.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        # --- Button Connections ---
        apply_btn = self.dialog.button_box.button(QDialogButtonBox.Apply)
        apply_btn.clicked.connect(self.apply_scale)
        
        close_btn = self.dialog.button_box.button(QDialogButtonBox.Close)
        close_btn.clicked.connect(self.deactivate)

        # --- Connect accumulation buttons ---
        self.dialog.line_add_button.clicked.connect(self.add_line_to_total)
        self.dialog.line_clear_button.clicked.connect(self.clear_line_total)
        
        self.dialog.rect_add_button.clicked.connect(self.add_rect_to_total)
        self.dialog.rect_clear_button.clicked.connect(self.clear_rect_total)
        
        self.dialog.remove_highlighted_button.clicked.connect(self.remove_scale_highlighted)

        # --- New Button Connection ---
        self.dialog.line_profile_button.clicked.connect(self._show_elevation_profile)

        # --- Drawing State ---
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        self.pixel_length = 0.0
        
        # --- Profile Data Management ---
        self.current_profile_data = None # Stores dict for the line *being drawn*
        self.accumulated_profiles = []   # Stores list of dicts for *saved* lines
        self.profile_plot_dialog = None  # Reference to pop-up
        
        # Color cycle for plots and lines
        # Note: Use width >= 3 for better cross-platform rendering compatibility
        self.color_cycle_pens = [
            pg.mkPen(color='#E63E00', width=4, cosmetic=True),  # Bright Orange (Current)
            pg.mkPen(color='#1f77b4', width=3, cosmetic=True),  # Matplotlib Blue
            pg.mkPen(color='#2ca02c', width=3, cosmetic=True),  # Matplotlib Green
            pg.mkPen(color='#d62728', width=3, cosmetic=True),  # Matplotlib Red
            pg.mkPen(color='#9467bd', width=3, cosmetic=True),  # Matplotlib Purple
            pg.mkPen(color='#8c564b', width=3, cosmetic=True),  # Matplotlib Brown
            pg.mkPen(color='#e377c2', width=3, cosmetic=True),  # Matplotlib Pink
            pg.mkPen(color='#7f7f7f', width=3, cosmetic=True),  # Matplotlib Gray
        ]
        self.current_color_index = 0 # Index for *accumulated* lines
        
        # Base pen for the *current* line (orange)
        self.base_pen = QPen(self.color_cycle_pens[0].color(), 4, Qt.DashLine)
        self.base_pen.setCosmetic(True)

        # --- Graphics Items ---
        # Line (for Set Scale and Measure Line)
        self.preview_line = QGraphicsLineItem()
        self.preview_line.setPen(self.base_pen)
        self.preview_line.setZValue(100)  # Draw on top
        
        # Rectangle
        self.preview_wireframe = QGraphicsItemGroup()
        self.preview_wireframe_base = QGraphicsRectItem(parent=self.preview_wireframe)
        self.preview_wireframe_base.setPen(self.base_pen)
        self.preview_wireframe_base.setZValue(2)  # Base rect on top of grid lines
        
        self.wireframe_grid_lines = []
        grid_size = 5  # Must match the grid_size in _update_wireframe_graphic
        num_lines_needed = (grid_size * (grid_size - 1)) * 2  # (5 * 4) * 2 = 40 lines
        
        for _ in range(num_lines_needed): 
            line = QGraphicsLineItem(parent=self.preview_wireframe)
            pen = QPen(QColor(230, 62, 0, 100), 1, Qt.DotLine)
            pen.setCosmetic(True)
            line.setPen(pen)
            line.setZValue(1)  # Grid lines below base rect
            self.wireframe_grid_lines.append(line)
        self.preview_wireframe.setZValue(100)
        
        self.show_crosshair = True  # Enable crosshair for precise measurements
        
        # --- Accumulation Variables ---
        self.current_line_length = 0.0
        self.total_line_length = 0.0
        
        self.current_rect_area = 0.0
        self.total_rect_area = 0.0
        
        # --- Accumulated Graphics ---
        self.accumulated_lines = []
        self.accumulated_rects = []
        self.accumulated_points = []

        # --- Total Locked Flags ---
        self.line_total_locked = False
        self.rect_total_locked = False

    def get_current_scale(self):
        """Helper to get current raster scale. Returns (scale, units)."""
        current_path = self.annotation_window.current_image_path
        if not current_path:
            return 1.0, "px"
            
        raster = self.main_window.image_window.raster_manager.get_raster(current_path)
        if raster and raster.scale_x is not None:
            # Assuming square pixels from this tool
            units = raster.scale_units
            if units == "metre":
                units = "m"  # Standardize
            return raster.scale_x, units
        else:
            return 1.0, "px"

    def get_current_z_data(self):
        """
        Helper to get z_channel, scale, and units for the current raster.
        Also enables/disables the 3D metric groups in the dialog.
        
        Returns:
            tuple: (raster, z_channel, scale_x, scale_y, z_unit) or (None, None, ...)
        """
        current_path = self.annotation_window.current_image_path
        if not current_path:
            self.dialog.line_3d_group.setEnabled(False)
            self.dialog.rect_3d_group.setEnabled(False)
            return None, None, None, None, None
            
        raster = self.main_window.image_window.raster_manager.get_raster(current_path)
        if not raster:
            self.dialog.line_3d_group.setEnabled(False)
            self.dialog.rect_3d_group.setEnabled(False)
            return None, None, None, None, None

        # Check for scale first
        scale_x = raster.scale_x
        scale_y = raster.scale_y
        
        # Now check for z_channel (lazily)
        z_channel = raster.z_channel_lazy
        z_unit = raster.z_unit
        
        # We need both scale and z-channel for 3D metrics
        if scale_x is not None and z_channel is not None:
            self.dialog.line_3d_group.setEnabled(True)
            self.dialog.rect_3d_group.setEnabled(True)
            return raster, z_channel, scale_x, scale_y, z_unit
        else:
            # If we're missing anything, disable the 3D fields
            self.dialog.line_3d_group.setEnabled(False)
            self.dialog.rect_3d_group.setEnabled(False)
            self.reset_3d_labels() # Explicitly clear them
            return raster, None, scale_x, scale_y, None # Return partial data

    def reset_3d_labels(self):
        """Resets just the 3D metric labels to N/A."""
        # Line Tab
        self.dialog.line_3d_length_label.setText("N/A")
        self.dialog.line_delta_z_label.setText("N/A")
        self.dialog.line_slope_label.setText("N/A")
        self.dialog.line_rugosity_label.setText("N/A")
        self.current_profile_data = None
        # Update profile button state (might still have accumulated profiles)
        self.update_profile_button_state()
        # Rect Tab
        self.dialog.rect_z_stats_label.setText("N/A")
        self.dialog.rect_3d_surface_area_label.setText("N/A")
        self.dialog.rect_volume_label.setText("N/A")
        self.dialog.rect_rugosity_label.setText("N/A")
    
    def update_profile_button_state(self):
        """
        Update the 'Show Elevation Profile' button state based on available profile data.
        The button should be enabled if:
        1. There are accumulated profiles with 3D data, OR
        2. There is a current profile with 3D data, OR
        3. Both
        """
        # Safety check: dialog and button must exist
        if (not self.dialog or
            not hasattr(self.dialog, 'line_profile_button') or
            self.dialog.line_profile_button is None):
            return
        
        has_accumulated = len(self.accumulated_profiles) > 0 if self.accumulated_profiles is not None else False
        has_current = self.current_profile_data and self.current_profile_data.get("has_3d", False)
        
        # Enable if there's any profile data to show
        should_enable = bool(has_accumulated or has_current)
            
        self.dialog.line_profile_button.setEnabled(should_enable)

    def load_existing_scale(self):
        """Loads and displays existing scale data for the current image if available."""
        current_path = self.annotation_window.current_image_path
        if not current_path:
            return
        
        raster = self.main_window.image_window.raster_manager.get_raster(current_path)
        if not raster or raster.scale_x is None:
            # No scale data available
            self.dialog.calculated_scale_label.setText("Scale: N/A")
            return
        
        # Display the existing scale
        scale_value = raster.scale_x  # Assuming square pixels
        units = raster.scale_units if raster.scale_units else "metre"
        
        # Standardize unit display
        if units == "metre":
            units = "m"
        
        # Format the scale text
        scale_text = f"{scale_value:.6f} {units}/pixel"
        self.dialog.calculated_scale_label.setText(f"Scale: {scale_text}")

    def activate(self):
        super().activate()
        # Set initial mode based on the currently selected tab
        self.on_tab_changed(self.dialog.tab_widget.currentIndex())
        
        # Add all preview items to scene
        if not self.preview_line.scene():
            self.annotation_window.scene.addItem(self.preview_line)
        if not self.preview_wireframe.scene():
            self.annotation_window.scene.addItem(self.preview_wireframe)
        
        self.stop_current_drawing()  # Resets all drawing
        self.dialog.reset_fields()
        
        # Connect signal to update highlighted count (only once)
        if not self.dialog._signal_connected:
            self.main_window.image_window.table_model.rowsChanged.connect(self.dialog.update_status_label)
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

        # Check for Z-Data to enable/disable 3D tabs
        self.get_current_z_data()
        
        # Load and display existing scale if present
        self.load_existing_scale()

        self.dialog.show()
        self.dialog.activateWindow()  # Bring it to the front

    def deactivate(self):
        # This function is called when another tool is selected
        # or when the dialog's "Close" button is clicked.
        if not self.active:
            return
            
        super().deactivate()
        self.dialog.hide()
        self.preview_line.hide()
        self.preview_wireframe.hide()
        
        # Clean up accumulated graphics
        for line in self.accumulated_lines:
            self.annotation_window.scene.removeItem(line)
        self.accumulated_lines.clear()
        for rect in self.accumulated_rects:
            self.annotation_window.scene.removeItem(rect)
        self.accumulated_rects.clear()
        for point in self.accumulated_points:
            self.annotation_window.scene.removeItem(point)
        self.accumulated_points.clear()
        
        # Clear profile data
        self.accumulated_profiles.clear()
        self.current_profile_data = None
        self.current_color_index = 0
        
        self.is_drawing = False
        
        # Close profile plot dialog if it's open
        if self.profile_plot_dialog:
            self.profile_plot_dialog.reject()
            self.profile_plot_dialog = None
        
        # Untoggle all tools when closing the scale tool
        self.main_window.untoggle_all_tools()

    def on_tab_changed(self, index):
        """Called when the user clicks a different tab."""
        self.current_mode = index
        self.stop_current_drawing()
        
        # Check for Z-Data to enable/disable 3D groups
        self.get_current_z_data()
        
        # Enable "Set Scale" button ONLY on the first tab
        if index == 0:
            self.dialog.set_scale_button.setEnabled(True)
            self.preview_wireframe.hide() # Hide rect
            self.preview_line.show() # Show line
        elif index == 1:
            self.dialog.set_scale_button.setEnabled(False)
            self.preview_wireframe.hide() # Hide rect
            self.preview_line.show() # Show line
        elif index == 2:
            self.dialog.set_scale_button.setEnabled(False)
            self.preview_line.hide() # Hide line
            self.preview_wireframe.show() # Show rect
            
    def stop_current_drawing(self):
        """Force stop all drawing, hide previews, and reset points."""
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        
        self.preview_line.hide()
        self.preview_line.setLine(QLineF())  # Clear the line geometry
        self.preview_wireframe.hide()
        self.preview_wireframe_base.setRect(QRectF())  # Clear the rect geometry
        
        # Reset labels
        self.dialog.pixel_length_label.setText("Draw a line on the image")
        self.dialog.line_length_label.setText("N/A")
        self.dialog.rect_perimeter_label.setText("N/A")
        self.dialog.rect_area_label.setText("N/A")
        
        # Reset 3D labels
        self.reset_3d_labels()
        
        # Reset current measurements
        self.current_line_length = 0.0
        self.current_rect_area = 0.0
        
        # Disable 'Add' buttons
        self.dialog.line_add_button.setEnabled(False)
        self.dialog.rect_add_button.setEnabled(False)
        
        # Close profile plot - but DO NOT clear accumulated data here
        # self.clear_line_total() handles that
        if self.profile_plot_dialog:
            # Just update it to show only accumulated lines
            self.profile_plot_dialog.update_plot(self.accumulated_profiles)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.LeftButton:
            return
        if not self.annotation_window.cursorInWindow(event.pos()):
            return
            
        scene_pos = self.annotation_window.mapToScene(event.pos())
        
        # --- Mode 0: Set Scale OR Mode 1: Measure Line ---
        if self.current_mode == 0 or self.current_mode == 1:
            if not self.is_drawing:
                # Start drawing
                self.start_point = scene_pos
                self.end_point = self.start_point
                self.is_drawing = True
                self.preview_line.setLine(QLineF(self.start_point, self.end_point))
                self.preview_line.show()
                if self.current_mode == 0:
                    self.dialog.pixel_length_label.setText("Drawing...")
                else:
                    self.dialog.line_length_label.setText("Drawing...")
                    self.reset_3d_labels() # Clear old graphics
            else:
                # Finish drawing
                self.is_drawing = False
                self.end_point = scene_pos
                line = QLineF(self.start_point, self.end_point)
                self.pixel_length = line.length()
                self.preview_line.setLine(line)
                
                if self.pixel_length == 0.0:
                    self.stop_current_drawing()
                    return

                # Update the correct label based on mode
                if self.current_mode == 0:
                    self.dialog.pixel_length_label.setText(f"{self.pixel_length:.2f} px")
                else:
                    self.calculate_line_measurement(final_calc=True) # <-- Final calc

        # --- Mode 2: Measure Rectangle ---
        elif self.current_mode == 2:
            if not self.is_drawing:
                # Start drawing
                self.start_point = scene_pos
                self.end_point = self.start_point
                self.is_drawing = True
                self._update_wireframe_graphic()
                self.preview_wireframe.show()
                self.dialog.rect_area_label.setText("Drawing...")
                self.dialog.rect_perimeter_label.setText("Drawing...")
                self.reset_3d_labels() # Clear old graphics
            else:
                # Finish drawing
                self.is_drawing = False
                self.end_point = scene_pos
                self._update_wireframe_graphic()
                self.calculate_rect_measurement(final_calc=True) # <-- Final calc

    def mouseMoveEvent(self, event: QMouseEvent):
        """
        Override to show crosshair for measurement tool regardless of selected label.
        """
        # Handle crosshair display without requiring selected_label
        scene_pos = self.annotation_window.mapToScene(event.pos())
        cursor_in_window = self.annotation_window.cursorInWindow(event.pos())
        
        if cursor_in_window and self.active and self.show_crosshair:
            self.update_crosshair(scene_pos)
        else:
            self.clear_crosshair()
        
        if not self.is_drawing:
            return
            
        scene_pos = self.annotation_window.mapToScene(event.pos())
        self.end_point = scene_pos

        # --- Mode 0: Set Scale OR Mode 1: Measure Line ---
        if self.current_mode == 0 or self.current_mode == 1:
            line = QLineF(self.start_point, self.end_point)
            self.pixel_length = line.length()
            self.preview_line.setLine(line)
            # Update correct label
            if self.current_mode == 0:
                self.dialog.pixel_length_label.setText(f"{self.pixel_length:.2f} px")
            else:
                # Live update for line length
                self.calculate_line_measurement(final_calc=False) # <-- Live calc

        # --- Mode 2: Measure Rectangle ---
        elif self.current_mode == 2:
            self._update_wireframe_graphic()
            # Live update for rect
            self.calculate_rect_measurement(final_calc=False) # <-- Live calc

    def mouseReleaseEvent(self, event: QMouseEvent):
        pass

    def keyPressEvent(self, event):
        """Handle key press events for canceling drawing."""
        
        # --- Cancel/Undo (Backspace) ---
        if event.key() == Qt.Key_Backspace:
            if self.is_drawing:
                # Cancel line or rect drawing
                self.stop_current_drawing()
        
        # --- Cancel (Escape) ---
        if event.key() == Qt.Key_Escape:
            self.stop_current_drawing()
            
    # --- New Graphic & Plotting Methods ---

    def _update_wireframe_graphic(self):
        """Updates the 3D wireframe grid being drawn."""
        rect = QRectF(self.start_point, self.end_point).normalized()
        self.preview_wireframe_base.setRect(rect)

        # Check for z-data. If none, just draw 2D rect and return
        _, z_channel, _, _, _ = self.get_current_z_data()
        if z_channel is None or not self.is_drawing: # Don't draw grid if not drawing
            # Hide all grid lines
            for line in self.wireframe_grid_lines:
                line.hide()
            return
        
        try:
            h, w = z_channel.shape
            grid_size = 5 # 5x5 grid
            line_idx = 0

            # Generate grid points
            x_points = np.linspace(rect.left(), rect.right(), grid_size)
            y_points = np.linspace(rect.top(), rect.bottom(), grid_size)
            
            # Get Z values for all grid points (vectorized)
            x_coords = np.clip(np.round(x_points).astype(int), 0, w - 1)
            y_coords = np.clip(np.round(y_points).astype(int), 0, h - 1)
            
            # Create a meshgrid for indexing
            xx, yy = np.meshgrid(x_coords, y_coords)
            z_grid = z_channel[yy, xx]
            
            # Simple Z scaling for visualization
            z_min, z_max = np.min(z_grid), np.max(z_grid)
            z_range = (z_max - z_min) if (z_max - z_min) > 0 else 1.0
            z_scale = -rect.height() / (z_range * 2.0) # Visual scale
            
            # Apply visual offset
            z_offsets = (z_grid - z_min) * z_scale
            
            # Create 3D points
            points_3d = []
            for i in range(grid_size):
                row = []
                for j in range(grid_size):
                    offset = z_offsets[i, j]
                    row.append(QPointF(x_points[j] + offset, y_points[i] + offset))
                points_3d.append(row)

            # Draw horizontal grid lines
            for i in range(grid_size):
                for j in range(grid_size - 1):
                    p1 = points_3d[i][j]
                    p2 = points_3d[i][j+1]
                    self.wireframe_grid_lines[line_idx].setLine(QLineF(p1, p2))
                    self.wireframe_grid_lines[line_idx].show()
                    line_idx += 1

            # Draw vertical grid lines
            for j in range(grid_size):
                for i in range(grid_size - 1):
                    p1 = points_3d[i][j]
                    p2 = points_3d[i+1][j]
                    self.wireframe_grid_lines[line_idx].setLine(QLineF(p1, p2))
                    self.wireframe_grid_lines[line_idx].show()
                    line_idx += 1
            
            # Hide unused lines
            for i in range(line_idx, len(self.wireframe_grid_lines)):
                self.wireframe_grid_lines[i].hide()

        except Exception as e:
            print(f"Error updating wireframe: {e}")
            # On error, just draw 2D rect
            for line in self.wireframe_grid_lines:
                line.hide()

    def _show_elevation_profile(self):
        """Shows the pop-up dialog with the elevation profile."""
        
        # Combine accumulated plots with the current (unsaved) plot
        all_plots_to_show = self.accumulated_profiles.copy()
        # Only add current profile if it has 3D data
        if self.current_profile_data and self.current_profile_data.get("has_3d", False):
            all_plots_to_show.append(self.current_profile_data)

        if not all_plots_to_show:
            QMessageBox.warning(self.dialog, "No Data", 
                                "No 3D elevation profile data to display.\n"
                                "Elevation profiles require depth/z-channel data.")
            return
            
        try:
            # Close existing dialog
            if self.profile_plot_dialog:
                self.profile_plot_dialog.reject()
            
            # Create and show new dialog
            self.profile_plot_dialog = ProfilePlotDialog(all_plots_to_show, self.dialog)
            self.profile_plot_dialog.show()
        except Exception as e:
            QMessageBox.critical(self.dialog, "Plot Error", f"Could not display plot: {e}")

    # --- Calculation and Accumulation Methods ---

    def calculate_line_measurement(self, final_calc=False):
        """Calculates and displays the length of the drawn line."""
        
        # --- 1. Get Data ---
        raster, z_channel, scale_x, scale_y, z_unit = self.get_current_z_data()
        
        # Check for invalid scale
        if scale_x is None or scale_y is None:
            self.dialog.line_length_label.setText("Scale Not Set")
            self.reset_3d_labels()
            return

        display_units = self.dialog.line_units_combo.currentText()
        
        # --- 2. 2D Calculations ---
        length_2d_meters = self.pixel_length * scale_x  # Assume square pixels
        
        # Convert to display units
        if display_units != "m":
            length_2d_display = convert_scale_units(length_2d_meters, 'metre', display_units)
        else:
            length_2d_display = length_2d_meters
        
        if final_calc:
            self.current_line_length = length_2d_display
            self.dialog.line_add_button.setEnabled(True)
            
            # Always create basic profile data (even without z-channel)
            # This ensures "Add to Total" works for 2D-only lines
            self.current_profile_data = {
                "name": "Current Line",
                "color": self.color_cycle_pens[0],  # Always orange for current
                "x_data": None,  # Will be populated if z_channel exists
                "y_data": None,
                "x_label": None,
                "y_label": None,
                "stats_str": None,
                "has_3d": False  # Flag to indicate if 3D data is present
            }
        
        self.dialog.line_length_label.setText(f"{length_2d_display:.3f} {display_units}")

        # --- 3. 3D Calculations ---
        # Stop if no z_channel
        if z_channel is None:
            self.reset_3d_labels()
            return
            
        try:
            h, w = z_channel.shape
            z_unit_str = z_unit if z_unit else 'z-units'
            
            # Convert z_unit to meters for 3D calculations
            # This ensures all spatial dimensions are in the same unit (meters)
            z_to_meters_factor = convert_scale_units(1.0, z_unit, 'metre') if z_unit else 1.0

            # Get Z start/end
            p1 = QPoint(int(self.start_point.x()), int(self.start_point.y()))
            p2 = QPoint(int(self.end_point.x()), int(self.end_point.y()))

            # Clamp points to be inside raster bounds
            p1.setX(max(0, min(p1.x(), w - 1)))
            p1.setY(max(0, min(p1.y(), h - 1)))
            p2.setX(max(0, min(p2.x(), w - 1)))
            p2.setY(max(0, min(p2.y(), h - 1)))

            z_start = z_channel[p1.y(), p1.x()]
            z_end = z_channel[p2.y(), p2.x()]
            
            delta_z = z_end - z_start
            self.dialog.line_delta_z_label.setText(f"{delta_z:.3f} {z_unit_str}")
            
            # Calculate Slope (convert delta_z to meters for meaningful percentage)
            if length_2d_meters > 0:
                delta_z_meters = delta_z * z_to_meters_factor
                slope = (delta_z_meters / length_2d_meters) * 100.0
                self.dialog.line_slope_label.setText(f"{slope:.2f} %")
            else:
                self.dialog.line_slope_label.setText("N/A")

            # --- 3D Length & Linear Rugosity ---
            # "Walk" the line
            num_samples = max(2, int(self.pixel_length / 2))  # Sample every 2 pixels
            x_samples = np.linspace(self.start_point.x(), self.end_point.x(), num_samples)
            y_samples = np.linspace(self.start_point.y(), self.end_point.y(), num_samples)
            
            profile_data_x = []  # For plot
            profile_data_y = []  # For plot
            
            total_3d_length = 0.0
            dist_2d_so_far = 0.0
            
            profile_data_x.append(0.0)
            z_a = z_channel[min(max(0, int(y_samples[0])), h - 1), min(max(0, int(x_samples[0])), w - 1)]
            profile_data_y.append(z_a)
            
            for i in range(num_samples - 1):
                # Get segment start/end points (pixel coords)
                x_a, y_a = x_samples[i], y_samples[i]
                x_b, y_b = x_samples[i + 1], y_samples[i + 1]
                
                # Get Z values (clamped)
                z_a = z_channel[min(max(0, int(y_a)), h - 1), min(max(0, int(x_a)), w - 1)]
                z_b = z_channel[min(max(0, int(y_b)), h - 1), min(max(0, int(x_b)), w - 1)]
                
                # Get segment components in real-world units (meters)
                dx_m = (x_b - x_a) * scale_x
                dy_m = (y_b - y_a) * scale_y
                dz_meters = (z_b - z_a) * z_to_meters_factor  # Convert z-units to meters
                
                # Add 3D segment length (all components now in meters)
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
            
            self.dialog.line_3d_length_label.setText(f"{length_3d_display:.3f} {display_units}")

            # Calculate Linear Rugosity
            if length_2d_meters > 0:
                linear_rugosity = total_3d_length / length_2d_meters
                self.dialog.line_rugosity_label.setText(f"{linear_rugosity:.3f}")
            else:
                self.dialog.line_rugosity_label.setText("N/A")
                
            # Store data for plot
            plot_y_label = f"Elevation ({z_unit_str})"
            stats_str = f"3D-Len: {length_3d_display:.2f}{display_units} | "
            stats_str += f"ΔZ: {delta_z:.2f}{z_unit_str} | "
            stats_str += f"Rugosity: {linear_rugosity:.3f}"
            
            # Update the current_profile_data with 3D information
            if self.current_profile_data:
                self.current_profile_data["x_data"] = plot_x_data
                self.current_profile_data["y_data"] = profile_data_y
                self.current_profile_data["x_label"] = plot_x_label
                self.current_profile_data["y_label"] = plot_y_label
                self.current_profile_data["stats_str"] = stats_str
                self.current_profile_data["has_3d"] = True

            if final_calc:
                # Update profile button state (we now have a current profile)
                self.update_profile_button_state()
                
                # --- Update plot ONLY on final click ---
                # Check if the plot dialog is already open and visible
                if self.profile_plot_dialog and self.profile_plot_dialog.isVisible():
                    # If it is, update it directly
                    all_plots = self.accumulated_profiles.copy()
                    if self.current_profile_data and self.current_profile_data.get("has_3d", False):
                        all_plots.append(self.current_profile_data)
                    self.profile_plot_dialog.update_plot(all_plots)
                    
        except Exception as e:
            print(f"Error in 3D line calculation: {e}")
            self.reset_3d_labels()

    def add_line_to_total(self):
        """Adds the current 2D line length and profile to the total."""
        
        # 1. Check if there is a line to add
        # We need either current_profile_data OR a valid current_line_length
        if self.current_line_length <= 0:
            QMessageBox.warning(self.dialog, "No Line", 
                                "Please draw and complete a line measurement first.")
            return
            
        # 2. Add 2D length to total
        self.total_line_length += self.current_line_length
        display_units = self.dialog.line_units_combo.currentText()
        self.dialog.line_total_length_label.setText(f"{self.total_line_length:.3f} {display_units}")
        
        # Lock the units combo if not already locked
        if not self.line_total_locked:
            self.line_total_locked = True
            self.dialog.line_units_combo.setEnabled(False)
        
        # 3. Get next color and name for the saved profile
        # We skip index 0 (orange) for saved lines
        color_index = (self.current_color_index % (len(self.color_cycle_pens) - 1)) + 1
        pen_to_use = self.color_cycle_pens[color_index]
        qcolor_to_use = pen_to_use.color()
        name_to_use = f"Line {self.current_color_index + 1}"
        
        # 4. Create the permanent line graphic on the map
        perm_pen = QPen(qcolor_to_use, 4, Qt.SolidLine)
        perm_pen.setCosmetic(True)
        perm_line = QGraphicsLineItem(self.preview_line.line())
        perm_line.setPen(perm_pen)
        perm_line.setZValue(99)  # Slightly below preview
        self.annotation_window.scene.addItem(perm_line)
        self.accumulated_lines.append(perm_line)
        
        # Add start and end points
        start_point = perm_line.line().p1()
        end_point = perm_line.line().p2()
        radius = 4  # 8 pixel diameter
        
        # White start point
        start_ellipse = QGraphicsEllipseItem(start_point.x() - radius, 
                                             start_point.y() - radius, 
                                             2 * radius, 2 * radius)
        
        start_ellipse.setBrush(QBrush(Qt.white))
        start_ellipse.setPen(QPen(Qt.black, 1))
        start_ellipse.setZValue(99)
        self.annotation_window.scene.addItem(start_ellipse)
        self.accumulated_points.append(start_ellipse)
        
        # Black end point
        end_ellipse = QGraphicsEllipseItem(end_point.x() - radius,
                                           end_point.y() - radius, 
                                           2 * radius, 2 * radius)
        
        end_ellipse.setBrush(QBrush(Qt.black))
        end_ellipse.setPen(QPen(Qt.black, 1))
        end_ellipse.setZValue(99)
        self.annotation_window.scene.addItem(end_ellipse)
        self.accumulated_points.append(end_ellipse)
        
        # 5. Promote the "current" profile to "accumulated" (only if it has 3D data)
        if self.current_profile_data and self.current_profile_data.get("has_3d", False):
            self.current_profile_data["name"] = name_to_use
            self.current_profile_data["color"] = pen_to_use
            self.accumulated_profiles.append(self.current_profile_data)
        
        # 6. Increment color index
        self.current_color_index += 1
        
        # 7. Stop the current drawing (this hides preview_line,
        #    resets UI, and updates the plot)
        self.stop_current_drawing()
        
        # 8. Update profile button state (we may have added a profile)
        self.update_profile_button_state()
        
        # 9. Update plot dialog if it's open (only if we have profiles to show)
        if self.profile_plot_dialog and self.profile_plot_dialog.isVisible():
            if self.accumulated_profiles:
                self.profile_plot_dialog.update_plot(self.accumulated_profiles)
            else:
                # Close the plot dialog if no 3D profiles exist
                self.profile_plot_dialog.reject()
                self.profile_plot_dialog = None

    def clear_line_total(self):
        # 1. Reset 2D total
        self.total_line_length = 0.0
        display_units = self.dialog.line_units_combo.currentText()
        self.dialog.line_total_length_label.setText(f"0.0 {display_units}")
        
        # Unlock the units combo
        self.line_total_locked = False
        self.dialog.line_units_combo.setEnabled(True)
        
        # 2. Remove accumulated graphics from map
        for line in self.accumulated_lines:
            self.annotation_window.scene.removeItem(line)
        self.accumulated_lines.clear()
        for point in self.accumulated_points:
            self.annotation_window.scene.removeItem(point)
        self.accumulated_points.clear()
        
        # 3. Clear profile data
        self.accumulated_profiles.clear()
        
        # Reset color index only if both lines and rects are cleared
        if not self.accumulated_rects:
            self.current_color_index = 0
        
        # 4. Stop any current drawing
        self.stop_current_drawing()
        
        # 5. Update profile button state (all profiles cleared)
        self.update_profile_button_state()
        
        # 6. Update plot if open
        if self.profile_plot_dialog and self.profile_plot_dialog.isVisible():
            self.profile_plot_dialog.update_plot([])

    def calculate_rect_measurement(self, final_calc=False):
        """Calculates and displays rect perimeter and area."""
        
        # --- 1. Get Data ---
        raster, z_channel, scale_x, scale_y, z_unit = self.get_current_z_data()

        # Check for invalid scale
        if scale_x is None or scale_y is None:
            self.dialog.rect_perimeter_label.setText("Scale Not Set")
            self.dialog.rect_area_label.setText("Scale Not Set")
            self.reset_3d_labels()
            return
            
        display_units = self.dialog.rect_units_combo.currentText()
        area_units = f"{display_units}²" if display_units != "px" else "px²"

        rect = QRectF(self.start_point, self.end_point).normalized()
        pixel_width = rect.width()
        pixel_height = rect.height()
        
        # Only reject tiny rectangles on final calculation
        if final_calc and (pixel_width < 1 or pixel_height < 1):
            # Don't stop_current_drawing here - just show a warning
            self.dialog.rect_perimeter_label.setText("Too Small")
            self.dialog.rect_area_label.setText("Too Small")
            self.reset_3d_labels()
            return
        elif not final_calc and (pixel_width < 1 or pixel_height < 1):
            # During live drawing, just don't calculate
            return

        # --- 2. 2D Calculations ---
        # Calculate dimensions in meters first
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
        
        if final_calc:
            # Store for accumulation
            self.current_rect_area = area_2d_display
            self.dialog.rect_add_button.setEnabled(True)

        self.dialog.rect_perimeter_label.setText(f"{perimeter_display:.3f} {display_units}")
        self.dialog.rect_area_label.setText(f"{area_2d_display:.3f} {area_units}")

        # --- 3. 3D Calculations ---
        # Stop if no z_channel
        if z_channel is None:
            self.reset_3d_labels()
            return

        try:
            h, w = z_channel.shape
            z_unit_str = z_unit if z_unit else 'z-units'
            
            # Convert z_unit to meters for 3D calculations
            z_to_meters_factor = convert_scale_units(1.0, z_unit, 'metre') if z_unit else 1.0
            
            # Get integer bounds for slicing, clamped to raster dims
            x1 = max(0, int(math.floor(rect.left())))
            y1 = max(0, int(math.floor(rect.top())))
            x2 = min(w, int(math.ceil(rect.right())))
            y2 = min(h, int(math.ceil(rect.bottom())))
            
            if x1 >= x2 or y1 >= y2:  # Check for zero-area slice
                self.reset_3d_labels()
                return

            # Get Z-Slice
            z_slice = z_channel[y1:y2, x1:x2]
            if z_slice.size == 0:
                self.reset_3d_labels()
                return

            # Calculate Z-Stats (in original z-units)
            z_min = np.min(z_slice)
            z_max = np.max(z_slice)
            z_mean = np.mean(z_slice)
            self.dialog.rect_z_stats_label.setText(
                f"Min: {z_min:.2f} | Max: {z_max:.2f} | Mean: {z_mean:.2f} ({z_unit_str})"
            )

            # Calculate Prismatic Volume
            pixel_area_2d = scale_x * scale_y
            volume = np.sum(z_slice) * pixel_area_2d
            vol_units = f"{area_units.replace('²', '')}² · {z_unit_str}"
            self.dialog.rect_volume_label.setText(f"{volume:.3f} {vol_units}")

            # Calculate 3D Surface Area
            # Convert z_slice to meters to maintain dimensional consistency
            z_slice_meters = z_slice * z_to_meters_factor
            
            # Calculate gradients with proper spacing (all in meters now)
            dz_dy, dz_dx = np.gradient(z_slice_meters, scale_y, scale_x)
            multiplier = np.sqrt(1.0 + dz_dx**2 + dz_dy**2)
            pixel_areas_3d = pixel_area_2d * multiplier
            surface_area_3d_meters = np.sum(pixel_areas_3d)

            # Convert to display units (area)
            if display_units != "m":
                conv_factor = convert_scale_units(1.0, 'metre', display_units)
                area_conv_factor = conv_factor * conv_factor
                surface_area_3d_display = surface_area_3d_meters * area_conv_factor
            else:
                surface_area_3d_display = surface_area_3d_meters
                
            self.dialog.rect_3d_surface_area_label.setText(f"{surface_area_3d_display:.3f} {area_units}")

            # Calculate Areal Rugosity
            if area_2d_meters > 0:
                areal_rugosity = surface_area_3d_meters / area_2d_meters
                self.dialog.rect_rugosity_label.setText(f"{areal_rugosity:.3f}")
            else:
                self.dialog.rect_rugosity_label.setText("N/A")
                
        except Exception as e:
            print(f"Error in 3D rect calculation: {e}")
            self.reset_3d_labels()

    def add_rect_to_total(self):
        """Adds the current 2D rect area to the total."""
        
        # 1. Check if there is a rect to add
        if self.current_rect_area <= 0:
            QMessageBox.warning(self.dialog, "No Rectangle", 
                                "Please draw and complete a rectangle measurement first.")
            return
            
        # 2. Add area to total
        self.total_rect_area += self.current_rect_area
        display_units = self.dialog.rect_units_combo.currentText()
        area_units = f"{display_units}²" if display_units != "px" else "px²"
        
        self.dialog.rect_total_area_label.setText(f"{self.total_rect_area:.3f} {area_units}")
        
        # Lock the units combo if not already locked
        if not self.rect_total_locked:
            self.rect_total_locked = True
            self.dialog.rect_units_combo.setEnabled(False)
        
        # 3. Get next color for the saved rectangle
        # We skip index 0 (orange) for saved shapes
        color_index = (self.current_color_index % (len(self.color_cycle_pens) - 1)) + 1
        pen_to_use = self.color_cycle_pens[color_index]
        qcolor_to_use = pen_to_use.color()
        
        # 4. Create a permanent rect item with solid colored line
        perm_rect = QGraphicsRectItem(self.preview_wireframe_base.rect())
        perm_pen = QPen(qcolor_to_use, 4, Qt.SolidLine)
        perm_pen.setCosmetic(True)
        perm_rect.setPen(perm_pen)
        perm_rect.setZValue(99)  # Slightly below preview
        self.annotation_window.scene.addItem(perm_rect)
        self.accumulated_rects.append(perm_rect)
        
        # 5. Increment color index (shared with lines)
        self.current_color_index += 1
        
        # 6. Reset current measurement and UI
        self.current_rect_area = 0.0
        self.dialog.rect_add_button.setEnabled(False)
        
        # Hide the preview but don't clear start/end points yet
        self.preview_wireframe.hide()
        
        # Reset labels
        self.dialog.rect_perimeter_label.setText("N/A")
        self.dialog.rect_area_label.setText("N/A")
        self.reset_3d_labels()  # Reset 3D fields too
        
        # Reset the drawing state to allow a new rectangle
        self.is_drawing = False
        self.start_point = None
        self.end_point = None

    def clear_rect_total(self):
        # 1. Reset total
        self.total_rect_area = 0.0
        display_units = self.dialog.rect_units_combo.currentText()
        area_units = f"{display_units}²" if display_units != "px" else "px²"
        self.dialog.rect_total_area_label.setText(f"0.0 {area_units}")
        
        # Unlock the units combo
        self.rect_total_locked = False
        self.dialog.rect_units_combo.setEnabled(True)
        
        # 2. Remove accumulated rects from map
        for rect in self.accumulated_rects:
            self.annotation_window.scene.removeItem(rect)
        self.accumulated_rects.clear()
        
        # Reset color index only if both lines and rects are cleared
        if not self.accumulated_lines and not self.accumulated_points:
            self.current_color_index = 0
        
        # 3. Stop any current drawing
        self.stop_current_drawing()

    def apply_scale(self):
        """
        Calculates and applies the new scale to the selected raster(s).
        """        
        # --- 1. Get User Input ---
        known_length = self.dialog.known_length_input.value()
        units = self.dialog.units_combo.currentText()
        
        # --- 2. Validate Inputs ---
        if self.pixel_length == 0.0 or not self.start_point:
            QMessageBox.warning(self.dialog, 
                                "No Line Drawn",
                                "Please draw a line on the image to set the pixel length.")
            return

        if known_length == 0.0:
            QMessageBox.warning(self.dialog, 
                                "Invalid Length",
                                "Known length cannot be zero.")
            return

        # --- 3. Get Current Raster and Warn on Overwrite ---
        current_path = self.annotation_window.current_image_path
        if not current_path:
            QMessageBox.warning(self.dialog, "No Image", "No image is currently loaded.")
            return
            
        current_raster = self.main_window.image_window.raster_manager.get_raster(current_path)
        
        if current_raster.scale_x is not None:
            reply = QMessageBox.question(self.dialog, 
                                         "Overwrite Scale?",
                                         "The current image already has a scale defined.\n\n"
                                         "Applying this new scale will overwrite the existing data "
                                         "for all selected images.\n\n"
                                         "Do you want to continue?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        # --- 4. Calculate New Scale ---
        try:
            known_length_meters = convert_scale_units(known_length, units, 'metre')
        except Exception as e:
            QMessageBox.critical(self.dialog, 
                                 "Unit Error", 
                                 f"Could not convert units: {e}")
            return
            
        new_scale = known_length_meters / self.pixel_length  # meters/pixel
        
        # --- 5. Get Target Images ---
        target_image_paths = self.dialog.get_selected_image_paths()
        if not target_image_paths:
            QMessageBox.warning(self.dialog, 
                                "No Images Selected",
                                "No images were found for the selected option.")
            return

        # --- 6. Loop and Apply ---
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Applying Scale")
        progress_bar.show()
        progress_bar.start_progress(len(target_image_paths))
        
        success_count = 0
        try:
            for path in target_image_paths:
                if progress_bar.wasCanceled():
                    break
                    
                raster = self.main_window.image_window.raster_manager.get_raster(path)
                if raster:
                    # We assume square pixels from this tool
                    raster.update_scale(new_scale, new_scale, 'm')
                    
                    # Update all annotations for this image with the new scale
                    self.annotation_window.set_annotations_scale(path)
                
                    success_count += 1
                
                progress_bar.update_progress()
                
        except Exception as e:
            QMessageBox.critical(self.dialog, "Error Applying Scale", f"An error occurred: {e}")
        finally:
            progress_bar.stop_progress()
            progress_bar.close()
            QApplication.restoreOverrideCursor()

        # --- 7. Finalize ---
        scale_text = f"{new_scale:.6f} m/pixel"
        self.dialog.calculated_scale_label.setText(f"Scale: {scale_text}")

        # Refresh the main window status bar if the current image was updated
        if current_path in target_image_paths:
            # Re-emit the signal to update scaled dimensions
            self.main_window.update_view_dimensions(current_raster.width, current_raster.height)

        QMessageBox.information(self.dialog, 
                                "Success",
                                f"Successfully applied new scale ({scale_text}) "
                                f"to {success_count} image(s).")
        
        # Clear accumulated measurements since scale changed
        self.clear_line_total()
        self.clear_rect_total()
        
        # Refresh the confidence window to update the tooltip, just in case
        if self.main_window.confidence_window.annotation:
            self.main_window.confidence_window.refresh_display()
        
    def remove_scale_highlighted(self):
        """Removes scale from all highlighted images."""
        highlighted_paths = self.dialog.get_selected_image_paths()
        
        if not highlighted_paths:
            QMessageBox.warning(self.dialog, "No Images", "No images are highlighted.")
            return
        
        # Count how many highlighted images actually have scale
        images_with_scale = []
        for path in highlighted_paths:
            raster = self.main_window.image_window.raster_manager.get_raster(path)
            if raster and raster.scale_x is not None:
                images_with_scale.append(path)
        
        if not images_with_scale:
            QMessageBox.information(self.dialog, "No Scale",
                                    "None of the highlighted images have scale data to remove.")
            return
        
        # Warn the user
        count = len(images_with_scale)
        if count == 1:
            message = "Are you sure you want to remove the scale from 1 highlighted image?\n"
        else:
            message = f"Are you sure you want to remove the scale from {count} highlighted images?\n"
        message += "This cannot be undone."
        
        reply = QMessageBox.question(self.dialog,
                                     "Confirm Removal",
                                     message,
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            progress_bar = ProgressBar(self.annotation_window, title="Removing Scale Data")
            progress_bar.show()
            progress_bar.start_progress(len(images_with_scale))
            
            current_image_was_updated = False
            
            try:
                for path in images_with_scale:
                    if progress_bar.wasCanceled():
                        break
                    
                    raster = self.main_window.image_window.raster_manager.get_raster(path)
                    if raster and raster.scale_x is not None:
                        # Remove scale from Raster
                        raster.remove_scale()
                        
                        # Update all associated annotations
                        self.annotation_window.set_annotations_scale(path)
                        
                        if path == self.annotation_window.current_image_path:
                            current_image_was_updated = True
                    
                    progress_bar.update_progress()
            finally:
                progress_bar.stop_progress()
                progress_bar.close()
                QApplication.restoreOverrideCursor()
            
            # Update UI if the current image was affected
            if current_image_was_updated:
                raster = self.main_window.image_window.raster_manager.get_raster(
                    self.annotation_window.current_image_path
                )
                self.main_window.update_view_dimensions(raster.width, raster.height)
                
                # Refresh the confidence window to update the tooltip
                if self.main_window.confidence_window.annotation:
                    self.main_window.confidence_window.refresh_display()
                
                # Update the scale display in the dialog
                self.load_existing_scale()
            
            # Clear accumulated measurements since scale changed
            self.clear_line_total()
            self.clear_rect_total()
            
            if count == 1:
                message = "Scale removed from 1 image."
            else:
                message = f"Scale removed from {count} images."
            QMessageBox.information(self.dialog, "Success", message)
