import warnings


import math
import numpy as np

from PyQt5.QtCore import Qt, QLineF, QPointF
from PyQt5.QtGui import QMouseEvent, QPen, QColor, QPainterPath, QPen, QColor, QBrush, QLinearGradient
from PyQt5.QtWidgets import (QDialog, QWidget, QVBoxLayout, QFormLayout, 
                             QDoubleSpinBox, QComboBox, QLabel, QHBoxLayout,
                             QDialogButtonBox, QMessageBox, QGraphicsLineItem,
                             QGroupBox, QPushButton, QTabWidget, QGraphicsPathItem)

from coralnet_toolbox.Tools.QtTool import Tool
from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ZProfilePathItem(QGraphicsPathItem):
    """
    Renders a 'Z-Fence' overlay along a drawn line.
    Visualizes depth/elevation profile by projecting it perpendicular to the drawing path.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setZValue(150)  # Below the main line (100) but above image
        
        # Visual Styling
        self.max_height_px = 80  # Max visual height of the profile in pixels
        
        # Gradient Brush (Cyan to Transparent Blue)
        self.start_color = QColor(0, 255, 255, 120)  # Bright Cyan
        self.end_color = QColor(0, 100, 255, 40)     # Faded Blue
        self.border_color = QColor(0, 255, 255, 200)

    def update_profile(self, p1, p2, raster):
        """
        Calculate and draw the profile polygon.
        
        Args:
            p1 (QPointF): Start point of the line
            p2 (QPointF): End point of the line
            raster (Raster): The raster object containing Z-data
        """
        if not raster or raster.z_channel_lazy is None:
            self.setPath(QPainterPath())
            return

        # 1. Geometry Setup
        line_vec = QPointF(p2.x() - p1.x(), p2.y() - p1.y())
        length = math.hypot(line_vec.x(), line_vec.y())
        if length < 5:  # Don't draw for tiny lines
            self.setPath(QPainterPath())
            return

        # Calculate Normal Vector (Perpendicular to line)
        # We normalize it to length 1
        dx, dy = line_vec.x() / length, line_vec.y() / length
        normal_vec = QPointF(-dy, dx) # Perpendicular rotation (-y, x)

        # 2. Sampling
        num_samples = min(int(length), 100) # Sample every pixel or capped at 100
        if num_samples < 2: 
            return
            
        x_steps = np.linspace(p1.x(), p2.x(), num_samples)
        y_steps = np.linspace(p1.y(), p2.y(), num_samples)
        
        z_values = []
        path_points = []
        
        # Collect Z data
        for x, y in zip(x_steps, y_steps):
            # Using get_z_value ensures we are visualizing the transformed (semantic) data
            z = raster.get_z_value(int(x), int(y))
            if z is not None:
                z_values.append(z)
                path_points.append(QPointF(x, y))
            else:
                # Handle gaps by replicating last known or 0
                val = z_values[-1] if z_values else 0
                z_values.append(val)
                path_points.append(QPointF(x, y))

        if not z_values:
            self.setPath(QPainterPath())
            return

        # 3. Normalization (Map Z range to pixel height)
        z_min = min(z_values)
        z_max = max(z_values)
        z_range = z_max - z_min
        
        if z_range == 0:
            z_range = 1.0

        # 4. Construct the "Fence" Polygon
        # The path starts at P1, goes along the line to P2,
        # then loops back "above" the line using the Z-offsets.
        
        path = QPainterPath()
        if not path_points:
            self.setPath(QPainterPath())
            return
            
        path.moveTo(path_points[0]) # Start at P1 (base)
        
        # Draw the base line
        for pt in path_points[1:]:
            path.lineTo(pt)
            
        # Draw the profile "top" (in reverse order to close loop)
        # We project OUTWARD using the normal vector
        for i in range(len(path_points) - 1, -1, -1):
            pt = path_points[i]
            z = z_values[i]
            
            # Normalize Z to 0.0 - 1.0 relative to min value in selection
            rel_z = (z - z_min) / z_range
            
            # Calculate pixel offset scaling
            pixel_offset = rel_z * self.max_height_px
            
            # Project point: Original + (Normal * Offset)
            proj_x = pt.x() + (normal_vec.x() * pixel_offset)
            proj_y = pt.y() + (normal_vec.y() * pixel_offset)
            
            path.lineTo(proj_x, proj_y)

        path.closeSubpath()
        self.setPath(path)
        
        # 5. Dynamic Gradient coloring
        # We set the gradient to align with the normal vector (base to top)
        grad_start = p1
        grad_end = QPointF(p1.x() + normal_vec.x() * self.max_height_px, 
                           p1.y() + normal_vec.y() * self.max_height_px)
                           
        grad = QLinearGradient(grad_start, grad_end)
        grad.setColorAt(0, self.end_color)     # Base of wall (low Z / base line)
        grad.setColorAt(1, self.start_color)   # Top of wall (high Z)
        
        self.setBrush(QBrush(grad))
        self.setPen(QPen(self.border_color, 1))
        
        
# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class ScaleToolDialog(QDialog):
    """
    A modeless dialog for the ScaleTool, allowing user input for scale calculation
    and propagation, including unified Z-channel calibration.
    """
    def __init__(self, tool, parent=None):
        super().__init__(parent)
        # Get references from the tool
        self.tool = tool
        self.annotation_window = self.tool.annotation_window
        self.main_window = self.annotation_window.main_window

        self.setWindowTitle("Scale Tool")
        self.setWindowIcon(get_icon("scale.png"))
        self.resize(450, 600)

        # This dialog is modeless
        self.setModal(False) 

        self.main_layout = QVBoxLayout(self)

        # --- Create Tab Widget ---
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.scale_tab = self.create_scale_tab()
        self.z_cal_tab = self.create_z_calibration_tab()
        
        # Add tabs to tab widget
        self.tab_widget.addTab(self.scale_tab, "XY Scale (Pixel Size)")
        self.tab_widget.addTab(self.z_cal_tab, "Z-Calibration (Depth/Elevation)")
        
        # Connect tab change signal
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        self.main_layout.addWidget(self.tab_widget)

        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Close)
        
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
        # Modes: 'xy_scale', 'z_scale', 'z_anchor'
        self.current_mode = 'xy_scale'

    def create_scale_tab(self):
        """Create the 'Set Scale' tab for XY calibration."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add Information groupbox
        info_groupbox = QGroupBox("Information")
        info_layout = QVBoxLayout()
        instruction_label = QLabel(
            "Draw a line across a known distance to calibrate image pixel size.\n\n"
            "1. Enter the known real-world length.\n"
            "2. Draw a line across that object in the image.\n"
            "3. Click 'Apply' to calibrate highlighted images."
        )
        instruction_label.setWordWrap(True)
        info_layout.addWidget(instruction_label)
        info_groupbox.setLayout(info_layout)
        layout.addWidget(info_groupbox)
        
        scale_layout = QFormLayout()
        
        # Units
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

        # Pixel length
        self.pixel_length_label = QLabel("Draw a line on the image")
        scale_layout.addRow("Pixel Length:", self.pixel_length_label)
        
        # Current/New scale
        self.calculated_scale_label = QLabel("N/A")
        scale_layout.addRow("Scale:", self.calculated_scale_label)
        
        layout.addLayout(scale_layout)
        
        # Clear Line button
        self.clear_line_button = QPushButton("Clear Line")
        self.clear_line_button.clicked.connect(self.clear_scale_line)
        layout.addWidget(self.clear_line_button)
        
        # Danger Zone
        self.danger_zone_group_box = QGroupBox("Danger Zone")
        self.danger_zone_group_box.setCheckable(True)
        self.danger_zone_group_box.setChecked(False)

        danger_zone_container = QWidget()
        danger_zone_layout = QVBoxLayout(danger_zone_container)
        danger_zone_layout.setContentsMargins(0, 0, 0, 0)

        self.remove_highlighted_button = QPushButton("Remove Scale from Highlighted Images")
        self.remove_highlighted_button.setStyleSheet("background-color: #D9534F; color: white; font-weight: bold;")
        danger_zone_layout.addWidget(self.remove_highlighted_button)

        group_layout = QVBoxLayout()
        group_layout.addWidget(danger_zone_container)
        self.danger_zone_group_box.setLayout(group_layout)
        self.danger_zone_group_box.toggled.connect(danger_zone_container.setVisible)
        danger_zone_container.setVisible(False)

        layout.addWidget(self.danger_zone_group_box)
        layout.addStretch()
        
        return tab

    def create_z_calibration_tab(self):
        """
        Create the unified 'Z-Calibration' tab.
        Handles NaN Setting, Vertical Scaling, View Mode (Depth/Elevation), and Anchoring.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # --- Information Groupbox (at top) ---
        info_groupbox = QGroupBox("Information")
        info_layout = QVBoxLayout()
        self.z_info_label = QLabel("Select a calibration step below.")
        self.z_info_label.setWordWrap(True)
        info_layout.addWidget(self.z_info_label)
        info_groupbox.setLayout(info_layout)
        layout.addWidget(info_groupbox)
        
        # --- Calibration Tool (Radio Buttons) ---
        interaction_group = QGroupBox("Calibration Tool")
        interaction_layout = QVBoxLayout()
        
        # Radio buttons to switch between View Mode, NaN, Scaling and Anchoring
        from PyQt5.QtWidgets import QRadioButton, QButtonGroup
        self.interaction_bg = QButtonGroup(self)
        
        self.radio_view = QRadioButton("Step A: View Mode")
        self.radio_view.setChecked(True)  # Default
        self.radio_view.setToolTip("Configure how Z-channel data is displayed (Depth vs Elevation).")
        self.interaction_bg.addButton(self.radio_view)
        
        self.radio_nan = QRadioButton("Step B: Set NaN Value (Click Point)")
        self.radio_nan.setToolTip("Click a pixel to set the NaN/NoData value for the Z-channel.")
        self.interaction_bg.addButton(self.radio_nan)
        
        self.radio_scale = QRadioButton("Step C: Vertical Scale (Draw Line)")
        self.radio_scale.setToolTip("Draw a line to define the vertical scale (magnitude).")
        self.interaction_bg.addButton(self.radio_scale)
        
        self.radio_anchor = QRadioButton("Step D: Reference Anchor (Click Point)")
        self.radio_anchor.setToolTip("Click a point to set the absolute reference value (offset).")
        self.interaction_bg.addButton(self.radio_anchor)
        
        self.interaction_bg.buttonClicked.connect(self.on_interaction_mode_changed)
        
        interaction_layout.addWidget(self.radio_view)
        interaction_layout.addWidget(self.radio_nan)
        interaction_layout.addWidget(self.radio_scale)
        interaction_layout.addWidget(self.radio_anchor)
        interaction_group.setLayout(interaction_layout)
        layout.addWidget(interaction_group)

        # --- Dynamic Controls (Stack) ---
        # We stack the View Mode, NaN, Scale, and Anchor controls and show only one set
        from PyQt5.QtWidgets import QStackedWidget
        self.controls_stack = QStackedWidget()
        
        # [Page 0] View Mode Controls
        view_widget = QWidget()
        view_form = QFormLayout(view_widget)
        view_form.setContentsMargins(0, 5, 0, 5)
        
        # Define view_mode_combo
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItem("Depth (from Camera)", "depth")
        self.view_mode_combo.addItem("Relative Elevation (from Bottom)", "elevation")
        self.view_mode_combo.setToolTip(
            "Depth: Standard depth map. Positive values = farther away.\n"
            "Elevation: Elevation map. Positive values = elevation above lowest point."
        )
        self.view_mode_combo.currentIndexChanged.connect(self.on_view_mode_changed)
        view_form.addRow("Display As:", self.view_mode_combo)
        
        # Elevation Reference
        self.z_inversion_ref_input = QDoubleSpinBox()
        self.z_inversion_ref_input.setRange(-10000.0, 10000.0)
        self.z_inversion_ref_input.setValue(0.0)
        self.z_inversion_ref_input.setDecimals(2)
        self.z_inversion_ref_input.setSuffix(" m")
        self.z_inversion_ref_input.setToolTip(
            "Reference elevation for converting depth to elevation (e.g., 0 for sea level)"
        )
        view_form.addRow("Elevation Reference:", self.z_inversion_ref_input)
        
        self.controls_stack.addWidget(view_widget)
        
        # [Page 1] NaN Setting Controls
        nan_widget = QWidget()
        nan_form = QFormLayout(nan_widget)
        nan_form.setContentsMargins(0, 5, 0, 5)
        
        self.z_nan_current_label = QLabel("Not Set")
        nan_form.addRow("Current NaN Value:", self.z_nan_current_label)
        
        self.z_nan_hover_label = QLabel("Hover over image...")
        nan_form.addRow("Hover Value:", self.z_nan_hover_label)
        
        self.z_nan_clicked_label = QLabel("Click to select...")
        nan_form.addRow("Clicked Value:", self.z_nan_clicked_label)
        
        self.controls_stack.addWidget(nan_widget)
        
        # [Page 2] Vertical Scaling Controls
        scale_widget = QWidget()
        scale_form = QFormLayout(scale_widget)
        scale_form.setContentsMargins(0, 5, 0, 5)
        
        self.z_units_combo = QComboBox()
        self.z_units_combo.addItems(["mm", "cm", "m", "in", "ft"])
        self.z_units_combo.setCurrentText("m")
        scale_form.addRow("Vertical Units:", self.z_units_combo)
        
        self.z_known_diff_input = QDoubleSpinBox()
        self.z_known_diff_input.setRange(0.001, 1000000.0)
        self.z_known_diff_input.setValue(1.0)
        self.z_known_diff_input.setDecimals(3)
        scale_form.addRow("Known Diff:", self.z_known_diff_input)
        
        self.z_raw_diff_label = QLabel("Draw a line...")
        scale_form.addRow("Measured Diff:", self.z_raw_diff_label)
        
        self.z_scalar_label = QLabel("N/A")
        scale_form.addRow("Calculated Scalar:", self.z_scalar_label)
        
        self.controls_stack.addWidget(scale_widget)
        
        # [Page 3] Anchor Controls
        anchor_widget = QWidget()
        anchor_form = QFormLayout(anchor_widget)
        anchor_form.setContentsMargins(0, 5, 0, 5)
        
        self.z_target_val_input = QDoubleSpinBox()
        self.z_target_val_input.setRange(-10000.0, 10000.0)
        self.z_target_val_input.setValue(0.0)
        self.z_target_val_input.setDecimals(3)
        anchor_form.addRow("Target Value:", self.z_target_val_input)
        
        # Buttons
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(0, 0, 0, 0)
        
        self.z_set_min_btn = QPushButton("Set to Min")
        self.z_set_min_btn.clicked.connect(self.set_target_to_min)
        button_layout.addWidget(self.z_set_min_btn)
        
        self.z_set_max_btn = QPushButton("Set to Max")
        self.z_set_max_btn.clicked.connect(self.set_target_to_max)
        button_layout.addWidget(self.z_set_max_btn)
        
        self.z_tare_btn = QPushButton("Tare to 0.0")
        self.z_tare_btn.clicked.connect(lambda: self.z_target_val_input.setValue(0.0))
        button_layout.addWidget(self.z_tare_btn)
        
        anchor_form.addRow("", button_widget)
        
        self.z_current_val_label = QLabel("Click a point...")
        anchor_form.addRow("Current Value:", self.z_current_val_label)
        
        self.z_offset_label = QLabel("N/A")
        anchor_form.addRow("Calculated Offset:", self.z_offset_label)
        
        self.controls_stack.addWidget(anchor_widget)
        
        layout.addWidget(self.controls_stack)
        
        # --- 4. Danger Zone (Reset) ---
        self.z_danger_zone = QGroupBox("Danger Zone")
        self.z_danger_zone.setCheckable(True)
        self.z_danger_zone.setChecked(False)

        danger_zone_container = QWidget()
        danger_zone_layout = QVBoxLayout(danger_zone_container)
        danger_zone_layout.setContentsMargins(0, 0, 0, 0)

        self.reset_z_btn = QPushButton("Reset All Z-Settings (Scalar=1, Offset=0)")
        self.reset_z_btn.setStyleSheet("background-color: #D9534F; color: white; font-weight: bold;")
        danger_zone_layout.addWidget(self.reset_z_btn)

        group_layout = QVBoxLayout()
        group_layout.addWidget(danger_zone_container)
        self.z_danger_zone.setLayout(group_layout)
        self.z_danger_zone.toggled.connect(danger_zone_container.setVisible)
        danger_zone_container.setVisible(False)
        
        layout.addWidget(self.z_danger_zone)
        layout.addStretch()
        
        return tab

    def set_target_to_min(self):
        """Set the anchor point to the location of the minimum Z value."""
        current_raster = self.main_window.image_window.current_raster
        if current_raster and current_raster.z_channel is not None:
            row, col = np.unravel_index(np.argmin(current_raster.z_channel), current_raster.z_channel.shape)
            pos = QPointF(col, row)
            self.tool.set_z_anchor_point(pos)
        else:
            QMessageBox.warning(self, "No Z-Channel", "No Z-channel data available for the current image.")

    def set_target_to_max(self):
        """Set the anchor point to the location of the maximum Z value."""
        current_raster = self.main_window.image_window.current_raster
        if current_raster and current_raster.z_channel is not None:
            row, col = np.unravel_index(np.argmax(current_raster.z_channel), current_raster.z_channel.shape)
            pos = QPointF(col, row)
            self.tool.set_z_anchor_point(pos)
        else:
            QMessageBox.warning(self, "No Z-Channel", "No Z-channel data available for the current image.")

    def on_tab_changed(self, index):
        """Handle tab changes to update interaction mode."""
        self.tool.stop_current_drawing()
        self.reset_fields()
        
        if index == 0:
            # XY Scale Tab
            self.current_mode = 'xy_scale'
            self.apply_button.setText("Apply")
            self.apply_button.setToolTip("Apply pixel calibration to highlighted images")
        else:
            # Z-Calibration Tab
            # Set mode based on active radio button
            self.on_interaction_mode_changed()
            
        self.update_z_tab_states()

    def on_interaction_mode_changed(self):
        """Switch between View Mode, NaN, Z-Scaling and Z-Anchoring modes."""
        if self.tab_widget.currentIndex() != 1:
            return

        self.tool.stop_current_drawing()
        
        if self.radio_view.isChecked():
            self.current_mode = 'z_view'
            self.controls_stack.setCurrentIndex(0)
            self.controls_stack.show()
            self.apply_button.setText("Apply")
            self.apply_button.setToolTip("Apply view mode settings to highlighted images")
            self.z_info_label.setText(
                "Choose how to display the Z-channel data. 'Depth' shows standard depth from the camera "
                "(positive values = farther away). 'Elevation' shows relative elevation from the bottom "
                "(positive values = elevation above the lowest point). When using Elevation mode, "
                "set the Elevation Reference to define what value represents the reference elevation "
                "(e.g., 0 for sea level)."
            )
        elif self.radio_nan.isChecked():
            self.current_mode = 'z_nan'
            self.controls_stack.setCurrentIndex(1)
            self.controls_stack.show()
            self.apply_button.setText("Apply")
            self.apply_button.setToolTip("Set NaN/NoData value for highlighted images")
            # Load and display current NaN value
            self.tool.load_current_nan_value()
            self.z_info_label.setText(
                "Click on a pixel in the image to sample the value that represents 'No Data' or invalid depth values. "
                "This value will be treated as NaN (Not a Number) in all Z-channel calculations and visualizations."
            )
        elif self.radio_scale.isChecked():
            self.current_mode = 'z_scale'
            self.controls_stack.setCurrentIndex(2)
            self.controls_stack.show()
            self.apply_button.setText("Apply")
            self.apply_button.setToolTip("Update the vertical multiplier (scalar) for highlighted images")
            self.z_info_label.setText(
                "Draw a line across a known vertical distance in the image to calibrate the Z-scale. "
                "Enter the real-world distance between the start and end points of your line, "
                "then draw the line on the image. This will calculate the appropriate scaling factor "
                "to convert raw Z-values to real units."
            )
        else:
            self.current_mode = 'z_anchor'
            self.controls_stack.setCurrentIndex(3)
            self.controls_stack.show()
            self.apply_button.setText("Apply")
            self.apply_button.setToolTip("Update the reference zero-point (offset) for highlighted images")
            self.z_info_label.setText(
                "Click on a point in the image to set the absolute reference value "
                "(e.g., sea level or ground level). Enter the desired Z-value for that point, "
                "and the system will adjust the offset so that the clicked point has that value. "
                "This sets the zero-point for your depth or elevation measurements."
            )

    def on_view_mode_changed(self, index):
        """
        Handle switching between Depth and Relative Elevation view modes.
        Triggers immediate non-destructive update on the current image.
        """
        mode = self.view_mode_combo.currentData()  # 'depth' or 'elevation'
        self.tool.set_z_view_mode(mode)
        
        # Load current inversion reference if available
        if mode == 'elevation':
            current_raster = self.main_window.image_window.current_raster
            if current_raster and current_raster.z_inversion_reference is not None:
                self.z_inversion_ref_input.setValue(current_raster.z_inversion_reference)

    def update_z_tab_states(self):
        """Enable/disable Z tab based on whether current image has Z-channel."""
        current_raster = self.main_window.image_window.current_raster
        has_z_channel = current_raster is not None and current_raster.z_channel is not None
        
        # Set tab enabled state
        self.tab_widget.setTabEnabled(1, has_z_channel)
        
        # If disabled and currently selected, switch to XY tab
        if not has_z_channel and self.tab_widget.currentIndex() == 1:
            self.tab_widget.setCurrentIndex(0)

    def get_selected_image_paths(self):
        """Get the selected image paths - only highlighted rows."""
        return self.main_window.image_window.table_model.get_highlighted_paths()

    def update_status_label(self):
        """Update the status label."""
        highlighted_paths = self.main_window.image_window.table_model.get_highlighted_paths()
        count = len(highlighted_paths)
        self.status_label.setText(f"{count} images highlighted" if count != 1 else "1 image highlighted")

    def reset_fields(self):
        """Resets the dialog fields."""
        # XY Scale
        self.pixel_length_label.setText("Draw a line on the image")
        # Don't reset scale label - keep showing current scale
        # Z NaN
        self.z_nan_hover_label.setText("Hover over image...")
        self.z_nan_clicked_label.setText("Click to select...")
        # Z Scale
        self.z_raw_diff_label.setText("Draw a line...")
        self.z_scalar_label.setText("N/A")
        # Z Anchor
        self.z_current_val_label.setText("Click a point...")
        self.z_offset_label.setText("N/A")

    def clear_scale_line(self):
        self.tool.stop_current_drawing()
        self.reset_fields()
        self.tool.load_existing_scale()

    def closeEvent(self, event):
        self.tool.stop_current_drawing()
        self.tool.deactivate()
        event.accept()


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
        
        # Z-Calibration Reset
        self.dialog.reset_z_btn.clicked.connect(self.reset_z_settings)

        # --- Drawing State ---
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        self.pixel_length = 0.0
        
        # Z-Anchor state
        self.z_anchor_point = None

        # --- Graphics Items ---
        # Line for scale setting
        self.preview_line = QGraphicsLineItem()
        pen = QPen(QColor(230, 62, 0), 3, Qt.DashLine)  # Blood red dashed line
        pen.setCosmetic(True)
        self.preview_line.setPen(pen)
        self.preview_line.setZValue(100)

        # Initialize the Z-Fence Overlay
        self.z_fence = ZProfilePathItem()
        self.z_fence.setVisible(False)
        
        self.show_crosshair = True

    def activate(self):
        super().activate()
        
        # Add items to scene
        if not self.preview_line.scene():
            self.annotation_window.scene.addItem(self.preview_line)
        if not self.z_fence.scene():
            self.annotation_window.scene.addItem(self.z_fence)
        
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
            highlighted_paths = self.main_window.image_window.table_model.get_highlighted_paths()
            if current_image_path not in highlighted_paths:
                self.main_window.image_window.table_model.set_highlighted_paths([current_image_path])
        
        self.dialog.update_status_label()
        self.load_existing_scale()
        self.dialog.update_z_tab_states()
        
        # Sync View Mode UI with current raster settings
        self.sync_view_mode_ui()

        self.dialog.show()
        self.dialog.activateWindow()

    def deactivate(self):
        if not self.active:
            return
        super().deactivate()
        self.dialog.hide()
        self.preview_line.hide()
        self.z_fence.setVisible(False)
        self.is_drawing = False
        self.main_window.untoggle_all_tools()

    def stop_current_drawing(self):
        """Stop any active drawing and clear graphics."""
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        self.pixel_length = 0.0
        
        if self.preview_line.scene():
            self.preview_line.hide()
            
        self.z_fence.setVisible(False)
        self.z_fence.setPath(QPainterPath())

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        if event.button() == Qt.LeftButton:
            scene_pos = self.annotation_window.mapToScene(event.pos())
            
            # Line Drawing Mode (XY Scale or Z-Scale)
            if self.dialog.current_mode in ['xy_scale', 'z_scale']:
                if not self.is_drawing:
                    self.start_point = scene_pos
                    self.end_point = scene_pos
                    self.is_drawing = True
                else:
                    self.end_point = scene_pos
                    self.is_drawing = False
                    
                    if self.dialog.current_mode == 'xy_scale':
                        self.calculate_scale()
                    else:
                        self.calculate_z_scale()
                        
            # Point Click Mode (Z-NaN or Z-Anchor)
            elif self.dialog.current_mode == 'z_nan':
                self.set_z_nan_point(scene_pos)
            elif self.dialog.current_mode == 'z_anchor':
                self.set_z_anchor_point(scene_pos)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for drawing."""
        # Call parent to handle crosshair
        super().mouseMoveEvent(event)
        
        scene_pos = self.annotation_window.mapToScene(event.pos())
        
        # Update current z-value display when in NaN mode
        if self.dialog.current_mode == 'z_nan':
            self.update_z_value_display(scene_pos)
        
        if self.is_drawing and self.start_point:
            self.end_point = scene_pos
            
            # Update Line
            line = QLineF(self.start_point, self.end_point)
            self.preview_line.setLine(line)
            self.preview_line.show()
            
            # Update Text
            if self.dialog.current_mode == 'xy_scale':
                pixel_length = line.length()
                self.dialog.pixel_length_label.setText(f"{pixel_length:.2f} pixels")
            
            # Update Z-Fence (only in Z modes)
            if self.dialog.current_mode in ['z_scale', 'z_anchor']:
                current_raster = self.main_window.image_window.current_raster
                if current_raster and current_raster.z_channel is not None:
                    self.z_fence.setVisible(True)
                    self.z_fence.update_profile(self.start_point, self.end_point, current_raster)

    def mouseReleaseEvent(self, event: QMouseEvent):
        pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Backspace:
            self.stop_current_drawing()
            self.dialog.reset_fields()

    # --- XY Scale Logic (Unchanged) ---
    def load_existing_scale(self):
        """Loads and displays existing scale data."""
        current_path = self.annotation_window.current_image_path
        if not current_path:
            self.dialog.calculated_scale_label.setText("N/A")
            return
        
        raster = self.main_window.image_window.raster_manager.get_raster(current_path)
        if not raster or raster.scale_x is None:
            self.dialog.calculated_scale_label.setText("N/A")
            return
        
        scale_value = raster.scale_x
        units = raster.scale_units if raster.scale_units else "metre"
        self.dialog.calculated_scale_label.setText(f"{scale_value:.6f} {units}/pixel")

    def calculate_scale(self):
        """Calculate pixel scale."""
        if not self.start_point or not self.end_point: return
        line = QLineF(self.start_point, self.end_point)
        pixel_length = line.length()
        if pixel_length == 0: return
        
        known_length = self.dialog.known_length_input.value()
        units = self.dialog.units_combo.currentText()
        scale = known_length / pixel_length
        self.dialog.calculated_scale_label.setText(f"{scale:.6f} {units}/pixel")
        self.dialog.pixel_length_label.setText(f"{pixel_length:.2f} pixels")

    def apply_scale(self):
        """Apply XY scale to highlighted images."""
        highlighted_paths = self.dialog.get_selected_image_paths()
        if not highlighted_paths: return
        
        scale_text = self.dialog.calculated_scale_label.text()
        if "N/A" in scale_text: return
        
        try:
            scale_value = float(scale_text.split()[0])
            units = scale_text.split()[1].split('/')[0]
        except: return

        # Standardize units
        unit_mapping = {'mm': 'millimetre', 'cm': 'centimetre', 'm': 'metre', 
                       'km': 'kilometre', 'in': 'inch', 'ft': 'foot', 'yd': 'yard', 'mi': 'mile'}
        scale_units = unit_mapping.get(units, 'metre')
        
        raster_manager = self.main_window.image_window.raster_manager
        current_path = self.annotation_window.current_image_path
        
        for path in highlighted_paths:
            raster = raster_manager.get_raster(path)
            if raster:
                raster.update_scale(scale_value, scale_value, scale_units)
                raster_manager.rasterUpdated.emit(path)
                
        if current_path in highlighted_paths:
            # Refresh view dimensions
            w, h = self.annotation_window.get_image_dimensions()
            self.main_window.update_view_dimensions(w, h)
            
        QMessageBox.information(self.dialog, "Applied", f"Scale applied to {len(highlighted_paths)} images.")
        self.stop_current_drawing()

    def remove_scale_highlighted(self):
        """Remove scale from highlighted."""
        highlighted_paths = self.dialog.get_selected_image_paths()
        if not highlighted_paths: 
            return
        
        if QMessageBox.question(self.dialog, "Confirm", "Remove scale?") != QMessageBox.Yes: return
        
        raster_manager = self.main_window.image_window.raster_manager
        current_path = self.annotation_window.current_image_path
        
        for path in highlighted_paths:
            raster = raster_manager.get_raster(path)
            if raster:
                raster.remove_scale()
                raster_manager.rasterUpdated.emit(path)
                
        if current_path in highlighted_paths:
            w, h = self.annotation_window.get_image_dimensions()
            self.main_window.update_view_dimensions(w, h)
        
        self.load_existing_scale()

    # --- Z-Calibration Logic (New) ---

    def load_current_nan_value(self):
        """Load and display current NaN value from the current raster."""
        current_raster = self.main_window.image_window.current_raster
        if not current_raster or current_raster.z_channel is None:
            self.dialog.z_nan_current_label.setText("Not Set")
            return
        
        if hasattr(current_raster, 'z_nodata') and current_raster.z_nodata is not None:
            self.dialog.z_nan_current_label.setText(f"{current_raster.z_nodata:.4f}")
        else:
            self.dialog.z_nan_current_label.setText("Not Set")

    def update_z_value_display(self, scene_pos):
        """Update the z-value display based on current mouse position."""
        current_raster = self.main_window.image_window.current_raster
        if not current_raster or current_raster.z_channel_lazy is None:
            self.dialog.z_nan_hover_label.setText("No Z-channel")
            return
        
        # Get pixel coordinates
        x, y = int(scene_pos.x()), int(scene_pos.y())
        
        # Ensure coordinates are within bounds
        if (x < 0 or y < 0 or 
            y >= current_raster.z_channel_lazy.shape[0] or 
            x >= current_raster.z_channel_lazy.shape[1]):
            self.dialog.z_nan_hover_label.setText("Out of bounds")
            return
        
        try:
            # Sample raw z-channel value at current position
            z_value = float(current_raster.z_channel_lazy[y, x])
            self.dialog.z_nan_hover_label.setText(f"{z_value:.4f}")
        except Exception:
            self.dialog.z_nan_hover_label.setText("Invalid")

    def set_z_nan_point(self, pos):
        """Handle NaN point click to sample pixel value."""
        current_raster = self.main_window.image_window.current_raster
        if not current_raster or current_raster.z_channel_lazy is None:
            return
        
        # Sample raw z-channel value at clicked position
        x, y = int(pos.x()), int(pos.y())
        
        # Ensure coordinates are within bounds
        if (x < 0 or y < 0 or 
            y >= current_raster.z_channel_lazy.shape[0] or 
            x >= current_raster.z_channel_lazy.shape[1]):
            return
        
        try:
            sampled_value = float(current_raster.z_channel_lazy[y, x])
            self.dialog.z_nan_clicked_label.setText(f"{sampled_value:.4f}")
            
            # Store sampled value for apply operation
            self.sampled_nan_value = sampled_value
            
        except Exception as e:
            QMessageBox.warning(self.dialog, "Error", f"Could not sample pixel value: {str(e)}")

    def apply_z_nan(self):
        """Apply NaN value to highlighted images with confirmation dialog."""
        highlighted = self.dialog.get_selected_image_paths()
        if not highlighted:
            return
        
        # Check if we have a sampled value
        if not hasattr(self, 'sampled_nan_value'):
            QMessageBox.warning(
                self.dialog, 
                "No Value Selected", 
                "Please click on a pixel to sample a NaN value first."
            )
            return
        
        new_nan = self.sampled_nan_value
        
        # Collect existing NaN values from highlighted images
        raster_manager = self.main_window.image_window.raster_manager
        existing_nans = []
        images_with_nan = 0
        images_without_nan = 0
        
        for path in highlighted:
            raster = raster_manager.get_raster(path)
            if raster and raster.z_channel is not None:
                if hasattr(raster, 'z_nodata') and raster.z_nodata is not None:
                    existing_nans.append(raster.z_nodata)
                    images_with_nan += 1
                else:
                    images_without_nan += 1
        
        # Build confirmation message
        if existing_nans:
            # Show unique existing values
            unique_nans = set(existing_nans)
            if len(unique_nans) == 1:
                prev_text = f"{list(unique_nans)[0]:.4f}"
            else:
                prev_text = f"Multiple values ({len(unique_nans)} unique)"
        else:
            prev_text = "Not Set"
        
        confirm_msg = (
            f"<b>Set NaN/NoData value for {len(highlighted)} image(s)?</b><br><br>"
            f"<b>Previous NaN Value:</b> {prev_text}<br>"
            f"<b>New NaN Value:</b> {new_nan:.4f}<br><br>"
            f"Images with existing NaN: {images_with_nan}<br>"
            f"Images without NaN: {images_without_nan}"
        )
        
        reply = QMessageBox.question(
            self.dialog,
            "Confirm NaN Value",
            confirm_msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Apply to all highlighted images
        current_path = self.annotation_window.current_image_path
        
        for path in highlighted:
            raster = raster_manager.get_raster(path)
            if raster and raster.z_channel is not None:
                raster.z_nodata = new_nan
                raster_manager.rasterUpdated.emit(path)
        
        # Refresh visualization if current image was updated
        if current_path in highlighted:
            self.annotation_window.refresh_z_channel_visualization()
            self.load_current_nan_value()
        
        # Success message
        success_msg = (
            f"<b>Successfully set NaN value for {len(highlighted)} image(s)</b><br><br>"
            f"<b>NaN Value:</b> {new_nan:.4f}<br>"
        )
        QMessageBox.information(self.dialog, "Success", success_msg)
        
        # Reset hover and clicked value displays
        self.dialog.z_nan_hover_label.setText("Hover over image...")
        self.dialog.z_nan_clicked_label.setText("Click to select...")
        if hasattr(self, 'sampled_nan_value'):
            delattr(self, 'sampled_nan_value')

    def calculate_z_scale(self):
        """Calculate Z scalar from line."""
        from coralnet_toolbox.utilities import calculate_z_scalar, validate_line_angle
        
        current_raster = self.main_window.image_window.current_raster
        if not current_raster or current_raster.z_channel is None: return

        # Validate line
        is_valid, _, _, warning = validate_line_angle(self.start_point, self.end_point)
        if not is_valid:
            QMessageBox.warning(self.dialog, "Invalid Line", warning)
            self.stop_current_drawing()
            return
            
        # Get Z values (using semantic values which respect current scalar)
        # Note: To calculate a NEW scalar, we ideally want raw difference.
        # But get_z_value applies current scalar. We can reverse it or just use raw data.
        # Let's use raw data for pure calibration.
        
        x1, y1 = int(self.start_point.x()), int(self.start_point.y())
        x2, y2 = int(self.end_point.x()), int(self.end_point.y())
        
        # Access raw z-channel directly
        try:
            z1 = float(current_raster.z_channel_lazy[y1, x1])
            z2 = float(current_raster.z_channel_lazy[y2, x2])
        except: return
        
        raw_diff = abs(z1 - z2)
        known_diff = self.dialog.z_known_diff_input.value()
        
        try:
            scalar = calculate_z_scalar(raw_diff, known_diff)
            self.dialog.z_raw_diff_label.setText(f"{raw_diff:.4f} (raw units)")
            self.dialog.z_scalar_label.setText(f"{scalar:.6f}")
        except ValueError as e:
            QMessageBox.warning(self.dialog, "Error", str(e))

    def apply_z_scale(self):
        """Apply scalar to highlighted images."""
        highlighted = self.dialog.get_selected_image_paths()
        if not highlighted: return
        
        scalar_text = self.dialog.z_scalar_label.text()
        if "N/A" in scalar_text: return
        
        try:
            scalar = float(scalar_text)
        except: return
        
        raster_manager = self.main_window.image_window.raster_manager
        current_path = self.annotation_window.current_image_path
        
        for path in highlighted:
            raster = raster_manager.get_raster(path)
            if raster and raster.z_channel is not None:
                # Update scalar, preserve offset/direction
                raster.z_settings['scalar'] = scalar
                # Save z_inversion_reference if in elevation mode
                if self.dialog.view_mode_combo.currentData() == 'elevation':
                    raster.z_inversion_reference = self.dialog.z_inversion_ref_input.value()
                raster_manager.rasterUpdated.emit(path)
                
        if current_path in highlighted:
            self.annotation_window.refresh_z_channel_visualization()
        
        # Success dialog with details
        view_mode = self.dialog.view_mode_combo.currentText()
        success_msg = (
            f"<b>Successfully applied calibration to {len(highlighted)} image(s)</b><br><br>"
            f"<b>Display Mode:</b> {view_mode}<br>"
            f"<b>Vertical Scalar:</b> {scalar:.6f}<br>"
        )
        QMessageBox.information(self.dialog, "Success", success_msg)
        self.stop_current_drawing()
        self.dialog.reset_fields()

    def apply_z_view(self):
        """Apply view mode and elevation reference to highlighted images."""
        highlighted = self.dialog.get_selected_image_paths()
        if not highlighted:
            return
        
        view_mode = self.dialog.view_mode_combo.currentData()
        elevation_ref = self.dialog.z_inversion_ref_input.value()
        
        raster_manager = self.main_window.image_window.raster_manager
        current_path = self.annotation_window.current_image_path
        
        for path in highlighted:
            raster = raster_manager.get_raster(path)
            if raster and raster.z_channel is not None:
                # Apply view mode
                settings = raster.z_settings
                scalar = settings.get('scalar', 1.0)
                
                if view_mode == 'depth':
                    settings['direction'] = 1
                    settings['offset'] = 0.0
                elif view_mode == 'elevation':
                    settings['direction'] = -1
                    # Auto-tare to max raw value
                    try:
                        max_raw = float(np.nanmax(raster.z_channel_lazy))
                    except Exception:
                        max_raw = 0.0
                    settings['offset'] = max_raw * scalar
                    raster.z_inversion_reference = elevation_ref
                
                raster_manager.rasterUpdated.emit(path)
        
        if current_path in highlighted:
            self.annotation_window.refresh_z_channel_visualization()
        
        # Success message
        view_mode_text = self.dialog.view_mode_combo.currentText()
        success_msg = (
            f"<b>Successfully applied view mode to {len(highlighted)} image(s)</b><br><br>"
            f"<b>Display Mode:</b> {view_mode_text}<br>"
        )
        if view_mode == 'elevation':
            success_msg += f"<b>Elevation Reference:</b> {elevation_ref:.2f} m<br>"
        
        QMessageBox.information(self.dialog, "Success", success_msg)

    def set_z_anchor_point(self, pos):
        """Handle anchor point click."""
        current_raster = self.main_window.image_window.current_raster
        if not current_raster or current_raster.z_channel is None:
            return
        
        # Get current semantic value (includes current offset/direction)
        current_z = current_raster.get_z_value(int(pos.x()), int(pos.y()))
        if current_z is None:
            return
        
        self.dialog.z_current_val_label.setText(f"{current_z:.3f}")
        
        # Calculate required offset change
        target = self.dialog.z_target_val_input.value()
        
        # The equation is: Z_display = Base + Offset
        # We want: Target = Base + NewOffset
        # Current = Base + OldOffset
        # So: NewOffset = OldOffset + (Target - Current)
        
        old_offset = current_raster.z_settings.get('offset', 0.0)
        delta = target - current_z
        new_offset = old_offset + delta
        
        self.dialog.z_offset_label.setText(f"{new_offset:.4f}")

    def apply_z_anchor(self):
        """Apply new offset to highlighted images."""
        highlighted = self.dialog.get_selected_image_paths()
        if not highlighted: 
            return
        
        offset_text = self.dialog.z_offset_label.text()
        if "N/A" in offset_text: 
            return
        
        try: 
            offset = float(offset_text)
        except: 
            return
        
        raster_manager = self.main_window.image_window.raster_manager
        current_path = self.annotation_window.current_image_path
        
        for path in highlighted:
            raster = raster_manager.get_raster(path)
            if raster and raster.z_channel is not None:
                raster.z_settings['offset'] = offset
                # Save z_inversion_reference if in elevation mode
                if self.dialog.view_mode_combo.currentData() == 'elevation':
                    raster.z_inversion_reference = self.dialog.z_inversion_ref_input.value()
                raster_manager.rasterUpdated.emit(path)
                
        if current_path in highlighted:
            self.annotation_window.refresh_z_channel_visualization()
        
        # Success dialog with details
        view_mode = self.dialog.view_mode_combo.currentText()
        success_msg = (
            f"<b>Successfully applied calibration to {len(highlighted)} image(s)</b><br><br>"
            f"<b>Display Mode:</b> {view_mode}<br>"
            f"<b>Reference Offset:</b> {offset:.4f}<br>"
        )
        QMessageBox.information(self.dialog, "Success", success_msg)
        self.dialog.reset_fields()

    def set_z_view_mode(self, mode):
        """
        Toggle between Depth and Elevation view modes non-destructively.
        Updates 'direction' and 'offset' in z_settings, and updates z_data_type metadata.
        """
        current_raster = self.main_window.image_window.current_raster
        if not current_raster or current_raster.z_channel_lazy is None: return
        
        settings = current_raster.z_settings
        scalar = settings.get('scalar', 1.0)
        
        if mode == 'depth':
            # Standard Depth Mode
            settings['direction'] = 1  # Positive = farther
            settings['offset'] = 0.0   # Zero at camera
            current_raster.z_data_type = 'depth'  # Update metadata
            
        elif mode == 'elevation':
            # Relative Elevation Mode
            # 1. Set Direction to -1 (Positive = closer/up)
            settings['direction'] = -1
            
            # 2. Auto-Tare: Find max raw value to set as zero-point (seafloor)
            # We use the raw data directly for robustness
            raw_data = current_raster.z_channel_lazy
            
            # Simple max ignoring NaNs
            try:
                max_raw = float(np.nanmax(raw_data))
            except:
                max_raw = 0.0
                
            # Offset = Max_Raw * Scalar
            # Explanation: Raw=10, Scalar=1. Depth=10. 
            # We want Elevation=0 at Depth=10.
            # Elev = -1 * (10*1) + Offset = 0  =>  Offset = 10
            settings['offset'] = max_raw * scalar
            current_raster.z_data_type = 'elevation'  # Update metadata
            
        # Refresh visuals
        self.annotation_window.refresh_z_channel_visualization()
        
        # Emit update so other UI components refresh
        self.main_window.image_window.raster_manager.rasterUpdated.emit(current_raster.image_path)

    def sync_view_mode_ui(self):
        """Update dropdown selection to match current raster's z_data_type."""
        current_raster = self.main_window.image_window.current_raster
        if not current_raster or current_raster.z_channel is None: 
            return
        
        # Use z_data_type to determine which mode to display
        # z_data_type can be 'depth' or 'elevation'
        data_type = getattr(current_raster, 'z_data_type', 'depth')
        
        # Set combobox to match the data type
        # Index 0 = Depth, Index 1 = Elevation
        if data_type == 'elevation':
            index = 1
        else:
            index = 0  # Default to depth
        
        self.dialog.view_mode_combo.blockSignals(True)
        self.dialog.view_mode_combo.setCurrentIndex(index)
        self.dialog.view_mode_combo.blockSignals(False)
        
        # Also load the elevation reference if available
        if hasattr(current_raster, 'z_inversion_reference') and current_raster.z_inversion_reference is not None:
            self.dialog.z_inversion_ref_input.blockSignals(True)
            self.dialog.z_inversion_ref_input.setValue(current_raster.z_inversion_reference)
            self.dialog.z_inversion_ref_input.blockSignals(False)

    def reset_z_settings(self):
        """Reset Z-settings to defaults."""
        highlighted = self.dialog.get_selected_image_paths()
        if not highlighted: 
            return
        
        if QMessageBox.question(self.dialog, "Confirm", "Reset Z-settings for highlighted?") != QMessageBox.Yes:
            return
            
        raster_manager = self.main_window.image_window.raster_manager
        current_path = self.annotation_window.current_image_path
        
        for path in highlighted:
            raster = raster_manager.get_raster(path)
            if raster and raster.z_channel is not None:
                raster.z_settings = {'scalar': 1.0, 'offset': 0.0, 'direction': 1}
                raster.z_inversion_reference = None  # Reset inversion reference
                raster_manager.rasterUpdated.emit(path)
                
        if current_path in highlighted:
            self.annotation_window.refresh_z_channel_visualization()
            self.sync_view_mode_ui()
        
        self.dialog.reset_fields()

    def handle_apply(self):
        """Route 'Apply' button to correct function."""
        if self.dialog.current_mode == 'xy_scale':
            self.apply_scale()
        elif self.dialog.current_mode == 'z_view':
            self.apply_z_view()
        elif self.dialog.current_mode == 'z_nan':
            self.apply_z_nan()
        elif self.dialog.current_mode == 'z_scale':
            self.apply_z_scale()
        elif self.dialog.current_mode == 'z_anchor':
            self.apply_z_anchor()

    def on_image_changed(self):
        """Handle image change events to update UI state."""
        self.dialog.update_z_tab_states()
        self.load_existing_scale()
        self.stop_current_drawing()
        self.sync_view_mode_ui()  # This now handles z_inversion_reference loading
