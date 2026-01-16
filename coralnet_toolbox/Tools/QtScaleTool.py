import warnings

from PyQt5.QtCore import Qt, QLineF
from PyQt5.QtGui import QMouseEvent, QPen, QColor
from PyQt5.QtWidgets import (QDialog, QWidget, QVBoxLayout, QFormLayout, 
                             QDoubleSpinBox, QComboBox, QLabel, QHBoxLayout,
                             QDialogButtonBox, QMessageBox, QGraphicsLineItem,
                             QGroupBox, QPushButton)

from coralnet_toolbox.Tools.QtTool import Tool
from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class ScaleToolDialog(QDialog):
    """
    A modeless dialog for the ScaleTool, allowing user input for scale calculation.
    """
    def __init__(self, tool, parent=None):
        super().__init__(parent)
        # Get references from the tool
        self.tool = tool
        self.annotation_window = self.tool.annotation_window
        self.main_window = self.annotation_window.main_window

        self.setWindowTitle("Scale Tool")
        self.setWindowIcon(get_icon("scale.png"))
        self.resize(450, 350)

        # This dialog is modeless
        self.setModal(False) 

        self.main_layout = QVBoxLayout(self)

        # Create scale tab
        self.scale_tab = self.create_scale_tab()
        self.main_layout.addWidget(self.scale_tab)

        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Close)
        
        self.apply_button = self.button_box.button(QDialogButtonBox.Apply)
        self.apply_button.setText("Apply")
        self.apply_button.setToolTip("Apply pixel calibration to highlighted images")
        
        self.main_layout.addWidget(self.button_box)
        
        # --- Status Label (at bottom of dialog) ---
        self.status_label = QLabel("No images highlighted")
        self.status_label.setAlignment(Qt.AlignLeft)
        self.main_layout.addWidget(self.status_label)
        
        # Signal connection will be made in activate() when image_window is guaranteed to exist
        self._signal_connected = False

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
        self.known_length_input.valueChanged.connect(self.tool.calculate_scale)
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
        self.pixel_length_label.setText("Draw a line on the image")
        # Don't reset scale label - keep showing current scale

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
    Tool for setting image scale.
    """
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.cursor = Qt.CrossCursor
        self.dialog = ScaleToolDialog(self, self.annotation_window)
        
        # --- Button Connections ---
        apply_btn = self.dialog.button_box.button(QDialogButtonBox.Apply)
        apply_btn.clicked.connect(self.apply_scale)
        
        close_btn = self.dialog.button_box.button(QDialogButtonBox.Close)
        close_btn.clicked.connect(self.deactivate)

        # XY Scale tab connections
        self.dialog.remove_highlighted_button.clicked.connect(self.remove_scale_highlighted)

        # --- Drawing State ---
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        self.pixel_length = 0.0

        # --- Graphics Items ---
        # Line for scale setting
        self.preview_line = QGraphicsLineItem()
        pen = QPen(QColor(230, 62, 0), 3, Qt.DashLine)  # Blood red dashed line
        pen.setCosmetic(True)
        self.preview_line.setPen(pen)
        self.preview_line.setZValue(100)
        
        self.show_crosshair = True

    def activate(self):
        super().activate()
        
        # Add items to scene
        if not self.preview_line.scene():
            self.annotation_window.scene.addItem(self.preview_line)
        
        self.stop_current_drawing()
        self.dialog.reset_fields()
        
        # Connect signal to update highlighted count (only once)
        if not self.dialog._signal_connected:
            self.main_window.image_window.table_model.rowsChanged.connect(self.dialog.update_status_label)
            self.main_window.image_window.imageChanged.connect(self.on_image_changed)
            # Connect rasterUpdated to ensure UI stays in sync with data changes
            self.main_window.image_window.raster_manager.rasterUpdated.connect(self.on_image_changed)
            self.dialog._signal_connected = True
        
        # Automatically highlight the current image if one is loaded
        current_image_path = self.annotation_window.current_image_path
        if current_image_path:
            highlighted_paths = self.main_window.image_window.table_model.get_highlighted_paths()
            if current_image_path not in highlighted_paths:
                self.main_window.image_window.table_model.set_highlighted_paths([current_image_path])
        
        self.dialog.update_status_label()
        self.load_existing_scale()

        self.dialog.show()
        self.dialog.activateWindow()

    def deactivate(self):
        if not self.active:
            return
        super().deactivate()
        self.dialog.hide()
        self.preview_line.hide()
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

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        if event.button() == Qt.LeftButton:
            scene_pos = self.annotation_window.mapToScene(event.pos())
            
            if not self.is_drawing:
                self.start_point = scene_pos
                self.end_point = scene_pos
                self.is_drawing = True
            else:
                self.end_point = scene_pos
                self.is_drawing = False
                self.calculate_scale()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for drawing."""
        # Call parent to handle crosshair
        super().mouseMoveEvent(event)
        
        scene_pos = self.annotation_window.mapToScene(event.pos())
        
        if self.is_drawing and self.start_point:
            self.end_point = scene_pos
            
            # Update Line
            line = QLineF(self.start_point, self.end_point)
            self.preview_line.setLine(line)
            self.preview_line.show()
            
            # Update Text
            pixel_length = line.length()
            self.dialog.pixel_length_label.setText(f"{pixel_length:.2f} pixels")

    def mouseReleaseEvent(self, event: QMouseEvent):
        pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Backspace:
            self.stop_current_drawing()
            self.dialog.reset_fields()

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
        if not self.start_point or not self.end_point: 
            return
        line = QLineF(self.start_point, self.end_point)
        pixel_length = line.length()
        if pixel_length == 0: 
            return
        
        known_length = self.dialog.known_length_input.value()
        units = self.dialog.units_combo.currentText()
        scale = known_length / pixel_length
        self.dialog.calculated_scale_label.setText(f"{scale:.6f} {units}/pixel")
        self.dialog.pixel_length_label.setText(f"{pixel_length:.2f} pixels")

    def apply_scale(self):
        """Apply XY scale to highlighted images."""
        highlighted_paths = self.dialog.get_selected_image_paths()
        if not highlighted_paths: 
            return
        
        scale_text = self.dialog.calculated_scale_label.text()
        if "N/A" in scale_text: 
            return
        
        try:
            scale_value = float(scale_text.split()[0])
            units = scale_text.split()[1].split('/')[0]
        except: 
            return

        # Convert the scale value to meters (standardized internal unit)
        from coralnet_toolbox.utilities import convert_scale_units
        scale_value_meters = convert_scale_units(scale_value, units, 'm')
        
        raster_manager = self.main_window.image_window.raster_manager
        current_path = self.annotation_window.current_image_path
        
        for path in highlighted_paths:
            raster = raster_manager.get_raster(path)
            if raster:
                # Always store in meters with 'm' as the unit
                raster.update_scale(scale_value_meters, scale_value_meters, 'm')
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
        
        if QMessageBox.question(self.dialog, "Confirm", "Remove scale?") != QMessageBox.Yes: 
            return
        
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

    def on_image_changed(self):
        """Handle image change events to update UI state."""
        self.load_existing_scale()
        self.stop_current_drawing()
