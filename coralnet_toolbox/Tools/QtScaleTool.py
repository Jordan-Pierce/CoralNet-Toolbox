import warnings

from PyQt5.QtCore import Qt, QLineF, QPointF
from PyQt5.QtGui import QMouseEvent, QPen, QColor
from PyQt5.QtWidgets import (QDialog, QWidget, QVBoxLayout, QFormLayout, 
                             QDoubleSpinBox, QComboBox, QLabel,
                             QDialogButtonBox, QMessageBox, QGraphicsLineItem,
                             QGroupBox, QPushButton)

from coralnet_toolbox.Tools.QtTool import Tool
from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# ScaleToolDialog Class
# ----------------------------------------------------------------------------------------------------------------------


class ScaleToolDialog(QDialog):
    """
    A modeless dialog for the ScaleTool, allowing user input for scale calculation
    and propagation.
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

        # --- Set Scale Tab Only ---
        self.setup_scale_tab(self.main_layout)

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

    def setup_scale_tab(self, layout):
        """Populates the 'Set Scale' section."""
        scale_layout = QFormLayout()
        
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

        scale_layout.addRow("Known Length:", self.known_length_input)
        scale_layout.addRow("Units:", self.units_combo)
        scale_layout.addRow("Pixel Length:", self.pixel_length_label)
        scale_layout.addRow("Scale:", self.calculated_scale_label)
        
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

        scale_layout.addRow(self.danger_zone_group_box)
        
        layout.addLayout(scale_layout)

    def get_selected_image_paths(self):
        """
        Get the selected image paths - only highlighted rows.
        
        :return: List of highlighted image paths
        """
        # Get highlighted image paths from the table model
        return self.main_window.image_window.table_model.get_highlighted_paths()

    def reset_fields(self):
        """Resets the dialog fields to their default state."""
        self.pixel_length_label.setText("Draw a line on the image")
        self.calculated_scale_label.setText("Scale: N/A")

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
        self.tool.deactivate()
        event.accept()


# ----------------------------------------------------------------------------------------------------------------------
# ScaleTool Class
# ----------------------------------------------------------------------------------------------------------------------


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

        self.dialog.remove_highlighted_button.clicked.connect(self.remove_scale_highlighted)

        # --- Drawing State ---
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        self.pixel_length = 0.0

        # --- Graphics Items ---
        # Line for scale setting
        self.preview_line = QGraphicsLineItem()
        pen = QPen(QColor(255, 0, 0), 2, Qt.DashLine)
        pen.setCosmetic(True)  # Make pen width independent of zoom level
        self.preview_line.setPen(pen)
        self.preview_line.setZValue(100)

        self.show_crosshair = True  # Enable crosshair for precise measurements

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
        
        # Add preview line to scene
        if not self.preview_line.scene():
            self.annotation_window.scene.addItem(self.preview_line)
        
        self.stop_current_drawing()
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

        # Load and display existing scale if present
        self.load_existing_scale()

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
        """Handle mouse press for starting scale line."""
        if event.button() == Qt.LeftButton:
            scene_pos = self.annotation_window.mapToScene(event.pos())
            
            if not self.is_drawing:
                # Start new line
                self.start_point = scene_pos
                self.end_point = scene_pos
                self.is_drawing = True
            else:
                # Finish line
                self.end_point = scene_pos
                self.is_drawing = False
                self.calculate_scale()

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
            self.dialog.calculated_scale_label.setText("Scale: N/A")

    def calculate_scale(self):
        """Calculate the scale based on the drawn line."""
        if not self.start_point or not self.end_point:
            return
        
        line = QLineF(self.start_point, self.end_point)
        pixel_length = line.length()
        
        if pixel_length == 0:
            QMessageBox.warning(self.dialog, "Invalid Line", 
                              "Please draw a line with non-zero length.")
            self.stop_current_drawing()
            return
        
        # Get known length and units from dialog
        known_length = self.dialog.known_length_input.value()
        units = self.dialog.units_combo.currentText()
        
        # Calculate scale (real-world units per pixel)
        scale = known_length / pixel_length
        
        # Update label
        scale_text = f"{scale:.6f} {units}/pixel"
        self.dialog.calculated_scale_label.setText(f"Scale: {scale_text}")
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
        for image_path in highlighted_paths:
            raster = raster_manager.get_raster(image_path)
            if raster:
                raster.scale_x = scale_value
                raster.scale_y = scale_value
                raster.scale_units = scale_units
        
        QMessageBox.information(self.dialog, "Scale Applied",
                              f"Scale applied to {len(highlighted_paths)} image(s).")
        
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
        
        reply = QMessageBox.question(self.dialog, "Confirm Removal",
                                   f"Remove scale from {len(highlighted_paths)} image(s)?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply != QMessageBox.Yes:
            return
        
        # Remove scale from each highlighted image
        raster_manager = self.main_window.image_window.raster_manager
        for image_path in highlighted_paths:
            raster = raster_manager.get_raster(image_path)
            if raster:
                raster.scale_x = None
                raster.scale_y = None
                raster.scale_units = None
        
        QMessageBox.information(self.dialog, "Scale Removed",
                              f"Scale removed from {len(highlighted_paths)} image(s).")
        
        self.load_existing_scale()
