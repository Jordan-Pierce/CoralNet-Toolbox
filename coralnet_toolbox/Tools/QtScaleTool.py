import warnings

from PyQt5.QtCore import Qt, QLineF, QRectF
from PyQt5.QtGui import QMouseEvent, QPen, QColor
from PyQt5.QtWidgets import (QApplication, QDialog, QWidget, QVBoxLayout, QTabWidget,
                             QFormLayout, QDoubleSpinBox, QComboBox, QLabel,
                             QDialogButtonBox, QMessageBox, QGraphicsLineItem,
                             QGroupBox, QCheckBox, QButtonGroup, QPushButton,
                             QGraphicsRectItem)

from coralnet_toolbox.Tools.QtTool import Tool

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon

from coralnet_toolbox.utilities import convert_scale_units

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
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

        # --- Scale Status (for reference on other tabs) ---
        self.current_scale_status_label = QLabel("Scale: N/A")
        self.current_scale_status_label.setToolTip("Current scale loaded from the image.")
        self.main_layout.addWidget(self.current_scale_status_label)

        # --- Image Options (Now outside the tabs) ---
        self.setup_options_layout()  
        self.main_layout.addWidget(self.options_group_box)

        # --- Dialog Buttons (Now outside the tabs) ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Close)
        
        # Rename "Apply" to "Set Scale" for clarity
        self.set_scale_button = self.button_box.button(QDialogButtonBox.Apply)
        self.set_scale_button.setText("Set Scale")
        
        self.main_layout.addWidget(self.button_box)

    def setup_scale_tab(self, tab_widget):
        """Populates the 'Set Scale' tab."""
        self.scale_layout = QFormLayout(tab_widget)
        
        self.known_length_input = QDoubleSpinBox()
        self.known_length_input.setRange(0.001, 1000000.0)
        self.known_length_input.setValue(1.0)
        self.known_length_input.setDecimals(3)
        self.known_length_input.setToolTip("Enter the real-world length of the line you will draw.")
        
        self.units_combo = QComboBox()
        self.units_combo.addItems(["mm", "cm", "m", "km"])
        self.units_combo.setCurrentText("m")
        self.units_combo.setToolTip("Select the units for the known length.")

        self.pixel_length_label = QLabel("Draw a line on the image")
        self.pixel_length_label.setToolTip("The length of the drawn line in pixels.")
        
        self.calculated_scale_label = QLabel("Scale: N/A")
        self.calculated_scale_label.setToolTip("The resulting scale in meters per pixel.")

        self.scale_layout.addRow("Known Length:", self.known_length_input)
        self.scale_layout.addRow("Units:", self.units_combo)
        self.scale_layout.addRow("Pixel Length:", self.pixel_length_label)
        self.scale_layout.addRow("Result:", self.calculated_scale_label)
        
        # Button for current image (styled in red)
        self.remove_current_button = QPushButton("Remove Scale from Current Image")
        self.remove_current_button.setToolTip("Removes the scale data from this image only.")
        self.remove_current_button.setStyleSheet(
            "background-color: #D9534F; color: white; font-weight: bold;"
        )
        self.scale_layout.addRow(self.remove_current_button)

        # Button for all images (styled in red)
        self.remove_all_button = QPushButton("Remove Scale from ALL Images")
        self.remove_all_button.setToolTip("Removes all scale data from every image in this project.")
        # Style the button to be red as a warning
        self.remove_all_button.setStyleSheet(
            "background-color: #D9534F; color: white; font-weight: bold;"
        )
        self.scale_layout.addRow(self.remove_all_button)

    def setup_line_tab(self, tab_widget):
        """Populates the 'Measure Line' tab."""
        layout = QFormLayout(tab_widget)
        
        self.line_length_label = QLabel("N/A")
        self.line_total_length_label = QLabel("0.0")

        self.line_add_button = QPushButton("Add to Total")
        self.line_clear_button = QPushButton("Clear Total")
        self.line_add_button.setEnabled(False)

        layout.addRow("Current Length:", self.line_length_label)
        layout.addRow("Total Length:", self.line_total_length_label)
        layout.addRow(self.line_add_button)
        layout.addRow(self.line_clear_button)

    def setup_rect_tab(self, tab_widget):
        """Populates the 'Measure Rectangle' tab."""
        layout = QFormLayout(tab_widget)
        
        self.rect_perimeter_label = QLabel("N/A")
        self.rect_area_label = QLabel("N/A")
        self.rect_total_area_label = QLabel("0.0")
        
        self.rect_add_button = QPushButton("Add Area to Total")
        self.rect_clear_button = QPushButton("Clear Total")
        self.rect_add_button.setEnabled(False)

        layout.addRow("Perimeter:", self.rect_perimeter_label)
        layout.addRow("Area:", self.rect_area_label)
        layout.addRow("Total Area:", self.rect_total_area_label)
        layout.addRow(self.rect_add_button)
        layout.addRow(self.rect_clear_button)

    def setup_options_layout(self):
        """
        Set up the layout with image options.
        (Copied and adapted from QtBase.py)
        """
        # Create a group box for image options
        self.options_group_box = QGroupBox("Image Options")
        layout = QVBoxLayout()

        # Create a button group for the image checkboxes
        self.apply_group = QButtonGroup(self)

        self.apply_filtered_checkbox = QCheckBox("▼ Apply to filtered images")
        self.apply_prev_checkbox = QCheckBox("↑ Apply to previous images")
        self.apply_next_checkbox = QCheckBox("↓ Apply to next images")
        self.apply_all_checkbox = QCheckBox("↕ Apply to all images")

        # Add the checkboxes to the button group
        self.apply_group.addButton(self.apply_filtered_checkbox)
        self.apply_group.addButton(self.apply_prev_checkbox)
        self.apply_group.addButton(self.apply_next_checkbox)
        self.apply_group.addButton(self.apply_all_checkbox)

        # Ensure only one checkbox can be checked at a time
        self.apply_group.setExclusive(True)

        # No default checkbox set, defaults to current image

        layout.addWidget(self.apply_filtered_checkbox)
        layout.addWidget(self.apply_prev_checkbox)
        layout.addWidget(self.apply_next_checkbox)
        layout.addWidget(self.apply_all_checkbox)

        self.options_group_box.setLayout(layout)

    def get_selected_image_paths(self):
        """
        Get the selected image paths based on the options.
        (Copied and adapted from QtBase.py)
        
        :return: List of selected image paths
        """
        # Current image path showing
        current_image_path = self.annotation_window.current_image_path
        if not current_image_path:
            return []

        # Determine which images to apply the scale to
        if self.apply_filtered_checkbox.isChecked():
            return self.main_window.image_window.table_model.filtered_paths
        elif self.apply_prev_checkbox.isChecked():
            if current_image_path in self.main_window.image_window.table_model.filtered_paths:
                current_index = self.main_window.image_window.table_model.get_row_for_path(current_image_path)
                return self.main_window.image_window.table_model.filtered_paths[:current_index + 1]
            else:
                return [current_image_path]
        elif self.apply_next_checkbox.isChecked():
            if current_image_path in self.main_window.image_window.table_model.filtered_paths:
                current_index = self.main_window.image_window.table_model.get_row_for_path(current_image_path)
                return self.main_window.image_window.table_model.filtered_paths[current_index:]
            else:
                return [current_image_path]
        elif self.apply_all_checkbox.isChecked():
            return self.main_window.image_window.raster_manager.image_paths
        else:
            # Default to "Apply to current image only"
            return [current_image_path]

    def reset_fields(self):
        """Resets the dialog fields to their default state."""
        # Reset Set Scale Tab
        self.pixel_length_label.setText("Draw a line on the image")
        self.calculated_scale_label.setText("Scale: N/A")
        
        # Reset Line Tab
        self.line_length_label.setText("N/A")
        self.line_total_length_label.setText("0.0")
        self.line_add_button.setEnabled(False)
        
        # Reset Rect Tab
        self.rect_perimeter_label.setText("N/A")
        self.rect_area_label.setText("N/A")
        self.rect_total_area_label.setText("0.0")
        self.rect_add_button.setEnabled(False)

        self.update_checkboxes()

    def update_checkboxes(self):
        """Clear the checkboxes states."""
        # Temporarily disable exclusivity to allow unchecking all checkboxes
        self.apply_group.setExclusive(False)
        self.apply_filtered_checkbox.setChecked(False)
        self.apply_prev_checkbox.setChecked(False)
        self.apply_next_checkbox.setChecked(False)
        self.apply_all_checkbox.setChecked(False)
        # Restore exclusivity
        self.apply_group.setExclusive(True)

    def closeEvent(self, event):
        """
        Handle the dialog close event (e.g., user clicks 'X').
        This should deactivate the tool.
        """
        # Get the tool and call its deactivate method
        # self.tool is ScaleTool
        self.tool.deactivate()
        event.accept()


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
        
        self.dialog.remove_current_button.clicked.connect(self.remove_scale_current)
        self.dialog.remove_all_button.clicked.connect(self.remove_scale_all)

        # --- Drawing State ---
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        self.pixel_length = 0.0
        
        # --- Graphics Items ---
        pen = QPen(QColor(230, 62, 0, 255), 4, Qt.DashLine)
        pen.setCosmetic(True)  # Ensures line is visible at all zoom levels
        
        # Line (for Set Scale and Measure Line)
        self.preview_line = QGraphicsLineItem()
        self.preview_line.setPen(pen)
        self.preview_line.setZValue(100)  # Draw on top
        
        # Rectangle
        self.preview_rect = QGraphicsRectItem()
        self.preview_rect.setPen(pen)
        self.preview_rect.setZValue(100)
        
        self.show_crosshair = True  # Enable crosshair for precise measurements
        
        # --- Accumulation Variables ---
        self.current_line_length = 0.0
        self.total_line_length = 0.0
        
        self.current_rect_area = 0.0
        self.total_rect_area = 0.0
        
        # --- Accumulated Graphics ---
        self.accumulated_lines = []
        self.accumulated_rects = []

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

    def activate(self):
        super().activate()
        # Set initial mode based on the currently selected tab
        self.on_tab_changed(self.dialog.tab_widget.currentIndex())
        
        # Add all preview items to scene
        if not self.preview_line.scene():
            self.annotation_window.scene.addItem(self.preview_line)
        if not self.preview_rect.scene():
            self.annotation_window.scene.addItem(self.preview_rect)
        
        self.stop_current_drawing()  # Resets all drawing
        self.dialog.reset_fields()
        
        # Update status label with current scale
        scale, units = self.get_current_scale()
        if units != "px":
            self.dialog.current_scale_status_label.setText(f"Scale: {scale:.6f} {units}/pixel")
        else:
            self.dialog.current_scale_status_label.setText("Scale: Not Set (units in pixels)")

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
        self.preview_rect.hide()
        
        # Clean up accumulated graphics
        for line in self.accumulated_lines:
            self.annotation_window.scene.removeItem(line)
        self.accumulated_lines.clear()
        for rect in self.accumulated_rects:
            self.annotation_window.scene.removeItem(rect)
        self.accumulated_rects.clear()
        
        self.is_drawing = False
        
        # Untoggle all tools when closing the scale tool
        self.main_window.untoggle_all_tools()

    def on_tab_changed(self, index):
        """Called when the user clicks a different tab."""
        self.current_mode = index
        self.stop_current_drawing()
        
        # Clear accumulated graphics when switching tabs
        for line in self.accumulated_lines:
            self.annotation_window.scene.removeItem(line)
        self.accumulated_lines.clear()
        for rect in self.accumulated_rects:
            self.annotation_window.scene.removeItem(rect)
        self.accumulated_rects.clear()
        
        # Enable "Set Scale" button ONLY on the first tab
        if index == 0:
            self.dialog.set_scale_button.setEnabled(True)
        else:
            self.dialog.set_scale_button.setEnabled(False)

    def stop_current_drawing(self):
        """Force stop all drawing, hide previews, and reset points."""
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        
        self.preview_line.hide()
        self.preview_rect.hide()
        
        # Reset labels
        self.dialog.pixel_length_label.setText("Draw a line on the image")
        self.dialog.line_length_label.setText("N/A")
        self.dialog.rect_perimeter_label.setText("N/A")
        self.dialog.rect_area_label.setText("N/A")
        
        # Reset current measurements
        self.current_line_length = 0.0
        self.current_rect_area = 0.0
        
        # Disable 'Add' buttons
        self.dialog.line_add_button.setEnabled(False)
        self.dialog.rect_add_button.setEnabled(False)

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
                    self.calculate_line_measurement()

        # --- Mode 2: Measure Rectangle ---
        elif self.current_mode == 2:
            if not self.is_drawing:
                # Start drawing
                self.start_point = scene_pos
                self.end_point = self.start_point
                self.is_drawing = True
                self.preview_rect.setRect(QRectF(self.start_point, self.end_point).normalized())
                self.preview_rect.show()
                self.dialog.rect_area_label.setText("Drawing...")
                self.dialog.rect_perimeter_label.setText("Drawing...")
            else:
                # Finish drawing
                self.is_drawing = False
                self.end_point = scene_pos
                self.preview_rect.setRect(QRectF(self.start_point, self.end_point).normalized())
                self.calculate_rect_measurement()

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

        # --- Mode 0: Set Scale OR Mode 1: Measure Line ---
        if self.current_mode == 0 or self.current_mode == 1:
            self.end_point = scene_pos
            line = QLineF(self.start_point, self.end_point)
            self.pixel_length = line.length()
            self.preview_line.setLine(line)
            # Update correct label
            if self.current_mode == 0:
                self.dialog.pixel_length_label.setText(f"{self.pixel_length:.2f} px")
            else:
                # Live update for line length
                scale, units = self.get_current_scale()
                length = self.pixel_length * scale
                self.dialog.line_length_label.setText(f"{length:.3f} {units}")

        # --- Mode 2: Measure Rectangle ---
        elif self.current_mode == 2:
            self.end_point = scene_pos
            self.preview_rect.setRect(QRectF(self.start_point, self.end_point).normalized())
            # Live update for rect
            scale, units = self.get_current_scale()
            area_units = f"{units}²" if units != "px" else "px²"
            rect = QRectF(self.start_point, self.end_point).normalized()
            real_width = rect.width() * scale
            real_height = rect.height() * scale
            perimeter = 2 * (real_width + real_height)
            area = real_width * real_height
            self.dialog.rect_perimeter_label.setText(f"{perimeter:.3f} {units}")
            self.dialog.rect_area_label.setText(f"{area:.3f} {area_units}")

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

    # --- Calculation and Accumulation Methods ---

    def calculate_line_measurement(self):
        """Calculates and displays the length of the drawn line."""
        scale, units = self.get_current_scale()
        
        self.current_line_length = self.pixel_length * scale
        
        self.dialog.line_length_label.setText(f"{self.current_line_length:.3f} {units}")
        self.dialog.line_add_button.setEnabled(True)

    def add_line_to_total(self):
        """Adds the current line length to the total."""
        self.total_line_length += self.current_line_length
        scale, units = self.get_current_scale()
            
        self.dialog.line_total_length_label.setText(f"{self.total_line_length:.3f} {units}")
        
        # Create a permanent line item to keep visible
        perm_line = QGraphicsLineItem(self.preview_line.line())
        perm_line.setPen(self.preview_line.pen())
        perm_line.setZValue(99)  # Slightly below preview
        self.annotation_window.scene.addItem(perm_line)
        self.accumulated_lines.append(perm_line)
        
        self.current_line_length = 0.0  # Reset current
        self.dialog.line_add_button.setEnabled(False)
        self.dialog.line_length_label.setText("N/A")

    def clear_line_total(self):
        self.total_line_length = 0.0
        scale, units = self.get_current_scale()
        self.dialog.line_total_length_label.setText(f"0.0 {units}")
        
        # Remove accumulated lines
        for line in self.accumulated_lines:
            self.annotation_window.scene.removeItem(line)
        self.accumulated_lines.clear()
        
        self.stop_current_drawing()

    def calculate_rect_measurement(self):
        """Calculates and displays rect perimeter and area."""
        scale, units = self.get_current_scale()
        area_units = f"{units}²" if units != "px" else "px²"

        rect = QRectF(self.start_point, self.end_point).normalized()
        pixel_width = rect.width()
        pixel_height = rect.height()

        # We assume square pixels from this tool
        real_width = pixel_width * scale
        real_height = pixel_height * scale
        
        perimeter = 2 * (real_width + real_height)
        area = real_width * real_height
        
        # Store for accumulation
        self.current_rect_area = area 

        self.dialog.rect_perimeter_label.setText(f"{perimeter:.3f} {units}")
        self.dialog.rect_area_label.setText(f"{area:.3f} {area_units}")
        self.dialog.rect_add_button.setEnabled(True)

    def add_rect_to_total(self):
        self.total_rect_area += self.current_rect_area
        scale, units = self.get_current_scale()
        area_units = f"{units}²" if units != "px" else "px²"
        
        self.dialog.rect_total_area_label.setText(f"{self.total_rect_area:.3f} {area_units}")
        
        # Create a permanent rect item to keep visible
        perm_rect = QGraphicsRectItem(self.preview_rect.rect())
        perm_rect.setPen(self.preview_rect.pen())
        perm_rect.setZValue(99)  # Slightly below preview
        self.annotation_window.scene.addItem(perm_rect)
        self.accumulated_rects.append(perm_rect)
        
        self.current_rect_area = 0.0
        self.dialog.rect_add_button.setEnabled(False)
        self.dialog.rect_perimeter_label.setText("N/A")
        self.dialog.rect_area_label.setText("N/A")

    def clear_rect_total(self):
        self.total_rect_area = 0.0
        scale, units = self.get_current_scale()
        area_units = f"{units}²" if units != "px" else "px²"
        self.dialog.rect_total_area_label.setText(f"0.0 {area_units}")
        
        # Remove accumulated rects
        for rect in self.accumulated_rects:
            self.annotation_window.scene.removeItem(rect)
        self.accumulated_rects.clear()
        
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
                    raster.update_scale(new_scale, new_scale, 'metre')
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
        
        # Update the status label
        self.dialog.current_scale_status_label.setText(f"Scale: {new_scale:.6f} m/pixel")

        # Refresh the main window status bar if the current image was updated
        if current_path in target_image_paths:
            # Re-emit the signal to update scaled dimensions
            self.main_window.update_view_dimensions(current_raster.width, current_raster.height)

        QMessageBox.information(self.dialog, 
                                "Success",
                                f"Successfully applied new scale ({scale_text}) "
                                f"to {success_count} image(s).")
        
    def remove_scale_current(self):
        """Removes scale from the currently loaded image."""
        current_path = self.annotation_window.current_image_path
        if not current_path:
            QMessageBox.warning(self.dialog, "No Image", "No image is currently loaded.")
            return

        raster = self.main_window.image_window.raster_manager.get_raster(current_path)
        if not raster or raster.scale_x is None:
            QMessageBox.information(self.dialog, "No Scale", "The current image has no scale data to remove.")
            return

        # Warn the user
        reply = QMessageBox.question(self.dialog,
                                     "Confirm Removal",
                                     "Are you sure you want to remove the scale from the current image?\n"
                                     "This cannot be undone.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            # 1. Remove scale from Raster
            raster.remove_scale()
            
            # 2. Update all associated annotations
            self.annotation_window.update_annotations_scale(current_path)
            
            # 3. Update UI
            self.dialog.current_scale_status_label.setText("Scale: Not Set (units in pixels)")
            self.main_window.update_view_dimensions(raster.width, raster.height)
            QMessageBox.information(self.dialog, "Success", "Scale removed from the current image.")

    def remove_scale_all(self):
        """Removes scale from ALL images in the project."""
        all_paths = self.main_window.image_window.raster_manager.image_paths
        if not all_paths:
            QMessageBox.warning(self.dialog, "No Images", "There are no images in the project.")
            return

        # CRITICAL warning for the user
        reply = QMessageBox.warning(self.dialog,
                                    "Confirm Global Removal",
                                    "ARE YOU SURE?\n\n"
                                    "This will remove scale data from ALL images in this project.\n"
                                    "This action cannot be undone.",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            progress_bar = ProgressBar(self.annotation_window, title="Removing All Scale Data")
            progress_bar.show()
            progress_bar.start_progress(len(all_paths))

            current_image_was_updated = False
            
            try:
                for path in all_paths:
                    if progress_bar.wasCanceled():
                        break
                    
                    raster = self.main_window.image_window.raster_manager.get_raster(path)
                    if raster and raster.scale_x is not None:
                        # 1. Remove scale from Raster
                        raster.remove_scale()
                        
                        # 2. Update all associated annotations
                        self.annotation_window.update_annotations_scale(path)

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
                self.dialog.current_scale_status_label.setText("Scale: Not Set (units in pixels)")
                self.main_window.update_view_dimensions(raster.width, raster.height)

            QMessageBox.information(self.dialog, "Success", "Scale data has been removed from all images.")