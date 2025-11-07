import warnings

from PyQt5.QtCore import Qt, QLineF
from PyQt5.QtGui import QMouseEvent, QPen, QColor
from PyQt5.QtWidgets import (QApplication, QDialog, QWidget, QVBoxLayout, QTabWidget,
                             QFormLayout, QDoubleSpinBox, QComboBox, QLabel,
                             QDialogButtonBox, QMessageBox, QGraphicsLineItem,
                             QGroupBox, QCheckBox, QButtonGroup)

from coralnet_toolbox.Tools.QtTool import Tool
from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.utilities import convert_scale_units

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
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
        self.setMinimumWidth(350)
        
        # This dialog is modeless
        self.setModal(False) 

        self.main_layout = QVBoxLayout(self)

        # --- Tab Widget for future extensibility ---
        self.tab_widget = QTabWidget()
        self.scale_tab = QWidget()
        self.tab_widget.addTab(self.scale_tab, "Set Scale")
        self.main_layout.addWidget(self.tab_widget)

        # --- Set Scale Tab UI ---
        self.scale_layout = QFormLayout(self.scale_tab)
        
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

        # --- Image Options (copied from QtBase.py) ---
        self.setup_options_layout()
        self.main_layout.addWidget(self.options_group_box)

        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Close)
        # Connections are made in the ScaleTool class
        self.main_layout.addWidget(self.button_box)

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
        self.pixel_length_label.setText("Draw a line on the image")
        self.calculated_scale_label.setText("Scale: N/A")
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
    Tool for setting the image scale by drawing a line of a known length.
    """
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        self.cursor = Qt.CrossCursor
        self.dialog = ScaleToolDialog(self, self.annotation_window)  # tool, parent
        
        # Connect dialog signals
        apply_btn = self.dialog.button_box.button(QDialogButtonBox.Apply)
        apply_btn.clicked.connect(self.apply_scale)
        
        close_btn = self.dialog.button_box.button(QDialogButtonBox.Close)
        close_btn.clicked.connect(self.deactivate)  # Deactivate tool when user clicks "Close"

        # Drawing state
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        self.pixel_length = 0.0
        
        # Graphics item for the preview line
        self.preview_line = QGraphicsLineItem()
        pen = QPen(QColor(230, 62, 0, 255), 2, Qt.DashLine)
        pen.setCosmetic(True)  # Ensures line is visible at all zoom levels
        self.preview_line.setPen(pen)
        self.preview_line.setZValue(100)  # Draw on top
        
        self.show_crosshair = False  # Disable default crosshair

    def activate(self):
        super().activate()
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        self.pixel_length = 0.0
        
        self.dialog.reset_fields()
        
        # Add preview line to scene if not already added
        if not self.preview_line.scene():
            self.annotation_window.scene.addItem(self.preview_line)
        self.preview_line.hide()

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
        self.is_drawing = False
        
        # Untoggle all tools when closing the scale tool
        self.main_window.untoggle_all_tools()

    def stop_current_drawing(self):
        """Force stop of current line drawing if in progress."""
        if self.is_drawing:
            self.is_drawing = False
        self.preview_line.hide()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            if not self.annotation_window.cursorInWindow(event.pos()):
                return
            
            if not self.is_drawing:
                # Start drawing the line
                self.start_point = self.annotation_window.mapToScene(event.pos())
                self.end_point = self.start_point
                self.is_drawing = True
                
                self.preview_line.setLine(QLineF(self.start_point, self.end_point))
                self.preview_line.show()
                self.dialog.pixel_length_label.setText("Drawing...")
            else:
                # Finish drawing the line
                self.is_drawing = False
                self.end_point = self.annotation_window.mapToScene(event.pos())
                line = QLineF(self.start_point, self.end_point)
                self.pixel_length = line.length()
                
                self.preview_line.setLine(line)
                self.dialog.pixel_length_label.setText(f"{self.pixel_length:.2f} px")
                
                if self.pixel_length == 0.0:
                    self.start_point = None
                    self.end_point = None
                    self.preview_line.hide()
                    self.dialog.pixel_length_label.setText("Draw a line on the image")

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.is_drawing:
            self.end_point = self.annotation_window.mapToScene(event.pos())
            line = QLineF(self.start_point, self.end_point)
            self.pixel_length = line.length()
            
            self.preview_line.setLine(line)
            self.dialog.pixel_length_label.setText(f"{self.pixel_length:.2f} px")

    def mouseReleaseEvent(self, event: QMouseEvent):
        pass

    def keyPressEvent(self, event):
        """Handle key press events for canceling drawing."""
        if self.is_drawing and event.key() == Qt.Key_Backspace:
            self.stop_current_drawing()
            self.start_point = None
            self.end_point = None
            self.pixel_length = 0.0
            self.dialog.pixel_length_label.setText("Draw a line on the image")

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

        # Refresh the main window status bar if the current image was updated
        if current_path in target_image_paths:
            # Re-emit the signal to update scaled dimensions
            self.main_window.update_view_dimensions(current_raster.width, current_raster.height)

        QMessageBox.information(self.dialog, 
                                "Success",
                                f"Successfully applied new scale ({scale_text}) "
                                f"to {success_count} image(s).")
