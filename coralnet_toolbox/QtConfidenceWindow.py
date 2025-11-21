import os
import warnings

from PyQt5.QtGui import QPixmap, QColor, QPainter, QCursor
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPropertyAnimation, QEasingCurve, pyqtProperty, QTimer
from PyQt5.QtWidgets import (QGraphicsView, QGraphicsScene, QWidget, QVBoxLayout, QSizePolicy,
                             QLabel, QHBoxLayout, QFrame, QGroupBox, QPushButton, QStyle)

from coralnet_toolbox.utilities import scale_pixmap
from coralnet_toolbox.utilities import convert_scale_units

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ConfidenceBar(QFrame):
    barClicked = pyqtSignal(object)  # Define a signal that takes an object (label)

    def __init__(self, confidence_window, label, confidence, parent=None):
        """Initialize the ConfidenceBar widget."""
        super().__init__(parent)
        self.confidence_window = confidence_window

        self.label = label
        self.confidence = confidence
        self.color = label.color
        self.setFixedHeight(20)  # Set a fixed height for the bars

        self._fill_width = 0
        self.target_fill_width = 0  # Will be set in resizeEvent

        # Animation will be created and started in the first resizeEvent
        self.animation = None

    def get_fill_width(self):
        """Getter for the fill_width property used by the animation."""
        return self._fill_width

    def set_fill_width(self, value):
        """Setter for the fill_width property used by the animation."""
        self._fill_width = value
        self.update()  # Trigger a repaint whenever the value changes

    # This property allows QPropertyAnimation to animate the fill width
    fill_width = pyqtProperty(int, fget=get_fill_width, fset=set_fill_width)

    def resizeEvent(self, event):
        """Handle resize to recalculate target fill width and start animation."""
        super().resizeEvent(event)
        # Calculate the target fill width based on the current widget width and confidence
        self.target_fill_width = int(self.width() * (self.confidence / 100))
        
        # Stop any existing animation
        if self.animation is not None:
            self.animation.stop()
            
        # Start animation from current position to target
        self.start_animation()

    def start_animation(self):
        """Start the fill animation."""
        if self.target_fill_width <= 0:
            # Stop any existing animation
            if self.animation is not None:
                self.animation.stop()
            
            # Explicitly set the fill width to 0 and trigger a repaint
            self._fill_width = 0
            self.update()
            return
            
        self.animation = QPropertyAnimation(self, b"fill_width")
        self.animation.setDuration(500)  # 500ms duration
        self.animation.setStartValue(0)  # Note: This could be self._fill_width for a smoother resume
        self.animation.setEndValue(self.target_fill_width)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)  # Smooth easing
        self.animation.start()

    def paintEvent(self, event):
        """Handle the paint event to draw the confidence bar."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        current_width = self.width()
        current_height = self.height()

        # Guard against painting on a widget with no size
        if current_width < 1 or current_height < 1:
            return

        # Draw the border for the entire bar area
        painter.setPen(self.color)
        painter.drawRect(0, 0, current_width - 1, current_height - 1)

        # Draw the filled part of the bar
        painter.setBrush(QColor(self.color.red(), self.color.green(), self.color.blue(), 192))
        
        # Clamp the _fill_width to be valid and within the widget's bounds
        fill_w = min(self._fill_width, current_width - 1)
        
        if fill_w > 0:
            painter.drawRect(0, 0, fill_w, current_height - 1)

    def mousePressEvent(self, event):
        """Handle mouse press events on the bar."""
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            # self.handle_click() # <-- DO NOT CALL DIRECTLY

            # Defer the click handling. This lets the mousePressEvent finish
            # before the widget is potentially deleted by the click's action.
            QTimer.singleShot(0, self.handle_click)

    def handle_click(self):
        """Handle the logic when the bar is clicked."""
        # Check if the Selector tool is active
        if self.confidence_window.main_window.annotation_window.selected_tool == "select":
            # Emit the signal with the label object
            self.barClicked.emit(self.label)
            # Set focus to the confidence window for keyboard events
            self.confidence_window.setFocus()

    def enterEvent(self, event):
        """Handle mouse enter events to change the cursor."""
        super().enterEvent(event)
        # Change cursor based on the active tool
        if self.confidence_window.main_window.annotation_window.selected_tool == "select":
            self.setCursor(QCursor(Qt.PointingHandCursor))
        else:
            self.setCursor(QCursor(Qt.ForbiddenCursor))

    def leaveEvent(self, event):
        """Handle mouse leave events to reset the cursor."""
        super().leaveEvent(event)
        self.setCursor(QCursor(Qt.ArrowCursor))  # Reset to the default cursor


class ConfidenceWindow(QWidget):
    def __init__(self, main_window, parent=None):
        """Initialize the ConfidenceWindow widget."""
        super().__init__(parent)
        self.main_window = main_window
        self.label_window = main_window.label_window

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Create a groupbox and set its title
        self.groupBox = QGroupBox("Confidence Window")
        self.groupBoxLayout = QVBoxLayout()
        self.groupBox.setLayout(self.groupBoxLayout)

        self.max_graphic_size = 256
        self.graphics_view = None
        self.scene = None

        self.bar_chart_widget = None
        self.bar_chart_layout = None

        self.init_graphics_view()

        self.annotation = None
        self.user_confidence = None
        self.machine_confidence = None
        self.chart_dict = None
        self.confidence_bar_labels = []
        
        # Get and store the icons
        self.user_icon = get_icon("user.png")
        self.machine_icon = get_icon("machine.png")
        self.prev_icon = self.style().standardIcon(QStyle.SP_ArrowLeft)
        self.next_icon = self.style().standardIcon(QStyle.SP_ArrowRight)
        
        self.top_k_icons = {
            "1": get_icon("1.png").pixmap(12, 12),
            "2": get_icon("2.png").pixmap(12, 12),
            "3": get_icon("3.png").pixmap(12, 12),
            "4": get_icon("4.png").pixmap(12, 12),
            "5": get_icon("5.png").pixmap(12, 12)
        }

        # --- Graphics View Controls Layout ---
        
        # Create navigation buttons for previous/next annotation
        self.prev_button = QPushButton(self.prev_icon, " Prev")
        self.prev_button.setToolTip("Select an annotation to enable navigation")
        self.prev_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.prev_button.clicked.connect(self.on_prev_clicked)
        
        self.next_button = QPushButton(self.next_icon, " Next")
        self.next_button.setToolTip("Select an annotation to enable navigation")
        self.next_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.next_button.clicked.connect(self.on_next_clicked)
        
        nav_layout = QHBoxLayout()
        nav_layout.addStretch()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        nav_layout.addStretch()
        
        self.groupBoxLayout.addLayout(nav_layout)
        
        # --- End Graphics View Controls ---

        # Initialize the bar chart (adds it to the layout)
        self.init_bar_chart_widget()

        # Create a label for the dimensions and a toggle button (NOW AT THE BOTTOM)
        self.dimensions_label = QLabel(self)
        self.dimensions_label.setAlignment(Qt.AlignCenter)

        self.toggle_button = QPushButton(self)
        self.toggle_button.setFixedSize(24, 24)
        self.toggle_state = False  # False = user, True = machine
        self.toggle_button.setIcon(get_icon("user.png"))
        self.toggle_button.clicked.connect(self.toggle_user_machine_confidence_icon)
        self.set_user_icon(False)  # Set to disabled user mode by default

        dim_layout = QHBoxLayout()
        dim_layout.addWidget(self.dimensions_label)
        dim_layout.addWidget(self.toggle_button)
        self.groupBoxLayout.addLayout(dim_layout)  # Add layout to the bottom

        # Add the groupbox to the main layout
        self.layout.addWidget(self.groupBox)
        
        # Set initial state of buttons
        self.set_navigation_enabled(False)
        
    def resizeEvent(self, event):
        """Handle resize events for the widget."""
        super().resizeEvent(event)
        self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        
    def keyPressEvent(self, event):
        """Handle key press events for 1-5 to select a confidence bar."""
        key = event.key()
        if Qt.Key_1 <= key <= Qt.Key_5:
            idx = (key - Qt.Key_1)  # 0-based index
            if hasattr(self, "confidence_bar_labels") and idx < len(self.confidence_bar_labels):
                label = self.confidence_bar_labels[idx]
                self.handle_bar_click(label)
        else:
            super().keyPressEvent(event)
            
    def set_navigation_enabled(self, enabled):
        """Enable or disable annotation navigation buttons."""
        self.prev_button.setEnabled(enabled)
        self.next_button.setEnabled(enabled)
        
        if not enabled:
            self.prev_button.setToolTip("Select an annotation to enable navigation")
            self.next_button.setToolTip("Select an annotation to enable navigation")
        else:
            self.prev_button.setToolTip("Previous Annotation")
            self.next_button.setToolTip("Next Annotation")

    def on_prev_clicked(self):
        """Handle previous button click."""
        self.main_window.annotation_window.cycle_annotations(-1)
        self.setFocus()

    def on_next_clicked(self):
        """Handle next button click."""
        self.main_window.annotation_window.cycle_annotations(1)
        self.setFocus()

    def init_graphics_view(self):
        """Initialize the graphics view for displaying the cropped image."""
        self.graphics_view = QGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.scene)
        self.groupBoxLayout.addWidget(self.graphics_view, 2)  # 2 for stretch factor

    def init_bar_chart_widget(self):
        """Initialize the widget and layout for the confidence bar chart."""
        self.bar_chart_widget = QWidget()
        self.bar_chart_layout = QVBoxLayout(self.bar_chart_widget)
        self.bar_chart_layout.setContentsMargins(0, 0, 0, 0)
        self.bar_chart_layout.setSpacing(2)  # Set spacing to make bars closer
        self.groupBoxLayout.addWidget(self.bar_chart_widget, 1)  # 1 for stretch factor
        
    def toggle_user_machine_confidence_icon(self):
        """Toggle the button icon and switch between user/machine confidences."""
        if not (self.user_confidence and self.machine_confidence):
            return  # Nothing to toggle

        self.toggle_state = not self.toggle_state
        if self.toggle_state:
            self.chart_dict = self.machine_confidence
            self.set_machine_icon(enabled=True)
        else:
            self.chart_dict = self.user_confidence
            self.set_user_icon(enabled=True)
        self.create_bar_chart()
            
    def set_user_icon(self, enabled=True):
        """Set the button icon to user mode."""
        self.toggle_button.setIcon(self.user_icon)
        self.toggle_button.setToolTip("Viewing User Confidences")
        self.toggle_button.setEnabled(enabled)
        self.toggle_state = False
        
    def set_machine_icon(self, enabled=True):
        """Set the button icon to machine mode."""
        self.toggle_button.setIcon(self.machine_icon)
        self.toggle_button.setToolTip("Viewing Machine Confidences")
        self.toggle_button.setEnabled(enabled)
        self.toggle_state = True

    def update_blank_pixmap(self):
        """Update the graphics view with a blank transparent pixmap."""
        view_size = self.graphics_view.size()
        new_pixmap = QPixmap(view_size)
        new_pixmap.fill(Qt.transparent)
        self.scene.clear()
        self.scene.addPixmap(new_pixmap)

    def update_annotation(self, annotation):
        """Update the currently displayed annotation data."""
        if annotation:
            self.annotation = annotation
            self.user_confidence = annotation.user_confidence
            self.machine_confidence = annotation.machine_confidence
            
            # Annotation is verified and contains machine confidences
            if annotation.verified and self.machine_confidence:
                self.chart_dict = self.user_confidence
                self.set_user_icon(annotation.verified)         # enabled user icon
            
            # Annotation is not verified and contains machine confidences
            elif not annotation.verified and self.machine_confidence:
                self.chart_dict = self.machine_confidence
                self.set_machine_icon(annotation.verified)      # disabled machine icon
                
            # Annotation is verified and does not contain machine confidences
            elif annotation.verified and not self.machine_confidence:
                self.chart_dict = self.user_confidence
                self.set_user_icon(not annotation.verified)     # disabled user icon
        
        else:
            self.set_user_icon(False)  # Disable user icon if no annotation is provided
            
    def refresh_display(self):
        """Refresh the confidence window display for the current annotation."""
        if self.annotation:
            # Update annotation data
            self.update_annotation(self.annotation)
            
            # Recreate the bar chart with updated data
            self.create_bar_chart()
            
            # Recreate the tooltip to reflect any data changes (like scale)
            self.create_annotation_tooltip(self.annotation)
            
            # Update the graphics view border color based on top confidence
            if self.chart_dict:
                labels, confidences = self.get_chart_data()
                if labels and confidences:
                    max_color = labels[confidences.index(max(confidences))].color
                    self.graphics_view.setStyleSheet(f"border: 2px solid {max_color.name()};")
    
    def on_annotation_updated(self, updated_annotation):
        """Handle annotation update signal - refresh display if it's the currently shown annotation."""
        if self.annotation and updated_annotation.id == self.annotation.id:
            self.refresh_display()

    def display_cropped_image(self, annotation):
        """Display the cropped image and update the bar chart."""
        try:
            self.clear_display()
            self.update_annotation(annotation)
            if self.annotation.cropped_image:
                # Get the cropped image graphic
                cropped_image_graphic = scale_pixmap(annotation.get_cropped_image_graphic(), self.max_graphic_size)
                # Add the scaled annotation graphic (as pixmap)
                self.scene.addPixmap(cropped_image_graphic)
                # Add the border color with increased width
                self.scene.setSceneRect(QRectF(cropped_image_graphic.rect()))
                self.graphics_view.setStyleSheet("QGraphicsView { border: 3px solid transparent; }")
                # Fit the view to the scene
                self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
                self.graphics_view.centerOn(self.scene.sceneRect().center())
                
                # Create tooltip with annotation information
                self.create_annotation_tooltip(annotation)
                
                # Create the bar charts
                self.create_bar_chart()

                # Update dimensions label with original and scaled dimensions
                orig_height = annotation.get_cropped_image().height()
                orig_width = annotation.get_cropped_image().width()
                scaled_height = cropped_image_graphic.height()
                scaled_width = cropped_image_graphic.width()
                
                if orig_height != scaled_height:
                    text = f"Original: {orig_height} x {orig_width} → Scaled: {scaled_height} x {scaled_width}"
                    self.dimensions_label.setText(text)
                else:
                    self.dimensions_label.setText(f"Crop: {orig_height} x {orig_width}")

                # Enable navigation buttons
                self.set_navigation_enabled(True)

        except Exception as e:
            # Cropped image is None or some other error occurred
            print(f"Error displaying cropped image: {e}")
            # Ensure buttons are disabled if loading fails
            self.set_navigation_enabled(False)

    def create_annotation_tooltip(self, annotation):
        """Create a formatted tooltip for the annotation displayed in the graphics view."""
        tooltip_parts = []
        
        # Annotation ID
        tooltip_parts.append(f"<b>Annotation ID:</b> {annotation.id}")
        
        # Label information
        if annotation.label:
            tooltip_parts.append(f"<b>Label:</b> {annotation.label.short_label_code}")
            if annotation.label.long_label_code != annotation.label.short_label_code:
                tooltip_parts.append(f"<b>Full Name:</b> {annotation.label.long_label_code}")
        
        # Confidence information
        if annotation.user_confidence:
            # Get the label with highest confidence
            top_label = max(annotation.user_confidence.keys(), key=lambda k: annotation.user_confidence[k])
            top_confidence = annotation.user_confidence[top_label] * 100
            tooltip_parts.append(f"<b>User Confidence:</b> {top_confidence:.1f}% ({top_label.short_label_code})")
        
        if annotation.machine_confidence:
            # Get the label with highest confidence
            top_label = max(annotation.machine_confidence.keys(), key=lambda k: annotation.machine_confidence[k])
            top_confidence = annotation.machine_confidence[top_label] * 100
            tooltip_parts.append(f"<b>Machine Confidence:</b> {top_confidence:.1f}% ({top_label.short_label_code})")
        
        # Verification status
        tooltip_parts.append(f"<b>Verified:</b> {'Yes' if annotation.verified else 'No'}")
        
        # Image path
        if annotation.image_path:
            tooltip_parts.append(f"<b>Source Image:</b> {os.path.basename(annotation.image_path)}")
        
        # Cropped image dimensions
        if annotation.cropped_image:
            width = annotation.cropped_image.width()
            height = annotation.cropped_image.height()
            tooltip_parts.append(f"<b>Cropped Dimensions:</b> {width} x {height}")
            
        # Area
        try:
            # Check for new scaled method first
            scaled_area_data = annotation.get_scaled_area()
            if scaled_area_data:
                base_area_value, base_linear_unit = scaled_area_data
                
                # Get the target unit from MainWindow's dropdown
                target_unit = self.main_window.current_unit_scale
                
                # Get the linear conversion factor (e.g., 1 'metre' to 'cm' = 100)
                linear_conv_factor = convert_scale_units(1.0, base_linear_unit, target_unit)
                # Area factor is the square of the linear factor
                area_conv_factor = linear_conv_factor * linear_conv_factor
                
                # Calculate the final area in the target units
                converted_area = base_area_value * area_conv_factor
                
                tooltip_parts.append(f"<b>Area:</b> {converted_area:.2f} {target_unit}²")
            else:
                # Fallback to pixel area
                area = annotation.get_area()
                if area is not None:
                    tooltip_parts.append(f"<b>Area:</b> {area:.2f} pixels²")
        except (NotImplementedError, AttributeError):
            pass  # No area method available
        
        # Perimeter
        try:
            # Check for new scaled method first
            scaled_perimeter_data = annotation.get_scaled_perimeter()
            if scaled_perimeter_data:
                base_perimeter_value, base_linear_unit = scaled_perimeter_data
                
                # Get the target unit from MainWindow's dropdown
                target_unit = self.main_window.current_unit_scale
                
                # Convert the perimeter value (linear)
                converted_perimeter = convert_scale_units(base_perimeter_value, base_linear_unit, target_unit)
                
                tooltip_parts.append(f"<b>Perimeter:</b> {converted_perimeter:.2f} {target_unit}")
            else:
                # Fallback to pixel perimeter
                perimeter = annotation.get_perimeter()
                if perimeter is not None:
                    tooltip_parts.append(f"<b>Perimeter:</b> {perimeter:.2f} pixels")
        except (NotImplementedError, AttributeError):
            pass  # No perimeter method available
                
        # Get the raster to access z_channel and scale
        raster = self.main_window.image_window.raster_manager.get_raster(annotation.image_path)
        
        if raster:
            # Lazily load the z_channel
            z_channel = raster.z_channel_lazy
            scale_x = raster.scale_x
            scale_y = raster.scale_y
            scale_units = raster.scale_units
            z_unit = raster.z_unit
            
            # Check if all required data is available
            if z_channel is not None and scale_x is not None and scale_y is not None and scale_units is not None:
                try:
                    # --- Volume Calculation ---
                    # Pass z_unit to ensure proper unit conversion in the calculation
                    volume = annotation.get_scaled_volume(z_channel, scale_x, scale_y, z_unit)
                    if volume is not None:
                        # Volume is now in cubic meters
                        vol_units = f"{scale_units}² · m"
                        tooltip_parts.append(f"<b>Volume:</b> {volume:.2f} {vol_units}")
                    
                    # --- 3D Surface Area Calculation ---
                    # Pass z_unit to ensure proper unit conversion in the calculation
                    surface_area = annotation.get_scaled_surface_area(z_channel, scale_x, scale_y, z_unit)
                    if surface_area is not None:
                        # Surface area is now in square meters
                        tooltip_parts.append(f"<b>3D Surface Area:</b> {surface_area:.2f} {scale_units}²")
                
                except Exception as e:
                    print(f"Error calculating 3D metrics for tooltip: {e}")
                    # Don't add to tooltip if calculation fails
                
        # Additional data
        if hasattr(annotation, 'data') and annotation.data:
            data_items = []
            for key, value in annotation.data.items():
                data_items.append(f"<li><b>{key}:</b> {value}</li>")
            if data_items:
                tooltip_parts.append(f"<b>Additional Data:</b><ul>{''.join(data_items)}</ul>")
        
        # Set the tooltip
        tooltip_text = "<br>".join(tooltip_parts)
        self.graphics_view.setToolTip(tooltip_text)

    def create_bar_chart(self):
        """Create and populate the confidence bar chart."""
        self.clear_layout(self.bar_chart_layout)
        self.confidence_bar_labels = []

        if not self.chart_dict:
            return

        labels, confidences = self.get_chart_data()
        if not confidences:
            return

        # Find the highest confidence value for border color
        max_confidence = max(confidences) if confidences else 0

        # Set border color based on the top prediction
        max_color = labels[confidences.index(max_confidence)].color
        self.graphics_view.setStyleSheet(f"border: 2px solid {max_color.name()};")

        # Use actual confidence values for both bar fill and display
        for idx, (label, confidence) in enumerate(zip(labels, confidences)):
            # Use the actual confidence for both display and bar fill
            self.add_bar_to_layout(label, confidence, confidence, idx + 1)
            self.confidence_bar_labels.append(label)

    def get_chart_data(self):
        """Retrieve the top 5 labels and confidences from the current chart dictionary."""
        keys = list(self.chart_dict.keys())[:5]
        return (
            keys,
            [conf_value * 100 for conf_value in self.chart_dict.values()][:5]
        )

    def add_bar_to_layout(self, label, display_confidence, bar_confidence, top_k):
        """Create and add a composite widget for the confidence bar to the layout."""
        # 1. Create a container widget for the entire row
        container_widget = QWidget()
        row_layout = QHBoxLayout(container_widget)
        row_layout.setContentsMargins(5, 2, 5, 2)
        row_layout.setSpacing(5)

        # 2. Create the individual components
        icon_label = QLabel()
        icon_label.setPixmap(self.top_k_icons[str(top_k)])
        icon_label.setFixedSize(14, 14)

        # Truncate the label text to 10 characters and add "..." if longer
        label_text = label.short_label_code
        if len(label_text) > 10:
            label_text = label_text[:10] + "..."
        
        class_label = QLabel(label_text)
        class_label.setFixedWidth(80)
        class_label.setToolTip(label.short_label_code)  # Show full text on hover

        # Use the actual confidence value for the bar's visual fill
        bar_widget = ConfidenceBar(self, label, bar_confidence)
        bar_widget.barClicked.connect(self.handle_bar_click)

        # Use the actual confidence value for the text display
        percentage_label = QLabel(f"{display_confidence:.2f}%")
        percentage_label.setFixedWidth(55)
        percentage_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # 3. Add the components to the row's layout
        row_layout.addWidget(icon_label)
        row_layout.addWidget(class_label)
        row_layout.addWidget(bar_widget, 1)  # Add bar_widget with stretch factor
        row_layout.addWidget(percentage_label)

        # 4. Add the container for the whole row to the main vertical layout
        self.bar_chart_layout.addWidget(container_widget)

    def handle_bar_click(self, label):
        """Handle clicks on a confidence bar to update the annotation."""
        # Guard clause: If no annotation is selected, do nothing.
        if not self.annotation:
            return

        # Store a local reference to the annotation.
        # This is crucial because unselect_annotation() will call clear_display()
        # and set self.annotation to None.
        annotation_to_update = self.annotation

        # Update the confidences to whichever bar was selected
        annotation_to_update.update_user_confidence(label)
        # Update the label to whichever bar was selected
        annotation_to_update.update_label(label)
        
        # Update the search bars
        self.main_window.image_window.update_search_bars()
        
        # Update everything else (essentially)
        # This next line will set self.annotation to None via clear_display()
        self.main_window.annotation_window.unselect_annotation(annotation_to_update)
        
        # Reselect the annotation using our saved local reference
        self.main_window.annotation_window.select_annotation(annotation_to_update)

    def clear_layout(self, layout):
        """Remove all widgets from the specified layout."""
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)

    def clear_display(self):
        """
        Clears the current scene and bar chart layout.
        """
        # Clear the scene
        self.scene.clear()
        # Clear the bar chart layout
        self.clear_layout(self.bar_chart_layout)
        # Reset the style sheet to default
        self.graphics_view.setStyleSheet("")
        # Clear the dimensions label
        self.dimensions_label.setText("")
        # Clear the tooltip
        self.graphics_view.setToolTip("")
        # Set the toggle button to user mode
        self.set_user_icon(False)
        # Disable navigation buttons
        self.set_navigation_enabled(False)
        # Clear the annotation reference
        self.annotation = None