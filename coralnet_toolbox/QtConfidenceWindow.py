import os
import warnings

from PyQt5.QtGui import QPixmap, QColor, QPainter, QCursor
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt5.QtWidgets import (QGraphicsView, QGraphicsScene, QWidget, QVBoxLayout,
                             QLabel, QHBoxLayout, QFrame, QGroupBox, QPushButton)

from coralnet_toolbox.Icons import get_icon
from coralnet_toolbox.utilities import scale_pixmap

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
            return
            
        self.animation = QPropertyAnimation(self, b"fill_width")
        self.animation.setDuration(500)  # 500ms duration
        self.animation.setStartValue(0)
        self.animation.setEndValue(self.target_fill_width)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)  # Smooth easing
        self.animation.start()

    def paintEvent(self, event):
        """Handle the paint event to draw the confidence bar."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw the border for the entire bar area
        painter.setPen(self.color)
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

        # Draw the filled part of the bar from left to the current fill_width
        painter.setBrush(QColor(self.color.red(), self.color.green(), self.color.blue(), 192))
        painter.drawRect(0, 0, self._fill_width, self.height() - 1)

    def mousePressEvent(self, event):
        """Handle mouse press events on the bar."""
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            self.handle_click()

    def handle_click(self):
        """Handle the logic when the bar is clicked."""
        # Check if the Selector tool is active
        if self.confidence_window.main_window.annotation_window.selected_tool == "select":
            # Emit the signal with the label object
            self.barClicked.emit(self.label)

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
        self.init_bar_chart_widget()

        self.annotation = None
        self.user_confidence = None
        self.machine_confidence = None
        self.chart_dict = None
        self.confidence_bar_labels = []
        
        # Get and store the icons
        self.user_icon = get_icon("user.png")
        self.machine_icon = get_icon("machine.png")
        
        self.top_k_icons = {
            "1": get_icon("1.png").pixmap(12, 12),
            "2": get_icon("2.png").pixmap(12, 12),
            "3": get_icon("3.png").pixmap(12, 12),
            "4": get_icon("4.png").pixmap(12, 12),
            "5": get_icon("5.png").pixmap(12, 12)
        }

        # Create a label for the dimensions and a toggle button
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
        self.groupBoxLayout.addLayout(dim_layout)

        # Add the groupbox to the main layout
        self.layout.addWidget(self.groupBox)
        
    def resizeEvent(self, event):
        """Handle resize events for the widget."""
        super().resizeEvent(event)
        self.update_blank_pixmap()
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

    def init_graphics_view(self):
        """Initialize the graphics view for displaying the cropped image."""
        self.graphics_view = QGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.scene)
        self.groupBoxLayout.addWidget(self.graphics_view, 2)  # 2 for stretch factor
        self.update_blank_pixmap()

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

        except Exception as e:
            # Cropped image is None or some other error occurred
            print(f"Error displaying cropped image: {e}")

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
        
        # Area and perimeter
        try:
            area = annotation.get_area()
            if area is not None:
                tooltip_parts.append(f"<b>Area:</b> {area:.2f} pixels²")
        except (NotImplementedError, AttributeError):
            pass
        
        try:
            perimeter = annotation.get_perimeter()
            if perimeter is not None:
                tooltip_parts.append(f"<b>Perimeter:</b> {perimeter:.2f} pixels")
        except (NotImplementedError, AttributeError):
            pass
        
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

        # Calculate the sum of all displayed confidences for relative scaling
        total_displayed_confidence = sum(confidences)
        
        # Find the highest confidence value for border color
        max_confidence = max(confidences) if confidences else 0

        # Set border color based on the top prediction
        max_color = labels[confidences.index(max_confidence)].color
        self.graphics_view.setStyleSheet(f"border: 2px solid {max_color.name()};")

        # Use relative confidence values for bar fill, but original values for display
        for idx, (label, confidence) in enumerate(zip(labels, confidences)):
            # Calculate relative confidence for bar fill (as percentage of total displayed)
            if total_displayed_confidence > 0:
                relative_confidence = (confidence / total_displayed_confidence) * 100
            else:
                relative_confidence = 0
            
            # Use original confidence for display text, relative for bar fill
            self.add_bar_to_layout(label, confidence, relative_confidence, idx + 1)
            self.confidence_bar_labels.append(label)

    def get_chart_data(self):
        """Retrieve the top 5 labels and confidences from the current chart dictionary."""
        keys = list(self.chart_dict.keys())[:5]
        return (
            keys,
            [conf_value * 100 for conf_value in self.chart_dict.values()][:5]
        )

    def add_bar_to_layout(self, label, original_confidence, bar_confidence, top_k):
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

        class_label = QLabel(label.short_label_code)
        class_label.setFixedWidth(80)

        # Use the actual confidence value for the bar's visual fill
        bar_widget = ConfidenceBar(self, label, bar_confidence)
        bar_widget.barClicked.connect(self.handle_bar_click)

        # Use the actual confidence value for the text display
        percentage_label = QLabel(f"{original_confidence:.2f}%")
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
        # Update the confidences to whichever bar was selected
        self.annotation.update_user_confidence(label)
        # Update the label to whichever bar was selected
        self.annotation.update_label(label)
        # Update the search bars
        self.main_window.image_window.update_search_bars()
        # Update everything else (essentially)
        self.main_window.annotation_window.unselect_annotation(self.annotation)
        self.main_window.annotation_window.select_annotation(self.annotation)

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
        # Set the toggle button to user mode
        self.set_user_icon(False)
