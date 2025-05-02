import warnings

from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from PyQt5.QtGui import QPixmap, QColor, QPainter, QCursor
from PyQt5.QtWidgets import (QGraphicsView, QGraphicsScene, QWidget, QVBoxLayout,
                             QLabel, QHBoxLayout, QFrame, QGroupBox, QPushButton)

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

    def paintEvent(self, event):
        """Handle the paint event to draw the confidence bar."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Calculate the middle point
        mid_width = self.width() // 2

        # Draw the border for both halves
        painter.setPen(self.color)
        painter.drawRect(0, 0, mid_width - 1, self.height() - 1)  # Left half
        painter.drawRect(mid_width, 0, self.width() - mid_width - 1, self.height() - 1)  # Right half

        # Draw the filled part of the bar from middle to confidence width
        filled_width = int((self.width() - mid_width) * (self.confidence / 100))
        painter.setBrush(QColor(self.color.red(), self.color.green(), self.color.blue(), 192))  # 75% transparency
        painter.drawRect(mid_width, 0, filled_width, self.height() - 1)

        # Set text color to black
        painter.setPen(Qt.black)

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
            self.setCursor(QCursor(Qt.ForbiddenCursor))  # Use a forbidden cursor icon

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
            
    def scale_pixmap(self, pixmap):
        """Scale pixmap and graphic if they exceed max dimension while preserving aspect ratio"""
        width = pixmap.width()
        height = pixmap.height()
        
        # Check if scaling is needed
        if width <= self.max_graphic_size and height <= self.max_graphic_size:
            return pixmap
            
        # Calculate scale factor based on largest dimension
        scale = self.max_graphic_size / max(width, height)
        
        # Scale pixmap
        scaled_pixmap = pixmap.scaled(
            int(width * scale), 
            int(height * scale),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        return scaled_pixmap

    def display_cropped_image(self, annotation):
        """Display the cropped image and update the bar chart."""
        try:
            self.clear_display()
            self.update_annotation(annotation)
            if self.annotation.cropped_image:
                # Get the cropped image graphic
                cropped_image_graphic = self.scale_pixmap(annotation.get_cropped_image_graphic())
                # Add the scaled annotation graphic (as pixmap)
                self.scene.addPixmap(cropped_image_graphic)
                # Add the border color with increased width
                self.scene.setSceneRect(QRectF(cropped_image_graphic.rect()))
                self.graphics_view.setStyleSheet("QGraphicsView { border: 3px solid transparent; }")
                # Fit the view to the scene
                self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
                self.graphics_view.centerOn(self.scene.sceneRect().center())
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

    def create_bar_chart(self):
        """Create and populate the confidence bar chart."""
        self.clear_layout(self.bar_chart_layout)
        self.confidence_bar_labels = []

        labels, confidences = self.get_chart_data()
        max_color = labels[confidences.index(max(confidences))].color
        self.graphics_view.setStyleSheet(f"border: 2px solid {max_color.name()};")

        for idx, (label, confidence) in enumerate(zip(labels, confidences)):
            bar_widget = ConfidenceBar(self, label, confidence, self.bar_chart_widget)
            bar_widget.barClicked.connect(self.handle_bar_click)  # Connect the signal to the slot
            self.add_bar_to_layout(bar_widget, label, confidence, idx + 1)
            self.confidence_bar_labels.append(label)

    def get_chart_data(self):
        """Retrieve the top 5 labels and confidences from the current chart dictionary."""
        keys = list(self.chart_dict.keys())[:5]
        return (
            keys,
            [conf_value * 100 for conf_value in self.chart_dict.values()][:5]
        )

    def add_bar_to_layout(self, bar_widget, label, confidence, top_k):
        """Add a single confidence bar widget to the bar chart layout."""
        bar_layout = QHBoxLayout(bar_widget)
        bar_layout.setContentsMargins(5, 2, 5, 2)
        
        # Create a top-k icon for the label
        icon_label = QLabel(bar_widget)
        icon_label.setPixmap(self.top_k_icons[str(top_k)])
        icon_label.setFixedSize(14, 14)
        bar_layout.addWidget(icon_label)

        # Create and style the class label
        class_label = QLabel(label.short_label_code, bar_widget)
        class_label.setAlignment(Qt.AlignCenter)
        bar_layout.addWidget(class_label)

        # Create and style the percentage label
        percentage_label = QLabel(f"{confidence:.2f}%", bar_widget)
        percentage_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        bar_layout.addWidget(percentage_label)

        self.bar_chart_layout.addWidget(bar_widget)

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