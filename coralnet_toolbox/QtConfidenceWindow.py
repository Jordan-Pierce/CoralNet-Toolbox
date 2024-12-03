import warnings

from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from PyQt5.QtGui import QPixmap, QColor, QPainter, QCursor
from PyQt5.QtWidgets import (QGraphicsView, QGraphicsScene, QWidget, QVBoxLayout,
                             QLabel, QHBoxLayout, QFrame)

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ConfidenceBar(QFrame):
    barClicked = pyqtSignal(object)  # Define a signal that takes an object (label)

    def __init__(self, confidence_window, label, confidence, parent=None):
        super().__init__(parent)
        self.confidence_window = confidence_window

        self.label = label
        self.confidence = confidence
        self.color = label.color
        self.setFixedHeight(20)  # Set a fixed height for the bars

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw the border
        painter.setPen(self.color)
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

        # Draw the filled part of the bar
        filled_width = int(self.width() * (self.confidence / 100))
        painter.setBrush(QColor(self.color.red(), self.color.green(), self.color.blue(), 192))  # 75% transparency
        painter.drawRect(0, 0, filled_width, self.height() - 1)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            self.handle_click()

    def handle_click(self):
        # Check if the Selector tool is active
        if self.confidence_window.main_window.annotation_window.selected_tool == "select":
            # Emit the signal with the label object
            self.barClicked.emit(self.label)

    def enterEvent(self, event):
        super().enterEvent(event)
        # Change cursor based on the active tool
        if self.confidence_window.main_window.annotation_window.selected_tool == "select":
            self.setCursor(QCursor(Qt.PointingHandCursor))
        else:
            self.setCursor(QCursor(Qt.ForbiddenCursor))  # Use a forbidden cursor icon

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.setCursor(QCursor(Qt.ArrowCursor))  # Reset to the default cursor


class ConfidenceWindow(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.label_window = main_window.label_window

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.graphics_view = None
        self.scene = None
        self.downscale_factor = 1.0

        self.bar_chart_widget = None
        self.bar_chart_layout = None

        self.init_graphics_view()
        self.init_bar_chart_widget()

        self.annotation = None
        self.user_confidence = None
        self.machine_confidence = None
        self.cropped_image = None
        self.chart_dict = None

        # Add QLabel for dimensions
        self.dimensions_label = QLabel(self)
        self.dimensions_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.dimensions_label)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_blank_pixmap()
        self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def init_graphics_view(self):
        self.graphics_view = QGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.scene)
        self.layout.addWidget(self.graphics_view, 2)  # 2 for stretch factor
        self.update_blank_pixmap()

    def init_bar_chart_widget(self):
        self.bar_chart_widget = QWidget()
        self.bar_chart_layout = QVBoxLayout(self.bar_chart_widget)
        self.bar_chart_layout.setContentsMargins(0, 0, 0, 0)
        self.bar_chart_layout.setSpacing(2)  # Set spacing to make bars closer
        self.layout.addWidget(self.bar_chart_widget, 1)  # 1 for stretch factor

    def update_blank_pixmap(self):
        view_size = self.graphics_view.size()
        new_pixmap = QPixmap(view_size)
        new_pixmap.fill(Qt.transparent)
        self.scene.clear()
        self.scene.addPixmap(new_pixmap)

    def update_annotation(self, annotation):
        if annotation:
            self.annotation = annotation
            self.user_confidence = annotation.user_confidence
            self.machine_confidence = annotation.machine_confidence
            self.cropped_image = annotation.cropped_image.copy()
            self.chart_dict = self.machine_confidence if self.machine_confidence else self.user_confidence

    def display_cropped_image(self, annotation):
        try:
            self.clear_display()  # Clear the current display before updating
            self.update_annotation(annotation)
            if self.cropped_image:  # Ensure cropped_image is not None
                cropped_image = annotation.get_cropped_image(self.downscale_factor)
                cropped_image_graphic = annotation.get_cropped_image_graphic()
                # Add the image
                self.scene.addPixmap(cropped_image)
                # Add the annotation graphic
                self.scene.addItem(cropped_image_graphic)
                # Add the border color
                self.scene.setSceneRect(QRectF(cropped_image.rect()))
                # Fit the view to the scene
                self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
                self.graphics_view.centerOn(self.scene.sceneRect().center())
                # Create the bar charts
                self.create_bar_chart()

                # Update dimensions label
                height = cropped_image.height()
                width = cropped_image.width()
                self.dimensions_label.setText(f"Crop: {height} x {width}")

        except:
            # Cropped image is None or some other error occurred
            pass

    def create_bar_chart(self):
        self.clear_layout(self.bar_chart_layout)

        labels, confidences = self.get_chart_data()
        max_color = labels[confidences.index(max(confidences))].color
        self.graphics_view.setStyleSheet(f"border: 2px solid {max_color.name()};")

        for label, confidence in zip(labels, confidences):
            bar_widget = ConfidenceBar(self, label, confidence, self.bar_chart_widget)
            bar_widget.barClicked.connect(self.handle_bar_click)  # Connect the signal to the slot
            self.add_bar_to_layout(bar_widget, label, confidence)

    def get_chart_data(self):
        keys = list(self.chart_dict.keys())[:5]
        return (
            keys,
            [conf_value * 100 for conf_value in self.chart_dict.values()][:5]
        )

    def add_bar_to_layout(self, bar_widget, label, confidence):
        bar_layout = QHBoxLayout(bar_widget)
        bar_layout.setContentsMargins(5, 2, 5, 2)

        class_label = QLabel(label.short_label_code, bar_widget)
        class_label.setAlignment(Qt.AlignCenter)
        bar_layout.addWidget(class_label)

        percentage_label = QLabel(f"{confidence:.2f}%", bar_widget)
        percentage_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        bar_layout.addWidget(percentage_label)

        self.bar_chart_layout.addWidget(bar_widget)

    def clear_layout(self, layout):
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

    def handle_bar_click(self, label):
        # Update the confidences to whichever bar was selected
        self.annotation.update_user_confidence(label)
        # Update the label to whichever bar was selected
        self.annotation.update_label(label)
        # Update everything (essentially)
        self.main_window.annotation_window.unselect_annotation(self.annotation)
        self.main_window.annotation_window.select_annotation(self.annotation)