import warnings

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QToolButton, QFrame, QGroupBox, QAction

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class CollapsibleSection(QWidget):
    """
    A collapsible section widget that displays a button and shows a popup when clicked.
    
    The popup can contain multiple widgets organized in group boxes.
    """
    
    def __init__(self, title, icon, parent=None, position='bottomleft'):
        """
        Initialize the collapsible section.
        
        Args:
            title (str): The title to display on the button
            icon (str): The icon filename to display on the button
            parent (QWidget): Parent widget
        """
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        # Position of popup relative to the button. One of:
        # 'topleft', 'topright', 'bottomleft', 'bottomright'
        self.position = position if isinstance(position, str) else 'bottomleft'
        if self.position not in ('topleft', 'topright', 'bottomleft', 'bottomright'):
            self.position = 'bottomleft'

        # Create the action
        self.toggle_action = QAction(QIcon(get_icon(icon)), title, self)
        self.toggle_action.setCheckable(False)
        self.toggle_action.triggered.connect(self.toggle_content)

        # Header button using the action
        self.toggle_button = QToolButton()
        self.toggle_button.setDefaultAction(self.toggle_action)
        self.toggle_button.setCheckable(False)
        self.toggle_button.setAutoRaise(True)  # Gives a flat appearance until clicked

        # Popup frame
        self.popup = QFrame(self.window())
        self.popup.setWindowFlags(Qt.Popup)
        self.popup.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.popup.setLayout(QVBoxLayout())
        self.popup.layout().setContentsMargins(5, 5, 5, 5)
        self.popup.hide()

        # Add button to layout
        self.layout().addWidget(self.toggle_button)

    def toggle_content(self):
        """Toggle the visibility of the popup content."""
        if self.popup.isVisible():
            self.popup.hide()
        else:
            pos = self.toggle_button.mapToGlobal(QPoint(0, 0))
            popup_size = self.popup.sizeHint()
            popup_width = popup_size.width()
            popup_height = popup_size.height()

            # Compute coordinates based on desired position
            if self.position == 'topleft':
                # Align popup left edge with button left edge, bottom with button top
                new_x = pos.x()
                new_y = pos.y() - popup_height
            elif self.position == 'topright':
                # Align popup left edge with button right edge, bottom with button top
                new_x = pos.x() + self.toggle_button.width()
                new_y = pos.y() - popup_height
            elif self.position == 'bottomright':
                # Align popup left edge with button right edge, top with button bottom
                new_x = pos.x() + self.toggle_button.width()
                new_y = pos.y() + self.toggle_button.height()
            else:  # 'bottomleft'
                # Align popup right edge with button right edge, top with button bottom
                new_x = pos.x() - popup_width + self.toggle_button.width()
                new_y = pos.y() + self.toggle_button.height()

            self.popup.move(new_x, new_y)
            self.popup.show()

    def add_widget(self, widget, title=None):
        """
        Add a widget to the popup.
        
        Args:
            widget (QWidget): The widget to add
            title (str): Optional title for the group box containing the widget
        """
        group_box = QGroupBox()
        group_box.setTitle(title)
        group_box.setLayout(QVBoxLayout())
        group_box.layout().addWidget(widget)
        self.popup.layout().addWidget(group_box)

    def hideEvent(self, event):
        """Handle hide event to also hide the popup."""
        self.popup.hide()
        self.toggle_action.setChecked(False)
        super().hideEvent(event)
