# Critial: order of imports matter
import PyQt5.QtCore
import PyQtAds
from PyQtAds import ads

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QMenu, QMenuBar, QToolBar, QWidget, 
                             QStatusBar, QSizePolicy, QVBoxLayout)

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DockWrapper(ads.CDockWidget):
    """
    Safely encapsulates a widget inside an ADS dock without crashing the C++ layout engine.
    Uses a standard QVBoxLayout to safely stack menus, toolbars, and the payload.
    """
    def __init__(self, title: str, object_name: str, main_widget: QWidget, parent=None, icon=None):
        # Do NOT pass parent here. ADS must take exclusive memory ownership.
        super().__init__(title)
        
        self.setObjectName(object_name)
        self.setWindowTitle(title)
        
        # If icon not provided, try to get it from parent (MainWindow)
        if icon is None and parent is not None and hasattr(parent, 'coralnet_icon'):
            icon = parent.coralnet_icon
        
        # Store icon reference to prevent garbage collection
        self._window_icon = icon
        
        # Set window icon if provided (displays when dock is floated)
        if icon is not None:
            self.setWindowIcon(icon)
        
        self.setFeature(ads.CDockWidget.DockWidgetClosable, True)
        self.setFeature(ads.CDockWidget.DockWidgetFloatable, True)
        self.setFeature(ads.CDockWidget.DockWidgetMovable, True)
        
        # Disable the "List all tabs" button that appears in the tab bar
        self.setFeature(ads.CDockWidget.DockWidgetDeleteOnClose, False)

        # Use a simple QWidget + QVBoxLayout instead of QMainWindow
        self.inner_widget = QWidget()
        self.layout = QVBoxLayout(self.inner_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0) # Keeps toolbars flush against each other
        
        self.payload_widget = main_widget
        
        # Add payload to layout. Toolbars and menus will be inserted ABOVE this dynamically.
        self.layout.addWidget(self.payload_widget)
        
        # Mount the safe widget to the ADS Dock
        self.setWidget(self.inner_widget)
        
        self.inner_widget.setStyleSheet("""
            QMenuBar { background-color: rgb(248, 249, 250); border-bottom: 1px solid #ddd; }
            QToolBar { background-color: rgb(248, 249, 250); border: none; padding: 2px; }
        """)

    # --- UI COMPONENT MOUNTING ---
    
    def add_menu(self, menu: QMenu):
        if not hasattr(self, '_local_menubar'):
            self._local_menubar = QMenuBar()
            self._local_menubar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._local_menubar.setStyleSheet("""
                QMenuBar { color: black; }
                QMenuBar::item { background-color: transparent; padding: 4px 8px; }
                QMenuBar::item:selected { background-color: #e2e6ea; }
            """)
            # Insert menu at the absolute top (index 0)
            self.layout.insertWidget(0, self._local_menubar)
            
        if not hasattr(self, '_menus'):
            self._menus = []
        self._menus.append(menu)
        self._local_menubar.addMenu(menu)

    def add_toolbar(self, toolbar: QToolBar, area=Qt.TopToolBarArea):
        """Attaches a QToolBar to the dock, respecting Top or Bottom placement."""
        if area == Qt.BottomToolBarArea:
            # Append it to the layout so it sits below the payload widget
            self.layout.addWidget(toolbar)
        else:
            # Insert it right above the payload widget
            payload_index = self.layout.indexOf(self.payload_widget)
            self.layout.insertWidget(payload_index, toolbar)

    def add_toolbar_break(self, area=None):
        # QVBoxLayout stacks toolbars automatically, so "breaks" aren't strictly necessary,
        # but you can add a small spacer line here if you want visual separation.
        pass

    def set_status_bar(self, status_bar: QStatusBar):
        # Add status bar to the absolute bottom (after the payload)
        self.layout.addWidget(status_bar)

    def get_payload(self) -> QWidget:
        return self.payload_widget

    def update_title(self, new_title: str):
        self.setWindowTitle(new_title)

    def toggle_toolbars(self, visible: bool):
        for toolbar in self.inner_widget.findChildren(QToolBar):
            toolbar.setVisible(visible)

    # --- LIFECYCLE EVENT FORWARDING ---
    def closeEvent(self, event):
        if hasattr(self.payload_widget, 'closeEvent'):
            self.payload_widget.closeEvent(event)
        super().closeEvent(event)

    def showEvent(self, event):
        if hasattr(self.payload_widget, 'showEvent'):
            self.payload_widget.showEvent(event)
        super().showEvent(event)

    def hideEvent(self, event):
        if hasattr(self.payload_widget, 'hideEvent'):
            self.payload_widget.hideEvent(event)
        super().hideEvent(event)