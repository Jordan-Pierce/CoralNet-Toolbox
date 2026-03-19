from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (QDockWidget, QMainWindow, QMenu, QMenuBar, 
                             QToolBar, QWidget, QStatusBar, QSizePolicy)

# -----------------------------------------------------------------------------------------
# Secondary Window Host 
# -----------------------------------------------------------------------------------------

class SecondaryDockHost(QMainWindow):
    """
    A temporary floating window that catches torn-off docks.
    If it becomes empty, it automatically cleans itself up.
    """
    def __init__(self, main_window_ref, parent=None):
        super().__init__(parent)
        self.main_app_window = main_window_ref
        
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("Floating Workspace")
        self.setDockNestingEnabled(True)
        
        # Zero-height central widget to yield all space to the docks
        self.central_widget = QWidget()
        self.central_widget.setFixedHeight(0)
        self.setCentralWidget(self.central_widget)

    def check_empty(self):
        """Checks if there are any docks left in this host. If not, destroy it."""
        docks = self.findChildren(QDockWidget)
        if not docks:
            self.deleteLater() # Safely destroy the empty shell

    def closeEvent(self, event):
        """If the user explicitly hits the red X on the host, send docks home."""
        for dock in self.findChildren(QDockWidget):
            self.main_app_window.addDockWidget(Qt.RightDockWidgetArea, dock)
            dock.setFloating(False)
        super().closeEvent(event)


# -----------------------------------------------------------------------------------------
# Main Wrapper Class
# -----------------------------------------------------------------------------------------

class DockWrapper(QDockWidget):
    def __init__(self, title: str, object_name: str, main_widget: QWidget, parent=None):
        super().__init__(title, parent)
        
        # Store a hard reference to the original main window for snap-back detection
        self.main_app_window = parent
        
        self.setObjectName(object_name)
        self.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.setFeatures(QDockWidget.DockWidgetMovable | 
                         QDockWidget.DockWidgetFloatable | 
                         QDockWidget.DockWidgetClosable)

        # Catch the exact moment the dock becomes floating (dropped outside its host)
        self.topLevelChanged.connect(self._on_float_state_changed)

        # -- UI Mounting (Your existing QMainWindow Trick) --
        self.inner_window = QMainWindow()
        self.inner_window.setWindowFlags(Qt.Widget) 
        self.inner_window.setContentsMargins(0, 0, 0, 0)
        
        self.payload_widget = main_widget
        self.inner_window.setCentralWidget(self.payload_widget)
        self.setWidget(self.inner_window)
        
        self.inner_window.setStyleSheet("""
            QMenuBar { background-color: rgb(248, 249, 250); border-bottom: 1px solid #ddd; }
            QToolBar { background-color: rgb(248, 249, 250); border: none; spacing: 4px; }
        """)

    # --- THE LIFECYCLE MAGIC ---

    def _on_float_state_changed(self, is_floating: bool):
        if is_floating:
            # We must use a slight delay. When Qt emits this signal, the internal 
            # C++ drag-and-drop state machine is still finishing. Reparenting 
            # instantly will crash the UI.
            QTimer.singleShot(10, self._handle_drop_location)

    def _handle_drop_location(self):
        # 1. Get the current global position of the mouse where they dropped it
        cursor_pos = QCursor.pos()
        
        # Keep track of where we came from to clean up empty shells
        previous_parent = self.parent()

        # 2. THE SNAP BACK: Did they drop it over the main window?
        # frameGeometry() gets the global screen coordinates of the main window
        if self.main_app_window.frameGeometry().contains(cursor_pos):
            self.setParent(self.main_app_window)
            # Default it to the right side. The user can drag it again to place it precisely.
            self.main_app_window.addDockWidget(Qt.RightDockWidgetArea, self)
            self.setFloating(False)
            
            # Clean up the shell we just left
            if isinstance(previous_parent, SecondaryDockHost):
                previous_parent.check_empty()
            return

        # 3. THE WRAP: They dropped it on the desktop or second monitor
        # If it's already in a secondary host, leave it alone (they just moved the floating dock)
        if isinstance(previous_parent, SecondaryDockHost):
            return

        # Create a new secondary host exactly where they dropped it
        new_host = SecondaryDockHost(self.main_app_window, parent=self.main_app_window)
        new_host.move(self.pos())  # Move host to match the floating dock's position
        new_host.resize(self.size())
        
        # Mount this dock into the new host
        self.setParent(new_host)
        new_host.addDockWidget(Qt.LeftDockWidgetArea, self)
        self.setFloating(False) 
        new_host.show()

        # Clean up the shell we just left (in case they dragged from Host A to Desktop to make Host B)
        if isinstance(previous_parent, SecondaryDockHost):
            previous_parent.check_empty()

    # --- YOUR EXISTING METHODS ---
    
    def add_menu(self, menu: QMenu):
        if not hasattr(self, '_local_menubar'):
            self._local_menubar = QMenuBar(self.inner_window)
            self._local_menubar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.inner_window.setMenuWidget(self._local_menubar)
        if not hasattr(self, '_menus'):
            self._menus = []
        self._menus.append(menu)
        menu.setParent(self._local_menubar)
        self._local_menubar.addMenu(menu)

    def add_toolbar(self, toolbar: QToolBar, area=Qt.TopToolBarArea):
        self.inner_window.addToolBar(area, toolbar)

    def add_toolbar_break(self, area=Qt.TopToolBarArea):
        self.inner_window.addToolBarBreak(area)

    def set_status_bar(self, status_bar: QStatusBar):
        self.inner_window.setStatusBar(status_bar)

    def get_payload(self) -> QWidget:
        return self.payload_widget

    def update_title(self, new_title: str):
        self.setWindowTitle(new_title)

    def set_locked(self, locked: bool = True):
        if locked:
            self.setFeatures(QDockWidget.NoDockWidgetFeatures)
        else:
            self.setFeatures(QDockWidget.DockWidgetMovable | 
                             QDockWidget.DockWidgetFloatable | 
                             QDockWidget.DockWidgetClosable)

    def closeEvent(self, event):
        if hasattr(self.payload_widget, 'closeEvent'):
            self.payload_widget.closeEvent(event)
        super().closeEvent(event)