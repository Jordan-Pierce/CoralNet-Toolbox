from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDockWidget, QMainWindow, QMenu, QMenuBar, QToolBar, QWidget, QStatusBar, QSizePolicy

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DockWrapper(QDockWidget):
    """
    A universal wrapper that encapsulates a widget inside a QDockWidget.
    Uses an internal QMainWindow to natively support toolbars, menu bars, 
    status bars, and lifecycle event forwarding specific to the wrapped widget.
    """
    def __init__(self, title: str, object_name: str, main_widget: QWidget, parent=None):
        super().__init__(title, parent)
        
        # Strictly required to allow PyQt to save/restore layout states
        self.setObjectName(object_name)
        
        # Standardize dock features across the app
        self.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.setFeatures(QDockWidget.DockWidgetMovable | 
                         QDockWidget.DockWidgetFloatable | 
                         QDockWidget.DockWidgetClosable)

        # THE TRICK: Use a QMainWindow as the single widget inside the dock
        self.inner_window = QMainWindow()
        self.inner_window.setWindowFlags(Qt.Widget) 
        
        # Strip default margins so the payload sits flush against the dock borders
        self.inner_window.setContentsMargins(0, 0, 0, 0)
        
        # Store the payload and set it as the central widget
        self.payload_widget = main_widget
        self.inner_window.setCentralWidget(self.payload_widget)
        
        self.setWidget(self.inner_window)
        
        # This applies the specific background color to ANY QToolBar 
        # that gets attached to this specific dock's inner window.
        # Add QMenuBar to the stylesheet so it isn't transparent
        self.inner_window.setStyleSheet("""
            QMenuBar {
                background-color: rgb(248, 249, 250);
                border-bottom: 1px solid #ddd;
            }
            QToolBar {
                background-color: rgb(248, 249, 250);
                border: none;
                spacing: 4px;
            }
        """)

    # --- UI COMPONENT MOUNTING ---
    
    def add_menu(self, menu: QMenu):
        """Attaches a QMenu to the dock's internal menu bar using setMenuWidget."""
        
        # 1. Create a physical QMenuBar and force it into the layout as a standard widget
        if not hasattr(self, '_local_menubar'):
            self._local_menubar = QMenuBar(self.inner_window)
            self._local_menubar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            
            # CRITICAL: setMenuWidget completely bypasses OS native-menubar bugs
            self.inner_window.setMenuWidget(self._local_menubar)
            
            # Ensure text is visible even if the app uses Dark Mode
            self._local_menubar.setStyleSheet("""
                QMenuBar {
                    background-color: rgb(248, 249, 250);
                    color: black;
                    border-bottom: 1px solid #ddd;
                }
                QMenuBar::item {
                    background-color: transparent;
                    padding: 4px 8px;
                }
                QMenuBar::item:selected {
                    background-color: #e2e6ea;
                }
            """)

        # 2. Prevent Python Garbage Collection from deleting the menu!
        if not hasattr(self, '_menus'):
            self._menus = []
        self._menus.append(menu)
        
        # 3. Bind coordinates to the menubar to fix the multi-monitor offset bug
        menu.setParent(self._local_menubar)
        
        # 4. Mount the menu
        self._local_menubar.addMenu(menu)

    def add_toolbar(self, toolbar: QToolBar, area=Qt.TopToolBarArea):
        """
        Attaches a QToolBar to the dock. 
        Use Qt.BottomToolBarArea, Qt.LeftToolBarArea, etc., for positioning.
        """
        self.inner_window.addToolBar(area, toolbar)

    def add_toolbar_break(self, area=Qt.TopToolBarArea):
        """Forces the next toolbar added to this area to start on a new row."""
        self.inner_window.addToolBarBreak(area)

    def set_status_bar(self, status_bar: QStatusBar):
        """Attaches a true QStatusBar to the absolute bottom of the dock."""
        self.inner_window.setStatusBar(status_bar)

    def get_payload(self) -> QWidget:
        """Returns the encapsulated main widget."""
        return self.payload_widget

    # --- DYNAMIC UI CONTROLS ---

    def show_status_message(self, message: str, timeout: int = 3000):
        """
        Flashes a temporary message on the dock's status bar (if it has one).
        Timeout is in milliseconds (0 means it stays until overwritten).
        """
        status_bar = self.inner_window.statusBar()
        if status_bar:
            status_bar.showMessage(message, timeout)

    def update_title(self, new_title: str):
        """Dynamically updates the title of the dock tab/header."""
        self.setWindowTitle(new_title)

    def set_locked(self, locked: bool = True):
        """Locks or unlocks the dock's ability to be moved, floated, or closed."""
        if locked:
            # Remove all features to lock it in place
            self.setFeatures(QDockWidget.NoDockWidgetFeatures)
        else:
            # Restore standard features
            self.setFeatures(QDockWidget.DockWidgetMovable | 
                             QDockWidget.DockWidgetFloatable | 
                             QDockWidget.DockWidgetClosable)

    def toggle_toolbars(self, visible: bool):
        """Hides or shows all toolbars attached to this dock (e.g., for Zen Mode)."""
        for toolbar in self.inner_window.findChildren(QToolBar):
            toolbar.setVisible(visible)

    def replace_payload(self, new_widget: QWidget):
        """
        Swaps out the central widget of the inner window without destroying the dock,
        menus, or toolbars.
        """
        # Remove the old payload
        old_widget = self.inner_window.takeCentralWidget()
        if old_widget:
            old_widget.setParent(None)
            
        # Set the new payload
        self.payload_widget = new_widget
        self.inner_window.setCentralWidget(self.payload_widget)

    # --- LIFECYCLE EVENT FORWARDING ---

    def closeEvent(self, event):
        """
        Intercepts the dock closing and forwards the event to the payload.
        Critical for widgets that need to stop threads or save state.
        """
        if hasattr(self.payload_widget, 'closeEvent'):
            self.payload_widget.closeEvent(event)
        super().closeEvent(event)

    def showEvent(self, event):
        """Forwards the show event to the payload to resume heavy tasks/animations."""
        if hasattr(self.payload_widget, 'showEvent'):
            self.payload_widget.showEvent(event)
        super().showEvent(event)

    def hideEvent(self, event):
        """Forwards the hide event to the payload to pause heavy tasks/animations."""
        if hasattr(self.payload_widget, 'hideEvent'):
            self.payload_widget.hideEvent(event)
        super().hideEvent(event)