from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDockWidget, QMainWindow, QMenu, QToolBar, QWidget


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DockWrapper(QDockWidget):
    """
    A universal wrapper that encapsulates a widget inside a QDockWidget.
    Uses an internal QMainWindow to natively support toolbars and menu bars
    specific to the wrapped widget.
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
        
        # Ensure it behaves as a child widget, not a pop-up desktop window
        self.inner_window.setWindowFlags(Qt.Widget) 
        
        # Set the core payload (e.g., TimerWindow, MVATViewer)
        self.inner_window.setCentralWidget(main_widget)
        self.setWidget(self.inner_window)

    def add_menu(self, menu: QMenu):
        """Attaches a QMenu to the dock's internal menu bar."""
        self.inner_window.menuBar().addMenu(menu)

    def add_toolbar(self, toolbar: QToolBar, area=Qt.TopToolBarArea):
        """Attaches a QToolBar to the dock's internal layout."""
        self.inner_window.addToolBar(area, toolbar)