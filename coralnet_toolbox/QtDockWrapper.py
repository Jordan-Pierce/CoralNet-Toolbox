from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDockWidget, QMainWindow, QMenu, QToolBar, QWidget, QStatusBar


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DockWrapper(QDockWidget):
    """
    A universal wrapper that encapsulates a widget inside a QDockWidget.
    Uses an internal QMainWindow to natively support toolbars, menu bars, 
    and status bars specific to the wrapped widget.
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

    def add_menu(self, menu: QMenu):
        """Attaches a QMenu to the dock's internal menu bar."""
        self.inner_window.menuBar().addMenu(menu)

    def add_toolbar(self, toolbar: QToolBar, area=Qt.TopToolBarArea):
        """
        Attaches a QToolBar to the dock. 
        Use Qt.BottomToolBarArea, Qt.LeftToolBarArea, etc., for positioning.
        """
        self.inner_window.addToolBar(area, toolbar)

    def set_status_bar(self, status_bar: QStatusBar):
        """Attaches a true QStatusBar to the absolute bottom of the dock."""
        self.inner_window.setStatusBar(status_bar)

    def get_payload(self) -> QWidget:
        """Returns the encapsulated main widget."""
        return self.payload_widget

    def closeEvent(self, event):
        """
        Intercepts the dock closing and forwards the event to the payload.
        Critical for widgets that need to stop threads or save state (like TimerWindow).
        """
        if hasattr(self.payload_widget, 'closeEvent'):
            self.payload_widget.closeEvent(event)
        super().closeEvent(event)