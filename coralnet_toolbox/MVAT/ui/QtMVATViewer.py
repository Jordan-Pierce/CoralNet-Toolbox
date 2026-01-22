from pyvistaqt import QtInteractor

from PyQt5.QtWidgets import QFrame, QVBoxLayout

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MVATViewer(QFrame):
    """
    A dedicated widget for holding the PyVista 3D Interactor.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        
        # Layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create PyVista QtInteractor
        self.plotter = QtInteractor(self)
        self.plotter.set_background('white')
        self.plotter.enable_trackball_style()
        
        # Add to layout
        self.layout.addWidget(self.plotter.interactor)

    def close(self):
        """Clean up the plotter resources."""
        if self.plotter:
            self.plotter.close()