import pyvista as pv

from pyvistaqt import QtInteractor

from PyQt5.QtWidgets import QFrame, QVBoxLayout
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MVATViewer(QFrame):
    """
    A dedicated widget for holding the PyVista 3D Interactor.
    """
    def __init__(self, parent=None, point_size=1):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setAcceptDrops(True)
        
        # Layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create PyVista QtInteractor
        self.plotter = QtInteractor(self)
        self.plotter.set_background('white')
        self.plotter.enable_trackball_style()
        
        # Point cloud management
        self.point_cloud_mesh = None
        self.point_cloud_actor = None
        self.point_size = point_size
        
        # Add to layout
        self.layout.addWidget(self.plotter.interactor)

    def dragEnterEvent(self, event):
        """Accept drag if a single 3D file is being dragged."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1:
                file_path = urls[0].toLocalFile()
                if any(file_path.lower().endswith(ext) for ext in ['.ply', '.stl', '.obj', '.vtk', '.pcd']):
                    event.acceptProposedAction()
                else:
                    event.ignore()
            else:
                event.ignore()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Load the dropped 3D file into the viewer."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            file_path = event.mimeData().urls()[0].toLocalFile()
            mesh = pv.read(file_path)
            # Remove existing point cloud if any
            if self.point_cloud_actor is not None:
                self.plotter.remove_actor(self.point_cloud_actor)
            # Handle styling for point cloud vs meshes
            if 'RGB' in mesh.point_data:
                self.point_cloud_actor = self.plotter.add_mesh(mesh, 
                                                               scalars='RGB', 
                                                               rgb=True, 
                                                               point_size=self.point_size)
            else:
                point_size = self.point_size if mesh.n_cells == 0 else None
                self.point_cloud_actor = self.plotter.add_mesh(mesh, 
                                                               color='cyan', 
                                                               point_size=point_size)
            # Store for re-adding after clears
            self.point_cloud_mesh = mesh
            self.plotter.reset_camera()
            event.acceptProposedAction()
        except Exception as e:
            print(f"Failed to load 3D file: {e}")
            event.ignore()
        finally:
            QApplication.restoreOverrideCursor()

    def add_point_cloud(self):
        """Re-add the stored point cloud to the plotter."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if self.point_cloud_mesh is not None:
                if 'RGB' in self.point_cloud_mesh.point_data:
                    self.point_cloud_actor = self.plotter.add_mesh(self.point_cloud_mesh, 
                                                                   scalars='RGB', 
                                                                   rgb=True, 
                                                                   point_size=self.point_size)
                else:
                    point_size = self.point_size if self.point_cloud_mesh.n_cells == 0 else None
                    self.point_cloud_actor = self.plotter.add_mesh(self.point_cloud_mesh, 
                                                                   color='cyan', 
                                                                   point_size=point_size)
        finally:
            QApplication.restoreOverrideCursor()

    def set_point_cloud_visible(self, visible):
        """Set visibility of the point cloud actor."""
        if self.point_cloud_actor is not None:
            self.point_cloud_actor.SetVisibility(visible)

    def set_point_size(self, size):
        """Update the point size for point clouds."""
        self.point_size = size
        # If point cloud is loaded, update the actor
        if self.point_cloud_actor is not None:
            self.point_cloud_actor.GetProperty().SetPointSize(size)
            self.plotter.render()  # Force re-render

    def close(self):
        """Clean up the plotter resources."""
        if self.plotter:
            self.plotter.close()
