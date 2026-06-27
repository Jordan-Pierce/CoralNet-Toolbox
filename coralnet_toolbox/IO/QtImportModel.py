"""
Import 3D Model dialog.

Provides a unified dialog for importing 3D models (meshes, point clouds,
Gaussian splats) into the MVAT viewer.  Used both from the File > Import menu
and from the drag-and-drop handler in QtMVATViewer so the import UI is never
duplicated.
"""

import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
)


SUPPORTED_EXTENSIONS = ['.ply', '.obj', '.stl', '.vtk', '.pcd']
FILE_FILTER = (
    "3D Files (*.ply *.obj *.stl *.vtk *.pcd);;"
    "PLY Files (*.ply);;"
    "OBJ Files (*.obj);;"
    "STL Files (*.stl);;"
    "VTK Files (*.vtk);;"
    "PCD Files (*.pcd);;"
    "All Files (*)"
)


def detect_ply_type(file_path: str) -> str:
    """Peek at the PLY header to choose a sensible default.

    Returns one of ``'mesh'``, ``'gaussian'``, or ``'pointcloud'``.
    """
    try:
        header_lines = []
        with open(file_path, 'rb') as fh:
            for raw_line in fh:
                line = raw_line.decode('ascii', errors='ignore').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break
        header = '\n'.join(header_lines)
        if 'element face' in header:
            return 'mesh'
        if 'f_dc_0' in header:
            return 'gaussian'
        return 'pointcloud'
    except Exception:
        return 'mesh'


def find_texture_file(file_path: str) -> bool:
    """Check if a texture image exists in the same directory as the mesh file.

    Looks for common texture file names (texture.png, texture.jpg, etc.) in the
    same directory as the input file.

    Args:
        file_path: Path to the 3D model file.

    Returns:
        True if a texture file is found, False otherwise.
    """
    try:
        dir_name = os.path.dirname(file_path)
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            tex_path = os.path.join(dir_name, f"texture{ext}")
            if os.path.exists(tex_path):
                return True
        return False
    except Exception:
        return False


class ImportModelDialog(QDialog):
    """Modal dialog for importing a 3D model file.

    Can be opened in two modes:

    * **Browse mode** (default): shows a file-browser group box at the top so
      the user can pick a file.  Used from ``File > Import > 3D Model``.
    * **Pre-filled mode**: call with ``file_path`` already set (e.g. from
      drag-and-drop) and the file-browser is hidden.

    On accept the caller reads back the chosen parameters via
    :pymethod:`get_result`.
    """

    def __init__(self, parent=None, file_path: str = None):
        super().__init__(parent)
        self.setWindowTitle("Import 3D Model")
        self.setModal(True)
        self.resize(480, 0)

        self._file_path = file_path

        layout = QVBoxLayout(self)

        # --- File browser (hidden when pre-filled) ---
        self._browse_group = QGroupBox("File")
        browse_form = QFormLayout(self._browse_group)
        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setReadOnly(True)
        path_row.addWidget(self._path_edit)
        self._browse_btn = QPushButton("Browse")
        self._browse_btn.clicked.connect(self._browse_file)
        path_row.addWidget(self._browse_btn)
        browse_form.addRow("File:", path_row)
        layout.addWidget(self._browse_group)

        if file_path:
            self._path_edit.setText(file_path)

        # --- Import settings ---
        self._settings_group = QGroupBox("Settings")
        form = QFormLayout(self._settings_group)

        self._type_combo = QComboBox()
        self._type_combo.addItems(["Mesh", "Point Cloud", "3D Gaussian Splatting"])
        form.addRow("Product type:", self._type_combo)

        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["True", "False"])
        self._sort_combo.setToolTip(
            "Spatially sort the 3D elements using a Morton Z-order curve after loading.\n"
            "Meshes: sorts faces for GPU cache coherence + index-map quality.\n"
            "Point clouds: sorts points so index maps compress better. Recommended."
        )
        form.addRow("Sort Data:", self._sort_combo)
        
        # --- Texture loading (mesh-only) ---
        self._texture_combo = QComboBox()
        self._texture_combo.addItems(["True", "False"])
        self._texture_combo.setToolTip(
            "Load associated texture image if present in the same directory as the mesh.\n"
            "Requires UV coordinates to be present in the mesh file.\n"
            "Only available for mesh products."
        )
        self._texture_combo.setEnabled(False)
        form.addRow("Load Texture:", self._texture_combo)

        self._kd_combo = QComboBox()
        self._kd_combo.addItems(["True", "False"])
        form.addRow("Calculate KD-Tree:", self._kd_combo)

        slider_row = QHBoxLayout()
        self._simplify_slider = QSlider(Qt.Horizontal)
        self._simplify_slider.setRange(0, 100)
        self._simplify_slider.setValue(0)
        self._simplify_slider.setTickInterval(10)
        self._simplify_slider.setTickPosition(QSlider.TicksBelow)
        self._simplify_slider.setToolTip(
            "Fraction of elements to remove before loading (0 = no simplification).\n\n"
            "0.0  — Full detail, slowest to render.\n"
            "0.5  — Remove 50%; good balance.\n"
            "0.9  — Remove 90%; fastest, lowest detail."
        )
        self._simplify_label = QLabel("0.00")
        self._simplify_label.setMinimumWidth(36)
        self._simplify_slider.valueChanged.connect(
            lambda v: self._simplify_label.setText(f"{v / 100:.2f}")
        )
        slider_row.addWidget(self._simplify_slider)
        slider_row.addWidget(self._simplify_label)
        form.addRow("Simplification:", slider_row)

        layout.addWidget(self._settings_group)

        # --- Buttons ---
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        # Apply auto-detection if a file is already set.
        if file_path:
            self._apply_file(file_path)

    # ------------------------------------------------------------------
    # File browsing
    # ------------------------------------------------------------------

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select 3D Model", "", FILE_FILTER
        )
        if path:
            self._path_edit.setText(path)
            self._file_path = path
            self._apply_file(path)

    def _apply_file(self, path: str):
        """Update the form to reflect the chosen file.

        Auto-detects the product type based on the file extension and format,
        and enables/disables the texture-loading combo based on whether the
        selected type is a mesh and a texture file exists in the same directory.
        """
        self._file_path = path
        ext = os.path.splitext(path)[1].lower()
        is_mesh = False

        if ext == '.ply':
            detected = detect_ply_type(path)
            idx = {'mesh': 0, 'pointcloud': 1, 'gaussian': 2}.get(detected, 0)
            self._type_combo.setCurrentIndex(idx)
            self._type_combo.setEnabled(True)
            is_mesh = (detected == 'mesh')
        elif ext == '.pcd':
            self._type_combo.setCurrentIndex(1)
            self._type_combo.setEnabled(False)
            is_mesh = False
        elif ext in ('.stl', '.obj'):
            self._type_combo.setCurrentIndex(0)
            self._type_combo.setEnabled(False)
            is_mesh = True
        elif ext == '.vtk':
            self._type_combo.setCurrentIndex(0)
            self._type_combo.setEnabled(True)
            is_mesh = True
        else:
            self._type_combo.setCurrentIndex(0)
            self._type_combo.setEnabled(True)
            is_mesh = True

        # Update texture combo: enabled only for meshes with a texture file present
        if is_mesh:
            has_texture = find_texture_file(path)
            self._texture_combo.setEnabled(has_texture)
            # Default to True if texture exists, False otherwise
            self._texture_combo.setCurrentIndex(0 if has_texture else 1)
        else:
            self._texture_combo.setEnabled(False)
            self._texture_combo.setCurrentIndex(1)  # Default to False

    # ------------------------------------------------------------------
    # Accept / result
    # ------------------------------------------------------------------

    def _on_accept(self):
        if not self._file_path:
            return
        self.accept()

    def get_result(self):
        """Return the import parameters chosen by the user.

        Returns:
            ``(file_path, ply_type, calculate_kd_tree, sort_data, simplification_ratio, load_texture)``
            where *ply_type* is one of ``'Mesh'``, ``'Point Cloud'``, or
            ``'3D Gaussian Splatting'``, and *load_texture* is a boolean indicating
            whether to load an associated texture file (only relevant for meshes).
        """
        return (
            self._file_path,
            self._type_combo.currentText(),
            self._kd_combo.currentText() == "True",
            self._sort_combo.currentText() == "True",
            self._simplify_slider.value() / 100.0,
            self._texture_combo.currentText() == "True",
        )


class ImportModel:
    """Thin wrapper instantiated by the main window (matches other IO classes).

    Also used by MVATViewer's drag-and-drop via :pymethod:`import_model_file`.
    """

    def __init__(self, main_window):
        self.main_window = main_window

    def import_model(self):
        """Show the browse dialog and load the chosen 3D model."""
        self.import_model_file()

    def import_model_file(self, file_path: str = None) -> bool:
        """Show the import dialog (optionally pre-filled) and load the model.

        Args:
            file_path: When given the file browser is hidden and the dialog
                       opens with this file pre-selected (drag-and-drop path).

        Returns:
            True if a product was successfully loaded, False otherwise.
        """
        dialog = ImportModelDialog(self.main_window, file_path=file_path)
        if dialog.exec_() != QDialog.Accepted:
            return False

        fp, ply_type, calc_kd, sort_data, simplify, load_texture = dialog.get_result()
        if not fp:
            return False

        viewer = getattr(self.main_window, 'mvat_viewer', None)
        if viewer is None:
            return False

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            product = viewer._create_product_from_file(
                fp,
                ply_type=ply_type,
                sort_data=sort_data,
                simplification_ratio=simplify,
                load_texture=load_texture,
            )
            if product is None:
                return False

            if hasattr(product, 'prepare_geometry'):
                product.prepare_geometry()

            if calc_kd:
                manager = getattr(viewer, 'mvat_manager', None)
                if manager is not None and hasattr(manager, '_prewarm_spatial_caches'):
                    manager._prewarm_spatial_caches(product)

            viewer.add_product(product)
            viewer.render_scene()
            return True
        finally:
            QApplication.restoreOverrideCursor()
