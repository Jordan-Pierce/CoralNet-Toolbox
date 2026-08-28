import warnings

import os

import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
                             QLineEdit, QPushButton, QFileDialog, QApplication, QMessageBox,
                             QLabel, QWidget, QListWidget, QListWidgetItem, QTabWidget)

from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation

from coralnet_toolbox.MetaData.QtMetrics import ALL_METRICS
from coralnet_toolbox.MetaData.QtMetrics import METRIC_CATEGORIES
from coralnet_toolbox.MetaData.QtMetrics import calculate_metrics_for_annotation

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_window_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------


# Metrics that need the raster's z-channel loaded before they can be computed.
Z_METRICS = ('volume', 'surface_area', 'min_z', 'max_z')


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ExportMetadataTable(QDialog):
    """Export a table of annotation metadata to CSV, one row per annotation.

    Covers both halves of an annotation's metadata: the built-in values derived
    from its geometry (area, perimeter, morphology, 3D) and the custom fields
    the user defined in the Metadata panel. Both are offered in a single column
    list, since from the user's point of view they are all just metadata.

    This replaces the former standalone Spatial Metrics export, which produced
    the same columns minus the custom ones -- the identity and metric columns
    are unchanged, so existing downstream scripts keep working.
    """

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_window_icon("coralnet.svg"))
        self.setWindowTitle("Export Metadata (CSV)")
        self.resize(500, 600)

        self.layout = QVBoxLayout(self)

        # 1. Info Section
        self.setup_info_layout()

        # 2. File Path Selection
        self.setup_file_path_layout()

        # 3. Column Selection (Tabbed)
        self.setup_column_selection_layout()

        # 4. Action Buttons
        self.setup_buttons_layout()

    def showEvent(self, event):
        """Handle the show event to refresh lists."""
        super().showEvent(event)
        self.update_images_list()
        self.update_labels_list()
        self.update_columns_list()

    # ----------------------------------------------------------------------
    # UI Setup Methods
    # ----------------------------------------------------------------------

    def setup_info_layout(self):
        """Simple information header."""
        info_label = QLabel(
            "<b>Export Metadata to CSV</b><br>"
            "Write one row per annotation, combining the built-in values derived from "
            "its geometry (area, perimeter, morphology, 3D) with the custom fields you "
            "defined. Filter by images, labels, and annotation types, and choose which "
            "columns to include."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("margin-bottom: 5px;")
        self.layout.addWidget(info_label)

    def setup_file_path_layout(self):
        """Setup file path selection."""
        groupbox = QGroupBox("Output File")
        layout = QFormLayout()

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setToolTip("Path where the metadata CSV file will be saved.")
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_file_path)
        self.browse_button.setToolTip("Browse for the output CSV file location.")

        file_layout = QHBoxLayout()
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.browse_button)

        layout.addRow("File Path:", file_layout)
        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)

    def _make_list_tab(self, title, tooltip):
        """Build a tab holding one extended-selection list plus its buttons."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        list_widget.setToolTip(tooltip)
        layout.addWidget(list_widget)

        buttons = QHBoxLayout()
        select_all = QPushButton("Select All")
        select_all.clicked.connect(lambda: self.select_all_in(list_widget))
        select_all.setToolTip("Select every entry.")
        deselect_all = QPushButton("Deselect All")
        deselect_all.clicked.connect(list_widget.clearSelection)
        deselect_all.setToolTip("Deselect every entry.")
        buttons.addWidget(select_all)
        buttons.addWidget(deselect_all)
        layout.addLayout(buttons)

        self.tab_widget.addTab(tab, title)
        return list_widget

    def setup_column_selection_layout(self):
        """Setup tabbed filters and column selection."""
        groupbox = QGroupBox("Column Selection")
        layout = QVBoxLayout()

        self.tab_widget = QTabWidget()

        self.images_list = self._make_list_tab(
            "Images", "Select which images to include in the export.")
        self.labels_list = self._make_list_tab(
            "Labels", "Select which annotation labels to include in the export.")

        self.types_list = self._make_list_tab(
            "Annotation Types",
            "Select which annotation types to include.\n"
            "Patch: Point circles.\nRectangle: Bounding boxes.\n"
            "Polygon: Free-form shapes.\nMultiPolygon: Multiple connected polygons.")
        self.types_list.addItems([
            "PatchAnnotation",
            "RectangleAnnotation",
            "PolygonAnnotation",
            "MultiPolygonAnnotation"
        ])
        self.types_list.selectAll()

        self.columns_list = self._make_list_tab(
            "Columns", "Select which metadata columns to include in the export.")

        layout.addWidget(self.tab_widget)
        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)

    def setup_buttons_layout(self):
        """Setup export and cancel buttons."""
        button_layout = QHBoxLayout()

        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.export_metadata_table)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.cancel_button)

        self.layout.addLayout(button_layout)

    # ----------------------------------------------------------------------
    # List Population Methods
    # ----------------------------------------------------------------------

    def update_images_list(self):
        """Populate the images list from the image window."""
        self.images_list.clear()
        if hasattr(self.image_window, 'raster_manager'):
            for image_path in self.image_window.raster_manager.image_paths:
                item = QListWidgetItem(os.path.basename(image_path))
                item.setData(Qt.UserRole, image_path)  # Store full path
                self.images_list.addItem(item)
        self.images_list.selectAll()

    def update_labels_list(self):
        """Populate the labels list from the label window."""
        self.labels_list.clear()
        if hasattr(self.label_window, 'labels'):
            for label in self.label_window.labels:
                item = QListWidgetItem(f"{label.short_label_code} - {label.long_label_code}")
                item.setData(Qt.UserRole, label.short_label_code)  # Store short code
                self.labels_list.addItem(item)
        self.labels_list.selectAll()

    def add_header(self, title):
        """Add a non-selectable category separator."""
        item = QListWidgetItem(f"── {title} ──")
        item.setFlags(Qt.NoItemFlags)
        self.columns_list.addItem(item)

    def update_columns_list(self):
        """Populate the combined column list: built-in metrics, then custom fields.

        Built-in and custom entries live in one list because the distinction is
        an implementation detail -- to the user they are all just metadata. The
        stored role keeps them apart when the row is built.
        """
        self.columns_list.clear()

        for category, metrics in METRIC_CATEGORIES.items():
            self.add_header(category)
            for metric in metrics:
                item = QListWidgetItem(metric)
                item.setData(Qt.UserRole, ('metric', metric))
                self.columns_list.addItem(item)

        self.add_header("Custom Fields")
        schema = self.get_schema()
        if schema is None or not len(schema):
            placeholder = QListWidgetItem("(none defined)")
            placeholder.setFlags(Qt.NoItemFlags)
            self.columns_list.addItem(placeholder)
        else:
            for field in schema:
                item = QListWidgetItem(field.label)
                item.setData(Qt.UserRole, ('custom', field.name))
                if field.description:
                    item.setToolTip(field.description)
                self.columns_list.addItem(item)

        self.select_all_in(self.columns_list)

    # ----------------------------------------------------------------------
    # Helper Methods
    # ----------------------------------------------------------------------

    def get_schema(self):
        """Return the project's metadata schema, or None when unavailable."""
        try:
            return self.main_window.metadata_window.schema
        except AttributeError:
            return None

    def select_all_in(self, list_widget):
        """Select every selectable row, skipping the category headers."""
        for index in range(list_widget.count()):
            item = list_widget.item(index)
            if item.flags() & Qt.ItemIsSelectable:
                item.setSelected(True)

    def browse_file_path(self):
        """Open file dialog to select output CSV path."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Metadata CSV",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )
        if file_path:
            if not file_path.lower().endswith('.csv'):
                file_path += '.csv'
            self.file_path_edit.setText(file_path)

    def _selected_data(self, list_widget, role=Qt.UserRole):
        """Return the stored value of each selected row, or all rows if none."""
        selected = list_widget.selectedItems()
        if not selected:
            # If none selected, fall back to everything selectable.
            return [list_widget.item(i).data(role)
                    for i in range(list_widget.count())
                    if list_widget.item(i).flags() & Qt.ItemIsSelectable]
        return [item.data(role) for item in selected]

    def get_selected_images(self):
        """Get list of selected image paths."""
        return self._selected_data(self.images_list)

    def get_selected_labels(self):
        """Get list of selected label short codes."""
        return self._selected_data(self.labels_list)

    def get_selected_types(self):
        """Get list of selected annotation types."""
        selected = self.types_list.selectedItems()
        if not selected:
            return [self.types_list.item(i).text() for i in range(self.types_list.count())]
        return [item.text() for item in selected]

    def _selected_columns(self, kind):
        """Return the selected column names of one kind, in list order."""
        names = []
        for index in range(self.columns_list.count()):
            item = self.columns_list.item(index)
            data = item.data(Qt.UserRole)
            if data and data[0] == kind and item.isSelected():
                names.append(data[1])
        return names

    def get_selected_metrics(self):
        """Get list of selected built-in metric names."""
        return [name for name in self._selected_columns('metric') if name in ALL_METRICS]

    def get_selected_fields(self):
        """Get the selected custom metadata field names."""
        schema = self.get_schema()
        if schema is None:
            return []
        return [name for name in self._selected_columns('custom') if schema.has_field(name)]

    # ----------------------------------------------------------------------
    # Row Building
    # ----------------------------------------------------------------------

    def build_row(self, annotation, parent_id, annotation_type, metrics, fields,
                  schema, z_data):
        """Build one CSV row for an annotation."""
        # Base columns (always included)
        row = {
            'annotation_id': annotation.id,
            'parent_annotation_id': parent_id,
            'image_path': annotation.image_path,
            'image_name': os.path.basename(annotation.image_path or ''),
            'annotation_type': annotation_type,
            'label_short': annotation.label.short_label_code,
            'label_long': annotation.label.long_label_code,
            'color_rgb': str(annotation.label.color.getRgb()[:3]),
        }

        if metrics:
            row.update(calculate_metrics_for_annotation(
                annotation, metrics,
                z_data.get('z_channel'), z_data.get('z_unit'),
                z_nodata=z_data.get('z_nodata'), z_data_type=z_data.get('z_data_type')))

        for name in fields:
            field = schema.get_field(name)
            if field is None:
                continue
            value = schema.get_value(annotation, name)
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)
            row[field.label] = value

        return row

    # ----------------------------------------------------------------------
    # Export Method
    # ----------------------------------------------------------------------

    def export_metadata_table(self):
        """Export the metadata table to CSV."""
        file_path = self.file_path_edit.text()
        if not file_path:
            QMessageBox.warning(self, "No File Selected", "Please select an output file path.")
            return

        selected_images = self.get_selected_images()
        selected_labels = self.get_selected_labels()
        selected_types = self.get_selected_types()
        selected_metrics = self.get_selected_metrics()
        selected_fields = self.get_selected_fields()

        if not selected_images:
            QMessageBox.warning(self, "No Images", "Please select at least one image.")
            return
        if not selected_labels:
            QMessageBox.warning(self, "No Labels", "Please select at least one label.")
            return
        if not selected_types:
            QMessageBox.warning(self, "No Types", "Please select at least one annotation type.")
            return
        if not selected_metrics and not selected_fields:
            QMessageBox.warning(self, "No Columns",
                                "Please select at least one column to export.")
            return

        schema = self.get_schema()

        # Set cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # Filter annotations
            filtered_annotations = []
            for annotation in self.annotation_window.annotations_dict.values():
                if getattr(annotation, 'is_mask_annotation', False):
                    continue
                if type(annotation).__name__ not in selected_types:
                    continue
                if annotation.image_path not in selected_images:
                    continue
                if annotation.label.short_label_code not in selected_labels:
                    continue
                filtered_annotations.append(annotation)

            if not filtered_annotations:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(self, "No Annotations",
                                    "No annotations match the selected filters.")
                return

            # Start progress bar
            progress_bar = ProgressBar(self, "Exporting Metadata")
            progress_bar.show()
            progress_bar.start_progress(len(filtered_annotations))

            # Cache z-channel data per image to avoid repeated lookups
            z_channel_cache = {}
            needs_z = any(metric in selected_metrics for metric in Z_METRICS)

            rows = []
            for annotation in filtered_annotations:
                if progress_bar.wasCanceled():
                    break

                z_data = {}
                if needs_z:
                    if annotation.image_path not in z_channel_cache:
                        raster = self.image_window.raster_manager.get_raster(annotation.image_path)
                        if raster:
                            z_channel_cache[annotation.image_path] = {
                                'z_channel': raster.z_channel_lazy,
                                'z_unit': raster.z_unit,
                                'z_nodata': raster.z_nodata,
                                'z_data_type': raster.z_data_type
                            }
                        else:
                            z_channel_cache[annotation.image_path] = {}
                    z_data = z_channel_cache.get(annotation.image_path, {})

                # A MultiPolygon is a container; its constituents carry the
                # geometry, so each becomes its own row pointing at the parent.
                if isinstance(annotation, MultiPolygonAnnotation):
                    for polygon in annotation.polygons:
                        rows.append(self.build_row(polygon, annotation.id, 'Polygon',
                                                   selected_metrics, selected_fields,
                                                   schema, z_data))
                else:
                    rows.append(self.build_row(
                        annotation, None,
                        type(annotation).__name__.replace('Annotation', ''),
                        selected_metrics, selected_fields, schema, z_data))

                progress_bar.update_progress()

            # Create DataFrame and export
            if rows:
                pd.DataFrame(rows).to_csv(file_path, index=False)

            progress_bar.stop_progress()
            progress_bar.close()

            QApplication.restoreOverrideCursor()

            message = f"Exported metadata for {len(rows)} annotations."
            try:
                self.main_window.status_bar.showMessage(message, 5000)
            except Exception:
                pass

            QMessageBox.information(
                self, "Export Complete",
                f"Successfully exported {len(rows)} annotations to:\n{os.path.basename(file_path)}"
            )
            self.accept()

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Export Error", f"An error occurred during export:\n{str(e)}")
