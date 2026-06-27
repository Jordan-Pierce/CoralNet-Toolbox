import warnings

import os
from datetime import datetime

import numpy as np

import geopandas as gpd
import pandas as pd
from pyproj import CRS as PyprojCRS
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import transform as shapely_transform

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QIntValidator
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QCheckBox, QComboBox, QLineEdit, QPushButton, QFileDialog,
    QApplication, QMessageBox, QLabel, QWidget,
    QListWidget, QListWidgetItem, QTabWidget
)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_window_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Format registry
# ----------------------------------------------------------------------------------------------------------------------
# Each entry: key -> (display label, file extension, GDAL/pyogrio driver or None for parquet)
# "mixed_ok" formats can hold Points and Polygons in one file/layer; the others need
# geometry-type splitting (separate GeoPackage layers, or suffixed files).

GEO_FORMATS = {
    "geojson": ("GeoJSON", ".geojson", "GeoJSON"),
    "parquet": ("GeoParquet", ".parquet", None),
    "gpkg":    ("GeoPackage", ".gpkg", "GPKG"),
    "fgb":     ("FlatGeobuf", ".fgb", "FlatGeobuf"),
    "shp":     ("Shapefile", ".shp", "ESRI Shapefile"),
}

# Formats that happily store mixed geometry types in a single file.
MIXED_GEOMETRY_OK = {"geojson", "parquet"}

WGS84_EPSG = 4326


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ExportGeoJSONAnnotations(QDialog):
    """Export annotations to geospatial vector formats (GeoJSON, GeoParquet, GeoPackage, etc.)."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

        self.setWindowIcon(get_window_icon("coralnet.svg"))
        self.setWindowTitle("Export Annotations to Geospatial Vector")
        self.resize(420, 560)

        # Create the main layout
        self.layout = QVBoxLayout(self)

        # 1. Info Section
        self.setup_info_layout()

        # 2. Format + output mode (single vs individual) & output path
        self.setup_output_configuration_layout()

        # 3. Annotations configuration
        self.setup_annotation_layout()

        # 4. Coordinate system + extra options
        self.setup_advanced_options_layout()

        # 5. Label selection
        self.setup_label_layout()

        # 6. Action buttons
        self.setup_buttons_layout()

    def showEvent(self, event):
        """Handle the show event to refresh label list."""
        super().showEvent(event)
        self.update_label_selection_list()

    # ----------------------------------------------------------------------
    # UI Setup Methods
    # ----------------------------------------------------------------------

    def setup_info_layout(self):
        """Simple information header."""
        info_label = QLabel(
            "<b>Export Annotations to Geospatial Vector</b><br>"
            "Export annotations for rasterio-compatible images (e.g., .tif, .jpg, .png) to a GIS "
            "vector format. Georeferenced images export in real-world coordinates; non-georeferenced "
            "images can export in pixel coordinates. Choose a single merged file or one file per image."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("margin-bottom: 5px;")
        self.layout.addWidget(info_label)

    def setup_output_configuration_layout(self):
        """Setup the format selector, export mode and output file/folder selection."""
        groupbox = QGroupBox("Output Configuration")
        layout = QVBoxLayout()

        # Format selector
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Output format:"))
        self.format_combo = QComboBox()
        for key, (label, ext, _driver) in GEO_FORMATS.items():
            self.format_combo.addItem(f"{label} ({ext})", key)
        self.format_combo.currentIndexChanged.connect(self.on_format_changed)
        self.format_combo.setToolTip("Vector format for exporting annotations.\nGeoJSON: Web-friendly, compatible with GIS software.\nGeoPackage: Stores multiple layers efficiently.\nShapefile: Legacy GIS format.")
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        layout.addLayout(format_layout)

        # Tab Widget for single vs multiple
        self.tab_widget = QTabWidget()

        # Single File Tab
        single_tab = QWidget()
        single_layout = QFormLayout(single_tab)
        self.output_file_edit = QLineEdit()
        self.output_file_edit.setToolTip("Path to the output GeoJSON/GeoPackage/Shapefile.\nAll annotations will be saved to this single file.")
        self.browse_output_file_button = QPushButton("Browse...")
        self.browse_output_file_button.setToolTip("Browse for the output file location.")
        self.browse_output_file_button.clicked.connect(self.browse_output_file)
        file_field_layout = QHBoxLayout()
        file_field_layout.addWidget(self.output_file_edit)
        file_field_layout.addWidget(self.browse_output_file_button)
        single_layout.addRow("Output File:", file_field_layout)
        self.tab_widget.addTab(single_tab, "Single File")

        # Multiple Files Tab
        multi_tab = QWidget()
        multi_layout = QFormLayout(multi_tab)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setToolTip("Directory where one output file per image will be created.\nEach image's annotations will be saved separately.")
        self.browse_output_dir_button = QPushButton("Browse...")
        self.browse_output_dir_button.setToolTip("Browse for the output directory location.")
        self.browse_output_dir_button.clicked.connect(self.browse_output_dir)
        dir_field_layout = QHBoxLayout()
        dir_field_layout.addWidget(self.output_dir_edit)
        dir_field_layout.addWidget(self.browse_output_dir_button)
        multi_layout.addRow("Output Directory:", dir_field_layout)
        self.tab_widget.addTab(multi_tab, "Multiple Files")

        layout.addWidget(self.tab_widget)
        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)

    def setup_annotation_layout(self):
        """Setup annotation types in a vertical layout."""
        groupbox = QGroupBox("Annotations to Include")
        layout = QVBoxLayout()

        # Points (patch annotations) — always exported as the patch's center-most point
        self.patch_checkbox = QCheckBox("Points")
        self.patch_checkbox.setChecked(True)
        self.patch_checkbox.setToolTip(
            "Patch annotations, exported as the patch's center-most point."
        )
        layout.addWidget(self.patch_checkbox)

        # Rectangle
        self.rectangle_checkbox = QCheckBox("Rectangle")
        self.rectangle_checkbox.setChecked(True)
        self.rectangle_checkbox.setToolTip("Rectangle annotations (bounding boxes).")
        layout.addWidget(self.rectangle_checkbox)

        # Polygon
        self.polygon_checkbox = QCheckBox("Polygon")
        self.polygon_checkbox.setChecked(True)
        self.polygon_checkbox.setToolTip("Polygon annotations (free-form shapes).")
        layout.addWidget(self.polygon_checkbox)

        # MultiPolygon
        self.multipolygon_checkbox = QCheckBox("MultiPolygon")
        self.multipolygon_checkbox.setChecked(True)
        self.multipolygon_checkbox.setToolTip("MultiPolygon annotations (multiple connected polygons).")
        layout.addWidget(self.multipolygon_checkbox)

        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)

    def setup_advanced_options_layout(self):
        """Setup coordinate system selection and extra output options."""
        groupbox = QGroupBox("Options")
        layout = QVBoxLayout()

        # Coordinate system: collapses the old WGS84 / allow-pixel / force-pixel checkboxes
        # into a single, unambiguous choice.
        coord_layout = QHBoxLayout()
        coord_layout.addWidget(QLabel("Coordinate system:"))
        self.coord_combo = QComboBox()
        self.coord_combo.addItem("WGS84 (lat/lon)", "wgs84")
        self.coord_combo.addItem("Native raster CRS", "native")
        self.coord_combo.addItem("Pixel coordinates", "pixel")
        self.coord_combo.setToolTip(
            "WGS84 (lat/lon): reproject everything to EPSG:4326 for web maps (Leaflet/Mapbox).\n"
            "Native raster CRS: keep each image's own coordinate reference system.\n"
            "Pixel coordinates: ignore georeferencing and export raw image-space coordinates."
        )
        coord_layout.addWidget(self.coord_combo)
        coord_layout.addStretch()
        layout.addLayout(coord_layout)

        # EPSG override for images that are georeferenced (e.g. a .jgw world file) but
        # carry no CRS. Supplying a code lets those images be reprojected.
        epsg_layout = QHBoxLayout()
        epsg_layout.addWidget(QLabel("Assume EPSG (no-CRS images):"))
        self.epsg_edit = QLineEdit()
        self.epsg_edit.setValidator(QIntValidator(1, 999999, self))
        self.epsg_edit.setPlaceholderText("e.g. 32717  (optional)")
        self.epsg_edit.setToolTip(
            "For images that have georeferencing (e.g. a .jgw world file) but no CRS, assume this "
            "EPSG code as their coordinate reference system so they can be reprojected.\n"
            "Images that already have a CRS are unaffected. Leave blank to skip."
        )
        epsg_layout.addWidget(self.epsg_edit)
        epsg_layout.addStretch()
        layout.addLayout(epsg_layout)

        # Styling
        self.style_checkbox = QCheckBox("Include label color/style fields")
        self.style_checkbox.setChecked(True)
        self.style_checkbox.setToolTip(
            "Adds 'marker-color', 'stroke', and 'fill' fields from the label color.\n"
            "These render automatically in GeoJSON web maps; in other formats they are plain columns."
        )
        layout.addWidget(self.style_checkbox)

        # Metadata
        self.metadata_checkbox = QCheckBox("Include extra metadata (area, export timestamp)")
        self.metadata_checkbox.setChecked(True)
        self.metadata_checkbox.setToolTip(
            "Adds a pixel-area field and an export timestamp to each feature."
        )
        layout.addWidget(self.metadata_checkbox)

        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)

    def setup_label_layout(self):
        """Setup label selection using a QListWidget."""
        groupbox = QGroupBox("Filter by Label")
        layout = QVBoxLayout()

        # Tools layout (Select All/None)
        tools_layout = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.setToolTip("Select all labels for export.")
        btn_none = QPushButton("Select None")
        btn_none.setToolTip("Deselect all labels.")
        btn_all.clicked.connect(lambda: self.toggle_labels(True))
        btn_none.clicked.connect(lambda: self.toggle_labels(False))
        tools_layout.addWidget(btn_all)
        tools_layout.addWidget(btn_none)
        tools_layout.addStretch()
        layout.addLayout(tools_layout)

        # List Widget
        self.label_list_widget = QListWidget()
        layout.addWidget(self.label_list_widget)

        groupbox.setLayout(layout)
        self.layout.addWidget(groupbox)

    def setup_buttons_layout(self):
        """Standard Export/Cancel buttons."""
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.export_button = QPushButton("Export")
        self.export_button.setObjectName("primaryButton")
        self.export_button.clicked.connect(self.export_data)
        self.export_button.setMinimumWidth(120)
        self.export_button.setMinimumHeight(30)
        self.export_button.setToolTip("Export annotations to the selected vector format (GeoJSON, GeoPackage, or Shapefile).")

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setToolTip("Close this dialog without exporting.")

        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.export_button)

        self.layout.addLayout(button_layout)

    # ----------------------------------------------------------------------
    # UI Interaction Methods
    # ----------------------------------------------------------------------

    def current_format(self):
        """Return (key, label, ext, driver) for the selected output format."""
        key = self.format_combo.currentData()
        label, ext, driver = GEO_FORMATS[key]
        return key, label, ext, driver

    def current_coord_mode(self):
        """Return the selected coordinate-system mode key: 'wgs84' | 'native' | 'pixel'."""
        return self.coord_combo.currentData()

    def get_assumed_epsg(self):
        """
        Parse the EPSG override field.

        Returns (epsg_int_or_None, error_message_or_None). An empty field is valid and
        yields (None, None).
        """
        text = self.epsg_edit.text().strip()
        if not text:
            return None, None
        try:
            code = int(text)
            PyprojCRS.from_epsg(code)  # raises if the code can't be resolved
        except Exception:
            return None, f"'{text}' is not a valid EPSG code."
        return code, None

    def on_format_changed(self):
        """Keep the single-file output path's extension in sync with the chosen format."""
        _key, _label, ext, _driver = self.current_format()
        current = self.output_file_edit.text().strip()
        if current:
            base, _old_ext = os.path.splitext(current)
            self.output_file_edit.setText(base + ext)

    def browse_output_dir(self):
        """Directory chooser for output."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def browse_output_file(self):
        """File chooser for output, filtered to the selected format."""
        _key, label, ext, _driver = self.current_format()
        file_filter = f"{label} (*{ext});;All Files (*)"
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Output File", "", file_filter)
        if file_path:
            # Ensure the chosen extension is present.
            base, existing_ext = os.path.splitext(file_path)
            if existing_ext.lower() != ext:
                file_path = base + ext
            self.output_file_edit.setText(file_path)

    def update_label_selection_list(self):
        """Populate the QListWidget with labels."""
        self.label_list_widget.clear()
        for label in self.label_window.labels:
            item = QListWidgetItem(f"{label.short_label_code} - {label.long_label_code}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, label)
            self.label_list_widget.addItem(item)

    def toggle_labels(self, select_all):
        """Helper to check/uncheck all items."""
        for i in range(self.label_list_widget.count()):
            item = self.label_list_widget.item(i)
            item.setCheckState(Qt.Checked if select_all else Qt.Unchecked)

    # ----------------------------------------------------------------------
    # Logic Helper Methods
    # ----------------------------------------------------------------------

    def get_selected_labels(self):
        """Retrieve list of label objects from checked list items."""
        selected = []
        for i in range(self.label_list_widget.count()):
            item = self.label_list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))
        return selected

    def get_label_color_hex(self, label):
        """Safely extract hex color string from label object."""
        try:
            if hasattr(label, 'color'):
                c = label.color
                if isinstance(c, QColor):
                    return c.name()
                elif isinstance(c, str):
                    return c
        except Exception:
            pass
        return "#555555"  # Fallback gray

    def get_annotations_for_image(self, image_path, selected_label_codes):
        """Get filtered annotations for a specific image."""
        raw_annotations = self.annotation_window.get_image_annotations(image_path)
        if not raw_annotations:
            return []

        annotations = []
        for annotation in raw_annotations:
            if annotation.label.short_label_code not in selected_label_codes:
                continue

            if self.patch_checkbox.isChecked() and isinstance(annotation, PatchAnnotation):
                annotations.append(annotation)
            elif self.rectangle_checkbox.isChecked() and isinstance(annotation, RectangleAnnotation):
                annotations.append(annotation)
            elif self.polygon_checkbox.isChecked() and isinstance(annotation, PolygonAnnotation):
                annotations.append(annotation)
            elif self.multipolygon_checkbox.isChecked() and isinstance(annotation, MultiPolygonAnnotation):
                annotations.append(annotation)

        return annotations

    def convert_annotation_to_polygon(self, annotation):
        """Convert a rectangle/polygon annotation to a list of pixel-coordinate rings."""
        if isinstance(annotation, RectangleAnnotation):
            tl, br = annotation.top_left, annotation.bottom_right
            exterior = [(tl.x(), tl.y()), (br.x(), tl.y()),
                        (br.x(), br.y()), (tl.x(), br.y())]
            return [exterior]

        elif isinstance(annotation, PolygonAnnotation):
            rings = [[(p.x(), p.y()) for p in annotation.points]]
            for hole in annotation.holes:
                rings.append([(p.x(), p.y()) for p in hole])
            return rings

        return []

    def annotation_to_pixel_geometry(self, annotation):
        """Build a shapely geometry in pixel space for an annotation, or None if degenerate.

        Patch annotations are always exported as their center-most point.
        """
        if isinstance(annotation, PatchAnnotation):
            return Point(annotation.center_xy.x(), annotation.center_xy.y())

        if isinstance(annotation, MultiPolygonAnnotation):
            polys = []
            for poly in annotation.polygons:
                rings = self.convert_annotation_to_polygon(poly)
                if rings and len(rings[0]) >= 3:
                    polys.append(Polygon(rings[0], rings[1:]))
            return MultiPolygon(polys) if polys else None

        rings = self.convert_annotation_to_polygon(annotation)
        if not rings or len(rings[0]) < 3:
            return None
        return Polygon(rings[0], rings[1:])

    @staticmethod
    def _affine_func(transform):
        """Return a vectorized pixel->world function for a rasterio Affine transform."""
        a, b, c = transform.a, transform.b, transform.c
        d, e, f = transform.d, transform.e, transform.f

        def fn(xs, ys, zs=None):
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            wx = a * xs + b * ys + c
            wy = d * xs + e * ys + f
            if zs is not None:
                return wx, wy, zs
            return wx, wy

        return fn

    def build_feature_record(self, annotation, image_path):
        """Build the attribute dict for one annotation feature."""
        props = {
            "short_label_code": annotation.label.short_label_code,
            "long_label_code": annotation.label.long_label_code,
            "source_image": os.path.basename(image_path),
        }
        if self.style_checkbox.isChecked():
            color_hex = self.get_label_color_hex(annotation.label)
            props["marker-color"] = color_hex
            props["stroke"] = color_hex
            props["fill"] = color_hex
            props["fill-opacity"] = 0.5
            props["stroke-width"] = 2
        return props

    def build_image_gdf(self, image_path, annotations, transform, src_crs, coord_mode, assumed_epsg=None):
        """
        Build a GeoDataFrame for a single image's annotations, or None if there are none.

        Geometries are produced in the raster's native CRS (world coordinates) for the
        'wgs84' and 'native' modes, or in raw pixel coordinates for 'pixel' mode. A
        world-file image (affine transform but no CRS) still yields world coordinates;
        when it has no CRS, ``assumed_epsg`` (if given) is used as its source CRS so it
        can be reprojected.
        """
        geoms = []
        records = []
        include_metadata = self.metadata_checkbox.isChecked()
        exported_at = datetime.now().isoformat() if include_metadata else None
        affine_fn = self._affine_func(transform) if (coord_mode != "pixel" and transform is not None) else None

        for ann in annotations:
            pixel_geom = self.annotation_to_pixel_geometry(ann)
            if pixel_geom is None or pixel_geom.is_empty:
                continue

            props = self.build_feature_record(ann, image_path)
            if include_metadata:
                props["area_pixels"] = round(float(pixel_geom.area), 2)
                props["exported_at"] = exported_at

            geom = pixel_geom if affine_fn is None else shapely_transform(affine_fn, pixel_geom)
            geoms.append(geom)
            records.append(props)

        if not geoms:
            return None

        crs = None
        if coord_mode != "pixel":
            if src_crs is not None:
                crs = src_crs.to_wkt()
            elif assumed_epsg is not None:
                crs = f"EPSG:{assumed_epsg}"
        gdf = gpd.GeoDataFrame(records, geometry=geoms, crs=crs)

        # WGS84 mode: reproject as soon as we can (only possible when a CRS is known).
        if coord_mode == "wgs84" and gdf.crs is not None:
            gdf = gdf.to_crs(epsg=WGS84_EPSG)

        return gdf

    def combine_gdfs(self, gdfs, coord_mode):
        """Concatenate per-image GeoDataFrames into one, reprojecting to a common CRS."""
        gdfs = [g for g in gdfs if g is not None and len(g) > 0]
        if not gdfs:
            return None

        if coord_mode == "pixel":
            combined = pd.concat(gdfs, ignore_index=True)
            return gpd.GeoDataFrame(combined, geometry="geometry", crs=None)

        # Target CRS: WGS84 mode is already reprojected to 4326; native mode adopts
        # the first georeferenced raster's CRS so all parts share one frame.
        known_crs = [g.crs for g in gdfs if g.crs is not None]
        target = known_crs[0] if known_crs else None

        reprojected = []
        for g in gdfs:
            if target is not None and g.crs is not None and g.crs != target:
                g = g.to_crs(target)
            reprojected.append(g)

        combined = pd.concat(reprojected, ignore_index=True)
        return gpd.GeoDataFrame(combined, geometry="geometry", crs=target)

    @staticmethod
    def _geometry_families(gdf):
        """Split a GeoDataFrame into {'points': gdf, 'polygons': gdf} by geometry type."""
        geom_types = gdf.geometry.geom_type
        families = {}
        point_mask = geom_types.isin(["Point", "MultiPoint"])
        if point_mask.any():
            families["points"] = gdf[point_mask]
        poly_mask = ~point_mask
        if poly_mask.any():
            families["polygons"] = gdf[poly_mask]
        return families

    def write_gdf(self, gdf, final_path, fmt_key, driver):
        """
        Write a GeoDataFrame to disk in the chosen format.

        GeoJSON/GeoParquet store mixed geometry types directly. GeoPackage writes one
        layer per geometry family; Shapefile/FlatGeobuf write one suffixed file per family.
        Returns the list of paths actually written.
        """
        if fmt_key == "parquet":
            gdf.to_parquet(final_path)
            return [final_path]

        if fmt_key in MIXED_GEOMETRY_OK:
            gdf.to_file(final_path, driver=driver)
            return [final_path]

        families = self._geometry_families(gdf)
        if len(families) <= 1:
            gdf.to_file(final_path, driver=driver)
            return [final_path]

        written = []
        if fmt_key == "gpkg":
            for family, sub in families.items():
                sub.to_file(final_path, driver=driver, layer=family)
                written.append(f"{final_path} [{family}]")
        else:  # shp, fgb: split into separate files
            base, ext = os.path.splitext(final_path)
            for family, sub in families.items():
                part_path = f"{base}_{family}{ext}"
                sub.to_file(part_path, driver=driver)
                written.append(part_path)
        return written

    # ----------------------------------------------------------------------
    # Main Export Execution
    # ----------------------------------------------------------------------

    def validate_inputs(self):
        """Check if output paths and selections are valid."""
        mode = "single" if self.tab_widget.currentIndex() == 0 else "individual"

        if mode == "single":
            if not self.output_file_edit.text().strip():
                QMessageBox.warning(self, "Output Error", "Please select an output file.")
                return False
        else:
            if not self.output_dir_edit.text().strip():
                QMessageBox.warning(self, "Output Error", "Please select an output directory.")
                return False

        if not any([self.patch_checkbox.isChecked(), self.rectangle_checkbox.isChecked(),
                    self.polygon_checkbox.isChecked(), self.multipolygon_checkbox.isChecked()]):
            QMessageBox.warning(self, "Selection Error", "Select at least one annotation type.")
            return False

        if not self.get_selected_labels():
            QMessageBox.warning(self, "Selection Error", "Select at least one label.")
            return False

        return True

    def export_data(self):
        """Main export execution method."""
        if not self.validate_inputs():
            return

        mode = "single" if self.tab_widget.currentIndex() == 0 else "individual"
        fmt_key, fmt_label, fmt_ext, fmt_driver = self.current_format()
        coord_mode = self.current_coord_mode()

        assumed_epsg, epsg_error = self.get_assumed_epsg()
        if epsg_error:
            QMessageBox.warning(self, "Invalid EPSG", epsg_error)
            return

        selected_label_objects = self.get_selected_labels()
        selected_label_codes = [label.short_label_code for label in selected_label_objects]

        # Resolve output target
        if mode == "single":
            final_output_path = self.output_file_edit.text().strip()
            base, ext = os.path.splitext(final_output_path)
            if ext.lower() != fmt_ext:
                final_output_path = base + fmt_ext
        else:
            final_output_dir = self.output_dir_edit.text().strip()
            if not os.path.exists(final_output_dir):
                try:
                    os.makedirs(final_output_dir)
                except OSError as e:
                    QMessageBox.critical(self, "Error", f"Could not create directory: {e}")
                    return

        all_images = self.image_window.raster_manager.image_paths

        # Pre-flight: WGS84 reprojection needs a source CRS. World-file images (e.g. .jgw)
        # carry an affine transform but no CRS, so they cannot be reprojected. Surface this
        # BEFORE writing anything and let the user decide, rather than warning after the fact.
        # An assumed EPSG supplies the missing CRS, so the warning is unnecessary then.
        if coord_mode == "wgs84" and assumed_epsg is None:
            no_crs_count = 0
            for image_path in all_images:
                if not self.get_annotations_for_image(image_path, selected_label_codes):
                    continue
                raster = self.image_window.raster_manager.get_raster(image_path)
                if raster and raster.rasterio_src and raster.rasterio_src.crs is None:
                    no_crs_count += 1
            if no_crs_count:
                resp = QMessageBox.question(
                    self, "Cannot Reproject to WGS84",
                    f"{no_crs_count} image(s) have georeferencing (e.g., a world file) but no CRS, "
                    "so their coordinates cannot be reprojected to WGS84. They will be exported in "
                    "their native world/raster coordinates instead.\n\n"
                    "Tip: add a .prj sidecar, or choose 'Native raster CRS', to avoid this.\n\n"
                    "Continue with the export?",
                    QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Yes
                )
                if resp != QMessageBox.Yes:
                    return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, "Exporting Geospatial Vector")
        progress_bar.show()
        progress_bar.start_progress(len(all_images))

        combined_gdfs = []  # for single mode
        exported_count = 0
        skipped_no_annotations = 0
        skipped_no_raster = 0
        skipped_no_features = 0

        try:
            for image_path in all_images:
                # 1. Filter annotations
                annotations = self.get_annotations_for_image(image_path, selected_label_codes)
                if not annotations:
                    skipped_no_annotations += 1
                    progress_bar.update_progress()
                    continue

                # 2. Load raster
                raster = self.image_window.raster_manager.get_raster(image_path)
                if not raster or not raster.rasterio_src:
                    skipped_no_raster += 1
                    progress_bar.update_progress()
                    continue

                transform = raster.rasterio_src.transform
                src_crs = raster.rasterio_src.crs

                # 3. Build GeoDataFrame for this image
                gdf = self.build_image_gdf(
                    image_path, annotations, transform, src_crs, coord_mode, assumed_epsg
                )
                if gdf is None or len(gdf) == 0:
                    skipped_no_features += 1
                    progress_bar.update_progress()
                    continue

                # 4. Write (individual) or accumulate (single)
                if mode == "single":
                    combined_gdfs.append(gdf)
                    exported_count += 1
                else:
                    filename = os.path.splitext(os.path.basename(image_path))[0] + fmt_ext
                    out_file = os.path.join(final_output_dir, filename)
                    self.write_gdf(gdf, out_file, fmt_key, fmt_driver)
                    exported_count += 1

                progress_bar.update_progress()

            # 5. Final write for single mode
            feature_count = 0
            if mode == "single":
                combined = self.combine_gdfs(combined_gdfs, coord_mode)
                if combined is None or len(combined) == 0:
                    QApplication.restoreOverrideCursor()
                    QMessageBox.warning(self, "Export", "No features to export.")
                    return
                feature_count = len(combined)
                self.write_gdf(combined, final_output_path, fmt_key, fmt_driver)

            # 6. Summary
            QApplication.restoreOverrideCursor()

            if fmt_key == "shp":
                QMessageBox.information(
                    self, "Shapefile Note",
                    "Shapefile field names are truncated to 10 characters and a single file holds "
                    "only one geometry type. Mixed point/polygon exports were split into "
                    "'_points' and '_polygons' files. Consider GeoPackage or GeoParquet to avoid this."
                )

            feature_summary = f"{feature_count} features" if mode == "single" else "individual files"
            QMessageBox.information(
                self, "Export Completed",
                f"Exported {exported_count} images ({feature_summary}) as {fmt_label}.\n"
                f"Skipped: {skipped_no_annotations} (no annotations), "
                f"{skipped_no_raster} (no raster), {skipped_no_features} (no valid features)."
            )
            self.accept()

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"An error occurred during export: {str(e)}")
            import traceback
            traceback.print_exc()

        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.finish_progress()
            progress_bar.stop_progress()
            progress_bar.close()
