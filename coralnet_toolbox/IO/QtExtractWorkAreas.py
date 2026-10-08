import os
import uuid

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import box
from shapely.strtree import STRtree

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDialog,
                             QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
                             QLabel, QLineEdit, QMessageBox, QPushButton, QTableWidget,
                             QTableWidgetItem, QVBoxLayout)

from coralnet_toolbox.Annotations.QtMaskAnnotation import MaskAnnotation
from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation
from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.annotation_clipping import (annotation_dict_bounds, annotation_fingerprint,
                                                              clip_annotation_dict, shift_annotation_dict,
                                                              SUPPORTED_TYPES)
from coralnet_toolbox.IO.QtImportFrames import _get_write_params
from coralnet_toolbox.Rasters import extracted_images
from coralnet_toolbox.Rasters.merge_back import mask_base_path, save_mask_base, stamp_copy
from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_window_icon
from coralnet_toolbox.utilities import raster_display_max, read_window_rgb


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

IMAGE_FORMATS = ['tif', 'png', 'jpg']

# Rebuilds copied annotations from their dicts
ANNOTATION_CLASSES = {
    'PatchAnnotation': PatchAnnotation,
    'RectangleAnnotation': RectangleAnnotation,
    'PolygonAnnotation': PolygonAnnotation,
    'MultiPolygonAnnotation': MultiPolygonAnnotation,
}

# Worker result statuses
WRITTEN = 'written'
REUSED = 'reused'
EMPTY = 'empty'


# ----------------------------------------------------------------------------------------------------------------------
# Extraction Worker
# ----------------------------------------------------------------------------------------------------------------------


def _is_georeferenced(src):
    transform = src.transform
    return transform is not None and not transform.is_identity


def _is_empty(src, window, rgb=None):
    """No data anywhere in the window (by the dataset's mask), or one flat color."""
    if rgb is None:
        return not src.read_masks(1, window=window).any()
    return bool((rgb == rgb[0, 0]).all())


def _write_tile(path, rgb, src, window, image_format, georeferenced):
    """Write one tile, complete or not at all, so an interrupted run never leaves a file that looks reusable."""
    directory, name = os.path.split(path)
    stem, extension = os.path.splitext(name)
    partial_path = os.path.join(directory, f"{stem}.partial{extension}")
    os.makedirs(directory, exist_ok=True)

    if image_format == 'tif' and georeferenced:
        height, width = rgb.shape[:2]
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 3,
            'dtype': 'uint8',
            'crs': src.crs,
            'transform': window_transform(window, src.transform),
            'compress': 'deflate',
            'photometric': 'RGB',
        }
        with rasterio.open(partial_path, 'w', **profile) as dst:
            dst.write(np.transpose(rgb, (2, 0, 1)))
    else:
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        if not cv2.imwrite(partial_path, bgr, _get_write_params(image_format)):
            raise IOError(f"Could not write {path}")

    os.replace(partial_path, path)


class WorkAreaExtractorThread(QThread):
    """Write each work area of each raster as an image, off the GUI thread.

    Opens its own dataset per raster: a Raster's dataset is shared with the
    GUI thread, and rasterio datasets are not thread-safe.
    """
    progress_updated = pyqtSignal(int)
    extraction_completed = pyqtSignal(list)
    extraction_error = pyqtSignal(str)

    def __init__(self, jobs, image_format, skip_empty):
        """
        Args:
            jobs: list of {'parent_path', 'output_dir', 'prefix', 'rects'}.
            image_format: 'tif', 'png' or 'jpg'.
            skip_empty: leave out tiles with no data, or one flat color.
        """
        super().__init__()
        self.jobs = jobs
        self.image_format = image_format
        self.skip_empty = skip_empty
        self.canceled = False

    def cancel(self):
        self.canceled = True

    def run(self):
        results = []
        done = 0
        try:
            for job in self.jobs:
                if self.canceled:
                    break
                with rasterio.open(job['parent_path']) as src:
                    georeferenced = _is_georeferenced(src)
                    # One brightness scale for every tile of this raster, computed
                    # only once a tile actually has to be read
                    max_value = None
                    max_value_known = False

                    for rect in job['rects']:
                        if self.canceled:
                            break

                        name = extracted_images.tile_filename(job['prefix'], rect, self.image_format)
                        path = extracted_images.normalize_path(os.path.join(job['output_dir'], name))
                        window = Window(*rect)

                        if os.path.exists(path):
                            status = REUSED
                        elif self.skip_empty and _is_empty(src, window):
                            status = EMPTY
                        else:
                            if not max_value_known:
                                max_value = raster_display_max(src)
                                max_value_known = True
                            rgb = read_window_rgb(src, window, max_value)
                            if self.skip_empty and _is_empty(src, window, rgb):
                                status = EMPTY
                            else:
                                _write_tile(path, rgb, src, window, self.image_format, georeferenced)
                                status = WRITTEN

                        results.append({
                            'parent_path': job['parent_path'],
                            'rect': rect,
                            'path': None if status == EMPTY else path,
                            'status': status,
                        })
                        done += 1
                        self.progress_updated.emit(done)

            self.extraction_completed.emit(results)

        except Exception as e:
            self.extraction_error.emit(str(e))


# ----------------------------------------------------------------------------------------------------------------------
# Extract Work Areas Dialog
# ----------------------------------------------------------------------------------------------------------------------


class ExtractWorkAreas(QDialog):
    """Save each work area of the highlighted rasters as its own image, and optionally import them.

    Laid out like Extract Frames. Extract and Import links the images to the
    raster they came from (see Rasters/extracted_images.py), and can copy the
    raster's annotations onto them, clipped to each work area.
    """

    def __init__(self, main_window, raster_paths, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window
        self.label_window = main_window.label_window
        self.raster_manager = self.image_window.raster_manager

        self.setWindowIcon(get_window_icon("tile.svg"))
        self.setWindowTitle("Extract Work Areas")
        self.resize(650, 650)

        self.rows = self._build_rows(raster_paths)
        self.eligible_rows = [row for row in self.rows if row['reason'] is None]

        self.worker = None
        self.progress = None

        layout = QVBoxLayout(self)
        layout.addWidget(self.create_info_group())
        layout.addWidget(self.create_source_group())
        layout.addWidget(self.create_output_group())
        layout.addWidget(self.create_options_group())
        layout.addLayout(self.create_buttons_layout())

        self.update_georeference_note()

    # ------------------------------------------------------------------
    # Source rasters
    # ------------------------------------------------------------------

    def _build_rows(self, raster_paths):
        """One row per highlighted raster, with why it is skipped or what it will produce."""
        rows = []
        for path in raster_paths:
            raster = self.raster_manager.get_raster(path)
            if raster is None:
                continue
            reason = extracted_images.extraction_block_reason(raster)
            rects = []
            overlap_count, max_overlap = 0, 0
            if reason is None:
                rects = extracted_images.work_area_rects(raster.get_work_areas(), raster.width, raster.height)
                overlap_count, max_overlap = extracted_images.overlap_summary(rects)
            rows.append({
                'path': raster.image_path,
                'raster': raster,
                'reason': reason,
                'rects': rects,
                'overlap_count': overlap_count,
                'max_overlap': max_overlap,
                'georeferenced': self._raster_is_georeferenced(raster),
            })
        return rows

    @staticmethod
    def _raster_is_georeferenced(raster):
        try:
            return _is_georeferenced(raster.rasterio_src)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # UI Group Builders
    # ------------------------------------------------------------------

    def create_info_group(self):
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()
        info_label = QLabel(
            "Saves each work area as its own image. Extract writes the images to disk. "
            "Extract and Import also adds them to this project, linked to the raster they came from."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        group_box.setLayout(layout)
        return group_box

    def create_source_group(self):
        group_box = QGroupBox("Source")
        layout = QVBoxLayout()

        self.source_table = QTableWidget(len(self.rows), 4)
        self.source_table.setHorizontalHeaderLabels(["Raster", "Type", "Work Areas", "Status"])
        self.source_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.source_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.source_table.verticalHeader().setVisible(False)
        # Long names are cut off with an ellipsis rather than wrapped; the
        # tooltip has the full text
        self.source_table.setWordWrap(False)
        self.source_table.setTextElideMode(Qt.ElideRight)
        header = self.source_table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignCenter)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)

        for row_index, row in enumerate(self.rows):
            raster = row['raster']
            if row['reason'] is None:
                count = len(row['rects'])
                status = f"{count} image{'s' if count != 1 else ''}"
            else:
                status = f"Skipped: {row['reason']}"
            work_area_count = len(raster.get_work_areas()) if row['reason'] != extracted_images.REASON_NOT_IMAGE else 0
            items = [os.path.basename(row['path']), raster.raster_type, str(work_area_count), status]
            for column, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                item.setToolTip(row['path'] if column == 0 else text)
                self.source_table.setItem(row_index, column, item)

        layout.addWidget(self.source_table)

        group_box.setLayout(layout)
        return group_box

    def create_output_group(self):
        group_box = QGroupBox("Output")
        layout = QFormLayout()

        self.output_dir_edit = QLineEdit()
        if self.eligible_rows:
            self.output_dir_edit.setText(os.path.dirname(self.eligible_rows[0]['path']))
        self.output_dir_edit.setToolTip("Folder to save into. Each raster's images go in a subfolder named after it.")
        self.output_dir_button = QPushButton("Browse...")
        self.output_dir_button.clicked.connect(self.browse_output_dir)
        self.output_dir_button.setToolTip("Browse for an output folder.")
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_dir_button)
        layout.addRow("Output Folder:", output_dir_layout)

        # One raster: the prefix is editable. Several: each uses its own file name.
        self.prefix_edit = QLineEdit()
        if len(self.eligible_rows) == 1:
            self.prefix_edit.setText(os.path.splitext(os.path.basename(self.eligible_rows[0]['path']))[0])
        else:
            self.prefix_edit.setEnabled(False)
            self.prefix_edit.setPlaceholderText("Each raster's file name")
        self.prefix_edit.setToolTip("Images are named prefix_x_y_width_height, from where each work area sits.")
        layout.addRow("Prefix:", self.prefix_edit)

        self.format_combo = QComboBox()
        self.format_combo.addItems(IMAGE_FORMATS)
        self.format_combo.setToolTip(
            "tif keeps the georeference of a georeferenced raster.\n"
            "png is lossless. jpg is smallest, with some loss of quality."
        )
        self.format_combo.currentIndexChanged.connect(self.update_georeference_note)
        layout.addRow("Format:", self.format_combo)

        self.georeference_note = QLabel("png and jpg do not keep the georeference.")
        self.georeference_note.setWordWrap(True)
        layout.addRow("", self.georeference_note)

        group_box.setLayout(layout)
        return group_box

    def create_options_group(self):
        group_box = QGroupBox("Options")
        layout = QFormLayout()

        self.skip_empty_combo = QComboBox()
        self.skip_empty_combo.addItems(["True", "False"])
        self.skip_empty_combo.setToolTip(
            "If True, leave out work areas with no data, such as the blank corners of an orthomosaic,\n"
            "or that are a single flat color."
        )
        layout.addRow("Skip Empty Tiles:", self.skip_empty_combo)

        self.include_annotations_combo = QComboBox()
        self.include_annotations_combo.addItems(["True", "False"])
        self.include_annotations_combo.setToolTip(
            "If True, Extract and Import copies the raster's annotations and mask onto the new\n"
            "images, clipped to each work area. The originals are not changed."
        )
        layout.addRow("Include Annotations:", self.include_annotations_combo)

        group_box.setLayout(layout)
        return group_box

    def create_buttons_layout(self):
        buttons_layout = QHBoxLayout()

        self.extract_button = QPushButton("Extract")
        self.extract_button.setToolTip("Save the images to the output folder without adding them to the project.")
        self.extract_button.clicked.connect(lambda: self.extract(import_after=False))

        self.extract_import_button = QPushButton("Extract and Import")
        self.extract_import_button.setToolTip(
            "Save the images and add them to this project, linked to the raster they came from."
        )
        self.extract_import_button.clicked.connect(lambda: self.extract(import_after=True))

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        for button in (self.extract_button, self.extract_import_button, self.cancel_button):
            buttons_layout.addWidget(button)

        enabled = bool(self.eligible_rows)
        self.extract_button.setEnabled(enabled)
        self.extract_import_button.setEnabled(enabled)
        return buttons_layout

    def update_georeference_note(self):
        georeferenced = any(row['georeferenced'] for row in self.eligible_rows)
        self.georeference_note.setVisible(georeferenced and self.format_combo.currentText() != 'tif')

    def browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.output_dir_edit.text())
        if directory:
            self.output_dir_edit.setText(directory)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, import_after):
        """Check, confirm, then extract on a background thread."""
        if not self.eligible_rows:
            return

        output_root = self.output_dir_edit.text().strip()
        if not output_root:
            QMessageBox.warning(self, "No Output Folder", "Please choose an output folder.")
            return

        if import_after and not self._confirm_import_allowed():
            return

        image_format = self.format_combo.currentText()
        jobs = self._build_jobs(output_root)

        # Confirm, saying how much is already on disk
        total = sum(len(job['rects']) for job in jobs)
        existing = sum(
            1 for job in jobs for rect in job['rects']
            if os.path.exists(os.path.join(job['output_dir'],
                                           extracted_images.tile_filename(job['prefix'], rect, image_format)))
        )
        raster_count = len(jobs)
        message = f"Extract {total} images from {raster_count} raster{'s' if raster_count != 1 else ''}?\n\n"
        if existing:
            message += f"({existing} already exist on disk and will be reused.)\n\n"
        message += f"Files will be named: prefix_x_y_width_height.{image_format}"
        reply = QMessageBox.question(self, "Confirm Extraction", message,
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply != QMessageBox.Yes:
            return

        self.progress = ProgressBar(self.annotation_window, "Extracting Work Areas")
        self.progress.show()
        self.progress.start_progress(total)
        QApplication.setOverrideCursor(Qt.WaitCursor)

        self.worker = WorkAreaExtractorThread(jobs, image_format, self.skip_empty_combo.currentText() == "True")
        self.worker.progress_updated.connect(self._on_progress)
        # Deferred out of the signal dispatch: finishing pumps events through
        # progress bars, which must not run nested inside a worker's slot
        self.worker.extraction_completed.connect(
            lambda results: QTimer.singleShot(0, lambda: self._finish(results, jobs, import_after, image_format))
        )
        self.worker.extraction_error.connect(self._on_error)
        self.worker.start()

    def _confirm_import_allowed(self):
        """Warn about overlap before creating a set. True to go ahead.

        A raster that already has a set never gets here: extraction_block_reason
        marks it "already extracted", so it is skipped rather than eligible.
        """
        overlapping = [row for row in self.eligible_rows if row['overlap_count']]
        if overlapping:
            max_overlap = max(row['max_overlap'] for row in overlapping)
            message_box = QMessageBox(QMessageBox.Warning, "Overlapping Work Areas",
                                      f"Some work areas overlap, by up to {max_overlap} px. Extracting works, "
                                      "but merging these images back later will ask you about each annotation "
                                      "in the overlapping strips. To avoid that, use Smaller edge tiles with no "
                                      "overlap in the Work Area Manager.", parent=self)
            continue_button = message_box.addButton("Continue", QMessageBox.AcceptRole)
            message_box.addButton("Cancel", QMessageBox.RejectRole)
            message_box.exec_()
            if message_box.clickedButton() is not continue_button:
                return False

        return True

    def _build_jobs(self, output_root):
        single = len(self.eligible_rows) == 1
        jobs = []
        for row in self.eligible_rows:
            stem = os.path.splitext(os.path.basename(row['path']))[0]
            prefix = (self.prefix_edit.text().strip() or stem) if single else stem
            jobs.append({
                'parent_path': row['path'],
                'output_dir': extracted_images.normalize_path(os.path.join(output_root, stem)),
                'prefix': prefix,
                'rects': row['rects'],
                'row': row,
            })
        return jobs

    def _on_progress(self, value):
        """Runs on the GUI thread. set_value does not pump events, so this stays safe as a queued slot."""
        if self.progress is None:
            return
        self.progress.set_value(value)
        if self.progress.wasCanceled() and self.worker is not None:
            self.worker.cancel()

    def _on_error(self, message):
        self._close_progress()
        QMessageBox.critical(self, "Extraction Error", f"An error occurred while extracting work areas:\n{message}")

    def _close_progress(self):
        QApplication.restoreOverrideCursor()
        if self.progress is not None:
            self.progress.stop_progress()
            self.progress.close()
            self.progress = None

    # ------------------------------------------------------------------
    # Finishing: manifests, import, annotations
    # ------------------------------------------------------------------

    def _finish(self, results, jobs, import_after, image_format):
        self._close_progress()
        canceled = self.worker is not None and self.worker.canceled

        by_parent = {}
        for result in results:
            by_parent.setdefault(result['parent_path'], []).append(result)

        include_annotations = import_after and not canceled and self.include_annotations_combo.currentText() == "True"
        set_ids = {job['parent_path']: extracted_images.new_set_id() for job in jobs} if import_after and not canceled else {}

        # Manifests next to the images, for every raster that produced any
        for job in jobs:
            tiles = [r for r in by_parent.get(job['parent_path'], []) if r['path']]
            if tiles:
                self._write_manifest(job, tiles, set_ids.get(job['parent_path']), image_format, include_annotations)

        written = sum(1 for r in results if r['status'] == WRITTEN)
        reused = sum(1 for r in results if r['status'] == REUSED)
        empty = sum(1 for r in results if r['status'] == EMPTY)

        copied = 0
        masks_copied = 0
        imported = 0
        already_in_project = set()
        if import_after and not canceled:
            all_paths = [r['path'] for r in results if r['path']]
            # Images already in the project (from an earlier, since unlinked,
            # extraction) keep the annotations they have; copying again would double them
            already_in_project = {path for path in all_paths if self.raster_manager.has_image_path(path)}
            self.accept()
            if all_paths:
                self.main_window.import_images._process_image_files(all_paths)
                for job in jobs:
                    tiles = [r for r in by_parent.get(job['parent_path'], []) if r['path']]
                    linked = self._link_set(job, tiles, set_ids[job['parent_path']], include_annotations)
                    imported += len(linked)
                    to_copy = [(rect, path) for rect, path in linked if path not in already_in_project]
                    if include_annotations and to_copy:
                        count, copies = self._copy_annotations(job['parent_path'], to_copy)
                        copied += count
                        given, mask_bases, mask_codes = self._copy_masks(job['parent_path'], to_copy)
                        masks_copied += given
                        # Merge Back reads these: copies deleted on their image, and
                        # each image's mask as extracted
                        parent = self.raster_manager.get_raster(job['parent_path'])
                        record = extracted_images.active_set(parent) if parent is not None else None
                        if record is not None:
                            record['copies'] = copies
                            record['mask_bases'] = mask_bases
                            record['mask_label_map'] = {str(k): v for k, v in mask_codes.items()}

        self._show_summary(written, reused, empty, imported, copied, canceled, import_after, masks_copied)

        if not import_after or canceled:
            self.accept()

    def _write_manifest(self, job, tiles, set_id, image_format, include_annotations):
        raster = job['row']['raster']
        crs_wkt, transform = None, None
        try:
            src = raster.rasterio_src
            if src is not None and _is_georeferenced(src):
                crs_wkt = src.crs.to_wkt() if src.crs else None
                transform = tuple(src.transform)[:6]
        except Exception:
            pass

        manifest = extracted_images.make_manifest(
            parent_path=job['parent_path'],
            parent_width=raster.width,
            parent_height=raster.height,
            crs_wkt=crs_wkt,
            transform=transform,
            set_id=set_id,
            tiles=[{'path': r['path'], 'x': r['rect'][0], 'y': r['rect'][1],
                    'width': r['rect'][2], 'height': r['rect'][3]} for r in tiles],
            image_format=image_format,
            annotations_included=include_annotations,
            has_overlap=bool(job['row']['overlap_count']),
            max_overlap_px=job['row']['max_overlap'],
        )
        manifest_path = extracted_images.normalize_path(
            os.path.join(job['output_dir'], extracted_images.MANIFEST_FILENAME))
        try:
            extracted_images.write_manifest(manifest_path, manifest)
        except Exception as e:
            print(f"[ExtractWorkAreas] Could not write manifest {manifest_path}: {e}")
        job['manifest_path'] = manifest_path

    def _link_set(self, job, tiles, set_id, include_annotations):
        """Record the set on the parent and each imported image. Returns [(rect, path)] that imported."""
        parent = self.raster_manager.get_raster(job['parent_path'])
        if parent is None:
            return []

        linked = []
        for result in tiles:
            tile = self.raster_manager.get_raster(result['path'])
            if tile is None:
                continue
            tile.tile_of = extracted_images.make_tile_of(parent.image_path, set_id, result['rect'])
            # A tile is a crop of the parent at full resolution, so the parent's
            # scale always holds for it. Copied even over the scale a GeoTIFF tile
            # reads from its georeference: the parent's may be a Scale Tool override
            if parent.scale_x is not None and parent.scale_units is not None:
                source = f"copied from {parent.basename}"
                if parent.scale_source:
                    source += f": {parent.scale_source}"
                tile.update_scale(parent.scale_x, parent.scale_y, parent.scale_units, source=source)
            linked.append((result['rect'], tile.image_path))

        if not linked:
            return []

        parent.tile_sets.append(extracted_images.make_set_record(
            set_id=set_id,
            output_dir=job['output_dir'],
            manifest_path=job.get('manifest_path'),
            tile_paths=[path for _, path in linked],
            annotations_included=include_annotations,
            has_overlap=bool(job['row']['overlap_count']),
            max_overlap_px=job['row']['max_overlap'],
        ))
        self.raster_manager.rasterUpdated.emit(parent.image_path)
        return linked

    def _copy_annotations(self, parent_path, linked):
        """Copy the parent's vector annotations onto each image, clipped to its work area.

        Each copy gets a fresh id, and its data records the original's id and
        fingerprint and its own fingerprint, so Merge Back can tell which side
        changed (see Rasters/merge_back.py). The originals are not changed.

        Returns:
            (number of copies, {copy_id: [origin_id, tile_path]})
        """
        sources = []
        boxes = []
        for annotation in self.annotation_window.image_annotations_dict.get(parent_path, []):
            type_name = type(annotation).__name__
            if type_name not in SUPPORTED_TYPES:
                continue
            data = annotation.to_dict()
            bounds = annotation_dict_bounds(type_name, data)
            if bounds is None:
                continue
            sources.append((type_name, data, annotation.id, annotation_fingerprint(type_name, data)))
            boxes.append(box(*bounds))

        if not sources:
            return 0, {}

        # Only clip the annotations whose bounds touch each tile
        tree = STRtree(boxes)
        copies = []
        copies_record = {}

        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress = ProgressBar(self.annotation_window, "Copying Annotations")
        progress.show()
        progress.start_progress(len(linked))
        try:
            for rect, tile_path in linked:
                x, y, width, height = rect
                for index in tree.query(box(x, y, x + width, y + height)):
                    type_name, data, origin_id, origin_fingerprint = sources[int(index)]
                    for piece_type, piece in clip_annotation_dict(type_name, data, rect):
                        piece = shift_annotation_dict(piece_type, piece, -x, -y)
                        piece['id'] = str(uuid.uuid4())
                        piece['image_path'] = tile_path
                        stamp_copy(piece, origin_id, origin_fingerprint)
                        for part in piece.get('polygons', []):
                            part.pop('id', None)
                            part['image_path'] = tile_path
                        try:
                            annotation = ANNOTATION_CLASSES[piece_type].from_dict(piece, self.label_window)
                        except Exception as e:
                            print(f"[ExtractWorkAreas] Could not copy annotation {origin_id}: {e}")
                            continue
                        # Fingerprinted as built, so anything the constructor
                        # normalizes does not later read as an edit
                        annotation.data['tile_copy_fingerprint'] = annotation_fingerprint(
                            piece_type, annotation.to_dict())
                        copies.append(annotation)
                        copies_record[annotation.id] = [origin_id, tile_path]
                progress.update_progress()

            if copies:
                self.annotation_window.add_annotations(copies, record_action=False)
        finally:
            progress.finish_progress()
            progress.stop_progress()
            progress.close()
            QApplication.restoreOverrideCursor()

        return len(copies), copies_record

    def _copy_masks(self, parent_path, linked):
        """Give each image its slice of the parent's mask, and save that slice as the merge base.

        The image's mask keeps the parent's class ids and label map, lock bits
        included, so it means exactly what the parent's pixels meant. The
        snapshot next to each image is what Merge Back compares against to
        tell which side changed a pixel (see Rasters/merge_back.py).

        Returns:
            (images given a mask, {tile_path: snapshot path, or '' for an empty slice},
             {class_id: short_code} of the parent's mask)
        """
        parent = self.raster_manager.get_raster(parent_path)
        parent_mask = getattr(parent, 'mask_annotation', None) if parent is not None else None
        if parent_mask is None or not np.any(parent_mask.mask_data):
            return 0, {path: "" for _, path in linked}, {}

        label_codes = {int(cid): label.short_label_code for cid, label in parent_mask.class_id_to_label_map.items()}
        all_labels = list(self.label_window.labels)
        bases = {}
        given = 0

        for rect, tile_path in linked:
            x, y, width, height = rect
            region = parent_mask.mask_data[y:y + height, x:x + width]
            if region.shape != (height, width) or not region.any():
                bases[tile_path] = ""
                continue

            base_path = extracted_images.normalize_path(mask_base_path(tile_path))
            try:
                save_mask_base(base_path, region, label_codes)
                bases[tile_path] = base_path
            except Exception as e:
                # Without a snapshot the merge falls back to filling only empty pixels
                print(f"[ExtractWorkAreas] Could not save mask snapshot {base_path}: {e}")

            tile = self.raster_manager.get_raster(tile_path)
            if tile is None or not all_labels:
                continue
            tile_mask = MaskAnnotation(image_path=tile_path, mask_data=region.copy(), initial_labels=all_labels)
            # The parent's class-id space, then any project labels it lacks
            tile_mask.class_id_to_label_map.clear()
            tile_mask.label_id_to_class_id_map.clear()
            for class_id, label in parent_mask.class_id_to_label_map.items():
                tile_mask.class_id_to_label_map[class_id] = label
                tile_mask.label_id_to_class_id_map[label.id] = class_id
            tile_mask.next_class_id = max((int(c) for c in parent_mask.class_id_to_label_map), default=0) + 1
            tile_mask.visible_label_ids = set(tile_mask.label_id_to_class_id_map)
            tile_mask.sync_label_map(all_labels)
            tile_mask.invalidate_color_map()

            # Attached the way project load attaches masks: not through
            # add_annotation, which would file it with the vector annotations
            tile.mask_annotation = tile_mask
            self.annotation_window.annotation_manager.register_mask_annotation(tile_mask)
            self.image_window.update_image_annotations(tile_path, update_counts=False)
            given += 1

        return given, bases, label_codes

    def _show_summary(self, written, reused, empty, imported, copied, canceled, import_after, masks_copied=0):
        """Report the extraction in the main window's status bar, e.g.
        Extracted: 46 | Reused: 2 | Empty: 3 | Imported: 48 | Annotations: 312
        """
        parts = [f"Extracted: {written}", f"Reused: {reused}"]
        if empty:
            parts.append(f"Empty: {empty}")
        if import_after and not canceled:
            parts.append(f"Imported: {imported}")
            parts.append(f"Annotations: {copied}")
            if masks_copied:
                parts.append(f"Masks: {masks_copied}")
        message = " | ".join(parts)
        if canceled:
            message = "Extraction canceled | " + message
        self.main_window.status_bar.showMessage(message, 10000)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        """Stop the worker before the dialog goes away."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(5000)
        super().closeEvent(event)
