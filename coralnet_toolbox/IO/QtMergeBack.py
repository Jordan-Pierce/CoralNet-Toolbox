import os
import uuid

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel,
                             QMessageBox, QVBoxLayout)

from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation
from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.annotation_clipping import (MULTIPOLYGON, PATCH, POLYGON, RECTANGLE,
                                                              SUPPORTED_TYPES)
from coralnet_toolbox.Rasters import extracted_images
from coralnet_toolbox.Rasters.merge_back import (CHOICE_LABELS, DEFAULT_OVERLAP_THRESHOLD, KEEP_BOTH, KEEP_NEW,
                                                 KEEP_OLD, MERGE, TILE_DATA_KEYS, TileState, apply_actions,
                                                 class_lut, load_mask_base, mask_base_path, mask_merge_regions,
                                                 plan_merge_back, resolve)
from coralnet_toolbox.Icons import get_window_icon


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

ANNOTATION_CLASSES = {
    PATCH: PatchAnnotation,
    RECTANGLE: RectangleAnnotation,
    POLYGON: PolygonAnnotation,
    MULTIPOLYGON: MultiPolygonAnnotation,
}

# The one question: what to do where an annotation changed on both sides
CONFLICT_CHOICES = (KEEP_NEW, KEEP_OLD, KEEP_BOTH, MERGE)

CONFLICT_TOOLTIP = ("For annotations changed both on the images and on the original:\n"
                    "Keep new: the images' version replaces the original's.\n"
                    "Keep old: the original's version stays.\n"
                    "Keep both: both are kept.\n"
                    "Merge: shapes with the same label are joined into one; different labels\n"
                    "and patches are kept both.\n"
                    "Mask pixels changed on both take the images' class, unless you choose Keep old.")

# After merging
REMOVE_IMAGES = 'remove'
DELETE_FILES = 'delete'
KEEP_IMAGES = 'keep'

AFTER_MERGING_CHOICES = [
    (REMOVE_IMAGES, "Remove images from project"),
    (DELETE_FILES, "Remove images and delete files"),
    (KEEP_IMAGES, "Keep images as ordinary images"),
]

AFTER_MERGING_TOOLTIP = (
    "Remove images from project: the images leave the project; their files stay in the output folder.\n"
    "Remove images and delete files: the images, their tile record and mask snapshots are also deleted\n"
    "from disk (only files the extraction created). This cannot be undone.\n"
    "Keep images as ordinary images: the images stay in the project, unlinked, with their annotations."
)


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def _fit(array, height, width):
    """An array cropped or zero-padded to (height, width), so mismatched shapes compare safely."""
    array = np.asarray(array, dtype=np.uint8)
    if array.shape == (height, width):
        return array
    fitted = np.zeros((height, width), np.uint8)
    rows, cols = min(height, array.shape[0]), min(width, array.shape[1])
    fitted[:rows, :cols] = array[:rows, :cols]
    return fitted


# ----------------------------------------------------------------------------------------------------------------------
# Merge Back Dialog
# ----------------------------------------------------------------------------------------------------------------------


class MergeBack(QDialog):
    """Bring the annotations on a set of extracted images back onto the raster they came from.

    The whole set at once. Whatever changed on only one side is merged as it
    is (see Rasters/merge_back.py), so the dialog asks one thing: what to do
    where an annotation changed on both sides. That answer applies to every
    such annotation, and to mask pixels changed on both.
    """

    def __init__(self, main_window, parent_path, record, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window
        self.label_window = main_window.label_window
        self.raster_manager = self.image_window.raster_manager

        self.parent_path = parent_path
        self.record = record
        self.parent_name = os.path.basename(parent_path)

        self.setWindowIcon(get_window_icon("tile.svg"))
        self.setWindowTitle("Merge Back")

        self._gather_tiles()
        self._gather_masks()
        self.parent_annotations = self._gather_parent()

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.plan = plan_merge_back(self.parent_path, self.parent_annotations, self.tiles,
                                        self.record.get('copies'), DEFAULT_OVERLAP_THRESHOLD)
        finally:
            QApplication.restoreOverrideCursor()

        layout = QVBoxLayout(self)

        count = len(self.tiles)
        message = QLabel(f"Merge the annotations on {count} extracted image{'s' if count != 1 else ''} "
                         f"back into {self.parent_name}?")
        message.setWordWrap(True)
        layout.addWidget(message)

        form = QFormLayout()
        self.choice_combo = QComboBox()
        for choice in CONFLICT_CHOICES:
            self.choice_combo.addItem(CHOICE_LABELS[choice], choice)
        self.choice_combo.setToolTip(CONFLICT_TOOLTIP)
        form.addRow("When both changed:", self.choice_combo)

        self.after_merging_combo = QComboBox()
        for choice, label in AFTER_MERGING_CHOICES:
            self.after_merging_combo.addItem(label, choice)
        self.after_merging_combo.setToolTip(AFTER_MERGING_TOOLTIP)
        form.addRow("After merging:", self.after_merging_combo)
        layout.addLayout(form)

        buttons = QDialogButtonBox()
        buttons.addButton("Merge", QDialogButtonBox.AcceptRole)
        buttons.addButton(QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.merge)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setMinimumWidth(440)
        self.resize(self.minimumWidth(), self.sizeHint().height())

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _gather_tiles(self):
        """The set's images still in the project, as TileStates, plus what will not come back."""
        self.tiles = []
        self.missing_count = 0
        self.work_area_count = 0
        for path in self.record.get('tile_paths', []):
            raster = self.raster_manager.get_raster(path)
            tile_of = getattr(raster, 'tile_of', None) if raster is not None else None
            if raster is None or not tile_of:
                self.missing_count += 1
                continue
            rect = (tile_of['x'], tile_of['y'], tile_of['width'], tile_of['height'])
            annotations = [(type(a).__name__, a.to_dict())
                           for a in self.annotation_window.image_annotations_dict.get(path, [])
                           if type(a).__name__ in SUPPORTED_TYPES]
            self.tiles.append(TileState(raster.image_path, rect, annotations))
            self.work_area_count += len(raster.get_work_areas())

    def _gather_masks(self):
        """Each image's mask now, and as extracted, in the original's class ids.

        Only images with something to say take part: a mask now, or a non-empty
        snapshot (whose pixels may all have been erased since). Counts are
        worked out here against the original as it is now; the merge repeats
        the comparison live, so an overlapping image already merged counts as a
        change on the original.
        """
        self.mask_jobs = []
        self.unknown_mask_bases = 0
        labels = list(self.label_window.labels)

        parent = self.raster_manager.get_raster(self.parent_path)
        self.parent_mask = getattr(parent, 'mask_annotation', None)
        if self.parent_mask is not None:
            self.parent_mask.sync_label_map(labels)
            target_ids = {label.short_label_code: int(cid)
                          for cid, label in self.parent_mask.class_id_to_label_map.items()}
        else:
            # What get_mask_annotation will create at merge: ids from 1 in project label order
            target_ids = {label.short_label_code: index + 1 for index, label in enumerate(labels)}

        bases = self.record.get('mask_bases')
        included = self.record.get('annotations_included', False)

        for tile_state in self.tiles:
            x, y, width, height = tile_state.rect
            raster = self.raster_manager.get_raster(tile_state.path)
            tile_mask = getattr(raster, 'mask_annotation', None)

            tile_values = None
            if tile_mask is not None:
                tile_codes = {int(cid): label.short_label_code for cid, label in tile_mask.class_id_to_label_map.items()}
                tile_values = class_lut(tile_codes, target_ids)[_fit(tile_mask.mask_data, height, width)]

            if not included:
                base = np.zeros((height, width), np.uint8)  # extracted without annotations: started empty
            elif bases is None or tile_state.path not in bases:
                base = None  # extracted before masks were tracked
            elif not bases[tile_state.path]:
                base = np.zeros((height, width), np.uint8)
            else:
                base_mask, base_codes = load_mask_base(bases[tile_state.path])
                base = (class_lut(base_codes, target_ids)[_fit(base_mask, height, width)]
                        if base_mask is not None else None)

            if tile_values is None and (base is None or not base.any()):
                continue
            if tile_values is None:
                tile_values = np.zeros((height, width), np.uint8)  # its mask was removed outright
            if base is None:
                self.unknown_mask_bases += 1
            self.mask_jobs.append({'rect': tile_state.rect, 'tile': tile_values, 'base': base})

        self.mask_applied = 0
        self.mask_conflicts = 0
        for job in self.mask_jobs:
            x, y, width, height = job['rect']
            region = (_fit(self.parent_mask.mask_data[y:y + height, x:x + width], height, width)
                      if self.parent_mask is not None else np.zeros((height, width), np.uint8))
            apply, conflict = mask_merge_regions(region, job['tile'], job['base'])
            self.mask_applied += int(np.count_nonzero(apply))
            self.mask_conflicts += int(np.count_nonzero(conflict))

    def _gather_parent(self):
        return {a.id: (type(a).__name__, a.to_dict())
                for a in self.annotation_window.image_annotations_dict.get(self.parent_path, [])
                if type(a).__name__ in SUPPORTED_TYPES}

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def merge(self):
        after = self.after_merging_combo.currentData()
        if after == DELETE_FILES:
            reply = QMessageBox.question(
                self, "Delete Files",
                "After merging, the extracted images, their tile record and mask snapshots will be deleted "
                "from disk. This cannot be undone.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        # Every change made on only one side, then each conflict settled the same way
        choice = self.choice_combo.currentData()
        deletes, adds, conflicts = apply_actions(self.plan)
        for conflict in conflicts:
            conflict_deletes, conflict_adds = resolve(conflict, choice, self.parent_path)
            deletes.extend(conflict_deletes)
            adds.extend(conflict_adds)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            deleted, added, failed = self._apply(deletes, adds)
            mask_pixels = self._apply_masks(keep_new=choice != KEEP_OLD)
            if after == KEEP_IMAGES:
                kept = self._keep_images()
            else:
                removed, files_deleted = self._remove_images(after == DELETE_FILES)
        finally:
            QApplication.restoreOverrideCursor()

        self.accept()
        self.image_window.load_image_by_path(self.parent_path)

        # Reported in the status bar, e.g.
        # Merged into ortho.tif | Added: 14 | Removed: 9 | Conflicts: 2 (Keep new) | Images removed: 48
        parts = [f"Merged into {self.parent_name}", f"Added: {added}", f"Removed: {deleted}"]
        if conflicts:
            parts.append(f"Conflicts: {len(conflicts)} ({CHOICE_LABELS[choice]})")
        if mask_pixels:
            parts.append(f"Mask pixels: {mask_pixels:,}")
        if failed:
            parts.append(f"Skipped: {failed}")  # could not be rebuilt, e.g. their label was deleted
        if after == KEEP_IMAGES:
            parts.append(f"Images kept: {kept}")
        else:
            parts.append(f"Images removed: {removed}")
            if after == DELETE_FILES:
                parts.append(f"Files deleted: {files_deleted}")
        self.main_window.status_bar.showMessage(" | ".join(parts), 10000)

    def _apply(self, deletes, adds):
        """Swap annotations on the original. Returns (deleted, added, failed)."""
        window = self.annotation_window
        self.main_window.untoggle_all_tools()

        # The undo history refers to annotations by id, and ids return to the
        # original here; an old entry could act on the wrong one
        window.action_stack.undo_stack.clear()
        window.action_stack.redo_stack.clear()

        to_delete = [window.annotations_dict[i] for i in dict.fromkeys(deletes) if i in window.annotations_dict]
        window.unselect_annotations()
        if to_delete:
            window.delete_annotations(to_delete, record_action=False)

        new_annotations = []
        restored_ids = []
        failed = 0
        for type_name, data, wanted_id in adds:
            data = dict(data)
            # An id still in use cannot be reused: two annotations sharing one
            # silently replace each other in the annotation dictionary
            if wanted_id and wanted_id not in window.annotations_dict:
                data['id'] = wanted_id
                restored_ids.append(wanted_id)
            else:
                data['id'] = str(uuid.uuid4())
            data['image_path'] = self.parent_path
            for part in data.get('polygons', []):
                part.pop('id', None)
                part['image_path'] = self.parent_path
            try:
                new_annotations.append(ANNOTATION_CLASSES[type_name].from_dict(data, self.label_window))
            except Exception as e:
                failed += 1
                print(f"[MergeBack] Could not rebuild annotation: {e}")

        if new_annotations:
            window.add_annotations(new_annotations, record_action=False)

        self._purge_feature_cache([a.id for a in to_delete] + restored_ids)
        return len(to_delete), len(new_annotations), failed

    def _apply_masks(self, keep_new):
        """Write the images' mask changes onto the original. Returns how many pixels changed.

        Compared live, image by image: with overlapping work areas, a pixel an
        earlier image already changed counts as changed on the original, so a
        later image that changed it differently is a conflict, not a silent win.
        """
        if not self.mask_jobs:
            return 0

        parent = self.raster_manager.get_raster(self.parent_path)
        parent_mask = parent.get_mask_annotation(list(self.label_window.labels))
        try:
            self.annotation_window.annotation_manager.register_mask_annotation(parent_mask)
        except Exception:
            pass

        width = parent_mask.mask_data.shape[1]
        changed = 0
        for job in self.mask_jobs:
            x, y, tile_width, tile_height = job['rect']
            region = parent_mask.mask_data[y:y + tile_height, x:x + tile_width]
            rows_available, cols_available = region.shape
            tile = job['tile'][:rows_available, :cols_available]
            base = job['base'][:rows_available, :cols_available] if job['base'] is not None else None

            apply, conflict = mask_merge_regions(region, tile, base)
            take = (apply | conflict) if keep_new else apply
            rows, cols = np.nonzero(take)
            if rows.size == 0:
                continue
            flat_indices = (y + rows).astype(np.int64) * width + (x + cols)
            result = parent_mask.apply_flat_values_at_indices(flat_indices, tile[rows, cols], silent=True)
            if result:
                changed += int(result['flat_indices'].size)

        if changed:
            parent_mask.refresh_graphics()
        self.image_window.update_image_annotations(self.parent_path)
        return changed

    def _purge_feature_cache(self, annotation_ids):
        """Drop cached Explorer features for annotations whose shape changed under the same id.

        The Explorer only invalidates entries while its viewer is open, and the
        cache is on disk, keyed by annotation id.
        """
        if not annotation_ids:
            return
        try:
            cache_manager = getattr(self.main_window, 'cache_manager', None)
            if cache_manager is None:
                from coralnet_toolbox.Explorer.managers.CacheManager import CacheManager
                cache_manager = CacheManager()
            cache_manager.remove_features_for_annotations(annotation_ids)
        except Exception as e:
            print(f"[MergeBack] Could not purge feature cache: {e}")

    def _keep_images(self):
        """Leave the set's images in the project, unlinked, as ordinary images. Returns how many."""
        tile_paths = [path for path in self.record.get('tile_paths', []) if self.raster_manager.has_image_path(path)]

        # Their annotations' provenance means nothing once the set is gone
        for path in tile_paths:
            for annotation in self.annotation_window.image_annotations_dict.get(path, []):
                if isinstance(getattr(annotation, 'data', None), dict):
                    for key in TILE_DATA_KEYS:
                        annotation.data.pop(key, None)

        for changed_path in extracted_images.unlink_set(self.raster_manager, self.parent_path,
                                                        self.record.get('set_id')):
            self.raster_manager.rasterUpdated.emit(changed_path)
        return len(tile_paths)

    def _remove_images(self, delete_files):
        """Take the set's images out of the project, and off disk if asked. Returns (removed, files deleted)."""
        set_id = self.record.get('set_id')
        output_dir = self.record.get('output_dir')
        manifest_path = self.record.get('manifest_path')

        # Captured first: removing images shrinks the record's list as it goes
        files = list(self.record.get('tile_paths', []))
        if manifest_path and os.path.isfile(manifest_path):
            try:
                files.extend(tile['path'] for tile in extracted_images.read_manifest(manifest_path).get('tiles', []))
            except Exception as e:
                print(f"[MergeBack] Could not read {manifest_path}: {e}")
        files = list(dict.fromkeys(files))
        in_project = [path for path in files if self.raster_manager.has_image_path(path)]

        # Mask snapshots sit next to their images
        snapshots = [path for path in (self.record.get('mask_bases') or {}).values() if path]
        snapshots += [mask_base_path(path) for path in files]
        files = list(dict.fromkeys(files + snapshots))

        if in_project:
            self.image_window.delete_images(in_project)
        extracted_images.unlink_set(self.raster_manager, self.parent_path, set_id)

        files_deleted = 0
        if delete_files:
            for path in files + ([manifest_path] if manifest_path else []):
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                        files_deleted += 1
                except OSError as e:
                    print(f"[MergeBack] Could not delete {path}: {e}")
            if output_dir and os.path.isdir(output_dir):
                try:
                    os.rmdir(output_dir)  # only succeeds when empty
                except OSError:
                    pass

        return len(in_project), files_deleted
