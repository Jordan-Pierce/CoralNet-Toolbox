import os
import uuid

import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox, QGroupBox, QLabel,
                             QMessageBox, QVBoxLayout)

from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation
from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.annotation_clipping import (MULTIPOLYGON, PATCH, POLYGON, RECTANGLE,
                                                              SUPPORTED_TYPES)
from coralnet_toolbox.Rasters import extracted_images
from coralnet_toolbox.Rasters.merge_back import (DELETED_KINDS, EDITED_BOTH_KINDS, KEEP_BOTH, KEEP_NEW, KEEP_OLD,
                                                 MASK_ADD, MASK_LOCK_BIT, MASK_REPLACE_ALL, OWNER, REVIEW,
                                                 REVIEW_LABEL_ID, TileState, apply_actions, class_lut,
                                                 load_mask_base, mask_base_path, mask_pixels_to_take,
                                                 masks_overlap, plan_merge_back, resolve, settle_mask_overlaps)
from coralnet_toolbox.Icons import get_window_icon
from coralnet_toolbox import theme as app_theme
from coralnet_toolbox.utilities import STATUS_SUMMARY_MS, plural as _plural, show_status as _show_status


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

ANNOTATION_CLASSES = {
    PATCH: PatchAnnotation,
    RECTANGLE: RectangleAnnotation,
    POLYGON: PolygonAnnotation,
    MULTIPOLYGON: MultiPolygonAnnotation,
}

# Each question is a list of (value, option text, line shown under it). The
# lines are formatted with {folder}.
#
# The copy names two things only, the same way everywhere:
#   Work Area Images  the images extracted from the work areas
#   Parent Image      the raster they were extracted from
# "Patch" always means a Patch annotation, never an area of an image.
#
# Annotations are never merged or joined into one: every version and every
# piece comes back, for the person to combine or delete afterwards.

DELETED_CHOICES = (
    (KEEP_NEW, "Bring back from the Work Area Images",
     "Restored as the Work Area Images have it now, with the original ID."),
    (KEEP_OLD, "Keep deleted",
     "The Work Area Images' copies are dropped."),
)

BOTH_CHANGED_CHOICES = (
    (KEEP_NEW, "Use the Work Area Images' version",
     "It replaces the Parent Image's version."),
    (KEEP_OLD, "Keep the Parent Image's version",
     "The Work Area Images' changes to these are dropped."),
    (KEEP_BOTH, "Keep both",
     "Both versions are kept, overlapping."),
)

# Masks only: a pixel holds one label, so two Work Area Images cannot both be kept there
DISAGREEMENT_CHOICES = (
    (OWNER, "Use the most central Work Area Image",
     "Where two Work Area Images changed the same mask pixel differently, the one where it sits farthest "
     "from the edge wins."),
    (REVIEW, "Relabel as Review",
     "Where two Work Area Images changed the same mask pixel differently, it is relabeled Review. Small "
     "differences still go to the most central one. Export with Treat as Ignore to keep Review out of "
     "training."),
)

MASK_CHOICES = (
    (MASK_ADD, "(Add) Only update pixels that are currently empty",
     "Labels from the Work Area Images fill every unlabeled pixel on the Parent Image. Nothing is "
     "overwritten or erased."),
    (MASK_REPLACE_ALL, "(Overwrite) Update all pixels",
     "Every pixel the Work Area Images cover takes their label, erased pixels included. Locked pixels "
     "are kept."),
)

# After merging
REMOVE_IMAGES = 'remove'
DELETE_FILES = 'delete'

AFTER_MERGING_CHOICES = (
    (REMOVE_IMAGES, "Remove Work Area Images, but do not delete the files",
     "Files stay in {folder}. To edit these areas again, extract them again from the Parent Image."),
    (DELETE_FILES, "Remove Work Area Images, and delete the files",
     "The Work Area Images and the files this extraction made are deleted."),
)


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


# The dialog keeps one width; its text wraps to fill each group
DIALOG_WIDTH = 560


def _wrapped_label(text="", color=None):
    label = QLabel(text)
    label.setWordWrap(True)
    # Top aligned: a shorter option's text sits under its dropdown, not mid way down
    label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    if color is not None:
        label.setStyleSheet(f"color: {color.name()};")
    return label



def _show_again_after_full_resolution(main_window, image_path, message):
    """Show message again once the image's full resolution arrives, if it is still on its way.

    A large image opens at low resolution and decodes the full one on a
    worker thread, then says "Loaded image" in the status bar, which would
    replace the merge summary seconds after it appeared.
    """
    window = main_window.annotation_window
    loading = (getattr(window, '_pending_full_res_path', None) == image_path
               or getattr(window, '_full_res_worker', None) is not None)
    if not loading:
        return
    status_bar = main_window.status_bar
    loaded = f"Loaded image: {os.path.basename(image_path)}"

    def on_change(text):
        if text == loaded:
            stop()
            # After that message is up, not from inside its own signal
            QTimer.singleShot(0, lambda: _show_status(main_window, message, STATUS_SUMMARY_MS))

    def stop():
        try:
            status_bar.messageChanged.disconnect(on_change)
        except TypeError:
            pass  # already shown again

    status_bar.messageChanged.connect(on_change)
    QTimer.singleShot(60000, stop)


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
    is (see Rasters/merge_back.py). The dialog asks one question per kind of
    decision, and only the questions that apply to this set, each with a line
    under it saying what the chosen option will do.
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

        parent = self.raster_manager.get_raster(self.parent_path)
        self.parent_size = (parent.width, parent.height) if parent is not None else None

        # Reading every image's annotations, mask and mask snapshot can take a
        # while on a large set. Planned once, to learn which questions apply;
        # the answers only choose how its conflicts are settled
        QApplication.setOverrideCursor(Qt.WaitCursor)
        _show_status(main_window, f"Reading the Work Area Images of {self.parent_name}...")
        try:
            self._gather_tiles()
            self._gather_masks()
            self.parent_annotations = self._gather_parent()
            self.plan = self._plan()
        finally:
            QApplication.restoreOverrideCursor()
            _show_status(main_window, "")

        self.deleted_combo = None
        self.both_changed_combo = None
        self.disagreement_combo = None
        self.mask_combo = None

        # Every wrapped label, with every text it can show: on first show each is
        # given the height of its longest text, so changing an option never
        # resizes the dialog (a top-level dialog does not size itself by a
        # label's height for width, and would cut wrapped text off)
        self._wrapped_labels = []
        self._text_fitted = False
        self.setFixedWidth(DIALOG_WIDTH)

        layout = QVBoxLayout(self)
        layout.addWidget(self._create_info_group())

        deleted = self.plan.conflict_count(DELETED_KINDS)
        if deleted:
            self.deleted_combo = self._add_question(
                layout, "Deleted on the Parent Image", DELETED_CHOICES,
                f"{_plural(deleted, 'annotation was', 'annotations were')} deleted on the Parent Image since "
                f"extraction, but {'is' if deleted == 1 else 'are'} still on the Work Area Images.")

        both = self.plan.conflict_count(EDITED_BOTH_KINDS)
        if both:
            self.both_changed_combo = self._add_question(
                layout, "Also Edited on the Parent Image", BOTH_CHANGED_CHOICES,
                f"{_plural(both, 'annotation was', 'annotations were')} changed on the Work Area Images and "
                "on the Parent Image.")

        # Only masks on overlapping Work Area Images can conflict
        if masks_overlap(self.mask_jobs):
            self.disagreement_combo = self._add_question(
                layout, "When Overlapping Work Area Images Conflict", DISAGREEMENT_CHOICES)

        if self.mask_jobs:
            note = None
            if self.unknown_mask_bases:
                note = (f"{_plural(self.unknown_mask_bases, 'Work Area Image predates', 'Work Area Images predate')} "
                        "mask tracking, so erasing where they overlap may not come through.")
            self.mask_combo = self._add_question(layout, "Parent's Mask Annotation", MASK_CHOICES, note)

        self.after_merging_combo = self._add_question(layout, "After Merging", AFTER_MERGING_CHOICES)

        buttons = QDialogButtonBox()
        buttons.addButton("Merge", QDialogButtonBox.AcceptRole)
        buttons.addButton(QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.merge)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Questions
    # ------------------------------------------------------------------

    def _create_info_group(self):
        """What is about to happen, what cannot be decided here, and what follows."""
        group_box = QGroupBox("Information")
        group_layout = QVBoxLayout(group_box)

        count = len(self.tiles)
        texts = [(f"Merges {count} Work Area Image{'s' if count != 1 else ''} back to the Parent Image:\n"
                  f"\t- {self.parent_name}\n\n"
                  "Work Area Images are first merged together, and then added to or overwrite the Parent Image. "
                  "Choose the options below.", None)]

        warnings = []
        if self.missing_count:
            warnings.append(f"{self.missing_count} Work Area Image{'s are' if self.missing_count != 1 else ' is'} "
                            "no longer in the project. Their changes will not come back.")
        if not (self.plan.edits or self.plan.news or self.plan.removals or self.plan.conflicts or self.mask_jobs):
            warnings.append("Nothing on the Work Area Images has changed since extraction.")
        if warnings:
            texts.append(("\n".join(warnings), app_theme.ATTENTION_COLOR))

        # Nothing is ever joined or merged; say where that leaves more than one annotation
        notes = []
        cut = self.plan.cut_annotations
        if cut:
            notes.append(f"{_plural(cut, 'edited annotation crosses', 'edited annotations cross')} a Work Area "
                         f"Image edge and come{'s' if cut == 1 else ''} back as separate pieces.")
        stacked = self.plan.stacked_patches
        if stacked:
            notes.append(f"{_plural(stacked, 'Patch annotation was', 'Patch annotations were')} changed "
                         "differently on overlapping Work Area Images. Each version is kept, stacked on the "
                         "same spot.")
        notes.append("This cannot be undone with Ctrl+Z.")
        texts.append(("\n".join(notes), app_theme.TEXT_SECONDARY_COLOR))

        for text, color in texts:
            label = _wrapped_label(text, color)
            self._wrapped_labels.append((label, [text]))
            group_layout.addWidget(label)
        return group_box

    def _add_question(self, layout, title, choices, context=None):
        """A group with a dropdown and a line under it saying what the chosen option does. Returns the combo.

        Args:
            title: the group's title, the question being asked.
            choices: [(value, option text, description)]; texts may use {folder}.
            context: a fixed sentence put before the description, such as how many this is about.
        """
        names = {'folder': self.record.get('output_dir') or "the output folder"}
        combo = QComboBox()
        for value, text, _ in choices:
            combo.addItem(text.format(**names), value)

        lines = [line.format(**names) for _, _, line in choices]
        lines = [f"{context} {line}" if context else line for line in lines]
        description = _wrapped_label(lines[combo.currentIndex()], app_theme.TEXT_SECONDARY_COLOR)
        self._wrapped_labels.append((description, lines))
        combo.currentIndexChanged.connect(lambda index: description.setText(lines[index]))

        group_box = QGroupBox(title)
        group_layout = QVBoxLayout(group_box)
        group_layout.addWidget(combo)
        group_layout.addWidget(description)
        layout.addWidget(group_box)
        return combo

    def showEvent(self, event):
        """Fit the wrapped text once, now that each label has its real width."""
        super().showEvent(event)
        if not self._text_fitted:
            self._text_fitted = True
            self._fit_wrapped_text()

    def _fit_wrapped_text(self):
        """Give each wrapped label the height of its longest text, then the dialog the height of it all.

        Done once: from then on, changing an option only swaps a label's text,
        and the dialog keeps its size.
        """
        self.layout().activate()
        for label, texts in self._wrapped_labels:
            width = label.width() if label.width() > 0 else DIALOG_WIDTH - 60
            shown = label.text()
            heights = []
            for text in texts:
                label.setText(text)
                heights.append(label.heightForWidth(width))
            label.setText(shown)
            label.setFixedHeight(max(heights))
        # The new heights only queue layout updates, and each group keeps its old
        # size until its own runs; recompute them all before reading the size
        for group_box in self.findChildren(QGroupBox):
            group_box.layout().invalidate()
            group_box.updateGeometry()  # drops the size the dialog's layout cached for it
        self.layout().invalidate()
        self.layout().activate()
        self.setFixedHeight(self.layout().sizeHint().height())

    @staticmethod
    def _answer(combo, default):
        return combo.currentData() if combo is not None else default

    def _plan(self):
        return plan_merge_back(self.parent_path, self.parent_annotations, self.tiles, self.record.get('copies'))

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
        snapshot (whose pixels may all have been erased since).
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
                f"After merging, the Work Area Images and the files this extraction made (the images, their "
                f"record and mask snapshots) will be deleted from {self.record.get('output_dir') or 'disk'}. "
                "This cannot be undone.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        deleted = self._answer(self.deleted_combo, KEEP_NEW)
        both_changed = self._answer(self.both_changed_combo, KEEP_NEW)
        disagreement = self._answer(self.disagreement_combo, OWNER)
        mask_mode = self._answer(self.mask_combo, MASK_ADD)

        # Every change made on only one side, then each conflict settled by its question
        deletes, adds, conflicts = apply_actions(self.plan)
        for conflict in conflicts:
            choice = deleted if conflict.kind in DELETED_KINDS else both_changed
            conflict_deletes, conflict_adds = resolve(conflict, choice)
            deletes.extend(conflict_deletes)
            adds.extend(conflict_adds)

        images = _plural(len(self.tiles), 'Work Area Image', 'Work Area Images')
        error = None
        stage = None
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            stage = "merging annotations"
            _show_status(self.main_window, f"Merging annotations from {images} into {self.parent_name}...")
            updated, added, removed_annotations, failed = self._apply(deletes, adds)

            stage = "merging masks"
            if self.mask_jobs:
                _show_status(self.main_window, f"Merging masks from {images} into {self.parent_name}...")
            mask_pixels, review_pixels = self._apply_masks(mask_mode, disagreement)

            stage = "removing the Work Area Images"
            _show_status(self.main_window, "Removing the Work Area Images and deleting their files..."
                         if after == DELETE_FILES else "Removing the Work Area Images...")
            removed, files_deleted = self._remove_images(after == DELETE_FILES)

            stage = f"opening {self.parent_name}"
            self.accept()
            self.image_window.load_image_by_path(self.parent_path)
        except Exception as e:
            error = e
        finally:
            QApplication.restoreOverrideCursor()

        if error is not None:
            print(f"[MergeBack] Failed while {stage}: {error}")
            _show_status(self.main_window, f"Merge Back failed while {stage}: {error}", STATUS_SUMMARY_MS)
            QMessageBox.critical(self, "Merge Back Failed",
                                 f"Merge Back stopped while {stage}:\n{error}\n\n"
                                 f"Some changes may already be on {self.parent_name}. Check it before saving "
                                 "the project.")
            # Not offered again: the plan no longer matches what is on the Parent Image
            self.reject()
            return

        # Reported in the status bar, e.g.
        # Merged 48 Work Area Images into ortho.tif | Annotations updated: 3 | Annotations added: 14 |
        # Conflicts settled: 2 | Mask pixels changed: 1,204 | Work Area Images removed: 48
        parts = [f"Merged {images} into {self.parent_name}"]
        if updated:
            parts.append(f"Annotations updated: {updated}")
        if added:
            parts.append(f"Annotations added: {added}")
        if removed_annotations:
            parts.append(f"Annotations removed: {removed_annotations}")
        if not (updated or added or removed_annotations):
            parts.append("Annotations unchanged")
        if conflicts:
            parts.append(f"Conflicts settled: {len(conflicts)}")
        if mask_pixels:
            parts.append(f"Mask pixels changed: {mask_pixels:,}")
        if review_pixels:
            parts.append(f"Mask pixels relabeled Review: {review_pixels:,}")
        if failed:
            # Could not be rebuilt, e.g. their label was deleted
            parts.append(f"Annotations skipped: {failed}")
        parts.append(f"Work Area Images removed: {removed}")
        if after == DELETE_FILES:
            parts.append(f"Files deleted: {files_deleted}")
        summary = " | ".join(parts)
        _show_status(self.main_window, summary, STATUS_SUMMARY_MS)
        _show_again_after_full_resolution(self.main_window, self.parent_path, summary)

    def _apply(self, deletes, adds):
        """Swap annotations on the original. Returns (updated, added, removed, failed).

        Updated ones came back under the ID they had, in place of the
        original's version; added ones are new to it, or brought back after it
        deleted them; removed ones are gone with nothing in their place.
        """
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
            # An id still in use, or already given to another add in this merge,
            # cannot be reused: two annotations sharing one silently replace each
            # other in the annotation dictionary
            if wanted_id and wanted_id not in window.annotations_dict and wanted_id not in restored_ids:
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
        deleted_ids = {a.id for a in to_delete}
        updated = sum(1 for a in new_annotations if a.id in deleted_ids)
        return updated, len(new_annotations) - updated, len(to_delete) - updated, failed

    def _apply_masks(self, mode=MASK_ADD, disagreement=OWNER):
        """Write the images' mask changes onto the original. Returns (pixels changed, pixels marked Review).

        Two stages. The images are settled among themselves first: where two
        overlapping images changed a pixel differently, one is held back from
        it, or both are and it is marked Review. Then each image is compared
        with the original, under the mask mode (see mask_pixels_to_take).
        Without the first stage, a pixel an earlier image had just written
        would look like an edit on the original to the next.
        """
        if not self.mask_jobs:
            return 0, 0

        parent = self.raster_manager.get_raster(self.parent_path)
        parent_mask = parent.get_mask_annotation(list(self.label_window.labels))
        try:
            self.annotation_window.annotation_manager.register_mask_annotation(parent_mask)
        except Exception:
            pass

        held_back, review = settle_mask_overlaps(self.mask_jobs, self.parent_size, disagreement)

        width = parent_mask.mask_data.shape[1]
        changed = 0
        for index, job in enumerate(self.mask_jobs):
            x, y, tile_width, tile_height = job['rect']
            region = parent_mask.mask_data[y:y + tile_height, x:x + tile_width]
            rows_available, cols_available = region.shape
            tile = job['tile'][:rows_available, :cols_available]
            base = job['base'][:rows_available, :cols_available] if job['base'] is not None else None

            take = mask_pixels_to_take(region, tile, base, mode)
            if index in held_back:
                take &= ~held_back[index][:rows_available, :cols_available]
            changed += self._write_mask_pixels(parent_mask, take, tile, x, y, width)

        # Review areas last, compared with the original like any image's change;
        # never over a locked pixel, since a lock is a decision and Review is not
        review_pixels = 0
        review_id = parent_mask.label_id_to_class_id_map.get(REVIEW_LABEL_ID)
        if review and review_id is not None:
            for x, y, area, base in review:
                region = parent_mask.mask_data[y:y + area.shape[0], x:x + area.shape[1]]
                rows_available, cols_available = region.shape
                area = area[:rows_available, :cols_available]
                base = base[:rows_available, :cols_available] if base is not None else None
                marked = (base if base is not None else region).copy()
                marked[area] = review_id

                take = mask_pixels_to_take(region, marked, base, mode) & area & (region < MASK_LOCK_BIT)
                written = self._write_mask_pixels(parent_mask, take, marked, x, y, width)
                changed += written
                review_pixels += written

        if changed:
            parent_mask.refresh_graphics()
        self.image_window.update_image_annotations(self.parent_path)
        return changed, review_pixels

    @staticmethod
    def _write_mask_pixels(parent_mask, take, values, x, y, width):
        """Write values[take] onto the original's mask at (x, y). Returns how many pixels changed."""
        rows, cols = np.nonzero(take)
        if rows.size == 0:
            return 0
        flat_indices = (y + rows).astype(np.int64) * width + (x + cols)
        result = parent_mask.apply_flat_values_at_indices(flat_indices, values[rows, cols], silent=True)
        return int(result['flat_indices'].size) if result else 0

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

        # Kept files are marked merged, so importing them again by hand asks first
        if not delete_files and manifest_path:
            extracted_images.mark_manifest_merged(manifest_path)

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
