import os
import gc
import yaml

import numpy as np
from PIL import Image

from rasterio.features import rasterize
from shapely.geometry import Polygon

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QGroupBox, QVBoxLayout, QLabel, QApplication, QCheckBox,
                             QRadioButton)

from coralnet_toolbox.MachineLearning.ExportDataset.QtBase import Base
from coralnet_toolbox.MachineLearning.ExportDataset.export_dataset_utils import (
    build_sample_export_name,
    busy_cursor,
    closing_progress_bar,
    frame_matches_stride,
    materialize_sample_image,
    normalize_source_path,
    resolve_sample_source,
    sample_dimensions,
)

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Rasters.extracted_images import has_active_set

from coralnet_toolbox.Icons import get_icon, get_window_icon


def shoelace_area(points):
    """Polygon area from its vertices, by the shoelace formula.

    The same formula PolygonAnnotation.get_area() already falls back to when
    Shapely refuses a geometry, done without building a Shapely polygon first.

    Args:
        points: Sequence of QPointF.

    Returns:
        float: Unsigned area, or 0.0 for fewer than three points.
    """
    count = len(points)
    if count < 3:
        return 0.0

    total = 0.0
    for index in range(count):
        current = points[index]
        following = points[(index + 1) % count]
        total += current.x() * following.y() - following.x() * current.y()
    return abs(total) / 2.0


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Semantic(Base):
    """
    Export semantic segmentation masks from MaskAnnotations.
    
    Unlike vector annotations (patches, rectangles, polygons), this class works with 
    mask annotations that contain pixel-level class information. The table shows 
    boolean presence or counts of class labels within mask annotations.
    """

    supports_unlabeled_video_frames = True
    task = "semantic"

    def __init__(self, main_window, parent=None):
        super(Semantic, self).__init__(main_window, parent)
        self.setWindowTitle("Export Semantic Segmentation Dataset")
        self.setWindowIcon(get_window_icon("mask.svg"))
        
        self._stats_cache = {} 
        self._project_labels = []
        # A mask's pixel stats only change when the mask does, so the cache is
        # filled in and kept rather than rebuilt. Set when this dialog's own
        # copy has to be re-read: on opening, and on Refresh.
        self._stats_cache_force = True
        # Set only by Refresh, which goes back to the pixels for every mask
        # rather than trusting the mask's own cache. The escape hatch if a mask
        # ever changes without invalidating itself.
        self._stats_mask_recompute = False

    def setup_info_layout(self):
        """Setup the info layout"""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text
        info_text = ("Export semantic segmentation masks in YOLO format. "
                     "Supports both MaskAnnotations and vector annotations (Patches, Rectangles, Polygons). "
                     "Vector annotations will be rasterized into semantic masks. "
                     "This exports pixel-level class labels for semantic segmentation training.")
        info_label = QLabel(info_text)

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def create_annotation_layout(self):
        """Creates the annotation type checkboxes layout group box - supports all annotation types for semantic."""
        group_box = QGroupBox("Annotation Types")
        layout = QVBoxLayout()

        # For semantic segmentation, we support both mask and vector annotations
        self.include_masks_checkbox = QCheckBox("Include Mask Annotations")
        self.include_masks_checkbox.setChecked(True)
        self.include_masks_checkbox.setEnabled(True)
        
        # Enable vector annotation types - these will be rasterized into semantic masks
        self.include_patches_checkbox = QCheckBox("Include Patch Annotations")
        self.include_patches_checkbox.setChecked(True)
        self.include_patches_checkbox.setEnabled(True)
        
        self.include_rectangles_checkbox = QCheckBox("Include Rectangle Annotations") 
        self.include_rectangles_checkbox.setChecked(True)
        self.include_rectangles_checkbox.setEnabled(True)
        
        self.include_polygons_checkbox = QCheckBox("Include Polygon Annotations")
        self.include_polygons_checkbox.setChecked(True)
        self.include_polygons_checkbox.setEnabled(True)

        # Each of these decides which annotations exist to index, so the index
        # has to go. All four are wired here: this override replaces the base
        # class's layout, so the base class's connections never ran and the
        # three vector boxes changed nothing until Refresh was pressed.
        for checkbox in (self.include_masks_checkbox,
                         self.include_patches_checkbox,
                         self.include_rectangles_checkbox,
                         self.include_polygons_checkbox):
            checkbox.stateChanged.connect(self.refresh_structure)

        layout.addWidget(self.include_masks_checkbox)
        layout.addWidget(self.include_patches_checkbox)
        layout.addWidget(self.include_rectangles_checkbox)
        layout.addWidget(self.include_polygons_checkbox)

        group_box.setLayout(layout)
        return group_box

    def update_annotation_type_checkboxes(self):
        """
        Update the state of annotation type checkboxes for semantic segmentation.
        """
        # Enable all annotation types for semantic segmentation
        self.include_masks_checkbox.setChecked(True)
        self.include_masks_checkbox.setEnabled(True)
        
        # Enable vector annotation types - they will be rasterized
        self.include_patches_checkbox.setChecked(True)
        self.include_patches_checkbox.setEnabled(True)
        self.include_rectangles_checkbox.setChecked(True)
        self.include_rectangles_checkbox.setEnabled(True)
        self.include_polygons_checkbox.setChecked(True)
        self.include_polygons_checkbox.setEnabled(True)

        # Enable negative sample options for semantic segmentation
        self.include_negatives_radio.setEnabled(True)
        self.exclude_negatives_radio.setEnabled(True)

    def setup_unlabeled_handling_layout(self):
        """
        Setup the layout for determining how unlabeled pixels are treated, 
        including explanatory text for the user.
        """
        group_box = QGroupBox("Unlabeled Pixel Handling")
        layout = QVBoxLayout()

        # Explanatory text to guide the user
        info_text = (
            "<b>How should unlabeled pixels be handled during training?</b><br><br>"
            "• <b>Ignore (Standard):</b> The model is not penalized for its predictions in unlabeled areas. "
            "Use this if your images are <i>sparsely labeled</i> (i.e., you didn't label every single object).<br>"
            "• <b>Background:</b> Unlabeled areas are explicitly taught to the model as 'negative space'. "
            "Use this if what is <i>not labeled</i> should be learned as 'Background'."
        )
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        # Optional: Add a little margin at the bottom of the label
        info_label.setStyleSheet("margin-bottom: 10px;")
        layout.addWidget(info_label)

        # Radio buttons
        self.ignore_radio = QRadioButton("Treat as Ignore (Index 255)")

        self.background_radio = QRadioButton("Treat as Background (Index 0)")
        self.background_radio.setChecked(True) # Default to background for semantic segmentation

        layout.addWidget(self.ignore_radio)
        layout.addWidget(self.background_radio)
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    # Set for the length of one index rebuild, which needs the mask list twice:
    # once for the statistics pass and once for _indexable_annotations. Building
    # it walks every image in the source with a get_raster() per image. A class
    # attribute so it exists before __init__ has run.
    _cached_mask_annotations = None

    def get_mask_annotations(self):
        """
        Get all mask annotations from the current images.

        Checks both raster.mask_annotation (set when an image is open in the
        annotation window) AND the annotation manager's image_annotations_dict
        (which holds all loaded annotations, including those for images the user
        hasn't navigated to yet).  The union ensures masks are found for every
        image in the project regardless of which images have been viewed.

        Returns:
            list: Deduplicated list of MaskAnnotation objects
        """
        if self._cached_mask_annotations is not None:
            return self._cached_mask_annotations

        seen_ids = set()
        mask_annotations = []

        if self.filtered_images_radio.isChecked():
            images = self.image_window.table_model.filtered_paths
        else:
            images = self.image_window.raster_manager.image_paths

        for image_path in images:
            # Source 1: raster attribute (fastest lookup)
            raster = self.image_window.raster_manager.get_raster(image_path)
            # Its work areas are extracted: the images carry its mask now, and
            # exporting both counts those pixels twice (see get_selected_image_paths)
            if raster is not None and has_active_set(raster):
                continue
            if raster and raster.mask_annotation:
                ann = raster.mask_annotation
                if ann.id not in seen_ids:
                    mask_annotations.append(ann)
                    seen_ids.add(ann.id)

            # Source 2: annotation manager (covers images not yet opened in the canvas)
            for ann in self.annotation_window.image_annotations_dict.get(image_path, []):
                if (getattr(ann, 'is_mask_annotation', False) and ann.id not in seen_ids):
                    mask_annotations.append(ann)
                    seen_ids.add(ann.id)

        return mask_annotations

    def _update_annotation_stats_cache(self, force=False, recompute_masks=False):
        """
        Fill in the per-annotation statistics cache the index is built against.

        Entries already held are kept unless `force`, which re-reads them.

        Either way a mask's pixel pass is only paid when the mask needs it.
        MaskAnnotation sets its own `_stats_cache` to None from every method
        that touches mask_data, and exposes `cached_statistics` to read that
        without triggering work, so None means stale and a dict - empty
        included, which is a mask with nothing painted - means current. Reading
        it is what makes opening this dialog cheap on a project whose masks have
        not changed.

        `recompute_masks` ignores that and goes back to the pixels for every
        mask. Refresh passes it, so the button still means recount everything.

        MaskAnnotations manage their own stats cache internally; we call
        recalculate_class_statistics() directly to guarantee fresh data
        regardless of whether the internal cache was previously set to an
        empty dict (e.g. computed before any pixels were painted).

        Vector annotations go through _vector_class_statistics() rather than
        their own get_class_statistics(), which is anything but cheap: see that
        method.
        """
        if force:
            self._stats_cache.clear()
        self._project_labels = list(self.main_window.label_window.labels)

        all_annotations = list(self.annotation_window.annotations_dict.values())
        all_annotations.extend(self.get_mask_annotations())
        unique_annotations = {anno.id: anno for anno in all_annotations}.values()

        for anno in unique_annotations:
            is_mask = anno.__class__.__name__ == 'MaskAnnotation'

            # A mask is always re-read, because reading is a property access and
            # it is the only way to notice one edited since the last pass. A
            # vector's statistics are its own label and area, which cannot change
            # while this dialog is up, so holding an entry is reason to skip it.
            if not is_mask and anno.id in self._stats_cache:
                continue

            try:
                if is_mask:
                    stats = None if recompute_masks else getattr(anno, 'cached_statistics', None)
                    if stats is None:
                        stats = anno.recalculate_class_statistics()
                    self._stats_cache[anno.id] = stats or {}
                else:
                    self._stats_cache[anno.id] = self._vector_class_statistics(anno)
            except Exception as e:
                if anno.id not in self._stats_cache:
                    self._stats_cache[anno.id] = {}
                print(f"Error caching stats for annotation {anno.id}: {e}")

    @staticmethod
    def _vector_area(annotation):
        """A vector annotation's area, without constructing a Shapely polygon.

        Each branch is the value that annotation's own get_area() computes: a
        patch is a square of side annotation_size, a rectangle's box area is its
        width times its height, and a polygon's is the shoelace of its shell
        less its holes. Anything else falls through to get_area() itself, so an
        annotation type added later is correct before it is fast.

        Args:
            annotation: The vector annotation to measure.

        Returns:
            float: Area in pixels.
        """
        kind = annotation.__class__.__name__

        if kind == 'PatchAnnotation':
            size = float(annotation.annotation_size)
            return size * size

        if kind == 'RectangleAnnotation':
            width = annotation.bottom_right.x() - annotation.top_left.x()
            height = annotation.bottom_right.y() - annotation.top_left.y()
            return abs(width) * abs(height)

        if kind == 'PolygonAnnotation':
            area = shoelace_area(annotation.points)
            for hole in (annotation.holes or ()):
                area -= shoelace_area(hole)
            return max(area, 0.0)

        return float(annotation.get_area())

    def _vector_class_statistics(self, annotation):
        """One vector annotation's statistics entry, without Shapely.

        get_class_statistics() routes through get_area(), which builds a Shapely
        polygon per call: measured at 16 to 24 us, so a project of a million
        annotations spent most of twenty seconds here on every Refresh, which
        clears this cache and so re-reads every one of them.

        The area itself is never read. _annotation_labels, the readiness check
        and the table all ask only whether pixel_count is above zero, so the
        whole of that time went on computing exact polygon areas to answer a
        question about zero.

        Args:
            annotation: The vector annotation to describe.

        Returns:
            dict: {label code: {pixel_count, percentage}}, as
                get_class_statistics() returns for a vector.
        """
        return {
            annotation.label.short_label_code: {
                "pixel_count": int(self._vector_area(annotation)),
                "percentage": 100.0,
            }
        }

    def refresh_all(self):
        """Refresh recomputes the mask statistics as well as the index.

        The only way to pick up a mask edited since this dialog opened, so the
        button keeps meaning "recount everything".
        """
        self._stats_cache_force = True
        self._stats_mask_recompute = True
        super().refresh_all()

    def _indexable_annotations(self):
        """Vector annotations by type, plus the masks, deduplicated by id.

        Masks are not in annotations_dict, so the base class would never see
        them. get_mask_annotations() reads the image source itself, which is one
        reason the image source radio has to invalidate the index.
        """
        allowed_types = set()
        if self.include_patches_checkbox.isChecked():
            allowed_types.add('PatchAnnotation')
        if self.include_rectangles_checkbox.isChecked():
            allowed_types.add('RectangleAnnotation')
        if self.include_polygons_checkbox.isChecked():
            allowed_types.add('PolygonAnnotation')

        unique = {}
        for annotation in self.annotation_window.annotations_dict.values():
            if annotation.__class__.__name__ in allowed_types:
                unique[annotation.id] = annotation

        if self.include_masks_checkbox.isChecked():
            for mask in self.get_mask_annotations():
                unique[mask.id] = mask

        return list(unique.values())

    def _annotation_labels(self, annotation):
        """Every label this annotation actually paints pixels for.

        A mask holds one entry per painted class, so it is indexed under each.
        The pixel_count test is the one mask_contains_selected_labels and the
        readiness check already used; applying it here too means a class that
        appears in a mask's statistics with nothing painted stops counting
        towards that label in the table while being ignored everywhere else.
        """
        stats = self._stats_cache.get(annotation.id) or {}
        return tuple(label for label, entry in stats.items()
                     if (entry or {}).get('pixel_count', 0) > 0)

    def rebuild_annotation_index(self):
        """Make sure the statistics are in hand, then index against them."""
        # Scanned once here and read from the cache by both passes below
        self._cached_mask_annotations = self.get_mask_annotations()
        try:
            self._update_annotation_stats_cache(force=self._stats_cache_force,
                                                recompute_masks=self._stats_mask_recompute)
            self._stats_cache_force = False
            self._stats_mask_recompute = False
            super().rebuild_annotation_index()
        finally:
            # Held for one rebuild only, so a mask edited later is still seen
            self._cached_mask_annotations = None

    def compute_split_label_counts(self):
        """Sum the splits, counting masks that sit on a source path.

        A mask's image_path is the source ('video.mp4'), never a frame path
        ('video.mp4::frame_7'), so a video split made of frame paths never
        matches it and the base class's per-path sum misses it. Each split's
        distinct sources are added once, and only where the source is not
        already one of its paths: a static source is its own sample path and the
        base pass has counted it.
        """
        super().compute_split_label_counts()

        path_counts = self._path_counts
        path_source = self._path_source
        selected = self._applied_labels

        split_counts = list(self._split_label_counts)
        split_totals = list(self._split_totals)

        for index, image_paths in enumerate((self.train_images, self.val_images, self.test_images)):
            counted = set(image_paths)
            counts = split_counts[index]
            total = split_totals[index]

            for source in {path_source.get(path) for path in counted}:
                if source is None or source in counted:
                    continue
                for label, count in path_counts.get(source, {}).items():
                    if label in selected:
                        counts[label] = counts.get(label, 0) + count
                        total += count

            split_totals[index] = total

        self._split_label_counts = tuple(split_counts)
        self._split_totals = tuple(split_totals)

    def mask_contains_selected_labels(self, mask_annotation):
        """
        Check if a mask annotation contains any of the selected labels.

        Reads from the pre-built export stats cache.  If the cache entry is
        absent or empty (e.g. due to a cache miss), falls back to asking the
        annotation directly so the answer is always authoritative.

        Args:
            mask_annotation (MaskAnnotation): The mask annotation to check

        Returns:
            bool: True if mask contains at least one selected label with pixels
        """
        if not self.selected_labels:
            return False

        class_stats = self._stats_cache.get(mask_annotation.id) or {}

        # Cache miss or stale empty entry — ask the annotation directly
        if not class_stats:
            class_stats = mask_annotation.get_class_statistics()

        for label_code in self.selected_labels:
            if label_code in class_stats and class_stats[label_code].get('pixel_count', 0) > 0:
                return True

        return False

    def filter_annotations(self):
        """
        Filter both mask and vector annotations based on the selected types and labels.
        This version now calls the MODIFIED mask_contains_selected_labels.

        Returns:
            list: List of filtered annotations (both mask and vector types).
        """
        annotations = []
        selected_sources = set(self.get_selected_source_paths())
        frame_stride = self._frame_stride()
        
        # Get and filter MASK annotations if selected
        if self.include_masks_checkbox.isChecked():
            mask_annotations = self.get_mask_annotations()  # Gets ALL masks
            # This call now uses the FAST, cached version
            for mask in mask_annotations:
                if normalize_source_path(mask.image_path) in selected_sources and self.mask_contains_selected_labels(mask):
                    annotations.append(mask)

        # Get and filter VECTOR annotations in a single pass
        allowed_types = set()
        if self.include_patches_checkbox.isChecked():
            allowed_types.add('PatchAnnotation')
        if self.include_rectangles_checkbox.isChecked():
            allowed_types.add('RectangleAnnotation')
        if self.include_polygons_checkbox.isChecked():
            allowed_types.add('PolygonAnnotation')

        selected_set = set(self.selected_labels)

        filtered_vectors = [
            a for a in self.annotation_window.annotations_dict.values()
            if a.__class__.__name__ in allowed_types
            and a.label.short_label_code in selected_set
            and normalize_source_path(a.image_path) in selected_sources
            and frame_matches_stride(a.image_path, frame_stride)
        ]

        annotations.extend(filtered_vectors)
            
        # Return a list of unique annotations
        return list({anno.id: anno for anno in annotations}.values())

    def populate_class_filter_list(self):
        """
        Populate the class filter list from the cache.
        """
        with busy_cursor():
            # Set the row count to 0
            self.label_counts_table.setRowCount(0)

            # One mask scan for this whole pass: the statistics below and the
            # annotation list after them both want the same list, and building
            # it walks every image with a get_raster() per image.
            self._cached_mask_annotations = self.get_mask_annotations()
            try:
                # The index is built right after this, against these statistics
                self._update_annotation_stats_cache(force=self._stats_cache_force,
                                                    recompute_masks=self._stats_mask_recompute)
                self._stats_cache_force = False
                self._stats_mask_recompute = False

                label_counts = {}  # Number of annotations/masks containing each label
                label_image_counts = {}  # Set of unique images containing each label

                # Get all annotations we have stats for
                all_annotations = (list(self.annotation_window.annotations_dict.values())
                                   + self._cached_mask_annotations)
            finally:
                # Held for this pass only, so a mask edited later is still seen
                self._cached_mask_annotations = None

            unique_annotations = {anno.id: anno for anno in all_annotations}.values()

            unique_annotations_list = list(unique_annotations)
            # Create a progress bar
            progress_bar = ProgressBar(self, "Populating Class Lists")
            progress_bar.show()
            progress_bar.start_progress(len(unique_annotations_list))

            with closing_progress_bar(progress_bar):
                for annotation in unique_annotations_list:
                    image_path = annotation.image_path

                    # --- Read from the cache ---
                    class_stats = self._stats_cache.get(annotation.id, {})

                    # Handle MaskAnnotation: iterate through its internal labels
                    if annotation.__class__.__name__ == 'MaskAnnotation':
                        for label_code, stats in class_stats.items():
                            if stats.get('pixel_count', 0) > 0:
                                if label_code in label_counts:
                                    label_counts[label_code] += 1
                                    label_image_counts[label_code].add(image_path)
                                else:
                                    label_counts[label_code] = 1
                                    label_image_counts[label_code] = {image_path}

                    # Handle Vector Annotations
                    else:
                        for label_code in class_stats.keys():
                            if label_code != 'Review':
                                if label_code in label_counts:
                                    label_counts[label_code] += 1
                                    label_image_counts[label_code].add(image_path)
                                else:
                                    label_counts[label_code] = 1
                                    label_image_counts[label_code] = {image_path}

                    progress_bar.update_progress()

                # If no annotations are found, populate with all available project labels
                if not label_counts:
                    for label in self.main_window.label_window.labels:
                        if label.short_label_code != 'Review':
                            label_counts[label.short_label_code] = 0
                            label_image_counts[label.short_label_code] = set()

                # Sort and populate the table
                sorted_label_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)

                self.label_counts_table.setColumnCount(len(self.TABLE_HEADERS))
                self.label_counts_table.setHorizontalHeaderLabels(self.TABLE_HEADERS)
                self.label_counts_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)

                # Labels hidden in the Label Window start unchecked, the same way
                # the Image Source defaults to the filtered table.
                hidden_codes = self.get_hidden_label_codes()
                target_codes = self.remap_target_codes()
                self._prune_label_remap(target_codes)

                label_rows = []
                self.label_counts_table.setUpdatesEnabled(False)
                for row, (label, count) in enumerate(sorted_label_counts):
                    label_rows.append(self.add_label_row(row,
                                                         label,
                                                         count,
                                                         len(label_image_counts.get(label, set())),
                                                         hidden_codes,
                                                         target_codes))
                self.label_counts_table.setUpdatesEnabled(True)
                progress_bar.finish_progress()

        # The base class's table loop reads these rows, and the index has to be
        # rebuilt against the statistics just gathered.
        self._label_rows = label_rows
        self._index_dirty = True
        self._reset_selection_state()

    def check_label_distribution(self):
        """
        Check the label distribution in the splits using the cache.
        Override base class method to work with mask annotations.
        
        Returns:
            bool: True if all labels are present in all splits and split config is allowed, False otherwise.
        """
        # Get the ratios from the spinboxes
        train_ratio = self.train_ratio_spinbox.value()
        val_ratio = self.val_ratio_spinbox.value()
        test_ratio = self.test_ratio_spinbox.value()
    
        # Only allow these split combinations:
        # - Train only
        # - Test only
        # - Train/Val
        # - Train/Val/Test
        
        allowed = False

        # Train only
        if train_ratio == 1.0 and val_ratio == 0 and test_ratio == 0:
            allowed = True
            
        # Test only
        elif train_ratio == 0 and val_ratio == 0 and test_ratio == 1.0:
            allowed = True
            
        # Train/Val
        elif train_ratio > 0 and val_ratio > 0 and test_ratio == 0:
            if abs(train_ratio + val_ratio - 1.0) < 1e-9:
                allowed = True
                
        # Train/Val/Test
        elif train_ratio > 0 and val_ratio > 0 and test_ratio > 0:
            if abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9:
                allowed = True

        if not allowed:
            return False
    
        # Summed for the table already, straight out of the index. The
        # three passes over the split annotation lists that stood here
        # were the same arithmetic over the same statistics.
        # Folded onto the export class names first: pointing a sparse label at
        # a well covered one is the reason the remap exists, so the question is
        # whether the class that reaches disk is covered, not the source label.
        train_label_counts, val_label_counts, test_label_counts = [
            self._fold_counts_to_targets(counts) for counts in self._split_label_counts]
        train_total, val_total, test_total = self._split_totals

        # Check the conditions for each split
        for label in self.export_class_names():
            if train_ratio > 0 and train_label_counts.get(label, 0) == 0:
                return False
            if val_ratio > 0 and val_label_counts.get(label, 0) == 0:
                return False
            if test_ratio > 0 and test_label_counts.get(label, 0) == 0:
                return False
    
        # Additional checks to ensure no empty splits
        if train_ratio > 0 and train_total == 0:
            return False
        if val_ratio > 0 and val_total == 0:
            return False
        if test_ratio > 0 and test_total == 0:
            return False
    
        if self.allows_unlabeled_video_export() and self.include_negatives_radio.isChecked():
            if train_ratio > 0 and len(self.train_images) == 0:
                return False
            if val_ratio > 0 and len(self.val_images) == 0:
                return False
            if test_ratio > 0 and len(self.test_images) == 0:
                return False
            return True

        return True

    def determine_splits(self):
        """
        Assign annotations in self.selected_annotations to train/val/test lists.

        Overrides the base class to handle MaskAnnotation paths correctly.
        A mask annotation's image_path is always the underlying source path
        (e.g. 'video.mp4' for video sources) rather than a virtual frame path
        ('video.mp4::frame_N').  The base class does an exact set membership
        check, which fails for video masks.  This override also accepts a match
        when the normalized source paths agree.
        """
        train_set = set(self.train_images)
        val_set = set(self.val_images)
        test_set = set(self.test_images)

        # Pre-build normalized source sets so video masks can match
        train_sources = {normalize_source_path(p) for p in train_set}
        val_sources = {normalize_source_path(p) for p in val_set}
        test_sources = {normalize_source_path(p) for p in test_set}

        def _in_split(ann, exact_set, source_set):
            return (ann.image_path in exact_set or
                    normalize_source_path(ann.image_path) in source_set)

        self.train_annotations = [
            a for a in self.selected_annotations if _in_split(a, train_set, train_sources)
        ]
        self.val_annotations = [
            a for a in self.selected_annotations if _in_split(a, val_set, val_sources)
        ]
        self.test_annotations = [
            a for a in self.selected_annotations if _in_split(a, test_set, test_sources)
        ]

    def create_dataset(self, output_dir_path):
        """
        Create the semantic segmentation dataset in YOLO format.
        
        Args:
            output_dir_path (str): Path to the output directory.
        """
        # Create the yaml file
        yaml_path = os.path.join(output_dir_path, 'data.yaml')
        train_dir = os.path.join(output_dir_path, 'train')
        val_dir = os.path.join(output_dir_path, 'valid')
        test_dir = os.path.join(output_dir_path, 'test')
        
        names = self.export_class_names()
        treat_as_background = self.background_radio.isChecked()

        # SHIFT LOGIC: Inject Background at 0 if selected
        names_dict = {}
        if treat_as_background:
            names_dict[0] = "background"
            for i, name in enumerate(names):
                names_dict[i + 1] = name
            num_classes = len(names) + 1
        else:
            for i, name in enumerate(names):
                names_dict[i] = name
            num_classes = len(names)

        # Paths are relative to this YAML file's own location (no 'path' key), so the
        # dataset still resolves correctly if this folder is later moved, copied, or
        # shared to a different machine.
        data = {
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': num_classes,
            'names': names_dict,
            'masks_dir': 'masks'
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        # Create the train, val, and test directories with images and masks subdirectories
        os.makedirs(f"{train_dir}/images", exist_ok=True)
        os.makedirs(f"{train_dir}/masks", exist_ok=True)
        os.makedirs(f"{val_dir}/images", exist_ok=True)
        os.makedirs(f"{val_dir}/masks", exist_ok=True)
        os.makedirs(f"{test_dir}/images", exist_ok=True)
        os.makedirs(f"{test_dir}/masks", exist_ok=True)

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self, "Creating Semantic Segmentation Dataset")
        progress_bar.show()

        try:
            # Process each split
            self.process_annotations(self.train_annotations, train_dir, "Training")
            self.process_annotations(self.val_annotations, val_dir, "Validation")
            self.process_annotations(self.test_annotations, test_dir, "Testing")

        finally:
            # Restore the cursor to the default cursor
            QApplication.restoreOverrideCursor()
            progress_bar.finish_progress()
            progress_bar.close()
            gc.collect()

    def process_mask_annotations(self, annotations, images_dir, labels_dir, progress_bar, image_paths):
        """
        Process both mask and vector annotations for a specific split in YOLO format.
        
        Args:
            annotations (list): List of annotation objects (both mask and vector types) for this split
            images_dir (str): Directory to save images
            labels_dir (str): Directory to save label masks
            progress_bar: Progress bar object
            image_paths (list): All image paths for this split (including negatives)
        """
        # Separate mask and vector annotations
        mask_annotations = [ann for ann in annotations if ann.__class__.__name__ == 'MaskAnnotation']
        vector_annotation_types = ['PatchAnnotation', 'RectangleAnnotation', 'PolygonAnnotation']
        vector_annotations = [ann for ann in annotations if ann.__class__.__name__ in vector_annotation_types]
        
        # Create mappings by image path
        image_to_mask = {normalize_source_path(ann.image_path): ann for ann in mask_annotations}
        
        # Group vector annotations by image path
        image_to_vectors = {}
        for ann in vector_annotations:
            if ann.image_path not in image_to_vectors:
                image_to_vectors[ann.image_path] = []
            image_to_vectors[ann.image_path].append(ann)
        
        for i, image_path in enumerate(image_paths):
            try:
                source_path, frame_idx, raster = resolve_sample_source(image_path, self.image_window.raster_manager)

                # Copy or materialize the image into the export directory.
                image_filename = build_sample_export_name(image_path)
                image_output_path = os.path.join(images_dir, image_filename)
                if not materialize_sample_image(image_path, self.image_window.raster_manager, image_output_path):
                    raise RuntimeError(f"Failed to export image sample: {image_path}")
                
                # Create semantic segmentation mask
                mask_output_path = os.path.join(labels_dir, build_sample_export_name(image_path, ".png"))
                
                # Combine mask and vector annotations for this image
                self.create_combined_semantic_mask(
                    image_path,
                    image_to_mask.get(normalize_source_path(image_path)),
                    image_to_vectors.get(image_path, image_to_vectors.get(source_path, [])),
                    mask_output_path
                )
                
                # Update progress
                progress_bar.update_progress()
                
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")

    def create_semantic_mask(self, mask_annotation, output_path):
        """
        Create a semantic segmentation mask from a MaskAnnotation in YOLO format.
        Single channel PNG with uint8 values where 0 is the first class and 255 is ignore/unlabeled.
        
        Args:
            mask_annotation (MaskAnnotation): The mask annotation to process
            output_path (str): Path to save the mask
        """
        # Get the mask data
        mask_data = mask_annotation.mask_data.copy()
        
        # Determine fill value and index offset
        treat_as_background = getattr(self, 'background_radio', None) and self.background_radio.isChecked()
        fill_value = 0 if treat_as_background else 255
        index_offset = 1 if treat_as_background else 0
        
        # One pass over the mask's own classes, resolved through the export
        # index, so a remapped pair of labels lands on one value.
        export_index = self.export_label_index()
        label_to_index = {}
        for class_id, label_obj in mask_annotation.class_id_to_label_map.items():
            index = export_index.get(label_obj.short_label_code)
            if index is not None:
                # Apply offset here
                label_to_index[class_id] = index + index_offset

        # Create output mask defaulting to our chosen fill_value
        output_mask = np.full_like(mask_data, fill_value, dtype=np.uint8)
        
        for class_id, label_index in label_to_index.items():
            class_mask = (mask_data == class_id) | (mask_data == class_id + mask_annotation.LOCK_BIT)
            output_mask[class_mask] = label_index
        
        mask_image = Image.fromarray(output_mask, mode='L')
        mask_image.save(output_path)

    def create_combined_semantic_mask(self, image_path, mask_annotation, vector_annotations, output_path):
        """
        Create a combined semantic segmentation mask from both mask and vector annotations.
        
        Args:
            image_path (str): Path to the source image
            mask_annotation (MaskAnnotation or None): Mask annotation for this image
            vector_annotations (list): List of vector annotations for this image
            output_path (str): Path to save the combined mask
        """
        try:
            height, width, _ = sample_dimensions(image_path, self.image_window.raster_manager)
            
            # Determine fill value and index offset
            treat_as_background = getattr(self, 'background_radio', None) and self.background_radio.isChecked()
            fill_value = 0 if treat_as_background else 255
            index_offset = 1 if treat_as_background else 0

            # Start with chosen fill value
            combined_mask = np.full((height, width), fill_value, dtype=np.uint8)
            
            if mask_annotation is not None:
                mask_data = mask_annotation.mask_data.copy()
                export_index = self.export_label_index()
                label_to_index = {}
                for class_id, label_obj in mask_annotation.class_id_to_label_map.items():
                    index = export_index.get(label_obj.short_label_code)
                    if index is not None:
                        label_to_index[class_id] = index + index_offset
                
                for class_id, label_index in label_to_index.items():
                    class_mask = ((mask_data == class_id) |
                                  (mask_data == class_id + mask_annotation.LOCK_BIT))
                    combined_mask[class_mask] = label_index
            
            if vector_annotations:
                vector_mask = self.rasterize_vector_annotations(vector_annotations, image_path)
                # Overlay non-fill values
                combined_mask = np.where(vector_mask != fill_value, vector_mask, combined_mask)
            
            mask_image = Image.fromarray(combined_mask, mode='L')
            mask_image.save(output_path)
            
        except Exception as e:
            print(f"Error creating combined semantic mask for {image_path}: {e}")
            self.create_empty_mask(image_path, output_path)
    
    def create_empty_mask(self, image_path, output_path):
        """
        Create an empty semantic segmentation mask (all background) for images without annotations.
        
        Args:
            image_path (str): Path to the original image to get dimensions
            output_path (str): Path to save the empty mask
        """
        try:
            height, width, _ = sample_dimensions(image_path, self.image_window.raster_manager)
            
            # Check toggle for background vs ignore
            treat_as_background = getattr(self, 'background_radio', None) and self.background_radio.isChecked()
            fill_value = 0 if treat_as_background else 255
            
            empty_mask = np.full((height, width), fill_value, dtype=np.uint8)
            
            mask_image = Image.fromarray(empty_mask, mode='L')
            mask_image.save(output_path)
            
        except Exception as e:
            print(f"Error creating empty mask for {image_path}: {e}")

    def rasterize_vector_annotations(self, vector_annotations, image_path):
        """
        Rasterize vector annotations (patches, rectangles, polygons) into a semantic mask.
        
        Args:
            vector_annotations (list): List of vector annotation objects
            image_path (str): Path to the image for getting dimensions
            
        Returns:
            np.ndarray: Semantic mask with rasterized annotations
        """
        try:
            height, width, _ = sample_dimensions(image_path, self.image_window.raster_manager)
            
            # Determine fill value and index offset
            treat_as_background = getattr(self, 'background_radio', None) and self.background_radio.isChecked()
            fill_value = 0 if treat_as_background else 255
            index_offset = 1 if treat_as_background else 0

            output_mask = np.full((height, width), fill_value, dtype=np.uint8)
            
            annotations_by_label = {}
            for annotation in vector_annotations:
                label_code = annotation.label.short_label_code
                if label_code not in annotations_by_label:
                    annotations_by_label[label_code] = []
                annotations_by_label[label_code].append(annotation)
            
            export_index = self.export_label_index()
            for label_code, label_annotations in annotations_by_label.items():
                if label_code in export_index:
                    # Apply offset to the class index
                    class_index = export_index[label_code] + index_offset
                    
                    geometries = []
                    for annotation in label_annotations:
                        geom = self._annotation_to_geometry(annotation)
                        if geom is not None:
                            geometries.append(geom)
                    
                    if geometries:
                        class_mask = rasterize(
                            [(geom, class_index) for geom in geometries],
                            out_shape=(height, width),
                            fill=fill_value,
                            dtype=np.uint8
                        )

                        output_mask = np.where(class_mask != fill_value, class_mask, output_mask)
            
            return output_mask
            
        except Exception as e:
            print(f"Error rasterizing vector annotations for {image_path}: {e}")
            fill_val = 0 if (getattr(self, 'background_radio', None) and self.background_radio.isChecked()) else 255
            return np.full((height, width), fill_val, dtype=np.uint8)

    def _annotation_to_geometry(self, annotation):
        """
        Convert an annotation object to a shapely geometry.
        
        Args:
            annotation: Annotation object (patch, rectangle, or polygon)
            
        Returns:
            shapely geometry or None
        """
        try:
            polygon = annotation.get_polygon()
            points = [(polygon.at(i).x(), polygon.at(i).y()) for i in range(polygon.count())]
            if len(points) >= 3:
                return Polygon(points)
        except Exception as e:
            print(f"Error converting annotation to geometry: {e}")
            
        return None

    def process_annotations(self, annotations, split_dir, split):
        """
        Process annotations for semantic segmentation export in YOLO format.
        
        Args:
            annotations (list): List of MaskAnnotation objects for this split
            split_dir (str): Directory for this split  
            split (str): Split name (e.g., "Training", "Validation", "Testing")
        """
        # Determine the full list of images for this split (including negatives)
        if split == "Training":
            image_paths = self.train_images
        elif split == "Validation":
            image_paths = self.val_images
        elif split == "Testing":
            image_paths = self.test_images
        else:
            image_paths = []

        if not image_paths:
            return

        # Set up progress bar
        progress_bar = ProgressBar(self, title=f"Creating {split} Dataset")
        progress_bar.show()
        progress_bar.start_progress(len(image_paths))

        try:
            # Create images and labels directories
            images_dir = os.path.join(split_dir, 'images')
            labels_dir = os.path.join(split_dir, 'masks')
            
            # Process all images in this split
            self.process_mask_annotations(annotations, images_dir, labels_dir, progress_bar, image_paths)
            
        finally:
            progress_bar.stop_progress()
            progress_bar.close()