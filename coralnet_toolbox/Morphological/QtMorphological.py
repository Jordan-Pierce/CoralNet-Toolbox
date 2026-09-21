import warnings

import numpy as np

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import (
    QApplication, QDialog, QMessageBox, QLabel, QVBoxLayout,
    QHBoxLayout, QPushButton, QGroupBox, QSizePolicy, QTabWidget, QWidget,
)

from coralnet_toolbox.QtActions import (
    AddAnnotationsAction,
    DeleteAnnotationsAction,
    CompoundAction,
    MaskEditAction,
    MergeAnnotationsAction,
)

from coralnet_toolbox.Annotations import MultiPolygonAnnotation, PolygonAnnotation

from coralnet_toolbox.Morphological.overlap_ops import (
    MERGE,
    SUBTRACT,
    ImagePlan,
    clean_geometry,
    plan_merge,
    plan_remove,
    plan_subtract,
    polygon_parts,
)
from coralnet_toolbox.Morphological.QtOverlapOperations import OverlapOperationsTab

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_window_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class AnnotationOperationsDialog(QDialog):
    """Modeless dialog for whole-image annotation operations.

    The Bake / Unbake tab converts between vector annotations and the mask; the
    Overlaps tab subtracts, removes or merges vector annotations by label. Both
    operate on whichever image rows are highlighted in the ImageWindow, matching the
    multi-image workflow used by PatchSamplingDialog. Defaults to the current image only.
    """

    # The "bake options" groupbox has been removed; this is the value it used to control.
    MIN_HOLE_AREA = 500

    def __init__(self, annotation_window, parent=None):
        super().__init__(parent or annotation_window)
        self.annotation_window = annotation_window
        self.main_window = annotation_window.main_window
        self.image_window = annotation_window.main_window.image_window

        self.setWindowTitle("Annotation Operations")
        self.setWindowIcon(get_window_icon("coralnet.svg"))
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setMinimumWidth(340)

        self.layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.build_bake_tab(), "Bake / Unbake")
        self.overlap_tab = OverlapOperationsTab(self)
        self.tabs.addTab(self.overlap_tab, "Overlaps")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.layout.addWidget(self.tabs)

        self.setup_footer_layout()
        self.on_tab_changed(self.tabs.currentIndex())

        # Keep the status label in sync with row highlighting in the ImageWindow
        self.image_window.table_model.rowsChanged.connect(self.update_status_label)

    def build_bake_tab(self):
        """Build the Bake / Unbake tab's explanatory text, and its buttons for the bottom row."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        group_box = QGroupBox("Information")
        layout = QVBoxLayout(group_box)

        info_label = QLabel(
            "Bake rasterizes vector annotations into the mask.\n"
            "Unbake vectorizes mask regions into vector annotations.\n\n"
            "Highlight one or more image rows in the Raster Window to apply to those "
            "images; otherwise only the current image is used."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        tab_layout.addWidget(group_box)
        tab_layout.addStretch()

        # Shown in the dialog's bottom row while this tab is current
        self.bake_actions = QWidget()
        button_layout = QHBoxLayout(self.bake_actions)
        button_layout.setContentsMargins(0, 0, 0, 0)

        self.bake_button = QPushButton("Bake")
        self.bake_button.setToolTip("Rasterize vector annotations into the mask on the applicable images.")
        self.bake_button.clicked.connect(lambda: self.run_bulk_operation("bake"))
        button_layout.addWidget(self.bake_button)

        self.unbake_button = QPushButton("Unbake")
        self.unbake_button.setToolTip("Vectorize mask regions into annotations on the applicable images.")
        self.unbake_button.clicked.connect(lambda: self.run_bulk_operation("unbake"))
        button_layout.addWidget(self.unbake_button)

        return tab

    def setup_footer_layout(self):
        """The one bottom row: highlighted image count, the current tab's buttons, and Close."""
        footer_layout = QHBoxLayout()

        # Status label showing the number of highlighted images
        self.status_label = QLabel("No images highlighted")
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        footer_layout.addWidget(self.status_label)
        footer_layout.addStretch()

        footer_layout.addWidget(self.bake_actions)
        footer_layout.addWidget(self.overlap_tab.action_widget)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)
        footer_layout.addWidget(close_button)

        self.layout.addLayout(footer_layout)

    def on_tab_changed(self, index):
        """Show the current tab's buttons and size the dialog to it; QTabWidget
        otherwise sizes every tab to the largest."""
        self.bake_actions.setVisible(self.tabs.widget(index) is not self.overlap_tab)
        self.overlap_tab.action_widget.setVisible(self.tabs.widget(index) is self.overlap_tab)

        for i in range(self.tabs.count()):
            policy = QSizePolicy.Preferred if i == index else QSizePolicy.Ignored
            self.tabs.widget(i).setSizePolicy(policy, policy)
        # The tab widget caches its size hint; drop it so adjustSize sees the new policies.
        self.tabs.updateGeometry()
        self.layout.activate()
        self.adjustSize()

    def showEvent(self, event):
        """Handle dialog show event."""
        super().showEvent(event)

        # Automatically highlight the current image if nothing relevant is highlighted yet
        current_image_path = self.annotation_window.current_image_path
        if current_image_path:
            highlighted_paths = self.image_window.table_model.get_highlighted_paths()
            if current_image_path not in highlighted_paths:
                self.image_window.table_model.set_highlighted_paths([current_image_path])

        # Labels may have been added, removed or recoloured since the last show
        self.overlap_tab.refresh_labels()
        self.update_status_label()

    def update_status_label(self):
        """Update the status label to show the number of images highlighted."""
        highlighted_paths = self.image_window.table_model.get_highlighted_paths()
        count = len(highlighted_paths)
        if count == 0:
            self.status_label.setText("No images highlighted")
        elif count == 1:
            self.status_label.setText("1 image highlighted")
        else:
            self.status_label.setText(f"{count} images highlighted")

    def run_bulk_operation(self, mode):
        """Bake or unbake annotations across the highlighted images, filtering out
        images the operation doesn't apply to.

        Args:
            mode (str): Either "bake" or "unbake".
        """
        image_paths = self.image_window.table_model.get_highlighted_paths()
        if not image_paths:
            QMessageBox.warning(self, "No Selection", "Please highlight at least one image row.")
            return

        aw = self.annotation_window

        # Check what can be done before doing anything, filtering out images that don't apply
        if mode == "bake":
            applicable_paths = [p for p in image_paths if aw.get_bakeable_annotations(p)]
        else:
            applicable_paths = [p for p in image_paths if aw.has_unbakeable_mask_regions(p)]

        if not applicable_paths:
            noun = "vector annotations to bake" if mode == "bake" else "mask regions to unbake"
            QMessageBox.information(self, "Nothing To Do", f"None of the highlighted images have {noun}.")
            return

        skipped_images = len(image_paths) - len(applicable_paths)

        progress_bar = None
        if len(applicable_paths) > 1:
            title = "Baking Annotations" if mode == "bake" else "Unbaking Annotations"
            progress_bar = ProgressBar(self, title=title)
            progress_bar.show()
            progress_bar.start_progress(len(applicable_paths))

        self.bake_button.setEnabled(False)
        self.unbake_button.setEnabled(False)

        total_changed = 0
        total_inner_skipped = 0
        try:
            for image_path in applicable_paths:
                if mode == "bake":
                    result = aw.bake_vector_annotations(image_path=image_path, show_status_message=False)
                    key = "baked"
                else:
                    result = aw.vectorize_mask_annotations(
                        image_path=image_path,
                        min_hole_area=self.MIN_HOLE_AREA,
                        show_status_message=False,
                    )
                    key = "vectorized"

                if result:
                    total_changed += result.get(key, 0)
                    total_inner_skipped += result.get("skipped", 0)

                if progress_bar is not None:
                    progress_bar.update_progress()
        finally:
            self.bake_button.setEnabled(True)
            self.unbake_button.setEnabled(True)
            if progress_bar is not None:
                progress_bar.stop_progress()
                progress_bar.close()

        self.update_status_label()

        verb = "Baked" if mode == "bake" else "Unbaked"
        noun = "vector annotations" if mode == "bake" else "mask regions"
        message = f"{verb} {total_changed} {noun} across {len(applicable_paths)} image(s)."
        if total_inner_skipped:
            message += f" Skipped {total_inner_skipped} that could not be converted."
        if skipped_images:
            message += f" {skipped_images} highlighted image(s) had nothing to {mode}."

        try:
            self.main_window.status_bar.showMessage(message, 4000)
        except Exception:
            pass


class MorphologicalMixin:
    """Mixin class providing bake/unbake and overlap annotation operations for AnnotationWindow."""

    def prompt_bake_or_unbake_annotations(self):
        """Show the modeless Annotation Operations dialog."""
        if not self.current_image_path:
            return False

        if getattr(self, '_bake_unbake_dialog', None) is None:
            self._bake_unbake_dialog = AnnotationOperationsDialog(self)

        dialog = self._bake_unbake_dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return True

    def get_bakeable_annotations(self, image_path):
        """Return the vector annotations on image_path that can be baked into its mask."""
        annotations = []
        for annotation in self.get_image_annotations(image_path):
            if getattr(annotation, 'is_mask_annotation', False):
                continue

            geometry_getter = getattr(annotation, 'get_rasterization_geometry', None)
            geometry = None
            if callable(geometry_getter):
                try:
                    geometry = geometry_getter()
                except Exception:
                    geometry = None

            if geometry is not None and not getattr(geometry, 'is_empty', False):
                annotations.append(annotation)

        return annotations

    def has_unbakeable_mask_regions(self, image_path):
        """Return True if image_path already has mask pixels that can be vectorized."""
        raster = self._resolve_raster_for_bake(image_path)
        mask_annotation = getattr(raster, 'mask_annotation', None) if raster is not None else None
        if mask_annotation is None:
            return False
        try:
            return bool(np.any(mask_annotation.mask_data % mask_annotation.LOCK_BIT))
        except Exception:
            return True

    def _resolve_raster_for_bake(self, image_path):
        """Return the raster for image_path if it is eligible for bake/unbake.

        Video frames are excluded except for the currently displayed frame, whose mask
        can only be read/written safely through current_mask_annotation (it re-seeds the
        shared per-video buffer from this frame's cached prediction). A bare VideoRaster
        path never carries its own mask.
        """
        if not image_path:
            return None
        if '::frame_' in str(image_path) and image_path != self.current_image_path:
            return None

        raster_manager = getattr(self.main_window.image_window, 'raster_manager', None)
        raster = raster_manager.get_raster(image_path) if raster_manager is not None else None
        if raster is None:
            return None
        if getattr(raster, 'raster_type', '') == 'VideoRaster':
            return None
        return raster

    def _get_mask_annotation_for_bake(self, image_path):
        """Get (creating if needed) the MaskAnnotation to bake/unbake for image_path."""
        if image_path == self.current_image_path:
            return self.current_mask_annotation

        raster = self._resolve_raster_for_bake(image_path)
        if raster is None:
            return None

        project_labels = self.main_window.label_window.labels
        mask_annotation = raster.get_mask_annotation(project_labels)
        try:
            self.annotation_manager.register_mask_annotation(mask_annotation)
        except Exception:
            pass
        return mask_annotation

    def bake_vector_annotations(self, image_path=None, show_status_message=True):
        """Bake one image's vector annotations into its mask and delete the vectors.

        This is the destructive counterpart to rasterize_annotations(): it
        permanently writes vector labels into the semantic mask and then removes
        the vector annotations from the image.

        Returns:
            dict | None: {"baked": n, "skipped": n} on success, else None.
        """
        image_path = image_path or self.current_image_path
        if not image_path:
            return None

        annotations = self.get_bakeable_annotations(image_path)
        if not annotations:
            if show_status_message:
                try:
                    self.main_window.status_bar.showMessage(
                        "No vector annotations on the current image can be baked into the mask.",
                        3000,
                    )
                except Exception:
                    pass
            return None

        mask_annotation = self._get_mask_annotation_for_bake(image_path)
        if mask_annotation is None:
            return None

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            _annotation_manager = getattr(self, 'annotation_manager', None)

            mask_annotation.blockSignals(True)
            if _annotation_manager is not None:
                _annotation_manager.blockSignals(True)

            baked_annotations = []
            skipped_annotations = []
            history_action = None
            delete_action = None
            try:
                history_action = MaskEditAction(mask_annotation, description="Bake vector annotations")
                bake_summary = mask_annotation.bake_annotations(annotations, history_action=history_action)

                baked_annotations = bake_summary.get("baked_annotations", []) if bake_summary else []
                skipped_annotations = bake_summary.get("skipped_annotations", []) if bake_summary else []

                if not baked_annotations:
                    try:
                        self.main_window.status_bar.showMessage(
                            "No vector annotations could be baked into the current mask.",
                            3000,
                        )
                    except Exception:
                        pass
                    return None

                self.unselect_annotations()

                delete_action = DeleteAnnotationsAction(self, baked_annotations)
                self.delete_annotations(baked_annotations, record_action=False)
            finally:
                if _annotation_manager is not None:
                    _annotation_manager.blockSignals(False)
                mask_annotation.blockSignals(False)

                try:
                    mask_annotation.refresh_graphics()
                    self.refresh_mask_annotation_view(mask_annotation)
                except Exception:
                    pass

            compound_action = CompoundAction(
                [history_action, delete_action],
                description="Bake vector annotations",
            )
            if history_action is not None and delete_action is not None:
                self.action_stack.push(compound_action)

            if show_status_message:
                try:
                    if skipped_annotations:
                        self.main_window.status_bar.showMessage(
                            f"Baked {len(baked_annotations)} vector annotations; skipped "
                            f"{len(skipped_annotations)} that could not be rasterized.",
                            4000,
                        )
                    else:
                        self.main_window.status_bar.showMessage(
                            f"Baked {len(baked_annotations)} vector annotations into the mask.",
                            3000,
                        )
                except Exception:
                    pass

            return {"baked": len(baked_annotations), "skipped": len(skipped_annotations)}
        finally:
            QApplication.restoreOverrideCursor()

    def vectorize_mask_annotations(self, image_path=None, min_hole_area: int = 500, show_status_message=True):
        """Convert an image's mask regions into vector annotations.

        Args:
            image_path: Image to operate on; defaults to the current image.
            min_hole_area: Minimum hole area in pixels to preserve as an
                interior ring. Holes smaller than this threshold are filled.
            show_status_message: Whether to post a status-bar message for this call.

        Returns:
            dict | None: {"vectorized": n, "skipped": n} on success, else None.
        """
        image_path = image_path or self.current_image_path
        if not image_path:
            return None

        mask_annotation = self._get_mask_annotation_for_bake(image_path)
        if mask_annotation is None:
            if show_status_message:
                try:
                    self.main_window.status_bar.showMessage(
                        "No mask annotation is available for the current image.",
                        3000,
                    )
                except Exception:
                    pass
            return None

        rejected_indices = []
        try:
            vector_annotations = mask_annotation.to_vector_annotations(
                transparency=self.main_window.get_transparency_value(),
                show_confidence=False,
                min_hole_area=min_hole_area,
                rejected_indices_out=rejected_indices,
                image_path=image_path,
            )
        except Exception:
            vector_annotations = []
            rejected_indices = []

        if not vector_annotations and not rejected_indices:
            if show_status_message:
                try:
                    self.main_window.status_bar.showMessage(
                        "No mask regions could be vectorized from the current image.",
                        3000,
                    )
                except Exception:
                    pass
            return None

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)

            _annotation_manager = getattr(self, 'annotation_manager', None)

            mask_annotation.blockSignals(True)
            if _annotation_manager is not None:
                _annotation_manager.blockSignals(True)

            add_action = None
            clear_action = None
            try:
                self.unselect_annotations()

                if vector_annotations:
                    add_action = AddAnnotationsAction(self, vector_annotations)
                    add_action.do()

                clear_action = MaskEditAction(mask_annotation, description="Vectorize mask annotations")
                mask_annotation.clear_pixels_for_annotations(
                    vector_annotations,
                    history_action=clear_action,
                    extra_flat_indices=rejected_indices,
                )
            finally:
                if _annotation_manager is not None:
                    _annotation_manager.blockSignals(False)
                mask_annotation.blockSignals(False)

                try:
                    mask_annotation.refresh_graphics()
                    self.refresh_mask_annotation_view(mask_annotation)
                except Exception:
                    pass

                try:
                    if '::frame_' in str(image_path):
                        self._sync_video_mask_to_cache()
                except Exception:
                    pass

            if clear_action is None or clear_action.is_empty():
                try:
                    if vector_annotations:
                        self.delete_annotations(vector_annotations, record_action=False)
                    if show_status_message:
                        self.main_window.status_bar.showMessage(
                            "No editable mask pixels were changed during vectorization.",
                            3000,
                        )
                except Exception:
                    pass
                return None

            actions = [action for action in (add_action, clear_action) if action is not None]
            if len(actions) > 1:
                self.action_stack.push(CompoundAction(
                    actions,
                    description="Vectorize mask annotations",
                ))
            else:
                self.action_stack.push(actions[0])

            if show_status_message:
                try:
                    message = f"Vectorized {len(vector_annotations)} mask regions into annotations."
                    if rejected_indices:
                        message += f" Discarded {len(rejected_indices)} sub-threshold regions."
                    self.main_window.status_bar.showMessage(message, 3000)
                except Exception:
                    pass

            return {"vectorized": len(vector_annotations), "skipped": len(rejected_indices)}
        finally:
            QApplication.restoreOverrideCursor()

    # ------------------------------------------------------------------------------------------------------------------
    # Overlap operations
    # ------------------------------------------------------------------------------------------------------------------

    # Overlap operations are polygon-only on both sides; rectangles, patches and
    # masks are left alone.
    _OVERLAP_TYPES = (PolygonAnnotation, MultiPolygonAnnotation)

    def _overlap_candidates(self, image_path, label_ids, include_unverified):
        """Return [(annotation, cleaned geometry)] for polygons on image_path matching the filters."""
        candidates = []
        for annotation in self.get_image_annotations(image_path):
            if not isinstance(annotation, self._OVERLAP_TYPES):
                continue
            if annotation.label.id not in label_ids:
                continue
            if not include_unverified and not annotation.verified:
                continue
            try:
                geometry = clean_geometry(annotation.get_rasterization_geometry())
            except Exception:
                geometry = None
            if geometry is not None:
                candidates.append((annotation, geometry))
        return candidates

    def plan_overlap_operation(self, image_path, spec):
        """Work out what an OverlapSpec would change on image_path, changing nothing.

        Returns:
            ImagePlan: the annotations to delete and the geometry replacing them.
        """
        plan = ImagePlan(image_path)
        targets = self._overlap_candidates(image_path, spec.target_label_ids, spec.include_unverified)
        plan.checked = len(targets)
        if not targets:
            return plan

        target_geoms = [geometry for _annotation, geometry in targets]

        if spec.operation == MERGE:
            keys = [annotation.label.id for annotation, _geometry in targets]
            for indices, geometry in plan_merge(target_geoms, keys):
                members = [targets[i][0] for i in indices]
                # Take confidence from the biggest member, and from an unverified one
                # if there is any, so a merge never quietly verifies a prediction.
                pool = [i for i in indices if not targets[i][0].verified] or indices
                template = targets[max(pool, key=lambda i: target_geoms[i].area)][0]
                plan.replaced.append((members, geometry, template))
            return plan

        references = self._overlap_candidates(image_path, spec.reference_label_ids, spec.include_unverified)
        if not references:
            return plan
        reference_geoms = [geometry for _annotation, geometry in references]

        if spec.operation == SUBTRACT:
            for index, geometry in plan_subtract(
                    target_geoms, reference_geoms, spec.min_overlap, spec.min_piece_area):
                annotation = targets[index][0]
                if geometry is None:
                    plan.removed.append(annotation)
                else:
                    plan.replaced.append(([annotation], geometry, annotation))
        else:
            for index in plan_remove(target_geoms, reference_geoms, spec.min_overlap):
                plan.removed.append(targets[index][0])

        return plan

    @staticmethod
    def _copy_annotation_state(template, annotation):
        """Carry label confidence and metadata over from the annotation being replaced.

        New annotations start verified with no machine confidence, so without this
        a clipped prediction would come back looking like a verified annotation.
        """
        annotation.verified = template.verified
        annotation.user_confidence = dict(template.user_confidence)
        annotation.machine_confidence = dict(template.machine_confidence)
        annotation.data = dict(template.data)
        annotation.metadata = dict(template.metadata)

    def _annotation_from_geometry(self, geometry, template):
        """Build a Polygon or MultiPolygon annotation from shapely geometry, styled after template."""
        parts = polygon_parts(geometry)
        if not parts:
            return None

        common_args = {
            "label": template.label,
            "image_path": template.image_path,
            "transparency": template.transparency,
            "show_confidence": template.show_confidence,
        }

        polygons = []
        for part in parts:
            # Shapely rings repeat their first point at the end; annotations don't.
            polygon = PolygonAnnotation(
                points=[QPointF(x, y) for x, y in part.exterior.coords[:-1]],
                holes=[[QPointF(x, y) for x, y in ring.coords[:-1]] for ring in part.interiors],
                simplify=False,
                **common_args,
            )
            self._copy_annotation_state(template, polygon)
            polygons.append(polygon)

        if len(polygons) == 1:
            return polygons[0]

        annotation = MultiPolygonAnnotation(polygons=polygons, **common_args)
        self._copy_annotation_state(template, annotation)
        return annotation

    def apply_overlap_plans(self, plans):
        """Make the changes in plans across all their images as a single undo step.

        Returns:
            tuple | None: (removed, added) annotation lists, or None if nothing changed.
        """
        removed = []
        added = []
        for plan in plans:
            removed.extend(plan.removed)
            for sources, geometry, template in plan.replaced:
                annotation = self._annotation_from_geometry(geometry, template)
                if annotation is None:
                    # Leave the sources alone rather than delete them for nothing.
                    continue
                removed.extend(sources)
                added.append(annotation)

        if not removed and not added:
            return None

        # One bulk delete and add for every image: per-annotation calls repaint and
        # rebuild the phantom layer each time.
        self.unselect_annotations()
        self.delete_annotations(removed, record_action=False)
        self.add_annotations(added, record_action=False)
        self.action_stack.push(MergeAnnotationsAction(self, removed, added))

        current = [a for a in added if a.image_path == self.current_image_path]
        if current:
            self.select_annotations_bulk(current)

        return removed, added
