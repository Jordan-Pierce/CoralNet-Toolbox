import warnings

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QDialog, QMessageBox, QSpinBox, QLabel, QVBoxLayout,
    QFormLayout, QDialogButtonBox, QGroupBox,
)

from coralnet_toolbox.QtActions import (
    AddAnnotationsAction,
    DeleteAnnotationsAction,
    CompoundAction,
    MaskEditAction,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MorphologicalMixin:
    """Mixin class providing bake/unbake annotation operations for AnnotationWindow."""

    def prompt_bake_or_unbake_annotations(self):
        """Offer a choice to bake vectors into the mask or unbake the mask into vectors."""
        if not self.current_image_path:
            return False

        vector_annotations = []
        for annotation in self.get_image_annotations():
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
                vector_annotations.append(annotation)

        raster = None
        try:
            raster_manager = getattr(self.main_window.image_window, 'raster_manager', None)
            if raster_manager is not None:
                raster = raster_manager.get_raster(self.current_image_path)
        except Exception:
            raster = None

        mask_annotation = getattr(raster, 'mask_annotation', None) if raster is not None else None
        has_mask_regions = False
        if mask_annotation is not None:
            try:
                has_mask_regions = bool(np.any(mask_annotation.mask_data % mask_annotation.LOCK_BIT))
            except Exception:
                has_mask_regions = True

        if not vector_annotations and not has_mask_regions:
            try:
                self.main_window.status_bar.showMessage(
                    "No vector or mask annotations are available on the current image.",
                    3000,
                )
            except Exception:
                pass
            return False

        dialog = QDialog(self)
        dialog.setWindowTitle("Convert Current Image Annotations")
        dialog.setModal(True)

        root_layout = QVBoxLayout(dialog)
        root_layout.setSpacing(12)

        info_group = QGroupBox("Information")
        info_layout = QVBoxLayout(info_group)
        info_layout.setContentsMargins(8, 8, 8, 8)
        desc_label = QLabel(
            "<b>Choose how to convert annotations for the current image.</b><br>"
            "Bake rasterizes vector annotations into the mask.<br>"
            "Unbake vectorizes the current mask regions into vector annotations."
        )
        desc_label.setWordWrap(True)
        info_layout.addWidget(desc_label)
        root_layout.addWidget(info_group)

        unbake_group = QGroupBox("Unbake Options")
        unbake_group.setEnabled(has_mask_regions)
        form_layout = QFormLayout(unbake_group)
        form_layout.setContentsMargins(8, 8, 8, 8)

        min_hole_spinbox = QSpinBox()
        min_hole_spinbox.setRange(0, 1_000_000)
        min_hole_spinbox.setValue(500)
        min_hole_spinbox.setSingleStep(100)
        min_hole_spinbox.setSuffix(" px²")
        min_hole_spinbox.setToolTip(
            "When vectorizing (unbaking) a mask, interior voids — holes — inside\n"
            "each region are traced as interior rings in the resulting polygon.\n\n"
            "Holes smaller than this area are silently filled, preventing the\n"
            "vertex explosion that comes from tracing every small gap or\n"
            "noise-level void in the mask.\n\n"
            "Holes at or above this threshold are preserved as true polygon\n"
            "holes, keeping significant voids (e.g. a sand patch inside a coral\n"
            "colony) accurately represented.\n\n"
            "0 = preserve all holes (maximum detail, most vertices).\n"
            "Higher values = fewer, larger holes kept (smoother polygons)."
        )
        min_hole_label = QLabel("Min hole area to preserve:")
        min_hole_label.setToolTip(min_hole_spinbox.toolTip())
        form_layout.addRow(min_hole_label, min_hole_spinbox)

        root_layout.addWidget(unbake_group)

        button_box = QDialogButtonBox()
        bake_button = button_box.addButton("Bake", QDialogButtonBox.AcceptRole)
        unbake_button = button_box.addButton("Unbake", QDialogButtonBox.AcceptRole)
        cancel_button = button_box.addButton(QDialogButtonBox.Cancel)

        bake_button.setEnabled(bool(vector_annotations))
        unbake_button.setEnabled(has_mask_regions)

        chosen = [None]

        def _on_bake():
            chosen[0] = "bake"
            dialog.accept()

        def _on_unbake():
            chosen[0] = "unbake"
            dialog.accept()

        bake_button.clicked.connect(_on_bake)
        unbake_button.clicked.connect(_on_unbake)
        cancel_button.clicked.connect(dialog.reject)

        root_layout.addWidget(button_box)
        dialog.setMinimumWidth(380)

        if dialog.exec_() != QDialog.Accepted or chosen[0] is None:
            return False
        if chosen[0] == "bake":
            return self.bake_vector_annotations(prompt_user=False)
        if chosen[0] == "unbake":
            return self.vectorize_mask_annotations(min_hole_area=min_hole_spinbox.value())
        return False

    def bake_vector_annotations(self, prompt_user=True):
        """Bake current-image vector annotations into the mask and delete them.

        This is the destructive counterpart to rasterize_annotations(): it
        permanently writes vector labels into the semantic mask and then removes
        the vector annotations from the current image.
        """
        if not self.current_image_path:
            return False

        annotations = []
        for annotation in self.get_image_annotations():
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

        if not annotations:
            try:
                self.main_window.status_bar.showMessage(
                    "No vector annotations on the current image can be baked into the mask.",
                    3000,
                )
            except Exception:
                pass
            return False

        if prompt_user:
            reply = QMessageBox.question(
                self,
                "Bake Vector Annotations",
                "Bake all vector annotations in the current image into the mask and remove the vectors?\n\n"
                "Undo will restore both the mask pixels and the vector annotations.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply != QMessageBox.Yes:
                return False

        mask_annotation = self.current_mask_annotation
        if mask_annotation is None:
            return False

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
                    return False

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

            try:
                if skipped_annotations:
                    self.main_window.status_bar.showMessage(
                        f"Baked {len(baked_annotations)} vector annotations; skipped {len(skipped_annotations)} that could not be rasterized.",
                        4000,
                    )
                else:
                    self.main_window.status_bar.showMessage(
                        f"Baked {len(baked_annotations)} vector annotations into the mask.",
                        3000,
                    )
            except Exception:
                pass
        finally:
            QApplication.restoreOverrideCursor()

        return True

    def vectorize_mask_annotations(self, min_hole_area: int = 500):
        """Convert the current image's mask regions into vector annotations.

        Args:
            min_hole_area: Minimum hole area in pixels to preserve as an
                interior ring. Holes smaller than this threshold are filled.
        """
        mask_annotation = self.current_mask_annotation
        if mask_annotation is None:
            try:
                self.main_window.status_bar.showMessage(
                    "No mask annotation is available for the current image.",
                    3000,
                )
            except Exception:
                pass
            return False

        rejected_indices = []
        try:
            vector_annotations = mask_annotation.to_vector_annotations(
                transparency=self.main_window.get_transparency_value(),
                show_confidence=False,
                min_hole_area=min_hole_area,
                rejected_indices_out=rejected_indices,
                image_path=self.current_image_path,
            )
        except Exception:
            vector_annotations = []
            rejected_indices = []

        if not vector_annotations and not rejected_indices:
            try:
                self.main_window.status_bar.showMessage(
                    "No mask regions could be vectorized from the current image.",
                    3000,
                )
            except Exception:
                pass
            return False

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
                    if '::frame_' in str(self.current_image_path):
                        self._sync_video_mask_to_cache()
                except Exception:
                    pass

            if clear_action is None or clear_action.is_empty():
                try:
                    if vector_annotations:
                        self.delete_annotations(vector_annotations, record_action=False)
                    self.main_window.status_bar.showMessage(
                        "No editable mask pixels were changed during vectorization.",
                        3000,
                    )
                except Exception:
                    pass
                return False

            actions = [action for action in (add_action, clear_action) if action is not None]
            if len(actions) > 1:
                self.action_stack.push(CompoundAction(
                    actions,
                    description="Vectorize mask annotations",
                ))
            else:
                self.action_stack.push(actions[0])

            try:
                message = f"Vectorized {len(vector_annotations)} mask regions into annotations."
                if rejected_indices:
                    message += f" Discarded {len(rejected_indices)} sub-threshold regions."
                self.main_window.status_bar.showMessage(message, 3000)
            except Exception:
                pass
        finally:
            QApplication.restoreOverrideCursor()

        return True
