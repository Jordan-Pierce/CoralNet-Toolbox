"""
FeatureSelectTool3D — Tier-2 feature similarity query tool for the 3D scene.

Works across product types — meshes (per-face), point clouds and Gaussian
splats (per-point/per-splat). The query is element-agnostic; only the recolor
sink differs (FeatureMeshManager.recolor_by_similarity dispatches to the
product). "face"/"mesh" wording below applies equally to points/splats.

Interaction (mirrors the 2D FeatureSelectTool):
  - Selecting the tool button does NOTHING visually. Press **Space** in the 3D
    viewer to ENGAGE the similarity overlay (the 3D analogue of creating a 2D
    work area): the heatmap appears over the current base array, and the
    shared annotation-window colormap dropdown + opacity slider drive it.
  - While engaged:
      * Hover: live preview of similarity to the element under the cursor
        (unioned with any committed prototypes).
      * Ctrl + left-click: commit a positive prototype.
      * Ctrl + right-click: commit a negative prototype.
      * Ctrl + wheel: adjust the selection threshold (live thresholded preview).
      * Space with prototypes: finalize — paint the scene (and propagate to
        cameras if multi-annotate); stays engaged for the next query.
      * Space with no prototypes: DISENGAGE (overlay off, controls released).
      * Backspace: clear prototypes; with none, disengage.
      * Ctrl + Alt: toggle MULTI-CLASS mode (same gesture as the 2D tool):
        Ctrl+click assigns the element to the CURRENTLY selected label (switch
        labels to add more classes; Ctrl+right-click removes the selected
        label's last prototype), Ctrl+wheel adjusts the reject floor, Space
        commits one paint per label (nearest-prototype argmax).

Engagement is mutually exclusive with the 2D tool's work area (either/or).
Both modes render through the same LUT-indexed display paths (mesh
SimilarityShader disp texture on the overlay actor / splat display channel),
so a preview tick is always "C matvecs on the GPU + an [N]-byte upload" —
multi-class costs the same as the binary gradient.
"""

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor

from coralnet_toolbox.MVAT.tools.Tool3D import Tool3D


# Hover updates are coalesced to at most one per this interval (~60 Hz) so a
# burst of mouse-move events can't queue up multiple full recolors.
_HOVER_INTERVAL_MS = 16


class FeatureSelectTool3D(Tool3D):
    """
    Click-to-query feature similarity tool.

    Maintains state: positive/negative prototypes, similarity cache, threshold.
    """

    tool_kind = "feature"
    # Opt in to receiving right-button presses (for Ctrl+right = negative).
    # The viewer only forwards right-clicks to tools that set this.
    wants_right_button = True

    def __init__(self, mvat_viewer, mvat_manager):
        super().__init__(mvat_viewer, mvat_manager)
        self._preview_color = 'cyan'
        self._threshold = 0.5
        # Once the user engages the threshold (Ctrl+wheel), keep showing the
        # thresholded preview as they add more points — adding a point should
        # refine the existing thresholded view, not reset it back to the full
        # gradient. Reset on clear/commit.
        self._threshold_active = False

        # ---- Multi-class mode (toggled with Ctrl+Alt while the tool is active) ----
        # In "multiclass" mode a Ctrl+click assigns the element to the CURRENTLY
        # selected label instead of a fixed positive/negative bucket; Space
        # classifies the whole scene (nearest-prototype argmax + reject floor)
        # and paints one label per class. Mirrors the 2D FeatureSelectTool.
        self.mode = "binary"            # "binary" | "multiclass"
        self.class_prototypes = {}      # label_id -> list[element_id]
        self.class_labels = {}          # label_id -> Label (resolved at commit)
        self.class_colors = {}          # label_id -> (r, g, b) (LUT palette rows)
        # Reject floor on raw cosine similarity: elements whose best-class score
        # is below this render the scrim and stay unlabeled at commit (separate
        # from the binary `_threshold`, which lives on the same raw scale here).
        self.multiclass_threshold = 0.5

        # Live-hover state. mouseMoveEvent just records the latest cursor world
        # position and marks dirty; the timer coalesces those into one recolor
        # per tick so a flood of move events can't back up the render thread.
        self._hover_timer = QTimer()
        self._hover_timer.setInterval(_HOVER_INTERVAL_MS)
        self._hover_timer.timeout.connect(self._process_hover)
        self._pending_hover_world = None
        self._hover_dirty = False

    def activate(self):
        """Activate the tool. Nothing changes visually until the user engages.

        Mirrors the 2D FeatureSelectTool: selecting the tool button alone never
        flips what the viewer displays. The similarity overlay appears only
        when the user presses Space (engage_overlay — the 3D analogue of
        creating a 2D work area).
        """
        super().activate()
        # No brush sphere for this tool — hover drives the similarity preview.
        self._hide_preview_sphere()
        self._pending_hover_world = None
        self._hover_dirty = False
        self._hover_timer.start()

    def deactivate(self):
        """Deactivate: disengage the overlay and clear the query."""
        self._hover_timer.stop()
        self._pending_hover_world = None
        self._hover_dirty = False
        super().deactivate()

        fmm = getattr(self.mvat_manager, 'feature_mesh_manager', None)

        # Drop the query and multi-class prototypes; the LUT returns to the
        # colormap (it may hold the label palette when leaving multiclass).
        self._clear_class_prototypes()
        if fmm is not None and fmm.query_engine is not None:
            fmm.query_engine.clear()
            self._threshold_active = False
            fmm.apply_colormap()

        # Tear the overlay down and release the shared colormap controls
        # (mirrors the 2D tool's deactivate → _release_colormap_controls).
        if fmm is not None:
            fmm.disengage_overlay()

    def _engaged(self) -> bool:
        """True while the similarity overlay is engaged (Space was pressed)."""
        fmm = getattr(self.mvat_manager, 'feature_mesh_manager', None)
        return bool(fmm is not None and getattr(fmm, 'overlay_engaged', False))

    def _has_prompts(self) -> bool:
        """Whether any prompt exists in the active mode (gates commit vs
        disengage on Space/Backspace — 2D `_has_prompts` parity)."""
        if self.mode == "multiclass":
            return any(self.class_prototypes.values())
        feature_manager = getattr(self.mvat_manager, 'feature_mesh_manager', None)
        qe = getattr(feature_manager, 'query_engine', None)
        if qe is None:
            return False
        return bool(qe.positive_ids or qe.negative_ids)

    def _clear_class_prototypes(self):
        """Drop all committed multi-class prototypes (bookkeeping only)."""
        self.class_prototypes = {}
        self.class_labels = {}
        self.class_colors = {}

    def _status(self, message, msecs=4000):
        status_bar = getattr(self.mvat_manager.main_window, 'status_bar', None)
        if status_bar is not None:
            status_bar.showMessage(message, msecs)

    def _toggle_multiclass_mode(self):
        """Ctrl+Alt: switch between binary pos/neg and multi-class labeling.

        Clears any in-progress prompts so the two interaction models never
        bleed into each other (same policy as the 2D tool); the feature buffer
        and threshold values are preserved.
        """
        feature_manager = getattr(self.mvat_manager, 'feature_mesh_manager', None)
        if feature_manager is None or feature_manager.query_engine is None:
            return
        self.mode = "multiclass" if self.mode == "binary" else "binary"
        feature_manager.query_engine.clear()
        self._clear_class_prototypes()
        self._threshold_active = False
        # Shared dropdown → 'None' (multiclass) / colormap (binary); no-op
        # while not engaged (2D _apply_mode_colormap parity).
        feature_manager.apply_mode_colormap(self.mode)
        if self.mode == "multiclass":
            if self._engaged():
                self._update_class_display()
            self._status("Feature Select 3D: MULTI-CLASS mode — Ctrl+click assigns "
                         "the selected label; switch labels to add more classes. "
                         "Space to commit, Ctrl+Alt to exit.", 6000)
        else:
            # Restore the colormap LUT (the class palette may be loaded).
            feature_manager.apply_colormap()
            if self._engaged():
                self._update_similarity_display()
            self._status("Feature Select 3D: BINARY mode — Ctrl+click positive, "
                         "Ctrl+right-click negative. Ctrl+Alt for multi-class.", 4000)

    def mousePressEvent(self, event, _face_id: int, world_pos):
        """
        Handle mouse clicks to add prototypes.

        Interaction:
            - Plain left / right press → ignored here so the viewer keeps its
              normal navigation (left = rotate, right = pan).
            - Ctrl + left  → add positive prototype.
            - Ctrl + right → add negative prototype.

        Args:
            event: The forwarded event. Presses are dispatched from VTK observers
                (QtMVATViewer._on_left_press / _on_right_press), so `event` is the
                VTK event *string* ("LeftButtonPressEvent"/"RightButtonPressEvent"),
                NOT a QMouseEvent — button/modifier state is resolved defensively.
            _face_id: VTK cell ID under the cursor (not used directly — the
                viewer's pick dispatch does not resolve a real cell id, so the
                element is instead resolved from world_pos below, matching the
                pattern used by DropperTool3D/FillTool3D).
            world_pos: World position [3]
        """
        if world_pos is None:
            return

        # No querying without engagement (2D parity: no work area → no clicks).
        if not self._engaged():
            return

        # Only Ctrl-modified clicks drive the query; plain clicks fall through
        # to the viewer's camera navigation.
        if not self._is_ctrl_pressed(event):
            return

        feature_manager = self.mvat_manager.feature_mesh_manager
        if feature_manager.query_engine is None:
            return

        primary_target = self.mvat_viewer.scene_context.get_primary_target()
        if primary_target is None:
            return

        tree = getattr(primary_target, '_hover_face_kdtree', None)
        if tree is None:
            return

        try:
            _, closest_idx = tree.query(world_pos, k=1)
        except Exception:
            return

        element_id = int(closest_idx)
        if element_id < 0:
            return

        if self.mode == "multiclass":
            self._handle_multiclass_click(event, element_id)
            return

        if self._is_right_button(event):
            # Ctrl + right-click: add negative prototype.
            feature_manager.query_engine.add_negative(element_id)
        else:
            # Ctrl + left-click: add positive prototype.
            feature_manager.query_engine.add_positive(element_id)

        # Preserve the thresholded preview across point additions once the
        # user has started thresholding (don't snap back to the full gradient).
        self._update_similarity_display(thresholded=self._threshold_active)

    def _handle_multiclass_click(self, event, element_id: int):
        """Ctrl+click in multi-class mode: (de)assign an element prototype to the
        currently selected label (left adds, right removes the label's last)."""
        label = self._get_selected_label()
        if label is None:
            self._status("Select a label before adding a class prototype.")
            return

        if self._is_right_button(event):
            ids = self.class_prototypes.get(label.id)
            if ids:
                ids.pop()
                if not ids:
                    # Drop empty bookkeeping so the class leaves the palette.
                    self.class_prototypes.pop(label.id, None)
                    self.class_labels.pop(label.id, None)
                    self.class_colors.pop(label.id, None)
        else:
            self.class_prototypes.setdefault(label.id, []).append(element_id)
            self.class_labels[label.id] = label
            color = QColor(label.color)
            self.class_colors[label.id] = (color.red(), color.green(), color.blue())

        self._update_class_display()

    def mouseMoveEvent(self, event, face_id: int, world_pos):
        """
        Record the cursor position for a coalesced live hover preview.

        Deliberately does NOT call super().mouseMoveEvent — this tool has no
        brush sphere; hover drives the similarity preview instead. The actual
        recolor happens in _process_hover on the timer, so a flood of move
        events only ever schedules work, never performs it inline.
        """
        if not self.active:
            return
        # Skip while a mouse button is held (camera rotate / pan drag).
        try:
            if event is not None and hasattr(event, 'buttons') and event.buttons() != Qt.NoButton:
                return
        except Exception:
            pass
        self._pending_hover_world = (
            np.asarray(world_pos, dtype=np.float64) if world_pos is not None else None
        )
        self._hover_dirty = True

    def _process_hover(self) -> None:
        """Timer slot: recolor the live preview for the latest hovered face."""
        if not self.active or not self._hover_dirty:
            return
        self._hover_dirty = False

        # No live preview until the overlay is engaged (Space).
        if not self._engaged():
            return

        feature_manager = self.mvat_manager.feature_mesh_manager
        if feature_manager.buffer is None or feature_manager.query_engine is None:
            return

        hover_id = None
        world = self._pending_hover_world
        if world is not None:
            primary_target = self.mvat_viewer.scene_context.get_primary_target()
            tree = getattr(primary_target, '_hover_face_kdtree', None) if primary_target else None
            if tree is not None:
                try:
                    _, closest_idx = tree.query(world, k=1)
                    hover_id = int(closest_idx)
                except Exception:
                    hover_id = None

        # Off-mesh (hover_id None) falls back to the committed query view.
        if self.mode == "multiclass":
            self._update_class_display(hover_id=hover_id)
        else:
            self._update_similarity_display(thresholded=self._threshold_active,
                                            hover_id=hover_id)

    def _is_right_button(self, event) -> bool:
        """Return True for a right-button press across the Qt and VTK paths."""
        # VTK dispatch passes the event name string.
        if isinstance(event, str):
            return "Right" in event
        try:
            event_button = getattr(event, 'button', None)
            if callable(event_button):
                return event_button() == Qt.RightButton
        except Exception:
            pass
        return False

    def _is_ctrl_pressed(self, event) -> bool:
        """Return True if Ctrl is held, handling both Qt and VTK dispatch paths.

        Left-button presses arrive from a VTK observer with no Qt modifiers, so
        the Control-key state is read from the VTK interactor in that case.
        """
        try:
            mods = getattr(event, 'modifiers', None)
            if callable(mods):
                return bool(mods() & Qt.ControlModifier)
        except Exception:
            pass
        try:
            interactor = self.mvat_viewer.plotter.interactor
            if interactor is not None and hasattr(interactor, 'GetControlKey'):
                return bool(interactor.GetControlKey())
        except Exception:
            pass
        return False

    def wheelEvent(self, event, delta_y: int):
        """
        Ctrl+wheel: adjust the selection threshold and show a live preview.

        The viewer's eventFilter only forwards wheel events to the active tool
        when Ctrl is held, so no modifier re-check is needed here. We deliberately
        do NOT call super().wheelEvent (which resizes the brush radius) — this
        tool has no brush.

        Args:
            event: QWheelEvent (real Qt event on this path).
            delta_y: Wheel delta in pixels.
        """
        if not self._engaged():
            return

        step = 0.02
        if self.mode == "multiclass":
            # Adjust the reject floor and refresh the class preview.
            if delta_y > 0:
                self.multiclass_threshold = min(1.0, self.multiclass_threshold + step)
            else:
                self.multiclass_threshold = max(0.0, self.multiclass_threshold - step)
            self._update_class_display()
            self._status(f"Feature select reject threshold: "
                         f"{self.multiclass_threshold:.2f}", 2000)
            return

        if delta_y > 0:
            self._threshold = min(1.0, self._threshold + step)
        else:
            self._threshold = max(0.0, self._threshold - step)

        # The threshold view is now engaged; subsequent point additions keep it.
        self._threshold_active = True

        # Live thresholded preview: highlight exactly what Space would commit.
        self._update_similarity_display(thresholded=True)

        self._status(f"Feature select threshold: {self._threshold:.2f}", 2000)

    def keyPressEvent(self, event):
        """
        Handle keyboard.

        Args:
            event: QKeyEvent
        """
        feature_manager = self.mvat_manager.feature_mesh_manager

        if event.key() == Qt.Key_Space:
            # Space state machine (2D work-area parity):
            #   not engaged            → engage the similarity overlay
            #   engaged + prototypes   → commit (stays engaged for the next query)
            #   engaged, no prototypes → disengage (overlay off, controls back)
            if not self._engaged():
                if feature_manager.engage_overlay(mode=self.mode):
                    self._status("Feature Select 3D: overlay engaged — Ctrl+click "
                                 "to query, Space to commit, Backspace to clear. "
                                 "Colormap + opacity: annotation-window controls.",
                                 5000)
            elif self._has_prompts():
                if self.mode == "multiclass":
                    self._commit_multiclass()
                else:
                    self._commit_selection_to_label()
            else:
                feature_manager.disengage_overlay()
            event.accept()
        elif event.key() == Qt.Key_Backspace:
            # Backspace: clear prototypes; with none, disengage (2D parity).
            if not self._engaged():
                event.accept()
                return
            if self._has_prompts():
                if self.mode == "multiclass":
                    self._clear_class_prototypes()
                    if feature_manager.query_engine is not None:
                        self._update_class_display()
                elif feature_manager.query_engine is not None:
                    feature_manager.query_engine.clear()
                    self._threshold_active = False
                    self._update_similarity_display()
            else:
                feature_manager.disengage_overlay()
            event.accept()
        else:
            super().keyPressEvent(event)

    def _update_similarity_display(self, thresholded: bool = False,
                                   hover_id: int = None) -> None:
        """Refresh the engaged similarity overlay (binary gradient/threshold view).

        Args:
            thresholded: When True, show a live preview of the thresholded
                selection (only faces that Space would commit are lit). When
                False, show the raw similarity gradient.
            hover_id: optional transient prototype (face under the cursor) folded
                into the query for live hover preview, without committing it.
        """
        feature_manager = self.mvat_manager.feature_mesh_manager
        if feature_manager.buffer is None or feature_manager.query_engine is None:
            return

        # Write the (optionally thresholded) display values; the overlay actor's
        # shader (or the splat display channel) picks them up on the render.
        threshold = self._threshold if thresholded else None
        feature_manager.recolor_by_similarity(threshold=threshold, hover_id=hover_id)
        self.mvat_viewer.plotter.render()

    def _commit_selection_to_label(self) -> None:
        """
        Finalize the highlighted (thresholded) selection.

        1. Paint the selected faces on the mesh with the active label.
        2. If multi-annotate is enabled, propagate those face IDs to the context
           cameras through the same pipeline the 3D brush uses
           (MVATManager._on_3d_brush_stroke_applied → PropagationEngine).
        """
        feature_manager = self.mvat_manager.feature_mesh_manager
        if feature_manager.buffer is None or feature_manager.query_engine is None:
            return

        # Get the active label and resolve it to the mesh's small class index.
        # NOTE: label.id is a project-level identifier (often a UUID/large int),
        # NOT the compact class_id the mesh's class_ids/Labels arrays use — so we
        # must resolve it through the same mask-annotation map the brush uses.
        label = self._get_selected_label()
        if label is None:
            return

        class_id, color_rgb = self._resolve_label(label)
        if class_id is None or color_rgb is None:
            return

        # Select elements by threshold
        selected_ids = feature_manager.query_engine.select(threshold=self._threshold)

        if selected_ids.size == 0:
            return

        # 1. Paint the mesh with the selection. The overlay stays engaged — the
        # paint is flushed to the GPU so it shows through the label overlay /
        # Labels views, while the query view is preserved here.
        primary_target = self.mvat_viewer.scene_context.get_primary_target()
        if primary_target and hasattr(primary_target, 'apply_labels'):
            primary_target.apply_labels(selected_ids, class_id, color_rgb)
            primary_target.flush_labels_to_gpu()

            # Emit repaint signal to update the viewer
            if hasattr(self.mvat_manager, '_universal_repaint_signal'):
                self.mvat_manager._universal_repaint_signal.emit([])

        # 2. Multi-annotate sync — propagate the painted faces to the context
        # cameras, mirroring BrushTool3D._finish_stroke().
        if getattr(self.mvat_manager, 'multi_annotate_enabled', False):
            handler = getattr(self.mvat_manager, '_on_3d_brush_stroke_applied', None)
            if callable(handler):
                try:
                    face_ids = np.asarray(selected_ids, dtype=np.int32)
                    handler(face_ids, label)
                except Exception as e:
                    print(f"[FeatureSelectTool3D] propagation failed: {e}")

        # Clear the query for the next selection (back to gradient view).
        feature_manager.query_engine.clear()
        self._threshold_active = False
        self._update_similarity_display()

    # ==================== Multi-class mode ====================

    def _update_class_display(self, hover_id: int = None) -> None:
        """Recolor the scene by nearest-prototype class (multi-class preview).

        The element under the cursor is folded into the CURRENTLY selected
        label's prototype set so hovering previews "what would this click do",
        mirroring the binary hover union. Renders through the same LUT paths as
        the gradient view, so per-tick cost is unchanged.
        """
        feature_manager = self.mvat_manager.feature_mesh_manager
        if feature_manager.buffer is None or feature_manager.query_engine is None:
            return

        label = self._get_selected_label()
        hover_key = None
        colors = dict(self.class_colors)
        if hover_id is not None and label is not None:
            hover_key = label.id
            if hover_key not in colors:
                color = QColor(label.color)
                colors[hover_key] = (color.red(), color.green(), color.blue())

        proto = {k: v for k, v in self.class_prototypes.items() if v}
        feature_manager.recolor_by_class(
            proto, colors, reject_threshold=self.multiclass_threshold,
            hover_key=hover_key, hover_id=hover_id)
        self.mvat_viewer.plotter.render()

    def _commit_multiclass(self) -> None:
        """Finalize the multi-class preview: paint one label per class.

        Classifies every element (nearest-prototype argmax + reject floor —
        exactly the preview field, no hover) and paints each class's elements
        with its label, then propagates per label when multi-annotate is on.
        """
        feature_manager = self.mvat_manager.feature_mesh_manager
        if feature_manager.buffer is None or feature_manager.query_engine is None:
            return

        proto = {k: v for k, v in self.class_prototypes.items() if v}
        if not proto:
            self._status("Feature Select: add at least one class prototype to commit.")
            return

        primary_target = self.mvat_viewer.scene_context.get_primary_target()
        if primary_target is None:
            return

        disp, keys = feature_manager.query_engine.class_display_scalars(
            proto, reject_threshold=self.multiclass_threshold)
        if not keys or not (disp > 0).any():
            self._status("Feature Select: nothing above the reject threshold to commit.")
            return

        committed = []  # (face_ids, label) per class, for propagation
        for k, key in enumerate(keys):
            label = self.class_labels.get(key)
            if label is None:
                continue
            class_id, color_rgb = self._resolve_label(label)
            if class_id is None or color_rgb is None:
                continue
            ids = np.flatnonzero(disp == (k + 1)).astype(np.int32)
            if ids.size == 0:
                continue
            if hasattr(primary_target, 'apply_labels'):
                primary_target.apply_labels(ids, class_id, color_rgb)
                committed.append((ids, label))

        if not committed:
            return

        if hasattr(primary_target, 'flush_labels_to_gpu'):
            primary_target.flush_labels_to_gpu()
        if hasattr(self.mvat_manager, '_universal_repaint_signal'):
            self.mvat_manager._universal_repaint_signal.emit([])

        # Multi-annotate sync — one propagation per label, mirroring the
        # binary commit path.
        if getattr(self.mvat_manager, 'multi_annotate_enabled', False):
            handler = getattr(self.mvat_manager, '_on_3d_brush_stroke_applied', None)
            if callable(handler):
                for ids, label in committed:
                    try:
                        handler(ids, label)
                    except Exception as e:
                        print(f"[FeatureSelectTool3D] propagation failed: {e}")

        # Clear the prototypes for the next selection (keep multiclass mode).
        self._clear_class_prototypes()
        self._update_class_display()
