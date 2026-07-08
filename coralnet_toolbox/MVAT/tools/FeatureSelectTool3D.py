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
  - While engaged, Ctrl+click / Ctrl+right-click also work on the CURRENT
    image in the 2D annotation window (see add_seed_from_2d_click), regardless
    of which 2D tool is active there — the pixel is unprojected to an element
    id via that image's camera index map and fed through the same prototype
    handling as a direct 3D click.

Engagement is mutually exclusive with the 2D tool's work area (either/or).
Both modes render through the same LUT-indexed display paths (mesh
SimilarityShader disp texture on the overlay actor / splat display channel),
so a preview tick is always "C matvecs on the GPU + an [N]-byte upload" —
multi-class costs the same as the binary gradient.
"""

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication

from coralnet_toolbox.MVAT.tools.Tool3D import Tool3D
from coralnet_toolbox.MVAT.core.PointMarkers3D import ColoredPointMarkers3D


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
        # A hover can come from either window: a 3D mouseMoveEvent resolves an
        # element id lazily (KD-tree over world_pos, done in _process_hover);
        # a 2D mouseMoveEvent (hover_from_2d_pixel) already has the element id
        # (via the camera's index map), so it's stored directly. Whichever
        # fired most recently wins — each setter clears the other's pending
        # value.
        self._hover_timer = QTimer()
        self._hover_timer.setInterval(_HOVER_INTERVAL_MS)
        self._hover_timer.timeout.connect(self._process_hover)
        self._pending_hover_world = None
        self._pending_hover_element_id = None
        self._hover_dirty = False

        # In-scene colored prototype markers (Feature 2): one label-colored sphere
        # per committed prototype (multi-class) or green/red per positive/negative
        # (binary). Its actor is added on activate() and refreshed by the display
        # updates. Mirrors the 2D FeatureSelectTool's per-prototype dots.
        self._proto_markers = ColoredPointMarkers3D(
            point_size=18.0, name='_feature_proto_markers')

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
        self._pending_hover_element_id = None
        self._hover_dirty = False
        self._hover_timer.start()
        # Register the prototype-marker actor (created lazily on first set_markers).
        self._proto_markers.add_to_plotter(self.mvat_viewer.plotter)

    def deactivate(self):
        """Deactivate: disengage the overlay and clear the query."""
        self._hover_timer.stop()
        self._pending_hover_world = None
        self._hover_dirty = False
        # Remove the in-scene prototype markers.
        try:
            self._proto_markers.remove_from_plotter(self.mvat_viewer.plotter)
        except Exception:
            pass
        super().deactivate()

        fmm = getattr(self.mvat_manager, 'feature_mesh_manager', None)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
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
        finally:
            QApplication.restoreOverrideCursor()

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

        self._apply_click_to_element(element_id, self._is_right_button(event))

    def _apply_click_to_element(self, element_id: int, is_right_button: bool) -> None:
        """Shared tail of a committed Ctrl+click: dispatch to multiclass or
        binary prototype handling and refresh the display.

        Used by mousePressEvent (element_id resolved via a 3D KD-tree query on
        world_pos) and add_seed_from_2d_click (element_id resolved via the
        camera's index map from a 2D pixel) — the only two callers.
        """
        if self.mode == "multiclass":
            self._handle_multiclass_click(element_id, is_right_button)
            return

        feature_manager = self.mvat_manager.feature_mesh_manager
        if is_right_button:
            # Ctrl + right-click: add negative prototype.
            feature_manager.query_engine.add_negative(element_id)
        else:
            # Ctrl + left-click: add positive prototype.
            feature_manager.query_engine.add_positive(element_id)

        # Preserve the thresholded preview across point additions once the
        # user has started thresholding (don't snap back to the full gradient).
        self._update_similarity_display(thresholded=self._threshold_active)

    def add_seed_from_2d_click(self, image_path: str, px: int, py: int,
                               is_right_button: bool) -> bool:
        """Feed a Ctrl+click on the 2D annotation window's current image into
        the same query engine a 3D Ctrl+click drives, by unprojecting the 2D
        pixel to a 3D element id via the camera's index map.

        Returns True when the click was claimed (overlay engaged, camera +
        element id resolved) — the caller should swallow the event. Returns
        False when the overlay isn't engaged, the current image isn't a
        loaded MVAT camera, or the pixel has no mesh coverage there — in all
        of those cases the caller should let the click fall through to
        normal 2D behavior.
        """
        if not self._engaged():
            return False
        feature_manager = getattr(self.mvat_manager, 'feature_mesh_manager', None)
        if feature_manager is None or feature_manager.query_engine is None:
            return False
        propagation_engine = getattr(self.mvat_manager, 'propagation_engine', None)
        if propagation_engine is None:
            return False
        camera = propagation_engine._get_camera_for_path(image_path)
        if camera is None:
            return False
        element_id = propagation_engine.pixel_to_element_id(camera, int(px), int(py))
        if element_id is None:
            return False
        self._apply_click_to_element(element_id, is_right_button)
        return True

    def _handle_multiclass_click(self, element_id: int, is_right_button: bool):
        """(De)assign an element prototype to the currently selected label
        (left adds, right removes the label's last) — mode-agnostic to the
        gesture's origin (3D click or 2D unprojected click)."""
        label = self._get_selected_label()
        if label is None:
            self._status("Select a label before adding a class prototype.")
            return

        if is_right_button:
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
        self._pending_hover_element_id = None
        self._hover_dirty = True

    def hover_from_2d_pixel(self, image_path: str, px: int, py: int) -> None:
        """Live-preview hover sourced from the 2D annotation window.

        Mirrors mouseMoveEvent's role, but the element id is resolved via the
        camera's index map (see PropagationEngine.pixel_to_element_id) instead
        of a 3D KD-tree query on world_pos — same coalesced-timer path either
        way. No-op while the tool isn't active or the overlay isn't engaged.
        """
        if not self.active or not self._engaged():
            return
        propagation_engine = getattr(self.mvat_manager, 'propagation_engine', None)
        camera = propagation_engine._get_camera_for_path(image_path) if propagation_engine else None
        element_id = (propagation_engine.pixel_to_element_id(camera, int(px), int(py))
                     if camera is not None else None)
        self._pending_hover_element_id = element_id
        self._pending_hover_world = None
        self._hover_dirty = True

    def clear_2d_hover(self) -> None:
        """Clear a 2D-sourced hover (e.g. the cursor left the annotation window)."""
        if self._pending_hover_element_id is None:
            return
        self._pending_hover_element_id = None
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

        hover_id = self._pending_hover_element_id
        world = self._pending_hover_world
        if hover_id is None and world is not None:
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
                if (feature_manager is None or feature_manager.buffer is None
                        or feature_manager.query_engine is None):
                    # No baked features yet — say so instead of doing nothing.
                    self._status("Feature Select 3D: no baked features for this "
                                 "model — run 'Bake Mesh Features' first.", 6000)
                else:
                    QApplication.setOverrideCursor(Qt.WaitCursor)
                    try:
                        engaged = feature_manager.engage_overlay(mode=self.mode)
                    finally:
                        QApplication.restoreOverrideCursor()
                    if engaged:
                        self._status("Feature Select 3D: overlay engaged — Ctrl+click "
                                     "to query, Space to commit, Backspace to clear. "
                                     "Colormap + opacity: annotation-window controls.",
                                     5000)
            elif self._has_prompts():
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    if self.mode == "multiclass":
                        self._commit_multiclass()
                    else:
                        self._commit_selection_to_label()
                finally:
                    QApplication.restoreOverrideCursor()
            else:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    feature_manager.disengage_overlay()
                finally:
                    QApplication.restoreOverrideCursor()
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
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    feature_manager.disengage_overlay()
                finally:
                    QApplication.restoreOverrideCursor()
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
        self._refresh_prototype_markers()
        self.mvat_viewer.plotter.render()

    def _paint_elements(self, primary_target, element_ids, class_id,
                        color_rgb, label) -> None:
        """Commit painted elements through the SAME pipeline the 3D brush uses.

        Meshes / point clouds go through MVATManager.submit_3d_face_paint /
        submit_3d_point_paint — that path updates the paint-shader class-id
        texture (what the label-overlay actor actually renders), so committed
        labels are visible immediately. A bare apply_labels + flush only
        updates the product caches / VTK array, which the overlay shader never
        reads. Splats keep the direct product path (their GPU label channel is
        the render path).
        """
        element_ids = np.asarray(element_ids, dtype=np.int32).ravel()
        element_type = getattr(primary_target, 'get_element_type', lambda: None)()
        label_id = getattr(label, 'id', None)
        if element_type == 'face':
            self.mvat_manager.submit_3d_face_paint(
                element_ids, color_rgb, int(class_id),
                primary_target=primary_target, label_id=label_id)
        elif element_type == 'point':
            self.mvat_manager.submit_3d_point_paint(
                element_ids, color_rgb, int(class_id),
                primary_target=primary_target, label_id=label_id)
        else:
            # Splats: the product's label channel IS the render path.
            if hasattr(primary_target, 'apply_labels'):
                primary_target.apply_labels(element_ids, int(class_id), color_rgb)
            if hasattr(primary_target, 'flush_labels_to_gpu'):
                primary_target.flush_labels_to_gpu()

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

        # 1. Paint the selection through the brush pipeline so it shows up in
        # the label overlay immediately (labels draw on top of the heatmap;
        # the overlay stays engaged for the next query).
        primary_target = self.mvat_viewer.scene_context.get_primary_target()
        if primary_target is not None:
            self._paint_elements(primary_target, selected_ids, class_id,
                                 color_rgb, label)

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
        self._refresh_prototype_markers()
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
            # Brush-pipeline paint so each class shows in the label overlay
            # immediately (see _paint_elements).
            self._paint_elements(primary_target, ids, class_id, color_rgb, label)
            committed.append((ids, label))

        if not committed:
            return

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

    # ==================== Prototype markers (Feature 2) ====================

    def _refresh_prototype_markers(self) -> None:
        """Sync the in-scene colored prototype markers with the current prompts.

        Multi-class: one label-colored sphere per prototype element (class_colors).
        Binary: green positive / red negative spheres (query_engine pos/neg ids).
        Hidden when the overlay is not engaged or there are no prompts. Called from
        the display-update methods, so the markers share the existing render (no
        extra render) and stay in sync on add / remove / clear / commit / backspace.
        """
        marker = getattr(self, '_proto_markers', None)
        if marker is None:
            return

        # No markers unless engaged (mirrors "no work area → no markers").
        if not self._engaged():
            marker.set_markers(None, None)
            return

        primary_target = self.mvat_viewer.scene_context.get_primary_target()
        if primary_target is None:
            marker.set_markers(None, None)
            return
        get_coord = getattr(primary_target, 'get_element_coordinate', None)

        # Collect (element_id, rgb) for the active mode.
        entries = []
        if self.mode == "multiclass":
            for label_id, ids in self.class_prototypes.items():
                if not ids:
                    continue
                rgb = self.class_colors.get(label_id, (255, 255, 255))
                entries.extend((eid, rgb) for eid in ids)
        else:
            fm = getattr(self.mvat_manager, 'feature_mesh_manager', None)
            qe = getattr(fm, 'query_engine', None)
            if qe is not None:
                entries.extend((eid, (0, 255, 0)) for eid in (qe.positive_ids or []))
                entries.extend((eid, (255, 0, 0)) for eid in (qe.negative_ids or []))

        # element_id → on-surface world coordinate. get_element_coordinate returns
        # the element's centroid on the rendered geometry (verified on-surface via
        # diagnostics), unlike the click pick which floats above the surface while
        # the similarity overlay biases the depth buffer toward the camera.
        if get_coord is None:
            marker.set_markers(None, None)
            return
        points, colors = [], []
        for eid, rgb in entries:
            try:
                pt = get_coord(int(eid))
            except Exception:
                pt = None
            if pt is None:
                continue
            pt = np.asarray(pt, dtype=np.float64).reshape(-1)
            if pt.size < 3 or not np.all(np.isfinite(pt[:3])):
                continue
            points.append(pt[:3])
            colors.append(rgb)

        if points:
            marker.set_markers(np.asarray(points, dtype=np.float32),
                               np.asarray(colors, dtype=np.uint8))
        else:
            marker.set_markers(None, None)
