"""
FeatureSelectTool3D — Tier-2 feature similarity query tool for the 3D scene.

Works across product types — meshes (per-face), point clouds and Gaussian
splats (per-point/per-splat). The query is element-agnostic; only the recolor
sink differs (FeatureMeshManager.recolor_by_similarity dispatches to the
product). "face"/"mesh" wording below applies equally to points/splats.

Interaction:
  - Hover over the scene: live preview of similarity to the element under the
    cursor (unioned with any committed prototypes), updated on the fly.
  - Ctrl + left-click on an element: commit a positive prototype
  - Ctrl + right-click: commit a negative prototype
  - Ctrl + wheel: adjust the selection threshold (live thresholded preview)
  - Space: finalize — paint the scene (and propagate to cameras if multi-annotate)
  - Backspace: clear query
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
        self._previous_array = None
        # Once the user engages the threshold (Ctrl+wheel), keep showing the
        # thresholded preview as they add more points — adding a point should
        # refine the existing thresholded view, not reset it back to the full
        # gradient. Reset on clear/commit.
        self._threshold_active = False

        # Live-hover state. mouseMoveEvent just records the latest cursor world
        # position and marks dirty; the timer coalesces those into one recolor
        # per tick so a flood of move events can't back up the render thread.
        self._hover_timer = QTimer()
        self._hover_timer.setInterval(_HOVER_INTERVAL_MS)
        self._hover_timer.timeout.connect(self._process_hover)
        self._pending_hover_world = None
        self._hover_dirty = False

    def activate(self):
        """Activate the tool and switch to Similarity array if available."""
        super().activate()
        # No brush sphere for this tool — hover drives the similarity preview.
        self._hide_preview_sphere()
        # Save the previously selected array so we can restore it on deactivate
        primary_target = self.mvat_viewer.scene_context.get_primary_target()
        if primary_target:
            self._previous_array = getattr(primary_target, 'selected_array', 'Labels')
        self._pending_hover_world = None
        self._hover_dirty = False
        self._hover_timer.start()

    def deactivate(self):
        """Deactivate and restore the previous array."""
        self._hover_timer.stop()
        self._pending_hover_world = None
        self._hover_dirty = False
        super().deactivate()
        primary_target = self.mvat_viewer.scene_context.get_primary_target()

        # Phase-2: tear the similarity GPU shader off the mesh actor so the mesh
        # renders normally under the restored array / other tools.
        fmm = getattr(self.mvat_manager, 'feature_mesh_manager', None)
        if fmm is not None and primary_target is not None:
            actor = getattr(self.mvat_viewer, '_product_actors', {}).get(
                getattr(primary_target, 'product_id', None)
            )
            if actor is not None:
                fmm.uninstall_shader(actor)

        # Clear any active query so the similarity array resets to baseline.
        if fmm is not None and fmm.query_engine is not None:
            fmm.query_engine.clear()
            self._threshold_active = False
            fmm.recolor_by_similarity()

        if self._previous_array:
            if primary_target and hasattr(primary_target, 'set_selected_array'):
                primary_target.set_selected_array(self._previous_array)

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

        if self._is_right_button(event):
            # Ctrl + right-click: add negative prototype.
            feature_manager.query_engine.add_negative(element_id)
        else:
            # Ctrl + left-click: add positive prototype.
            feature_manager.query_engine.add_positive(element_id)

        # Preserve the thresholded preview across point additions once the
        # user has started thresholding (don't snap back to the full gradient).
        self._update_similarity_display(thresholded=self._threshold_active)

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
        step = 0.02
        if delta_y > 0:
            self._threshold = min(1.0, self._threshold + step)
        else:
            self._threshold = max(0.0, self._threshold - step)

        # The threshold view is now engaged; subsequent point additions keep it.
        self._threshold_active = True

        # Live thresholded preview: highlight exactly what Space would commit.
        self._update_similarity_display(thresholded=True)

        status_bar = getattr(self.mvat_manager.main_window, 'status_bar', None)
        if status_bar is not None:
            status_bar.showMessage(f"Feature select threshold: {self._threshold:.2f}", 2000)

    def keyPressEvent(self, event):
        """
        Handle keyboard.

        Args:
            event: QKeyEvent
        """
        if event.key() == Qt.Key_Space:
            # Space: finalize the highlighted selection — paint the mesh (and,
            # when multi-annotate is on, propagate to the context cameras).
            self._commit_selection_to_label()
            event.accept()
        elif event.key() == Qt.Key_Backspace:
            # Clear query and reset back to the gradient view.
            feature_manager = self.mvat_manager.feature_mesh_manager
            if feature_manager.query_engine is not None:
                feature_manager.query_engine.clear()
                self._threshold_active = False
                self._update_similarity_display()
            event.accept()
        else:
            super().keyPressEvent(event)

    def _update_similarity_display(self, thresholded: bool = False,
                                   hover_id: int = None) -> None:
        """Recolor the mesh by similarity and ensure the Similarity array is shown.

        Args:
            thresholded: When True, show a live preview of the thresholded
                selection (only faces that Enter would commit are lit). When
                False, show the raw similarity gradient.
            hover_id: optional transient prototype (face under the cursor) folded
                into the query for live hover preview, without committing it.
        """
        feature_manager = self.mvat_manager.feature_mesh_manager
        if feature_manager.buffer is None or feature_manager.query_engine is None:
            return

        # Write the (optionally thresholded) similarity scalars into the mesh.
        threshold = self._threshold if thresholded else None
        feature_manager.recolor_by_similarity(threshold=threshold, hover_id=hover_id)

        # The mesh actor's mapper is only bound to the Similarity scalar (with
        # the similarity colormap) when the viewer rebuilds the actor via
        # get_render_style() — set_selected_array() alone just stores an
        # attribute. So the first time we show similarity we must drive the
        # viewer's array-selection path (render_scene); afterwards the in-place
        # scalar write + a cheap render() is enough.
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

        # 1. Paint the mesh with the selection. We deliberately stay on the
        # current array (Similarity) rather than switching to Labels — the
        # paint is flushed to the GPU so it shows the moment the user switches
        # the dropdown to Labels, but the query view is preserved here.
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
