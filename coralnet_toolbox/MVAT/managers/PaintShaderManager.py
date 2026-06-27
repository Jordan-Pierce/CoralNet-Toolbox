"""
PaintShaderManager — owns the GPU label-paint shader state for MVAT products.

Mirrors FeatureMeshManager's Similarity-shader plumbing, but for label paint: it
keeps a per-product PaintShaderState (the packed class-id texture + the label-color
LUT) and re-uploads the LUT on every install so label recolors/additions are picked
up.

"Labels" is no longer a selectable base array — labels are rendered by a translucent
overlay actor that shares the product geometry and draws the paint shader in discard
mode (unpainted fragments discarded) over whatever base array is shown (RGB / Texture
/ ...), blended at the transparency slider's opacity. See ``install_label_overlay_shader``
and QtMVATViewer._sync_label_overlay_actor. Paint therefore never bleeds onto the base
array (the old floating-overlay actors did, which the user did not want).

The LUT reuses the mask annotation's cached color map
(``QtMaskAnnotation._get_color_map()`` — a (>=256, 4) uint8 RGBA array indexed by
class_id, invalidated whenever labels change), so there's no second source of truth
for label colors.

Any ShaderUnavailable disables the shader for the session; the caller falls back to
the LabelWorker overlay path.
"""

from __future__ import annotations

import numpy as np

from coralnet_toolbox.MVAT.shaders.PaintShader import (
    ShaderUnavailable,
    build_state,
    install_paint_shader,
    uninstall_paint_shader,
)


class PaintShaderManager:
    """Per-product label-paint shader state + install/uninstall + LUT upload."""

    def __init__(self, mvat_manager):
        self.mvat_manager = mvat_manager
        self.viewer = mvat_manager.viewer
        self.annotation_window = mvat_manager.annotation_window

        self.shader_enabled = True
        # product_id -> PaintShaderState
        self._states: dict = {}

        # Label opacity (0..1) for showing labels over non-Labels arrays (RGB /
        # Texture / ...). 0 == off: no overlay actor is added, so those arrays render
        # natively. >0 adds a translucent label-overlay actor at this opacity.
        self.paint_opacity: float = 0.0

    # ------------------------------------------------------------------
    # Label palette (LUT) — reuse the mask annotation's cached color map
    # ------------------------------------------------------------------
    def _label_palette(self):
        """Return the (M, 4) uint8 RGBA palette indexed by class_id, or None.

        Built in the canonical mesh class-id space (PropagationEngine) so it stays
        consistent with the cid texture: that is the SAME id->label mapping the
        paint paths use, so a full LUT re-upload reproduces every painted color and
        no per-paint pin is needed. Falls back to the mask annotation's color map
        only when the engine is unavailable.
        """
        engine = getattr(self.mvat_manager, 'propagation_engine', None)
        if engine is not None:
            try:
                palette = engine.canonical_label_palette()
                if palette is not None and palette.shape[0] > 0:
                    return palette
            except Exception:
                pass

        try:
            ma = self.annotation_window.current_mask_annotation
        except Exception:
            ma = None
        if ma is None:
            return None
        try:
            return ma._get_color_map()
        except Exception:
            return None

    def _upload_lut(self, state, palette=None) -> None:
        if palette is None:
            palette = self._label_palette()
        if palette is None:
            return
        try:
            state.update_lut(palette)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # State lifecycle
    # ------------------------------------------------------------------
    def _is_paintable(self, product) -> bool:
        """Meshes, point clouds, and Gaussian splats are all paintable by the
        class-id shader (gl_PrimitiveID for meshes/splats, gl_VertexID for points)."""
        try:
            from coralnet_toolbox.MVAT.core.Products import MeshProduct, PointCloudProduct, GaussianSplattingProduct
            return isinstance(product, (MeshProduct, PointCloudProduct, GaussianSplattingProduct))
        except Exception:
            return False

    def _get_or_build_state(self, product):
        if not self.shader_enabled:
            return None
        pid = getattr(product, "product_id", None)
        try:
            n = int(product.get_element_count())
        except Exception:
            n = 0
        if n <= 0:
            return None

        palette = self._label_palette()
        n_labels = (palette.shape[0] - 1) if palette is not None else 255

        state = self._states.get(pid)
        need_rebuild = (
            state is None
            or state.N != n
            or not state.can_hold_labels(n_labels)
        )
        if need_rebuild:
            try:
                state = build_state(n, n_labels)
            except ShaderUnavailable as e:
                self._disable(f"build_state failed: {e}")
                return None
            self._states[pid] = state
            # Seed the texture with the product's existing class_ids so painted
            # labels show immediately on the first switch to the Labels array.
            try:
                class_ids = getattr(product, "class_ids", None)
                if class_ids is not None:
                    state.update_class_ids_full(np.asarray(class_ids))
            except Exception:
                pass
        return state

    # ------------------------------------------------------------------
    # Label-overlay actor (labels shown over a non-Labels base array)
    # ------------------------------------------------------------------
    def should_show_label_overlay(self, product) -> bool:
        """True when a translucent label-overlay actor should be drawn over the
        product's base array (slider open, not the Labels/Similarity array)."""
        if not self.shader_enabled or self.paint_opacity <= 0.0:
            return False
        if not self._is_paintable(product):
            return False
        sa = getattr(product, "selected_array", None)
        return sa not in ("Labels", "Similarity")

    def install_label_overlay_shader(self, actor, product) -> bool:
        """Install the replace+discard shader on a translucent overlay actor that
        shares the product's geometry. Returns True on success."""
        if actor is None or not self.shader_enabled:
            return False
        state = self._get_or_build_state(product)
        if state is None:
            return False
        self._upload_lut(state)

        element_type = getattr(product, "get_element_type", lambda: 'face')()

        try:
            install_paint_shader(actor, state, discard_unpainted=True, element_type=element_type)
            return True
        except ShaderUnavailable as e:
            self._disable(f"label overlay install failed: {e}")
            return False

    def uninstall_paint_shader(self, actor) -> None:
        """Remove the shader from an actor (array switch / tool deactivate)."""
        if actor is None:
            return
        try:
            uninstall_paint_shader(actor)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Paint hook (used in Phase 4) and maintenance
    # ------------------------------------------------------------------
    def update_class_ids_subset(self, product, element_ids, class_id, color_rgb=None) -> bool:
        """O(painted) class-id texture write for a stroke. Returns True if applied.

        When ``color_rgb`` is given, also pins ``LUT[class_id]`` to that exact color
        so the painted region always renders the color the user painted with —
        independent of any upstream class_id -> palette skew (e.g. the default
        'Review' label) that could otherwise make the displayed color off by one.
        """
        state = self._states.get(getattr(product, "product_id", None))
        if state is None:
            return False
        try:
            if color_rgb is not None and int(class_id) != 0:
                state.set_lut_entry(int(class_id), color_rgb)
            state.update_class_ids_subset(element_ids, class_id)
            return True
        except Exception:
            return False

    def get_state(self, product):
        return self._states.get(getattr(product, "product_id", None))

    def refresh_lut(self) -> None:
        """Re-upload the LUT to all live states (labels added/recolored)."""
        palette = self._label_palette()
        if palette is None:
            return
        for state in self._states.values():
            self._upload_lut(state, palette)

    def set_paint_opacity(self, value) -> bool:
        """Store the label-overlay opacity (0..1).

        Returns True when the 0 boundary was crossed — the caller should run
        render_scene to add/remove the overlay actors. Within (0, 1] the caller just
        updates each overlay actor's property opacity (no rebuild).
        """
        try:
            value = max(0.0, min(1.0, float(value)))
        except Exception:
            return False
        old = self.paint_opacity
        self.paint_opacity = value
        return (old <= 0.0) != (value <= 0.0)

    def forget_product(self, product_id) -> None:
        self._states.pop(product_id, None)

    def _disable(self, reason: str) -> None:
        print(f"[PaintShaderManager] disabled → overlay fallback: {reason}")
        self.shader_enabled = False
        self._states.clear()
