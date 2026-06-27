"""
PaintShader — GPU label colormap for the MVAT "Labels" array.

Painting labels onto a mesh/point cloud used to render through floating overlay
actors built by ``LabelWorker`` (per-stroke PolyData + an O(P log P) vertex weld
of *all* painted faces on every stroke end). That decoupled the paint from the
array dropdown (paint floated over RGB / Texture / etc.) and got heavier the more
you painted.

This module renders labels the same way ``SimilarityShader`` renders the feature
heatmap — by reading a per-element value in a fragment-shader replacement and
looking it up in a small color LUT, bypassing VTK's ``MapScalars`` entirely:

  * the per-element ``class_id`` [N] (int32 in RAM, the source of truth) is packed
    16-bit, two elements per RGBA8 texel, into a small "cid" texture;
  * a fragment-shader replacement reads that value by ``gl_PrimitiveID`` (== face
    id for an all-triangle mesh; == point id for a GL_POINTS cloud), decodes the
    16-bit class id, and looks it up in a label-color LUT texture;
  * ``class_id == 0`` (unlabeled) renders white — the current unpainted look.

Differences from ``SimilarityShader`` (which this is modeled on):
  * class ids are NOT bounded to 256 (projects can have 500+ labels), so the value
    is packed 16-bit (2-per-texel) instead of uint8 (4-per-texel), and the LUT is
    sized to the project label count instead of a fixed 256-wide colormap.

Painting becomes ``update_class_ids_subset(ids, class_id)`` — an O(painted) byte
write + ``Modified()`` — with no geometry and no actor churn.

Everything is best-effort: any failure raises ``ShaderUnavailable`` (re-exported
from SimilarityShader so callers can catch one type) and the caller falls back to
the LabelWorker overlay path.

NOTE: the 16-bit pack assumes a little-endian host (x86 / ARM); the uint16 view of
the class-id array maps low byte -> R/B, high byte -> G/A directly.

Knobs (mirror SimilarityShader; flip while bringing the shader up on a real build):
  * IMPL_REPLACE_FIRST — toggle if our color write is overwritten by VTK's.
  * DUMP_SHADER — print the generated fragment source once for diagnosis.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import vtk  # type: ignore
except Exception:  # pragma: no cover
    vtk = None

# Reuse the texture helper, the texture-dimension guard, and the exception type
# so callers can ``except ShaderUnavailable`` uniformly across both shaders.
from coralnet_toolbox.MVAT.shaders.SimilarityShader import (
    ShaderUnavailable,
    MAX_TEX_DIM,
    _make_uint8_texture,
)


IMPL_REPLACE_FIRST = True
DUMP_SHADER = False

# Class id 0 is reserved for "unlabeled" (see QtMaskAnnotation); render it white,
# matching the default unpainted look on the Labels array.
UNLABELED_COLOR = (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Texel-grid sizing
# ---------------------------------------------------------------------------
def _cid_grid(n_elements: int) -> Tuple[int, int, int]:
    """Texel grid for N elements packed 2 (16-bit) per RGBA8 texel.

    Returns (W, H, n_texels).
    """
    n_texels = (n_elements + 1) // 2
    W = min(MAX_TEX_DIM, n_texels) if n_texels > 0 else 1
    W = max(1, W)
    H = (n_texels + W - 1) // W
    if H > MAX_TEX_DIM:
        raise ShaderUnavailable(
            f"cid texture {W}x{H} exceeds {MAX_TEX_DIM} (N={n_elements})"
        )
    return W, H, n_texels


def _lut_grid(n_labels: int) -> Tuple[int, int, int]:
    """Texel grid for the label LUT, indexed directly by class_id.

    Capacity is ``max(256, n_labels + 1)`` (index 0 reserved for unlabeled, plus
    headroom so a few new labels don't force a rebuild). Returns (W, H, cap).
    """
    cap = max(256, int(n_labels) + 1)
    W = min(MAX_TEX_DIM, cap)
    W = max(1, W)
    H = (cap + W - 1) // W
    if H > MAX_TEX_DIM:
        raise ShaderUnavailable(
            f"lut texture {W}x{H} exceeds {MAX_TEX_DIM} (n_labels={n_labels})"
        )
    return W, H, cap


# ---------------------------------------------------------------------------
# GLSL — read packed 16-bit class id by gl_PrimitiveID, look up the label LUT.
# The cid-texture width and LUT width are baked in as literals per install (the
# install rebuilds the Impl on every actor rebuild, so a resized state is safe).
# Samplers (cidTex, lutTex) are declared by VTK for the property textures.
# ---------------------------------------------------------------------------
def _decode_cid_glsl(cid_w: int, element_type: str = 'face') -> str:
    """GLSL that decodes the 16-bit ``cid`` for this fragment from ``cidTex``."""
    cw = str(int(cid_w))
    id_source = "vertID" if element_type == 'point' else "gl_PrimitiveID"
    return (
        f"  int eid = {id_source};\n"
        "  int texIdx = eid >> 1;\n"            # 2 elements packed per RGBA texel
        "  int which = eid & 1;\n"
        "  ivec2 cc = ivec2(texIdx % " + cw + ", texIdx / " + cw + ");\n"
        "  vec4 texel = texelFetch(cidTex, cc, 0);\n"   # RGBA8 UNORM -> [0,1]
        "  float lo; float hi;\n"
        "  if (which == 0) { lo = texel.r; hi = texel.g; }\n"
        "  else            { lo = texel.b; hi = texel.a; }\n"
        "  int cid = int(lo * 255.0 + 0.5) + int(hi * 255.0 + 0.5) * 256;\n"
    )


def _color_impl(cid_w: int, lut_w: int, discard_unpainted: bool = False, element_type: str = 'face') -> str:
    """Replace-mode color: the shader owns the color completely (flat label color).

    Gaussians blend the paint over their SH colors with mix() to preserve
    view-dependent shading; meshes and point clouds overwrite entirely.
    """
    lw = str(int(lut_w))
    ur, ug, ub = UNLABELED_COLOR

    if discard_unpainted:
        unpainted = "    discard;\n"
    else:
        unpainted = "    ambientColor = vec3(" + f"{ur}, {ug}, {ub}" + ");\n"

    # Gaussians blend paint over SH colors; meshes/points overwrite.
    apply_color = (
        "    ambientColor = mix(ambientColor, labelColor, 0.8);\n"
        if element_type == 'splat' else
        "    ambientColor = labelColor;\n"
    )

    return (
        "//VTK::Color::Impl\n"
        "{\n"
        + _decode_cid_glsl(cid_w, element_type) +
        "  if (cid == 0) {\n"
        + unpainted +
        "  } else {\n"
        "    ivec2 lc = ivec2(cid % " + lw + ", cid / " + lw + ");\n"
        "    vec3 labelColor = texelFetch(lutTex, lc, 0).rgb;\n"
        + apply_color +
        "  }\n"
        "  diffuseColor = vec3(0.0);\n"
        "}\n"
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class PaintShaderState:
    """Holds the GPU class-id value texture and the label LUT for one product."""

    def __init__(self, n_elements: int, n_labels: int):
        if vtk is None:
            raise ShaderUnavailable("VTK not importable")

        self.N = int(n_elements)
        self.n_labels = int(n_labels)

        # Per-element class-id texture: 2 elements (16-bit each) per RGBA8 texel.
        self.cid_W, self.cid_H, self.cid_texels = _cid_grid(self.N)
        cid_buf = np.zeros((self.cid_H, self.cid_W, 4), dtype=np.uint8)
        self.cid_tex, self.cid_img, self.cid_np = _make_uint8_texture(cid_buf, deep=False)

        # Label LUT: indexed directly by class_id (row 0 == unlabeled sentinel).
        self.lut_W, self.lut_H, self.lut_cap = _lut_grid(self.n_labels)
        lut_buf = np.zeros((self.lut_H, self.lut_W, 4), dtype=np.uint8)
        self.lut_tex, self.lut_img, self.lut_np = _make_uint8_texture(lut_buf, deep=False)

    # -- class-id texture ------------------------------------------------
    def update_class_ids_full(self, class_ids: np.ndarray) -> None:
        """Repack the entire per-element class-id array [N] into the cid texture."""
        cid = np.ascontiguousarray(class_ids, dtype=np.uint16).ravel()
        if cid.shape[0] != self.N:
            raise ShaderUnavailable(f"class_ids size {cid.shape[0]} != N {self.N}")
        pad = self.cid_texels * 2 - self.N
        if pad:
            cid = np.concatenate([cid, np.zeros(pad, dtype=np.uint16)])
        # uint16 little-endian view -> bytes laid out exactly as RGBA per texel:
        # [e0 lo, e0 hi, e1 lo, e1 hi] == [R, G, B, A].
        as_bytes = cid.view(np.uint8).reshape(self.cid_texels, 4)
        flat = self.cid_np  # [H*W, 4]
        flat[:self.cid_texels] = as_bytes
        if self.cid_texels < flat.shape[0]:
            flat[self.cid_texels:] = 0
        self._mark_cid_dirty()

    def update_class_ids_subset(self, element_ids: np.ndarray, class_id: int) -> None:
        """Set ``class_id`` on a subset of elements (the painted stroke). O(painted)."""
        ids = np.asarray(element_ids, dtype=np.int64).ravel()
        if ids.size == 0:
            return
        # Guard against stray ids past the buffer (defensive; brush should not).
        if ids.max(initial=-1) >= self.N or ids.min(initial=0) < 0:
            ids = ids[(ids >= 0) & (ids < self.N)]
            if ids.size == 0:
                return
        cid = int(class_id) & 0xFFFF
        lo = cid & 0xFF
        hi = (cid >> 8) & 0xFF
        rows = ids >> 1
        col = (ids & 1) * 2          # 0 -> R/G, 1 -> B/A
        flat = self.cid_np
        flat[rows, col] = lo
        flat[rows, col + 1] = hi
        self._mark_cid_dirty()

    def _mark_cid_dirty(self) -> None:
        self.cid_img.GetPointData().GetScalars().Modified()
        self.cid_img.Modified()

    # -- label LUT -------------------------------------------------------
    def can_hold_labels(self, n_labels: int) -> bool:
        """True if the current LUT capacity covers ``n_labels`` (else rebuild state)."""
        return int(n_labels) + 1 <= self.lut_cap

    def update_lut(self, palette_rgb: np.ndarray) -> None:
        """Upload the label palette; row i == RGB color for ``class_id == i``."""
        p = np.asarray(palette_rgb, dtype=np.uint8)
        if p.ndim != 2 or p.shape[1] < 3:
            raise ShaderUnavailable(f"palette must be (n,3+), got {p.shape}")
        n = min(p.shape[0], self.lut_W * self.lut_H)
        flat = self.lut_np  # [H*W, 4]
        flat[:] = 0
        flat[:n, :3] = p[:n, :3]
        flat[:n, 3] = 255
        self._mark_lut_dirty()

    def set_lut_entry(self, class_id: int, rgb) -> None:
        """Set a single LUT row from the exact color a stroke painted with.

        Called on every paint so the rendered color of a painted region is always
        the color the user painted with — independent of any upstream class_id ->
        color-map skew (e.g. the default 'Review' label offsetting the palette).
        """
        cid = int(class_id)
        if cid <= 0 or cid >= self.lut_W * self.lut_H:
            return
        try:
            r, g, b = int(rgb[0]) & 255, int(rgb[1]) & 255, int(rgb[2]) & 255
        except Exception:
            return
        flat = self.lut_np
        flat[cid, 0] = r
        flat[cid, 1] = g
        flat[cid, 2] = b
        flat[cid, 3] = 255
        self._mark_lut_dirty()

    def _mark_lut_dirty(self) -> None:
        self.lut_img.GetPointData().GetScalars().Modified()
        self.lut_img.Modified()


def build_state(n_elements: int, n_labels: int) -> "PaintShaderState":
    """Build the paint textures once per product (rebuild when capacity grows)."""
    if vtk is None:
        raise ShaderUnavailable("VTK not importable")
    return PaintShaderState(n_elements, n_labels)


# ---------------------------------------------------------------------------
# Install / uninstall (actor-side shader property, VTK 9.x) — mirrors
# SimilarityShader.install_similarity_shader.
# ---------------------------------------------------------------------------
def install_paint_shader(actor, state: "PaintShaderState",
                         discard_unpainted: bool = False,
                         element_type: str = 'face') -> bool:
    """
    Attach the label fragment shader + textures to ``actor`` (replace mode).
    Injects gl_VertexID piping if the actor is a point cloud.
    Idempotent; raises ShaderUnavailable on failure (caller falls back).
    """
    if vtk is None:
        raise ShaderUnavailable("VTK not importable")
    if actor is None or state is None:
        raise ShaderUnavailable("actor/state missing")
    if getattr(actor, "_paint_shader_installed", False):
        return True

    try:
        sp = getattr(actor, "GetShaderProperty", lambda: None)()
        prop = actor.GetProperty()
        mapper = actor.GetMapper()
        if sp is None or prop is None:
            raise ShaderUnavailable("actor has no shader property")

        prop.SetTexture("cidTex", state.cid_tex)
        prop.SetTexture("lutTex", state.lut_tex)

        if mapper is not None:
            try:
                mapper.ScalarVisibilityOff()
            except Exception:
                pass
        actor._paint_prev_light = (
            prop.GetAmbient(), prop.GetDiffuse(), prop.GetSpecular()
        )
        prop.SetAmbient(1.0)
        prop.SetDiffuse(0.0)
        prop.SetSpecular(0.0)

        impl = _color_impl(state.cid_W, state.lut_W, discard_unpainted=discard_unpainted, element_type=element_type)

        if element_type == 'point':
            # Pipe gl_VertexID out of the VTK vertex stage (PRESERVE VTK'S MARKERS!)
            sp.AddVertexShaderReplacement(
                "//VTK::PositionVC::Dec", True,
                "//VTK::PositionVC::Dec\nflat out int vertID;\n", False
            )
            sp.AddVertexShaderReplacement(
                "//VTK::PositionVC::Impl", True,
                "//VTK::PositionVC::Impl\nvertID = gl_VertexID;\n", False
            )
            sp.AddFragmentShaderReplacement(
                "//VTK::Color::Dec", True,
                "//VTK::Color::Dec\nflat in int vertID;\n", False
            )

        sp.AddFragmentShaderReplacement(
            "//VTK::Color::Impl", IMPL_REPLACE_FIRST, impl, False
        )
        sp.AddFragmentShaderReplacement("//VTK::TCoord::Impl", True, "", False)

        if DUMP_SHADER:
            print(f"[PaintShader] installed (discard={discard_unpainted}): "
                  f"N={state.N} cidTex={state.cid_W}x{state.cid_H} "
                  f"lut={state.lut_W}x{state.lut_H} type={element_type}")

        actor._paint_shader_installed = True
        actor._paint_shader_sp = sp
        actor._paint_shader_state = state
        return True
    except ShaderUnavailable:
        raise
    except Exception as e:
        try:
            uninstall_paint_shader(actor)
        except Exception:
            pass
        raise ShaderUnavailable(f"install failed: {e}")


def uninstall_paint_shader(actor) -> None:
    """Remove the label-colormap shader + textures and restore normal rendering."""
    if actor is None:
        return
    sp = getattr(actor, "_paint_shader_sp", None)
    if sp is None:
        sp = getattr(actor, "GetShaderProperty", lambda: None)()
    try:
        if sp is not None:
            sp.ClearAllFragmentShaderReplacements()
    except Exception:
        pass
    try:
        prop = actor.GetProperty()
        for name in ("cidTex", "lutTex"):
            try:
                prop.RemoveTexture(name)
            except Exception:
                pass
        prev = getattr(actor, "_paint_prev_light", None)
        if prev is not None:
            prop.SetAmbient(prev[0]); prop.SetDiffuse(prev[1]); prop.SetSpecular(prev[2])
    except Exception:
        pass
    try:
        mapper = actor.GetMapper()
        if mapper is not None:
            mapper.ScalarVisibilityOn()
    except Exception:
        pass
    for attr in ("_paint_shader_installed", "_paint_shader_sp", "_paint_shader_state",
                 "_paint_prev_light", "_paint_shader_mode"):
        if hasattr(actor, attr):
            try:
                delattr(actor, attr)
            except Exception:
                pass
