"""
SimilarityShader — Phase-2 GPU colormap for the Tier-2 feature-similarity recolor.

The bottleneck on large meshes is NOT the similarity math (that's already on the
GPU in torch and costs a few ms) — it's VTK's per-change ``MapScalars``: for CELL
data VTK maps scalars -> RGBA on the CPU every time the array changes, expands to
per-vertex colors, and re-uploads. That is O(N) CPU work (~125 ms at 4M faces,
seconds at 76M) and runs on every click / threshold tick.

This module bypasses ``MapScalars`` entirely:

  * the per-face display value [N] (uint8, produced exactly as before by
    QueryEngine.display_scalars) is packed 4-faces-per-texel into a small RGBA8
    texture and uploaded directly (no CPU color mapping, no vertex expansion);
  * a fragment-shader replacement reads that value by ``gl_PrimitiveID`` (== face
    id for an all-triangle mesh), looks it up in a 256-wide colormap LUT texture,
    and writes the color — with the mapper's ScalarVisibilityOff so VTK never
    touches the cell scalar on the GPU.

The shader does only the colormap; the similarity/threshold logic stays in torch.
Per click we still upload [N] bytes, but as a raw texture (a few ms) instead of
the ~125 ms CPU MapScalars path.

VTK 9.6 specifics (verified against the live build):
  * shader replacement lives on the ACTOR: ``actor.GetShaderProperty()`` returns
    a vtkOpenGLShaderProperty with ``AddFragmentShaderReplacement`` — the mapper
    has no shader API at all.
  * custom uniforms via ``sp.GetFragmentCustomUniforms().SetUniform*`` (these are
    auto-declared in the shader; no //VTK::Color::Dec uniform block, no
    UpdateShaderEvent observer needed).
  * textures via ``actor.GetProperty().SetTexture(name, vtkTexture)``; the sampler
    is exposed to the shader under ``name``.

Everything is best-effort: any failure raises ShaderUnavailable and the caller
falls back to the Phase-1 uint8 cell-scalar path.

Knobs (flip while bringing the shader up on a real build):
  * DECLARE_SAMPLERS — True if VTK does NOT auto-declare the property samplers.
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

try:
    from vtk.util.numpy_support import numpy_to_vtk  # type: ignore
except Exception:  # pragma: no cover
    try:
        from vtkmodules.util.numpy_support import numpy_to_vtk  # type: ignore
    except Exception:
        numpy_to_vtk = None


MAX_TEX_DIM = 16384        # most desktop GL supports 16384; packing guard
# VTK declares the samplers itself for property textures bound via
# prop.SetTexture (it also adds tcoord plumbing). Declaring them again here
# causes a duplicate-declaration compile error, so leave this False.
DECLARE_SAMPLERS = False
IMPL_REPLACE_FIRST = True
DUMP_SHADER = False


class ShaderUnavailable(RuntimeError):
    """Raised when the GPU colormap shader can't be used; caller falls back."""


# ---------------------------------------------------------------------------
# GLSL — colormap only. Reads the packed per-face value by gl_PrimitiveID and
# maps it through the colormap LUT.
#
# The disp-texture width is baked in as a literal constant per install (avoids a
# custom uniform, whose auto-declaration isn't reliable across VTK versions).
# Samplers (dispTex, cmapTex) are declared by VTK for the property textures,
# so we don't declare them. The marker is echoed so VTK's own Color code still
# runs first (declaring/setting ambientColor/diffuseColor); ours overwrites it.
# ---------------------------------------------------------------------------
def _color_dec() -> str:
    decl = "//VTK::Color::Dec\n"
    if DECLARE_SAMPLERS:
        decl += "uniform sampler2D dispTex;\n"
        decl += "uniform sampler2D cmapTex;\n"
    return decl


def _color_impl(disp_w: int, element_type: str = 'face') -> str:
    # NOTE: avoid GLSL reserved words — `packed` is reserved, so the texel var is
    # named `texel`. VTK auto-declares dispTex/cmapTex (property textures), so
    # DECLARE_SAMPLERS must stay False to avoid a duplicate-declaration error.
    w = str(int(disp_w))
    # Meshes natively use gl_PrimitiveID. Point clouds use our injected vertID.
    id_source = "vertID" if element_type == 'point' else "gl_PrimitiveID"

    return (
        "//VTK::Color::Impl\n"
        "{\n"
        f"  int fid = {id_source};\n"
        "  int texIdx = fid >> 2;\n"          # 4 faces packed per RGBA texel
        "  int ch = fid & 3;\n"
        "  ivec2 dc = ivec2(texIdx % " + w + ", texIdx / " + w + ");\n"
        "  vec4 texel = texelFetch(dispTex, dc, 0);\n"   # RGBA8 UNORM -> [0,1]
        "  float dispv = texel[ch];\n"
        "  int ci = int(clamp(dispv, 0.0, 1.0) * 255.0 + 0.5);\n"
        "  vec3 simColor = texelFetch(cmapTex, ivec2(ci, 0), 0).rgb;\n"
        "  ambientColor = simColor;\n"
        "  diffuseColor = vec3(0.0);\n"
        "}\n"
    )


# ---------------------------------------------------------------------------
# Texture builders
# ---------------------------------------------------------------------------
def _make_uint8_texture(rgba: np.ndarray, deep: bool = True):
    """Wrap an [H, W, 4] uint8 array in a nearest-sampled RGBA8 vtkTexture."""
    if vtk is None or numpy_to_vtk is None:
        raise ShaderUnavailable("VTK / numpy_support not importable")

    H, W, _ = rgba.shape
    flat = np.ascontiguousarray(rgba.reshape(-1, 4).astype(np.uint8))

    img = vtk.vtkImageData()
    img.SetDimensions(W, H, 1)
    arr = numpy_to_vtk(flat, deep=deep, array_type=vtk.VTK_UNSIGNED_CHAR)
    arr.SetNumberOfComponents(4)
    img.GetPointData().SetScalars(arr)

    tex = vtk.vtkTexture()
    tex.SetInputData(img)
    tex.InterpolateOff()
    tex.MipmapOff()
    try:
        tex.SetColorModeToDirectScalars()  # use bytes as-is (UNORM), no LUT
    except Exception:
        pass
    try:
        tex.SetWrapMode(vtk.vtkTexture.ClampToEdge)
    except Exception:
        pass
    tex._img = img
    tex._np = flat
    return tex, img, flat


def _disp_grid(n_faces: int) -> Tuple[int, int, int]:
    """Texel grid for N faces packed 4-per-RGBA-texel. Returns (W, H, n_texels)."""
    n_texels = (n_faces + 3) // 4
    W = min(MAX_TEX_DIM, n_texels) if n_texels > 0 else 1
    W = max(1, W)
    H = (n_texels + W - 1) // W
    if H > MAX_TEX_DIM:
        raise ShaderUnavailable(
            f"disp texture {W}x{H} exceeds {MAX_TEX_DIM} (N={n_faces})"
        )
    return W, H, n_texels


DEFAULT_COLORMAP = "plasma"


def build_colormap_texture(colormap_name: str = None):
    """Build a 256x1 RGBA8 LUT texture for the given matplotlib colormap."""
    if colormap_name is None:
        colormap_name = DEFAULT_COLORMAP
    try:
        import matplotlib
        cmap = matplotlib.colormaps[colormap_name]
    except Exception:
        try:
            import matplotlib.cm as cm
            cmap = cm.get_cmap(colormap_name)
        except Exception as e:  # pragma: no cover
            raise ShaderUnavailable(f"{colormap_name} colormap unavailable: {e}")
    lut = (np.asarray(cmap(np.linspace(0.0, 1.0, 256))) * 255.0).round()
    lut = lut.astype(np.uint8).reshape(1, 256, 4)
    tex, _img, _flat = _make_uint8_texture(lut)
    return tex


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class SimilarityShaderState:
    """Holds the GPU colormap textures and the packed disp buffer for a mesh."""

    def __init__(self, n_faces: int):
        self.N = int(n_faces)
        self.W, self.H, self.n_texels = _disp_grid(self.N)

        buf = np.zeros((self.H, self.W, 4), dtype=np.uint8)
        # Shallow wrap: update_disp writes ``disp_np`` in place each click.
        self.disp_tex, self.disp_img, self.disp_np = _make_uint8_texture(buf, deep=False)
        self.cmap_tex = build_colormap_texture()

    def update_disp(self, disp_uint8: np.ndarray) -> None:
        """Pack the per-face display values [N] uint8 into the disp texture."""
        d = np.ascontiguousarray(disp_uint8, dtype=np.uint8).ravel()
        if d.shape[0] != self.N:
            raise ShaderUnavailable(
                f"disp size {d.shape[0]} != N {self.N}"
            )
        pad = self.n_texels * 4 - self.N
        if pad:
            d = np.concatenate([d, np.zeros(pad, dtype=np.uint8)])
        packed = d.reshape(self.n_texels, 4)
        flat = self.disp_np  # [H*W, 4]
        flat[:self.n_texels] = packed
        if self.n_texels < flat.shape[0]:
            flat[self.n_texels:] = 0
        self.disp_img.GetPointData().GetScalars().Modified()
        self.disp_img.Modified()


def build_state(n_faces: int) -> "SimilarityShaderState":
    """Build the colormap textures once per bake / cache-load. May raise."""
    if vtk is None:
        raise ShaderUnavailable("VTK not importable")
    return SimilarityShaderState(n_faces)


# ---------------------------------------------------------------------------
# Install / uninstall (actor-side shader property, VTK 9.x)
# ---------------------------------------------------------------------------
def install_similarity_shader(actor, state: "SimilarityShaderState", element_type: str = 'face') -> bool:
    """
    Attach the colormap fragment shader + textures to ``actor`` via its shader
    property. Injects gl_VertexID piping if the actor is a point cloud.
    Idempotent; raises ShaderUnavailable on failure (caller falls back).
    """
    if vtk is None:
        raise ShaderUnavailable("VTK not importable")
    if actor is None or state is None:
        raise ShaderUnavailable("actor/state missing")
    if getattr(actor, "_sim_shader_installed", False):
        return True

    try:
        sp = getattr(actor, "GetShaderProperty", lambda: None)()
        prop = actor.GetProperty()
        mapper = actor.GetMapper()
        if sp is None or prop is None:
            raise ShaderUnavailable("actor has no shader property")

        # Shader owns the color: stop VTK mapping/uploading the cell scalar.
        if mapper is not None:
            try:
                mapper.ScalarVisibilityOff()
            except Exception:
                pass

        # Flat, unlit output: ambientColor (which we set) is the final color.
        actor._sim_prev_light = (
            prop.GetAmbient(), prop.GetDiffuse(), prop.GetSpecular()
        )
        prop.SetAmbient(1.0)
        prop.SetDiffuse(0.0)
        prop.SetSpecular(0.0)

        prop.SetTexture("dispTex", state.disp_tex)
        prop.SetTexture("cmapTex", state.cmap_tex)

        impl = _color_impl(state.W, element_type)
        dec = _color_dec()

        if element_type == 'point':
            dec += "flat in int vertID;\n"
            # Pipe gl_VertexID out of the VTK vertex stage (PRESERVE VTK'S MARKERS!)
            sp.AddVertexShaderReplacement(
                "//VTK::PositionVC::Dec", True,
                "//VTK::PositionVC::Dec\nflat out int vertID;\n", False
            )
            sp.AddVertexShaderReplacement(
                "//VTK::PositionVC::Impl", True,
                "//VTK::PositionVC::Impl\nvertID = gl_VertexID;\n", False
            )

        if DECLARE_SAMPLERS or element_type == 'point':
            sp.AddFragmentShaderReplacement("//VTK::Color::Dec", True, dec, False)
        sp.AddFragmentShaderReplacement(
            "//VTK::Color::Impl", IMPL_REPLACE_FIRST, impl, False
        )
        # Our dispTex/cmapTex are DATA textures sampled by gl_PrimitiveID via
        # texelFetch — not surface textures. On a mesh WITH texture coordinates
        # (e.g. a textured photogrammetry mesh) VTK would otherwise emit tcoord
        # sampling code here that multiplies our color by texture(dispTex, tcoord),
        # corrupting it (wrong colors / black). Suppress that block.
        sp.AddFragmentShaderReplacement("//VTK::TCoord::Impl", True, "", False)

        if DUMP_SHADER:
            print(f"[SimilarityShader] installed: N={state.N} dispTex={state.W}x{state.H} type={element_type}")
            print("[SimilarityShader] FRAG Impl:\n" + impl)

        actor._sim_shader_installed = True
        actor._sim_shader_sp = sp
        actor._sim_shader_state = state
        return True
    except ShaderUnavailable:
        raise
    except Exception as e:
        try:
            uninstall_similarity_shader(actor)
        except Exception:
            pass
        raise ShaderUnavailable(f"install failed: {e}")


def uninstall_similarity_shader(actor) -> None:
    """Remove the colormap shader + textures and restore normal rendering."""
    if actor is None:
        return
    sp = getattr(actor, "_sim_shader_sp", None)
    if sp is None:
        sp = getattr(actor, "GetShaderProperty", lambda: None)()
    try:
        if sp is not None:
            sp.ClearAllFragmentShaderReplacements()
    except Exception:
        pass
    try:
        prop = actor.GetProperty()
        for name in ("dispTex", "cmapTex"):
            try:
                prop.RemoveTexture(name)
            except Exception:
                pass
        prev = getattr(actor, "_sim_prev_light", None)
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
    for attr in ("_sim_shader_installed", "_sim_shader_sp", "_sim_shader_state",
                 "_sim_prev_light"):
        if hasattr(actor, attr):
            try:
                delattr(actor, attr)
            except Exception:
                pass
