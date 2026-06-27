"""GLSL shader sources for MVAT GPU rasterization."""

VERT = """
#version 330 core
in vec3 position;
uniform mat4 mvp;
void main() {
    vec4 clip = mvp * vec4(position, 1.0);
    // Bake the vertical flip into clip space so the framebuffer is already in
    // top-to-bottom image order. This removes the CPU [::-1] flip + contiguity
    // copy after readback. gl_PrimitiveID and depth (z/w) are unaffected, and
    // there is no face culling so the reversed winding is harmless.
    clip.y = -clip.y;
    gl_Position = clip;
}
"""

# Single-channel R32I face-ID target. Face count is always int32, so the tiered
# byte-packed encodings (R8/RG16/RGB24) and their CPU bit-unpack decoders are gone.
#
# A second R32F color attachment (location 1) carries the *linearized* camera-space
# depth. Computing it here (instead of reading the raw depth attachment and running
# the z_ndc → linear formula per pixel on the CPU) keeps the depth math on the GPU;
# the readback only reshapes and masks the background. u_near/u_far mirror the
# clipping planes baked into _build_mvp.
FRAG_FACE_ID_INT = """
#version 330 core
layout(location = 0) out int fragFaceID;
layout(location = 1) out float fragLinearDepth;
uniform float u_near;
uniform float u_far;
void main() {
    fragFaceID = gl_PrimitiveID + 1;
    float z_ndc = 2.0 * gl_FragCoord.z - 1.0;
    fragLinearDepth = (2.0 * u_near * u_far) / (u_far + u_near - z_ndc * (u_far - u_near));
}
"""

# ---------------------------------------------------------------------------
# Point-cloud GL_POINTS shaders
# ---------------------------------------------------------------------------
# Point clouds render through the SAME R32I face-ID pipeline as meshes, only the
# draw primitive (GL_POINTS) and the ID source (gl_VertexID instead of
# gl_PrimitiveID) differ. This unifies depth-buffer occlusion, viewport
# cropping, FBO caching, CUDA-GL readback and the distortion warp across both
# geometry types. ``point_size`` (gl_PointSize, render-resolution pixels) gives
# each point volume so a foreground cloud actually occludes the background;
# ``splat_round`` discards the corners of the square point sprite to make a disc.
VERT_POINT = """
#version 330 core
in vec3 position;
uniform mat4 mvp;
uniform float point_size;
flat out int vertID;
void main() {
    vertID = gl_VertexID;
    vec4 clip = mvp * vec4(position, 1.0);
    // Bake the vertical flip into clip space (matches the mesh VERT shader) so
    // the framebuffer is already top-to-bottom and no CPU [::-1] is needed.
    clip.y = -clip.y;
    gl_Position = clip;
    gl_PointSize = point_size;  // requires ctx.enable(PROGRAM_POINT_SIZE)
}
"""

# Single-channel R32I point-ID target. Mirrors FRAG_FACE_ID_INT: writes
# gl_VertexID + 1 (so 0 stays background, reversed to -1 on readback). The R32F
# location-1 output carries linearized camera-space depth (see FRAG_FACE_ID_INT).
FRAG_POINT_ID_INT = """
#version 330 core
flat in int vertID;
uniform int splat_round;   // 1 = round disc, 0 = square sprite
uniform float u_near;
uniform float u_far;
layout(location = 0) out int fragPointID;
layout(location = 1) out float fragLinearDepth;
void main() {
    if (splat_round == 1) {
        // gl_PointCoord is [0,1]^2 across the sprite; discard outside the
        // inscribed circle (radius 0.5) to render a filled disc.
        vec2 d = gl_PointCoord - vec2(0.5);
        if (dot(d, d) > 0.25) discard;
    }
    fragPointID = vertID + 1;
    float z_ndc = 2.0 * gl_FragCoord.z - 1.0;
    fragLinearDepth = (2.0 * u_near * u_far) / (u_far + u_near - z_ndc * (u_far - u_near));
}
"""

# ---------------------------------------------------------------------------
# 3D Gaussian Splatting (3DGS) index-map shader
# ---------------------------------------------------------------------------
# Approach B (accurate solid ellipsoids): reuse pyvista_gs' gau_vert.glsl to
# project the 3D covariances exactly as the live renderer does, but swap the
# color-blend fragment shader for a hard-cutoff integer Splat-ID writer. With
# DEPTH_TEST enabled and a hard alpha cutoff each splat becomes an opaque
# ellipsoid, so the nearest splat wins per pixel (no CPU depth sorting needed).
#
# NOTE: ``gl_InstanceID`` is a VERTEX-stage builtin and is NOT readable in the
# fragment stage. gau_vert.glsl is patched at load time (see
# VisibilityManager.setup_batch_splat_moderngl_context) to emit the resolved
# splat index ``boxid`` through the flat varying ``v_splatID``, which this
# shader reads. IDs are written 1-based so background stays 0 (→ -1 after the
# readback's `-1` decode, matching the mesh/point paths).
FRAG_SPLAT_ID_INT = """
#version 430 core
in vec3 color;
in float alpha;
in vec3 conic;
in vec2 coordxy;
flat in int v_splatID;

uniform float u_near;
uniform float u_far;

layout(location = 0) out int fragFaceID;
layout(location = 1) out float fragLinearDepth;

void main() {
    // Exact ellipsoid boundary math (mirrors gau_frag.glsl).
    float power = -0.5f * (conic.x * coordxy.x * coordxy.x + conic.z * coordxy.y * coordxy.y) - conic.y * coordxy.x * coordxy.y;
    if (power > 0.f) discard;

    // Hard cutoff for index-map solidity.
    float opacity = min(0.99f, alpha * exp(power));
    if (opacity < 0.1f) discard;

    fragFaceID = v_splatID + 1;
    float z_ndc = 2.0 * gl_FragCoord.z - 1.0;
    fragLinearDepth = (2.0 * u_near * u_far) / (u_far + u_near - z_ndc * (u_far - u_near));
}
"""

# ---------------------------------------------------------------------------
# Visible-element coverage (compute shader)
# ---------------------------------------------------------------------------
# After rasterization the 1-based element-ID map lives in the FBO's R32I color
# attachment (still on the GPU). This compute pass scans it and sets cov[id]=1
# for every visible element, so the CPU replaces an O(num_valid_pixels)
# np.bincount over the whole image (~66-109ms at production resolution) with a
# small O(N_elements) buffer readback + np.flatnonzero (~0.4-18ms).
#
# Requires GL 4.3 (compute + SSBO). Callers must compile this defensively and
# fall back to bincount when unavailable (e.g. legacy macOS 4.1) — the standalone
# context reports version 3.3 even on drivers that support compute, so capability
# must be detected by trying to compile, not by version number.
# ---------------------------------------------------------------------------
# Lens-distortion warp (fullscreen index/depth remap)
# ---------------------------------------------------------------------------
# Re-projects an undistorted index/depth map back into the distorted photo space
# on the GPU, replacing the CPU cv2.remap / torch grid_sample post-step. Each
# output (distorted) pixel reads its source coordinate from a precomputed RG32F
# map texture (Raster._map_x/_map_y, in *output*-resolution pixel units) and
# samples the source map with NEAREST addressing — integer element IDs must never
# be interpolated. Matches grid_sample(mode='nearest', align_corners=True,
# padding_mode='zeros') + a border fill for out-of-bounds source coords.
#
# The vertex stage emits a single full-screen triangle from gl_VertexID, so the
# pass needs no vertex buffer: vao = ctx.vertex_array(prog, []) ; render(vertices=3).
WARP_VERT = """
#version 330 core
void main() {
    // Oversized triangle covering clip space: (-1,-1), (3,-1), (-1,3).
    vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0,
                  (gl_VertexID == 2) ? 3.0 : -1.0);
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

# Combined index + linear-depth warp, used to fuse distortion into the render pass:
# the source textures are the *resident* render-FBO attachments (1-based R32I index
# in color 0, R32F linear depth in color 1), so distortion costs only the existing
# single readback instead of a separate cv2.remap / grid_sample round-trip.
#
# The index source is 1-based (0 = background), so out-of-bounds writes 0 and the
# normal `-1` decode on readback turns it into background. Depth out-of-bounds writes
# 0.0 and is masked to NaN later via the decoded index == -1, matching the
# non-distorted path. NEAREST/texelFetch only — element IDs must never interpolate.
WARP_FRAG = """
#version 330 core
uniform isampler2D srcIdx;     // render FBO color 0 (1-based index, render-res)
uniform sampler2D  srcDepth;   // render FBO color 1 (linear depth, render-res)
uniform sampler2D  mapTex;     // RG32F: source (x, y) per output pixel, output-res units
uniform ivec2 in_size;         // source texture size (render-res)
uniform ivec2 out_size;        // output texture size (native == map size)
uniform int   have_depth;
layout(location = 0) out int   fragIdx;
layout(location = 1) out float fragDepth;
void main() {
    ivec2 o = ivec2(gl_FragCoord.xy);
    vec2 src = texelFetch(mapTex, o, 0).rg;
    if (src.x < 0.0 || src.x > float(out_size.x - 1) ||
        src.y < 0.0 || src.y > float(out_size.y - 1)) {
        fragIdx = 0;        // 1-based background → -1 after decode
        fragDepth = 0.0;    // masked to NaN later via index == -1
        return;
    }
    // align_corners=True normalization (range size-1), then nearest-round. When
    // in_size == out_size this is just round(src).
    int ix = clamp(int(round(src.x / float(out_size.x - 1) * float(in_size.x - 1))), 0, in_size.x - 1);
    int iy = clamp(int(round(src.y / float(out_size.y - 1) * float(in_size.y - 1))), 0, in_size.y - 1);
    fragIdx = texelFetch(srcIdx, ivec2(ix, iy), 0).r;
    fragDepth = (have_depth == 1) ? texelFetch(srcDepth, ivec2(ix, iy), 0).r : 0.0;
}
"""

COVERAGE_CS = """
#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(r32i) uniform readonly iimage2D idxTex;   // FBO color attachment 0 (1-based IDs)
// Binding 5 is deliberately high: the 3DGS render path occupies SSBO bindings 0
// and 1 (gaussian data + draw order), and these binding points are global GL
// state shared across programs. Using 5 keeps the coverage buffer from clobbering
// the splat bindings between the compute dispatch and the next camera's draw.
layout(std430, binding = 5) buffer Cov { uint cov[]; };
uniform ivec2 dims;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= dims.x || p.y >= dims.y) return;
    int id = imageLoad(idxTex, p).r - 1;         // decode 1-based → 0-based
    if (id >= 0) cov[id] = 1u;                    // all writers store the same value; no atomics needed
}
"""
