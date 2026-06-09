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
FRAG_FACE_ID_INT = """
#version 330 core
out int fragFaceID;
void main() {
    fragFaceID = gl_PrimitiveID + 1;
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
# gl_VertexID + 1 (so 0 stays background, reversed to -1 on readback).
FRAG_POINT_ID_INT = """
#version 330 core
flat in int vertID;
uniform int splat_round;   // 1 = round disc, 0 = square sprite
out int fragPointID;
void main() {
    if (splat_round == 1) {
        // gl_PointCoord is [0,1]^2 across the sprite; discard outside the
        // inscribed circle (radius 0.5) to render a filled disc.
        vec2 d = gl_PointCoord - vec2(0.5);
        if (dot(d, d) > 0.25) discard;
    }
    fragPointID = vertID + 1;
}
"""
