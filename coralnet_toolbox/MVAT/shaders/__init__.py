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
