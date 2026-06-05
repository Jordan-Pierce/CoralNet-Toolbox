"""GLSL shader sources for MVAT GPU rasterization."""

VERT = """
#version 330 core
in vec3 position;
uniform mat4 mvp;
void main() {
    gl_Position = mvp * vec4(position, 1.0);
}
"""

FRAG_FACE_ID_LOW = """
#version 330 core
out vec4 fragColor;
void main() {
    // Encode face ID (1-based) into RGB so _decode_face_id_screenshot works unchanged.
    // background = RGB(0,0,0); face 0 = RGB(1,0,0); face 16M = high-pass needed.
    int encoded = gl_PrimitiveID + 1;
    fragColor = vec4(
        float( encoded        & 0xFF) / 255.0,
        float((encoded >>  8) & 0xFF) / 255.0,
        float((encoded >> 16) & 0xFF) / 255.0,
        1.0
    );
}
"""

FRAG_FACE_ID_HIGH = """
#version 330 core
out vec4 fragColor;
void main() {
    // Second pass for meshes with > 16,777,215 faces.
    // Encodes bits 24-31 of the face ID into the R channel.
    int encoded = gl_PrimitiveID + 1;
    fragColor = vec4(
        float((encoded >> 24) & 0xFF) / 255.0,
        0.0, 0.0, 1.0
    );
}
"""
