"""GLSL shader sources for MVAT GPU rasterization."""

VERT = """
#version 330 core
in vec3 position;
uniform mat4 mvp;
void main() {
    gl_Position = mvp * vec4(position, 1.0);
}
"""

FRAG_FACE_ID_INT = """
#version 330 core
out int fragFaceID;
void main() {
    fragFaceID = gl_PrimitiveID + 1;
}
"""
