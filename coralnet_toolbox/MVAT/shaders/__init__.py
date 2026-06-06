"""GLSL shader sources for MVAT GPU rasterization."""

VERT = """
#version 330 core
in vec3 position;
uniform mat4 mvp;
void main() {
    gl_Position = mvp * vec4(position, 1.0);
}
"""

# Tiered encoding shaders for different mesh element counts
FRAG_FACE_ID_R8 = """
#version 330 core
out uint fragFaceID;
void main() {
    fragFaceID = uint(gl_PrimitiveID + 1);
}
"""

FRAG_FACE_ID_RG16 = """
#version 330 core
out uvec2 fragFaceID;
void main() {
    uint id = uint(gl_PrimitiveID + 1);
    fragFaceID = uvec2(
        (id >> 8) & 0xFFu,
        id & 0xFFu
    );
}
"""

FRAG_FACE_ID_RGB24 = """
#version 330 core
out vec3 fragFaceID;
void main() {
    uint id = uint(gl_PrimitiveID + 1);
    fragFaceID = vec3(
        float((id >> 16) & 0xFFu) / 255.0,
        float((id >> 8) & 0xFFu) / 255.0,
        float(id & 0xFFu) / 255.0
    );
}
"""

FRAG_FACE_ID_INT = """
#version 330 core
out int fragFaceID;
void main() {
    fragFaceID = gl_PrimitiveID + 1;
}
"""
