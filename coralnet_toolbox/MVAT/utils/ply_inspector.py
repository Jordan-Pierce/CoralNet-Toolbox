"""
Fast PLY header inspection to read element counts without loading geometry.

Useful for showing encoding hints and simplification recommendations before
mesh loading.
"""

import re
from pathlib import Path


def read_ply_header(ply_path: str) -> dict:
    """
    Parse PLY file header to extract element counts.

    Args:
        ply_path: Path to .ply file

    Returns:
        dict with keys: 'vertices', 'faces', 'encoding_hint'

    Raises:
        ValueError: If file is not a valid PLY or header is malformed
    """
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    # Read header (PLY files have text headers, max 100KB should cover it)
    with open(ply_path, 'rb') as f:
        header_data = f.read(102400).decode('utf-8', errors='ignore')

    # Find header end (marked by "end_header" line)
    header_end_idx = header_data.find('end_header')
    if header_end_idx == -1:
        raise ValueError("Invalid PLY file: no 'end_header' found")

    header = header_data[:header_end_idx]

    # Verify it's a PLY file
    if not header.startswith('ply'):
        raise ValueError("Invalid PLY file: missing 'ply' magic number")

    # Extract element counts from "element <name> <count>" lines
    elements = {}
    for line in header.split('\n'):
        line = line.strip()
        if line.startswith('element '):
            match = re.match(r'element\s+(\w+)\s+(\d+)', line)
            if match:
                name, count = match.groups()
                elements[name] = int(count)

    n_vertices = elements.get('vertex', 0)
    n_faces = elements.get('face', 0)

    # Determine encoding tier
    encoding_hint = _get_encoding_hint(n_faces)

    return {
        'vertices': n_vertices,
        'faces': n_faces,
        'encoding_hint': encoding_hint,
    }


def _get_encoding_hint(n_faces: int) -> str:
    """Return human-readable encoding hint and optimization suggestion."""
    if n_faces <= 255:
        return "8-bit encoding — Tiny mesh ✓"
    elif n_faces <= 65535:
        return "16-bit encoding — Small mesh ✓"
    elif n_faces <= 1000000:
        return "24-bit encoding — Medium mesh ✓"
    elif n_faces <= 16777215:
        return "32-bit encoding — Large mesh (consider simplification for faster processing)"
    else:
        return "32-bit encoding — Very large mesh (recommend simplification to < 1M faces)"


def suggest_simplification_factor(n_faces: int, target_faces: int = 1000000) -> float:
    """
    Suggest simplification factor to reach target face count.

    Args:
        n_faces: Current face count
        target_faces: Desired face count (default 1M for optimal perf)

    Returns:
        Simplification factor as float in (0, 1). 1.0 = no simplification.
    """
    if n_faces <= target_faces:
        return 1.0
    return target_faces / n_faces


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ply_inspector.py <ply_file>")
        sys.exit(1)

    ply_file = sys.argv[1]
    info = read_ply_header(ply_file)

    print(f"\n{'='*60}")
    print(f"PLY File: {Path(ply_file).name}")
    print(f"{'='*60}")
    print(f"Vertices: {info['vertices']:,}")
    print(f"Faces:    {info['faces']:,}")
    print(f"Encoding: {info['encoding_hint']}")

    if info['faces'] > 1000000:
        factor = suggest_simplification_factor(info['faces'])
        print(f"\n💡 Suggestion:")
        print(f"   Simplify to {1000000:,} faces ({factor*100:.1f}% of original)")
        print(f"   This enables 24-bit encoding and faster performance.")

    print(f"{'='*60}\n")
