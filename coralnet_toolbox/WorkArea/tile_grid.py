"""Tile grid layout for work areas.

One pure function decides where tiles go, so the Work Area Manager's preview
and its Apply step can never disagree, and later callers (Extract Work Areas)
reuse the same layout. No Qt here: rectangles are plain (x, y, width, height)
tuples in image pixels.
"""

from dataclasses import dataclass, field


# ----------------------------------------------------------------------------------------------------------------------
# Edge modes
# ----------------------------------------------------------------------------------------------------------------------

# Drop tiles that would not fit whole inside the usable area.
EDGE_SKIP = "skip"
# Keep every tile full size; add a last column / row slid inward against the
# far edge so the whole usable area is covered. Those tiles overlap their
# neighbours by more than the requested overlap.
EDGE_SHIFT = "shift"
# Cut the last column / row smaller so coverage is exact and no tile overlaps
# more than requested. With zero overlap every pixel is in exactly one tile,
# which is what merging tiles back into the original needs.
EDGE_SHRINK = "shrink"

EDGE_MODES = (EDGE_SKIP, EDGE_SHIFT, EDGE_SHRINK)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TileGrid:
    """Result of compute_tile_grid.

    rects: (x, y, width, height) per tile, row by row.
    cols / rows: size of the regular grid. With EDGE_SHIFT the extra edge
        column / row is not counted, matching what the Work Area Manager has
        always reported.
    """
    rects: list = field(default_factory=list)
    cols: int = 0
    rows: int = 0

    def __len__(self):
        return len(self.rects)


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def compute_tile_grid(image_width, image_height, tile_width, tile_height,
                      overlap_width=0, overlap_height=0, margins=(0, 0, 0, 0),
                      edge_mode=EDGE_SHIFT):
    """Lay out tiles over the usable area of an image.

    Args:
        image_width, image_height: image size in pixels.
        tile_width, tile_height: tile size in pixels.
        overlap_width, overlap_height: overlap between neighbouring tiles in pixels.
        margins: (left, top, right, bottom) in pixels, excluded from tiling.
        edge_mode: EDGE_SKIP, EDGE_SHIFT or EDGE_SHRINK.

    Returns:
        TileGrid

    Raises:
        ValueError: unknown edge_mode, overlap not smaller than the tile, or
            margins that leave no usable area.
    """
    if edge_mode not in EDGE_MODES:
        raise ValueError(f"Unknown edge mode: {edge_mode!r}")

    left, top, right, bottom = margins
    usable_width = image_width - left - right
    usable_height = image_height - top - bottom
    if usable_width <= 0 or usable_height <= 0:
        raise ValueError("Margins are too large for the image size.")

    effective_width = tile_width - overlap_width
    effective_height = tile_height - overlap_height
    if effective_width <= 0 or effective_height <= 0:
        raise ValueError("Effective tile size must be positive. Reduce overlap.")

    if edge_mode == EDGE_SHRINK:
        return _shrink_grid(left, top, usable_width, usable_height,
                            tile_width, tile_height, effective_width, effective_height)

    if edge_mode == EDGE_SHIFT:
        return _shift_grid(image_width, image_height, left, top, usable_width, usable_height,
                           tile_width, tile_height, overlap_width, overlap_height,
                           effective_width, effective_height)

    return _skip_grid(image_width, image_height, left, top, usable_width, usable_height,
                      tile_width, tile_height, overlap_width, overlap_height,
                      effective_width, effective_height)


def _skip_grid(image_width, image_height, left, top, usable_width, usable_height,
               tile_width, tile_height, overlap_width, overlap_height,
               effective_width, effective_height):
    """Regular grid; tiles that would cross the usable area are dropped."""
    num_tiles_x = max(1, int((usable_width - overlap_width) / effective_width) + 1)
    num_tiles_y = max(1, int((usable_height - overlap_height) / effective_height) + 1)

    rects = []
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            x = left + j * effective_width
            y = top + i * effective_height

            # Skip tiles that would exceed the usable area OR image boundaries
            if (x + tile_width > left + usable_width or y + tile_height > top + usable_height or
                    x + tile_width > image_width or y + tile_height > image_height):
                continue

            rects.append((x, y, tile_width, tile_height))

    return TileGrid(rects=rects, cols=num_tiles_x, rows=num_tiles_y)


def _shift_grid(image_width, image_height, left, top, usable_width, usable_height,
                tile_width, tile_height, overlap_width, overlap_height,
                effective_width, effective_height):
    """Regular grid plus a right column / bottom row slid inward for full coverage."""
    num_tiles_x = max(1, int(usable_width / effective_width))
    num_tiles_y = max(1, int(usable_height / effective_height))

    # Area the regular grid covers
    covered_width = num_tiles_x * effective_width + overlap_width
    covered_height = num_tiles_y * effective_height + overlap_height

    rects = []
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            x = left + j * effective_width
            y = top + i * effective_height

            # Ensure tile doesn't exceed image boundaries
            if x + tile_width > image_width or y + tile_height > image_height:
                continue

            rects.append((x, y, tile_width, tile_height))

    # Positions of the extra column / row, aligned to the far edge and kept inside the image
    right_edge = min(left + usable_width - tile_width, image_width - tile_width)
    bottom_edge = min(top + usable_height - tile_height, image_height - tile_height)
    right_edge_valid = right_edge >= left and right_edge + tile_width <= image_width
    bottom_edge_valid = bottom_edge >= top and bottom_edge + tile_height <= image_height

    # Extra column at the right edge
    if covered_width < usable_width and right_edge_valid:
        for i in range(num_tiles_y):
            y = top + i * effective_height
            if y + tile_height > image_height:
                continue
            rects.append((right_edge, y, tile_width, tile_height))

    # Extra row at the bottom edge, then the corner tile when both edges needed one
    if covered_height < usable_height and bottom_edge_valid:
        for j in range(num_tiles_x):
            x = left + j * effective_width
            if x + tile_width > image_width:
                continue
            rects.append((x, bottom_edge, tile_width, tile_height))

        if covered_width < usable_width and right_edge_valid:
            rects.append((right_edge, bottom_edge, tile_width, tile_height))

    return TileGrid(rects=rects, cols=num_tiles_x, rows=num_tiles_y)


def _shrink_grid(left, top, usable_width, usable_height,
                 tile_width, tile_height, effective_width, effective_height):
    """Regular grid whose last column / row is cut to end exactly at the far edge."""
    columns = _shrink_spans(left, usable_width, tile_width, effective_width)
    rows = _shrink_spans(top, usable_height, tile_height, effective_height)

    rects = [(x, y, w, h) for y, h in rows for x, w in columns]
    return TileGrid(rects=rects, cols=len(columns), rows=len(rows))


def _shrink_spans(start, length, tile, step):
    """(position, size) spans along one axis, stopping at the first that reaches the end.

    Stopping there matters with overlap: once a tile reaches the far edge, the
    next step would only produce a sliver already inside it.
    """
    end = start + length
    spans = []
    position = start
    while True:
        spans.append((position, min(tile, end - position)))
        if position + tile >= end:
            break
        position += step
    return spans
