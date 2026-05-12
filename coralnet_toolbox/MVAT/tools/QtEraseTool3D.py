"""
Erase3DTool — erase-specific 3D tool logic built on top of Brush3DTool.

The shared preview sphere, hover batching, and label-colored highlight are
owned by Tool3D.  Erase3DTool only changes the preview fallback color and the
label resolution so the brush plumbing would erase rather than paint.
"""

import warnings

import numpy as np

from coralnet_toolbox.MVAT.tools.QtBrushTool3D import Brush3DTool

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Erase3DTool(Brush3DTool):
    """
    Previews the erase tool within the brush radius.

    Inherits the full stroke / preview sphere / projection machinery
    from Brush3DTool.  Only the fallback preview color and the label
    resolution (always background) differ.

    Mirrors EraseTool._on_math_finished which hardcodes class_id = 0, and
    EraseTool.create_cursor_preview_item which renders a red dashed outline.
    """

    # Red wireframe to match the 2D EraseTool's dashed-red cursor convention.
    _PREVIEW_COLOR   = 'red'
    _PREVIEW_OPACITY = 0.45

    def _use_active_label_preview_color(self):
        return False

    def _resolve_label(self, label):
        """
        Eraser always writes class_id = 0 (background) with white color,
        mirroring EraseTool._on_math_finished which hardcodes class_id = 0.
        The `label` argument is intentionally ignored.
        """
        return 0, (255, 255, 255)
