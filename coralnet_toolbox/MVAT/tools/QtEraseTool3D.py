"""
Erase3DTool — erases labels from mesh faces (writes class_id = 0 / background).

Extends Brush3DTool, mirroring exactly how Tools/QtEraseTool.EraseTool extends
Tools/QtBrushTool.BrushTool in 2D.  The only semantic difference is that
_resolve_label() always returns class_id=0 with a white color, so the painter
thread erases rather than paints regardless of which label is selected.
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
    Erases labels from mesh faces within the brush radius.

    Inherits the full stroke / KD-tree / preview sphere / projection machinery
    from Brush3DTool.  Only the visual style (red wireframe sphere) and the
    label resolution (always background) differ.

    Mirrors EraseTool._on_math_finished which hardcodes class_id = 0, and
    EraseTool.create_cursor_preview_item which renders a red dashed outline.
    """

    # Red wireframe to match the 2D EraseTool's dashed-red cursor convention.
    _PREVIEW_COLOR   = 'red'
    _PREVIEW_OPACITY = 0.45

    def _resolve_label(self, label):
        """
        Eraser always writes class_id = 0 (background) with white color,
        mirroring EraseTool._on_math_finished which hardcodes class_id = 0.
        The `label` argument is intentionally ignored.
        """
        return 0, (255, 255, 255)
