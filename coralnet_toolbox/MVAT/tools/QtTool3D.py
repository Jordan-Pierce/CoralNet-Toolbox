"""
Base class for 3D mesh interaction tools in the MVATViewer.

Mirrors the structure of Tools/QtTool.py but operates in VTK viewport space:
  - Event handlers receive (event, face_id, world_pos) alongside the raw QMouseEvent.
  - No Qt scene, annotation window, or 2D crosshair concepts.
  - activate() / deactivate() are managed by MVATManager.set_selected_3d_tool(),
    exactly mirroring how AnnotationWindow.set_selected_tool() manages 2D tools.

Naming mirrors Tools/QtTool.py intentionally so the two hierarchies are easy
to read side-by-side.
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Tool3D:
    """
    Abstract base class for all 3D mesh interaction tools.

    Analogous to Tools/QtTool.Tool but designed for the MVATViewer (PyVista / VTK).
    Subclasses override the mouse/wheel event methods and activate / deactivate
    to set up / tear down any VTK actors used for visual feedback (e.g. a
    preview sphere).

    Attributes:
        mvat_viewer:  The MVATViewer widget (owns the PyVista plotter and scene).
        mvat_manager: The MVATManager (owns labels, cameras, multi-annotate state).
        active (bool): True while this tool is the selected_3d_tool.
                       Mirrors Tool.active.
    """

    def __init__(self, mvat_viewer, mvat_manager):
        """
        Args:
            mvat_viewer:  MVATViewer instance — provides plotter, scene_context,
                          and set_active_3d_tool().
            mvat_manager: MVATManager instance — provides annotation_window,
                          cameras, multi_annotate_enabled, and the label painter.
        """
        self.mvat_viewer = mvat_viewer
        self.mvat_manager = mvat_manager

        self.active = False

    # ------------------------------------------------------------------
    # Lifecycle  (mirrors Tool.activate / Tool.deactivate)
    # ------------------------------------------------------------------

    def activate(self):
        """
        Activate this tool.
        Called by MVATManager.set_selected_3d_tool() after deactivating the
        previous tool — mirrors AnnotationWindow.set_selected_tool() calling
        tool.activate().
        """
        self.active = True

    def deactivate(self):
        """
        Deactivate this tool and clean up any VTK actors / state.
        Called by MVATManager.set_selected_3d_tool() before switching to a
        different tool — mirrors AnnotationWindow.set_selected_tool() calling
        previous_tool.deactivate().
        """
        self.active = False
        self.stop_current_drawing()

    def stop_current_drawing(self):
        """
        Force-stop any in-progress drawing / stroke operation.
        Subclasses should override this to commit or discard the current stroke.
        Mirrors Tool.stop_current_drawing().
        """
        pass

    # ------------------------------------------------------------------
    # Event handlers — called by MVATViewer.eventFilter when this tool is
    # the active_3d_tool.  Signatures differ from the 2D Tool equivalents
    # because VTK picking enriches each event with a resolved face_id and
    # world_pos before it reaches the tool.
    # ------------------------------------------------------------------

    def mousePressEvent(self, event, face_id: int, world_pos):
        """
        Handle a left-button press on the 3D viewport.

        Args:
            event:     The original QMouseEvent forwarded from eventFilter.
            face_id:   VTK cell ID of the mesh face under the cursor, or -1 if
                       the cursor is over empty space.
            world_pos: np.ndarray (3,) world-space coordinate of the pick, or
                       None when face_id == -1.
        """
        pass

    def mouseMoveEvent(self, event, face_id: int, world_pos):
        """
        Handle a mouse-move event in the 3D viewport.

        Args:
            event:     The original QMouseEvent forwarded from eventFilter.
            face_id:   VTK cell ID under the cursor, or -1.
            world_pos: np.ndarray (3,) world coordinate, or None.
        """
        pass

    def mouseReleaseEvent(self, event):
        """
        Handle a left-button release in the 3D viewport.

        Args:
            event: The original QMouseEvent forwarded from eventFilter.
        """
        pass

    def wheelEvent(self, event, delta_y: int):
        """
        Handle a Ctrl+wheel event forwarded from MVATViewer.eventFilter.
        Typically used to resize the brush radius.

        Args:
            event:   The original QWheelEvent.
            delta_y: angleDelta().y() (positive = scroll up / zoom in).
        """
        pass
