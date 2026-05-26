"""
Batched Frustum Manager for MVAT.

Manages batched rendering of camera frustums for efficient 3D visualization.
Extracted from core/Frustum.py — geometry lives in core/Frustum.Frustum;
coordination and batching lives here.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv

from coralnet_toolbox.MVAT.core.constants import (
    STATE_DEFAULT,
    STATE_HIGHLIGHTED,
    STATE_SELECTED,
    STATE_HOVER,
    STATE_COLORS,
)


def _rgb_to_hex(rgb):
    """Convert RGB tuple (0-1) to hex string."""
    r, g, b = [int(c * 255) for c in rgb]
    return f'#{r:02x}{g:02x}{b:02x}'


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class BatchedFrustumManager:
    """
    Manages batched rendering of camera frustums for efficient 3D visualization.

    Instead of creating individual actors per camera (O(N) draw calls), this class
    merges all frustum wireframes into a single PolyData mesh with scalar-based
    coloring, reducing draw calls to O(1).

    Selection and highlight states are managed via scalar arrays that map to
    a discrete color lookup table. Updating state only requires modifying the
    scalar array and calling mesh.Modified(), avoiding expensive actor property changes.

    Attributes:
        cameras: Dict mapping image_path -> Camera object
        camera_indices: Dict mapping image_path -> index in merged mesh
        merged_wireframe: Combined PyVista PolyData for all wireframes
        wireframe_actor: Single actor for the merged wireframe mesh
        point_counts: List of point counts per camera (for scalar array indexing)
    """

    def __init__(self):
        """Initialize the BatchedFrustumManager."""
        self.cameras: Dict[str, 'Camera'] = {}
        self.camera_indices: Dict[str, int] = {}
        self.camera_paths: List[str] = []  # Ordered list of camera paths

        # Merged geometry
        self.merged_wireframe: Optional[pv.PolyData] = None
        self.wireframe_actor = None

        # Track point ranges for each camera in the merged mesh
        self.point_ranges: List[Tuple[int, int]] = []  # (start_idx, end_idx) for each camera

        # Current scale for geometry generation
        self._current_scale = None

    def build_frustum_batch(self,
                            cameras: Dict[str, 'Camera'],
                            scale: float = 0.1) -> Optional[pv.PolyData]:
        """
        Build a merged PolyData mesh from all camera frustums.

        Args:
            cameras: Dict mapping image_path -> Camera object
            scale: Scale factor for frustum size

        Returns:
            pv.PolyData: Merged wireframe mesh with 'state' scalar array, or None if empty
        """
        if not cameras:
            print(f"   🐛 build_frustum_batch: No cameras provided")
            return None

        print(f"   🐛 build_frustum_batch: Processing {len(cameras)} cameras")

        self.cameras = cameras
        self.camera_indices.clear()
        self.camera_paths.clear()
        self.point_ranges.clear()
        self._current_scale = scale

        meshes = []
        point_offset = 0

        for idx, (path, camera) in enumerate(cameras.items()):

            self.camera_indices[path] = idx
            self.camera_paths.append(path)

            # Get wireframe mesh from frustum (geometry only, no actor creation)
            mesh = camera.frustum.get_mesh(scale)

            if mesh is not None:
                wireframe_mesh = self._frustum_to_wireframe_polydata(camera, scale)

                if wireframe_mesh is not None:
                    n_points = wireframe_mesh.n_points
                    self.point_ranges.append((point_offset, point_offset + n_points))
                    point_offset += n_points
                    meshes.append(wireframe_mesh)
                    print(f"   🐛   Camera {idx}: Created wireframe ({n_points} points)")
                else:
                    print(f"   🐛   Camera {idx}: _frustum_to_wireframe_polydata returned None")
                    self.point_ranges.append((point_offset, point_offset))
            else:
                print(f"   🐛   Camera {idx}: camera.frustum.get_mesh returned None")
                self.point_ranges.append((point_offset, point_offset))

        if not meshes:
            print(f"   🐛 build_frustum_batch: No meshes generated - returning None")
            self.merged_wireframe = None
            return None

        # Merge all wireframe meshes into one
        if len(meshes) == 1:
            self.merged_wireframe = meshes[0]
        else:
            self.merged_wireframe = pv.merge(meshes)

        # Initialize state array (all default/white)
        n_points = self.merged_wireframe.n_points
        self.merged_wireframe['state'] = np.zeros(n_points, dtype=np.uint8)

        return self.merged_wireframe

    def _frustum_to_wireframe_polydata(self, camera: 'Camera', scale: float) -> Optional[pv.PolyData]:
        """
        Convert a camera's frustum to wireframe PolyData (lines only).

        Creates line segments for the frustum edges:
        - 4 lines for the near plane quad
        - 4 lines from center to corners

        Args:
            camera: Camera object with frustum geometry
            scale: Scale factor for frustum size

        Returns:
            pv.PolyData: Wireframe as line segments
        """
        w, h = camera.width, camera.height

        corners_pix = np.array([
            [0, 0, 1],   # Top-Left (0)
            [w, 0, 1],   # Top-Right (1)
            [w, h, 1],   # Bottom-Right (2)
            [0, h, 1]    # Bottom-Left (3)
        ])

        # Unproject to camera space
        frustum_points_cam = scale * (camera.K_inv @ corners_pix.T).T

        # Add camera center
        all_points_cam = np.vstack([frustum_points_cam, [0, 0, 0]])  # 5 points

        # Transform to world coordinates
        R_inv = camera.R.T
        t_vec = camera.t.reshape(3, 1)
        all_points_world = (R_inv @ (all_points_cam.T - t_vec)).T

        # Define line segments (VTK format: [n_pts, idx0, idx1, ...])
        lines = np.array([
            2, 0, 1,  # Top edge
            2, 1, 2,  # Right edge
            2, 2, 3,  # Bottom edge
            2, 3, 0,  # Left edge
            2, 4, 0,  # Center to TL
            2, 4, 1,  # Center to TR
            2, 4, 2,  # Center to BR
            2, 4, 3,  # Center to BL
        ])

        return pv.PolyData(all_points_world, lines=lines)

    def add_to_plotter(self,
                       plotter,
                       line_width: float = 1.5) -> Optional['vtkActor']:
        """
        Add the merged wireframe mesh to a PyVista plotter.

        Args:
            plotter: PyVista plotter instance
            line_width: Width of wireframe lines

        Returns:
            vtkActor: The wireframe actor, or None if no geometry
        """
        if self.merged_wireframe is None:
            return None

        if self.wireframe_actor is not None:
            try:
                plotter.remove_actor(self.wireframe_actor)
            except:
                pass

        cmap = [_rgb_to_hex(STATE_COLORS[STATE_DEFAULT]),
                _rgb_to_hex(STATE_COLORS[STATE_HIGHLIGHTED]),
                _rgb_to_hex(STATE_COLORS[STATE_SELECTED]),
                _rgb_to_hex(STATE_COLORS[STATE_HOVER])]

        self.wireframe_actor = plotter.add_mesh(
            self.merged_wireframe,
            scalars='state',
            cmap=cmap,
            clim=[0, 3],
            show_scalar_bar=False,
            line_width=line_width,
            render_lines_as_tubes=False,
            style='wireframe',
            name='_batched_frustums',
            reset_camera=False
        )
        self.wireframe_actor.SetPickable(False)

        return self.wireframe_actor

    def update_camera_state(self, path: str, state: int):
        """
        Update the visual state of a single camera's frustum.

        Args:
            path: Image path of the camera
            state: STATE_DEFAULT (0), STATE_HIGHLIGHTED (1), or STATE_SELECTED (2)
        """
        if self.merged_wireframe is None:
            return
        if path not in self.camera_indices:
            return

        idx = self.camera_indices[path]
        if idx >= len(self.point_ranges):
            return

        start, end = self.point_ranges[idx]
        if start < end:
            self.merged_wireframe['state'][start:end] = state

    def update_camera_states(self,
                             selected_path: Optional[str] = None,
                             highlighted_paths: Optional[List[str]] = None,
                             hovered_path: Optional[str] = None,
                             context_highlighted_paths: Optional[List[str]] = None):
        """
        Batch update visual states for all cameras.

        Args:
            selected_path: Path of the selected camera (lime green)
            highlighted_paths: List of highlighted camera paths (cyan)
            hovered_path: Path of the hovered camera (red, with Ctrl)
            context_highlighted_paths: List of matrix-visible camera paths that
                should render cyan while the selected camera remains green.
        """
        if self.merged_wireframe is None:
            return

        highlighted_paths = highlighted_paths or []
        context_highlighted_paths = context_highlighted_paths or []

        # Reset all to default
        self.merged_wireframe['state'][:] = STATE_DEFAULT

        cyan_paths = []
        seen_paths = set()
        for path in list(highlighted_paths) + list(context_highlighted_paths):
            if path not in seen_paths:
                cyan_paths.append(path)
                seen_paths.add(path)

        for path in cyan_paths:
            if path in self.camera_indices and path != hovered_path and path != selected_path:
                idx = self.camera_indices[path]
                if idx < len(self.point_ranges):
                    start, end = self.point_ranges[idx]
                    if start < end:
                        self.merged_wireframe['state'][start:end] = STATE_HIGHLIGHTED

        if selected_path and selected_path in self.camera_indices and selected_path != hovered_path:
            idx = self.camera_indices[selected_path]
            if idx < len(self.point_ranges):
                start, end = self.point_ranges[idx]
                if start < end:
                    self.merged_wireframe['state'][start:end] = STATE_SELECTED

        if hovered_path and hovered_path in self.camera_indices:
            idx = self.camera_indices[hovered_path]
            if idx < len(self.point_ranges):
                start, end = self.point_ranges[idx]
                if start < end:
                    self.merged_wireframe['state'][start:end] = STATE_HOVER

    def mark_modified(self):
        """Mark the merged mesh as modified to trigger re-render."""
        if self.merged_wireframe is not None:
            self.merged_wireframe.Modified()

    def set_visibility(self, visible: bool):
        """Set visibility of the batched wireframe actor."""
        if self.wireframe_actor is not None:
            self.wireframe_actor.SetVisibility(visible)

    def remove_from_plotter(self, plotter):
        """Remove the wireframe actor from the plotter."""
        if self.wireframe_actor is not None:
            try:
                plotter.remove_actor(self.wireframe_actor)
            except:
                pass
            self.wireframe_actor = None

    def clear(self):
        """Clear all cached data."""
        self.cameras.clear()
        self.camera_indices.clear()
        self.camera_paths.clear()
        self.point_ranges.clear()
        self.merged_wireframe = None
        self.wireframe_actor = None
        self._current_scale = None
