"""
MousePositionBridge — bridges AnnotationWindow mouse-move events into the MVAT
controller, building camera rays and updating context-canvas markers.
"""
import time

import numpy as np

from PyQt5.QtCore import QObject

from coralnet_toolbox.MVAT.core.Ray import CameraRay
from coralnet_toolbox.MVAT.core.constants import (
    RAY_COLOR_SELECTED,
    RAY_COLOR_HIGHLIGHTED,
    RAY_COLOR_INVALID,
)


# -------------------------------------------------------------------------------------
# Classes
# -------------------------------------------------------------------------------------


class MousePositionBridge(QObject):
    """
    Bridges mouse position events from the AnnotationWindow to the MVAT
    controller.

    Responsibilities:
    - Build a 3D ray from a selected camera and a 2D image pixel (uses depth
        from a z-channel when available, otherwise falls back to a scene
        median/default depth).
    - Create corresponding rays from highlighted cameras to the same world
        point and choose colors based on depth accuracy.
    - Project the selected ray into other camera image spaces and forward
        marker/visibility updates to the ContextMatrix for UI presentation.

    This class mirrors the behavior previously implemented on the window
    layer but is now manager-owned so it can operate without direct UI
    responsibilities.
    """
    def __init__(self, manager: 'MVATManager'):
        super().__init__()
        self.manager = manager
        self.enabled = True
        self._last_update_time = 0
        self._last_mouse_x = -1
        self._last_mouse_y = -1

    def on_mouse_moved(self, x: int, y: int):
        if not self.enabled:
            return
        # Skip duplicate pixels (Qt can fire multiple events for the same position)
        if x == self._last_mouse_x and y == self._last_mouse_y:
            return
        # Time-gate: cap at ~60 fps to avoid flooding the ortho ray-trace / perspective
        # projection loop on every raw mouse event.
        now = time.monotonic()
        if now - self._last_update_time < 0.016:
            return
        self._last_update_time = now
        self._last_mouse_x = x
        self._last_mouse_y = y
        self._process_pending_position(x, y)
            
    def _process_pending_position(self, x: int, y: int):
        # --- Ortho path: route to ortho handler when OrthoRaster is displayed ---
        ortho_camera = self.manager.ortho_camera
        if ortho_camera is not None:
            current_path = getattr(self.manager.annotation_window, 'current_image_path', None)
            if current_path == ortho_camera.image_path:
                self.manager.viewer.clear_ray()
                self._process_ortho_position(x, y, ortho_camera)
                return

        self.manager.viewer.clear_ortho_ray()

        camera = self.manager.selected_camera
        if camera is None or not (0 <= x < camera.width and 0 <= y < camera.height):
            self.clear_all_markers()
            self.manager.viewer.clear_ray()
            return

        visible_cameras = self.manager._get_visible_context_cameras()
        visible_paths = {cam.image_path for cam in visible_cameras}
        if camera.image_path not in visible_paths:
            self.clear_all_markers()
            self.manager.viewer.clear_ray()
            return

        primary_target = self.manager.viewer.scene_context.get_primary_target()

        # --- Path A: Index Map (preferred) ---
        ray = None
        candidate_id = -1
        index_map = camera._raster.index_map
        if index_map is not None:
            candidate_id = int(index_map[y, x])

        if candidate_id > -1 and primary_target is not None:
            coord = primary_target.get_element_coordinate(candidate_id)
            if coord is not None:
                # ---> Perspective logic <---
                origin = camera.position.copy()
                direction = coord - origin
                norm = np.linalg.norm(direction)
                direction = direction / norm if norm > 0 else camera.R.T @ np.array([0, 0, 1])
                
                ray = CameraRay(
                    origin=origin,
                    direction=direction,
                    terminal_point=coord,
                    has_accurate_depth=True,
                    pixel_coord=(x, y),
                    source_camera=camera,
                    element_id=candidate_id
                )

        # --- Path B: Z-channel / depth fallback ---
        if ray is None:
            raster = camera._raster
            depth = None
            z_data_type = raster.z_data_type if hasattr(raster, 'z_data_type') else None
            
            if raster.z_channel is not None:
                z_value = raster.get_z_value(x, y)
                # For depth maps, only accept positive values
                # For elevation maps, accept any value (including negative)
                if z_value is not None:
                    if z_data_type == 'elevation' or (z_data_type == 'depth' and z_value > 0) or z_data_type is None:
                        depth = z_value
            
            if depth is None or np.isnan(depth):
                default_depth = self.manager.viewer.get_scene_median_depth(camera.position)
            else:
                default_depth = depth
            
            ray = CameraRay.from_pixel_and_camera(
                pixel_xy=(x, y),
                camera=camera,
                depth=depth,
                default_depth=default_depth,
            )

        primary_ray_valid = ray.has_accurate_depth or ray.element_id > -1
        rays_with_colors = [(ray, RAY_COLOR_SELECTED if primary_ray_valid else RAY_COLOR_INVALID)]
        
        # --- Short-Circuit Invalid Primary Rays ---
        # If the primary ray did not hit real scene geometry, skip secondary rays entirely.
        if not primary_ray_valid:
            # Primary ray is invalid: keep only the selected-camera ray, clear markers, and return.
            self.manager.viewer.show_rays(rays_with_colors)
            self.clear_all_markers()
            return
        
        # --- Primary ray is valid: proceed with secondary rays ---
        visibility_status = {}
        accuracies = {camera.image_path: ray.has_accurate_depth}
        highlighted_cameras = visible_cameras

        for target_cam in highlighted_cameras:
            if target_cam.image_path == camera.image_path:
                continue

            # Project primary terminal point into this camera
            proj = target_cam.project(ray.terminal_point)
            u_proj = int(round(float(proj[0]))) if not np.isnan(proj[0]) else -1
            v_proj = int(round(float(proj[1]))) if not np.isnan(proj[1]) else -1
            in_bounds = (
                not np.isnan(proj[0])
                and 0 <= u_proj < target_cam.width
                and 0 <= v_proj < target_cam.height
            )

            # If the projected point is out of bounds, skip this camera entirely
            # (no secondary ray should be drawn into empty space outside the image)
            if not in_bounds:
                continue

            target_terminal = ray.terminal_point
            ray_color = RAY_COLOR_INVALID
            is_occluded = True
            found_id = -1

            if getattr(target_cam, '_raster', None) is not None and target_cam._raster.index_map is not None and ray.element_id > -1:
                found_id = int(target_cam._raster.index_map[v_proj, u_proj])

                # Determine visibility with spatial tolerance.
                # METHOD A: 3D Distance Threshold (ACTIVE)
                # Resolves false-positive occlusions caused by sub-pixel rounding when
                # projecting the primary 3D point into the secondary camera.  A found_id
                # that differs from primary_element_id may still belong to the same
                # physical surface patch; we accept it as visible when its 3D centre is
                # within 5% of the camera-to-point distance.
                is_visible = False
                if found_id == ray.element_id:
                    is_visible = True
                elif found_id > -1 and primary_target is not None:
                    found_coord = primary_target.get_element_coordinate(found_id)
                    if found_coord is not None:
                        surface_dist = np.linalg.norm(found_coord - ray.terminal_point)
                        cam_to_point_dist = np.linalg.norm(target_cam.position - ray.terminal_point)
                        tolerance = 0.05 * cam_to_point_dist
                        if surface_dist <= tolerance:
                            is_visible = True

                # METHOD B: 2D Neighbourhood Search (COMMENTED OUT — for comparison testing)
                # Checks whether primary_element_id appears anywhere in a 5×5 pixel window
                # around the projected pixel, catching cases where aliasing shifts the hit
                # by 1-2 pixels.
                # -----------------------------------------------------------------------
                # HALF = 2  # half-width of the search window (full window = 2*HALF+1)
                # v_lo = max(0, v_proj - HALF)
                # v_hi = min(target_cam.height, v_proj + HALF + 1)
                # u_lo = max(0, u_proj - HALF)
                # u_hi = min(target_cam.width,  u_proj + HALF + 1)
                # neighbourhood = target_cam.index_map[v_lo:v_hi, u_lo:u_hi]
                # is_visible = int(ray.element_id) in neighbourhood
                # -----------------------------------------------------------------------

                if is_visible:
                    target_terminal = ray.terminal_point
                    is_occluded = False
                    accuracies[target_cam.image_path] = True

                elif found_id > -1:                                 # TRUE OCCLUSION
                    occluder = primary_target.get_element_coordinate(found_id) if primary_target else None
                    target_terminal = occluder if occluder is not None else ray.terminal_point
                    accuracies[target_cam.image_path] = False

                else:                                               # BACKGROUND (-1)
                    accuracies[target_cam.image_path] = False

            else:
                # Legacy fallback: depth-based occlusion test
                is_occluded = target_cam.is_point_occluded_depth_based(
                    ray.terminal_point, depth_threshold=0.15)
                accuracies[target_cam.image_path] = target_cam._raster.z_channel is not None

            visibility_status[target_cam.image_path] = is_occluded
            ray_color = RAY_COLOR_HIGHLIGHTED if (accuracies.get(target_cam.image_path, False) and not is_occluded) else RAY_COLOR_INVALID

            # Build secondary ray directly
            t_origin = target_cam.position.copy()
            t_direction = target_terminal - t_origin
            t_norm = np.linalg.norm(t_direction)
            t_direction = t_direction / t_norm if t_norm > 0 else target_cam.R.T @ np.array([0, 0, 1])
            is_ray_accurate = accuracies.get(target_cam.image_path, False) and not is_occluded
            target_ray = CameraRay(
                origin=t_origin,
                direction=t_direction,
                terminal_point=target_terminal,
                has_accurate_depth=is_ray_accurate,
                source_camera=target_cam,
                element_id=found_id,
            )
            rays_with_colors.append((target_ray, ray_color))

        self.manager.viewer.show_rays(rays_with_colors)
        # Project only into visible cameras — avoids O(all_cameras) work per frame
        visible_cam_dict = {cam.image_path: cam for cam in highlighted_cameras}
        visible_cam_dict[camera.image_path] = camera  # include the primary camera
        projections = ray.project_to_cameras(visible_cam_dict)

        # Update context matrix canvases (Phase 4)
        if self.manager.context_matrix is not None:
            try:
                self.manager.context_matrix.update_dynamic_markers(
                    projections, accuracies, visibility_status
                )
            except Exception:
                self.manager.context_matrix.clear_all_dynamic_markers()
                
    def _process_ortho_position(self, x: int, y: int, ortho_camera):
        """
        Handle mouse-move events when the AnnotationWindow is showing an OrthoRaster.

        Resolves the 3D world point via z-channel lookup (O(1)) instead of mesh
        ray tracing.  The z-channel is stored at full ortho resolution so pixel
        coords map directly; the raw CRS elevation is fed into geo_to_world.
        """

        if not (0 <= x < ortho_camera.width and 0 <= y < ortho_camera.height):
            self.manager.viewer.clear_ortho_ray()
            self.clear_all_markers()
            return

        # Geo XY from affine transform
        X, Y = ortho_camera.pixel_to_geo(x, y)

        # CRS elevation from z-channel (stored at full ortho resolution, raw units)
        Z = ortho_camera._raster.get_z_value(x, y)
        if Z is None:
            self.manager.viewer.clear_ortho_ray()
            self.clear_all_markers()
            return

        world_pt = ortho_camera.geo_to_world(X, Y, Z)

        try:
            self.manager.viewer.show_ortho_ray(
                world_pt,
                ortho_camera.get_vertical_direction_world(),
            )
        except Exception:
            pass

        # Project to all visible context cameras and build marker dicts
        projections = {}
        accuracies = {}
        visibility_status = {}

        for cam in self.manager._get_visible_context_cameras():
            try:
                proj = cam.project(world_pt)
                if np.isnan(proj).any():
                    continue
                u = float(proj[0])
                v = float(proj[1])
                in_bounds = 0 <= u < cam.width and 0 <= v < cam.height
                projections[cam.image_path] = (u, v, in_bounds)
                accuracies[cam.image_path] = True
                visibility_status[cam.image_path] = False
            except Exception:
                pass

        if self.manager.context_matrix is not None:
            try:
                self.manager.context_matrix.update_dynamic_markers(
                    projections, accuracies, visibility_status
                )
            except Exception:
                self.manager.context_matrix.clear_all_dynamic_markers()

    def clear_all_markers(self):
        if self.manager.context_matrix is not None:
            try:
                self.manager.context_matrix.clear_all_dynamic_markers()
            except Exception:
                pass

    def cleanup(self):
        self.clear_all_markers()
