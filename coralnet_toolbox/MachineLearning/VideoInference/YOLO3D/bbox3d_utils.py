import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
from collections import defaultdict


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------


# Default camera intrinsic matrix (can be overridden)
DEFAULT_K = np.array([
    [718.856, 0.0, 607.1928],
    [0.0, 718.856, 185.2157],
    [0.0, 0.0, 1.0]
])

# Default camera projection matrix (can be overridden)
DEFAULT_P = np.array([
    [718.856, 0.0, 607.1928, 45.38225],
    [0.0, 718.856, 185.2157, -0.1130887],
    [0.0, 0.0, 1.0, 0.003779761]
])


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class BBox3DEstimator:
    """
    3D bounding box estimation from 2D detections and depth
    """
    def __init__(self, camera_matrix=None, projection_matrix=None):
        """
        Initialize the 3D bounding box estimator
        
        Args:
            camera_matrix (numpy.ndarray): Camera intrinsic matrix (3x3)
            projection_matrix (numpy.ndarray): Camera projection matrix (3x4)
        """
        self.K = camera_matrix if camera_matrix is not None else DEFAULT_K
        self.P = projection_matrix if projection_matrix is not None else DEFAULT_P
        
        # Initialize Kalman filters for tracking 3D boxes
        self.kf_trackers = {}
        
        # Store history of 3D boxes for filtering
        self.box_history = defaultdict(list)
        self.max_history = 5
    
    def estimate_3d_box(self, bbox_2d, depth_value, class_name=None, object_id=None):
        """
        Estimate 3D bounding box from 2D bounding box and depth
        
        Args:
            bbox_2d (list): 2D bounding box [x1, y1, x2, y2]
            depth_value (float): Depth value at the center of the bounding box
            class_name (str): Class name of the object (optional, not used for dimensions)
            object_id (int): Object ID for tracking (None for no tracking)
            
        Returns:
            dict: 3D bounding box parameters
        """
        # Get 2D box center
        x1, y1, x2, y2 = bbox_2d
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Use default dimensions (height, width, length)
        default_dim = 1.0
        dimensions = [default_dim, default_dim, default_dim]
        
        # Convert depth to distance - use a larger range for better visualization
        distance = 1.0 + depth_value * 9.0
        
        # Calculate 3D location
        location = self._backproject_point(center_x, center_y, distance)
        
        # Estimate orientation
        orientation = self._estimate_orientation(bbox_2d, location, class_name)
        
        # Create 3D box
        box_3d = {
            'dimensions': dimensions,
            'location': location,
            'orientation': orientation,
            'bbox_2d': bbox_2d,
            'object_id': object_id,
            'class_name': class_name
        }
        
        # Apply Kalman filtering if tracking is enabled
        if object_id is not None:
            box_3d = self._apply_kalman_filter(box_3d, object_id)
            
            # Add to history for temporal filtering
            self.box_history[object_id].append(box_3d)
            if len(self.box_history[object_id]) > self.max_history:
                self.box_history[object_id].pop(0)
            
            # Apply temporal filtering
            box_3d = self._apply_temporal_filter(object_id)
        
        return box_3d
    
    def _backproject_point(self, x, y, depth):
        """
        Backproject a 2D point to 3D space
        
        Args:
            x (float): X coordinate in image space
            y (float): Y coordinate in image space
            depth (float): Depth value
            
        Returns:
            numpy.ndarray: 3D point (x, y, z) in camera coordinates
        """
        # Create homogeneous coordinates
        point_2d = np.array([x, y, 1.0])
        
        # Backproject to 3D
        # The z-coordinate is the depth
        # The x and y coordinates are calculated using the inverse of the camera matrix
        point_3d = np.linalg.inv(self.K) @ point_2d * depth
        
        # For indoor scenes, adjust the y-coordinate to be more realistic
        # In camera coordinates, y is typically pointing down
        # Adjust y to place objects at a reasonable height
        # This is a simplification - in a real system, this would be more sophisticated
        point_3d[1] = point_3d[1] * 0.5  # Scale down y-coordinate
        
        return point_3d
    
    def _estimate_orientation(self, bbox_2d, location, class_name):
        """
        Estimate orientation of the object
        
        Args:
            bbox_2d (list): 2D bounding box [x1, y1, x2, y2]
            location (numpy.ndarray): 3D location of the object
            class_name (str): Class name of the object
            
        Returns:
            float: Orientation angle in radians
        """
        # Calculate ray from camera to object center
        theta_ray = np.arctan2(location[0], location[2])
        
        # For other objects, use the 2D box aspect ratio to estimate orientation
        x1, y1, x2, y2 = bbox_2d
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 1.0
        
        # If the object is wide, it might be facing sideways
        if aspect_ratio > 1.5:
            # Object is wide, might be facing sideways
            # Use the position relative to the image center to guess orientation
            image_center_x = self.K[0, 2]  # Principal point x
            if (x1 + x2) / 2 < image_center_x:
                # Object is on the left side of the image
                alpha = np.pi / 2  # Facing right
            else:
                # Object is on the right side of the image
                alpha = -np.pi / 2  # Facing left
        else:
            # Object has normal proportions, assume it's facing the camera
            alpha = 0.0
        
        # Global orientation
        rot_y = alpha + theta_ray
        
        return rot_y
    
    def _init_kalman_filter(self, box_3d):
        """
        Initialize a Kalman filter for a new object
        
        Args:
            box_3d (dict): 3D bounding box parameters
            
        Returns:
            filterpy.kalman.KalmanFilter: Initialized Kalman filter
        """
        # State: [x, y, z, width, height, length, yaw, vx, vy, vz, vyaw]
        kf = KalmanFilter(dim_x=11, dim_z=7)
        
        # Initial state
        kf.x = np.array([
            box_3d['location'][0],
            box_3d['location'][1],
            box_3d['location'][2],
            box_3d['dimensions'][1],  # width
            box_3d['dimensions'][0],  # height
            box_3d['dimensions'][2],  # length
            box_3d['orientation'],
            0, 0, 0, 0  # Initial velocities
        ])
        
        # State transition matrix (motion model)
        dt = 1.0  # Time step
        kf.F = np.eye(11)
        kf.F[0, 7] = dt  # x += vx * dt
        kf.F[1, 8] = dt  # y += vy * dt
        kf.F[2, 9] = dt  # z += vz * dt
        kf.F[6, 10] = dt  # yaw += vyaw * dt
        
        # Measurement function
        kf.H = np.zeros((7, 11))
        kf.H[0, 0] = 1  # x
        kf.H[1, 1] = 1  # y
        kf.H[2, 2] = 1  # z
        kf.H[3, 3] = 1  # width
        kf.H[4, 4] = 1  # height
        kf.H[5, 5] = 1  # length
        kf.H[6, 6] = 1  # yaw
        
        # Measurement uncertainty
        kf.R = np.eye(7) * 0.1
        kf.R[0:3, 0:3] *= 1.0  # Location uncertainty
        kf.R[3:6, 3:6] *= 0.1  # Dimension uncertainty
        kf.R[6, 6] = 0.3  # Orientation uncertainty
        
        # Process uncertainty
        kf.Q = np.eye(11) * 0.1
        kf.Q[7:11, 7:11] *= 0.5  # Velocity uncertainty
        
        # Initial state uncertainty
        kf.P = np.eye(11) * 1.0
        kf.P[7:11, 7:11] *= 10.0  # Velocity uncertainty
        
        return kf
    
    def _apply_kalman_filter(self, box_3d, object_id):
        """
        Apply Kalman filtering to smooth 3D box parameters
        
        Args:
            box_3d (dict): 3D bounding box parameters
            object_id (int): Object ID for tracking
            
        Returns:
            dict: Filtered 3D bounding box parameters
        """
        # Initialize Kalman filter if this is a new object
        if object_id not in self.kf_trackers:
            self.kf_trackers[object_id] = self._init_kalman_filter(box_3d)
        
        # Get the Kalman filter for this object
        kf = self.kf_trackers[object_id]
        
        # Predict
        kf.predict()
        
        # Update with measurement
        measurement = np.array([
            box_3d['location'][0],
            box_3d['location'][1],
            box_3d['location'][2],
            box_3d['dimensions'][1],  # width
            box_3d['dimensions'][0],  # height
            box_3d['dimensions'][2],  # length
            box_3d['orientation']
        ])
        
        kf.update(measurement)
        
        # Update box_3d with filtered values
        filtered_box = box_3d.copy()
        filtered_box['location'] = np.array([kf.x[0], kf.x[1], kf.x[2]])
        filtered_box['dimensions'] = np.array([kf.x[4], kf.x[3], kf.x[5]])  # height, width, length
        filtered_box['orientation'] = kf.x[6]
        
        return filtered_box
    
    def _apply_temporal_filter(self, object_id):
        """
        Apply temporal filtering to smooth 3D box parameters over time
        
        Args:
            object_id (int): Object ID for tracking
            
        Returns:
            dict: Temporally filtered 3D bounding box parameters
        """
        history = self.box_history[object_id]
        
        if len(history) < 2:
            return history[-1]
        
        # Get the most recent box
        current_box = history[-1]
        
        # Apply exponential moving average to location and orientation
        alpha = 0.7  # Weight for current measurement (higher = less smoothing)
        
        # Initialize with current values
        filtered_box = current_box.copy()
        
        # Apply EMA to location and orientation
        for i in range(len(history) - 2, -1, -1):
            weight = alpha * (1 - alpha) ** (len(history) - i - 2)
            filtered_box['location'] = filtered_box['location'] * (1 - weight) + history[i]['location'] * weight
            
            # Handle orientation wrapping
            angle_diff = history[i]['orientation'] - filtered_box['orientation']
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            filtered_box['orientation'] += angle_diff * weight
        
        return filtered_box
    
    def project_box_3d_to_2d(self, box_3d):
        """
        Project 3D bounding box corners to 2D image space
        
        Args:
            box_3d (dict): 3D bounding box parameters
            
        Returns:
            numpy.ndarray: 2D points of the 3D box corners (8x2)
        """
        # Extract parameters
        h, w, length = box_3d['dimensions']
        x, y, z = box_3d['location']
        rot_y = box_3d['orientation']
        
        # Get 2D box for reference
        x1, y1, x2, y2 = box_3d['bbox_2d']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width_2d = x2 - x1
        height_2d = y2 - y1
        
        # Create rotation matrix
        R_mat = np.array([
            [np.cos(rot_y), 0, np.sin(rot_y)],
            [0, 1, 0],
            [-np.sin(rot_y), 0, np.cos(rot_y)]
        ])

        # For other objects, use standard box configuration
        x_corners = np.array([
            length / 2, length / 2, -length / 2, -length / 2,
            length / 2, length / 2, -length / 2, -length / 2
        ])
        y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])  # Bottom at y=0
        z_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

        # Rotate and translate corners
        corners_3d = np.vstack([x_corners, y_corners, z_corners])
        corners_3d = R_mat @ corners_3d
        corners_3d[0, :] += x
        corners_3d[1, :] += y
        corners_3d[2, :] += z
        
        # Project to 2D
        corners_3d_homo = np.vstack([corners_3d, np.ones((1, 8))])
        corners_2d_homo = self.P @ corners_3d_homo
        corners_2d = corners_2d_homo[:2, :] / corners_2d_homo[2, :]
        
        # Constrain the 3D box to be within a reasonable distance of the 2D box
        mean_x = np.mean(corners_2d[0, :])
        mean_y = np.mean(corners_2d[1, :])
        
        # If the projected box is too far from the 2D box center, adjust it
        if abs(mean_x - center_x) > width_2d or abs(mean_y - center_y) > height_2d:
            # Shift the projected points to center on the 2D box
            shift_x = center_x - mean_x
            shift_y = center_y - mean_y
            corners_2d[0, :] += shift_x
            corners_2d[1, :] += shift_y
        
        return corners_2d.T
    
    def draw_box_3d(self, image, box_3d, color=(0, 255, 0), thickness=2):
        """
        Draw enhanced 3D bounding box on image with better depth perception
        
        Args:
            image (numpy.ndarray): Image to draw on
            box_3d (dict): 3D bounding box parameters
            color (tuple): Color in BGR format
            thickness (int): Line thickness
            
        Returns:
            numpy.ndarray: Image with 3D box drawn
        """
        # Get 2D box coordinates
        x1, y1, x2, y2 = [int(coord) for coord in box_3d['bbox_2d']]
        
        # Get depth value for scaling
        depth_value = box_3d.get('depth_value', 0.5)
        
        # Calculate box dimensions
        width = x2 - x1
        height = y2 - y1
        
        # Calculate the offset for the 3D effect (deeper objects have smaller offset)
        # Inverse relationship with depth - closer objects have larger offset
        depth_alpha = 0.7  # Adjust this to control depth effect
        offset_factor = 1.0 - depth_value
        offset_x = int(width * depth_alpha * offset_factor)
        offset_y = int(height * depth_alpha * offset_factor)

        # Make min/max offset proportional to box size
        min_offset = int(min(width, height) * 0.08)
        max_offset = int(min(width, height) * 0.25)
        offset_x = np.clip(offset_x, min_offset, max_offset)
        offset_y = np.clip(offset_y, min_offset, max_offset)
                
        # Create points for the 3D box
        # Front face (the 2D bounding box)
        front_tl = (x1, y1)
        front_tr = (x2, y1)
        front_br = (x2, y2)
        front_bl = (x1, y2)
        
        # Back face (offset by depth)
        back_tl = (x1 + offset_x, y1 - offset_y)
        back_tr = (x2 + offset_x, y1 - offset_y)
        back_br = (x2 + offset_x, y2 - offset_y)
        back_bl = (x1 + offset_x, y2 - offset_y)
        
        # Create a slightly transparent copy of the image for the 3D effect
        overlay = image.copy()
        
        # Draw the front face (2D bounding box)
        cv2.rectangle(image, front_tl, front_br, color, thickness)
        
        # Draw the connecting lines between front and back faces
        cv2.line(image, front_tl, back_tl, color, thickness)
        cv2.line(image, front_tr, back_tr, color, thickness)
        cv2.line(image, front_br, back_br, color, thickness)
        cv2.line(image, front_bl, back_bl, color, thickness)
        
        # Draw the back face
        cv2.line(image, back_tl, back_tr, color, thickness)
        cv2.line(image, back_tr, back_br, color, thickness)
        cv2.line(image, back_br, back_bl, color, thickness)
        cv2.line(image, back_bl, back_tl, color, thickness)
        
        # Fill the top face with a semi-transparent color to enhance 3D effect
        pts_top = np.array([front_tl, front_tr, back_tr, back_tl], np.int32)
        pts_top = pts_top.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts_top], color)
        
        # Fill the right face with a semi-transparent color
        pts_right = np.array([front_tr, front_br, back_br, back_tr], np.int32)
        pts_right = pts_right.reshape((-1, 1, 2))
        
        # Darken the right face color for better 3D effect
        right_color = (int(color[0] * 0.7), int(color[1] * 0.7), int(color[2] * 0.7))
        cv2.fillPoly(overlay, [pts_right], right_color)
        
        # Apply the overlay with transparency
        alpha = 0.3  # Transparency factor
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # Get class name and object ID
        class_name = box_3d['class_name']
        obj_id = box_3d['object_id'] if 'object_id' in box_3d else None
        
        # Draw text information
        text_y = y1 - 10
        font_scale = 0.35  # Reduced font size
        font_thickness = 1
        if obj_id is not None:
            cv2.putText(image, f"ID:{obj_id}", (x1, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
            text_y -= 12

        cv2.putText(image, class_name, (x1, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
        text_y -= 12

        # Get depth information if available
        if 'depth_value' in box_3d:
            depth_value = box_3d['depth_value']
            depth_text = f"D:{depth_value:.2f}"
            cv2.putText(image, depth_text, (x1, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
            text_y -= 12

        # Get score if available
        if 'score' in box_3d:
            score = box_3d['score']
            score_text = f"S:{score:.2f}"
            cv2.putText(image, score_text, (x1, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
        
        # Draw a vertical line from the bottom of the box to the ground
        # This helps with depth perception
        ground_y = y2 + int(height * 0.2)  # A bit below the bottom of the box
        cv2.line(image, (int((x1 + x2) / 2), y2), (int((x1 + x2) / 2), ground_y), color, thickness)
        
        # Draw a small circle at the bottom to represent the ground contact point
        cv2.circle(image, (int((x1 + x2) / 2), ground_y), thickness * 2, color, -1)
        
        return image
    
    def cleanup_trackers(self, active_ids):
        """
        Clean up Kalman filters and history for objects that are no longer tracked
        
        Args:
            active_ids (list): List of active object IDs
        """
        # Convert to set for faster lookup
        active_ids_set = set(active_ids)
        
        # Clean up Kalman filters
        for obj_id in list(self.kf_trackers.keys()):
            if obj_id not in active_ids_set:
                del self.kf_trackers[obj_id]
        
        # Clean up box history
        for obj_id in list(self.box_history.keys()):
            if obj_id not in active_ids_set:
                del self.box_history[obj_id]


class BirdEyeView:
    """
    Refactored Bird's Eye View (BEV) visualization for 3D bounding boxes.
    This class renders a top-down view based on heuristic estimates of depth
    and position from 2D data.

    World Coordinate System Assumption:
    - X axis: Forward (rendered as 'up' in the BEV image)
    - Y axis: Right (rendered as 'right' in the BEV image)

    Image Coordinate System:
    - Origin (0,0) is at the top-left. Ego vehicle is at the bottom-center.
    """
    # --- Constants for Styling ---
    COLOR_BACKGROUND = (20, 20, 20)
    COLOR_GRID = (80, 80, 80)
    COLOR_AXIS_X = (0, 200, 0)  # Green for Forward (X)
    COLOR_AXIS_Y = (0, 0, 200)  # Red for Right (Y)
    COLOR_MARKERS = (180, 180, 180)
    COLOR_TEXT = (220, 220, 220)
    COLOR_BOX_DEFAULT = (255, 255, 255)
    COLOR_BOX_LINE = (70, 70, 70)

    FONT_STYLE = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_AXIS = 0.3
    FONT_SCALE_MARKER = 0.2
    FONT_SCALE_ID = 0.2

    THICKNESS_GRID = 1
    THICKNESS_AXIS = 2
    THICKNESS_MARKER = 1
    
    def __init__(self, image_shape=(800, 600), scale=30, camera_height=1.2, max_distance=5.0):
        """
        Args:
            image_shape (tuple): (width, height) of the BEV image.
            scale (float): Pixels per meter.
            camera_height (float): Camera height above ground (meters). Not used in this implementation.
            max_distance (float): Max distance (meters) to map `depth_value` to.
        """
        self.width, self.height = image_shape
        self.scale = float(scale)
        self.camera_height = camera_height
        self.max_distance = max_distance

        # Origin point (ego vehicle location) in the image
        self.origin_x = self.width // 2
        self.origin_y = self.height

        self.bev_image = None
        self.reset()

    def reset(self):
        """Resets the BEV image to a blank canvas with grid, axes, and markers."""
        self.bev_image = np.full((self.height, self.width, 3), self.COLOR_BACKGROUND, dtype=np.uint8)
        self._draw_grid()
        self._draw_axes()
        self._draw_distance_markers()

    def _draw_grid(self, step_m=0.5):
        """Draws the grid on the BEV canvas."""
        # Horizontal lines (distance from camera)
        for dist in np.arange(step_m, self.max_distance * 2, step_m):
            y = int(self.origin_y - dist * self.scale)
            if y < 0: 
                break
            cv2.line(self.bev_image, (0, y), (self.width, y), self.COLOR_GRID, self.THICKNESS_GRID)

        # Vertical lines (sideways distance from center)
        max_side_dist_m = self.origin_x / self.scale
        for dist in np.arange(step_m, max_side_dist_m, step_m):
            # Lines to the right of center
            x_right = int(self.origin_x + dist * self.scale)
            cv2.line(self.bev_image, (x_right, 0), (x_right, self.height), self.COLOR_GRID, self.THICKNESS_GRID)
            # Lines to the left of center
            x_left = int(self.origin_x - dist * self.scale)
            cv2.line(self.bev_image, (x_left, 0), (x_left, self.height), self.COLOR_GRID, self.THICKNESS_GRID)

    def _draw_axes(self):
        """Draws the X (forward) and Y (right) axes."""
        axis_length_px = min(80, self.height // 5)
        
        # Forward (X, up)
        cv2.line(self.bev_image, (self.origin_x, self.origin_y),
                 (self.origin_x, self.origin_y - axis_length_px), self.COLOR_AXIS_X, self.THICKNESS_AXIS)
        cv2.putText(self.bev_image, "X", (self.origin_x - 15, self.origin_y - axis_length_px + 15),
                    self.FONT_STYLE, self.FONT_SCALE_AXIS, self.COLOR_AXIS_X, 1)
        
        # Right (Y, right)
        cv2.line(self.bev_image, (self.origin_x, self.origin_y),
                 (self.origin_x + axis_length_px, self.origin_y), self.COLOR_AXIS_Y, self.THICKNESS_AXIS)
        cv2.putText(self.bev_image, "Y", (self.origin_x + axis_length_px - 15, self.origin_y - 10),
                    self.FONT_STYLE, self.FONT_SCALE_AXIS, self.COLOR_AXIS_Y, 1)

    def _draw_distance_markers(self, step_m=0.5, label_interval_m=1.0):
        """Draws distance markers along the forward axis."""
        for dist in np.arange(step_m, self.max_distance, step_m):
            y = int(self.origin_y - dist * self.scale)
            if y < 0: 
                break
            
            cv2.line(self.bev_image, (self.origin_x - 5, y), (self.origin_x + 5, y), 
                     self.COLOR_MARKERS, self.THICKNESS_MARKER)

            # Add text label at specified intervals
            if dist % label_interval_m < 1e-3:
                cv2.putText(self.bev_image, f"{int(dist)}m", (self.origin_x + 10, y + 4),
                            self.FONT_STYLE, self.FONT_SCALE_MARKER, self.COLOR_TEXT, 1)

    def draw_box(self, box_3d: dict, color: tuple = None):
        """
        Draws an object on the BEV image using heuristic estimates.
        NOTE: This method does NOT perform a true 3D to 2D projection.
              It places objects based on a normalized 'depth_value' and
              the 2D bounding box's screen position.

        Args:
            box_3d (dict): Dictionary with object parameters. Expected keys:
                - 'depth_value' (float): Normalized depth [0, 1].
                - 'bbox_2d' (list): 2D bounding box [x1, y1, x2, y2].
                - 'object_id' (any, optional): ID to display.
            color (tuple, optional): BGR color. If None, uses a default.
        """
        try:
            box_color = color if color is not None else self.COLOR_BOX_DEFAULT
            
            # --- Heuristic Depth Calculation ---
            # Linearly map the normalized depth_value to a distance in meters.
            depth_value = 1.0 - float(box_3d.get('depth_value', 0.5))
            # Map depth_value [0,1] to [0, max_distance] meters, then to [origin_y, 0]
            depth_m = depth_value * self.max_distance
            bev_y = int(self.origin_y - (depth_m / self.max_distance) * self.height)
            
            # --- Heuristic Sideways Position Calculation ---
            # Use the 2D box's horizontal center to estimate side position.
            bbox_2d = box_3d.get('bbox_2d')
            if bbox_2d:
                x1, _, x2, _ = bbox_2d
                center_x_2d = (x1 + x2) / 2
                # Map screen position (-0.5 to 0.5) to a scaled BEV position.
                rel_x = (center_x_2d / self.width) - 0.5
                bev_x = int(self.origin_x + rel_x * self.width * 0.8)
            else:
                bev_x = self.origin_x  # Default to center if no 2D box

            # --- Heuristic Size Calculation ---
            # Scale the drawn box size based on the 2D box width.
            size_factor = 1.0
            if bbox_2d:
                width_2d = bbox_2d[2] - bbox_2d[0]
                size_factor = max(0.5, min(width_2d / 100.0, 2.0))
            size_px = int(8 * size_factor)

            # Clamp coordinates to be visible within the image
            bev_x = np.clip(bev_x, size_px, self.width - size_px)
            bev_y = np.clip(bev_y, size_px, self.origin_y - size_px)

            # Draw the object as a square
            cv2.rectangle(self.bev_image, (bev_x - size_px, bev_y - size_px), 
                          (bev_x + size_px, bev_y + size_px), box_color, -1)
            
            # Draw line from origin to object
            cv2.line(self.bev_image, (self.origin_x, self.origin_y), (bev_x, bev_y),
                     self.COLOR_BOX_LINE, 1)

            # Draw object ID if present
            if 'object_id' in box_3d:
                cv2.putText(self.bev_image, str(box_3d['object_id']), (bev_x - 5, bev_y - 5),
                            self.FONT_STYLE, self.FONT_SCALE_ID, (0, 0, 0), 1)

        except (KeyError, TypeError) as e:
            print(f"Error drawing box: Invalid or missing data in 'box_3d' dict -> {e}")
        except Exception as e:
            print(f"An unexpected error occurred in draw_box: {e}")

    def get_image(self) -> np.ndarray:
        """Returns the current BEV image."""
        return self.bev_image
        
    def get_resized_image(self, frame_width, frame_height, size_factor=0.25, aspect_ratio=4 / 3):
        """
        Returns a properly resized BEV image suitable for overlay on a frame.
        
        Args:
            frame_width (int): Width of the target frame
            frame_height (int): Height of the target frame
            size_factor (float): Size as a fraction of frame height (default: 0.25 = 25%)
            aspect_ratio (float): Width:height ratio for BEV (default: 4/3)
            
        Returns:
            np.ndarray: Resized BEV image
        """
        # Calculate BEV dimensions based on frame size
        bev_height = int(frame_height * size_factor)
        
        # Use specified aspect ratio for BEV
        bev_width = int(bev_height * aspect_ratio)
        
        # Make sure BEV width doesn't exceed 1/3 of the frame width
        if bev_width > frame_width // 3:
            bev_width = frame_width // 3
            bev_height = int(bev_width / aspect_ratio)
        
        # Ensure dimensions are valid
        if bev_height <= 0 or bev_width <= 0:
            # Return a small valid image if dimensions are invalid
            return np.zeros((10, 10, 3), dtype=np.uint8)
            
        # Resize and return the BEV image
        return cv2.resize(self.bev_image, (bev_width, bev_height))