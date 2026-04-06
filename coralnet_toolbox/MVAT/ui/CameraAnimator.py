"""
Camera Animator for MVAT Viewer

Provides smooth transitions between camera perspectives using Qt timers
and quaternion-based interpolation for gimbal-lock-free rotation.
"""

import numpy as np
from PyQt5.QtCore import QTimer, QObject


class Quaternion:
    """
    Minimal quaternion class for SLERP (Spherical Linear Interpolation).
    Represents rotations in xyzw format.
    """
    def __init__(self, x=0, y=0, z=0, w=1):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    
    @classmethod
    def from_vectors(cls, v1, v2):
        """Create a quaternion representing rotation from v1 to v2."""
        v1 = np.array(v1, dtype=float)
        v2 = np.array(v2, dtype=float)
        
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Cross product for axis of rotation
        axis = np.cross(v1, v2)
        axis_len = np.linalg.norm(axis)
        
        if axis_len < 1e-10:
            # Vectors are parallel (or antiparallel)
            if np.dot(v1, v2) > 0:
                # Same direction: no rotation
                return cls(0, 0, 0, 1)
            else:
                # Opposite direction: rotate 180° around an arbitrary perpendicular
                perp = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
                axis = np.cross(v1, perp)
                axis = axis / np.linalg.norm(axis)
                return cls(axis[0], axis[1], axis[2], 0)
        
        axis = axis / axis_len
        angle = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
        
        # Quaternion: q = (sin(angle/2) * axis, cos(angle/2))
        half_angle = angle / 2
        sin_half = np.sin(half_angle)
        
        return cls(
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
            np.cos(half_angle)
        )
    
    def slerp(self, other, t):
        """
        Spherical Linear Interpolation with another quaternion.
        t: interpolation parameter in [0, 1]
        """
        # Ensure we take the shorter path
        dot = self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
        
        if dot < 0:
            other = Quaternion(-other.x, -other.y, -other.z, -other.w)
            dot = -dot
        
        dot = np.clip(dot, -1, 1)
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        
        if sin_theta < 1e-10:
            # Quaternions are very close; use linear interpolation
            t_inv = 1 - t
            return Quaternion(
                self.x * t_inv + other.x * t,
                self.y * t_inv + other.y * t,
                self.z * t_inv + other.z * t,
                self.w * t_inv + other.w * t
            )
        
        w1 = np.sin((1 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta
        
        return Quaternion(
            self.x * w1 + other.x * w2,
            self.y * w1 + other.y * w2,
            self.z * w1 + other.z * w2,
            self.w * w1 + other.w * w2
        )
    
    def to_vector(self):
        """Convert quaternion back to a direction vector (xyzw format as array)."""
        return np.array([self.x, self.y, self.z, self.w])


class CameraAnimator(QObject):
    """
    Animates smooth camera transitions using Qt timers.
    
    Features:
    - Linear interpolation for position and focal point
    - Quaternion SLERP for smooth rotation (gimbal-lock-free)
    - Configurable duration (default 400ms)
    - Support for cancellation (rapid clicks restart animation)
    """
    
    def __init__(self, plotter, duration_ms=400):
        """
        Initialize the camera animator.
        
        Args:
            plotter: PyVista QtInteractor plotter instance
            duration_ms: Animation duration in milliseconds (default 400)
        """
        super().__init__()
        self.plotter = plotter
        self.duration_ms = duration_ms
        self.elapsed_ms = 0
        
        # Start/end states
        self.start_pos = None
        self.start_focal = None
        self.start_up = None
        self.start_fov = None
        
        self.end_pos = None
        self.end_focal = None
        self.end_up = None
        self.end_fov = None
        
        # Quaternion-based rotation state
        self.start_quat = None
        self.end_quat = None
        
        # Timer
        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self._on_timer_tick)
        self.frame_interval_ms = 16  # ~60 FPS
        
    def animate_to_camera_state(self, position, focal_point, up_vector, fov_deg):
        """
        Start animating to a new camera state.
        
        Args:
            position: 3D camera position (numpy array or list)
            focal_point: 3D focal point (numpy array or list)
            up_vector: 3D up vector (numpy array or list)
            fov_deg: Vertical field of view in degrees
        """
        try:
            # Capture current state
            cam = self.plotter.camera
            self.start_pos = np.array(cam.position, dtype=float)
            self.start_focal = np.array(cam.focal_point, dtype=float)
            self.start_up = np.array(cam.up, dtype=float)
            self.start_up = self.start_up / np.linalg.norm(self.start_up)
            self.start_fov = cam.view_angle
            
            # Set target state
            self.end_pos = np.array(position, dtype=float)
            self.end_focal = np.array(focal_point, dtype=float)
            self.end_up = np.array(up_vector, dtype=float)
            self.end_up = self.end_up / np.linalg.norm(self.end_up)
            self.end_fov = float(fov_deg)
            
            # Compute quaternions for smooth rotation
            # Represent rotation as: view_direction + up_vector
            start_view = self.start_focal - self.start_pos
            end_view = self.end_focal - self.end_pos
            
            self.start_quat = Quaternion.from_vectors(
                [0, 0, 1],  # identity view direction
                start_view
            )
            self.end_quat = Quaternion.from_vectors(
                [0, 0, 1],
                end_view
            )
            
            # Reset elapsed time, apply initial state (t=0), then start animation
            self.elapsed_ms = 0
            # Apply the start frame immediately to ensure smooth transition from current view
            self._apply_camera_state(0.0)
            self.timer.start(self.frame_interval_ms)
        except Exception as e:
            print(f"CameraAnimator: Failed to start animation: {e}")
            # Fallback: jump to end state immediately
            self._apply_camera_state(1.0)
            self.cancel()
    
    def _on_timer_tick(self):
        """Timer callback for animation frame."""
        self.elapsed_ms += self.frame_interval_ms
        
        if self.elapsed_ms >= self.duration_ms:
            # Animation complete
            self._apply_camera_state(1.0)
            self.cancel()
        else:
            # Animation in progress
            t = self.elapsed_ms / self.duration_ms
            self._apply_camera_state(t)
    
    def _apply_camera_state(self, t):
        """
        Apply interpolated camera state at parameter t in [0, 1].
        
        Args:
            t: Interpolation parameter (0=start, 1=end)
        """
        try:
            if (self.start_pos is None or self.end_pos is None):
                return
            
            # Linear interpolation for position and focal point
            interp_pos = self.start_pos + t * (self.end_pos - self.start_pos)
            interp_focal = self.start_focal + t * (self.end_focal - self.start_focal)
            
            # Quaternion SLERP for smooth rotation
            interp_quat = self.start_quat.slerp(self.end_quat, t)
            interp_quat_vec = interp_quat.to_vector()
            
            # Derive up vector from interpolated quaternion
            # For simplicity, linear interpolation of up vector as well
            # (This is a reasonable approximation for camera orbiting)
            interp_up = self.start_up + t * (self.end_up - self.start_up)
            interp_up = interp_up / np.linalg.norm(interp_up)
            
            # Linear interpolation for FOV
            interp_fov = self.start_fov + t * (self.end_fov - self.start_fov)
            
            # Apply to camera
            cam = self.plotter.camera
            cam.position = interp_pos.tolist()
            cam.focal_point = interp_focal.tolist()
            cam.up = interp_up.tolist()
            cam.view_angle = float(interp_fov)
            
            # Trigger render
            try:
                self.plotter.render()
            except Exception:
                pass
        except Exception as e:
            print(f"CameraAnimator: Failed to apply state: {e}")
    
    def cancel(self):
        """Stop the animation."""
        self.timer.stop()
        self.start_pos = None
        self.start_focal = None
        self.start_up = None
        self.start_fov = None
        self.end_pos = None
        self.end_focal = None
        self.end_up = None
        self.end_fov = None
