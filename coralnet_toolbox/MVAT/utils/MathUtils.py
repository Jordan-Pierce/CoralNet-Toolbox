"""
Math utilities for MVAT.

Provides the Quaternion class used for gimbal-lock-free camera interpolation
in QtCameraAnimator.
"""

import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


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
