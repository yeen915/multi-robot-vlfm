"""Conversions between the simulator's world frame and the shared map frame W.

W is R1's episodic frame as VLFM uses it: the origin is the episode start position,
x points along the start heading, y to the left and z up. The simulator's world frame
has y up, and an agent with identity rotation faces -z.

Poses of both robots are expressed in W, because Habitat's GPS and compass sensors
are computed relative to the episode start whichever robot the agent is standing in
for. These functions do the same conversion directly from simulator poses.
"""

from typing import List, Tuple

import numpy as np
import quaternion


def _quat(rotation_xyzw: List[float]) -> quaternion.quaternion:
    x, y, z, w = rotation_xyzw
    return quaternion.quaternion(w, x, y, z)


def _rotate(q: quaternion.quaternion, v: np.ndarray) -> np.ndarray:
    return (q * quaternion.quaternion(0, *v) * q.inverse()).imag


def world_to_shared(
    position: np.ndarray,
    rotation_xyzw: List[float],
    start_position: np.ndarray,
    start_rotation_xyzw: List[float],
) -> Tuple[np.ndarray, float]:
    """Converts a simulator pose to W.

    Returns:
        (x, y, z) in W, where z is the height above the episode start, and the yaw in
        radians (counter-clockwise from the start heading, seen from above).
    """
    q_start = _quat(start_rotation_xyzw)
    local = _rotate(q_start.inverse(), np.asarray(position, dtype=np.float64) - np.asarray(start_position))
    xyz = np.array([-local[2], -local[0], local[1]])

    forward = _rotate(q_start.inverse() * _quat(rotation_xyzw), np.array([0.0, 0.0, -1.0]))
    yaw = float(np.arctan2(-forward[0], -forward[2]))
    return xyz, yaw


def shared_to_world(
    points: np.ndarray,
    start_position: np.ndarray,
    start_rotation_xyzw: List[float],
) -> np.ndarray:
    """Converts (N, 3) points in W to the simulator's world frame."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    local = np.stack([-points[:, 1], points[:, 2], -points[:, 0]], axis=1)
    q_start = _quat(start_rotation_xyzw)
    rotated = np.array([_rotate(q_start, p) for p in local])
    return rotated + np.asarray(start_position, dtype=np.float64)
