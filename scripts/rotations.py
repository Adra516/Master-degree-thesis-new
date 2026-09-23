"""Quaternion helpers — the subset of tf.transformations this module uses.

Ported from rework-attempt-1's src/locobot_prog/geometry/rotations.py, which
existed there so the geometry layer imports with no ROS installed (`tf` is a
ROS package). Kept here mainly for its one non-cosmetic function,
world_to_body(): the ORIGINAL yaw-only ground-truth transform in
locobot_motion_node.py's _object_in_base put a point at grasp height ~12 mm
off along the approach, because the robot rests pitched ~2 deg nose-down on
its caster — a real, measured bug (fixed here, in arm_diagnostics.py; NOT
touched in locobot_motion_node.py itself, which keeps its pre-existing
yaw-only version).

Same conventions as tf.transformations: quaternions are [x, y, z, w], Euler
angles are static-frame x-y-z (tf's default 'sxyz'), and quaternion_matrix
returns a 4x4 homogeneous matrix.
"""
import math

import numpy as np


def quaternion_from_euler(roll, pitch, yaw):
    """[x, y, z, w] for static-frame roll (x), pitch (y), yaw (z)."""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.array([sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy,
                     cr * cp * cy + sr * sp * sy])


def quaternion_matrix(quat):
    """4x4 homogeneous rotation matrix for quaternion [x, y, z, w]."""
    q = np.asarray(quat, dtype=float)
    n = float(np.dot(q, q))
    if n < 1e-12:
        return np.identity(4)
    q = q * math.sqrt(2.0 / n)
    q = np.outer(q, q)
    x, y, z, w = 0, 1, 2, 3
    return np.array([
        [1.0 - q[y, y] - q[z, z], q[x, y] - q[z, w], q[x, z] + q[y, w], 0.0],
        [q[x, y] + q[z, w], 1.0 - q[x, x] - q[z, z], q[y, z] - q[x, w], 0.0],
        [q[x, z] - q[y, w], q[y, z] + q[x, w], 1.0 - q[x, x] - q[y, y], 0.0],
        [0.0, 0.0, 0.0, 1.0]])


def world_to_body(body_position, body_quat, point):
    """`point` (world frame) in the frame of a body at `body_position` with
    orientation `body_quat` [x, y, z, w]. The FULL orientation: the LoCoBot
    rests pitched ~2° nose-down on its caster, and a yaw-only transform puts a
    point at grasp height ~12 mm off along the approach."""
    R = quaternion_matrix(body_quat)[:3, :3]
    return R.T @ (np.asarray(point, float) - np.asarray(body_position, float))


def yaw_from_quaternion(x, y, z, w):
    """Heading (rotation about z) of a quaternion, in (-pi, pi]."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
