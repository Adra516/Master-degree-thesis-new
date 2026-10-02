"""Pure geometry for auto-labeling training frames from Gazebo ground truth —
no ROS imports, so it can be tested offline (see test/test_auto_label.py).

The idea: a catalogued object's true size and pose are known exactly in
Gazebo. Projecting its 3D box into the camera image gives an exact 2D
bounding box for free, with no manual annotation — used by
collect_training_data.py to build a fine-tuning dataset for YOLO3D's
underlying detector (a plain axis-aligned detector, not OBB; see that
script's own docstring for why).

Frame conventions:
  - "cam" = the camera OPTICAL frame (ROS convention: +Z forward into the
    scene, +X right, +Y down) — the same frame yolo_detection_node.py's
    pos_cam/_backproject_point already use.
  - K is the 3x3 camera intrinsic matrix from sensor_msgs/CameraInfo.
"""
import itertools

import numpy as np


def project_point(K, p_cam):
    """Pinhole-project a 3D point in the camera optical frame to pixel
    (u, v), or None if it is behind the camera (z <= 0 — cannot appear in
    the image at all)."""
    x, y, z = p_cam
    if z <= 1e-6:
        return None
    return (K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2])


def box_corners(center, half_extent, R=None):
    """The 8 corners of an oriented box: center + R @ (+-hx, +-hy, +-hz),
    `half_extent` = (hx, hy, hz) in the box's own local axes. `R` is the
    box's rotation matrix (world or camera frame, whichever `center` is in);
    identity if the box is axis-aligned in that frame."""
    hx, hy, hz = half_extent
    R = np.eye(3) if R is None else np.asarray(R, dtype=float)
    center = np.asarray(center, dtype=float)
    return [center + R @ np.array([sx * hx, sy * hy, sz * hz])
            for sx, sy, sz in itertools.product((-1.0, 1.0), repeat=3)]


def project_box_to_bbox2d(K, corners_cam, img_w, img_h):
    """The enclosing axis-aligned pixel box for a set of 3D corners already
    in the camera optical frame, clipped to the image bounds. None if every
    corner is behind the camera, or the clipped box is degenerate (the
    object is entirely outside the frame)."""
    pts = [p for p in (project_point(K, c) for c in corners_cam) if p is not None]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = max(0.0, min(xs)), min(float(img_w), max(xs))
    y1, y2 = max(0.0, min(ys)), min(float(img_h), max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def relative_pose(ref_pos, ref_quat, obj_pos, obj_quat, quaternion_matrix):
    """`obj`'s position and rotation matrix in the frame of a body at
    `ref_pos`/`ref_quat` — both Gazebo world poses (quaternions [x,y,z,w]).
    Position uses the full orientation (not yaw-only), matching
    rotations.world_to_body. `quaternion_matrix` is passed in (rotations.py)
    rather than imported, keeping this module free of any project-specific
    import path setup.
    """
    Rref = quaternion_matrix(ref_quat)[:3, :3]
    Robj = quaternion_matrix(obj_quat)[:3, :3]
    pos_rel = Rref.T @ (np.asarray(obj_pos, dtype=float) - np.asarray(ref_pos, dtype=float))
    R_rel = Rref.T @ Robj
    return pos_rel, R_rel


def yolo_label_line(class_id, bbox2d, img_w, img_h):
    """One line of a YOLO-format label file: "class cx cy w h", all but
    `class` normalized to [0, 1]."""
    x1, y1, x2, y2 = bbox2d
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return "%d %.6f %.6f %.6f %.6f" % (class_id, cx, cy, w, h)
