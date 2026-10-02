"""auto_label.py: pure-geometry checks against independently hand-computed
expected pixel boxes, plus the relative-pose transform against
rotations.world_to_body (already trusted, used throughout the live pipeline
for ground-truth checks)."""
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import auto_label as al  # noqa: E402
import rotations  # noqa: E402

# A simple pinhole camera: 100 px focal length, 200x200 image, centered.
K = np.array([[100.0, 0.0, 100.0],
              [0.0, 100.0, 100.0],
              [0.0, 0.0, 1.0]])


def test_project_point_matches_hand_computed_pixel():
    # A point 1m in front of the camera, 0.5m to the right (+X), 0.25m down
    # (+Y, optical convention): u = fx*X/Z + cx = 100*0.5/1 + 100 = 150;
    # v = fy*Y/Z + cy = 100*0.25/1 + 100 = 125.
    u, v = al.project_point(K, (0.5, 0.25, 1.0))
    assert math.isclose(u, 150.0) and math.isclose(v, 125.0)


def test_project_point_behind_camera_is_none():
    assert al.project_point(K, (0.1, 0.0, -0.5)) is None
    assert al.project_point(K, (0.1, 0.0, 0.0)) is None


def test_box_corners_axis_aligned_at_origin():
    corners = al.box_corners((0.0, 0.0, 1.0), (0.1, 0.2, 0.3))
    arr = np.array(corners)
    assert len(corners) == 8
    assert np.isclose(arr[:, 0].min(), -0.1) and np.isclose(arr[:, 0].max(), 0.1)
    assert np.isclose(arr[:, 1].min(), -0.2) and np.isclose(arr[:, 1].max(), 0.2)
    assert np.isclose(arr[:, 2].min(), 0.7) and np.isclose(arr[:, 2].max(), 1.3)


def test_project_box_to_bbox2d_hand_computed_flat_plane():
    # A flat (zero-depth) 0.2 x 0.2 plane centered at (0, 0, 2), all 4
    # corners at the same Z: half-width in pixels = fx*half_extent/Z =
    # 100*0.1/2 = 5 exactly -> x in [95, 105], y in [95, 105].
    corners = al.box_corners((0.0, 0.0, 2.0), (0.1, 0.1, 0.0))
    box = al.project_box_to_bbox2d(K, corners, img_w=200, img_h=200)
    assert box is not None
    x1, y1, x2, y2 = box
    assert math.isclose(x1, 95.0, abs_tol=1e-6) and math.isclose(x2, 105.0, abs_tol=1e-6)
    assert math.isclose(y1, 95.0, abs_tol=1e-6) and math.isclose(y2, 105.0, abs_tol=1e-6)


def test_project_box_to_bbox2d_hand_computed_cube_near_face_dominates():
    # A true 0.2 x 0.2 x 0.2 cube centered at (0, 0, 2): perspective means
    # the NEAR face (z=1.9) projects to a wider box than the far face
    # (z=2.1), and the enclosing box is set by the near face:
    # half-width = fx*0.1/1.9 = 100*0.1/1.9 = 5.263157...
    corners = al.box_corners((0.0, 0.0, 2.0), (0.1, 0.1, 0.1))
    box = al.project_box_to_bbox2d(K, corners, img_w=200, img_h=200)
    assert box is not None
    x1, y1, x2, y2 = box
    half = 100.0 * 0.1 / 1.9
    assert math.isclose(x1, 100.0 - half, abs_tol=1e-9)
    assert math.isclose(x2, 100.0 + half, abs_tol=1e-9)
    assert math.isclose(y1, 100.0 - half, abs_tol=1e-9)
    assert math.isclose(y2, 100.0 + half, abs_tol=1e-9)


def test_project_box_to_bbox2d_clips_to_image_and_can_vanish():
    # Far off to one side: the 2D box should clip to the image edge, not
    # read negative/over-bound pixel values.
    corners = al.box_corners((5.0, 0.0, 2.0), (0.1, 0.1, 0.1))
    box = al.project_box_to_bbox2d(K, corners, img_w=200, img_h=200)
    assert box is None   # entirely outside the 200x200 frame

    # Straddling the left edge: expect the box clipped at x=0, not negative.
    corners = al.box_corners((-1.9, 0.0, 2.0), (0.3, 0.1, 0.1))
    box = al.project_box_to_bbox2d(K, corners, img_w=200, img_h=200)
    assert box is not None
    x1, y1, x2, y2 = box
    assert x1 == 0.0
    assert x2 > 0.0


def test_project_box_to_bbox2d_wholly_behind_camera_is_none():
    corners = al.box_corners((0.0, 0.0, -2.0), (0.1, 0.1, 0.1))
    assert al.project_box_to_bbox2d(K, corners, img_w=200, img_h=200) is None


def test_relative_pose_matches_trusted_world_to_body_for_position():
    ref_pos = (1.0, 2.0, 0.3)
    ref_quat = rotations.quaternion_from_euler(0.0, math.radians(-2.0), math.radians(30.0))
    obj_pos = (1.5, 1.6, 0.35)
    obj_quat = rotations.quaternion_from_euler(0.0, 0.0, math.radians(10.0))

    pos_rel, R_rel = al.relative_pose(ref_pos, ref_quat, obj_pos, obj_quat,
                                      rotations.quaternion_matrix)
    expected_pos = rotations.world_to_body(ref_pos, ref_quat, obj_pos)
    assert np.allclose(pos_rel, expected_pos)

    # R_rel must be a proper rotation (orthonormal, det +1), not an
    # arbitrary 3x3.
    assert np.allclose(R_rel @ R_rel.T, np.eye(3), atol=1e-9)
    assert math.isclose(np.linalg.det(R_rel), 1.0, abs_tol=1e-9)


def test_yolo_label_line_hand_computed():
    # A box from x=[40,60], y=[80,120] in a 200x100 image: centre (50,100),
    # size (20,40) -> normalized cx=0.25, cy=1.0, w=0.1, h=0.4.
    line = al.yolo_label_line(2, (40.0, 80.0, 60.0, 120.0), img_w=200, img_h=100)
    cls, cx, cy, w, h = line.split()
    assert cls == "2"
    assert math.isclose(float(cx), 0.25) and math.isclose(float(cy), 1.0)
    assert math.isclose(float(w), 0.1) and math.isclose(float(h), 0.4)


if __name__ == "__main__":
    import sys as _sys
    mod = _sys.modules[__name__]
    for name in dir(mod):
        if name.startswith("test_"):
            getattr(mod, name)()
    print("OK")
