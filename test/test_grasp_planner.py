"""The grasp planner on rendered depth images whose right answer is known.

Each scene is ray-cast from a camera placed like the LoCoBot's (above and
behind, looking down at the table), then goes through the same steps the node
runs: back-project the detection box, move to base_link, cut to the object,
plan. No ROS, no Gazebo.
"""
import math
import re
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, _HERE)
import synthetic_depth as sd  # noqa: E402
import footprint  # noqa: E402
import grasp_geometry as gg  # noqa: E402
import grasp_planner as gp  # noqa: E402

TABLE_Z = 0.25
EYE = (0.06, 0.0, 0.55)
TABLE = sd.Table(TABLE_Z, 0.30, 1.00, -0.50, 0.50)
MM = 1000.0


def plan_scene(shapes, box_corners, centre, dims_hwl, max_dev=0.0,
               centre_error=(0.01, -0.005), noise=None):
    """Render, back-project, cut and plan. `centre` is the object's true (x, y);
    the planner is given it with `centre_error` added, as the world model's
    estimate would be. noise = (rng, sigma) adds Gaussian depth noise."""
    look = (centre[0], centre[1], TABLE_Z + dims_hwl[0] / 2.0)
    T = sd.look_at(EYE, look)
    depth = sd.render(T, list(shapes) + [TABLE])
    if noise is not None:
        rng, sigma = noise
        depth = np.where(depth > 0, depth + rng.normal(0.0, sigma, depth.shape), 0.0)
    bbox = sd.bbox_of(T, box_corners)
    pts = footprint.transform(footprint.backproject(depth, bbox, sd.K), T)
    guess = (centre[0] + centre_error[0], centre[1] + centre_error[1])
    pts = footprint.object_points(pts, TABLE_Z, center_xy=guess,
                                  radius=gp.cut_radius(dims_hwl))
    bearing = math.atan2(guess[1], guess[0])
    return gp.plan_from_points(pts, TABLE_Z, bearing, dims_hwl, max_dev=max_dev)


def cylinder(cx, cy, r, h):
    return (sd.Cylinder(cx, cy, r, TABLE_Z, TABLE_Z + h),
            sd.corners(cx, cy, r, r, TABLE_Z, TABLE_Z + h))


def box(cx, cy, lx, ly, h, yaw=0.0):
    half = math.hypot(lx, ly) / 2.0 if yaw else None
    c = (sd.corners(cx, cy, half, half, TABLE_Z, TABLE_Z + h) if yaw
         else sd.corners(cx, cy, lx / 2, ly / 2, TABLE_Z, TABLE_Z + h))
    return sd.Box(cx, cy, lx, ly, TABLE_Z, TABLE_Z + h, yaw), c


def assert_width(measured, true_width):
    """A single view measures the silhouette to the last object pixel, up to a
    pixel short on each side (0.76 mm at 0.45 m): never wider, at most ~3
    pixels narrower."""
    assert -0.0025 < measured - true_width < 0.0005, (measured, true_width)


def grasp_xy_error(plan, centre):
    return math.hypot(plan["grasp_point"][0] - centre[0], plan["grasp_point"][1] - centre[1])


def test_bottle_body():
    """68 mm cylinder, like the bottle's body: width from the silhouette (a
    single view measures the chord between the tangent points, a hair under
    the diameter), grip at the anchor height, jaws on the centre."""
    c = (0.45, 0.02)
    shape, corners = cylinder(*c, 0.034, 0.25)
    plan, diag = plan_scene([shape], corners, c, [0.25, 0.068, 0.068])
    assert plan is not None, diag
    assert_width(diag["width"], 0.068)
    assert grasp_xy_error(plan, c) < 0.003, (plan["grasp_point"], c)
    anchor = gg.grasp_height_above_table([0.25, 0.068, 0.068])
    assert abs(diag["band_z"] - anchor) <= 0.005, (diag["band_z"], anchor)
    assert abs(plan["grasp_point"][2] - (TABLE_Z + diag["band_z"])) < 1e-9
    assert diag["quality"] > 0.97, diag["quality"]
    assert not plan["full_close"]
    assert abs(plan["gripper_close"] - (diag["width"] / 2 - 0.006)) < 1e-9
    assert plan["gripper_open"] == gg.FINGER_OPEN


def test_box_face_on():
    """Only the front face is visible; the sides the jaws close on are not.
    Completing the slice by reflection puts them back, and the depth prior (L)
    puts the grasp point at the box's centre instead of on its front face."""
    c = (0.45, 0.0)
    shape, corners = box(*c, 0.06, 0.06, 0.09)
    plan, diag = plan_scene([shape], corners, c, [0.09, 0.06, 0.06])
    assert plan is not None, diag
    assert diag["visible_depth"] < 0.005, diag["visible_depth"]   # it really is one face
    assert_width(diag["width"], 0.06)
    assert grasp_xy_error(plan, c) < 0.003, (plan["grasp_point"], c)
    assert diag["quality"] > 0.99, diag["quality"]
    assert diag["axis_dev"] == 0.0
    assert gg.GRIP_Z_FLOOR <= diag["band_z"] <= 0.09 - gg.GRIP_TOP_MARGIN


def _turned_box(side, max_dev):
    c = (0.45, 0.0)
    shape, corners = box(*c, side, side, 0.09, yaw=math.radians(30))
    plan, diag = plan_scene([shape], corners, c, [0.09, side, side], max_dev=max_dev)
    return c, plan, diag


def _check_face_grasp(c, plan, diag, side):
    """The jaws close within a few degrees of the box's face normal (the box
    is turned 30°): the axis search runs in 2° steps, and a couple of degrees
    off a face only turns the box that much as the jaws seat."""
    assert plan is not None, diag
    assert diag["turn"] <= math.radians(3.0), math.degrees(diag["turn"])
    _, _, yaw = _euler_z(plan["grasp_orientation"])
    bearing = math.atan2(-0.005, 0.46)
    assert abs(yaw - (bearing + diag["axis_dev"])) < 1e-6
    off = math.radians(3.0)
    assert abs(yaw - math.radians(30)) <= off, math.degrees(yaw)
    assert side - 0.0025 < diag["width"] < side * (math.cos(off) + math.sin(off)) + 0.0005, \
        diag["width"]
    assert grasp_xy_error(plan, c) < 0.004, (plan["grasp_point"], c)


def test_turned_box_would_turn_in_the_jaws():
    """A 40 mm box turned 30°: with the closing axis fixed across the approach
    the jaws land on two corners and would turn the box 30° before a face
    seats (more than MAX_SEAT_TURN), so the planner refuses. Allowed to turn
    the axis, it closes across the faces. The centre comes from the
    silhouette corners, not the depth prior."""
    c, plan, diag = _turned_box(0.04, 0.0)
    assert plan is None, diag
    assert re.search(r"would turn (29|30|31)°", diag["reason"]), diag["reason"]
    c, plan, diag = _turned_box(0.04, math.radians(30))
    _check_face_grasp(c, plan, diag, 0.04)


def test_turned_tea_box_fits_only_across_its_faces():
    """The tea box turned 30° is 0.06 (cos 30° + sin 30°) = 82 mm across the
    approach — wider than the jaws open. Across its faces it is 60 mm."""
    c, plan, diag = _turned_box(0.06, 0.0)
    assert plan is None, diag
    assert "too wide" in diag["reason"], diag["reason"]
    c, plan, diag = _turned_box(0.06, math.radians(30))
    _check_face_grasp(c, plan, diag, 0.06)


def test_too_wide_for_the_jaws():
    """80 mm: the jaws open to 74 mm. The old fit gate (GRIPPER_W_M + 0.03 =
    110 mm) called this graspable."""
    c = (0.45, 0.0)
    shape, corners = cylinder(*c, 0.04, 0.25)
    plan, diag = plan_scene([shape], corners, c, [0.25, 0.08, 0.08])
    assert plan is None
    assert "too wide" in diag["reason"], diag["reason"]


def test_neck_above_the_grip_closes_fully():
    """Body 60 mm up to 95 mm, then a 24 mm neck: the grip band sits just
    below the narrowing, so the jaws close fully rather than stopping at the
    body's width (they could not follow the object if it slipped)."""
    c = (0.45, 0.0)
    body = sd.Cylinder(*c, 0.03, TABLE_Z, TABLE_Z + 0.095)
    neck = sd.Cylinder(*c, 0.012, TABLE_Z + 0.095, TABLE_Z + 0.25)
    corners = sd.corners(*c, 0.03, 0.03, TABLE_Z, TABLE_Z + 0.25)
    plan, diag = plan_scene([body, neck], corners, c, [0.25, 0.06, 0.06])
    assert plan is not None, diag
    assert diag["band_z"] < 0.095
    assert_width(diag["width"], 0.06)
    assert plan["full_close"] and diag["tapered"]
    assert plan["gripper_close"] == gg.FINGER_CLOSE


def test_uniform_can_is_not_tapered():
    c = (0.44, -0.03)
    shape, corners = cylinder(*c, 0.0275, 0.096)
    plan, diag = plan_scene([shape], corners, c, [0.096, 0.055, 0.055])
    assert plan is not None, diag
    assert not plan["full_close"]
    assert_width(diag["width"], 0.055)


def test_too_short_for_a_side_grasp():
    c = (0.45, 0.0)
    shape, corners = box(*c, 0.06, 0.06, 0.07)
    plan, diag = plan_scene([shape], corners, c, [0.07, 0.06, 0.06])
    assert plan is None
    assert "too short" in diag["reason"], diag["reason"]


def test_neighbour_in_the_detection_box_is_cut():
    """A can 10 cm to the side shares the detection box (bbox of both); the
    radius cut (68 mm for the bottle) keeps it out of the bottle's slices."""
    c = (0.45, 0.02)
    bottle, b_corners = cylinder(*c, 0.034, 0.25)
    can, c_corners = cylinder(c[0] + 0.01, c[1] + 0.10, 0.0275, 0.096)
    alone, _ = plan_scene([bottle], b_corners, c, [0.25, 0.068, 0.068])
    both, diag = plan_scene([bottle, can], b_corners + c_corners, c, [0.25, 0.068, 0.068])
    assert both is not None, diag
    assert_width(diag["width"], 0.068)
    assert np.allclose(both["grasp_point"], alone["grasp_point"], atol=0.001)


def test_depth_noise_does_not_change_the_decision():
    """2 mm of depth noise (a real depth camera at half a metre; Gazebo's has
    none) on the bottle and on the turned 40 mm box: the bottle is still
    planned with its width and centre, the box still refused. Face quality
    taken from hull edges failed this: noise puts millimetre edges on the hull
    facing every direction."""
    c = (0.45, 0.02)
    bottle, b_corners = cylinder(*c, 0.034, 0.25)
    box40, x_corners = box(0.45, 0.0, 0.04, 0.04, 0.09, yaw=math.radians(30))
    for seed in range(3):
        rng = np.random.default_rng(seed)
        plan, diag = plan_scene([bottle], b_corners, c, [0.25, 0.068, 0.068], noise=(rng, 0.002))
        assert plan is not None, diag
        assert abs(diag["width"] - 0.068) < 0.002 and grasp_xy_error(plan, c) < 0.005
        plan, diag = plan_scene([box40], x_corners, (0.45, 0.0), [0.09, 0.04, 0.04],
                                noise=(rng, 0.002))
        assert plan is None and "would turn" in diag["reason"], diag


def test_no_points_degrades_to_none():
    plan, diag = gp.plan_from_points(np.empty((0, 3)), TABLE_Z, 0.0, [0.25, 0.068, 0.068])
    assert plan is None and "no points" in diag["reason"], diag


def test_bands_measure_width_across_the_approach():
    """footprint.bands on its own: a 60x30 mm box seen along its long side has
    lateral width 60 mm whichever way it is approached from."""
    c = (0.45, 0.10)
    shape, corners = box(*c, 0.03, 0.06, 0.12)
    T = sd.look_at(EYE, (c[0], c[1], TABLE_Z + 0.06))
    depth = sd.render(T, [shape, TABLE])
    pts = footprint.transform(footprint.backproject(depth, sd.bbox_of(T, corners), sd.K), T)
    pts = footprint.object_points(pts, TABLE_Z)
    bearing = math.atan2(c[1], c[0])
    bands = footprint.bands(pts, TABLE_Z, bearing)
    assert len(bands) > 15
    mid = [b for b in bands if 0.03 < b.z < 0.09]
    # Seen from 12.5° off its axis the silhouette is a bit wider than 60 mm:
    # 0.06 cos 12.5° + 0.03 sin 12.5° = 65 mm. That is what the jaws close over.
    expected = 0.06 * math.cos(bearing) + 0.03 * math.sin(bearing)
    for b in mid:
        assert -0.003 < b.width - expected < 0.0005, (b.z, b.width, expected)


def _euler_z(q):
    x, y, z, w = q
    return (0.0, 0.0, math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL PASS")
