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


def test_uniform_cylinder():
    """48 mm cylinder: width from the silhouette (a single view measures the
    chord between the tangent points, a hair under the diameter), grip at the
    anchor height, jaws on the centre."""
    c = (0.45, 0.02)
    shape, corners = cylinder(*c, 0.024, 0.25)
    plan, diag = plan_scene([shape], corners, c, [0.25, 0.048, 0.048])
    assert plan is not None, diag
    assert_width(diag["width"], 0.048)
    assert grasp_xy_error(plan, c) < 0.003, (plan["grasp_point"], c)
    anchor = gg.grasp_height_above_table([0.25, 0.048, 0.048])
    assert abs(diag["band_z"] - anchor) <= 0.005, (diag["band_z"], anchor)
    assert abs(plan["grasp_point"][2] - (TABLE_Z + diag["band_z"])) < 1e-9
    assert diag["quality"] > 0.97, diag["quality"]
    assert not plan["full_close"]
    # a uniform body: where the pads touch is the band's own width, to the noise
    # of the widest band within the pad
    assert 0.0 <= diag["contact_width"] - diag["width"] < 0.002
    expected = gg.finger_pos_for_width(diag["contact_width"]) - gp.SQUEEZE
    assert abs(plan["gripper_close"] - expected) < 1e-9
    assert plan["gripper_open"] == gg.FINGER_OPEN


def test_close_target_accounts_for_the_pad_inset():
    """The pads sit gg.PAD_INSET inside the finger joint position (measured in
    Gazebo: a 40 mm block stops the fingers at 28.9 mm, not 20). The close
    target must put the pads SQUEEZE into the object, not SQUEEZE + 9 mm."""
    assert abs(gg.finger_pos_for_width(0.040) - 0.029) < 1e-9
    assert abs(gg.jaw_width(gg.FINGER_OPEN) - 0.056) < 1e-9
    c = (0.45, 0.0)
    shape, corners = cylinder(*c, 0.020, 0.25)
    plan, diag = plan_scene([shape], corners, c, [0.25, 0.040, 0.040])
    assert plan is not None, diag
    pad_depth = gg.jaw_width(gg.finger_pos_for_width(diag["contact_width"])) / 2 \
        - gg.jaw_width(plan["gripper_close"]) / 2
    assert abs(pad_depth - gp.SQUEEZE) < 1e-9, pad_depth


def test_baseline_refuses_a_width_the_jaws_can_never_open_to():
    """baseline_plan() has no per-band search to fall back on — it's one guess
    at the object's mid-height — so unlike the depth planner it must refuse
    outright rather than confidently aim at a width (e.g. a bottle's 64 mm
    body) the 56 mm jaws can never straddle, which is what let that body look
    plausible before gg.PAD_INSET existed (64 < the old, wrong 74 mm max)."""
    plan = gg.baseline_plan((0.45, 0.0), [0.25, 0.064, 0.064], "along_x", 0.25)
    assert not plan["valid"] and "exceeds the jaws" in plan["reason"], plan
    # A width that DOES fit is unaffected.
    plan = gg.baseline_plan((0.45, 0.0), [0.25, 0.028, 0.028], "along_x", 0.25)
    assert plan["valid"], plan


def test_catalogued_can_and_tea_box_are_too_wide_for_the_real_jaws():
    """The jaws open 56 mm pad to pad. The 55 mm coke can would leave 0.5 mm a
    side and the 60 mm tea box does not fit at all — in any orientation."""
    c = (0.45, 0.0)
    can, can_corners = cylinder(*c, 0.0275, 0.096)
    plan, diag = plan_scene([can], can_corners, c, [0.096, 0.055, 0.055])
    assert plan is None and "too wide" in diag["reason"], diag
    for yaw_dev in (0.0, math.radians(30)):
        c, plan, diag = _turned_box(0.06, yaw_dev)
        assert plan is None and "too wide" in diag["reason"], diag


def test_box_face_on():
    """Only the front face is visible; the sides the jaws close on are not.
    Completing the slice by reflection puts them back, and the depth prior (L)
    puts the grasp point at the box's centre instead of on its front face."""
    c = (0.45, 0.0)
    shape, corners = box(*c, 0.045, 0.045, 0.09)
    plan, diag = plan_scene([shape], corners, c, [0.09, 0.045, 0.045])
    assert plan is not None, diag
    assert diag["visible_depth"] < 0.005, diag["visible_depth"]   # it really is one face
    assert_width(diag["width"], 0.045)
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
    """A 34 mm box turned 30°: with the closing axis fixed across the approach
    the jaws land on two corners and would turn the box 30° before a face
    seats (more than MAX_SEAT_TURN), so the planner refuses. Allowed to turn
    the axis, it closes across the faces. The centre comes from the
    silhouette corners, not the depth prior."""
    c, plan, diag = _turned_box(0.034, 0.0)
    assert plan is None, diag
    assert re.search(r"would turn (29|30|31)°", diag["reason"]), diag["reason"]
    c, plan, diag = _turned_box(0.034, math.radians(30))
    _check_face_grasp(c, plan, diag, 0.034)


def test_turned_box_fits_only_across_its_faces():
    """A 44 mm box turned 30° is 0.044 (cos 30° + sin 30°) = 60 mm across the
    approach — wider than the jaws open. Across its faces it is 44 mm."""
    c, plan, diag = _turned_box(0.044, 0.0)
    assert plan is None, diag
    assert "too wide" in diag["reason"], diag["reason"]
    c, plan, diag = _turned_box(0.044, math.radians(30))
    _check_face_grasp(c, plan, diag, 0.044)


def test_too_wide_for_the_jaws():
    """80 mm: the jaws open to 56 mm. The old fit gate (GRIPPER_W_M + 0.03 =
    110 mm) called this graspable."""
    c = (0.45, 0.0)
    shape, corners = cylinder(*c, 0.04, 0.25)
    plan, diag = plan_scene([shape], corners, c, [0.25, 0.08, 0.08])
    assert plan is None
    assert "too wide" in diag["reason"], diag["reason"]


def test_close_squeezes_a_little_past_the_contact_never_fully():
    """Body 48 mm up to 95 mm, then a 10 mm spike (under the jaws' 12 mm
    minimum): the grip band sits just below the narrowing. The jaws are
    commanded SQUEEZE past where the pads touch — not fully closed (10 mm past a
    60 mm body's surface ejected the object in the sim)."""
    c = (0.45, 0.0)
    body = sd.Cylinder(*c, 0.024, TABLE_Z, TABLE_Z + 0.095)
    neck = sd.Cylinder(*c, 0.005, TABLE_Z + 0.095, TABLE_Z + 0.25)
    corners = sd.corners(*c, 0.024, 0.024, TABLE_Z, TABLE_Z + 0.25)
    plan, diag = plan_scene([body, neck], corners, c, [0.25, 0.048, 0.048])
    assert plan is not None, diag
    assert diag["band_z"] < 0.095
    assert_width(diag["width"], 0.048)
    assert diag["tapered"] and not plan["full_close"]
    expected = gg.finger_pos_for_width(diag["contact_width"]) - gp.SQUEEZE
    assert abs(plan["gripper_close"] - expected) < 1e-9
    assert plan["gripper_close"] > gg.FINGER_CLOSE + 0.005


def test_close_target_is_measured_from_where_a_taper_is_touched():
    """On a cone the pad touches at its LOWER edge, wider than the band the grip is
    centred on, so the squeeze is counted from there. Counting it from the centre
    band (the old rule, and a full close before that) commands the fingers
    further in than the intended squeeze — 8-10 mm on the live bottle."""
    c = (0.46, 0.0)
    steps = [(0.0335, 0.000, 0.140)]
    z, r = 0.140, 0.0335
    while z < 0.250:
        z1 = min(0.250, z + 0.005)
        r = max(0.0135, r - 0.00117 * (z1 - z) / 0.005 * 1.0)   # ~0.234 mm/mm of diameter
        steps.append((r, z, z1)); z = z1
    shapes = [sd.Cylinder(*c, rr, TABLE_Z + lo, TABLE_Z + hi) for rr, lo, hi in steps]
    corners = sd.corners(*c, 0.0335, 0.0335, TABLE_Z, TABLE_Z + 0.25)
    plan, diag = plan_scene(shapes, corners, c, [0.25, 0.067, 0.067])
    assert plan is not None, diag
    assert diag["contact_width"] > diag["width"] + 0.003, (diag["contact_width"], diag["width"])
    assert plan["gripper_close"] > gg.FINGER_CLOSE, plan["gripper_close"]
    interference = gg.finger_pos_for_width(diag["contact_width"]) - plan["gripper_close"]
    assert interference <= gp.SQUEEZE + 1e-9, interference


def test_neck_above_a_too_wide_body_is_reachable():
    """A real bottle-shaped case: an 80 mm body (wider than the jaws can ever
    open around, even before clearance) up to 130 mm, then a 46 mm neck the
    jaws fit easily. The body alone would make the whole object ungraspable
    under a search limited to a narrow window around the formula's guessed
    (mid-body) anchor height — this is the regression test for that bug: the
    real per-band measurement must be free to look past the body and find the
    neck, not be blind to anything outside a guessed height."""
    c = (0.45, 0.02)
    body = sd.Cylinder(*c, 0.04, TABLE_Z, TABLE_Z + 0.13)
    neck = sd.Cylinder(*c, 0.023, TABLE_Z + 0.13, TABLE_Z + 0.23)
    corners = sd.corners(*c, 0.04, 0.04, TABLE_Z, TABLE_Z + 0.23)
    plan, diag = plan_scene([body, neck], corners, c, [0.23, 0.08, 0.08])
    assert plan is not None, diag
    assert diag["band_z"] > 0.13, diag   # in the neck, not the too-wide body
    assert_width(diag["width"], 0.046)
    assert not plan["full_close"]        # uniform there, no taper just above it
    assert grasp_xy_error(plan, c) < 0.003, (plan["grasp_point"], c)


def test_fitting_body_loses_to_a_roomier_neck():
    """A bottle scaled to fit the real 56 mm jaws: a 50 mm body up to 165 mm
    (fits, but with only 3 mm a side), then a shoulder and a neck narrowing to
    27 mm. The body FITS, so the "too wide" test never covers this: the planner
    has to prefer the neck for its room, not only take it when the body is
    measured over the limit. With the formula's height guess weighted 0.5 the
    body outscored the neck and the arm gripped the body every run."""
    c = (0.45, 0.0)
    steps = [(0.0250, 0.000, 0.165), (0.0233, 0.165, 0.170), (0.0218, 0.170, 0.175),
             (0.0195, 0.175, 0.180), (0.0173, 0.180, 0.186), (0.0150, 0.186, 0.196),
             (0.0135, 0.196, 0.233)]
    shapes = [sd.Cylinder(*c, r, TABLE_Z + lo, TABLE_Z + hi) for r, lo, hi in steps]
    corners = sd.corners(*c, 0.025, 0.025, TABLE_Z, TABLE_Z + 0.233)
    plan, diag = plan_scene(shapes, corners, c, [0.233, 0.050, 0.050])
    assert plan is not None, diag
    assert diag["band_z"] > 0.165, ("gripped the body", diag["band_z"], diag["width"])
    clearance = (gp.JAW_MAX - diag["width"]) / 2.0
    assert clearance >= 0.008, ("not roomier than the body's ~4 mm", clearance)
    assert gg.GRIP_Z_FLOOR <= diag["band_z"] <= 0.233 - gg.GRIP_TOP_MARGIN
    assert grasp_xy_error(plan, c) < 0.004, (plan["grasp_point"], c)


def test_pad_height_is_checked_not_just_one_band():
    """A band whose own width fits can still be a bad grip if the 29 mm-tall jaw
    pad also spans a much wider part: a 46 mm neck starting 5 mm above an 80 mm
    body has a fitting band right above the body, with the pad overlapping it."""
    c = (0.45, 0.0)
    body = sd.Cylinder(*c, 0.040, TABLE_Z, TABLE_Z + 0.130)
    neck = sd.Cylinder(*c, 0.023, TABLE_Z + 0.130, TABLE_Z + 0.230)
    corners = sd.corners(*c, 0.04, 0.04, TABLE_Z, TABLE_Z + 0.23)
    plan, diag = plan_scene([body, neck], corners, c, [0.23, 0.08, 0.08])
    assert plan is not None, diag
    assert diag["band_z"] - 0.130 >= gp.PAD_HALF + gp.PAD_MARGIN - 0.0051, \
        ("pad would overlap the 80 mm body", diag["band_z"])


def test_uniform_can_is_not_tapered():
    c = (0.44, -0.03)
    shape, corners = cylinder(*c, 0.0225, 0.096)
    plan, diag = plan_scene([shape], corners, c, [0.096, 0.045, 0.045])
    assert plan is not None, diag
    assert not plan["full_close"]
    assert_width(diag["width"], 0.045)


def test_too_short_for_a_side_grasp():
    c = (0.45, 0.0)
    shape, corners = box(*c, 0.06, 0.06, 0.07)
    plan, diag = plan_scene([shape], corners, c, [0.07, 0.06, 0.06])
    assert plan is None
    assert "too short" in diag["reason"], diag["reason"]


def test_neighbour_in_the_detection_box_is_cut():
    """A can 10 cm to the side shares the detection box (bbox of both); the
    radius cut keeps it out of the target's slices."""
    c = (0.45, 0.02)
    bottle, b_corners = cylinder(*c, 0.024, 0.25)
    can, c_corners = cylinder(c[0] + 0.01, c[1] + 0.10, 0.0275, 0.096)
    alone, _ = plan_scene([bottle], b_corners, c, [0.25, 0.048, 0.048])
    both, diag = plan_scene([bottle, can], b_corners + c_corners, c, [0.25, 0.048, 0.048])
    assert both is not None, diag
    assert_width(diag["width"], 0.048)
    assert np.allclose(both["grasp_point"], alone["grasp_point"], atol=0.001)


def test_depth_noise_does_not_change_the_decision():
    """2 mm of depth noise (a real depth camera at half a metre; Gazebo's has
    none) on a 48 mm cylinder and on the turned 34 mm box: the cylinder is still
    planned with its width and centre, the box still refused. Face quality
    taken from hull edges failed this: noise puts millimetre edges on the hull
    facing every direction."""
    c = (0.45, 0.02)
    bottle, b_corners = cylinder(*c, 0.024, 0.25)
    box34, x_corners = box(0.45, 0.0, 0.034, 0.034, 0.09, yaw=math.radians(30))
    for seed in range(3):
        rng = np.random.default_rng(seed)
        plan, diag = plan_scene([bottle], b_corners, c, [0.25, 0.048, 0.048], noise=(rng, 0.002))
        assert plan is not None, diag
        assert abs(diag["width"] - 0.048) < 0.002 and grasp_xy_error(plan, c) < 0.005
        plan, diag = plan_scene([box34], x_corners, (0.45, 0.0), [0.09, 0.034, 0.034],
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


def test_table_height_is_measured_not_assumed():
    """The prior said 0.250; the table the camera sees is at 0.272 (the real
    sim). The support surface next to the target must come out at the table's,
    not the prior's — and not the target's own top or a neighbour's."""
    real_z = 0.272
    c = (0.50, 0.0)
    table = sd.Table(real_z, 0.30, 1.00, -0.50, 0.50)
    bottle = sd.Cylinder(*c, 0.034, real_z, real_z + 0.253)
    neighbour = sd.Box(0.55, 0.14, 0.08, 0.08, real_z, real_z + 0.10)
    T = sd.look_at(EYE, (c[0], c[1], real_z + 0.13))
    depth = sd.render(T, [bottle, neighbour, table])
    bbox = sd.bbox_of(T, sd.corners(*c, 0.034, 0.034, real_z, real_z + 0.253))
    got = footprint.table_height(depth, bbox, sd.K, T, c, 0.068, guess_z=0.250)
    assert got is not None
    assert abs(got - real_z) < 0.004, (got, real_z)


def test_table_height_declines_with_nothing_to_measure():
    """No depth in the strip (or nothing near the prior): None, so the caller
    keeps the prior instead of trusting an empty histogram."""
    c = (0.50, 0.0)
    T = sd.look_at(EYE, (c[0], c[1], 0.38))
    empty = np.zeros((sd.H, sd.W))
    assert footprint.table_height(empty, [280, 150, 360, 330], sd.K, T, c, 0.068, 0.25) is None
    far_table = sd.Table(0.60, 0.30, 1.00, -0.50, 0.50)   # nowhere near the 0.25 +/- 0.08 prior
    depth = sd.render(T, [far_table])
    assert footprint.table_height(depth, [280, 150, 360, 330], sd.K, T, c, 0.068, 0.25) is None


def test_planner_uses_the_true_table_when_the_prior_is_low():
    """With the table really at 0.272 the planner must work in heights above THAT:
    the same bottle, planned once with the true table height and once with the
    0.250 prior, must be gripped at the same absolute height (the depth is what
    decides it), and its band heights must be measured from the true table."""
    real_z = 0.272
    c = (0.45, 0.0)
    steps = [(0.0333, 0.000, 0.165), (0.0310, 0.165, 0.170), (0.0290, 0.170, 0.175),
             (0.0260, 0.175, 0.180), (0.0230, 0.180, 0.186), (0.0200, 0.186, 0.196),
             (0.0180, 0.196, 0.233)]
    shapes = [sd.Cylinder(*c, r, real_z + lo, real_z + hi) for r, lo, hi in steps]
    corners = sd.corners(*c, 0.0333, 0.0333, real_z, real_z + 0.233)
    table = sd.Table(real_z, 0.30, 1.00, -0.50, 0.50)
    T = sd.look_at(EYE, (c[0], c[1], real_z + 0.1165))
    depth = sd.render(T, shapes + [table])
    bbox = sd.bbox_of(T, corners)
    dims = [0.233, 0.066, 0.066]
    cloud = footprint.transform(footprint.backproject(depth, bbox, sd.K), T)
    bearing = math.atan2(c[1], c[0])
    measured = footprint.table_height(depth, bbox, sd.K, T, c, gp.cut_radius(dims), 0.250)
    assert measured is not None and abs(measured - real_z) < 0.004, measured
    pts = footprint.object_points(cloud, measured, center_xy=c, radius=gp.cut_radius(dims))
    plan, diag = gp.plan_from_points(pts, measured, bearing, dims)
    assert plan is not None, diag
    # the reported band height is above the TRUE table: absolute z = table + band
    assert abs(plan["grasp_point"][2] - (measured + diag["band_z"])) < 1e-6
    assert diag["band_z"] > 0.165


def test_neck_grip_clears_the_shoulder_with_the_measured_aim_error():
    """The bottle's width profile as the depth camera measured it live (heights above
    the true table). At 178 mm the planner used to grip with a pad that reached
    down to the 46-53 mm shoulder: room on paper (~10 mm/side) but none once the
    7-10 mm aim error the contact check measures is taken off it — the lower finger
    corner grazed the bottle. The chosen band must leave the aim error on each side
    of the WHOLE pad: with the real 56 mm jaws this bottle offers ~12 mm a side at
    best, so 10 mm (the worst aim error ever measured, 9.4 mm) is the bar."""
    c = (0.46, 0.0)
    profile = [(0.000, 0.0665), (0.130, 0.0665), (0.138, 0.0651), (0.143, 0.0630),
               (0.148, 0.0587), (0.152, 0.0531), (0.158, 0.0459), (0.162, 0.0404),
               (0.168, 0.0385), (0.173, 0.0380), (0.178, 0.0370), (0.182, 0.0353),
               (0.188, 0.0348), (0.192, 0.0331), (0.198, 0.0332), (0.202, 0.0317),
               (0.208, 0.0300), (0.212, 0.0295), (0.218, 0.0280), (0.222, 0.0271),
               (0.228, 0.0265), (0.253, 0.0265)]
    shapes = []
    for (z0, w0), (z1, _) in zip(profile, profile[1:]):
        shapes.append(sd.Cylinder(*c, w0 / 2.0, TABLE_Z + z0, TABLE_Z + z1))
    corners = sd.corners(*c, 0.0333, 0.0333, TABLE_Z, TABLE_Z + 0.253)
    plan, diag = plan_scene(shapes, corners, c, [0.253, 0.067, 0.067])
    assert plan is not None, diag
    lower_edge = diag["band_z"] - gp.PAD_HALF - gp.PAD_MARGIN
    widest_under_pad = max(w for (z, w) in profile if z >= lower_edge - 0.005) if lower_edge > 0 else 0.0665
    room = (gp.JAW_MAX - widest_under_pad) / 2.0
    assert diag["band_z"] >= 0.185, ("still gripping low, on the shoulder", diag["band_z"])
    assert room >= 0.010, ("less room than the aim error needs", room, diag["band_z"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL PASS")
