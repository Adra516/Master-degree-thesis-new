"""The 2026-10-05 fix: a square-footprint box (tea_box, 45x45mm) is NOT the
same thing as a round object, even though both can have footprint ratio 1.0.
Confirmed live by ground truth: a baseline-planned tea_box grasp at the wrong
yaw caught a corner and toppled it 45-67 degrees in roll/pitch while being
lifted. Two things had to change together, or the fix is a no-op for exactly
the object it targets:

1. locobot_geometry.estimate_grasp_orientation must take the catalogue's own
   `round` flag (config/objects.yaml) instead of inferring "round" from the
   footprint being square — the old inference was wrong for any square but
   non-round object.
2. grasp_geometry.baseline_plan/grasp_pose must actually use that refined
   orientation for the gripper's yaw, instead of always facing the robot's
   bearing to the object (atan2(y, x)), which only coincidentally lines up
   with a box's real faces.
"""
import math
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import grasp_geometry as gg  # noqa: E402
from locobot_geometry import estimate_grasp_orientation  # noqa: E402
from resolver import ObjectCatalog  # noqa: E402

CATALOG_PATH = os.path.join(_HERE, "..", "config", "objects.yaml")


def _rotated_square_image(angle_deg, size=400, side=80):
    """A filled white square, rotated by angle_deg, on a black background —
    a synthetic silhouette with a KNOWN true rotation, for the Canny-edge
    fallback path (no depth needed)."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx = cy = size // 2
    box = cv2.boxPoints(((cx, cy), (side, side), angle_deg)).astype(np.int32)
    cv2.fillConvexPoly(img, box, (255, 255, 255))
    return img


def _angles_equivalent_mod_90(a_deg, b_deg, tol_deg=3.0):
    """True when a and b describe the same face-alignment of a SQUARE
    cross-section — any of a, a+90, a-90, a+180... is an equally valid
    closing axis, since opposite faces are identical."""
    diff = (a_deg - b_deg + 45.0) % 90.0 - 45.0
    return abs(diff) < tol_deg


# ── The catalogue's own round flag ───────────────────────────────────────────

def test_catalog_round_flag():
    catalog = ObjectCatalog.from_yaml(CATALOG_PATH)
    assert catalog.is_round("bottle") is True
    assert catalog.is_round("can_coke") is True
    assert catalog.is_round("ball") is True
    assert catalog.is_round("tea_box") is False   # square, NOT round
    assert catalog.is_round("table") is False


# ── estimate_grasp_orientation: catalogue override vs. the old dims-ratio guess ──

def test_round_true_short_circuits_regardless_of_image():
    img = _rotated_square_image(37.0)   # a real rotation present in the image
    bbox = [100, 100, 300, 300]
    yaw = estimate_grasp_orientation(img, bbox, dims_hwl=[0.09, 0.045, 0.045],
                                     symmetric=True)
    assert yaw == 0.0


def test_square_but_not_round_is_measured_not_zeroed():
    """The specific bug: tea_box's footprint ratio is 1.0, same as a round
    object's, but it is NOT round — a real rotation must come back, not the
    old hardcoded 0.0."""
    for true_angle in (20.0, -15.0, 40.0):
        img = _rotated_square_image(true_angle)
        bbox = [100, 100, 300, 300]
        yaw = estimate_grasp_orientation(img, bbox, dims_hwl=[0.09, 0.045, 0.045],
                                         symmetric=False)
        assert yaw != 0.0, "a real rotation was zeroed out"
        assert _angles_equivalent_mod_90(math.degrees(yaw), true_angle), \
            (true_angle, math.degrees(yaw))


def test_uncatalogued_square_keeps_old_behaviour():
    """symmetric=None (no catalogue entry) — unchanged from before this fix:
    falls back to the footprint-ratio guess, which zeroes a square."""
    img = _rotated_square_image(37.0)
    bbox = [100, 100, 300, 300]
    yaw = estimate_grasp_orientation(img, bbox, dims_hwl=[0.09, 0.045, 0.045],
                                     symmetric=None)
    assert yaw == 0.0


def test_round_object_still_short_circuits_via_dims_when_uncatalogued():
    """A genuinely round, uncatalogued object (symmetric=None) still gets the
    old dims-ratio-based shortcut — this fix must not regress that case."""
    img = _rotated_square_image(37.0)
    bbox = [100, 100, 300, 300]
    yaw = estimate_grasp_orientation(img, bbox, dims_hwl=[0.23, 0.062, 0.062],
                                     symmetric=None)
    assert yaw == 0.0


# ── baseline_plan / grasp_pose: orientation actually changes the commanded yaw ──

def test_grasp_pose_uses_orientation_when_given():
    xyz, quat = gg.grasp_pose((0.45, 0.10), [0.09, 0.045, 0.045], 0.25,
                              orientation=math.radians(30.0))
    yaw = 2.0 * math.atan2(quat[2], quat[3])   # z/w -> yaw for a pure-yaw quaternion
    assert math.isclose(yaw, math.radians(30.0), abs_tol=1e-6)


def test_grasp_pose_falls_back_to_bearing_without_orientation():
    x, y = 0.45, 0.10
    xyz, quat = gg.grasp_pose((x, y), [0.09, 0.045, 0.045], 0.25, orientation=None)
    yaw = 2.0 * math.atan2(quat[2], quat[3])
    assert math.isclose(yaw, math.atan2(y, x), abs_tol=1e-6)


def test_baseline_plan_reason_reflects_which_yaw_source_was_used():
    plan_bearing = gg.baseline_plan((0.45, 0.10), [0.09, 0.045, 0.045], "along_x", 0.25)
    assert "bearing yaw" in plan_bearing["reason"]

    plan_measured = gg.baseline_plan((0.45, 0.10), [0.09, 0.045, 0.045], "along_x", 0.25,
                                     orientation=math.radians(12.0))
    assert "measured yaw" in plan_measured["reason"]
    yaw = 2.0 * math.atan2(plan_measured["grasp_orientation"][2],
                           plan_measured["grasp_orientation"][3])
    assert math.isclose(yaw, math.radians(12.0), abs_tol=1e-6)


def test_baseline_plan_corner_catch_scenario_now_aligns_to_the_box():
    """The actual failing geometry from the live reproduction: robot bearing
    off by a few degrees from the box's true (near-zero) yaw. Without the
    fix the commanded yaw would be the bearing (~a few degrees); with it,
    the box's own measured yaw wins."""
    target_xy = (0.421, -0.033)              # from the live log
    bearing = math.atan2(target_xy[1], target_xy[0])
    assert abs(math.degrees(bearing)) > 2.0   # the mismatch that caused the topple

    box_true_yaw = math.radians(1.0)          # the box's real, near-zero world/base yaw
    plan = gg.baseline_plan(target_xy, [0.09, 0.045, 0.045], "along_x", 0.25,
                            orientation=box_true_yaw)
    yaw = 2.0 * math.atan2(plan["grasp_orientation"][2], plan["grasp_orientation"][3])
    assert math.isclose(yaw, box_true_yaw, abs_tol=1e-6)
    assert not math.isclose(yaw, bearing, abs_tol=1e-6)


def test_square_orientation_snaps_to_the_bearing_reachable_equivalent():
    """The actual failure found live after the first version of this fix
    shipped: minAreaRect's choice between a square's two 90-degree-apart
    equivalent faces is arbitrary, and the 'wrong' one sends the standoff
    ~100mm to the side of the object (unreachable) instead of in front of
    it. -85deg and 90deg are both exactly 90deg from these real bearings
    (-1.2deg, 0.85deg) measured live; the snap must prefer the equivalent
    within the same 90-degree family that is actually close to bearing."""
    cases = [
        (0.421, -0.033, math.radians(-85.0)),
        (0.339, 0.005, math.radians(90.0)),
    ]
    for x, y, far_equivalent in cases:
        bearing = math.atan2(y, x)
        xyz, quat = gg.grasp_pose((x, y), [0.09, 0.045, 0.045], 0.25,
                                  orientation=far_equivalent)
        yaw = 2.0 * math.atan2(quat[2], quat[3])
        # the snapped yaw must be within 45 degrees of bearing...
        diff = abs(((yaw - bearing + math.pi) % (2 * math.pi)) - math.pi)
        assert diff < math.radians(45.0), (math.degrees(yaw), math.degrees(bearing))
        # ...and still physically the same grasp as the original estimate
        # (a multiple of 90 degrees away).
        snapped = abs(((yaw - far_equivalent) % (math.pi / 2.0)))
        assert snapped < 1e-6 or abs(snapped - math.pi / 2.0) < 1e-6


def test_elongated_object_orientation_is_not_snapped():
    """Only a near-square footprint has this extra 90-degree ambiguity — an
    elongated box (bottle-cap-on-its-side-shaped) must keep its measured yaw
    exactly, even when that happens to sit far from bearing."""
    far_from_bearing = math.radians(80.0)
    xyz, quat = gg.grasp_pose((0.45, 0.0), [0.09, 0.03, 0.09], 0.25,
                              orientation=far_from_bearing)   # w=30mm, l=90mm: 3:1
    yaw = 2.0 * math.atan2(quat[2], quat[3])
    assert math.isclose(yaw, far_from_bearing, abs_tol=1e-6)


if __name__ == "__main__":
    import sys as _sys
    mod = _sys.modules[__name__]
    for name in dir(mod):
        if name.startswith("test_"):
            getattr(mod, name)()
            print("ok", name)
    print("ALL PASS")
