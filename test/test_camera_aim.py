"""camera_aim: where the head points and in what order a held object is turned.

Numbers are the geometry measured from TF on 2026-10-19: the camera at (0.087, 0.017, 0.561),
and a bottle held at the retreat pose with its hand at about (0.515, 0.027, 0.634)."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import camera_aim as ca  # noqa: E402


def test_straight_ahead_at_camera_height_is_pan_zero_tilt_zero():
    pan, tilt = ca.aim_angles((0.5, ca.CAMERA_POS[1], ca.CAMERA_POS[2]))
    assert abs(pan) < 1e-9 and abs(tilt) < 1e-9


def test_a_point_below_is_a_positive_tilt_and_a_point_above_negative():
    assert ca.aim_angles((0.5, 0.017, 0.30))[1] > 0.3       # looking down at the table
    assert ca.aim_angles((0.5, 0.017, 0.80))[1] < -0.3      # looking up at a raised hand


def test_the_held_bottle_was_outside_the_view_at_the_table_tilt_and_is_centred_now():
    # With the head at its near tilt (0.45 rad) the hand at z 0.634 sat ~36 deg above the
    # optical axis; the vertical half field of view is atan(240/589) = 22 deg.
    ee = (0.515, 0.027, 0.634)
    pan, tilt = ca.aim_angles(ee)
    off_axis = abs(math.degrees(tilt - 0.45))
    assert off_axis > 22.0, off_axis
    assert abs(math.degrees(tilt)) < 12.0, tilt


def test_pan_follows_the_sign_of_the_lateral_offset():
    assert ca.aim_angles((0.4, 0.2, 0.5))[0] > 0.2
    assert ca.aim_angles((0.4, -0.2, 0.5))[0] < -0.2


def test_angles_are_clamped_to_the_joint_limits():
    pan, tilt = ca.aim_angles((0.0, 2.0, 0.561))          # abeam of the camera
    assert abs(pan) <= ca.PAN_LIMIT + 1e-12
    _, tilt_up = ca.aim_angles((0.09, 0.017, 5.0))        # straight overhead
    assert tilt_up >= ca.TILT_MIN - 1e-12


def test_held_centre_is_below_the_hand_by_the_grip_height_minus_half_the_object():
    # bottle 0.25 tall gripped 0.20 above the table: centre 0.20 - 0.125 = 0.075 below the hand
    c = ca.held_centre((0.5, 0.0, 0.63), grasp_above_table=0.20, height=0.25)
    assert abs(c[2] - (0.63 - 0.075)) < 1e-9
    # unknown size: fixed drop; a grip above the object's centre never raises the aim point
    assert ca.held_centre((0.5, 0.0, 0.63))[2] == 0.63 - ca.DEFAULT_CENTRE_DROP_M
    assert ca.held_centre((0.5, 0.0, 0.63), 0.05, 0.25)[2] == 0.63


def _t(tour):
    return [(p.yaw, p.pitch, p.dx) if p else None for p in tour]


def test_the_tour_visits_each_side_from_the_smallest_turn_out_returning_home_between_sides():
    tour = _t(ca.inspect_tour())
    assert tour == [(-0.35, 0.0, 0.0), (-0.4, -0.3, 0.0), (-0.5, 0.0, 0.0), None,
                    (0.35, 0.0, 0.0), (0.4, -0.3, 0.0), (0.5, 0.0, -0.06), (0.5, 0.0, 0.0), None]


def test_the_tilted_top_poses_are_dropped_unless_asked_for():
    plain = [p for p in ca.inspect_tour(include_tilt=False) if p]
    assert len(plain) == 5 and all(abs(p.pitch) < 1e-9 for p in plain)
    assert len([p for p in ca.inspect_tour(include_tilt=True) if p]) == 7


def test_a_view_cap_keeps_the_highest_priority_poses():
    assert _t(ca.inspect_tour(2)) == [(-0.5, 0.0, 0.0), None, (0.5, 0.0, 0.0), None]
    assert _t(ca.inspect_tour(1)) == [(-0.5, 0.0, 0.0), None]
    assert ca.inspect_tour(0) == []
    assert len([p for p in ca.inspect_tour() if p]) == len(ca.INSPECT_POSES)


def test_every_tour_ends_back_at_the_start():
    for n in range(0, 8):
        tour = ca.inspect_tour(n)
        assert not tour or tour[-1] is None


def test_poses_in_the_tour_stay_inside_what_was_measured_reachable():
    # yaw <= 0.5, pitch <= 0 (the opposite tilt had no IK solution), never higher than the start
    for p in ca.INSPECT_POSES:
        assert abs(p.yaw) <= 0.5 and p.pitch <= 0.0 and p.dz <= 0.0 and p.dx <= 0.0


def test_a_hand_pitch_swings_the_held_centre_about_the_held_point():
    flat = ca.held_centre((0.5, 0.0, 0.63), 0.20, 0.25, pitch=0.0)
    tipped = ca.held_centre((0.5, 0.0, 0.63), 0.20, 0.25, pitch=-0.3)    # top toward the camera
    assert tipped[0] > flat[0] and tipped[2] > flat[2]                    # bottom swings away, centre rises
    assert abs(math.hypot(tipped[0] - 0.5, tipped[2] - 0.63) - 0.075) < 1e-9   # same distance from the hand


def test_head_path_starts_at_the_start_ends_on_the_target_and_is_monotone():
    pts, dt = ca.head_path((0.0, 0.35), (0.4, -0.1))
    assert pts[-1] == (0.4, -0.1)
    pans = [0.0] + [p for p, _ in pts]
    tilts = [0.35] + [t for _, t in pts]
    assert all(b >= a - 1e-12 for a, b in zip(pans, pans[1:]))
    assert all(b <= a + 1e-12 for a, b in zip(tilts, tilts[1:]))
    assert 0.0 < dt <= 0.1


def test_head_path_never_exceeds_the_peak_speed_and_eases_in_and_out():
    pts, dt = ca.head_path((0.0, 0.0), (0.6, 0.0))
    pans = [0.0] + [p for p, _ in pts]
    speeds = [abs(b - a) / dt for a, b in zip(pans, pans[1:])]
    assert max(speeds) <= ca.HEAD_MAX_SPEED * 1.02, max(speeds)     # discretisation margin
    assert speeds[0] < 0.1 * ca.HEAD_MAX_SPEED and speeds[-1] < 0.1 * ca.HEAD_MAX_SPEED
    accel = [abs(b - a) / dt for a, b in zip(speeds, speeds[1:])]
    # minimum-jerk peak acceleration is 5.77 x distance / duration^2 (0.68 rad/s^2 here)
    assert max(accel) < 0.8, max(accel)


def test_a_tiny_re_aim_is_still_eased_over_the_minimum_duration():
    pts, dt = ca.head_path((0.0, 0.0), (0.01, 0.0))
    assert len(pts) * dt >= ca.HEAD_MIN_DURATION - dt
    assert len(pts) >= 8


def test_no_movement_is_a_single_waypoint_on_the_spot():
    pts, dt = ca.head_path((0.2, 0.1), (0.2, 0.1))
    assert pts[-1] == (0.2, 0.1)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL PASS")
