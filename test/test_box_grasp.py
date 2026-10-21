"""Offline: the two pure helpers behind the box grasp (grasp_geometry.box_grip_from_width and
lateral_offset). No ROS master, no Gazebo."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import tf.transformations as tft  # noqa: E402

import base_controller as bc  # noqa: E402
import grasp_geometry as gg  # noqa: E402


def test_the_box_grip_opens_fully_and_closes_a_squeeze_inside_the_face_width():
    opening, close = gg.box_grip_from_width(0.045)
    assert opening == gg.FINGER_OPEN
    assert abs(close - (gg.finger_pos_for_width(0.045) - gg.PLANNER_SQUEEZE)) < 1e-9
    assert abs(gg.planned_width_mm(close, False) - 45.0) < 1e-6      # the contact gate recovers 45 mm
    room_mm = (gg.jaw_width(gg.FINGER_OPEN) - 0.045) / 2 * 1000
    assert abs(room_mm - 5.5) < 1e-6                                  # the clearance the box can ever have


def test_the_box_grip_stays_inside_the_fingers_travel():
    for w in (0.0, 0.01, 0.045, 0.2):
        opening, close = gg.box_grip_from_width(w)
        assert gg.FINGER_CLOSE <= close <= gg.FINGER_OPEN - 0.005
        assert gg.FINGER_CLOSE + 0.008 <= opening <= gg.FINGER_OPEN


def test_lateral_offset_is_measured_along_the_hands_closing_axis():
    level = tft.quaternion_from_euler(0.0, 0.0, 0.0)
    e, u = gg.lateral_offset((0.5, 0.02), (0.4, 0.0), level)
    assert abs(e - 0.02) < 1e-9 and u == (0.0, 1.0)                  # left of a hand pointing along +x
    e, _ = gg.lateral_offset((0.5, -0.013), (0.4, 0.0), level)
    assert abs(e + 0.013) < 1e-9
    e, _ = gg.lateral_offset((0.5, 0.0), (0.4, 0.0), level)
    assert e == 0.0                                                   # dead ahead, however far


def test_lateral_offset_ignores_distance_along_the_approach_and_follows_the_hands_yaw():
    yawed = tft.quaternion_from_euler(0.0, 0.0, math.radians(30.0))
    # An aim straight down world +x is 50 mm to the right of a hand yawed 30 deg left, 0.1 m away.
    e, u = gg.lateral_offset((0.5, 0.0), (0.4, 0.0), yawed)
    assert abs(e + 0.05) < 1e-9
    assert abs(math.hypot(*u) - 1.0) < 1e-9
    # The same aim point exactly on the hand's own line is zero whatever the distance.
    ahead = (0.4 + 0.2 * math.cos(math.radians(30)), 0.2 * math.sin(math.radians(30)))
    e, _ = gg.lateral_offset(ahead, (0.4, 0.0), yawed)
    assert abs(e) < 1e-9


def test_the_heading_hold_pushes_back_gently_and_is_capped():
    assert bc.heading_hold_speed(0.0) == 0.0
    assert bc.heading_hold_speed(math.radians(0.04)) == 0.0                 # inside the dead band
    small = bc.heading_hold_speed(math.radians(0.2))
    assert 0.0 < small < bc.HOLD_MAX_SPEED
    assert abs(small - bc.HOLD_GAIN * math.radians(0.2)) < 1e-12            # proportional
    assert bc.heading_hold_speed(math.radians(5.0)) == bc.HOLD_MAX_SPEED    # capped
    assert bc.heading_hold_speed(-math.radians(5.0)) == -bc.HOLD_MAX_SPEED  # and symmetric
    assert bc.heading_hold_speed(-math.radians(0.2)) == -small
    assert bc.HOLD_MAX_SPEED <= 0.05            # never a turn the base makes on purpose (MIN_TURN_SPEED 0.03..)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL PASS")
