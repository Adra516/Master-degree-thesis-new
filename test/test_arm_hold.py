"""ArmActuator.hold must never re-command a garbage joint read.

Regression for: a read of all-zero joints right before pre-grasp was fed back to
the arm as a "settle hold", swinging it to (0.55, 0, 0.47) through the target.
No ROS master needed: rospy's publisher and sleep are stubbed.
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import rospy  # noqa: E402

sent = []


class _FakePub:
    def __init__(self, *a, **k):
        pass

    def publish(self, traj):
        sent.append(list(traj.points[-1].positions))


rospy.Publisher = _FakePub
rospy.sleep = lambda *_: None
warnings = []
rospy.logwarn = lambda msg, *a: warnings.append(msg % a if a else msg)

import arm_actuators as aa  # noqa: E402


class _Group:
    def __init__(self, values):
        self.values = values

    def get_current_joint_values(self):
        return self.values


def _arm(reading):
    sent.clear()
    warnings.clear()
    return aa.ArmActuator("locobot", _Group(reading))


READY = list(aa.GRASP_READY_POSE)


def test_plausible_reading_is_followed():
    arm = _arm([0.01, -0.99, 1.26, 0.0, -0.24, 0.0])
    arm.send_joints(READY, 0.1)
    sent.clear()
    arm.hold(0.1)
    assert sent == [[0.01, -0.99, 1.26, 0.0, -0.24, 0.0]], sent
    assert not warnings


def test_all_zero_reading_is_not_commanded():
    arm = _arm([0.0] * 6)
    arm.send_joints(READY, 0.1)
    sent.clear()
    arm.hold(0.1)
    assert sent == [READY], sent
    assert any("implausible" in w for w in warnings), warnings


def test_nan_and_wrong_length_are_rejected():
    for bad in ([float("nan")] * 6, [0.0, -1.0, 1.25]):
        arm = _arm(bad)
        arm.send_joints(READY, 0.1)
        sent.clear()
        arm.hold(0.1)
        assert sent == [READY], (bad, sent)


def test_no_prior_command_and_bad_read_moves_nothing():
    arm = _arm([float("nan")] * 6)
    arm.hold(0.1)
    assert sent == [], sent


class _Traj:
    def __init__(self, names, positions):
        pt = type("P", (), {"positions": positions,
                            "time_from_start": type("D", (), {"to_sec": lambda s: 1.3})()})()
        self.joint_trajectory = type("J", (), {"joint_names": names, "points": [pt]})()


def test_published_trajectory_updates_the_reference():
    """After a straight-in move the arm is far from the last send_joints pose; the
    check must compare against where the trajectory ended, not the stale pose."""
    end = [-0.08, -0.12, 0.73, 0.0, -0.58, 0.0]
    arm = _arm(end)
    arm.send_joints(READY, 0.1)
    arm.publish_trajectory(_Traj(aa.ARM_JOINTS, end))
    sent.clear()
    arm.hold(0.1)
    assert sent == [end], sent
    assert not warnings, warnings


def test_trajectory_joint_order_is_mapped_by_name():
    names = list(reversed(aa.ARM_JOINTS))
    end = [-0.08, -0.12, 0.73, 0.0, -0.58, 0.0]
    arm = _arm(end)
    arm.publish_trajectory(_Traj(names, list(reversed(end))))
    assert all(math.isclose(a, b) for a, b in zip(arm.last_target, end)), arm.last_target


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL PASS")
