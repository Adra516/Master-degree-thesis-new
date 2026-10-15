"""GraspExecutor._outcome: a partially-closed gripper is only a grasp if WE closed it.

Regression for 2026-10-14: the contact gate refused to close (jaws 24 mm off along
the approach) after the jaws had already grazed and welded the bottle; the fingers
rested at 29.7 mm and get_object reported success with the bottle hanging tilted
off the gripper. No close was ever commanded.
"""
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import rospy  # noqa: E402

rospy.loginfo = lambda *a, **k: None
rospy.logwarn = lambda *a, **k: None
rospy.get_param = lambda name, default=None: default
rospy.sleep = lambda *_: None

import grasp_executor as ge  # noqa: E402


def _executor(gw_mm, right_mm=None):
    ex = object.__new__(ge.GraspExecutor)
    ex.gripper = NS(position=gw_mm / 1000.0,
                    right_position=(right_mm if right_mm is not None else gw_mm) / 1000.0,
                    send=lambda *a, **k: None)
    ex.scene = NS(clear=lambda: None)
    ex._pending_grasp = None
    ex.retracted = False
    ex._retract_clear = lambda *a, **k: setattr(ex, "retracted", True)
    return ex


def test_refused_contact_gate_with_partly_closed_fingers_is_not_a_grasp():
    ex = _executor(29.7)
    info = {"phase": "descent", "verified": False,
            "reason": "jaws 24mm off the object along the approach direction"}
    ok, data = ex._outcome(False, 0.037, info)
    assert ok is False, data
    assert ex.retracted
    assert "24mm" in data["reason"]


def test_a_real_close_that_later_failed_planning_still_counts():
    """The old rule's purpose is kept: the fingers closed on something and a LATER
    step (e.g. the retreat path) failed."""
    ex = _executor(28.5, 27.0)
    info = {"phase": "descent", "verified": True, "close_target_m": 0.0232}
    ok, _ = ex._outcome(False, 0.037, info)
    assert ok is True


def test_a_clean_grasp_is_still_a_success():
    ex = _executor(28.5, 27.0)
    ok, _ = ex._outcome(True, 0.037, {"verified": True, "close_target_m": 0.0232})
    assert ok is True


def test_ok_true_but_gate_unverified_is_not_a_success():
    ex = _executor(37.0)
    ok, _ = ex._outcome(True, 0.037, {"verified": False})
    assert ok is False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL PASS")
