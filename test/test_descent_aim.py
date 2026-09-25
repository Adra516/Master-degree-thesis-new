"""descent_aim: re-aim the straight-in leg when the base has crept, refuse when far.

Regression for: the arm reached the standoff (2.5 mm) and then stopped, every
run, because the odometry-held object had shifted 15-30 mm in base_link (the base
creeps while the arm holds its poses) and the gate refused anything over 15 mm.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import grasp_geometry as gg  # noqa: E402


def test_within_tolerance_keeps_the_commanded_point():
    aim, verdict, drift = gg.descent_aim((0.452, -0.032), (0.447, -0.030), 15.0)
    assert verdict == "ok" and aim == (0.452, -0.032), (aim, verdict)
    assert 5 < drift < 6


def test_the_logged_25mm_case_is_reaimed_not_refused():
    """Values from the real run: commanded (0.452,-0.032), odometry (0.428,-0.037)."""
    aim, verdict, drift = gg.descent_aim((0.452, -0.032), (0.428, -0.037), 15.0)
    assert verdict == "reaimed", verdict
    assert aim == (0.428, -0.037), aim
    assert 24 < drift < 26, drift


def test_a_large_disagreement_is_still_refused():
    aim, verdict, drift = gg.descent_aim((0.452, -0.032), (0.380, -0.032), 15.0)
    assert verdict == "refused" and aim == (0.452, -0.032), (aim, verdict)
    assert drift > gg.REAIM_MAX_MM


def test_limits_are_inclusive_and_tunable():
    assert gg.descent_aim((0, 0), (0.015, 0), 15.0)[1] == "ok"
    assert gg.descent_aim((0, 0), (0.0451, 0), 15.0)[1] == "refused"
    assert gg.descent_aim((0, 0), (0.0451, 0), 15.0, max_reaim_mm=60.0)[1] == "reaimed"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL PASS")
