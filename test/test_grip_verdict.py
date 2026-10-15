"""grip_verdict: is the object still between the fingers?

Every number here is a logged value, not a made-up fixture: the 10 fresh-launch
bottle picks of 2026-10-13 (ground truth showed none held cleanly for 30 s while
get_object reported success on 7), the same day's tea box / can picks, and the
user's own bottle run. Regression for: get_object reporting success with the
fingers fully open (37.1 mm), past their own limit (64 mm), or sitting at their
commanded close target (nothing between them).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import grasp_geometry as gg  # noqa: E402

MM = 0.001


def held(gw_mm, target_mm, baseline_mm=None):
    base = None if baseline_mm is None else baseline_mm * MM
    return gg.grip_verdict(gw_mm * MM, target_mm * MM, gg.FINGER_OPEN, base)


def test_real_holds_are_not_flagged():
    # (finger position, commanded target) — fingers stopped 3.4-5.3 mm short.
    for gw, tgt in [(28.1, 24.3), (27.1, 22.7), (28.5, 23.2),    # bottle, grasped
                    (30.5, 27.1), (31.9, 27.3)]:                 # tea box, can
        ok, why = held(gw, tgt)
        assert ok, (gw, tgt, why)


def test_fingers_that_reached_their_target_are_empty():
    # B1/B2/B3 of the 10-trial run: all reported success=True.
    for gw, tgt in [(23.0, 22.6), (17.6, 22.6), (14.5, 22.4)]:
        ok, why = held(gw, tgt)
        assert not ok and "target" in why, (gw, tgt, why)


def test_fully_open_fingers_are_empty():
    ok, why = held(37.1, 23.6)     # first fresh trial: success=True with open jaws
    assert not ok and "fully open" in why, why


def test_a_reading_beyond_the_finger_limit_is_not_a_hold():
    ok, why = held(64.0, 23.7)     # the bottle thrown through the jaws
    assert not ok and "beyond" in why, why


def test_full_close_target_below_the_joint_limit_still_detects_empty():
    """A full close commands FINGER_CLOSE (10 mm) but a finger cannot go below its
    15 mm joint limit, so empty fingers rest AT the limit, never at the target."""
    ok, why = gg.grip_verdict(0.0151, gg.FINGER_CLOSE, gg.FINGER_OPEN)
    assert not ok, why
    # An object wider than the limit still stops them short of it.
    ok, why = gg.grip_verdict(0.0185, gg.FINGER_CLOSE, gg.FINGER_OPEN)
    assert ok, why


def test_no_target_means_only_the_open_and_limit_rules_apply():
    assert gg.grip_verdict(0.020, None, gg.FINGER_OPEN) == (True, "")
    assert gg.grip_verdict(0.0371, None, gg.FINGER_OPEN)[0] is False


def test_a_snap_since_the_lift_is_a_lost_grip():
    # Delayed failures: both fingers slammed 28 -> 15 mm in 0.2 s with the arm still.
    ok, why = gg.grip_verdict(0.0210, 0.0150, gg.FINGER_OPEN, baseline_m=0.0280)
    assert not ok and "snapped" in why, why
    # Ordinary settling jitter is not a snap.
    assert gg.grip_verdict(0.0275, 0.0243, gg.FINGER_OPEN, baseline_m=0.0280)[0]


def test_bottle_gone_signature_with_both_fingers_is_empty():
    """Fresh-launch verification trial 2 (2026-10-13): get_object said success=True
    while the bottle was already on the floor. Left finger 23.5 vs a 22.6 target
    (+0.9), right finger 21.7 — 0.9 mm PAST its target. The original 0.8 mm margin
    missed it by 0.1 mm."""
    ok, why = gg.grip_verdict(0.0235, 0.0226, gg.FINGER_OPEN, right_m=0.0217)
    assert not ok and "target" in why, why


def test_the_users_own_run_is_now_called_empty():
    """24.4 mm vs a 23.4 target = +1.0: the bottle was found 16 cm away upright
    within 17 s of success=True. Left finger only (the right was not logged)."""
    ok, why = held(24.4, 23.4)
    assert not ok, why


def test_a_lopsided_real_grip_is_not_called_empty():
    """First fresh trial: left finger 24.7 vs a 23.6 target (+1.1, would read
    'empty' on its own) with the right finger 31.4 — the bottle genuinely gripped,
    friction pinning it so only one finger could close. Needs both fingers."""
    ok, why = gg.grip_verdict(0.0247, 0.0236, gg.FINGER_OPEN, right_m=0.0314)
    assert ok, why


def test_the_gap_between_the_observed_clusters_is_respected():
    """Empties were within 1.0 mm, every real hold >= 3.0 mm short: the margin must
    sit strictly between, or it starts rejecting real holds / missing real empties."""
    assert 1.0 < gg.GRIP_TARGET_REACH_MM < 3.0, gg.GRIP_TARGET_REACH_MM
    # worst real holds seen: both fingers 3.0+ short
    assert gg.grip_verdict(0.0256, 0.0226, gg.FINGER_OPEN, right_m=0.0256)[0]


def test_off_centre_can_is_not_called_fully_open():
    """Can, 2026-10-14: left 33.8 / right 29.3 mm (pad gap 45.1 = the can), can rising
    in the jaws. The left finger alone is within 3 mm of open; the mean is not."""
    ok, why = gg.grip_verdict(0.0338, 0.0270, gg.FINGER_OPEN, right_m=0.0293)
    assert ok, why
    # both fingers really open -> still empty
    assert not gg.grip_verdict(0.0362, 0.0270, gg.FINGER_OPEN, right_m=0.0360)[0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL PASS")
