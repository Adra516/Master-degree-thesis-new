"""contact_verdict's width-aware straddle gate + follow_reference.

Values are logged ones (2026-10-13). The user's own run: the jaws closed on a bottle
with along-jaw +14.8 mm and the forearm_roll was thrown to its -pi limit. A fresh
launch with -10.1 mm threw the bottle 227 mm in 0.4 s (ground truth: undisturbed to
within 0.4 mm until the instant the fingers started to close). Every descent within
+-2.6 mm was fine. The old gate only asked whether the object's centre lay between
the fingertips (|along| < gap/2 = 28 mm), so it passed both.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import grasp_geometry as gg  # noqa: E402


def metrics(along, gap=55.7, miss=0.0, straddled=True):
    return {"along_jaw_mm": along, "in_plane_miss_mm": miss, "straddled": straddled,
            "jaw_gap_mm": gap}


BOTTLE_W = gg.planned_width_mm(0.0230, False)     # the user's close target: 23.0 mm


def test_the_users_run_is_refused_not_verified():
    assert 35.5 < BOTTLE_W < 36.5, BOTTLE_W        # 36 mm contact width, room 9.9 mm
    ok, why = gg.contact_verdict(metrics(+14.8), 15.0, BOTTLE_W)
    assert not ok and "room" in why, why


def test_the_thrown_bottle_case_is_refused():
    ok, why = gg.contact_verdict(metrics(-10.1, gap=55.0), 15.0, BOTTLE_W)
    assert not ok, why


def test_every_good_arrival_still_passes():
    # +-2.6 mm was the worst clean one (tea box); bottle clean ones were <= 1.8 mm.
    for along in (0.2, -1.1, 1.8, 0.7):
        ok, why = gg.contact_verdict(metrics(along), 15.0, BOTTLE_W)
        assert ok, (along, why)
    tea_w = gg.planned_width_mm(0.0271, False)      # tea box close target 27.1 mm
    ok, why = gg.contact_verdict(metrics(2.6, gap=55.3), 15.0, tea_w)
    assert ok, why


def test_without_a_width_the_old_rule_applies():
    """No plan width (full close / unknown): fall back to straddled-only, so this
    never refuses something the previous gate accepted for want of data."""
    assert gg.contact_verdict(metrics(+14.8), 15.0, None)[0]
    assert gg.contact_verdict(metrics(+14.8), 15.0, None)[1].startswith("ok")


def test_full_close_plans_have_no_width():
    assert gg.planned_width_mm(gg.FINGER_CLOSE, True) is None


def test_an_object_wider_than_the_gap_is_left_to_the_planner():
    # room <= 0: cannot straddle at all; refusing it is the planner's job, not this gate's.
    ok, _ = gg.contact_verdict(metrics(1.0, gap=50.0), 15.0, 56.0)
    assert ok


def test_the_old_checks_still_win_first():
    assert not gg.contact_verdict(metrics(+24.4, straddled=False), 15.0, BOTTLE_W)[0]
    assert not gg.contact_verdict(metrics(0.5, miss=20.0), 15.0, BOTTLE_W)[0]


def test_follow_reference_applies_the_creep_since_the_aim_was_taken():
    # aim (0.410, 0.008, 0.468); the reference then moved -12 mm in y and +1 mm in z.
    aim, verdict, (dxy, dz) = gg.follow_reference(
        (0.410, 0.008, 0.468), (0.430, 0.020, 0.470), (0.430, 0.008, 0.471))
    assert verdict == "followed", verdict
    assert abs(aim[1] - (-0.004)) < 1e-9 and abs(aim[0] - 0.410) < 1e-9, aim
    assert abs(aim[2] - 0.469) < 1e-9, aim
    assert 11.5 < dxy < 12.5, dxy


def test_follow_reference_ignores_noise_and_refuses_jumps():
    aim = (0.410, 0.008, 0.468)
    r0 = (0.430, 0.020, 0.470)
    assert gg.follow_reference(aim, r0, (0.4305, 0.0205, 0.4705))[1] == "ok"
    out, verdict, _ = gg.follow_reference(aim, r0, (0.500, 0.020, 0.470))
    assert verdict == "refused" and out == aim, (out, verdict)
    out, verdict, _ = gg.follow_reference(aim, None, r0)
    assert verdict == "unavailable" and out == aim
    out, verdict, _ = gg.follow_reference(aim, r0, None)
    assert verdict == "unavailable" and out == aim


def test_slightly_off_box_is_centred_not_refused():
    """2026-10-14 user run: 44 mm tea box, jaw gap 55.3, along -4.7 mm, refused by the
    room rule (4 mm allowed) although the arm was in place. The centring slide moves
    the jaws -4.7 mm along the closing axis."""
    m = metrics(-4.7, gap=55.3)
    m["axis_xy"] = [0.0, 1.0]
    w = 45.3
    assert not gg.contact_verdict(m, 15.0, w)[0]
    dx, dy = gg.centring_shift(m, 15.0, w)
    assert abs(dx) < 1e-9 and abs(dy - (-0.0047)) < 1e-9, (dx, dy)


def test_no_centring_when_already_fine_or_far_out_or_miss_too_big():
    w = 36.0
    m = metrics(+1.0); m["axis_xy"] = [0.0, 1.0]
    assert gg.centring_shift(m, 15.0, w) is None          # inside the allowed band
    m = metrics(+14.8); m["axis_xy"] = [0.0, 1.0]
    assert gg.centring_shift(m, 15.0, w) is None          # pad already deep in the object
    m = metrics(+8.5, miss=20.0); m["axis_xy"] = [0.0, 1.0]
    assert gg.centring_shift(m, 15.0, w) is None          # approach miss: not a sideways fix
    m = metrics(+8.5); m.pop("jaw_gap_mm")
    assert gg.centring_shift(m, 15.0, w) is None          # nothing to size it with


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL PASS")
