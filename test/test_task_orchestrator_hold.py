"""Offline, no ROS master, no Gazebo: TaskOrchestrator's held-object state
machine — get_object's "already holding" refusal, inspect_object's "nothing
held" refusal, and inspect_object's unconditional place (even when the
rotation sweep partly fails).

Same rospy-stubbing technique as test_arm_hold.py / test_vlm_agent_plan.py:
mutate the already-imported `rospy` module's attributes before importing
world_model_node, so its own `rospy.X` calls resolve to the fakes. Every arm/
world-model service TaskOrchestrator calls is a configurable fake keyed by the
service name's last path segment (e.g. "execute_place", "set_joint") — this
file deliberately does NOT replay a real pick-and-place (that needs Gazebo;
see the live verification step), only the hold/refuse/place bookkeeping this
session added around it.
"""
import math
import os
import sys
from types import SimpleNamespace as NS

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import rospy  # noqa: E402

PARAMS = {}
rospy.get_param = lambda name, default=None: PARAMS.get(name, default)
rospy.loginfo = lambda *a, **k: None
rospy.logwarn = lambda *a, **k: None
rospy.logerr = lambda *a, **k: None
rospy.sleep = lambda *_: None
rospy.is_shutdown = lambda: False
rospy.Subscriber = lambda *a, **k: None
PUBLISHED = []          # (topic, value) for every Publisher.publish
rospy.Publisher = lambda topic, *a, **k: NS(publish=lambda m, *x, **y: PUBLISHED.append((topic, m)))
rospy.wait_for_service = lambda *a, **k: None
rospy.Service = lambda *a, **k: None

CALLS = []              # (key, kwargs) for every fake service call made
SERVICE_HANDLERS = {}    # key -> callable(**kwargs) -> response


class _FakeProxy:
    def __init__(self, name):
        self.key = name.rsplit("/", 1)[-1]

    def __call__(self, **kwargs):
        CALLS.append((self.key, kwargs))
        handler = SERVICE_HANDLERS.get(self.key)
        if handler is None:
            raise AssertionError(f"no fake handler registered for '{self.key}'")
        return handler(**kwargs)


rospy.ServiceProxy = lambda name, srv_type: _FakeProxy(name)

import world_model_node as wmn  # noqa: E402


def _orchestrator(obstacles=()):
    CALLS.clear()
    PUBLISHED.clear()
    PARAMS.clear()
    PARAMS["~inspect_mode"] = "wrist"      # the older sweep; the hand-yaw tests switch it
    SERVICE_HANDLERS.clear()
    SERVICE_HANDLERS["set_joint"] = lambda **kw: NS(
        success=True, message="", joint_positions=[0.0] * 6)
    SERVICE_HANDLERS["execute_place"] = lambda **kw: NS(success=True, message="placed")
    SERVICE_HANDLERS["target_memory"] = lambda **kw: NS(ok=False, position=NS(x=0.0, y=0.0, z=0.0))
    SERVICE_HANDLERS["drive_relative"] = lambda **kw: NS(success=True, message="")
    SERVICE_HANDLERS["get_scene"] = lambda **kw: NS(obstacles=list(obstacles))
    SERVICE_HANDLERS["shift_arm"] = lambda **kw: NS(
        success=True, message="", ee_position=NS(x=0.0, y=0.0, z=0.0))
    node = wmn.TaskOrchestrator()
    node._latest_color = NS(data=b"\x00")   # a frame is "available" for every sweep step
    return node


def _box(name, x, y):
    return NS(name=name, center=NS(x=x, y=y, z=0.25), size=NS(x=0.05, y=0.05, z=0.1))


def test_get_object_refuses_while_something_is_held():
    node = _orchestrator()
    node._held = {"target_name": "bottle", "pos": [0.5, 0.0]}
    node._navigate = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("must not navigate while something is held"))
    r = node._handle_get_object(NS(target_name="can_coke"))
    assert r.success is False
    assert "already holding" in r.description.lower(), r.description
    assert "bottle" in r.description


def test_inspect_object_refuses_when_nothing_held():
    node = _orchestrator()
    node._held = None
    r = node._handle_inspect_object(NS())
    assert r.success is False
    assert "nothing is currently held" in r.message.lower(), r.message
    assert not CALLS, CALLS   # no set_joint/execute_place calls at all


def test_inspect_object_rotates_places_and_clears_held_state():
    node = _orchestrator()
    node._held = {"target_name": "bottle", "pos": [0.42, 0.01]}
    r = node._handle_inspect_object(NS())
    assert r.success is True
    assert len(r.views) == len(wmn.INSPECT_OFFSET_FRACTIONS), r.views
    assert node._held is None   # cleared regardless of outcome
    place_calls = [kw for key, kw in CALLS if key == "execute_place"]
    assert len(place_calls) == 1, place_calls
    assert place_calls[0]["place_point"].x == 0.42
    assert place_calls[0]["place_point"].y == 0.01
    assert place_calls[0]["at_grasp_height"] is True


def test_inspect_object_still_places_when_the_rotation_partly_fails():
    """RELEASE IS THE PRIORITY (grasp_executor.place's own stated rule) applies
    here too: a held object must come back down even if some sweep steps
    failed to move, not be left hanging indefinitely."""
    node = _orchestrator()
    node._held = {"target_name": "bottle", "pos": [0.42, 0.01]}
    calls = {"n": 0}

    def flaky_set_joint(**kw):
        calls["n"] += 1
        return NS(success=calls["n"] != 2, message="simulated failure", joint_positions=[0.0] * 6)

    SERVICE_HANDLERS["set_joint"] = flaky_set_joint
    r = node._handle_inspect_object(NS())
    assert len(r.views) == len(wmn.INSPECT_OFFSET_FRACTIONS) - 1, r.views
    place_calls = [kw for key, kw in CALLS if key == "execute_place"]
    assert len(place_calls) == 1, place_calls   # still placed despite the one failed step
    assert node._held is None


def test_inspect_object_still_places_when_no_views_were_captured_at_all():
    node = _orchestrator()
    node._held = {"target_name": "bottle", "pos": [0.42, 0.01]}
    node._latest_color = None   # no frame ever available
    r = node._handle_inspect_object(NS())
    assert r.views == []
    place_calls = [kw for key, kw in CALLS if key == "execute_place"]
    assert len(place_calls) == 1, place_calls
    assert node._held is None
    assert r.success is False   # nothing to show the VLM, but the object is safely back down


def test_inspect_object_dodges_a_too_close_obstacle_before_rotating():
    """Regression for the live-observed arm/finger instability: a bottle
    swinging during the wrist_rotate sweep can reach a neighbour (e.g. the
    ball, 0.12m away in locobot_table.world) since the post-grasp retreat is
    vertical-only. A catalogued object nearer than INSPECT_OBSTACLE_CLEAR_M
    must trigger a shift_arm call, AWAY from that obstacle, before any
    set_joint sweep step."""
    held_pos = [0.42, 0.01]
    close_obstacle = _box("ball", 0.42, 0.13)   # 0.12m away in y — too close
    node = _orchestrator(obstacles=[close_obstacle])
    node._held = {"target_name": "bottle", "pos": held_pos}
    node._handle_inspect_object(NS())
    shift_calls = [kw for key, kw in CALLS if key == "shift_arm"]
    assert len(shift_calls) == 1, shift_calls
    # The obstacle is at +y relative to the held point, so "away" is -y.
    assert shift_calls[0]["dy"] < 0, shift_calls
    assert shift_calls[0]["dz"] == 0.0
    # The shift must be requested BEFORE the first rotation step.
    keys = [key for key, _ in CALLS]
    assert keys.index("shift_arm") < keys.index("set_joint"), keys


def test_inspect_object_does_not_dodge_when_nothing_is_close():
    held_pos = [0.42, 0.01]
    far_obstacle = _box("tea_box", 0.70, 0.15)   # ~0.28m away — clear
    node = _orchestrator(obstacles=[far_obstacle])
    node._held = {"target_name": "bottle", "pos": held_pos}
    node._handle_inspect_object(NS())
    shift_calls = [kw for key, kw in CALLS if key == "shift_arm"]
    assert shift_calls == [], shift_calls


def test_inspect_offsets_scale_down_for_a_taller_object():
    """Regression for the live-observed run-to-run bottle instability with no
    obstacle involved: wrist_rotate tilts (not spins) a side-gripped object,
    so the same angle swings a taller object's far end through a wider arc.
    can_coke's own height is the reference the scaling is calibrated to leave
    UNCHANGED (never observed unstable); the ~2.8x taller bottle must get a
    proportionally smaller sweep."""
    can_coke_offsets = wmn.inspect_offsets_for(0.082)
    bottle_offsets = wmn.inspect_offsets_for(0.23)
    assert can_coke_offsets == [f * wmn.INSPECT_MAX_OFFSET_RAD for f in wmn.INSPECT_OFFSET_FRACTIONS]
    assert max(abs(o) for o in bottle_offsets) < max(abs(o) for o in can_coke_offsets)
    # Still symmetric, still 5 steps, still centred on zero.
    assert len(bottle_offsets) == 5
    assert bottle_offsets[2] == 0.0
    assert bottle_offsets[0] == -bottle_offsets[-1]


def test_inspect_offsets_floor_and_unknown_height_fallback():
    # A very tall hypothetical object is floored, not scaled to ~0.
    tiny_offsets = wmn.inspect_offsets_for(5.0)
    assert max(abs(o) for o in tiny_offsets) == wmn.INSPECT_MIN_OFFSET_RAD
    # Unknown height (None, or 0/negative from a bad measurement) falls back
    # to the original full range — the already-tested pre-scaling behaviour.
    full = [f * wmn.INSPECT_MAX_OFFSET_RAD for f in wmn.INSPECT_OFFSET_FRACTIONS]
    assert wmn.inspect_offsets_for(None) == full
    assert wmn.inspect_offsets_for(0.0) == full
    assert wmn.inspect_offsets_for(-0.1) == full


def test_inspect_offsets_scale_down_for_a_taller_object():
    """Regression for the live-observed run-to-run bottle instability with no
    obstacle involved: wrist_rotate tilts (not spins) a side-gripped object,
    so the same angle swings a taller object's far end through a wider arc.
    can_coke's own height is the reference the scaling is calibrated to leave
    UNCHANGED (never observed unstable); the ~2.8x taller bottle must get a
    proportionally smaller sweep."""
    can_coke_offsets = wmn.inspect_offsets_for(0.082)
    bottle_offsets = wmn.inspect_offsets_for(0.23)
    assert can_coke_offsets == [f * wmn.INSPECT_MAX_OFFSET_RAD for f in wmn.INSPECT_OFFSET_FRACTIONS]
    assert max(abs(o) for o in bottle_offsets) < max(abs(o) for o in can_coke_offsets)
    # Still symmetric, still 5 steps, still centred on zero.
    assert len(bottle_offsets) == 5
    assert bottle_offsets[2] == 0.0
    assert bottle_offsets[0] == -bottle_offsets[-1]


def test_inspect_offsets_floor_and_unknown_height_fallback():
    # A very tall hypothetical object is floored, not scaled to ~0.
    tiny_offsets = wmn.inspect_offsets_for(5.0)
    assert max(abs(o) for o in tiny_offsets) == wmn.INSPECT_MIN_OFFSET_RAD
    # Unknown height (None, or 0/negative from a bad measurement) falls back
    # to the original full range — the already-tested pre-scaling behaviour.
    full = [f * wmn.INSPECT_MAX_OFFSET_RAD for f in wmn.INSPECT_OFFSET_FRACTIONS]
    assert wmn.inspect_offsets_for(None) == full
    assert wmn.inspect_offsets_for(0.0) == full
    assert wmn.inspect_offsets_for(-0.1) == full


def _hand_fake(unreachable=()):
    """turn_hand fake: records the yaws asked for; `unreachable` yaws have no IK solution."""
    asked = []

    def turn_hand(yaw=0.0, move_time=0.0):
        asked.append(yaw)
        if yaw in unreachable:
            return NS(success=False, message="no IK solution", ee_position=NS(x=0, y=0, z=0), joint_positions=[])
        return NS(success=True, message="", ee_position=NS(x=0.50, y=0.02, z=0.63), joint_positions=[0.0] * 6)

    SERVICE_HANDLERS["turn_hand"] = turn_hand
    return asked


def _held():
    return {"target_name": "bottle", "pos": [0.42, 0.01], "height": 0.25, "grasp_above_table": 0.20}


def test_inspect_by_hand_yaw_turns_each_side_returns_home_and_places():
    node = _orchestrator()
    PARAMS["~inspect_mode"] = "yaw"
    asked = _hand_fake()
    node._held = _held()
    r = node._handle_inspect_object(NS())
    assert r.success is True
    assert len(r.views) == 4, r.views
    # one side outward, back to the start, the other side, back to the start; first call is the read
    assert asked == [0.0, -0.35, -0.5, 0.0, 0.35, 0.5, 0.0, 0.0], asked
    assert not [1 for key, _ in CALLS if key == "set_joint"], "the wrist sweep must not run"
    place_calls = [kw for key, kw in CALLS if key == "execute_place"]
    assert len(place_calls) == 1 and node._held is None
    assert "hand yaw -0.50 to +0.50" in r.description, r.description


def test_inspect_by_hand_yaw_aims_the_camera_at_the_object_then_restores_it():
    node = _orchestrator()
    PARAMS["~inspect_mode"] = "yaw"
    PARAMS["~camera_smooth"] = False          # one command per aim: easy to count
    _hand_fake()
    node._held = _held()
    node._handle_inspect_object(NS())
    pans = [m.data for t, m in PUBLISHED if t.endswith("pan_controller/command")]
    tilts = [m.data for t, m in PUBLISHED if t.endswith("tilt_controller/command")]
    assert len(pans) == 5, pans              # 4 aims + the restore
    assert pans[-1] == 0.0                   # back to the fixed forward view
    assert tilts[-1] == node.tilt_near       # and the near tilt the next step assumes
    # held bottle: hand at z 0.63, centre 0.075 lower -> camera at 0.561 looks slightly UP
    assert all(t < 0.2 for t in tilts[:-1]), tilts


def test_the_camera_head_moves_in_a_smooth_ramp_by_default():
    node = _orchestrator()
    PARAMS["~inspect_mode"] = "yaw"
    _hand_fake()
    node._held = _held()
    node._handle_inspect_object(NS())
    pans = [m.data for t, m in PUBLISHED if t.endswith("pan_controller/command")]
    tilts = [m.data for t, m in PUBLISHED if t.endswith("tilt_controller/command")]
    assert len(pans) > 40, len(pans)          # many small commands, not five jumps
    assert pans[-1] == 0.0 and tilts[-1] == node.tilt_near   # the restore ends exactly on the fixed view
    # no single command moves the head more than a few hundredths of a radian
    steps = [abs(b - a) for a, b in zip(pans, pans[1:])] + [abs(b - a) for a, b in zip(tilts, tilts[1:])]
    assert max(steps) < 0.06, max(steps)


def test_inspect_by_hand_yaw_skips_unreachable_yaws():
    node = _orchestrator()
    PARAMS["~inspect_mode"] = "yaw"
    _hand_fake(unreachable=(-0.5, 0.5))
    node._held = _held()
    r = node._handle_inspect_object(NS())
    assert len(r.views) == 2, r.views
    assert "-0.35 to +0.35" in r.description, r.description
    assert node._held is None and len([1 for k, _ in CALLS if k == "execute_place"]) == 1


def test_inspect_by_hand_yaw_falls_back_to_the_wrist_sweep_when_nothing_is_reachable():
    node = _orchestrator()
    PARAMS["~inspect_mode"] = "yaw"
    _hand_fake(unreachable=(-0.5, -0.35, 0.35, 0.5))
    node._held = _held()
    r = node._handle_inspect_object(NS())
    assert len(r.views) == len(wmn.INSPECT_OFFSET_FRACTIONS), r.views
    assert [1 for k, _ in CALLS if k == "set_joint"], "the wrist sweep should have run"
    assert node._held is None


def _memory(x, y, z=0.47, ok=True):
    SERVICE_HANDLERS["target_memory"] = lambda **kw: NS(ok=ok, position=NS(x=x, y=y, z=z))


def test_inspect_places_the_object_at_its_original_position_reprojected_from_odometry():
    """Measured 2026-10-19: the base creeps 6-10 cm and yaws a few degrees between a grasp and the
    end of an inspection, so the object came back 10 cm from where it was picked up. The memory
    of the original position is reprojected into the CURRENT base frame and used exactly."""
    node = _orchestrator()
    _memory(0.40, -0.03, 0.47)                       # 4 cm from where the grasp noted it, 0.42/0.01
    node._held = {"target_name": "bottle", "pos": [0.42, 0.01]}
    r = node._handle_inspect_object(NS())
    place = [kw for key, kw in CALLS if key == "execute_place"][0]
    assert place["exact"] is True and place["at_grasp_height"] is False
    assert (place["place_point"].x, place["place_point"].y, place["place_point"].z) == (0.40, -0.03, 0.47)
    assert [kw for key, kw in CALLS if key == "target_memory"][-1]["action"] == 3   # FORGET, after the place
    assert not [1 for key, _ in CALLS if key == "drive_relative"], "within reach: the base must not move"
    assert r.success is True


def test_inspect_falls_back_to_the_grasp_point_when_the_memory_is_gone_or_implausible():
    for ok, x, y in [(False, 0.4, 0.0), (True, 0.70, 0.0), (True, 0.60, 0.0), (True, 0.20, 0.0)]:
        node = _orchestrator()
        _memory(x, y, ok=ok)    # gone / 28 cm away / still out of reach after the move / too close
        node._held = {"target_name": "bottle", "pos": [0.42, 0.01]}
        node._handle_inspect_object(NS())
        place = [kw for key, kw in CALLS if key == "execute_place"][0]
        assert place["exact"] is False and place["at_grasp_height"] is True, (ok, x, y)
        assert (place["place_point"].x, place["place_point"].y) == (0.42, 0.01)


def test_inspect_drives_the_base_back_when_the_original_position_has_crept_out_of_reach():
    """q1-q4, 2026-10-19: the base rolled back 5-11 cm and yawed, putting the original spot at
    0.60-0.63 m. One turn + drive back to the parking spot, then reproject and place exactly."""
    node = _orchestrator()
    readings = [(0.60, -0.05, 0.47), (0.50, -0.01, 0.47)]       # before / after the move

    def memory(**kw):
        x, y, z = readings[0] if not [1 for k, _ in CALLS if k == "drive_relative"] else readings[1]
        return NS(ok=True, position=NS(x=x, y=y, z=z))

    SERVICE_HANDLERS["target_memory"] = memory
    node._held = {"target_name": "bottle", "pos": [0.50, 0.02]}
    node._handle_inspect_object(NS())
    drive = [kw for key, kw in CALLS if key == "drive_relative"]
    assert len(drive) == 1, drive
    assert abs(drive[0]["turn"] - math.atan2(-0.05, 0.60)) < 1e-9      # face the original spot
    assert abs(drive[0]["distance"] - (math.hypot(0.60, -0.05) - 0.50)) < 1e-9   # back to the 0.50 m park
    place = [kw for key, kw in CALLS if key == "execute_place"][0]
    assert place["exact"] is True
    assert (place["place_point"].x, place["place_point"].y) == (0.50, -0.01)


def test_the_place_drive_is_bounded():
    node = _orchestrator()
    readings = [(0.73, 0.0, 0.47), (0.58, 0.0, 0.47)]
    SERVICE_HANDLERS["target_memory"] = lambda **kw: NS(ok=True, position=NS(
        x=readings[0 if not [1 for k, _ in CALLS if k == "drive_relative"] else 1][0], y=0.0, z=0.47))
    node._held = {"target_name": "bottle", "pos": [0.60, 0.0]}
    node._handle_inspect_object(NS())
    drive = [kw for key, kw in CALLS if key == "drive_relative"][0]
    assert drive["distance"] == wmn.PLACE_MAX_DRIVE_M                  # not the 0.23 m the position asks for
    place = [kw for key, kw in CALLS if key == "execute_place"][0]
    assert place["exact"] is False                                      # 0.58 m: still out of reach -> grasp point


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL PASS")
