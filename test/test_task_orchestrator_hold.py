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
import os
import sys
from types import SimpleNamespace as NS

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import rospy  # noqa: E402

rospy.get_param = lambda name, default=None: default
rospy.loginfo = lambda *a, **k: None
rospy.logwarn = lambda *a, **k: None
rospy.logerr = lambda *a, **k: None
rospy.sleep = lambda *_: None
rospy.is_shutdown = lambda: False
rospy.Subscriber = lambda *a, **k: None
rospy.Publisher = lambda *a, **k: NS(publish=lambda *a, **k: None)
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
    SERVICE_HANDLERS.clear()
    SERVICE_HANDLERS["set_joint"] = lambda **kw: NS(
        success=True, message="", joint_positions=[0.0] * 6)
    SERVICE_HANDLERS["execute_place"] = lambda **kw: NS(success=True, message="placed")
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


def test_pretip_clear_uses_the_height_it_reaches_over_the_object():
    """Neighbours' boxes top out at ~0.385 m. A tea box / can grasp at z~0.33 used to
    be excluded because its GRASP height is below that; the over-the-top reach goes
    to ~0.45-0.5 m, which is clear."""
    obstacles = [_box("ball", 0.48, -0.15), _box("tea_box", 0.45, 0.22)]
    for o in obstacles:
        o.center.z, o.size.z = 0.31, 0.15          # top 0.385
    assert wmn.TaskOrchestrator._pretip_clear(0.49, obstacles)       # tea box reach height
    assert not wmn.TaskOrchestrator._pretip_clear(0.40, obstacles)   # would be too close
    assert wmn.TaskOrchestrator._pretip_clear(0.33, [])


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


def test_pretip_only_swings_when_the_jaws_are_above_every_neighbour():
    """The pre-tip swing is a joint move with no collision check. Measured scene:
    neighbours' boxes top out at ~0.385 m (ball/tea box). A bottle neck grasp at
    0.478 m is clear; a can or tea-box grasp at ~0.336 m is below their tops."""
    obstacles = [_box("ball", 0.48, -0.15), _box("tea_box", 0.45, 0.22)]
    for o in obstacles:
        o.center.z, o.size.z = 0.31, 0.15          # top 0.385
    assert wmn.TaskOrchestrator._pretip_clear(0.478, obstacles)
    assert not wmn.TaskOrchestrator._pretip_clear(0.336, obstacles)
    assert wmn.TaskOrchestrator._pretip_clear(0.336, [])   # nothing around: free to swing


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL PASS")
