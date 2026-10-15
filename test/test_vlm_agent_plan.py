"""Offline, no ROS master, no Ollama: the new get_object/inspect_object split,
the plan/state JSON, and run_task's tool-use loop, driven by a scripted fake
LLM. Regression target: get_object no longer places (it only reports "held"),
inspect_object's multiple rotation-sweep views must all reach the VLM from one
tool call (not just one image), and update_plan must merge facts rather than
replace them.

rospy is stubbed the same way test_arm_hold.py stubs it: mutate the already-
imported `rospy` module's attributes before importing the module under test,
so `vlm_agent_node.rospy.X` resolves to the fakes. Real sensor_msgs/cv_bridge
types ARE used (no ROS master needed to construct or encode an Image message),
so inspect_object's view-encoding path (CvBridge -> cv2.imencode) runs for
real, not mocked away.
"""
import json
import os
import sys
from types import SimpleNamespace as NS

import numpy as np

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
rospy.wait_for_service = lambda *a, **k: None
rospy.Service = lambda *a, **k: None

SERVICE_HANDLERS = {}   # short name ("get_object", ...) -> callable(**kwargs) -> response


class _FakeProxy:
    def __init__(self, name):
        self.key = name.rsplit("/", 1)[-1]

    def __call__(self, **kwargs):
        handler = SERVICE_HANDLERS.get(self.key)
        if handler is None:
            raise AssertionError(f"no fake handler registered for '{self.key}'")
        return handler(**kwargs)


rospy.ServiceProxy = lambda name, srv_type: _FakeProxy(name)

import vlm_agent_node as van  # noqa: E402
from cv_bridge import CvBridge  # noqa: E402


# ── Fake LLM: a scripted queue of responses, ollama's own shape ────────────

class _Func:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, name, arguments):
        self.function = _Func(name, arguments)


class _Msg:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _Resp:
    def __init__(self, message):
        self.message = message


_SCRIPT = []


def _fake_ollama_chat(model, messages, tools, stream=False, think=None):
    return _SCRIPT.pop(0)


def _bridge_image():
    return CvBridge().cv2_to_imgmsg(np.zeros((4, 4, 3), dtype=np.uint8), "bgr8")


def _get_object_ok(**kw):
    return NS(success=True, gripper_width=0.022, zone="grasp_ready", scene_summary="ok",
              visible_objects=[], phase="descent",
              description="Grasped and holding 'bottle' (width 22mm). Call inspect_object.",
              attach_camera_frame=True, bev=NS(data=b""))


def _inspect_object_three_views(**kw):
    return NS(success=True, message="", views=[_bridge_image() for _ in range(3)],
              description="Rotated 'bottle' through 3 view(s) for inspection.")


def _node():
    SERVICE_HANDLERS["get_object"] = _get_object_ok
    SERVICE_HANDLERS["inspect_object"] = _inspect_object_three_views
    SERVICE_HANDLERS["navigate_to"] = lambda **kw: NS(
        success=True, distance=0.3, zone="grasp_ready", scene_summary="ok",
        description="Positioned.", visible_objects=[], attach_camera_frame=True, bev=NS(data=b""))
    node = van.VlmAgentNode()
    node.latest_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    van.ollama_chat = _fake_ollama_chat
    return node


# ── Pure logic: PlanState merges facts, replaces steps ─────────────────────

def test_plan_state_merges_facts_and_replaces_steps():
    p = van.PlanState()
    p.update(goal="pick the bottle", steps=["navigate", "get_object"])
    p.update(facts={"a": 1})
    p.update(facts={"b": 2})
    assert p.facts == {"a": 1, "b": 2}, p.facts
    p.update(steps=["done"])
    assert p.steps == ["done"], p.steps
    assert p.goal == "pick the bottle"
    assert p.done is False
    p.update(done=True)
    assert p.done is True


# ── Full loop: get_object (hold) -> inspect_object (3 views) -> update_plan
#    -> final answer ─────────────────────────────────────────────────────────

def test_run_task_holds_inspects_and_records_a_fact():
    node = _node()
    _SCRIPT[:] = [
        _Resp(_Msg(tool_calls=[_ToolCall("get_object", {"target_name": "bottle"})])),
        _Resp(_Msg(tool_calls=[_ToolCall("inspect_object", {})])),
        _Resp(_Msg(tool_calls=[_ToolCall(
            "update_plan", {"facts": {"bottle_has_marker": False}, "done": True})])),
        _Resp(_Msg(content="Done — no marker found.")),
    ]
    answer = node.run_task("Take the bottle, has it got a marker?")
    assert answer == "Done — no marker found.", answer
    assert node.plan.facts == {"bottle_has_marker": False}, node.plan.facts
    assert node.plan.done is True

    # get_object's own result must say HELD, not placed:
    tool_msgs = [m for m in node.messages
                if isinstance(m, dict) and m.get("role") == "tool"]
    get_object_result = json.loads(next(m for m in tool_msgs if m["name"] == "get_object")["content"])
    assert get_object_result["success"] is True
    assert "holding" in get_object_result["description"].lower()

    # all 3 rotation-sweep views from inspect_object must reach the VLM in one
    # images-only message, not just the first one:
    image_msgs = [m for m in node.messages
                 if isinstance(m, dict) and m.get("role") == "user" and m.get("images")]
    assert any(len(m["images"]) == 3 for m in image_msgs), image_msgs


def test_run_task_is_a_continuing_conversation_across_calls():
    """A second run_task call must still see the first one's messages — the
    REPL's whole point is a standing conversation, not a fresh one per task."""
    node = _node()
    _SCRIPT[:] = [_Resp(_Msg(content="First answer."))]
    node.run_task("First task.")
    n_before = len(node.messages)
    _SCRIPT[:] = [_Resp(_Msg(content="Second answer."))]
    answer = node.run_task("Second task.")
    assert answer == "Second answer."
    assert len(node.messages) > n_before
    assert any(m.get("content") == "First task." for m in node.messages
              if isinstance(m, dict))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL PASS")
