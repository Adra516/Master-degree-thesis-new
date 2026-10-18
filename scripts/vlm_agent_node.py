#!/usr/bin/env python3
"""vlm_agent_node — the VLM planning layer, thin.

this file only runs the VLM loop.

The VLM (Ollama, qwen3-vl) is given a task and these tools, and decides what to
call. Each tool is one service call to task_orchestrator (the TaskOrchestrator
half of world_model_node.py); this node knows nothing about perception,
geometry or motion. Tool results go back to the VLM as JSON, with the bird's-eye
view and/or the camera image(s) attached so it can check the outcome itself.

  get_object(target_name)  -> /locobot/task/get_object      pick up and HOLD
  inspect_object()         -> /locobot/task/inspect_object  rotate the held
                              object through a few angles, then always place it
  navigate_to(target_name) -> /locobot/task/navigate_to
  update_plan(...)         -> in-process only; mutates the shared plan/state JSON

This node also serves run_task (RunTask.srv): give it one natural-language task,
it runs the loop above to completion and returns the VLM's final answer plus
the plan JSON. Tasks share one ongoing conversation across calls (self.messages
is never reset), so a later task can refer back to what an earlier one found —
see agent_repl.py for a terminal client.
"""
import base64
import json
import time
from typing import Optional

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from ollama import chat as ollama_chat
from sensor_msgs.msg import Image

from locobot_prog.srv import GetObject, InspectObject, NavigateTo, RunTask, RunTaskResponse

# A grasp counts as holding the object above this finger position 
GRIP_HELD_MIN_WIDTH = 0.015

SYSTEM_PROMPT = f"""You are an autonomous LoCoBot manipulation agent.
You have these tools:
  get_object(target_name) — finds the target, drives to it, and picks it up. Success
                            means it is now HELD (gripper_width_m > {GRIP_HELD_MIN_WIDTH:.3f}).
                            It is NOT put back down by this call, and it refuses if
                            something is already held.
  inspect_object()         — rotates the currently-held object through a few angles so
                            you can see it (one feedback image per angle), then ALWAYS
                            places it back down at the end, whether or not you learned
                            what you needed. Call this after get_object.
  navigate_to(target_name) — only drives to the target and stops in front of it, without
                            grasping. get_object already does this itself.
  update_plan(...)         — updates a shared plan/state JSON (goal, steps, facts, done).
                            Record decisions and what you learn here, e.g. after
                            inspect_object: update_plan(facts={{"bottle_has_marker": false}}).

Workflow for a task that needs to pick something up and look at it:
1. Call get_object to attempt the pick.
2. Call inspect_object to look at it, then record what you found with update_plan.
3. If get_object failed, call it again (up to 3 times total) before giving up.
End the episode by calling update_plan(done=True) and stating the outcome in your
final (non-tool-call) reply."""


class PlanState:
    """The shared plan/state JSON the VLM reads and writes through update_plan.
    Deliberately a loose scratchpad, not a schema — a free-text goal/steps list
    plus a facts dict the VLM fills in itself, not a validated structure this node interprets or acts on."""

    def __init__(self):
        self.goal = None
        self.steps = []
        self.facts = {}
        self.done = False

    def to_dict(self):
        return {"goal": self.goal, "steps": self.steps, "facts": self.facts, "done": self.done}

    def to_json(self):
        return json.dumps(self.to_dict())

    def update(self, goal=None, steps=None, facts=None, done=None):
        if goal is not None:
            self.goal = goal
        if steps is not None:
            self.steps = list(steps)
        if facts is not None:
            self.facts.update(facts)
        if done is not None:
            self.done = bool(done)
        return self.to_dict()


class VlmAgentNode:

    def __init__(self):
        ns = rospy.get_param("~robot_name", "locobot")
        self.target_object = rospy.get_param("~target_object", "bottle")
        self.vlm_model = rospy.get_param("~vlm_model", "qwen3-vl:2b-instruct")
        self.max_steps = int(rospy.get_param("~max_steps", 20))
        self._interval = float(rospy.get_param("~grasp_min_interval_s", 5.0))
        self._last_call_ts = 0.0

        self.bridge = CvBridge()
        self.latest_frame = None
        self._pending_images = []
        rospy.Subscriber(f"/{ns}/camera/color/image_raw", Image, self._on_image, queue_size=1)

        for name in ("get_object", "navigate_to", "inspect_object"):
            try:
                rospy.wait_for_service(f"/{ns}/task/{name}", timeout=60.0)
            except rospy.ROSException:
                rospy.logerr("[agent] /%s/task/%s unavailable", ns, name)
        self._get_object = rospy.ServiceProxy(f"/{ns}/task/get_object", GetObject)
        self._navigate_to = rospy.ServiceProxy(f"/{ns}/task/navigate_to", NavigateTo)
        self._inspect_object = rospy.ServiceProxy(f"/{ns}/task/inspect_object", InspectObject)

        # Persistent across run_task calls on purpose: a follow-up task can
        # refer back to what an earlier one found (the user driving the REPL
        # is having one ongoing conversation, not a fresh one per task).
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.plan = PlanState()
        self.tools = {"get_object": self.get_object, "navigate_to": self.navigate_to,
                      "inspect_object": self.inspect_object, "update_plan": self.update_plan}

        rospy.Service(f"/{ns}/agent/run_task", RunTask, self._handle_run_task)
        rospy.loginfo("[agent] target='%s' model=%s max_steps=%d",
                      self.target_object, self.vlm_model, self.max_steps)

    def _on_image(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError:
            pass

    @staticmethod
    def _encode(img_bgr, width: int = 0, height: int = 0, quality: int = 75) -> Optional[str]:
        if img_bgr is None:
            return None
        if width and height:
            img_bgr = cv2.resize(img_bgr, (width, height))
        _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf).decode("utf-8")

    def _feedback_images(self, res):
        """The bird's-eye view (when the task produced one), then the camera."""
        images = []
        if res.bev.data:
            images.append(self._encode(self.bridge.imgmsg_to_cv2(res.bev, "bgr8")))
        if res.attach_camera_frame:
            images.append(self._encode(self.latest_frame, width=320, height=240))
        return [i for i in images if i is not None]

    def _pace(self):
        """Leave at least grasp_min_interval_s between tool calls."""
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self._interval:
            rospy.sleep(self._interval - elapsed)
        self._last_call_ts = time.monotonic()

    # ── The VLM's tools. The docstrings are the tool descriptions it reads. ───

    def get_object(self, target_name: str) -> dict:
        """Observe the scene, navigate to target_name, and pick it up — it is then HELD,
        not placed back down. Refuses if something is already held (call inspect_object
        first to finish with it). The feedback image shows the gripper after the attempt
        so you can verify success.
        Args:
            target_name: Name of the target object to pick up (e.g. 'bottle').
        Returns:
            dict with success, gripper_width_m, zone, scene_summary, description.
        """
        rospy.loginfo("[agent] TOOL get_object('%s')", target_name)
        self._pace()
        try:
            r = self._get_object(target_name=target_name)
        except rospy.ServiceException as e:
            return {"success": False, "gripper_width_m": 0.0, "description": f"get_object failed: {e}"}
        self._pending_images = self._feedback_images(r)
        out = {"success": r.success, "gripper_width_m": r.gripper_width, "zone": r.zone,
               "scene_summary": r.scene_summary}
        if r.visible_objects:
            out["visible_objects"] = list(r.visible_objects)
        if r.phase:
            out["phase"] = r.phase
        out["description"] = r.description
        return out

    def inspect_object(self) -> dict:
        """Rotate the currently-held object (from a prior get_object) through a few
        angles so you can see it, then ALWAYS place it back down — whether or not you
        learned what you needed. Returns one feedback image per angle; look at all of
        them before answering. Call update_plan afterwards to record what you found.
        Returns:
            dict with success, n_views, description.
        """
        rospy.loginfo("[agent] TOOL inspect_object()")
        self._pace()
        try:
            r = self._inspect_object()
        except rospy.ServiceException as e:
            return {"success": False, "description": f"inspect_object failed: {e}"}
        images = [self._encode(self._to_bgr(v), width=320, height=240) for v in r.views]
        self._pending_images = [i for i in images if i is not None]
        return {"success": r.success, "n_views": len(r.views), "description": r.description}

    def update_plan(self, goal: str = "", steps: Optional[list] = None,
                    facts: Optional[dict] = None, done: bool = False) -> dict:
        """Update the shared plan/state JSON. Call this whenever you decide on a plan,
        finish a step, or learn something worth remembering (e.g. after inspect_object,
        record what you observed). Only given fields change — omit the rest to leave
        them as they are.
        Args:
            goal: optional, the overall objective in one sentence.
            steps: optional, replaces the full step list (a list of short strings).
            facts: optional, merged into the existing facts, e.g. {"bottle_has_marker": false}.
            done: true once the task is fully complete.
        Returns:
            dict: the full plan/state JSON after the update.
        """
        rospy.loginfo("[agent] TOOL update_plan(goal=%r, steps=%s, facts=%s, done=%s)",
                      goal, steps, facts, done)
        return self.plan.update(goal=goal or None, steps=steps, facts=facts, done=done or None)

    def _to_bgr(self, img_msg):
        try:
            return self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError:
            return None

    def navigate_to(self, target_name: str) -> dict:
        """Drive the robot to target_name and stop in front of it, facing it, without grasping.
        Use it to approach or look at an object. get_object already navigates by itself.
        Args:
            target_name: Name of the object to drive to (e.g. 'bottle').
        Returns:
            dict with success, distance_m, zone, scene_summary, description.
        """
        rospy.loginfo("[agent] TOOL navigate_to('%s')", target_name)
        self._pace()
        try:
            r = self._navigate_to(target_name=target_name)
        except rospy.ServiceException as e:
            return {"success": False, "description": f"navigate_to failed: {e}"}
        self._pending_images = self._feedback_images(r)
        out = {"success": r.success, "distance_m": round(r.distance, 3), "zone": r.zone,
               "scene_summary": r.scene_summary, "description": r.description}
        if r.visible_objects:
            out["visible_objects"] = list(r.visible_objects)
        return out

    # ── Agent loop ───────────────────────────────────────────────────────────

    def run_task(self, task_text):
        """Run one natural-language task through the VLM tool-use loop to
        completion. Appends to self.messages (never reset) so a later task can
        refer back to what an earlier one found. Returns the VLM's final text
        answer — never raises; a VLM error or exhausting max_steps without a
        final answer is reported back as that answer's text, not an exception."""
        cam_b64 = self._encode(self.latest_frame, width=320, height=240)
        self.messages.append({"role": "user", "content": task_text,
                              "images": [cam_b64] if cam_b64 else []})
        rospy.loginfo("[agent] run_task: %r (max_steps=%d)", task_text, self.max_steps)

        final_text = "(max_steps reached without a final answer)"
        for step in range(self.max_steps):
            if rospy.is_shutdown():
                break
            rospy.loginfo("[agent] Step %d/%d — calling VLM", step + 1, self.max_steps)
            try:
                resp = ollama_chat(model=self.vlm_model, messages=self.messages,
                                   tools=list(self.tools.values()), stream=False, think=False)
            except TypeError:
                # Older ollama clients have no `think` argument.
                try:
                    resp = ollama_chat(model=self.vlm_model, messages=self.messages,
                                       tools=list(self.tools.values()), stream=False)
                except Exception as e:
                    rospy.logerr("[agent] VLM error: %s", e)
                    final_text = f"(VLM error: {e})"
                    break
            except Exception as e:
                rospy.logerr("[agent] VLM error: %s", e)
                final_text = f"(VLM error: {e})"
                break

            msg = resp.message
            rospy.loginfo("[agent] VLM content: %s  tool_calls: %s",
                          (msg.content or "")[:120], bool(msg.tool_calls))
            self.messages.append(msg)

            if not msg.tool_calls:
                if step < self.max_steps - 1 and not (msg.content or "").strip():
                    rospy.logwarn("[agent] Empty reply, no tool call — nudging model")
                    self.messages.append({"role": "user",
                                          "content": "Continue the task: call a tool, "
                                                     "or give your final answer."})
                    continue
                final_text = msg.content or ""
                rospy.loginfo("[agent] VLM final: %s", final_text or "(empty)")
                break

            for tc in msg.tool_calls:
                name, args = tc.function.name, tc.function.arguments
                rospy.loginfo("[agent] → %s(%s)", name, args)
                self._pending_images = []
                fn = self.tools.get(name)
                result = fn(**args) if fn else {"error": f"Unknown tool: {name}"}
                rospy.loginfo("[agent] ← success=%s zone=%s", result.get("success"),
                              result.get("zone", "?"))
                self.messages.append({"role": "tool", "name": name, "content": json.dumps(result)})
                if self._pending_images:
                    self.messages.append({"role": "user", "content": "",
                                          "images": self._pending_images})

        return final_text

    def _handle_run_task(self, req):
        answer = self.run_task(req.task)
        return RunTaskResponse(success=True, answer=answer, plan_json=self.plan.to_json())

    def run(self):
        rospy.loginfo("[agent] Waiting for the camera…")
        rospy.sleep(3.0)
        deadline = rospy.Time.now() + rospy.Duration(15.0)
        while self.latest_frame is None and not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                rospy.logerr("[agent] No camera frame — aborting")
                return
            rospy.sleep(0.1)

        if bool(rospy.get_param("~auto_run", False)):
            rospy.loginfo("[agent] auto_run — running one task at startup")
            self.run_task(f"Pick up the {self.target_object}.")

        rospy.loginfo("[agent] Ready — waiting for run_task calls on ~/agent/run_task.")
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("vlm_agent")
    VlmAgentNode().run()
