#!/usr/bin/env python3
"""task_agent_node — the VLM planning layer, thin.

Ported near-verbatim from rework-attempt-1's scripts/vlm_agent_node.py.
This is now the ONLY agent node — it replaced the old, monolithic
vlm_agent_node.py (1064 lines; retired, see git history) as vision_nodes.
launch's live entry point. All the geometry/navigation/grasp logic that used
to live inline in that file now lives in world_model_node.py's
TaskOrchestrator, reached through exactly two service calls.

The VLM (Ollama, qwen3-vl) is given the user's request and two tools, and
decides what to call. Each tool is one service call to task_orchestrator (the
TaskOrchestrator half of world_model_node.py); this node knows nothing about
perception, geometry or motion. Tool results go back to the VLM as JSON, with
the bird's-eye view and the camera image attached so it can check the outcome
itself.

  get_object(target_name)  -> /locobot/task/get_object
  navigate_to(target_name) -> /locobot/task/navigate_to
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

from locobot_prog.srv import GetObject, NavigateTo

# A grasp counts as holding the object above this finger position — the value
# task_orchestrator (world_model_node.py) uses to set "success". Stated to the
# VLM from here so the prompt cannot drift from the code (it once said 0.020
# while the code used 0.015).
GRIP_HELD_MIN_WIDTH = 0.015

SYSTEM_PROMPT = f"""You are an autonomous LoCoBot manipulation agent.
You have two tools:
  get_object(target_name)  — finds the target, drives to it, picks it up and puts it
                             back down, and returns a feedback image so you can verify.
  navigate_to(target_name) — only drives to the target and stops in front of it,
                             without grasping. get_object already does this itself.

Workflow:
1. Call get_object to attempt the pick.
2. Check the feedback: success is true (gripper_width_m > {GRIP_HELD_MIN_WIDTH:.3f}) when
   the object was secured.
3. If it failed, call get_object again (up to 3 times total).
End the episode by stating the outcome."""


class TaskAgentNode:

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

        for name in ("get_object", "navigate_to"):
            try:
                rospy.wait_for_service(f"/{ns}/task/{name}", timeout=60.0)
            except rospy.ROSException:
                rospy.logerr("[agent] /%s/task/%s unavailable", ns, name)
        self._get_object = rospy.ServiceProxy(f"/{ns}/task/get_object", GetObject)
        self._navigate_to = rospy.ServiceProxy(f"/{ns}/task/navigate_to", NavigateTo)

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
        """Observe the scene, navigate to target_name, grasp it, and return a feedback image.
        Internally handles scene observation, zone classification, navigation, and grasping.
        The feedback image shows the gripper after the attempt so you can verify success.
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

    def run(self):
        rospy.loginfo("[agent] Waiting for the camera…")
        rospy.sleep(3.0)
        deadline = rospy.Time.now() + rospy.Duration(15.0)
        while self.latest_frame is None and not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                rospy.logerr("[agent] No camera frame — aborting")
                return
            rospy.sleep(0.1)

        tools = {"get_object": self.get_object, "navigate_to": self.navigate_to}
        cam_b64 = self._encode(self.latest_frame, width=320, height=240)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Pick up the {self.target_object}.",
             "images": [cam_b64] if cam_b64 else []},
        ]
        rospy.loginfo("[agent] Starting loop (max_steps=%d target='%s')",
                      self.max_steps, self.target_object)

        step = 0
        for step in range(self.max_steps):
            if rospy.is_shutdown():
                break
            rospy.loginfo("[agent] Step %d/%d — calling VLM", step + 1, self.max_steps)
            try:
                resp = ollama_chat(model=self.vlm_model, messages=messages,
                                   tools=list(tools.values()), stream=False, think=False)
            except TypeError:
                # Older ollama clients have no `think` argument.
                try:
                    resp = ollama_chat(model=self.vlm_model, messages=messages,
                                       tools=list(tools.values()), stream=False)
                except Exception as e:
                    rospy.logerr("[agent] VLM error: %s", e)
                    break
            except Exception as e:
                rospy.logerr("[agent] VLM error: %s", e)
                break

            msg = resp.message
            rospy.loginfo("[agent] VLM content: %s  tool_calls: %s",
                          (msg.content or "")[:120], bool(msg.tool_calls))
            messages.append(msg)

            if not msg.tool_calls:
                if step < self.max_steps - 1:
                    rospy.logwarn("[agent] No tool call — nudging model")
                    messages.append({"role": "user", "content": "Call get_object now."})
                    continue
                rospy.loginfo("[agent] VLM final: %s", msg.content or "(empty)")
                break

            for tc in msg.tool_calls:
                name, args = tc.function.name, tc.function.arguments
                rospy.loginfo("[agent] → %s(%s)", name, args)
                self._pending_images = []
                fn = tools.get(name)
                result = fn(**args) if fn else {"error": f"Unknown tool: {name}"}
                rospy.loginfo("[agent] ← success=%s zone=%s", result.get("success"),
                              result.get("zone", "?"))
                messages.append({"role": "tool", "name": name, "content": json.dumps(result)})
                if self._pending_images:
                    messages.append({"role": "user", "content": "",
                                     "images": self._pending_images})

        rospy.loginfo("[agent] Loop finished after %d steps.", step + 1)


if __name__ == "__main__":
    rospy.init_node("task_agent")
    TaskAgentNode().run()
