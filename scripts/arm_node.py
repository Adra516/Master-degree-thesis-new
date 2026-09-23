#!/usr/bin/env python3
"""arm_node — the manipulator, as services. The only MoveGroupCommander.

  ~set_arm_pose       (SetArmPose)       named joint pose
  ~set_gripper        (SetGripper)       open / close
  ~get_gripper_state  (GetGripperState)
  ~execute_pregrasp   (ExecutePregrasp)  PHASE 1: every object an obstacle, stop at the standoff
  ~execute_grasp      (ExecuteGrasp)     PHASE 2: target released, straight in, close, lift
  ~execute_place      (ExecutePlace)     lower and release

Ported near-verbatim from rework-attempt-1's scripts/arm_node.py — see
grasp_executor.py's module docstring for what "ported" means here: the logic,
timings and checks are the two-phase grasp originally proven in
locobot_motion_node.py (retired — see git history), just executing a
caller-supplied GraspPlan instead of deriving one inline.

Independently runnable and testable on its own:
`rosservice call /locobot/arm/set_arm_pose "pose_name: 'grasp_ready'"`.

Why one MoveIt owner: a second MoveGroupCommander doubles the planning-scene
subscriptions, and a second PlanningSceneInterface can race this one on
collision-object diffs — this node adds and removes the target mid-grasp, so a
competing diff would make the two-phase collision semantics nondeterministic.
"""
import os
import sys
import threading

import actionlib
import moveit_commander
import rospy
import tf2_ros
from moveit_commander import PlanningSceneInterface
from moveit_msgs.msg import MoveGroupAction

# roslaunch execs this file's source through a tiny generated wrapper in
# devel/lib/, so __file__'s directory (not Python's own auto-prepended
# sys.path[0], which points at the WRAPPER's directory in that case) must be
# added explicitly for the flat sibling imports below to resolve. Same
# pattern already used by the now-retired locobot_motion_node.py/
# vlm_agent_node.py — missing here caused a real ModuleNotFoundError under
# `roslaunch` (never surfaced by direct `python3 arm_node.py` testing, where
# the auto-prepend happens to already be correct).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_actuators import ARM_POSES, CAMERA_CLEAR_POSE, ArmActuator, Gripper
from arm_diagnostics import GraspDiagnostics
from collision_scene import CollisionScene
from grasp_executor import GraspExecutor
from grasp_geometry import FINGER_CLOSE, FINGER_OPEN
from locobot_prog.srv import (ExecuteGrasp, ExecuteGraspResponse, ExecutePlace,
                              ExecutePlaceResponse, ExecutePregrasp,
                              ExecutePregraspResponse, GetGripperState,
                              GetGripperStateResponse, SetArmPose, SetArmPoseResponse,
                              SetGripper, SetGripperResponse)

# How far short of its commanded close target a finger must stop before the grip
# counts as holding something. Above joint/controller settling noise, well below
# the half-width of any graspable object.
GRIP_HELD_MARGIN = 0.003


def nan_if_none(value, scale=1.0):
    """A missing measurement travels as NaN (float .srv fields have no null)."""
    return float("nan") if value is None else float(value) * scale


def _box_to_dict(box_msg):
    """locobot_prog/CollisionBox -> grasp_geometry's plain-dict box shape
    ({name, x, y, z, sx, sy, sz}) — the format collision_scene.py and
    grasp_geometry.py's own box builders already use."""
    c, s = box_msg.center, box_msg.size
    return {"name": box_msg.name, "x": c.x, "y": c.y, "z": c.z,
            "sx": s.x, "sy": s.y, "sz": s.z}


def _plan_to_dict(plan_msg):
    """locobot_prog/GraspPlan -> the plain-dict shape grasp_geometry.baseline_plan
    returns and grasp_executor.py consumes."""
    g, q, s = plan_msg.grasp_point, plan_msg.grasp_orientation, plan_msg.object_size
    return {"valid": bool(plan_msg.valid), "reason": plan_msg.reason,
            "grasp_point": [g.x, g.y, g.z],
            "grasp_orientation": [q.x, q.y, q.z, q.w],
            "standoff_offset": plan_msg.standoff_offset,
            "standoff_offset_min": plan_msg.standoff_offset_min,
            "gripper_open": plan_msg.gripper_open,
            "gripper_close": plan_msg.gripper_close,
            "full_close": bool(plan_msg.full_close),
            "object_size": [s.x, s.y, s.z]}


class ArmNode:

    def __init__(self):
        ns = rospy.get_param("~robot_name", "locobot")

        # MoveIt — wait for move_group before trying to connect.
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.loginfo("[arm] Waiting for move_group action server…")
        client = actionlib.SimpleActionClient("move_group", MoveGroupAction)
        if not client.wait_for_server(rospy.Duration(30.0)):
            rospy.logerr("[arm] move_group not available after 30 s — is MoveIt running?")
            raise RuntimeError("move_group action server not found")
        move_group = moveit_commander.MoveGroupCommander("interbotix_arm")
        move_group.set_end_effector_link(f"{ns}/ee_gripper_link")
        move_group.set_pose_reference_frame(f"{ns}/base_link")
        move_group.set_planning_time(8.0)
        move_group.set_num_planning_attempts(20)
        move_group.set_goal_position_tolerance(0.01)     # 1 cm
        move_group.set_goal_orientation_tolerance(0.05)  # ~3°
        move_group.set_max_velocity_scaling_factor(0.3)
        move_group.set_max_acceleration_scaling_factor(0.3)
        robot = moveit_commander.RobotCommander()  # reference state for retime_trajectory()

        tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(tf_buffer)

        self.arm = ArmActuator(ns, move_group)
        self.gripper = Gripper(ns)
        self.executor = GraspExecutor(
            ns, move_group, robot, self.arm, self.gripper,
            CollisionScene(ns, PlanningSceneInterface()),
            GraspDiagnostics(ns, move_group, tf_buffer))
        rospy.sleep(1.0)

        # One command at a time, as the old topic interface did: two steps must
        # never drive the arm at once.
        self._lock = threading.Lock()
        rospy.Service("~set_arm_pose", SetArmPose, self._guard(self._set_arm_pose, SetArmPoseResponse))
        rospy.Service("~set_gripper", SetGripper, self._guard(self._set_gripper, SetGripperResponse))
        rospy.Service("~get_gripper_state", GetGripperState, self._get_gripper_state)
        rospy.Service("~execute_pregrasp", ExecutePregrasp,
                      self._guard(self._execute_pregrasp, ExecutePregraspResponse))
        rospy.Service("~execute_grasp", ExecuteGrasp,
                      self._guard(self._execute_grasp, ExecuteGraspResponse))
        rospy.Service("~execute_place", ExecutePlace,
                      self._guard(self._execute_place, ExecutePlaceResponse))
        rospy.loginfo("[arm] Ready.")

    def _guard(self, handler, response_class):
        """Serialise calls, and turn an exception into a failed reply (the caller
        gets success=false and the error, not a broken service call)."""
        def wrapped(req):
            with self._lock:
                try:
                    return handler(req)
                except Exception as e:
                    rospy.logerr("[arm] %s raised: %s", handler.__name__, e)
                    return response_class(success=False, message=str(e))
        return wrapped

    # ── Handlers ─────────────────────────────────────────────────────────────

    def _set_arm_pose(self, req):
        positions = ARM_POSES.get(req.pose_name)
        if positions is None:
            return SetArmPoseResponse(success=False, message=f"Unknown pose: {req.pose_name} "
                                                             f"(known: {', '.join(ARM_POSES)})")
        self.arm.send_joints(positions, move_time=2.0)
        return SetArmPoseResponse(success=True)

    def _set_gripper(self, req):
        if req.open:
            self.gripper.send(FINGER_OPEN)
        else:
            self.gripper.send(FINGER_CLOSE, move_time=1.5)
        return SetGripperResponse(success=True, finger_position=self.gripper.position)

    def _get_gripper_state(self, req):
        """"Held" is judged against the COMMANDED close target, not a fixed
        width: with an adaptive close the resting finger position depends on the
        object. An object STOPS the fingers short of their target; empty jaws
        travel all the way to it. Before any grasp there is no target to compare
        against, so the old fixed 20 mm threshold is used."""
        pos = self.gripper.position
        ref = self.executor.last_close_target
        holding = pos > ref + GRIP_HELD_MARGIN if ref is not None else pos * 1000.0 < 20.0
        return GetGripperStateResponse(finger_position=pos, holding=holding)

    def _execute_pregrasp(self, req):
        plan = _plan_to_dict(req.plan)
        obstacles = [_box_to_dict(b) for b in req.obstacles]
        target_box = _box_to_dict(req.target_box)
        ok, info = self.executor.pregrasp(plan, obstacles, target_box, req.table_z)
        return ExecutePregraspResponse(
            success=ok,
            position_error=nan_if_none(info.get("err_norm_mm"), 1e-3),
            orientation_error=nan_if_none(info.get("ori_err_deg"), 0.017453292519943295),
            message=info.get("reason", ""))

    def _execute_grasp(self, req):
        plan = _plan_to_dict(req.plan)
        obj_now = (req.object_now.x, req.object_now.y) if req.object_now_valid else None
        ok, info = self.executor.grasp(plan, obj_now)
        return ExecuteGraspResponse(
            success=ok,
            # False only when a contact check refused to close.
            verified=info.get("verified", True),
            descended=info.get("descended", False),
            straddled=info.get("straddled", False),
            along_jaw=nan_if_none(info.get("along_jaw_mm"), 1e-3),
            in_plane_miss=nan_if_none(info.get("in_plane_miss_mm"), 1e-3),
            gripper_width=info.get("gripper_width_after_m", 0.0),
            message=info.get("reason", ""))

    def _execute_place(self, req):
        p = req.place_point
        ok, info = self.executor.place(p.x, p.y, None if req.at_grasp_height else p.z)
        return ExecutePlaceResponse(success=ok, message=f"placed at {info['placed_at']}")

    def run(self):
        rospy.loginfo("[arm] Moving arm to camera-clear pose.")
        self.arm.send_joints(CAMERA_CLEAR_POSE, move_time=2.5)
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("arm")
    ArmNode().run()
