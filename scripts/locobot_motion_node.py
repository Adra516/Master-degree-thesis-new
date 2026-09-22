#!/usr/bin/env python3
import json
import math
import os
import sys

import moveit_commander
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Pose, PoseStamped, Twist
from moveit_commander import PlanningSceneInterface
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grasp_geometry

# Arm joint presets (6-DOF: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate)
CAMERA_CLEAR_POSE = [0.0, -1.20,  1.00, 0.0,  0.80, 0.0]
LIFT_POSE         = [0.0, -0.80,  1.00, 0.0,  0.50, 0.0]
GRASP_READY_POSE  = [0.0, -1.00,  1.25, 0.0, -0.25, 0.0]

GRIPPER_OPEN  = 0.037   # metres, left_finger
GRIPPER_CLOSE = 0.010
# How far short of its commanded close target a finger must stop before the grip
# counts as holding something.  Above joint/controller settling noise, well below
# the half-width of any graspable object (>= ~27 mm/side short for a 55 mm can).
GRIP_HELD_MARGIN = 0.003
PREGRASP_POS_TOL_MM  = 15.0
PREGRASP_ORI_TOL_DEG = 10.0
CONTACT_MISS_TOL_MM  = 15.0
# A pending pre-grasp older than this is refused by execute_descent — the arm may
# have sagged and the scene/target may no longer be what was verified.
PENDING_GRASP_TTL_S  = 30.0

ARM_POSES = {
    "camera_clear": CAMERA_CLEAR_POSE,
    "grasp_ready":  GRASP_READY_POSE,
    "lift":         LIFT_POSE,
}

class LocobotMotionNode:

    def __init__(self):
        ns = rospy.get_param("~robot_name", "locobot")
        self.ns            = ns
        self.linear_speed  = float(rospy.get_param("~linear_speed",  0.12))
        self.angular_speed = float(rospy.get_param("~angular_speed", 0.30))

        # Actuator publishers 
        self.cmd_vel_pub = rospy.Publisher( f"/{ns}/cmd_vel", Twist, queue_size=1)
        self.arm_pub = rospy.Publisher( f"/{ns}/arm_controller/command", JointTrajectory, queue_size=1, latch=True)
        self.grip_pub = rospy.Publisher( f"/{ns}/gripper_controller/command", JointTrajectory, queue_size=1, latch=True)

        # Command channel
        self.result_pub = rospy.Publisher( f"/{ns}/locobot_prog/motion_result", String, queue_size=1)
        rospy.Subscriber( f"/{ns}/locobot_prog/motion_cmd", String, self._on_command, queue_size=1)

        # Sensor subscribers
        rospy.Subscriber(f"/{ns}/joint_states", JointState, self._on_joint_states, queue_size=1)

        # TF, used to close the loop on rotation (see _rotate_closed_loop)
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # State
        self.gripper_position = GRIPPER_OPEN   # cached from joint_states

        # Table geometry (updated per execute_pregrasp/execute_descent command)
        self.table_surface_z  = 0.25
        self.table_thickness  = 0.05
        self._grasp_target_x  = 0.50           # used to centre table collision box
        self._table_in_scene  = False          # tracks whether MoveIt scene has a "table" box
        self._obstacle_names  = []             # neighbour-object collision boxes in the scene
        self._target_obstacle_in_scene = False # target-as-obstacle box (pre-grasp phase)
        self._last_grasp_z    = None           # EE height of the last grasp (for "place")
        self._last_close_target = None         # last commanded per-finger close (m)
        # Two-phase grasp: execute_pregrasp leaves the arm holding at the standoff
        # and stashes everything execute_descent needs here (the MoveIt scene and
        # the arm pose both persist between the two commands). 
        self._pending_grasp   = None # None = no verified pre-grasp is pending, so a descent command must be refused.
        self._grasp_diag        = bool(rospy.get_param("~grasp_diag", True))
        self._diag_object_model = str(rospy.get_param("~diag_object_model", ""))
        rospy.loginfo("[motion] Grasp gates: pre-grasp <= %.1fmm / %.1f°, "
                      "contact miss <= %.1fmm (all live-tunable via rosparam)",
                      self._tol("pregrasp_pos_tol_mm",  PREGRASP_POS_TOL_MM),
                      self._tol("pregrasp_ori_tol_deg", PREGRASP_ORI_TOL_DEG),
                      self._tol("contact_miss_tol_mm",  CONTACT_MISS_TOL_MM))
        self._diag_pub = rospy.Publisher(f"/{ns}/locobot_prog/grasp_diag",
                                         String, queue_size=4, latch=True)
        self._gt_srv = None
        if self._grasp_diag:
            try:
                from gazebo_msgs.srv import GetModelState
                rospy.wait_for_service("/gazebo/get_model_state", timeout=3.0)
                self._gt_srv = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
                rospy.loginfo("[diag] Gazebo ground-truth available — contact-vs-object check enabled")
            except (rospy.ROSException, ImportError):
                rospy.loginfo("[diag] Gazebo GT unavailable — pre-grasp + fingers-vs-commanded still logged")

        # MoveIt — wait for move_group before trying to connect
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.loginfo("[motion] Waiting for move_group action server…")
        import actionlib
        from moveit_msgs.msg import MoveGroupAction
        client = actionlib.SimpleActionClient("move_group", MoveGroupAction)
        if not client.wait_for_server(rospy.Duration(30.0)):
            rospy.logerr("[motion] move_group not available after 30 s — is MoveIt running?")
            raise RuntimeError("move_group action server not found")
        rospy.loginfo("[motion] move_group ready.")
        self.move_group = moveit_commander.MoveGroupCommander("interbotix_arm")
        self.move_group.set_end_effector_link(f"{ns}/ee_gripper_link")
        self.move_group.set_pose_reference_frame(f"{ns}/base_link")
        self.move_group.set_planning_time(8.0)
        self.move_group.set_num_planning_attempts(20)
        self.move_group.set_goal_position_tolerance(0.01)    # 1 cm
        self.move_group.set_goal_orientation_tolerance(0.05) # ~3°
        self.move_group.set_max_velocity_scaling_factor(0.3)
        self.move_group.set_max_acceleration_scaling_factor(0.3)
        self.scene = PlanningSceneInterface()
        # RobotCommander needed as the reference state for retime_trajectory()
        self.robot = moveit_commander.RobotCommander()
        rospy.sleep(1.0)
        rospy.loginfo("[motion] Service node ready.")

    # Sensor callbacks
    def _on_joint_states(self, msg):
        if "left_finger" in msg.name:
            idx = msg.name.index("left_finger")
            self.gripper_position = msg.position[idx]

    # Command dispatcher
    def _on_command(self, msg):
        try:
            cmd = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError) as e:
            self._publish_result(False, error=f"JSON parse error: {e}")
            return

        name = cmd.get("cmd", "")
        rospy.loginfo("[motion] CMD: %s", name)

        dispatch = {
            "navigate":          self._handle_navigate,
            "set_arm_pose":      self._handle_set_arm_pose,
            "open_gripper":      self._handle_open_gripper,
            "close_gripper":     self._handle_close_gripper,
            "get_gripper_state": self._handle_get_gripper_state,
            # Two-phase grasp: each phase verifies what it achieved before handing off.
            "execute_pregrasp":  self._handle_execute_pregrasp,
            "execute_descent":   self._handle_execute_descent,
            "place":             self._handle_place,
        }

        handler = dispatch.get(name)
        if handler is None:
            self._publish_result(False, error=f"Unknown command: {name}")
            return

        try:
            handler(cmd)
        except Exception as e:
            rospy.logerr("[motion] Handler %s raised: %s", name, e)
            self._publish_result(False, error=str(e))

    def _publish_result(self, success: bool, data: dict = None, error: str = ""):
        payload = {"success": success, "data": data or {}}
        if error:
            payload["error"] = error
        self.result_pub.publish(String(data=json.dumps(payload)))

    # Command handlers
    def _handle_navigate(self, cmd: dict):
        linear_x  = float(cmd.get("linear_x",  0.0))
        angular_z = float(cmd.get("angular_z",  0.0))
        duration  = float(cmd.get("duration_s", 1.0))

        # Pure rotation: close the loop on the MEASURED base yaw (TF) instead of dead-reckoning by time.
        # speed*duration assumes instant velocity with no accel/decel ramp or cmd_vel processing latency,
        # so an open-loop turn always misses the commanded angle by some slack —
        # that slack is exactly what forced the caller into many hunt-and + recheck iterations.
        if linear_x == 0.0 and angular_z != 0.0:
            self._rotate_closed_loop(angular_z, duration)
        else:
            self._cmd_vel(linear_x, angular_z)
            rospy.sleep(duration)
            self._stop()
        self._publish_result(True)

    def _get_base_yaw(self):
        """Current base_link yaw in the ODOM frame, or None if TF is unavailable.
        Deliberately odom, not map: rtabmap corrects map->odom on every new
        keyframe, which can inject a discontinuous jump into "map" mid-turn —
        that looks to a velocity feedback loop like the base suddenly rotated
        (or un-rotated) on its own. odom is raw, continuous wheel odometry
        with no such jumps, which is what a tight rotation loop needs; map is
        for global goals, not for closing a millisecond-scale control loop.
        """
        try:
            t = self.tf_buffer.lookup_transform(
                f"{self.ns}/odom", f"{self.ns}/base_link", rospy.Time(0), rospy.Duration(0.2))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        q = t.transform.rotation
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _rotate_closed_loop(self, angular_z: float, duration: float):
        """Turn by angular_z*duration, but stop on MEASURED yaw, not elapsed time."""
        start_yaw = self._get_base_yaw()
        if start_yaw is None:
            rospy.logwarn("[motion] No TF for closed-loop rotation — falling back to timed turn")
            self._cmd_vel(0.0, angular_z)
            rospy.sleep(duration)
            self._stop()
            return
        self._rotate_to_target_yaw(start_yaw + angular_z * duration,
                                   max_speed=max(abs(angular_z), 0.05))

    def _rotate_to_target_yaw(self, target_yaw: float, max_speed: float = 0.25):
        """P-controlled turn to an ABSOLUTE odom-frame yaw (decelerates into
        the target, so the full residual can be commanded in one shot)."""
        tol      = math.radians(0.15)
        deadline = rospy.Time.now() + rospy.Duration(20.0)
        rate     = rospy.Rate(50)
        last_speed = 0.05
        misses   = 0
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            yaw = self._get_base_yaw()
            if yaw is None:
                # odom TF can stall under CPU contention; a miss must not cut
                # the turn short (silent undershoot).  Keep the last command,
                # bail loudly only after ~1.2s of solid silence.
                misses += 1
                if misses > 60:
                    rospy.logwarn("[motion] Closed-loop rotation: TF stalled "
                                  "%.1fs, bailing early (undershoot likely)",
                                  misses / 50.0)
                    break
                self._cmd_vel(0.0, last_speed)
                rate.sleep()
                continue
            misses = 0
            err = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
            if abs(err) < tol:
                break
            last_speed = float(np.clip(abs(err) * 2.0, 0.03, max_speed))
            last_speed = math.copysign(last_speed, err)
            self._cmd_vel(0.0, last_speed)
            rate.sleep()
        self._stop()

    def _handle_set_arm_pose(self, cmd: dict):
        pose_name = cmd.get("pose_name", "camera_clear")
        positions = ARM_POSES.get(pose_name)
        if positions is None:
            self._publish_result(False, error=f"Unknown pose: {pose_name}")
            return
        self._send_arm(positions, move_time=2.0)
        self._publish_result(True)

    def _handle_open_gripper(self, cmd: dict):
        self._send_gripper(GRIPPER_OPEN)
        self._publish_result(True)

    def _handle_close_gripper(self, cmd: dict):
        self._send_gripper(GRIPPER_CLOSE, move_time=1.5)
        self._publish_result(True)

    def _handle_get_gripper_state(self, cmd: dict):
        """Report the gripper gap and whether something is actually held.
        "Held" is judged against the COMMANDED close target, not a fixed width:
        with an adaptive close the resting finger position depends on the object,
        so a hard 20 mm threshold misreports.  The physical signal is that an
        object STOPS the fingers short of their target, while empty jaws travel
        all the way to it (a measured width equal to the close command means
        EMPTY fingers).  Falls back to the old fixed threshold before any grasp
        has run, when there is no commanded target to compare against.
        """
        gap_mm = self.gripper_position * 1000.0
        ref    = self._last_close_target
        if ref is not None:
            has_object = self.gripper_position > ref + GRIP_HELD_MARGIN
        else:
            has_object = gap_mm < 20.0
        self._publish_result(True, data={
            "has_object": has_object,
            "gap_mm":     round(gap_mm, 1),
        })

    def _setup_grasp(self, cmd: dict):
        """Shared preamble for the grasp phases: validate the target, build the
        MoveIt collision scene, point the waist at it and raise to grasp_ready.
        Returns a dict of the parsed/derived parameters, or None when the target
        was rejected — in which case the failure result has ALREADY been
        published, so the caller must simply return.
        """
        pos_base      = [float(cmd["x"]), float(cmd["y"]), float(cmd["z"])]
        angle_rad     = float(cmd.get("angle_rad",  0.0))
        dims_hwl      = cmd.get("dims_hwl")   # None when unknown 
        grasp_axis    = cmd.get("grasp_axis", "along_x")
        table_width_m = float(cmd.get("table_width_m", 0.80))
        table_depth_m = float(cmd.get("table_depth_m", 0.40))
        table_near_x  = cmd.get("table_near_x")   # float or None

        self.table_surface_z = float(cmd.get("table_z", 0.25))
        self._grasp_target_x = pos_base[0]

        # Reject targets outside the validated distance window before any setup.
        # Too far (> 0.52 = ARM_REACH_M − 3 cm margin): kinematically unreachable.
        # Too close (< 0.33): the STANDOFF, not the target, is what runs out of
        # arm — it sits PREGRASP_STANDOFF_OFFSET nearer still (0.33 → r=0.25),
        # and the measured joint-limit margin for a level gripper collapses as
        # the arm retracts (0.384 rad at r=0.36 → 0.181 at r=0.28), so the
        # pre-grasp becomes unreachable before the grasp itself does.  
        _dist_xy = math.hypot(pos_base[0], pos_base[1])
        if _dist_xy > 0.52:
            rospy.logwarn("[motion] Target at %.3fm — beyond arm reach (0.52m), aborting",
                          _dist_xy)
            self._publish_result(False, data={"gripper_width_after_m": 0.0},
                                 error=f"Target {_dist_xy:.2f}m exceeds arm reach — drive closer")
            return
        if _dist_xy < 0.33:
            rospy.logwarn("[motion] Target at %.3fm — too close for safe arm swing "
                          "(min 0.33m), aborting", _dist_xy)
            self._publish_result(False, data={"gripper_width_after_m": 0.0},
                                 error=f"Target {_dist_xy:.2f}m too close — back up")
            return
        # Reject objects too SHORT for a side grasp BEFORE any arm motion: the
        # jaws' lowest safe height would already be above the object's top edge,
        # so the grip would close in empty air.  Checked here rather than mid-sequence 
        try:
            grasp_geometry.grasp_height_above_table(dims_hwl)
        except grasp_geometry.ObjectTooShortForSideGrasp as e:
            rospy.logwarn("[motion] Too short for a side grasp (%s), aborting", e)
            self._publish_result(False, data={"gripper_width_after_m": 0.0},
                                 error=f"Object too short for a side grasp: {e}")
            return

        rospy.loginfo("[motion] Table: z=%.3f  w=%.2f  d=%.2f",
                      self.table_surface_z, table_width_m, table_depth_m)

        # Add table collision box BEFORE moving the arm to avoid START_STATE_IN_COLLISION
        self._add_table_collision(table_width_m=table_width_m,
                                  table_depth_m=table_depth_m,
                                  table_near_x=table_near_x)
        # Phase 1/2 collision scene: neighbour objects (can, ball, other cubes) as
        # padded axis-aligned boxes so the arm plans around them, PLUS the target
        # itself as an obstacle for the pre-grasp phase.  The target box is removed
        # again before the final linear grasp. these only affect the collision-aware swing + pre-grasp move.
        self._obstacle_names = self._add_obstacles(cmd.get("obstacles", []))
        self._add_target_obstacle(pos_base, dims_hwl)
        rospy.sleep(0.30)   # let MoveIt register the collision objects

        # Rotate waist to face the target before raising to GRASP_READY_POSE.
        # This gives the Cartesian planner a starting configuration that is already
        # pointing at the target, avoiding large waist swings during descent.
        waist_for_target = float(np.clip(
            math.atan2(pos_base[1], pos_base[0]), -1.5, 1.5))
        rospy.loginfo("[motion] Waist to target: %.3f rad (%.1f°)",
                      waist_for_target, math.degrees(waist_for_target))
        ready = [waist_for_target] + list(GRASP_READY_POSE[1:])
        self._send_arm(ready, move_time=3.5)

        return {"pos_base": pos_base, "dims_hwl": dims_hwl, "angle_rad": angle_rad,
                "grasp_axis": grasp_axis, "waist_for_target": waist_for_target,
                "ready_pose": ready, "gw_before": self.gripper_position}

    def _cleanup_grasp_scene(self):
        """Tear down every collision object this grasp added.
        remove_attached_object only DETACHES (leaving the box in the world at the
        gripper's last pose), so the world object is deleted too — a stale box
        would otherwise trip the collision-aware pre-grasp of the NEXT grasp.
        """
        self.scene.remove_attached_object(f"{self.ns}/ee_gripper_link",
                                          name="target_object")
        self.scene.remove_world_object("target_object")
        if self._table_in_scene:
            self.scene.remove_world_object("table")
            self._table_in_scene = False
        self._remove_obstacles()
        self._remove_target_obstacle()   # in case the grasp aborted before release

    def _grasp_outcome(self, ok, gw_before, info):
        """Publish the final result of a grasp, cleaning up the scene first.
        Keeps the long-standing rule that a MEASURABLY CLOSED gripper counts as
        success even when planning reported a failure — the physical evidence
        outranks the planner's opinion.
        """
        gw_m = self.gripper_position
        gripper_closed_on_object = (
            gw_before > GRIPPER_OPEN - 0.005          # was open before
            and GRIPPER_CLOSE + 0.008 < gw_m          # now partially closed
            and gw_m < GRIPPER_OPEN - 0.005           # but not fully open (held something)
        )
        success = ok or gripper_closed_on_object
        self._cleanup_grasp_scene()
        self._pending_grasp = None
        if not success:
            # Retract so the camera can see the scene for the next attempt 
            self._retract_clear()

        data = dict(info or {}, gripper_width_after_m=round(gw_m, 4))
        if success:
            rospy.loginfo("[motion] Grasp succeeded (gripper=%.1fmm  plan_ok=%s  grip_check=%s)",
                          gw_m * 1000, ok, gripper_closed_on_object)
            self._publish_result(True, data=data)
        else:
            rospy.logwarn("[motion] Grasp failed (gripper=%.1fmm  plan_ok=%s): %s",
                          gw_m * 1000, ok, (info or {}).get("reason", "?"))
            self._publish_result(False, data=data,
                                 error=(info or {}).get("reason",
                                                        "MoveIt planning or execution failed"))

    def _handle_execute_pregrasp(self, cmd: dict):
        """PHASE 1 command — approach the standoff and verify the arm reached it.
        On success the arm HOLDS at the verified standoff with the scene intact,
        waiting for execute_descent. On failure the scene is torn down and the
        arm retracts, so the caller can re-measure and re-position the base.
        """
        setup = self._setup_grasp(cmd)
        if setup is None:
            return                      # rejected; result already published
        ok, info = self._do_pregrasp(setup["pos_base"], setup["dims_hwl"],
                                     setup["angle_rad"], setup["grasp_axis"],
                                     setup["waist_for_target"], setup["ready_pose"])
        if not ok:
            self._cleanup_grasp_scene()
            self._retract_clear()
            self._publish_result(False, data=info, error=info.get("reason"))
            return
        # Remember what the descent will need from the setup half.
        self._pending_grasp["gw_before"] = setup["gw_before"]
        self._publish_result(True, data=info)

    def _handle_execute_descent(self, cmd: dict):
        """PHASE 2 command — descend, verify the jaws straddle the object, close.
        Refuses to run without a fresh, matching pre-grasp: the arm pose AND the
        MoveIt scene from phase 1 are both preconditions, and neither is
        re-established here.
        obj_xy (optional) — the object's CURRENT position in base_link, from the
        caller's odom reprojection.  Drives the contact gate; omitted → the gate
        is unverified and passes.
        """
        pg = self._pending_grasp
        if pg is None:
            self._publish_result(False, error="No pending pre-grasp — run "
                                              "execute_pregrasp first")
            return
        age = rospy.Time.now().to_sec() - pg["stamp"]
        if age > PENDING_GRASP_TTL_S:
            self._pending_grasp = None
            self._cleanup_grasp_scene()
            self._publish_result(False, error=f"Pending pre-grasp is stale "
                                              f"({age:.0f}s) — re-run execute_pregrasp")
            return
        # Identity guard: the descent must be for the target that was verified.
        if "x" in cmd and "y" in cmd:
            drift = math.hypot(float(cmd["x"]) - pg["x"], float(cmd["y"]) - pg["y"])
            if drift > 0.02:
                self._publish_result(False, error=f"Descent target moved {drift*100:.0f}cm "
                                                  f"from the verified pre-grasp — re-run "
                                                  f"execute_pregrasp")
                return

        obj_xy = cmd.get("obj_xy")     # [x, y] or [x, y, z] in base_link
        obj_base = None
        if obj_xy and len(obj_xy) >= 2:
            # Compare at the grasp height: the jaws are level there, so only the
            # in-plane geometry is meaningful for the straddle test.
            obj_base = [float(obj_xy[0]), float(obj_xy[1]), pg["grasp_z"]]

        # Pass the CURRENT dims (as of this descent command) through, so
        # _do_descent's adjustment checkpoint can recompute grasp_z if they've
        # changed since pre-grasp ran.
        ok, info = self._do_descent(obj_base, cmd.get("dims_hwl"))
        self._grasp_outcome(ok, pg.get("gw_before", self.gripper_position), info)

    def _handle_place(self, cmd: dict):
        """Lower the held object straight down at (x, y) and release it.
        Called right after execute_descent succeeds, while the arm holds the
        object at the retreat height straight above the grasp point.  Because the place point
        is the grasp point, NO pre-place swing is needed: we just descend the
        same vertical line (constant EE orientation, no twist) and open.
        RELEASE IS THE PRIORITY.  A partial or failed descent must NOT skip the
        open (that left the object welded to the arm forever) — we always open
        the gripper, even if the arm only got part way down.
        """
        x = float(cmd["x"])
        y = float(cmd["y"])
        # Descend to 3 mm below the grasp height: the object's base rests on the
        # table while still held, so any tilt is squared up before release.
        place_z = float(cmd.get("place_z",
                                (self._last_grasp_z or self.table_surface_z + 0.10)
                                - 0.003))
        approach_z = place_z + 0.10

        rospy.loginfo("[motion] Place: xy=(%.3f,%.3f)  place_z=%.3f", x, y, place_z)

        current_ori = self.move_group.get_current_pose().pose.orientation

        def _wp(px, py, pz):
            p = Pose()
            p.position.x, p.position.y, p.position.z = px, py, pz
            p.orientation = current_ori
            return p

        # 1. Descend straight down. placing the object a couple of cm high and letting it settle beats not releasing.
        plan, frac = self.move_group.compute_cartesian_path(
            [_wp(x, y, approach_z), _wp(x, y, place_z)], 0.005, False)
        rospy.loginfo("[motion] Place descent coverage: %.0f%%", frac * 100)
        if frac > 0.30:
            plan = self.move_group.retime_trajectory(
                self.robot.get_current_state(), plan,
                velocity_scaling_factor=0.15, acceleration_scaling_factor=0.15)
            dur = self._publish_arm_trajectory(plan)
            rospy.sleep(dur + 1.0)
        else:
            rospy.logwarn("[motion] Place descent only %.0f%% — releasing in place",
                          frac * 100)

        # 2. Settle, then OPEN — always.  Opening the fingers wide (per-finger travel ~10 mm)
        #    exceeds the grasp-fix release tolerance (8 mm) so the weld detaches. 
        #    _send_gripper() already blocks for move_time+0.2s internally.
        #    A released, low-gripped (top-heavy) object can topple immediately, and
        #    gazebo_grasp_fix re-attaches on ANY new contact within a single
        #    20Hz update (grip_count_threshold=1 — required for the original
        #    grasp to ever attach at all). 
        settled = self.move_group.get_current_joint_values()
        self._send_arm(settled, move_time=0.8)
        self._send_gripper(GRIPPER_OPEN, move_time=1.0)
        # Also drop the MoveIt planning-scene attachment so future plans are clean.
        self.scene.remove_attached_object(f"{self.ns}/ee_gripper_link",
                                          name="target_object")

        # 3. Retreat FAR and IMMEDIATELY — back to the original post-grasp lift
        #    height (not a token 5cm), at real speed, so the gripper clears the
        #    toppling object's space before it can re-contact anything. 
        retreat_up_z = (self._last_grasp_z or place_z) + 0.15
        up_plan, up_frac = self.move_group.compute_cartesian_path(
            [_wp(x, y, retreat_up_z)], 0.01, False)
        if up_frac > 0.5:
            up_plan = self.move_group.retime_trajectory(
                self.robot.get_current_state(), up_plan,
                velocity_scaling_factor=0.3, acceleration_scaling_factor=0.3)
            up_dur = self._publish_arm_trajectory(up_plan)
            rospy.sleep(up_dur + 0.3)
        self._send_arm(CAMERA_CLEAR_POSE, move_time=3.0)

        self._publish_result(True, data={"placed_at": [round(x, 3), round(y, 3)]})

    # Low-level actuators
    def _gripper_span_for(self, dims_hwl, grasp_axis,
                          open_clearance=0.022, grip_squeeze=0.018):
        """Compute per-finger OPEN and CLOSE positions adapted to the object size.
        The gripper is antagonistic: the left_finger joint position (metres) is the
        finger's displacement from centre and the right_finger mirrors it (-1×), so
        the object's half-width maps directly to the finger position needed to touch
        it.  Sizing both motions off the measured object makes the same code grasp a
        thin bottle or a wide box — only the numbers change.
        dims_hwl  : [H, W, L] camera-measured object dimensions in metres.
        grasp_axis: "along_x" → fingers close across W (dims_hwl[1]),
                    "along_y" → fingers close across L (dims_hwl[2]).
        open_clearance: extra gap per side on approach so the jaws descend around the
                    object without clipping it.
        grip_squeeze : how far past the object surface (per side) to command on close,
                    so the effort controller loads onto and holds the object.
        Returns (open_pos, close_pos), both clipped to the mechanical gripper range.
        Falls back to full open / full close when dimensions are unknown.
        """
        if dims_hwl is None:
            return GRIPPER_OPEN, GRIPPER_CLOSE
        narrow_m = float(dims_hwl[1] if grasp_axis == "along_x" else dims_hwl[2])
        half     = narrow_m / 2.0                       # finger displacement at contact
        open_pos  = float(np.clip(half + open_clearance,
                                  GRIPPER_CLOSE + 0.008, GRIPPER_OPEN))
        close_pos = float(np.clip(half - grip_squeeze,
                                  GRIPPER_CLOSE, GRIPPER_OPEN - 0.005))
        return open_pos, close_pos

    def _gripper_target_close(self, dims_hwl, grasp_axis, squeeze=0.006):
        """Per-finger close target — CONDITIONAL on the object's shape.
        TAPERED or unknown-width (tall slender bottle): FULL close.  Its width at
        the grip height is not measured — the bbox width is the wide body — so a
        body-sized target would stop the fingers in mid-air at the narrower grip
        point (no contact, no weld).  Closing fully lets them reach whatever
        cross-section is actually there; the gentle vendor gain (p=100) plus the
        weld hold it without shoving.  GRIPPER_CLOSE is clamped by the controller
        to the joint minimum, so this never over-drives the joint.
        PRISMATIC with a trustworthy measured width (can, box): ADAPTIVE close to
        just inside the surface.  A full close on a uniform object slams shut, so
        the first finger contacts and shoves the object sideways before the second
        lands; stopping a few mm inside the surface lets the jaws meet the object
        together and self-centre.  The width -> finger-position arithmetic lives
        only in _gripper_span_for; `squeeze` is the tuned 6 mm reach-through of
        the bbox width over-estimate, not a force knob.
        """
        if grasp_geometry.is_tapered(dims_hwl, grasp_axis):
            return GRIPPER_CLOSE
        _, close_pos = self._gripper_span_for(dims_hwl, grasp_axis,
                                              grip_squeeze=squeeze)
        return close_pos

    def _send_arm(self, positions, move_time=2.0):
        traj = JointTrajectory()
        traj.joint_names = ["waist", "shoulder", "elbow",
                            "forearm_roll", "wrist_angle", "wrist_rotate"]
        pt = JointTrajectoryPoint()
        pt.positions       = positions
        pt.time_from_start = rospy.Duration(move_time)
        traj.points = [pt]
        self.arm_pub.publish(traj)
        rospy.sleep(move_time + 0.2)

    def _publish_arm_trajectory(self, robot_traj):
        """Run a MoveIt RobotTrajectory by publishing it straight to the arm
        controller's command topic, bypassing MoveIt's execute() (the
        FollowJointTrajectory ACTION interface).
        Why: the action interface enforces per-joint path tolerances and ABORTS
        the whole trajectory the moment forearm_roll lags past its tolerance —
        which the low-gain roll joint does on every descent in this sim.  The
        command topic has no such monitoring: the controller simply chases the
        setpoints and forearm_roll settles as best it can.  Wrist roll doesn't
        affect the gripper's XYZ position (only which way the fingers face, which
        is irrelevant for a roughly cylindrical object), so this trades a perfect
        roll angle for a descent that actually completes.

        Returns the trajectory's nominal duration in seconds.
        """
        jt = robot_traj.joint_trajectory
        if not jt.points:
            return 0.0
        self.arm_pub.publish(jt)
        return jt.points[-1].time_from_start.to_sec()

    def _send_gripper(self, position, move_time=1.0):
        # Controller requires both joints; right_finger mirrors left_finger
        traj = JointTrajectory()
        traj.joint_names = ["left_finger", "right_finger"]
        pt = JointTrajectoryPoint()
        pt.positions       = [position, -position]
        pt.time_from_start = rospy.Duration(move_time)
        traj.points = [pt]
        self.grip_pub.publish(traj)
        rospy.sleep(move_time + 0.2)

    def _cmd_vel(self, linear=0.0, angular=0.0):
        t = Twist()
        t.linear.x  = linear
        t.angular.z = angular
        self.cmd_vel_pub.publish(t)

    def _stop(self):
        self._cmd_vel(0.0, 0.0)

    # MoveIt helpers 
    def _add_table_collision(self,
                             table_width_m: float = 0.80,
                             table_depth_m: float = 0.40,
                             table_near_x:  float = None,
                             padding:       float = 0.02):
        """Add (or replace) MoveIt table collision box.
        table_near_x: near-edge X of the table in base_link.  When provided the box
        is anchored to the actual table edge (near face = table_near_x) so that arm
        links passing OVER the edge do not collide with the box.  When absent the box
        is centred on _grasp_target_x (old behaviour — box may clip the elbow).
        Adds `padding` safety margin above the physical surface.
        """
        if self._table_in_scene:
            self.scene.remove_world_object("table")
            rospy.sleep(0.15)

        if table_near_x is not None and float(table_near_x) > 0.05:
            # Box near face = table near edge; centre = near_edge + depth/2.
            # Guard: only use near_x if it is in front of the robot (> 5 cm).
            # If the robot accidentally overshot the table edge, fall back to the object-centred estimate so the box stays in a sane position.
            bx = float(table_near_x) + table_depth_m / 2.0
        else:
            bx = self._grasp_target_x

        thickness = self.table_thickness + padding
        # Box top surface should be at table_surface_z + padding → centre = (table_surface_z + padding) - thickness/2
        table_cz  = self.table_surface_z + padding - thickness / 2.0

        pose = PoseStamped()
        pose.header.frame_id    = f"{self.ns}/base_link"
        pose.pose.position.x    = bx
        pose.pose.position.y    = 0.0
        pose.pose.position.z    = table_cz
        pose.pose.orientation.w = 1.0
        # Apply minimum size floors: a wide, deep box is safer than a narrow wrong one
        safe_depth = max(table_depth_m, 0.50)
        safe_width = max(table_width_m, 1.00)
        self.scene.add_box("table", pose, size=(safe_depth, safe_width, thickness))
        self._table_in_scene = True
        rospy.sleep(0.30)
        rospy.loginfo("[motion] Table collision: bx=%.3f  near_x=%s  centre_z=%.3f"
                      "  surface_z=%.3f  thick=%.3f  w=%.2f  d=%.2f",
                      bx, f"{table_near_x:.3f}" if table_near_x is not None else "None",
                      table_cz, self.table_surface_z, thickness,
                      table_width_m, table_depth_m)

    def _add_obstacles(self, obstacles):
        """Add neighbour-object collision boxes to the MoveIt planning scene.
        `obstacles` is the list built by grasp_geometry.obstacle_boxes on the
        agent side — one dict per NON-target detected object with a padded,
        axis-aligned square footprint (orientation unknown), centred and sized
        in base_link.  The target object is NOT here (it is grasped, not
        avoided).  Returns the list of names added so they can be removed later.
        """
        # clear any leftovers from a prior grasp before re-adding
        self._remove_obstacles()
        names = []
        for ob in obstacles or []:
            try:
                pose = PoseStamped()
                pose.header.frame_id    = f"{self.ns}/base_link"
                pose.pose.position.x    = float(ob["x"])
                pose.pose.position.y    = float(ob["y"])
                pose.pose.position.z    = float(ob["z"])
                pose.pose.orientation.w = 1.0
                self.scene.add_box(ob["name"], pose,
                                   size=(float(ob["sx"]), float(ob["sy"]), float(ob["sz"])))
                names.append(ob["name"])
            except (KeyError, TypeError, ValueError) as e:
                rospy.logwarn("[motion] skipping malformed obstacle %r: %s", ob, e)
        if names:
            rospy.loginfo("[motion] Obstacles added (%d): %s", len(names), ", ".join(names))
        return names

    def _remove_obstacles(self):
        """Remove every neighbour-object collision box added by _add_obstacles."""
        for name in self._obstacle_names:
            self.scene.remove_world_object(name)
        self._obstacle_names = []

    def _add_target_obstacle(self, pos_base, dims_hwl):
        """Add the TARGET as a collision box for the pre-grasp phase (the "pick"
        pattern: the arm avoids it while approaching, then it is released before
        the final linear grasp). Lightly padded (we deliberately approach it),
        square footprint (yaw unknown), body resting on the table."""
        h, w, l = dims_hwl if dims_hwl is not None else (0.15, 0.06, 0.06)
        # Shared with grasp_geometry.min_standoff_offset, which sizes the closest
        # the standoff may ever sit to THIS box — they must not drift apart.
        pad  = grasp_geometry.TARGET_BOX_PAD
        foot = max(float(w), float(l)) + 2 * pad
        pose = PoseStamped()
        pose.header.frame_id    = f"{self.ns}/base_link"
        pose.pose.position.x    = float(pos_base[0])
        pose.pose.position.y    = float(pos_base[1])
        pose.pose.position.z    = self.table_surface_z + float(h) / 2.0
        pose.pose.orientation.w = 1.0
        self.scene.add_box("target_obstacle", pose, size=(foot, foot, float(h) + 2 * pad))
        self._target_obstacle_in_scene = True
        rospy.loginfo("[motion] Target-as-obstacle added (pre-grasp): "
                      "xy=(%.3f,%.3f) foot=%.3f h=%.3f", pos_base[0], pos_base[1], foot, h)

    def _remove_target_obstacle(self):
        """Release the target from collision (it becomes the grasp target)."""
        if self._target_obstacle_in_scene:
            self.scene.remove_world_object("target_obstacle")
            self._target_obstacle_in_scene = False

    # ── Grasp instrumentation: pre-grasp pose + finger contact position ──────────
    def _finger_xyz(self, side):
        """(x,y,z) of a finger link in base_link via TF, or None."""
        try:
            t = self.tf_buffer.lookup_transform(
                f"{self.ns}/base_link", f"{self.ns}/{side}_finger_link",
                rospy.Time(0), rospy.Duration(0.3))
            tr = t.transform.translation
            return np.array([tr.x, tr.y, tr.z])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None

    def _object_in_base(self, model):
        """Ground-truth object centre in base_link via Gazebo, or None.  Standard
        world→base transform (robot model yaw about base)."""
        if self._gt_srv is None or not model:
            return None
        try:
            r = self._gt_srv(self.ns, "")
            o = self._gt_srv(model, "")
        except rospy.ServiceException:
            return None
        if not (getattr(r, "success", False) and getattr(o, "success", False)):
            return None
        q = r.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        dx = o.pose.position.x - r.pose.position.x
        dy = o.pose.position.y - r.pose.position.y
        bx = math.cos(-yaw)*dx - math.sin(-yaw)*dy
        by = math.sin(-yaw)*dx + math.cos(-yaw)*dy
        return np.array([bx, by, o.pose.position.z])

    @staticmethod
    def _tol(name, default):
        """Read a gate tolerance LIVE, so `rosparam set …/<name> <v>` takes effect
        without relaunching the node — the same treatment ~diag_object_model gets,
        and the reason the gates can be relaxed or disabled mid-evaluation."""
        return float(rospy.get_param(f"~{name}", default))

    def _publish_diag(self, payload):
        try:
            self._diag_pub.publish(String(data=json.dumps(payload)))
        except Exception:
            pass

    def _diag_pregrasp(self, cx, cy, cz, cori):
        """VERIFY the manipulator's pre-grasp pose: actual EE (base_link, MoveIt FK
        from the live joint state) vs the COMMANDED pre-grasp pose.  Pure FK, no
        frame ambiguity, so this works on hardware as well as in sim.
        Returns the measurement dict (err_norm_mm, ori_err_deg, …) so the caller
        can GATE on it, or None when no measurement could be taken (diagnostics
        disabled, or FK/TF failed).  None means UNVERIFIED, not "failed" — the
        caller must not treat a missing measurement as a bad pose.  Measurement
        must never break a grasp, hence the blanket except.
        """
        if not self._grasp_diag:
            return None
        try:
            p = self.move_group.get_current_pose().pose
            dx, dy, dz = p.position.x - cx, p.position.y - cy, p.position.z - cz
            norm_mm = math.sqrt(dx*dx + dy*dy + dz*dz) * 1000.0
            a, b = p.orientation, cori
            dot = abs(a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w)
            ori_deg = math.degrees(2.0 * math.acos(min(1.0, dot)))
            rospy.loginfo(
                "[diag] PRE-GRASP EE  cmd=(%.3f,%.3f,%.3f)  act=(%.3f,%.3f,%.3f)  "
                "err dx=%+.1f dy=%+.1f dz=%+.1f |Δ|=%.1fmm  ori_err=%.1f°",
                cx, cy, cz, p.position.x, p.position.y, p.position.z,
                dx*1000, dy*1000, dz*1000, norm_mm, ori_deg)
            payload = {"phase": "pre_grasp",
                       "cmd": [cx, cy, cz],
                       "act": [p.position.x, p.position.y, p.position.z],
                       "err_mm": [dx*1000, dy*1000, dz*1000],
                       "err_norm_mm": norm_mm, "ori_err_deg": ori_deg}
            self._publish_diag(payload)
            return payload
        except Exception as e:
            rospy.logwarn("[diag] pre-grasp measurement failed: %s", e)
            return None

    @staticmethod
    def _jaw_metrics(centre, u, gap, obj, cmd_xy):
        """Where an object sits relative to the jaws that are about to close.
        centre/u/gap describe the gripper (finger-link midpoint, closing-axis
        unit vector, physical jaw gap); obj is the object centre in base_link;
        cmd_xy is the COMMANDED grasp point.  Returns (along, perp, straddled):

          along     — object offset along the CLOSING axis, measured from the
                      finger midpoint.  0 = perfectly centred, |along| < gap/2
                      means it lies between the fingertips.  Measured from the
                      fingers because that is the real geometry of straddling,
                      and this axis carries no frame offset.
          perp      — miss along the APPROACH direction, measured from the
                      COMMANDED grasp point, NOT the finger midpoint.  The finger
                      link frames sit ~28 mm behind ee_gripper_link along exactly
                      this axis, so measuring it from the fingers would report a
                      constant ~28 mm "miss" on a perfectly executed grasp and
                      the gate would reject everything.  The commanded point is
                      where the jaws actually close, and the pre-grasp gate has
                      already verified the arm tracks its commands.
          straddled — object lies between the jaws along the closing axis.
        """
        along     = float(np.dot(obj - centre, u))
        straddled = bool(abs(along) < gap / 2.0)
        d_cmd     = np.asarray(obj[:2], float) - np.asarray(cmd_xy, float)
        perp      = float(np.linalg.norm(d_cmd - float(np.dot(d_cmd, u[:2])) * u[:2]))
        return along, perp, straddled

    def _diag_contact(self, cx, cy, cz, obj_base=None):
        """CHECK where the object sits relative to the jaws, BEFORE closing them.
        Logs the gripper centre, physical jaw gap and the offset vs the COMMANDED
        grasp point, then measures the object offset from up to two sources:

        * obj_base — the object position supplied by the caller (the agent's
          odom-cache reprojection).  HARDWARE-PORTABLE: needs neither the camera
          (the arm occludes it at grasp height) nor ground truth.  These are the
          metrics the caller GATES on.
        * Gazebo ground truth, when ~diag_object_model is set — logged alongside
          as an extra so the odom estimate's accuracy can be measured against
          truth in sim.  Never the gate, so the gate stays hardware-portable.

        Returns the gate metrics dict, or None when no object position was
        available from either source (UNVERIFIED, not "failed").  Measurement
        must never break a grasp, hence the blanket except.
        """
        if not self._grasp_diag:
            return None
        try:
            lf, rf = self._finger_xyz("left"), self._finger_xyz("right")
            if lf is None or rf is None:
                rospy.logwarn("[diag] contact: finger TF unavailable — skipped")
                return None
            centre = (lf + rf) / 2.0
            axis   = rf - lf
            gap    = float(np.linalg.norm(axis))
            u      = axis / (gap + 1e-9)              # finger-closing axis unit vector
            dcx, dcy, dcz = centre[0]-cx, centre[1]-cy, centre[2]-cz
            rospy.loginfo(
                "[diag] CONTACT gripper-centre=(%.3f,%.3f,%.3f)  jaw_gap=%.1fmm",
                centre[0], centre[1], centre[2], gap*1000)
            rospy.loginfo(
                "[diag] CONTACT vs CMD grasp point: dx=%+.1f dy=%+.1f dz=%+.1f mm",
                dcx*1000, dcy*1000, dcz*1000)
            payload = {"phase": "contact",
                       "gripper_centre": centre.tolist(), "jaw_gap_mm": gap*1000,
                       "left_finger": lf.tolist(), "right_finger": rf.tolist(),
                       "vs_cmd_mm": [dcx*1000, dcy*1000, dcz*1000]}
            # ── GATE source: the caller-supplied object position ────────────────
            # Comes from the agent's odom-cache reprojection, so it needs neither
            # the camera nor ground truth and the gate behaves the same on
            # hardware as in sim.
            gate = None
            if obj_base is not None:
                obj   = np.asarray(obj_base, dtype=float)
                rel   = obj - centre
                along, perp, straddled = self._jaw_metrics(centre, u, gap, obj, (cx, cy))
                rospy.loginfo(
                    "[diag] CONTACT vs OBJECT (odom): object=(%.3f,%.3f,%.3f)  "
                    "along-jaw=%+.1fmm (0=centred)  in-plane-miss=%.1fmm  straddled=%s",
                    obj[0], obj[1], obj[2], along*1000, perp*1000, straddled)
                gate = {"along_jaw_mm": along*1000, "in_plane_miss_mm": perp*1000,
                        "straddled": straddled, "source": "odom"}
                payload.update(dict(gate, object_base=obj.tolist(),
                                    offset_mm=(rel*1000).tolist()))

            # ── EXTRA: Gazebo ground truth, for accuracy comparison only ───────
            # Never gates — gating on GT would validate a check that cannot exist
            # on the real robot.  Model name read LIVE so you can
            # `rosparam set …/diag_object_model <name>` without relaunching.
            model  = rospy.get_param("~diag_object_model", self._diag_object_model)
            obj_gt = self._object_in_base(model)
            if obj_gt is not None:
                a_gt, p_gt, s_gt = self._jaw_metrics(centre, u, gap, obj_gt, (cx, cy))
                rospy.loginfo(
                    "[diag] CONTACT vs OBJECT (GT '%s'): object=(%.3f,%.3f,%.3f)  "
                    "along-jaw=%+.1fmm  in-plane-miss=%.1fmm  straddled=%s  → %s",
                    model, obj_gt[0], obj_gt[1], obj_gt[2], a_gt*1000, p_gt*1000, s_gt,
                    "OK/centred" if (s_gt and p_gt < 0.010) else "OFF-CENTRE (finger-push risk)")
                payload.update({"gt_object_base": obj_gt.tolist(),
                                "gt_along_jaw_mm": a_gt*1000,
                                "gt_in_plane_miss_mm": p_gt*1000,
                                "gt_straddled": s_gt})
                if obj_base is not None:
                    # How far the hardware-portable estimate is from truth — the
                    # number that says whether the odom path can be trusted.
                    est_err = float(np.linalg.norm(np.asarray(obj_base, float) - obj_gt))
                    rospy.loginfo("[diag] CONTACT odom-vs-GT object error: %.1fmm",
                                  est_err * 1000)
                    payload["odom_vs_gt_mm"] = est_err * 1000
            elif obj_base is None:
                rospy.loginfo("[diag] CONTACT vs OBJECT: no object position — pass one "
                              "in, or set ~diag_object_model for the GT reference")
            self._publish_diag(payload)
            return gate
        except Exception as e:
            rospy.logwarn("[diag] contact measurement failed: %s", e)
            return None

    def _approach_pregrasp(self, x, y, z, ori):
        """Collision-aware Cartesian move to the pre-grasp offset pose, with the
        target + neighbours still obstacles — the arm stops CLEAR at the standoff.
        Returns True if a collision-free path was found & executed; on low
        coverage it logs and returns False (the caller still runs the descent, so
        no regression — same reach as before, minus the collision guarantee)."""
        p = Pose()
        p.position.x, p.position.y, p.position.z = x, y, z
        p.orientation = ori
        frac = 0.0
        for attempt in (1, 2):
            plan, frac = self.move_group.compute_cartesian_path([p], 0.005, True)
            if frac >= 0.80:
                plan = self.move_group.retime_trajectory(
                    self.robot.get_current_state(), plan, 0.2, 0.2)
                dur = self._publish_arm_trajectory(plan)
                rospy.loginfo("[motion] Pre-grasp: collision-aware move to offset "
                              "(cov=%.0f%%, %.1fs, attempt %d)", frac * 100, dur, attempt)
                rospy.sleep(dur + 0.4)
                return True
            # Re-settle to zero the low-gain joints' position error and retry once
            # (collision-aware Cartesian coverage is start-state sensitive, same as
            # the descent's proven re-seat retry).
            settled = self.move_group.get_current_joint_values()
            self._send_arm(settled, move_time=0.6)
        rospy.logwarn("[motion] Pre-grasp collision-aware path low cov (%.0f%%) — "
                      "standoff may be blocked; proceeding to descent", frac * 100)
        return False

    def _compute_grasp_z(self, pos_base, dims_hwl):
        """Return the target gripper Z for grasping.
        Delegates to grasp_geometry.grasp_height_above_table — THE single source
        of truth for grip height.  The agent's reachability sweep consumes the
        same helper to choose the base standoff, so the base is positioned for
        the height the arm actually executes.  Do not re-derive the formula here.

        The height is object-RELATIVE: a fraction of the object's own height,
        floored by what the arm can reach and ceilinged by both the object's rim
        and the measured GRIP_Z_CEILING (which preserves the proven ~8.5 cm grip
        on the tall bottle, whose body ends ~10-11 cm — above that the neck is
        narrower than the jaws' minimum close and they shut on air).

        Raises grasp_geometry.ObjectTooShortForSideGrasp when the object is too
        short for any valid side grip.  Deliberately not clamped: a clamp would
        command a grip in the air above the object and report it as normal.
        """
        return self.table_surface_z + grasp_geometry.grasp_height_above_table(dims_hwl)

    def _wp_for(self, ori):
        """Waypoint factory bound to a fixed EE orientation."""
        def _wp(px, py, pz):
            p = Pose()
            p.position.x, p.position.y, p.position.z = px, py, pz
            p.orientation = ori
            return p
        return _wp

    def _retreat_up(self, x, y, z, ori):
        """Lift the gripper straight up to z, holding orientation.
        avoid_collisions=False: any attached-object box may overlap the table box
        at grasp height, but moving directly away from the table cannot collide.
        Direct publish (not execute()) so forearm_roll lag can't abort the lift.
        """
        plan, frac = self.move_group.compute_cartesian_path(
            [self._wp_for(ori)(x, y, z)], 0.01, False)
        if frac > 0.5:
            plan = self.move_group.retime_trajectory(
                self.robot.get_current_state(), plan,
                velocity_scaling_factor=0.15, acceleration_scaling_factor=0.15)
            rospy.sleep(self._publish_arm_trajectory(plan) + 0.4)
            return True
        rospy.logwarn("[motion] Retreat path only %.0f%% — holding position", frac * 100)
        return False

    def _retract_clear(self, min_lift=0.12):
        """Get the arm out of the way WITHOUT sweeping the object.

        A joint-space swing straight to CAMERA_CLEAR_POSE rotates the waist back
        to zero and reconfigures shoulder/elbow, dragging the gripper sideways
        through the space the object occupies.  MEASURED on aborted grasps: the
        target was tipped 90° over, or dragged 251 mm across the table — which
        defeats the point of aborting, since the whole reason to refuse the close
        is to leave the object recoverable for a retry.

        Lifting straight up first puts the gripper clear above the object before
        any reconfiguration begins.  Same reasoning the post-grasp lift already
        uses when it avoids a joint-space LIFT_POSE while carrying an object.
        """
        try:
            p = self.move_group.get_current_pose().pose
            self._retreat_up(p.position.x, p.position.y,
                             max(p.position.z + min_lift,
                                 self.table_surface_z + 0.25),
                             p.orientation)
        except Exception as e:
            rospy.logwarn("[motion] Clear-lift before retract failed: %s", e)
        self._send_arm(CAMERA_CLEAR_POSE, move_time=2.5)

    def _do_pregrasp(self, pos_base, dims_hwl, yaw, grasp_axis, waist_angle=0.0,
                     ready_pose=None):
        """PHASE 1 of 2 — approach the standoff and VERIFY the arm got there.
        The arm is already at GRASP_READY_POSE (waist facing target) when called.
        Opens the gripper, moves collision-aware to the pre-grasp standoff IN
        FRONT OF the object, then measures the achieved EE pose against the
        commanded one and gates on it.  On failure it adapts and re-plans once —
        Cartesian coverage is start-state sensitive, the same reason the descent
        has a re-seat retry.

        The standoff sits back along the approach axis AT THE GRASP HEIGHT, so
        the arm never passes over the object in either phase and the grasp that
        follows is a straight horizontal line.

        The target stays a collision obstacle throughout: the descent removes it.
        On success the state the descent needs is stashed in self._pending_grasp
        and the arm HOLDS at the standoff.

        Returns (ok, info) where info carries the measurement and a reason string.
        """
        x, y = pos_base[0], pos_base[1]
        # ONE call produces both points: grasp_pose fixes the grip height and the
        # yaw that faces the object, pregrasp_pose backs off along that same
        # approach axis.  Neither is re-derived here.
        grasp_xyz, grasp_quat = grasp_geometry.grasp_pose((x, y), dims_hwl,
                                                          self.table_surface_z)
        grasp_z = grasp_xyz[2]
        offset  = grasp_geometry.PREGRASP_STANDOFF_OFFSET
        pre_xyz, _ = grasp_geometry.pregrasp_pose(grasp_xyz, grasp_quat, offset)
        # Remembered for the "place" command: putting the object down at the same
        # EE height it was grasped at sets its base gently back onto the table.
        self._last_grasp_z = grasp_z

        rospy.loginfo("[motion] Horizontal grasp: grasp=(%.3f,%.3f,%.3f)  "
                      "standoff=(%.3f,%.3f,%.3f) offset=%.3f  axis=%s  yaw=%.3f",
                      x, y, grasp_z, pre_xyz[0], pre_xyz[1], pre_xyz[2],
                      offset, grasp_axis, yaw)

        
        settled = self.move_group.get_current_joint_values()
        self._send_arm(settled, move_time=0.8)

        # Descent EE orientation: read from the current (grasp_ready) state —
        # level gripper, guarantees Cartesian IK feasibility.
        descent_ori = self.move_group.get_current_pose().pose.orientation

        gripper_open_pos, _ = self._gripper_span_for(dims_hwl, grasp_axis)
        narrow_mm = (float(dims_hwl[1] if grasp_axis == "along_x" else dims_hwl[2]) * 1000
                     if dims_hwl is not None else 0.0)
        rospy.loginfo("[motion] Gripper open=%.1fmm (object narrow=%.1fmm axis=%s)",
                      gripper_open_pos * 1000, narrow_mm, grasp_axis)
        self._send_gripper(gripper_open_pos)
        rospy.sleep(0.3)   # let arm settle after gripper motion before planning

        # Pre-grasp (Phase 2): stop CLEAR at the offset standoff, target still an
        # obstacle 
        
        verdict, reason, meas = None, "", None
        approach_ok = False
        for attempt in (1, 2):
            approach_ok = self._approach_pregrasp(*pre_xyz, descent_ori)
            meas = self._diag_pregrasp(*pre_xyz, descent_ori)
            verdict, reason = grasp_geometry.pregrasp_verdict(
                meas, self._tol("pregrasp_pos_tol_mm", PREGRASP_POS_TOL_MM),
                self._tol("pregrasp_ori_tol_deg", PREGRASP_ORI_TOL_DEG))
            if verdict:
                rospy.loginfo("[motion] Pre-grasp VERIFIED: %s (attempt %d)",
                              reason, attempt)
                break
            rospy.logwarn("[motion] Pre-grasp NOT verified: %s (attempt %d, "
                          "approach_coverage_ok=%s)", reason, attempt, approach_ok)
            if attempt == 1:
                if approach_ok:
                    rospy.loginfo("[motion] Near miss — refining from the current "
                                  "pose (no re-seat)")
                else:
                    floor = grasp_geometry.min_standoff_offset(dims_hwl)
                    new_offset = max(floor, offset - 0.02)
                    if new_offset < offset - 1e-6:
                        offset = new_offset
                        pre_xyz, _ = grasp_geometry.pregrasp_pose(
                            grasp_xyz, grasp_quat, offset)
                        rospy.logwarn("[motion] Standoff unreachable from here — "
                                      "closing the offset to %.3fm (floor %.3fm, "
                                      "still outside the padded target box) and "
                                      "re-seating", offset, floor)
                    else:
                        rospy.logwarn("[motion] Standoff unreachable and already at "
                                      "the minimum safe offset (%.3fm) — re-seating "
                                      "and retrying unchanged", offset)
                    self._send_arm(ready_pose or ([waist_angle] + list(GRASP_READY_POSE[1:])),
                                   move_time=2.5)
                    settled = self.move_group.get_current_joint_values()
                    self._send_arm(settled, move_time=0.8)
                    descent_ori = self.move_group.get_current_pose().pose.orientation

        info = {"phase": "pregrasp", "verified": bool(verdict), "reason": reason,
                "err_norm_mm": round(float(meas["err_norm_mm"]), 1) if meas else None,
                "ori_err_deg": round(float(meas["ori_err_deg"]), 1) if meas else None}
        if not verdict:
            # Leave the target obstacle in the scene — nothing was released, so
            # the caller can retry or abort from a consistent planning scene.
            self._pending_grasp = None
            return False, info

        # Hold at the verified standoff; the descent command picks up from here.
        self._pending_grasp = {
            "x": x, "y": y, "grasp_z": grasp_z,
            "grasp_quat": grasp_quat, "offset": offset, "pre_xyz": pre_xyz,
            "descent_ori": descent_ori, "dims_hwl": dims_hwl,
            "grasp_axis": grasp_axis, "waist_angle": waist_angle,
            "ready_pose": ready_pose,
            "stamp": rospy.Time.now().to_sec(),
        }
        return True, info

    def _do_descent(self, obj_base=None, dims_hwl=None):
        """PHASE 2 of 2 — move straight in onto the object, VERIFY the jaws
        straddle it, then close.

        Requires a verified pre-grasp (self._pending_grasp) — the arm is holding
        at the standoff, level and facing the object, and the MoveIt scene still
        has the target as an obstacle.

        This leg is a SINGLE straight horizontal line: the standoff already sits
        at the grasp height, so closing the gap is one Cartesian waypoint at
        constant z.  (It used to be two waypoints — across to the target's xy,
        then vertically down onto it — which is why this method is still called
        "descent" throughout the command protocol.)

        obj_base : the object's current position in base_link, supplied by the
                   caller (the agent's odom-cache reprojection).  This is what
                   the contact gate measures against; it needs neither the camera
                   (the arm occludes it here) nor ground truth, so the gate
                   behaves identically on hardware.  None → contact unverified,
                   which passes.
        dims_hwl : the CURRENT best dimension estimate, as of right now —
                   separate from pg["dims_hwl"], which is whatever was known when
                   pre-grasp ran.  This is the genuine "does the arm still agree
                   on the object's size before committing to the grasp"
                   checkpoint between the two phases: if it differs from what
                   pre-grasp used (e.g. the agent's calibration gathered more
                   far-vantage samples in the meantime), grasp_z is RECOMPUTED
                   here rather than blindly trusting the frozen pre-grasp-time
                   value.  None -> reuse the pre-grasp value unchanged.

        Returns (ok, info).
        """
        pg          = self._pending_grasp
        x, y        = pg["x"], pg["y"]
        grasp_axis  = pg["grasp_axis"]
        waist_angle = pg["waist_angle"]
        descent_ori = pg["descent_ori"]

        # ── ADJUSTMENT CHECKPOINT: pre-grasp -> grasp ────────────────────────
        # Recompute from the CURRENT dims rather than trusting what pre-grasp
        # planned with.  If they agree, this is a no-op (same number back out).
        # If they differ, the straight-in move simply ends at the corrected
        # height: because the approach is horizontal, a dims change moves the
        # goal by that difference in z and the single Cartesian leg absorbs it as
        # a slight slope, instead of needing a separate re-verification round
        # trip.
        if dims_hwl is None:
            dims_hwl = pg["dims_hwl"]
        grasp_z = self._compute_grasp_z((x, y, 0.0), dims_hwl)
        if abs(grasp_z - pg["grasp_z"]) > 0.005:
            rospy.loginfo("[motion] Pre-grasp→grasp ADJUSTMENT: grasp_z %.3f->%.3f "
                          "(dims recalibrated since pre-grasp)", pg["grasp_z"], grasp_z)
        self._last_grasp_z = grasp_z   # so "place" descends to the ADJUSTED height
        retreat_z = grasp_z + 0.15
        _wp       = self._wp_for(descent_ori)

        # ── GATE 2a — BEFORE any motion: is the object still where we aimed? ──
        # The commanded grasp point was measured before the grasp_ready swing;
        # obj_base is where the object is NOW.  Catching a mismatch here, rather
        # than after descending, is what makes an abort genuinely clean: the jaws
        # never come down around the object, so they cannot brush it — and
        # gazebo_grasp_fix welds on first two-sided contact with NO close command,
        # so a brushed object would be carried away by the retreat (measured: the
        # target was lifted to z+0.15 and dragged 254 mm before dropping).
        miss_tol = self._tol("contact_miss_tol_mm", CONTACT_MISS_TOL_MM)
        if obj_base is not None:
            drift_mm = math.hypot(obj_base[0] - x, obj_base[1] - y) * 1000.0
            if drift_mm > miss_tol:
                reason = (f"object has moved {drift_mm:.0f}mm from the commanded "
                          f"grasp point (tolerance {miss_tol:.0f}mm) — not descending")
                rospy.logwarn("[motion] Contact NOT verified pre-descent: %s", reason)
                return False, {"phase": "descent", "verified": False,
                               "reason": reason, "drift_mm": round(drift_mm, 1),
                               "descended": False}

        self._remove_target_obstacle()     # target becomes the grasp target
        # Re-settle at the standoff so the descent plans from a stable start state.
        settled = self.move_group.get_current_joint_values()
        self._send_arm(settled, move_time=0.6)

        # The grasp leg — ONE straight horizontal waypoint.  Computed by MoveIt
        # (for the IK along the path) but EXECUTED by publishing directly to the
        # controller command topic, NOT through move_group.execute().  execute()
        # uses the FollowJointTrajectory action, which aborts whenever
        # forearm_roll lags past its path tolerance (every time in this sim).
        # Direct publishing has no such abort, so the move runs to completion.
        # Retime to 15% first: compute_cartesian_path ignores the velocity scaling
        # factor, so without this it would run at full speed and the arm would
        # overshoot the setpoints badly.
        #
        # avoid_collisions=False: this leg deliberately drives the OPEN jaws
        # around the object, so a collision check could only reject it — and the
        # table collision box's position is only estimated, so with the check on,
        # identical commands planned 100%/53%/28% depending on tiny start-state
        # differences near the box's front corner.  It is safe by construction
        # here: it starts from a pose verified to within PREGRASP_POS_TOL_MM, it
        # is only PREGRASP_STANDOFF_OFFSET long, and grasp_z is floor-clamped
        # well above the table.
        cart_plan, fraction = None, 0.0
        for attempt in (1, 2):
            cart_plan, fraction = self.move_group.compute_cartesian_path(
                [_wp(x, y, grasp_z)], 0.005, False)
            rospy.loginfo("[motion] Grasp path coverage: %.0f%% (attempt %d)",
                          fraction * 100, attempt)
            if fraction >= 0.80:
                break
            if attempt == 2:
                break
            # IK failed partway from this start state.  Re-seat, then RE-ESTABLISH
            # THE STANDOFF before retrying — do not simply re-plan from the ready
            # pose.  The ready pose is retracted, so a straight line from there to
            # the grasp point would be a long UNVERIFIED move driving the jaws
            # into the object, which is the exact failure the pre-grasp gate
            # exists to prevent.  Re-approaching the standoff first keeps the
            # invariant this phase depends on: the straight-in leg always starts
            # from a measured-good pose.
            rospy.logwarn("[motion] Grasp path short — re-seating and re-taking "
                          "the standoff before retrying")
            self._send_arm(pg.get("ready_pose")
                           or ([waist_angle] + list(GRASP_READY_POSE[1:])),
                           move_time=2.5)
            settled = self.move_group.get_current_joint_values()
            self._send_arm(settled, move_time=0.8)
            descent_ori = self.move_group.get_current_pose().pose.orientation
            _wp = self._wp_for(descent_ori)   # rebind: waypoints carry the orientation
            if not self._approach_pregrasp(*pg["pre_xyz"], descent_ori):
                rospy.logwarn("[motion] Could not re-take the standoff — aborting "
                              "rather than driving in from the ready pose")
                return False, {"phase": "descent", "verified": False,
                               "reason": "lost the verified standoff and could "
                                         "not re-establish it",
                               "descended": False}
            settled = self.move_group.get_current_joint_values()
            self._send_arm(settled, move_time=0.8)
        if fraction < 0.80:
            rospy.logwarn("[motion] Grasp path too short (%.0f%%) — aborting",
                          fraction * 100)
            return False, {"phase": "descent", "verified": False,
                           "reason": f"descent path only {fraction*100:.0f}% reachable"}

        cart_plan = self.move_group.retime_trajectory(
            self.robot.get_current_state(), cart_plan,
            velocity_scaling_factor=0.15, acceleration_scaling_factor=0.15)
        descent_dur = self._publish_arm_trajectory(cart_plan)
        rospy.loginfo("[motion] Descent executing directly (%.1fs, no action abort)",
                      descent_dur)
        rospy.sleep(descent_dur + 0.8)   # let the descent trajectory finish playing

        # Actively HOLD the arm at its own measured pose before gripping.  The
        # open-loop descent leaves the effort controllers chasing a goal they can't
        # fully reach under gravity, so they keep fighting (integral wind-up) and
        # the gripper wobbles on contact.  Re-commanding the *actual* joint values
        # drives the position error to ~0, so the arm stops fighting and holds
        # rigid while the jaws close.  This is the key stabilizer.
        settled = self.move_group.get_current_joint_values()
        self._send_arm(settled, move_time=0.8)

        # GATE: with the jaws OPEN around the object at grasp height, check that
        # the object is actually between them BEFORE closing.  Measured here, not
        # after the close, because that is the only moment the straddle question
        # is meaningful — and because closing on a known off-centre object is the
        # failure that shoves the target across the table rather than gripping it.
        contact = self._diag_contact(x, y, grasp_z, obj_base)
        c_ok, c_reason = grasp_geometry.contact_verdict(contact, miss_tol)
        info = {"phase": "descent", "verified": bool(c_ok), "reason": c_reason,
                "descended": True}
        if contact:
            info.update({"along_jaw_mm": round(contact["along_jaw_mm"], 1),
                         "in_plane_miss_mm": round(contact["in_plane_miss_mm"], 1),
                         "straddled": contact["straddled"]})
        if not c_ok:
            rospy.logwarn("[motion] Contact NOT verified: %s — retreating WITHOUT "
                          "closing", c_reason)
            # The jaws are already down around the object and may have BRUSHED it
            # on the way (74 mm open vs a 60 mm box leaves ~7 mm a side), and
            # gazebo_grasp_fix welds on first two-sided contact without any close
            # command — so a plain lift would carry the object off.  Cycling the
            # fingers past the plugin's ~8 mm release tolerance drops anything
            # that attached.  Safe specifically in this branch: it fires only when
            # the object is NOT between the jaws, so moving inward cannot squeeze
            # it.
            self._send_gripper(GRIPPER_OPEN - 0.012, move_time=0.8)
            self._send_gripper(GRIPPER_OPEN, move_time=0.8)
            self._retreat_up(x, y, retreat_z, descent_ori)
            return False, info
        rospy.loginfo("[motion] Contact VERIFIED: %s", c_reason)

        # Close so the jaws contact the object and gazebo_grasp_fix attaches it —
        # full close for a tapered object, adaptive for a uniform prism (see
        # _gripper_target_close).  Slowly, so contact ramps up without jolting the
        # arm.  The commanded target is remembered: "held" is then a comparison
        # against what was actually commanded (fingers stop SHORT of the target on
        # an object, reach it on air) instead of a fixed width threshold.
        close_target = self._gripper_target_close(dims_hwl, grasp_axis)
        self._last_close_target = close_target
        rospy.loginfo("[motion] Gripper close target=%.1fmm (%s)",
                      close_target * 1000,
                      "tapered/unknown -> full close"
                      if grasp_geometry.is_tapered(dims_hwl, grasp_axis)
                      else "prismatic -> adaptive")
        self._send_gripper(close_target, move_time=2.0)
        rospy.sleep(0.6)   # let the grip settle and grasp-fix attach

        # Re-hold at the new pose: the grasped object shifts the load, so re-command
        # the measured joints again to absorb that shift before lifting (avoids a
        # lurch when the arm starts carrying the extra weight).
        settled = self.move_group.get_current_joint_values()
        self._send_arm(settled, move_time=0.6)

        # Attach object in planning scene so retreat motion avoids table collision.
        # Use the measured dims when known, else a small default box (dims unknown).
        obj_pose = PoseStamped()
        obj_pose.header.frame_id    = f"{self.ns}/base_link"
        obj_pose.pose.position.x    = x
        obj_pose.pose.position.y    = y
        obj_pose.pose.position.z    = grasp_z
        obj_pose.pose.orientation.w = 1.0
        h, w, l = dims_hwl if dims_hwl is not None else (0.15, 0.06, 0.06)
        self.scene.attach_box(
            f"{self.ns}/ee_gripper_link", "target_object",
            pose=obj_pose, size=(w, l, h),
            touch_links=[f"{self.ns}/left_finger_link",
                         f"{self.ns}/right_finger_link",
                         f"{self.ns}/gripper_link"],
        )

        # "Lifted a little" IS this straight-up Cartesian retreat (grasp_z + 0.15).
        # We deliberately do NOT swing to a big joint-space LIFT_POSE: that
        # reconfigures shoulder/elbow/wrist_angle while carrying the welded
        # object, tilting and shaking it — the "twist/oscillation" seen on lift.
        # Holding straight above the grasp point keeps the gripper orientation
        # constant, so the object rides up level and still.  A settle-hold at the
        # measured pose stops the effort controllers fighting gravity.
        self._retreat_up(x, y, retreat_z, descent_ori)
        settled = self.move_group.get_current_joint_values()
        self._send_arm(settled, move_time=1.2)

        self._pending_grasp = None   # consumed
        return True, info

    # Main

    def run(self):
        rospy.loginfo("[motion] Starting. Moving arm to camera-clear pose.")
        self._send_arm(CAMERA_CLEAR_POSE, move_time=2.5)
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("locobot_motion_node")
    LocobotMotionNode().run()
