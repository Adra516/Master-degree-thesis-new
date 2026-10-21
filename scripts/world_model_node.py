#!/usr/bin/env python3
"""world_model_node — the world model AND the pick-and-place sequence.

Two classes in one process, ported near-verbatim
from rework-attempt-1's scripts/world_model_node.py + scripts/task_orchestrator_node.py:

  WorldModelNode (~world_model/...):
    ~get_scene      (GetScene)      objects, target, what blocks it, zone,
                                    table, collision boxes, bird's-eye view
    ~get_target     (GetTarget)     every source for where the target is
    ~target_memory  (TargetMemory)  hold the target in odometry / forget it
    ~bev_image      (Image, 1 Hz)   the bird's-eye view, live (when subscribed)

  TaskOrchestrator (/task/...):
    /task/get_object   (GetObject)   observe -> navigate -> pregrasp -> grasp -> place
    /task/navigate_to  (NavigateTo)  observe -> navigate, stop in front of the object

Try either half with nothing else running:
  rosservice call /locobot/world_model/get_scene "{target_name: 'tea box', include_bev: false}"
  rosservice call /locobot/task/get_object "target_name: 'bottle'"
"""
import collections
import copy
import math
import os
import statistics
import sys
import threading
import time

import numpy as np
import rospkg
import rospy
import tf2_geometry_msgs  # noqa: F401 — registers PointStamped with tf2
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped, Vector3, PointStamped
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64
from std_srvs.srv import SetBool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera_aim
import footprint
import grasp_geometry as gg
import grasp_planner as gp
import reach
import rotations
import world_bev
from arm_actuators import ARM_JOINTS
from conversions import (box_to_msg, detection_from_msg, dims_hwl_from_size,
                         plan_to_msg, size_from_dims_hwl, table_to_msg,
                         vec_or_zero, world_object_to_msg)
from ground_truth import GroundTruth
from resolver import CatalogResolver, ObjectCatalog
from target_odom_cache import TargetOdomCache
from target_state import measured_position
from world_model import WorldModel
from zones import ARM_GRASP_DIST, TABLE_DEPTH_M, ZONE_BLOCKED, ZONE_GRASP_READY
from locobot_prog.msg import CollisionBox, DetectionArray, TargetState
from locobot_prog.srv import (ApproachTarget, ApproachTargetRequest,
                              ApproachTargetResponse, DriveRelative, ExecuteGrasp, ExecutePlace,
                              ExecutePregrasp, GetGripperState, GetObject, GetObjectResponse,
                              GetScene, GetSceneResponse, GetTarget,
                              GetTargetResponse, InspectObject, InspectObjectResponse,
                              NavigateTo, NavigateToResponse,
                              PlanGrasp, PlanGraspResponse, SetArmPose, SetJoint,
                              ShiftArm, TargetMemory, TargetMemoryRequest,
                              TargetMemoryResponse, TurnHand)

# The grasp point trusts a target lock at most this old (positioning uses 2 s):
# the robot has been standing still since it aligned.
GRASP_MAX_LOCK_AGE = 3.0
TILT_SETTLE_S = 2.0
# how long to wait for a frame that contains the target, so the depth planner can measure it 
IN_FRAME_WAIT_TRIES = 10
IN_FRAME_WAIT_S = 0.4
# A grasp counts as holding the object above this measured finger position.
# A base tilt/creep smaller than this is left to the arm's own re-aim; a larger one
# is worth re-taking the pre-grasp for (see TaskOrchestrator._settled_shift).
SHIFT_FOLLOW_XY_MM = 10.0

# BASE TILT FLIP. The base rests either upright (~2.1 deg) or leaning forward on its
# stop (~5.2 deg); it flips forward when the arm's shoulder passes about -0.69 rad and
# stays there until the arm is almost fully retracted (shoulder ~-1.15). With the old
# 0.44-0.47 m park the grasp point sat ~0.42 m out, the base was still upright at the
# standoff and flipped 1.5-2.2 s into the straight-in leg, just as the pads reached the
# neck: the end effector dropped ~20 mm, lunged ~25 mm, and the jaws knocked the bottle
# (2026-10-17, 5/5 descents refused at the contact gate, 4/5 bottles knocked over).
# With the grasp point ~0.50 m out the base flips during the FIRST pre-grasp instead,
# the tilt re-take measures it (+26 mm) and the leg starts on an already-flipped base
# (2026-10-18: 2/2 clean lifts when parked there). See PARK_GRASP_* below. The earlier
# over-the-top "pre-tip" was removed 2026-10-17: it ended the pre-grasp above the target.
# How far the depth planner may turn the jaws' closing axis away from "perpendicular to
# the line to the object", so that they close on a box's real faces.
#   * 0 for a ROUND object (bottle, can: any angle meets the same curved surface) and for
#     anything not in the catalogue.
#   * GRASP_BOX_MAX_AXIS_DEV_DEG for a catalogued object that is NOT round (the tea box).
# Why (2026-10-21, ground truth): the tea box sits 3-16 deg off the approach, so across the
# jaws it is 47-56 mm wide against the 56 mm the open pads leave (0.2-4.5 mm a side), and
# a pad that meets a corner turns the box 40-50 deg. Aligned it is its true 45 mm (5.5 mm
# a side). The hand was within 3 deg of the faces in every pre-grasp with the axis free, and
# those pre-grasps verified at 2.2-2.7 mm. The 40 deg tried on 2026-10-14 was blocked (33-63 mm
# off in 5/5 launches); 25 deg turns the hand about its own point through the multi-seed
# wrist and was not. Override with ~grasp_max_axis_dev_deg (applies to every object).
GRASP_MAX_AXIS_DEV_DEG = 0.0
GRASP_BOX_MAX_AXIS_DEV_DEG = 25.0
# A catalogued box's face width is a fact; the planner's width is a measurement from one side
# and read 18-51 mm for the same 45 mm tea box. Past this difference the catalogue wins for
# the gripper opening and the close target (a 19 mm reading closed the jaws to 14.6 mm and
# threw the box out; the contact gate's room is computed from the same target).
BOX_WIDTH_TRUST_M = 0.006

RETAKE_REAIM_MAX_MM = 65.0     # the descent re-aim limit (/arm/reaim_max_mm) for a pick whose
                               # tilt-corrected pre-grasp was blocked and fell back to the original
                               # standoff: the descent must follow the whole tilt shift (40-49 mm
                               # measured; the default limit refused one at 45)
# Where the base must be parked, judged from the DEPTH PLANNER's grasp point (the
# navigation estimate was 4-6 cm off in both directions, 2026-10-18): closer than MIN,
# the base is still upright at the standoff and flips during the straight-in leg (see
# BASE TILT FLIP above); farther than MAX, past the arm's reach check.
PARK_GRASP_MIN_M = 0.49
PARK_GRASP_MAX_M = 0.53
PARK_GRASP_TARGET_M = 0.505
GRIP_HELD_MIN_WIDTH = 0.015

# inspect_object's rotation sweep: offsets (rad) from the wrist_rotate value the
# grasp ended at, swept in order and one frame captured per step. wrist_rotate is
# the LAST joint (arm_actuators.ARM_JOINTS) — its axis is the gripper's own roll
# axis, so moving only it spins a held object about its own long axis in place
# (no other joint changes, so the EE position does not move) rather than
# swinging the whole arm. A BOUNDED sweep, not a full rotation, per the project
# requirement that inspection shows a few angles, not a flip. FIRST-PASS values —
# not yet live-verified in Gazebo; expect to retune after watching it run.
# Sweep amplitude is HEIGHT-SCALED, not flat, as of 2026-10-07: wrist_rotate
# tilts rather than cleanly spins a side-gripped object, so a given angle
# swings a TALLER object's far end through a proportionally wider arc (basic
# lever-arm geometry — the grip point is nowhere near either end). Live-
# observed: the bottle (0.23m) was intermittently unstable mid-sweep even
# with no nearby obstacle to hit, run to run, with no other change — exactly
# the signature of a swing that sometimes, not always, stresses something
# past a threshold (here: the grasp-fix weld's own release_tolerance, a soft
# joint that can lag a fast/wide rotation). can_coke (0.082m) was never
# observed to have this problem even once, so its height is the reference
# point this scaling is calibrated to leave UNCHANGED; everything taller
# gets proportionally less rotation, everything shorter (floored) a bit more
# — not yet re-verified live, since this needs exactly the kind of run-to-run
# variance it's meant to fix to confirm either way.
INSPECT_REFERENCE_HEIGHT_M = 0.082    # can_coke's own catalogued height
INSPECT_MAX_OFFSET_RAD = 0.5          # amplitude AT the reference height (the old flat value)
INSPECT_MIN_OFFSET_RAD = 0.12         # floor, so a very tall object still gets SOME sweep
INSPECT_OFFSET_FRACTIONS = [-1.0, -0.5, 0.0, 0.5, 1.0]   # 5 steps, symmetric, as before
INSPECT_STEP_MOVE_TIME = 1.0
INSPECT_STEP_SETTLE_S = 0.5
INSPECT_CAMERA_SETTLE_S = 2.0   # head (effort position controller) to settle after a single step command
INSPECT_CAMERA_SETTLE_SMOOTH_S = 0.6   # ... after a ramp, which already arrives slowly


def inspect_offsets_for(height):
    """wrist_rotate sweep offsets (rad), scaled by the held object's height —
    see INSPECT_REFERENCE_HEIGHT_M's comment for why. Falls back to the full
    (pre-scaling) range when height is unknown — the original, already-tested
    behaviour for that case, not a new untested path."""
    if not height or height <= 0:
        max_off = INSPECT_MAX_OFFSET_RAD
    else:
        max_off = max(INSPECT_MIN_OFFSET_RAD,
                      min(INSPECT_MAX_OFFSET_RAD,
                          INSPECT_MAX_OFFSET_RAD * INSPECT_REFERENCE_HEIGHT_M / height))
    return [f * max_off for f in INSPECT_OFFSET_FRACTIONS]


# Minimum clearance (m, XY-plane) a held object's grasp point must have from
# any OTHER catalogued object's centre before it's considered safe to rotate
# in place — below this, _dodge_nearby_obstacle shifts the arm away first.
# First-pass value, sized to catch the live case this was built for: the ball
# sits only 0.12m from the bottle's own position in locobot_table.world
# (retreat is vertical-only, so the held object never moves away from it on
# its own); the bottle's other neighbours (tea_box/can_coke) are 0.25m+ away
# and were never observed to collide.
INSPECT_OBSTACLE_CLEAR_M = 0.20
INSPECT_MAX_S = 110.0         # time budget of the inspection tour (s)
PLACE_MAX_SHIFT_M = 0.25      # a reprojected original position farther than this from the grasp point is not trusted
PLACE_MIN_REACH_M = 0.33      # the arm's own reach window for a placement (grasp_executor REACH_MIN/MAX)
PLACE_MAX_REACH_M = 0.55
PLACE_COMFORT_MAX_M = 0.53    # farther than this the base is moved back first (it turns to face the spot when >3 deg off)
PLACE_PARK_DIST_M = 0.50      # where the base parks for a pick (reach.py), so where it returns to
PLACE_MAX_DRIVE_M = 0.15
PLACE_SETTLE_S = 1.5
# How far (m) to shift the arm horizontally away from the closest too-near
# obstacle before rotating. Also first-pass/not yet live-verified.
INSPECT_SHIFT_AWAY_M = 0.12


class WorldModelNode:

    def __init__(self):
        ns = rospy.get_param("~robot_name", "locobot")
        self.ns = ns
        default_catalog = rospkg.RosPack().get_path("locobot_prog") + "/config/objects.yaml"
        self.catalog = ObjectCatalog.from_yaml(rospy.get_param("~catalog", default_catalog))
        self.model = WorldModel(CatalogResolver(self.catalog), self.catalog,
                                rospy.get_param("~target_object", "bottle"))
        self._lock = threading.Lock()        # detection callback vs service threads
        self.K = None
        self.camera_frame = None             # camera optical frame, from camera_info

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.odom_cache = TargetOdomCache(self.tf_buffer, ns)
        self._map_point = None               # the target lock, in the map frame
        self._frame_stamp = rospy.Time()     # camera stamp of model.latest
        self.bridge = CvBridge()

        # Recent depth frames, for ~plan_grasp: a detection's bbox/stamp lags its
        # source image (YOLO on the CPU, ~1.1s), so the depth frame it belongs to
        # is found by nearest stamp, not "whatever's latest" — same as
        # rework-attempt-1's grasp_planner_node. Raw Image messages only (decoded
        # lazily, once, for whichever single frame a plan actually uses); at this
        # launch's 320x240 depth resolution ~150KB/frame, so 3s at 10Hz (the old
        # node's own sizing) is ~4.6MB — negligible next to this process's RSS,
        # and cv2/numpy are already loaded here via world_bev.py/cv_bridge.
        self._depth_lock = threading.Lock()  # separate from self._lock: different
                                             # rospy callback thread than detections
        self._depth_frame_step = float(rospy.get_param("~grasp_depth_frame_step_s", 0.1))
        buffer_s = float(rospy.get_param("~grasp_depth_buffer_s", 3.0))
        self._depth_frames = collections.deque(maxlen=int(buffer_s / self._depth_frame_step) + 1)
        self._plan_max_frame_offset = float(rospy.get_param("~grasp_plan_max_frame_offset", 0.25))

        self.bev_pub = rospy.Publisher("~bev_image", Image, queue_size=1, latch=True)
        rospy.Subscriber(f"/{ns}/locobot_prog/detections", DetectionArray,
                         self._on_detections, queue_size=1)
        rospy.Subscriber(f"/{ns}/camera/color/camera_info", CameraInfo,
                         self._on_camera_info, queue_size=1)
        rospy.Subscriber(f"/{ns}/camera/aligned_depth_to_color/image_raw", Image,
                         self._on_depth, queue_size=2, buff_size=2 ** 24)
        rospy.Service("~get_scene", GetScene, self._get_scene)
        rospy.Service("~get_target", GetTarget, self._get_target)
        rospy.Service("~target_memory", TargetMemory, self._target_memory)
        rospy.Service("~plan_grasp", PlanGrasp, self._handle_plan_grasp)
        rospy.Timer(rospy.Duration(1.0), self._publish_live_bev)
        rospy.loginfo("[world] Ready. Tracking '%s'; catalogue: %s",
                      self.model.target_name, ", ".join(self.catalog.objects))

    # ── Inputs ───────────────────────────────────────────────────────────────

    def _on_camera_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
            self.camera_frame = msg.header.frame_id

    def _on_depth(self, msg):
        """Buffer raw depth frames, throttled, for ~plan_grasp's nearest-stamp
        lookup. Ported from rework-attempt-1's grasp_planner_node._on_depth."""
        with self._depth_lock:
            if self._depth_frames:
                dt = (msg.header.stamp - self._depth_frames[-1].header.stamp).to_sec()
                if dt < 0:                    # sim time went back (reset): start over
                    self._depth_frames.clear()
                elif dt < self._depth_frame_step:
                    return
            self._depth_frames.append(msg)

    def _depth_frame_near(self, stamp):
        """The buffered depth frame closest to `stamp`, and how far from it in
        seconds; (None, inf) when empty."""
        with self._depth_lock:
            frames = list(self._depth_frames)
        if not frames:
            return None, math.inf
        best = min(frames, key=lambda f: abs((f.header.stamp - stamp).to_sec()))
        return best, abs((best.header.stamp - stamp).to_sec())

    def _camera_to_base(self, header):
        """4x4 transform from the depth frame to base_link as it is NOW, with the camera taken
        AT THE FRAME'S STAMP (a Time(0) lookup would use wherever the camera is now).

        Through odom, so the object the planner finds is placed in the frame the arm is about to
        work in: after the base moved between that image and now (a park correction), the
        old base frame put the grasp point where it had been, by the length of the move."""
        t = self.tf_buffer.lookup_transform_full(
            f"{self.ns}/base_link", rospy.Time(0), header.frame_id, header.stamp,
            f"{self.ns}/odom", rospy.Duration(0.3)).transform
        q, p = t.rotation, t.translation
        T = rotations.quaternion_matrix([q.x, q.y, q.z, q.w])
        T[:3, 3] = [p.x, p.y, p.z]
        return T

    def _handle_plan_grasp(self, req):
        try:
            return self._plan_grasp(req)
        except Exception as e:   # a planner fault must degrade to the baseline, not abort
            rospy.logerr("[world] plan_grasp failed: %r", e)
            return PlanGraspResponse(planned=False, reason=f"planner error: {e!r}")

    def _plan_grasp(self, req):
        res = PlanGraspResponse(planned=False, table_z=float("nan"))
        frame, offset = self._depth_frame_near(req.stamp)
        res.frame_offset = offset if math.isfinite(offset) else -1.0
        if frame is None or self.K is None or self.camera_frame is None:
            res.reason = "no depth frame or camera_info yet"
            return res
        if offset > self._plan_max_frame_offset:
            res.reason = f"no depth frame within {self._plan_max_frame_offset:.2f}s of the detection"
            return res
        try:
            T = self._camera_to_base(frame.header)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            res.reason = f"no camera pose at the frame's stamp: {e}"
            return res

        depth = self.bridge.imgmsg_to_cv2(frame, desired_encoding="passthrough")
        depth = depth.astype(np.float32) / 1000.0 if depth.dtype == np.uint16 \
            else depth.astype(np.float32)
        s = req.size
        dims = dims_hwl_from_size(s) if (s.x > 0 and s.y > 0 and s.z > 0) else None
        centre = (req.position.x, req.position.y)
        radius = gp.cut_radius(dims)
        # The table the caller assumed is a constant. Falls back to the caller's when it cannot be seen.
        table_z = req.table_z
        measured = footprint.table_height(depth, req.bbox, self.K, T, centre, radius, req.table_z)
        if measured is not None:
            res.table_z = measured
            table_z = measured
            rospy.loginfo("[world] table surface measured at z=%.3f (assumed %.3f, %+.0f mm)",
                          measured, req.table_z, (measured - req.table_z) * 1000)
        cloud = footprint.transform(footprint.backproject(depth, req.bbox, self.K), T)
        pts = footprint.object_points(cloud, table_z, center_xy=centre, radius=radius)
        bearing = math.atan2(centre[1], centre[0])
        # Opt-in per-band width dump: `rosparam set .../plan_grasp_debug_bands
        # true` to see the object's real measured width-by-height profile —
        # this is what actually decides the grasp height now (no artificial
        # window), so seeing it is the fast way to answer "why did/didn't it
        # pick that band" without re-deriving it from a point-cloud dump.
        if rospy.get_param("~plan_grasp_debug_bands", False):
            for b in footprint.bands(pts, table_z, bearing):
                rospy.loginfo("[world] plan_grasp band: z=%4.0fmm n=%4d width=%5.1fmm",
                              b.z * 1000, b.n, b.width * 1000)
        plan, diag = gp.plan_from_points(pts, table_z, bearing, dims,
                                         max_dev=req.max_axis_deviation)
        res.points = diag["points"]
        if plan is None:
            res.reason = diag["reason"]
            rospy.loginfo("[world] plan_grasp: no plan (%d points): %s", res.points, res.reason)
            return res
        res.planned = True
        res.reason = plan["reason"]
        res.plan = plan_to_msg(plan)
        res.contact_left = Point(*diag["contact_left"])
        res.contact_right = Point(*diag["contact_right"])
        res.width = diag["width"]
        res.seat_turn = diag["turn"]
        g = plan["grasp_point"]
        rospy.loginfo("[world] plan_grasp: (%d points, frame %+.2fs) (%.3f, %.3f, %.3f) "
                      "width %.0fmm turn %.0f°  %s", res.points, offset, g[0], g[1], g[2],
                      res.width * 1000, math.degrees(res.seat_turn), res.reason)
        return res

    def _on_detections(self, msg):
        dets = [detection_from_msg(d) for d in msg.detections]
        with self._lock:
            locked = self.model.update(dets, time.monotonic(), self.K)
            self._frame_stamp = msg.header.stamp
            table = self.model.table
        if locked is None:
            return
        # Hold the target in odometry (for when the arm occludes the camera) and in
        # the map (so it survives the base turning away from it).
        self.odom_cache.add_detection(locked["position_base"])
        self._map_point = self._transform(locked["position_base"], "base_link", "map")
        rospy.loginfo_throttle(5.0, "[world] Target locked: %s @ %s  (table z=%.3f near_x=%s)",
                               locked["class_name"], locked["position_base"],
                               table.table_z_for_grasp(),
                               "%.3f" % table.table_near_x if table.table_near_x else "?")

    def _transform(self, xyz, src, dst):
        """xyz from frame src to frame dst (base_link frames get the robot
        prefix), or None when TF cannot do it."""
        frame = lambda f: f"{self.ns}/{f}" if f == "base_link" else f  # noqa: E731
        pt = PointStamped()
        pt.header.frame_id = frame(src)
        pt.header.stamp = rospy.Time(0)
        pt.point.x, pt.point.y, pt.point.z = xyz
        try:
            out = self.tf_buffer.transform(pt, frame(dst), rospy.Duration(0.5))
            return (out.point.x, out.point.y, out.point.z)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "[world] TF %s -> %s failed: %s", src, dst, e)
            return None

    def _set_target(self, name):
        """Switch the tracked target when a request names a different one. Its
        odometry and map memory describe the old object, so they go too."""
        if self.model.set_target(name):
            self.odom_cache = TargetOdomCache(self.tf_buffer, self.ns)
            self._map_point = None
            rospy.loginfo("[world] Now tracking '%s'", name)

    def _key(self, label):
        keys = self.catalog.keys_for_label(label)
        return keys[0] if len(keys) == 1 else ""

    # ── Services ─────────────────────────────────────────────────────────────

    def _scene(self, req_target, nav_x_limit=0.0, standoff=0.0, use_cache=False):
        with self._lock:
            self._set_target(req_target)
            s = self.model.scene(time.monotonic(), nav_x_limit=nav_x_limit or None,
                                 standoff=standoff or None, use_cache=use_cache)
            obstacles = self.model.obstacles()
            table = self.model.table
            locked = self.model.lock is not None
            key = self.catalog.key_for_name(self.model.target_name) or ""
        return s, obstacles, table, locked, key

    def _render(self, s, table):
        img = world_bev.render_birdseye(s.objects, s.target, s.blocking, s.zone, s.grasp,
                                        table_bbox_m=table.table_bbox_m)
        msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        self.bev_pub.publish(msg)
        return msg

    def _get_scene(self, req):
        s, obstacles, table, locked, key = self._scene(
            req.target_name, req.nav_x_limit, req.standoff, req.use_cache)
        res = GetSceneResponse(
            catalog_key=key, target_found=s.target is not None,
            objects=[world_object_to_msg(o, self._key(o["name"])) for o in s.objects],
            blocking=[b["name"] for b in s.blocking], zone=s.zone,
            table=table_to_msg(table), summary=s.summary, visible=s.visible,
            detection_count=s.detection_count, from_cache=s.from_cache,
            target_locked=locked, obstacles=[box_to_msg(b) for b in obstacles])
        if s.target is not None:
            res.target = world_object_to_msg(s.target, key)
        if req.include_bev:
            res.bev = self._render(s, table)
        return res

    def _get_target(self, req):
        with self._lock:
            self._set_target(req.target_name)
            now = time.monotonic()
            m = self.model
            st = TargetState(target_name=m.target_name,
                             catalog_key=self.catalog.key_for_name(m.target_name) or "")
            if m.lock is not None:
                st.locked, st.lock_age = True, now - m.lock_t
                st.lock_position = Point(*m.lock["position_base"])
                st.lock_size, _ = vec_or_zero(m.lock["dims_hwl"])
                st.lock_orientation = float(m.lock.get("orientation", 0.0))
            recent = m.recent_positions(now)
            st.recent_count = len(recent)
            if recent:
                st.recent_median = Point(statistics.median(x for x, _ in recent),
                                         statistics.median(y for _, y in recent), 0.0)
            ft = m.frame_target()
            if ft is not None:
                st.in_frame = True
                st.frame_position = Point(*ft["position_base"])
                st.frame_size, _ = vec_or_zero(ft["dims_hwl"])
                st.frame_orientation = float(ft.get("orientation", 0.0))
                st.frame_bbox = [float(v) for v in ft["bbox_2d"][:4]]
                st.frame_stamp = self._frame_stamp
            st.calibrated_size, st.calibrated = vec_or_zero(m.target_calibrated_dims())
            map_point = self._map_point
        # Outside the lock: a TF lookup can block for up to 0.5 s.
        if map_point is not None:
            p = self._transform(map_point, "map", "base_link")
            if p is not None:
                st.map_valid, st.map_position = True, Point(p[0], p[1], 0.0)
        return GetTargetResponse(state=st)

    def _target_memory(self, req):
        if req.action == req.MEASURE:
            p = self.odom_cache.measure()
        elif req.action == req.REPROJECT:
            p = self.odom_cache.reproject()
        elif req.action == req.SET:
            p = self.odom_cache.set([req.position.x, req.position.y, req.position.z])
        elif req.action == req.FORGET:
            with self._lock:
                self.model.forget_target()
                self._map_point = None
            return TargetMemoryResponse(ok=True)
        else:
            return TargetMemoryResponse(ok=False)
        if p is None:
            return TargetMemoryResponse(ok=False)
        return TargetMemoryResponse(ok=True, position=Point(*p))

    def _publish_live_bev(self, _event):
        if self.bev_pub.get_num_connections() == 0:
            return
        s, _, table, _, _ = self._scene("")
        self._render(s, table)


class _Failed(Exception):
    """A step failed; `fields` is the reply to send (success is always false)."""

    def __init__(self, **fields):
        super().__init__(fields.get("description", ""))
        self.fields = fields


class TaskOrchestrator:

    def __init__(self):
        ns = rospy.get_param("~robot_name", "locobot")
        self.ns = ns
        self.grasp_dist = float(rospy.get_param("~arm_grasp_dist_m", ARM_GRASP_DIST))
        self._lock = threading.Lock()          # one task at a time
        # Own copy of the object catalogue (the same file the WorldModelNode loads): the
        # grasp plan needs to know whether the target is round and what its true size is.
        try:
            default_catalog = rospkg.RosPack().get_path("locobot_prog") + "/config/objects.yaml"
        except rospkg.ResourceNotFound:       # e.g. an offline test with no ROS package path
            default_catalog = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "..", "config", "objects.yaml")
        self.catalog = ObjectCatalog.from_yaml(rospy.get_param("~catalog", default_catalog))

        def proxy(name, srv_type):
            name = f"/{ns}/{name}"
            try:
                rospy.wait_for_service(name, timeout=30.0)
            except rospy.ROSException:
                rospy.logerr("[task] %s unavailable", name)
            return rospy.ServiceProxy(name, srv_type)

        self.get_scene = proxy("world_model/get_scene", GetScene)
        self.get_target = proxy("world_model/get_target", GetTarget)
        self.target_memory = proxy("world_model/target_memory", TargetMemory)
        self.plan_grasp = proxy("world_model/plan_grasp", PlanGrasp)
        self.approach = proxy("navigation/approach_target", ApproachTarget)
        self.drive_relative = proxy("navigation/drive_relative", DriveRelative)
        self.hold_heading = proxy("navigation/hold_heading", SetBool)
        self.set_arm_pose = proxy("arm/set_arm_pose", SetArmPose)
        self.execute_pregrasp = proxy("arm/execute_pregrasp", ExecutePregrasp)
        self.execute_grasp = proxy("arm/execute_grasp", ExecuteGrasp)
        self.execute_place = proxy("arm/execute_place", ExecutePlace)
        self.set_joint = proxy("arm/set_joint", SetJoint)
        self.shift_arm = proxy("arm/shift_arm", ShiftArm)
        self.turn_hand = proxy("arm/turn_hand", TurnHand)
        self.get_gripper_state = proxy("arm/get_gripper_state", GetGripperState)

        # Held-object state
        self._held = None
        # Raw colour frames, for inspect_object's view captures.
        self._latest_color = None
        rospy.Subscriber(f"/{ns}/camera/color/image_raw", Image,
                         self._on_color, queue_size=1, buff_size=2 ** 24)

        # MoveIt IK, only to CHOOSE the standoff (inverse reachability). Optional: without it the default standoff is used.
        self._joint_state = None
        rospy.Subscriber(f"/{ns}/joint_states", JointState, self._on_joint_state, queue_size=1)
        self._ik_srv = None
        try:
            rospy.wait_for_service(f"/{ns}/compute_ik", timeout=5.0)
            self._ik_srv = rospy.ServiceProxy(f"/{ns}/compute_ik", GetPositionIK)
            rospy.loginfo("[task] compute_ik available — reachability standoff enabled")
        except rospy.ROSException:
            rospy.logwarn("[task] compute_ik unavailable — using fixed grasp_dist")

        # Camera tilt, by range.
        self.tilt_far = float(rospy.get_param("~camera_tilt_far", 0.35))
        self.tilt_near = float(rospy.get_param("~camera_tilt_near", 0.45))
        self._tilt = None
        self._head = None             # (pan, tilt) last commanded by _move_head
        self._tilt_pub = rospy.Publisher(f"/{ns}/tilt_controller/command", Float64,
                                         queue_size=1, latch=True)
        self._pan_pub = rospy.Publisher(f"/{ns}/pan_controller/command", Float64,
                                        queue_size=1, latch=True)

        self._ground_truth = None     # Gazebo, only when a diag model is named
        self._ref_point = None        # (x, y, z) in base_link of the point last cached in the odometry memory

        # ABSOLUTE names, not "~get_object"/"~navigate_to" — see this file's
        # module docstring for why the merge requires this.
        rospy.Service(f"/{ns}/task/get_object", GetObject, self._handle_get_object)
        rospy.Service(f"/{ns}/task/navigate_to", NavigateTo, self._handle_navigate_to)
        rospy.Service(f"/{ns}/task/inspect_object", InspectObject, self._handle_inspect_object)
        rospy.loginfo("[task] Ready.")

    def _on_joint_state(self, msg):
        self._joint_state = msg

    def _on_color(self, msg):
        self._latest_color = msg

    def _current_joint(self, name):
        js = self._joint_state
        if js is None or name not in js.name:
            return None
        return js.position[js.name.index(name)]

    def _set_tilt(self, tilt):
        """Point the camera at `tilt` rad and wait for frames from the new view.
        Does nothing when it is already there."""
        if self._tilt == tilt:
            return
        self._tilt_pub.publish(Float64(tilt))
        self._tilt = tilt
        rospy.sleep(TILT_SETTLE_S)

    def _wait_in_frame(self, target_name, st):
        """`st`, or a newer target state in which the target is in the current
        camera frame, waiting briefly for one. Returns the last state read when
        none arrives."""
        for _ in range(IN_FRAME_WAIT_TRIES):
            if st.in_frame:
                break
            rospy.sleep(IN_FRAME_WAIT_S)
            st = self.get_target(target_name=target_name).state
        return st

    def _remember(self, action, position=None):
        """target_memory MEASURE/SET that also keeps the cached point. REPROJECT
        later returns where that same world point is in the CURRENT base frame;
        the difference is how far the base has tilted/crept since (see
        _follow_tilt)."""
        kwargs = {"action": action}
        if position is not None:
            kwargs["position"] = position
        r = self.target_memory(**kwargs)
        if r.ok:
            self._ref_point = (r.position.x, r.position.y, r.position.z)
        return r

    def _settled_shift(self):
        """How far the target has moved in the CURRENT base frame since the plan
        was measured, as (dx, dy, dz) metres — or None when it cannot be measured
        or is too small to matter.

        The arm's own weight tilts the base nose-down once it extends to the
        standoff, and that tilt STAYS after the arm retracts (ground truth,
        Gazebo): the bottle rose ~25 mm and moved ~14 mm back relative to the
        arm, all of it between grasp_ready and the standoff — none at
        grasp_ready. So it can only be measured once the arm is loaded, i.e.
        after the first pre-grasp. The odometry memory holds the target fixed in
        the world, so REPROJECT sees exactly that shift (camera-free, unlike a
        fresh detection, which the arm now occludes).

        Absorbing it here, by re-taking the pre-grasp at the corrected pose,
        keeps the straight-in leg horizontal at a standoff that was verified at
        the right height. Absorbing it in the leg instead meant a slope that
        dragged the pads over the bottle's shoulder, or a raise that hit the IK
        limits (73-77% path)."""
        if self._ref_point is None:
            return None
        mem = self.target_memory(action=TargetMemoryRequest.REPROJECT)
        if not mem.ok:
            return None
        ref, now = self._ref_point, (mem.position.x, mem.position.y, mem.position.z)
        # Same verdicts the arm's own descent gate uses: "ok" = within tolerance,
        # "refused" = too far to be a tilt/creep at all.
        _, vxy, dxy_mm = gg.descent_aim(ref[:2], now[:2], SHIFT_FOLLOW_XY_MM, gg.REAIM_MAX_MM)
        _, vz, dz_mm = gg.descent_height(ref[2], now[2], gg.REAIM_Z_TOL_MM, gg.REAIM_Z_MAX_MM)
        if "refused" in (vxy, vz):
            rospy.logwarn("[task] Target moved (%.0f mm sideways, %+.0f mm in height) since it "
                          "was measured — too much to follow; the arm's own gate decides",
                          dxy_mm, dz_mm)
            return None
        if vxy == "ok" and vz == "ok":
            return None
        return tuple(n - r for n, r in zip(now, ref))

    def _handle_navigate_to(self, req):
        with self._lock:
            try:
                ctx = self._navigate(req.target_name)
            except _Failed as f:
                return NavigateToResponse(success=False, **_only(f.fields, NavigateToResponse))
            bev = self.get_scene(target_name=req.target_name, include_bev=True).bev
            return NavigateToResponse(
                success=True, distance=ctx["distance"], zone=ctx["zone"],
                scene_summary=ctx["summary"], attach_camera_frame=True, bev=bev,
                description=(f"Positioned {ctx['distance']:.2f}m in front of the "
                             f"{req.target_name}, facing it. Nothing was grasped."))

    def _handle_get_object(self, req):
        with self._lock:
            if self._held is not None:
                return GetObjectResponse(
                    success=False, gripper_width=0.0,
                    description=(f"Already holding '{self._held['target_name']}' — call "
                                 f"inspect_object first (it places it back down) before "
                                 f"picking anything else."))
            try:
                ctx = self._navigate(req.target_name)
                r = self._pick(req.target_name, ctx)
            except _Failed as f:
                return GetObjectResponse(success=False, **_only(f.fields, GetObjectResponse))
            except Exception:
                self._hold(False)
                raise
            if self._held is None:
                self._hold(False)              # nothing in the gripper: nothing to keep steady
            return r

    def _handle_inspect_object(self, req):
        with self._lock:
            if self._held is None:
                return InspectObjectResponse(
                    success=False,
                    message="Nothing is currently held — call get_object first.")
            held = self._held
            self._held = None   # cleared up front: inspect_object always places,
                                # so nothing stays "held" past this call either way
            views, desc = self._rotate_and_capture(held)
            point, exact = self._place_point(held)
            placed = self.execute_place(place_point=point, at_grasp_height=not exact, exact=exact)
            self._hold(False)
            try:
                self.target_memory(action=TargetMemoryRequest.FORGET)
            except rospy.ServiceException:
                pass
            if not placed.success:
                desc += f" Place step failed: {placed.message or 'unknown'}."
            rospy.loginfo("[task] inspect_object: %d view(s), placed=%s",
                          len(views), placed.success)
            return InspectObjectResponse(success=bool(views) and placed.success,
                                         message="" if views else "No views captured.",
                                         views=views, description=desc)

    def _place_point(self, held):
        """(Point, exact): where to put the held object down. The object's ORIGINAL position,
        reprojected from the odometry memory into the base frame as it is now.

        Measured 2026-10-19: between the grasp and the end of an inspection the base rolls
        back 5-11 cm and yaws 6-11 degrees, so the original spot ends up 0.60-0.63 m from the
        base, beyond the arm's reach; placing at the grasp point in the OLD base frame put the
        object 10-15 cm from where it was picked up. So when the original position is too far
        or off to the side, the base first turns and drives back to where it parked (one
        move, never more than PLACE_MAX_DRIVE_M forward), then the position is reprojected
        again. Odometry under-reads the backward roll by 3-4 cm, so a few cm of error remain.

        Falls back to the position noted at the grasp, in the old base frame, when the memory is
        gone, implausible, or still out of reach."""
        fallback = (Point(held["pos"][0], held["pos"][1], 0.0), False)

        def reproject():
            try:
                mem = self.target_memory(action=TargetMemoryRequest.REPROJECT)
            except rospy.ServiceException:
                return None
            return (mem.position.x, mem.position.y, mem.position.z) if mem.ok else None

        ref = reproject()
        if ref is None:
            return fallback
        moved = math.hypot(ref[0] - held["pos"][0], ref[1] - held["pos"][1])
        if moved > PLACE_MAX_SHIFT_M:
            rospy.logwarn("[task] inspect_object: the remembered original position (%.3f, %.3f) is "
                          "%.0f mm from the grasp point — not trusted, placing at the grasp point",
                          ref[0], ref[1], moved * 1000)
            return fallback
        dist, bearing = math.hypot(ref[0], ref[1]), math.atan2(ref[1], ref[0])
        if dist > PLACE_COMFORT_MAX_M:       # sideways offsets are the arm's waist; only distance needs the base
            # A straight drive only: turning to face the spot is not needed (the waist takes the
            # sideways offset) and the closed-loop turn ran to its 20 s deadline every time
            # (13 degrees in 20 s, 2026-10-20) while the bottle hung in the gripper. The drive
            # brings the straight-line distance to the park distance: the point's x moves by
            # d with hypot(x - d, y) = PLACE_PARK_DIST_M.
            ahead = math.sqrt(max(0.0, PLACE_PARK_DIST_M ** 2 - ref[1] ** 2))
            drive = max(-PLACE_MAX_DRIVE_M, min(PLACE_MAX_DRIVE_M, ref[0] - ahead))
            rospy.loginfo("[task] inspect_object: the base has crept — the original position is now "
                          "%.3fm away at %+.1f deg; driving %+.3fm straight back",
                          dist, math.degrees(bearing), drive)
            self._hold(False)                  # the drive-back moves the base
            try:
                r = self.drive_relative(turn=0.0, distance=drive, max_angular=0.0, max_linear=0.0)
                if not r.success:
                    rospy.logwarn("[task] inspect_object: the base move was refused: %s", r.message)
            except rospy.ServiceException as e:
                rospy.logwarn("[task] inspect_object: the base move failed: %s", e)
            rospy.sleep(PLACE_SETTLE_S)
            self._hold(True)                   # and hold where it ended up
            ref = reproject()
            if ref is None:
                return fallback
        if not (PLACE_MIN_REACH_M <= math.hypot(ref[0], ref[1]) <= PLACE_MAX_REACH_M):
            rospy.logwarn("[task] inspect_object: the original position (%.3f, %.3f) is out of the "
                          "arm's reach — placing at the grasp point instead", ref[0], ref[1])
            return fallback
        rospy.loginfo("[task] inspect_object: placing back at the original position (%.3f, %.3f)",
                      ref[0], ref[1])
        return Point(ref[0], ref[1], ref[2]), True

    def _dodge_nearby_obstacle(self, held):
        """If another catalogued object sits closer than INSPECT_OBSTACLE_CLEAR_M
        to the held object's own (unchanged since retreat is vertical-only)
        grasp point, shift the arm horizontally away from it before rotating.
        Never raises and never blocks the sweep — a failed scene lookup or a
        failed shift just means it proceeds unshifted, same as before this
        existed."""
        try:
            obstacles = list(self.get_scene(target_name=held["target_name"]).obstacles)
        except rospy.ServiceException as e:
            rospy.logwarn("[task] inspect_object: could not check for nearby "
                          "obstacles: %s", e)
            return
        hx, hy = held["pos"]
        closest, closest_d = None, math.inf
        for o in obstacles:
            d = math.hypot(o.center.x - hx, o.center.y - hy)
            if d < closest_d:
                closest, closest_d = o, d
        if closest is None or closest_d >= INSPECT_OBSTACLE_CLEAR_M:
            return
        away_x, away_y = hx - closest.center.x, hy - closest.center.y
        norm = math.hypot(away_x, away_y)
        if norm < 1e-6:
            return   # degenerate — coincide with the obstacle's centre; nothing sane to dodge toward
        away_x, away_y = away_x / norm * INSPECT_SHIFT_AWAY_M, away_y / norm * INSPECT_SHIFT_AWAY_M
        rospy.loginfo("[task] inspect_object: '%s' is %.0fmm from the held object — "
                      "shifting the arm (%+.0f,%+.0f)mm away before rotating",
                      closest.name, closest_d * 1000, away_x * 1000, away_y * 1000)
        r = self.shift_arm(dx=away_x, dy=away_y, dz=0.0, move_time=1.5)
        if not r.success:
            rospy.logwarn("[task] inspect_object: obstacle-dodge shift failed: %s", r.message)

    def _rotate_and_capture_wrist(self, held):
        """Sweep wrist_rotate through inspect_offsets_for(held height) around
        wherever the grasp left it, capturing one frame per step. Never
        raises — a failed step is just skipped, so the caller can still
        place what was held.

        wrist_rotate is the gripper's own roll axis, but a SIDE grip holds the
        object with that axis roughly horizontal — rotating it tilts the held
        object rather than spinning it about its own vertical axis (live-
        verified: a bottle came out of a +-0.5 rad sweep lying on its side).
        That tilt is fine for inspection (different angles are still shown),
        but place() always lowers straight down assuming the object is back in
        its AT-GRASP orientation — so the sweep must end back at `center`
        before placing, not at the last offset, or the object gets set down
        tilted and topples.

        Before any of that: the post-grasp retreat is vertical-only (same x,y
        as the grasp point, see grasp_executor.place's retreat comment), so a
        held object sits at essentially its ORIGINAL table position for the
        whole sweep. Live-observed 2026-10-07: a bottle (0.23m, tilts through
        a wide arc for a given wrist_rotate angle since the grip is nowhere
        near its far end) destabilised the whole arm mid-sweep, matching
        GazeboGraspFix attach/detach chatter and wild camera jumps — and
        locobot_table.world happens to put the ball only 0.12m from the
        bottle's own position, well within reach of that swing; a shorter
        object (can_coke, 0.082m) swings far less for the same angle and
        tested clean. _dodge_nearby_obstacle (below) checks for exactly this
        and shifts the arm horizontally clear of the nearest catalogued
        neighbour first, without driving the robot base.

        `baseline` (the full 6-joint vector, not just wrist_rotate) comes from
        ONE NaN set_joint call (pure read, no motion) and is then passed back
        as hold_positions on every subsequent call, rather than letting
        arm_node re-read its own last_target fresh each time. Live-observed
        2026-10-07: without this, a held bottle came out of the full sweep
        visibly tilted even though wrist_rotate itself correctly returned to
        its exact starting value — hold()'s settle-to-measured behaviour
        between calls let tiny per-step measurement noise in the other 5
        joints (forearm_roll, wrist_angle, ...) accumulate call over call, so
        locking one snapshot for the whole sweep is what actually keeps the
        rest of the arm still. (The trusted-read reasoning for not using this
        process's own joint_states for the read itself still applies — see
        arm_node.py's hold() docstring: the wrist is spherical, two joint-
        space solutions per EE pose.)"""
        target_name = held["target_name"]
        self._dodge_nearby_obstacle(held)
        r0 = self.set_joint(joint_name="wrist_rotate", position=float("nan"), move_time=0.0)
        if r0.success:
            baseline = list(r0.joint_positions)
        else:
            baseline = [self._current_joint(j) or 0.0 for j in ARM_JOINTS]
        center = baseline[ARM_JOINTS.index("wrist_rotate")]
        offsets = inspect_offsets_for(held.get("height"))
        rospy.loginfo("[task] inspect_object: sweeping wrist_rotate +-%.2f rad "
                      "(held height %s m)", max(abs(o) for o in offsets),
                      "%.3f" % held["height"] if held.get("height") else "unknown")
        views = []
        for off in offsets:
            r = self.set_joint(joint_name="wrist_rotate", position=center + off,
                               move_time=INSPECT_STEP_MOVE_TIME, hold_positions=baseline)
            if not r.success:
                rospy.logwarn("[task] inspect_object: set_joint(%.2f) failed: %s",
                              center + off, r.message)
                continue
            rospy.sleep(INSPECT_STEP_SETTLE_S)
            if self._latest_color is not None:
                views.append(self._latest_color)
        r = self.set_joint(joint_name="wrist_rotate", position=center,
                           move_time=INSPECT_STEP_MOVE_TIME, hold_positions=baseline)
        if not r.success:
            rospy.logwarn("[task] inspect_object: failed to return to centre (%.2f) "
                          "before placing: %s", center, r.message)
        else:
            rospy.sleep(INSPECT_STEP_SETTLE_S)
        desc = f"Rotated '{target_name}' through {len(views)} view(s) for inspection."
        return views, desc

    def _head_now(self):
        """(pan, tilt) the head is at: measured when joint_states has it, else the last
        angles commanded here, else the fixed forward view."""
        pan, tilt = self._current_joint("pan"), self._current_joint("tilt")
        if pan is None or tilt is None:
            pan, tilt = self._head if self._head is not None else (0.0, self._tilt or self.tilt_far)
        return pan, tilt

    def _move_head(self, target, settle):
        """Take the head to `target` (pan, tilt): as a minimum-jerk ramp of position commands
        (`~camera_smooth`, default true), else as one step command like before."""
        if rospy.get_param("~camera_smooth", True):
            pts, dt = camera_aim.head_path(self._head_now(), target)
            for pan, tilt in pts:
                self._pan_pub.publish(Float64(pan))
                self._tilt_pub.publish(Float64(tilt))
                rospy.sleep(dt)
        else:
            self._pan_pub.publish(Float64(target[0]))
            self._tilt_pub.publish(Float64(target[1]))
        self._head = target
        rospy.sleep(settle)

    def _aim_camera(self, point, settle=INSPECT_CAMERA_SETTLE_S):
        """Point the camera head at `point` (base_link), smoothly, and let it settle."""
        pan, tilt = camera_aim.aim_angles(point)
        self._tilt = None            # the head is no longer at one of the fixed tilts
        smooth = rospy.get_param("~camera_smooth", True)
        self._move_head((pan, tilt), INSPECT_CAMERA_SETTLE_SMOOTH_S if smooth else settle)
        return pan, tilt

    def _restore_camera(self):
        """Back to the fixed view every other step assumes (pan 0, near tilt)."""
        self._move_head((0.0, self.tilt_near), 0.3)
        self._tilt = self.tilt_near

    def _rotate_and_capture(self, held):
        """The inspection sweep: `~inspect_mode` "yaw" (default) turns the HAND in place and
        keeps the camera on the held object; "wrist" is the older wrist_rotate sweep. The
        hand sweep falls back to the wrist sweep when no hand yaw can be reached."""
        if rospy.get_param("~inspect_mode", "yaw") == "wrist":
            return self._rotate_and_capture_wrist(held)
        views, desc = self._rotate_and_capture_yaw(held)
        if views:
            return views, desc
        rospy.logwarn("[task] inspect_object: no hand yaw was reachable — falling back to the "
                      "wrist sweep")
        return self._rotate_and_capture_wrist(held)

    def _rotate_and_capture_yaw(self, held):
        """Show the held object from several sides and angles with the arm out of the way.

        The camera sits right above the arm base, so an object held in front is seen past
        the elbow and forearm (measured 2026-10-19: even aimed at it, the forearm covers the
        middle of the frame, and pulling the object closer makes it worse). Yawing the hand
        about the vertical through the held point (srv/TurnHand) turns the object about its
        own axis AND swings the forearm to one side; at +-0.5 rad the whole bottle is clear
        in the frame. A tour of such poses (camera_aim.INSPECT_POSES: turns to both sides, a
        closer one, and views with the top tilted toward the camera) is walked in the order of
        camera_aim.inspect_tour, returning to the start between sides, and the head is aimed at
        the object's centre before every capture.

        Bounded on purpose: `~inspect_max_views` poses at most, and `~inspect_max_s` seconds
        (a hold of about eight poses lost the bottle in one run; the tour stops early rather
        than risk it), and it also stops when the gripper no longer reports holding. Never
        raises: an unreachable pose is skipped, and the hand is always returned to where it
        started before the caller places the object."""
        target_name = held["target_name"]
        self._dodge_nearby_obstacle(held)
        try:
            start = self.turn_hand(yaw=0.0, pitch=0.0, dx=0.0, dy=0.0, dz=0.0, move_time=0.0)  # no sequence open: a pure read
        except rospy.ServiceException as e:
            rospy.logwarn("[task] inspect_object: turn_hand unavailable: %s", e)
            return [], ""
        if not start.success:
            return [], ""
        include_tilt = bool(rospy.get_param("~inspect_tilt_views", False))
        max_views = int(rospy.get_param("~inspect_max_views", len(camera_aim.INSPECT_POSES)))
        max_s = float(rospy.get_param("~inspect_max_s", INSPECT_MAX_S))
        t_start = time.time()
        views, labels = [], []
        try:
            for pose in camera_aim.inspect_tour(max_views, include_tilt=include_tilt):
                if pose is None:
                    self.turn_hand(yaw=0.0, pitch=0.0, dx=0.0, dy=0.0, dz=0.0, move_time=0.0)   # exactly back to the start
                    continue
                if time.time() - t_start > max_s:
                    rospy.logwarn("[task] inspect_object: %.0f s budget used — stopping the tour "
                                  "after %d view(s)", max_s, len(views))
                    break
                r = self.turn_hand(yaw=pose.yaw, pitch=pose.pitch, dx=pose.dx, dy=0.0, dz=pose.dz,
                                   move_time=0.0)
                if not r.success:
                    rospy.loginfo("[task] inspect_object: pose '%s' skipped: %s", pose.label, r.message)
                    continue
                ee = (r.ee_position.x, r.ee_position.y, r.ee_position.z)
                centre = camera_aim.held_centre(ee, held.get("grasp_above_table"), held.get("height"),
                                                pose.pitch)
                pan, tilt = self._aim_camera(centre)
                rospy.sleep(INSPECT_STEP_SETTLE_S)
                if self._latest_color is not None:
                    views.append(self._latest_color)
                    labels.append(pose.label)
                rospy.loginfo("[task] inspect_object: view %d '%s', camera pan %+.2f tilt %+.2f",
                              len(views), pose.label, pan, tilt)
                try:
                    if not self.get_gripper_state().holding:
                        rospy.logwarn("[task] inspect_object: the gripper no longer reports holding "
                                      "— ending the tour")
                        break
                except rospy.ServiceException:
                    pass
        finally:
            try:
                self.turn_hand(yaw=0.0, pitch=0.0, dx=0.0, dy=0.0, dz=0.0, move_time=0.0)   # no-op when already back
            except rospy.ServiceException:
                pass
            self._restore_camera()
        if not views:
            return [], ""
        listing = "; ".join(f"{i + 1}) {lab}" for i, lab in enumerate(labels))
        desc = (f"Showed '{target_name}' in {len(views)} view(s) with the camera aimed at it, in this "
                f"order: {listing}.")
        return views, desc

    # ── Steps ────────────────────────────────────────────────────────────────

    def _navigate(self, target_name):
        """Observe, choose the standoff, and put the base there. Returns the
        context the grasp needs; raises _Failed with the reply otherwise."""
        self._set_tilt(self.tilt_far)
        # 1. Observe. An empty frame is answered from the recent cache; with no
        #    detections and no lock at all, give perception a moment.
        s = self.get_scene(target_name=target_name, use_cache=True)
        if s.from_cache:
            rospy.loginfo("[task] Using cached detections")
        if s.detection_count == 0 and not s.target_locked:
            rospy.loginfo("[task] No detections — waiting 2 s")
            rospy.sleep(2.0)
            s = self.get_scene(target_name=target_name)
        rospy.loginfo("[task] Scene: zone=%s target=%s detections=%d locked=%s",
                      s.zone, s.target_found, s.detection_count, s.target_locked)
        if not s.target_found:
            visible = list(s.visible)
            raise _Failed(gripper_width=0.0, zone=s.zone, scene_summary=s.summary,
                          visible_objects=visible, attach_camera_frame=True,
                          description=(f"'{target_name}' not detected. "
                                       f"Visible objects: {visible if visible else 'none'}. "
                                       "Check the image — call get_object with the correct "
                                       "name if needed."))

        # 2. The standoff where the arm is most comfortable for THIS object.
        st = self.get_target(target_name=target_name).state
        dims = (dims_hwl_from_size(st.lock_size) if st.locked
                else dims_hwl_from_size(st.frame_size) if st.in_frame else None)
        # ~standoff_override_m > 0 parks the base at that distance instead (experiments).
        override = float(rospy.get_param("~standoff_override_m", 0.0))
        standoff = override if override > 0 else self._reach_optimal_standoff(dims, s.table.surface_z)
        if override > 0:
            rospy.loginfo("[task] Standoff override: %.2fm", override)

        raw_near_x = (s.table.near_x if s.table.edge_known
                      else s.target.position.x - TABLE_DEPTH_M / 2.0)
        table_near_x = max(raw_near_x, ARM_GRASP_DIST + 0.05)
        grasp_table_near_x = max(raw_near_x, 0.40)
        s = self.get_scene(target_name=target_name, nav_x_limit=table_near_x, standoff=standoff)
        zone, summary = s.zone, s.summary

        # 3. Coarse approach when out of range.
        if zone != ZONE_GRASP_READY:
            if zone == ZONE_BLOCKED:
                b = s.blocking[0] if s.blocking else "unknown"
                raise _Failed(gripper_width=0.0, zone=zone, scene_summary=summary,
                              description=f"Path blocked by {b} — cannot proceed.")
            r = self._approach(target_name, standoff, table_near_x, ApproachTargetRequest.COARSE)
            if not r.success:
                raise _Failed(gripper_width=0.0, zone=zone, scene_summary=summary,
                              description=r.message)
            rospy.sleep(0.5)
            s = self.get_scene(target_name=target_name, nav_x_limit=table_near_x,
                               standoff=standoff)
            zone, summary = s.zone, s.summary
            if not s.target_found:
                raise _Failed(gripper_width=0.0, zone=zone, scene_summary=summary,
                              description=f"'{target_name}' lost after navigation.")

        # 4. Single-shot fine positioning.
        r = self._approach(target_name, standoff, table_near_x, ApproachTargetRequest.FINE)
        if not r.success:
            fields = dict(gripper_width=0.0, zone=r.zone or zone,
                          scene_summary=r.scene_summary or summary, description=r.message)
            if r.too_close:
                fields["phase"] = "positioning"
            raise _Failed(**fields)
        return {"scene": s, "standoff": standoff, "table_near_x": table_near_x,
                "grasp_table_near_x": grasp_table_near_x, "zone": ZONE_GRASP_READY,
                "summary": summary, "distance": r.distance}

    def _pick(self, target_name, ctx):
        s, standoff = ctx["scene"], ctx["standoff"]
        zone, summary = ctx["zone"], ctx["summary"]

        # 5. Where exactly to grasp. Tuck the arm FIRST so the camera has an
        #    unobstructed view, then read the target with the robot standing
        #    still: a lock made 12 s earlier and reprojected through open-loop
        #    drive odometry once put the grasp 5-6 cm past the real bottle.
        self.set_arm_pose(pose_name="camera_clear")
        self._set_tilt(self.tilt_near)   # the whole target in frame, base to top
        rospy.sleep(1.2)   # let perception deliver frames of the unobstructed scene
        st = self.get_target(target_name=target_name).state
        # A single read here can catch perception between frames
        # The depth planner needs the target in the CURRENT frame, and a small
        # object close up is detected only intermittently (the can at 0.40 m
        # scores ~0.12 against a 0.10 threshold). Without this wait a fresh
        # lock was enough to proceed, the plan fell to the bbox baseline, and
        # the baseline's ASSUMED table height put the jaws ~20 mm too low on
        # the can. Standing still, a detecting frame is a fraction of a second
        # away.
        st = self._wait_in_frame(target_name, st)
        for _ in range(4):
            if st.in_frame or (st.locked and st.lock_age < GRASP_MAX_LOCK_AGE):
                break
            rospy.logwarn("[task] No fresh detection of '%s' yet (lock_age=%.1fs) — "
                          "waiting for perception to catch up", target_name, st.lock_age)
            rospy.sleep(1.2)
            st = self.get_target(target_name=target_name).state
        p = measured_position(st, GRASP_MAX_LOCK_AGE)
        if p is not None:
            pos = [round(p[0], 3), round(p[1], 3)]
            rospy.loginfo("[task] Grasp position from %s: (%.3f, %.3f)", p[2], *pos)
        else:
            pos = [s.target.position.x, s.target.position.y]
            rospy.loginfo("[task] Grasp position from the last scene: (%.3f, %.3f)", *pos)
        # Hold the target in odometry now, while the robot is still and the camera
        # clear: at grasp height the arm occludes the camera.
        if not self._remember(TargetMemoryRequest.MEASURE).ok:
            rospy.logwarn("[task] Odometry target memory not primed (too few fresh "
                          "frames) — the contact check will be unverified")
        # Size: the far-vantage calibration is the most accurate source and the
        # only one that does not degrade as the robot closes in.
        dims = (dims_hwl_from_size(st.calibrated_size) if st.calibrated
                else dims_hwl_from_size(st.frame_size) if st.in_frame
                else dims_hwl_from_size(st.lock_size) if st.locked else None)
        # Orientation has no calibrated-over-time analogue — same frame/lock priority as dims, one level simpler.
        orientation = (st.frame_orientation if st.in_frame
                      else st.lock_orientation if st.locked else None)
        grasp_axis = s.target.grasp_axis or "along_x"
        tz = s.table.surface_z
        scene_obstacles = list(self.get_scene(target_name=target_name).obstacles)
        # How far everything has moved in base_link since it was measured (the base
        # tilts/creeps under the arm) — the collision scene must move with it.
        tilt = (0.0, 0.0, 0.0)

        def moved(box):
            c = box.center
            return CollisionBox(name=box.name, size=box.size,
                                center=Point(c.x + tilt[0], c.y + tilt[1], c.z + tilt[2]))

        def table_msg():
            return box_to_msg(gg.table_box(tz, s.table.width or 0.80, s.table.depth or 0.40,
                                           ctx["grasp_table_near_x"], pos[0]))

        def target_box():
            return box_to_msg(gg.target_box(pos, dims, tz + tilt[2]))

        # 6. PRE-GRASP — measured, and refused from a bad pose. A failed one
        #    retracts the arm, so the base can be corrected and it can retry once.
        # The plan is chosen once per attempt and reused for this attempt's
        # execute_grasp call below — NOT recomputed there (see _choose_plan).
        plan_msg, tz = self._choose_plan(st, pos, dims, grasp_axis, tz, orientation,
                                          target_name=target_name)
        # 5b. PARK DISTANCE, judged from the planner's grasp point (see PARK_GRASP_*):
        #     one straight drive to bring it into the window, then measure and plan again.
        gd = math.hypot(plan_msg.grasp_point.x, plan_msg.grasp_point.y)
        if not (PARK_GRASP_MIN_M <= gd <= PARK_GRASP_MAX_M):
            move = round(gd - PARK_GRASP_TARGET_M, 3)
            rospy.loginfo("[task] Grasp point %.3fm from the base (window %.2f-%.2fm) — "
                          "driving %+.3fm before the pre-grasp", gd, PARK_GRASP_MIN_M,
                          PARK_GRASP_MAX_M, move)
            try:
                dr = self.drive_relative(turn=0.0, distance=move, max_angular=0.0, max_linear=0.0)
                if not dr.success:
                    rospy.logwarn("[task] Park correction refused: %s", dr.message)
            except rospy.ServiceException as e:
                rospy.logwarn("[task] Park correction failed: %s", e)
            rospy.sleep(1.5)
            ctx["grasp_table_near_x"] -= move
            st = self._wait_in_frame(target_name, self.get_target(target_name=target_name).state)
            p = measured_position(st, GRASP_MAX_LOCK_AGE)
            if p is not None:
                pos = [round(p[0], 3), round(p[1], 3)]
            self._remember(TargetMemoryRequest.MEASURE)
            scene_obstacles = list(self.get_scene(target_name=target_name).obstacles)
            plan_msg, tz = self._choose_plan(st, pos, dims, grasp_axis, tz, orientation,
                                          target_name=target_name)
            rospy.loginfo("[task] After the park correction: grasp point %.3fm from the base",
                          math.hypot(plan_msg.grasp_point.x, plan_msg.grasp_point.y))
        # The base is parked: hold its heading from here through the lift, and through the
        # inspection until the object is put down (released in the handlers below).
        self._hold(True)
        pre, failures, followed, reaim_widened = None, 0, False, False
        retake_blocked, saved = False, None
        while True:
            pre = self.execute_pregrasp(
                plan=plan_msg, obstacles=[moved(table_msg())] + [moved(o) for o in scene_obstacles],
                target_box=target_box(), table_z=tz + tilt[2])
            if pre.success:
                rospy.loginfo("[task] Pre-grasp verified (%s): %s",
                              "tilt-corrected" if followed else f"attempt {failures + 1}",
                              pre.message or "ok")
                # The arm is loaded now, so the base has tilted: is the target where
                # this pre-grasp assumed? If not, take the pre-grasp again THERE (once).
                if retake_blocked:
                    # Back at the ORIGINAL standoff after the tilt-corrected one was
                    # blocked: let the descent's re-aim follow the tilt instead.
                    rospy.set_param(f"/{self.ns}/arm/reaim_max_mm", RETAKE_REAIM_MAX_MM)
                    reaim_widened = True
                    break
                d = None if followed else self._settled_shift()
                if d is None:
                    break
                followed = True
                saved = (plan_msg.grasp_point.x, plan_msg.grasp_point.y, plan_msg.grasp_point.z,
                         list(pos), self._ref_point, tilt)
                tilt = (tilt[0] + d[0], tilt[1] + d[1], tilt[2] + d[2])
                plan_msg.grasp_point.x += d[0]
                plan_msg.grasp_point.y += d[1]
                plan_msg.grasp_point.z += d[2]
                pos = [round(pos[0] + d[0], 3), round(pos[1] + d[1], 3)]
                self._ref_point = (self._ref_point[0] + d[0], self._ref_point[1] + d[1],
                                   self._ref_point[2] + d[2])
                rospy.loginfo("[task] Base tilted under the arm: target is (%+.0f, %+.0f, %+.0f)mm "
                              "from where it was planned — re-taking the pre-grasp at "
                              "(%.3f, %.3f, %.3f)", d[0] * 1000, d[1] * 1000, d[2] * 1000,
                              plan_msg.grasp_point.x, plan_msg.grasp_point.y,
                              plan_msg.grasp_point.z)
                continue
            if followed and not retake_blocked:
                # The tilt-corrected standoff could not be reached (measured: the
                # collision-aware approach blocked at 35-47% for the angled tea-box
                # approach). The original standoff WAS verified a moment ago: go back
                # to it and let the descent follow the tilt (~30 mm, +20 mm).
                retake_blocked = True
                (plan_msg.grasp_point.x, plan_msg.grasp_point.y, plan_msg.grasp_point.z,
                 pos, self._ref_point, tilt) = saved
                rospy.logwarn("[task] Tilt-corrected pre-grasp blocked (%s) — going back to the "
                              "original standoff; the descent will follow the tilt", pre.message)
                continue
            failures += 1
            rospy.logwarn("[task] Pre-grasp NOT verified (attempt %d): %s", failures, pre.message)
            if failures == 2:
                break
            rospy.sleep(1.2)
            self._hold(False)                  # the correction turns/drives the base
            r = self._approach(target_name, standoff, ctx["table_near_x"],
                               ApproachTargetRequest.CORRECT)
            if not r.success:
                break
            self._hold(True)                   # hold the corrected heading
            pos = [round(r.target.x, 3), round(r.target.y, 3)]
            self._remember(TargetMemoryRequest.MEASURE)   # re-prime here
            # The base moved, so what was measured before no longer applies.
            tilt, followed = (0.0, 0.0, 0.0), False
            # Refresh st too: a corrected base pose means a new camera pose, so
            # the depth plan's buffered frame/TF lookup must describe THIS pose,
            # not the pre-correction one the original st still holds.
            st = self._wait_in_frame(target_name,
                                     self.get_target(target_name=target_name).state)
            plan_msg, tz = self._choose_plan(st, pos, dims, grasp_axis, tz, orientation,
                                          target_name=target_name)
            rospy.loginfo("[task] Re-positioned for pre-grasp retry: (%.3f, %.3f)", *pos)
        if not pre.success:
            return GetObjectResponse(
                success=False, gripper_width=0.0, zone=zone, scene_summary=summary,
                phase="pregrasp", attach_camera_frame=True,
                description=(f"Pre-grasp positioning failed ({pre.message or 'unknown'}). "
                             f"The arm never reached the standoff, so nothing was grasped "
                             f"and the object was not disturbed."))

        # 7. GRASP — the arm occludes the camera now, so the contact check gets
        #    the object's position from the odometry memory.
        mem = self.target_memory(action=TargetMemoryRequest.REPROJECT)
        # z = the grasp height on the object NOW: the planned height plus however
        # far the cached point has moved vertically since it was cached (base
        # tilt). NaN when unknown -> the arm then keeps the planned height.
        obj_z = float("nan")
        if mem.ok and self._ref_point is not None:
            obj_z = round(plan_msg.grasp_point.z + (mem.position.z - self._ref_point[2]), 3)
        obj_now = (Point(round(mem.position.x, 3), round(mem.position.y, 3), obj_z)
                   if mem.ok else None)
        if obj_now is not None:
            rospy.loginfo("[task] Contact check reference (odometry): (%.3f, %.3f)  "
                          "grasp height %.3f (planned %.3f)",
                          obj_now.x, obj_now.y, obj_z, plan_msg.grasp_point.z)
        else:
            rospy.logwarn("[task] No odometry reprojection — contact check unverified")
        try:
            g = self.execute_grasp(plan=plan_msg, object_now=obj_now or Point(),
                                   object_now_valid=obj_now is not None)
        finally:
            if reaim_widened:   # back to the arm's own default for every other pick
                try:
                    rospy.delete_param(f"/{self.ns}/arm/reaim_max_mm")
                except KeyError:
                    pass
        gw = g.gripper_width

        # 8. HOLD — stop here; placing is now inspect_object's job (it always
        #    places, win or lose). get_object's success means "picked up and
        #    held", not "picked, placed back down".
        rospy.sleep(0.3)
        bev = self.get_scene(target_name=target_name, include_bev=True).bev
        # g.success is the arm's own verdict and already folds in the finger evidence
        # (gg.grip_verdict). It used to be OR'd with "fingers between 15 and 34 mm",
        # which re-promoted exactly the outcomes the arm had just rejected: fingers
        # resting at 23.0 / 17.6 mm after the bottle was thrown clear are "in the
        # window" and were reported as a held object.
        grasped = g.success
        if grasped and gw > GRIP_HELD_MIN_WIDTH:
            # The odometry memory is KEPT while the object is held: inspect_object reprojects it
            # to put the object back where it was, however far the base has crept since.
            self._held = {"target_name": target_name, "pos": pos,
                         "height": float(dims[0]) if dims else None,
                         "grasp_above_table": float(plan_msg.grasp_point.z - tz)}
            desc = (f"Grasped and holding '{target_name}' (width {gw * 1000:.0f}mm). "
                    f"Call inspect_object to look at it — it will be placed back down "
                    f"at the end of that call.")
        elif g.success:
            desc = f"Gripper closed but nearly empty ({gw * 1000:.0f}mm) — object may have slipped."
            # Tuck the arm: the grasp leaves it EXTENDED at the retreat hold, and
            # without a place step nothing else retracts it — it would block the
            # camera for every following attempt.
            self.set_arm_pose(pose_name="camera_clear")
        elif not g.verified:
            # The contact check refused to close: the object was left UNDISTURBED,
            # so retrying from a corrected base position is worth doing.
            desc = (f"Reached the object but the jaws were not centred on it "
                    f"({g.message or 'off-centre'}) — did not close, to avoid pushing it "
                    f"out of reach.")
        else:
            desc = f"Grasp failed: {g.message or 'unknown'}."
        rospy.loginfo("[task] get_object: ok=%s gw=%.1fmm", g.success, gw * 1000)
        return GetObjectResponse(
            success=grasped and gw > GRIP_HELD_MIN_WIDTH, gripper_width=round(gw, 4),
            zone=zone, scene_summary=summary, phase="descent", description=desc,
            attach_camera_frame=True, bev=bev)

    def _approach(self, target_name, standoff, table_near_x, mode):
        try:
            return self.approach(target_name=target_name, standoff=standoff,
                                 table_near_x=table_near_x, mode=mode)
        except rospy.ServiceException as e:
            return ApproachTargetResponse(success=False, message=f"Navigation failed: {e}")

    # ── Grasp planning ───────────────────────────────────────────────────────

    def _hold(self, on):
        """Keep the base's heading (navigation ~hold_heading) from now on, or release it. The base
        yaws on its own while the arm is out and turned (see base_controller.HOLD_*), which moved
        the target 8 mm sideways during the last seconds of the approach. Never raises: a base that
        cannot hold is the situation before this existed."""
        try:
            r = self.hold_heading(data=bool(on))
            if on and not r.success:
                rospy.logwarn("[task] Heading hold refused: %s", r.message)
        except rospy.ServiceException as e:
            rospy.logwarn("[task] Heading hold %s failed: %s", "on" if on else "release", e)

    def _box_key(self, target_name):
        """Catalogue key of a catalogued, NON-round target (a cuboid like the tea box), else None."""
        key = self.catalog.key_for_name(target_name) if target_name else None
        if (key is None or self.catalog.is_round(key)
                or self.catalog.objects[key].get("role") == "surface"
                or self.catalog.size(key) is None):
            return None
        return key

    def _axis_dev_deg(self, target_name):
        """How far the planner may turn the closing axis for this target (see GRASP_BOX_*)."""
        override = rospy.get_param("~grasp_max_axis_dev_deg", None)
        if override is not None:
            return float(override)
        return GRASP_BOX_MAX_AXIS_DEV_DEG if self._box_key(target_name) else GRASP_MAX_AXIS_DEV_DEG

    def _trust_box_width(self, plan_msg, target_name, planned_width):
        """For a catalogued box, replace a measured width that disagrees with the catalogue by the
        catalogue's face width: the opening and the close target (and so the contact gate's room)
        then follow the real box instead of one camera view of it. Round objects are untouched
        (a bottle's neck is narrower than its catalogue size, and that is what is wanted)."""
        key = self._box_key(target_name)
        if key is None:
            return plan_msg
        sx, sy, _ = self.catalog.size(key)
        face = min(sx, sy)
        if abs(planned_width - face) <= BOX_WIDTH_TRUST_M:
            return plan_msg
        opening, close = gg.box_grip_from_width(face)
        rospy.loginfo("[task] Planned width %.0f mm disagrees with the %s's %.0f mm faces — "
                      "opening %.1f mm and closing to %.1f mm from the catalogue width",
                      planned_width * 1000, key, face * 1000, opening * 1000, close * 1000)
        plan_msg.gripper_open, plan_msg.gripper_close, plan_msg.full_close = opening, close, False
        return plan_msg

    def _choose_plan(self, st, pos, dims, grasp_axis, tz, orientation=None, target_name=None):
        """(plan, table_z): the grasp to execute — the depth-image planner's measurement of this
        specific object's real shape when it can support one, else the bbox-
        centre baseline. Never worse than the baseline — a planner that is
        unavailable, declines, or errors just falls back silently. Called once
        per attempt and memoized by the caller: a second live measurement is
        not guaranteed to reproduce the point the arm already positioned
        itself relative to, and re-querying it for the descent would be both
        physically wrong and liable to trip grasp_executor's drift guard on
        measurement noise rather than a real problem.

        orientation: the refined per-detection yaw (None when never measured),
        used only by the baseline fallback — the depth planner measures its
        own orientation-aware contact axis directly from real 3D geometry and
        has no use for this 2D-image-derived estimate."""
        r = None
        if rospy.get_param("~force_baseline_plan", False):
            rospy.loginfo("[task] ~force_baseline_plan set — skipping the depth planner")
        elif st.in_frame:
            try:
                r = self.plan_grasp(bbox=st.frame_bbox, stamp=st.frame_stamp,
                                    position=Point(pos[0], pos[1], 0.0),
                                    size=size_from_dims_hwl(dims) if dims else Vector3(),
                                    table_z=tz,
                                    max_axis_deviation=math.radians(
                                        self._axis_dev_deg(target_name)))
            except rospy.ServiceException as e:
                rospy.logwarn("[task] Grasp planner call failed: %s — using baseline", e)
        else:
            rospy.loginfo("[task] Target not in the latest frame — using baseline")

        # The table height the planner MEASURED beats the assumed constant — for the
        # baseline fallback and for every box and height the caller derives from it.
        if r is not None and math.isfinite(r.table_z) and r.table_z > 0.05:
            if abs(r.table_z - tz) > 0.005:
                rospy.loginfo("[task] Table surface z=%.3f measured (assumed %.3f)", r.table_z, tz)
            tz = r.table_z

        if r is not None and r.planned:
            rospy.loginfo("[task] Grasp plan from the DEPTH PLANNER: %s "
                          "(width %.0fmm, turn %.0f°)", r.reason, r.width * 1000,
                          math.degrees(r.seat_turn))
            plan_msg, source = self._trust_box_width(r.plan, target_name, r.width), "planner"
            # Re-prime the odometry cache with THIS measurement: it's real
            # geometry from the actual band picked
            # The contact-check gate compares its commanded point against
            # whatever the cache holds — if the cache still held the older,
            # cruder estimate, a genuinely more accurate plan would disagree
            # with it by that same bias and look like the object had moved.
            try:
                self._remember(TargetMemoryRequest.SET, plan_msg.grasp_point)
            except rospy.ServiceException as e:
                rospy.logwarn("[task] Could not re-prime odometry cache: %s", e)
        else:
            if r is not None:
                rospy.loginfo("[task] Depth planner declined (%s) — using baseline", r.reason)
            plan_msg = plan_to_msg(gg.baseline_plan(pos, dims, grasp_axis, tz,
                                                    orientation=orientation))
            source = "baseline"

        model = rospy.get_param("~diag_object_model",
                                rospy.get_param(f"/{self.ns}/arm/diag_object_model", ""))
        if model:
            if self._ground_truth is None:
                self._ground_truth = GroundTruth(self.ns)
            gt = self._ground_truth.object_in_base(model)
            if gt is not None:
                g = plan_msg.grasp_point
                rospy.loginfo("[task] Grasp point (%s) vs GT '%s' (%.3f, %.3f): %.1fmm",
                              source, model, gt[0], gt[1],
                              math.hypot(g.x - gt[0], g.y - gt[1]) * 1000)
        return plan_msg, tz

    # ── Inverse reachability ─────────────────────────────────────────────────

    def _ik(self, pos, quat):
        """MoveIt IK for a gripper pose in base_link (no collisions), as
        (joint_names, joint_positions), or None."""
        ps = PoseStamped()
        ps.header.frame_id = f"{self.ns}/base_link"
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = pos
        (ps.pose.orientation.x, ps.pose.orientation.y,
         ps.pose.orientation.z, ps.pose.orientation.w) = quat
        req = GetPositionIKRequest()
        req.ik_request.group_name = "interbotix_arm"
        req.ik_request.ik_link_name = f"{self.ns}/ee_gripper_link"
        req.ik_request.pose_stamped = ps
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout = rospy.Duration(0.3)
        # Seed with the current joints — without it MoveIt logs a stream of
        # "Found empty JointState message" errors.
        if self._joint_state is not None:
            req.ik_request.robot_state.joint_state = self._joint_state
        try:
            r = self._ik_srv(req)
        except rospy.ServiceException:
            return None
        if r.error_code.val != 1:
            return None
        return r.solution.joint_state.name, r.solution.joint_state.position

    def _reach_optimal_standoff(self, dims_hwl, table_z):
        if self._ik_srv is None:
            return self.grasp_dist
        best = reach.reach_optimal_standoff(self._ik, dims_hwl, table_z)
        if best is None:
            rospy.logwarn("[task] Reachability: no standoff scored — fallback %.2fm",
                          self.grasp_dist)
            return self.grasp_dist
        rospy.loginfo("[task] Reachability standoff: %.2fm (approach margin %.3f, "
                      "plateau score %.3f)", *best)
        return best[0]


def _only(fields, response_class):
    """The subset of `fields` that response_class has."""
    slots = set(response_class.__slots__)
    return {k: v for k, v in fields.items() if k in slots}


if __name__ == "__main__":
    rospy.init_node("world_model")
    WorldModelNode()
    TaskOrchestrator()
    rospy.spin()
