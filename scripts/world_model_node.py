#!/usr/bin/env python3
"""world_model_node — the world model AND the pick-and-place sequence.

Two classes in one process (see the merge note below), ported near-verbatim
from rework-attempt-1's scripts/world_model_node.py + scripts/
task_orchestrator_node.py:

  WorldModelNode (~world_model/...):
    ~get_scene      (GetScene)      objects, target, what blocks it, zone,
                                    table, collision boxes, bird's-eye view
    ~get_target     (GetTarget)     every source for where the target is
    ~target_memory  (TargetMemory)  hold the target in odometry / forget it
    ~bev_image      (Image, 1 Hz)   the bird's-eye view, live (when subscribed)

  TaskOrchestrator (/task/...):
    /task/get_object   (GetObject)   observe -> navigate -> pregrasp -> grasp -> place
    /task/navigate_to  (NavigateTo)  observe -> navigate, stop in front of the object

WHY MERGED, NOT TWO PROCESSES (the plan's RAM cap): rework-attempt-1 ran these
as separate nodes; merging them back recovers the typed-service interface
without paying for a third extra process (see the Step 3/4 commit messages
for the measured RAM cost per process on this machine). Confirmed mechanically
clean before merging (Explore pass, step 3): disjoint import graphs, no shared
mutable globals, different-purpose locks that don't deadlock.

The ONE thing the merge requires: TaskOrchestrator's services were registered
`~`-relative (rospy.Service("~get_object", ...)), which resolves under
whichever single node name this process ends up with — under "world_model"
that would silently become /locobot/world_model/get_object instead of
/locobot/task/get_object. Fixed here by registering them with explicit
absolute names instead, so external callers (vlm_agent_node.py,
tools/e2e_sim.sh-equivalents) can still find them at the path the original
two-process design used.

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

# See arm_node.py's identical comment: roslaunch execs this file through a
# devel/lib/ wrapper, so __file__'s own directory must be added explicitly
# for the flat sibling imports below to resolve.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import footprint
import grasp_geometry as gg
import grasp_planner as gp
import reach
import rotations
import world_bev
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
                              ApproachTargetResponse, ExecuteGrasp, ExecutePlace,
                              ExecutePregrasp, GetObject, GetObjectResponse,
                              GetScene, GetSceneResponse, GetTarget,
                              GetTargetResponse, NavigateTo, NavigateToResponse,
                              PlanGrasp, PlanGraspResponse, SetArmPose, TargetMemory,
                              TargetMemoryRequest, TargetMemoryResponse)

# The grasp point trusts a target lock at most this old (positioning uses 2 s):
# the robot has been standing still since it aligned.
GRASP_MAX_LOCK_AGE = 3.0
# A grasp counts as holding the object above this measured finger position.
# A base tilt/creep smaller than this is left to the arm's own re-aim; a larger one
# is worth re-taking the pre-grasp for (see TaskOrchestrator._settled_shift).
SHIFT_FOLLOW_XY_MM = 10.0
GRIP_HELD_MIN_WIDTH = 0.015


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
        """4x4 transform from the depth frame to base_link AT THE FRAME'S STAMP
        (a Time(0) lookup would use wherever the camera is now)."""
        t = self.tf_buffer.lookup_transform(f"{self.ns}/base_link", header.frame_id,
                                            header.stamp, rospy.Duration(0.3)).transform
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
        # The table the caller assumed is a constant; the one the camera sees is
        # what the jaws must clear, so measure it (footprint.table_height) and plan
        # in heights above THAT. Falls back to the caller's when it cannot be seen.
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
        self.set_arm_pose = proxy("arm/set_arm_pose", SetArmPose)
        self.execute_pregrasp = proxy("arm/execute_pregrasp", ExecutePregrasp)
        self.execute_grasp = proxy("arm/execute_grasp", ExecuteGrasp)
        self.execute_place = proxy("arm/execute_place", ExecutePlace)

        # MoveIt IK, only to CHOOSE the standoff (inverse reachability). Optional:
        # without it the default standoff is used.
        self._joint_state = None
        rospy.Subscriber(f"/{ns}/joint_states", JointState, self._on_joint_state, queue_size=1)
        self._ik_srv = None
        try:
            rospy.wait_for_service(f"/{ns}/compute_ik", timeout=5.0)
            self._ik_srv = rospy.ServiceProxy(f"/{ns}/compute_ik", GetPositionIK)
            rospy.loginfo("[task] compute_ik available — reachability standoff enabled")
        except rospy.ROSException:
            rospy.logwarn("[task] compute_ik unavailable — using fixed grasp_dist")

        self._ground_truth = None     # Gazebo, only when a diag model is named
        self._ref_point = None        # (x, y, z) in base_link of the point last cached in the odometry memory

        # ABSOLUTE names, not "~get_object"/"~navigate_to" — see this file's
        # module docstring for why the merge requires this.
        rospy.Service(f"/{ns}/task/get_object", GetObject, self._handle_get_object)
        rospy.Service(f"/{ns}/task/navigate_to", NavigateTo, self._handle_navigate_to)
        rospy.loginfo("[task] Ready.")

    def _on_joint_state(self, msg):
        self._joint_state = msg

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
            try:
                ctx = self._navigate(req.target_name)
                return self._pick_and_place(req.target_name, ctx)
            except _Failed as f:
                return GetObjectResponse(success=False, **_only(f.fields, GetObjectResponse))

    # ── Steps ────────────────────────────────────────────────────────────────

    def _navigate(self, target_name):
        """Observe, choose the standoff, and put the base there. Returns the
        context the grasp needs; raises _Failed with the reply otherwise."""
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
        standoff = self._reach_optimal_standoff(dims, s.table.surface_z)

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

    def _pick_and_place(self, target_name, ctx):
        s, standoff = ctx["scene"], ctx["standoff"]
        zone, summary = ctx["zone"], ctx["summary"]

        # 5. Where exactly to grasp. Tuck the arm FIRST so the camera has an
        #    unobstructed view, then read the target with the robot standing
        #    still: a lock made 12 s earlier and reprojected through open-loop
        #    drive odometry once put the grasp 5-6 cm past the real bottle.
        self.set_arm_pose(pose_name="camera_clear")
        rospy.sleep(1.2)   # let perception deliver frames of the unobstructed scene
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
        plan_msg, tz = self._choose_plan(st, pos, dims, grasp_axis, tz)
        pre, failures, followed = None, 0, False
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
                d = None if followed else self._settled_shift()
                if d is None:
                    break
                followed = True
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
            failures += 1
            rospy.logwarn("[task] Pre-grasp NOT verified (attempt %d): %s", failures, pre.message)
            if failures == 2:
                break
            rospy.sleep(1.2)
            r = self._approach(target_name, standoff, ctx["table_near_x"],
                               ApproachTargetRequest.CORRECT)
            if not r.success:
                break
            pos = [round(r.target.x, 3), round(r.target.y, 3)]
            self._remember(TargetMemoryRequest.MEASURE)   # re-prime here
            # The base moved, so what was measured before no longer applies.
            tilt, followed = (0.0, 0.0, 0.0), False
            # Refresh st too: a corrected base pose means a new camera pose, so
            # the depth plan's buffered frame/TF lookup must describe THIS pose,
            # not the pre-correction one the original st still holds.
            st = self.get_target(target_name=target_name).state
            plan_msg, tz = self._choose_plan(st, pos, dims, grasp_axis, tz)
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
        g = self.execute_grasp(plan=plan_msg, object_now=obj_now or Point(),
                               object_now_valid=obj_now is not None)
        gw = g.gripper_width

        # 8. PLACE — or report why not.
        rospy.sleep(0.3)
        bev = self.get_scene(target_name=target_name, include_bev=True).bev
        grasped = g.success or (GRIP_HELD_MIN_WIDTH < gw < 0.034)
        if grasped and gw > GRIP_HELD_MIN_WIDTH:
            self.target_memory(action=TargetMemoryRequest.FORGET)
            # Place BACK at the grasp point: the arm is held straight above it, so
            # it just lowers the object down the same vertical line — reachable
            # (it was just grasped there) and on the table.
            rospy.loginfo("[task] Place point (grasp point): (%.3f, %.3f)", *pos)
            placed = self.execute_place(place_point=Point(pos[0], pos[1], 0.0),
                                        at_grasp_height=True)
            if placed.success:
                desc = (f"Pick-and-place complete — grasped (width {gw * 1000:.0f}mm), "
                        f"lifted, and placed back down.")
            else:
                desc = (f"Grasp succeeded (width {gw * 1000:.0f}mm) but the place step "
                        f"failed: {placed.message or 'unknown'}.")
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

    def _choose_plan(self, st, pos, dims, grasp_axis, tz):
        """(plan, table_z): the grasp to execute — the depth-image planner's measurement of this
        specific object's real shape when it can support one, else the bbox-
        centre baseline. Never worse than the baseline — a planner that is
        unavailable, declines, or errors just falls back silently. Called once
        per attempt and memoized by the caller: a second live measurement is
        not guaranteed to reproduce the point the arm already positioned
        itself relative to, and re-querying it for the descent would be both
        physically wrong and liable to trip grasp_executor's drift guard on
        measurement noise rather than a real problem."""
        r = None
        if st.in_frame:
            try:
                r = self.plan_grasp(bbox=st.frame_bbox, stamp=st.frame_stamp,
                                    position=Point(pos[0], pos[1], 0.0),
                                    size=size_from_dims_hwl(dims) if dims else Vector3(),
                                    table_z=tz, max_axis_deviation=0.0)
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
            plan_msg, source = r.plan, "planner"
            # Re-prime the odometry cache with THIS measurement: it's real
            # geometry from the actual band picked, not YOLO3D's bbox-based
            # position (which has a documented range-dependent radial bias).
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
            plan_msg = plan_to_msg(gg.baseline_plan(pos, dims, grasp_axis, tz))
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
