#!/usr/bin/env python3
"""VLM agent-in-the-loop for LoCoBot pick tasks."""
import base64, json, math, os, statistics, sys, threading, time
from collections import deque
from typing import Optional

import cv2
import numpy as np
from ollama import chat as ollama_chat
import rospy
import tf2_ros
import tf2_geometry_msgs
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import CameraInfo, Image, LaserScan, JointState
from std_msgs.msg import String
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from locobot_scene_model import SceneModel
from target_odom_cache import TargetOdomCache
import grasp_geometry
from locobot_geometry import (
    ARM_GRASP_DIST, GRIPPER_W_M, TABLE_DEPTH_M, TABLE_ROBOT_MARGIN,
    ZONE_GRASP_READY, ZONE_BLOCKED,
    parse_object, find_blocking, classify_zone,
    compute_grasp, build_summary, render_birdseye, estimate_grasp_orientation,
)

DIM_CALIB_MIN_DIST   = 0.47   # measured good-vantage floor
DIM_CALIB_MIN_SAMPLES = 3

# WidowX-250s arm joint limits (rad), verified from the URDF — used to score how
# comfortable an IK solution is (distance to the nearest joint limit).
ARM_LIMITS = {
    "waist":        (-3.142, 3.142),
    "shoulder":     (-1.885, 1.990),
    "elbow":        (-2.147, 1.606),
    "forearm_roll": (-3.142, 3.142),
    "wrist_angle":  (-1.745, 2.147),
    "wrist_rotate": (-3.142, 3.142),
}
# Inverse-reachability standoff sweep — a genuine PER-TARGET distance, 
STANDOFF_MIN, STANDOFF_MAX, STANDOFF_STEP = 0.44, 0.47, 0.01
# The standoff offset itself is NOT duplicated here: _approach_margin gets it
# from grasp_geometry.pregrasp_pose's own default, so the distance scored is by
# construction the distance the motion node flies.
APPROACH_SAMPLES = 3     # points sampled from standoff to grasp point (inclusive)
PLATEAU_WEIGHT  = 0.25   # weight on the mean margin of the ±1 neighbour standoffs

SYSTEM_PROMPT = """You are an autonomous LoCoBot manipulation agent.
You have one tool:
  get_object(target_name) — observes the scene, navigates if needed, grasps the target,
                            and returns a feedback image so you can verify success.

Workflow:
1. Call get_object to attempt the pick.
2. Check the feedback: gripper_width_m > 0.020 means the object is secured.
3. If it failed, call get_object again (up to 3 times total).
End the episode by stating the outcome."""

class VLMAgentNode:

    def __init__(self):
        ns = rospy.get_param("~robot_name", "locobot")
        self.robot_name    = ns
        self.target_object = rospy.get_param("~target_object", "bottle")
        self.vlm_model     = rospy.get_param("~vlm_model",     "qwen3-vl:2b-instruct")
        self.max_steps     = int(rospy.get_param("~max_steps",  20))
        self.grasp_dist    = float(rospy.get_param("~arm_grasp_dist_m", ARM_GRASP_DIST))
        # Per-target standoff chosen by inverse reachability (see _reach_optimal_standoff).
        self._active_grasp_dist = self.grasp_dist
        self._grasp_interval = float(rospy.get_param("~grasp_min_interval_s", 5.0))
        self._last_grasp_ts  = 0.0

        self.bridge            = CvBridge()
        self.latest_frame      = None
        self.latest_depth      = None   # float32 metres, aligned to color frame
        self.latest_detections = []
        self._det_cache        = []
        self._det_cache_ts     = 0.0
        self._det_cache_ttl    = float(rospy.get_param("~det_cache_ttl_s", 30.0))
        self._locked_target    = None   # best detection of target seen so far
        self._locked_target_ts = 0.0    # monotonic time of the last accepted lock
        self._target_pos_buf   = deque(maxlen=8)   # recent (t, x, y, z) base-frame
        # Per-class dimension CALIBRATION, accumulated ONLY from good-vantage
        # frames (dist >= DIM_CALIB_MIN_DIST), for EVERY detected class — not
        # just the target — so neighbour objects get accurately-sized collision
        # boxes too, not just the grasp target.  See _calibrated_dims().
        self._dim_calib        = {}   # class_name -> deque[(dist, H, W, L)]
        self.front_dist        = float("inf")
        self._pending_images   = []
        self._motion_result    = None
        self._motion_event     = threading.Event()
        self.K                 = None   # 3×3 camera intrinsics — filled once by camera_info
        self.scene_model       = SceneModel()
        self.tf_buffer         = tf2_ros.Buffer()
        self.tf_listener       = tf2_ros.TransformListener(self.tf_buffer)
        self._locked_target_map = None  # target position stored in /map frame
        self._target_cache     = TargetOdomCache(self.tf_buffer, ns)

        self.motion_pub = rospy.Publisher(f"/{ns}/locobot_prog/motion_cmd", String, queue_size=1)
        rospy.Subscriber(f"/{ns}/locobot_prog/motion_result", String, self._on_motion_result, queue_size=1)

        # MoveIt IK service (move_group under /{ns}) — for inverse-reachability standoff selection. 
        self._joint_state = None   # current arm state, used as the IK seed
        rospy.Subscriber(f"/{ns}/joint_states", JointState,
                         self._on_joint_state, queue_size=1)
        self._ik_srv = None
        try:
            rospy.wait_for_service(f"/{ns}/compute_ik", timeout=5.0)
            self._ik_srv = rospy.ServiceProxy(f"/{ns}/compute_ik", GetPositionIK)
            rospy.loginfo("[agent] compute_ik available — reachability standoff enabled")
        except rospy.ROSException:
            rospy.logwarn("[agent] compute_ik unavailable — using fixed grasp_dist")
        rospy.Subscriber(f"/{ns}/camera/color/image_raw",                  Image,      self._on_image,       queue_size=1)
        rospy.Subscriber(f"/{ns}/camera/aligned_depth_to_color/image_raw", Image,      self._on_depth,       queue_size=1)
        rospy.Subscriber(f"/{ns}/locobot_prog/detections_3d",              String,     self._on_detections,  queue_size=1)
        rospy.Subscriber(f"/{ns}/scan",                                     LaserScan,  self._on_scan,        queue_size=1)
        rospy.Subscriber(f"/{ns}/camera/color/camera_info",                 CameraInfo, self._on_camera_info, queue_size=1)

        rospy.loginfo("[agent] target='%s' model=%s max_steps=%d",
                      self.target_object, self.vlm_model, self.max_steps)

    # ROS callbacks
    def _on_camera_info(self, msg):
        if self.K is not None:
            return
        self.K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        rospy.loginfo("[agent] Camera intrinsics: fx=%.1f fy=%.1f", self.K[0, 0], self.K[1, 1])

    def _on_image(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError:
            pass

    def _on_depth(self, msg):
        try:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            # RealSense aligned depth is uint16 millimetres → convert to float32 metres
            if raw.dtype == np.uint16:
                self.latest_depth = raw.astype(np.float32) / 1000.0
            else:
                self.latest_depth = raw.astype(np.float32)
        except CvBridgeError:
            pass

    def _on_detections(self, msg):
        try:
            dets = json.loads(msg.data)
            if not isinstance(dets, list):
                return
            self.latest_detections = dets
            if dets:
                self._det_cache    = dets
                self._det_cache_ts = time.monotonic()
            for d in dets:
                cls = d.get("class_name", "").lower()
                # Dimension calibration — EVERY class, not just the target, so
                # neighbour objects (obstacle boxes) benefit too.  Gated to
                # dist >= DIM_CALIB_MIN_DIST: see the measured table at the top
                # of the file for why closer frames are actively worse, not just
                # noisier.  Accumulates for the whole node lifetime (dimensions
                # don't change between grasp attempts), so once calibrated near
                # the start of an approach, later close-range frames never
                # overwrite it.
                if d.get("position_base") and d.get("dims_hwl") and cls:
                    _px, _py, _ = d["position_base"]
                    _dist = math.hypot(_px, _py)
                    if _dist >= DIM_CALIB_MIN_DIST:
                        h, w, l = (float(v) for v in d["dims_hwl"])
                        sample = (_dist, h, w, l)
                        self._dim_calib.setdefault(cls, deque(maxlen=20)).append(sample)
                        # ALSO index under the target's own logical name when
                        # this detection matched via an ALIAS (e.g. target_object
                        # "bottle" measured as YOLO class "vase") — otherwise
                        # _calibrated_dims(target_name) can never find it, since
                        # it looks up by the human-chosen name, not the raw class.
                        if d.get("is_target") and cls != self.target_object.lower():
                            self._dim_calib.setdefault(
                                self.target_object.lower(),
                                deque(maxlen=20)).append(sample)
                # Update live table geometry whenever the dining table is detected
                if cls == "dining table" and self.K is not None:
                    self.scene_model.update_table(d, self.K)
                    rospy.loginfo_throttle(5.0,
                        "[agent] Table updated: surface_z=%.3f  near_x=%.3f  w=%.2fm  d=%.2fm",
                        self.scene_model.table_z_for_grasp(),
                        self.scene_model.table_near_x if self.scene_model.table_near_x is not None else -1.0,
                        self.scene_model.table_width_m or 0.0,
                        self.scene_model.table_depth_m or 0.0)
                is_tgt = (d.get("is_target") or cls == self.target_object.lower())
                if is_tgt and d.get("position_base") and d.get("dims_hwl"):
                    # Sanity gate: reject implausible detections before locking.
                    px, py, pz = d["position_base"]
                    dh, dw = float(d["dims_hwl"][0]), float(d["dims_hwl"][1])
                    if not (0.0 <= pz <= 0.8 and math.hypot(px, py) < 2.0
                            and dh < 0.45 and dw < 0.35):
                        rospy.logwarn_throttle(5.0,
                            "[agent] Rejecting implausible target det: pos=%s dims=%s",
                            d["position_base"], d["dims_hwl"])
                        continue
                    locked = dict(d)
                    locked["is_target"] = True
                    self._locked_target    = locked
                    self._locked_target_ts = time.monotonic()   # freshness marker
                    # Buffer recent base-frame positions so the grasp point can be
                    # a MEDIAN over several frames — a single YOLO frame's bbox
                    # jitter (~cm) is what occasionally clips the tight-clearance
                    # bottle grasp; the median cancels it.
                    self._target_pos_buf.append((time.monotonic(), px, py, pz))
                    self._target_cache.add_detection([px, py, pz])
                    self._lock_target_in_map(locked)
                    rospy.loginfo_throttle(5.0, "[agent] Target locked: %s @ %s",
                                           locked.get("class_name"), locked.get("position_base"))
        except (json.JSONDecodeError, TypeError):
            pass

    def _on_scan(self, msg):
        n = len(msg.ranges)
        k = max(1, n // 24)   # ±15° cone
        cone  = list(msg.ranges[:k]) + list(msg.ranges[n - k:])
        valid = [r for r in cone if msg.range_min < r < msg.range_max]
        self.front_dist = min(valid) if valid else float("inf")

    def _on_joint_state(self, msg):
        self._joint_state = msg

    def _calibrated_dims(self, class_name: str, min_samples: int = DIM_CALIB_MIN_SAMPLES):
        """Median [H, W, L] for `class_name` from good-vantage frames only
        (dist >= DIM_CALIB_MIN_DIST), or None if too few have been seen yet.

        This is the AUTHORITATIVE dimension source once available: measured
        (see the table at the top of this file) that both the plain bbox height
        and the centre-derived "safe" fallback get WORSE the closer the robot
        is, with the centre one biased badly (33-62% over true in the 0.40-0.55m
        band).  A locked-in calibration from farther away avoids re-deriving a
        worse number right when it matters most, at the grasp standoff.
        """
        buf = self._dim_calib.get(class_name.lower())
        if not buf or len(buf) < min_samples:
            return None
        hs = [h for (_, h, _, _) in buf]
        ws = [w for (_, _, w, _) in buf]
        ls = [l for (_, _, _, l) in buf]
        return [statistics.median(hs), statistics.median(ws), statistics.median(ls)]

    def _lock_target_in_map(self, detection: dict) -> None:
        """Store target position in /map frame so it survives robot rotation."""
        pb = detection.get("position_base")
        if pb is None:
            return
        pt = PointStamped()
        pt.header.frame_id = f"{self.robot_name}/base_link"
        pt.header.stamp    = rospy.Time(0)
        pt.point.x, pt.point.y, pt.point.z = pb[0], pb[1], pb[2]
        try:
            pt_map = self.tf_buffer.transform(pt, "map", rospy.Duration(0.5))
            self._locked_target_map = (pt_map.point.x, pt_map.point.y, pt_map.point.z)
            rospy.logdebug("[agent] Target locked in map: (%.2f, %.2f, %.2f)",
                           pt_map.point.x, pt_map.point.y, pt_map.point.z)
        except Exception as e:
            rospy.logwarn("[agent] TF lock failed: %s", e)

    def _target_in_base(self) -> Optional[tuple]:
        """Re-project locked map-frame target position into current base_link."""
        if self._locked_target_map is None:
            return None
        mx, my, mz = self._locked_target_map
        pt = PointStamped()
        pt.header.frame_id = "map"
        pt.header.stamp    = rospy.Time(0)
        pt.point.x, pt.point.y, pt.point.z = mx, my, mz
        try:
            pt_base = self.tf_buffer.transform(
                pt, f"{self.robot_name}/base_link", rospy.Duration(0.5))
            return (pt_base.point.x, pt_base.point.y)
        except Exception as e:
            rospy.logwarn("[agent] TF reproject failed: %s", e)
            return None

    def _on_motion_result(self, msg):
        try:
            self._motion_result = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self._motion_result = {"success": False, "error": "unparseable"}
        self._motion_event.set()

    # Helpers
    def _call_motion(self, cmd: dict, timeout: float = 30.0) -> dict:
        self._motion_result = None
        self._motion_event.clear()
        self.motion_pub.publish(String(data=json.dumps(cmd)))
        if self._motion_event.wait(timeout):
            return self._motion_result
        rospy.logwarn("[agent] Motion timeout: %s", cmd.get("cmd"))
        return {"success": False, "error": "timeout"}

    # Inverse-reachability standoff selection 
    def _ik(self, pos, quat):
        """Query MoveIt /compute_ik for a gripper pose in base_link (pure
        kinematics, no collisions).  Returns the solution JointState or None."""
        if self._ik_srv is None:
            return None
        ps = PoseStamped()
        ps.header.frame_id = f"{self.robot_name}/base_link"
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = pos
        (ps.pose.orientation.x, ps.pose.orientation.y,
         ps.pose.orientation.z, ps.pose.orientation.w) = quat
        req = GetPositionIKRequest()
        req.ik_request.group_name    = "interbotix_arm"
        req.ik_request.ik_link_name  = f"{self.robot_name}/ee_gripper_link"
        req.ik_request.pose_stamped  = ps
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout       = rospy.Duration(0.3)
        # Seed with the current joint state — without it MoveIt logs a stream of
        # "Found empty JointState message" errors (empty default robot_state).
        if self._joint_state is not None:
            req.ik_request.robot_state.joint_state = self._joint_state
        try:
            r = self._ik_srv(req)
            return r.solution.joint_state if r.error_code.val == 1 else None
        except rospy.ServiceException:
            return None

    @staticmethod
    def _joint_margin(joint_state):
        """Min over arm joints of the normalized distance to the nearest joint
        limit (0 = at a limit, 0.5 = mid-range).  Higher = more comfortable."""
        m = 1.0
        for name, pos in zip(joint_state.name, joint_state.position):
            key = name.split("/")[-1]
            if key in ARM_LIMITS:
                lo, hi = ARM_LIMITS[key]
                m = min(m, min(pos - lo, hi - pos) / (hi - lo))
        return m

    def _approach_margin(self, dims_hwl, table_z, d):
        """Min joint-limit margin over the ENTIRE straight-line approach at standoff
        `d`: sample APPROACH_SAMPLES points from the pre-grasp standoff forward to
        the grasp point (the path the wrist actually travels), IK each, and take
        the worst.  Returns None if any sampled pose is unreachable (the whole
        approach must be feasible).
        Base-independent: the target is evaluated straight ahead at distance `d`.

        Both endpoints come from the SHARED grasp_geometry helpers, so the
        standoff scored here is the one the arm actually flies to.  This matters
        more than it looks: the margin along this line is NOT dominated by the
        grasp point but by the STANDOFF, which is the folded end of the move
        (measured over the validated window: 0.384 rad at the standoff vs 0.679
        at the grasp point).  Scoring only the grasp point would pick a base
        distance for the easy half of the path."""
        try:
            gp, gq = grasp_geometry.grasp_pose((d, 0.0), dims_hwl, table_z)
        except grasp_geometry.ObjectTooShortForSideGrasp:
            # No valid side-grip height at any standoff — score this candidate
            # unreachable so the sweep falls back, rather than optimising a
            # standoff for a grasp the motion node will refuse (it runs the same
            # check before moving).
            return None
        pre, _ = grasp_geometry.pregrasp_pose(gp, gq)
        margin = 1.0
        for k in range(APPROACH_SAMPLES):
            frac = k / (APPROACH_SAMPLES - 1) if APPROACH_SAMPLES > 1 else 0.0
            pt = [pre[i] + (gp[i] - pre[i]) * frac for i in range(3)]
            sol = self._ik(pt, gq)
            if sol is None:
                return None
            margin = min(margin, self._joint_margin(sol))
        return margin

    def _reach_optimal_standoff(self, dims_hwl, table_z):
        """Inverse-reachability standoff: sweep the safe band and pick the distance
        with the best DESCENT-PATH joint margin, biased toward a reachability
        PLATEAU (neighbouring standoffs also reachable) so a small base error still
        lands in a comfortable pose.  Falls back to self.grasp_dist when IK is
        unavailable or nothing scores."""
        if self._ik_srv is None:
            return self.grasp_dist
        # Score every candidate first (descent-path min margin, or None if blocked)…
        grid = []
        d = STANDOFF_MIN
        while d <= STANDOFF_MAX + 1e-9:
            grid.append((round(d, 2), self._approach_margin(dims_hwl, table_z, d)))
            d += STANDOFF_STEP
        # …then choose: own margin + PLATEAU_WEIGHT × mean(reachable neighbours).
        best_d, best_val, best_own = None, -1.0, 0.0
        for i, (di, si) in enumerate(grid):
            if si is None:
                continue
            neigh = [grid[j][1] for j in (i - 1, i + 1)
                     if 0 <= j < len(grid) and grid[j][1] is not None]
            val = si + PLATEAU_WEIGHT * (sum(neigh) / len(neigh) if neigh else 0.0)
            if val > best_val:
                best_val, best_d, best_own = val, di, si
        if best_d is None:
            rospy.logwarn("[agent] Reachability: no standoff scored — fallback %.2fm",
                          self.grasp_dist)
            return self.grasp_dist
        rospy.loginfo("[agent] Reachability standoff: %.2fm (descent-margin %.3f, "
                      "plateau-score %.3f)", best_d, best_own, best_val)
        return best_d

    def _geometry(self, target_name: str, extra: list = None, nav_x_limit: float = None):
        src = list(self.latest_detections) + (extra or [])
        objects = []
        for d in src:
            if not (d.get("position_base") and d.get("dims_hwl")):
                continue
            d = dict(d)
            d["is_target"] = (d.get("class_name", "").lower() == target_name.lower()
                              or d.get("is_target", False))
            objects.append(parse_object(d))
        target = next((o for o in objects if o["is_target"]), None)

        # (the arm goes over the table; MoveIt handles table avoidance for the arm)
        nav_obstacles = [o for o in objects if not o["is_target"]
                         and (nav_x_limit is None or o["pos_xy"][0] < nav_x_limit)]
        blocking = find_blocking(target, nav_obstacles, GRIPPER_W_M)
        zone     = classify_zone(target, blocking, self._active_grasp_dist, table_near_x=nav_x_limit)
        tsz      = self.scene_model.table_z_for_grasp()
        grasp    = compute_grasp(target, table_surface_z=tsz)
        summary  = build_summary(target, zone, blocking, grasp,
                                 table_surface_z=tsz if self.scene_model.table_surface_z is not None else None)
        return objects, target, blocking, zone, grasp, summary

    def _encode(self, img_bgr, width: int = 0, height: int = 0, quality: int = 75) -> Optional[str]:
        if img_bgr is None:
            return None
        if width and height:
            img_bgr = cv2.resize(img_bgr, (width, height))
        _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf).decode("utf-8")

    def _navigate_to_target(self, target: dict, table_near_x: float) -> dict:
        """Drive toward the target: pure rotation to face it, then straight forward drive.
        Separating rotation from translation. TF reprojection is preferred over live YOLO so the
        target position stays stable across robot rotations.
        """
        tx, ty = self._target_in_base() or target["pos_xy"]
        dist   = math.hypot(tx, ty)
        alpha  = math.atan2(ty, tx)
        drive_m = max(0.0, min(dist - self._active_grasp_dist,
                               table_near_x - TABLE_ROBOT_MARGIN))

        rospy.loginfo("[agent] Navigate: target=(%.2f, %.2f)  dist=%.2fm"
                      "  alpha=%.1f deg  drive=%.2fm",
                      tx, ty, dist, math.degrees(alpha), drive_m)

        if drive_m < 0.03 and abs(alpha) <= math.radians(3.0):
            rospy.loginfo("[agent] Already within grasp range — skipping drive")
            return {"success": True}

        if self.front_dist < 0.30:
            return {"success": False,
                    "description": "Obstacle ahead — cannot drive forward."}

        # ── Step 1: rotation to face the target ──────────────────────
        if abs(alpha) > math.radians(3.0):
            rot_speed = 0.30  # rad/s
            rot_dur   = round(abs(alpha) / rot_speed, 2)
            rot_dir   = 1.0 if alpha > 0 else -1.0
            rospy.loginfo("[agent] Rotate: dir=%.0f  angle=%.1f°  dur=%.1fs",
                          rot_dir, math.degrees(alpha), rot_dur)
            self._call_motion({"cmd": "navigate", "linear_x": 0.0,
                               "angular_z": rot_dir * rot_speed,
                               "duration_s": rot_dur})
            rospy.sleep(0.3)
            # Re-read position after rotation using TF
            reproj = self._target_in_base()
            if reproj:
                tx, ty  = reproj
                dist    = math.hypot(tx, ty)
                drive_m = max(0.0, min(dist - self._active_grasp_dist,
                                       table_near_x - TABLE_ROBOT_MARGIN))

        # ── Step 2: straight forward drive (no angular component) ─────────
        if drive_m > 0.03:
            drive_dur = round(drive_m / 0.08, 2)
            rospy.loginfo("[agent] Drive straight: %.2fm  dur=%.1fs", drive_m, drive_dur)
            self._call_motion({"cmd": "navigate", "linear_x": 0.08,
                               "angular_z": 0.0,
                               "duration_s": drive_dur})
        return {"success": True}


    # The one VLM-facing tool
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

        elapsed = time.monotonic() - self._last_grasp_ts
        if elapsed < self._grasp_interval:
            rospy.sleep(self._grasp_interval - elapsed)
        self._last_grasp_ts = time.monotonic()
        # Reset the per-target standoff to the default; recomputed by inverse
        # reachability below once the target and its dims are known.
        self._active_grasp_dist = self.grasp_dist

        # ── 1. Observe scene ────────────────────────────────────────────────
        if not self.latest_detections:
            age = time.monotonic() - self._det_cache_ts
            if self._det_cache and age < self._det_cache_ttl:
                rospy.loginfo("[agent] Using cached detections (%.1fs old)", age)
                self.latest_detections = list(self._det_cache)
            elif not self._locked_target:
                rospy.loginfo("[agent] No detections — waiting 2 s")
                rospy.sleep(2.0)

        locked = [self._locked_target] if self._locked_target else []
        objects, target, blocking, zone, grasp, summary = self._geometry(target_name, locked)
        rospy.loginfo("[agent] Scene: zone=%s target=%s  detections=%d  locked=%s",
                      zone, target is not None, len(self.latest_detections),
                      bool(self._locked_target))

        if target is None:
            visible = list({d.get("class_name", "?") for d in self.latest_detections
                            if d.get("class_name")})
            fb_b64 = self._encode(self.latest_frame, width=320, height=240)
            if fb_b64:
                self._pending_images = [fb_b64]
            return {"success": False, "gripper_width_m": 0.0,
                    "zone": zone, "scene_summary": summary,
                    "visible_objects": visible,
                    "description": (
                        f"'{target_name}' not detected. "
                        f"Visible objects: {visible if visible else 'none'}. "
                        "Check the image — call get_object with the correct name if needed.")}

        # ── Inverse reachability (decision level): choose the per-target standoff
        _raw_reach = self._locked_target or next(
            (d for d in self.latest_detections
             if d.get("class_name", "").lower() == target_name.lower()
             or d.get("is_target")), None)
        _dims_reach = (_raw_reach or {}).get("dims_hwl")
        self._active_grasp_dist = self._reach_optimal_standoff(
            _dims_reach, self.scene_model.table_z_for_grasp())

        raw_near_x   = (self.scene_model.table_near_edge()
                        or target["pos_xy"][0] - TABLE_DEPTH_M / 2.0)
        table_near_x       = max(raw_near_x, ARM_GRASP_DIST + 0.05)
        grasp_table_near_x = max(raw_near_x, 0.40)
        objects, target, blocking, zone, grasp, summary = self._geometry(
            target_name, locked, table_near_x)

        # ── 2. Navigate to grasp pose ────────────────────────────────────────
        if zone != ZONE_GRASP_READY:
            if zone == ZONE_BLOCKED:
                b = blocking[0]["name"] if blocking else "unknown"
                return {"success": False, "gripper_width_m": 0.0,
                        "zone": zone, "scene_summary": summary,
                        "description": f"Path blocked by {b} — cannot proceed."}

            nav = self._navigate_to_target(target, table_near_x)
            if not nav["success"]:
                return {"success": False, "gripper_width_m": 0.0,
                        "zone": zone, "scene_summary": summary,
                        "description": nav["description"]}

            rospy.sleep(0.5)
            objects, target, blocking, zone, grasp, summary = self._geometry(
                target_name, locked, table_near_x)
            if target is None:
                return {"success": False, "gripper_width_m": 0.0,
                        "zone": zone, "scene_summary": summary,
                        "description": f"'{target_name}' lost after navigation."}

        # ── Single-shot positioning (no convergence loop) ────────────────────
        COARSE_TOL = math.radians(1.5)   # face-target threshold (above bearing noise)
        REOPEN_TOL = math.radians(6.0)   # only a real veer earns the one heading trim
        DIST_TOL   = 0.03                # distance dead-band around the standoff
       
        grasp_dist = self._active_grasp_dist
        dist_min, dist_max = grasp_dist - DIST_TOL, grasp_dist + DIST_TOL
        def _measure_xy():
            """(tx, ty, err): MEDIAN of recent (<1.5s) fresh frames → fresh lock →
            TF reprojection → YOLO re-read.  err is a failure dict or None.  Same
            priority the loop used, extracted so the single-shot flow reads clean."""
            now = time.monotonic()
            recent = [(x, y) for (t, x, y, z) in self._target_pos_buf if now - t < 1.5]
            if self._locked_target and time.monotonic() - self._locked_target_ts < 2.0 and len(recent) >= 3:
                import statistics
                return (statistics.median(x for x, _ in recent),
                        statistics.median(y for _, y in recent), None)
            if self._locked_target and time.monotonic() - self._locked_target_ts < 2.0:
                p = self._locked_target["position_base"]
                return p[0], p[1], None
            reproj = self._target_in_base()
            if reproj:
                return reproj[0], reproj[1], None
            objs, tgt, blk, z2, g2, s2 = self._geometry(target_name, locked, table_near_x)
            if tgt is None:
                return None, None, {"success": False, "gripper_width_m": 0.0, "zone": z2,
                                    "scene_summary": s2,
                                    "description": f"'{target_name}' lost during positioning."}
            if z2 == ZONE_BLOCKED:
                bn = blk[0]["name"] if blk else "unknown"
                return None, None, {"success": False, "gripper_width_m": 0.0, "zone": z2,
                                    "scene_summary": s2, "description": f"Path blocked by {bn}."}
            return tgt["pos_xy"][0], tgt["pos_xy"][1], None

        def _face(angle):
            spd = float(np.clip(1.5 * abs(angle), 0.10, 0.20))
            self._call_motion({"cmd": "navigate", "linear_x": 0.0,
                               "angular_z": (1.0 if angle > 0 else -1.0) * spd,
                               "duration_s": max(0.15, round(abs(angle) / spd, 2))})
            rospy.sleep(0.6)

        def _drive(signed_m):
            if signed_m > 0 and self.front_dist < 0.30:      # table / obstacle guard
                rospy.logwarn("[agent] Forward drive held — %.2fm ahead", self.front_dist)
                return
            m = max(-0.15, min(0.15, signed_m))              # one bounded correction
            self._call_motion({"cmd": "navigate",
                               "linear_x": 0.06 if m > 0 else -0.06, "angular_z": 0.0,
                               "duration_s": max(0.25, round(abs(m) / 0.06, 2))})
            rospy.sleep(0.6)

        # STEP 1 — GO: face the target once, then drive to the standoff once.
        tx, ty, err = _measure_xy()
        if err:
            return err
        angle = math.atan2(ty, tx)
        if abs(angle) > COARSE_TOL:
            rospy.loginfo("[agent] Position GO: face %.1f°", math.degrees(angle))
            _face(angle)
            tx, ty, err = _measure_xy()
            if err:
                return err
        dist = math.hypot(tx, ty)
        if not (dist_min <= dist <= dist_max):
            rospy.loginfo("[agent] Position GO: drive %+.2fm (%.2f→%.2fm)",
                          grasp_dist - dist, dist, grasp_dist)
            _drive(dist - grasp_dist)

        # STEP 2 — VERIFY (one pass): one heading trim only on a genuine veer, one distance correction, then re-read for the decision.
        tx, ty, err = _measure_xy()
        if err:
            return err
        dist, angle = math.hypot(tx, ty), math.atan2(ty, tx)
        if abs(angle) > REOPEN_TOL:
            rospy.loginfo("[agent] Position VERIFY: heading drifted %.1f° — one trim",
                          math.degrees(angle))
            _face(angle)
            tx, ty, err = _measure_xy()
            if err:
                return err
            dist, angle = math.hypot(tx, ty), math.atan2(ty, tx)
        if not (dist_min <= dist <= dist_max):
            _drive(dist - grasp_dist)
            tx, ty, err = _measure_xy()
            if err:
                return err
            dist, angle = math.hypot(tx, ty), math.atan2(ty, tx)

        # DECISION — proceed if the standoff is in the arm's safe window; the waist pre-rotation absorbs the heading residual. 
        # Too close → the swing clips the object; too far → the motion node hard-aborts at 0.52 m.
        if dist < 0.40:
            # Back up, then CONFIRM it worked. 
            rospy.logwarn("[agent] Positioned TOO CLOSE (%.2fm) — backing up", dist)
            for _ in range(2):
                # BUG FIX (2026-09-22): was `grasp_dist - dist`, which is POSITIVE
                # whenever dist < grasp_dist — i.e. exactly the case this branch
                # exists for — and _drive() treats positive as FORWARD. That drove
                # the robot further INTO the object it was meant to be backing away
                # from. `dist - grasp_dist` is negative here, matching the correct
                # sign convention already used one branch over (STEP 1's `_drive
                # (dist - grasp_dist)`, line ~643) and actually backs up.
                _drive(dist - grasp_dist)
                tx, ty, err = _measure_xy()
                if err:
                    return err
                dist, angle = math.hypot(tx, ty), math.atan2(ty, tx)
                rospy.loginfo("[agent] After back-up: dist=%.2fm", dist)
                if dist >= 0.40:
                    break
            if dist < 0.40:
                return {"success": False, "gripper_width_m": 0.0, "zone": zone,
                        "scene_summary": summary, "phase": "positioning",
                        "description": (f"Still too close to the target ({dist:.2f}m) "
                                        f"after backing up — the arm cannot reach it "
                                        f"without knocking it over. Move the base back "
                                        f"and retry.")}
            zone = ZONE_GRASP_READY
        elif dist <= 0.50:
            rospy.loginfo("[agent] Positioned (single-shot): dist=%.2fm heading=%.1f° "
                          "— waist compensates the rest", dist, math.degrees(angle))
            zone = ZONE_GRASP_READY
        else:
            return {"success": False, "gripper_width_m": 0.0, "zone": zone,
                    "scene_summary": summary,
                    "description": f"Target too far after positioning ({dist:.2f}m)."}

        # ── 3. Compute grasp parameters ──────────────────────────────────────
        # Tuck the arm FIRST so the camera has an unobstructed view, then wait
        # for fresh detections with the robot stationary.  A real failure mode:
        # the grasp position came from a lock made 12 s earlier (robot far away)
        # reprojected through open-loop drive odometry — 5-6 cm past the real
        # bottle, so the descending wrist flipped it.  The freshest close-range
        # detection, taken while stationary, is the most accurate source.
        self._call_motion({"cmd": "set_arm_pose", "pose_name": "camera_clear"})
        rospy.sleep(1.2)   # let YOLO deliver detections of the unobstructed scene

        raw = next((d for d in self.latest_detections
                    if d.get("class_name", "").lower() == target_name.lower()
                    or d.get("is_target")), None)

        # Position priority:
        #  1. FRESH lock (< 3 s): robot has been stationary since alignment, so
        #     its base-frame position needs no reprojection and carries no
        #     odometry drift.
        #  2. TF reprojection of the map-frame lock (robot may have moved since
        #     the last sighting).
        #  3. Raw parsed target as last resort.
        # Z always comes from the scene model (table surface): measured target z
        # is bbox-centre height, not a graspable height.
        _tz = self.scene_model.table_z_for_grasp() + 0.05
        _lock_age = time.monotonic() - self._locked_target_ts
        # MEDIAN of the recent (< 1.5 s) base-frame detections: a single frame's
        # bbox jitter (~cm) is what occasionally clips the tight bottle grasp;
        # the median over ~5 stationary frames cancels it.  Measured: lateral
        # error dropped to sub-mm when the base is aligned, so the remaining miss
        # is per-frame noise — exactly what a median removes.
        now = time.monotonic()
        recent = [(x, y) for (t, x, y, z) in self._target_pos_buf if now - t < 1.5]
        if self._locked_target and _lock_age < 3.0 and len(recent) >= 3:
            import statistics
            _lx = statistics.median(x for x, _ in recent)
            _ly = statistics.median(y for _, y in recent)
            pos_base = [round(_lx, 3), round(_ly, 3), _tz]
            rospy.loginfo("[agent] Grasp pos from MEDIAN of %d frames: (%.3f, %.3f)",
                          len(recent), _lx, _ly)
        elif self._locked_target and _lock_age < 3.0:
            _lx, _ly = self._locked_target["position_base"][:2]
            pos_base = [round(_lx, 3), round(_ly, 3), _tz]
            rospy.loginfo("[agent] Grasp pos from FRESH lock (%.1fs old): (%.3f, %.3f)",
                          _lock_age, _lx, _ly)
        else:
            _reproj = self._target_in_base()
            if _reproj:
                _px, _py = _reproj
                pos_base = [round(_px, 3), round(_py, 3), _tz]
                rospy.loginfo("[agent] Grasp pos from TF reprojection (lock %.1fs old):"
                              " (%.3f, %.3f)", _lock_age, _px, _py)
            else:
                pos_base = target.get("pos_xyz") or [target["pos_xy"][0],
                                                     target["pos_xy"][1], _tz]
                rospy.loginfo("[agent] Grasp pos from raw target (no TF): %s", pos_base)
        # Cache the target in ODOM while the robot is stationary and the camera is
        # still clear.  The contact gate later needs the object's position at a
        # moment when the arm occludes the camera, and an odom-held point survives
        # the base motion the grasp_ready swing induces (wheel odometry tracks it;
        # a fresh detection is simply not available then).
        if self._target_cache.measure() is None:
            rospy.logwarn("[agent] Odom target cache not primed (too few fresh "
                          "frames) — the contact gate will be unverified")
        # Dimensions drive the gripper OPEN/CLOSE width and grasp height.  PREFER
        # the CALIBRATED dims (median over good-vantage frames, collected while
        # the robot was still farther away — see _calibrated_dims): measured to
        # be the most accurate source, and the only one that does not degrade as
        # the robot closes in.  Falls back to live detection / locked target only
        # when no calibration exists yet (e.g. the very first attempt started
        # already close); NO fake default, so the motion node full-closes on
        # contact rather than stopping the jaws in mid-air on a bogus width.
        _calib = self._calibrated_dims(target_name)
        if _calib is not None:
            dims_hwl = _calib
        elif raw and raw.get("dims_hwl"):
            dims_hwl = raw["dims_hwl"]
        elif self._locked_target and self._locked_target.get("dims_hwl"):
            dims_hwl = self._locked_target["dims_hwl"]
        else:
            dims_hwl = None
        grasp_axis = target.get("grasp_axis", "along_x")
        angle_rad  = target.get("yaw", 0.0)
        # NOTE: there is deliberately no separate "clearance height" estimate here
        # any more.  It existed only to tell the arm how high to fly to clear the
        # object's TOP before descending onto it; the horizontal approach never
        # goes over the object, so the question no longer arises.  dims_hwl above
        # (calibrated where possible) is the single height input, and it feeds the
        # grip height and the gripper width — both of which want the ACCURATE
        # value, not a deliberately over-estimated one.
        bbox_2d     = raw.get("bbox_2d") if raw else None
        depth_m     = float(raw["depth_m"])    if raw and raw.get("depth_m")    else None
        orientation = float(raw["orientation"]) if raw and raw.get("orientation") is not None else None
        if self.latest_frame is not None and bbox_2d and len(bbox_2d) >= 4:
            angle_rad = estimate_grasp_orientation(
                self.latest_frame, bbox_2d,
                fallback_yaw=angle_rad,
                depth_img=self.latest_depth,
                depth_m=depth_m,
                dims_hwl=dims_hwl,       # physical footprint → fast symmetry check
                orientation=orientation,  # theta_ray → resolves 180° flip
            )
        # Safety clamp: symmetry guard already returns 0.0 for circular objects;
        # this caps any residual outlier on genuinely elongated objects
        angle_rad = max(-math.pi / 3, min(math.pi / 3, angle_rad))

        # ── 4. Execute grasp ─────────────────────────────────────────────────
        # (arm already at camera_clear — done at the top of section 3)
        # Obstacle scene: every OTHER detected object (not the target) becomes a
        # padded MoveIt collision box so the arm plans around it (Phase 1); the
        # target itself is added obstacle-then-released by the motion node's
        # two-phase pre-grasp (Phase 2).  Swap in CALIBRATED dims where available
        # for these too — a neighbour's obstacle box is sized from whatever frame
        # is current at grasp time, which by the same measured degradation can be
        # a close-range (worse) reading; the calibration buffer is filled for
        # every class, not just the target, for exactly this reason.
        _obstacle_dets = []
        for d in self.latest_detections:
            _cn = (d.get("class_name") or "").lower()
            _cd = self._calibrated_dims(_cn) if _cn else None
            _obstacle_dets.append(dict(d, dims_hwl=_cd) if _cd is not None else d)
        obstacles = grasp_geometry.obstacle_boxes(
            _obstacle_dets, target_name,
            self.scene_model.table_z_for_grasp())

        def _grasp_cmd(**extra):
            """Common payload for both grasp phases (they describe the same
            target and the same scene; only the verb and obj_xy differ)."""
            cmd = {"x":            pos_base[0],
                   "y":            pos_base[1],
                   "z":            pos_base[2],
                   "angle_rad":    angle_rad,
                   "dims_hwl":     dims_hwl,
                   "grasp_axis":   grasp_axis,
                   "table_z":      self.scene_model.table_z_for_grasp(),
                   "table_near_x":  grasp_table_near_x,
                   "table_width_m": self.scene_model.table_width_m or 0.80,
                   "table_depth_m": self.scene_model.table_depth_m or 0.40,
                   "obstacles":    obstacles}
            cmd.update(extra)
            return cmd

        # ── 4a. PRE-GRASP — approach the standoff, and VERIFY the arm got there.
        # Split from the descent on purpose: the motion node measures the achieved
        # standoff pose and refuses to descend from a bad one.  Real runs put the
        # arm 100-150 mm off here (0-77% Cartesian coverage) and the descent then
        # silently replanned into a diagonal sweep.  Failing here instead lets the
        # BASE be corrected — something the motion node cannot do on its own.
        pre_res, pre_data = {}, {}
        for attempt in (1, 2):
            pre_res  = self._call_motion(_grasp_cmd(cmd="execute_pregrasp"),
                                         timeout=60.0)
            pre_data = pre_res.get("data") or {}
            if pre_res.get("success"):
                rospy.loginfo("[agent] Pre-grasp verified (attempt %d): %s",
                              attempt, pre_data.get("reason", "ok"))
                break
            rospy.logwarn("[agent] Pre-grasp NOT verified (attempt %d): %s",
                          attempt, pre_data.get("reason") or pre_res.get("error"))
            if attempt == 2:
                break
            # The failed pre-grasp retracted the arm, so the camera is clear
            # again: re-measure and make ONE bounded base correction, then retry.
            rospy.sleep(1.2)
            _tx, _ty, _merr = _measure_xy()
            if _merr:
                break
            _d, _a = math.hypot(_tx, _ty), math.atan2(_ty, _tx)
            if abs(_a) > COARSE_TOL:
                _face(_a)
            if not (dist_min <= _d <= dist_max):
                _drive(_d - grasp_dist)
            _tx, _ty, _merr = _measure_xy()
            if _merr:
                break
            pos_base = [round(_tx, 3), round(_ty, 3), _tz]
            self._target_cache.measure()      # re-prime at the corrected pose
            rospy.loginfo("[agent] Re-positioned for pre-grasp retry: (%.3f, %.3f)",
                          pos_base[0], pos_base[1])

        if not pre_res.get("success"):
            reason = pre_data.get("reason") or pre_res.get("error", "unknown")
            fb_b64 = self._encode(self.latest_frame, width=320, height=240)
            if fb_b64:
                self._pending_images = [fb_b64]
            return {"success": False, "gripper_width_m": 0.0, "zone": zone,
                    "scene_summary": summary, "phase": "pregrasp",
                    "description": (f"Pre-grasp positioning failed ({reason}). The arm "
                                    f"never reached the standoff, so nothing was "
                                    f"grasped and the object was not disturbed.")}

        # ── 4b. DESCENT — descend, verify the jaws straddle the object, close.
        # The arm is now holding at the standoff and OCCLUDES the camera, so the
        # object's position for the contact gate cannot come from a fresh
        # detection.  Reproject the odom-cached measurement instead.
        obj_now = self._target_cache.reproject()
        descent = _grasp_cmd(cmd="execute_descent")
        if obj_now is not None:
            descent["obj_xy"] = [round(obj_now[0], 3), round(obj_now[1], 3)]
            rospy.loginfo("[agent] Contact gate reference (odom reproject): "
                          "(%.3f, %.3f)", obj_now[0], obj_now[1])
        else:
            rospy.logwarn("[agent] No odom reprojection — contact gate unverified")
        grasp_res = self._call_motion(descent, timeout=60.0)

        ok      = grasp_res.get("success", False)
        g_data  = grasp_res.get("data") or {}
        gw_m    = g_data.get("gripper_width_after_m", 0.0)

        # ── 5. Capture BEV + feedback image → inject into VLM context ────────
        rospy.sleep(0.3)
        bev    = render_birdseye(objects, target, blocking, zone, grasp,
                               table_bbox_m=self.scene_model.table_bbox_m)
        bev_b64 = self._encode(bev)
        fb_b64  = self._encode(self.latest_frame, width=320, height=240)
        self._pending_images = [x for x in [bev_b64, fb_b64] if x is not None]

        # ── 6. Build result ──────────────────────────────────────────────────
        # Treat as success if motion node reports ok, OR if the gripper is
        # measurably closed (object in hand) even if MoveIt planning failed.
        grasped = ok or (0.015 < gw_m < 0.034)
        if grasped and gw_m > 0.015:
            self._locked_target     = None
            self._locked_target_map = None
            # Grasp confirmed: complete the pick-and-place by putting the object
            # down ON THE TABLE.  The place point is anchored to the LIVE table
            # measurement at place time, NOT the grasp-time coordinates: the base
            # rotates/drifts several degrees during the ~30 s grasp+lift, and
            # stale base-relative coordinates once pointed past the table edge —
            # the bottle was released over the void and fell to the floor.
            # Place BACK at the grasp point: the arm is held straight above it, so
            # the motion node just lowers the object down the same vertical line —
            # guaranteed reachable (we just grasped there) and on the table.  The
            # earlier "live table anchor" put the drop at an awkward close/off-side
            # reach whose descent aborted, so the object was never released.
            place_x, place_y = pos_base[0], pos_base[1]
            rospy.loginfo("[agent] Place point (grasp point): (%.3f, %.3f)",
                          place_x, place_y)
            place_res = self._call_motion({
                "cmd": "place",
                "x":   place_x,
                "y":   place_y,
            }, timeout=60.0)
            if place_res.get("success"):
                desc = (f"Pick-and-place complete — grasped (width "
                        f"{gw_m * 1000:.0f}mm), lifted, and placed back down.")
            else:
                desc = (f"Grasp succeeded (width {gw_m * 1000:.0f}mm) but the "
                        f"place step failed: {place_res.get('error', 'unknown')}.")
        elif ok:
            desc = f"Gripper closed but nearly empty ({gw_m * 1000:.0f}mm) — object may have slipped."
            # Tuck the arm: the grasp sequence leaves it EXTENDED at the
            # retreat hold, and without a place step nothing else retracts it
            # — it then blocks the camera for every following attempt (and can
            # sweep the target while the base moves).  Measured failure mode.
            self._call_motion({"cmd": "set_arm_pose", "pose_name": "camera_clear"})
        elif not g_data.get("verified", True):
            # The contact gate refused to close: the jaws reached grasp height but
            # the object was not between them.  Reported distinctly from a plain
            # planning failure because the object was deliberately left UNDISTURBED
            # — retrying from a corrected base position is worth doing.
            desc = (f"Reached the object but the jaws were not centred on it "
                    f"({g_data.get('reason', 'off-centre')}) — did not close, to "
                    f"avoid pushing it out of reach.")
        else:
            desc = (f"Grasp failed: "
                    f"{g_data.get('reason') or grasp_res.get('error', 'unknown')}.")

        rospy.loginfo("[agent] get_object: ok=%s gw=%.1fmm phase=%s", ok, gw_m * 1000,
                      g_data.get("phase", "descent"))
        return {
            "success":        grasped and gw_m > 0.015,
            "gripper_width_m": round(gw_m, 4),
            "zone":           zone,
            "scene_summary":  summary,
            "phase":          g_data.get("phase", "descent"),
            "description":    desc,
        }

    # Agent loop
    def run(self):
        rospy.loginfo("[agent] Waiting for camera and YOLO…")
        rospy.sleep(3.0)

        deadline = rospy.Time.now() + rospy.Duration(15.0)
        while self.latest_frame is None and not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                rospy.logerr("[agent] No camera frame — aborting")
                return
            rospy.sleep(0.1)

        cam_b64  = self._encode(self.latest_frame, width=320, height=240)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Pick up the {self.target_object}.",
             "images":  [cam_b64] if cam_b64 else []},
        ]

        rospy.loginfo("[agent] Starting loop (max_steps=%d target='%s')",
                      self.max_steps, self.target_object)

        for step in range(self.max_steps):
            if rospy.is_shutdown():
                break
            rospy.loginfo("[agent] Step %d/%d — calling VLM", step + 1, self.max_steps)
            try:
                resp = ollama_chat(
                    model=self.vlm_model,
                    messages=messages,
                    tools=[self.get_object],
                    stream=False,
                    think=False,
                )
            except TypeError:
                try:
                    resp = ollama_chat(
                        model=self.vlm_model,
                        messages=messages,
                        tools=[self.get_object],
                        stream=False,
                    )
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
                name = tc.function.name
                args = tc.function.arguments
                rospy.loginfo("[agent] → %s(%s)", name, args)

                self._pending_images = []
                fn     = self.get_object if name == "get_object" else None
                result = fn(**args) if fn else {"error": f"Unknown tool: {name}"}

                rospy.loginfo("[agent] ← success=%s gw=%.1fmm zone=%s",
                              result.get("success"),
                              result.get("gripper_width_m", 0) * 1000,
                              result.get("zone", "?"))

                # inject tool result into conversation
                messages.append({"role": "tool", "name": name, "content": json.dumps(result)})

                # inject images so VLM can see BEV + feedback
                if self._pending_images:
                    messages.append({"role": "user", "content": "",
                                     "images": self._pending_images})

        rospy.loginfo("[agent] Loop finished after %d steps.", step + 1)


if __name__ == "__main__":
    rospy.init_node("vlm_agent_node")
    VLMAgentNode().run()
