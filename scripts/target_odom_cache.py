"""target_odom_cache.py — stationary-measure / odom-hold target position cache.

Design (measure -> move -> verify, no iterations):

  * `position_base` from yolo_detection_node stays the PRIMARY signal — it is
    only trusted while the robot is STATIONARY, where it is accurate.
  * measure(): median over the recent detection buffer (robust to the ~1 cm
    per-frame YOLO bbox jitter), then the median point is cached in the ODOM
    frame.  Odom is continuous wheel odometry (no rtabmap map-correction
    jumps), so the cached point stays valid while the robot moves.
  * reproject(): cached odom point -> CURRENT base frame, valid mid-motion and
    when the camera is blind (bottle fills the frame at < 0.45 m, or the arm
    occludes it after the grasp_ready swing).
  * goal(): absolute odom-frame bearing + distance for the ONE-SHOT
    closed-loop rotate/drive — this is what replaces the iteration loop: the
    goal is fixed in odom at measure time, so nothing chases stale
    `position_base` frames during the motion.

Deliberately NOT a continuous estimator: detections arriving mid-motion are
never fused (3 Hz + ~0.3 s latency makes them 3-4 deg stale during a turn —
the documented "chased TF lag for 6 rotations" failure).  For a static target
a fresh stationary measurement beats any mid-motion frame.
"""
import math
import time
from collections import deque

import numpy as np
import rospy
import tf2_geometry_msgs                      # registers PointStamped support
from geometry_msgs.msg import PointStamped


class TargetOdomCache:

    def __init__(self, tf_buffer, robot_name="locobot", maxlen=10):
        self.tf = tf_buffer
        self.base_frame = f"{robot_name}/base_link"
        self.odom_frame = f"{robot_name}/odom"
        self.buf = deque(maxlen=maxlen)       # (t_mono, x, y, z) base frame
        self.odom_pt = None                   # cached (x, y, z) in odom
        self.meas_range = None                # robot->target range AT measure time:
                                              # YOLO's radial bias grows with range
                                              # (~5 cm at 0.65 m), so consumers must
                                              # know how far out the cache was made

    # ── ingestion ────────────────────────────────────────────────────────────

    def add_detection(self, pos_base):
        """Push one detection's base-frame position (call from _on_detections)."""
        self.buf.append((time.monotonic(), float(pos_base[0]),
                         float(pos_base[1]), float(pos_base[2])))

    def fresh_frames(self, max_age=1.5):
        now = time.monotonic()
        return [(x, y, z) for (t, x, y, z) in self.buf if now - t <= max_age]

    # ── measure (stationary) ─────────────────────────────────────────────────

    def measure(self, max_age=1.5, min_frames=3):
        """Median of fresh frames; caches the result in odom.  Robot must be
        stationary (caller's responsibility).  Returns [x, y, z] base-frame
        median, or None if not enough fresh frames."""
        fr = self.fresh_frames(max_age)
        if len(fr) < min_frames:
            return None
        med = np.median(np.array(fr), axis=0)
        pt = PointStamped()
        pt.header.frame_id = self.base_frame
        pt.header.stamp = rospy.Time(0)
        pt.point.x, pt.point.y, pt.point.z = med
        try:
            po = self.tf.transform(pt, self.odom_frame, rospy.Duration(0.5))
            self.odom_pt = (po.point.x, po.point.y, po.point.z)
            self.meas_range = float(math.hypot(med[0], med[1]))
        except Exception as e:
            rospy.logwarn("[target_cache] odom cache failed: %s", e)
            self.odom_pt = None
        return [float(v) for v in med]

    # ── odom-held queries (valid mid-motion / camera-blind) ─────────────────

    def reproject(self):
        """Cached odom point -> CURRENT base frame (x, y, z), or None."""
        if self.odom_pt is None:
            return None
        pt = PointStamped()
        pt.header.frame_id = self.odom_frame
        pt.header.stamp = rospy.Time(0)
        pt.point.x, pt.point.y, pt.point.z = self.odom_pt
        try:
            pb = self.tf.transform(pt, self.base_frame, rospy.Duration(0.5))
            return (pb.point.x, pb.point.y, pb.point.z)
        except Exception as e:
            rospy.logwarn("[target_cache] reproject failed: %s", e)
            return None

    def robot_pose_odom(self):
        """(x, y, yaw) of base_link in odom, or None."""
        try:
            t = self.tf.lookup_transform(self.odom_frame, self.base_frame,
                                         rospy.Time(0), rospy.Duration(0.5))
        except Exception as e:
            rospy.logwarn("[target_cache] robot pose lookup failed: %s", e)
            return None
        q = t.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (t.transform.translation.x, t.transform.translation.y, yaw)

    def goal(self, standoff):
        """Absolute odom-frame goal for the one-shot move, from the cache.

        Returns dict(bearing_abs, yaw_err, dist, drive) or None:
          bearing_abs — odom-frame bearing robot->target (rotate to face this)
          yaw_err     — signed rotation needed from current yaw (rad)
          dist        — current robot->target distance (m)
          drive       — signed forward distance to reach `standoff` (m)
        """
        if self.odom_pt is None:
            return None
        rp = self.robot_pose_odom()
        if rp is None:
            return None
        rx, ry, ryaw = rp
        tx, ty = self.odom_pt[0], self.odom_pt[1]
        bearing = math.atan2(ty - ry, tx - rx)
        dist = math.hypot(tx - rx, ty - ry)
        yaw_err = math.atan2(math.sin(bearing - ryaw), math.cos(bearing - ryaw))
        return {"bearing_abs": bearing, "yaw_err": yaw_err,
                "dist": dist, "drive": dist - standoff}
