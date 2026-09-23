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
    occludes it after the grasp_ready swing).  This is what replaces the
    iteration loop: the goal is fixed in odom at measure time, so nothing
    chases stale `position_base` frames during the motion.

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

    def _cache(self, pos_base):
        """Transform a base-frame point to odom and cache it there — the part
        measure() and set() share. Returns True on success."""
        pt = PointStamped()
        pt.header.frame_id = self.base_frame
        pt.header.stamp = rospy.Time(0)
        pt.point.x, pt.point.y, pt.point.z = pos_base
        try:
            po = self.tf.transform(pt, self.odom_frame, rospy.Duration(0.5))
            self.odom_pt = (po.point.x, po.point.y, po.point.z)
            self.meas_range = float(math.hypot(pos_base[0], pos_base[1]))
            return True
        except Exception as e:
            rospy.logwarn("[target_cache] odom cache failed: %s", e)
            self.odom_pt = None
            return False

    def measure(self, max_age=1.5, min_frames=3):
        """Median of fresh frames; caches the result in odom.  Robot must be
        stationary (caller's responsibility).  Returns [x, y, z] base-frame
        median, or None if not enough fresh frames."""
        fr = self.fresh_frames(max_age)
        if len(fr) < min_frames:
            return None
        med = np.median(np.array(fr), axis=0)
        self._cache(med)
        return [float(v) for v in med]

    def set(self, pos_base):
        """Cache an EXPLICIT base-frame point directly, instead of the median
        of raw detections — for when the caller already has something more
        precise than a per-frame YOLO3D position (e.g. the grasp planner's own
        real-geometry measurement of whichever band it picked; YOLO3D's own
        bbox-based position has a documented radial bias that grows with
        range). Robot must be stationary, same as measure(). Returns
        [x, y, z] on success, None if the TF lookup failed."""
        return [float(v) for v in pos_base] if self._cache(pos_base) else None

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

