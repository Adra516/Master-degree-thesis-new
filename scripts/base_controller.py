"""Closed-loop turning and timed straight driving for the mobile base.

Ported near-verbatim from rework-attempt-1's src/locobot_prog/navigation/
base_controller.py, itself "moved verbatim from locobot_motion_node.py" —
VERIFIED against the current locobot_motion_node.py's own
_rotate_closed_loop/_rotate_to_target_yaw/_get_base_yaw: identical gains
(TURN_GAIN=2.0), tolerances (YAW_TOLERANCE=0.15deg), deadlines (20s), TF-miss
handling (60 misses ~1.2s), and odom-not-map rationale, before porting.

Like cobot's Pick_And_Place class, this owns its connections (the cmd_vel
publisher) and exposes one motion per method; the node decides what order to
call them in.
"""
import math

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Twist

from rotations import yaw_from_quaternion

# Stop within this of the target yaw. The controller measures ~0.15 deg
# accurate against Gazebo ground truth.
YAW_TOLERANCE = math.radians(0.15)
TURN_DEADLINE_S = 20.0
CONTROL_RATE_HZ = 50
# Odometry TF can stall under CPU load. A stall must not cut a turn short
# (a silent undershoot), so keep the last command through short gaps and bail
# loudly only after this many consecutive misses (~1.2 s at 50 Hz).
MAX_TF_MISSES = 60
TURN_GAIN = 2.0
MIN_TURN_SPEED = 0.03


class BaseController:

    def __init__(self, robot_name, tf_buffer):
        self.ns = robot_name
        self.tf_buffer = tf_buffer
        self.cmd_vel_pub = rospy.Publisher(f"/{robot_name}/cmd_vel", Twist, queue_size=1)

    def yaw(self):
        """Current base_link yaw in the ODOM frame, or None if TF is unavailable.

        Deliberately odom, not map: rtabmap corrects map->odom on every new
        keyframe, which injects a discontinuous jump into "map" mid-turn — to a
        velocity feedback loop that looks like the base suddenly rotated on its
        own. odom is raw, continuous wheel odometry, which is what a tight
        rotation loop needs; map is for global goals.
        """
        try:
            t = self.tf_buffer.lookup_transform(
                f"{self.ns}/odom", f"{self.ns}/base_link", rospy.Time(0), rospy.Duration(0.2))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        q = t.transform.rotation
        return yaw_from_quaternion(q.x, q.y, q.z, q.w)

    def rotate_by(self, angle, max_speed):
        """Turn in place by `angle` (rad), stopping on MEASURED yaw, not time.

        Timed turns miss by the accel/decel ramp and cmd_vel latency, which is
        what used to force repeated re-measure-and-turn iterations. Falls back
        to a timed turn only when odometry is unavailable at the start.
        """
        start = self.yaw()
        if start is None:
            rospy.logwarn("[nav] No TF for closed-loop rotation — falling back to timed turn")
            self.send(0.0, math.copysign(max_speed, angle))
            rospy.sleep(abs(angle) / max_speed)
            self.stop()
            return
        self.rotate_to_yaw(start + angle, max_speed=max(max_speed, 0.05))

    def rotate_to_yaw(self, target_yaw, max_speed=0.25):
        """P-controlled turn to an ABSOLUTE odom-frame yaw. Decelerates into the
        target, so the full residual can be commanded in one shot."""
        deadline = rospy.Time.now() + rospy.Duration(TURN_DEADLINE_S)
        rate = rospy.Rate(CONTROL_RATE_HZ)
        last_speed = 0.05
        misses = 0
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            yaw = self.yaw()
            if yaw is None:
                misses += 1
                if misses > MAX_TF_MISSES:
                    rospy.logwarn("[nav] Closed-loop rotation: TF stalled %.1fs, "
                                  "bailing early (undershoot likely)",
                                  misses / float(CONTROL_RATE_HZ))
                    break
                self.send(0.0, last_speed)
                rate.sleep()
                continue
            misses = 0
            err = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
            if abs(err) < YAW_TOLERANCE:
                break
            last_speed = float(np.clip(abs(err) * TURN_GAIN, MIN_TURN_SPEED, max_speed))
            last_speed = math.copysign(last_speed, err)
            self.send(0.0, last_speed)
            rate.sleep()
        self.stop()

    def drive(self, distance, speed):
        """Drive straight by `distance` (m, negative = backwards) at `speed`.

        Timed, not closed-loop: this is the translation the pick sequence was
        tuned with. (A closed-loop odometry drive was tried in July and reverted
        after it failed on the real run.)
        """
        self.send(math.copysign(speed, distance), 0.0)
        rospy.sleep(abs(distance) / speed)
        self.stop()

    def send(self, linear=0.0, angular=0.0):
        t = Twist()
        t.linear.x = linear
        t.angular.z = angular
        self.cmd_vel_pub.publish(t)

    def stop(self):
        self.send(0.0, 0.0)
