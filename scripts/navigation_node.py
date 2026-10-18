#!/usr/bin/env python3
"""navigation_node — every base motion, as services. The only cmd_vel publisher.

  ~drive_relative  (DriveRelative)   turn in place, then drive straight
  ~approach_target (ApproachTarget)  put the base at the standoff in front of a
                                     target: COARSE approach, FINE single-shot
                                     positioning, or one CORRECTion

Positioning measures the target through the world model (world_model_node.py)
and moves the base through BaseController; the policy itself is approach.py.

  rosservice call /locobot/navigation/drive_relative "{turn: 0.5236}"

approach_target calls world_model/get_target and world_model/get_scene, so it
needs world_model_node running to do anything — if that node is down, the call
fails cleanly with "World model unavailable: ...", caught by _Robot and
reported as PositioningFailed, rather than crashing this node.
"""
import math
import os
import sys
import threading

import numpy as np
import rospy
import tf2_ros
from sensor_msgs.msg import LaserScan
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import approach
from base_controller import BaseController
from locobot_prog.srv import (ApproachTarget, ApproachTargetResponse, DriveRelative,
                              DriveRelativeResponse, GetScene, GetTarget)
from target_state import measured_position

# Positioning trusts a lock at most this old (the grasp point uses 3 s).
MEASURE_MAX_LOCK_AGE = 2.0


class NavigationNode:

    def __init__(self):
        ns = rospy.get_param("~robot_name", "locobot")
        # Defaults used when a request leaves max_angular / max_linear at 0.
        self.default_angular = float(rospy.get_param("~default_angular_speed", 0.30))
        self.default_linear = float(rospy.get_param("~default_linear_speed", 0.08))
        # Refuse to drive forward when the laser sees something this close ahead.
        self.front_veto_m = float(rospy.get_param("~front_veto_m", 0.30))

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.base = BaseController(ns, self.tf_buffer)
        self._lock = threading.Lock()          # one base motion at a time

        self.front_dist = float("inf")
        rospy.Subscriber(f"/{ns}/scan", LaserScan, self._on_scan, queue_size=1)
        self.get_target = rospy.ServiceProxy(f"/{ns}/world_model/get_target", GetTarget)
        self.get_scene = rospy.ServiceProxy(f"/{ns}/world_model/get_scene", GetScene)

        rospy.Service("~drive_relative", DriveRelative, self._handle_drive_relative)
        rospy.Service("~approach_target", ApproachTarget, self._handle_approach_target)
        rospy.loginfo("[nav] Ready (front veto %.2fm)", self.front_veto_m)

    def _on_scan(self, msg):
        """Nearest return in a ±15° cone straight ahead."""
        n = len(msg.ranges)
        k = max(1, n // 24)
        cone = list(msg.ranges[:k]) + list(msg.ranges[n - k:])
        valid = [r for r in cone if msg.range_min < r < msg.range_max]
        self.front_dist = min(valid) if valid else float("inf")

    def front_blocked(self):
        return self.front_dist < self.front_veto_m

    def _handle_drive_relative(self, req):
        with self._lock:
            res = DriveRelativeResponse(success=True, turn_achieved=float("nan"))
            w = req.max_angular if req.max_angular > 0.0 else self.default_angular
            v = req.max_linear if req.max_linear > 0.0 else self.default_linear

            if req.turn != 0.0:
                start = self.base.yaw()
                self.base.rotate_by(req.turn, w)
                end = self.base.yaw()
                if start is not None and end is not None:
                    res.turn_achieved = math.atan2(math.sin(end - start), math.cos(end - start))
                rospy.loginfo("[nav] Turn %.2f° (achieved %.2f°)", math.degrees(req.turn),
                              math.degrees(res.turn_achieved))

            if req.distance != 0.0:
                if req.distance > 0.0 and self.front_blocked():
                    rospy.logwarn("[nav] Forward drive refused — obstacle %.2fm ahead",
                                  self.front_dist)
                    res.success = False
                    res.front_blocked = True
                    res.message = f"Obstacle {self.front_dist:.2f}m ahead — cannot drive forward."
                    return res
                rospy.loginfo("[nav] Drive %+.3fm at %.2fm/s", req.distance, v)
                self.base.drive(req.distance, v)
            return res

    def _handle_approach_target(self, req):
        robot = _Robot(self, req.target_name, req.standoff, req.table_near_x)
        res = ApproachTargetResponse()
        with self._lock:
            try:
                if req.mode == req.COARSE:
                    approach.coarse(robot, req.standoff, req.table_near_x)
                    res.success = True
                    return res
                if req.mode == req.FINE:
                    # The safe-distance window is live-tunable so the standoff can be
                    # moved for experiments (rosparam ~too_close_m / ~too_far_m).
                    out = approach.fine(robot, req.standoff,
                                        float(rospy.get_param("~too_close_m", approach.TOO_CLOSE)),
                                        float(rospy.get_param("~too_far_m", approach.TOO_FAR)))
                else:
                    out = approach.correct(robot, req.standoff)
                res.distance, res.bearing, res.target.x, res.target.y = out
                res.success = True
            except approach.PositioningFailed as e:
                rospy.logwarn("[nav] Positioning failed: %s", e.message)
                res.message, res.zone, res.scene_summary = e.message, e.zone, e.summary
                res.too_close = e.too_close
        return res


class _Robot:
    """approach.py's robot interface, backed by the base controller and the
    world model. The pauses after each move are the ones the pick sequence was
    tuned with (the camera needs the base to settle before the next measure)."""

    def __init__(self, node, target_name, standoff, table_near_x):
        self.node, self.target_name = node, target_name
        self.standoff, self.table_near_x = standoff, table_near_x

    def _state(self):
        try:
            return self.node.get_target(target_name=self.target_name).state
        except rospy.ServiceException as e:
            raise approach.PositioningFailed(f"World model unavailable: {e}")

    def _scene(self):
        try:
            return self.node.get_scene(target_name=self.target_name,
                                       nav_x_limit=self.table_near_x, standoff=self.standoff)
        except rospy.ServiceException as e:
            raise approach.PositioningFailed(f"World model unavailable: {e}")

    def measure(self):
        p = measured_position(self._state(), MEASURE_MAX_LOCK_AGE)
        if p is not None:
            return p[0], p[1]
        s = self._scene()                       # last resort: re-read the scene
        if not s.target_found:
            raise approach.PositioningFailed(f"'{self.target_name}' lost during positioning.",
                                             s.zone, s.summary)
        if s.zone == "path_blocked":
            bn = s.blocking[0] if s.blocking else "unknown"
            raise approach.PositioningFailed(f"Path blocked by {bn}.", s.zone, s.summary)
        return s.target.position.x, s.target.position.y

    def map_target(self):
        st = self._state()
        return (st.map_position.x, st.map_position.y) if st.map_valid else None

    def coarse_target(self):
        """The map reprojection stays stable while the base turns; the live
        scene is the fallback."""
        m = self.map_target()
        if m is not None:
            return m
        s = self._scene()
        if not s.target_found:
            raise approach.PositioningFailed(f"'{self.target_name}' not detected.",
                                             s.zone, s.summary)
        return s.target.position.x, s.target.position.y

    def front_blocked(self):
        return self.node.front_blocked()

    def rotate(self, angle, speed):
        self.node.base.rotate_by(angle, speed)
        rospy.sleep(0.3)

    def drive_straight(self, m, speed):
        if m > 0 and self.front_blocked():
            raise approach.PositioningFailed(
                f"Obstacle {self.node.front_dist:.2f}m ahead — cannot drive forward.")
        self.node.base.drive(m, speed)

    def face(self, angle):
        spd = float(np.clip(1.5 * abs(angle), 0.10, 0.20))   # slow for small angles
        self.node.base.rotate_by(angle, spd)
        rospy.sleep(0.6)

    def drive(self, m):
        if m > 0 and self.front_blocked():                   # table / obstacle guard
            rospy.logwarn("[nav] Forward drive held — %.2fm ahead", self.node.front_dist)
            return
        self.node.base.drive(max(-0.15, min(0.15, m)), 0.06)  # one bounded correction
        rospy.sleep(0.6)

    def log(self, fmt, *args):
        rospy.loginfo("[nav] " + fmt, *args)


if __name__ == "__main__":
    rospy.init_node("navigation")
    NavigationNode()
    rospy.spin()
