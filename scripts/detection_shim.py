#!/usr/bin/env python3
"""detection_shim — bridges yolo_detection_node.py's JSON detections into the
typed DetectionArray world_model_node expects.

NOT a rework-attempt-1 file — that branch's world_model_node assumed
yolo_detection_node.py itself published DetectionArray natively. Porting
yolo_detection_node.py was explicitly ruled out (it's the one perception node
everything currently depends on, and touching the live production path was
the higher-risk option); this shim exists instead so that node stays
completely untouched.

Also the ONE place, by deliberate design, that keeps a real capability the
rework's own ported code silently dropped: task_orchestrator_node.py never
calls anything equivalent to locobot_geometry.estimate_grasp_orientation
(contour + symmetry-guard + 180-degree flip resolution via the raw
theta_ray) — it relies purely on the raw, unrefined orientation field. This
shim applies that SAME refinement (unchanged, imported from
locobot_geometry.py, still used identically by the still-live
vlm_agent_node.py) to every detection before publishing, so world_model_node
gets the refined yaw, not the raw one.

  /locobot/locobot_prog/detections_3d   (String, JSON)  <- from yolo_detection_node.py
        |
        |  + estimate_grasp_orientation refinement per detection
        v
  /locobot/locobot_prog/detections      (DetectionArray) -> world_model_node
"""
import json
import os
import sys

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String

# See arm_node.py's identical comment: roslaunch execs this file through a
# devel/lib/ wrapper, so __file__'s own directory must be added explicitly
# for the flat sibling imports below to resolve.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conversions import detection_to_msg
from locobot_geometry import estimate_grasp_orientation
from locobot_prog.msg import DetectionArray


class DetectionShim:

    def __init__(self):
        ns = rospy.get_param("~robot_name", "locobot")
        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_depth = None   # float32 metres, aligned to color frame

        self.pub = rospy.Publisher(f"/{ns}/locobot_prog/detections",
                                   DetectionArray, queue_size=1)
        rospy.Subscriber(f"/{ns}/camera/color/image_raw", Image,
                         self._on_image, queue_size=1)
        rospy.Subscriber(f"/{ns}/camera/aligned_depth_to_color/image_raw", Image,
                         self._on_depth, queue_size=1)
        rospy.Subscriber(f"/{ns}/locobot_prog/detections_3d", String,
                         self._on_detections, queue_size=1)
        rospy.loginfo("[shim] Ready — detections_3d (JSON) -> detections (DetectionArray)")

    def _on_image(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError:
            pass

    def _on_depth(self, msg):
        try:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            # RealSense aligned depth is uint16 millimetres -> float32 metres
            if raw.dtype == np.uint16:
                self.latest_depth = raw.astype(np.float32) / 1000.0
            else:
                self.latest_depth = raw.astype(np.float32)
        except CvBridgeError:
            pass

    def _on_detections(self, msg):
        try:
            dets = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(dets, list):
            return

        out = DetectionArray()
        out.header.stamp = rospy.Time.now()
        for d in dets:
            if not (d.get("position_base") and d.get("dims_hwl") and d.get("class_name")):
                continue
            bbox_2d = d.get("bbox_2d")
            if self.latest_frame is not None and bbox_2d and len(bbox_2d) >= 4:
                refined = estimate_grasp_orientation(
                    self.latest_frame, bbox_2d,
                    fallback_yaw=d.get("orientation", 0.0),
                    depth_img=self.latest_depth,
                    depth_m=d.get("depth_m"),
                    dims_hwl=d.get("dims_hwl"),
                    orientation=d.get("orientation"),
                )
                d = dict(d, orientation=refined)
            out.detections.append(detection_to_msg(d))
        self.pub.publish(out)


if __name__ == "__main__":
    rospy.init_node("detection_shim")
    DetectionShim()
    rospy.spin()
