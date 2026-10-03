#!/usr/bin/env python3
"""Auto-label training frames from Gazebo ground truth, for fine-tuning
YOLO3D's underlying detector (ObjectDetector / plain yolo11s — NOT an OBB
model: this project must keep the YOLO3D architecture, so the fine-tune
target is a standard axis-aligned detector, same as it loads today).

Why this needs no manual annotation: each catalogued object's true size and
pose are known exactly in Gazebo. Projecting its 3D box into the camera
image (auto_label.py, offline-tested) gives an exact 2D bounding box for
free. The geometry: for each accepted camera frame, get the robot's and
object's world poses from /gazebo/get_model_state, compute the object's pose
in base_link (auto_label.relative_pose, same full-orientation math as
rotations.world_to_body, already trusted elsewhere in this project), look up
base_link -> camera optical frame via TF (the same direction
yolo_detection_node.py's own _to_base_frame already proves reliable, just
reversed), build the object's 8 box corners from its catalogued size, and
project them (auto_label.project_box_to_bbox2d) to get the 2D label.

Usage: run a sim WITHOUT the vision nodes (no YOLO needed for collection,
only the camera + Gazebo + TF):
    roslaunch locobot_prog locobot.launch use_vision_nodes:=false
GOTCHA: set_camera_angle.py (which points the camera at the table, pan=0
tilt=0.35) lives inside vision_nodes.launch, so use_vision_nodes:=false
leaves the camera at its raw spawn angle — pointed at the robot's own
pan/tilt mast, not the table (confirmed: a saved frame showed only the mast
and sky). Set it manually once per launch before collecting:
    rostopic pub -1 /locobot/tilt_controller/command std_msgs/Float64 0.35
    rostopic pub -1 /locobot/pan_controller/command  std_msgs/Float64 0.0
then, in another terminal:
    rosrun locobot_prog collect_training_data.py _output_dir:=/path/to/dataset
Move the robot and/or objects between runs (teleport via
/gazebo/set_model_state, or drive it) for viewpoint variety; re-running
accumulates into the same output_dir rather than overwriting it.

Verified live: the projected boxes land pixel-tight on the real objects
(visual check against saved debug frames) across bottle/tea_box/can_coke,
including correct clipping at the frame edge and correctly dropping an
object that has gone fully out of view.
"""
import os
import sys
import time

import numpy as np
import rospkg
import rospy
import yaml
from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers PointStamped with tf2
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auto_label as al
import rotations
from resolver import ObjectCatalog

# Stable class-id order, written to data.yaml so a training run's class ids
# match this list regardless of collection order. Keep this list in sync with
# config/objects.yaml; a catalogued object missing here is simply never
# labeled (skipped with a one-time warning), not an error.
# "table", not "dining_table": matches this project's own catalogue key
# (config/objects.yaml) and the Gazebo model's own name (<model name="table">
# in locobot_table_arm.world), consistent with how the other 4 classes are
# named after their catalogue key rather than a COCO label. The table is
# `role: surface` in the catalogue but that only matters to the live
# pipeline's obstacle logic — catalog.size() ignores it, so the table is
# labeled exactly like any other catalogued object here (its static box:
# 0.40 x 0.80 x 0.25 m, origin at floor level, matching every other object's
# "origin is the BASE" convention already assumed below).
CLASS_NAMES = ["bottle", "tea_box", "can_coke", "ball", "table"]
# Catalogue key -> Gazebo model name. Matches this project's world file
# (locobot_table_arm.world's <name> tags) directly for every current class;
# override here if a future world spawns an object under a different name.
GAZEBO_MODEL_NAME = {k: k for k in CLASS_NAMES}


class TrainingDataCollector:

    def __init__(self):
        self.robot_name = rospy.get_param("~robot_name", "locobot")
        self.rate_hz = float(rospy.get_param("~rate", 0.75))
        self.max_frames = int(rospy.get_param("~max_frames", 0))   # 0 = unlimited
        self.save_debug = bool(rospy.get_param("~save_debug_images", True))
        out_dir = rospy.get_param("~output_dir",
                                  os.path.expanduser("~/th_ws/yolo_training_data"))
        self.images_dir = os.path.join(out_dir, "images")
        self.labels_dir = os.path.join(out_dir, "labels")
        self.debug_dir = os.path.join(out_dir, "debug")
        for d in (self.images_dir, self.labels_dir, self.debug_dir):
            os.makedirs(d, exist_ok=True)
        self._write_data_yaml(out_dir)

        default_catalog = rospkg.RosPack().get_path("locobot_prog") + "/config/objects.yaml"
        self.catalog = ObjectCatalog.from_yaml(rospy.get_param("~catalog", default_catalog))
        self.class_id = {name: i for i, name in enumerate(CLASS_NAMES)}
        missing = [n for n in CLASS_NAMES if n not in self.catalog.objects]
        if missing:
            rospy.logwarn("[collect] Not in the catalogue, will never be "
                          "labeled: %s", missing)

        self._get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bridge = CvBridge()

        self.K = None
        self.camera_frame = None
        self._last_save = 0.0
        self._n_saved = self._count_existing()

        ns = self.robot_name
        rospy.Subscriber(f"/{ns}/camera/color/camera_info", CameraInfo,
                         self._on_camera_info, queue_size=1)
        rospy.Subscriber(f"/{ns}/camera/color/image_raw", Image,
                         self._on_image, queue_size=1, buff_size=2**24)
        rospy.loginfo("[collect] Ready. output=%s  classes=%s  starting at frame #%d",
                      out_dir, CLASS_NAMES, self._n_saved)

    def _write_data_yaml(self, out_dir):
        path = os.path.join(out_dir, "data.yaml")
        with open(path, "w") as f:
            yaml.safe_dump({"path": out_dir, "train": "images", "val": "images",
                            "nc": len(CLASS_NAMES), "names": CLASS_NAMES}, f)

    def _count_existing(self):
        if not os.path.isdir(self.images_dir):
            return 0
        return len([n for n in os.listdir(self.images_dir) if n.endswith(".jpg")])

    def _on_camera_info(self, msg):
        if self.K is not None:
            return
        self.K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        self.camera_frame = msg.header.frame_id
        rospy.loginfo("[collect] Camera ready. frame=%s", self.camera_frame)

    def _on_image(self, msg):
        now = time.monotonic()
        if now - self._last_save < 1.0 / max(self.rate_hz, 0.01):
            return
        if self.K is None or self.camera_frame is None:
            return
        if self.max_frames and self._n_saved >= self.max_frames:
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logwarn("[collect] CvBridge: %s", e)
            return
        img_h, img_w = bgr.shape[:2]

        try:
            tf_cam_base = self.tf_buffer.lookup_transform(
                self.camera_frame, f"{self.robot_name}/base_link", msg.header.stamp,
                rospy.Duration(0.2))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5.0, "[collect] TF lookup failed: %s", e)
            return

        try:
            robot = self._get_model_state(self.robot_name, "")
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(5.0, "[collect] get_model_state(robot): %s", e)
            return
        if not robot.success:
            return
        robot_pos = (robot.pose.position.x, robot.pose.position.y, robot.pose.position.z)
        robot_quat = (robot.pose.orientation.x, robot.pose.orientation.y,
                      robot.pose.orientation.z, robot.pose.orientation.w)

        lines, boxes_for_debug = [], []
        for name in CLASS_NAMES:
            if name not in self.catalog.objects:
                continue
            size = self.catalog.size(name)   # (length, width, height)
            if size is None:
                continue
            model_name = GAZEBO_MODEL_NAME[name]
            try:
                obj = self._get_model_state(model_name, "")
            except rospy.ServiceException:
                continue
            if not obj.success:
                continue   # this object isn't spawned in the current world

            obj_pos = (obj.pose.position.x, obj.pose.position.y, obj.pose.position.z)
            obj_quat = (obj.pose.orientation.x, obj.pose.orientation.y,
                        obj.pose.orientation.z, obj.pose.orientation.w)
            pos_base, R_base = al.relative_pose(robot_pos, robot_quat, obj_pos, obj_quat,
                                                rotations.quaternion_matrix)
            length, width, height = size
            # Model origin is the object's BASE (see e.g. tea_box/model.sdf's
            # own header comment) — its geometric centre is height/2 up along
            # the object's OWN local z axis, hence R_base @ [0,0,h/2].
            center_base = pos_base + R_base @ np.array([0.0, 0.0, height / 2.0])
            corners_base = al.box_corners(center_base, (length / 2.0, width / 2.0, height / 2.0),
                                          R_base)
            corners_cam = []
            for c in corners_base:
                pt = PointStamped()
                pt.header.frame_id = f"{self.robot_name}/base_link"
                pt.header.stamp = msg.header.stamp
                pt.point.x, pt.point.y, pt.point.z = c
                pt_cam = tf2_geometry_msgs.do_transform_point(pt, tf_cam_base)
                corners_cam.append((pt_cam.point.x, pt_cam.point.y, pt_cam.point.z))

            box2d = al.project_box_to_bbox2d(self.K, corners_cam, img_w, img_h)
            if box2d is None:
                continue
            lines.append(al.yolo_label_line(self.class_id[name], box2d, img_w, img_h))
            boxes_for_debug.append((name, box2d))

        if not lines:
            if self.save_debug:   # keep a few misses too, for diagnosing why
                cv2.imwrite(os.path.join(self.debug_dir, "miss_%06d.jpg" % self._n_saved), bgr)
            return   # nothing labelable in this frame — don't save a blank one

        self._last_save = now
        idx = self._n_saved
        self._n_saved += 1
        img_path = os.path.join(self.images_dir, "frame_%06d.jpg" % idx)
        lbl_path = os.path.join(self.labels_dir, "frame_%06d.txt" % idx)
        cv2.imwrite(img_path, bgr)
        with open(lbl_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        if self.save_debug:
            dbg = bgr.copy()
            for name, (x1, y1, x2, y2) in boxes_for_debug:
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(dbg, p1, p2, (0, 255, 0), 2)
                cv2.putText(dbg, name, (p1[0], max(0, p1[1] - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(self.debug_dir, "frame_%06d.jpg" % idx), dbg)

        rospy.loginfo_throttle(2.0, "[collect] saved frame #%d (%d object(s): %s)",
                               idx, len(lines), [n for n, _ in boxes_for_debug])

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("collect_training_data")
    TrainingDataCollector().run()
