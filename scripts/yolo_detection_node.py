import ast, json, os, sys, time
import numpy as np
import message_filters
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped

# YOLO3D utilities — set up before importing 
_YOLO3D_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "YOLO3D"))
if _YOLO3D_DIR not in sys.path:
    sys.path.insert(0, _YOLO3D_DIR)

from bbox3d_utils import BBox3DEstimator, DEFAULT_DIMS    # DEFAULT_DIMS, _backproject_point, _estimate_orientation
from detection_model import ObjectDetector  # YOLOv11 + multi-object tracking


class YoloDetectionNode:
    # target_object -> its empirically-measured YOLO class-label alias(es).
    # is_target = (class_name == target_object OR class_name in aliases), 
    # so an alias for the WRONG target silently steals detections from whatever
    # is actually configured — e.g. bottle's alias "vase" was kept active
    # even while target_object='airplane' (the tissue-box target), so the
    # bottle (detected far more reliably) kept winning "is_target" over the
    # intended box. Auto-deriving aliases FROM target_object makes that
    # cross-contamination structurally impossible: only the currently
    # configured target's own alias(es) are ever active.
    _ALIAS_MAP = {
        "bottle":   ["vase"],
        "cancoke": ["cell phone"],
        "teabox": ["remote"], #remote
        "ball": ["sport ball"],
    }

    def __init__(self):
        self.robot_name = rospy.get_param("~robot_name",    "locobot")
        self.conf_thresh = float(rospy.get_param("~conf_threshold", 0.10))
        self.process_rate = float(rospy.get_param("~process_rate", 2.0))
        target_object = rospy.get_param("~target_object", "bottle")
        raw_classes = rospy.get_param("~target_classes", [])

        if isinstance(raw_classes, str):
            raw_classes = ast.literal_eval(raw_classes)
        self.target_classes = [c.lower() for c in raw_classes]
        self.target_object  = target_object.lower()

        # target_aliases param is an OPTIONAL override; when left empty (the
        # normal case) the alias set is looked up from target_object, never
        # a fixed list that could apply to the wrong target.
        raw_aliases = rospy.get_param("~target_aliases", [])
        if isinstance(raw_aliases, str):
            raw_aliases = ast.literal_eval(raw_aliases)
        if raw_aliases:
            self.target_aliases = {a.lower() for a in raw_aliases}
        else:
            self.target_aliases = {a.lower() for a in
                                   self._ALIAS_MAP.get(self.target_object, [])}

        ns = self.robot_name

        # ObjectDetector (chdir so YOLO finds yolo11s.pt in YOLO3D folder)
        _cwd = os.getcwd()
        os.chdir(_YOLO3D_DIR)
        self.detector = ObjectDetector(model_size='small', conf_thres=self.conf_thresh)
        os.chdir(_cwd)

        self.K = None   # filled from camera_info
        self.bbox3d = None   # BBox3DEstimator with real K
        self.bridge = CvBridge()

        self.det_pub = rospy.Publisher(f"/{ns}/locobot_prog/detections_3d", String, queue_size=5)
        self.img_pub = rospy.Publisher(f"/{ns}/locobot_prog/yolo_image",    Image,  queue_size=1)

        rospy.Subscriber(f"/{ns}/camera/color/camera_info", CameraInfo, self._on_camera_info, queue_size=1)

        rgb_sub = message_filters.Subscriber(f"/{ns}/camera/color/image_raw", Image)
        depth_sub = message_filters.Subscriber(f"/{ns}/camera/aligned_depth_to_color/image_raw", Image)
        sync = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.5)
        sync.registerCallback(self._on_rgbd)

        self._last_proc = 0.0
        self.camera_frame = None   # filled from camera_info header (optical frame)
        self.target_frame = f"{self.robot_name}/base_link"
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.loginfo("[yolo3d] Ready. target='%s' classes=%s", self.target_object, self.target_classes)

    def _on_camera_info(self, msg):
        if self.K is not None:
            return
        self.K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        self.bbox3d = BBox3DEstimator(camera_matrix=self.K)
        self.camera_frame = msg.header.frame_id   # e.g. locobot/camera_color_optical_frame
        rospy.loginfo("[yolo3d] Camera ready. frame=%s  dims: %s",
                      self.camera_frame, list(self.bbox3d.dims.keys()))

    def _on_rgbd(self, rgb_msg, depth_msg):
        now = time.monotonic()
        if now - self._last_proc < 1.0 / max(self.process_rate, 0.1):
            return
        if self.bbox3d is None or self.camera_frame is None:
            return
        if not self.tf_buffer.can_transform(
                self.target_frame, self.camera_frame,
                rospy.Time(0), rospy.Duration(0.0)):
            rospy.logwarn_throttle(5.0, "[yolo3d] Waiting for TF %s→%s",
                                   self.camera_frame, self.target_frame)
            return
        self._last_proc = now
        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logwarn("[yolo3d] CvBridge: %s", e)
            return

        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32)

        self._process(bgr, depth, rgb_msg.header.stamp)

    def _process(self, bgr, depth, stamp):
        # ObjectDetector.detect() → (annotated_bgr, [[bbox, score, cls_id, obj_id], ...])
        annotated, raw_dets = self.detector.detect(bgr, track=True)
        detections = []

        # Classes always published regardless of target_classes (needed for scene modelling)
        SCENE_CLASSES = {"dining table", "table"}

        for bbox, score, cls_id, obj_id in raw_dets:
            class_name = self.detector.model.names[cls_id].lower()
            if (self.target_classes
                    and class_name not in self.target_classes
                    and class_name not in SCENE_CLASSES):
                continue

            x1, y1, x2, y2 = [float(v) for v in bbox]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            # Metric depth from RealSense aligned depth image
            depth_m = self._sample_depth(depth, x1, y1, x2, y2)

            # Surface-to-center compensation: the sampled depth is the object's
            # FRONT SURFACE (median of the bbox region), not its central axis.
            # Grasping needs the axis, so push the point deeper along the camera
            # ray by the estimated radius (half the measured width).  Without
            # this every grasp lands ~one radius short and the fingers side-swipe
            # the object (measured bias vs Gazebo ground truth: -3.9 cm for the
            # bottle at 0.31 m depth).
            if depth_m > 0:
                fx0 = self.K[0, 0]
                w_est = ((x2 - x1) / fx0) * depth_m         # measured width (m)
                radius_est = float(np.clip(w_est / 2.0, 0.005, 0.05))
                depth_center = depth_m + radius_est
            else:
                depth_center = depth_m

            # 3D camera-frame position via BBox3DEstimator pinhole backprojection
            pos_cam = self.bbox3d._backproject_point(cx, cy, depth_center).tolist() if depth_m > 0 else [0., 0., 0.]

            # Transform to base_link frame for arm planning
            pos_base = self._to_base_frame(pos_cam, stamp, self.camera_frame, self.target_frame)
            if pos_base is None:
                continue   # TF failed on this stamp — skip rather than publish wrong coords
            # Camera-measured H and W; fall back to DEFAULT_DIMS only for L
            if depth_m > 0:
                fx  = self.K[0, 0];  fy = self.K[1, 1]
                W_m = ((x2 - x1) / fx) * depth_m
                H_m = ((y2 - y1) / fy) * depth_m
                # Rotationally symmetric objects: depth extent = width.  Using the
                # DEFAULT_DIMS class prior for L here (e.g. vase L=0.15 for a
                # 0.06 m bottle) breaks the downstream symmetry check — the
                # footprint looks elongated, yaw estimation runs, and the grasp
                # gets a bogus ±60° gripper angle.
                if class_name in ("bottle", "vase", "cup", "wine glass", "bowl"):
                    L_m = W_m
                else:
                    _L  = DEFAULT_DIMS.get(class_name)
                    L_m = float(_L[2]) if _L is not None else W_m
                dims_hwl = [round(H_m, 3), round(W_m, 3), round(L_m, 3)]
            else:
                _dims = self.bbox3d.dims.get(class_name)
                dims_hwl = (_dims if _dims is not None else self.bbox3d.dims["bottle"]).tolist()

            # Ray-based yaw orientation estimate
            orientation = float(self.bbox3d._estimate_orientation([x1, y1, x2, y2], np.array(pos_cam), class_name))

            detections.append({
                "class_name":   class_name,
                "confidence":   round(float(score), 3),
                "bbox_2d":      [round(v, 1) for v in (x1, y1, x2, y2)],
                "depth_m":      round(depth_m, 3),
                "position_cam":  [round(v, 3) for v in pos_cam],
                "position_base": [round(v, 3) for v in pos_base],
                "dims_hwl":     [round(v, 3) for v in dims_hwl],
                "orientation":  round(orientation, 4),
                "object_id":    int(obj_id) if obj_id is not None else None,
                "is_target":    class_name == self.target_object or class_name in self.target_aliases,
            })

        self.det_pub.publish(String(data=json.dumps(detections)))
        try:
            self.img_pub.publish(self.bridge.cv2_to_imgmsg(annotated, "bgr8"))
        except CvBridgeError:
            pass

        targets = [d for d in detections if d["is_target"]]
        if targets:
            t = targets[0]
            rospy.loginfo("[yolo3d] TARGET '%s' conf=%.2f d=%.2fm pos=%s dims=%s",
                          t["class_name"], t["confidence"], t["depth_m"],
                          t["position_cam"], t["dims_hwl"])

    def _sample_depth(self, depth, x1, y1, x2, y2):
        """Median depth from inner 50% of the bbox (ignores edges and holes)."""
        dh, dw = depth.shape
        bw, bh = x2 - x1, y2 - y1
        ix1 = int(max(0, x1 + bw * 0.25)); ix2 = int(min(dw, x2 - bw * 0.25))
        iy1 = int(max(0, y1 + bh * 0.25)); iy2 = int(min(dh, y2 - bh * 0.25))
        if ix2 <= ix1 or iy2 <= iy1:
            ix1, ix2, iy1, iy2 = int(x1), int(x2), int(y1), int(y2)
        region = depth[iy1:iy2, ix1:ix2]
        valid  = region[(region > 0.05) & (region < 10.0)]
        return float(np.median(valid)) if valid.size > 0 else 0.0

    def _to_base_frame(self, pos_cam, stamp, camera_frame, target_frame):
        """Transform [X,Y,Z] from camera optical frame to target_frame."""
        pt = PointStamped()
        pt.header.stamp    = stamp
        pt.header.frame_id = camera_frame
        pt.point.x, pt.point.y, pt.point.z = pos_cam
        try:
            pt_base = self.tf_buffer.transform(pt, target_frame, rospy.Duration(0.1))
            return [pt_base.point.x, pt_base.point.y, pt_base.point.z]
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5.0, "[yolo3d] TF lookup failed: %s", e)
            return None   # caller must discard this detection

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    rospy.init_node("yolo_detection_node")
    YoloDetectionNode().run()
