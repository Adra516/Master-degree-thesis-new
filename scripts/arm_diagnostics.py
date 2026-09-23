"""Grasp measurements: verify the pre-grasp pose and where the object sits
between the open jaws, BEFORE closing them.

Ported near-verbatim from rework-attempt-1's src/locobot_prog/arm/
diagnostics.py, itself "moved verbatim from locobot_motion_node.py" (retired,
see git history) — with ONE substantive change, carried over deliberately:
_object_in_base uses rotations.world_to_body (full orientation) instead of a
yaw-only transform. MEASURED bug (rework-attempt-1, commit ecea036): the
robot rests pitched ~2 deg nose-down on its caster in every arm pose, and a
yaw-only ground-truth transform put a point at grasp height ~12mm off along
the approach — this is a real, independent fix, worth having even though this
file's rest is otherwise an exact copy.

Every measurement returns None when it could not be taken; callers treat
None as UNVERIFIED (passes), never as a failure, so an instrumentation
outage cannot become a grasp outage. For the same reason every measurement is
wrapped in a blanket except.
"""
import json
import math

import numpy as np
import rospy
import tf2_ros
from std_msgs.msg import String

import rotations


class GraspDiagnostics:

    def __init__(self, robot_name, move_group, tf_buffer):
        self.ns = robot_name
        self.move_group = move_group
        self.tf_buffer = tf_buffer
        self._grasp_diag = bool(rospy.get_param("~grasp_diag", True))
        # Gazebo model to compare against (sim only); read live on every check.
        self._diag_object_model = str(rospy.get_param("~diag_object_model", ""))
        self._diag_pub = rospy.Publisher(f"/{robot_name}/locobot_prog/arm_grasp_diag",
                                         String, queue_size=4, latch=True)
        self._gt_srv = None
        if self._grasp_diag:
            try:
                from gazebo_msgs.srv import GetModelState
                rospy.wait_for_service("/gazebo/get_model_state", timeout=3.0)
                self._gt_srv = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
                rospy.loginfo("[diag] Gazebo ground-truth available — contact-vs-object check enabled")
            except (rospy.ROSException, ImportError):
                rospy.loginfo("[diag] Gazebo GT unavailable — pre-grasp + fingers-vs-commanded still logged")

    def _finger_xyz(self, side):
        """(x,y,z) of a finger link in base_link via TF, or None."""
        try:
            t = self.tf_buffer.lookup_transform(
                f"{self.ns}/base_link", f"{self.ns}/{side}_finger_link",
                rospy.Time(0), rospy.Duration(0.3))
            tr = t.transform.translation
            return np.array([tr.x, tr.y, tr.z])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None

    def _object_in_base(self, model):
        """Ground-truth object centre in base_link via Gazebo, or None.  Full
        orientation, not yaw only: the robot rests pitched ~2° nose-down, which
        a yaw-only transform turned into a ~12 mm error along the approach."""
        if self._gt_srv is None or not model:
            return None
        try:
            r = self._gt_srv(self.ns, "")
            o = self._gt_srv(model, "")
        except rospy.ServiceException:
            return None
        if not (getattr(r, "success", False) and getattr(o, "success", False)):
            return None
        p, q = r.pose.position, r.pose.orientation
        return rotations.world_to_body(
            (p.x, p.y, p.z), (q.x, q.y, q.z, q.w),
            (o.pose.position.x, o.pose.position.y, o.pose.position.z))

    def _publish_diag(self, payload):
        try:
            self._diag_pub.publish(String(data=json.dumps(payload)))
        except Exception:
            pass

    def pregrasp(self, cx, cy, cz, cori):
        """VERIFY the manipulator's pre-grasp pose: actual EE (base_link, MoveIt FK
        from the live joint state) vs the COMMANDED pre-grasp pose.  Pure FK, no
        frame ambiguity, so this works on hardware as well as in sim.
        Returns the measurement dict (err_norm_mm, ori_err_deg, …) so the caller
        can GATE on it, or None when no measurement could be taken (diagnostics
        disabled, or FK/TF failed).  None means UNVERIFIED, not "failed" — the
        caller must not treat a missing measurement as a bad pose.  Measurement
        must never break a grasp, hence the blanket except.
        """
        if not self._grasp_diag:
            return None
        try:
            p = self.move_group.get_current_pose().pose
            dx, dy, dz = p.position.x - cx, p.position.y - cy, p.position.z - cz
            norm_mm = math.sqrt(dx*dx + dy*dy + dz*dz) * 1000.0
            a, b = p.orientation, cori
            dot = abs(a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w)
            ori_deg = math.degrees(2.0 * math.acos(min(1.0, dot)))
            rospy.loginfo(
                "[diag] PRE-GRASP EE  cmd=(%.3f,%.3f,%.3f)  act=(%.3f,%.3f,%.3f)  "
                "err dx=%+.1f dy=%+.1f dz=%+.1f |Δ|=%.1fmm  ori_err=%.1f°",
                cx, cy, cz, p.position.x, p.position.y, p.position.z,
                dx*1000, dy*1000, dz*1000, norm_mm, ori_deg)
            payload = {"phase": "pre_grasp",
                       "cmd": [cx, cy, cz],
                       "act": [p.position.x, p.position.y, p.position.z],
                       "err_mm": [dx*1000, dy*1000, dz*1000],
                       "err_norm_mm": norm_mm, "ori_err_deg": ori_deg}
            self._publish_diag(payload)
            return payload
        except Exception as e:
            rospy.logwarn("[diag] pre-grasp measurement failed: %s", e)
            return None

    @staticmethod
    def _jaw_metrics(centre, u, gap, obj, cmd_xy):
        """Where an object sits relative to the jaws that are about to close.
        centre/u/gap describe the gripper (finger-link midpoint, closing-axis
        unit vector, physical jaw gap); obj is the object centre in base_link;
        cmd_xy is the COMMANDED grasp point.  Returns (along, perp, straddled):

          along     — object offset along the CLOSING axis, measured from the
                      finger midpoint.  0 = perfectly centred, |along| < gap/2
                      means it lies between the fingertips.  Measured from the
                      fingers because that is the real geometry of straddling,
                      and this axis carries no frame offset.
          perp      — miss along the APPROACH direction, measured from the
                      COMMANDED grasp point, NOT the finger midpoint.  The finger
                      link frames sit ~28 mm behind ee_gripper_link along exactly
                      this axis, so measuring it from the fingers would report a
                      constant ~28 mm "miss" on a perfectly executed grasp and
                      the gate would reject everything.  The commanded point is
                      where the jaws actually close, and the pre-grasp gate has
                      already verified the arm tracks its commands.
          straddled — object lies between the jaws along the closing axis.
        """
        along     = float(np.dot(obj - centre, u))
        straddled = bool(abs(along) < gap / 2.0)
        d_cmd     = np.asarray(obj[:2], float) - np.asarray(cmd_xy, float)
        perp      = float(np.linalg.norm(d_cmd - float(np.dot(d_cmd, u[:2])) * u[:2]))
        return along, perp, straddled

    def contact(self, cx, cy, cz, obj_base=None):
        """CHECK where the object sits relative to the jaws, BEFORE closing them.
        Logs the gripper centre, physical jaw gap and the offset vs the COMMANDED
        grasp point, then measures the object offset from up to two sources:

        * obj_base — the object position supplied by the caller (the agent's
          odom-cache reprojection).  HARDWARE-PORTABLE: needs neither the camera
          (the arm occludes it at grasp height) nor ground truth.  These are the
          metrics the caller GATES on.
        * Gazebo ground truth, when ~diag_object_model is set — logged alongside
          as an extra so the odom estimate's accuracy can be measured against
          truth in sim.  Never the gate, so the gate stays hardware-portable.

        Returns the gate metrics dict, or None when no object position was
        available from either source (UNVERIFIED, not "failed").  Measurement
        must never break a grasp, hence the blanket except.
        """
        if not self._grasp_diag:
            return None
        try:
            lf, rf = self._finger_xyz("left"), self._finger_xyz("right")
            if lf is None or rf is None:
                rospy.logwarn("[diag] contact: finger TF unavailable — skipped")
                return None
            centre = (lf + rf) / 2.0
            axis   = rf - lf
            gap    = float(np.linalg.norm(axis))
            u      = axis / (gap + 1e-9)              # finger-closing axis unit vector
            dcx, dcy, dcz = centre[0]-cx, centre[1]-cy, centre[2]-cz
            rospy.loginfo(
                "[diag] CONTACT gripper-centre=(%.3f,%.3f,%.3f)  jaw_gap=%.1fmm",
                centre[0], centre[1], centre[2], gap*1000)
            rospy.loginfo(
                "[diag] CONTACT vs CMD grasp point: dx=%+.1f dy=%+.1f dz=%+.1f mm",
                dcx*1000, dcy*1000, dcz*1000)
            payload = {"phase": "contact",
                       "gripper_centre": centre.tolist(), "jaw_gap_mm": gap*1000,
                       "left_finger": lf.tolist(), "right_finger": rf.tolist(),
                       "vs_cmd_mm": [dcx*1000, dcy*1000, dcz*1000]}
            # ── GATE source: the caller-supplied object position ────────────────
            # Comes from the agent's odom-cache reprojection, so it needs neither
            # the camera nor ground truth and the gate behaves the same on
            # hardware as in sim.
            gate = None
            if obj_base is not None:
                obj   = np.asarray(obj_base, dtype=float)
                rel   = obj - centre
                along, perp, straddled = self._jaw_metrics(centre, u, gap, obj, (cx, cy))
                rospy.loginfo(
                    "[diag] CONTACT vs OBJECT (odom): object=(%.3f,%.3f,%.3f)  "
                    "along-jaw=%+.1fmm (0=centred)  in-plane-miss=%.1fmm  straddled=%s",
                    obj[0], obj[1], obj[2], along*1000, perp*1000, straddled)
                gate = {"along_jaw_mm": along*1000, "in_plane_miss_mm": perp*1000,
                        "straddled": straddled, "source": "odom"}
                payload.update(dict(gate, object_base=obj.tolist(),
                                    offset_mm=(rel*1000).tolist()))

            # ── EXTRA: Gazebo ground truth, for accuracy comparison only ───────
            # Never gates — gating on GT would validate a check that cannot exist
            # on the real robot.  Model name read LIVE so you can
            # `rosparam set …/diag_object_model <name>` without relaunching.
            model  = rospy.get_param("~diag_object_model", self._diag_object_model)
            obj_gt = self._object_in_base(model)
            if obj_gt is not None:
                a_gt, p_gt, s_gt = self._jaw_metrics(centre, u, gap, obj_gt, (cx, cy))
                rospy.loginfo(
                    "[diag] CONTACT vs OBJECT (GT '%s'): object=(%.3f,%.3f,%.3f)  "
                    "along-jaw=%+.1fmm  in-plane-miss=%.1fmm  straddled=%s  → %s",
                    model, obj_gt[0], obj_gt[1], obj_gt[2], a_gt*1000, p_gt*1000, s_gt,
                    "OK/centred" if (s_gt and p_gt < 0.010) else "OFF-CENTRE (finger-push risk)")
                payload.update({"gt_object_base": obj_gt.tolist(),
                                "gt_along_jaw_mm": a_gt*1000,
                                "gt_in_plane_miss_mm": p_gt*1000,
                                "gt_straddled": s_gt})
                if obj_base is not None:
                    # How far the hardware-portable estimate is from truth — the
                    # number that says whether the odom path can be trusted.
                    est_err = float(np.linalg.norm(np.asarray(obj_base, float) - obj_gt))
                    rospy.loginfo("[diag] CONTACT odom-vs-GT object error: %.1fmm",
                                  est_err * 1000)
                    payload["odom_vs_gt_mm"] = est_err * 1000
            elif obj_base is None:
                rospy.loginfo("[diag] CONTACT vs OBJECT: no object position — pass one "
                              "in, or set ~diag_object_model for the GT reference")
            self._publish_diag(payload)
            return gate
        except Exception as e:
            rospy.logwarn("[diag] contact measurement failed: %s", e)
            return None
