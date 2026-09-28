"""The two-phase grasp, and the place that ends a pick-and-place.

Ported near-verbatim from rework-attempt-1's src/locobot_prog/arm/
grasp_executor.py, itself "moved from locobot_motion_node.py. The logic, the
timings and the checks are unchanged; what changed is where the geometry
comes from." The arm used to derive the grasp pose, standoff and gripper
widths itself from the object's dimensions (locobot_motion_node.py's
_do_pregrasp/_do_descent). It now executes a GraspPlan (a plain dict, the
fields grasp_geometry.baseline_plan() returns) exactly as given, so the
planning can be replaced without touching the arm.

Mirrors cobot's Pick_And_Place: one method per step (pregrasp, grasp, place),
the MoveIt handles injected, the order decided by the caller.

  pregrasp(): every object — the target included — is an obstacle. Raise the
              arm, open, stop at the standoff in front of the object, verify.
  grasp():    remove the target from the scene, move straight in, check the
              open jaws straddle it, close, attach, lift straight up.
  place():    lower it straight back down and release.
"""
import math

import numpy as np
import rospy
from geometry_msgs.msg import Pose

import grasp_geometry as gg
from arm_actuators import CAMERA_CLEAR_POSE, GRASP_READY_POSE
from grasp_geometry import FINGER_CLOSE, FINGER_OPEN

# Gate tolerances. Read LIVE (live_param), so `rosparam set ~<name> <v>` takes
# effect without relaunching — which is how a gate is relaxed or disabled
# mid-evaluation.
PREGRASP_POS_TOL_MM = 15.0
PREGRASP_ORI_TOL_DEG = 10.0
CONTACT_MISS_TOL_MM = 15.0
# A pending pre-grasp older than this is refused by grasp() — the arm may have
# sagged and the scene/target may no longer be what was verified.
PENDING_GRASP_TTL_S = 30.0
# Validated distance window for the target, in base_link. Too far (> 0.52):
# kinematically unreachable. Too close (< 0.33): the STANDOFF, not the target,
# runs out of arm — it sits the standoff offset nearer still (0.33 -> r=0.25),
# and the measured joint-limit margin for a level gripper collapses as the arm
# retracts (0.384 rad at r=0.36 -> 0.181 at r=0.28).
REACH_MAX = 0.52
REACH_MIN = 0.33


def live_param(name, default):
    return float(rospy.get_param(f"~{name}", default))


class GraspExecutor:

    def __init__(self, robot_name, move_group, robot, arm, gripper, scene, diag):
        self.ns = robot_name
        self.move_group = move_group      # the system's only MoveGroupCommander
        self.robot = robot                # RobotCommander: reference state for retiming
        self.arm = arm                     # arm_actuators.ArmActuator
        self.gripper = gripper             # arm_actuators.Gripper
        self.scene = scene                 # collision_scene.CollisionScene
        self.diag = diag                   # arm_diagnostics.GraspDiagnostics

        self.table_surface_z = 0.25
        self._last_grasp_z = None         # EE height of the last grasp (for place)
        self._last_grasp_xy = None        # where the jaws last closed (for place)
        self.last_close_target = None     # last commanded per-finger close (m)
        # pregrasp() leaves the arm holding at the standoff and stashes what
        # grasp() needs here. None = no verified pre-grasp: grasp() must refuse.
        self._pending_grasp = None

        rospy.loginfo("[arm] Grasp gates: pre-grasp <= %.1fmm / %.1f°, "
                      "contact miss <= %.1fmm (all live-tunable via rosparam)",
                      live_param("pregrasp_pos_tol_mm", PREGRASP_POS_TOL_MM),
                      live_param("pregrasp_ori_tol_deg", PREGRASP_ORI_TOL_DEG),
                      live_param("contact_miss_tol_mm", CONTACT_MISS_TOL_MM))

    # ── Public steps ─────────────────────────────────────────────────────────

    def pregrasp(self, plan, obstacles, target_box, table_z):
        """PHASE 1: approach the standoff and verify the arm reached it.

        plan: a GraspPlan dict (grasp_geometry.baseline_plan's shape).
        obstacles, target_box: grasp_geometry box dicts (table_box/obstacle_boxes,
        target_box respectively).

        On success the arm HOLDS at the verified standoff with the scene intact.
        On failure the scene is torn down and the arm retracts, so the caller
        can re-position the base and try again. Returns (ok, info).
        """
        self.table_surface_z = table_z
        x, y = plan["grasp_point"][0], plan["grasp_point"][1]
        dist = math.hypot(x, y)
        if dist > REACH_MAX:
            rospy.logwarn("[arm] Target at %.3fm — beyond arm reach (%.2fm), aborting",
                          dist, REACH_MAX)
            return False, {"reason": f"Target {dist:.2f}m exceeds arm reach — drive closer"}
        if dist < REACH_MIN:
            rospy.logwarn("[arm] Target at %.3fm — too close for safe arm swing "
                          "(min %.2fm), aborting", dist, REACH_MIN)
            return False, {"reason": f"Target {dist:.2f}m too close — back up"}
        # An invalid plan is refused BEFORE any arm motion — e.g. an object too
        # short for a side grasp, where the jaws would close in empty air.
        if not plan["valid"]:
            rospy.logwarn("[arm] Plan refused before moving: %s", plan["reason"])
            return False, {"reason": plan["reason"]}

        # Collision scene BEFORE moving the arm (avoids START_STATE_IN_COLLISION):
        # table and neighbours, plus the target itself for the pre-grasp phase.
        self.scene.set_obstacles(obstacles)
        self.scene.add_target(target_box)
        rospy.sleep(0.30)   # let MoveIt register the collision objects

        # Rotate the waist to face the target before raising to GRASP_READY_POSE,
        # so the Cartesian planner starts already pointing at the target.
        waist = float(np.clip(math.atan2(y, x), -1.5, 1.5))
        rospy.loginfo("[arm] Waist to target: %.3f rad (%.1f°)", waist, math.degrees(waist))
        ready = [waist] + list(GRASP_READY_POSE[1:])
        self.arm.send_joints(ready, move_time=3.5)
        gw_before = self.gripper.position

        ok, info = self._do_pregrasp(plan, waist, ready)
        if not ok:
            self.scene.clear()
            self._retract_clear()
            return False, info
        self._pending_grasp["gw_before"] = gw_before
        return True, info

    def grasp(self, plan, obj_now):
        """PHASE 2: move in, verify the jaws straddle the object, close, lift.

        Refuses without a fresh, matching pre-grasp: the arm pose AND the MoveIt
        scene from phase 1 are both preconditions, and neither is re-established
        here. obj_now: the object's CURRENT (x, y) in base_link, or None — it
        drives the contact check, which passes (unverified) without it.
        Returns (ok, info).
        """
        pg = self._pending_grasp
        if pg is None:
            return False, {"reason": "No pending pre-grasp — run execute_pregrasp first"}
        age = rospy.Time.now().to_sec() - pg["stamp"]
        if age > PENDING_GRASP_TTL_S:
            self._pending_grasp = None
            self.scene.clear()
            return False, {"reason": f"Pending pre-grasp is stale ({age:.0f}s) — "
                                     f"re-run execute_pregrasp"}
        # Identity guard: the grasp must be for the target that was verified.
        drift = math.hypot(plan["grasp_point"][0] - pg["x"], plan["grasp_point"][1] - pg["y"])
        if drift > 0.02:
            return False, {"reason": f"Descent target moved {drift*100:.0f}cm from the "
                                     f"verified pre-grasp — re-run execute_pregrasp"}
        # Compare at the grasp height: the jaws are level there, so only the
        # in-plane geometry is meaningful for the straddle test.
        obj_base = None
        if obj_now is not None:
            # z is the grasp height at the object's CURRENT position when the caller
            # knows it (NaN / absent / non-positive = unknown -> no height re-aim).
            oz = obj_now[2] if len(obj_now) > 2 else float("nan")
            obj_base = [obj_now[0], obj_now[1],
                        oz if math.isfinite(oz) and oz > 0.05 else plan["grasp_point"][2]]
        ok, info = self._do_descent(plan, obj_base)
        return self._outcome(ok, pg.get("gw_before", self.gripper.position), info)

    def place(self, x, y, place_z=None):
        """Lower the held object straight down at (x, y) and release it.

        Called right after grasp(), while the arm holds the object at the
        retreat height straight above the grasp point. Because the place point
        is the grasp point, NO pre-place swing is needed: descend the same
        vertical line (constant orientation, no twist) and open.

        RELEASE IS THE PRIORITY. A partial or failed descent must NOT skip the
        open (that left the object welded to the arm forever) — always open the
        gripper, even if the arm only got part way down.

        place_z None = 3 mm below the height it was grasped at: the object's
        base rests on the table while still held, so any tilt is squared up
        before release.
        """
        last = self._last_grasp_xy
        if last is not None and math.hypot(x - last[0], y - last[1]) < 0.06:
            x, y = last        # the arm is over the point it actually grasped
        if place_z is None:
            place_z = (self._last_grasp_z or self.table_surface_z + 0.10) - 0.003
        approach_z = place_z + 0.10

        rospy.loginfo("[arm] Place: xy=(%.3f,%.3f)  place_z=%.3f", x, y, place_z)

        current_ori = self.move_group.get_current_pose().pose.orientation

        def _wp(px, py, pz):
            p = Pose()
            p.position.x, p.position.y, p.position.z = px, py, pz
            p.orientation = current_ori
            return p

        # 1. Descend straight down. Placing the object a couple of cm high and
        #    letting it settle beats not releasing.
        plan, frac = self.move_group.compute_cartesian_path(
            [_wp(x, y, approach_z), _wp(x, y, place_z)], 0.005, False)
        rospy.loginfo("[arm] Place descent coverage: %.0f%%", frac * 100)
        if frac > 0.30:
            plan = self.move_group.retime_trajectory(
                self.robot.get_current_state(), plan,
                velocity_scaling_factor=0.15, acceleration_scaling_factor=0.15)
            dur = self.arm.publish_trajectory(plan)
            rospy.sleep(dur + 1.0)
        else:
            rospy.logwarn("[arm] Place descent only %.0f%% — releasing in place",
                          frac * 100)

        # 2. Settle, then OPEN — always. Opening wide moves the finger contact
        #    points past the grasp-fix release tolerance, so the weld detaches.
        #    A released, low-gripped (top-heavy) object can topple immediately,
        #    and gazebo_grasp_fix re-attaches on ANY new contact within one
        #    20 Hz update (grip_count_threshold=1 — required for the original
        #    grasp to attach at all).
        self.arm.hold(0.8)
        self.gripper.send(FINGER_OPEN, move_time=1.0)
        # Also drop the MoveIt attachment so future plans are clean.
        self.scene.detach()

        # 3. Retreat FAR and IMMEDIATELY — back to the post-grasp lift height (not
        #    a token 5 cm), at real speed, so the gripper clears the toppling
        #    object's space before it can re-contact anything.
        retreat_up_z = (self._last_grasp_z or place_z) + 0.15
        up_plan, up_frac = self.move_group.compute_cartesian_path(
            [_wp(x, y, retreat_up_z)], 0.01, False)
        if up_frac > 0.5:
            up_plan = self.move_group.retime_trajectory(
                self.robot.get_current_state(), up_plan,
                velocity_scaling_factor=0.3, acceleration_scaling_factor=0.3)
            up_dur = self.arm.publish_trajectory(up_plan)
            rospy.sleep(up_dur + 0.3)
        self.arm.send_joints(CAMERA_CLEAR_POSE, move_time=3.0)

        return True, {"placed_at": [round(x, 3), round(y, 3)]}

    # ── Internals ────────────────────────────────────────────────────────────

    def _outcome(self, ok, gw_before, info):
        """Final result of a grasp, after cleaning up the scene.

        Keeps the long-standing rule that a MEASURABLY CLOSED gripper counts as
        success even when planning reported a failure — the physical evidence
        outranks the planner's opinion.
        """
        gw_m = self.gripper.position
        gripper_closed_on_object = (
            gw_before > FINGER_OPEN - 0.005          # was open before
            and FINGER_CLOSE + 0.008 < gw_m          # now partially closed
            and gw_m < FINGER_OPEN - 0.005           # but not fully open (held something)
        )
        success = ok or gripper_closed_on_object
        self.scene.clear()
        self._pending_grasp = None
        if not success:
            # Retract so the camera can see the scene for the next attempt.
            self._retract_clear()

        data = dict(info or {}, gripper_width_after_m=round(gw_m, 4))
        if success:
            rospy.loginfo("[arm] Grasp succeeded (gripper=%.1fmm  plan_ok=%s  grip_check=%s)",
                          gw_m * 1000, ok, gripper_closed_on_object)
            if gw_m > FINGER_OPEN - 0.008:
                rospy.logwarn("[arm] Jaws closed only %.0fmm per finger from fully open "
                              "(a %.0fmm grip). Fine for a wide object, but if the planned "
                              "grip width was narrower than that the jaws are on a wider "
                              "part than planned — check the grasp height",
                              (FINGER_OPEN - gw_m) * 1000, gw_m * 2000)
        else:
            data.setdefault("reason", "MoveIt planning or execution failed")
            rospy.logwarn("[arm] Grasp failed (gripper=%.1fmm  plan_ok=%s): %s",
                          gw_m * 1000, ok, data["reason"])
        return success, data

    def _approach_pregrasp(self, x, y, z, ori):
        """Collision-aware Cartesian move to the pre-grasp offset pose, with the
        target + neighbours still obstacles — the arm stops CLEAR at the standoff.
        Returns True if a collision-free path was found & executed; on low
        coverage it logs and returns False."""
        p = Pose()
        p.position.x, p.position.y, p.position.z = x, y, z
        p.orientation = ori
        frac = 0.0
        for attempt in (1, 2):
            plan, frac = self.move_group.compute_cartesian_path([p], 0.005, True)
            if frac >= 0.80:
                plan = self.move_group.retime_trajectory(
                    self.robot.get_current_state(), plan, 0.2, 0.2)
                dur = self.arm.publish_trajectory(plan)
                rospy.loginfo("[arm] Pre-grasp: collision-aware move to offset "
                              "(cov=%.0f%%, %.1fs, attempt %d)", frac * 100, dur, attempt)
                rospy.sleep(dur + 0.4)
                return True
            # Re-settle to zero the low-gain joints' position error and retry once
            # (collision-aware Cartesian coverage is start-state sensitive, same as
            # the descent's proven re-seat retry).
            self.arm.hold(0.6)
        rospy.logwarn("[arm] Pre-grasp collision-aware path low cov (%.0f%%) — "
                      "standoff may be blocked; proceeding to descent", frac * 100)
        return False

    def _explain_short_leg(self, legs, cart_plan, fraction):
        """A short straight-in leg with collision checking OFF is an IK failure
        along the path, not an obstacle — so log what the arm looked like and how
        the same leg fares in variants, to say WHICH ingredient it is (the start
        state, the raise, or the height) the next time it happens."""
        try:
            now = [round(v, 3) for v in self.move_group.get_current_joint_values()]
            pts = cart_plan.joint_trajectory.points
            end = [round(v, 3) for v in pts[-1].positions] if pts else None
            last = legs[-1]
            _, f_on = self.move_group.compute_cartesian_path(legs, 0.005, True)
            rospy.logwarn("[arm] Short leg (%.0f%%, ends at z=%.3f): joints now %s, partial "
                          "path ends at %s (roll joints near +-pi = wrist singularity), "
                          "same leg with collisions ON: %.0f%%", fraction * 100,
                          last.position.z, now, end, f_on * 100)
        except Exception as e:      # diagnostics must never break a grasp
            rospy.logwarn("[arm] could not explain the short leg: %s", e)

    def _wp_for(self, ori):
        """Waypoint factory bound to a fixed EE orientation."""
        def _wp(px, py, pz):
            p = Pose()
            p.position.x, p.position.y, p.position.z = px, py, pz
            p.orientation = ori
            return p
        return _wp

    def _retreat_up(self, x, y, z, ori):
        """Lift the gripper straight up to z, holding orientation.
        avoid_collisions=False: any attached-object box may overlap the table box
        at grasp height, but moving directly away from the table cannot collide.
        Direct publish (not execute()) so forearm_roll lag can't abort the lift.
        """
        plan, frac = self.move_group.compute_cartesian_path(
            [self._wp_for(ori)(x, y, z)], 0.01, False)
        if frac > 0.5:
            plan = self.move_group.retime_trajectory(
                self.robot.get_current_state(), plan,
                velocity_scaling_factor=0.15, acceleration_scaling_factor=0.15)
            rospy.sleep(self.arm.publish_trajectory(plan) + 0.4)
            return True
        rospy.logwarn("[arm] Retreat path only %.0f%% — holding position", frac * 100)
        return False

    def _retract_clear(self, min_lift=0.12):
        """Get the arm out of the way WITHOUT sweeping the object.

        A joint-space swing straight to CAMERA_CLEAR_POSE rotates the waist back
        to zero and reconfigures shoulder/elbow, dragging the gripper sideways
        through the space the object occupies. MEASURED on aborted grasps: the
        target was tipped 90° over, or dragged 251 mm across the table — which
        defeats the point of aborting, since the whole reason to refuse the close
        is to leave the object recoverable for a retry.

        Lifting straight up first puts the gripper clear above the object before
        any reconfiguration begins.
        """
        try:
            p = self.move_group.get_current_pose().pose
            self._retreat_up(p.position.x, p.position.y,
                             max(p.position.z + min_lift, self.table_surface_z + 0.25),
                             p.orientation)
        except Exception as e:
            rospy.logwarn("[arm] Clear-lift before retract failed: %s", e)
        self.arm.send_joints(CAMERA_CLEAR_POSE, move_time=2.5)

    def _do_pregrasp(self, plan, waist_angle, ready_pose):
        """Approach the standoff and VERIFY the arm got there.

        The arm is already at GRASP_READY_POSE (waist facing target). Opens the
        gripper, moves collision-aware to the standoff IN FRONT OF the object,
        then measures the achieved EE pose against the commanded one and gates
        on it. On failure it adapts and re-plans once — Cartesian coverage is
        start-state sensitive.

        The standoff sits back along the approach axis AT THE GRASP HEIGHT, so
        the arm never passes over the object and the grasp that follows is a
        straight horizontal line. The target stays a collision obstacle
        throughout: _do_descent removes it.
        """
        gx, gy, gz = plan["grasp_point"]
        x, y = gx, gy
        grasp_xyz = [gx, gy, gz]
        grasp_quat = list(plan["grasp_orientation"])   # only defines the approach axis
        grasp_z = gz
        offset = plan["standoff_offset"]
        pre_xyz, _ = gg.pregrasp_pose(grasp_xyz, grasp_quat, offset)
        # Remembered for place(): putting the object down at the height it was
        # grasped at sets its base gently back onto the table.
        self._last_grasp_z = grasp_z

        rospy.loginfo("[arm] Horizontal grasp: grasp=(%.3f,%.3f,%.3f)  "
                      "standoff=(%.3f,%.3f,%.3f) offset=%.3f  [%s]",
                      x, y, grasp_z, pre_xyz[0], pre_xyz[1], pre_xyz[2],
                      offset, plan["reason"])

        self.arm.hold(0.8)

        # Approach orientation: read from the current (grasp_ready) state —
        # level gripper, guarantees Cartesian IK feasibility.
        descent_ori = self.move_group.get_current_pose().pose.orientation

        rospy.loginfo("[arm] Gripper open=%.1fmm", plan["gripper_open"] * 1000)
        self.gripper.send(plan["gripper_open"])
        rospy.sleep(0.3)   # let arm settle after gripper motion before planning

        verdict, reason, meas = None, "", None
        approach_ok = False
        for attempt in (1, 2):
            approach_ok = self._approach_pregrasp(*pre_xyz, descent_ori)
            meas = self.diag.pregrasp(*pre_xyz, descent_ori)
            verdict, reason = gg.pregrasp_verdict(
                meas, live_param("pregrasp_pos_tol_mm", PREGRASP_POS_TOL_MM),
                live_param("pregrasp_ori_tol_deg", PREGRASP_ORI_TOL_DEG))
            if verdict:
                rospy.loginfo("[arm] Pre-grasp VERIFIED: %s (attempt %d)", reason, attempt)
                break
            rospy.logwarn("[arm] Pre-grasp NOT verified: %s (attempt %d, "
                          "approach_coverage_ok=%s)", reason, attempt, approach_ok)
            if attempt == 1:
                if approach_ok:
                    rospy.loginfo("[arm] Near miss — refining from the current "
                                  "pose (no re-seat)")
                else:
                    floor = plan["standoff_offset_min"]
                    new_offset = max(floor, offset - 0.02)
                    if new_offset < offset - 1e-6:
                        offset = new_offset
                        pre_xyz, _ = gg.pregrasp_pose(grasp_xyz, grasp_quat, offset)
                        rospy.logwarn("[arm] Standoff unreachable from here — "
                                      "closing the offset to %.3fm (floor %.3fm, "
                                      "still outside the padded target box) and "
                                      "re-seating", offset, floor)
                    else:
                        rospy.logwarn("[arm] Standoff unreachable and already at "
                                      "the minimum safe offset (%.3fm) — re-seating "
                                      "and retrying unchanged", offset)
                    self.arm.send_joints(ready_pose, move_time=2.5)
                    self.arm.hold(0.8)
                    descent_ori = self.move_group.get_current_pose().pose.orientation

        info = {"phase": "pregrasp", "verified": bool(verdict), "reason": reason,
                "err_norm_mm": round(float(meas["err_norm_mm"]), 1) if meas else None,
                "ori_err_deg": round(float(meas["ori_err_deg"]), 1) if meas else None}
        if not verdict:
            self._pending_grasp = None
            return False, info

        # Hold at the verified standoff; grasp() picks up from here.
        self._pending_grasp = {
            "x": x, "y": y, "grasp_z": grasp_z,
            "grasp_quat": grasp_quat, "offset": offset, "pre_xyz": pre_xyz,
            "descent_ori": descent_ori, "waist_angle": waist_angle,
            "ready_pose": ready_pose,
            "stamp": rospy.Time.now().to_sec(),
        }
        return True, info

    def _do_descent(self, plan, obj_base=None):
        """Move straight in onto the object, VERIFY the jaws straddle it, close.

        Requires a verified pre-grasp: the arm is holding at the standoff,
        level and facing the object, and the scene still has the target as an
        obstacle. This leg is a SINGLE straight horizontal line: the standoff
        already sits at the grasp height. (It used to be across-then-down,
        which is why the command protocol still calls it "descent".)

        obj_base: the object's current position in base_link (the caller's
        odometry reprojection). The contact check measures against it; it needs
        neither the camera (the arm occludes it here) nor ground truth, so the
        check behaves identically on hardware. None -> unverified, which passes.

        `plan` is the caller's CURRENT plan. If the object's dimensions were
        refined since the pre-grasp, its grasp_point z differs and the straight
        leg simply ends at the corrected height (a slight slope) instead of
        needing another verification round trip.
        """
        pg = self._pending_grasp
        x, y = pg["x"], pg["y"]
        waist_angle = pg["waist_angle"]
        descent_ori = pg["descent_ori"]

        # ── ADJUSTMENT CHECKPOINT: pre-grasp -> grasp ────────────────────────
        grasp_z = plan["grasp_point"][2]
        if abs(grasp_z - pg["grasp_z"]) > 0.005:
            rospy.loginfo("[arm] Pre-grasp→grasp ADJUSTMENT: grasp_z %.3f->%.3f "
                          "(dims recalibrated since pre-grasp)", pg["grasp_z"], grasp_z)
        self._last_grasp_z = grasp_z   # so place() descends to the ADJUSTED height
        retreat_z = grasp_z + 0.15
        _wp = self._wp_for(descent_ori)

        # ── GATE 2a — BEFORE any motion: is the object still where we aimed? ──
        # Catching a mismatch here, rather than after moving in, is what makes an
        # abort genuinely clean: the jaws never come around the object, so they
        # cannot brush it — and gazebo_grasp_fix welds on first two-sided contact
        # with NO close command, so a brushed object would be carried away by the
        # retreat (measured: lifted to z+0.15 and dragged 254 mm before dropping).
        #
        # A moderate mismatch is not a reason to stop at the standoff: the base
        # creeps while the arm holds these poses, so the object genuinely shifts
        # in base_link between planning and now (the odometry reference is
        # world-fixed, so it sees that; the commanded point is not). Within
        # REAIM_MAX_MM the leg aims at where the object is now instead of
        # refusing; only a larger disagreement stops here. Gate 2b below still
        # checks the real finger positions before the jaws close.
        miss_tol = live_param("contact_miss_tol_mm", CONTACT_MISS_TOL_MM)
        if obj_base is not None:
            (x, y), verdict, drift_mm = gg.descent_aim(
                (x, y), (obj_base[0], obj_base[1]),
                live_param("reaim_xy_tol_mm", gg.REAIM_XY_TOL_MM),
                live_param("reaim_max_mm", gg.REAIM_MAX_MM))
            if verdict == "refused":
                reason = (f"object has moved {drift_mm:.0f}mm from the commanded "
                          f"grasp point (re-aim limit "
                          f"{live_param('reaim_max_mm', gg.REAIM_MAX_MM):.0f}mm) "
                          f"— not descending")
                rospy.logwarn("[arm] Contact NOT verified pre-descent: %s", reason)
                return False, {"phase": "descent", "verified": False,
                               "reason": reason, "drift_mm": round(drift_mm, 1),
                               "descended": False}
            if verdict == "reaimed":
                rospy.logwarn("[arm] Object is %.0fmm from the commanded grasp point "
                              "(base creep) — re-aiming the straight-in leg to "
                              "(%.3f, %.3f)", drift_mm, x, y)

        # Height, by the same reasoning: the base tilts nose-down as the arm
        # extends, which lifts the object relative to the arm as well as moving
        # it back (one rigid rotation). Measured: +31 mm, which left the pad
        # centre ~167 mm up a bottle planned at 198 mm — on its 58-65 mm shoulder.
        # The leg goes from the standoff (old height) straight to the re-aimed
        # point as ONE line, a slight slope. Raising at the standoff first and then
        # advancing looked cleaner but failed live (73-77% coverage): at the raised
        # pose the wrist angle is ~0, a wrist singularity where forearm_roll and
        # wrist_rotate line up, so IK swings both to +-2.85 rad and the path stops
        # at the joint limit. The direct line to the same point got 100%. The jaws
        # only reach the object in the last ~20 mm, where the line is within a few
        # mm of the final height.
        if obj_base is not None:
            aim_z, zverdict, dz_mm = gg.descent_height(
                grasp_z, obj_base[2], live_param("reaim_z_tol_mm", gg.REAIM_Z_TOL_MM),
                live_param("reaim_z_max_mm", gg.REAIM_Z_MAX_MM))
            if zverdict == "refused":
                reason = (f"object height has shifted {dz_mm:+.0f}mm from the planned "
                          f"grasp height (limit "
                          f"{live_param('reaim_z_max_mm', gg.REAIM_Z_MAX_MM):.0f}mm) "
                          f"— not descending")
                rospy.logwarn("[arm] Contact NOT verified pre-descent: %s", reason)
                return False, {"phase": "descent", "verified": False, "reason": reason,
                               "dz_mm": round(dz_mm, 1), "descended": False}
            if zverdict == "reaimed":
                rospy.logwarn("[arm] Object is %+.0fmm in height from the planned grasp "
                              "point (base tilt) — raising the jaws %.3f -> %.3f",
                              dz_mm, grasp_z, aim_z)
                grasp_z = aim_z
                retreat_z = grasp_z + 0.15
                self._last_grasp_z = grasp_z
            obj_base = [obj_base[0], obj_base[1], grasp_z]

        self._last_grasp_xy = (x, y)   # where the jaws really close (place() reuses it)
        self.scene.remove_target()     # the target becomes the grasp target
        # Re-settle at the standoff so the leg plans from a stable start state.
        self.arm.hold(0.6)

        # The grasp leg — ONE straight horizontal waypoint. Computed by MoveIt
        # (for the IK along the path) but EXECUTED by direct publish, not
        # move_group.execute() (see ArmActuator.publish_trajectory). Retimed to
        # 15% first: compute_cartesian_path ignores the velocity scaling factor.
        #
        # avoid_collisions=False: this leg deliberately drives the OPEN jaws
        # around the object, so a collision check could only reject it — and the
        # table box's position is only estimated, so with the check on, identical
        # commands planned 100%/53%/28% depending on tiny start-state differences
        # near the box's front corner. Safe by construction: it starts from a pose
        # verified to within PREGRASP_POS_TOL_MM, is only the standoff offset
        # long, and grasp_z is floor-clamped well above the table.
        cart_plan, fraction = None, 0.0
        for attempt in (1, 2):
            legs = [_wp(x, y, grasp_z)]
            cart_plan, fraction = self.move_group.compute_cartesian_path(
                legs, 0.005, False)
            rospy.loginfo("[arm] Grasp path coverage: %.0f%% (attempt %d)",
                          fraction * 100, attempt)
            if fraction >= 0.80:
                break
            self._explain_short_leg(legs, cart_plan, fraction)
            if attempt == 2:
                break
            # IK failed partway from this start state. Re-seat, then RE-ESTABLISH
            # THE STANDOFF before retrying — a straight line from the retracted
            # ready pose would be a long UNVERIFIED move driving the jaws into the
            # object, the exact failure the pre-grasp gate exists to prevent.
            rospy.logwarn("[arm] Grasp path short — re-seating and re-taking "
                          "the standoff before retrying")
            self.arm.send_joints(pg.get("ready_pose")
                                 or ([waist_angle] + list(GRASP_READY_POSE[1:])),
                                 move_time=2.5)
            self.arm.hold(0.8)
            descent_ori = self.move_group.get_current_pose().pose.orientation
            _wp = self._wp_for(descent_ori)   # rebind: waypoints carry the orientation
            if not self._approach_pregrasp(*pg["pre_xyz"], descent_ori):
                rospy.logwarn("[arm] Could not re-take the standoff — aborting "
                              "rather than driving in from the ready pose")
                return False, {"phase": "descent", "verified": False,
                               "reason": "lost the verified standoff and could "
                                         "not re-establish it",
                               "descended": False}
            self.arm.hold(0.8)
        if fraction < 0.80:
            rospy.logwarn("[arm] Grasp path too short (%.0f%%) — aborting",
                          fraction * 100)
            return False, {"phase": "descent", "verified": False,
                           "reason": f"descent path only {fraction*100:.0f}% reachable"}

        cart_plan = self.move_group.retime_trajectory(
            self.robot.get_current_state(), cart_plan,
            velocity_scaling_factor=0.15, acceleration_scaling_factor=0.15)
        descent_dur = self.arm.publish_trajectory(cart_plan)
        rospy.loginfo("[arm] Descent executing directly (%.1fs, no action abort)",
                      descent_dur)
        rospy.sleep(descent_dur + 0.8)   # let the trajectory finish playing

        # Actively HOLD the arm at its own measured pose before gripping — the
        # key stabilizer (see ArmActuator.hold).
        self.arm.hold(0.8)

        # GATE: with the jaws OPEN around the object at grasp height, check that
        # the object is actually between them BEFORE closing — the only moment
        # the straddle question is meaningful, and closing on a known off-centre
        # object is the failure that shoves the target across the table.
        contact = self.diag.contact(x, y, grasp_z, obj_base)
        c_ok, c_reason = gg.contact_verdict(contact, miss_tol)
        info = {"phase": "descent", "verified": bool(c_ok), "reason": c_reason,
                "descended": True}
        if contact:
            info.update({"along_jaw_mm": round(contact["along_jaw_mm"], 1),
                         "in_plane_miss_mm": round(contact["in_plane_miss_mm"], 1),
                         "straddled": contact["straddled"]})
        if not c_ok:
            rospy.logwarn("[arm] Contact NOT verified: %s — retreating WITHOUT "
                          "closing", c_reason)
            # The jaws are already around the object and may have BRUSHED it (74 mm
            # open vs a 60 mm box leaves ~7 mm a side), and gazebo_grasp_fix welds
            # on first two-sided contact without any close command — so a plain
            # lift would carry the object off. Cycling the fingers past the
            # plugin's release tolerance drops anything that attached. Safe here:
            # this branch fires only when the object is NOT between the jaws.
            self.gripper.send(FINGER_OPEN - 0.012, move_time=0.8)
            self.gripper.send(FINGER_OPEN, move_time=0.8)
            self._retreat_up(x, y, retreat_z, descent_ori)
            return False, info
        rospy.loginfo("[arm] Contact VERIFIED: %s", c_reason)

        # Close so the jaws contact the object and gazebo_grasp_fix attaches it —
        # full close for a tapered object, adaptive for a uniform prism. Slowly,
        # so contact ramps up without jolting the arm. The commanded target is
        # remembered: "held" is then a comparison against what was commanded
        # (fingers stop SHORT of it on an object, reach it on air).
        self.last_close_target = plan["gripper_close"]
        rospy.loginfo("[arm] Gripper close target=%.1fmm (%s)",
                      plan["gripper_close"] * 1000,
                      "unmeasured width -> full close" if plan["full_close"]
                      else "a squeeze past the contact")
        self.gripper.send(plan["gripper_close"], move_time=2.0)
        rospy.sleep(0.6)   # let the grip settle and grasp-fix attach

        # Re-hold at the new pose: the grasped object shifts the load, so
        # re-command the measured joints to absorb it before lifting.
        self.arm.hold(0.6)

        # Attach the object in the planning scene so the retreat avoids the table.
        # (W, L, H), the axis order this box has always been attached with.
        sx, sy, sz = plan["object_size"]
        self.scene.attach(x, y, grasp_z, (sy, sx, sz))

        # "Lifted a little" IS this straight-up Cartesian retreat (grasp_z + 0.15).
        # Deliberately NOT a joint-space swing to LIFT_POSE: that reconfigures
        # shoulder/elbow/wrist while carrying the welded object, tilting and
        # shaking it. Straight up keeps the orientation constant, so the object
        # rides up level; the settle-hold stops the controllers fighting gravity.
        self._retreat_up(x, y, retreat_z, descent_ori)
        self.arm.hold(1.2)

        self._pending_grasp = None   # consumed
        return True, info
