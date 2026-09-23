"""Putting the base at the grasp standoff in front of the target. Pure policy.

Ported near-verbatim from rework-attempt-1's src/locobot_prog/navigation/
approach.py, itself "moved from the VLM agent" — fine() is the single-shot
GO/VERIFY/DECISION positioning loop that used to live in the retired
vlm_agent_node.py's get_object() (COARSE_TOL/REOPEN_TOL/DIST_TOL, the
TOO_CLOSE back-up branch with the sign bug independently found and fixed
there this session — see that fix's commit message; confirmed here at
"robot.drive(dist - standoff)", already correct). coarse()/correct() are
additional capabilities used by
navigation_node's COARSE/CORRECT modes, not yet exercised by anything —
matching the earlier plan's navigate_to split, deferred to a later step.

No rospy import at all — fully unit-testable without ROS. The robot is
reached through a small interface, so the policy runs the same against the
real base (navigation_node) or a simulated one:

  robot.measure()          -> (x, y) of the target in base_link, or raises
                              PositioningFailed (target lost / path blocked)
  robot.coarse_target()    -> (x, y): map reprojection, else the scene's target
  robot.map_target()       -> (x, y) or None
  robot.front_blocked()    -> bool (laser)
  robot.rotate(angle, w)   closed-loop turn
  robot.drive_straight(m, v)
  robot.face(angle)        a positioning turn (slow for small angles), then settle
  robot.drive(m)           one bounded positioning drive, then settle
  robot.log(fmt, *args)

Single-shot on purpose: an earlier convergence loop re-rotated on sub-noise
bearing jitter every iteration. Here it is measure -> move -> verify once ->
decide.
"""
import math

COARSE_TOL = math.radians(1.5)    # face the target when off by more (above bearing noise)
REOPEN_TOL = math.radians(6.0)    # only a real veer earns the one verify-step heading trim
DIST_TOL = 0.03                   # distance dead-band around the standoff
TOO_CLOSE = 0.40                  # closer: the arm's swing clips the object
TOO_FAR = 0.50                    # farther: the arm node aborts at 0.52 m
TABLE_ROBOT_MARGIN = 0.12         # coarse drives keep this far from the table edge
COARSE_FACE_TOL = math.radians(3.0)
COARSE_TURN_SPEED = 0.30          # rad/s
COARSE_DRIVE_SPEED = 0.08         # m/s


class PositioningFailed(Exception):

    def __init__(self, message, zone="", summary="", too_close=False):
        super().__init__(message)
        self.message, self.zone, self.summary, self.too_close = message, zone, summary, too_close


def coarse(robot, standoff, table_near_x):
    """Face the target, then drive most of the way to the standoff, never past
    TABLE_ROBOT_MARGIN from the table edge. Rotation and translation are kept
    separate. Raises PositioningFailed."""
    tx, ty = robot.coarse_target()
    dist, alpha = math.hypot(tx, ty), math.atan2(ty, tx)
    drive_m = max(0.0, min(dist - standoff, table_near_x - TABLE_ROBOT_MARGIN))
    robot.log("Navigate: target=(%.2f, %.2f)  dist=%.2fm  alpha=%.1f deg  drive=%.2fm",
              tx, ty, dist, math.degrees(alpha), drive_m)

    if drive_m < 0.03 and abs(alpha) <= COARSE_FACE_TOL:
        robot.log("Already within grasp range — skipping drive")
        return
    if robot.front_blocked():
        raise PositioningFailed("Obstacle ahead — cannot drive forward.")

    if abs(alpha) > COARSE_FACE_TOL:
        robot.log("Rotate: angle=%.1f°", math.degrees(alpha))
        robot.rotate(alpha, COARSE_TURN_SPEED)
        reproj = robot.map_target()        # re-read through TF after turning
        if reproj:
            tx, ty = reproj
            dist = math.hypot(tx, ty)
            drive_m = max(0.0, min(dist - standoff, table_near_x - TABLE_ROBOT_MARGIN))

    if drive_m > 0.03:
        robot.log("Drive straight: %.2fm", drive_m)
        robot.drive_straight(drive_m, COARSE_DRIVE_SPEED)


def fine(robot, standoff):
    """GO (face once, drive once) -> VERIFY (one heading trim only on a genuine
    veer, one distance correction) -> DECISION. Returns (dist, angle, tx, ty) of
    the final measurement. Raises PositioningFailed."""
    dist_min, dist_max = standoff - DIST_TOL, standoff + DIST_TOL

    # STEP 1 — GO
    tx, ty = robot.measure()
    angle = math.atan2(ty, tx)
    if abs(angle) > COARSE_TOL:
        robot.log("Position GO: face %.1f°", math.degrees(angle))
        robot.face(angle)
        tx, ty = robot.measure()
    dist = math.hypot(tx, ty)
    if not (dist_min <= dist <= dist_max):
        robot.log("Position GO: drive %+.2fm (%.2f→%.2fm)", standoff - dist, dist, standoff)
        robot.drive(dist - standoff)

    # STEP 2 — VERIFY (one pass)
    tx, ty = robot.measure()
    dist, angle = math.hypot(tx, ty), math.atan2(ty, tx)
    if abs(angle) > REOPEN_TOL:
        robot.log("Position VERIFY: heading drifted %.1f° — one trim", math.degrees(angle))
        robot.face(angle)
        tx, ty = robot.measure()
        dist, angle = math.hypot(tx, ty), math.atan2(ty, tx)
    if not (dist_min <= dist <= dist_max):
        robot.drive(dist - standoff)
        tx, ty = robot.measure()
        dist, angle = math.hypot(tx, ty), math.atan2(ty, tx)

    # DECISION — proceed if the standoff is in the arm's safe window; the waist
    # pre-rotation absorbs the heading residual.
    if dist < TOO_CLOSE:
        robot.log("Positioned TOO CLOSE (%.2fm) — backing up", dist)
        for _ in range(2):
            robot.drive(dist - standoff)         # negative: away from the target
            tx, ty = robot.measure()
            dist, angle = math.hypot(tx, ty), math.atan2(ty, tx)
            robot.log("After back-up: dist=%.2fm", dist)
            if dist >= TOO_CLOSE:
                break
        if dist < TOO_CLOSE:
            raise PositioningFailed(
                f"Still too close to the target ({dist:.2f}m) after backing up — the arm "
                f"cannot reach it without knocking it over. Move the base back and retry.",
                too_close=True)
    elif dist <= TOO_FAR:
        robot.log("Positioned (single-shot): dist=%.2fm heading=%.1f° — waist compensates "
                  "the rest", dist, math.degrees(angle))
    else:
        raise PositioningFailed(f"Target too far after positioning ({dist:.2f}m).")
    return dist, angle, tx, ty


def correct(robot, standoff):
    """One bounded correction from a fresh measurement — used before retrying a
    failed pre-grasp. Returns (dist, angle, tx, ty) re-measured afterwards."""
    tx, ty = robot.measure()
    d, a = math.hypot(tx, ty), math.atan2(ty, tx)
    if abs(a) > COARSE_TOL:
        robot.face(a)
    if not (standoff - DIST_TOL <= d <= standoff + DIST_TOL):
        robot.drive(d - standoff)
    tx, ty = robot.measure()
    return math.hypot(tx, ty), math.atan2(ty, tx), tx, ty
