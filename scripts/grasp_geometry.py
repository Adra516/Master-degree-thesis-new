import math

import numpy as np
import tf.transformations as tft


# ── Grasp-height model — THE single source of truth ──────────────────────────
# Grip at a fraction of the object's own height instead of a near-constant band.
# This formula (and its GRIP_Z_CEILING) is the BLIND fallback: it has no real
# measurement to justify picking anywhere but a conservative guess. The real,
# shape-aware search that isn't bound by this ceiling lives in grasp_planner.py
# — see its module docstring for why it searches the object's whole physical
# range instead of trusting this formula's single guessed height.
GRIP_FRAC       = 0.50   # grip point as a fraction of object height (mid-body)
GRIP_Z_FLOOR    = 0.06   # lowest height above the table the jaws safely reach
GRIP_TOP_MARGIN = 0.018  # keep the jaws this far below the object's top edge
GRIP_Z_CEILING  = 0.085
# Height used when dims are unknown: middle of the proven band.
GRIP_Z_UNKNOWN  = 0.08

# Per-finger joint positions (m); right_finger mirrors left_finger. The URDF
# limits left_finger to [0.015, 0.037]. FINGER_CLOSE is deliberately below that
# minimum: the controller clamps it, so "close" always means fully closed.
FINGER_OPEN  = 0.037
FINGER_CLOSE = 0.010
# The pads' inner faces sit this far INSIDE each finger's joint position: the
# finger mesh's contact plate is at y=-9 mm in its own link frame. So the real
# gap between the pads is 2*(joint - PAD_INSET): 12-56 mm, NOT 30-74 mm.
# Measured in Gazebo by closing on floating blocks: 40 mm stopped the fingers at
# 28.9 mm, 30 mm at 24.0 mm (this model: 29.0/24.0; joint-equals-pad: 20/15).
PAD_INSET = 0.009


def finger_pos_for_width(width):
    """Finger joint position at which the pads just touch an object this wide."""
    return width / 2.0 + PAD_INSET


def jaw_width(finger_pos):
    """Real pad-to-pad gap for a finger joint position."""
    return 2.0 * (finger_pos - PAD_INSET)


# Mirrors grasp_planner.SIDE_CLEARANCE (duplicated, not imported: grasp_planner
# already imports this module, so the reverse would be circular) — the minimum
# room per side between the open jaws and the object for an approach that
# doesn't clip it.
JAW_CLEARANCE = 0.002
# Assumed object size [H, W, L] when nothing was measured — only ever used to
# size collision boxes, never to choose a grip.
UNKNOWN_OBJECT_DIMS = (0.15, 0.06, 0.06)


class ObjectTooWideForJaws(ValueError):
    """The measured width at the grasp axis exceeds what the jaws can ever open
    to, even with no clearance to spare.

    Raised instead of silently committing to a grip the jaws cannot physically
    close around. baseline_plan() has no per-band search to fall back on the
    way grasp_planner.plan_from_points does — it is one guessed point, at the
    object's own mid-height — so a target too wide there has no narrower spot
    for baseline to try instead; the caller is left to keep whatever plan it
    already had, or report the failure, rather than approach a point the jaws
    were always going to fail to straddle."""


class ObjectTooShortForSideGrasp(ValueError):
    """No valid horizontal (side) grip height exists: the jaws' lowest safe
    height is already above the object's top edge.

    Deliberately raised instead of silently clamping — a clamp would command a
    grip in empty air above the object and report it as a normal grasp.  The
    top-down approach axis that used to cover short objects was removed (see the
    module header), so this case has no fallback and must be surfaced."""


def grasp_height_above_table(dims_hwl,
                             grip_frac=GRIP_FRAC,
                             z_floor=GRIP_Z_FLOOR,
                             top_margin=GRIP_TOP_MARGIN,
                             z_ceiling=GRIP_Z_CEILING):
    """Height ABOVE THE TABLE at which the jaws should close, from object height:

        z = clip(grip_frac * H, z_floor, min(H - top_margin, z_ceiling))

    Floored by what the arm can safely reach; ceilinged by BOTH the object's own
    rim (so the jaws never land on the top edge) and the measured GRIP_Z_CEILING.

    THE single source of truth for grip height: the motion node executes it via
    _compute_grasp_z, and the agent's reachability sweep consumes it via
    _horiz_grasp_z, so the base standoff is chosen for the height the arm will
    actually use.  Never re-derive this formula elsewhere — if the two diverge,
    the agent optimises the standoff for one grip height while the arm executes
    another.

    dims_hwl : [H, W, L] metres, or None when unknown.
    Returns  : height above the table (m).
    Raises   : ObjectTooShortForSideGrasp when no valid grip height exists.
    """
    if dims_hwl is None:
        return GRIP_Z_UNKNOWN
    h = float(dims_hwl[0])
    upper = min(h - top_margin, z_ceiling)
    if upper < z_floor:
        raise ObjectTooShortForSideGrasp(
            f"object H={h:.3f}m: top minus margin ({h - top_margin:.3f}m) is below "
            f"the minimum grip height ({z_floor:.3f}m) — no valid side grasp")
    return float(np.clip(grip_frac * h, z_floor, upper))


def _horiz_grasp_z(table_z, dims_hwl):
    """Absolute grasp Z for a horizontal grasp: table height + the shared
    object-relative offset.  Thin wrapper so there is ONE height formula."""
    return table_z + grasp_height_above_table(dims_hwl)


TAPER_ASPECT_RATIO = 2.5


def is_tapered(dims_hwl, grasp_axis):
    """True when the object's width at the grip height cannot be trusted from the
    bbox, so the jaws must close FULLY to reach whatever cross-section is there.

    A slender, tall object (bottle) narrows above its base, so an adaptive close
    computed from the wide bbox stops the fingers in mid-air.  A squat uniform
    prism (can, box) has the same width all the way up, so the measured width is
    trustworthy and an adaptive close lets the jaws self-centre instead of
    slamming shut and shoving the object sideways.

    Unknown dims -> True (conservative: full close, the pre-existing behaviour).
    """
    if dims_hwl is None:
        return True
    h = float(dims_hwl[0])
    narrow = float(dims_hwl[1] if grasp_axis == "along_x" else dims_hwl[2])
    if narrow <= 1e-6:
        return True
    return (h / narrow) > TAPER_ASPECT_RATIO


# ── Pre-grasp standoff (HORIZONTAL) ──────────────────────────────────────────
# The pre-grasp pose sits BACK from the grasp point along the approach axis, at
# the SAME height the jaws will close at — the arm stops short of the object and
# then moves straight in.  It never passes over the object at any point.
#
# This replaced an earlier "hover above the object, then descend vertically"
# design.  That design needed a whole layer of machinery just to be safe — a
# measured collision-box clearance ceiling, a second high staging pose to keep
# the swing from sweeping over a tall object, and a separate conservative height
# estimator to avoid flying into the object's top.  Approaching from the side at
# grip height removes the reason all of it existed: there is no "top" to clear.
#
# Clearance now means horizontal room, not vertical: the standoff must sit
# outside the target's padded collision box along the approach axis.  A typical
# target's box half-extent is ~0.04-0.055 m (a 0.06-0.09 m footprint plus pad).
# MoveIt enforces the rest for free — the approach is planned collision-aware
# with the target still an obstacle, so a standoff that IS inside the box
# simply fails to plan; the caller's retry on failure shrinks the offset
# (moves closer), which only helps when the standoff was kinematically out of
# reach, not when a scene obstacle is blocking an otherwise-reachable one.
#
# MEASURED live 2026-10-01: at the old 0.08 m, ~30% of pre-grasp approaches hit
# this — every single one confirmed (same path re-planned with collisions OFF
# -> 100%) to be the target's OWN box clipped mid-swing, never a kinematics
# problem. compute_cartesian_path checks collision along the WHOLE interpolated
# path from wherever the arm currently is, not just the destination, so even a
# ~21 mm point-clearance margin at 0.08 m left room for the wrist/forearm to
# graze the box partway through, depending on the exact start pose. Widened to
# 0.10 m (~41 mm margin) — comfortably inside the already-measured reachability
# budget below (0.321 rad of joint-limit margin at 0.10 m vs 0.384 at 0.08 m).
TARGET_BOX_PAD = 0.01   # MUST match the motion node's _add_target_obstacle pad
PREGRASP_STANDOFF_OFFSET = 0.10   # metres back along the approach axis
# Air between the fingertips and the padded box face at the minimum offset.
# ee_gripper_link is FK-MEASURED to be the furthest-forward link of the whole
# gripper (nothing sticks out past it), so the offset is a fingertip clearance
# and this margin is the real gap, not an approximation.
PREGRASP_FACE_CLEAR = 0.010


def min_standoff_offset(dims_hwl):
    """Smallest standoff that still sits OUTSIDE the target's padded collision box.

    The retry in _do_pregrasp REDUCES the offset when the standoff is hard to
    reach, which is the opposite of the old vertical design's "raise to the
    ceiling" escape.  MEASURED (URDF FK + exact planar IK, level gripper, over
    the whole validated target window r=0.44-0.47 x z=table+0.060-0.085): the
    joint-limit margin falls off monotonically as the standoff retracts —

        offset  0.05   0.06   0.07   0.08   0.09   0.10   0.12   0.14
        margin  0.488  0.453  0.418  0.384  0.352  0.321  0.264  0.216  rad

    so moving FURTHER back folds the arm toward its elbow limit and makes an
    unreachable standoff worse, not better.  Moving IN is the direction that
    helps; this function is the floor that keeps doing so from ever putting the
    fingertips inside the box the pre-grasp is planned to avoid.

    The box is axis-aligned with a square footprint (yaw unknown), so the worst
    case along an arbitrary approach direction is its DIAGONAL half-extent.
    """
    if dims_hwl is None:
        return PREGRASP_STANDOFF_OFFSET
    half_foot = max(float(dims_hwl[1]), float(dims_hwl[2])) / 2.0 + TARGET_BOX_PAD
    return min(PREGRASP_STANDOFF_OFFSET,
               half_foot * math.sqrt(2.0) + PREGRASP_FACE_CLEAR)


# ── Verification verdicts for the two-phase grasp ────────────────────────────
# The grasp runs as two separately verified commands (execute_pregrasp →
# execute_descent).  These turn a raw measurement into a go/no-go decision.
# They live here, not in the motion node, so they stay importable without ROS
# and can be unit-tested against the real measured numbers.
#
# SHARED RULE — a MISSING measurement passes.  The gates refuse poses that were
# measured to be bad; they do not refuse poses that could not be measured at all
# (diagnostics disabled, TF stalled, no object position supplied).  Failing
# closed on a missing measurement would turn an instrumentation outage into a
# total grasp outage, which is strictly worse than the pre-gate behaviour.

def pregrasp_verdict(meas, pos_tol_mm, ori_tol_deg):
    """Did the manipulator actually reach the commanded pre-grasp pose?

    meas : the dict returned by the motion node's _diag_pregrasp (FK actual vs
           commanded), or None when unmeasured.
    Returns (ok, reason) — reason is human-readable and ends up in the result
    the VLM sees, so it names the phase and the magnitude.
    """
    if meas is None:
        return True, "unverified (no measurement)"
    err = float(meas.get("err_norm_mm", 0.0))
    ori = float(meas.get("ori_err_deg", 0.0))
    if err > pos_tol_mm:
        return False, (f"pre-grasp position off by {err:.0f}mm "
                       f"(tolerance {pos_tol_mm:.0f}mm)")
    if ori > ori_tol_deg:
        return False, (f"pre-grasp orientation off by {ori:.1f}° "
                       f"(tolerance {ori_tol_deg:.0f}°)")
    return True, f"ok ({err:.1f}mm, {ori:.1f}°)"


# How much of the free room each side of the object the arrival may use up. With a
# 56 mm jaw gap and a 33 mm neck there are 11.4 mm per side; a pad that lands on
# the neck interpenetrates it (kp=1e6 contact), and the depenetration burst is
# what throws the bottle and spins the weak forearm_roll. Measured 2026-10-13 on
# three bottle descents that arrived 10-25 mm off-centre: the old test only asked
# whether the object's centre lay between the fingertips (|along| < 28 mm), and
# passed -10.1 and +14.8 mm; both ended with the bottle thrown. Every descent
# within +-2.6 mm was fine. 3/4 of the room leaves 8.5 mm for the bottle and 4.0 mm
# for a 45 mm box, with the observed good arrivals (<= 2.6 mm) well inside both.
CONTACT_ROOM_FRACTION = 0.75


def contact_verdict(metrics, miss_tol_mm, width_mm=None, room_fraction=CONTACT_ROOM_FRACTION):
    """Are the OPEN jaws straddling the object, measured just before closing?

    metrics  : the dict returned by _diag_contact (along_jaw_mm, in_plane_miss_mm,
               straddled, and jaw_gap_mm), or None when no object position was
               available.
    width_mm : the object's width between the pads at the grip height, or None.
               With it (and jaw_gap_mm) the offset is judged against the room that
               is physically there on each side, not merely "between the tips".

    Refusing to close is the point: closing on a known off-centre object is the
    measured failure where the jaws shove the target across the table instead of
    gripping it (full close ejected a 60 mm box 221 mm at a perfect standoff).
    """
    if metrics is None:
        return True, "unverified (no object position)"
    along = float(metrics.get("along_jaw_mm", 0.0))
    miss  = float(metrics.get("in_plane_miss_mm", 0.0))
    if not metrics.get("straddled", True):
        return False, f"object not between the jaws (along-jaw {along:+.0f}mm)"
    if miss > miss_tol_mm:
        return False, (f"jaws {miss:.0f}mm off the object along the approach "
                       f"direction (tolerance {miss_tol_mm:.0f}mm)")
    gap = metrics.get("jaw_gap_mm")
    if width_mm is not None and gap is not None:
        room = (float(gap) - float(width_mm)) / 2.0
        if room > 0.0 and abs(along) > room_fraction * room:
            return False, (f"object is {along:+.0f}mm off the jaw centre but there is only "
                           f"{room:.0f}mm of room each side ({room_fraction * room:.0f}mm "
                           f"allowed) — a pad would land on it")
    return True, f"ok (along-jaw {along:+.1f}mm, miss {miss:.1f}mm)"


# Mirrors grasp_planner.SQUEEZE (duplicated, not imported: grasp_planner already
# imports this module) — the per-finger distance a planned close target sits
# inside the contact width it was derived from.
PLANNER_SQUEEZE = 0.004


def planned_width_mm(close_pos, full_close, squeeze=PLANNER_SQUEEZE):
    """The pad-to-pad width a plan's close target was derived from, in mm, or None
    when the plan is a full close (the width at the grip height is not trusted, so
    it cannot size anything). Inverts close = finger_pos_for_width(w) - squeeze.
    Measured on the logged bottle plans: the 33 mm band had a 36 mm contact width
    (pads land on the wider neck just below), close target 23.0 mm -> 36 mm here.
    Errs toward the WIDER object, i.e. toward less room, i.e. a stricter gate."""
    if full_close:
        return None
    return jaw_width(float(close_pos) + squeeze) * 1000.0


def centring_shift(metrics, miss_tol_mm, width_mm, room_fraction=CONTACT_ROOM_FRACTION,
                   margin_mm=-1.0):
    """(dx, dy) in metres that centres the OPEN jaws on the object, or None.

    Only when the ONLY thing wrong is the along-jaw offset: straddled, within the
    approach-miss tolerance, and offset beyond room_fraction of the room but still
    within the room (+1 mm: the near pad may just touch; sliding moves it AWAY, so
    that is still the safest move) — and the far pad stays clear. Needs the closing axis
    (metrics["axis_xy"]) and the object width."""
    if not metrics or width_mm is None or metrics.get("jaw_gap_mm") is None:
        return None
    if not metrics.get("straddled", True) or "axis_xy" not in metrics:
        return None
    if float(metrics.get("in_plane_miss_mm", 0.0)) > miss_tol_mm:
        return None
    along = float(metrics.get("along_jaw_mm", 0.0))
    room = (float(metrics["jaw_gap_mm"]) - float(width_mm)) / 2.0
    if room <= 0.0 or abs(along) <= room_fraction * room or abs(along) >= room - margin_mm:
        return None
    ux, uy = metrics["axis_xy"]
    return (along / 1000.0 * ux, along / 1000.0 * uy)


# ── Is the object actually still between the fingers? ────────────────────────
# Measured over 10 fresh-launch bottle picks (2026-10-13, ground truth): get_object
# reported success on 7 and NOT ONE held cleanly for 30 s. Every empty-gripper
# outcome among them had the same finger signature, so this is not a guess:
#   fingers at their commanded target (23.0 vs 22.6, 17.6 vs 22.6, 14.5 vs 22.4 mm)
#   — an object, however tight, stops them SHORT of a target set inside its width;
#   a finger reading past its own 37 mm limit (64 mm: the bottle thrown through
#   the jaws); fingers fully open (37.1 mm) with success still reported.
# A held bottle stalled 3.4-5.3 mm short of its target (28.1/24.3, 27.1/22.7,
# 28.5/23.2; tea box 30.5/27.1, can 31.9/27.3), and a grip that later let go
# showed the fingers snapping inward 10+ mm in 0.2 s.
#
# The margin below is set from the two clusters actually observed, with a gap
# between them: bottle-gone outcomes rested within 0.4-1.0 mm of the target
# (23.0/22.6, 24.4/23.4 in the user's own run, 23.5/22.6 with the bottle on the
# floor at return — and the right finger 0.9 mm PAST its target), every real
# hold at least 3.0 mm short on both fingers. 1.5 mm sits in that gap.
#
# KNOWN COST: a grip that is real but extremely marginal (both fingers < 1.5 mm
# short, i.e. under 3 mm of total interference with the plan) is called empty.
# Failing a barely-held object is the cheaper error here: the alternative, still
# measured today, is reporting "holding the bottle" with it on the floor.
FINGER_MIN_POS = 0.015        # a finger's own lower joint limit; commands below it clip
GRIP_OPEN_TRAVEL_MM = 3.0     # within this of fully open = closed on nothing
GRIP_TARGET_REACH_MM = 1.5    # within this of the commanded target = nothing stopped it
GRIP_SNAP_MM = 4.0            # drawn this far inward since the lift = the grip let go
GRIP_READING_SLACK_MM = 2.0   # above FINGER_OPEN by more than this = not a real reading


def grip_verdict(gw_m, close_target_m, finger_open_m=FINGER_OPEN, baseline_m=None, right_m=None):
    """Is the object still between the fingers? Returns (held, reason).

    gw_m           : the measured LEFT finger position now (m).
    close_target_m : the per-finger position the close was commanded to, or None
                     when no close happened (the target rule is then skipped).
    baseline_m     : the left finger position right after the lift, or None; with
                     it, a finger drawn inward by GRIP_SNAP_MM since is a grip that
                     let go even though the target was never reached.
    right_m        : the RIGHT finger as a positive magnitude, or None. The fingers
                     are independent joints and a lopsided real grip leaves one of
                     them near its target (measured: 24.7 vs a 23.6 target on the
                     left, 31.4 on the right, with the bottle genuinely gripped),
                     so "nothing stopped them" needs BOTH fingers at the target.

    "held" here means only "the fingers show something was stopping them" — it
    cannot prove the object is the right one, nor that it is not about to fall.
    """
    if gw_m > finger_open_m + GRIP_READING_SLACK_MM / 1000.0:
        return False, (f"finger reading {gw_m * 1000:.0f}mm is beyond the finger's own "
                       f"{finger_open_m * 1000:.0f}mm limit (the object was thrown through the jaws)")
    # "Fully open" is judged on the MEAN of both fingers when the right one is known:
    # an off-centre object stops one finger early. Measured 2026-10-14 on the can:
    # left 33.8 / right 29.3 mm (pad gap 45.1 = the can's width), can rising in the
    # jaws, yet the left finger alone read "fully open" and the jaws were opened on it.
    mean_m = gw_m if right_m is None else (gw_m + right_m) / 2.0
    if mean_m >= finger_open_m - GRIP_OPEN_TRAVEL_MM / 1000.0:
        return False, f"fingers are fully open ({gw_m * 1000:.1f}mm) — nothing is gripped"
    if close_target_m is not None:
        reach = max(float(close_target_m), FINGER_MIN_POS) + GRIP_TARGET_REACH_MM / 1000.0
        if gw_m <= reach and (right_m is None or right_m <= reach):
            return False, (f"fingers closed all the way to their {close_target_m * 1000:.1f}mm "
                           f"target ({gw_m * 1000:.1f}mm"
                           + (f" / {right_m * 1000:.1f}mm" if right_m is not None else "")
                           + ") — nothing stopped them")
    if baseline_m is not None and baseline_m - gw_m > GRIP_SNAP_MM / 1000.0:
        return False, (f"fingers snapped inward {(baseline_m - gw_m) * 1000:.0f}mm since the "
                       f"lift — the grip let go")
    return True, ""


# How far the odometry-held object position may sit from the commanded grasp point
# before the straight-in leg is REFUSED rather than re-aimed. The base creeps while
# the arm holds these poses (measured: +8 mm forward and 1.5° of yaw between
# camera_clear and grasp_ready, ~1.7 mm/s afterwards), so the object really does
# shift in base_link between planning and descent: 15-30 mm was seen every run.
REAIM_MAX_MM = 45.0
# Inside this the odometry reference and the commanded point are the same to within
# its own noise, so the commanded point stands. NOT the contact tolerance: that one
# (15 mm) is how far off the jaws may be and still close, and using it here left a
# 7-10 mm aim error uncorrected on every live pick, though the reference was right
# (its along-jaw reading matched ground truth to 0.4 mm).
REAIM_XY_TOL_MM = 3.0


def descent_aim(commanded_xy, object_xy, tol_mm, max_reaim_mm=REAIM_MAX_MM):
    """Where the straight-in leg should aim, before any motion.

    commanded_xy : the grasp point the pre-grasp was verified for (base_link).
    object_xy    : where the odometry-held reference says the object is NOW.
    Returns (aim_xy, verdict, drift_mm):
      "ok"      drift within tol_mm — aim at the commanded point as before.
      "reaimed" beyond tol_mm but within max_reaim_mm — the object shifted
                relative to the base (creep), so aim at where it is now. The
                post-descent contact check against the real finger positions
                still has to pass before the jaws close.
      "refused" further than max_reaim_mm — too large to trust either number.
    """
    dx = float(object_xy[0]) - float(commanded_xy[0])
    dy = float(object_xy[1]) - float(commanded_xy[1])
    drift = math.hypot(dx, dy) * 1000.0
    if drift <= tol_mm:
        return (float(commanded_xy[0]), float(commanded_xy[1])), "ok", drift
    if drift <= max_reaim_mm:
        return (float(object_xy[0]), float(object_xy[1])), "reaimed", drift
    return (float(commanded_xy[0]), float(commanded_xy[1])), "refused", drift


# The same tilt that shifts the object in x shifts it in HEIGHT. Measured with
# ground truth: the bottle's base rose 31 mm relative to the arm between planning
# and the grasp (3.8 deg nose-down at 0.47 m reach: dz = x*tilt = 31 mm, and
# dx = -z*tilt = -30 mm — the two shifts are one rigid rotation). Re-aiming only
# x/y left the pad centre 31 mm low: planned 198 mm up the bottle, actually
# ~167 mm, on the 58-65 mm shoulder, so the jaws closed at a 64 mm gap.
REAIM_Z_TOL_MM = 6.0       # arm sag is ~3 mm; inside this the verified height stands
REAIM_Z_MAX_MM = 45.0


def descent_height(commanded_z, object_z, tol_mm=REAIM_Z_TOL_MM, max_mm=REAIM_Z_MAX_MM):
    """The height the straight-in leg should aim at.

    commanded_z : the planned grasp height (base_link).
    object_z    : the same grasp point on the object NOW (odometry-held, in the
                  current tilted base frame), or None when unknown.
    Returns (aim_z, verdict, dz_mm): "ok" (keep commanded), "reaimed" (follow the
    object), or "refused" (shift too large to trust).
    """
    if object_z is None:
        return float(commanded_z), "ok", 0.0
    dz = (float(object_z) - float(commanded_z)) * 1000.0
    if abs(dz) <= tol_mm:
        return float(commanded_z), "ok", dz
    if abs(dz) <= max_mm:
        return float(object_z), "reaimed", dz
    return float(commanded_z), "refused", dz


def follow_reference(aim_xyz, ref0_xyz, ref_now_xyz, tol_xy_mm=2.0, tol_z_mm=3.0,
                     max_xy_mm=REAIM_MAX_MM, max_z_mm=REAIM_Z_MAX_MM):
    """Carry an aim point along with the object's odometry-held reference.

    The leg's aim is taken once, at the start of the descent, but the unwind of the
    wrist, the leg itself and the settle-holds follow it by 6-10 s, during which the
    base keeps moving relative to the world (measured: 11-30 mm on the three
    off-centre bottle arrivals; the aim and the contact gate both used the value
    from before). The reference is world-fixed, so re-reading it and applying the
    DIFFERENCE since the aim was taken follows the object without needing the
    camera, which the arm occludes.

    aim_xyz      : the point being aimed at, base_link, as taken at ref0.
    ref0_xyz     : the reference as read when the aim was taken (None = unknown).
    ref_now_xyz  : the reference read now (None = unavailable).
    Returns (new_aim, verdict, (dxy_mm, dz_mm)); verdict is "ok" (within noise),
    "followed", "unavailable", or "refused" (moved further than the re-aim limit
    — too much to trust, the aim is left alone).
    """
    if ref0_xyz is None or ref_now_xyz is None:
        return tuple(aim_xyz), "unavailable", (0.0, 0.0)
    dx = float(ref_now_xyz[0]) - float(ref0_xyz[0])
    dy = float(ref_now_xyz[1]) - float(ref0_xyz[1])
    dz = float(ref_now_xyz[2]) - float(ref0_xyz[2])
    dxy_mm, dz_mm = math.hypot(dx, dy) * 1000.0, dz * 1000.0
    if dxy_mm > max_xy_mm or abs(dz_mm) > max_z_mm:
        return tuple(aim_xyz), "refused", (dxy_mm, dz_mm)
    if dxy_mm <= tol_xy_mm and abs(dz_mm) <= tol_z_mm:
        return tuple(aim_xyz), "ok", (dxy_mm, dz_mm)
    return ((float(aim_xyz[0]) + dx, float(aim_xyz[1]) + dy, float(aim_xyz[2]) + dz),
            "followed", (dxy_mm, dz_mm))


def _nearest_square_equivalent(yaw, bearing):
    """For a SQUARE footprint, `yaw` and `yaw + k*90deg` (any integer k) are
    the same physical grasp — the jaws meet the same pair of opposite faces
    either way. Pick whichever representative is closest to `bearing` (the
    direction the robot is actually standing in), not whichever one
    minAreaRect happened to return.

    This exists because a square has an ambiguity estimate_grasp_orientation's
    own 180-degree disambiguation (against the camera ray) does not resolve:
    that logic only picks between a candidate and its opposite, never between
    a candidate and the SAME box rotated 90 degrees, which looks identical.
    Confirmed live (2026-10-05): the measured yaw landed ~90deg from bearing
    on tea_box in 2 of 2 trials after the orientation fix first shipped
    (-85deg vs a -1.2deg bearing; 90deg vs a 0.85deg bearing) — minAreaRect's
    choice between the two is effectively arbitrary, and backing the standoff
    away along the far one sends it ~100mm to the SIDE of the object instead
    of in front of it, which the approach/standoff machinery (built assuming
    the gripper roughly faces the same way the robot does) cannot reach."""
    best, best_diff = yaw, abs(((yaw - bearing + math.pi) % (2 * math.pi)) - math.pi)
    for k in (-2, -1, 1, 2):
        cand = yaw + k * math.pi / 2.0
        diff = abs(((cand - bearing + math.pi) % (2 * math.pi)) - math.pi)
        if diff < best_diff:
            best, best_diff = cand, diff
    return best


def grasp_pose(target_xy, dims_hwl, table_z, orientation=None):
    """EE horizontal grasp pose (position [x,y,z], quaternion [x,y,z,w]):
    gripper level, descends onto the object body.

    target_xy   : (x, y) of the object in base_link.
    orientation : the object's own measured yaw (radians, base_link), when
        known — the gripper then closes parallel to its real faces instead of
        just facing its centre. Without this, a non-round object (a box) is
        closed on at whatever angle the robot happens to be standing at,
        which only coincidentally lines up with a flat face; a few degrees of
        mismatch is enough for the jaws to meet a CORNER instead, and corner
        contact is exactly what toppled tea_box live (confirmed via a
        ground-truth pose trace, 2026-10-05: 45-67deg of roll/pitch while
        being lifted, after a bearing-only grasp). None (round object, or
        orientation never measured) keeps the original behaviour, which is
        correct for a round object — any angle meets the same curved surface.
        For a near-SQUARE footprint, snapped to whichever 90-degree-equivalent
        representation stays closest to the bearing (see
        _nearest_square_equivalent) — an elongated box only has the 180-degree
        ambiguity already resolved upstream, so this only applies near 1:1.
    """
    x, y = float(target_xy[0]), float(target_xy[1])
    bearing = math.atan2(y, x)
    if orientation is None:
        yaw = bearing
    else:
        yaw = float(orientation)
        if dims_hwl is not None:
            _, w, l = dims_hwl
            if max(w, l) / max(min(w, l), 1e-6) < 1.3:
                yaw = _nearest_square_equivalent(yaw, bearing)
    z = _horiz_grasp_z(table_z, dims_hwl)
    quat = tft.quaternion_from_euler(0.0, 0.0, yaw)
    return [x, y, z], list(quat)


def approach_vector(quat):
    """Unit vector the gripper points along — its local +x in base frame.

    Taken from the rotation matrix rather than re-derived from yaw so it stays
    correct if the grasp orientation ever gains pitch or roll.
    """
    R = tft.quaternion_matrix(quat)
    v = R[:3, 0]
    return v / (np.linalg.norm(v) + 1e-9)


def pregrasp_pose(grasp_xyz, quat, offset=PREGRASP_STANDOFF_OFFSET):
    """Pre-grasp standoff: `offset` metres BACK from the grasp point along the
    approach axis, at the SAME height and orientation.

    THE single source of truth for the standoff, for the same reason
    grasp_height_above_table is for the grip height: the motion node flies to it
    and the agent's reachability sweep scores the path to it, so if the two ever
    re-derived it independently the agent would optimise the base position for a
    standoff the arm never uses.

    Because the standoff shares the grasp height, the grasp itself is a straight
    horizontal line from here — no descent, and the arm never passes over the
    object at any point in either phase.
    """
    a = approach_vector(quat)
    pre = np.asarray(grasp_xyz, float) - float(offset) * a
    return [float(pre[0]), float(pre[1]), float(pre[2])], list(quat)


# Air added to each side of a neighbour's box. 20 mm, on top of YOLO widths that
# read 8-11 mm wide, left too narrow a corridor for the open gripper between
# objects 10 cm apart once the grasp is low enough to pass between them.
OBSTACLE_PAD = 0.015


def obstacle_boxes(detections, target_name, table_z, pad=OBSTACLE_PAD):
    """Collision boxes for every NON-target detected object.

    Orientation is unknown, so each is a conservative axis-aligned box with a
    square footprint (max of width/length + pad) covering any yaw, sitting on
    the table.  Returns a list of dicts: name, x, y, z (centre), sx, sy, sz.
    """
    boxes, seen = [], {}
    tname = (target_name or "").lower()
    for d in detections or []:
        cls = (d.get("class_name") or "").lower()
        pb = d.get("position_base")
        dims = d.get("dims_hwl")
        if not (pb and dims):
            continue
        if d.get("is_target") or cls == tname:
            continue                    # target is handled separately
        h = float(dims[0]); w = float(dims[1]); l = float(dims[2])
        foot = max(w, l) + 2 * pad
        # de-dup by rounded position so repeated frames of one object collapse
        key = (round(pb[0], 2), round(pb[1], 2))
        if key in seen:
            continue
        seen[key] = True
        name = f"obs_{cls or 'obj'}_{len(boxes)}"
        boxes.append({"name": name,
                      "x": float(pb[0]), "y": float(pb[1]),
                      "z": table_z + h / 2.0,
                      "sx": foot, "sy": foot, "sz": h + 2 * pad})
    return boxes


# ── Gripper widths ───────────────────────────────────────────────────────────
# Ported from the retired locobot_motion_node's _gripper_span_for /
# _gripper_target_close (same numbers), as plain functions instead of methods
# so arm_node's caller can build a full GraspPlan without an arm-node instance.

def gripper_span_for(dims_hwl, grasp_axis, open_clearance=0.022, grip_squeeze=0.018):
    """Per-finger OPEN and CLOSE positions adapted to the object size.

    The gripper is antagonistic: right_finger mirrors left_finger, and the pads
    touch the object at finger_pos_for_width(width) (half-width + PAD_INSET).
    Sizing both motions off the measured object makes the same code grasp a
    thin bottle or a wide box — only the numbers change.

    dims_hwl      : [H, W, L] camera-measured object dimensions in metres.
    grasp_axis    : "along_x" -> fingers close across W (dims_hwl[1]),
                    "along_y" -> fingers close across L (dims_hwl[2]).
    open_clearance: extra gap per side on approach so the jaws pass around the
                    object without clipping it.
    grip_squeeze  : how far past the object surface (per side) to command on
                    close, so the effort controller loads onto the object.
    Returns (open_pos, close_pos), both clipped to the mechanical range.
    Falls back to full open / full close when dimensions are unknown.
    """
    if dims_hwl is None:
        return FINGER_OPEN, FINGER_CLOSE
    narrow_m = float(dims_hwl[1] if grasp_axis == "along_x" else dims_hwl[2])
    contact = finger_pos_for_width(narrow_m)
    open_pos = float(np.clip(contact + open_clearance, FINGER_CLOSE + 0.008, FINGER_OPEN))
    close_pos = float(np.clip(contact - grip_squeeze, FINGER_CLOSE, FINGER_OPEN - 0.005))
    return open_pos, close_pos


def gripper_close_target(dims_hwl, grasp_axis, squeeze=0.006):
    """Per-finger close target — CONDITIONAL on the object's shape.

    TAPERED or unknown width (tall slender bottle): FULL close. Its width at the
    grip height is not measured — the bbox width is the wide body — so a
    body-sized target would stop the fingers in mid-air at the narrower grip
    point (no contact, no weld). Closing fully lets them reach whatever
    cross-section is there; the gentle vendor gain (p=100) plus the weld hold
    it without shoving.

    PRISMATIC with a trustworthy measured width (can, box): ADAPTIVE close to
    just inside the surface. A full close on a uniform object slams shut, so the
    first finger shoves the object sideways before the second lands; stopping a
    few mm inside the surface lets the jaws meet it together and self-centre.
    `squeeze` is the tuned 6 mm reach-through of the bbox width over-estimate,
    not a force knob.
    """
    if is_tapered(dims_hwl, grasp_axis):
        return FINGER_CLOSE
    _, close_pos = gripper_span_for(dims_hwl, grasp_axis, grip_squeeze=squeeze)
    return close_pos


# ── Collision boxes for the two-phase grasp ──────────────────────────────────
# Ported for arm_node.py from the retired locobot_motion_node's
# _add_target_obstacle/_add_table_collision, as plain functions returning the
# same dict shape obstacle_boxes() already uses (name, x, y, z, sx, sy, sz).

def target_box(target_xy, dims_hwl, table_z):
    """The TARGET as a collision box for the pre-grasp phase: the arm avoids it
    while approaching, then it is removed before the straight-in move. Lightly
    padded (we deliberately approach it), square footprint (yaw unknown), body
    resting on the table. The pad is shared with min_standoff_offset, which
    sizes the closest the standoff may ever sit to this box."""
    h, w, l = dims_hwl if dims_hwl is not None else UNKNOWN_OBJECT_DIMS
    foot = max(float(w), float(l)) + 2 * TARGET_BOX_PAD
    return {"name": "target_obstacle",
            "x": float(target_xy[0]), "y": float(target_xy[1]),
            "z": table_z + float(h) / 2.0,
            "sx": foot, "sy": foot, "sz": float(h) + 2 * TARGET_BOX_PAD}


def table_box(table_z, width, depth, near_x, target_x, thickness=0.05, padding=0.02):
    """The table as a collision box, `padding` above the real surface.

    With a measured near edge (in front of the robot, > 5 cm) the box's near
    face sits ON that edge, so arm links passing over the edge do not collide
    with it. Otherwise it is centred on the target's x (can clip the elbow; a
    fallback only). Minimum size floors: a wide, deep box is safer than a
    narrow wrong one.
    """
    if near_x is not None and float(near_x) > 0.05:
        cx = float(near_x) + depth / 2.0
    else:
        cx = float(target_x)
    thick = thickness + padding
    return {"name": "table", "x": cx, "y": 0.0,
            "z": table_z + padding - thick / 2.0,    # top face at table_z + padding
            "sx": max(depth, 0.50), "sy": max(width, 1.00), "sz": thick}


# ── The baseline plan ────────────────────────────────────────────────────────
# Added for arm_node.py: packages grasp_pose/pregrasp_pose/gripper sizing into
# one dict — the fields of locobot_prog/GraspPlan — so a caller can hand the
# arm a complete, self-contained plan instead of raw position+dims.  This is
# THE plan the pipeline has always executed; a caller with a better estimate
# (e.g. from real depth data) may build and pass a different one instead — the
# arm executes whatever GraspPlan it is given and derives no geometry itself.

def baseline_plan(target_xy, dims_hwl, grasp_axis, table_z, orientation=None):
    """The grasp the pipeline has always executed, as one plan: level gripper
    facing the object's centre (or aligned to its real faces, when
    `orientation` is given — see grasp_pose), grip height from
    grasp_height_above_table, gripper widths from the measured bbox.

    Returns a plain dict (the fields of locobot_prog/GraspPlan). When no side
    grasp exists the plan is returned with valid=False and the reason, rather
    than raising, so callers can report it.
    """
    size = dims_hwl if dims_hwl is not None else UNKNOWN_OBJECT_DIMS
    plan = {
        "valid": True,
        "reason": ("baseline: bbox centre, measured yaw" if orientation is not None
                  else "baseline: bbox centre, bearing yaw"),
        "standoff_offset": PREGRASP_STANDOFF_OFFSET,
        "standoff_offset_min": min_standoff_offset(dims_hwl),
        "full_close": is_tapered(dims_hwl, grasp_axis),
        # x = depth extent (L), y = width (W), z = height (H)
        "object_size": [float(size[2]), float(size[1]), float(size[0])],
    }
    if dims_hwl is not None:
        narrow_m = float(dims_hwl[1] if grasp_axis == "along_x" else dims_hwl[2])
        jaw_max = jaw_width(FINGER_OPEN)
        if narrow_m + 2 * JAW_CLEARANCE > jaw_max:
            plan.update(valid=False,
                        reason=(f"ObjectTooWideForJaws: {narrow_m*1000:.0f}mm "
                                f"exceeds the jaws' {jaw_max*1000:.0f}mm max "
                                "opening — no per-band search here to find a "
                                "narrower spot, unlike the depth planner"),
                        grasp_point=[float(target_xy[0]), float(target_xy[1]), table_z],
                        grasp_orientation=[0.0, 0.0, 0.0, 1.0],
                        gripper_open=FINGER_OPEN, gripper_close=FINGER_CLOSE)
            return plan
    try:
        xyz, quat = grasp_pose(target_xy, dims_hwl, table_z, orientation=orientation)
    except ObjectTooShortForSideGrasp as e:
        plan.update(valid=False, reason=f"Object too short for a side grasp: {e}",
                    grasp_point=[float(target_xy[0]), float(target_xy[1]), table_z],
                    grasp_orientation=[0.0, 0.0, 0.0, 1.0],
                    gripper_open=FINGER_OPEN, gripper_close=FINGER_CLOSE)
        return plan
    plan.update(grasp_point=[float(v) for v in xyz],
                grasp_orientation=[float(v) for v in quat],
                gripper_open=gripper_span_for(dims_hwl, grasp_axis)[0],
                gripper_close=gripper_close_target(dims_hwl, grasp_axis))
    return plan
