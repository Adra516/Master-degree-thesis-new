import math

import numpy as np
import tf.transformations as tft


# ── Grasp-height model — THE single source of truth ──────────────────────────
# Grip at a fraction of the object's own height instead of a near-constant band.
GRIP_FRAC       = 0.50   # grip point as a fraction of object height (mid-body)
GRIP_Z_FLOOR    = 0.06   # lowest height above the table the jaws safely reach
GRIP_TOP_MARGIN = 0.018  # keep the jaws this far below the object's top edge
GRIP_Z_CEILING  = 0.085
# Height used when dims are unknown: middle of the proven band.  
GRIP_Z_UNKNOWN  = 0.08


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
# target's box half-extent is ~0.04-0.055 m (a 0.06-0.09 m footprint plus pad),
# so 0.08 m clears it with margin.  MoveIt enforces the rest for free — the
# approach is planned collision-aware with the target still an obstacle, so a
# standoff that IS inside the box simply fails to plan, and the caller's
# adaptive retry escalates the offset rather than a hand-tuned formula having
# to predict it.
TARGET_BOX_PAD = 0.01   # MUST match the motion node's _add_target_obstacle pad
PREGRASP_STANDOFF_OFFSET = 0.08   # metres back along the approach axis
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


def contact_verdict(metrics, miss_tol_mm):
    """Are the OPEN jaws straddling the object, measured just before closing?

    metrics : the dict returned by _diag_contact (along_jaw_mm, in_plane_miss_mm,
              straddled), or None when no object position was available.

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
    return True, f"ok (along-jaw {along:+.1f}mm, miss {miss:.1f}mm)"


def grasp_pose(target_xy, dims_hwl, table_z):
    """EE horizontal grasp pose (position [x,y,z], quaternion [x,y,z,w]):
    gripper level, +x pointing at the object, descends onto the object body.

    target_xy : (x, y) of the object in base_link.
    """
    x, y = float(target_xy[0]), float(target_xy[1])
    yaw = math.atan2(y, x)                      # face the object
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


def obstacle_boxes(detections, target_name, table_z, pad=0.02):
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
