"""Where exactly the jaws close: the width profile picks the grip height, the
slice's width function picks the closing axis and the contact points.
Pure: numpy + OpenCV.

Input is the target's points in base_link (geometry.footprint), so this works
on what the camera actually measured instead of a bounding box:

1. Height. The object is sliced into 5 mm bands and each band's LATERAL width
   (across the approach — what the jaws close over, and what one camera
   measures well) is recorded. EVERY band across the object's whole physically
   reachable range is a candidate — from the arm's real floor (GRIP_Z_FLOOR) up
   to just below its own measured top (GRIP_TOP_MARGIN) — not just a narrow
   window around grasp_height_above_table's guessed height. That guess still
   matters, but only as a tie-break preference (in _band_score) among bands
   that already fit the jaws: real objects are not uniform (a bottle's neck can
   be 40mm narrower than its body 4cm lower), and a formula with no depth image
   has no way to know that — only the measured profile does. A band must fit
   the real jaw span (30-74 mm, from the URDF finger limits) with clearance.

2. The hidden back. A single camera sees only the front of the object: a box
   seen face-on shows its front face and nothing of the sides the jaws will
   touch. Each band is therefore completed by reflecting it through its
   centre — exact for anything centrally symmetric (cylinders, boxes, cans).
   The centre is laterally the middle of the silhouette (the silhouette IS
   visible); along the approach it is the larger of two estimates: the middle
   of the two silhouette points (exact for a turned box, a few mm short for a
   cylinder, half a box short for a box seen face-on) and the front surface
   plus half the known depth (right for the face-on box).

3. Closing axis. Parallel jaws squeezing an object turn it toward the nearest
   minimum of its width function w(theta) — the distance between the two
   support lines of its convex hull (OpenCV) closing along theta — until a jaw
   lies flush on a face. How far it has to turn is the grasp's quality measure
   (quality = cos(turn)): 0° for a box closed across its faces or for anything
   round, 30° for a box turned 30° to the jaws. Past MAX_SEAT_TURN the squeeze
   shoves the object more than it turns it. The width function is used rather
   than the hull's edge normals because it is a max over many points: depth
   noise that puts spurious millimetre edges on the hull barely moves it. The
   closing axis may deviate from perpendicular-to-approach by up to max_dev
   (0 = today's fixed axis).

4. Grasp point: the band's centre, which is also the midpoint of the contacts
   (the completed band is symmetric about it).
"""
import math

import cv2
import numpy as np

import footprint
import rotations
import grasp_geometry as gg

JAW_MIN = 2 * 0.015                 # URDF left_finger lower limit, both fingers
JAW_MAX = 2 * gg.FINGER_OPEN        # 0.074
SIDE_CLEARANCE = 0.002              # the open jaws must pass the object by this, per side
MAX_SEAT_TURN = math.radians(25.0)  # how far the object may turn in the closing jaws
ROUND_TOL = 0.002                   # w(theta) this close to its minimum: nothing to turn toward
# The jaw pads are a vertical strip, not a line: the finger mesh
# (mobile_wx250s_10_gripper_finger.stl) is 55.3 x 16.0 x 29.0 mm, and its 29 mm
# side is vertical for a level grip. Everything within half of that above and
# below the grip height has to fit between the open jaws too.
PAD_HALF = 0.0145
# ...plus what the arm actually does: it sits ~3 mm below the commanded height
# (pre-grasp dz was -2.3 to -3.1 mm on every run, gravity sag) and bands are 5 mm
# apart. A live trace with ground truth showed why the margin matters: at 188 mm
# the pad reached down to the 58 mm shoulder, the right finger came within
# ~1.6 mm of it, and with this model's friction (mu 5) that graze dragged the
# bottle 21 mm forward before the jaws closed.
PAD_MARGIN = 0.008
# Per-side clearance at which more room stops mattering. A tight fit is easy to
# miss by a few mm of lateral error (one finger lands on the object and pushes
# it), so clearance is the main preference after seating. It has to cover the
# aim error the contact check actually measured on live picks — the object sat
# 5.8, 8.1, 8.2 and 9.4 mm off the jaws' centre line (the along-jaw value, which
# matched Gazebo ground truth to 0.4 mm) — plus what the grip height cannot
# promise: the ~3 mm sag and the pad's lower edge. 10 mm/side (the earlier
# value) was consumed by the aim error alone: the planner took a band whose pad
# reached down to the bottle's shoulder with ~10 mm to spare on paper, and the
# lower finger corner grazed it in the real run. 18 = 10 + 8.
CLEARANCE_FULL = 0.018
# How far beyond the surface the fingers are commanded (per finger). The jaws stop
# at the object; the command only sets the squeeze. Measured on live bottle picks
# at a fixed grip height: ~4-6 mm held it upright; ~8-10 mm made the pads walk
# down the neck's taper to the 52-60 mm shoulder (gripper read 26-28 mm/finger) or
# shoved it out of the jaws (9 mm = empty). But on a genuine cone the SAME 6 mm
# does not mean the same thing at every band: two live picks on bands 4 mm apart
# (198 mm, 202 mm) put the fingers 6.8 mm and 2.8 mm short of a 6 mm target —
# the second one, with barely any margin between the measured contact and the
# commanded target, tipped the bottle >10 deg and dragged it 38 mm while
# closing. 4 mm is the safe end of that measured range; it still holds (measured
# clean: 0 deg / 0 mm disturbance at 198 mm) with more of the margin behind it.
SQUEEZE = 0.004
TAPER_LOOKUP = 0.02                 # diagnostic only: compare the width this far above the grip...
TAPER_DROP = 0.005                  # ...narrower by more than this = tapered
DEV_STEP = math.radians(2.0)
CUT_MARGIN = 0.02                   # the world model's horizontal position error, roughly

_DEG = np.radians(np.arange(180))
_AXES = np.column_stack((np.cos(_DEG), np.sin(_DEG)))    # closing axes, 1° apart


def cut_radius(dims_hwl):
    """How far from the target's estimated centre a point may lie and still
    belong to it (footprint.object_points): half the footprint's diagonal plus
    the position error. Anything farther is a neighbour sharing the detection
    box, or the background."""
    _, w, l = dims_hwl if dims_hwl is not None else gg.UNKNOWN_OBJECT_DIMS
    return math.hypot(float(w), float(l)) / 2.0 + CUT_MARGIN


def width_function(hull):
    """w(theta) for theta = 0..179°, smoothed over ±2° (w repeats every 180°)."""
    proj = hull @ _AXES.T
    w = proj.max(axis=0) - proj.min(axis=0)
    return sum(np.roll(w, k) for k in range(-2, 3)) / 5.0


def seat_turn(w, i):
    """How far (radians) an object closed along axis i (degrees) turns before
    it is seated: to the lowest w within ±45° (a box's basin), or not at all
    when w there is no lower than at i by more than ROUND_TOL."""
    basin = (i + np.arange(-45, 46)) % 180
    j = int(basin[np.argmin(w[basin])])
    if w[i] - w[j] <= ROUND_TOL:
        return 0.0
    return math.radians(abs((j - i + 90) % 180 - 90))


def slice_centre(band, bearing, depth):
    """Centre (x, y) of a band, from its visible points and the object's full
    depth along the approach (see the module header, step 2)."""
    ahat = np.array([math.cos(bearing), math.sin(bearing)])
    lhat = np.array([-math.sin(bearing), math.cos(bearing)])
    s, a = band.xy @ lhat, band.xy @ ahat
    a_silhouette = (a[np.argmax(s)] + a[np.argmin(s)]) / 2.0
    a_c = max(float(a_silhouette), band.a_front + depth / 2.0)
    s_c = (band.s_left + band.s_right) / 2.0
    return a_c * ahat + s_c * lhat


def contact_candidates(xy, bearing, max_dev=0.0, centre=None):
    """For each closing axis within max_dev of perpendicular-to-approach: the
    two support points of the slice's convex hull, the span between them, how
    far the object would turn in the jaws, and quality = cos(turn). With
    `centre`, the slice is first completed by reflection through it."""
    xy = np.asarray(xy, np.float64)
    if len(xy) < 3:
        return []
    if centre is not None:
        xy = np.vstack((xy, 2.0 * np.asarray(centre) - xy))
    hull = cv2.convexHull(xy.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    if len(hull) < 3:
        return []
    w = width_function(hull)
    steps = max(1, int(round(max_dev / DEV_STEP))) if max_dev > 0 else 0
    out = []
    for k in range(-steps, steps + 1):
        dev = k * DEV_STEP
        phi = bearing + math.pi / 2.0 + dev
        c = np.array([math.cos(phi), math.sin(phi)])
        proj = hull @ c
        i_l, i_r = int(np.argmax(proj)), int(np.argmin(proj))
        turn = seat_turn(w, int(round(math.degrees(phi))) % 180)
        out.append({"dev": dev, "axis": c, "width": float(proj[i_l] - proj[i_r]),
                    "left": hull[i_l], "right": hull[i_r],
                    "turn": turn, "quality": math.cos(turn)})
    return out


def _band_score(cand, z, z_ref, env_width):
    """Seating first (1 when the object will not turn in the jaws, 0 at
    MAX_SEAT_TURN); then clearance for the open jaws over the WHOLE pad height
    (env_width, the widest the object gets within PAD_HALF of z), which is what
    decides whether a few mm of lateral error pushes the object; then, only as a
    tie-break, staying near the formula's anchor height; then keeping the
    default closing axis.

    The anchor bonus is deliberately small. At 0.5 it outweighed the whole
    clearance term, so the bottle's 66 mm body (4 mm/side) beat its 52 mm neck
    (11 mm/side) whenever the body fit at all — the formula's height guess
    silently overruled the measurement it was meant to break ties between."""
    clearance = (JAW_MAX - max(env_width, cand["width"])) / 2.0
    return (1.0 - cand["turn"] / MAX_SEAT_TURN
            + 1.0 * min(1.0, clearance / CLEARANCE_FULL)
            + 0.1 * math.exp(-((z - z_ref) / 0.015) ** 2)
            - 0.2 * abs(cand["dev"]) / math.radians(30.0))


def plan_from_points(points, table_z, bearing, dims_hwl, depth_hint=None, max_dev=0.0):
    """Plan a grasp from the target's points (N x 3, base_link).

    dims_hwl  : the object's [H, W, L] (the anchor height and the fallback depth)
    depth_hint: the object's full depth along the approach, if known better
                than L (e.g. from the catalogue)
    Returns (plan, diag): plan has the fields of locobot_prog/GraspPlan (as
    geometry.grasp.baseline_plan returns them); diag says how it was chosen.
    plan is None when the points cannot support a grasp (diag["reason"] says
    why); the caller then keeps the baseline plan.
    """
    diag = {"points": int(len(points))}
    try:
        z_ref = gg.grasp_height_above_table(dims_hwl)
    except gg.ObjectTooShortForSideGrasp as e:
        diag["reason"] = f"too short for a side grasp: {e}"
        return None, diag
    H = float(dims_hwl[0]) if dims_hwl is not None else None
    # The object's WHOLE physically reachable range, not a narrow window around
    # z_ref: z_ref remains the tie-break anchor (in _band_score) among bands
    # that already fit the jaws, but nothing here stops the search from finding
    # a real, better-fitting band far from that guess (see module docstring).
    z_lo = gg.GRIP_Z_FLOOR
    z_hi = (H - gg.GRIP_TOP_MARGIN) if H is not None else math.inf

    depth_prior = (depth_hint if depth_hint is not None
                   else float(dims_hwl[2]) if dims_hwl is not None else 0.0)
    bands = footprint.bands(points, table_z, bearing)
    by_z = {round(b.z, 4): b for b in bands}
    best = None
    rejected = []
    for b in bands:
        if not (z_lo <= b.z <= z_hi):
            continue
        # depth_prior is a WHOLE-OBJECT guess (catalog L, or the widest part's
        # own depth): correct for a band that's genuinely as deep as that, but
        # capped by this band's own measured lateral width — a band that's
        # narrower than the prior (a neck, once the search can reach one) is
        # assumed roughly round, so its own front-to-back depth cannot exceed
        # what its silhouette already shows sideways. Without this cap a wide
        # body's depth_prior, applied to a narrow band far above it, overshoots
        # the along-approach centre by the same amount the two widths differ.
        # How much wider than this band the object gets within the pad's height
        # (a ratio, so it carries over to a rotated closing axis: a box turned
        # 30° is 82 mm across the approach but 60 mm across its faces).
        env_ratio = max(1.0, max(o.width for o in bands
                                 if abs(o.z - b.z) <= PAD_HALF + PAD_MARGIN + 1e-9)
                        / max(b.width, 1e-6))
        depth = max(b.a_back - b.a_front, min(depth_prior, b.width))
        centre = slice_centre(b, bearing, depth)
        for cand in contact_candidates(b.xy, bearing, max_dev, centre):
            w = cand["width"]
            if w <= JAW_MIN:
                rejected.append(f"{b.z*1000:.0f}mm: {w*1000:.0f}mm < jaw minimum")
                continue
            if w + 2 * SIDE_CLEARANCE > JAW_MAX:
                rejected.append(f"{b.z*1000:.0f}mm: {w*1000:.0f}mm too wide for the jaws")
                continue
            env_width = w * env_ratio
            # Where the pads will actually touch: the widest the object is anywhere
            # within the pad's own height (not the extra margin) — on a taper that
            # is the pad's lower edge, wider than the band the grip is centred on.
            contact_ratio = max(1.0, max(o.width for o in bands
                                         if abs(o.z - b.z) <= PAD_HALF + 0.0025)
                                / max(b.width, 1e-6))
            if env_width + 2 * SIDE_CLEARANCE > JAW_MAX:
                rejected.append(f"{b.z*1000:.0f}mm: jaw pad would overlap "
                                f"{env_width*1000:.0f}mm of the object")
                continue
            if cand["turn"] > MAX_SEAT_TURN:
                rejected.append(f"{b.z*1000:.0f}mm: would turn "
                                f"{math.degrees(cand['turn']):.0f}° in the jaws")
                continue
            score = _band_score(cand, b.z, z_ref, env_width)
            if best is None or score > best[0]:
                best = (score, b, cand, centre, depth, w * contact_ratio)
    if best is None:
        diag["reason"] = ("no band in %.0f-%.0f mm fits the jaws" % (z_lo * 1000, z_hi * 1000)
                          + (f" ({rejected[0]})" if rejected else " (no points there)"))
        return None, diag

    score, band, cand, xy, depth, contact_width = best
    width = cand["width"]
    z = table_z + band.z

    above = by_z.get(round(band.z + TAPER_LOOKUP, 4))
    tapered = above is not None and above.width < width - TAPER_DROP
    yaw = bearing + cand["dev"]
    quat = [float(v) for v in rotations.quaternion_from_euler(0.0, 0.0, yaw)]
    open_pos = float(np.clip(width / 2.0 + 0.022, gg.FINGER_CLOSE + 0.008, gg.FINGER_OPEN))
    close_pos = float(np.clip(contact_width / 2.0 - SQUEEZE, gg.FINGER_CLOSE,
                              gg.FINGER_OPEN - 0.005))
    size = dims_hwl if dims_hwl is not None else gg.UNKNOWN_OBJECT_DIMS

    plan = {"valid": True,
            "reason": (f"planner: {width*1000:.0f}mm at {band.z*1000:.0f}mm, "
                       f"turns {math.degrees(cand['turn']):.0f}°, axis {math.degrees(cand['dev']):+.0f}°"),
            "grasp_point": [float(xy[0]), float(xy[1]), float(z)],
            "grasp_orientation": quat,
            "standoff_offset": gg.PREGRASP_STANDOFF_OFFSET,
            "standoff_offset_min": gg.min_standoff_offset(dims_hwl),
            "gripper_open": open_pos, "gripper_close": close_pos, "full_close": False,
            "object_size": [float(size[2]), float(size[1]), float(size[0])]}
    left = np.append(cand["left"], z)
    right = np.append(cand["right"], z)
    diag.update(width=width, band_z=band.z, band_points=band.n, quality=cand["quality"],
                turn=cand["turn"], axis_dev=cand["dev"], tapered=tapered,
                contact_width=contact_width,
                visible_depth=band.a_back - band.a_front, depth_used=depth,
                contact_left=left.tolist(), contact_right=right.tolist(),
                anchor_z=z_ref, score=score)
    return plan, diag
