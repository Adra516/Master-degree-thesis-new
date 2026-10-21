"""Where to point the camera head, and in what order to turn a held object. Pure: no ROS.

The LoCoBot's camera sits on a pan/tilt head on the base tower, 0.56 m up and directly
ABOVE the arm base (x 0.087 vs 0.092): the arm's vertical plane contains the camera, so
whatever the hand holds in front is seen past the elbow and forearm. Aiming the head at
the held object fixes the framing (it was 36 deg off the optical axis, outside the 44 deg
vertical field of view, with the head at its table tilt), but not the occlusion; that is
what yawing the hand does (see srv/TurnHand.srv).

Geometry measured from TF on 2026-10-19 (base_link, metres). Tilt is positive LOOKING
DOWN, as the controllers expect.
"""
import math
from collections import namedtuple

CAMERA_POS = (0.087, 0.017, 0.561)       # camera_link at pan 0 / tilt ~0
PAN_LIMIT = math.radians(90.0) - 0.05
TILT_MIN = -math.radians(90.0) + 0.05
TILT_MAX = math.radians(75.0) - 0.05

# The poses of an inspection tour, in priority order (a cap on the number of views keeps the first
# ones). Measured 2026-10-19/20 on a held bottle, bottle pixels visible / outcome:
#   hand straight ahead 9k; yaw +-0.5 rad 31k; yaw +-0.35 22-25k; yaw +0.5 and 6 cm closer 28k;
#   yaw +-0.4 with the top tilted toward the camera (pitch -0.3) 16-19k and the cap in view;
#   NOT reachable (no IK solution): the opposite tilt (pitch +0.3), 5 cm higher, yaw +-0.9;
#   5 cm lower is reachable but the bottle slid down in the grip. The bottle was lost after about
#   eight poses in one run, so the tour is short and has a time budget.
# The two tilted-top poses (pitch != 0) are OPT-IN (`include_tilt`): in the simulator the soft
# grasp-fix weld lets the bottle ratchet out of the pads when the hand pitches (finger asymmetry
# grew until one finger closed on nothing: lost mid-tour in 4 of ~13 tours with them against 1 of
# ~12 without, 2026-10-20). They give the best view of the cap, so they stay available.
HandPose = namedtuple("HandPose", "yaw pitch dx dz label")
INSPECT_POSES = (
    HandPose(-0.5, 0.0, 0.0, 0.0, "turned 29 degrees to the left"),
    HandPose(0.5, 0.0, 0.0, 0.0, "turned 29 degrees to the right"),
    HandPose(-0.35, 0.0, 0.0, 0.0, "turned 20 degrees to the left"),
    HandPose(0.35, 0.0, 0.0, 0.0, "turned 20 degrees to the right"),
    HandPose(0.5, 0.0, -0.06, 0.0, "turned 29 degrees to the right, 6 cm closer"),
    HandPose(-0.4, -0.3, 0.0, 0.0, "top tilted toward the camera, turned 23 degrees to the left"),
    HandPose(0.4, -0.3, 0.0, 0.0, "top tilted toward the camera, turned 23 degrees to the right"),
)
# Used when the object's size is unknown: the held object's centre below the hand.
DEFAULT_CENTRE_DROP_M = 0.08


# Head motion. A single position command makes the effort controller (p 30) slam the head to
# the new angle; a minimum-jerk ramp (zero speed and acceleration at both ends) moves it
# smoothly instead. 0.5 rad/s peak keeps a 0.5 rad re-aim under 2 s.
HEAD_MAX_SPEED = 0.5          # rad/s, peak of the ramp
HEAD_MIN_DURATION = 0.4       # s; even a tiny re-aim is eased, not stepped
HEAD_RATE_HZ = 25.0


def aim_angles(point, camera_pos=CAMERA_POS):
    """(pan, tilt) in rad that put `point` (base_link) on the camera's optical axis."""
    dx, dy, dz = (point[i] - camera_pos[i] for i in range(3))
    pan = math.atan2(dy, dx)
    tilt = -math.atan2(dz, math.hypot(dx, dy))
    return (max(-PAN_LIMIT, min(PAN_LIMIT, pan)), max(TILT_MIN, min(TILT_MAX, tilt)))


def held_centre(ee, grasp_above_table=None, height=None, pitch=0.0):
    """Centre of a held object, given the hand's position. The grip is at
    `grasp_above_table` on an object `height` tall, so the centre is
    (grasp_above_table - height/2) below the hand; unknown size -> a fixed drop.
    A hand pitch swings that centre about the held point: positive pitch moves it toward the
    robot (the horizontal direction from the base to the hand), negative away from it."""
    if grasp_above_table is not None and height:
        drop = max(0.0, grasp_above_table - height / 2.0)
    else:
        drop = DEFAULT_CENTRE_DROP_M
    reach = math.hypot(ee[0], ee[1])
    ux, uy = (ee[0] / reach, ee[1] / reach) if reach > 1e-6 else (1.0, 0.0)
    return (ee[0] - ux * drop * math.sin(pitch), ee[1] - uy * drop * math.sin(pitch),
            ee[2] - drop * math.cos(pitch))


def inspect_tour(max_views=None, poses=INSPECT_POSES, include_tilt=True):
    """The moves of an inspection tour: HandPose items and None for 'return to the start'.
    The first `max_views` poses by priority are kept; they are then ordered so each side is
    visited from the smallest turn outward, with a return to the start between the sides and
    at the end; the tilted-top poses are dropped unless `include_tilt` (a direct swing from one side to the other would tilt the held object on the
    way)."""
    pool = [p for p in poses if include_tilt or abs(p.pitch) < 1e-9]
    keep = list(pool if max_views is None else pool[:max(0, int(max_views))])
    key = lambda p: (abs(p.yaw), abs(p.pitch), p.dx)   # noqa: E731
    left = sorted((p for p in keep if p.yaw < 0), key=key)
    right = sorted((p for p in keep if p.yaw >= 0), key=key)
    tour = list(left)
    if left and right:
        tour.append(None)
    tour += right
    if tour:
        tour.append(None)
    return tour


def _ease(s):
    """Minimum-jerk blend 0 -> 1 (zero velocity and acceleration at both ends)."""
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def head_path(start, end, max_speed=HEAD_MAX_SPEED, min_duration=HEAD_MIN_DURATION,
              rate_hz=HEAD_RATE_HZ):
    """A smooth ramp for the head from `start` to `end`, both (pan, tilt) in rad.
    Returns ([(pan, tilt), ...], dt): the waypoints after `start` ending exactly on `end`,
    one every dt seconds. The peak speed of the larger of the two joint moves is
    `max_speed`, reached mid-ramp (a minimum-jerk blend peaks at 1.875 x the mean speed)."""
    dist = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
    duration = max(min_duration, 1.875 * dist / max_speed)
    n = max(1, int(round(duration * rate_hz)))
    pts = []
    for i in range(1, n + 1):
        e = _ease(i / n)
        pts.append((start[0] + (end[0] - start[0]) * e, start[1] + (end[1] - start[1]) * e))
    pts[-1] = (end[0], end[1])
    return pts, duration / n
