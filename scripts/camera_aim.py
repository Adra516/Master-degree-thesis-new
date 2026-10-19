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

CAMERA_POS = (0.087, 0.017, 0.561)       # camera_link at pan 0 / tilt ~0
PAN_LIMIT = math.radians(90.0) - 0.05
TILT_MIN = -math.radians(90.0) + 0.05
TILT_MAX = math.radians(75.0) - 0.05

# Hand yaws tried during an inspection, rad about the vertical through the held point.
# +-0.5 solved for a held bottle, +-0.9 did not; unreachable ones are skipped. Visible
# bottle pixels (2026-10-19, same camera aim): hand straight 9.3k, +-0.25 rad 12k/22k, +-0.5 rad
# 31k/31k, so the arm only really leaves the view from about 0.35 rad outward.
INSPECT_YAWS = (-0.5, -0.35, 0.35, 0.5)
# Head motion. A single position command makes the effort controller (p 30) slam the head to
# the new angle; a minimum-jerk ramp (zero speed and acceleration at both ends) moves it
# smoothly instead. 0.5 rad/s peak keeps a 0.5 rad re-aim under 2 s.
HEAD_MAX_SPEED = 0.5          # rad/s, peak of the ramp
HEAD_MIN_DURATION = 0.4       # s; even a tiny re-aim is eased, not stepped
HEAD_RATE_HZ = 25.0
# Used when the object's size is unknown: the held object's centre below the hand.
DEFAULT_CENTRE_DROP_M = 0.08


def aim_angles(point, camera_pos=CAMERA_POS):
    """(pan, tilt) in rad that put `point` (base_link) on the camera's optical axis."""
    dx, dy, dz = (point[i] - camera_pos[i] for i in range(3))
    pan = math.atan2(dy, dx)
    tilt = -math.atan2(dz, math.hypot(dx, dy))
    return (max(-PAN_LIMIT, min(PAN_LIMIT, pan)), max(TILT_MIN, min(TILT_MAX, tilt)))


def held_centre(ee, grasp_above_table=None, height=None):
    """Centre of a held object, given the hand's position. The grip is at
    `grasp_above_table` on an object `height` tall, so the centre is
    (grasp_above_table - height/2) below the hand; unknown size -> a fixed drop."""
    if grasp_above_table is not None and height:
        drop = max(0.0, grasp_above_table - height / 2.0)
    else:
        drop = DEFAULT_CENTRE_DROP_M
    return (ee[0], ee[1], ee[2] - drop)


def inspect_order(yaws=INSPECT_YAWS):
    """Moves for a sweep, as signed yaws with 0.0 meaning 'return to the start': one
    side from the smallest turn outward, back to the start, then the other side. A
    direct swing from one side to the other would tilt the held object on the way."""
    neg = sorted((y for y in yaws if y < 0), key=abs)
    pos = sorted((y for y in yaws if y > 0), key=abs)
    order = list(neg)
    if neg and pos:
        order.append(0.0)
    order += pos
    order.append(0.0)
    return order


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
