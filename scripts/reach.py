"""Inverse reachability: where the base should stand so the arm is comfortable.

This is a relocation, not new
capability: ARM_LIMITS is still a hardcoded dict here, same as before — the
"derive reach from the URDF at startup" idea from the original refactor brief
was never actually implemented, and stays a genuinely open item, independent
of this port.

Pure: the IK solver is passed in as a function ik(position, quaternion) ->
(joint_names, joint_positions) or None, so the scoring can be tested without
MoveIt.

MoveIt cannot plan the base (non-holonomic, drifting odometry), but its IK
can CHOOSE the standoff the base then drives to: sweep candidate distances,
score each by how far the arm stays from its joint limits along the whole
approach, and prefer a plateau so a small base error still lands well.
"""
import grasp_geometry as gg

# WidowX-250s arm joint limits (rad), from the URDF.
ARM_LIMITS = {
    "waist":        (-3.142, 3.142),
    "shoulder":     (-1.885, 1.990),
    "elbow":        (-2.147, 1.606),
    "forearm_roll": (-3.142, 3.142),
    "wrist_angle":  (-1.745, 2.147),
    "wrist_rotate": (-3.142, 3.142),
}
# Candidate standoffs. 0.49-0.51 (was 0.44-0.47): parked nearer, the base is still
# upright at the pre-grasp and flips during the straight-in leg, knocking a tall object
# (see world_model_node.py, BASE TILT FLIP). The arm's reach abort is 0.55 m.
STANDOFF_MIN, STANDOFF_MAX, STANDOFF_STEP = 0.49, 0.51, 0.01
APPROACH_SAMPLES = 3     # points sampled from standoff to grasp point (inclusive)
PLATEAU_WEIGHT = 0.25    # weight on the mean margin of the +-1 neighbour standoffs


def joint_margin(names, positions):
    """Min over arm joints of the normalized distance to the nearest joint
    limit (0 = at a limit, 0.5 = mid-range). Higher = more comfortable."""
    m = 1.0
    for name, pos in zip(names, positions):
        key = name.split("/")[-1]
        if key in ARM_LIMITS:
            lo, hi = ARM_LIMITS[key]
            m = min(m, min(pos - lo, hi - pos) / (hi - lo))
    return m


def approach_margin(ik, dims_hwl, table_z, d):
    """Min joint-limit margin over the ENTIRE straight-line approach with the
    target straight ahead at distance d: APPROACH_SAMPLES points from the
    pre-grasp standoff to the grasp point, each IK-solved. None if any point is
    unreachable (the whole approach must be feasible), or if the object has no
    valid side-grip height at all.

    Both endpoints come from the SHARED grasp helpers, so the standoff scored
    is the one the arm flies to. That matters: the margin along this line is
    dominated by the STANDOFF, the folded end of the move (measured: 0.384 rad
    at the standoff vs 0.679 at the grasp point)."""
    try:
        gp, gq = gg.grasp_pose((d, 0.0), dims_hwl, table_z)
    except gg.ObjectTooShortForSideGrasp:
        return None
    pre, _ = gg.pregrasp_pose(gp, gq)
    margin = 1.0
    for k in range(APPROACH_SAMPLES):
        frac = k / (APPROACH_SAMPLES - 1) if APPROACH_SAMPLES > 1 else 0.0
        pt = [pre[i] + (gp[i] - pre[i]) * frac for i in range(3)]
        sol = ik(pt, gq)
        if sol is None:
            return None
        margin = min(margin, joint_margin(*sol))
    return margin


def reach_optimal_standoff(ik, dims_hwl, table_z):
    """Sweep the standoff band and return (distance, own_margin, score) for the
    best candidate — own margin plus PLATEAU_WEIGHT x the mean of its reachable
    neighbours — or None when no candidate is reachable."""
    grid = []
    d = STANDOFF_MIN
    while d <= STANDOFF_MAX + 1e-9:
        grid.append((round(d, 2), approach_margin(ik, dims_hwl, table_z, d)))
        d += STANDOFF_STEP
    best = None
    for i, (di, si) in enumerate(grid):
        if si is None:
            continue
        neigh = [grid[j][1] for j in (i - 1, i + 1)
                 if 0 <= j < len(grid) and grid[j][1] is not None]
        val = si + PLATEAU_WEIGHT * (sum(neigh) / len(neigh) if neigh else 0.0)
        if val > (best[2] if best else -1.0):
            best = (di, si, val)
    return best
