"""Scene geometry: parse detections, find what blocks the approach, classify
the approach zone, and summarise the scene for the VLM. Pure: no ROS.

Ported near-verbatim from rework-attempt-1's src/locobot_prog/geometry/
zones.py ("moved from scripts/locobot_geometry.py. The bird's-eye renderer is
now world/bev.py"), verified against locobot_geometry.py constant-for-constant
before porting. render_birdseye moved on to world_bev.py; everything from
locobot_geometry.py this file didn't take over (parse_object, find_blocking,
classify_zone, compute_grasp, build_summary) was later deleted there as dead
code once its only caller, vlm_agent_node.py, was retired — this file and
world_bev.py are their sole surviving copies. locobot_geometry.py itself
still exists, trimmed down to its one other survivor, estimate_grasp_
orientation, which detection_shim.py calls directly.
"""

import math
from typing import List, Optional

# physical constant
ARM_REACH_M        = 0.55   # maximum horizontal reach
ARM_GRASP_DIST     = 0.45   # comfortable operating distance — 3 cm margin for overshoot
GRIPPER_W_M        = 0.08
TABLE_SURFACE_Z    = 0.25   # simulation table top surface (world z)
TABLE_DEPTH_M      = 0.40   # table extent in the X (forward) direction
TABLE_ROBOT_MARGIN = 0.12   # min gap to keep between robot base and table near edge
LAT_THRESH_M       = 0.04
# Footprint max/min ratio below which an object is treated as effectively square,
# so its estimated yaw carries no usable information and the grasp axis is taken
# from the smaller dimension instead (see parse_object).  Matches the symmetry
# threshold the perception side uses for symmetric (round) footprints.
SQUARE_FOOTPRINT_RATIO = 1.3

# Zone constants
ZONE_GRASP_READY   = "grasp_ready"
ZONE_DRIVE_FORWARD = "drive_forward"
ZONE_ROTATE_NEEDED = "rotate_needed"
ZONE_BLOCKED       = "path_blocked"
ZONE_OUT_OF_SIGHT  = "target_not_found"

ZONE_TO_ACTION = {
    ZONE_GRASP_READY:   "execute_pregrasp",
    ZONE_DRIVE_FORWARD: "drive_forward",
    ZONE_ROTATE_NEEDED: "rotate_to_target",
    ZONE_BLOCKED:       "avoid_obstacle",
    ZONE_OUT_OF_SIGHT:  "search_for_target",
}


# Geometry functions
def parse_object(d: dict) -> dict:
    """Enrich a raw YOLO3D detection dict with geometry fields."""
    x, y, z = d["position_base"]
    h, w, l  = d["dims_hwl"]
    dist     = math.hypot(x, y)
    yaw      = d.get("orientation", 0.0)

    # Grasp axis = which measured dimension the jaws close across.
    # The yaw estimate comes from a single 2-D box + depth, which is unreliable
    # for a box-like object; trusting it on a near-square footprint can select
    # the WIDER face and try to span it.  For such footprints the yaw carries no
    # usable information anyway (a square looks the same at any rotation), so
    # close across the SMALLER of W and L instead.  Only a clearly elongated
    # footprint, where yaw is both meaningful and better constrained, keeps the
    # yaw-derived axis.
    footprint_ratio = max(w, l) / max(min(w, l), 1e-6)
    if footprint_ratio < SQUARE_FOOTPRINT_RATIO:
        grasp_axis = "along_x" if w <= l else "along_y"
    elif abs(yaw) < math.pi / 4 or abs(yaw) > 3 * math.pi / 4:
        grasp_axis = "along_x"
    else:
        grasp_axis = "along_y"

    # Recomputed from the CHOSEN axis, so gripper_fits below gates on the
    # dimension actually being spanned (including after the fallback above).
    narrow = w if grasp_axis == "along_x" else l

    return {
        "name":         d.get("class_name", "unknown"),
        "pos_xy":       [round(x, 3), round(y, 3)],
        "pos_xyz":      [round(x, 3), round(y, 3), round(z, 3)],
        "footprint_lw": [round(l, 3), round(w, 3)],
        "height_m":     round(h, 3),
        "dist_m":       round(dist, 3),
        "in_arm_reach": dist <= ARM_REACH_M,
        "is_target":    d.get("is_target", False),
        "yaw":          round(yaw, 4),
        "grasp_axis":   grasp_axis,
        "gripper_fits": narrow <= GRIPPER_W_M + 0.03,
    }


def find_blocking(target: Optional[dict], obstacles: List[dict],
                  gripper_w: float = GRIPPER_W_M) -> List[dict]:
    """Return obstacles that lie inside the gripper corridor to target."""
    if target is None:
        return []
    tx, ty = target["pos_xy"]
    dist   = target["dist_m"]
    if dist < 1e-6:
        return []

    dx, dy = tx / dist, ty / dist   # forward unit vector
    px, py = -dy, dx                # left perpendicular

    blocking = []
    for obs in obstacles:
        ox, oy  = obs["pos_xy"]
        ol, ow  = obs["footprint_lw"]
        obs_r   = math.hypot(ol, ow) / 2.0
        t_proj  = ox * dx + oy * dy
        lateral = ox * px + oy * py
        clearance = abs(lateral) - (gripper_w / 2.0 + obs_r)
        if 0.0 < t_proj < dist and clearance < 0.0:
            blocking.append({
                "name":        obs["name"],
                "clearance_m": round(clearance, 3),
                "side":        "left" if lateral > 0 else "right",
            })
    return blocking


def classify_zone(target: Optional[dict], blocking: List[dict], arm_reach: float = ARM_REACH_M,
                  table_near_x: float = None) -> str:
    """Classify the current approach zone.
    table_near_x: measured near edge of the table in base_link X.  When provided,
    drive_needed is also clamped so the robot stops at least TABLE_ROBOT_MARGIN
    before the table, whichever limit is tighter.
    """
    if target is None:
        return ZONE_OUT_OF_SIGHT
    if blocking:
        return ZONE_BLOCKED
    if abs(target["pos_xy"][1]) > LAT_THRESH_M:
        return ZONE_ROTATE_NEEDED
    drive_needed = max(0.0, target["dist_m"] - arm_reach + 0.05)
    if table_near_x is not None:
        # Cap drive distance so the robot keeps TABLE_ROBOT_MARGIN to the table edge
        drive_needed = min(drive_needed, max(0.0, table_near_x - TABLE_ROBOT_MARGIN))
    if drive_needed > 0.05:
        return ZONE_DRIVE_FORWARD
    return ZONE_GRASP_READY


def compute_grasp(target: Optional[dict],
                  table_surface_z: float = None) -> Optional[dict]:
    """Compute grasp geometry from a parsed target object, for the scene
    summary / bird's-eye preview shown to the VLM (NOT the executed grasp —
    that is computed independently from live dims, closer to execution time).
    Height uses grasp_geometry.grasp_height_above_table — THE single source of
    truth also used to actually execute the grasp — so this preview at least
    stays consistent with reality instead of using its own disconnected guess.
    table_surface_z: measured table surface height (base_link Z).
    Falls back to the module-level TABLE_SURFACE_Z constant when not provided.
    """
    if target is None:
        return None
    import grasp_geometry
    tx, ty = target["pos_xy"]
    l, w   = target["footprint_lw"]
    tz     = table_surface_z if table_surface_z is not None else TABLE_SURFACE_Z
    dims_hwl = [target.get("height_m"), w, l]
    try:
        grasp_z = tz + grasp_geometry.grasp_height_above_table(dims_hwl)
    except grasp_geometry.ObjectTooShortForSideGrasp:
        grasp_z = None   # no valid side-grip height; shown as "N/A" downstream
    return {
        "pre_grasp_xy":      [round(tx - 0.08, 3), round(ty, 3)],
        "grasp_xy":          [round(tx, 3),         round(ty, 3)],
        "grasp_z":           round(grasp_z, 3) if grasp_z is not None else None,
        "gripper_angle_deg": round(math.degrees(target["yaw"]), 1),
        "grasp_axis":        target["grasp_axis"],
        "gripper_fits":      target["gripper_fits"],
    }


def build_summary(target: Optional[dict], zone: str, blocking: List[dict],
                  grasp: Optional[dict],
                  table_surface_z: float = None) -> str:

    action = ZONE_TO_ACTION[zone].replace("_", " ").upper()

    if target is None:
        return f"[{action}] Target not detected."

    tx, ty = target["pos_xy"]
    dir_s  = ("ahead-left"  if ty < -LAT_THRESH_M else
              "ahead-right" if ty >  LAT_THRESH_M else
              "directly ahead")

    t_line = (f"{target['name'].capitalize()} {target['dist_m']:.2f}m "
              f"{dir_s} (lateral {ty:+.2f}m).")

    if blocking:
        b      = blocking[0]
        b_line = (f"{b['name']} blocks approach path "
                  f"(clearance {b['clearance_m']:.3f}m, on {b['side']}).")
    else:
        b_line = "Approach path clear."

    if grasp and zone == ZONE_GRASP_READY:
        z_str = f"{grasp['grasp_z']:.2f}m" if grasp['grasp_z'] is not None else "N/A (too short)"
        g_line = (f"Grasp: pre-grasp ({grasp['pre_grasp_xy'][0]:.2f},"
                  f"{grasp['pre_grasp_xy'][1]:.2f})m, "
                  f"descend to z={z_str}, "
                  f"gripper {grasp['gripper_angle_deg']:.0f}°, "
                  f"axis {grasp['grasp_axis'].replace('_', ' ')}.")
    else:
        g_line = ""

    tz_line = (f" [table z={table_surface_z:.3f}m]"
               if table_surface_z is not None else "")

    return f"[{action}] {t_line} {b_line} {g_line}{tz_line}".strip()
