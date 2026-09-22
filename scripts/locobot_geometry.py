"""locobot_geometry.py — pure geometry utilities for LoCoBot manipulation.
All functions are module-level (no ROS, no class) so they can be imported
by any node without pulling in ROS dependencies.
"""

import math
from typing import List, Optional, Tuple
import cv2
import numpy as np

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
# threshold in estimate_grasp_orientation, which returns yaw = 0 below this ratio.
SQUARE_FOOTPRINT_RATIO = 1.3

# Bird's-eye rendering constants 
PX_PER_M = 100
CANVAS_W = 600
CANVAS_H = 520
ROBOT_PX = (CANVAS_W // 2, CANVAS_H - 50)

# Zone constants
ZONE_GRASP_READY   = "grasp_ready"
ZONE_DRIVE_FORWARD = "drive_forward"
ZONE_ROTATE_NEEDED = "rotate_needed"
ZONE_BLOCKED       = "path_blocked"
ZONE_OUT_OF_SIGHT  = "target_not_found"

ZONE_COLORS = {
    ZONE_GRASP_READY:   (0, 220, 0),
    ZONE_DRIVE_FORWARD: (0, 180, 255),
    ZONE_ROTATE_NEEDED: (0, 120, 255),
    ZONE_BLOCKED:       (0, 0, 220),
    ZONE_OUT_OF_SIGHT:  (80, 80, 80),
}

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
    that is computed independently in vlm_agent_node.get_object() from live
    dims, closer to execution time).
    Height uses grasp_geometry.grasp_height_above_table — THE single source of
    truth also used to actually execute the grasp — so this preview at least
    stays consistent with reality instead of using its own disconnected guess.
    (Previously a hardcoded `table_z + 0.02`, unrelated to any object's real
    height; harmless to the grasp itself since this value was never sent to
    the motion node, but it showed a wrong number to the VLM.)
    table_surface_z: measured table surface height (base_link Z).
    Falls back to the module-level TABLE_SURFACE_Z constant when not provided.
    """
    if target is None:
        return None
    import grasp_geometry   # local: avoids a module-load-order dependency
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


def estimate_grasp_orientation(img_bgr: np.ndarray, bbox_2d: list,
                               fallback_yaw: float = 0.0,
                               depth_img: np.ndarray = None,
                               depth_m: float = None,
                               dims_hwl: list = None,
                               orientation: float = None) -> float:
    """Estimate gripper yaw fusing depth-masked minAreaRect with bbox3d priors.
    Two fusions over the plain image-only approach:
    1. Physical symmetry check (dims_hwl):
       The footprint ratio max(w,l)/min(w,l) from the known class dimensions is
       tested BEFORE any image processing.  If < 1.3 the object is symmetric
       (bottle, vase, cup) and yaw is meaningless → return 0.0 immediately.
       This is more reliable than measuring the contour itself, which can be
       distorted by depth holes or partial occlusion.
    2. 180° disambiguation (orientation / theta_ray from bbox3d):
       minAreaRect returns the long-axis angle but cannot tell which end is
       "forward" — both yaw and yaw±π fit the same rectangle.  The bbox3d
       orientation field (≈ camera ray angle to the object) is used to pick
       the candidate closest to the viewing direction, resolving the flip.

    Falls back gracefully at every step: dims_hwl absent → skip check 1;
    orientation absent → skip check 2; depth absent → use Canny edges.

    Args:
        img_bgr:      Full BGR image (any resolution).
        bbox_2d:      [x1, y1, x2, y2] pixel bounding box.
        fallback_yaw: Returned when estimation cannot produce a result.
        depth_img:    Float32 depth image in metres, aligned to img_bgr.
        depth_m:      Median depth of the object in metres (from _sample_depth).
        dims_hwl:     [height, width, length] physical dimensions from YOLO3D.
        orientation:  bbox3d global yaw estimate (theta_ray, radians).

    Returns:
        yaw_rad in [-π/2, π/2].
    """
    # ── 1. Physical symmetry check — skip image processing for round objects ──
    if dims_hwl is not None:
        _, w, l = dims_hwl
        footprint_ratio = max(w, l) / max(min(w, l), 1e-6)
        if footprint_ratio < 1.3:
            return 0.0   # symmetric footprint (bottle, vase, cup) → yaw undefined

    x1, y1, x2, y2 = (int(v) for v in bbox_2d[:4])
    h_img, w_img   = img_bgr.shape[:2]

    pad = 4
    x1c = max(0, x1 - pad);  y1c = max(0, y1 - pad)
    x2c = min(w_img, x2 + pad); y2c = min(h_img, y2 + pad)
    if (x2c - x1c) < 10 or (y2c - y1c) < 10:
        return fallback_yaw

    if depth_img is not None and depth_m is not None and depth_m > 0.0:
        # Depth-masked path: keep only pixels within ±4 cm of the object depth
        depth_crop = depth_img[y1c:y2c, x1c:x2c]
        mask = cv2.inRange(depth_crop,
                           float(depth_m - 0.04),
                           float(depth_m + 0.04))
    else:
        # Canny fallback when no depth is available
        crop  = img_bgr[y1c:y2c, x1c:x2c]
        gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mask  = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 120)

    # Take the largest contour — most likely the object silhouette
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fallback_yaw

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return fallback_yaw

    # Fit a minimum-area rectangle to the contour
    _, (rw, rh), angle_deg = cv2.minAreaRect(contour)

    if min(rw, rh) < 1.0:
        return fallback_yaw

    # Image-derived symmetry guard (used when dims_hwl was not provided)
    if max(rw, rh) / min(rw, rh) < 1.3:
        return 0.0

    # minAreaRect returns angle in [-90, 0]; when width < height the long axis
    # is the height direction, so add 90° to align with the long axis.
    if rw < rh:
        angle_deg += 90.0

    yaw_rad = math.radians(angle_deg)
    # Normalise to [-π/2, π/2]
    while yaw_rad >  math.pi / 2.0: yaw_rad -= math.pi
    while yaw_rad < -math.pi / 2.0: yaw_rad += math.pi

    # ── 2. 180° disambiguation using bbox3d global orientation (theta_ray) ───
    # minAreaRect can't distinguish yaw from yaw±π (same rectangle either way).
    # The bbox3d orientation ≈ camera ray angle, which is close to 0 for objects
    # roughly centred in front of the robot.  Pick whichever of the two
    # equivalent long-axis directions is closer to that global estimate.
    if orientation is not None:
        alt_rad = yaw_rad + (math.pi if yaw_rad <= 0.0 else -math.pi)
        if abs(alt_rad - orientation) < abs(yaw_rad - orientation):
            yaw_rad = alt_rad
        # Re-normalise after potential flip
        while yaw_rad >  math.pi / 2.0: yaw_rad -= math.pi
        while yaw_rad < -math.pi / 2.0: yaw_rad += math.pi

    return float(yaw_rad)


def _w2px(x: float, y: float) -> Tuple[int, int]:
    return (ROBOT_PX[0] - int(y * PX_PER_M),
            ROBOT_PX[1] - int(x * PX_PER_M))


def render_birdseye(objects: List[dict], target: Optional[dict],
                    blocking: List[dict], zone: str,
                    grasp: Optional[dict],
                    table_bbox_m: tuple = None) -> np.ndarray:
    """Render annotated top-down map; returns BGR ndarray (no ROS publishing).
    table_bbox_m: optional (near_x, far_x, left_y, right_y) in base_link metres.
    When provided, draws the table as a filled grey rectangle beneath all objects.
    """
    img        = np.full((CANVAS_H, CANVAS_W, 3), 18, dtype=np.uint8)
    zone_color = ZONE_COLORS[zone]
    block_names = {b["name"] for b in blocking}

    # Grid
    for d in range(7):
        yp = ROBOT_PX[1] - int(d * PX_PER_M)
        if 0 <= yp < CANVAS_H:
            cv2.line(img, (0, yp), (CANVAS_W, yp), (38, 38, 38), 1)
            cv2.putText(img, f"{d}m", (4, yp - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (65, 65, 65), 1)
    for yd in range(-3, 4):
        xp = ROBOT_PX[0] - int(yd * PX_PER_M)
        if 0 <= xp < CANVAS_W:
            cv2.line(img, (xp, 0), (xp, CANVAS_H), (38, 38, 38), 1)

    # Arm reach arc
    cv2.circle(img, ROBOT_PX, int(ARM_REACH_M * PX_PER_M),
               (0, 160, 160), 1, cv2.LINE_AA)

    # Zone border
    cv2.rectangle(img, (2, 2), (CANVAS_W - 2, CANVAS_H - 2), zone_color, 3)

    # Table footprint without object on it
    if table_bbox_m is not None:
        near_x, far_x, left_y, right_y = table_bbox_m
        tbl_pts = np.array([
            _w2px(near_x, left_y),
            _w2px(near_x, right_y),
            _w2px(far_x,  right_y),
            _w2px(far_x,  left_y),
        ], np.int32)
        ov = img.copy()
        cv2.fillPoly(ov, [tbl_pts], (70, 70, 70))
        cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
        cv2.polylines(img, [tbl_pts], True, (130, 130, 130), 1)

    # Gripper corridor + approach line
    if target:
        t_px    = _w2px(*target["pos_xy"])
        tx, ty  = target["pos_xy"]
        dist    = target["dist_m"]
        dx, dy  = (tx / dist, ty / dist) if dist > 1e-6 else (1, 0)
        half_w  = int(GRIPPER_W_M / 2.0 * PX_PER_M)
        perp_px = int(-dy * half_w)
        perp_py = int( dx * half_w)

        corridor = np.array([
            [ROBOT_PX[0] + perp_px, ROBOT_PX[1] + perp_py],
            [t_px[0]     + perp_px, t_px[1]     + perp_py],
            [t_px[0]     - perp_px, t_px[1]     - perp_py],
            [ROBOT_PX[0] - perp_px, ROBOT_PX[1] - perp_py],
        ], np.int32)
        ov = img.copy()
        cv2.fillPoly(ov, [corridor], (0, 55, 0))
        cv2.addWeighted(ov, 0.35, img, 0.65, 0, img)
        cv2.polylines(img, [corridor], True, zone_color, 1)
        cv2.line(img, ROBOT_PX, t_px, zone_color, 2, cv2.LINE_AA)

        if grasp:
            pg_px = _w2px(*grasp["pre_grasp_xy"])
            gp_px = _w2px(*grasp["grasp_xy"])
            cv2.circle(img, pg_px, 5, (255, 200, 0), -1)
            cv2.circle(img, gp_px, 6, (0, 255, 255), -1)
            cv2.putText(img, "pre",
                        (pg_px[0] + 5, pg_px[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.26, (255, 200, 0), 1)
            cv2.putText(img, "grasp",
                        (gp_px[0] + 5, gp_px[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.26, (0, 255, 255), 1)

    for obj in objects:
        cpx  = _w2px(*obj["pos_xy"])
        l_px = max(int(obj["footprint_lw"][0] * PX_PER_M), 6)
        w_px = max(int(obj["footprint_lw"][1] * PX_PER_M), 6)
        angle       = -(90.0 + math.degrees(obj["yaw"]))
        is_blocking = obj["name"] in block_names

        if obj["is_target"]:
            fill, lc = (180, 80, 0), (255, 200, 80)
        elif is_blocking:
            fill, lc = (0, 0, 180), (100, 100, 255)
        else:
            fill, lc = (30, 120, 30), (140, 220, 140)

        box = cv2.boxPoints((cpx, (w_px, l_px), angle)).astype(int)
        cv2.drawContours(img, [box], 0, fill, -1)
        cv2.drawContours(img, [box], 0, (190, 190, 190), 1)

        if obj["is_target"] and grasp:
            hl = max(l_px, w_px) // 2 + 4
            if obj["grasp_axis"] == "along_x":
                cv2.line(img, (cpx[0] - hl, cpx[1]),
                         (cpx[0] + hl, cpx[1]), (0, 240, 240), 1)
            else:
                cv2.line(img, (cpx[0], cpx[1] - hl),
                         (cpx[0], cpx[1] + hl), (0, 240, 240), 1)

        cv2.putText(img, obj["name"][:9],
                    (cpx[0] - 16, cpx[1] - l_px // 2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, lc, 1)
        cv2.putText(img, f"{obj['dist_m']:.2f}m",
                    (cpx[0] - 13, cpx[1] + l_px // 2 + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (190, 190, 90), 1)

    cv2.circle(img, ROBOT_PX, 7, (210, 210, 210), -1)
    cv2.arrowedLine(img, ROBOT_PX, (ROBOT_PX[0], ROBOT_PX[1] - 20),
                    (210, 210, 210), 2, tipLength=0.38)
    cv2.putText(img, zone.replace("_", " ").upper(), (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, zone_color, 1)
    return img
