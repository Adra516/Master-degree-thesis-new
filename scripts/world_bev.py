"""Bird's-eye view: a top-down picture of the world model for the VLM and for
debugging (rqt_image_view on the world model's bev_image topic).

Ported near-verbatim from rework-attempt-1's src/locobot_prog/world/bev.py,
itself "moved from scripts/locobot_geometry.py" — verified identical to
locobot_geometry.py's render_birdseye before porting. Named world_bev.py (not
bev.py) to avoid ambiguity with unrelated names elsewhere. locobot_geometry.
py's own copy was deleted once its only caller, vlm_agent_node.py, was
retired — this is its sole surviving copy. Pure: OpenCV and numpy, no ROS.
"""
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from zones import (ARM_REACH_M, GRIPPER_W_M, ZONE_BLOCKED, ZONE_DRIVE_FORWARD,
                   ZONE_GRASP_READY, ZONE_OUT_OF_SIGHT, ZONE_ROTATE_NEEDED)

# Bird's-eye rendering constants
PX_PER_M = 100
CANVAS_W = 600
CANVAS_H = 520
ROBOT_PX = (CANVAS_W // 2, CANVAS_H - 50)

ZONE_COLORS = {
    ZONE_GRASP_READY:   (0, 220, 0),
    ZONE_DRIVE_FORWARD: (0, 180, 255),
    ZONE_ROTATE_NEEDED: (0, 120, 255),
    ZONE_BLOCKED:       (0, 0, 220),
    ZONE_OUT_OF_SIGHT:  (80, 80, 80),
}


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
