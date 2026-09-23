"""locobot_geometry.py — 2D-image grasp-yaw estimation, the one piece of the
original monolithic geometry module still in use.

Everything else this file used to hold (parse_object, find_blocking,
classify_zone, compute_grasp, build_summary, render_birdseye/_w2px, and the
zone/canvas constants they used) was the scene-classification and
bird's-eye-preview logic written for the retired vlm_agent_node.py. It has
since been superseded by zones.py, world_table.py and world_bev.py in the
world_model_node graph, and was deleted here as dead code (no remaining
importer — verified via grep before removal). estimate_grasp_orientation is
the one function detection_shim.py still calls; it has no dependency on any
of the removed code.
"""

import math
import cv2
import numpy as np


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
