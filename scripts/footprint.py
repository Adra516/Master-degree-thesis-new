"""The target's points, seen from above: depth image -> 3-D points in base_link,
cut to the object and sliced by height. Pure: numpy only.

This is the "bird's-eye view obtained by geometric projection from above" at
object scale: every depth pixel inside the detection box is back-projected,
moved into the robot's frame, stripped of the table and the background, and
the remaining points are looked at from the top, one height band at a time.

A single camera sees only the FRONT of an object. What it measures well is the
lateral extent (the silhouette, which is exactly what the jaws close across);
the depth extent along the viewing ray is only partly visible. Consumers must
not treat a band's depth extent as the object's.
"""
from collections import namedtuple

import numpy as np

# One height band of the object, in approach coordinates:
#   a = distance along the approach direction (the bearing), s = lateral (left +).
Band = namedtuple("Band", "z n width s_left s_right a_front a_back xy")


def backproject(depth, bbox, K, pad=4, near=0.10, far=2.0):
    """Pixels of `depth` (metres) inside bbox [x1, y1, x2, y2] -> (N, 3) points in
    the camera's optical frame (x right, y down, z forward)."""
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox[:4])
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 3))
    d = depth[y1:y2, x1:x2].astype(np.float64)
    v, u = np.mgrid[y1:y2, x1:x2]
    ok = np.isfinite(d) & (d > near) & (d < far)
    d, u, v = d[ok], u[ok], v[ok]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.column_stack(((u - cx) * d / fx, (v - cy) * d / fy, d))


def transform(points, T):
    """(N, 3) points through a 4x4 homogeneous transform."""
    return points @ T[:3, :3].T + T[:3, 3]


def object_points(points_base, table_z, center_xy=None, radius=None,
                  min_above=0.005, max_above=0.5):
    """Keep the points that belong to the object: above the table surface (by
    min_above, so the table itself goes), below max_above, and — when a centre
    is known — within `radius` of it horizontally, which drops the background
    and any neighbour sharing the detection box."""
    p = points_base
    z = p[:, 2] - table_z
    keep = (z > min_above) & (z < max_above)
    if center_xy is not None and radius is not None:
        keep &= np.hypot(p[:, 0] - center_xy[0], p[:, 1] - center_xy[1]) < radius
    return p[keep]


def table_height(depth, bbox, K, T, center_xy, radius, guess_z, window=0.08,
                 bin_h=0.004, min_points=300):
    """Height (base_link z) of the surface the target stands on, MEASURED next to
    it — or None when the picture cannot say.

    The prior (zones.TABLE_SURFACE_Z) is a constant for one world file, and in
    the sim it was 22 mm off the table the depth camera actually sees (0.250 vs
    0.272 in base_link, checked against Gazebo ground truth: the camera's own
    back-projection is accurate to a few mm). Everything the grasp derives from
    "height above the table" (the search window, the anchor, the object's height)
    inherited that error, so the jaws came down too low on every tall object.

    The support surface is looked for in a strip of the depth image around and
    below the target's box (the table in front of and beside it), keeping the
    points within `window` of the prior and outside `radius` of the target. The
    table is the LOWEST surface there with real support: objects standing on it
    only add higher points, so the lowest well-populated height bin wins over the
    fullest one, and its points' median gives the height."""
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    bw, bh = x2 - x1, y2 - y1
    region = [x1 - bw, y1 + 0.5 * bh, x2 + bw, y2 + 0.6 * bh]
    cloud = transform(backproject(depth, region, K, pad=0), T)
    if len(cloud) == 0:
        return None
    near = np.hypot(cloud[:, 0] - center_xy[0], cloud[:, 1] - center_xy[1]) <= radius
    z = cloud[~near, 2]
    z = z[np.abs(z - guess_z) < window]
    if len(z) < min_points:
        return None
    edges = np.arange(guess_z - window, guess_z + window + bin_h, bin_h)
    counts, edges = np.histogram(z, bins=edges)
    solid = np.flatnonzero(counts >= max(0.3 * counts.max(), min_points / 4))
    if len(solid) == 0:
        return None
    k = int(solid[0])
    centre = 0.5 * (edges[k] + edges[k + 1])
    return float(np.median(z[np.abs(z - centre) < 1.5 * bin_h]))


def _extent(v, trim):
    v = np.sort(v)
    return v[trim], v[-1 - trim]


def bands(points, table_z, bearing, slice_h=0.005, min_points=12, trim=2):
    """Slice the object into horizontal bands and measure each in approach
    coordinates. The `trim` most extreme points are dropped at each end of
    every extent, so a couple of stray pixels cannot widen the object. A count,
    not a percentage: silhouette edges are sparse (seen at a grazing angle), so
    a percentage cuts millimetres off the real object, and more off the side
    seen more obliquely."""
    if len(points) == 0:
        return []
    ahat = np.array([np.cos(bearing), np.sin(bearing)])
    lhat = np.array([-np.sin(bearing), np.cos(bearing)])
    z = points[:, 2] - table_z
    out = []
    top = z.max()
    k = 0
    while k * slice_h < top:
        sel = (z >= k * slice_h) & (z < (k + 1) * slice_h)
        xy = points[sel, :2]
        if len(xy) >= min_points:
            s_lo, s_hi = _extent(xy @ lhat, trim)
            a_lo, a_hi = _extent(xy @ ahat, trim)
            out.append(Band(z=(k + 0.5) * slice_h, n=len(xy), width=s_hi - s_lo,
                            s_left=s_hi, s_right=s_lo, a_front=a_lo, a_back=a_hi, xy=xy))
        k += 1
    return out
