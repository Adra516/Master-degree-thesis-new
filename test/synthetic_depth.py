"""Render depth images of simple scenes with a known answer, for testing the
grasp planner offline. Pure numpy ray casting; not a test module itself.

Frames: base_link (x forward, y left, z up). The camera follows ROS optical
conventions (x right, y down, z forward); `depth` holds the optical z of the
first hit, 0 where the ray hits nothing.
"""
import math

import numpy as np

W, H = 640, 480
HFOV = 0.994837                      # the sim camera's horizontal field of view
FX = (W / 2) / math.tan(HFOV / 2)
K = np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1.0]])


def look_at(eye, target):
    """4x4 transform optical frame -> base_link, camera at eye looking at target."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])     # right = forward x up
    x /= np.linalg.norm(x)
    y = np.cross(z, x)                   # down
    T = np.eye(4)
    T[:3, :3] = np.column_stack((x, y, z))
    T[:3, 3] = eye
    return T


class Cylinder:
    def __init__(self, cx, cy, radius, z0, z1):
        self.c, self.r, self.z0, self.z1 = np.array([cx, cy]), radius, z0, z1

    def hit(self, o, d):
        """Smallest t > 0 where o + t d meets the side or the top cap."""
        oc = o[:2] - self.c
        a = d[:, 0] ** 2 + d[:, 1] ** 2
        b = 2 * (oc[0] * d[:, 0] + oc[1] * d[:, 1])
        c = oc @ oc - self.r ** 2
        disc = b * b - 4 * a * c
        t = np.full(len(d), np.inf)
        ok = disc >= 0
        t_side = (-b[ok] - np.sqrt(disc[ok])) / (2 * a[ok])
        z = o[2] + t_side * d[ok, 2]
        good = (t_side > 0) & (z >= self.z0) & (z <= self.z1)
        idx = np.where(ok)[0][good]
        t[idx] = t_side[good]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_top = (self.z1 - o[2]) / d[:, 2]
        p = o[:2] + t_top[:, None] * d[:, :2]
        top = (t_top > 0) & (np.hypot(*(p - self.c).T) <= self.r)
        t[top] = np.minimum(t[top], t_top[top])
        return t


class Box:
    """Upright box: centre (cx, cy), size lx (along its own x) by ly, height
    z0..z1, turned by yaw about z."""

    def __init__(self, cx, cy, lx, ly, z0, z1, yaw=0.0):
        self.c, self.half = np.array([cx, cy, (z0 + z1) / 2]), np.array([lx, ly, z1 - z0]) / 2
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        self.R = np.array([[cy_, -sy_, 0], [sy_, cy_, 0], [0, 0, 1.0]])

    def hit(self, o, d):
        ol = self.R.T @ (o - self.c)
        dl = d @ self.R
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-self.half - ol) / dl
            t2 = (self.half - ol) / dl
        tmin = np.nanmax(np.minimum(t1, t2), axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        t = np.where((tmax >= tmin) & (tmin > 0), tmin, np.inf)
        return t


class Table:
    def __init__(self, z, x0, x1, y0, y1):
        self.z, self.x, self.y = z, (x0, x1), (y0, y1)

    def hit(self, o, d):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (self.z - o[2]) / d[:, 2]
        p = o[:2] + t[:, None] * d[:, :2]
        ok = (t > 0) & (p[:, 0] >= self.x[0]) & (p[:, 0] <= self.x[1]) \
            & (p[:, 1] >= self.y[0]) & (p[:, 1] <= self.y[1])
        return np.where(ok, t, np.inf)


def render(T, shapes):
    """Depth image (H x W, metres of optical z) of `shapes` seen through T."""
    v, u = np.mgrid[0:H, 0:W]
    rays = np.column_stack((((u - K[0, 2]) / K[0, 0]).ravel(),
                            ((v - K[1, 2]) / K[1, 1]).ravel(),
                            np.ones(W * H)))
    d = rays @ T[:3, :3].T            # optical z = 1 per unit t, so t IS the depth
    o = T[:3, 3]
    t = np.full(W * H, np.inf)
    for s in shapes:
        t = np.minimum(t, s.hit(o, d))
    t[~np.isfinite(t)] = 0.0
    return t.reshape(H, W)


def bbox_of(T, shape_points):
    """Pixel bbox [x1, y1, x2, y2] of 3-D points (base_link) seen through T."""
    p = (np.asarray(shape_points) - T[:3, 3]) @ T[:3, :3]
    u = K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]
    v = K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]
    return [u.min(), v.min(), u.max(), v.max()]


def corners(cx, cy, half_x, half_y, z0, z1):
    return [(cx + sx * half_x, cy + sy * half_y, z)
            for sx in (-1, 1) for sy in (-1, 1) for z in (z0, z1)]
