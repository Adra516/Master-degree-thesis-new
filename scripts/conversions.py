"""The one place that knows both shapes: the pure library's plain dicts and
lists, and the locobot_prog messages.

Ported near-verbatim from rework-attempt-1's src/locobot_prog/ros/
conversions.py. Pure geometry keeps its own conventions (dims_hwl = [H, W, L]);
messages use named axes (Vector3: x = depth extent, y = width, z = height).
Converting only here keeps "index 0 is height" from leaking into anything that
talks ROS.
"""
import math

from geometry_msgs.msg import Point, Quaternion, Vector3

from locobot_prog.msg import (CollisionBox, Detection3D, GraspPlan, TableModel,
                              WorldObject)


def plan_to_msg(plan):
    """dict from grasp_geometry.baseline_plan -> locobot_prog/GraspPlan."""
    m = GraspPlan()
    m.valid = bool(plan["valid"])
    m.reason = plan.get("reason", "")
    m.grasp_point = Point(*plan["grasp_point"])
    m.grasp_orientation = Quaternion(*plan["grasp_orientation"])
    m.standoff_offset = plan["standoff_offset"]
    m.standoff_offset_min = plan["standoff_offset_min"]
    m.gripper_open = plan["gripper_open"]
    m.gripper_close = plan["gripper_close"]
    m.full_close = bool(plan["full_close"])
    m.object_size = Vector3(*plan["object_size"])
    return m


def box_to_msg(box):
    """dict {name, x, y, z, sx, sy, sz} -> locobot_prog/CollisionBox."""
    return CollisionBox(name=box["name"], center=Point(box["x"], box["y"], box["z"]),
                        size=Vector3(box["sx"], box["sy"], box["sz"]))


def dims_hwl_from_size(size):
    """Vector3 (x = depth, y = width, z = height) -> [H, W, L]."""
    return [size.z, size.y, size.x]


def size_from_dims_hwl(dims_hwl):
    """[H, W, L] -> Vector3 (x = depth, y = width, z = height)."""
    h, w, l = dims_hwl
    return Vector3(l, w, h)


def nan_if_none(value, scale=1.0):
    """A missing measurement travels as NaN (float fields have no null)."""
    return float("nan") if value is None else float(value) * scale


def none_if_nan(value, scale=1.0):
    return None if value is None or math.isnan(value) else value * scale


# ── Detections and the world model ───────────────────────────────────────────

def detection_to_msg(d):
    """Perception's detection dict -> locobot_prog/Detection3D."""
    oid = d.get("object_id")
    return Detection3D(class_name=d["class_name"], confidence=d["confidence"],
                       track_id=-1 if oid is None else int(oid),
                       bbox=list(d["bbox_2d"]), depth=d["depth_m"],
                       position=Point(*d["position_base"]),
                       size=size_from_dims_hwl(d["dims_hwl"]),
                       ray_yaw=d.get("orientation", 0.0))


def detection_from_msg(m):
    """locobot_prog/Detection3D -> the dict shape the pure world model uses."""
    return {"class_name": m.class_name, "confidence": m.confidence,
            "object_id": None if m.track_id < 0 else m.track_id,
            "bbox_2d": list(m.bbox), "depth_m": m.depth,
            "position_base": [m.position.x, m.position.y, m.position.z],
            "dims_hwl": dims_hwl_from_size(m.size), "orientation": m.ray_yaw}


def world_object_to_msg(o, catalog_key=""):
    """zones.parse_object dict -> locobot_prog/WorldObject."""
    length, width = o["footprint_lw"]
    return WorldObject(name=o["name"], catalog_key=catalog_key, is_target=o["is_target"],
                       position=Point(*o["pos_xyz"]),
                       size=Vector3(length, width, o["height_m"]),
                       distance=o["dist_m"], yaw=o["yaw"], grasp_axis=o["grasp_axis"],
                       gripper_fits=o["gripper_fits"], in_arm_reach=o["in_arm_reach"])


def table_to_msg(t):
    """world_table.TableEstimate -> locobot_prog/TableModel."""
    edge = t.table_near_x is not None
    return TableModel(surface_known=t.table_surface_z is not None,
                      surface_z=t.table_z_for_grasp(),
                      edge_known=edge,
                      near_x=t.table_near_x if edge else 0.0,
                      far_x=t.table_far_x if edge else 0.0,
                      width=t.table_width_m or 0.0, depth=t.table_depth_m or 0.0)


def vec_or_zero(values):
    """[H, W, L] or None -> (Vector3 size, known)."""
    if values is None:
        return Vector3(), False
    return size_from_dims_hwl(values), True
