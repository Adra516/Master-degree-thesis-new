"""The table, estimated from "dining table" detections.

Ported near-verbatim from rework-attempt-1's src/locobot_prog/world/table.py
("moved from scripts/locobot_scene_model.py, where the class was called
SceneModel although it only ever modelled the table" — that file's only
caller, vlm_agent_node.py, has since been retired, so locobot_scene_model.py
was deleted; this is its sole surviving copy). Named world_table.py here (not
table.py) to avoid colliding with the unrelated table_box() helper already in
grasp_geometry.py. Pure: no ROS.
"""
from typing import Optional

import numpy as np

from zones import TABLE_SURFACE_Z


class TableEstimate:

    def __init__(self):
        self.table_surface_z: Optional[float] = None
        self.table_near_x:    Optional[float] = None
        self.table_far_x:     Optional[float] = None
        self.table_width_m:   Optional[float] = None
        self.table_depth_m:   Optional[float] = None
        self._ty_center:      float           = 0.0

    def update_table(self, det: dict, K: np.ndarray) -> None:
        """Refresh table geometry from a 'dining table' detection + camera K matrix.

        Width (horizontal bbox extent) is reliable from pinhole projection because
        the camera sees the full lateral span of the table.

        Depth (forward extent) is NOT reliable this way: a forward-facing camera
        sees only the thin front edge of a horizontal surface, so
        (bbox_height_px / fy) * depth gives the edge thickness (~5 cm), not the
        true table depth.  We use a conservative 0.60 m prior and only accept
        larger measured values (from dims[2] if available from DEFAULT_DIMS).

        Surface Z is also unreliable because the bbox centre often aligns with
        objects sitting ON the table, not the table surface itself.  We accept the
        YOLO-derived value only if it is within ±10 cm of the world-file constant.
        """
        pos   = det.get("position_base")
        depth = float(det.get("depth_m", 0.0))
        bbox  = det.get("bbox_2d")
        dims  = det.get("dims_hwl")
        if not (pos and depth > 0.1 and bbox and K is not None):
            return

        x1, y1, x2, y2 = bbox
        fx = float(K[0, 0])

        # Width: reliable; floor at 0.60 m so the collision box is never too narrow
        W_m = max(((x2 - x1) / fx) * depth, 0.60)

        # Depth: use the DEFAULT_DIMS L value when available, else a 0.60 m prior.
        # Only accept values larger than the prior (measured value could be L from
        # DEFAULT_DIMS which might encode a more accurate table depth).
        D_prior = 0.60
        D_m = max(D_prior, float(dims[2]) if dims else D_prior)

        tx, ty, tz = pos
        h = float(dims[0]) if dims else 0.05

        self.table_near_x  = tx - D_m / 2.0
        self.table_far_x   = tx + D_m / 2.0
        self.table_depth_m = D_m
        self.table_width_m = W_m
        self._ty_center    = ty

        # surface_z: only use YOLO-measured value if it is close to the world-file
        # constant (TABLE_SURFACE_Z).  When objects are on the table the bbox
        # centre lands on those objects and tz is too high.
        measured_sz = tz + h / 2.0
        self.table_surface_z = (measured_sz
                                if abs(measured_sz - TABLE_SURFACE_Z) < 0.10
                                else TABLE_SURFACE_Z)

    def table_near_edge(self) -> Optional[float]:
        """Return the measured near-edge X in base_link, or None if not yet seen."""
        return self.table_near_x

    def table_z_for_grasp(self) -> float:
        """Return measured table surface Z; falls back to TABLE_SURFACE_Z constant."""
        return self.table_surface_z if self.table_surface_z is not None else TABLE_SURFACE_Z

    @property
    def table_bbox_m(self) -> Optional[tuple]:
        """(near_x, far_x, left_y, right_y) in base_link metres, or None."""
        if self.table_near_x is None or self.table_far_x is None or self.table_width_m is None:
            return None
        hw = self.table_width_m / 2.0
        return (self.table_near_x, self.table_far_x,
                self._ty_center + hw, self._ty_center - hw)
