"""The world model: what is where, built up from detections over time.

Ported near-verbatim from rework-attempt-1's src/locobot_prog/world/model.py
("moved from the VLM agent, which kept this state in a dozen attributes of
its own and rebuilt the scene on every call"). Here it is one object, fed
every detection frame, and asked for scenes and target estimates. Pure:
timestamps are passed in, TF-dependent pieces (map lock, odometry cache) stay
in the node.

Detections are plain dicts: class_name, confidence, position_base [x, y, z],
dims_hwl [H, W, L], bbox_2d, depth_m.
"""
import math
import statistics
from collections import deque, namedtuple

from grasp_geometry import obstacle_boxes
from zones import (ARM_GRASP_DIST, GRIPPER_W_M, classify_zone, compute_grasp,
                   build_summary, find_blocking, parse_object)
from resolver import normalize
from world_table import TableEstimate

# Size calibration uses only frames taken from at least this far away: MEASURED
# that both the bbox height and the centre-derived estimate get WORSE closer in
# (the centre one 33-62% over true in the 0.40-0.55 m band), so a size locked in
# from farther away beats re-deriving a worse one at the grasp standoff.
DIM_CALIB_MIN_DIST = 0.47
DIM_CALIB_MIN_SAMPLES = 3
DIM_CALIB_WINDOW = 20
# Recent target positions kept for the median: a single frame's bbox jitter
# (~cm) is what occasionally clipped the tight bottle grasp.
# YOLO3D cannot measure the size of a class it has no reason to know: it fills
# dims from a per-class prior (a "person" is 0.6 x 0.6 m). A small object the
# detector mislabels then becomes a table-sized obstacle — seen live as
# obs_person_0, 0.64 m square, whose box contained the arm's own ready pose, so
# every pre-grasp Cartesian path started "in collision" (0% coverage), and the
# same size blocked the approach corridor ("path blocked by person"). Anything
# that is neither a catalogued object nor the table surface is capped to what can
# plausibly stand on this table (the largest catalogued object is 0.07 m).
MAX_UNCATALOGUED_FOOTPRINT = 0.08
POSITION_BUFFER = 8
RECENT_WINDOW_S = 1.5
# How long an empty frame may be answered with the last non-empty one.
DETECTION_CACHE_TTL_S = 30.0

Scene = namedtuple("Scene", "objects target blocking zone grasp summary "
                            "detection_count from_cache visible")


def sane_target(d):
    """Reject implausible detections before locking onto them (a real run once
    locked 'vase @ [1.878, -0.201, -0.012] dims 0.80x0.25')."""
    px, py, pz = d["position_base"]
    h, w = float(d["dims_hwl"][0]), float(d["dims_hwl"][1])
    return 0.0 <= pz <= 0.8 and math.hypot(px, py) < 2.0 and h < 0.45 and w < 0.35


def _usable(d):
    return bool(d.get("position_base") and d.get("dims_hwl") and d.get("class_name"))


class WorldModel:

    def __init__(self, resolver, catalog, default_target):
        self.resolver = resolver
        self.catalog = catalog
        self.table = TableEstimate()
        self.latest = []                     # detections of the latest frame
        self._cache, self._cache_t = [], -math.inf
        self._calib = {}                     # calibration key -> deque[(dist, H, W, L)]
        self.target_name = default_target
        self._reset_target()

    # ── The target being tracked ─────────────────────────────────────────────

    def _reset_target(self):
        self.lock = None                     # last accepted detection of the target
        self.lock_t = -math.inf
        self.recent = deque(maxlen=POSITION_BUFFER)   # (t, x, y, z), base_link

    def set_target(self, name):
        """Track `name` from now on. Returns True if that changed the target
        (the lock and position history of the old one are dropped: they
        describe a different object)."""
        if not name or normalize(name) == normalize(self.target_name):
            return False
        self.target_name = name
        self._reset_target()
        return True

    def forget_target(self):
        """Drop the lock after a successful grasp: the object has moved."""
        self.lock = None
        self.lock_t = -math.inf

    # ── Ingest ───────────────────────────────────────────────────────────────

    def calib_key(self, label):
        """Calibrate per catalogue object when the label names exactly one, so
        'bottle' and 'vase' detections of the same bottle share one size."""
        keys = self.catalog.keys_for_label(label)
        return keys[0] if len(keys) == 1 else normalize(label)

    def update(self, detections, now, K=None):
        """Ingest one detection frame. Returns the accepted target detection
        (also stored as self.lock), or None."""
        surface = self.catalog.surface_labels()
        detections = [self._plausible_size(d, surface) for d in detections]
        self.latest = detections
        if detections:
            self._cache, self._cache_t = detections, now
        for d in detections:
            label = normalize(d.get("class_name", ""))
            if _usable(d):
                px, py, _ = d["position_base"]
                dist = math.hypot(px, py)
                if dist >= DIM_CALIB_MIN_DIST:
                    h, w, l = (float(v) for v in d["dims_hwl"])
                    self._calib.setdefault(self.calib_key(label),
                                           deque(maxlen=DIM_CALIB_WINDOW)).append((dist, h, w, l))
            if label in surface and K is not None:
                self.table.update_table(d, K)

        candidates = [d for d in detections if _usable(d) and sane_target(d)
                      and self.resolver.matches(self.target_name, d)]
        r = self.resolver.resolve(self.target_name, candidates)
        if r.index < 0:
            return None
        d = candidates[r.index]
        self.lock = dict(d, is_target=True)
        self.lock_t = now
        self.recent.append((now, *(float(v) for v in d["position_base"])))
        return self.lock

    def _plausible_size(self, d, surface):
        """`d` with an implausibly large footprint capped, for labels that are
        neither catalogued nor a surface (see MAX_UNCATALOGUED_FOOTPRINT)."""
        label = normalize(d.get("class_name", ""))
        dims = d.get("dims_hwl")
        if not dims or label in surface or self.catalog.keys_for_label(label):
            return d
        h, w, l = (float(v) for v in dims)
        if max(w, l) <= MAX_UNCATALOGUED_FOOTPRINT:
            return d
        return dict(d, dims_hwl=[h, min(w, MAX_UNCATALOGUED_FOOTPRINT),
                                 min(l, MAX_UNCATALOGUED_FOOTPRINT)])

    def calibrated_dims(self, key):
        """Median [H, W, L] from good-vantage frames, or None if too few."""
        buf = self._calib.get(key)
        if not buf or len(buf) < DIM_CALIB_MIN_SAMPLES:
            return None
        return [statistics.median(s[i] for s in buf) for i in (1, 2, 3)]

    def target_calibrated_dims(self):
        key = self.catalog.key_for_name(self.target_name) or normalize(self.target_name)
        return self.calibrated_dims(key)

    # ── Queries ──────────────────────────────────────────────────────────────

    def frame_target(self):
        """The target's best detection in the latest frame, or None."""
        cands = [d for d in self.latest if self.resolver.matches(self.target_name, d)]
        r = self.resolver.resolve(self.target_name, cands)
        return cands[r.index] if r.index >= 0 else None

    def recent_positions(self, now):
        return [(x, y) for (t, x, y, _) in self.recent if now - t < RECENT_WINDOW_S]

    def scene(self, now, nav_x_limit=None, standoff=None, use_cache=False):
        """Objects, the target, what blocks the approach and the zone — for the
        target being tracked. The lock is included even when the target is out
        of the current frame, so it survives the camera losing it."""
        src, from_cache = self.latest, False
        if use_cache and not src and self._cache and now - self._cache_t < DETECTION_CACHE_TTL_S:
            src, from_cache = self._cache, True
        dets = [dict(d, is_target=self.resolver.matches(self.target_name, d))
                for d in src if _usable(d)]
        best = self.resolver.resolve(self.target_name, [d for d in dets if d["is_target"]])
        if self.lock is not None:
            dets.append(dict(self.lock, is_target=True))
        objects = [parse_object(d) for d in dets]

        target = None
        if best.index >= 0:
            target = [o for o in objects if o["is_target"]][best.index]
        elif self.lock is not None:
            target = objects[-1]

        nav_obstacles = [o for o in objects if not o["is_target"]
                         and (nav_x_limit is None or o["pos_xy"][0] < nav_x_limit)]
        blocking = find_blocking(target, nav_obstacles, GRIPPER_W_M)
        # The agent always passed its current standoff here (default 0.45 m).
        zone = classify_zone(target, blocking,
                             ARM_GRASP_DIST if standoff is None else standoff,
                             table_near_x=nav_x_limit)
        tsz = self.table.table_z_for_grasp()
        grasp = compute_grasp(target, table_surface_z=tsz)
        summary = build_summary(target, zone, blocking, grasp,
                                table_surface_z=tsz if self.table.table_surface_z is not None
                                else None)
        visible = sorted({d.get("class_name", "?") for d in src if d.get("class_name")})
        return Scene(objects, target, blocking, zone, grasp, summary,
                     len(src), from_cache, visible)

    def obstacles(self):
        """Collision boxes for every NON-target, NON-surface object in the
        latest frame. A label that names exactly one catalogued object takes
        that object's catalogue size: YOLO reads these small objects 8-11 mm
        too wide (a 45 mm can as 56 mm), which narrows the corridor between
        neighbours for nothing. Otherwise the size comes from the calibration
        where available (a close-range frame is the worse size estimate, by
        the same measurement that motivates the calibration).

        Surface labels (the table) are excluded here, not just left uncapped:
        the table already has its own dedicated collision box (table_box(),
        built from TableEstimate). _plausible_size() deliberately does NOT cap
        a surface detection's size — capping it would corrupt the table
        model's own real, large footprint — but that same exemption means an
        imprecise "dining table" detection fed through here as a genericobject
        obstacle instead can come out as a huge, wrongly-centred phantom box
        (measured live: 0.51 x 0.51 x 0.21 m, centred squarely in the
        approach corridor) that blocks the whole pre-grasp path — a duplicate
        of the table that's also uncapped, the worst of both."""
        surface = self.catalog.surface_labels()
        dets = []
        for d in self.latest:
            label = normalize(d.get("class_name", ""))
            if label in surface:
                continue
            keys = self.catalog.keys_for_label(label) if label else []
            size = self.catalog.size(keys[0]) if len(keys) == 1 else None
            cd = ([size[2], size[1], size[0]] if size is not None      # (x, y, z) -> [H, W, L]
                  else self.calibrated_dims(self.calib_key(label)) if label else None)
            d2 = dict(d, is_target=self.resolver.matches(self.target_name, d))
            if cd is not None:
                d2["dims_hwl"] = cd
            dets.append(d2)
        return obstacle_boxes(dets, "", self.table.table_z_for_grasp())
