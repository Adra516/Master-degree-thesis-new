"""A misclassified small object must not become a table-sized obstacle.

Regression for obs_person_0: YOLO3D's per-class prior sized a "person" at
0.6 x 0.6 m, the resulting 0.64 m box contained the arm's ready pose, and every
pre-grasp Cartesian path then started in collision (0% coverage).
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
from resolver import CatalogResolver, ObjectCatalog  # noqa: E402
from grasp_geometry import OBSTACLE_PAD  # noqa: E402
from world_model import MAX_UNCATALOGUED_FOOTPRINT, WorldModel  # noqa: E402

CATALOG = ObjectCatalog.from_yaml(os.path.join(_HERE, "..", "config", "objects.yaml"))


def _model():
    return WorldModel(CatalogResolver(CATALOG), CATALOG, "bottle")


def _det(label, pos, dims, conf=0.5):
    return {"class_name": label, "confidence": conf, "object_id": None,
            "bbox_2d": [0, 0, 10, 10], "depth_m": pos[0], "position_base": list(pos),
            "dims_hwl": list(dims), "orientation": 0.0}


def _box_for(model, label):
    return next(b for b in model.obstacles() if label in b["name"])


def test_the_logged_phantom_person_is_capped():
    m = _model()
    m.update([_det("person", (0.498, 0.254, 0.306), (0.114, 0.6, 0.6))], 0.0, None)
    b = _box_for(m, "person")
    assert b["sx"] <= MAX_UNCATALOGUED_FOOTPRINT + 2 * OBSTACLE_PAD + 1e-9, b
    # the arm's ready pose (EE ~ (0.31, 0.01), the links a few cm either side)
    # must be clear of it in y: box spans y in [0.254 - sx/2, 0.254 + sx/2]
    assert b["y"] - b["sy"] / 2 > 0.10, b


def test_catalogued_obstacle_takes_its_catalogue_size_and_the_table_keeps_its_own():
    """YOLO reads the 45 mm can as 55 mm; as a neighbour's collision box it is
    sized from the catalogue instead. Label is "can_coke" (the fine-tuned
    model's own class, since the 2026-10-03 catalogue switch) rather than the
    old "cell phone" COCO alias — same object, current reality."""
    m = _model()
    m.update([_det("can_coke", (0.45, 0.20, 0.30), (0.096, 0.055, 0.055)),
              _det("dining table", (0.66, 0.0, 0.25), (0.02, 0.40, 0.80))], 0.0, None)
    can = _box_for(m, "can_coke")
    assert math.isclose(can["sx"], 0.045 + 2 * OBSTACLE_PAD), can
    tbl = next(d for d in m.latest if d["class_name"] == "dining table")
    assert tbl["dims_hwl"] == [0.02, 0.40, 0.80], tbl


def test_a_plausible_uncatalogued_object_is_untouched():
    m = _model()
    m.update([_det("cup", (0.45, 0.20, 0.30), (0.10, 0.08, 0.08))], 0.0, None)
    assert math.isclose(_box_for(m, "cup")["sx"], 0.08 + 2 * OBSTACLE_PAD)


def test_surface_label_is_not_a_generic_obstacle():
    """Regression for obs_dining table_0: the table is already modelled by its
    own dedicated table_box()/TableEstimate, and a surface label's size is
    deliberately NOT capped by _plausible_size (needed so the table keeps its
    real, large footprint there) — so feeding it through obstacle_boxes() too
    can produce a second, huge, uncapped phantom copy of the table. Measured
    live: 0.51 x 0.51 x 0.21 m, centred in the approach corridor, 0% pre-grasp
    Cartesian coverage. A "dining table" detection must not appear in
    obstacles() at all, whatever its reported size."""
    m = _model()
    m.update([_det("dining table", (0.562, 0.0, 0.336), (0.211, 0.511, 0.511)),
              _det("can_coke", (0.45, 0.20, 0.30), (0.096, 0.055, 0.055))], 0.0, None)
    names = [b["name"] for b in m.obstacles()]
    assert not any("dining table" in n or "table" in n for n in names), names
    assert any("can_coke" in n for n in names), names   # real obstacles still appear


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL PASS")
