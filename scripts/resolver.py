"""Which detection is the object the user asked for.

Ported near-verbatim from rework-attempt-1's src/locobot_prog/perception/
resolver.py. This is the concrete replacement for yolo_detection_node.py's
_ALIAS_MAP (a plain OR-check: is_target = class_name == target_object or
class_name in target_aliases) — that check has no way to prefer one candidate
over another when two DIFFERENT objects share a label; this scores each
matching candidate by confidence x plausibility(measured size vs the
catalogue's expected size), so a same-labelled wrong object loses to a
better-sized one instead of racing it on a coin flip. yolo_detection_node.py
itself is untouched — this is consumed by detection_shim.py/world_model.py,
not by the perception node.

The ONE place that answers "is this the target". Before, the answer was given
in five places against three different reference names (the perception
node's ~target_object, the agent's ~target_object and the VLM's tool
argument), and it worked only because the checks were OR-ed together.

TargetResolver is the interface; CatalogResolver answers from
config/objects.yaml. A VLM-based resolver can implement the same interface
later (it gets the candidates and the image, returns an index), and the world
model will not need to change.
"""
import math
from abc import ABC, abstractmethod
from collections import namedtuple

import yaml

# index: which candidate is the target (-1 = none); key: the catalogue key the
# name resolved to ("" when the name is not in the catalogue); reason: for logs
# and for the VLM.
Resolution = namedtuple("Resolution", "index key reason")

# How far a measured size may be from the catalogue size before it counts
# against a candidate. YOLO's bbox height is off by up to ~35% at close range,
# so this is deliberately loose, and size never rejects a sole candidate.
SIZE_TOLERANCE = 0.5


def normalize(name):
    """'Tea_Box ' -> 'tea box'."""
    return " ".join(str(name).lower().replace("_", " ").replace("-", " ").split())


class ObjectCatalog:

    def __init__(self, objects):
        """objects: the `objects:` mapping of config/objects.yaml."""
        self.objects = objects
        self._by_name = {}
        for key, spec in objects.items():
            for name in [key] + list(spec.get("names", [])):
                self._by_name[normalize(name).replace(" ", "")] = key

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            return cls(yaml.safe_load(f)["objects"])

    def key_for_name(self, name):
        """Catalogue key for what a user/VLM called the object, or None."""
        return self._by_name.get(normalize(name).replace(" ", ""))

    def labels(self, key):
        return {normalize(l) for l in self.objects[key].get("yolo_labels", [])}

    def size(self, key):
        """(x, y, z) metres, or None."""
        s = self.objects[key].get("size")
        return (float(s["x"]), float(s["y"]), float(s["z"])) if s else None

    def surface_labels(self):
        """Labels of objects with role: surface (the table)."""
        out = set()
        for key, spec in self.objects.items():
            if spec.get("role") == "surface":
                out |= self.labels(key)
        return out

    def keys_for_label(self, label):
        """Every non-surface catalogue key the detector label could be."""
        label = normalize(label)
        return [k for k, spec in self.objects.items()
                if spec.get("role") != "surface" and label in self.labels(k)]


class TargetResolver(ABC):

    @abstractmethod
    def resolve(self, target_name, candidates, image=None):
        """Pick the target among `candidates` (dicts with at least class_name and
        confidence; dims_hwl optional). Returns a Resolution."""


class CatalogResolver(TargetResolver):

    def __init__(self, catalog):
        self.catalog = catalog

    def matches(self, target_name, detection):
        """Could this detection be the named object?"""
        label = normalize(detection.get("class_name", ""))
        key = self.catalog.key_for_name(target_name)
        if key is None:
            # Not in the catalogue: the name must BE the detector label. This keeps
            # get_object("vase") working exactly as it always has.
            return label == normalize(target_name)
        return label in self.catalog.labels(key)

    def plausibility(self, key, dims_hwl):
        """1.0 when the measured size agrees with the catalogue, falling off
        smoothly with relative error. 1.0 when either size is unknown."""
        expected = self.catalog.size(key) if key else None
        if expected is None or not dims_hwl:
            return 1.0
        h, w, _ = (float(v) for v in dims_hwl)
        err_h = (h - expected[2]) / expected[2]
        err_w = (w - expected[1]) / expected[1]
        return math.exp(-(err_h ** 2 + err_w ** 2) / (2 * SIZE_TOLERANCE ** 2))

    def resolve(self, target_name, candidates, image=None):
        key = self.catalog.key_for_name(target_name) or ""
        scored = [(float(d.get("confidence", 0.0)) * self.plausibility(key, d.get("dims_hwl")), i)
                  for i, d in enumerate(candidates) if self.matches(target_name, d)]
        if not scored:
            return Resolution(-1, key, f"no detection matches '{target_name}'")
        score, i = max(scored)
        label = candidates[i].get("class_name")
        how = f"label '{label}'" + ("" if not key else f" is a {key}")
        if len(scored) > 1:
            how += f", best of {len(scored)} candidates (score {score:.2f})"
        return Resolution(i, key, how)
