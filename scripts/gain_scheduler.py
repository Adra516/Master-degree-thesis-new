"""Temporarily stiffen specific joints' PID gains while the arm is physically
holding a grasped object, then restore the exact tuned baseline on release.

STEP 1 only (see the session that added this): LOAD_GAIN_DELTA is
deliberately EMPTY for every joint below. This file's whole job right now is
to prove the plumbing works — that each joint's dynamic_reconfigure server is
reachable, that its baseline gains can be read back correctly, and that
apply_load()/release() can be called at the right moments in grasp_executor.py
— without changing a single number yet. Nothing in this file is allowed to
change arm behaviour until a specific delta is filled in, measured with the
same live ground-truth tracer used for every other gain change in this
project's history, and found to actually help.

Why a delta on top of a read baseline, not a second hardcoded table: this
project's real, hard-won tuning lives in config/arm_gains_override.yaml
(loaded into the parameter server by locobot.launch) — see that file's own
comments for the battery of measurements behind every number in it, and for
why guessed PID values on this arm have gone wrong before (forearm_roll's
p=20 rang the whole arm; see gazebo-effort-pid-tuning memory / that file's
history). A second, independently-hardcoded "loaded" gain table would drift
out of sync with it the same way yolo_detection_node's old _ALIAS_MAP drifted
out of sync with config/objects.yaml. Reading the live baseline once and
applying only a small delta on top of THAT keeps there being exactly one
source of truth for what "normal" is.

Each JointTrajectoryController joint gets its own dynamic_reconfigure server
automatically (control_toolbox::Pid, no code needed on the controller side):
  /<robot_name>/<controller>/gains/<joint>/set_parameters
"""
import rospy
from dynamic_reconfigure.client import Client

# (controller, joint) -> {gain: delta}. Empty dict = no change for that
# joint — apply_load()/release() skip it entirely. Fill in a real delta only
# after measuring it; see this file's module docstring.
LOAD_GAIN_DELTA = {
    ("arm_controller", "forearm_roll"): {},
    ("arm_controller", "shoulder"): {},
    ("arm_controller", "elbow"): {},
    ("gripper_controller", "left_finger"): {},
    ("gripper_controller", "right_finger"): {},
}


class GainScheduler:

    def __init__(self, robot_name="locobot", timeout=2.0):
        self.ns = robot_name
        self._clients = {}     # (controller, joint) -> dynamic_reconfigure.Client
        self._baseline = {}    # (controller, joint) -> {p, i, d, ...} at connect time

        for key in LOAD_GAIN_DELTA:
            ctrl, joint = key
            path = f"/{robot_name}/{ctrl}/gains/{joint}"
            try:
                client = Client(path, timeout=timeout)
                baseline = client.get_configuration(timeout=timeout)
            except rospy.ROSException as e:
                rospy.logwarn("[gains] %s dynamic_reconfigure unavailable (%s) — "
                              "gain scheduling disabled for this joint", path, e)
                continue
            self._clients[key] = client
            self._baseline[key] = baseline
            rospy.loginfo("[gains] %s baseline read: p=%.2f i=%.2f d=%.2f",
                          path, baseline["p"], baseline["i"], baseline["d"])

    def apply_load(self):
        """Call once the object is confirmed held, before the lift. No-op for
        any joint whose LOAD_GAIN_DELTA is empty (all of them, right now)."""
        for key, client in self._clients.items():
            delta = LOAD_GAIN_DELTA[key]
            if not delta:
                continue
            new = dict(self._baseline[key], **{k: self._baseline[key][k] + v
                                               for k, v in delta.items()})
            client.update_configuration(new)
            rospy.loginfo("[gains] %s/%s -> %s (loaded)", *key,
                          {k: round(new[k], 2) for k in delta})

    def release(self):
        """Call once the object is placed/released. Restores the EXACT values
        read at connect time — never a second guess at what 'normal' is."""
        for key, client in self._clients.items():
            if not LOAD_GAIN_DELTA[key]:
                continue
            client.update_configuration(self._baseline[key])
            rospy.loginfo("[gains] %s/%s -> baseline (released)", *key)
