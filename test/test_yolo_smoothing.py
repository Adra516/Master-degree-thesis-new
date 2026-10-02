"""_TunedBBox3DEstimator (yolo_detection_node.py): per-object Kalman +
temporal smoothing of position/size/orientation, retuned from the library's
street-scene-scaled defaults to this project's tabletop scale.

No ROS master needed: importing the node module only defines classes: it
calls rospy.init_node() inside `if __name__ == "__main__"`, not at import
time, so this just needs rospy et al. importable (they are, via the sourced
ROS environment), not running.

Regression context: an earlier version of this check compared raw vs
smoothed jitter and drift tracking using a SINGLE last frame's error, which
once flagged a false failure — a single frame still carries its own
measurement noise, on top of whatever bias the filter has. Judging
convergence on the mean of several trailing frames, averaged over many
random seeds, separates true bias from per-frame noise.
"""
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import yolo_detection_node as ydn  # noqa: E402

TRUE_LOC = np.array([0.45, 0.02, 0.33])
TRUE_DIMS = np.array([0.082, 0.045, 0.045])   # H, W, L (the resized can)
TRUE_ORI = 0.05
# Real base-creep magnitude measured live ("Base tilted under the arm:
# target is (-24,+7,+20)mm") — the kind of real, modest drift the filter
# must still track, as opposed to an object teleporting.
DRIFT_LOC = TRUE_LOC + np.array([0.020, -0.007, 0.010])

N_TRIALS = 40
LOC_NOISE, DIM_NOISE, ORI_NOISE = 0.010, 0.008, 0.15   # matches the project's
# documented per-frame jitter (world_model.py's calibration-buffer comment)


def _smoothed_run(est, rng, target, n):
    out = []
    for _ in range(n):
        loc = target + rng.normal(0, LOC_NOISE, 3)
        dims = TRUE_DIMS + rng.normal(0, DIM_NOISE, 3)
        ori = TRUE_ORI + rng.normal(0, ORI_NOISE)
        box = {"dimensions": dims, "location": loc, "orientation": ori,
               "bbox_2d": [0, 0, 10, 10], "object_id": 1, "class_name": "can_coke"}
        box = est._apply_kalman_filter(box, 1)
        hist = est.box_history[1]
        hist.append(box)
        if len(hist) > est.max_history:
            hist.pop(0)
        box = est._apply_temporal_filter(1)
        out.append(box["location"].copy())
    return np.array(out)


def _one_trial(seed):
    rng = np.random.default_rng(seed)
    est = ydn._TunedBBox3DEstimator(camera_matrix=np.eye(3))
    settle = _smoothed_run(est, rng, TRUE_LOC, 30)
    raw_settle = TRUE_LOC + rng.normal(0, LOC_NOISE, (30, 3))  # independent raw reference
    drift = _smoothed_run(est, rng, DRIFT_LOC, 15)
    return raw_settle, settle, drift, est


def test_smoothing_cuts_jitter_without_losing_accuracy():
    raw_jit, smooth_jit, settle_err, drift_err = [], [], [], []
    for seed in range(N_TRIALS):
        raw_settle, settle, drift, _ = _one_trial(seed)
        raw_jit.append(np.std(np.diff(raw_settle, axis=0), axis=0))
        smooth_jit.append(np.std(np.diff(settle, axis=0), axis=0))
        settle_err.append(np.linalg.norm(settle[-5:].mean(0) - TRUE_LOC))
        drift_err.append(np.linalg.norm(drift[-5:].mean(0) - DRIFT_LOC))
    raw_jit, smooth_jit = np.mean(raw_jit, axis=0), np.mean(smooth_jit, axis=0)
    settle_err, drift_err = np.mean(settle_err), np.mean(drift_err)

    assert (smooth_jit < raw_jit * 0.6).all(), (raw_jit, smooth_jit)
    assert settle_err < 0.012, settle_err
    assert drift_err < 0.012, drift_err


def test_cleanup_trackers_drops_state_for_ids_no_longer_seen():
    _, _, _, est = _one_trial(0)
    assert 1 in est.kf_trackers and 1 in est.box_history
    est.cleanup_trackers([])
    assert 1 not in est.kf_trackers
    assert 1 not in est.box_history


def test_untracked_detection_id_is_untouched_by_default_r():
    """A sanity check that the override actually took: the library's default
    R (location variance 1.0) would need ~100x more frames to converge this
    tightly, so a passing test above already implies this, but assert the
    concrete R values directly too as a regression anchor."""
    est = ydn._TunedBBox3DEstimator(camera_matrix=np.eye(3))
    box = {"dimensions": TRUE_DIMS, "location": TRUE_LOC, "orientation": TRUE_ORI,
           "bbox_2d": [0, 0, 10, 10], "object_id": 1, "class_name": "can_coke"}
    est._apply_kalman_filter(box, 1)
    kf = est.kf_trackers[1]
    assert math.isclose(kf.R[0, 0], 1e-4)
    assert math.isclose(kf.R[3, 3], 6e-5)
    assert math.isclose(kf.R[6, 6], 0.0225)


if __name__ == "__main__":
    test_smoothing_cuts_jitter_without_losing_accuracy()
    test_cleanup_trackers_drops_state_for_ids_no_longer_seen()
    test_untracked_detection_id_is_untouched_by_default_r()
    print("OK")
