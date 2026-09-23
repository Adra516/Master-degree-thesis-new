"""Choosing one target position from what the world model knows. Pure.

Ported near-verbatim from rework-attempt-1's src/locobot_prog/world/
target_state.py. The world model reports every source it has (a
locobot_prog/TargetState); the caller decides which to trust, like cobot's
pick-and-place client decides what to do with GetPosition. This is the
priority the pick sequence has always used, with the freshness limit as a
parameter because positioning (2 s) and the grasp point (3 s) have always
used different ones.
"""


def measured_position(state, max_lock_age, min_recent=3):
    """(x, y, source) of the target in base_link, or None when no source is
    usable. Priority:

      1. "median" — median of the recent positions, when the lock is fresh and
         there are at least `min_recent` of them. Measured: with the base
         aligned the remaining miss is per-frame bbox jitter, which a median
         over ~5 stationary frames cancels.
      2. "lock"   — the last accepted detection, when fresh. The robot has been
         still since, so it needs no reprojection.
      3. "map"    — the lock reprojected from the map frame, for when the
         camera has lost the target (e.g. it fills the frame up close).
    """
    fresh = state.locked and state.lock_age < max_lock_age
    if fresh and state.recent_count >= min_recent:
        return state.recent_median.x, state.recent_median.y, "median"
    if fresh:
        return state.lock_position.x, state.lock_position.y, "lock"
    if state.map_valid:
        return state.map_position.x, state.map_position.y, "map"
    return None
