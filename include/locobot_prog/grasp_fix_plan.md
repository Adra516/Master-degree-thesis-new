# Grasp Reliability Fix — Implementation Plan (hand-off to Claude in VS Code)

## Goal
Make side-grasps work for objects of **different heights and widths** (bottle *and*
short can / `tea_box`) by feeding the already-measured YOLO3D bbox dimensions into the
grasp **height** and **close** behaviour, without disturbing the parts of the stack that
are already carefully tuned.

The diagnosis behind this plan: `_compute_grasp_z` currently collapses to a near-constant
7–8.5 cm grip height for anything shorter than 20 cm, so short objects get gripped at/above
their own rim. Fixing the height to be object-relative is the single highest-value change.

---

## Ground rules (do NOT change these)
- **Do not touch `arm_gains_override.yaml`.** The PID gains are the output of a measured
  tuning battery (forearm_roll pinned to vendor p=5, gripper deliberately gentle p=100
  because `gazebo_grasp_fix` is the holder). Re-tuning risks reopening the oscillation and
  will not fix the accuracy problem, which is geometric.
- **Keep the open-loop descent + settle-hold pattern** in `_execute_grasp` (direct publish
  to the controller, re-command measured joints to zero position error). Do not route the
  descent back through `move_group.execute()`.
- **Do not re-introduce the fake `[0.25, 0.10, 0.10]` default dims.** When dims are unknown,
  keep the existing "unknown → conservative fallback" branches; do not inject a fabricated size.
- Keep the weld-based hold assumption (fingers only need gentle contact; the fix attaches).

---

## Files in scope
| File | Role in this plan |
|------|-------------------|
| `locobot_motion_node.py` | `_compute_grasp_z`, `_gripper_target_close`, `_gripper_span_for`, `_execute_grasp`, `_handle_get_gripper_state` |
| `grasp_geometry.py` | `_horiz_grasp_z`, `grasp_pose` — **duplicate height formula, must stay in sync** |
| `vlm_agent_node.py` | `_descent_margin`, `_reach_optimal_standoff` — consume the height formula for standoff choice |
| `locobot_geometry.py` | `parse_object` grasp-axis selection |

---

## Critical consistency constraint (read before Phase 1)
The grasp-height formula exists in **two places**:
- `locobot_motion_node._compute_grasp_z` (what the arm actually executes), and
- `grasp_geometry._horiz_grasp_z` (used by `vlm_agent_node._descent_margin` →
  `_reach_optimal_standoff` to decide how far the base should stand from the object).

If only one is changed, the agent will optimise the standoff for one grip height while the
arm executes another. **Both must change together.** Preferred approach: factor the height
formula into a single shared helper in `grasp_geometry.py` and have the motion node import
and call it, so there is one source of truth.

---

## Phase 0 — Measure first (no code changes yet)
Before editing anything, capture a baseline so the fixes are driven by data, not guessing.

- [ ] Run one grasp on a **bottle** and one on a **short can / `tea_box`**.
- [ ] For each, capture the existing instrumentation output:
  - `[diag] PRE-GRASP EE …` (from `_diag_pregrasp`) — EE actual vs commanded, `err_norm_mm`, `ori_err_deg`.
  - `[diag] CONTACT …` (from `_diag_contact`) — `jaw_gap_mm`, `along_jaw_mm` (0 = centred), `in_plane_miss_mm`, `straddled`.
  - Set `~diag_object_model` to the Gazebo model name first so the ground-truth object check runs.
- [ ] Record `grasp_z`, `dims_hwl`, and `grasp_axis` printed in the `[motion] Horizontal grasp:` log line.

**Interpretation gate:** if PRE-GRASP `err_norm_mm` is small but CONTACT `along_jaw_mm` /
`in_plane_miss_mm` is large, the problem is **centering / height**, not control gains — proceed
with Phases 1–3. Do not consider any gain change unless contact offsets remain large *after*
Phases 1–2.

---

## Phase 1 — Dimension-aware grasp height (highest value)
**Problem:** current formula
```python
raw_z = table_z + clip(0.35 * H, 0.070, 0.085)   # then clip to [table_z+0.06, table_z+0.14]
```
For `0.35*H` to clear the 0.070 floor, H must be ≥ 0.20 m. So every object shorter than
20 cm is gripped at a fixed ~7 cm regardless of height → short can/box gets gripped at/above
its rim → fingers close on air or the top edge.

**Target behaviour:** grip at a fraction of the object's own height, floored by the real
gripper/table clearance and ceilinged by the object top minus a margin.
```python
z_above = clip(grip_frac * H, z_floor, H - top_margin)
grasp_z = table_z + z_above
```
- `grip_frac ≈ 0.5` (mid-body) — expose as a module constant, not a magic number.
- `z_floor` = the genuine minimum reachable/​safe height above the table (reuse the existing
  `table_z + 0.06` intent).
- `top_margin` ≈ 1.5–2 cm so the jaws never land on the rim.

**Tasks**
- [ ] Introduce a single shared helper (suggested: `grasp_geometry.grasp_height_above_table(H, ...)`)
      implementing the formula above; expose `grip_frac`, `z_floor`, `top_margin` as named args/constants.
- [ ] Rewrite `_compute_grasp_z` (motion node) to call the shared helper.
- [ ] Rewrite `_horiz_grasp_z` (`grasp_geometry.py`) to call the same helper, so
      `_descent_margin` / `_reach_optimal_standoff` optimise for the *actual* grip height.
- [ ] **Degenerate-short guard:** if `H - top_margin < z_floor` (object too short for any valid
      side grip), do not silently clamp into an invalid grip. Log clearly and return a sentinel /
      raise a handled "too short for side grasp" condition. (Note: the top-down approach axis was
      removed — see the `grasp_geometry.py` module header — so this case currently has no fallback.
      Flag it; do not fake a grip.)

**Acceptance**
- [ ] Bottle grip height ≈ unchanged from baseline (regression check).
- [ ] `tea_box` grip height now lands at roughly mid-box, **below its top**, confirmed in the
      `[motion] Horizontal grasp: … grasp_z=…` log and in `_diag_contact` (`along_jaw_mm` small,
      `straddled=True`).

---

## Phase 2 — Reconcile the close strategy (shape-aware)
**Problem:** `_gripper_span_for` computes an *adaptive* close from measured width, but
`_gripper_target_close` **ignores it and always returns `GRIPPER_CLOSE` (full close)**. Full
close is correct for the tapered bottle neck (unknown local width), but for a uniform can/box
it slams shut — the first finger contacts and shoves the object sideways before the second lands.

**Target behaviour:** choose the close target by shape confidence.
- Roughly prismatic / uniform-width object with a reliable measured width → adaptive close to
  `half_width − small_squeeze` (use `_gripper_span_for`'s close value) so the jaws self-centre.
- Tapered / unknown local width (bottle) → keep the current full close.

**Tasks**
- [ ] Give `_gripper_target_close` access to `dims_hwl` + a shape/confidence flag and branch:
      adaptive for uniform, full-close for tapered/unknown.
- [ ] Keep `_gripper_span_for` as the single place that turns a width into finger positions
      (do not duplicate the arithmetic).
- [ ] Re-check `_handle_get_gripper_state`'s `has_object = gap_mm < 20.0` threshold: with an
      adaptive close the resting gap now depends on object width, so a fixed 20 mm threshold can
      misreport "empty". Make the "held" check dims-aware (e.g. gap noticeably less than the
      commanded open span) rather than a hard constant.

**Acceptance**
- [ ] Can/box: `_diag_contact` `along_jaw_mm` and `in_plane_miss_mm` smaller than Phase-0 baseline.
- [ ] Bottle: no regression (still welds and lifts level).

---

## Phase 3 — Grasp-axis reliability
**Problem:** `parse_object` (`locobot_geometry.py`) picks `grasp_axis` purely from the estimated
yaw, and that yaw comes from a single 2D box + depth (`_estimate_orientation`) — unreliable for a
box. Wrong axis on a `tea_box` = trying to span its long side.

**Tasks**
- [ ] When yaw confidence is low (or by default for near-square footprints), fall back to
      "close across the **smaller** of W and L" instead of trusting the yaw-derived axis.
- [ ] Keep `gripper_fits` (`narrow <= GRIPPER_W_M + 0.03`) as the gate; ensure `narrow` is
      recomputed from the chosen axis after any fallback.

**Acceptance**
- [ ] For a `tea_box` deliberately placed at an ambiguous yaw, the chosen `grasp_axis` closes
      across the narrower face (verify in the log + `_diag_contact`).

---

## Phase 4 — Reconcile stale docstrings/comments
Low-risk cleanup so the next reader isn't misled.
- [ ] `_compute_grasp_z` docstring narrates gripping the shoulder/neck at `0.65*H`, but the code
      grips low at `0.35*H` clamped. Update the docstring to match the new Phase-1 formula and
      delete the stale `0.65*H` narrative.
- [ ] Note in `_gripper_target_close` why the close is now conditional (Phase 2).

---

## Do-not-regress checklist (run after each phase)
- [ ] Bottle: welds on contact, lifts level, no twist/oscillation.
- [ ] Reach window still enforced (`0.33 m`–`0.52 m`) in `_handle_execute_grasp`.
- [ ] Descent still executes via direct publish (no `execute()` action abort).
- [ ] No fake default dims injected anywhere; unknown-dims fallbacks intact.
- [ ] `grasp_geometry` height helper and `_compute_grasp_z` produce identical numbers for the
      same inputs (add a tiny unit check if convenient — `grasp_geometry` is written to be
      unit-testable in isolation).

## Suggested order of work
Phase 0 (measure) → Phase 1 (height, both call-sites) → re-measure → Phase 2 (close) →
re-measure → Phase 3 (axis) → Phase 4 (docs). Only revisit gains if contact offsets are still
large after Phases 1–2, and if so treat it as an open-loop-descent settling question (does the EE
reach the commanded z under gravity?), not a gain-value question.
