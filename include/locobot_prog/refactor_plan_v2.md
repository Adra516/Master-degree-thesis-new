# locobot_proj refactor — grounded prompt for Claude Code

## Simplification scope (confirmed, do not deviate without asking)

The goal is to make this codebase **simpler**, taking the cobot project as
the model of what "simple" looks like structurally: one service per
action, small classes, no JSON-blob-over-topic dispatch. This is
explicitly a **structural and interface simplification**, not a behavioral
one:

- **Keep, verbatim**: every verification gate (`pregrasp_verdict`,
  `contact_verdict`, `_diag_pregrasp`, `_diag_contact`), every retry loop
  (the pre-grasp base-correction retry, the two-attempt pregrasp loop),
  odom caching (`TargetOdomCache`), and per-class dimension calibration
  (`_dim_calib`/`_calibrated_dims`). These exist because each one fixed a
  specific measured failure — cutting them reintroduces the failure, not
  just "extra code".
- **Simplify**: file/node boundaries, the topic+JSON dispatch convention
  (→ typed services), the single `get_object(target_name)` tool (→
  `get_object(name, mode)` + `navigate_to`), the auto-chained place (→
  separate call), the hardcoded YOLO alias map (→ `label_reconciler`),
  and the hand-copied `ARM_REACH_M`/`ARM_LIMITS` constants (→ derived from
  the URDF).

If, while implementing, a piece of "complexity" turns out to be pure
scaffolding rather than a fix for a measured failure, it's fair game to
simplify — but default to keeping behavior and ask before cutting
anything whose comment references a measurement, a failure mode, or a
"do not remove" warning.

## Read this first: the existing code is more advanced than a first glance
## at the launch files suggests

`locobot_prog/scripts/` already contains a working, heavily-tuned
pick-and-place pipeline. Do not treat this as a from-scratch build — it is
a **decomposition and interface-cleanup task** on working code, not a
redesign of the underlying behavior. Every empirically-tuned number and
verification gate below was arrived at through measurement (see the inline
comments in the files themselves) and must survive the refactor unchanged
in behavior, only moved to a new location.

## Current file inventory (what exists today)

- **`vlm_agent_node.py`** (~1065 lines) — everything: camera/detection
  subscriptions, per-class dimension calibration, IK-based inverse-
  reachability standoff selection, single-shot rotate/drive positioning,
  calling the motion node, auto-chaining a `place` after a successful
  grasp, and the Ollama agent loop itself. Currently exposes **one** VLM
  tool: `get_object(target_name)`. `stream=False` today (not the
  streaming loop from the Ollama docs — flag this as a deliberate decision
  point, not an oversight, when you touch it).
- **`locobot_motion_node.py`** (~1445 lines) — a single node reachable via
  **topics**, not services: subscribes to `motion_cmd` (a `String` carrying
  JSON `{"cmd": ..., ...}`), dispatches to handlers
  (`navigate`, `set_arm_pose`, `open_gripper`, `close_gripper`,
  `get_gripper_state`, `execute_pregrasp`, `execute_descent`,
  `execute_grasp` [legacy combined], `place`), and publishes results on
  `motion_result`. Already implements:
  - `_setup_grasp` / `_cleanup_grasp_scene` — the shared collision-scene
    helper we'd speculated about **already exists**.
  - `_add_obstacles` / `_remove_obstacles` (all non-target objects) and
    `_add_target_obstacle` / `_remove_target_obstacle` (the target itself)
    — the pre_grasp-adds-everything / grasp-removes-target toggle is
    already implemented as two paired helpers.
  - `_do_pregrasp` / `_do_descent` — FK-measured verification
    (`_diag_pregrasp`, `_diag_contact`) that actually checks the arm
    reached the commanded pose and the jaws straddle the object before
    closing, with retry logic in the caller.
  - `_pending_grasp` state (with a TTL and a drift/identity guard) held
    between the `execute_pregrasp` and `execute_descent` calls — this is
    how the two-phase sequencing is enforced today, and needs an
    equivalent mechanism after the split into services (see below).
- **`locobot_geometry.py`** (~457 lines) — pure functions, no ROS: physical
  constants (`ARM_REACH_M=0.55`, `ARM_GRASP_DIST=0.45`, `GRIPPER_W_M=0.08`,
  `TABLE_SURFACE_Z=0.25` fallback, `TABLE_DEPTH_M=0.40`,
  `TABLE_ROBOT_MARGIN=0.12`), zone classification
  (`grasp_ready`/`drive_forward`/`rotate_needed`/`path_blocked`/
  `target_not_found`), `parse_object`, `find_blocking`, `classify_zone`,
  `compute_grasp` (preview only, for the VLM-facing summary),
  `build_summary`, `render_birdseye` (the bird's-eye visualization fed
  back to the VLM as an image — this already exists and works),
  `estimate_grasp_orientation` (OpenCV contour/minAreaRect yaw estimation
  — this is most of what we called `grasp_point_selector`).
- **`grasp_geometry.py`** (~274 lines) — THE authoritative grasp math, and
  explicitly documented as the single source of truth not to be
  re-derived elsewhere:
  - `grasp_height_above_table` — grip point as a fraction (`GRIP_FRAC=0.5`)
    of object height, floored/ceilinged by measured safe bounds.
  - `pregrasp_pose` — standoff = grasp point minus `offset` along the
    approach vector, **at the same height**, exactly the geometry
    confirmed earlier in this conversation.
  - `min_standoff_offset`, `obstacle_boxes`,
    `pregrasp_verdict`/`contact_verdict` (the go/no-go gates, kept
    ROS-free so they're unit-testable — already satisfies the "testable
    without a full sim boot" goal for this piece).
- **`locobot_scene_model.py`** (~87 lines) — `SceneModel`: already a
  minimal `environment_model`. Derives live table geometry (near/far edge,
  width, depth, surface Z) from a `"dining table"` YOLO detection plus the
  camera intrinsics, falling back to `TABLE_SURFACE_Z` only when the
  measurement is implausible. Pure Python, no ROS — good, keep this
  property.
- **`target_odom_cache.py`** (~131 lines) — `TargetOdomCache`: caches the
  target's position in the odom frame so navigation and grasping stay
  correct while the base is rotating/driving and the camera can't see the
  target (occluded by the arm, or too close). This is load-bearing for
  navigation accuracy — do not simplify away.
- **`yolo_detection_node.py`** (~252 lines) — YOLO-3D + `BBox3DEstimator`,
  publishes `detections_3d` (String/JSON) and an annotated image. **This
  is where the alias map actually lives**, as a class attribute:
  ```python
  _ALIAS_MAP = {
      "bottle":  ["vase"],
      "cancoke": ["cell phone"],
      "teabox":  ["remote"],
      "ball":    ["sport ball"],
  }
  ```
  This is the concrete thing `label_reconciler` replaces.
- **`set_camera_angle.py`** (~27 lines) — one-shot pan/tilt publish. Fine
  as-is, no change needed.

## get_object's response contract to the VLM — keep it JSON, keep it rich

This corrects something implied too narrowly earlier in this
conversation. `get_object`'s **return value to the VLM** must stay a JSON
scene description, not shrink to a bare status/enum. Per the original
thesis brief: YOLO and the VLM inspect the room together, YOLO's
recognition results update a JSON scene description, and that JSON is
handed back to the VLM as the material it reasons over on its next turn —
this is already close to what `get_object` returns today (`success,
gripper_width_m, zone, scene_summary, visible_objects, description`) and
that shape should be **kept and extended by mode**, not replaced by a
terse `{success, status, message, pose}` contract.

Concretely, for each mode:

- `mode="locate"` — return the observed object list/positions/dims (what
  `_geometry`'s `objects` already produces) plus `zone` and
  `scene_summary`, so the VLM can reason about the room even when it
  didn't ask to pick anything up.
- `mode="pick"` — return the same scene-description shape as today's
  `get_object`, minus the auto-place fields, plus whatever `pre_grasp`/
  `grasp` status is needed to explain a failure (`phase`, `description`).
- `mode="place"` — return an updated scene description reflecting the
  object's new position, in the same JSON shape, not just a pass/fail.

This is a property of the **VLM-facing Python function only**
(`get_object` as passed into `tools=[...]`) — it doesn't mean the internal
ROS `.srv` contracts (`PreGrasp`, `Grasp`, `Place`) need to carry the full
scene JSON too; those stay typed and minimal per the services work
above. `get_object`'s Python implementation is what assembles the rich
JSON from the environment model's current scene state before returning it
to the VLM, same as `_geometry`/`build_summary` already do today — keep
that assembly step, just have it live in the new `get_object` node/
service instead of inline in `vlm_agent_node`.

## What must change, mapped to the actual code

### 1. Two VLM-facing tools, not one — `get_object` + `navigate_to`

Today only `get_object(target_name)` is a tool; navigation
(`_navigate_to_target`) is a private method called internally by
`get_object`. This needs to split into:

- **`get_object(name, mode)`** — keeps the fine, single-shot rotate/drive
  positioning loop that's already in `get_object` (the `COARSE_TOL`/
  `REOPEN_TOL`/`DIST_TOL` logic, `_measure_xy`/`_face`/`_drive`) — this is
  tightly coupled to grasp accuracy (it uses the odom cache and standoff
  distance) and should stay part of `get_object`'s `pick` mode, not move
  to `navigate_to`.
- **`navigate_to(location)`** — a new, separate tool for coarser
  scene-level moves (e.g. driving toward the table region before the
  target is even in frame), extracted from whatever currently sits inside
  `_navigate_to_target`'s rotation+drive-to-table-edge portion
  (`table_near_x`/`TABLE_ROBOT_MARGIN` logic) as a standalone capability.

**Open question to resolve with me before implementing**: should
`get_object(mode="pick")` still perform its own fine positioning
internally (current behavior), or should the VLM be expected to call
`navigate_to` first and `get_object` assume it's already in range? I lean
toward keeping fine positioning inside `get_object` (it needs the odom
cache and standoff distance that are specific to the grasp, not general
navigation) and using `navigate_to` only for coarse repositioning — but
confirm this before restructuring `_navigate_to_target`.

### 2. `mode` on `get_object`: `locate` / `pick` / `place`

Today `get_object` always does the full observe→navigate→grasp→**auto-place**
sequence in one call. Per our decision, `place` must become a **separate**
call, not auto-chained. Concretely: everything from `# ── 5. Capture BEV +
feedback image` onward in `get_object` that triggers `_call_motion({"cmd":
"place", ...})` on a successful grasp needs to be removed from the `pick`
path and become its own `mode="place"` branch, callable independently
once the object is confirmed held (`gw_m > 0.015` is the existing signal
for "object secured" — reuse it as the "am I holding something"
precondition for `place`).

Add a `locate` mode that runs steps 1 (`_geometry`) only — no navigation,
no motion commands — for when the VLM just needs to know where something
is.

### 3. Topics → services (matching the cobot project's actual pattern)

The cobot project's `object_position_base_link.py` exposes a plain
service (`GetPosition.srv`: `bool get_position` → `x/y/z`), called from
C++ via `ros::ServiceClient` — request/response, not the JSON-over-`String`-
topic convention currently used between `vlm_agent_node` and
`locobot_motion_node`. Convert `motion_cmd`/`motion_result` into real
services, one per command currently in the dispatch dict (or at minimum:
`pre_grasp`, `grasp`, `place`, `navigate`, `set_arm_pose`,
`get_gripper_state`), each with a proper `.srv` (typed fields, not a JSON
blob in a `String`) — this is what makes each command independently
`rosservice call`-able the way we discussed, and matches the cobot
project's actual pattern more literally than the current code does.

**Preserve `_pending_grasp`'s role across this change**: since
`pre_grasp`/`grasp` become two separate service calls on the same
long-lived node/process, holding `_pending_grasp` (with its TTL and
drift/identity guard) as instance state between the two service calls is
still correct and requires no redesign — just confirm the collapsed
`manipulation_node` (housing `pre_grasp`/`grasp`/`place`, per our earlier
process-layout decision) is one process, so this state isn't lost between
calls.

### 4. Replace `_ALIAS_MAP` with `label_reconciler`

The dict above, currently in `yolo_detection_node.py`, is the concrete
target. Move the reconciliation out of the detection node entirely (it
doesn't belong in the perception node) into its own service, following
the embedding/similarity + session-cached-VLM-confirmation approach
discussed earlier, replacing `self.target_aliases` and the `is_target =
class_name == target_object or class_name in target_aliases` check.

### 5. Derive `ARM_REACH_M` / `ARM_LIMITS` instead of hardcoding them

`locobot_geometry.py` hardcodes `ARM_REACH_M = 0.55` and
`ARM_GRASP_DIST = 0.45`; `vlm_agent_node.py` separately hardcodes a full
`ARM_LIMITS` dict (joint limits in radians) that's a manual copy from the
URDF, used only for scoring IK solution comfort in
`_reach_optimal_standoff`. Both should be computed from the URDF/
`robot_description` at node startup (e.g. via `urdf_parser_py` or MoveIt's
own kinematic model) instead of hand-copied — this is the main concrete
"hardcoded constant that should be derived from the environment" left in
the codebase; most other constants in `locobot_geometry.py` are already
reasonable fallback bounds around a live measurement (`SceneModel`
already derives table geometry live), not pure hardcoding.

### 6. Bird's-eye view is already correctly implemented; grasp-point
*selection* (not just orientation) is the actual gap

The brief's "bird's-eye view obtained by geometric projection from above"
has a literal ancestor in the cobot project: `opencv_tutorials/src/
stream_top_position.py` and `get_position_server.py` subscribe to a
physically overhead `rgbd_camera` topic and use HSV color masking +
`cv2.findContours` + `cv2.minAreaRect` to get each object's position,
size, and orientation directly from a true top-down image. LoCoBot has no
separate overhead camera — its single camera is forward-facing — so
`locobot_geometry.render_birdseye` is the correct equivalent: a
*synthetic* top-down rendering built from YOLO-3D's computed 3D
positions, not a literal overhead frame. **This part is already right —
relocate `render_birdseye` to wherever the new node boundaries put "give
me the bird's-eye scene," don't reimplement it.**

`estimate_grasp_orientation` (contour + `minAreaRect`, same technique as
cobot's overhead scripts, plus depth-masking and a 180°
theta-ray disambiguation LoCoBot's forward-facing setup needs that cobot's
overhead one doesn't) gives yaw/orientation only.

*Hardware confirmation*: per Trossen's LoCoBot WidowX-250 spec
(https://docs.trossenrobotics.com/interbotix_xslocobots_docs/specifications/locobot_wx250s.html),
servos 10/11 are dedicated `pan`/`tilt` joints — the camera sits on a
pan-tilt head (matching `set_camera_angle.py`'s
`pan_controller`/`tilt_controller` topics already in the code), angled
down toward the table (`tilt=0.35` in the current launch file), not a
fixed overhead camera. So `render_birdseye`'s *computed* top-down
projection from YOLO-3D's 3D positions is the right approach for this
hardware — there is no frame to read a literal top-down view from the way
cobot's `stream_top_position.py` does. Keep this distinction explicit if
you ever revisit `render_birdseye` — it's solving a harder problem
(reconstructing top-down from an angled view) than cobot's literal
overhead read, not a lesser version of it.

**The actual gap**, per
the brief, is that nothing today generates and verifies *candidate
contact points* against the gripper's geometry and picks the best one —
`estimate_grasp_orientation` tells you which way the object is turned,
not where on its surface the jaws should land or whether that point is
actually reachable/clear. This is genuinely new logic to build (the
brief says the current implementation "does not really do this properly"),
using `estimate_grasp_orientation`'s contour extraction as a starting
point but extending it to: (a) generate a small set of candidate contact
points along the object's surface consistent with `grasp_axis`, (b) check
each against gripper width / `is_tapered` / `min_standoff_offset`, and
(c) pick the best-scoring one — this becomes the new
`grasp_point_selector`'s actual job, not a relocation of existing code.

## What must NOT change in behavior (only in location)

- `grasp_geometry.py`'s formulas (`grasp_height_above_table`,
  `pregrasp_pose`, `min_standoff_offset`, `is_tapered`) and their
  measured constants (`GRIP_FRAC`, `GRIP_Z_FLOOR`, `TAPER_ASPECT_RATIO`,
  etc.) — verbatim.
- `pregrasp_verdict`/`contact_verdict` tolerance-gate logic.
- `_diag_pregrasp`/`_diag_contact`'s FK-measurement approach and the
  retry-with-base-correction logic in `get_object`'s pre-grasp loop.
- `TargetOdomCache`'s measure/reproject/goal behavior — this is what
  keeps navigation accurate while the camera is blind or the base is
  moving; do not replace with a simpler live-detection-only approach.
- Per-class dimension calibration (`_dim_calib`, `DIM_CALIB_MIN_DIST`,
  `_calibrated_dims`) — measured to be more accurate than live readings at
  close range; keep it, just relocate it to wherever detection/environment
  state ends up living.
- `arm_gains_override.yaml` and `grasp_fix_urdf.xacro`'s
  `gazebo_grasp_fix` block — unchanged, as established earlier.

## Reference project (cobot) — what to actually take from it

`cobot_IK/src/pick_place_simple_client_class.cpp` is the cleanest
reference: a small class (`Pick_And_Place`) with one method per action
(`pre_pick`, `pick`, `place`, `open_gripper`, `close_gripper`), each
calling a `ros::ServiceClient` (`GetPosition`) and `MoveGroupInterface`
directly, driven from a plain sequential `main()`. It is **much simpler**
than the current locobot code — no retries, no verification gates, no
collision-scene toggling, no two-phase split. Take from it:
- The **service-based request/response pattern** (vs. the current
  topic+JSON convention) — this is the main structural thing to adopt.
- One method/service per action, cleanly separated — the node-per-action
  boundary we've been designing.
- `position_tracker`'s `GetPosition.srv` as a minimal example of a typed
  service contract, though the actual services needed here will carry
  richer fields (dims, orientation, obstacles) than cobot's plain x/y/z.

Do **not** import cobot's simplicity where it means losing the locobot
code's verification/retry logic — cobot's version is a teaching example
without the measured failure modes locobot's code was built to handle.
All cobot code is C++; everything in `locobot_prog` is Python (rospy) —
translate the *pattern* (services, one-action-per-method, class
encapsulation), not the language or the simplified behavior.

## Order of work

1. Read `locobot_motion_node.py` in full detail (I've summarized its
   structure above from a `grep` of method names plus targeted reads —
   confirm the full bodies of `_do_pregrasp`, `_do_descent`, and
   `_handle_place` before touching them).
2. Define the new `.srv` files: `PreGrasp`, `Grasp`, `Place`, `Navigate`,
   `SetArmPose`, `GetGripperState` — typed fields replacing today's JSON
   `cmd` dicts. Check field lists against what `_grasp_cmd()` in
   `vlm_agent_node.py` currently packs (`x, y, z, angle_rad, dims_hwl,
   grasp_axis, table_z, table_near_x, table_width_m, table_depth_m,
   obstacles`) so nothing gets dropped.
3. Split `locobot_motion_node.py` into the collapsed `manipulation_node`
   (pre_grasp/grasp/place as services in one process, per the earlier
   process-layout decision) — port `_setup_grasp`/`_cleanup_grasp_scene`/
   `_add_obstacles`/`_add_target_obstacle` and the diagnostic gates
   unchanged.
4. Extract `SceneModel` + the table-geometry logic into a standalone
   `environment_model` node/service, adding arm-reach-from-URDF per
   point 5 above.
5. Replace `_ALIAS_MAP` in `yolo_detection_node.py` with a call to the new
   `label_reconciler` service.
6. Build `get_object`'s `locate`/`pick`/`place` mode dispatch, moving in
   the relevant sections of the current `get_object` method, and decouple
   place per point 2 above.
7. Extract `navigate_to` per point 1's resolution.
8. Rebuild `vlm_agent_node.py` last as the thin two-tool orchestrator,
   deciding then whether to adopt the Ollama streaming loop or keep
   `stream=False` — flag this choice explicitly rather than changing it
   silently.

## Appendix: original thesis brief (verbatim, for your own reference)

> The aim of this thesis is to let a LoCoBot explore its surrounding
> environment through its on-board camera and use a Vision-Language Model
> to interpret what it sees and update its plan accordingly. The user
> provides a prompt naming a target object; the robot must then locate
> that object in the scene, reach it, and manipulate it with the end
> effector. The longer-term objective is for the VLM to keep control of
> the manipulator after the object has been grasped, deciding the
> subsequent actions rather than running a fixed routine. [...] In
> particular, each "action" exposed to the VLM should become a ROS node
> of its own, so that it can be triggered and tested independently
> without having to call Qwen every time. [...] The same applies to
> navigation: it is currently handled partly inside the motion node,
> which is the wrong place for it, and it should be extracted and exposed
> as a proper VLM function alongside get_object. [...] get_object keeps
> its JSON interface, with the VLM and YOLO inspecting the room together,
> recognizing the objects and updating the JSON that is then passed back
> to the VLM as the scene description it reasons over. One hard
> constraint applies throughout: the PID controller data of the LoCoBot
> must not be modified.

Two points from this text to hold onto, since they're easy to blur once
you're deep in the code:

- **"each action exposed to the VLM should become a ROS node of its
  own"** — read literally, this means `get_object` and `navigate_to` are
  each their own node (not two services bundled in one process). This is
  slightly stricter than the process-collapsing discussed earlier for
  `pre_grasp`/`grasp`/`place` — those three stay collapsed into one
  `manipulation_node` because they are *not* VLM-exposed actions, only
  `get_object` and `navigate_to` are, and the brief's node-per-action rule
  applies specifically to what the VLM sees.
- **The post-grasp VLM control goal is explicitly out of scope for this
  stage** — don't add hooks, stubs, or extensibility machinery for it
  beyond what naturally falls out of decoupling `place` from `pick`
  (already planned). Keep the rework focused on the pick-and-place cycle.

## Dead code elimination

As part of the restructuring, audit every file being touched for code with
**zero remaining call sites** and remove it. This is a different bar from
the "simplify structure, keep behavior" rule above — unused code isn't
"behavior to preserve," it's just not called by anything, so deleting it
changes nothing at runtime. Rule: grep/trace actual callers across the
whole package before deleting anything; "looks legacy" or "looks unused"
is not sufficient on its own — confirm zero references first, including
from launch files and any `rospy.get_param` lookups.

Known candidates spotted while reading the code (confirm each before
removing, don't take this list as pre-verified):

- **`execute_grasp` in `locobot_motion_node.py`** — explicitly commented
  as `"Legacy single-shot command: both phases back to back, no decision
  point. Kept for direct-command callers that don't need the two-phase
  split."` `get_object` only ever calls `execute_pregrasp`/
  `execute_descent` today. If nothing else in the package sends this
  command, drop the handler (and the dispatch entry) once the services
  split is done — carrying a second, unused way to trigger a grasp forward
  into the new `manipulation_node` adds a maintenance path with no caller.
- **`target_aliases` rospy param plumbing in `yolo_detection_node.py`**
  (`raw_aliases`, the optional-override branch, `self.target_aliases`) —
  once `_ALIAS_MAP` is fully replaced by `label_reconciler`, this whole
  mechanism (and the corresponding comment block in `vision_nodes.launch`
  about auto-derived vs. fixed alias lists) has no reason to exist;
  remove it rather than leaving it as a second, now-redundant
  reconciliation path.
- **Stale launch-file references** — `locobot_table_arm.launch`'s header
  comment lists `vlm_process_rate:=0.5` as an optional override, but no
  such arg is declared or used anywhere in `vision_nodes.launch`. Either
  the param is genuinely gone (delete the comment) or it was meant to
  exist and never got wired up (confirm which with me before assuming).
- **Any import, constant, or helper that only existed to support code
  being deleted above** — e.g. if removing `execute_grasp` leaves some
  helper only it called, remove that too, transitively.
- **Do not treat the cobot project's own unused variants
  (`pick_place.cpp`, `pick_place_test.cpp`, `ur5_pick_place*.cpp`,
  `ik_test.cpp`, etc.) as something to clean up** — that's a separate
  repo kept only as a reference; `pick_place_simple_client_class.cpp` is
  the one file to actually look at, the rest can be ignored entirely.

After each node split (steps 3-7 in "Order of work" above), do a pass on
that specific file for now-dead imports and helpers before moving to the
next step, rather than deferring all dead-code cleanup to the end.

## What I want from you (Claude Code) in this session

1. Before writing any code, read the full bodies of `_do_pregrasp`,
   `_do_descent`, and `_handle_place` in `locobot_motion_node.py`, and the
   full `get_object` method in `vlm_agent_node.py` (I've only summarized
   these above) — confirm your understanding against mine before
   proposing `.srv` fields.
2. Propose the `.srv` definitions listed in step 2 above and check them
   with me before generating node code.
3. Implement in the order above, one step at a time, pausing for review
   after each — do not generate the whole restructure in one pass.
4. Anywhere you're unsure whether a piece of logic is "measured/tuned"
   (keep verbatim) vs. "structural scaffolding" (free to reorganize),
   ask rather than guess — the comments in the existing files usually say
   which, but flag genuine ambiguity.
5. Do not modify `arm_gains_override.yaml` or the `gazebo_grasp_fix` block
   in `grasp_fix_urdf.xacro` under any circumstances.
6. After each step, report what you removed as dead code (and why you
   confirmed it had zero callers) alongside what you built — this should
   be visible in the review, not silent.
