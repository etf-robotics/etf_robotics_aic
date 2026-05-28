---
scope: reference for every MDP term provided by aic_task — signatures, inputs, outputs, state, knobs, edge cases
audience: AI agents working in this repo
last_verified_commit: 8d9a44e
related:
  - 03_port_insertion_overview.md
  - 04_assembly_pattern.md
  - ../aic_task/tasks/manager_based/port_insertion/mdp/README.md
---

# 05 · MDP Terms

Every MDP term this package owns, plus the re-export pattern used by the
local `mdp/` submodule. Use this doc as a lookup; the behavioral spec for
the env as a whole lives in
[03_port_insertion_overview.md](03_port_insertion_overview.md).

All file:line references below were verified against
`last_verified_commit`. Re-verify before editing — the repo moves.

## Re-export pattern in `mdp/__init__.py`

[mdp/__init__.py](../aic_task/tasks/manager_based/port_insertion/mdp/__init__.py)
exists so the task's own term modules and `builders.py` can pull both the
core IsaacLab MDP library and the local terms from one namespace. It does
two things and nothing else:

```python
from isaaclab.envs.mdp import *                       # everything in the core
from isaaclab.envs.mdp import (                       # explicit, IDE-discoverable
    UniformPoseCommandCfg, action_rate_l2, body_pose_w, generated_commands,
    image, joint_pos_rel, joint_vel_l2, joint_vel_rel, last_action,
    reset_joints_by_scale, root_pos_w, root_quat_w, time_out,
)
from .commands import *                               # adds InsertionGoalCommand[Cfg]
```

Notes:

- The explicit list duplicates names already pulled in by the star import.
  It exists to make those names visible to static analyzers and to lock in
  a stable contract — anything dropped from IsaacLab core surfaces as an
  ImportError here rather than as a silent NameError at first use.
- Only `commands` is re-exported. `events` and `terminations` are imported
  by `builders.py` directly with their fully-qualified names; they are not
  surfaced via `mdp.<symbol>`. Stay consistent — when adding a new
  termination or event, leave it out of `mdp/__init__.py` unless you have
  a specific reason to expose it.
- The folder is blacklisted from task auto-discovery via `_BLACKLIST_PKGS`
  in [tasks/__init__.py](../aic_task/tasks/__init__.py); see
  [02_gym_registration.md](02_gym_registration.md). Importing a file under
  `mdp/` does not register a Gym ID, which is exactly what you want.

## `InsertionGoalCommand`

A stateful, world-frame command term that publishes the EEF goal poses
for the selected port's entrance and seat frames.

| Field | Value |
|---|---|
| Symbol | [`InsertionGoalCommand`](../aic_task/tasks/manager_based/port_insertion/mdp/commands.py#L19) |
| Cfg symbol | [`InsertionGoalCommandCfg`](../aic_task/tasks/manager_based/port_insertion/mdp/commands.py#L164) |
| Base class | `isaaclab.managers.CommandTerm` |
| Registered as | `insertion_goal` command term (name set by `InsertionGoalSpec.command_name` in [specs.py:54](../aic_task/tasks/manager_based/port_insertion/specs.py#L54)) |
| Built by | [`build_command_cfg`](../aic_task/tasks/manager_based/port_insertion/builders.py#L130) |

### Cfg fields (all default-valued)

[InsertionGoalCommandCfg](../aic_task/tasks/manager_based/port_insertion/mdp/commands.py#L164):

| Field | Type | Default | Set by builder from | Meaning |
|---|---|---|---|---|
| `target_scene_name` | `str` | `"target"` | `goal.target_slot` | Scene slot whose root pose carries the port. |
| `target_root_prim` | `str \| None` | `"nic_card_link"` | `target.usd.root_prim` | Root prim of the target USD; used to disambiguate prim resolution. |
| `port_name` | `str` | `"sfp_port_0"` | `port.name` | The named `TargetPortSpec` consumed. |
| `port_seat_frame_path` | `str` | `"/sfp_port_0_link"` | `port.seat_frame_path` | USD path (relative to target root) of the fully-inserted "seat" pose. |
| `port_entrance_frame_path` | `str \| None` | `"/sfp_port_0_link/sfp_port_0_link_entrance"` | `port.entrance_frame_path` | USD path of the "entrance" pose just outside the port. |
| `eef_pos_in_port_frame` | `(float,)*3` | `(0,0,0)` | `goal.eef_pose_in_port_frame.pos` | EEF target position expressed in the port frame. |
| `eef_quat_in_port_frame` | `(float,)*4` (wxyz) | `(1,0,0,0)` | `goal.eef_pose_in_port_frame.rot` | EEF target orientation expressed in the port frame. |
| `resampling_time_range` | `(float, float)` | `(1e9, 1e9+1)` | `goal.resampling_time_range` | Effectively never resampled mid-episode. Inherited from `CommandTermCfg`. |
| `debug_vis` | `bool` | `False` | `goal.debug_vis` | No built-in marker drawing. |

Builder-side validation lives in
[`_validate_command_cfg`](../aic_task/tasks/manager_based/port_insertion/builders.py#L382)
and rejects empty frame paths, mismatched slot/port names, and a zero
orientation quaternion.

### What it reads

- `env.scene[cfg.target_scene_name]` — the target rigid object; reads
  `.data.root_pos_w` and `.data.root_quat_w` each step.
- On resample: walks the live USD stage via `omni.usd.get_context()` to
  find the port entrance/seat prims relative to the asset root. The prim
  resolver tries the literal path, the path with the `target_root_prim`
  prepended, the path with that prefix stripped, and a basename descendant
  search (see
  [`_candidate_prim_paths`](../aic_task/tasks/manager_based/port_insertion/mdp/commands.py#L301)).
  This is what makes the term tolerant of "spawned with defaultPrim
  nesting" vs. "spawned flat" USD layouts.

### What it writes

The compact command tensor is 14-D `float`:

```text
self._command[:, 0:3]  = entrance_pos_w   # world position of the EEF entrance goal
self._command[:, 3:7]  = entrance_quat_w  # world orientation (wxyz)
self._command[:, 7:10] = seat_pos_w       # world position of the EEF seat goal
self._command[:, 10:14] = seat_quat_w     # world orientation (wxyz)
```

Set in
[`_update_command`](../aic_task/tasks/manager_based/port_insertion/mdp/commands.py#L141).
The named tensors are also exposed as attributes so other terms (and
agents) can read them without unpacking the 14-D layout:

| Attribute | Shape | Description |
|---|---|---|
| `entrance_pos_w`, `entrance_quat_w` | `(N, 3)`, `(N, 4)` | Entrance pose in world frame. |
| `seat_pos_w`, `seat_quat_w` | `(N, 3)`, `(N, 4)` | Seat pose in world frame. |
| `entrance_pos_root`, `entrance_quat_root` | `(N, 3)`, `(N, 4)` | Entrance pose in the target root frame; recomputed only on resample. |
| `seat_pos_root`, `seat_quat_root` | `(N, 3)`, `(N, 4)` | Same, for the seat. |
| `final_tip_pos_w` | `(N, 3)` | Alias for `seat_pos_w` ([commands.py:63](../aic_task/tasks/manager_based/port_insertion/mdp/commands.py#L63)). Consumed by `InsertionGoalReachedSuccess` / `InsertionGoalStationaryFailure`. |
| `target_tip_quat_w` | `(N, 4)` | Alias for `seat_quat_w` ([commands.py:69](../aic_task/tasks/manager_based/port_insertion/mdp/commands.py#L69)). Same consumers. |

### Internal state and lifecycle

- `entrance_pos_root` / `entrance_quat_root` / `seat_pos_root` /
  `seat_quat_root` are filled once per resample in
  [`_resample_command`](../aic_task/tasks/manager_based/port_insertion/mdp/commands.py#L78)
  by walking the live USD stage. They are the constant
  port-pose-in-target-root for this episode.
- `entrance_pos_w` / `seat_pos_w` (and quats) are recomputed every step by
  composing the constant root-frame poses with the current target world
  pose. This is what handles the board / target moving mid-episode (the
  goal "follows" the target).
- Default `resampling_time_range` of `(1e9, 1e9+1)` means the resample
  effectively never fires mid-episode; the goal is locked at the first
  step of the episode and follows the target pose thereafter.

### Edge cases worth knowing

- The term *requires* the port frame to exist as a USD prim under the
  target asset's root. The resolver falls back to a basename traversal,
  but if the prim doesn't exist under any candidate path the resample
  raises `KeyError` with the candidate list. That kills the env reset.
- `_resample_command` materializes a `torch.float64` tensor per env-id
  via per-env iteration. With a large `num_envs` and a high reset
  frequency this is a real cost; today the resample is effectively never
  called mid-episode.
- `_update_metrics` is a no-op
  ([commands.py:75](../aic_task/tasks/manager_based/port_insertion/mdp/commands.py#L75)).
  The base `CommandTerm` reporting hooks therefore have nothing to
  display.

## `randomize_dome_light`

| Field | Value |
|---|---|
| Symbol | [`randomize_dome_light`](../aic_task/tasks/manager_based/port_insertion/mdp/events.py#L27) |
| Mode | `"reset"` (wired by [build_event_cfg](../aic_task/tasks/manager_based/port_insertion/builders.py#L203)) |
| Registered as | `randomize_light` event term |

### Signature

```python
def randomize_dome_light(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    light_scene_name: str,
    intensity_range: tuple[float, float] = (1500.0, 3500.0),
    color_range: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.5, 0.5, 0.5),
        (1.0, 1.0, 1.0),
    ),
) -> None:
```

Knobs surfaced via `build_event_cfg`'s `params=...`
([builders.py:203](../aic_task/tasks/manager_based/port_insertion/builders.py#L203)):
`light_scene_name="light"` (the slot added inside `build_scene_cfg` —
see [builders.py:79](../aic_task/tasks/manager_based/port_insertion/builders.py#L79)),
plus the intensity / color ranges shown above. Defaults are repeated in
the builder; both must match if you change either.

### What it does

- Looks up the dome-light prim via the env's scene cfg
  (`env.scene.cfg.<light_scene_name>.prim_path`).
- Samples a single scalar intensity in `intensity_range` and a single RGB
  color tuple in `color_range` and writes them via USD attribute setters
  on the `UsdLux.DomeLight` schema.
- **Ignores `env_ids`** — the dome light is a single stage-level prim
  shared across all envs. If you spawn per-env lights you need a different
  event.

### State / failure

- Stateless. Returns `None`. If the light prim is invalid (e.g. spawned
  under a different name), the function returns early without raising,
  which means a typo in `light_scene_name` silently no-ops. Cross-check
  against the constant `_LIGHT_SCENE_NAME` in
  [builders.py:44](../aic_task/tasks/manager_based/port_insertion/builders.py#L44).

## `randomize_board_and_parts`

The board-and-parts reset event. The board's pose is jittered each reset,
and each board-relative part (the two SC ports and the NIC card) follows
the board with its own optional jitter and grid snap.

| Field | Value |
|---|---|
| Symbol | [`randomize_board_and_parts`](../aic_task/tasks/manager_based/port_insertion/mdp/events.py#L130) |
| Mode | `"reset"` |
| Registered as | `randomize_board_and_parts` event term |

### Signature

```python
def randomize_board_and_parts(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    board_slot_name: str,
    board_default_position: tuple[float, float, float],
    board_pose_ranges: Mapping[str, tuple[float, float]],
    board_relative_parts: Sequence[Mapping[str, object]],
    sync_usd_xforms: bool = True,
) -> None:
```

The builder
([builders.py:227](../aic_task/tasks/manager_based/port_insertion/builders.py#L227))
unpacks the layout's `randomization` spec (a
[`LayoutRandomizationSpec`](../aic_task/asset_specs/scene.py#L48)) into
the plain `Mapping[axis, bounds]` and `Mapping[axis, step]` dicts this
function accepts.

`board_relative_parts` is a list of dicts of the form:

```python
{
    "slot_name": "sc_port_1",
    "board_local_offset": (0.0067, -0.0362, 0.005),
    "pose_ranges": {"x": (-0.005, 0.02)},
    "snap_steps": {},
}
```

### What it does (per reset env)

1. Caches each randomized slot's initial root orientation in
   `_cached_orientations[name]` on first use
   ([events.py:24](../aic_task/tasks/manager_based/port_insertion/mdp/events.py#L24)).
   The cache lives at module scope and persists for the process lifetime.
2. Samples the board's `x`, `y`, and `yaw` offsets from
   `board_pose_ranges`; falls back to `(0, 0)` if an axis is missing.
3. Composes the new world board pose
   `(board_default_position + dxy, env_origin, yaw·cached_quat)` and
   writes it to the simulation via
   `board_asset.write_root_pose_to_sim(...)` and zero velocity.
4. For each part: samples per-axis offsets (with optional grid snap via
   [`_sample_axis`](../aic_task/tasks/manager_based/port_insertion/mdp/events.py#L55)),
   rotates the part's `(x, y)` offset by the board's yaw, and writes the
   composed world pose plus an additional yaw jitter on the part.
5. When `sync_usd_xforms=True`, also writes the new pose to the underlying
   USD `Xformable` ops via
   [`_write_usd_xform_pose`](../aic_task/tasks/manager_based/port_insertion/mdp/events.py#L90).
   This is what keeps the USD visualization in sync with the physics-side
   `RigidObject` state — required for any code that reads pose from the
   stage (e.g. `InsertionGoalCommand`'s USD walk on resample).

### State

- Module-level `_cached_orientations` dict: `slot_name → (num_envs, 4)`
  tensor of the slot's wxyz quat at first encounter. It is *not* reset
  between episodes; only the first reset for that slot populates it.
- Each call samples fresh; no per-env counter.

### Edge cases worth knowing

- Per-part `pose_ranges`/`snap_steps` are looped over `num_resets` in
  Python
  ([events.py:198](../aic_task/tasks/manager_based/port_insertion/mdp/events.py#L198))
  rather than vectorized. With many envs and many parts this dominates
  reset time.
- The function uses Python `random.randint` for snap sampling and
  `torch.empty(...).uniform_(...)` for continuous sampling. Neither
  participates in IsaacLab's seeding infrastructure, so reproducibility
  across runs depends on each library's separate seed state.
- If a part dict references a slot the layout doesn't define, the lookup
  inside the builder
  ([builders.py:218](../aic_task/tasks/manager_based/port_insertion/builders.py#L218))
  raises `KeyError` at build time, before the env starts. Good.
- `sync_usd_xforms=True` opens the live stage from inside the event,
  which is fine inside Isaac Sim but unusable in environments without an
  active stage. For tests that build cfgs without spawning, the event is
  never called.

## `InsertionGoalReachedSuccess`

The success termination. Stateful per-env counter — fires only after
*consecutive* in-tolerance steps add up to `required_seconds`.

| Field | Value |
|---|---|
| Symbol | [`InsertionGoalReachedSuccess`](../aic_task/tasks/manager_based/port_insertion/mdp/terminations.py#L23) |
| Base class | `isaaclab.managers.ManagerTermBase` |
| Registered as | `success` termination ([builders.py:250](../aic_task/tasks/manager_based/port_insertion/builders.py#L250)) |
| Failure-side counterpart | `InsertionGoalStationaryFailure` |

### Call signature

```python
def __call__(
    self,
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "insertion_goal",
    tip_body: str = "sfp_tip_link",
    position_threshold: float = 0.003,
    orientation_threshold: float = math.radians(4.0),
    required_seconds: float = 0.5,
) -> torch.Tensor
```

The builder populates `params={...}` from
`PortInsertionTerminationSpec` and the EEF body role
([builders.py:250](../aic_task/tasks/manager_based/port_insertion/builders.py#L250)):

- `asset_cfg = SceneEntityCfg(robot_slot_name, body_names=eef_body)`
- `tip_body = robot.body_name_for_role(ROBOT_ROLE_EEF)` (`"sfp_tip_link"` today)
- `position_threshold = 0.003`
- `orientation_threshold = math.radians(4.0)`
- `required_seconds = 0.5`

### What it reads

Via [`_insertion_goal_tip_errors`](../aic_task/tasks/manager_based/port_insertion/mdp/terminations.py#L110):

- `env.scene[asset_cfg.name]` — the robot articulation; reads
  `robot.data.body_pos_w[:, tip_id, :]` and `.body_quat_w[:, tip_id, :]`.
- `env.command_manager.get_term(command_name)` — the
  `InsertionGoalCommand` instance; reads `goal.final_tip_pos_w` and
  `goal.target_tip_quat_w` (i.e. the seat pose).

`tip_id` is resolved with
[`_first_body_id`](../aic_task/tasks/manager_based/port_insertion/mdp/terminations.py#L129),
which uses `asset_cfg.body_ids` if already resolved, then
`asset_cfg.body_names`, then the `tip_body` fallback. The builder sets
`body_names=tip_body`, so the fallback path covers a `SceneEntityCfg` that
the env hasn't yet resolved.

### What it writes

Return value: `torch.BoolTensor(shape=(num_envs,))`. `True` means the
termination should fire for that env.

### Internal state

- `self._success_counts: torch.Tensor` of shape `(num_envs,)`, `int32`.
  Each step:
  - `reached = (pos_err <= position_threshold) & (rot_err <= orientation_threshold)`
  - `_success_counts = where(reached, _success_counts + 1, 0)`
  - returns `_success_counts >= ceil(required_seconds / env_step_dt)`
- `reset(env_ids)` zeroes the counts for the resetting envs.
- `env_step_dt` is read via
  [`_env_step_dt`](../aic_task/tasks/manager_based/port_insertion/mdp/terminations.py#L148)
  which prefers `env.step_dt`, falls back to
  `env.cfg.sim.dt * env.cfg.decimation`, and finally `1.0` if both are
  missing.

### Edge cases worth knowing

- The required step count is `max(1, ceil(required_seconds / dt))`. With
  the default `sim.dt = 1/120`, `decimation = 4`, `required_seconds = 0.5`,
  this is `ceil(0.5 / (4/120)) = 15` steps.
- The counter resets to *zero* on any out-of-tolerance step. The success
  is not "average over N steps within tolerance"; it is "uninterrupted N
  steps within tolerance".
- `orientation_threshold` is in *radians*. The spec stores it as
  `math.radians(4.0)` ([specs.py:140](../aic_task/tasks/manager_based/port_insertion/specs.py#L140)).

## `InsertionGoalStationaryFailure`

The "robot is parked outside the goal" failure. Detects sustained
near-zero motion of the tip while still outside the goal's success
radius.

| Field | Value |
|---|---|
| Symbol | [`InsertionGoalStationaryFailure`](../aic_task/tasks/manager_based/port_insertion/mdp/terminations.py#L57) |
| Base class | `isaaclab.managers.ManagerTermBase` |
| Registered as | `failed_stationary` termination ([builders.py:261](../aic_task/tasks/manager_based/port_insertion/builders.py#L261)) |

### Call signature

```python
def __call__(
    self,
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "insertion_goal",
    tip_body: str = "sfp_tip_link",
    movement_threshold: float = 0.001,
    success_position_threshold: float = 0.003,
    required_seconds: float = 1.0,
) -> torch.Tensor
```

Builder-supplied params:

- `asset_cfg`, `command_name`, `tip_body` — identical to the success
  term (same robot, same command, same body).
- `movement_threshold = 0.001` — meters; "still" if `||tip - anchor|| ≤ threshold`.
- `success_position_threshold = 0.003` — meters; the radius around the
  *seat* outside which "stationary" counts as failure (so successes
  beat failures on a tie).
- `required_seconds = 1.0`.

### What it reads

Same as the success term — robot body pose and the command term's
`final_tip_pos_w`. Quaternion error is *not* used.

### What it writes

Return value: `torch.BoolTensor(shape=(num_envs,))`.

### Internal state

- `self._anchor_pos_w: (num_envs, 3) float`. The last known "stationary"
  position. Re-set to the current tip pose any time the tip moves more
  than `movement_threshold` from it.
- `self._stable_counts: (num_envs,) int32`. Counter of consecutive steps
  within `movement_threshold` of the anchor.
- `self._initialized: (num_envs,) bool`. First-step gate. On the very
  first call after a reset, the anchor is set to the current tip pose
  and the counter starts at 1.
- `reset(env_ids)` clears all three.

The fire condition is
`(_stable_counts >= ceil(required_seconds/dt)) AND (||tip - seat|| > success_position_threshold)`.
The second clause is what prevents this term from racing the success
term when the robot is parked *on* the goal.

### Edge cases worth knowing

- On the very first step the anchor is set to the current pose, but the
  counter is also set to 1 (via `reset_window | uninitialized` →
  `_stable_counts = one_step`). So even if the robot is perfectly still
  from t=0, the first 30 control steps build up the counter — the term
  cannot fire on step 1.
- The anchor uses `tip_pos_w.dtype/device` lazily — see the
  `if self._anchor_pos_w.dtype != tip_pos_w.dtype ...` block at
  [terminations.py:90](../aic_task/tasks/manager_based/port_insertion/mdp/terminations.py#L90).
  Don't change `__init__` to `float32` without keeping that re-cast in
  sync.
- `success_position_threshold` here and `position_threshold` on the
  success term are different fields with the same numeric value
  (`0.003`). The spec stores them separately in
  `PortInsertionTerminationSpec` ([specs.py:62](../aic_task/tasks/manager_based/port_insertion/specs.py#L62));
  there is no enforced relation between them. If you tighten one, decide
  whether you want the other to move too.
