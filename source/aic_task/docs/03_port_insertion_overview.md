---
scope: complete behavioral spec of AIC-Port-Insertion-v0 — scene, action, command, observations, events, terminations, episode contract
audience: AI agents working in this repo
last_verified_commit: bb6a606
related:
  - 01_package_structure.md
  - 02_gym_registration.md
  - ../aic_task/tasks/manager_based/port_insertion/README.md
  - ../aic_task/asset_specs/README.md
  - 06_diff_ik_contract.md
---

# 03 · `AIC-Port-Insertion-v0` Overview

The one Gym ID this package registers. Goal: drive a UR5e end-of-arm tool
(an SFP plug rigidly attached past the gripper) into a fixed NIC-card SFP
port mounted on a randomized task board.

Source files referenced throughout:

- Registration: [tasks/manager_based/port_insertion/__init__.py](../aic_task/tasks/manager_based/port_insertion/__init__.py)
- Env cfg: [tasks/manager_based/port_insertion/port_insertion_env_cfg.py](../aic_task/tasks/manager_based/port_insertion/port_insertion_env_cfg.py)
- Selection layer: [tasks/manager_based/port_insertion/specs.py](../aic_task/tasks/manager_based/port_insertion/specs.py)
- Spec-to-cfg builders: [tasks/manager_based/port_insertion/builders.py](../aic_task/tasks/manager_based/port_insertion/builders.py)
- MDP terms: [tasks/manager_based/port_insertion/mdp/](../aic_task/tasks/manager_based/port_insertion/mdp/)
- Asset/layout contracts: [asset_specs/](../aic_task/asset_specs/)

## TL;DR table

| Field | Value | Source |
|---|---|---|
| Gym ID | `AIC-Port-Insertion-v0` | `port_insertion/__init__.py` |
| Env class | `isaaclab.envs:ManagerBasedRLEnv` | `port_insertion/__init__.py` |
| Env cfg | `PortInsertionEnvCfg` (extends `ManagerBasedRLEnvCfg`) | `port_insertion_env_cfg.py` |
| `sim.dt` | `1/120 s` | `port_insertion_env_cfg.py:43` |
| `decimation` | `4` (⇒ control freq = 30 Hz) | `port_insertion_env_cfg.py:40` |
| `episode_length_s` | `120.0 s` | `port_insertion_env_cfg.py:42` |
| Action term | `arm_action`: relative-mode DiffIK pose on `gripper_tcp` (6-D) | `specs.py:UR5E_DIFF_IK_CONTROLLER`, `builders.build_action_cfg` |
| Command term | `insertion_goal`: world-frame entrance + seat poses (14-D) | `mdp/commands.py`, `specs.py:NIC_PORT_0_INSERTION_GOAL` |
| Observation group | `policy` (concatenated, no corruption) | `builders.build_observation_cfg` |
| Reward manager | **empty** — no shaping | `builders.build_empty_reward_cfg` |
| Termination terms | `success`, `failed_stationary`, `time_out` | `builders.build_termination_cfg` |
| Reset events | `reset_robot_joints`, `randomize_light`, `randomize_board_and_parts` | `builders.build_event_cfg` |
| RL runner cfg | `PPORunnerCfg` (RSL-RL on-policy PPO) | `agents/rsl_rl_ppo_cfg.py` |

## Scene

Built by `build_scene_cfg(AIC_PORT_INSERTION_ASSEMBLY)`. The assembly's
layout is `AIC_PORT_INSERTION_LAYOUT` from
[asset_specs/scene.py](../aic_task/asset_specs/scene.py).

Scene slots (stable names — use these as `env.scene[name]` keys):

| Slot key | Role | Asset | USD | Default pose `(x,y,z)`, quat `(w,x,y,z)` | Notes |
|---|---|---|---|---|---|
| `robot` | robot (articulation) | `UR5E_CABLE_ASSET` | `assets/robots/ur5e_cable/aic_unified_robot_cable_sdf.usd` | `(-0.18, -0.122, 0.0)`, `(0,0,0,1)` | TCP body = `gripper_tcp`, EEF body = `sfp_tip_link` |
| `workcell` | static workcell | `AIC_WORKCELL_ASSET` | `assets/workcells/aic/aic.usd` | `(0, 0, -1.15)`, identity | Visual surroundings; not in MDP |
| `board` | board (rigid, kinematic) | `TASK_BOARD_ASSET` | `assets/workcells/task_board/task_board_rigid.usd` | `(0.2837, 0.229, 0.0)`, identity | Anchor for board-relative randomization |
| `sc_port_1`, `sc_port_2` | auxiliary (rigid, kinematic) | `SC_PORT_ASSET` | `assets/targets/sc_port/sc_port.usd` | layout-specified | Distractor / passive parts |
| `target` | target (rigid, kinematic) | `NIC_CARD_ASSET` | `assets/targets/nic_card/nic_card.usd` | `(0.25135, 0.25229, 0.0743)`, `(0, 0, -0.7068, 0.7074)` | Contains the `sfp_port_0` we insert into |
| `ground` | infrastructure | — | `GroundPlaneCfg` at `z = -1.05` | (added by `build_scene_cfg`) | |
| `light` | infrastructure | — | dome light | (added by `build_scene_cfg`) | Randomized at reset |

Per-camera cfg comes from `UR5E_CABLE_ASSET.camera_frames` (`center_camera`,
`left_camera`, `right_camera`) and is built with `TiledCameraCfg`. The current
policy observation group does **not** consume images — cameras exist for
recording / replay scripts and for whoever wants vision later.

## Action — `arm_action`

`build_action_cfg` reads `UR5E_DIFF_IK_CONTROLLER` from `specs.py` and produces
a single action term named `arm_action`:

```python
ControllerSpec(
    name="ur5e_diff_ik_tcp",
    action_name="arm_action",
    action_type="diff_ik",
    robot_slot="robot",
    joint_group="arm",          # 6 UR5e arm joints
    controlled_body_role="tcp", # → "gripper_tcp"
    command_type="pose",
    use_relative_mode=True,
    ik_method="dls",
    ik_params={"lambda_val": 0.01},
    scale=(0.015, 0.015, 0.015, 0.025, 0.025, 0.025),
)
```

Concretely this means:

- 6-D action `(dx, dy, dz, drx, dry, drz)`.
- Multiplied by `scale` element-wise; the result is treated as a delta pose
  in the **robot root (base) frame**, then added to the current `gripper_tcp`
  pose via `apply_delta_pose`.
- The delta IK is damped-least-squares with `λ² = 0.01`.

> Subtle but important: errors must be computed in the robot root frame
> before being sent as actions. See
> [06_diff_ik_contract.md](06_diff_ik_contract.md)
> for the full contract and the world-vs-root pitfall.

The action does **not** include a gripper command — the SFP plug is rigidly
attached to `sfp_tip_link`, so there's nothing to close.

## Command — `insertion_goal`

Built by `build_command_cfg`. Implemented in
[mdp/commands.py](../aic_task/tasks/manager_based/port_insertion/mdp/commands.py)
as `InsertionGoalCommand`.

Configuration (from `NIC_PORT_0_INSERTION_GOAL`):

| Field | Value | Meaning |
|---|---|---|
| `target_scene_name` | `"target"` | Scene slot whose root pose carries the port. |
| `target_root_prim` | `"nic_card_link"` | Root prim of the NIC card USD. |
| `port_name` | `"sfp_port_0"` | The single port defined on `NIC_CARD_ASSET`. |
| `port_seat_frame_path` | `"/sfp_port_0_link"` | USD path (relative to target root) for the fully-inserted "seat" pose. |
| `port_entrance_frame_path` | `"/sfp_port_0_link/sfp_port_0_link_entrance"` | USD path for the "entrance" pose just outside the port. |
| `eef_pos_in_port_frame` | `(0.0, 0.0, 0.001)` | EEF position the goal targets, expressed in the port frame. |
| `eef_quat_in_port_frame` | `(√0.5, √0.5, 0, 0)` | EEF orientation in the port frame. |
| `resampling_time_range` | `(1e9, 1e9 + 1)` | Effectively never resampled mid-episode; the goal is locked at reset. |
| `debug_vis` | `False` | No built-in marker drawing. |

At runtime the term publishes a 14-D command tensor laid out as:

```text
[ entrance_pos_w (3) | entrance_quat_w (4) | seat_pos_w (3) | seat_quat_w (4) ]
```

It also exposes named tensors for direct use by other code:

- `goal.entrance_pos_w`, `goal.entrance_quat_w`
- `goal.seat_pos_w`, `goal.seat_quat_w`
- Compatibility aliases `goal.final_tip_pos_w` (== seat pos) and
  `goal.target_tip_quat_w` (== seat quat) used by the termination terms.

How an agent reads it (used by `scripts/random_agent.py`):

```python
goal = env.unwrapped.command_manager.get_term("insertion_goal")
target_pos = goal.entrance_pos_w     # or goal.seat_pos_w
target_quat = goal.entrance_quat_w   # or goal.seat_quat_w
```

## Observation group — `policy`

`build_observation_cfg` returns `{"policy": PolicyCfg()}`. The group is
concatenated (`concatenate_terms=True`) and corruption is off
(`enable_corruption=False`).

| Term | Function | Source | Dim (UR5e) |
|---|---|---|---|
| `joint_pos` | `joint_pos_rel` over the arm joint group | `isaaclab.envs.mdp` | 6 |
| `joint_vel` | `joint_vel_rel` over the arm joint group | `isaaclab.envs.mdp` | 6 |
| `actions` | `last_action(action_name="arm_action")` | `isaaclab.envs.mdp` | 6 |
| `insertion_goal` | `generated_commands(command_name="insertion_goal")` | `isaaclab.envs.mdp` | 14 |

Total flat observation size = 32 floats per env.

There is currently **no image observation** in the policy group. Camera USDs
are spawned for recording / data collection, not for the policy.

## Reward manager

Empty: `build_empty_reward_cfg()` returns `{}`. The task today is intended for
oracle / heuristic / scripted policies (and as a substrate to add reward
shaping on). RL training that expects nonzero rewards will need a custom
reward cfg before it learns anything useful.

## Termination terms

Built by `build_termination_cfg`. All three are returned in a dict.

### `success` — `InsertionGoalReachedSuccess`

Stateful per-env counter. Each step it computes:

- `position_error = ‖tip_pos_w − goal.seat_pos_w‖`
- `orientation_error = quat_error_magnitude(tip_quat_w, goal.seat_quat_w)`

where `tip_body = "sfp_tip_link"`. If both errors fall below their thresholds,
the success counter increments; otherwise it resets to 0. The term fires when
the counter reaches `ceil(required_seconds / env_step_dt)` consecutive steps.

Thresholds (from `AIC_PORT_INSERTION_TERMINATION`):

| Param | Value |
|---|---|
| `position_threshold` | `0.003 m` |
| `orientation_threshold` | `radians(4°)` |
| `required_seconds` | `0.5 s` (≈ 15 steps at 30 Hz control) |

### `failed_stationary` — `InsertionGoalStationaryFailure`

Stateful per-env anchor. Each step:

1. If the tip has moved more than `movement_threshold` from the anchor, reset
   the anchor to the current tip pose and reset the stable-step counter to 1.
2. Otherwise, increment the stable-step counter.
3. Fire when `stable_steps >= ceil(required_seconds / dt)` **and** the tip is
   still outside `success_position_threshold` of the goal.

Thresholds:

| Param | Value |
|---|---|
| `movement_threshold` | `0.001 m` |
| `success_position_threshold` | `0.003 m` |
| `required_seconds` | `1.0 s` (≈ 30 steps at 30 Hz control) |

### `time_out`

Standard IsaacLab `time_out` term with `time_out=True` — terminates when
`episode_length_s` is exceeded. Marked as a truncation, not a true failure,
in the env's `dones`/`truncations` split.

## Reset events

Built by `build_event_cfg`. Run on `mode="reset"` for each terminating env.

| Event | Function | What it does |
|---|---|---|
| `reset_robot_joints` | `reset_joints_by_offset` | Resets the UR5e arm joints to the default joint positions from `UR5E_ARM_JOINT_GROUP`, no random offset (`position_range=(0,0)`, `velocity_range=(0,0)`). |
| `randomize_light` | `randomize_dome_light` | Samples a new dome-light intensity in `(1500, 3500)` and color in `[(0.5,0.5,0.5), (1,1,1)]`. |
| `randomize_board_and_parts` | `randomize_board_and_parts` | Jitters the board in `x/y` and yaw, carries board-relative parts (the two SC ports + the NIC card target) along with the board, plus optional per-part jitter and grid snap. Optionally writes the new pose into USD xforms when `sync_usd_xforms=True`. |

Randomization ranges (from `AIC_PORT_INSERTION_RANDOMIZATION`):

- Board: `x ∈ [-0.04, 0.04] m`, `y ∈ [-0.04, 0.04] m`, `yaw ∈ [-0.35, 0.35] rad`.
- `sc_port_1`, `sc_port_2`: anchored at fixed board-local offsets, `x` jittered in `[-0.005, 0.02]`.
- `target` (NIC card): anchored at board-local offset `(-0.03235, 0.02329, 0.0743)`,
  `y` jittered in `[0.0, 0.12]` snapped to a `0.04` grid (so the card lands in one of four discrete y-slots).

## Episode contract

Putting it together for one episode of a single env:

1. On reset:
   - Robot joints snap to the default UR5e arm pose.
   - Dome light is resampled.
   - Board pose is sampled; `sc_port_1`, `sc_port_2`, and `target` are placed
     relative to the new board pose with their own jitter / snap.
   - `InsertionGoalCommand` reads the current target / port frames and locks
     in `entrance_pos_w / quat_w` and `seat_pos_w / quat_w` for the rest of
     the episode (resampling time is effectively infinite).
2. Each step (30 Hz control, 120 Hz physics):
   - Policy / agent emits 6-D `arm_action` in `[-1, 1]`.
   - DiffIK in pose+relative mode computes a target TCP pose in the root
     frame and solves for joint targets.
   - Observation group is published (`joint_pos`, `joint_vel`, `actions`,
     `insertion_goal`).
3. Termination:
   - `success` after `≥ 0.5 s` continuous within `3 mm` + `4°` of the seat
     pose.
   - `failed_stationary` after `≥ 1.0 s` of `< 1 mm` motion while still
     outside `3 mm` of the seat.
   - `time_out` after `120 s` of wall time.

`PortInsertionEnvCfg.__post_init__` is where `sim.dt`, `decimation`, and
`episode_length_s` are set; the assembly's specs control everything else.

## Things this task deliberately does **not** ship

- **No reward terms.** Use the empty reward manager and add your own cfg when
  training.
- **No image observations in the policy group.** Cameras are present in the
  scene; consume them at recording time.
- **No mid-episode goal resampling.** `resampling_time_range = (1e9, …)`
  locks the goal at reset. Don't expect curriculum-style goal changes.
- **No gripper action.** The plug is rigidly attached.
- **No oracle / scripted controller in the task itself.** Scripted policies
  live in `scripts/` (`random_agent.py` is a goal-driven demo, not a
  ground-truth oracle).
