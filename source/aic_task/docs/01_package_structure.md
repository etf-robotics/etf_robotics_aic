---
scope: directory map and module roles of the aic_task Python package; dependency direction
audience: AI agents working in this repo
last_verified_commit: aaaa911
related:
  - ../aic_task/README.md
  - 02_gym_registration.md
  - ../aic_task/asset_specs/README.md
---

# 01 · Package Structure

This document is the ground-truth map of the `aic_task` Python package as of
the `last_verified_commit` above. Every module listed here is either a public
entry point or a piece a contributor will need to touch when changing the
task. If you can't see a file here, it doesn't exist in this package today.

The package distinguishes two layers and nothing else:

- **Data contracts** — `asset_specs/` describes assets and scene layouts as
  frozen dataclasses with no IsaacLab imports.
- **Task assembly** — `tasks/manager_based/<task>/` turns those contracts into
  IsaacLab cfg objects and registers a Gym ID.

All runtime users (demo, teleop, recording, RL training) live outside this
package in [`scripts/`](../../../scripts/).

## Directory map

```text
source/aic_task/
├── config/
│   └── extension.toml         Isaac Sim extension metadata (entry point, deps)
├── docs/
│   ├── 01_package_structure.md   ← you are here
│   ├── 02_gym_registration.md    (planned)
│   ├── 03_port_insertion_overview.md   (planned)
│   ├── …                              (planned, see aic_task/README.md)
│   └── CHANGELOG.rst
├── pyproject.toml             Build backend (setuptools + toml)
├── setup.py                   Reads extension.toml for version + deps
└── aic_task/
    ├── README.md              Package-level entry point
    ├── __init__.py            from .tasks import *  →  triggers gym.register
    ├── extension.py           Isaac Sim UI extension; only active inside the Sim UI
    ├── assets/                Raw USD assets (pure data, no Python)
    │   ├── robots/ur5e_cable/aic_unified_robot_cable_sdf.usd
    │   ├── targets/nic_card/nic_card.usd
    │   ├── targets/sc_port/sc_port.usd
    │   ├── workcells/aic/aic.usd
    │   └── workcells/task_board/task_board_rigid.usd
    ├── asset_specs/           Python contracts for assets / layouts (see its own README)
    │   ├── README.md
    │   ├── __init__.py        Re-exports the public spec API
    │   ├── base.py            AssetSpec, SceneSlotSpec, PoseSpec, asset_path, ASSET_DIR
    │   ├── robots.py          UR5E_CABLE_ASSET, ROBOT_ROLE_TCP / ROBOT_ROLE_EEF
    │   ├── targets.py         NIC_CARD_ASSET, SC_PORT_ASSET, port frame paths
    │   ├── workcells.py       AIC_WORKCELL_ASSET, TASK_BOARD_ASSET (static USDs)
    │   └── scene.py           SCENE_SLOT_* constants, AIC_PORT_INSERTION_LAYOUT,
    │                          board/part randomization spec
    └── tasks/
        ├── __init__.py        Auto-discovers task subpackages via
        │                      isaaclab_tasks.utils.import_packages
        └── manager_based/
            ├── __init__.py    (only imports gymnasium for side-effect parity)
            └── port_insertion/
                ├── __init__.py            gym.register("AIC-Port-Insertion-v0", …)
                ├── port_insertion_env_cfg.py   PortInsertionEnvCfg (@configclass)
                ├── specs.py               Assembly choice + ControllerSpec + GoalSpec
                ├── builders.py            spec → IsaacLab cfg objects
                ├── mdp/
                │   ├── __init__.py
                │   ├── commands.py        InsertionGoalCommand (entrance/seat pose)
                │   ├── events.py          randomize_dome_light, randomize_board_and_parts
                │   ├── observations.py    body_pose_b, body_vel_b, insertion_goal_b,
                │   │                          seat_pos_err_b, seat_quat_delta_b, insertion_fraction
                │   ├── recorders.py       PreStepGroupedObservationsRecorder +
                │   │                          GroupedActionStateRecorderManagerCfg
                │   └── terminations.py    InsertionGoalReachedSuccess, …StationaryFailure
                └── agents/
                    ├── __init__.py
                    └── rsl_rl_ppo_cfg.py  Training-side cfg consumed by scripts/rsl_rl/
```

## Module responsibilities

### Top of package

| Module | Owns | Touch when… |
|---|---|---|
| `aic_task/__init__.py` | Side-effect chain that registers Gym IDs and conditionally loads the Sim UI extension. | Adding a new top-level subpackage that needs eager import. |
| `aic_task/extension.py` | Isaac Sim UI buttons for asset-import debugging. Wrapped in a `try/except ModuleNotFoundError` in `__init__.py`, so it is a no-op in headless runs. | Only when working inside the Isaac Sim UI. |
| `aic_task/assets/` | USD files referenced by `asset_specs.base.asset_path(...)`. Pure data — no Python. | Replacing or adding USDs (and updating the matching spec). |

### `asset_specs/`

Pure-Python contracts. No IsaacLab cfg classes, no `omni.*` imports. This is
the layer that the rest of the codebase imports to ask "what's the USD path
for the robot?", "what body name is the TCP?", "where does the NIC card go in
the scene?", etc. The package has its own
[README](../aic_task/asset_specs/README.md) — read it before changing an asset.

| File | Owns |
|---|---|
| `base.py` | `AssetSpec`, `SceneSlotSpec`, `PoseSpec`, `BodyRoleSpec`, `JointGroupSpec`, `UsdAssetInterface`, plus `asset_path()` and `ASSET_DIR`. The dataclass primitives used everywhere else. |
| `robots.py` | `UR5E_CABLE_ASSET`, the `ROBOT_ROLE_TCP` / `ROBOT_ROLE_EEF` constants, arm joint group, actuator defaults. |
| `targets.py` | `NIC_CARD_ASSET` and `SC_PORT_ASSET` with their port frame paths (entrance / seat) and insertion-axis conventions. |
| `workcells.py` | `AIC_WORKCELL_ASSET`, `TASK_BOARD_ASSET` — static USDs that don't participate in the MDP. |
| `scene.py` | Scene-slot string constants, the `AIC_PORT_INSERTION_LAYOUT` (assignment of assets to slots, default poses), and `AIC_PORT_INSERTION_RANDOMIZATION` (board / part randomization spec). |

### `tasks/`

Task discovery glue + the concrete task package.

| File | Owns |
|---|---|
| `tasks/__init__.py` | Calls `isaaclab_tasks.utils.import_packages(__name__, _BLACKLIST_PKGS)` with `_BLACKLIST_PKGS = ["utils", ".mdp"]`. This is how Gym IDs get registered side-effectfully. Details in `02_gym_registration.md`. |
| `tasks/manager_based/__init__.py` | Imports `gymnasium` so the namespace is import-safe. No registration here. |

#### `tasks/manager_based/port_insertion/`

This is the only concrete task today. Inside it:

| File | Owns |
|---|---|
| `__init__.py` | The `gym.register("AIC-Port-Insertion-v0", …)` call. Provides `env_cfg_entry_point` and `rsl_rl_cfg_entry_point` kwargs. |
| `port_insertion_env_cfg.py` | `PortInsertionEnvCfg(ManagerBasedRLEnvCfg)` — thin `@configclass` that wires together the seven `build_*_cfg` outputs and sets `decimation`, `episode_length_s`, `sim.dt`. |
| `specs.py` | The selection layer: `PortInsertionAssemblySpec`, `ControllerSpec`, `InsertionGoalSpec`, `PortInsertionTerminationSpec`, `PortInsertionObservationSpec`, and the concrete `AIC_PORT_INSERTION_ASSEMBLY` that ties them together. Calls `validate()` at import time. |
| `builders.py` | Pure functions `build_scene_cfg`, `build_action_cfg`, `build_command_cfg`, `build_observation_cfg`, `build_event_cfg`, `build_termination_cfg`, `build_empty_reward_cfg`. Each takes an `AssemblySpec` and returns an IsaacLab cfg object. This is where the spec-vs-cfg boundary lives. |
| `mdp/commands.py` | `InsertionGoalCommand` term — publishes the entrance/seat pose of the selected port to the env. |
| `mdp/events.py` | Reset-mode events: `randomize_dome_light`, `randomize_board_and_parts` (board pose + board-relative part poses with optional snap). |
| `mdp/observations.py` | Root-frame pose/velocity (`body_pose_b`, `body_vel_b`) for the `policy` group; privileged goal + errors (`insertion_goal_b`, `seat_pos_err_b`, `seat_quat_delta_b`, `insertion_fraction`) for the `cheatcode` group. |
| `mdp/recorders.py` | `PreStepGroupedObservationsRecorder` + `GroupedActionStateRecorderManagerCfg` — captures every obs group (policy + cheatcode) into HDF5 instead of just the policy group. Wired by `scripts/record_demos.py`. |
| `mdp/terminations.py` | `InsertionGoalReachedSuccess`, `InsertionGoalStationaryFailure` — both consume the command term plus the EEF body pose. |
| `agents/rsl_rl_ppo_cfg.py` | PPO runner cfg consumed by `scripts/rsl_rl/`. Not used by the env at sim time. |

## Dependency direction

The import graph is one-way. The layer below never imports the layer above.

```text
                ┌────────────────────────────────────┐
                │  scripts/  (random_agent, teleop,  │
                │            record, replay, rsl_rl) │
                └────────────────┬───────────────────┘
                                 │ import aic_task   (triggers .tasks/*)
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │  tasks/manager_based/port_insertion/__init__.py  │  ← gym.register
        └────────────────┬─────────────────────────────────┘
                         │
                         ▼
        ┌──────────────────────────────────────────────────┐
        │  port_insertion_env_cfg.py  (PortInsertionEnvCfg)│
        └────────────────┬─────────────────────────────────┘
                         │ uses build_*_cfg(ASSEMBLY)
                         ▼
        ┌──────────────────────────────────────────────────┐
        │  builders.py  +  mdp/*.py  (IsaacLab cfg layer)  │
        └────────────────┬─────────────────────────────────┘
                         │ reads AIC_PORT_INSERTION_ASSEMBLY
                         ▼
        ┌──────────────────────────────────────────────────┐
        │  specs.py  (PortInsertionAssemblySpec, …)        │
        └────────────────┬─────────────────────────────────┘
                         │ composes from asset_specs
                         ▼
        ┌──────────────────────────────────────────────────┐
        │  asset_specs/  (UR5E_CABLE_ASSET, NIC_CARD_ASSET,│
        │                 AIC_PORT_INSERTION_LAYOUT, …)    │
        └────────────────┬─────────────────────────────────┘
                         │ asset_path(...)
                         ▼
        ┌──────────────────────────────────────────────────┐
        │  assets/  (USD files on disk)                    │
        └──────────────────────────────────────────────────┘
```

Practical consequences:

- `asset_specs/` cannot import from `tasks/`. If you need task-only knowledge
  in an asset spec, redesign — the asset spec is the wrong place.
- `builders.py` is the *only* file in the task package that talks to both
  `aic_task.asset_specs` **and** `isaaclab.*` cfg classes. If you find IsaacLab
  cfg construction sneaking into `specs.py` or `mdp/*.py`, it's a smell.
- The `mdp/*.py` term implementations may import IsaacLab runtime types (e.g.
  `ManagerTermBase`, `Articulation`), but they receive their parameters from
  `builders.py` — they don't read `AIC_PORT_INSERTION_ASSEMBLY` directly.

## Entry points for an AI working in this package

Use these as fixed reading anchors before doing anything else:

1. [`aic_task/__init__.py`](../aic_task/__init__.py) — to understand what
   `import aic_task` triggers.
2. [`aic_task/tasks/__init__.py`](../aic_task/tasks/__init__.py) — to see how
   tasks are auto-discovered.
3. [`tasks/manager_based/port_insertion/__init__.py`](../aic_task/tasks/manager_based/port_insertion/__init__.py)
   — to find the Gym ID and cfg entry-point string.
4. [`tasks/manager_based/port_insertion/specs.py`](../aic_task/tasks/manager_based/port_insertion/specs.py)
   — the single source of truth for what the task *selects*.
5. [`tasks/manager_based/port_insertion/builders.py`](../aic_task/tasks/manager_based/port_insertion/builders.py)
   — to see how a selection becomes an IsaacLab cfg.

If `asset_specs/` becomes relevant, read its
[README](../aic_task/asset_specs/README.md) before the source — it's already
well-documented.
