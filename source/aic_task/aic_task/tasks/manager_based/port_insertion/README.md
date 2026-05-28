---
scope: short index of the AIC-Port-Insertion-v0 task package; points at the deep-dive doc and the files inside this folder
audience: AI agents working in this repo
last_verified_commit: bb6a606
related:
  - ../../../../docs/03_port_insertion_overview.md
  - ../../../../docs/02_gym_registration.md
  - ../../../../docs/01_package_structure.md
---

# `port_insertion/` — `AIC-Port-Insertion-v0`

This folder is the entire implementation of the `AIC-Port-Insertion-v0` Gym
environment. Importing `aic_task` triggers `__init__.py` here via the task
walker (see [02_gym_registration.md](../../../../docs/02_gym_registration.md)).

For a complete behavioral spec of the env (scene, action, command,
observations, events, terminations, episode contract), read
[03_port_insertion_overview.md](../../../../docs/03_port_insertion_overview.md).
This README is just a directory index.

## Files

| File | Role |
|---|---|
| [`__init__.py`](__init__.py) | `gym.register("AIC-Port-Insertion-v0", …)`; provides `env_cfg_entry_point` and `rsl_rl_cfg_entry_point`. |
| [`port_insertion_env_cfg.py`](port_insertion_env_cfg.py) | `PortInsertionEnvCfg(ManagerBasedRLEnvCfg)` — wires the seven `build_*_cfg(ASSEMBLY)` outputs and sets `sim.dt = 1/120`, `decimation = 4`, `episode_length_s = 120.0`. |
| [`specs.py`](specs.py) | **Selection layer.** Defines `ControllerSpec`, `InsertionGoalSpec`, `PortInsertionTerminationSpec`, and the concrete `AIC_PORT_INSERTION_ASSEMBLY` that picks the UR5e + NIC card + AIC layout + DiffIK controller + success/stationary thresholds. `validate()` runs at import time. |
| [`builders.py`](builders.py) | **Spec → IsaacLab cfg layer.** Pure functions `build_scene_cfg`, `build_action_cfg`, `build_command_cfg`, `build_observation_cfg`, `build_event_cfg`, `build_termination_cfg`, `build_empty_reward_cfg`. The only file in this folder that imports both `aic_task.asset_specs` and `isaaclab.*` cfg classes. |
| [`mdp/`](mdp/) | Task-specific MDP terms (see table below). |
| [`agents/rsl_rl_ppo_cfg.py`](agents/rsl_rl_ppo_cfg.py) | `PPORunnerCfg(RslRlOnPolicyRunnerCfg)` — training-side cfg consumed by `scripts/rsl_rl/`. Not used at sim time. |

## `mdp/` submodules

| File | Owns |
|---|---|
| [`mdp/__init__.py`](mdp/__init__.py) | Re-exports `isaaclab.envs.mdp` plus the local `commands` symbols. |
| [`mdp/commands.py`](mdp/commands.py) | `InsertionGoalCommand` + `InsertionGoalCommandCfg`. Publishes a 14-D tensor `[entrance_pos_w, entrance_quat_w, seat_pos_w, seat_quat_w]` plus named tensor properties on the term itself. |
| [`mdp/events.py`](mdp/events.py) | Reset-mode events: `randomize_dome_light`, `randomize_board_and_parts` (carries board-relative parts under board jitter, with optional grid snap and USD-xform sync). |
| [`mdp/terminations.py`](mdp/terminations.py) | `InsertionGoalReachedSuccess`, `InsertionGoalStationaryFailure` — both stateful, both consume the `insertion_goal` command and the `sfp_tip_link` body. |

## What lives where (cheat sheet)

| If you want to change… | Edit |
|---|---|
| Success / failure thresholds | `PortInsertionTerminationSpec` in `specs.py` |
| DiffIK scale, IK method, controlled body | `UR5E_DIFF_IK_CONTROLLER` in `specs.py` |
| The selected port or its EEF-in-port pose | `NIC_PORT_0_INSERTION_GOAL` in `specs.py` |
| Robot / target asset choice or layout | swap the constants imported at the top of `specs.py` (`UR5E_CABLE_ASSET`, `NIC_CARD_ASSET`, `AIC_PORT_INSERTION_LAYOUT`) — change `AIC_PORT_INSERTION_ASSEMBLY` accordingly |
| Observation group composition | `build_observation_cfg` in `builders.py` |
| Reset event roster or randomization ranges | `build_event_cfg` in `builders.py` (ranges live on the layout in `asset_specs/scene.py`) |
| `sim.dt`, `decimation`, `episode_length_s` | `PortInsertionEnvCfg.__post_init__` in `port_insertion_env_cfg.py` |
| What `insertion_goal` publishes / how the term computes it | `mdp/commands.py` |
| When the env reports success / failure | `mdp/terminations.py` |
| Training (PPO) hyperparameters | `agents/rsl_rl_ppo_cfg.py` |

## Read these before editing

1. [03_port_insertion_overview.md](../../../../docs/03_port_insertion_overview.md) — the full behavioral spec.
2. [`../../../asset_specs/README.md`](../../../asset_specs/README.md) — slot/role/body conventions you'll need to keep consistent.
3. [06_diff_ik_contract.md](../../../../docs/06_diff_ik_contract.md) — the DiffIK action contract; required reading if you touch `build_action_cfg` or anything that emits actions.
