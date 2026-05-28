---
scope: aic_task Python package — what it is, how it's imported, where to look next
audience: AI agents working in this repo
last_verified_commit: bb6a606
related:
  - ../docs/01_package_structure.md
  - asset_specs/README.md
---

# aic_task

IsaacLab extension/package that defines the AIC robotics tasks. Today it ships
a single Gym-registered task, `AIC-Port-Insertion-v0`, plus the reusable asset
contracts, scene layout, and config builders behind it.

The package has two responsibilities, and nothing else:

1. **Describe assets and scenes** as plain-Python contracts (`asset_specs/`).
2. **Assemble those contracts into IsaacLab env cfgs and register Gym IDs**
   (`tasks/`).

Anything that uses a task at runtime (demo agents, teleop, recording, training)
lives in the top-level [`scripts/`](../../../scripts/) folder, not here.

## Install / import

The package is installed in editable mode via the workspace `pyproject.toml`
and registered as the `aic_task` Isaac Sim extension via
[`../config/extension.toml`](../config/extension.toml). At runtime:

```python
import aic_task             # triggers task registration as a side-effect
import gymnasium as gym
env = gym.make("AIC-Port-Insertion-v0")
```

The side-effect chain is:

```text
aic_task/__init__.py
  └── from .tasks import *
        └── tasks/__init__.py  (calls isaaclab_tasks.utils.import_packages)
              └── tasks/manager_based/port_insertion/__init__.py  (gym.register)
```

Details and the discovery blacklist are in
[`../docs/02_gym_registration.md`](../docs/02_gym_registration.md) *(written
in a later phase).*

## Layout at a glance

```text
aic_task/
├── __init__.py          Triggers task registration; loads UI extension if inside Isaac Sim
├── extension.py         Isaac Sim UI extension (debug buttons); irrelevant for headless runs
├── assets/              USD files (robots, targets, workcells) — pure data
├── asset_specs/         Python contracts describing assets, slots, layouts
└── tasks/
    └── manager_based/
        └── port_insertion/   Single ManagerBasedRLEnv: AIC-Port-Insertion-v0
```

Full directory map, per-module role, and dependency direction:
[`../docs/01_package_structure.md`](../docs/01_package_structure.md).

## Where to start when working in this package

| Goal | Start here |
|---|---|
| Understand what the one task does | [`../docs/03_port_insertion_overview.md`](../docs/03_port_insertion_overview.md) *(later phase)* |
| Add a new task | [`../docs/08_adding_a_new_task.md`](../docs/08_adding_a_new_task.md) *(later phase)* |
| Modify the existing task | [`../docs/09_modifying_existing_task.md`](../docs/09_modifying_existing_task.md) *(later phase)* |
| Change an asset's body roles, joints, or USD path | [`asset_specs/README.md`](asset_specs/README.md) |
| Wire a new controller / action term | [`tasks/manager_based/port_insertion/builders.py`](tasks/manager_based/port_insertion/builders.py) — see [`../docs/06_diff_ik_contract.md`](../docs/06_diff_ik_contract.md) for the DiffIK contract |

## What this package deliberately does **not** contain

- No demo / teleop / recording / training scripts — those live in
  [`scripts/`](../../../scripts/) at the repo root.
- No live camera streaming, oracles, or computer-vision helpers — they were
  removed; scripts that need them implement them locally.
- No reward functions yet — `AIC-Port-Insertion-v0` runs with an empty reward
  manager (see `build_empty_reward_cfg` in
  [`tasks/manager_based/port_insertion/builders.py`](tasks/manager_based/port_insertion/builders.py)).
