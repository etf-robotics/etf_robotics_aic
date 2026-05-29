---
scope: file-by-file recipe for creating a new Gym-registered task subpackage under aic_task/tasks/manager_based/, in dependency order
audience: AI agents working in this repo
last_verified_commit: cfb23ef
related:
  - 01_package_structure.md
  - 02_gym_registration.md
  - 04_assembly_pattern.md
  - 05_mdp_terms.md
---

# 08 · Adding a New Task

This doc is the concrete checklist for adding a second Gym-registered task
(say `AIC-Foo-v0`) to this package. It is not about *running* the task —
that belongs in the future `scripts/` doc set. It is also not about
*modifying* the existing port-insertion task — for that, go to
[09_modifying_existing_task.md](09_modifying_existing_task.md).

Before you start, read [04_assembly_pattern.md](04_assembly_pattern.md). The
recipe below is just that pattern, instantiated. Skim
[port_insertion/](../aic_task/tasks/manager_based/port_insertion/) as a
reference *shape* — do not clone its behavioral choices into your task
without first asking whether they fit.

## What you are creating

A new sibling of `port_insertion/` under
[tasks/manager_based/](../aic_task/tasks/manager_based/):

```text
aic_task/tasks/manager_based/<new_task>/
├── __init__.py              gym.register("AIC-<New>-v0", …)
├── <new_task>_env_cfg.py    @configclass env cfg, wires builders
├── specs.py                 Assembly selection (no IsaacLab imports)
├── builders.py              spec → IsaacLab cfg objects
├── mdp/                     (only if you need task-specific MDP terms)
│   ├── __init__.py          re-export pattern
│   ├── commands.py          (optional)
│   ├── events.py            (optional)
│   └── terminations.py      (optional)
└── agents/                  (only if you want training-side cfgs)
    ├── __init__.py
    └── rsl_rl_ppo_cfg.py
```

You do not edit any file outside this folder. In particular:

- **Do not** edit [tasks/__init__.py](../aic_task/tasks/__init__.py). The
  `import_packages` walker finds your subpackage automatically. See
  [02_gym_registration.md](02_gym_registration.md).
- **Do not** edit [aic_task/__init__.py](../aic_task/__init__.py). The
  `from .tasks import *` line above already triggers your registration via
  the walker.
- The `_BLACKLIST_PKGS = ["utils", ".mdp"]` entries in
  [tasks/__init__.py](../aic_task/tasks/__init__.py) **already cover your
  new `mdp/` submodule**. Anything you put under `<new_task>/mdp/` will be
  silently skipped by the walker; it gets imported on demand by
  `builders.py`. You do not register an MDP term as a Gym ID.

A trap to avoid: if you name an internal helper module `utils` (anywhere in
the dotted path), the walker will skip it too — the blacklist is a
substring match, not a path match. Common workaround: put helpers in
`builders.py` or a `_helpers.py`. See
[02_gym_registration.md](02_gym_registration.md).

## Build order

Build bottom-up so each file's imports already exist when you write it.

### 1. (If needed) Add or extend asset specs

If the task uses assets the existing
[asset_specs/](../aic_task/asset_specs/) already describes
(`UR5E_CABLE_ASSET`, `NIC_CARD_ASSET`, `SC_PORT_ASSET`, `TASK_BOARD_ASSET`,
`AIC_WORKCELL_ASSET`), skip this step.

If you need a new robot / target / workcell:

- Add it under [asset_specs/](../aic_task/asset_specs/) in the file that
  matches its role: `robots.py`, `targets.py`, `workcells.py`.
- Put its USD under [assets/](../aic_task/assets/) at a path consistent
  with `asset_path("category", "subdir", "file.usd")`.
- For target USDs with insertion frames, define a `TargetPortSpec` (see
  [NIC_SFP_PORT_0](../aic_task/asset_specs/targets.py#L45)) so your task's
  goal can reference the port by name.
- Re-export the new constant from
  [asset_specs/\_\_init\_\_.py](../aic_task/asset_specs/__init__.py).

Asset-spec rules from
[04_assembly_pattern.md](04_assembly_pattern.md): the file must not
`import isaaclab.*` or `omni.*`, and asset facts (USD path, body name,
joint group, root prim) live here and nowhere else.

### 2. (Optional) Add a new scene layout

If your task arranges assets differently from
[AIC_PORT_INSERTION_LAYOUT](../aic_task/asset_specs/scene.py#L175) (a new
board, new auxiliary slots, different randomization), add a new
`SceneLayoutSpec` next to it in
[asset_specs/scene.py](../aic_task/asset_specs/scene.py). Define a new
`SCENE_SLOT_<NAME>` string constant for each new stable scene key — those
strings become `env.scene[<key>]` and must not change after they ship.

Re-use existing layouts when you can; a layout is heavyweight and worth
sharing across tasks.

### 3. `<new_task>/specs.py` — the selection layer

This is where the task says what it picks, in plain frozen dataclasses.
Use [port_insertion/specs.py](../aic_task/tasks/manager_based/port_insertion/specs.py)
as the shape, not the content. The minimum a task spec needs:

- A `ControllerSpec`-shaped record describing the action. If your action
  is also relative-mode pose DiffIK on a robot's TCP body, you can reuse
  the existing `ControllerSpec`; otherwise define a new dataclass for your
  controller (and let `builders.py` translate it). The asset-side facts
  (joint group name, controlled body role) come from the robot spec.
- A goal / target dataclass that names the scene slot, the port or
  target frame, and any per-task knobs (e.g. EEF-in-port pose).
- A termination dataclass holding success / failure thresholds as plain
  numbers — no IsaacLab cfg types here.
- An assembly dataclass that bundles the robot, target, layout, controller,
  goal, and termination, with a `validate()` method that fails fast on
  cross-spec mismatches (controller's `robot_slot` vs. layout's robot
  slot, goal's `target_slot` vs. layout's target slot, port exists on
  target). See
  [PortInsertionAssemblySpec.validate](../aic_task/tasks/manager_based/port_insertion/specs.py#L91)
  for the pattern.
- A concrete `<NEW>_ASSEMBLY = ...AssemblySpec(...)` constant at the
  bottom of the module, followed by `<NEW>_ASSEMBLY.validate()` so the
  mismatch fires at *import* time, not at sim start.

Forbidden imports in this file: `isaaclab.*`, `omni.*`, anything from
`mdp/` or `builders.py`. Allowed: `aic_task.asset_specs.*`, stdlib.

### 4. `<new_task>/mdp/` — only if you need task-specific terms

Skip this folder entirely if your task is built from terms IsaacLab already
ships. `builders.py` can pull `joint_pos_rel`, `last_action`,
`generated_commands`, `time_out`, `reset_joints_by_offset`, etc. directly
from `isaaclab.envs.mdp` (see how the existing
[builders.py](../aic_task/tasks/manager_based/port_insertion/builders.py#L9)
imports them).

If you do need new terms (a new command, a new event mode, a new
termination), put each in its own file:

- `mdp/commands.py` for `CommandTerm` subclasses + their `CommandTermCfg`
  classes. Reference:
  [InsertionGoalCommand](../aic_task/tasks/manager_based/port_insertion/mdp/commands.py#L19).
- `mdp/events.py` for `mode="reset"` (or other-mode) event functions.
  Reference:
  [randomize_dome_light](../aic_task/tasks/manager_based/port_insertion/mdp/events.py#L50).
- `mdp/terminations.py` for `ManagerTermBase` subclasses with stateful
  per-env counters. Reference:
  [InsertionGoalReachedSuccess](../aic_task/tasks/manager_based/port_insertion/mdp/terminations.py#L23).

Then write `mdp/__init__.py`. It does exactly two things and nothing
else (see
[port_insertion/mdp/\_\_init\_\_.py](../aic_task/tasks/manager_based/port_insertion/mdp/__init__.py)):

```python
from isaaclab.envs.mdp import *      # noqa: F401, F403
from .commands import *              # noqa: F401, F403  (only if you defined commands)
```

Note: `events` and `terminations` are **not** re-exported from
`mdp/__init__.py`. `builders.py` imports them by their fully-qualified
names. Stay consistent — leave them out of the re-export unless you have
a specific reason. Detail in
[05_mdp_terms.md](05_mdp_terms.md#re-export-pattern-in-mdp__init__py).

The contract for the term implementations themselves (from
[04_assembly_pattern.md](04_assembly_pattern.md)): the term must not
import `<NEW>_ASSEMBLY` or read any module-level constant from `specs.py`.
It receives its knobs via `params={...}` passed by `builders.py`.

### 5. `<new_task>/builders.py` — spec → cfg

This is the only file in your task that imports both
`aic_task.asset_specs.*` and `isaaclab.*` cfg classes. Write one pure
function per env-cfg field, each taking your assembly and returning the
corresponding IsaacLab cfg object. Pattern (see
[port_insertion/builders.py](../aic_task/tasks/manager_based/port_insertion/builders.py)):

```python
def build_scene_cfg(assembly: <New>AssemblySpec, *, num_envs: int = 1, ...) -> InteractiveSceneCfg: ...
def build_action_cfg(assembly: <New>AssemblySpec) -> ActionsCfg: ...
def build_command_cfg(assembly: <New>AssemblySpec) -> CommandsCfg: ...
def build_observation_cfg(assembly: <New>AssemblySpec) -> dict[str, ObsGroup]: ...
def build_event_cfg(assembly: <New>AssemblySpec) -> dict[str, EventTerm]: ...
def build_termination_cfg(assembly: <New>AssemblySpec) -> dict[str, DoneTerm]: ...
def build_empty_reward_cfg() -> dict: ...
```

Rules:

- Call `assembly.validate()` at the top of each `build_*_cfg` so a stale
  assembly fails fast even if `specs.py` skipped its top-level validate
  call.
- Pass term parameters via `params={...}` on `EventTerm` / `DoneTerm` /
  `ObsTerm` / `CommandTermCfg`. Term implementations never reach for the
  assembly directly.
- Do not import `<NEW>_ASSEMBLY` here. The assembly comes in as the
  `assembly` parameter; the constant is the env cfg's choice, not the
  builder's.

### 6. `<new_task>/<new_task>_env_cfg.py` — the env cfg

A thin `@configclass` that wires the seven `build_*_cfg` outputs into a
`ManagerBasedRLEnvCfg` subclass and sets `decimation`, `sim.dt`,
`episode_length_s`. Pattern:

```python
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .builders import (
    build_action_cfg,
    build_command_cfg,
    build_empty_reward_cfg,
    build_event_cfg,
    build_observation_cfg,
    build_scene_cfg,
    build_termination_cfg,
)
from .specs import <NEW>_ASSEMBLY


ASSEMBLY = <NEW>_ASSEMBLY


@configclass
class <New>EnvCfg(ManagerBasedRLEnvCfg):
    scene = build_scene_cfg(ASSEMBLY)
    actions = build_action_cfg(ASSEMBLY)
    commands = build_command_cfg(ASSEMBLY)
    observations = build_observation_cfg(ASSEMBLY)
    events = build_event_cfg(ASSEMBLY)
    rewards = build_empty_reward_cfg()
    terminations = build_termination_cfg(ASSEMBLY)

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 120.0
        self.sim.dt = 1.0 / 120.0
```

If your task wants different physics / control rates, override them here.
This file should not construct IsaacLab cfg objects directly; if you find
yourself doing that, push it down into a new `build_*_cfg` in
`builders.py`.

### 7. `<new_task>/agents/` — only if you want training-side cfgs

`agents/rsl_rl_ppo_cfg.py` is consumed by `scripts/rsl_rl/`, not by the
env at sim time. Mirror
[agents/rsl_rl_ppo_cfg.py](../aic_task/tasks/manager_based/port_insertion/agents/rsl_rl_ppo_cfg.py)
if you want PPO; skip the folder entirely if not. Remember to add an
`agents/__init__.py` so the `agents` namespace is importable from
`__init__.py` (the registration line below).

### 8. `<new_task>/__init__.py` — registration

Last, because nothing else should run during the walker's import of this
file *except* the registration. Pattern (matching
[port_insertion/\_\_init\_\_.py](../aic_task/tasks/manager_based/port_insertion/__init__.py)):

```python
import gymnasium as gym

from . import agents  # only if you created the agents subpackage


gym.register(
    id="AIC-<New>-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.<new_task>_env_cfg:<New>EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:<New>PPORunnerCfg",
    },
)
```

`f"{__name__}.<new_task>_env_cfg:<New>EnvCfg"` resolves to the dotted path
of the cfg class; consumers turn it into an instance via
`isaaclab_tasks.utils.parse_env_cfg`. Field semantics live in
[02_gym_registration.md](02_gym_registration.md).

If you skipped the agents subpackage, drop the `import` line and remove
the `rsl_rl_cfg_entry_point` kwarg.

## Verifying registration without launching a sim

You don't need a stage to confirm the walker picked up your task. From the
repo root:

```bash
python -c "import aic_task; import gymnasium; print('AIC-<New>-v0' in gymnasium.registry)"
```

If this prints `False`, the most common causes are:

1. A typo or `ImportError` in your `__init__.py`. The walker swallows the
   import error silently; `python -c "import aic_task.tasks.manager_based.<new_task>"`
   surfaces the real exception.
2. Your subpackage name contains `utils` or `.mdp` somewhere in its dotted
   path — the walker's blacklist substring-matches and silently skips it.
3. Your `specs.py` raises in `.validate()` at import time. That *will* show
   up as a real ImportError; fix the spec mismatch.

The behavioral contract for the registered env (sim rate, observation
layout, etc.) does not need to match the port-insertion task. The
behavioral contract for the *registration* does — Gym ID style, env class,
`disable_env_checker=True`, and the `env_cfg_entry_point` kwarg are all
load-bearing for `parse_env_cfg`.

## What this doc does not cover

- How to actually `gym.make` and step your task. That's a `scripts/`-side
  concern.
- How to write reward shaping. The pattern's empty
  `build_empty_reward_cfg()` returns `{}`; replacing it with a real reward
  manager is an open design choice, not part of the recipe here.
- How to wire training. `agents/rsl_rl_ppo_cfg.py` is a starting point;
  hyperparameter selection is downstream.
