---
name: aic-task-add
description: Route requests to add a second Gym-registered task subpackage under source/aic_task/aic_task/tasks/manager_based/ — a new ManagerBasedRLEnvCfg, a new Gym ID, alongside the existing AIC-Port-Insertion-v0. Invoke when the user says "add a new task", "register a new Gym ID", "new ManagerBasedRLEnv", or names a Gym ID other than AIC-Port-Insertion-v0. Skip for tweaking the existing task (use aic-task-edit) or editing an mdp/*.py term (use aic-mdp-term-work).
---

# aic-task-add

For creating a **new** task subpackage, not for tweaking the existing
port-insertion one. The full recipe — every file, in dependency order —
is
[08_adding_a_new_task.md](../../../source/aic_task/docs/08_adding_a_new_task.md).
Open it once; follow its build order section by section. Pair it with
[04_assembly_pattern.md](../../../source/aic_task/docs/04_assembly_pattern.md)
for the layering rules so you know which file owns which responsibility.

## Read order

1. [08_adding_a_new_task.md](../../../source/aic_task/docs/08_adding_a_new_task.md)
   — recipe and file list.
2. [04_assembly_pattern.md](../../../source/aic_task/docs/04_assembly_pattern.md)
   — the four-layer rule and the forbidden-imports table.
3. The existing
   [port_insertion/](../../../source/aic_task/aic_task/tasks/manager_based/port_insertion/)
   subpackage **as a reference shape**, not as a behavioral spec to
   clone. Don't import its constants into your new task; don't copy its
   thresholds verbatim "because they were there". Pick its file layout
   and the assembly pattern; pick your own numbers.

## Walker invariant

Drop the new package under
[source/aic_task/aic_task/tasks/manager_based/](../../../source/aic_task/aic_task/tasks/manager_based/)
`<new_name>/`, put `gym.register(...)` in its `__init__.py`, and **do
not touch
[tasks/\_\_init\_\_.py](../../../source/aic_task/aic_task/tasks/__init__.py)**
— the `import_packages` walker finds your subpackage automatically.

The `_BLACKLIST_PKGS = ["utils", ".mdp"]` entries already cover your new
`mdp/` submodule (substring match on dotted names). One trap: naming an
internal helper module `utils` anywhere in the dotted path gets it
**silently skipped** by the walker. If you need helpers, put them in
`builders.py` or a `_helpers.py`. Reference:
[02 §"Practical implications for adding a task"](../../../source/aic_task/docs/02_gym_registration.md#practical-implications-for-adding-a-task).

## Files you'll create

The four mandatory ones, plus two optional folders. Order matches the
build order in
[08 §"Build order"](../../../source/aic_task/docs/08_adding_a_new_task.md#build-order):

| File / folder | Mandatory? |
|---|---|
| `<new_task>/specs.py` | yes |
| `<new_task>/builders.py` | yes |
| `<new_task>/<new_task>_env_cfg.py` | yes |
| `<new_task>/__init__.py` (with `gym.register(...)`) | yes |
| `<new_task>/mdp/` (only if you need task-specific MDP terms) | no |
| `<new_task>/agents/` (only if you want a training-side cfg) | no |

Skip `mdp/` if your task is built entirely from terms IsaacLab already
ships — `joint_pos_rel`, `last_action`, `generated_commands`,
`time_out`, `reset_joints_by_offset`. `builders.py` can pull them
directly from `isaaclab.envs.mdp` (see the existing
[builders.py](../../../source/aic_task/aic_task/tasks/manager_based/port_insertion/builders.py#L9)).

Skip `agents/` if the task does not (yet) have a training-side
hyperparameter cfg. The registration kwarg `rsl_rl_cfg_entry_point` is
not load-bearing for the env at sim time — drop the kwarg and the
import.

## Verify registration without launching sim

```bash
python -c "import aic_task; import gymnasium; print('AIC-<New>-v0' in gymnasium.registry)"
```

If `False`, the three usual suspects, per
[08 §"Verifying registration without launching a sim"](../../../source/aic_task/docs/08_adding_a_new_task.md#verifying-registration-without-launching-a-sim):

1. `ImportError` in your `__init__.py` swallowed by the walker. Surface
   it with
   `python -c "import aic_task.tasks.manager_based.<new_name>"`.
2. Your subpackage name contains `utils` or `.mdp` somewhere in its
   dotted path — the blacklist substring-matches.
3. Your `specs.py` raises in `.validate()` at import time. That shows up
   as a real ImportError; fix the spec mismatch.

## What this skill does NOT route to

- [03_port_insertion_overview.md](../../../source/aic_task/docs/03_port_insertion_overview.md)
  — describes the *existing* task's behavioral choices. Reading it for
  the new task will lead you to clone choices that may not fit.
- [09_modifying_existing_task.md](../../../source/aic_task/docs/09_modifying_existing_task.md)
  — that's the wrong skill; switch to
  [aic-task-edit](../aic-task-edit/SKILL.md) for any tweak to the
  existing task.
- [10_glossary.md](../../../source/aic_task/docs/10_glossary.md) —
  vocabulary index, open if a term in
  [08_adding_a_new_task.md](../../../source/aic_task/docs/08_adding_a_new_task.md)
  is unfamiliar; otherwise skip.

## Source-of-truth SHAs

- [08_adding_a_new_task.md](../../../source/aic_task/docs/08_adding_a_new_task.md):
  `cfb23ef`
- [04_assembly_pattern.md](../../../source/aic_task/docs/04_assembly_pattern.md):
  `8d9a44e`
- [02_gym_registration.md](../../../source/aic_task/docs/02_gym_registration.md):
  `bb6a606`

If those have drifted, the file-list and dependency-order sections of 08
may have moved. Run [docs-task-sync](../docs-task-sync/SKILL.md) first
if something doesn't match what you see in the source tree.
