---
name: aic-mdp-term-work
description: Route substantive edits to (or additions of) MDP terms in source/aic_task/aic_task/tasks/manager_based/port_insertion/mdp/ — new termination, new event, new command, or non-trivial change to an existing term's logic. Invoke when the user mentions "termination", "command term", "event", "reset event", "success/failure condition", or names InsertionGoalCommand / InsertionGoalReachedSuccess / InsertionGoalStationaryFailure / randomize_dome_light / randomize_board_and_parts / reset_robot_to_default_joint_pose. Skip for pure threshold/number tweaks (use aic-task-edit) and for creating a new Gym ID (use aic-task-add).
---

# aic-mdp-term-work

For writing or editing a class/function inside
[source/aic_task/aic_task/tasks/manager_based/port_insertion/mdp/](../../../source/aic_task/aic_task/aic_task/tasks/manager_based/port_insertion/mdp/).
The full per-term reference is
[05_mdp_terms.md](../../../source/aic_task/docs/05_mdp_terms.md) — open
**only the section for the term you are touching**. Do not read the
whole doc.

## The contract (4 lines)

Terms receive their knobs via `params={...}` passed by `builders.py`.
Terms **never** import `AIC_PORT_INSERTION_ASSEMBLY` and never read any
constant from `specs.py`. The builder is the only place that knows about
the assembly. Adding a new knob: extend the spec dataclass, plumb it
through `build_*_cfg`, accept it as a function/method parameter on the
term.

## Term-to-section index

Open the matching entry in
[05_mdp_terms.md](../../../source/aic_task/docs/05_mdp_terms.md):

| Term you're touching | Section anchor |
|---|---|
| `InsertionGoalCommand[Cfg]` | [#insertiongoalcommand](../../../source/aic_task/docs/05_mdp_terms.md#insertiongoalcommand) |
| `reset_robot_to_default_joint_pose` (available, not wired) | [#reset_robot_to_default_joint_pose](../../../source/aic_task/docs/05_mdp_terms.md#reset_robot_to_default_joint_pose) |
| `randomize_dome_light` | [#randomize_dome_light](../../../source/aic_task/docs/05_mdp_terms.md#randomize_dome_light) |
| `randomize_board_and_parts` | [#randomize_board_and_parts](../../../source/aic_task/docs/05_mdp_terms.md#randomize_board_and_parts) |
| `InsertionGoalReachedSuccess` | [#insertiongoalreachedsuccess](../../../source/aic_task/docs/05_mdp_terms.md#insertiongoalreachedsuccess) |
| `InsertionGoalStationaryFailure` | [#insertiongoalstationaryfailure](../../../source/aic_task/docs/05_mdp_terms.md#insertiongoalstationaryfailure) |

For a brand-new term, read
[#re-export-pattern-in-mdp__init__py](../../../source/aic_task/docs/05_mdp_terms.md#re-export-pattern-in-mdp__init__py)
first — it covers where the file goes and how `mdp/__init__.py` should
treat it.

## Re-export invariants

[mdp/\_\_init\_\_.py](../../../source/aic_task/aic_task/tasks/manager_based/port_insertion/mdp/__init__.py)
already does two things and nothing else:

```python
from isaaclab.envs.mdp import *      # noqa: F401, F403
from .commands import *              # noqa: F401, F403
```

- `from isaaclab.envs.mdp import *` already covers IsaacLab's core terms
  (`generated_commands`, `joint_pos_rel`, `time_out`,
  `reset_joints_by_offset`, …). Don't duplicate those.
- Events and terminations are **not** re-exported via
  `mdp/__init__.py`. They are imported by `builders.py` directly with
  fully-qualified names (`from .mdp.events import randomize_dome_light`).
  When adding a new termination or event, leave it out of
  `mdp/__init__.py` unless you have a specific reason to expose it.

## Walker invariant

The `mdp/` folder is blacklisted from task auto-discovery — the entry
`".mdp"` in `_BLACKLIST_PKGS` at
[tasks/\_\_init\_\_.py:15](../../../source/aic_task/aic_task/tasks/__init__.py#L15)
makes the walker skip any module whose dotted name contains `.mdp`.
**You cannot accidentally register a new Gym ID from a file under
`mdp/`.** Equally, a stray `gym.register(...)` inside `mdp/*.py` will be
silently ignored — that's a clear smell. Reference:
[02 §"The blacklist used here"](../../../source/aic_task/docs/02_gym_registration.md#the-blacklist-used-here).

## After editing

If you added a new parameter, the path is:

1. Add the field to the matching dataclass in
   [specs.py](../../../source/aic_task/aic_task/tasks/manager_based/port_insertion/specs.py)
   (e.g. `PortInsertionTerminationSpec` for a new success condition).
2. Wire it into `params={...}` in
   [builders.py](../../../source/aic_task/aic_task/tasks/manager_based/port_insertion/builders.py)
   inside the matching `build_*_cfg`.
3. Accept it as a parameter in the term's call signature; default it
   conservatively.

The next `import aic_task` runs `AIC_PORT_INSERTION_ASSEMBLY.validate()`
at module scope — that catches cross-spec mismatches without launching
sim.

## What this skill does NOT route to

- [01_package_structure.md](../../../source/aic_task/docs/01_package_structure.md),
  [02_gym_registration.md](../../../source/aic_task/docs/02_gym_registration.md)
  (other than the blacklist anchor above),
  [03_port_insertion_overview.md](../../../source/aic_task/docs/03_port_insertion_overview.md),
  [04_assembly_pattern.md](../../../source/aic_task/docs/04_assembly_pattern.md):
  background. Open only if the term's section anchor sends you there.
- [06_diff_ik_contract.md](../../../source/aic_task/docs/06_diff_ik_contract.md):
  about emitting actions, not about MDP terms. Wrong skill unless you
  are also editing a script-side controller.
- [aic-task-edit](../aic-task-edit/SKILL.md): for tweaking *parameter
  values* (thresholds, ranges, seconds). This skill is only for *logic*
  changes. If the user wants `0.003 m → 0.002 m`, use the other skill.

## Source-of-truth SHAs

- [05_mdp_terms.md](../../../source/aic_task/docs/05_mdp_terms.md):
  `8d9a44e`
- [02_gym_registration.md](../../../source/aic_task/docs/02_gym_registration.md):
  `bb6a606`

If the docs have moved, the section-anchor line numbers may have
drifted. Run [docs-task-sync](../docs-task-sync/SKILL.md) first if
anything looks off.
