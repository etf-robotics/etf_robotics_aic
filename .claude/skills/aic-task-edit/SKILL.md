---
name: aic-task-edit
description: Route an edit of the existing AIC-Port-Insertion-v0 task (any knob tweak — thresholds, randomization, DiffIK scale, controlled body, swapped port, new observation term, sim rate) to the single file that owns the change, using the layered cheat sheet. Invoke for any "fix / tune / tweak the existing port-insertion task" request. Skip for: creating a new Gym ID (use aic-task-add), substantive edits to an mdp/*.py term's logic (use aic-mdp-term-work), or anything outside the task directory.
---

# aic-task-edit

The package's central rule: **specs.py selects, builders.py translates,
mdp/*.py implements — touch one layer per change.** Almost every
"tighten this", "swap that", "tweak the other" request lands in exactly
one file. This skill points you at that file.

The routing is the cheat sheet from
[port_insertion/README.md](../../../source/aic_task/aic_task/tasks/manager_based/port_insertion/README.md#what-lives-where-cheat-sheet).
The README is the canonical copy. The table is reproduced below so you
don't have to open it.

## Read order

1. The matching row in the table below.
2. The named file. Edit only that field / function.
3. **Stop.** Do not also open `04_assembly_pattern.md`,
   `03_port_insertion_overview.md`, or `01_package_structure.md` unless
   the row directly sends you there. The whole point of this skill is
   to spare you that reading.

## Cheat sheet

Anchors are relative to the repo root.

| If the user wants to change… | Edit |
|---|---|
| Success / failure thresholds | `PortInsertionTerminationSpec` in [source/aic_task/aic_task/tasks/manager_based/port_insertion/specs.py:138](../../../source/aic_task/aic_task/tasks/manager_based/port_insertion/specs.py#L138) |
| DiffIK scale, IK method, controlled body | `UR5E_DIFF_IK_CONTROLLER` in [source/aic_task/aic_task/tasks/manager_based/port_insertion/specs.py:112](../../../source/aic_task/aic_task/tasks/manager_based/port_insertion/specs.py#L112) |
| The selected port or its EEF-in-port pose | `NIC_PORT_0_INSERTION_GOAL` in [source/aic_task/aic_task/tasks/manager_based/port_insertion/specs.py:126](../../../source/aic_task/aic_task/tasks/manager_based/port_insertion/specs.py#L126) |
| Robot / target asset choice or layout | Imports at the top of [source/aic_task/aic_task/tasks/manager_based/port_insertion/specs.py:14](../../../source/aic_task/aic_task/tasks/manager_based/port_insertion/specs.py#L14) |
| Body name, joint group, USD path of an asset | The matching file under [source/aic_task/aic_task/asset_specs/](../../../source/aic_task/aic_task/asset_specs/) |
| Observation group composition | `build_observation_cfg` in [source/aic_task/aic_task/tasks/manager_based/port_insertion/builders.py:159](../../../source/aic_task/aic_task/tasks/manager_based/port_insertion/builders.py#L159) |
| Reset event roster | `build_event_cfg` in [source/aic_task/aic_task/tasks/manager_based/port_insertion/builders.py:188](../../../source/aic_task/aic_task/tasks/manager_based/port_insertion/builders.py#L188) |
| Randomization ranges (board / parts) | [source/aic_task/aic_task/asset_specs/scene.py:148](../../../source/aic_task/aic_task/asset_specs/scene.py#L148) (`AIC_PORT_INSERTION_RANDOMIZATION`) |
| `sim.dt`, `decimation`, `episode_length_s` | `PortInsertionEnvCfg.__post_init__` in [source/aic_task/aic_task/tasks/manager_based/port_insertion/port_insertion_env_cfg.py:38](../../../source/aic_task/aic_task/tasks/manager_based/port_insertion/port_insertion_env_cfg.py#L38) |

If the user's request matches a row above, you are done after the edit
plus a sanity read of the file. The next two cases route elsewhere:

- **Logic change inside a term** (what `insertion_goal` publishes, when
  `success`/`failed_stationary` fires, a new event function): switch to
  [aic-mdp-term-work](../aic-mdp-term-work/SKILL.md). Threshold/number
  tweaks stay in `specs.py` per the rows above.
- **A new Gym ID** (not a tweak of the existing one): switch to
  [aic-task-add](../aic-task-add/SKILL.md).

If the request doesn't match any row above and isn't one of those two
cases, you are doing something the assembly pattern wasn't built for —
ask the user before adding an abstraction.

## Invariants (true today; if they break, you've made the wrong edit)

- **The assembly's `.validate()` at
  [specs.py:156](../../../source/aic_task/aic_task/tasks/manager_based/port_insertion/specs.py#L156)
  re-runs at module import time and catches most cross-spec
  mismatches.** Do not pre-validate by hand; trust the call. If the
  next `python -c "import aic_task"` raises, fix the spec mismatch the
  exception names.
- **`builders.py` never imports `AIC_PORT_INSERTION_ASSEMBLY` by name.**
  It takes `assembly` as a parameter. If your edit reaches for the
  constant inside a builder, that's the smell from
  [04 §"What an assembly violation looks like"](../../../source/aic_task/docs/04_assembly_pattern.md#what-an-assembly-violation-looks-like).
- **`mdp/*.py` terms read their knobs from `params={...}`, never from the
  spec.** A new termination/event/command knob is added to the spec
  dataclass, then plumbed through `builders.py`. A `from ..specs import
  ...` line inside an `mdp/*.py` is a smell.
- **Asset facts (USD path, body name, joint group, root prim) live in
  exactly one place** — the matching `asset_specs/*.py`. A second copy
  inside `specs.py` or `builders.py` is the smell.

## What this skill does NOT route to

- [01_package_structure.md](../../../source/aic_task/docs/01_package_structure.md),
  [02_gym_registration.md](../../../source/aic_task/docs/02_gym_registration.md),
  [03_port_insertion_overview.md](../../../source/aic_task/docs/03_port_insertion_overview.md):
  background reading. Open only if a row above explicitly sends you
  there.
- [04_assembly_pattern.md](../../../source/aic_task/docs/04_assembly_pattern.md):
  design rationale. Open *only* when the user disputes which file the
  edit belongs in.
- [08_adding_a_new_task.md](../../../source/aic_task/docs/08_adding_a_new_task.md):
  recipe for a new Gym ID. Wrong skill — use
  [aic-task-add](../aic-task-add/SKILL.md).

## Worked walkthroughs

For each row above,
[09_modifying_existing_task.md](../../../source/aic_task/docs/09_modifying_existing_task.md)
has a worked diff (5–10 lines). If the user asks for a change the table
covers and you want to confirm the *shape* of the diff before writing
it, read only the matching entry there — do not skim the whole doc.

## Source-of-truth SHAs

Pointers above were verified against these doc commits. If the docs have
moved since, the line numbers may have drifted; check before trusting
deep anchors. Use [docs-task-sync](../docs-task-sync/SKILL.md) to
re-verify.

- `port_insertion/README.md` cheat sheet: `bb6a606`
- 04 (assembly pattern), 05 (MDP terms): `8d9a44e`
- 09 (modifying existing task): `cfb23ef`
