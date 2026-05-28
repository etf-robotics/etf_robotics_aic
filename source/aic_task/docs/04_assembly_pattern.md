---
scope: design rationale for the asset_specs → specs → builders → env_cfg layering used by aic_task
audience: AI agents working in this repo
last_verified_commit: 8d9a44e
related:
  - 01_package_structure.md
  - 03_port_insertion_overview.md
  - ../aic_task/asset_specs/README.md
  - ../aic_task/tasks/manager_based/port_insertion/README.md
---

# 04 · The Assembly Pattern

This package never builds an IsaacLab cfg in a single place. It splits the
construction into four layers, each with one responsibility. After reading
this doc you should know which layer to touch when, why the boundary exists,
and how to add a new assembly variant without rewriting the env cfg or any
builder.

The running example throughout is `AIC_PORT_INSERTION_ASSEMBLY`, defined at
[specs.py:147](../aic_task/tasks/manager_based/port_insertion/specs.py#L147).

## The four layers

```text
asset_specs/                       Pure-Python contracts.
   │                                No isaaclab / omni imports.
   ▼
tasks/<task>/specs.py              Selection layer.
   │                                Picks assets + layout + controller + goal
   │                                + thresholds into one AssemblySpec.
   │                                Calls .validate() at import time.
   ▼
tasks/<task>/builders.py           Spec → IsaacLab cfg conversion.
   │                                Only file in the task that imports BOTH
   │                                aic_task.asset_specs AND isaaclab.*.
   ▼
tasks/<task>/<task>_env_cfg.py     @configclass env cfg.
                                    Wires the build_*_cfg outputs into a
                                    ManagerBasedRLEnvCfg subclass.
```

| Layer | File | Imports allowed | Imports forbidden |
|---|---|---|---|
| 1. asset contracts | [asset_specs/](../aic_task/asset_specs/) | stdlib only | `isaaclab.*`, `omni.*`, anything under `tasks/` |
| 2. assembly selection | [specs.py](../aic_task/tasks/manager_based/port_insertion/specs.py) | `aic_task.asset_specs`, stdlib | `isaaclab.*`, `omni.*`, anything under `mdp/` or `builders.py` |
| 3. cfg builders | [builders.py](../aic_task/tasks/manager_based/port_insertion/builders.py) | `aic_task.asset_specs`, `isaaclab.*`, local `mdp/*` | reading `AIC_PORT_INSERTION_ASSEMBLY` directly (it must be passed in) |
| 4. env cfg | [port_insertion_env_cfg.py](../aic_task/tasks/manager_based/port_insertion/port_insertion_env_cfg.py) | everything above | nothing useful below — never bypass the builders |

The arrows are one-way. The asset-spec layer cannot know that a task exists;
the spec layer cannot know what an `ArticulationCfg` is; the builder layer
cannot know that `AIC_PORT_INSERTION_ASSEMBLY` is the chosen assembly.

## Why split layer 1 from layer 2 — the spec layer

The asset-spec layer (`asset_specs/`) is the set of facts that stay true
wherever an asset is used. The selection layer (`specs.py`) is the set of
facts that are true *for this task's assembly only*. Keeping them separate
buys four things:

- **Readability and diff-ability.** The spec layer is plain frozen
  dataclasses. A diff of [specs.py](../aic_task/tasks/manager_based/port_insertion/specs.py)
  shows exactly what changed about the task, with no IsaacLab cfg noise
  mixed in.
- **Import-time validation.**
  [`AIC_PORT_INSERTION_ASSEMBLY.validate()`](../aic_task/tasks/manager_based/port_insertion/specs.py#L91)
  runs at module import. It fails fast if the controller's `robot_slot`
  doesn't match the layout's robot slot, if the goal targets the wrong
  scene slot, or if the chosen port doesn't exist on the chosen target.
  These are exactly the cross-spec mismatches a builder would otherwise
  catch only at scene-construction time, after a slow sim init.
- **Alternate assemblies are cheap.** A new assembly is one new
  `PortInsertionAssemblySpec(...)` value at the bottom of `specs.py` — the
  builders never change. See "Adding a new assembly" below.
- **No `omni.*` at config-class definition time.** `isaaclab.*` is fine to
  import outside Isaac Sim, but `omni.*` is not always available. Keeping
  the spec layer omni-free means importing the spec to inspect or test the
  assembly does not require a running stage.

The smell test for a spec-layer change: if your edit imports something from
`isaaclab.*`, you're in the wrong file. Move the logic down into
`builders.py` and have the spec describe the *parameters* the builder
needs.

## Why split layer 2 from layer 3 — the builder layer

The selection layer says *what* was chosen; the builder layer says *how*
that becomes an IsaacLab cfg. Splitting buys:

- **A single audit point for "spec → IsaacLab translation."** Every
  `build_*_cfg` in [builders.py](../aic_task/tasks/manager_based/port_insertion/builders.py)
  takes a `PortInsertionAssemblySpec` and returns one IsaacLab cfg object.
  If you want to know which IsaacLab type owns which spec field, this is
  the one file you read.
- **No hidden cross-references.** Builders receive the assembly as a
  parameter; they never reach for the module-level constant. That means a
  test can `build_action_cfg(MY_TEST_ASSEMBLY)` without monkey-patching.
- **MDP terms get parameters, not constants.** `build_termination_cfg`
  passes `position_threshold`, `orientation_threshold`, `required_seconds`,
  etc. as `params={...}` on the `DoneTerm`
  ([builders.py:241](../aic_task/tasks/manager_based/port_insertion/builders.py#L241)).
  The term classes in `mdp/terminations.py` never read the assembly. This
  is the rule that keeps `mdp/*.py` cleanly reusable.

The smell test for a builder-layer change: if your edit reaches for
`AIC_PORT_INSERTION_ASSEMBLY` by name instead of using the `assembly`
parameter, it's wrong. Same for any `mdp/*.py` that grows an
`from ..specs import ...` line.

## Why split layer 3 from layer 4 — the env cfg

The env cfg ([port_insertion_env_cfg.py](../aic_task/tasks/manager_based/port_insertion/port_insertion_env_cfg.py))
exists only to be a `@configclass` IsaacLab can resolve. It wires the
seven `build_*_cfg(ASSEMBLY)` outputs into a `ManagerBasedRLEnvCfg`
subclass and sets `sim.dt`, `decimation`, and `episode_length_s`. That's
all. There is no spec selection in this file — swapping `ASSEMBLY` is a
one-line change.

If you find yourself constructing IsaacLab cfg objects directly inside
the env cfg's `__post_init__`, push that construction into a new
`build_*_cfg` function on `builders.py`. The env cfg should remain a
declarative wiring layer.

## Where each kind of change belongs

The cheat sheet from the port-insertion
[task README](../aic_task/tasks/manager_based/port_insertion/README.md#what-lives-where-cheat-sheet)
is reproduced below with the *why*, so you can judge edge cases instead of
just following the table.

| If you want to change… | Edit | Why here |
|---|---|---|
| Success / failure thresholds | `PortInsertionTerminationSpec` in [specs.py](../aic_task/tasks/manager_based/port_insertion/specs.py#L62) | Pure numbers per assembly — no IsaacLab type involved. Builder copies them onto `DoneTerm.params`. |
| DiffIK scale, IK method, controlled body | `UR5E_DIFF_IK_CONTROLLER` in [specs.py](../aic_task/tasks/manager_based/port_insertion/specs.py#L112) | Same — controller knobs are spec data; `build_action_cfg` translates them. |
| The selected port or its EEF-in-port pose | `NIC_PORT_0_INSERTION_GOAL` in [specs.py](../aic_task/tasks/manager_based/port_insertion/specs.py#L126) | The goal is a *selection* (which port, which pose-in-port). Port frame paths live one layer down on `NIC_CARD_ASSET`. |
| Robot / target asset choice or layout | Imports at the top of [specs.py](../aic_task/tasks/manager_based/port_insertion/specs.py#L14) | Swap the constants from `asset_specs/`; rebuild `AIC_PORT_INSERTION_ASSEMBLY`. |
| Body name, joint group, USD path of an asset | [asset_specs/*.py](../aic_task/asset_specs/) | Asset facts. Never duplicate into the spec or builder layer. |
| Observation group composition | `build_observation_cfg` in [builders.py](../aic_task/tasks/manager_based/port_insertion/builders.py#L159) | The choice of `ObsTerm`s is an IsaacLab cfg concern; the spec only tells you the asset / command / action names to wire in. |
| Reset event roster | `build_event_cfg` in [builders.py](../aic_task/tasks/manager_based/port_insertion/builders.py#L188) | Same — events are IsaacLab `EventTerm`s. Randomization *ranges* live on the layout in [asset_specs/scene.py](../aic_task/asset_specs/scene.py). |
| `sim.dt`, `decimation`, `episode_length_s` | `PortInsertionEnvCfg.__post_init__` in [port_insertion_env_cfg.py](../aic_task/tasks/manager_based/port_insertion/port_insertion_env_cfg.py) | These are env-level scalars with no assembly variation today. |
| What the `insertion_goal` term publishes | [mdp/commands.py](../aic_task/tasks/manager_based/port_insertion/mdp/commands.py) | The term implementation. Builder only constructs its cfg from the spec. |
| When the env reports success / failure | [mdp/terminations.py](../aic_task/tasks/manager_based/port_insertion/mdp/terminations.py) | Same — the logic is in the term. Thresholds come from the spec via the builder. |

The reason every row points at exactly one file is the import direction:
no layer can substitute for the one below it. If the row above does not
fit, you are doing something the assembly pattern wasn't designed for —
think twice before adding a new abstraction.

## Adding a new assembly without touching builders or the env cfg

Concrete recipe — selecting a different port for the existing task:

1. In [asset_specs/targets.py](../aic_task/asset_specs/targets.py), confirm
   the port you want exists on `NIC_CARD_ASSET.ports`. (Today only
   `sfp_port_0` is defined — add a new `TargetPortSpec` first if not.)
2. In [specs.py](../aic_task/tasks/manager_based/port_insertion/specs.py),
   add a new `InsertionGoalSpec` next to `NIC_PORT_0_INSERTION_GOAL`,
   referencing the new port name and EEF-in-port pose.
3. Add a new `PortInsertionAssemblySpec(...)` constant next to
   `AIC_PORT_INSERTION_ASSEMBLY`, reusing the existing
   `UR5E_DIFF_IK_CONTROLLER` / `AIC_PORT_INSERTION_LAYOUT` /
   `AIC_PORT_INSERTION_TERMINATION` and substituting the new goal.
4. Call `.validate()` on the new constant.
5. Choose which assembly the env cfg uses. Two ways:
   - Hard-coded: import the new constant in
     [port_insertion_env_cfg.py](../aic_task/tasks/manager_based/port_insertion/port_insertion_env_cfg.py)
     and pass it to every `build_*_cfg`.
   - Environment-driven: read an env var (e.g. `AIC_PORT_INSERTION_ASSEMBLY`)
     in `specs.py` and pick the assembly there. The rest of the package
     stays untouched.

Either way, you never opened `builders.py`. Every cfg is reconstructed from
the new assembly by the same builder functions. If you find yourself
opening a builder, ask whether your change is really an *assembly* change —
if it is a *capability* change (new IK method, new observation term, new
event), the builder is the correct place.

## What an assembly violation looks like

Watch for these patterns in code review; they all break the layering:

- A `from isaaclab.*` line appearing in
  [asset_specs/](../aic_task/asset_specs/) or
  [specs.py](../aic_task/tasks/manager_based/port_insertion/specs.py).
  The spec layers must remain importable without IsaacLab.
- A `from ..specs import AIC_PORT_INSERTION_ASSEMBLY` line in
  [builders.py](../aic_task/tasks/manager_based/port_insertion/builders.py)
  or any `mdp/*.py`. The builder receives the assembly as a parameter; the
  term receives its knobs as `params={...}`.
- A direct `ArticulationCfg(...)` / `RigidObjectCfg(...)` /
  `DifferentialInverseKinematicsActionCfg(...)` construction inside
  [port_insertion_env_cfg.py](../aic_task/tasks/manager_based/port_insertion/port_insertion_env_cfg.py).
  Move it into a new `build_*_cfg` and call that.
- A second source of truth for an asset fact (USD path, body name, joint
  list) appearing somewhere other than the matching `asset_specs/*.py`
  file. Asset facts live in exactly one place.

When in doubt, follow the import direction: lower layer never imports
upper. The validate-at-import step on the assembly catches the rest.
