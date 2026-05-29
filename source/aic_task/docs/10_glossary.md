---
scope: vocabulary used by the rest of the aic_task docs — one definition per term plus a canonical example anchor
audience: AI agents working in this repo
last_verified_commit: cfb23ef
related:
  - 01_package_structure.md
  - 04_assembly_pattern.md
  - 05_mdp_terms.md
---

# 10 · Glossary

Reference for terms the other docs use without defining. Each entry is
one paragraph plus a file:line anchor pointing at the canonical example.
For *operational* use of the term, follow the cross-link to the doc that
walks through it; this file only fixes the meaning.

## Terms owned by this package

### assembly

The single dataclass value that ties a task's robot, target, layout,
controller, goal, and termination together — the **selection layer**'s
output. Each task module defines exactly one assembly constant; the env
cfg's `ASSEMBLY = …` binds the cfg to that constant. The pattern is the
focus of [04_assembly_pattern.md](04_assembly_pattern.md).

Canonical example:
[AIC_PORT_INSERTION_ASSEMBLY](../aic_task/tasks/manager_based/port_insertion/specs.py#L147).
Its `.validate()` call on the next line is what makes import-time spec
mismatches blow up before sim start.

### spec layer

The dataclasses inside a task's `specs.py` that describe *what* the task
selects, in plain frozen records — no `isaaclab.*` or `omni.*` imports.
The classes
[`ControllerSpec`](../aic_task/tasks/manager_based/port_insertion/specs.py#L34),
[`InsertionGoalSpec`](../aic_task/tasks/manager_based/port_insertion/specs.py#L51),
[`PortInsertionTerminationSpec`](../aic_task/tasks/manager_based/port_insertion/specs.py#L63),
and the bundling
[`PortInsertionAssemblySpec`](../aic_task/tasks/manager_based/port_insertion/specs.py#L75)
are the canonical example. Operational rules in
[04_assembly_pattern.md](04_assembly_pattern.md#why-split-layer-1-from-layer-2--the-spec-layer).

### builder layer

The pure functions inside a task's `builders.py` that translate the spec
layer into IsaacLab cfg objects. One function per env-cfg field
(`build_scene_cfg`, `build_action_cfg`, …). The builder layer is the
**only** file in a task that imports both `aic_task.asset_specs.*` and
`isaaclab.*` cfg classes.

Canonical example:
[builders.py](../aic_task/tasks/manager_based/port_insertion/builders.py).
The function-by-function map is in
[04_assembly_pattern.md](04_assembly_pattern.md#why-split-layer-2-from-layer-3--the-builder-layer).

### body role

A semantic name for a robot body, decoupled from the concrete body name
the USD uses. The robot spec maps a role string to a body name via
`body_name_for_role(role)`. The package uses two roles today: `tcp` and
`eef`. `tcp` is the body the controller drives; `eef` is the physical
insertion tip used by the goal and the terminations. Why both: the SFP
plug is rigidly attached past the gripper, so the IK target (TCP) and
the goal target (EEF) are different rigid links.

Canonical example:
[BodyRoleSpec for UR5e](../aic_task/asset_specs/robots.py#L105). On the
current UR5e cable asset, `tcp → "gripper_tcp"` and `eef → "sfp_tip_link"`.

### joint group

A named subset of an articulation's joints, with default positions, used
by the controller and the observation group as a single unit. Today
there is exactly one joint group, the UR5e
[`arm` group](../aic_task/asset_specs/robots.py#L72). `ControllerSpec.joint_group`
is the string that selects it.

### scene slot

A named, role-tagged instance of an asset inside a `SceneLayoutSpec`.
The `name` field is the stable scene key — that is what scripts use as
`env.scene[name]`. The `prim_path` is the USD path the spawner writes to
(usually templated with `{ENV_REGEX_NS}`). Scene slots also carry a
default pose, a `kinematic` flag, and a human-readable `purpose`.

Canonical example:
[AIC_NIC_CARD_TARGET_SLOT](../aic_task/asset_specs/scene.py#L138).
List of stable slot names in
[asset_specs/scene.py](../aic_task/asset_specs/scene.py#L13).

### scene layout

The `SceneLayoutSpec` that groups all scene slots a task uses (robot,
target, board, workcell, auxiliaries) plus its reset randomization. The
selection layer picks one layout per assembly. Today there is exactly
one,
[`AIC_PORT_INSERTION_LAYOUT`](../aic_task/asset_specs/scene.py#L175).

### asset spec

A frozen dataclass at the bottom of the layering that describes facts
about an asset that stay true wherever it is used — USD path, root prim,
joint groups, body roles, ports. No knowledge of any task. Subclasses:
`RobotAssetSpec`, `TargetAssetSpec`, `StaticAssetSpec`. Rules and
examples in the
[asset_specs/ README](../aic_task/asset_specs/README.md).

### command term

A `CommandTerm` subclass plus its `CommandTermCfg` — the mechanism
IsaacLab uses to publish per-step "what the policy is trying to do"
tensors. The task's
[`InsertionGoalCommand`](../aic_task/tasks/manager_based/port_insertion/mdp/commands.py#L19)
publishes the entrance / seat poses of the selected port. Full
reference in
[05_mdp_terms.md](05_mdp_terms.md#insertiongoalcommand).

### manager term

Umbrella name for the IsaacLab term types the manager-based env cfg
wires up: `ObservationTermCfg`, `EventTermCfg`, `TerminationTermCfg`,
`RewardTermCfg`, `CommandTermCfg`. Each is constructed with `func=…`
plus `params={…}`; the function does the work, the dict carries the
parameters. The single rule the assembly pattern adds: those `params`
come from the builder layer reading the spec, never from the term
itself reaching for the assembly. Detail in
[04_assembly_pattern.md](04_assembly_pattern.md#why-split-layer-2-from-layer-3--the-builder-layer).

### insertion goal

The episode-level target of the existing task — the EEF pose, expressed
in the port frame, that the agent must reach. Encoded as an
[`InsertionGoalSpec`](../aic_task/tasks/manager_based/port_insertion/specs.py#L51)
and rendered into world frame each step by the
[`InsertionGoalCommand`](../aic_task/tasks/manager_based/port_insertion/mdp/commands.py#L19)
term. The current goal is
[`NIC_PORT_0_INSERTION_GOAL`](../aic_task/tasks/manager_based/port_insertion/specs.py#L126).

### entrance frame / seat frame

The two USD prims on the target asset that the insertion goal command
resolves to. `seat_frame_path` is the fully-inserted pose; the agent
"wins" when the EEF holds this for `success_required_seconds`.
`entrance_frame_path` is the just-outside-the-port pose; scripted
controllers typically aim here first, then switch to seat. Defined on
each [`TargetPortSpec`](../aic_task/asset_specs/targets.py#L11).

### `validate()`

The method on the assembly that checks cross-spec consistency at module
import time — controller's `robot_slot` matches layout's robot slot,
goal's `target_slot` matches layout's target slot, the chosen port
exists on the target. The reason these are caught fast instead of at
scene-construction time.

Canonical example:
[PortInsertionAssemblySpec.validate](../aic_task/tasks/manager_based/port_insertion/specs.py#L91),
called on the next line at module scope. Builders also call
`assembly.validate()` at the top of every `build_*_cfg` so a stale
assembly fails fast even if the top-level call was skipped.

## Terms from the surrounding IsaacLab tree

### `ManagerBasedRLEnv` / `ManagerBasedRLEnvCfg`

The IsaacLab env class and cfg base that this package targets. `gym.make`
of an AIC task instantiates `ManagerBasedRLEnv` with the
`env_cfg_entry_point` cfg. Cfg subclasses are `@configclass`-decorated
and group seven fields: `scene`, `actions`, `commands`, `observations`,
`events`, `rewards`, `terminations`. Canonical task-side example:
[`PortInsertionEnvCfg`](../aic_task/tasks/manager_based/port_insertion/port_insertion_env_cfg.py#L29).

### `parse_env_cfg`

The `isaaclab_tasks.utils.parse_env_cfg(id, ...)` helper that resolves a
registered Gym ID's `env_cfg_entry_point` string to a cfg instance,
applying device / num_envs / fabric overrides. Used by every script that
launches a task. Detail in
[02_gym_registration.md](02_gym_registration.md#how-a-consumer-actually-launches-the-env).

### `import_packages`

The `isaaclab_tasks.utils.import_packages(name, blacklist)` walker that
recursively imports every subpackage under `name`, **skipping any whose
dotted name contains any string in `blacklist`**. This is how this
package's `gym.register` calls fire side-effectfully. Used by
[tasks/\_\_init\_\_.py:17](../aic_task/tasks/__init__.py#L17). Walker
contract and the substring-match gotcha in
[02_gym_registration.md](02_gym_registration.md#import_packages--what-it-actually-does).

### `_BLACKLIST_PKGS`

The list passed into `import_packages`. In this package it is
`["utils", ".mdp"]` — every task's `mdp/` submodule is skipped, and any
future `utils/` helper is also skipped. The match is *substring*, not
*path*, so naming any internal module `utils` will silently skip it. See
[02_gym_registration.md](02_gym_registration.md#the-blacklist-used-here).

### `ENV_REGEX_NS`

The `{ENV_REGEX_NS}` template in a `SceneSlotSpec.prim_path` is an
IsaacLab placeholder that the env replaces with the per-env namespace at
spawn time (so each env in a vectorized run gets its own prim subtree).
You write `prim_path="{ENV_REGEX_NS}/target"`; IsaacLab spawns
`/World/envs/env_0/target`, `/World/envs/env_1/target`, …

Canonical example: every slot in
[asset_specs/scene.py](../aic_task/asset_specs/scene.py#L94).

### `SceneEntityCfg`

The IsaacLab cfg type that a manager term uses to refer to an asset (and
optionally its joints / bodies) by scene-slot name. `params={"asset_cfg":
SceneEntityCfg("robot", body_names="sfp_tip_link")}` is how a
termination term tells IsaacLab "the body I need lives on the robot
slot, and it's called `sfp_tip_link`". The env resolves the name to
indices the first time the term runs.

Canonical example: the robot-cfg construction inside
[`build_termination_cfg`](../aic_task/tasks/manager_based/port_insertion/builders.py#L247).

### `@configclass`

IsaacLab's `dataclass`-equivalent decorator (from
`isaaclab.utils.configclass`). Used on every cfg subclass — the env cfg,
the inline `PolicyCfg` inside `build_observation_cfg`, every
`<New>EnvCfg` you write under
[08_adding_a_new_task.md](08_adding_a_new_task.md). It adds the
`__post_init__` hook that `ManagerBasedRLEnvCfg` uses to finalize the
cfg after construction.
