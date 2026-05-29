---
scope: operational walkthrough for common edits to AIC-Port-Insertion-v0; one entry per cheat-sheet row, each pointing at the single file to edit
audience: AI agents working in this repo
last_verified_commit: cfb23ef
related:
  - 03_port_insertion_overview.md
  - 04_assembly_pattern.md
  - 05_mdp_terms.md
  - ../aic_task/tasks/manager_based/port_insertion/README.md
---

# 09 · Modifying the Existing Task

The companion to [04_assembly_pattern.md](04_assembly_pattern.md). The
assembly pattern says *why* a change lives in one file; this doc shows
exactly *what* the diff looks like.

Workflow for any change you find here:

1. Find the row that matches your intent.
2. Open the named file and edit only that field / function.
3. The assembly's `.validate()` call at
   [specs.py:156](../aic_task/tasks/manager_based/port_insertion/specs.py#L156)
   re-runs on the next `import aic_task` and catches most cross-spec
   mismatches before any sim starts.

If your edit doesn't fit any row, **stop**. The pattern's promise is that
edits are single-file. If yours isn't, you may be conflating two changes
(e.g. "swap the robot *and* change the goal pose"); split it. Or you may
be doing something the pattern wasn't designed for, in which case the
right next step is [08_adding_a_new_task.md](08_adding_a_new_task.md),
not a multi-file patch to the existing task.

## Cheat-sheet entries

Each entry below is one cheat-sheet row from
[port_insertion/README.md](../aic_task/tasks/manager_based/port_insertion/README.md#what-lives-where-cheat-sheet),
unpacked with a worked example.

### Tighten the success position threshold

Cheat-sheet row: *Success / failure thresholds → `PortInsertionTerminationSpec` in `specs.py`*.

Goal: require ≤ 2 mm position error (currently 3 mm).

Edit
[specs.py:138](../aic_task/tasks/manager_based/port_insertion/specs.py#L138):

```diff
 AIC_PORT_INSERTION_TERMINATION = PortInsertionTerminationSpec(
-    success_position_threshold=0.003,
+    success_position_threshold=0.002,
     success_orientation_threshold_rad=math.radians(4.0),
     success_required_seconds=0.5,
     stationary_movement_threshold=0.001,
     stationary_success_position_threshold=0.003,
     stationary_required_seconds=1.0,
 )
```

Why here: thresholds are pure numbers per assembly. The builder copies
them onto `DoneTerm.params` at
[builders.py:256](../aic_task/tasks/manager_based/port_insertion/builders.py#L256);
the termination term reads them from `params` and never imports the
spec. Detail in
[05_mdp_terms.md](05_mdp_terms.md#insertiongoalreachedsuccess).

Subtle: `success_position_threshold` and `stationary_success_position_threshold`
are *different* fields with the same default value. The latter sets the
"outside the goal" boundary that prevents `failed_stationary` from firing
when the robot is parked on the goal. If you tighten success, decide
whether you want stationary's boundary to move too — see
[05_mdp_terms.md](05_mdp_terms.md#edge-cases-worth-knowing-3) for the
race-condition rationale.

### Loosen / tighten the success time

Same file, same dataclass — change `success_required_seconds`. With the
current `sim.dt = 1/120` and `decimation = 4`, the required consecutive
step count is `max(1, ceil(seconds / (4/120)))` (so `0.5 s → 15 steps`).
Detail at
[05_mdp_terms.md](05_mdp_terms.md#insertiongoalreachedsuccess).

### Change the DiffIK scale (per-step position cap)

Cheat-sheet row: *DiffIK scale, IK method, controlled body → `UR5E_DIFF_IK_CONTROLLER` in `specs.py`*.

Goal: shrink the per-step XYZ cap from 15 mm to 10 mm.

Edit
[specs.py:112](../aic_task/tasks/manager_based/port_insertion/specs.py#L112):

```diff
 UR5E_DIFF_IK_CONTROLLER = ControllerSpec(
     name="ur5e_diff_ik_tcp",
     action_name="arm_action",
     action_type="diff_ik",
     ...
-    scale=(0.015, 0.015, 0.015, 0.025, 0.025, 0.025),
+    scale=(0.010, 0.010, 0.010, 0.025, 0.025, 0.025),
 )
```

Why here: `build_action_cfg` reads the spec and constructs
`DifferentialInverseKinematicsActionCfg(scale=controller.scale, ...)` at
[builders.py:118](../aic_task/tasks/manager_based/port_insertion/builders.py#L118).
The IsaacLab type lives there; the *value* lives in the spec.

After the edit, downstream code that emits actions (e.g. scripted
controllers in `scripts/`) must match the new scale. See
[06_diff_ik_contract.md](06_diff_ik_contract.md) for the contract.

### Swap the controlled body

Same file, same dataclass — change `controlled_body_role`. The current
value is `ROBOT_ROLE_TCP` (= `"gripper_tcp"`). To drive the EEF directly
instead:

```diff
-    controlled_body_role=ROBOT_ROLE_TCP,
+    controlled_body_role=ROBOT_ROLE_EEF,
```

(You will also need to import `ROBOT_ROLE_EEF` from `aic_task.asset_specs`
at the top of `specs.py`.) The actual body name comes from
[`UR5E_CABLE_ASSET.body_name_for_role`](../aic_task/asset_specs/robots.py#L105),
which today returns `"gripper_tcp"` / `"sfp_tip_link"` for TCP / EEF
respectively. You do not edit `builders.py`; it reads the role and asks
the robot spec for the body name at
[builders.py:107](../aic_task/tasks/manager_based/port_insertion/builders.py#L107).

Heads up: the success / stationary terminations still target the **EEF
body** (`tip_body = robot.body_name_for_role(ROBOT_ROLE_EEF)`, set in
[builders.py:246](../aic_task/tasks/manager_based/port_insertion/builders.py#L246)).
That's by design — the EEF is what physically must reach the seat. Don't
chase the role swap into the termination builder.

### Retarget which port the goal points at

Cheat-sheet row: *The selected port or its EEF-in-port pose → `NIC_PORT_0_INSERTION_GOAL` in `specs.py`*.

Today the NIC target asset defines exactly one port,
[NIC_SFP_PORT_0](../aic_task/asset_specs/targets.py#L45). To swap to a
different port you must first add the new `TargetPortSpec` to
`NIC_CARD_ASSET.ports` in
[asset_specs/targets.py](../aic_task/asset_specs/targets.py) (an asset
fact — the port frame paths come from the USD), then change the goal:

```diff
 NIC_PORT_0_INSERTION_GOAL = InsertionGoalSpec(
     command_name="insertion_goal",
     target_slot=SCENE_SLOT_TARGET,
-    port_name="sfp_port_0",
+    port_name="sfp_port_1",
     eef_pose_in_port_frame=PoseSpec(
         pos=(0.0, 0.0, 0.001),
         rot=(math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),
     ),
     ...
 )
```

The asset-spec edit is non-negotiable: the port frame USD paths
(`seat_frame_path`, `entrance_frame_path`) must exist as prims under the
target, or the command term's resample raises `KeyError` on first reset.
Detail at
[05_mdp_terms.md](05_mdp_terms.md#edge-cases-worth-knowing).

### Move the EEF-in-port goal pose

Same file, same `InsertionGoalSpec` — change `eef_pose_in_port_frame`.
This is the pose the EEF aims at, expressed in the port frame:

```diff
     eef_pose_in_port_frame=PoseSpec(
-        pos=(0.0, 0.0, 0.001),
+        pos=(0.0, 0.0, 0.003),
         rot=(math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),
     ),
```

The quaternion is `wxyz`. See
[05_mdp_terms.md](05_mdp_terms.md#insertiongoalcommand) for what the
command term does with this and how the goal is recomposed in world
frame each step.

### Swap the robot, target, or layout asset

Cheat-sheet row: *Robot / target asset choice or layout → imports at the top of `specs.py`*.

To swap, e.g., the target asset constant:

```diff
 from aic_task.asset_specs import (
     AIC_PORT_INSERTION_LAYOUT,
-    NIC_CARD_ASSET,
+    MY_NEW_TARGET_ASSET as NIC_CARD_ASSET,
     ...
 )
```

(Or rename your new target everywhere it is referenced — same effect.)
The bottom-of-file `AIC_PORT_INSERTION_ASSEMBLY` then uses the new
constant. `validate()` will fail at import if the new target lacks the
port named by `NIC_PORT_0_INSERTION_GOAL.port_name`; that's the desired
behavior — fix one side before declaring the swap done.

If your "swap" is actually "use a different asset *and* a different
behavior", split it: do the asset swap first, get the existing task
running on the new asset, then make behavioral changes in their own
commits.

### Add a new observation term

Cheat-sheet row: *Observation group composition → `build_observation_cfg` in `builders.py`*.

Goal: add `joint_torque` to the policy observation group.

Edit
[builders.py:159](../aic_task/tasks/manager_based/port_insertion/builders.py#L159).
The function builds a single `ObservationGroupCfg` (named `policy`) by
declaring `ObsTerm` fields inside a local `@configclass`. Append a new
`ObsTerm`:

```diff
     @configclass
     class PolicyCfg(ObsGroup):
         joint_pos = ObsTerm(...)
         joint_vel = ObsTerm(...)
         actions = ObsTerm(func=last_action, params={"action_name": action_name})
         insertion_goal = ObsTerm(func=generated_commands, params={"command_name": command_name})
+        joint_torque = ObsTerm(
+            func=joint_torque_rel,  # add the import at the top of builders.py
+            params={"asset_cfg": SceneEntityCfg(robot_name, joint_names=joint_names)},
+        )

         def __post_init__(self) -> None:
             self.enable_corruption = False
             self.concatenate_terms = True
```

Why here: which `ObsTerm`s to wire is an IsaacLab cfg concern. The spec
only tells you the asset / command / action names to plug in; you read
them off `assembly.robot`, `assembly.controller`, `assembly.goal` already
at the top of the function. Don't add a "what observations to include"
field to the spec — that's a builder concern.

If your new term needs parameters the assembly doesn't carry today,
*then* the change crosses into the spec layer (add the field to a
spec dataclass first, then plumb it through the builder). Most adds
don't.

### Add a non-empty reward manager

Cheat-sheet row: similar to *Observation group composition* — the empty
reward today is a placeholder, not a spec'd choice.

Add a `build_reward_cfg(assembly)` in
[builders.py](../aic_task/tasks/manager_based/port_insertion/builders.py)
that returns a dict of `RewardTermCfg` keyed by name, mirroring the
shape of `build_termination_cfg`. Wire it from
[port_insertion_env_cfg.py:35](../aic_task/tasks/manager_based/port_insertion/port_insertion_env_cfg.py#L35):

```diff
-    rewards = build_empty_reward_cfg()
+    rewards = build_reward_cfg(ASSEMBLY)
```

Reward terms behave like termination terms in the layering: their
parameters come from `params={...}`, populated by the builder from the
assembly. Don't read `AIC_PORT_INSERTION_ASSEMBLY` from inside a reward
function.

### Change a randomization range

Cheat-sheet row: *Reset event roster or randomization ranges → `build_event_cfg` in `builders.py` (ranges live on the layout in `asset_specs/scene.py`)*.

Ranges are layout-owned. Goal: widen the board's yaw range from ±0.35 rad
to ±0.5 rad.

Edit
[asset_specs/scene.py:148](../aic_task/asset_specs/scene.py#L148):

```diff
 AIC_PORT_INSERTION_RANDOMIZATION = LayoutRandomizationSpec(
     board_slot_name=SCENE_SLOT_BOARD,
     board_ranges=(
         AxisRangeSpec("x", (-0.04, 0.04)),
         AxisRangeSpec("y", (-0.04, 0.04)),
-        AxisRangeSpec("yaw", (-0.35, 0.35)),
+        AxisRangeSpec("yaw", (-0.5, 0.5)),
     ),
     ...
 )
```

Why here: the asset / layout layer owns where things go and how much they
move. `build_event_cfg` translates those into the `params={...}` of
`randomize_board_and_parts` at
[builders.py:227](../aic_task/tasks/manager_based/port_insertion/builders.py#L227).
No builder change needed.

If you want to drop or add a randomized axis (e.g. randomize `z` too),
add it as a new `AxisRangeSpec` here. The event function loops over the
provided dict.

### Replace `reset_joints_by_offset` with a deterministic reset

Cheat-sheet row: *Reset event roster → `build_event_cfg` in `builders.py`*.

Today the event uses
[`reset_joints_by_offset`](../aic_task/tasks/manager_based/port_insertion/builders.py#L194)
with `position_range=(-0.3, 0.3)`. To switch to a deterministic reset
that snaps every joint to the spec default (no random offset), use the
unwired-but-available term documented in
[05_mdp_terms.md](05_mdp_terms.md#reset_robot_to_default_joint_pose):

```diff
+from .mdp.events import reset_robot_to_default_joint_pose

     events = {
         "reset_robot_joints": EventTerm(
-            func=reset_joints_by_offset,
+            func=reset_robot_to_default_joint_pose,
             mode="reset",
             params={
                 "asset_cfg": SceneEntityCfg(robot_slot_name),
-                "position_range": (-0.3, 0.3),
-                "velocity_range": (0.0, 0.0),
             },
         ),
         ...
     }
```

Don't keep the `position_range` / `velocity_range` keys — the
deterministic reset function takes no randomization knobs. Removing them
is the signal that this is a behavior change, not a parameter tweak.

### Change `sim.dt` / `decimation` / `episode_length_s`

Cheat-sheet row: *`sim.dt`, `decimation`, `episode_length_s` → `PortInsertionEnvCfg.__post_init__`*.

Edit
[port_insertion_env_cfg.py:38](../aic_task/tasks/manager_based/port_insertion/port_insertion_env_cfg.py#L38):

```diff
     def __post_init__(self):
         super().__post_init__()
-        self.decimation = 4
+        self.decimation = 2
         self.sim.render_interval = self.decimation
         self.episode_length_s = 120.0
         self.sim.dt = 1.0 / 120.0
```

Knock-on: success / failure thresholds are stored in *seconds*, not
steps. Changing `decimation` or `sim.dt` changes the step count the
terminations require but does not change the seconds. If you're tuning
both, do them together with the seconds-side number in mind.

### Change what `insertion_goal` publishes

Cheat-sheet row: *What `insertion_goal` publishes / how the term computes it → `mdp/commands.py`*.

This is a *capability* change, not a tuning change — you are modifying
the term's logic, not its parameters. Edit
[mdp/commands.py](../aic_task/tasks/manager_based/port_insertion/mdp/commands.py).
For the full reference (cfg fields, internal state, the prim resolver,
named tensors), see
[05_mdp_terms.md](05_mdp_terms.md#insertiongoalcommand). Two anchors
that catch most edits:

- The 14-D command tensor layout is set in
  [`_update_command`](../aic_task/tasks/manager_based/port_insertion/mdp/commands.py#L141).
  If you change the layout, every consumer (terminations, observation
  group, downstream agents) has to follow.
- The `final_tip_pos_w` / `target_tip_quat_w` aliases at
  [commands.py:63](../aic_task/tasks/manager_based/port_insertion/mdp/commands.py#L63)
  are what the termination terms read. Don't break the alias names
  without updating the consumers in
  [mdp/terminations.py](../aic_task/tasks/manager_based/port_insertion/mdp/terminations.py).

Once you've decided this is a capability change, the
[`aic-mdp-term-work`](../../../.claude/skills/aic-mdp-term-work/SKILL.md)
skill is the more focused entry point than this row.

### Change when success / failure fires

Cheat-sheet row: *When the env reports success / failure → `mdp/terminations.py`*.

Same capability-vs-tuning split. Threshold values live in `specs.py`;
*logic* lives here. If you are tweaking numbers, go back to the
"thresholds" row at the top of this doc. If you are changing the
condition itself (e.g. "success requires position AND velocity below
threshold"), edit
[mdp/terminations.py](../aic_task/tasks/manager_based/port_insertion/mdp/terminations.py)
and add the new knob to the call signature, the
`PortInsertionTerminationSpec` dataclass, and the `params={...}` block
in `build_termination_cfg` — in that order.

## When the assembly violates the pattern

These are the smells listed in
[04_assembly_pattern.md](04_assembly_pattern.md#what-an-assembly-violation-looks-like)
— if your edit introduces one, you've put the change in the wrong layer:

- `from isaaclab.*` in
  [asset_specs/](../aic_task/asset_specs/) or
  [specs.py](../aic_task/tasks/manager_based/port_insertion/specs.py).
- `from ..specs import AIC_PORT_INSERTION_ASSEMBLY` in
  [builders.py](../aic_task/tasks/manager_based/port_insertion/builders.py)
  or any `mdp/*.py`.
- An `ArticulationCfg(...)` / `DoneTerm(func=...)` / etc. constructed
  directly in
  [port_insertion_env_cfg.py](../aic_task/tasks/manager_based/port_insertion/port_insertion_env_cfg.py)
  instead of via a `build_*_cfg`.
- A USD path or body name appearing somewhere other than the matching
  `asset_specs/*.py`.

The fix is always "move it back to the right layer" — don't disable the
pattern.
