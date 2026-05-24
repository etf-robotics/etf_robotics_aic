# Port Insertion Command Goal Notes

These notes describe the command/termination/oracle refactor for
`AIC-Port-Insertion-v0`.

The main change is that the port insertion target now lives in the task as an
IsaacLab command term named `insertion_goal`.  The oracle no longer owns the
selected port or the target offsets.  It reads the same command that
terminations read, so data collection, imitation learning, and later RL can
share one definition of "the task goal".

## Purpose

Before this change, the oracle script had its own target settings
(`port_index`, `target_xz_offset`, `approach_offset`) while the environment
termination used a different success definition.  That worked for a demo, but
it made the task brittle: the expert could aim at one target while the
environment judged success with another.

Now the flow is:

```text
commands.insertion_goal
  -> defines desired sfp_tip_link pose and insertion path
  -> oracle reads it and computes gripper_tcp actions
  -> success/failure terminations read it and reset the env
```

The controlled action is still relative DiffIK on `gripper_tcp`.  The task goal
is still for `sfp_tip_link`.  The oracle bridges those frames every step with:

```text
T_world_tcp_des = T_world_tip_des * T_tip_tcp_live
```

`T_tip_tcp_live` is recomputed from the live simulated poses each step because
the cable/tool can flex.

## Files Changed

- `source/aic_task/aic_task/tasks/manager_based/port_insertion/mdp/commands.py`
  adds `InsertionGoalCommand` and `InsertionGoalCommandCfg`.
- `source/aic_task/aic_task/tasks/manager_based/port_insertion/mdp/__init__.py`
  exports the local command helpers.
- `source/aic_task/aic_task/tasks/manager_based/port_insertion/port_insertion_env_cfg.py`
  adds `PortInsertionCommandsCfg`, command-aware success/failure terminations,
  and keeps timeout active.
- `source/aic_task/aic_task/tasks/manager_based/port_insertion/mdp/terminations.py`
  adds command-aware success and stationary-failure termination terms.
- `source/aic_task/aic_task/controllers/nic_card_insert_oracle.py`
  reads the command goal and keeps the TCP-from-tip control math.
- `scripts/il/nic_card_insert_oracle.py`
  stops disabling terminations and continuously resets oracle state for envs
  that terminate or time out.

## Goal Architecture

`InsertionGoalCommand` caches the selected port pose in the NIC-card root frame
on reset, then updates world-frame command buffers from the live `nic_card`
pose.  This keeps the command compatible with reset randomization of the board
or card.

The command exposes:

- final `sfp_tip_link` target position in world frame
- approach `sfp_tip_link` target position in world frame
- target `sfp_tip_link` quaternion in world frame
- insertion path vector, axis, and length
- port X/Y/Z axes in world frame
- compact 24-D command tensor for future observation/debug use

The 24-D tensor schema is currently:

```text
0:3    final_tip_pos_w
3:7    target_tip_quat_w
7:10   approach_tip_pos_w
10:13  path_axis_w
13:14  path_length
14:17  port_x_w
17:20  port_y_w
20:23  port_z_w
23:24  port_index
```

The policy observation was intentionally not changed in this pass.  Since the
task is fixed to port 0, the policy does not yet need a goal observation.  If
port 0 and port 1 are trained together later, the command tensor or a smaller
goal observation should be added to the policy group.

## Termination Behavior

Success is now based on the commanded final `sfp_tip_link` pose:

- position error must be within the configured success radius
- orientation error must be within the configured orientation threshold
- the condition must hold for a short stable window

Failure is now based on being stationary in the wrong place:

- only `sfp_tip_link` position is checked
- rotation is ignored
- if the tip stays within the movement window for the required time while still
  outside the success radius, the env terminates as failure
- this is a normal termination, not a timeout

Timeout remains the IsaacLab `time_out` termination.

## Hardcoded Values

These are the important hardcoded values after the refactor.

### Task goal

| Item | Value | Location |
| --- | --- | --- |
| Command name | `insertion_goal` | env config, oracle, terminations |
| Target asset | `nic_card` | `PortInsertionCommandsCfg` |
| Port name | `sfp_port_0` | `PortInsertionCommandsCfg` |
| Port index | `0` | `PortInsertionCommandsCfg` |
| Tip body | `sfp_tip_link` | terminations and oracle defaults |
| Controlled body | `gripper_tcp` | oracle and DiffIK action config |
| Target X/Z offset | `(0.0, 0.001)` m | `PortInsertionCommandsCfg` |
| Approach offset | `(0.0, -0.09, 0.0)` m | `PortInsertionCommandsCfg` |
| Orientation correction | local `R_x(+90 deg)` | `InsertionGoalCommand` and legacy oracle helper |
| Command resampling range | `(1.0e9, 1.0e9 + 1.0)` seconds | `InsertionGoalCommandCfg` |
| Compact command dim | `24` | `InsertionGoalCommand` |

### Terminations

| Item | Value | Location |
| --- | --- | --- |
| Success position threshold | `0.003` m | `PortInsertionTerminationsCfg.success` |
| Success orientation threshold | `4 deg` | `PortInsertionTerminationsCfg.success` |
| Success hold time | `0.5` s | `PortInsertionTerminationsCfg.success` |
| Failure movement threshold | `0.001` m | `PortInsertionTerminationsCfg.failed_stationary` |
| Failure wrong-position threshold | outside `0.003` m | `PortInsertionTerminationsCfg.failed_stationary` |
| Failure stationary time | `1.0` s | `PortInsertionTerminationsCfg.failed_stationary` |
| Timeout | `120.0` s episode length | `PortInsertionEnvCfg.__post_init__` |

### Scene and physics choices

| Item | Value | Reason |
| --- | --- | --- |
| Default env count | `1` | task config default |
| Env spacing | `4.0` | keeps envs separated |
| `replicate_physics` | `False` | needed because cable/rope collision group does not replicate cleanly |
| `filter_collisions` | `False` | avoids adding IsaacLab collision groups on top of the cable collision setup |
| Arm effort limit | `300.0` | extra insertion authority for cable/contact demo |
| Arm stiffness | `6000.0` | extra insertion authority |
| Arm damping | `300.0` | extra insertion authority |

### Oracle runner defaults

These are script control/debug defaults, not task-goal definitions:

| CLI/default | Value |
| --- | --- |
| `--step_hz` | `30` |
| `--max_episode_steps` | `1200` |
| `--approach_threshold` | `0.015` m |
| `--insert_lateral_threshold` | `0.010` m |
| `--insert_orientation_threshold_deg` | `4.0` deg |
| `--insert_lookahead` | `0.002` m |
| `--final_threshold` | `0.003` m |
| `--insert_speed` | `0.010` m/s |
| `--pos_gain` | `1.2` |
| `--rot_gain` | `0.2` |
| `--max_pos_delta` | `0.020` m |
| `--insert_max_pos_delta` | `0.02` m |
| `--max_rot_delta` | `2.5` |
| `--log_every` | `5` steps |
| point logging | disabled by default |
| point-log env | `0` |
| start joints | `(0.55, -1.3642, -1.6648, -1.6933, 1.5710, 1.4110)` |
| start settle steps | `20` |
| Fabric | enabled by default |

The runner no longer has `--port_index`, `--target_xz_offset`,
`--approach_offset`, or `--hold_steps`.  Those were removed so the script cannot
quietly disagree with the task definition.

## What To Change Later

For training both ports, make the command sample or assign `port_index` per env
and expose a compact command observation to the policy.  Then the oracle,
success, and failure terms should continue to work because they already read
the command buffers.

For force-aware insertion, add contact or wrench observations separately.  The
command should remain the geometric goal; forces should inform rewards,
observations, or safety/failure logic rather than replacing the goal.
