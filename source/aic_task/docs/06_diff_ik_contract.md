---
scope: how to drive the relative-mode pose DiffIK action used by AIC-Port-Insertion-v0 toward the insertion_goal command
audience: AI agents working in this repo
last_verified_commit: 8d9a44e
related:
  - 03_port_insertion_overview.md
  - 05_mdp_terms.md
---

# 06 · DiffIK Goal-Driving Contract

How to drive the robot toward the `insertion_goal` (or any EEF-frame pose)
with the relative-mode pose DiffIK action used by `AIC-Port-Insertion-v0`.
Distilled from
[`scripts/direct_entrance_approach.py`](../../../scripts/direct_entrance_approach.py).

## Action contract

The action term is `DifferentialInverseKinematicsActionCfg` with:

- `body_name = "gripper_tcp"` (the controlled TCP body, set via
  `ROBOT_ROLE_TCP` in
  [`asset_specs/robots.py`](../aic_task/asset_specs/robots.py)
  and wired up in
  [`tasks/manager_based/port_insertion/builders.py`](../aic_task/tasks/manager_based/port_insertion/builders.py))
- `command_type = "pose"`, `use_relative_mode = True`
- `scale = (0.015, 0.015, 0.015, 0.025, 0.025, 0.025)` — per-step caps on the
  delta the controller will accept (xyz in meters, rotvec in radians)

In this mode the 6-D action is `(dx, dy, dz, drx, dry, drz)` applied as a delta
**to the current TCP pose, expressed in the robot root (base) frame** — the
IsaacLab implementation in the parent tree at
`/home/etfrobot/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py`
(`_compute_frame_pose`) subtracts `root_pos_w/root_quat_w` before handing the
pose to the controller, which then calls `apply_delta_pose`.

> The controller does not look at the world frame. If you compute the delta in
> world frame and the robot base is rotated, the arm aligns the wrong thing.

## Recipe

The root-frame entrance goal is exposed for the **EEF body
(`sfp_tip_link`)**, not the TCP. Scripted agents should read the named
root-frame observations returned by `env.reset()` / `env.step()`, then:

1. Compute the static `EEF → TCP` offset from the current root-frame poses.
2. Shift the EEF goal by that offset to get the TCP goal.
3. Take the TCP pose error in root frame, divide by `scale`, clamp to `[-1, 1]`.

```python
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    subtract_frame_transforms,
)

# 1. Read root-frame poses from the observation API.
policy = obs["policy"]
tcp_pos_b = policy["tcp_pos_b"]    # gripper_tcp
tcp_quat_b = policy["tcp_quat_b"]
eef_pos_b = policy["eef_pos_b"]    # sfp_tip_link
eef_quat_b = policy["eef_quat_b"]

# 2. Static EEF→TCP offset (rigid link, constant per step).
tcp_in_eef_pos, tcp_in_eef_quat = subtract_frame_transforms(
    eef_pos_b, eef_quat_b, tcp_pos_b, tcp_quat_b,
)

# 3. Pull the root-frame EEF entrance goal from the cheatcode observations
#    and shift it by EEF→TCP to obtain the root-frame TCP goal.
cheatcode = obs["cheatcode"]
eef_goal_pos_b = cheatcode["entrance_pos_b"]
eef_goal_quat_b = cheatcode["entrance_quat_b"]
tcp_goal_pos_b, tcp_goal_quat_b = combine_frame_transforms(
    eef_goal_pos_b, eef_goal_quat_b, tcp_in_eef_pos, tcp_in_eef_quat,
)

# 4. Pose error in root frame, then scale + clamp to one action unit.
pos_err, rot_err = compute_pose_error(
    tcp_pos_b, tcp_quat_b, tcp_goal_pos_b, tcp_goal_quat_b,
    rot_error_type="axis_angle",
)
scale = torch.tensor(
    [0.015, 0.015, 0.015, 0.025, 0.025, 0.025], device=env.unwrapped.device,
)
action = (torch.cat([pos_err, rot_err], dim=-1) / scale).clamp(-1.0, 1.0)
obs, _, _, _, _ = env.step(action)
```

## Gotchas

- **World vs. root frame.** This is the easy bug. `compute_pose_error` returns
  the delta in whatever frame you fed it — feed it root-frame poses, not world.
- **TCP vs. EEF.** The controller drives `gripper_tcp`, but the command term
  publishes `sfp_tip_link`. Always apply the static `EEF→TCP` offset; don't aim
  the TCP straight at the EEF goal.
- **Observation groups.** Current TCP/EEF poses live in `obs["policy"]`; the
  privileged entrance goal lives in `obs["cheatcode"]`. Keep the observation
  dict returned by each `env.step()` call.
- **Scale.** It must match `ControllerSpec.scale` for the env. The clamp to
  `[-1, 1]` is what gives "one increment toward the goal per step"; without
  the clamp a large initial error produces a giant requested delta that the
  IK can't actually realize.
