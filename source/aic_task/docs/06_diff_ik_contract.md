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
Distilled from the fix in
[`scripts/random_agent.py`](../../../scripts/random_agent.py).

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

The goal command (`insertion_goal`) is published for the **EEF body
(`sfp_tip_link`)**, not the TCP. So we have to:

1. Compute the static `EEF → TCP` offset from the live world poses.
2. Shift the EEF goal by that offset to get the TCP goal.
3. Re-express both the current TCP pose and the TCP goal in the **robot root
   frame**.
4. Take the pose error in root frame, divide by `scale`, clamp to `[-1, 1]`.

```python
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    subtract_frame_transforms,
)

# 1. Read world-frame poses from the articulation.
tcp_pos_w  = robot.data.body_pos_w[:, tcp_idx, :]   # gripper_tcp
tcp_quat_w = robot.data.body_quat_w[:, tcp_idx, :]
eef_pos_w  = robot.data.body_pos_w[:, eef_idx, :]   # sfp_tip_link
eef_quat_w = robot.data.body_quat_w[:, eef_idx, :]
root_pos_w  = robot.data.root_pos_w
root_quat_w = robot.data.root_quat_w

# 2. Static EEF→TCP offset (rigid link, constant per step).
tcp_in_eef_pos, tcp_in_eef_quat = subtract_frame_transforms(
    eef_pos_w, eef_quat_w, tcp_pos_w, tcp_quat_w,
)

# 3. Pull the EEF-frame goal from the command term and shift it by EEF→TCP
#    to obtain the TCP goal in world frame.
eef_goal_pos_w, eef_goal_quat_w = goal_term.entrance_pos_w, goal_term.entrance_quat_w
tcp_goal_pos_w, tcp_goal_quat_w = combine_frame_transforms(
    eef_goal_pos_w, eef_goal_quat_w, tcp_in_eef_pos, tcp_in_eef_quat,
)

# 4. Re-express current TCP and TCP goal in the ROBOT ROOT frame — this is the
#    frame DiffIK relative-mode applies the delta in.
tcp_pos_b, tcp_quat_b = subtract_frame_transforms(
    root_pos_w, root_quat_w, tcp_pos_w, tcp_quat_w,
)
tcp_goal_pos_b, tcp_goal_quat_b = subtract_frame_transforms(
    root_pos_w, root_quat_w, tcp_goal_pos_w, tcp_goal_quat_w,
)

# 5. Pose error in root frame, then scale + clamp to one action unit.
pos_err, rot_err = compute_pose_error(
    tcp_pos_b, tcp_quat_b, tcp_goal_pos_b, tcp_goal_quat_b,
    rot_error_type="axis_angle",
)
scale = torch.tensor(
    [0.015, 0.015, 0.015, 0.025, 0.025, 0.025], device=env.unwrapped.device,
)
action = (torch.cat([pos_err, rot_err], dim=-1) / scale).clamp(-1.0, 1.0)
env.step(action)
```

## Gotchas

- **World vs. root frame.** This is the easy bug. `compute_pose_error` returns
  the delta in whatever frame you fed it — feed it root-frame poses, not world.
- **TCP vs. EEF.** The controller drives `gripper_tcp`, but the command term
  publishes `sfp_tip_link`. Always apply the static `EEF→TCP` offset; don't aim
  the TCP straight at the EEF goal.
- **Body indices.** Resolve with `robot.find_bodies(name)` once outside the
  loop. The `gripper_tcp` / `sfp_tip_link` names come from the
  `BodyRoleSpec` entries on `UR5E_CABLE_ASSET`.
- **Scale.** It must match `ControllerSpec.scale` for the env. The clamp to
  `[-1, 1]` is what gives "one increment toward the goal per step"; without
  the clamp a large initial error produces a giant requested delta that the
  IK can't actually realize.
- **Markers for sanity.** Visualizing the EEF and goal frames (`--markers` in
  `random_agent.py`) is the fastest way to confirm you're aiming at the right
  thing — they should drift toward overlap as the arm converges.
