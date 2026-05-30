# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the port-insertion task.

Provides split root-frame pose and velocity observations consumed by the
policy obs group, plus privileged root-frame command-target observations
consumed by the ``cheatcode`` obs group used for BC dataset recording and
asymmetric critics.

Frame convention: pose, velocity, and command-target outputs of ``*_b``
functions are expressed in the asset root-link frame. The port-insertion
task mounts the UR5e with a 180-degree rotation about Z, so env-frame and
root-frame differ; root-frame is what the robot "sees" via forward
kinematics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_pos_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Body positions expressed in the asset's root-link frame."""

    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = _select_body_tensor(asset.data.body_pos_w, asset_cfg)
    num_bodies = body_pos_w.shape[1]
    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, num_bodies, -1)
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, num_bodies, -1)

    pos_b = quat_apply_inverse(root_quat_w, body_pos_w - root_pos_w)
    return pos_b.reshape(env.num_envs, -1)


def body_quat_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Body orientations expressed in the asset's root-link frame."""

    asset: Articulation = env.scene[asset_cfg.name]
    body_quat_w = _select_body_tensor(asset.data.body_quat_w, asset_cfg)
    num_bodies = body_quat_w.shape[1]
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, num_bodies, -1)

    quat_b = quat_mul(quat_inv(root_quat_w), body_quat_w)
    return quat_b.reshape(env.num_envs, -1)


def body_lin_vel_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Body linear velocity expressed in the asset's root-link frame."""

    asset: Articulation = env.scene[asset_cfg.name]
    body_lin_w = _select_body_tensor(asset.data.body_lin_vel_w, asset_cfg)
    num_bodies = body_lin_w.shape[1]
    root_lin_w = asset.data.root_lin_vel_w.unsqueeze(1).expand(-1, num_bodies, -1)
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, num_bodies, -1)

    lin_b = quat_apply_inverse(root_quat_w, body_lin_w - root_lin_w)
    return lin_b.reshape(env.num_envs, -1)


def body_ang_vel_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Body angular velocity expressed in the asset's root-link frame."""

    asset: Articulation = env.scene[asset_cfg.name]
    body_ang_w = _select_body_tensor(asset.data.body_ang_vel_w, asset_cfg)
    num_bodies = body_ang_w.shape[1]
    root_ang_w = asset.data.root_ang_vel_w.unsqueeze(1).expand(-1, num_bodies, -1)
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, num_bodies, -1)

    ang_b = quat_apply_inverse(root_quat_w, body_ang_w - root_ang_w)
    return ang_b.reshape(env.num_envs, -1)


def joint_applied_torque(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint torques applied by the actuator model (N·m, signed per joint).

    Reads ``Articulation.data.applied_torque`` — the post-clip torque the
    actuator model says it is exerting. For ``ImplicitActuator`` (PD handled
    by PhysX), IsaacLab recomputes this approximately from the PD law one
    step late, since PhysX does not expose the exerted joint torque directly.
    Shape is ``(num_envs, len(asset_cfg.joint_ids))``.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]


def body_incoming_wrench(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint-reaction wrench at each selected body, in that body's local frame.

    Reads ``Articulation.data.body_incoming_joint_wrench_b`` — the 6D wrench
    ``(Fx, Fy, Fz, Mx, My, Mz)`` the parent joint transmits to each body.
    The frame is the **body's own local frame** (IsaacLab convention), which
    matches what a wrist-mounted F/T sensor reports in hardware. This is
    *different* from the ``_b`` suffix on the pose/velocity helpers above,
    which means asset-root frame — hence the explicit name without ``_b``.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    wrench = _select_body_tensor(asset.data.body_incoming_joint_wrench_b, asset_cfg)
    return wrench.reshape(env.num_envs, -1)


def entrance_pos_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Entrance goal position expressed in the asset's root-link frame."""

    return _goal_pos_b(env, command_name, asset_cfg, goal_name="entrance")


def entrance_quat_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Entrance goal orientation expressed in the asset's root-link frame."""

    return _goal_quat_b(env, command_name, asset_cfg, goal_name="entrance")


def seat_pos_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Seat goal position expressed in the asset's root-link frame."""

    return _goal_pos_b(env, command_name, asset_cfg, goal_name="seat")


def seat_quat_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Seat goal orientation expressed in the asset's root-link frame."""

    return _goal_quat_b(env, command_name, asset_cfg, goal_name="seat")


def insertion_fraction(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "sfp_tip_link",
    lateral_threshold_m: float = 0.002,
) -> torch.Tensor:
    """Position-only insertion progress scalar, perpendicular-gated.

    Decomposes the seat-to-body world-frame error along the
    entrance->seat axis. With ``L = ||seat - entrance||`` and
    ``axis = (seat - entrance) / L``:

    - ``axial = (seat - body) . axis`` (positive when the body is behind the
      seat along the axis).
    - ``perp_dist = ||(seat - body) - axial * axis||``.

    Returns ``(1 - clamp(axial / L, 0, 1))`` gated to zero when
    ``perp_dist > lateral_threshold_m``. Holds at 1 past the seat (clamp).

    Internal math stays in world frame because the output is a frame-invariant
    scalar.

    Args:
        env: The environment.
        command_name: Name of the command term publishing ``entrance_pos_w``
            and ``seat_pos_w``.
        asset_cfg: The articulation owning the reference body.
        body_name: Name of the body used as the insertion tip.
        lateral_threshold_m: Max perpendicular distance from the
            entrance->seat line at which progress is reported.

    Returns:
        Progress scalar in ``[0, 1]``. Shape ``(num_envs, 1)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    goal = env.command_manager.get_term(command_name)
    body_id = _resolve_body_id(asset, asset_cfg, body_name)
    tip_pos_w = asset.data.body_pos_w[:, body_id, :]

    insertion_vec = goal.seat_pos_w - goal.entrance_pos_w
    length = insertion_vec.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    axis = insertion_vec / length

    err = goal.seat_pos_w - tip_pos_w
    axial = (err * axis).sum(dim=-1, keepdim=True)
    perp = err - axial * axis
    perp_dist = perp.norm(dim=-1, keepdim=True)

    progress = 1.0 - (axial / length).clamp(0.0, 1.0)
    gate = (perp_dist <= lateral_threshold_m).to(progress.dtype)
    return progress * gate


def _goal_pos_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    *,
    goal_name: str,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    goal_pos_w = _goal_tensor_w(env, command_name, goal_name, suffix="pos")
    return quat_apply_inverse(asset.data.root_quat_w, goal_pos_w - asset.data.root_pos_w)


def _goal_quat_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    *,
    goal_name: str,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    goal_quat_w = _goal_tensor_w(env, command_name, goal_name, suffix="quat")
    return quat_mul(quat_inv(asset.data.root_quat_w), goal_quat_w)


def _select_body_tensor(data: torch.Tensor, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    selected = data[:, asset_cfg.body_ids, :]
    if selected.ndim == 2:
        selected = selected.unsqueeze(1)
    return selected


def _goal_tensor_w(
    env: ManagerBasedRLEnv,
    command_name: str,
    goal_name: str,
    *,
    suffix: str,
) -> torch.Tensor:
    if goal_name not in {"entrance", "seat"}:
        raise ValueError(f"Unsupported insertion goal name: {goal_name}")
    if suffix not in {"pos", "quat"}:
        raise ValueError(f"Unsupported insertion goal tensor suffix: {suffix}")
    goal = env.command_manager.get_term(command_name)
    return getattr(goal, f"{goal_name}_{suffix}_w")


def _resolve_body_id(asset: Articulation, asset_cfg: SceneEntityCfg, body_name: str) -> int:
    """Return the first body id for ``asset_cfg``, falling back to ``body_name``.

    Prefers pre-resolved ``asset_cfg.body_ids`` (populated by IsaacLab at
    env init when ``body_names`` is configured) to avoid a per-step name
    lookup on the articulation.
    """
    if isinstance(asset_cfg.body_ids, int):
        return asset_cfg.body_ids
    if asset_cfg.body_ids is not None and not isinstance(asset_cfg.body_ids, slice):
        return int(asset_cfg.body_ids[0])
    body_ids, _ = asset.find_bodies(body_name)
    if len(body_ids) == 0:
        raise KeyError(f"Body '{body_name}' not found on asset '{asset_cfg.name}'.")
    return int(body_ids[0])


__all__ = [
    "body_ang_vel_b",
    "body_incoming_wrench",
    "body_lin_vel_b",
    "body_pos_b",
    "body_quat_b",
    "entrance_pos_b",
    "entrance_quat_b",
    "joint_applied_torque",
    "seat_pos_b",
    "seat_quat_b",
    "insertion_fraction",
]
