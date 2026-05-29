# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the port-insertion task.

Provides root-frame pose and velocity observations consumed by the policy
obs group, plus privileged goal/error observations consumed by the
``cheatcode`` obs group used for BC dataset recording and asymmetric
critics.

Frame convention: outputs of ``*_b`` functions are expressed in the asset
root-link frame. The port-insertion task mounts the UR5e with a 180-degree
rotation about Z, so env-frame and root-frame differ; root-frame is what
the robot "sees" via forward kinematics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_pose_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Body poses expressed in the asset's root-link frame.

    Args:
        env: The environment.
        asset_cfg: The articulation and target body indices.

    Returns:
        Flattened poses ``[x, y, z, qw, qx, qy, qz]`` per body in the root
        frame, stacked horizontally. Shape ``(num_envs, 7 * num_bodies)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    num_bodies = body_pos_w.shape[1]
    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, num_bodies, -1)
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, num_bodies, -1)

    pos_b = quat_apply_inverse(root_quat_w, body_pos_w - root_pos_w)
    quat_b = quat_mul(quat_inv(root_quat_w), body_quat_w)
    return torch.cat([pos_b, quat_b], dim=-1).reshape(env.num_envs, -1)


def body_vel_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Body 6D velocity ``[lin(3), ang(3)]`` in the asset's root-link frame.

    Subtracts the root-link world velocity then rotates into the root
    frame. For a fixed-base articulation the root terms are zero; the
    subtraction is kept for correctness with moving-base assets.

    Args:
        env: The environment.
        asset_cfg: The articulation and target body indices.

    Returns:
        Flattened 6D velocity per body. Shape ``(num_envs, 6 * num_bodies)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_lin_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    body_ang_w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :]
    num_bodies = body_lin_w.shape[1]
    root_lin_w = asset.data.root_lin_vel_w.unsqueeze(1).expand(-1, num_bodies, -1)
    root_ang_w = asset.data.root_ang_vel_w.unsqueeze(1).expand(-1, num_bodies, -1)
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, num_bodies, -1)

    lin_b = quat_apply_inverse(root_quat_w, body_lin_w - root_lin_w)
    ang_b = quat_apply_inverse(root_quat_w, body_ang_w - root_ang_w)
    return torch.cat([lin_b, ang_b], dim=-1).reshape(env.num_envs, -1)


def insertion_goal_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Insertion goal poses (entrance + seat) in the asset's root frame.

    Reads the world-frame entrance and seat poses published by the named
    command term and transforms each pose into the asset's root frame using
    the same convention as :func:`body_pose_b`.

    Args:
        env: The environment.
        command_name: Name of the registered command term that exposes
            ``entrance_pos_w``, ``entrance_quat_w``, ``seat_pos_w``,
            ``seat_quat_w``.
        asset_cfg: The articulation whose root frame is the target frame.

    Returns:
        Concatenated ``[entr_pos_b(3), entr_quat_b(4), seat_pos_b(3),
        seat_quat_b(4)]``. Shape ``(num_envs, 14)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    goal = env.command_manager.get_term(command_name)

    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w
    root_quat_inv = quat_inv(root_quat_w)

    entr_pos_b = quat_apply_inverse(root_quat_w, goal.entrance_pos_w - root_pos_w)
    entr_quat_b = quat_mul(root_quat_inv, goal.entrance_quat_w)
    seat_pos_b = quat_apply_inverse(root_quat_w, goal.seat_pos_w - root_pos_w)
    seat_quat_b = quat_mul(root_quat_inv, goal.seat_quat_w)

    return torch.cat([entr_pos_b, entr_quat_b, seat_pos_b, seat_quat_b], dim=-1)


def seat_pos_err_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "sfp_tip_link",
) -> torch.Tensor:
    """Body-frame position error from the named body to the goal seat.

    Computed as ``(seat_pos_w - body_pos_w)`` rotated into the asset root
    frame.

    Args:
        env: The environment.
        command_name: Name of the command term publishing ``seat_pos_w``.
        asset_cfg: The articulation owning the reference body.
        body_name: Name of the body subtracted from the seat position
            (typically the EEF / insertion tip).

    Returns:
        Position error per env in root frame. Shape ``(num_envs, 3)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    goal = env.command_manager.get_term(command_name)
    body_id = _resolve_body_id(asset, asset_cfg, body_name)

    eef_pos_w = asset.data.body_pos_w[:, body_id, :]
    return quat_apply_inverse(asset.data.root_quat_w, goal.seat_pos_w - eef_pos_w)


def seat_quat_delta_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "sfp_tip_link",
) -> torch.Tensor:
    """Body-frame quaternion delta from the named body to the goal seat.

    Computes ``inv(q_eef_b) * q_seat_b`` (the relative rotation expressed in
    the EEF frame), then canonicalizes the result to enforce ``qw >= 0`` so
    the SO(3) double-cover does not appear as a sign-flip in regression
    targets.

    Args:
        env: The environment.
        command_name: Name of the command term publishing ``seat_quat_w``.
        asset_cfg: The articulation owning the reference body.
        body_name: Name of the body compared against the seat orientation
            (typically the EEF / insertion tip).

    Returns:
        Quaternion delta with ``qw >= 0``. Shape ``(num_envs, 4)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    goal = env.command_manager.get_term(command_name)
    body_id = _resolve_body_id(asset, asset_cfg, body_name)

    root_quat_inv = quat_inv(asset.data.root_quat_w)
    eef_quat_b = quat_mul(root_quat_inv, asset.data.body_quat_w[:, body_id, :])
    seat_quat_b = quat_mul(root_quat_inv, goal.seat_quat_w)

    delta = quat_mul(quat_inv(eef_quat_b), seat_quat_b)
    return quat_unique(delta)


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
    "body_pose_b",
    "body_vel_b",
    "insertion_goal_b",
    "seat_pos_err_b",
    "seat_quat_delta_b",
    "insertion_fraction",
]
