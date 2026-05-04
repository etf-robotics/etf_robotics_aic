# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable reward functions for AIC manager-based tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_mul,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _first_body_id(asset: Articulation | RigidObject, asset_cfg: SceneEntityCfg, fallback_body_name: str) -> int:
    """Resolve the first configured body id, including before manager resolution."""
    if isinstance(asset_cfg.body_ids, int):
        return asset_cfg.body_ids
    if asset_cfg.body_ids is not None and not isinstance(asset_cfg.body_ids, slice):
        return asset_cfg.body_ids[0]

    body_names = asset_cfg.body_names
    if body_names is None:
        body_names = [fallback_body_name]
    elif isinstance(body_names, str):
        body_names = [body_names]

    body_ids, _ = asset.find_bodies(body_names, preserve_order=True)
    return body_ids[0]


def _target_position_w(
    target: RigidObject,
    target_cfg: SceneEntityCfg,
    fallback_offset: tuple[float, float, float],
) -> torch.Tensor:
    """Return a target body position or the asset root with a world-frame offset."""
    if isinstance(target_cfg.body_ids, int):
        return target.data.body_pos_w[:, target_cfg.body_ids]
    if target_cfg.body_ids is not None and not isinstance(target_cfg.body_ids, slice):
        return target.data.body_pos_w[:, target_cfg.body_ids[0]]

    offset_w = torch.tensor(fallback_offset, dtype=target.data.root_pos_w.dtype, device=target.data.root_pos_w.device)
    return target.data.root_pos_w + offset_w


def _target_quat_w(target: RigidObject, target_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return a target body orientation or the asset root orientation."""
    if isinstance(target_cfg.body_ids, int):
        return target.data.body_quat_w[:, target_cfg.body_ids]
    if target_cfg.body_ids is not None and not isinstance(target_cfg.body_ids, slice):
        return target.data.body_quat_w[:, target_cfg.body_ids[0]]
    return target.data.root_quat_w


def _z_axis_quat(num_envs: int, angle_deg: float, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Create a wxyz quaternion for a local Z-axis rotation."""
    half_angle = torch.tensor(angle_deg * torch.pi / 360.0, dtype=dtype, device=device)
    quat = torch.stack(
        [
            torch.cos(half_angle),
            torch.zeros_like(half_angle),
            torch.zeros_like(half_angle),
            torch.sin(half_angle),
        ]
    )
    return quat.unsqueeze(0).expand(num_envs, -1)


def _constant_vec(
    values: tuple[float, float, float],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create a 3D tensor constant on the correct device."""
    return torch.tensor(values, dtype=dtype, device=device).unsqueeze(0)


def _constant_rpy_quat(
    rpy: tuple[float, float, float],
    num_envs: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create a batched wxyz quaternion from constant XYZ Euler angles in radians."""
    rpy_tensor = torch.tensor(rpy, dtype=dtype, device=device)
    quat = quat_from_euler_xyz(rpy_tensor[0], rpy_tensor[1], rpy_tensor[2])
    return quat.unsqueeze(0).expand(num_envs, -1)


def _body_point_pos_w(
    asset: Articulation,
    asset_cfg: SceneEntityCfg,
    point_offset: tuple[float, float, float],
    fallback_body_name: str,
    offset_in_body_frame: bool,
) -> torch.Tensor:
    """Return a point attached to an articulation body in world coordinates."""
    body_id = _first_body_id(asset, asset_cfg, fallback_body_name)
    body_pos_w = asset.data.body_pos_w[:, body_id, :]
    point_offset_tensor = _constant_vec(
        point_offset,
        dtype=body_pos_w.dtype,
        device=body_pos_w.device,
    )
    if offset_in_body_frame:
        body_quat_w = asset.data.body_quat_w[:, body_id]
        point_offset_tensor = quat_apply(body_quat_w, point_offset_tensor.expand(body_pos_w.shape[0], -1))
    return body_pos_w + point_offset_tensor


def _asset_point_pos_w(
    target: RigidObject,
    target_cfg: SceneEntityCfg,
    point_offset: tuple[float, float, float],
    offset_in_target_frame: bool,
) -> torch.Tensor:
    """Return a point attached to a target asset/body in world coordinates."""
    target_pos_w = _target_position_w(target, target_cfg, (0.0, 0.0, 0.0))
    offset_tensor = _constant_vec(
        point_offset,
        dtype=target_pos_w.dtype,
        device=target_pos_w.device,
    )
    if offset_in_target_frame:
        target_quat_w = _target_quat_w(target, target_cfg)
        offset_tensor = quat_apply(target_quat_w, offset_tensor.expand(target_pos_w.shape[0], -1))
    return target_pos_w + offset_tensor


def _body_point_quat_w(
    asset: Articulation,
    asset_cfg: SceneEntityCfg,
    point_rpy: tuple[float, float, float],
    fallback_body_name: str,
) -> torch.Tensor:
    """Return the orientation of a point frame attached to an articulation body."""
    body_id = _first_body_id(asset, asset_cfg, fallback_body_name)
    body_quat_w = asset.data.body_quat_w[:, body_id]
    point_quat_b = _constant_rpy_quat(
        point_rpy,
        body_quat_w.shape[0],
        dtype=body_quat_w.dtype,
        device=body_quat_w.device,
    )
    return quat_mul(body_quat_w, point_quat_b)


def _asset_point_quat_w(
    target: RigidObject,
    target_cfg: SceneEntityCfg,
    point_rpy: tuple[float, float, float],
) -> torch.Tensor:
    """Return the orientation of a point frame attached to a target asset/body."""
    target_quat_w = _target_quat_w(target, target_cfg)
    point_quat_t = _constant_rpy_quat(
        point_rpy,
        target_quat_w.shape[0],
        dtype=target_quat_w.dtype,
        device=target_quat_w.device,
    )
    return quat_mul(target_quat_w, point_quat_t)


def position_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize tracking of the commanded position using L2 norm."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    return torch.linalg.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward commanded position tracking using ``1 - tanh(error / std)``."""
    return 1.0 - torch.tanh(position_command_error(env, command_name, asset_cfg) / std)


def position_command_error_exp(
    env: ManagerBasedRLEnv, sigma: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward commanded position tracking using a Gaussian kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    dist_sq = torch.sum(torch.square(curr_pos_w - des_pos_w), dim=1)
    return torch.exp(-dist_sq / (sigma**2))


def orientation_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize commanded orientation error as shortest-path angular distance in radians."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward commanded orientation tracking using ``1 - tanh(error / std)``."""
    return 1.0 - torch.tanh(orientation_command_error(env, command_name, asset_cfg) / std)


def ee_reaching_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Sparse +1 bonus when the body is within ``threshold`` meters of the command position."""
    distance = position_command_error(env, command_name, asset_cfg)
    return (distance < threshold).float()


def joint_torques_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize applied joint torques with squared L2 norm."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint accelerations with squared L2 norm."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_pos_limits(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joints that exceed their soft position limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def body_lin_acc_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize linear acceleration of selected bodies."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.linalg.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1)


def cheat_distance_to_asset_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    asset_fallback_body_name: str = "gripper_tcp",
    target_fallback_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Penalize distance from an articulation body to a target asset/body ground-truth pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    body_id = _first_body_id(asset, asset_cfg, asset_fallback_body_name)
    body_pos_w = asset.data.body_pos_w[:, body_id, :]
    target_pos_w = _target_position_w(target, target_cfg, target_fallback_offset)
    return torch.linalg.norm(body_pos_w - target_pos_w, dim=1)


def cheat_body_reaching_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    threshold: float,
    asset_fallback_body_name: str = "gripper_tcp",
    target_fallback_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Sparse +1 bonus when an articulation body reaches a target asset/body pose."""
    distance = cheat_distance_to_asset_l2(
        env,
        asset_cfg=asset_cfg,
        target_cfg=target_cfg,
        asset_fallback_body_name=asset_fallback_body_name,
        target_fallback_offset=target_fallback_offset,
    )
    return (distance < threshold).float()


def cheat_body_point_to_asset_point_distance_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    asset_point_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_point_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_fallback_body_name: str = "gripper_tcp",
    asset_offset_in_body_frame: bool = True,
    target_offset_in_target_frame: bool = True,
) -> torch.Tensor:
    """Penalize distance between a point on a robot body and a point on a target asset."""
    asset: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    asset_point_w = _body_point_pos_w(
        asset,
        asset_cfg,
        asset_point_offset,
        asset_fallback_body_name,
        asset_offset_in_body_frame,
    )
    target_point_w = _asset_point_pos_w(
        target,
        target_cfg,
        target_point_offset,
        target_offset_in_target_frame,
    )
    return torch.linalg.norm(asset_point_w - target_point_w, dim=1)


def cheat_body_point_to_asset_point_reaching_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    threshold: float,
    asset_point_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_point_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_fallback_body_name: str = "gripper_tcp",
    asset_offset_in_body_frame: bool = True,
    target_offset_in_target_frame: bool = True,
) -> torch.Tensor:
    """Sparse +1 bonus when two configured points are close enough."""
    distance = cheat_body_point_to_asset_point_distance_l2(
        env,
        asset_cfg=asset_cfg,
        target_cfg=target_cfg,
        asset_point_offset=asset_point_offset,
        target_point_offset=target_point_offset,
        asset_fallback_body_name=asset_fallback_body_name,
        asset_offset_in_body_frame=asset_offset_in_body_frame,
        target_offset_in_target_frame=target_offset_in_target_frame,
    )
    return (distance < threshold).float()


def cheat_body_point_orientation_error_to_asset_point(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    asset_point_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_point_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_fallback_body_name: str = "gripper_tcp",
) -> torch.Tensor:
    """Penalize orientation error between point frames attached to a robot body and target asset."""
    asset: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    asset_point_quat_w = _body_point_quat_w(
        asset,
        asset_cfg,
        asset_point_rpy,
        asset_fallback_body_name,
    )
    target_point_quat_w = _asset_point_quat_w(target, target_cfg, target_point_rpy)
    return quat_error_magnitude(asset_point_quat_w, target_point_quat_w)


def cheat_distance_to_nic_card_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    nic_card_cfg: SceneEntityCfg = SceneEntityCfg("nic_card"),
    nic_card_offset: tuple[float, float, float] = (0.0, 0.0, 0.05),
    asset_fallback_body_name: str = "gripper_tcp",
) -> torch.Tensor:
    """Penalize ground-truth distance from the end-effector to the NIC card/port pose."""
    return cheat_distance_to_asset_l2(
        env,
        asset_cfg=asset_cfg,
        target_cfg=nic_card_cfg,
        asset_fallback_body_name=asset_fallback_body_name,
        target_fallback_offset=nic_card_offset,
    )


def cheat_orientation_error_to_asset(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    asset_fallback_body_name: str = "gripper_tcp",
    z_rot_offset_deg: float = 0.0,
) -> torch.Tensor:
    """Penalize ground-truth orientation error from an articulation body to a target asset/body."""
    asset: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    body_id = _first_body_id(asset, asset_cfg, asset_fallback_body_name)
    body_quat_w = asset.data.body_quat_w[:, body_id]
    target_quat_w = _target_quat_w(target, target_cfg)
    if z_rot_offset_deg != 0.0:
        target_quat_w = quat_mul(
            target_quat_w,
            _z_axis_quat(target_quat_w.shape[0], z_rot_offset_deg, dtype=target_quat_w.dtype, device=target_quat_w.device),
        )
    return quat_error_magnitude(body_quat_w, target_quat_w)


def cheat_ee_orientation_error_to_nic(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    nic_card_cfg: SceneEntityCfg = SceneEntityCfg("nic_card"),
    z_rot_offset_deg: float = 180.0,
    asset_fallback_body_name: str = "gripper_tcp",
) -> torch.Tensor:
    """Penalize ground-truth misalignment between the end-effector and NIC card/port orientation."""
    return cheat_orientation_error_to_asset(
        env,
        asset_cfg=asset_cfg,
        target_cfg=nic_card_cfg,
        asset_fallback_body_name=asset_fallback_body_name,
        z_rot_offset_deg=z_rot_offset_deg,
    )
