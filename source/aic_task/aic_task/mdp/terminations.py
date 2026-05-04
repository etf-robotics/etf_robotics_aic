# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable termination functions for AIC manager-based tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude, quat_mul

from .rewards import (
    _asset_point_pos_w,
    _asset_point_quat_w,
    _body_point_pos_w,
    _body_point_quat_w,
    _first_body_id,
    _target_position_w,
    _z_axis_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cheat_body_distance_success(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    distance_threshold: float,
    asset_fallback_body_name: str = "gripper_tcp",
    target_fallback_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Return success from ground-truth distance between a body and target asset/body."""
    asset: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    body_id = _first_body_id(asset, asset_cfg, asset_fallback_body_name)
    body_pos_w = asset.data.body_pos_w[:, body_id, :]
    target_pos_w = _target_position_w(target, target_cfg, target_fallback_offset)
    distance = torch.linalg.norm(body_pos_w - target_pos_w, dim=1)
    return distance < distance_threshold


def cheat_ee_distance_to_port_success(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    port_cfg: SceneEntityCfg,
    distance_threshold: float = 0.5,
    asset_fallback_body_name: str = "gripper_tcp",
) -> torch.Tensor:
    """Return success when the end-effector is close to a port ground-truth pose."""
    return cheat_body_distance_success(
        env,
        asset_cfg=asset_cfg,
        target_cfg=port_cfg,
        distance_threshold=distance_threshold,
        asset_fallback_body_name=asset_fallback_body_name,
    )


def cheat_body_point_to_asset_point_pose_success(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    position_threshold: float,
    orientation_threshold: float,
    asset_point_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_point_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_point_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_point_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_fallback_body_name: str = "gripper_tcp",
    asset_offset_in_body_frame: bool = True,
    target_offset_in_target_frame: bool = True,
) -> torch.Tensor:
    """Return success when two configured point frames match within position and orientation tolerances."""
    asset: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    asset_point_pos_w = _body_point_pos_w(
        asset,
        asset_cfg,
        asset_point_offset,
        asset_fallback_body_name,
        asset_offset_in_body_frame,
    )
    target_point_pos_w = _asset_point_pos_w(
        target,
        target_cfg,
        target_point_offset,
        target_offset_in_target_frame,
    )
    position_error = torch.linalg.norm(asset_point_pos_w - target_point_pos_w, dim=1)

    asset_point_quat_w = _body_point_quat_w(
        asset,
        asset_cfg,
        asset_point_rpy,
        asset_fallback_body_name,
    )
    target_point_quat_w = _asset_point_quat_w(target, target_cfg, target_point_rpy)
    orientation_error = quat_error_magnitude(asset_point_quat_w, target_point_quat_w)

    return (position_error < position_threshold) & (orientation_error < orientation_threshold)


def cheat_sfp_module_above_nic_card_success(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    nic_card_cfg: SceneEntityCfg,
    z_offset: float = 0.10,
    position_threshold: float = 0.05,
    asset_fallback_body_name: str = "sfp_module_visual",
) -> torch.Tensor:
    """Return success when an SFP module is above the NIC card ground-truth position."""
    robot: Articulation = env.scene[asset_cfg.name]
    nic_card: RigidObject = env.scene[nic_card_cfg.name]

    module_id = _first_body_id(robot, asset_cfg, asset_fallback_body_name)
    module_pos_w = robot.data.body_pos_w[:, module_id, :]

    target_pos_w = nic_card.data.root_pos_w.clone()
    target_pos_w[:, 2] += z_offset

    position_error = torch.linalg.norm(module_pos_w - target_pos_w, dim=1)
    return position_error < position_threshold


def cheat_ee_above_port_success(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    port_cfg: SceneEntityCfg,
    port_offset: tuple[float, float, float] = (0.0, 0.0, 0.07),
    xy_threshold: float = 0.5,
    z_threshold: float = 0.5,
    orientation_threshold: float = 1.0,
    z_rot_offset_deg: float = 180.0,
    asset_fallback_body_name: str = "gripper_tcp",
) -> torch.Tensor:
    """Return success when the end-effector is above and aligned to a port ground-truth pose."""
    robot: Articulation = env.scene[asset_cfg.name]
    port: RigidObject = env.scene[port_cfg.name]

    ee_id = _first_body_id(robot, asset_cfg, asset_fallback_body_name)
    ee_pos_w = robot.data.body_pos_w[:, ee_id, :]
    ee_quat_w = robot.data.body_quat_w[:, ee_id]

    offset_w = torch.tensor(port_offset, dtype=port.data.root_pos_w.dtype, device=port.data.root_pos_w.device)
    target_pos_w = port.data.root_pos_w + offset_w

    xy_error = torch.linalg.norm(ee_pos_w[:, :2] - target_pos_w[:, :2], dim=1)
    z_error = torch.abs(ee_pos_w[:, 2] - target_pos_w[:, 2])

    target_quat_w = quat_mul(
        port.data.root_quat_w,
        _z_axis_quat(port.data.root_quat_w.shape[0], z_rot_offset_deg, dtype=port.data.root_quat_w.dtype, device=port.data.root_quat_w.device),
    )
    orientation_error = quat_error_magnitude(ee_quat_w, target_quat_w)

    return (xy_error < xy_threshold) & (z_error < z_threshold) & (orientation_error < orientation_threshold)
