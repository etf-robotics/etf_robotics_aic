# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for the AIC task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _first_body_id(asset: Articulation | RigidObject, asset_cfg: SceneEntityCfg, fallback_body_name: str) -> int:
    """Resolve the first body id even when SceneEntityCfg was not manager-resolved."""
    if not isinstance(asset_cfg.body_ids, slice):
        return asset_cfg.body_ids[0]

    body_names = asset_cfg.body_names
    if body_names is None:
        body_names = [fallback_body_name]
    elif isinstance(body_names, str):
        body_names = [body_names]

    body_ids, _ = asset.find_bodies(body_names, preserve_order=True)
    return body_ids[0]


def ee_distance_to_port_success(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    port_cfg: SceneEntityCfg,
    distance_threshold: float = 0.5,
) -> torch.Tensor:
    """Return success when the end-effector is within ``distance_threshold`` of the port root."""
    robot: Articulation = env.scene[asset_cfg.name]
    port: RigidObject = env.scene[port_cfg.name]

    ee_idx = _first_body_id(robot, asset_cfg, "gripper_tcp")
    ee_pos_w = robot.data.body_pos_w[:, ee_idx, :]
    port_pos_w = port.data.root_pos_w

    distance = torch.linalg.norm(ee_pos_w - port_pos_w, dim=1)
    return distance < distance_threshold


def sfp_module_above_nic_card_success(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    nic_card_cfg: SceneEntityCfg,
    z_offset: float = 0.10,
    position_threshold: float = 0.05,
) -> torch.Tensor:
    """Return success when the SFP module is close to the NIC-card XY position with a Z offset."""
    robot: Articulation = env.scene[asset_cfg.name]
    nic_card: RigidObject = env.scene[nic_card_cfg.name]

    module_idx = _first_body_id(robot, asset_cfg, "sfp_module_visual")
    module_pos_w = robot.data.body_pos_w[:, module_idx, :]

    target_pos_w = nic_card.data.root_pos_w.clone()
    target_pos_w[:, 2] += z_offset

    position_error = torch.linalg.norm(module_pos_w - target_pos_w, dim=1)
    return position_error < position_threshold


def ee_above_port_success(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    port_cfg: SceneEntityCfg,
    port_offset: tuple[float, float, float] = (0.0, 0.0, 0.07),
    xy_threshold: float = 0.5,
    z_threshold: float = 0.5,
    orientation_threshold: float = 1,
    z_rot_offset_deg: float = 180.0,
) -> torch.Tensor:
    """Return success when the end-effector is above a port and roughly aligned.

    The target position is the port root pose plus ``port_offset`` in world axes.
    ``orientation_threshold`` is an angular error in radians.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    port: RigidObject = env.scene[port_cfg.name]

    ee_idx = _first_body_id(robot, asset_cfg, "gripper_tcp")
    ee_pos_w = robot.data.body_pos_w[:, ee_idx, :]
    ee_quat_w = robot.data.body_quat_w[:, ee_idx]

    port_pos_w = port.data.root_pos_w
    port_quat_w = port.data.root_quat_w
    offset_w = torch.tensor(port_offset, dtype=port_pos_w.dtype, device=port_pos_w.device)
    target_pos_w = port_pos_w + offset_w

    xy_error = torch.linalg.norm(ee_pos_w[:, :2] - target_pos_w[:, :2], dim=1)
    z_error = torch.abs(ee_pos_w[:, 2] - target_pos_w[:, 2])

    half_angle = torch.tensor(
        z_rot_offset_deg * torch.pi / 360.0,
        dtype=port_quat_w.dtype,
        device=port_quat_w.device,
    )
    q_correction = torch.stack(
        [
            torch.cos(half_angle),
            torch.zeros_like(half_angle),
            torch.zeros_like(half_angle),
            torch.sin(half_angle),
        ]
    ).unsqueeze(0).expand(port_quat_w.shape[0], -1)
    target_quat_w = quat_mul(port_quat_w, q_correction)
    orientation_error = quat_error_magnitude(ee_quat_w, target_quat_w)

    return (
        (xy_error < xy_threshold)
        & (z_error < z_threshold)
        & (orientation_error < orientation_threshold)
    )
