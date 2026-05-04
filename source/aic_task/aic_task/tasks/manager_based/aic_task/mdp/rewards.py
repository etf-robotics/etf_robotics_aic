# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the aic task (UR5e assembly with task board).

Includes:
- Command-tracking rewards with exponential / tanh kernels (inspired by the
  gear-assembly deploy environment).
- A sparse reaching bonus.
- Smoothness and safety penalties (torques, joint acceleration, action rate).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Command-pose tracking (position)
# ---------------------------------------------------------------------------


def position_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def position_command_error_exp(
    env: ManagerBasedRLEnv, sigma: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward position tracking using a Gaussian (exponential) kernel.

    Unlike tanh, this kernel drops off very steeply beyond *sigma*, providing
    almost no gradient far from the target while giving a strong signal
    close-in — ideal for fine insertion tasks.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    dist_sq = torch.sum(torch.square(curr_pos_w - des_pos_w), dim=1)
    return torch.exp(-dist_sq / (sigma**2))


# ---------------------------------------------------------------------------
# Command-pose tracking (orientation)
# ---------------------------------------------------------------------------


def orientation_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize orientation error (shortest-path angular distance in rad)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward orientation tracking using the tanh kernel.

    Maps the angular error through ``1 - tanh(error / std)`` so that perfectly
    aligned orientations yield 1.0.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    ang_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    return 1.0 - torch.tanh(ang_error / std)


# ---------------------------------------------------------------------------
# Sparse reaching bonus
# ---------------------------------------------------------------------------


def ee_reaching_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Sparse +1 bonus when the EE is within *threshold* (m) of the command position."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return (distance < threshold).float()


# ---------------------------------------------------------------------------
# Smoothness / safety penalties
# ---------------------------------------------------------------------------


def joint_torques_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize applied joint torques (L2 squared)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1
    )


def joint_acc_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint accelerations (L2 squared) for smoother motion."""
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
    """Penalize linear acceleration of selected bodies (encourages gentle motion)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1
    )

#DODATO: Distance from end-effector to NIC card

def distance_to_nic_card_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, nic_card_cfg: SceneEntityCfg = SceneEntityCfg("nic_card")
) -> torch.Tensor:
    """Penalize the distance from the end-effector to the NIC card using L2-norm."""
    robot: Articulation = env.scene[asset_cfg.name]
    nic_card: RigidObject = env.scene[nic_card_cfg.name]
    
    ee_idx = robot.find_bodies("gripper_tcp")[0][0]
    ee_pos_w = robot.data.body_pos_w[:, ee_idx, :]  # type: ignore
    # Koristi body_ids ako je specificiran (port), inače fallback na root
    PORT_OFFSET = torch.tensor([0.0, 0.0, 0.05], device=env.device)  # prilagodi
    # body_ids može biti None, slice(None), ili lista/int
    body_ids = nic_card_cfg.body_ids
    if body_ids is not None and not isinstance(body_ids, slice):
        nic_pos_w = nic_card.data.body_pos_w[:, body_ids[0]]  # type: ignore
    else:
        nic_pos_w = nic_card.data.root_pos_w + PORT_OFFSET

    return torch.norm(ee_pos_w - nic_pos_w, dim=1)

def ee_orientation_error_to_nic(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    nic_card_cfg: SceneEntityCfg = SceneEntityCfg("nic_card"),
    z_rot_offset_deg: float = 180.0,  # <- lako menjivo ako ikad treba
) -> torch.Tensor:
    """Penalize misalignment between EE orientation and NIC card port orientation.
    
    Accounts for coordinate frame mismatch between gripper_tcp and NIC body
    by pre-rotating the NIC quaternion by z_rot_offset_deg around its local Z axis.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    nic_card: RigidObject = env.scene[nic_card_cfg.name]

    # EE orijentacija
    ee_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids[0]]  # (N, 4) wxyz

    # NIC orijentacija
    body_ids = nic_card_cfg.body_ids
    if body_ids is not None and not isinstance(body_ids, slice):
        nic_quat_w = nic_card.data.body_quat_w[:, body_ids[0]]
    else:
        nic_quat_w = nic_card.data.root_quat_w

    # --- Korekcija koordinatnog sistema ---
    # 180° oko Z: q = (w=0, x=0, y=0, z=1)
    # Generalno za bilo koji ugao:
    half_angle = torch.tensor(z_rot_offset_deg / 2.0 * torch.pi / 180.0)
    q_correction = torch.tensor(
        [torch.cos(half_angle), 0.0, 0.0, torch.sin(half_angle)],  # wxyz
        dtype=nic_quat_w.dtype,
        device=nic_quat_w.device,
    ).unsqueeze(0).expand(nic_quat_w.shape[0], -1)  # (N, 4)

    # Rotiramo NIC frame: q_corrected = q_nic * q_offset  (local Z rot)
    nic_quat_corrected = quat_mul(nic_quat_w, q_correction)
    # --------------------------------------

    return quat_error_magnitude(ee_quat_w, nic_quat_corrected)