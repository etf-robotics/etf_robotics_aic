# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable observation functions for AIC manager-based tasks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _first_body_id(asset: Articulation | RigidObject, asset_cfg: SceneEntityCfg, fallback_body_name: str) -> int:
    """Resolve a body id even if the manager has not populated ``body_ids`` yet."""
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


def body_to_target_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    target_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_fallback_body_name: str = "gripper_tcp",
) -> torch.Tensor:
    """Vector from an articulation body to a target asset root plus a world-frame offset."""
    asset: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    body_id = _first_body_id(asset, asset_cfg, asset_fallback_body_name)
    body_pos_w = asset.data.body_pos_w[:, body_id, :]
    offset_w = torch.tensor(target_offset, dtype=target.data.root_pos_w.dtype, device=target.data.root_pos_w.device)
    target_pos_w = target.data.root_pos_w + offset_w
    return target_pos_w - body_pos_w


def body_point_to_asset_point_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    asset_point_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_point_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_fallback_body_name: str = "gripper_tcp",
    asset_offset_in_body_frame: bool = True,
    target_offset_in_target_frame: bool = True,
) -> torch.Tensor:
    """Vector from a point on an articulation body to a point on a target asset."""
    asset: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    body_id = _first_body_id(asset, asset_cfg, asset_fallback_body_name)
    body_pos_w = asset.data.body_pos_w[:, body_id, :]

    asset_offset = torch.tensor(asset_point_offset, dtype=body_pos_w.dtype, device=body_pos_w.device).unsqueeze(0)
    if asset_offset_in_body_frame:
        body_quat_w = asset.data.body_quat_w[:, body_id]
        asset_offset = quat_apply(body_quat_w, asset_offset.expand(body_pos_w.shape[0], -1))
    asset_point_w = body_pos_w + asset_offset

    target_offset = torch.tensor(
        target_point_offset,
        dtype=target.data.root_pos_w.dtype,
        device=target.data.root_pos_w.device,
    ).unsqueeze(0)
    if target_offset_in_target_frame:
        target_offset = quat_apply(target.data.root_quat_w, target_offset.expand(target.data.root_pos_w.shape[0], -1))
    target_point_w = target.data.root_pos_w + target_offset

    return target_point_w - asset_point_w


def contact_net_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return contact sensor net forces in world frame, flattened for policy observations."""
    from isaaclab.sensors import ContactSensor

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces_w = contact_sensor.data.net_forces_w
    body_ids = sensor_cfg.body_ids

    if body_ids is None or body_ids == slice(None):
        if getattr(sensor_cfg, "body_names", None) is not None:
            body_names = sensor_cfg.body_names
            names = [body_names] if isinstance(body_names, str) else body_names
            pattern = re.compile(names[0] if len(names) == 1 else "|".join(names))
            body_ids = [i for i, body_name in enumerate(contact_sensor.body_names) if pattern.search(body_name)]
            if body_ids:
                net_forces_w = net_forces_w[:, body_ids, :]
    else:
        net_forces_w = net_forces_w[:, body_ids, :]

    return net_forces_w.reshape(env.num_envs, -1)
