# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the port-insertion task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class InsertionGoalReachedSuccess(ManagerTermBase):
    """Stateful success when the insertion tip reaches the commanded goal."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._success_counts = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._success_counts[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        command_name: str = "insertion_goal",
        tip_body: str = "sfp_tip_link",
        position_threshold: float = 0.003,
        orientation_threshold: float = math.radians(4.0),
        required_seconds: float = 0.5,
    ) -> torch.Tensor:
        position_error, orientation_error, _ = _insertion_goal_tip_errors(
            env,
            asset_cfg=asset_cfg,
            command_name=command_name,
            tip_body=tip_body,
        )
        reached = (position_error <= position_threshold) & (orientation_error <= orientation_threshold)
        self._success_counts = torch.where(reached, self._success_counts + 1, torch.zeros_like(self._success_counts))
        required_steps = max(1, int(math.ceil(required_seconds / _env_step_dt(env))))
        return self._success_counts >= required_steps


class InsertionGoalStationaryFailure(ManagerTermBase):
    """Failure when the insertion tip is stationary outside the goal."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._anchor_pos_w = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)
        self._stable_counts = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._anchor_pos_w[env_ids] = 0.0
        self._stable_counts[env_ids] = 0
        self._initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        command_name: str = "insertion_goal",
        tip_body: str = "sfp_tip_link",
        movement_threshold: float = 0.001,
        success_position_threshold: float = 0.003,
        required_seconds: float = 1.0,
    ) -> torch.Tensor:
        position_error, _, tip_pos_w = _insertion_goal_tip_errors(
            env,
            asset_cfg=asset_cfg,
            command_name=command_name,
            tip_body=tip_body,
        )

        uninitialized = ~self._initialized
        self._anchor_pos_w = torch.where(uninitialized.unsqueeze(1), tip_pos_w, self._anchor_pos_w)
        self._initialized = torch.ones_like(self._initialized)

        moved_too_far = torch.linalg.norm(tip_pos_w - self._anchor_pos_w, dim=1) > movement_threshold
        reset_window = uninitialized | moved_too_far
        self._anchor_pos_w = torch.where(reset_window.unsqueeze(1), tip_pos_w, self._anchor_pos_w)

        one_step = torch.ones_like(self._stable_counts)
        self._stable_counts = torch.where(reset_window, one_step, self._stable_counts + 1)

        required_steps = max(1, int(math.ceil(required_seconds / _env_step_dt(env))))
        stationary = self._stable_counts >= required_steps
        outside_goal = position_error > success_position_threshold
        return stationary & outside_goal


def _insertion_goal_tip_errors(
    env: ManagerBasedRLEnv,
    *,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    tip_body: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene[asset_cfg.name]
    goal = env.command_manager.get_term(command_name)
    tip_id = _first_body_id(robot, asset_cfg, tip_body)

    tip_pos_w = robot.data.body_pos_w[:, tip_id, :]
    tip_quat_w = robot.data.body_quat_w[:, tip_id, :]

    position_error = torch.linalg.norm(tip_pos_w - goal.final_tip_pos_w, dim=1)
    orientation_error = quat_error_magnitude(tip_quat_w, goal.target_tip_quat_w)
    return position_error, orientation_error, tip_pos_w


def _first_body_id(asset: Articulation, asset_cfg: SceneEntityCfg, fallback_body_name: str) -> int:
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
    if len(body_ids) == 0:
        available = ", ".join(getattr(asset, "body_names", []))
        raise KeyError(f"Body '{fallback_body_name}' not found. Available bodies: {available}")
    return int(body_ids[0])


def _env_step_dt(env: ManagerBasedRLEnv) -> float:
    """Return manager step time in seconds, with config fallback for tests."""

    step_dt = getattr(env, "step_dt", None)
    if step_dt is not None:
        step_dt = float(step_dt)
        if step_dt > 0.0:
            return step_dt

    cfg = getattr(env, "cfg", None)
    sim_cfg = getattr(cfg, "sim", None)
    sim_dt = getattr(sim_cfg, "dt", None)
    decimation = getattr(cfg, "decimation", None)
    if sim_dt is not None and decimation is not None:
        step_dt = float(sim_dt) * float(decimation)
        if step_dt > 0.0:
            return step_dt

    return 1.0


__all__ = [
    "InsertionGoalReachedSuccess",
    "InsertionGoalStationaryFailure",
]
