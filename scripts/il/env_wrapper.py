# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thin wrapper around the AIC port-insertion Gym env.

Exposes ``PortInsertionEnv.make()`` and ``tcp_goal_to_action()``. The
wrapper's only per-step job is converting a TCP-frame pose goal into a
clamped 6D delta-pose action for the DiffIK controller. Frame work
(EEF→TCP) lives inside the planner, applied once per plan; the wrapper
does not touch it.
"""

from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import compute_pose_error
from isaaclab_tasks.utils import parse_env_cfg


class PortInsertionEnv:
    """Owns the Gym env and the TCP-goal → action conversion."""

    def __init__(self, gym_env: gym.Env, action_name: str = "arm_action"):
        unwrapped: ManagerBasedRLEnv = gym_env.unwrapped
        scale = unwrapped.action_manager.get_term(action_name).cfg.scale
        if len(scale) != 6:
            raise ValueError(
                f"PortInsertionEnv expects a 6D pose action; action "
                f"'{action_name}' has scale of length {len(scale)}."
            )
        self.gym_env = gym_env
        self.action_name = action_name
        self._scale = torch.tensor(scale, device=unwrapped.device, dtype=torch.float32)

    @classmethod
    def make(
        cls,
        task: str,
        num_envs: int,
        device: str,
        use_fabric: bool = True,
        action_name: str = "arm_action",
        extra_sensors: dict | None = None,
    ) -> "PortInsertionEnv":
        """Build the Gym env.

        ``extra_sensors`` (name -> SensorBaseCfg) are attached to the scene
        cfg *before* construction. They live in the scene but are **not** added
        to any observation group, so the policy never sees them — use this to
        bolt on eval-only sensors (e.g. a third-person overview camera for
        video) without touching the committed task config or the BC dataset
        schema. Caller reads them via ``env.unwrapped.scene[<name>]``.
        """
        cfg = parse_env_cfg(task, device=device, num_envs=num_envs, use_fabric=use_fabric)
        for name, sensor_cfg in (extra_sensors or {}).items():
            setattr(cfg.scene, name, sensor_cfg)
        gym_env = gym.make(task, cfg=cfg)
        return cls(gym_env, action_name=action_name)

    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        return self.gym_env.unwrapped

    @property
    def num_envs(self) -> int:
        return self.unwrapped.num_envs

    @property
    def device(self) -> str:
        return str(self.unwrapped.device)

    @property
    def action_scale(self) -> torch.Tensor:
        """6D action scale (m for pos, rad for rotvec). Shape (6,)."""
        return self._scale

    @property
    def policy_dt(self) -> float:
        """Seconds between policy steps (`sim.dt * decimation`)."""
        cfg = self.unwrapped.cfg
        return cfg.sim.dt * cfg.decimation

    def reset(self, *args, **kwargs):
        return self.gym_env.reset(*args, **kwargs)

    def step(self, action: torch.Tensor):
        return self.gym_env.step(action)

    def close(self) -> None:
        self.gym_env.close()

    def tcp_goal_to_action(
        self,
        tcp_goal_pos_b: torch.Tensor,
        tcp_goal_quat_b: torch.Tensor,
        obs: dict,
    ) -> torch.Tensor:
        """Per-env TCP root-frame pose goal → clamped 6D delta-pose action.

        Args:
            tcp_goal_pos_b: (N, 3) target TCP position in robot root frame.
            tcp_goal_quat_b: (N, 4) target TCP orientation (w, x, y, z).
            obs: full obs dict from ``env.step``/``env.reset``; provides
                current TCP pose via ``obs["policy"]["tcp_pos_b"]`` and
                ``obs["policy"]["tcp_quat_b"]``.

        Returns:
            (N, 6) action in ``[-1, 1]`` per axis, ready for ``env.step``.
        """
        policy = obs["policy"]
        tcp_pos_b = policy["tcp_pos_b"]
        tcp_quat_b = policy["tcp_quat_b"]
        pos_err, rot_err = compute_pose_error(
            tcp_pos_b, tcp_quat_b, tcp_goal_pos_b, tcp_goal_quat_b,
            rot_error_type="axis_angle",
        )
        action = torch.cat([pos_err, rot_err], dim=-1) / self._scale
        return action.clamp(-1.0, 1.0)
