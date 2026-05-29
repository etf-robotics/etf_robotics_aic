"""Abstract base class for scripted expert path planners."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from isaaclab.envs import ManagerBasedRLEnv


class PathPlanner(ABC):
    """Drives a `ManagerBasedRLEnv` toward a task goal via the env's action term.

    Subclasses implement `reset` (per-env trajectory init) and `act` (per-step
    action). The base reads the DiffIK action term's `scale` so subclasses can
    cap per-step pose deltas against the same constants the controller uses.
    """

    def __init__(self, env: ManagerBasedRLEnv, action_name: str = "arm_action"):
        self.env = env
        self.action_name = action_name
        self.device = env.device
        self.num_envs = env.num_envs

        scale = env.action_manager.get_term(action_name).cfg.scale
        if len(scale) != 6:
            raise ValueError(
                f"PathPlanner expects a 6D pose action (pos xyz + rotvec xyz); "
                f"action '{action_name}' has scale of length {len(scale)}."
            )
        scale_t = torch.tensor(scale, device=self.device, dtype=torch.float32)
        self._pos_scale = scale_t[:3]
        self._rot_scale = scale_t[3:]

    @property
    def pos_scale(self) -> torch.Tensor:
        """Per-step position delta at `raw_action = 1.0`, shape (3,), meters."""
        return self._pos_scale

    @property
    def rot_scale(self) -> torch.Tensor:
        """Per-step rotvec delta at `raw_action = 1.0`, shape (3,), radians."""
        return self._rot_scale

    @property
    def policy_dt(self) -> float:
        """Seconds between policy steps (`sim.dt * decimation`)."""
        return self.env.cfg.sim.dt * self.env.cfg.decimation

    @abstractmethod
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Initialize per-env planner state. `None` means all envs."""

    @abstractmethod
    def act(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return `(action, info)`.

        `action` has shape `(num_envs, action_dim)`. `info` should at minimum
        contain `subtask_id` as an int tensor of shape `(num_envs,)` so
        downstream recorders can write MimicGen-compatible labels.
        """
