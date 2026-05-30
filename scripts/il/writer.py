# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeRobot dataset writer for the port-insertion demo collector.

Buffers ``(state, action)`` per env each step; on episode end, flushes a
buffer to disk iff the ``success`` termination fired for that env. Buffers
for ``failed_stationary`` / ``time_out`` ends are discarded, and so are
any in-flight buffers at ``close()``.

Output layout under ``root_dir``:

    <root_dir>/
      001_<timestamp>/   # one LeRobot dataset per run
      002_<timestamp>/
      ...

Pass ``append=True`` to reopen the most recent ``NNN_*`` run and continue
incrementing its episode indices instead of creating a fresh run.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import torch

from .env_wrapper import PortInsertionEnv

_POLICY_GROUP = "policy"
_RUN_RE = re.compile(r"^(\d{3})_")


class PortInsertionWriter:
    """Writes successful port-insertion episodes to a LeRobot dataset."""

    def __init__(
        self,
        env: PortInsertionEnv,
        root_dir: str | Path,
        append: bool = False,
        task: str = "AIC-Port-Insertion-v0",
    ):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as e:
            raise ImportError(
                "PortInsertionWriter requires the `lerobot` package "
                "(pip install lerobot)."
            ) from e

        self._env = env
        self._task = task
        self._num_envs = env.num_envs

        policy = env.unwrapped.obs_buf[_POLICY_GROUP]
        self._state_keys: list[str] = [k for k in policy if not k.endswith("_rgb")]
        self._image_keys: list[str] = [k for k in policy if k.endswith("_rgb")]
        if not self._state_keys:
            raise RuntimeError("policy obs group has no non-image terms; cannot build state vector.")

        state_dim = sum(int(policy[k].shape[-1]) for k in self._state_keys)
        action_dim = int(env.gym_env.action_space.shape[-1])
        fps = max(1, round(1.0 / env.policy_dt))

        features: dict[str, dict] = {
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": list(self._state_keys),
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": [f"a{i}" for i in range(action_dim)],
            },
        }
        self._image_features: dict[str, str] = {}
        for k in self._image_keys:
            cam = k.removesuffix("_rgb").removesuffix("_camera")
            feat_name = f"observation.images.{cam}"
            self._image_features[k] = feat_name
            h, w, c = (int(d) for d in policy[k].shape[1:])
            features[feat_name] = {
                "dtype": "video",
                "shape": (h, w, c),
                "names": ["height", "width", "channels"],
            }

        root_dir = Path(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(
            d for d in root_dir.iterdir() if d.is_dir() and _RUN_RE.match(d.name)
        )
        if append and existing:
            run_dir = existing[-1]
            self._dataset = LeRobotDataset(repo_id=run_dir.name, root=run_dir)
            print(f"[writer]: appending to existing run {run_dir.name}")
        else:
            next_idx = int(_RUN_RE.match(existing[-1].name).group(1)) + 1 if existing else 1
            run_name = f"{next_idx:03d}_{time.strftime('%Y%m%d-%H%M%S')}"
            run_dir = root_dir / run_name
            self._dataset = LeRobotDataset.create(
                repo_id=run_name,
                root=run_dir,
                fps=fps,
                features=features,
                use_videos=True,
            )
            print(f"[writer]: creating new run {run_name} at {run_dir} (fps={fps}, state_dim={state_dim})")

        self._buffers: list[list[dict]] = [[] for _ in range(self._num_envs)]

    def record(self, obs: dict, action: torch.Tensor) -> None:
        """Append ``(s_t, a_t)`` for every env to its in-flight buffer."""
        policy = obs[_POLICY_GROUP]
        state = torch.cat([policy[k] for k in self._state_keys], dim=-1)
        state_np = state.detach().to(dtype=torch.float32, device="cpu").numpy()
        action_np = action.detach().to(dtype=torch.float32, device="cpu").numpy()
        images_np = {
            feat: policy[k].detach().to(device="cpu").contiguous().numpy()
            for k, feat in self._image_features.items()
        }
        for i in range(self._num_envs):
            frame = {
                "observation.state": state_np[i],
                "action": action_np[i],
                "task": self._task,
            }
            for feat, arr in images_np.items():
                frame[feat] = arr[i]
            self._buffers[i].append(frame)

    def commit(self, env_ids: torch.Tensor, success_mask: torch.Tensor) -> None:
        """Flush successful envs' buffers as completed episodes; drop the rest."""
        ids: list[int] = env_ids.detach().cpu().tolist()
        success: list[bool] = success_mask.detach().cpu().tolist()
        for eid in ids:
            buf = self._buffers[eid]
            if success[eid] and buf:
                for frame in buf:
                    self._dataset.add_frame(frame)
                self._dataset.save_episode()
            self._buffers[eid] = []

    def close(self) -> None:
        """Drop any in-flight buffers. Successful episodes are already on disk."""
        for i in range(self._num_envs):
            self._buffers[i] = []
