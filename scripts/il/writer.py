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
import sys
import time
from pathlib import Path

import torch

from .env_wrapper import PortInsertionEnv

_POLICY_GROUP = "policy"
_RUN_RE = re.compile(r"^(\d{3})_")


def _log(msg: str) -> None:
    """Stderr, line-flushed. Kit eats stdout on teardown — stderr survives."""
    print(f"[writer]: {msg}", file=sys.stderr, flush=True)


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
            "annotation.phase": {
                # Training-time label, NOT a policy input — at inference there
                # is no planner to provide a phase. One-hot per frame
                # (0=APPROACH, 1=ALIGN, 2=INSERT) so it can drive an auxiliary
                # loss head, dataset filtering, or per-phase loss weighting.
                "dtype": "float32",
                "shape": (3,),
                "names": ["APPROACH", "ALIGN", "INSERT"],
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
            _log(f"appending to existing run {run_dir.name}")
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
            _log(f"creating new run {run_name} at {run_dir} (fps={fps}, state_dim={state_dim})")

        self._run_dir = run_dir
        self._buffers: list[list[dict]] = [[] for _ in range(self._num_envs)]
        self._saved_episodes = 0
        self._saved_frames = 0

    def record(self, obs: dict, action: torch.Tensor, phase: torch.Tensor) -> None:
        """Append ``(s_t, a_t, phase_t)`` for every env to its in-flight buffer.

        ``phase`` is an ``int8/int64`` tensor of shape ``(num_envs,)`` with
        values in ``{0=APPROACH, 1=ALIGN, 2=INSERT}``. It is one-hot encoded
        into a ``float32[3]`` row so a LeRobot policy can consume it as input.
        """
        policy = obs[_POLICY_GROUP]
        state = torch.cat([policy[k] for k in self._state_keys], dim=-1)
        state_np = state.detach().to(dtype=torch.float32, device="cpu").numpy()
        action_np = action.detach().to(dtype=torch.float32, device="cpu").numpy()
        phase_onehot = torch.nn.functional.one_hot(phase.detach().to(torch.long), num_classes=3)
        phase_np = phase_onehot.to(dtype=torch.float32, device="cpu").numpy()
        images_np = {
            feat: policy[k].detach().to(device="cpu").contiguous().numpy()
            for k, feat in self._image_features.items()
        }
        for i in range(self._num_envs):
            frame = {
                "observation.state": state_np[i],
                "annotation.phase": phase_np[i],
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
                n_frames = len(buf)
                for frame in buf:
                    self._dataset.add_frame(frame)
                self._dataset.save_episode()
                self._saved_episodes += 1
                self._saved_frames += n_frames
                _log(
                    f"saved episode {self._saved_episodes - 1} from env {eid} "
                    f"({n_frames} frames) — totals: {self._saved_episodes} eps / "
                    f"{self._saved_frames} frames"
                )
            elif buf:
                _log(f"dropped env {eid} buffer ({len(buf)} frames, success=False)")
            self._buffers[eid] = []

    def close(self) -> None:
        """Drop in-flight buffers and finalize the LeRobot dataset.

        `LeRobotDataset.finalize()` flushes the parquet footers. Without it
        the data/episodes parquet files are left footerless and the dataset
        cannot be reopened.
        """
        dropped = sum(len(b) for b in self._buffers)
        if dropped:
            _log(f"dropping {dropped} in-flight frames across in-progress envs")
        for i in range(self._num_envs):
            self._buffers[i] = []

        _log(f"finalizing dataset (flushing parquet footers)...")
        t0 = time.monotonic()
        self._dataset.finalize()
        dt = time.monotonic() - t0
        _log(
            f"FINALIZED in {dt:.2f}s — {self._saved_episodes} episodes, "
            f"{self._saved_frames} frames at {self._run_dir}"
        )
        _log("dataset is safe to read now.")
