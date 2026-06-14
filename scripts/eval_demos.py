# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained ACT policy on AIC-Port-Insertion-v0.

Closed-loop rollout under the same env the dataset was collected from:
the policy drives the robot, the env reports success / failure /
timeout via the named termination terms. Reports an aggregate success
rate and (optionally) writes one mp4 per rollout under ``outputs/eval/``
so you can scrub through them on the host.

Modes:

- **Default**: headless, fast batched rollouts. Per-episode outcomes
  printed; sample mp4s saved with ``--save_videos``.
- **``--gui``**: drops ``--headless`` so the Isaac Sim viewport opens
  via X11 and you watch the policy drive the robot live. Requires
  ``DISPLAY`` set inside the container (already true:1 in our setup).

Example (headless, 20 rollouts, save videos):

    docker exec -w /workspace/isaaclab isaac-lab-base \\
      ./isaaclab.sh -p etf_robotics_aic/scripts/eval_demos.py \\
      --headless --enable_cameras \\
      --n_episodes 20 --save_videos

Example (live GUI, watch the policy run):

    docker exec -w /workspace/isaaclab isaac-lab-base \\
      ./isaaclab.sh -p etf_robotics_aic/scripts/eval_demos.py \\
      --enable_cameras --gui --n_episodes 5
"""

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

_DEFAULT_CKPT = str(
    _SCRIPT_DIR.parent
    / "outputs" / "train" / "act_phase_port_insertion"
    / "checkpoints" / "100000" / "pretrained_model"
)
_DEFAULT_EVAL_DIR = str(_SCRIPT_DIR.parent / "outputs" / "eval")

# Order MUST match scripts/il/writer.py's policy-group iteration order at
# collection time. If you change the obs schema there, mirror it here.
_STATE_KEYS = (
    "joint_pos", "joint_vel", "joint_torque",
    "tcp_pos_b", "tcp_quat_b",
    "eef_pos_b", "eef_quat_b",
    "tcp_lin_vel_b", "tcp_ang_vel_b",
    "eef_lin_vel_b", "eef_ang_vel_b",
    "wrist_wrench",
    "actions",
)
_IMAGE_BINDINGS = (
    # (lerobot feature, isaac obs key)
    ("observation.images.center", "center_camera_rgb"),
    ("observation.images.left", "left_camera_rgb"),
    ("observation.images.right", "right_camera_rgb"),
)

parser = argparse.ArgumentParser(description="Closed-loop eval of a trained ACT policy.")
parser.add_argument("--ckpt", type=str, default=_DEFAULT_CKPT,
                    help="Path to a `pretrained_model/` dir from training.")
parser.add_argument("--task", type=str, default="AIC-Port-Insertion-v0")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Parallel envs. >1 is faster but the ACT action queue is shared, "
                         "so per-env temporal consistency suffers slightly on async resets. "
                         "Keep at 1 for clean per-rollout metrics; raise for throughput.")
parser.add_argument("--n_episodes", type=int, default=20)
parser.add_argument("--gui", action="store_true",
                    help="Skip --headless so the Isaac Sim viewport opens (uses container DISPLAY).")
parser.add_argument("--save_videos", action="store_true",
                    help="Write one mp4 per episode under --eval_dir.")
parser.add_argument("--eval_dir", type=str, default=_DEFAULT_EVAL_DIR,
                    help="Where to put videos / metrics. Each invocation makes a fresh NNN_<ts>/ subdir.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--temporal_ensemble_coeff", type=float, default=0.01,
                    help="ACT temporal-ensembling coefficient (original ACT uses 0.01). When >0 the "
                         "policy re-queries the network every step and blends the overlapping chunk "
                         "predictions (exp-weighted, wᵢ=exp(-coeff*i)) — this is the reactive, closed-loop "
                         "ACT mode and the default here. It forces n_action_steps=1. Pass 0 (or negative) "
                         "to fall back to the checkpoint's own n_action_steps (open-loop action chunking).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# `--gui` is a convenience: drop the --headless launcher arg if the user asked for it.
if args_cli.gui:
    args_cli.headless = False
if not getattr(args_cli, "enable_cameras", False):
    # The policy needs the three RGB cameras; without --enable_cameras the
    # sensors fail to initialize and obs construction errors out.
    print("[eval]: forcing --enable_cameras (policy reads three RGB streams).", file=sys.stderr)
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import aic_task.tasks  # noqa: F401

from il.env_wrapper import PortInsertionEnv

# Importing this only registers the class; the factory monkey-patch isn't
# triggered (it lives inside main() in train_demos.py).
from train_demos import ACTPolicyWithPhaseHead

from lerobot.processor.pipeline import DataProcessorPipeline
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)


def _isaac_to_lerobot_batch(obs: dict, device: torch.device | str) -> dict:
    """Convert one Isaac obs step into a single LeRobot inference batch.

    Isaac returns ``obs["policy"][<term>]`` with shape ``(N, ...)`` per term.
    LeRobot's preprocessor expects:

    - ``observation.state``  : ``(N, 56)`` float32 (concatenation of the
      non-image policy terms in the writer-time order).
    - ``observation.images.X``: ``(N, 3, 224, 224)`` float32 in ``[0, 1]``
      (Isaac gives uint8 NHWC; we permute + scale).
    """
    policy = obs["policy"]
    state = torch.cat([policy[k] for k in _STATE_KEYS], dim=-1).float()
    batch = {"observation.state": state}
    for feat, src in _IMAGE_BINDINGS:
        img = policy[src]  # (N, H, W, C)
        img = img.permute(0, 3, 1, 2).contiguous()  # (N, C, H, W)
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        else:
            img = img.float()
        batch[feat] = img
    return batch


def _make_eval_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(d for d in root.iterdir() if d.is_dir() and d.name[:3].isdigit())
    next_idx = (int(existing[-1].name[:3]) + 1) if existing else 1
    run_dir = root / f"{next_idx:03d}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir()
    return run_dir


def _chown_to_host(path: Path, uid: int = 1000, gid: int = 1000) -> None:
    try:
        os.chown(path, uid, gid)
        if path.is_dir():
            for p in path.rglob("*"):
                try:
                    os.chown(p, uid, gid)
                except (PermissionError, FileNotFoundError):
                    pass
    except (PermissionError, FileNotFoundError):
        pass


def main() -> None:
    env = PortInsertionEnv.make(
        task=args_cli.task,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
    )
    device = env.device

    print(f"[eval]: loading policy from {args_cli.ckpt}", file=sys.stderr)
    policy = ACTPolicyWithPhaseHead.from_pretrained(args_cli.ckpt)
    # Default to closed-loop temporal ensembling regardless of how the checkpoint
    # was trained. The original ACT inference re-queries the net every step and
    # blends overlapping chunk predictions — far more robust for contact-rich
    # insertion than committing to a full chunk open-loop. Build the ensembler
    # even if the ckpt's config carried `temporal_ensemble_coeff=None`.
    if args_cli.temporal_ensemble_coeff and args_cli.temporal_ensemble_coeff > 0:
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        policy.config.temporal_ensemble_coeff = args_cli.temporal_ensemble_coeff
        policy.config.n_action_steps = 1
        policy.temporal_ensembler = ACTTemporalEnsembler(
            args_cli.temporal_ensemble_coeff, policy.config.chunk_size
        )
        print(f"[eval]: temporal ensembling ON (coeff={args_cli.temporal_ensemble_coeff}, "
              f"n_action_steps=1, chunk_size={policy.config.chunk_size}).", file=sys.stderr)
    else:
        print(f"[eval]: temporal ensembling OFF — open-loop chunking with "
              f"n_action_steps={policy.config.n_action_steps}.", file=sys.stderr)
    policy.to(device).eval()

    preprocessor = DataProcessorPipeline.from_pretrained(
        args_cli.ckpt, config_filename="policy_preprocessor.json"
    )
    # The action postprocessor takes a Tensor in / out (not a dict), so we
    # have to tell the pipeline how to wrap and unwrap. Without these the
    # default `batch_to_transition` is used and chokes on a raw Tensor.
    postprocessor = DataProcessorPipeline.from_pretrained(
        args_cli.ckpt,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    eval_dir = _make_eval_dir(Path(args_cli.eval_dir))
    print(f"[eval]: outputs at {eval_dir}", file=sys.stderr)

    obs, _ = env.reset(seed=args_cli.seed)
    policy.reset()

    # Per-env step counters and (if --save_videos) frame buffers.
    step_counts = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    video_buffers: list[list[np.ndarray]] = [[] for _ in range(env.num_envs)]

    # Aggregate metrics.
    n_total = 0
    n_success = 0
    n_failed = 0
    n_timeout = 0
    ep_lengths: list[int] = []

    target = args_cli.n_episodes

    while n_total < target and simulation_app.is_running():
        with torch.inference_mode():
            batch = _isaac_to_lerobot_batch(obs, device)
            batch = preprocessor(batch)
            action = policy.select_action(batch)
            action = postprocessor(action)
            if args_cli.save_videos:
                # Snapshot the center camera frame for each env (HWC uint8 host copy).
                centers = obs["policy"]["center_camera_rgb"].detach().cpu().numpy()
                for i in range(env.num_envs):
                    video_buffers[i].append(centers[i])
            obs, _, terminated, truncated, _ = env.step(action)
            step_counts += 1

            done = (terminated | truncated).nonzero(as_tuple=False).flatten()
            if done.numel() == 0:
                continue

            success_mask = env.unwrapped.termination_manager.get_term("success")
            failed_mask = env.unwrapped.termination_manager.get_term("failed_stationary")
            for eid in done.tolist():
                if n_total >= target:
                    break
                length = int(step_counts[eid].item())
                if success_mask[eid].item():
                    outcome, n_success = "success", n_success + 1
                elif failed_mask[eid].item():
                    outcome, n_failed = "failed_stationary", n_failed + 1
                else:
                    outcome, n_timeout = "time_out", n_timeout + 1
                n_total += 1
                ep_lengths.append(length)
                print(
                    f"[eval] ep {n_total}/{target}  env={eid}  outcome={outcome:<18}  len={length}",
                    file=sys.stderr,
                )
                if args_cli.save_videos and video_buffers[eid]:
                    fpath = eval_dir / f"episode_{n_total - 1:03d}_env{eid}_{outcome}.mp4"
                    fps = max(1, round(1.0 / env.policy_dt))
                    import imageio.v3 as iio
                    iio.imwrite(fpath, np.stack(video_buffers[eid]), fps=fps, codec="libx264")
                step_counts[eid] = 0
                video_buffers[eid] = []

            # ACT's action queue is shared across batch. Reset whenever any
            # env resets so the chunk re-starts cleanly from the new obs.
            policy.reset()

    sr = n_success / max(n_total, 1)
    mean_len = sum(ep_lengths) / max(len(ep_lengths), 1)
    summary = (
        f"\n{'=' * 60}\n"
        f"Eval over {n_total} episodes (ckpt={args_cli.ckpt}):\n"
        f"  success rate:        {n_success}/{n_total}  ({sr * 100:.1f}%)\n"
        f"  failed_stationary:   {n_failed}/{n_total}  ({n_failed / max(n_total, 1) * 100:.1f}%)\n"
        f"  time_out:            {n_timeout}/{n_total}  ({n_timeout / max(n_total, 1) * 100:.1f}%)\n"
        f"  mean episode length: {mean_len:.1f} steps  (fps={1.0 / env.policy_dt:.1f})\n"
        f"{'=' * 60}\n"
    )
    print(summary)
    (eval_dir / "summary.txt").write_text(summary)

    _chown_to_host(eval_dir)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
