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
  printed. With ``--save_videos`` each rollout writes two mp4s: a
  third-person ``_overview`` view of the whole robot executing the policy
  (great for a thesis figure) and a ``_cams`` strip of the three policy
  cameras (center|left|right). Pass ``--no_overview`` to skip the extra
  overview render if GPU memory is tight.
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
                    help="Write per-episode mp4s under --eval_dir: a third-person `overview` "
                         "view of the whole robot plus a `cams` strip of the three policy cameras.")
parser.add_argument("--no_overview", action="store_true",
                    help="With --save_videos, skip the extra third-person overview camera "
                         "(only save the policy-camera strip). Use if GPU memory is tight.")
parser.add_argument("--overview_res", type=int, nargs=2, default=(720, 1280),
                    metavar=("H", "W"),
                    help="Overview camera resolution as `H W` (default 720 1280, 16:9 for slides).")
parser.add_argument("--overview_eye", type=float, nargs=3, default=(1.4, 0.2, 0.7),
                    metavar=("X", "Y", "Z"),
                    help="Overview camera eye position, as an offset (m) from each env origin. "
                         "Default is a front 3/4 shot with the whole arm on the left and the task "
                         "board on the right, nothing occluding (tuned from rendered candidates).")
parser.add_argument("--overview_target", type=float, nargs=3, default=(0.0, 0.0, 0.25),
                    metavar=("X", "Y", "Z"),
                    help="Point the overview camera looks at, as an offset (m) from each env origin "
                         "(roughly the robot/board workspace centre).")
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

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

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


def _progress_metrics(obs: dict):
    """Per-env insertion progress from the privileged ``cheatcode`` obs group.

    Returns ``(insertion_fraction, tcp_to_seat_dist_m)``, each shape ``(N,)``,
    or ``(None, None)`` if the cheatcode group / required terms aren't present.

    - ``insertion_fraction``: the env's perpendicular-gated 0..1 progress scalar
      (1.0 = fully seated, 0.0 = at/behind the entrance or laterally off-axis).
    - distance: Euclidean TCP->seat error in the robot base frame, in metres —
      the raw "how far from the hole did it get" number.

    These are read for *reporting only*; they never touch the policy input, so
    eval stays an honest closed-loop test (the policy still sees only ``policy``
    group obs).
    """
    cheat = obs.get("cheatcode")
    if cheat is None or "insertion_fraction" not in cheat or "seat_pos_b" not in cheat:
        return None, None
    frac = cheat["insertion_fraction"].reshape(-1).float()
    seat = cheat["seat_pos_b"]          # (N, 3) base frame
    tcp = obs["policy"]["tcp_pos_b"]    # (N, 3) base frame
    dist = torch.linalg.norm(seat - tcp, dim=-1).float()
    return frac, dist


_OVERVIEW_CAM = "overview_camera"


def _overview_camera_cfg(height: int, width: int, eye, target) -> TiledCameraCfg:
    """A third-person scene camera that frames the whole robot + task board.

    Spawned per-env at ``{ENV_REGEX_NS}/overview_cam`` but kept out of every
    observation group, so it never reaches the policy or the BC dataset — it
    exists purely so eval can render a human-watchable "robot in full" video.

    The look-at pose is baked straight into the spawn ``OffsetCfg`` (computed
    from ``eye``/``target``) rather than set at runtime. That matters: with
    Fabric on (which we keep so the rollout dynamics are identical to a normal
    eval) the renderer reads the camera pose from Fabric, and a post-spawn
    ``set_world_poses_from_view`` (USD-only) would be ignored — the camera would
    render empty sky. ``eye``/``target`` are offsets (m) from each env origin;
    since ``OffsetCfg.pos`` is parent-relative (parent = the env prim) the same
    cfg frames every env identically. The look direction is axis-aligned across
    envs, so one orientation is correct for all of them.

    Focal length is shortened vs. the 22.48 mm policy cameras to widen the FOV
    (~70° horizontal) so the entire arm and board fit in frame.
    """
    # OpenGL convention: create_rotation_matrix_from_view returns an opengl-frame
    # (-Z forward, +Y up) rotation, which is also USD's native camera convention.
    rot = create_rotation_matrix_from_view(
        torch.tensor([eye], dtype=torch.float32),
        torch.tensor([target], dtype=torch.float32),
        up_axis="Z",
    )
    quat = tuple(quat_from_matrix(rot)[0].tolist())
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/overview_cam",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.0,
            focus_distance=0.0,
            horizontal_aperture=20.955,
            vertical_aperture=20.955 * height / width,
            clipping_range=(0.05, 50.0),
        ),
        height=height,
        width=width,
        data_types=["rgb"],
        offset=TiledCameraCfg.OffsetCfg(pos=tuple(eye), rot=quat, convention="opengl"),
    )


def _overview_frames(env) -> np.ndarray:
    """Current overview RGB for every env as host uint8, shape (N, H, W, 3)."""
    rgb = env.unwrapped.scene[_OVERVIEW_CAM].data.output["rgb"]  # (N, H, W, C) uint8
    return rgb[..., :3].detach().cpu().numpy()


def _cams_strip(obs: dict) -> np.ndarray:
    """Center|left|right policy cameras stitched horizontally, (N, H, 3W, 3)."""
    policy = obs["policy"]
    views = [policy[k] for k in ("center_camera_rgb", "left_camera_rgb", "right_camera_rgb")]
    strip = torch.cat(views, dim=2)  # concat along width
    return strip[..., :3].detach().cpu().numpy()


def _write_mp4(fpath: Path, frames: list[np.ndarray], fps: int) -> None:
    import imageio.v3 as iio
    # `-crf 18` is near-visually-lossless and keeps the default yuv420p pixel
    # format (no extra -pix_fmt -> no duplicate-option warning). libx264 needs
    # even dimensions for yuv420p, so crop a stray odd row/column if present.
    arr = np.stack(frames)
    h, w = arr.shape[1], arr.shape[2]
    arr = arr[:, : h - (h % 2), : w - (w % 2)]
    iio.imwrite(fpath, arr, fps=fps, codec="libx264", output_params=["-crf", "18"])


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
    # Add the third-person overview camera only when we're actually saving
    # videos and the user hasn't opted out — it costs an extra render pass.
    want_overview = args_cli.save_videos and not args_cli.no_overview
    extra_sensors = None
    if want_overview:
        extra_sensors = {
            _OVERVIEW_CAM: _overview_camera_cfg(
                *args_cli.overview_res, args_cli.overview_eye, args_cli.overview_target
            )
        }

    # Keep Fabric on (the default) so the rollout dynamics — and therefore the
    # eval metrics — are identical to a normal eval; the overview camera's pose
    # is baked into its spawn offset, so no runtime re-aim is needed.
    env = PortInsertionEnv.make(
        task=args_cli.task,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        extra_sensors=extra_sensors,
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

    # Per-episode metrics, written incrementally (flushed each episode) so a
    # long run still yields usable stats if it's interrupted. One row per
    # rollout; `closest_mm` is blank if the privileged cheatcode obs is absent.
    metrics_csv = (eval_dir / "metrics.csv").open("w")
    metrics_csv.write("episode,env,outcome,length,best_insert,closest_mm\n")
    metrics_csv.flush()

    obs, _ = env.reset(seed=args_cli.seed)
    policy.reset()

    # Per-env step counters and (if --save_videos) frame buffers. Each env gets
    # one buffer per saved view: the third-person `overview` and the policy
    # `cams` strip (center|left|right).
    step_counts = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    overview_buffers: list[list[np.ndarray]] = [[] for _ in range(env.num_envs)]
    cams_buffers: list[list[np.ndarray]] = [[] for _ in range(env.num_envs)]

    # Aggregate metrics.
    n_total = 0
    n_success = 0
    n_failed = 0
    n_timeout = 0
    ep_lengths: list[int] = []
    ep_best_frac: list[float] = []   # per-episode max insertion_fraction reached
    ep_min_dist: list[float] = []    # per-episode min TCP->seat distance (m)

    # Running per-env "closest approach" trackers, frozen into the lists above
    # when each episode ends, then reset for the next rollout in that env.
    best_frac = torch.zeros(env.num_envs, device=device)
    min_dist = torch.full((env.num_envs,), float("inf"), device=device)

    target = args_cli.n_episodes

    while n_total < target and simulation_app.is_running():
        with torch.inference_mode():
            batch = _isaac_to_lerobot_batch(obs, device)
            batch = preprocessor(batch)
            action = policy.select_action(batch)
            action = postprocessor(action)
            # Track closest approach from the pre-step (in-episode) obs.
            frac, dist = _progress_metrics(obs)
            if frac is not None:
                best_frac = torch.maximum(best_frac, frac)
                min_dist = torch.minimum(min_dist, dist)
            if args_cli.save_videos:
                # Snapshot host uint8 frames for each env: the policy-camera
                # strip and (if enabled) the third-person overview.
                cams = _cams_strip(obs)
                overview = _overview_frames(env) if want_overview else None
                for i in range(env.num_envs):
                    cams_buffers[i].append(cams[i])
                    if overview is not None:
                        overview_buffers[i].append(overview[i])
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
                bf = float(best_frac[eid].item())
                md = float(min_dist[eid].item())
                ep_best_frac.append(bf)
                ep_min_dist.append(md)
                md_str = "n/a" if md == float("inf") else f"{md * 1000:.1f}mm"
                print(
                    f"[eval] ep {n_total}/{target}  env={eid}  outcome={outcome:<18}  "
                    f"len={length:<5}  best_insert={bf:.2f}  closest={md_str}",
                    file=sys.stderr,
                )
                metrics_csv.write(
                    f"{n_total - 1},{eid},{outcome},{length},{bf:.4f},"
                    f"{'' if md == float('inf') else f'{md * 1000:.2f}'}\n"
                )
                metrics_csv.flush()
                # Reset closest-approach trackers for this env's next rollout.
                best_frac[eid] = 0.0
                min_dist[eid] = float("inf")
                if args_cli.save_videos and cams_buffers[eid]:
                    fps = max(1, round(1.0 / env.policy_dt))
                    stem = f"episode_{n_total - 1:03d}_env{eid}_{outcome}"
                    _write_mp4(eval_dir / f"{stem}_cams.mp4", cams_buffers[eid], fps)
                    if want_overview and overview_buffers[eid]:
                        _write_mp4(eval_dir / f"{stem}_overview.mp4", overview_buffers[eid], fps)
                step_counts[eid] = 0
                cams_buffers[eid] = []
                overview_buffers[eid] = []

            # ACT's action queue is shared across batch. Reset whenever any
            # env resets so the chunk re-starts cleanly from the new obs.
            policy.reset()

    sr = n_success / max(n_total, 1)
    mean_len = sum(ep_lengths) / max(len(ep_lengths), 1)
    # Continuous "how close did it get" stats — informative even at 0% success.
    progress_lines = ""
    finite_dist = [d for d in ep_min_dist if d != float("inf")]
    if ep_best_frac and finite_dist:
        import statistics
        mean_frac = sum(ep_best_frac) / len(ep_best_frac)
        med_frac = statistics.median(ep_best_frac)
        mean_md = sum(finite_dist) / len(finite_dist)
        best_md = min(finite_dist)
        progress_lines = (
            f"  insertion progress (best, 1.0=seated):  mean {mean_frac:.2f}   median {med_frac:.2f}\n"
            f"  closest TCP->seat distance:             mean {mean_md * 1000:.1f} mm   best {best_md * 1000:.1f} mm\n"
        )
    summary = (
        f"\n{'=' * 60}\n"
        f"Eval over {n_total} episodes (ckpt={args_cli.ckpt}):\n"
        f"  success rate:        {n_success}/{n_total}  ({sr * 100:.1f}%)\n"
        f"  failed_stationary:   {n_failed}/{n_total}  ({n_failed / max(n_total, 1) * 100:.1f}%)\n"
        f"  time_out:            {n_timeout}/{n_total}  ({n_timeout / max(n_total, 1) * 100:.1f}%)\n"
        f"  mean episode length: {mean_len:.1f} steps  (fps={1.0 / env.policy_dt:.1f})\n"
        f"{progress_lines}"
        f"{'=' * 60}\n"
    )
    print(summary)
    (eval_dir / "summary.txt").write_text(summary)
    metrics_csv.close()

    _chown_to_host(eval_dir)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
