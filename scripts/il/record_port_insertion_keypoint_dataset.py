# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record AIC port-insertion demonstrations with a scripted visual oracle.

The oracle uses USD-resolved port frames and projected vision keypoints to
choose its insertion phase, but neither the keypoints nor the privileged
oracle state are stored. Isaac Lab's ``ActionStateRecorderManagerCfg``
serialises the env's observation dict (RGB + proprio) and the env-applied
6-D DiffIK action — exactly what ``IsaacLabPolicy`` will see at inference.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record AIC port-insertion demonstrations.")
parser.add_argument("--task", type=str, default="AIC-Port-Insertion-v0")
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/port_insertion.hdf5",
    help="Output HDF5 file. Name it <module>__<port>__<plug>.hdf5 to match the IsaacLabPolicy dispatch.",
)
parser.add_argument("--num_demos", type=int, default=10, help="0 = infinite.")
parser.add_argument("--max_episode_steps", type=int, default=900)
parser.add_argument("--step_hz", type=int, default=30)
parser.add_argument("--settle_seconds", type=float, default=1.0)
# Oracle gains/limits. Deltas chosen so raw = processed / scale stays in [-1, 1]
# for the env's per-axis scale (0.015 m, 0.025 rad).
parser.add_argument("--pos_gain", type=float, default=0.7)
parser.add_argument("--rot_gain", type=float, default=0.45)
parser.add_argument("--max_pos_delta", type=float, default=0.012)
parser.add_argument("--max_rot_delta", type=float, default=0.045)
parser.add_argument("--insert_max_pos_delta", type=float, default=0.0015)
parser.add_argument("--insert_max_rot_delta", type=float, default=0.01)
# Success: stationary-gripper window + tail to record
parser.add_argument("--success_stationary_distance", type=float, default=0.01)
parser.add_argument("--success_stationary_seconds", type=float, default=1.0)
parser.add_argument("--success_record_tail_seconds", type=float, default=0.0)
# Oracle target tuning
parser.add_argument(
    "--target_offset_port_frame",
    type=float,
    nargs=3,
    default=(-0.001, 0.003, 0.0),
    metavar=("DX", "DY", "DZ"),
)
parser.add_argument("--target_roll_offset_deg", type=float, default=-13.0)
# Misc
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--stream", action="store_true", default=False)
parser.add_argument("--log_every", type=int, default=50, help="0 disables.")
parser.set_defaults(use_fabric=True)
parser.add_argument("--disable_fabric", action="store_false", dest="use_fabric")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import math
import time

import gymnasium as gym
import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import aic_task.tasks  # noqa: F401
from aic_task.controllers.port_insertion_oracle import (
    apply_insertion_phase_gate,
    compute_port_insertion_oracle,
    get_action_scale,
)
from aic_task.vision import compute_port_keypoint_labels, make_default_port_keypoint_layout


TARGET_NAME = "nic_card"
PORT_NAME = "sfp_port_0"
TCP_BODY = "gripper_tcp"
PLUG_CENTER_BODY = "sfp_module_link"
PLUG_TIP_BODY = "sfp_tip_link"
CAMERA_NAMES = ("left_camera", "center_camera", "right_camera")


class RateLimiter:
    """Enforce a target loop rate by polling and rendering."""

    def __init__(self, hz: int):
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env) -> None:
        next_wakeup = self.last_time + self.sleep_duration
        while time.time() < next_wakeup:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time += self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def main() -> None:
    if args_cli.max_episode_steps <= 0:
        raise ValueError(f"--max_episode_steps must be positive, got {args_cli.max_episode_steps}.")

    output_dir = os.path.dirname(args_cli.dataset_file) or "."
    output_file = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Created output directory: {output_dir}")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=args_cli.use_fabric,
    )
    env_cfg.env_name = args_cli.task.split(":")[-1]
    # Our stationary-gripper success drives the recorder; let the env-side
    # terms not auto-reset, so we get the success tail in the demo.
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False
    _disable_rewards(env_cfg)

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if args_cli.stream:
        from aic_task.utils.live_camera_stream import attach_default_camera_stream
        attach_default_camera_stream(env)

    layout = make_default_port_keypoint_layout(
        port_name=PORT_NAME,
        use_usd_geometry=True,
    )
    action_scale = get_action_scale(env, env.action_space.shape[-1])
    rate_limiter = RateLimiter(args_cli.step_hz)

    print(f"[INFO] Recording {args_cli.task} -> {args_cli.dataset_file}")
    print(f"[INFO] Action scale: {action_scale[0].cpu().tolist()}")
    print(
        f"[INFO] Success: {TCP_BODY} stationary within "
        f"{args_cli.success_stationary_distance:.3f} m for "
        f"{args_cli.success_stationary_seconds:.1f} s + "
        f"{args_cli.success_record_tail_seconds:.1f} s tail"
    )

    env.sim.reset()
    env.reset()
    recorded = 0
    try:
        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while simulation_app.is_running():
                if args_cli.num_demos > 0 and recorded >= args_cli.num_demos:
                    break
                
                success_map = _run_episode(env, layout, action_scale, rate_limiter)
                
                # Export only successful env_ids
                successful_env_ids = [eid for eid, success in success_map.items() if success]
                if successful_env_ids:
                    env.recorder_manager.record_pre_reset(successful_env_ids, force_export_or_skip=False)
                    success_tensor = torch.ones(
                        (len(successful_env_ids), 1), dtype=torch.bool, device=env.device
                    )
                    env.recorder_manager.set_success_to_episodes(successful_env_ids, success_tensor)
                    env.recorder_manager.export_episodes(successful_env_ids)
                    # Get total recorded across all env_ids
                    recorded = sum(env.recorder_manager.exported_successful_episode_count.values()) \
                        if isinstance(env.recorder_manager.exported_successful_episode_count, dict) \
                        else env.recorder_manager.exported_successful_episode_count
                    num_successful = len(successful_env_ids)
                    print(f"[INFO] Recorded {recorded}/{args_cli.num_demos} demos ({num_successful} envs succeeded).")
                
                # Report failures
                failed_env_ids = [eid for eid, success in success_map.items() if not success]
                if failed_env_ids:
                    print(f"[INFO] Timeout — discarding {len(failed_env_ids)} attempt(s).")
                
                # Reset failed envs (or all if needed)
                if failed_env_ids:
                    env.recorder_manager.reset(failed_env_ids)
                
                # Global sim reset and env reset
                env.sim.reset()
                env.reset()
                
                if env.sim.is_stopped():
                    break
    finally:
        env.close()
        print(f"[INFO] Done. Dataset: {args_cli.dataset_file}")


def _run_episode(env, layout, action_scale, rate_limiter) -> dict[int, bool]:
    """Run one episode step per environment in parallel.
    
    Returns:
        dict[int, bool]: Success status for each env_id.
    """
    warmup = torch.zeros(env.action_space.shape, dtype=torch.float32, device=env.device)
    env.step(warmup)
    _settle_episode(env, warmup, rate_limiter, seconds=args_cli.settle_seconds)

    step_dt = _env_step_dt(env)
    required_stable = max(1, int(math.ceil(args_cli.success_stationary_seconds / step_dt)))
    record_tail = max(0, int(math.ceil(args_cli.success_record_tail_seconds / step_dt)))

    # Per-env tracking (batched)
    num_envs = env.num_envs
    stationary_steps = torch.zeros(num_envs, dtype=torch.int32, device=env.device)
    stationary_anchor = torch.zeros((num_envs, 3), dtype=torch.float32, device=env.device)
    has_anchor = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    finished = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    for step in range(args_cli.max_episode_steps):
        if finished.all():
            break

        # Compute labels and oracle for all environments
        labels = compute_port_keypoint_labels(env, CAMERA_NAMES, layout, target_name=TARGET_NAME)
        oracle = compute_port_insertion_oracle(
            env,
            action_scale,
            labels=labels,
            layout=layout,
            plug_center_body=PLUG_CENTER_BODY,
            plug_tip_body=PLUG_TIP_BODY,
            tcp_body=TCP_BODY,
            target_name=TARGET_NAME,
            port_name=PORT_NAME,
            pos_gain=args_cli.pos_gain,
            rot_gain=args_cli.rot_gain,
            max_pos_delta=args_cli.max_pos_delta,
            max_rot_delta=args_cli.max_rot_delta,
            insert_max_pos_delta=args_cli.insert_max_pos_delta,
            insert_max_rot_delta=args_cli.insert_max_rot_delta,
            target_offset_port_frame=tuple(args_cli.target_offset_port_frame),
            target_roll_offset=math.radians(args_cli.target_roll_offset_deg),
            force_phase_backoff=False,
        )
        action = apply_insertion_phase_gate(oracle.raw_action, oracle.phase)

        # Log only first env
        if args_cli.log_every > 0 and step % args_cli.log_every == 0:
            phase_str = str(oracle.phase[0]) if isinstance(oracle.phase, (list, tuple)) else str(oracle.phase)
            print(
                f"[INFO] step={step:04d} phase={phase_str} "
                f"tip_err={float(oracle.tip_to_target[0]):.4f} m "
                f"axis_err={float(torch.rad2deg(oracle.axis_error[0])):.1f}° "
                f"stable={stationary_steps[0]}/{required_stable} "
                f"|a|={float(torch.linalg.norm(action[0])):.3f} "
                f"active_envs={(~finished).sum()}"
            )

        env.step(action)

        # Per-env TCP position tracking
        tcp_after = _body_pos(env, TCP_BODY, None)  # (num_envs, 3)

        for env_id in range(num_envs):
            if finished[env_id]:
                continue

            if not has_anchor[env_id]:
                stationary_anchor[env_id] = tcp_after[env_id].clone()
                stationary_steps[env_id] = 1
                has_anchor[env_id] = True
            elif torch.linalg.norm(tcp_after[env_id] - stationary_anchor[env_id]) > args_cli.success_stationary_distance:
                stationary_anchor[env_id] = tcp_after[env_id].clone()
                stationary_steps[env_id] = 1
            else:
                stationary_steps[env_id] += 1

            if stationary_steps[env_id] >= required_stable + record_tail:
                finished[env_id] = True

        if env.sim.is_stopped():
            break

        rate_limiter.sleep(env)

    # Return success map indexed by env_id
    success_map = {env_id: bool(finished[env_id]) for env_id in range(num_envs)}
    return success_map


def _settle_episode(env, action, rate_limiter, *, seconds):
    steps = max(0, int(round(seconds * args_cli.step_hz)))
    for _ in range(steps):
        env.step(action)
        if env.sim.is_stopped():
            break
        rate_limiter.sleep(env)


def _env_step_dt(env):
    step_dt = getattr(env, "step_dt", None)
    if step_dt is not None and float(step_dt) > 0.0:
        return float(step_dt)
    return 1.0 / max(1, args_cli.step_hz)


def _disable_rewards(env_cfg):
    for name, value in vars(env_cfg.rewards).items():
        if name.startswith("_") or value is None:
            continue
        setattr(env_cfg.rewards, name, None)


def _body_pos(env, body_name, env_index):
    """Return TCP position. If env_index is None, return all envs (num_envs, 3)."""
    robot = env.scene["robot"]
    body_ids = robot.find_bodies(body_name, preserve_order=True)[0]
    if len(body_ids) == 0:
        available = ", ".join(getattr(robot, "body_names", []))
        raise KeyError(f"Robot body '{body_name}' not found. Available: {available}")
    if env_index is None:
        return robot.data.body_pos_w[:, int(body_ids[0])].clone()  # (num_envs, 3)
    return robot.data.body_pos_w[env_index, int(body_ids[0])].clone()  # (3,)


if __name__ == "__main__":
    main()
    simulation_app.close()
