# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record visual-oracle port keypoint datasets for perception-first imitation.

This recorder stores the observations the student may use later (RGB cameras
and proprioception) together with privileged labels generated from simulator
ground truth: port keypoints projected into each camera, visibility, depth,
teacher phase, and a gated oracle action.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record AIC visual port keypoint datasets.")
parser.add_argument("--task", type=str, default="AIC-Port-Approach-v0", help="Name of the task.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/visual_port_keypoints.hdf5",
    help="HDF5 file to write.",
)
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to record. 0 = infinite.")
parser.add_argument("--max_episode_steps", type=int, default=350, help="Maximum control steps per episode.")
parser.add_argument("--step_hz", type=int, default=30, help="Control/render loop rate.")
parser.add_argument("--save_every", type=int, default=1, help="Save every N-th control step.")
parser.add_argument("--env_index", type=int, default=0, help="Environment index to serialize.")
parser.add_argument(
    "--camera_names",
    nargs="+",
    default=["left_camera", "center_camera", "right_camera"],
    help="Camera sensor names to record.",
)
parser.set_defaults(enable_depth_labels=True)
parser.add_argument("--no_depth_labels", action="store_false", dest="enable_depth_labels", help="Disable depth labels.")
parser.add_argument(
    "--stream",
    action="store_true",
    default=False,
    help="Attach the browser camera stream while recording.",
)
parser.add_argument("--pos_gain", type=float, default=0.8, help="Oracle position gain.")
parser.add_argument("--rot_gain", type=float, default=0.6, help="Oracle rotation gain.")
parser.add_argument("--max_pos_delta", type=float, default=0.025, help="Max processed position delta.")
parser.add_argument("--max_rot_delta", type=float, default=0.20, help="Max processed rotation delta.")
parser.add_argument("--num_success_steps", type=int, default=12, help="Consecutive HOLD steps to mark success.")
parser.add_argument("--min_depth", type=float, default=0.01, help="Minimum positive camera depth for in-frame labels.")
parser.add_argument(
    "--occlusion_depth_tolerance",
    type=float,
    default=0.015,
    help="Depth mismatch allowed for strict visible labels.",
)
parser.add_argument("--mouth_half_width", type=float, default=0.012, help="Half width of the port mouth keypoint box.")
parser.add_argument(
    "--mouth_half_height",
    type=float,
    default=0.006,
    help="Half height of the port mouth keypoint box.",
)
parser.add_argument("--axis_length", type=float, default=0.025, help="Length of local axis helper keypoints.")
parser.add_argument(
    "--keypoint_offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help="Offset added to all keypoints in the NIC-card frame.",
)
parser.add_argument(
    "--entry_offset",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Override entry_center in the NIC-card frame.",
)
parser.add_argument(
    "--approach_offset",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Override approach_center in the NIC-card frame.",
)
parser.add_argument(
    "--log_every",
    type=int,
    default=50,
    help="Print one recorder debug line every N steps. 0 disables it.",
)
parser.add_argument(
    "--log_projection_details",
    action="store_true",
    default=False,
    help="Print per-camera projected UV/depth ranges in debug logs.",
)
parser.add_argument(
    "--debug_keypoint",
    type=str,
    default="entry_center",
    help="Named keypoint to print in detailed projection logs.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import time

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import aic_task.tasks  # noqa: F401
from aic_task.controllers import (
    TeacherPhase,
    apply_phase_gate,
    choose_preapproach_phase,
    compute_port_approach_oracle,
    get_action_scale,
)
from aic_task.vision import compute_port_keypoint_labels, labels_for_env, make_default_port_keypoint_layout
from aic_task.vision.dataset_writer import PortKeypointDatasetWriter


class RateLimiter:
    """Convenience class for enforcing a target loop rate."""

    def __init__(self, hz: int):
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env) -> None:
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time += self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def main() -> None:
    """Record visual-oracle episodes."""
    if args_cli.save_every <= 0:
        raise ValueError(f"--save_every must be positive, got {args_cli.save_every}.")
    if args_cli.max_episode_steps <= 0:
        raise ValueError(f"--max_episode_steps must be positive, got {args_cli.max_episode_steps}.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=max(1, args_cli.env_index + 1),
        use_fabric=True,
    )
    env_cfg.env_name = args_cli.task.split(":")[-1]
    env_cfg.observations.policy.concatenate_terms = False
    _configure_camera_data_types(env_cfg, args_cli.camera_names, enable_depth=args_cli.enable_depth_labels)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if args_cli.stream:
        from aic_task.utils.live_camera_stream import attach_default_camera_stream

        attach_default_camera_stream(env)
    if args_cli.env_index < 0 or args_cli.env_index >= env.num_envs:
        raise ValueError(f"--env_index must be in [0, {env.num_envs - 1}], got {args_cli.env_index}.")

    layout = make_default_port_keypoint_layout(
        entry_offset=_optional_tuple(args_cli.entry_offset),
        approach_offset=_optional_tuple(args_cli.approach_offset),
        keypoint_offset=tuple(args_cli.keypoint_offset),
        mouth_half_width=args_cli.mouth_half_width,
        mouth_half_height=args_cli.mouth_half_height,
        axis_length=args_cli.axis_length,
    )
    action_scale = get_action_scale(env, env.action_space.shape[-1])
    writer = PortKeypointDatasetWriter(
        args_cli.dataset_file,
        task_name=args_cli.task,
        camera_names=args_cli.camera_names,
        keypoint_names=layout.names,
        phase_names={int(phase): phase.name for phase in TeacherPhase},
        step_hz=args_cli.step_hz,
        env_index=args_cli.env_index,
    )
    rate_limiter = RateLimiter(args_cli.step_hz)

    print(f"[INFO] Recording visual port keypoint dataset for {args_cli.task}")
    print(f"[INFO] Cameras: {', '.join(args_cli.camera_names)}")
    print("[INFO] Camera poses: robot articulation bodies named <camera>_optical")
    print(f"[INFO] Saving to: {args_cli.dataset_file}")
    print(f"[INFO] Keypoints: {', '.join(layout.names)}")

    recorded_episodes = 0
    try:
        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while simulation_app.is_running():
                if args_cli.num_episodes > 0 and recorded_episodes >= args_cli.num_episodes:
                    break
                success = _record_episode(env, writer, layout, action_scale, rate_limiter)
                recorded_episodes += 1
                print(
                    f"[INFO] Episode {recorded_episodes} recorded "
                    f"({'success' if success else 'timeout'}). "
                    f"Total samples: {writer.sample_count}"
                )
    finally:
        writer.close()
        env.close()
        print(f"[INFO] Wrote {writer.episode_count} episodes / {writer.sample_count} samples.")
        print(f"[INFO] Dataset: {writer.file_path}")


def _record_episode(
    env: gym.Env,
    writer: PortKeypointDatasetWriter,
    layout,
    action_scale: torch.Tensor,
    rate_limiter: RateLimiter,
) -> bool:
    env.sim.reset()
    env.reset()
    warmup_action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=env.device)
    env.step(warmup_action)

    writer.start_episode()
    hold_step_count = 0
    success = False

    for step in range(args_cli.max_episode_steps):
        labels = compute_port_keypoint_labels(
            env,
            args_cli.camera_names,
            layout,
            min_depth=args_cli.min_depth,
            occlusion_depth_tolerance=args_cli.occlusion_depth_tolerance,
        )
        oracle = compute_port_approach_oracle(
            env,
            action_scale,
            pos_gain=args_cli.pos_gain,
            rot_gain=args_cli.rot_gain,
            max_pos_delta=args_cli.max_pos_delta,
            max_rot_delta=args_cli.max_rot_delta,
        )
        phase = choose_preapproach_phase(
            labels,
            layout,
            env_index=args_cli.env_index,
            position_error=float(oracle.position_error[args_cli.env_index]),
            orientation_error=float(oracle.orientation_error[args_cli.env_index]),
        )
        action = apply_phase_gate(oracle.raw_action, phase, env_index=args_cli.env_index)
        if env.num_envs > 1:
            selected_action = torch.zeros_like(action)
            selected_action[args_cli.env_index] = action[args_cli.env_index]
            action = selected_action

        if step % args_cli.save_every == 0:
            writer.append(
                frames=_read_camera_frames(env, args_cli.camera_names, args_cli.env_index),
                labels=labels_for_env(labels, args_cli.env_index),
                proprio=_read_proprio(env, args_cli.env_index),
                action=action[args_cli.env_index],
                phase=int(phase),
                oracle=_oracle_to_record(oracle, args_cli.env_index),
            )
        if args_cli.log_every > 0 and step % args_cli.log_every == 0:
            print(
                f"[INFO] step={step:04d} phase={phase.name} "
                f"visible={_visible_keypoint_count(labels, args_cli.env_index)} "
                f"in_frame={_keypoint_count(labels, args_cli.env_index, 'in_frame')} "
                f"pos_err={float(oracle.position_error[args_cli.env_index]):.4f} "
                f"rot_err={float(torch.rad2deg(oracle.orientation_error[args_cli.env_index])):.1f}deg "
                f"action_norm={float(torch.linalg.norm(action[args_cli.env_index])):.4f}"
            )
            if args_cli.log_projection_details:
                print(_projection_debug_line(labels, args_cli.env_index, args_cli.debug_keypoint))

        env.step(action)
        if phase == TeacherPhase.HOLD:
            hold_step_count += 1
        else:
            hold_step_count = 0
        if hold_step_count >= args_cli.num_success_steps:
            success = True
            break
        if env.sim.is_stopped():
            break
        rate_limiter.sleep(env)

    writer.close_episode(success=success)
    return success


def _configure_camera_data_types(env_cfg, camera_names: list[str], *, enable_depth: bool) -> None:
    data_types = ["rgb", "distance_to_image_plane"] if enable_depth else ["rgb"]
    for camera_name in camera_names:
        if hasattr(env_cfg.scene, camera_name):
            getattr(env_cfg.scene, camera_name).data_types = list(data_types)


def _read_camera_frames(env: gym.Env, camera_names: list[str], env_index: int) -> dict[str, np.ndarray]:
    frames = {}
    for camera_name in camera_names:
        camera = env.scene.sensors[camera_name]
        frames[camera_name] = camera.data.output["rgb"][env_index].detach().cpu().numpy()
    return frames


def _read_proprio(env: gym.Env, env_index: int) -> dict[str, torch.Tensor]:
    robot = env.scene["robot"]
    tcp_body_id = robot.find_bodies("gripper_tcp", preserve_order=True)[0][0]
    tcp_pos_w = robot.data.body_pos_w[env_index, tcp_body_id]
    tcp_quat_w = robot.data.body_quat_w[env_index, tcp_body_id]
    return {
        "joint_pos": robot.data.joint_pos[env_index],
        "joint_vel": robot.data.joint_vel[env_index],
        "tcp_pose_w": torch.cat((tcp_pos_w, tcp_quat_w), dim=0),
    }


def _visible_keypoint_count(labels: dict, env_index: int) -> int:
    return _keypoint_count(labels, env_index, "visible")


def _keypoint_count(labels: dict, env_index: int, mask_key: str) -> int:
    return int(
        sum(
            camera_labels[mask_key][env_index].sum().item()
            for camera_labels in labels["cameras"].values()
        )
    )


def _projection_debug_line(labels: dict, env_index: int, debug_keypoint: str) -> str:
    keypoint_names = labels["keypoint_names"]
    if debug_keypoint not in keypoint_names:
        debug_keypoint = keypoint_names[0]
    keypoint_index = keypoint_names.index(debug_keypoint)
    camera_parts = []
    for camera_name, camera_labels in labels["cameras"].items():
        uv = camera_labels["uv"][env_index]
        depth = camera_labels["depth"][env_index]
        keypoint_uv = uv[keypoint_index]
        keypoint_depth = depth[keypoint_index]
        point_camera = camera_labels["points_camera"][env_index, keypoint_index]
        keypoint_front = bool(keypoint_depth > args_cli.min_depth)
        keypoint_in = bool(camera_labels["in_frame"][env_index, keypoint_index])
        keypoint_visible = bool(camera_labels["visible"][env_index, keypoint_index])
        in_front = int((depth > args_cli.min_depth).sum().item())
        in_frame = int(camera_labels["in_frame"][env_index].sum().item())
        camera_parts.append(
            f"{camera_name}:front={in_front} in={in_frame} "
            f"u=[{float(uv[:, 0].min()):.1f},{float(uv[:, 0].max()):.1f}] "
            f"v=[{float(uv[:, 1].min()):.1f},{float(uv[:, 1].max()):.1f}] "
            f"z=[{float(depth.min()):.3f},{float(depth.max()):.3f}] "
            f"{debug_keypoint}:front={keypoint_front} in={keypoint_in} "
            f"vis={keypoint_visible} uv=({float(keypoint_uv[0]):.1f},{float(keypoint_uv[1]):.1f}) "
            f"z={float(keypoint_depth):.3f} "
            f"p_cam=({float(point_camera[0]):.3f},{float(point_camera[1]):.3f},{float(point_camera[2]):.3f})"
        )
    return "[INFO] projection " + " | ".join(camera_parts)


def _optional_tuple(values: list[float] | None) -> tuple[float, float, float] | None:
    if values is None:
        return None
    return (float(values[0]), float(values[1]), float(values[2]))


def _oracle_to_record(oracle, env_index: int) -> dict[str, torch.Tensor]:
    return {
        "raw_action": oracle.raw_action[env_index],
        "processed_action": oracle.processed_action[env_index],
        "position_error": oracle.position_error[env_index],
        "orientation_error": oracle.orientation_error[env_index],
        "desired_tcp_pose_w": torch.cat((oracle.desired_tcp_pos_w[env_index], oracle.desired_tcp_quat_w[env_index])),
        "target_tip_pose_w": torch.cat((oracle.target_tip_pos_w[env_index], oracle.target_tip_quat_w[env_index])),
        "current_tip_pose_w": torch.cat((oracle.current_tip_pos_w[env_index], oracle.current_tip_quat_w[env_index])),
    }


if __name__ == "__main__":
    main()
    simulation_app.close()
