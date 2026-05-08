# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug AIC eye-in-hand camera tracking and simple target projection.

The camera pose source is intentionally fixed: each camera is read from the
robot articulation body named ``<camera_name>_optical``.  This matches the
working path used by the visual keypoint labeler.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug AIC camera body tracking and target projection.")
parser.add_argument("--task", type=str, default="AIC-Port-Approach-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--env_index", type=int, default=0, help="Environment index to inspect.")
parser.add_argument("--max_steps", type=int, default=180, help="Maximum steps to run. 0 = until closed.")
parser.add_argument("--step_hz", type=int, default=30, help="Loop rate.")
parser.add_argument("--log_every", type=int, default=10, help="Print debug info every N steps.")
parser.add_argument("--target_name", type=str, default="nic_card", help="Scene asset to project.")
parser.add_argument(
    "--target_offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help="Target point offset in the target asset frame.",
)
parser.add_argument(
    "--camera_names",
    nargs="+",
    default=["left_camera", "center_camera", "right_camera"],
    help="Camera sensor names to inspect.",
)
parser.add_argument(
    "--controller",
    choices=["oracle", "zero", "random"],
    default="oracle",
    help="Action source. 'oracle' is the existing ground-truth port approach action.",
)
parser.add_argument("--stream", action="store_true", default=False, help="Attach browser camera stream.")
parser.add_argument("--preview_dir", type=str, default=None, help="Write PPM mosaics with projected target crosses.")
parser.add_argument("--preview_every", type=int, default=10, help="Write preview every N steps.")
parser.add_argument("--pos_gain", type=float, default=0.8, help="Oracle position gain.")
parser.add_argument("--rot_gain", type=float, default=0.6, help="Oracle rotation gain.")
parser.add_argument("--max_pos_delta", type=float, default=0.025, help="Oracle max processed position delta.")
parser.add_argument("--max_rot_delta", type=float, default=0.20, help="Oracle max processed rotation delta.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os
import time

import gymnasium as gym
import numpy as np
import torch

import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import aic_task.tasks  # noqa: F401
from aic_task.controllers import compute_port_approach_oracle, get_action_scale


class RateLimiter:
    """Simple wall-clock rate limiter."""

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
    """Run camera tracking diagnostics."""
    if args_cli.log_every <= 0:
        raise ValueError(f"--log_every must be positive, got {args_cli.log_every}.")
    if args_cli.preview_every <= 0:
        raise ValueError(f"--preview_every must be positive, got {args_cli.preview_every}.")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env_cfg.env_name = args_cli.task.split(":")[-1]
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    if args_cli.stream:
        from aic_task.utils.live_camera_stream import attach_default_camera_stream

        attach_default_camera_stream(env)

    if args_cli.env_index < 0 or args_cli.env_index >= env.num_envs:
        raise ValueError(f"--env_index must be in [0, {env.num_envs - 1}], got {args_cli.env_index}.")
    for camera_name in args_cli.camera_names:
        if camera_name not in env.scene.sensors:
            available = ", ".join(env.scene.sensors.keys())
            raise KeyError(f"Camera '{camera_name}' not found. Available sensors: {available}")
        _camera_body_id(env, camera_name)
    _target_asset(env)

    if args_cli.preview_dir is not None:
        os.makedirs(args_cli.preview_dir, exist_ok=True)

    action_scale = None
    if args_cli.controller == "oracle":
        action_scale = get_action_scale(env, env.action_space.shape[-1])

    rate_limiter = RateLimiter(args_cli.step_hz)
    env.sim.reset()
    env.reset()
    env.step(torch.zeros(env.action_space.shape, dtype=torch.float32, device=env.device))

    reference = _capture_reference(env)
    previous_frames = _read_camera_frames(env)

    print("[INFO] Camera tracking diagnostic")
    print(f"[INFO] task={args_cli.task} target={args_cli.target_name} controller={args_cli.controller}")
    print(f"[INFO] cameras={', '.join(args_cli.camera_names)}")
    for camera_name in args_cli.camera_names:
        print(f"[INFO]   {camera_name} body={camera_name}_optical")

    step = 0
    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running():
            action = _compute_action(env, action_scale)
            env.step(action)

            if step % args_cli.log_every == 0:
                frames = _read_camera_frames(env)
                _print_step_debug(env, step, reference, previous_frames, frames)
                previous_frames = frames

            if args_cli.preview_dir is not None and step % args_cli.preview_every == 0:
                frames = _read_camera_frames(env)
                _write_preview(env, frames, step)

            step += 1
            if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                break
            if env.sim.is_stopped():
                break
            rate_limiter.sleep(env)

    env.close()


def _compute_action(env: gym.Env, action_scale: torch.Tensor | None) -> torch.Tensor:
    if args_cli.controller == "zero":
        return torch.zeros(env.action_space.shape, dtype=torch.float32, device=env.device)
    if args_cli.controller == "random":
        return 2.0 * torch.rand(env.action_space.shape, device=env.device) - 1.0
    oracle = compute_port_approach_oracle(
        env,
        action_scale,
        pos_gain=args_cli.pos_gain,
        rot_gain=args_cli.rot_gain,
        max_pos_delta=args_cli.max_pos_delta,
        max_rot_delta=args_cli.max_rot_delta,
    )
    return oracle.raw_action


def _capture_reference(env: gym.Env) -> dict:
    env_index = args_cli.env_index
    tcp_pos_w, tcp_quat_w = _tcp_pose_w(env, env_index)
    reference = {"tcp_pos_w": tcp_pos_w.clone(), "target_pos_w": _target_point_w(env, env_index).clone()}
    for camera_name in args_cli.camera_names:
        camera_pos, _ = _camera_body_pose(env, camera_name, env_index)
        reference[camera_name] = {
            "pos_w": camera_pos.clone(),
            "pos_tcp": _point_in_frame(camera_pos, tcp_pos_w, tcp_quat_w).clone(),
        }
    return reference


def _print_step_debug(
    env: gym.Env,
    step: int,
    reference: dict,
    previous_frames: dict[str, np.ndarray],
    frames: dict[str, np.ndarray],
) -> None:
    env_index = args_cli.env_index
    target_point_w = _target_point_w(env, env_index)
    target_delta = torch.linalg.norm(target_point_w - reference["target_pos_w"]).item()
    tcp_pos_w, tcp_quat_w = _tcp_pose_w(env, env_index)
    tcp_delta = torch.linalg.norm(tcp_pos_w - reference["tcp_pos_w"]).item()
    print(f"[INFO] step={step:04d} target_w={_vec(target_point_w)} target_delta={target_delta:.4f}m")
    print(f"[INFO]   gripper_tcp: tcp_w={_vec(tcp_pos_w)} tcp_delta={tcp_delta:.4f}m")

    for camera_name in args_cli.camera_names:
        camera_pos, camera_quat = _camera_body_pose(env, camera_name, env_index)
        camera_delta = torch.linalg.norm(camera_pos - reference[camera_name]["pos_w"]).item()
        camera_pos_tcp = _point_in_frame(camera_pos, tcp_pos_w, tcp_quat_w)
        relative_delta = torch.linalg.norm(camera_pos_tcp - reference[camera_name]["pos_tcp"]).item()
        rgb_delta = _frame_delta(previous_frames[camera_name], frames[camera_name])
        projection = _project_point(env, camera_name, target_point_w, env_index)
        print(
            f"[INFO]   {camera_name}: cam_w={_vec(camera_pos)} cam_delta={camera_delta:.4f}m "
            f"rgb_delta={rgb_delta:.2f} cam_in_tcp={_vec(camera_pos_tcp)} "
            f"rel_delta={relative_delta:.5f}m uv=({projection['u']:.1f},{projection['v']:.1f}) "
            f"z={projection['z']:.3f} in={projection['in_frame']}"
        )


def _project_point(env: gym.Env, camera_name: str, target_point_w: torch.Tensor, env_index: int) -> dict:
    camera = env.scene.sensors[camera_name]
    camera_pos, camera_quat = _camera_body_pose(env, camera_name, env_index)
    intrinsic = camera.data.intrinsic_matrices[env_index]
    point_camera = math_utils.quat_apply_inverse(
        camera_quat.unsqueeze(0),
        (target_point_w - camera_pos).unsqueeze(0),
    )[0]

    z = float(point_camera[2])
    if abs(z) < 1.0e-9:
        u = float("inf")
        v = float("inf")
    else:
        projected = intrinsic @ point_camera
        u = float(projected[0] / projected[2])
        v = float(projected[1] / projected[2])

    image = camera.data.output["rgb"]
    height = int(image.shape[1] if image.dim() == 4 else image.shape[0])
    width = int(image.shape[2] if image.dim() == 4 else image.shape[1])
    in_frame = bool(z > 0.0 and 0.0 <= u <= width - 1 and 0.0 <= v <= height - 1)
    return {"u": u, "v": v, "z": z, "in_frame": in_frame, "point_camera": point_camera}


def _camera_body_pose(env: gym.Env, camera_name: str, env_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    body_id = _camera_body_id(env, camera_name)
    return robot.data.body_pos_w[env_index, body_id], robot.data.body_quat_w[env_index, body_id]


def _camera_body_id(env: gym.Env, camera_name: str) -> int:
    robot = env.scene["robot"]
    body_name = f"{camera_name}_optical"
    body_ids = robot.find_bodies(body_name, preserve_order=True)[0]
    if len(body_ids) == 0:
        available = ", ".join(getattr(robot, "body_names", []))
        raise KeyError(f"Camera body '{body_name}' not found. Available robot bodies: {available}")
    return int(body_ids[0])


def _target_point_w(env: gym.Env, env_index: int) -> torch.Tensor:
    target = _target_asset(env)
    offset = torch.tensor(args_cli.target_offset, dtype=target.data.root_pos_w.dtype, device=env.device)
    return target.data.root_pos_w[env_index] + math_utils.quat_apply(
        target.data.root_quat_w[env_index].unsqueeze(0),
        offset.unsqueeze(0),
    )[0]


def _target_asset(env: gym.Env):
    try:
        return env.scene[args_cli.target_name]
    except Exception as exc:
        available = ", ".join(_scene_entry_names(env.scene)) or "unknown"
        raise KeyError(
            f"Target '{args_cli.target_name}' not found. Available scene entries: {available}"
        ) from exc


def _scene_entry_names(scene) -> list[str]:
    names: list[str] = []
    for attr_name in ("articulations", "rigid_objects", "sensors", "extras"):
        entries = getattr(scene, attr_name, None)
        if hasattr(entries, "keys"):
            names.extend(str(name) for name in entries.keys())
    return sorted(set(names))


def _tcp_pose_w(env: gym.Env, env_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    tcp_body_id = robot.find_bodies("gripper_tcp", preserve_order=True)[0][0]
    return robot.data.body_pos_w[env_index, tcp_body_id], robot.data.body_quat_w[env_index, tcp_body_id]


def _point_in_frame(point_w: torch.Tensor, frame_pos_w: torch.Tensor, frame_quat_w: torch.Tensor) -> torch.Tensor:
    return math_utils.quat_apply_inverse(
        frame_quat_w.unsqueeze(0),
        (point_w - frame_pos_w).unsqueeze(0),
    )[0]


def _read_camera_frames(env: gym.Env) -> dict[str, np.ndarray]:
    frames = {}
    env_index = args_cli.env_index
    for camera_name in args_cli.camera_names:
        camera = env.scene.sensors[camera_name]
        frames[camera_name] = camera.data.output["rgb"][env_index].detach().cpu().numpy()
    return frames


def _write_preview(env: gym.Env, frames: dict[str, np.ndarray], step: int) -> None:
    env_index = args_cli.env_index
    target_point_w = _target_point_w(env, env_index)
    annotated = []
    for camera_name in args_cli.camera_names:
        frame = frames[camera_name].copy()
        projection = _project_point(env, camera_name, target_point_w, env_index)
        _draw_cross(frame, int(round(projection["u"])), int(round(projection["v"])))
        annotated.append(frame)
    mosaic = np.concatenate(annotated, axis=1)
    _write_ppm(os.path.join(args_cli.preview_dir, f"frame_{step:06d}.ppm"), mosaic)
    _write_ppm(os.path.join(args_cli.preview_dir, "latest.ppm"), mosaic)


def _draw_cross(frame: np.ndarray, x: int, y: int, radius: int = 5) -> None:
    height, width = frame.shape[:2]
    if x < 0 or x >= width or y < 0 or y >= height:
        return
    color = np.array([255, 0, 0], dtype=np.uint8)
    frame[y, max(0, x - radius) : min(width, x + radius + 1)] = color
    frame[max(0, y - radius) : min(height, y + radius + 1), x] = color


def _write_ppm(file_path: str, rgb: np.ndarray) -> None:
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got {rgb.shape}.")
    with open(file_path, "wb") as file:
        file.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        file.write(np.ascontiguousarray(rgb).tobytes())


def _frame_delta(previous: np.ndarray, current: np.ndarray) -> float:
    return float(np.mean(np.abs(current.astype(np.float32) - previous.astype(np.float32))))


def _vec(value: torch.Tensor) -> str:
    data = value.detach().cpu().tolist()
    return f"({data[0]:.3f},{data[1]:.3f},{data[2]:.3f})"


if __name__ == "__main__":
    main()
    simulation_app.close()
