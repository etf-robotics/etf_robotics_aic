# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""View or record raw RGB frames from the AIC robot cameras.

The default path uses the AIC port-approach scripted controller so the scene
changes while the cameras are streaming. Use ``--controller zero`` for a static
smoke test or ``--controller random`` for a quick motion test.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View or save AIC camera RGB streams.")
parser.add_argument("--task", type=str, default="AIC-Port-Approach-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--env_index", type=int, default=0, help="Environment index to display/save.")
parser.add_argument("--step_hz", type=int, default=30, help="Loop rate in Hz.")
parser.add_argument("--max_steps", type=int, default=0, help="Maximum sim steps to run. 0 = until closed.")
parser.add_argument(
    "--controller",
    choices=["cheat", "zero", "random"],
    default="cheat",
    help="Action source. 'cheat' is implemented for AIC-Port-Approach-v0.",
)
parser.add_argument(
    "--camera_names",
    nargs="+",
    default=["center_camera", "left_camera", "right_camera"],
    help="Scene camera sensor names to read.",
)
parser.add_argument("--display", action="store_true", default=False, help="Show a real-time OpenCV mosaic.")
parser.add_argument(
    "--allow_no_display",
    action="store_true",
    default=False,
    help="Continue when --display was requested but no OpenCV GUI window can be opened.",
)
parser.add_argument(
    "--preview_dir",
    type=str,
    default=None,
    help="Optional directory for RGB mosaic previews. Writes latest.ppm and frame_*.ppm without GUI dependencies.",
)
parser.add_argument("--preview_every", type=int, default=10, help="Write one preview mosaic every N steps.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default=None,
    help="Optional HDF5 file for raw camera frames, e.g. ./datasets/cameras.hdf5.",
)
parser.add_argument("--save_every", type=int, default=1, help="Save every N-th step when --dataset_file is set.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--pos_gain", type=float, default=0.8, help="Cheat controller position gain.")
parser.add_argument("--rot_gain", type=float, default=0.6, help="Cheat controller rotation gain.")
parser.add_argument("--max_pos_delta", type=float, default=0.025, help="Cheat controller max position delta.")
parser.add_argument("--max_rot_delta", type=float, default=0.20, help="Cheat controller max rotation delta.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# These cameras are actual render sensors, so Isaac Lab must enable camera rendering.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import json
import os
import time

import gymnasium as gym
import h5py
import numpy as np
import torch

import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import aic_task.tasks  # noqa: F401


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


class CameraHDF5Writer:
    """Streaming HDF5 writer for raw RGB camera frames."""

    def __init__(self, file_path: str, task_name: str, camera_names: list[str], step_hz: int, env_index: int):
        self.file_path = file_path
        self.camera_names = camera_names
        self.step_count = 0
        output_dir = os.path.dirname(file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self._file = h5py.File(file_path, "w")
        data_group = self._file.create_group("data")
        data_group.attrs["env_args"] = json.dumps({"env_name": task_name, "type": "raw_camera_stream"})
        self._episode_group = data_group.create_group("demo_0")
        self._episode_group.attrs["task"] = task_name
        self._episode_group.attrs["step_hz"] = step_hz
        self._episode_group.attrs["env_index"] = env_index
        self._episode_group.attrs["camera_names"] = json.dumps(camera_names)
        self._obs_group = self._episode_group.create_group("obs")
        self._action_dataset = None
        self._datasets: dict[str, h5py.Dataset] = {}

    def append(self, frames: dict[str, np.ndarray], action: np.ndarray | None = None) -> None:
        if not self._datasets:
            self._create_frame_datasets(frames)
        for camera_name, frame in frames.items():
            dataset = self._datasets[camera_name]
            dataset.resize((self.step_count + 1, *dataset.shape[1:]))
            dataset[self.step_count] = frame

        if action is not None:
            if self._action_dataset is None:
                self._action_dataset = self._episode_group.create_dataset(
                    "actions",
                    shape=(0, action.shape[-1]),
                    maxshape=(None, action.shape[-1]),
                    chunks=(1, action.shape[-1]),
                    dtype=np.float32,
                    compression="gzip",
                )
            self._action_dataset.resize((self.step_count + 1, action.shape[-1]))
            self._action_dataset[self.step_count] = action.astype(np.float32)

        self.step_count += 1

    def close(self) -> None:
        self._episode_group.attrs["num_samples"] = self.step_count
        self._file.flush()
        self._file.close()

    def _create_frame_datasets(self, frames: dict[str, np.ndarray]) -> None:
        for camera_name, frame in frames.items():
            camera_group = self._obs_group.create_group(camera_name)
            camera_group.attrs["format"] = "rgb_uint8_nhwc"
            self._datasets[camera_name] = camera_group.create_dataset(
                "rgb",
                shape=(0, *frame.shape),
                maxshape=(None, *frame.shape),
                chunks=(1, *frame.shape),
                dtype=np.uint8,
                compression="gzip",
            )


class LiveCameraViewer:
    """Small OpenCV viewer that stitches camera frames horizontally."""

    def __init__(self, camera_names: list[str]):
        self.camera_names = camera_names
        self.enabled = False
        self._cv2 = None
        self.error_message = None
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            self.error_message = (
                "No DISPLAY/WAYLAND_DISPLAY was found, so OpenCV has no GUI display to use. "
                "Run with a desktop session/X forwarding, or use --preview_dir/--dataset_file."
            )
            print(f"[WARN] {self.error_message}")
            return
        try:
            import cv2

            self._cv2 = cv2
            cv2.namedWindow("AIC cameras", cv2.WINDOW_NORMAL)
            print(
                "[INFO] OpenCV display backend ready "
                f"(DISPLAY={os.environ.get('DISPLAY')}, WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY')})."
            )
            self.enabled = True
        except Exception as exc:
            self.error_message = f"OpenCV live display is unavailable: {exc}"
            print(f"[WARN] {self.error_message}")

    def show(self, frames: dict[str, np.ndarray]) -> bool:
        """Show frames and return False when the user requests exit."""
        if not self.enabled:
            return True

        cv2 = self._cv2
        labeled_frames = []
        for camera_name in self.camera_names:
            frame = frames[camera_name].copy()
            cv2.putText(frame, camera_name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            labeled_frames.append(frame)
        mosaic_rgb = np.concatenate(labeled_frames, axis=1)
        mosaic_bgr = cv2.cvtColor(mosaic_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("AIC cameras", mosaic_bgr)
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)

    def close(self) -> None:
        if self.enabled:
            self._cv2.destroyAllWindows()


class PreviewWriter:
    """Writes dependency-free RGB PPM mosaics for remote/headless inspection."""

    def __init__(self, output_dir: str, camera_names: list[str]):
        self.output_dir = output_dir
        self.camera_names = camera_names
        os.makedirs(output_dir, exist_ok=True)

    def write(self, step: int, frames: dict[str, np.ndarray]) -> None:
        mosaic = np.concatenate([frames[camera_name] for camera_name in self.camera_names], axis=1)
        frame_path = os.path.join(self.output_dir, f"frame_{step:06d}.ppm")
        latest_path = os.path.join(self.output_dir, "latest.ppm")
        self._write_ppm(frame_path, mosaic)
        self._write_ppm(latest_path, mosaic)

    @staticmethod
    def _write_ppm(file_path: str, rgb: np.ndarray) -> None:
        height, width, channels = rgb.shape
        if channels != 3:
            raise ValueError(f"Expected RGB image with 3 channels, got shape {rgb.shape}.")
        with open(file_path, "wb") as file:
            file.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            file.write(np.ascontiguousarray(rgb).tobytes())


def _constant_vec(values: tuple[float, float, float], like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(values, dtype=like.dtype, device=like.device).unsqueeze(0)


def _constant_rpy_quat(rpy: tuple[float, float, float], num_envs: int, like: torch.Tensor) -> torch.Tensor:
    rpy_tensor = torch.tensor(rpy, dtype=like.dtype, device=like.device)
    quat = math_utils.quat_from_euler_xyz(rpy_tensor[0], rpy_tensor[1], rpy_tensor[2])
    return quat.unsqueeze(0).expand(num_envs, -1)


def _clamp_vector_norm(vector: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.linalg.norm(vector, dim=1, keepdim=True)
    scale = torch.clamp(max_norm / torch.clamp(norm, min=1.0e-9), max=1.0)
    return vector * scale


def _get_action_scale(env: gym.Env, action_dim: int) -> torch.Tensor:
    action_term = env.action_manager.get_term("arm_action")
    scale = getattr(action_term, "_scale", None)
    if scale is None:
        return torch.ones((env.num_envs, action_dim), device=env.device)
    return scale[:, :action_dim]


def compute_cheat_action(env: gym.Env, action_scale: torch.Tensor) -> torch.Tensor:
    """Compute a raw relative IK action toward the configured port-approach pose."""
    from aic_task.tasks.manager_based.port_approach.port_approach_env_cfg import (
        CABLE_TIP_OFFSET_FROM_TCP,
        CABLE_TIP_RPY_FROM_TCP,
        NIC_PORT_APPROACH_OFFSET,
        NIC_PORT_APPROACH_RPY,
        TARGET_NAME,
    )

    robot = env.scene["robot"]
    target = env.scene[TARGET_NAME]
    num_envs = env.num_envs

    tcp_body_id = robot.find_bodies("gripper_tcp", preserve_order=True)[0][0]
    tcp_pos_w = robot.data.body_pos_w[:, tcp_body_id, :]
    tcp_quat_w = robot.data.body_quat_w[:, tcp_body_id, :]

    target_offset = _constant_vec(NIC_PORT_APPROACH_OFFSET, target.data.root_pos_w)
    target_tip_pos_w = target.data.root_pos_w + math_utils.quat_apply(
        target.data.root_quat_w, target_offset.expand(num_envs, -1)
    )
    target_tip_quat_w = math_utils.quat_mul(
        target.data.root_quat_w,
        _constant_rpy_quat(NIC_PORT_APPROACH_RPY, num_envs, target.data.root_quat_w),
    )

    tcp_tip_quat = _constant_rpy_quat(CABLE_TIP_RPY_FROM_TCP, num_envs, tcp_quat_w)
    desired_tcp_quat_w = math_utils.quat_mul(target_tip_quat_w, math_utils.quat_inv(tcp_tip_quat))
    tcp_tip_offset = _constant_vec(CABLE_TIP_OFFSET_FROM_TCP, tcp_pos_w)
    desired_tcp_pos_w = target_tip_pos_w - math_utils.quat_apply(
        desired_tcp_quat_w, tcp_tip_offset.expand(num_envs, -1)
    )

    tcp_pos_b, tcp_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, tcp_pos_w, tcp_quat_w
    )
    desired_tcp_pos_b, desired_tcp_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, desired_tcp_pos_w, desired_tcp_quat_w
    )
    pos_error_b, rot_error_b = math_utils.compute_pose_error(
        tcp_pos_b, tcp_quat_b, desired_tcp_pos_b, desired_tcp_quat_b, rot_error_type="axis_angle"
    )

    processed_action = torch.zeros((num_envs, action_scale.shape[1]), dtype=tcp_pos_w.dtype, device=env.device)
    processed_action[:, 0:3] = _clamp_vector_norm(pos_error_b * args_cli.pos_gain, args_cli.max_pos_delta)
    processed_action[:, 3:6] = _clamp_vector_norm(rot_error_b * args_cli.rot_gain, args_cli.max_rot_delta)
    return processed_action / torch.clamp(action_scale, min=1.0e-9)


def compute_action(env: gym.Env, controller: str, action_scale: torch.Tensor | None) -> torch.Tensor:
    if controller == "zero":
        return torch.zeros(env.action_space.shape, device=env.device)
    if controller == "random":
        return 2.0 * torch.rand(env.action_space.shape, device=env.device) - 1.0
    if action_scale is None:
        raise RuntimeError("Cheat controller requires an action scale.")
    return compute_cheat_action(env, action_scale)


def read_camera_frames(env: gym.Env, camera_names: list[str], env_index: int) -> dict[str, np.ndarray]:
    frames = {}
    for camera_name in camera_names:
        if camera_name not in env.scene.sensors:
            available = ", ".join(env.scene.sensors.keys())
            raise KeyError(f"Camera '{camera_name}' was not found. Available sensors: {available}")
        camera = env.scene.sensors[camera_name]
        frames[camera_name] = camera.data.output["rgb"][env_index].detach().cpu().numpy()
    return frames


def main() -> None:
    if not args_cli.display and args_cli.dataset_file is None:
        print("[WARN] Neither --display nor --dataset_file was set; enabling --display for a useful default.")
        args_cli.display = True
    if args_cli.save_every <= 0:
        raise ValueError(f"--save_every must be positive, got {args_cli.save_every}.")
    if args_cli.preview_every <= 0:
        raise ValueError(f"--preview_every must be positive, got {args_cli.preview_every}.")
    if args_cli.controller == "cheat" and "Port-Approach" not in args_cli.task:
        raise ValueError(
            "--controller cheat currently expects the AIC-Port-Approach task. Use --controller zero/random."
        )

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.env_name = args_cli.task.split(":")[-1]

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if args_cli.env_index < 0 or args_cli.env_index >= env.num_envs:
        raise ValueError(f"--env_index must be in [0, {env.num_envs - 1}], got {args_cli.env_index}.")

    action_scale = None
    if args_cli.controller == "cheat":
        action_scale = _get_action_scale(env, env.action_space.shape[-1])

    writer = None
    viewer = None
    preview_writer = None
    if args_cli.dataset_file is not None:
        writer = CameraHDF5Writer(
            args_cli.dataset_file,
            args_cli.task,
            args_cli.camera_names,
            args_cli.step_hz,
            args_cli.env_index,
        )
    if args_cli.display:
        viewer = LiveCameraViewer(args_cli.camera_names)
        if not viewer.enabled and not args_cli.allow_no_display:
            raise RuntimeError(
                f"{viewer.error_message}\n"
                "If you only want to record images on a headless machine, run without --display and use "
                "--dataset_file or --preview_dir. If you want to ignore this and keep running, add "
                "--allow_no_display."
            )
    if args_cli.preview_dir is not None:
        preview_writer = PreviewWriter(args_cli.preview_dir, args_cli.camera_names)

    rate_limiter = RateLimiter(args_cli.step_hz)
    env.sim.reset()
    env.reset()

    print(f"[INFO] Streaming cameras: {', '.join(args_cli.camera_names)}")
    if writer is not None:
        print(f"[INFO] Saving raw RGB frames to: {args_cli.dataset_file}")
    if preview_writer is not None:
        print(f"[INFO] Saving preview mosaics to: {args_cli.preview_dir}")
    if viewer is not None and viewer.enabled:
        print("[INFO] Live view is open. Press q or Esc in the camera window to stop.")

    step = 0
    try:
        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while simulation_app.is_running():
                action = compute_action(env, args_cli.controller, action_scale)
                env.step(action)
                frames = read_camera_frames(env, args_cli.camera_names, args_cli.env_index)

                if viewer is not None and not viewer.show(frames):
                    break
                if writer is not None and step % args_cli.save_every == 0:
                    writer.append(frames, action[args_cli.env_index].detach().cpu().numpy())
                if preview_writer is not None and step % args_cli.preview_every == 0:
                    preview_writer.write(step, frames)

                step += 1
                if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                    break
                if env.sim.is_stopped():
                    break
                rate_limiter.sleep(env)
    finally:
        if writer is not None:
            writer.close()
            print(f"[INFO] Saved {writer.step_count} camera samples to: {writer.file_path}")
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
