#!/usr/bin/env python3
"""Play a visual behavior-cloning policy trained by scripts/il/train.py."""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run an IL visual BC checkpoint in an Isaac Lab task.")
parser.add_argument("--task", type=str, default="AIC-Port-Insertion-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt/last.pt from scripts/il/train.py.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--max_steps", type=int, default=0, help="Maximum play steps. 0 = run until app closes.")
parser.add_argument("--step_hz", type=int, default=30, help="Wall-clock loop rate. 0 = run as fast as possible.")
parser.add_argument("--camera_names", nargs="+", default=None, help="Override checkpoint camera names.")
parser.add_argument("--proprio_keys", nargs="+", default=None, help="Override checkpoint proprio keys.")
parser.add_argument("--image_size", type=int, default=0, help="Override checkpoint image size. 0 = checkpoint value.")
parser.add_argument("--action_clip", type=float, default=1.0, help="Clamp raw env actions to [-clip, clip]. <=0 disables.")
parser.add_argument("--action_scale", type=float, default=1.0, help="Extra multiplier on denormalized actions.")
parser.add_argument("--zero_rot", action="store_true", default=False, help="Zero rotational action components for debugging.")
parser.add_argument("--stream", action="store_true", default=False, help="Attach browser camera stream.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Fabric and use USD I/O.")
parser.add_argument("--disable_randomization", action="store_true", default=False, help="Disable reset randomization events.")
parser.add_argument("--disable_success_termination", action="store_true", default=False, help="Disable success termination while playing.")
parser.add_argument("--tcp_body", type=str, default="gripper_tcp", help="TCP body name used for tcp_pose_w proprio.")
parser.add_argument("--plug_center_body", type=str, default="sfp_module_link", help="Optional plug center body for legacy proprio.")
parser.add_argument("--plug_tip_body", type=str, default="sfp_tip_link", help="Optional plug tip body for legacy proprio.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import aic_task.tasks  # noqa: F401
from aic_task.utils.live_camera_stream import attach_default_camera_stream

from train import VisualBCPolicy, prepare_images


DEFAULT_CAMERAS = ["left_camera", "center_camera", "right_camera"]
DEFAULT_PROPRIO_KEYS = ["joint_pos", "joint_vel", "tcp_pose_w"]


class RateLimiter:
    """Convenience class for enforcing a target loop rate."""

    def __init__(self, hz: int):
        self.sleep_duration = 0.0 if hz <= 0 else 1.0 / hz
        self.last_time = time.time()

    def sleep(self, env: gym.Env) -> None:
        if self.sleep_duration <= 0.0:
            return
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(min(0.01, self.sleep_duration))
            env.sim.render()
        self.last_time = max(next_wakeup_time, time.time())


def main() -> None:
    checkpoint_path = Path(args_cli.checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.env_name = args_cli.task.split(":")[-1]
    if args_cli.disable_success_termination and hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    if args_cli.disable_randomization:
        _disable_randomization(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if args_cli.stream:
        attach_default_camera_stream(env)

    device = torch.device(env.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    camera_names = list(args_cli.camera_names or config.get("camera_names", DEFAULT_CAMERAS))
    proprio_keys = list(args_cli.proprio_keys or config.get("proprio_keys", DEFAULT_PROPRIO_KEYS))
    image_size = int(args_cli.image_size or config.get("image_size", 128))

    _validate_cameras(env, camera_names)
    model = _load_model(checkpoint, config, camera_names, proprio_keys, device)
    stats = _load_stats(checkpoint, device)
    rate_limiter = RateLimiter(args_cli.step_hz)

    env.sim.reset()
    env.reset()

    print(f"[INFO] Playing checkpoint: {checkpoint_path}")
    print(f"[INFO] Task: {args_cli.task} | envs={env.num_envs} | device={device}")
    print(f"[INFO] Cameras: {', '.join(camera_names)}")
    print(f"[INFO] Proprio: {', '.join(proprio_keys)}")
    print(f"[INFO] Image size: {image_size} | action_clip={args_cli.action_clip} | action_scale={args_cli.action_scale}")
    if args_cli.stream:
        print("[INFO] Camera stream attached.")

    step = 0
    try:
        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while simulation_app.is_running():
                image = _read_camera_batch(env, camera_names, device)
                proprio = _read_proprio_batch(env, proprio_keys, device)

                image = prepare_images(image, image_size, augment=False)
                proprio = (proprio - stats["proprio_mean"]) / stats["proprio_std"]
                normalized_action = model(image, proprio)["action"]
                action = normalized_action * stats["action_std"] + stats["action_mean"]
                action = action * args_cli.action_scale
                if args_cli.zero_rot and action.shape[-1] >= 6:
                    action[:, 3:6] = 0.0
                if args_cli.action_clip > 0.0:
                    action = torch.clamp(action, -args_cli.action_clip, args_cli.action_clip)

                if action.shape != env.action_space.shape:
                    raise RuntimeError(f"Model action shape {tuple(action.shape)} != env action shape {env.action_space.shape}.")
                env.step(action)

                if step % 30 == 0:
                    action_norm = torch.linalg.norm(action, dim=1).mean()
                    print(f"[INFO] step={step:05d} action_norm={float(action_norm):.4f}")

                step += 1
                if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                    break
                if env.sim.is_stopped():
                    break
                rate_limiter.sleep(env)
    finally:
        env.close()


def _load_model(
    checkpoint: dict,
    config: dict,
    camera_names: list[str],
    proprio_keys: list[str],
    device: torch.device,
) -> VisualBCPolicy:
    del proprio_keys
    model = VisualBCPolicy(
        image_channels=int(config.get("image_channels", len(camera_names) * 3)),
        proprio_dim=int(config["proprio_dim"]),
        action_dim=int(config["action_dim"]),
        num_cameras=len(camera_names),
        num_keypoints=int(config.get("num_keypoints", 0)),
        phase_count=int(config.get("phase_count", 0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _load_stats(checkpoint: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "proprio_mean": checkpoint["proprio_mean"].to(device=device, dtype=torch.float32),
        "proprio_std": checkpoint["proprio_std"].to(device=device, dtype=torch.float32).clamp_min(1.0e-6),
        "action_mean": checkpoint["action_mean"].to(device=device, dtype=torch.float32),
        "action_std": checkpoint["action_std"].to(device=device, dtype=torch.float32).clamp_min(1.0e-6),
    }


def _read_camera_batch(env: gym.Env, camera_names: list[str], device: torch.device) -> torch.Tensor:
    channels = []
    for camera_name in camera_names:
        camera = env.scene.sensors[camera_name]
        rgb = camera.data.output["rgb"][..., :3].to(device)
        if rgb.dtype == torch.uint8:
            rgb = rgb.to(dtype=torch.float32) / 255.0
        else:
            rgb = rgb.to(dtype=torch.float32)
            if float(rgb.max()) > 1.5:
                rgb = rgb / 255.0
        channels.append(rgb.permute(0, 3, 1, 2))
    return torch.cat(channels, dim=1)


def _read_proprio_batch(env: gym.Env, proprio_keys: list[str], device: torch.device) -> torch.Tensor:
    robot = env.scene["robot"]
    values = []
    for key in proprio_keys:
        if key == "joint_pos":
            values.append(robot.data.joint_pos.to(device))
        elif key == "joint_vel":
            values.append(robot.data.joint_vel.to(device))
        elif key == "tcp_pose_w":
            body_id = _first_body_id(robot, args_cli.tcp_body)
            values.append(torch.cat((robot.data.body_pos_w[:, body_id], robot.data.body_quat_w[:, body_id]), dim=1).to(device))
        elif key == "plug_center_pose_w":
            body_id = _first_body_id(robot, args_cli.plug_center_body)
            values.append(torch.cat((robot.data.body_pos_w[:, body_id], robot.data.body_quat_w[:, body_id]), dim=1).to(device))
        elif key == "plug_tip_pose_w":
            body_id = _first_body_id(robot, args_cli.plug_tip_body)
            values.append(torch.cat((robot.data.body_pos_w[:, body_id], robot.data.body_quat_w[:, body_id]), dim=1).to(device))
        elif key == "plug_axis_w":
            center_id = _first_body_id(robot, args_cli.plug_center_body)
            tip_id = _first_body_id(robot, args_cli.plug_tip_body)
            axis = robot.data.body_pos_w[:, tip_id] - robot.data.body_pos_w[:, center_id]
            axis = axis / torch.linalg.norm(axis, dim=1, keepdim=True).clamp_min(1.0e-9)
            values.append(axis.to(device))
        else:
            raise KeyError(f"Unsupported proprio key '{key}' in checkpoint. Override with --proprio_keys if needed.")
    return torch.cat(values, dim=1).to(dtype=torch.float32)


def _first_body_id(robot, body_name: str) -> int:
    body_ids = robot.find_bodies(body_name, preserve_order=True)[0]
    if len(body_ids) == 0:
        available = ", ".join(getattr(robot, "body_names", []))
        raise KeyError(f"Robot body '{body_name}' not found. Available robot bodies: {available}")
    return int(body_ids[0])


def _validate_cameras(env: gym.Env, camera_names: list[str]) -> None:
    missing = [name for name in camera_names if name not in env.scene.sensors]
    if missing:
        available = ", ".join(env.scene.sensors.keys())
        raise KeyError(f"Missing camera sensors {missing}. Available sensors: {available}")


def _disable_randomization(env_cfg) -> None:
    for name in ("randomize_light", "randomize_board_and_parts"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)


if __name__ == "__main__":
    main()
    simulation_app.close()
