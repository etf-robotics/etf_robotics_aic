#!/usr/bin/env python3
"""Play an AIC robomimic BC checkpoint in Isaac Lab."""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import time
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run an AIC robomimic checkpoint in an Isaac Lab task.")
parser.add_argument("--task", type=str, default="AIC-Port-Insertion-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a robomimic .pth checkpoint.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--max_steps", type=int, default=0, help="Maximum play steps. 0 = run until app closes.")
parser.add_argument("--step_hz", type=int, default=30, help="Wall-clock loop rate. 0 = run as fast as possible.")
parser.add_argument("--action_clip", type=float, default=1.0, help="Clamp raw actions to [-clip, clip]. <=0 disables.")
parser.add_argument("--action_scale", type=float, default=1.0, help="Extra raw-action multiplier for debugging.")
parser.add_argument("--zero_rot", action="store_true", default=False, help="Zero rotational action components.")
parser.add_argument("--stream", action="store_true", default=False, help="Attach browser camera stream.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Fabric and use USD I/O.")
parser.add_argument(
    "--disable_randomization",
    action="store_true",
    default=False,
    help="Disable reset randomization events.",
)
parser.add_argument(
    "--disable_success_termination",
    action="store_true",
    default=False,
    help="Disable success termination.",
)
parser.add_argument("--seed", type=int, default=101)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Everything below runs after Isaac Sim starts."""

import gymnasium as gym
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import aic_task.tasks  # noqa: F401
from aic_task.utils.live_camera_stream import attach_default_camera_stream


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

    ckpt_dict = torch_load_checkpoint(checkpoint_path)
    obs_keys = checkpoint_obs_keys(ckpt_dict)
    rgb_keys = checkpoint_rgb_keys(ckpt_dict)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.env_name = args_cli.task.split(":")[-1]
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = None
    env_cfg.image_obs_list = list(rgb_keys)
    if args_cli.disable_success_termination and hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    if args_cli.disable_randomization:
        disable_randomization(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if args_cli.stream:
        attach_default_camera_stream(env)

    seed_everything(args_cli.seed, env)
    device = TorchUtils.get_torch_device(try_to_use_cuda=(args_cli.device != "cpu"))
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)
    rate_limiter = RateLimiter(args_cli.step_hz)

    env.sim.reset()
    obs_dict, _ = env.reset()
    policy.start_episode()

    print(f"[INFO] Playing checkpoint: {checkpoint_path}")
    print(f"[INFO] Task: {args_cli.task} | envs={env.num_envs} | device={device}")
    print(f"[INFO] Obs keys: {', '.join(obs_keys)}")
    print(f"[INFO] RGB keys: {', '.join(rgb_keys)}")
    print(f"[INFO] action_clip={args_cli.action_clip} action_scale={args_cli.action_scale}")

    step = 0
    try:
        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while simulation_app.is_running():
                actions = infer_actions(policy, obs_dict["policy"], obs_keys, rgb_keys, env.num_envs, env.device)
                actions = actions * args_cli.action_scale
                if args_cli.zero_rot and actions.shape[-1] >= 6:
                    actions[:, 3:6] = 0.0
                if args_cli.action_clip > 0.0:
                    actions = torch.clamp(actions, -args_cli.action_clip, args_cli.action_clip)
                if tuple(actions.shape) != tuple(env.action_space.shape):
                    raise RuntimeError(
                        f"Policy action shape {tuple(actions.shape)} != env action shape {env.action_space.shape}."
                    )

                obs_dict, _, terminated, truncated, _ = env.step(actions)

                if step % 30 == 0:
                    action_norm = torch.linalg.norm(actions, dim=1).mean()
                    print(f"[INFO] step={step:05d} action_norm={float(action_norm):.4f}")

                step += 1
                if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                    break
                if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                    obs_dict, _ = env.reset()
                    policy.start_episode()
                if env.sim.is_stopped():
                    break
                rate_limiter.sleep(env)
    finally:
        env.close()


def infer_actions(policy, policy_obs: dict[str, torch.Tensor], obs_keys, rgb_keys, num_envs: int, device: str):
    """Run a robomimic RolloutPolicy for each env from Isaac Lab dict observations."""
    actions = []
    for env_id in range(num_envs):
        obs = {}
        for key in obs_keys:
            if key not in policy_obs:
                raise KeyError(f"Live env observation is missing checkpoint key '{key}'.")
            value = policy_obs[key][env_id]
            if key in rgb_keys:
                value = prepare_rgb(value)
            else:
                value = value.detach().cpu().numpy().astype(np.float32, copy=False)
            obs[key] = value
        action = np.asarray(policy(obs), dtype=np.float32).reshape(-1)
        if action.size < 6:
            raise RuntimeError(f"Policy returned action of size {action.size}; expected at least 6.")
        actions.append(torch.from_numpy(action[:6]))
    return torch.stack(actions, dim=0).to(device=device, dtype=torch.float32)


def prepare_rgb(value: torch.Tensor) -> np.ndarray:
    """Convert Isaac Lab HWC uint8/float RGB to robomimic CHW float in [0, 1]."""
    image = value.detach().cpu()
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Expected HWC RGB image, got shape {tuple(image.shape)}.")
    image = image[..., :3]
    if image.dtype == torch.uint8:
        image = image.to(dtype=torch.float32) / 255.0
    else:
        image = image.to(dtype=torch.float32)
        if float(image.max()) > 1.5:
            image = image / 255.0
    image = image.clamp(0.0, 1.0)
    return image.permute(2, 0, 1).contiguous().numpy()


def checkpoint_obs_keys(ckpt_dict: dict[str, Any]) -> tuple[str, ...]:
    shape_meta = ckpt_dict.get("shape_metadata") or ckpt_dict.get("shape_meta") or {}
    keys = shape_meta.get("all_obs_keys")
    if keys:
        return tuple(keys)
    config = checkpoint_config_dict(ckpt_dict)
    modalities = config.get("observation", {}).get("modalities", {}).get("obs", {})
    return tuple(modalities.get("low_dim", []) + modalities.get("rgb", []))


def checkpoint_rgb_keys(ckpt_dict: dict[str, Any]) -> tuple[str, ...]:
    config = checkpoint_config_dict(ckpt_dict)
    modalities = config.get("observation", {}).get("modalities", {}).get("obs", {})
    rgb = modalities.get("rgb", [])
    if rgb:
        return tuple(rgb)
    shape_meta = ckpt_dict.get("shape_metadata") or ckpt_dict.get("shape_meta") or {}
    all_shapes = shape_meta.get("all_shapes", {})
    return tuple(key for key, shape in all_shapes.items() if len(tuple(shape)) == 3 and tuple(shape)[0] == 3)


def checkpoint_config_dict(ckpt_dict: dict[str, Any]) -> dict[str, Any]:
    raw = ckpt_dict.get("config", {})
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw)


def torch_load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def disable_randomization(env_cfg) -> None:
    for name in ("randomize_light", "randomize_board_and_parts"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)


def seed_everything(seed: int, env) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if hasattr(env, "seed"):
        env.seed(seed)


if __name__ == "__main__":
    main()
    simulation_app.close()
