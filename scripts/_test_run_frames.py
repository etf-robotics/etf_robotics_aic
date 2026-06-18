# SPDX-License-Identifier: BSD-3-Clause
"""Goal-driven test run on the fixed cable asset; saves periodic camera frames.

Drives the TCP toward the insertion-goal entrance pose (same controller as
direct_entrance_approach.py) and dumps RGB from the three task cameras every
--every steps so the connectors can be inspected on fresh frames.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="AIC-Port-Insertion-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=2)
parser.add_argument("--steps", type=int, default=260, help="control steps per episode")
parser.add_argument("--every", type=int, default=20, help="capture interval in steps")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg
import aic_task.tasks  # noqa: F401

_POS_SCALE = 0.015
_ROT_SCALE = 0.025
OUT = "/workspace/isaaclab/etf_robotics_aic/datasets/_testrun"


def _save(rgb, path):
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgb[..., :3]).save(path)


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make(args_cli.task, cfg=cfg)
    u = env.unwrapped
    scale = torch.tensor([_POS_SCALE] * 3 + [_ROT_SCALE] * 3, device=u.device)

    for ep in range(args_cli.episodes):
        ep_dir = os.path.join(OUT, f"ep{ep}")
        os.makedirs(ep_dir, exist_ok=True)
        obs, _ = env.reset()
        for step in range(args_cli.steps):
            with torch.inference_mode():
                p = obs["policy"]
                tcp_in_eef_pos, tcp_in_eef_quat = subtract_frame_transforms(
                    p["eef_pos_b"], p["eef_quat_b"], p["tcp_pos_b"], p["tcp_quat_b"]
                )
                goal_pos_b = obs["cheatcode"]["entrance_pos_b"]
                goal_quat_b = obs["cheatcode"]["entrance_quat_b"]
                tcp_goal_pos_b, tcp_goal_quat_b = combine_frame_transforms(
                    goal_pos_b, goal_quat_b, tcp_in_eef_pos, tcp_in_eef_quat
                )
                pos_err, rot_err = compute_pose_error(
                    p["tcp_pos_b"], p["tcp_quat_b"], tcp_goal_pos_b, tcp_goal_quat_b, rot_error_type="axis_angle"
                )
                action = (torch.cat([pos_err, rot_err], dim=-1) / scale).clamp(-1.0, 1.0)
                obs, _, _, _, _ = env.step(action)

            if step % args_cli.every == 0 or step == args_cli.steps - 1:
                for cam in ("center_camera", "left_camera", "right_camera"):
                    rgb = u.scene.sensors[cam].data.output["rgb"][0].detach().cpu().numpy()
                    _save(rgb, os.path.join(ep_dir, f"step{step:04d}_{cam}.png"))
                print(f"[testrun] ep{ep} step{step:04d} captured")
        print(f"[testrun] episode {ep} done -> {ep_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
