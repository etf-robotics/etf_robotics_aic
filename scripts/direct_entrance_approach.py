# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal-driven agent that drives the TCP toward the insertion-goal entrance pose."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Goal-driven agent for the port-insertion task.")
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default="AIC-Port-Insertion-v0", help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    subtract_frame_transforms,
)
from isaaclab_tasks.utils import parse_env_cfg

import aic_task.tasks  # noqa: F401


# DiffIK action scales must match the controller spec used at env build time.
_POS_SCALE = 0.015
_ROT_SCALE = 0.025


def _entrance_pose_b(obs: dict):
    cheatcode = obs["cheatcode"]
    return cheatcode["entrance_pos_b"], cheatcode["entrance_quat_b"]


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    obs, _ = env.reset()

    unwrapped = env.unwrapped

    scale = torch.tensor(
        [_POS_SCALE, _POS_SCALE, _POS_SCALE, _ROT_SCALE, _ROT_SCALE, _ROT_SCALE],
        device=unwrapped.device,
    )

    while simulation_app.is_running():
        with torch.inference_mode():
            policy = obs["policy"]
            tcp_pos_b = policy["tcp_pos_b"]
            tcp_quat_b = policy["tcp_quat_b"]
            eef_pos_b = policy["eef_pos_b"]
            eef_quat_b = policy["eef_quat_b"]

            # TCP expressed in the EEF frame, derived from root-frame obs.
            tcp_in_eef_pos, tcp_in_eef_quat = subtract_frame_transforms(
                eef_pos_b, eef_quat_b, tcp_pos_b, tcp_quat_b
            )

            # Shift the EEF goal by EEF→TCP to get the desired TCP pose, all
            # expressed in the robot root frame expected by relative DiffIK.
            eef_goal_pos_b, eef_goal_quat_b = _entrance_pose_b(obs)
            tcp_goal_pos_b, tcp_goal_quat_b = combine_frame_transforms(
                eef_goal_pos_b, eef_goal_quat_b, tcp_in_eef_pos, tcp_in_eef_quat
            )

            pos_err, rot_err = compute_pose_error(
                tcp_pos_b, tcp_quat_b, tcp_goal_pos_b, tcp_goal_quat_b, rot_error_type="axis_angle"
            )

            # Convert root-frame error into raw action units; clamp so the env
            # clips each step to one scale increment toward the goal.
            action = torch.cat([pos_err, rot_err], dim=-1) / scale
            action = action.clamp(-1.0, 1.0)

            obs, _, _, _, _ = env.step(action)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
