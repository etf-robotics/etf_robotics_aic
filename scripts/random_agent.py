# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal-driven agent that drives the TCP toward the insertion-goal entrance pose."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Goal-driven agent for the port-insertion task.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default="AIC-Port-Insertion-v0", help="Name of the task.")
parser.add_argument(
    "--goal", type=str, default="entrance", choices=["entrance", "seat"],
    help="Which insertion-goal frame to drive toward.",
)
parser.add_argument(
    "--command_name", type=str, default="insertion_goal",
    help="Name of the insertion goal command term.",
)
parser.add_argument(
    "--tcp_body", type=str, default="gripper_tcp",
    help="Robot body the differential IK action controls.",
)
parser.add_argument(
    "--eef_body", type=str, default="sfp_tip_link",
    help="Robot body the insertion goal is expressed for (the plug tip).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if os.environ.get("AIC_CAMERA_STREAM", "1").strip().lower() not in {"0", "false", "no", "off"}:
    args_cli.enable_cameras = True

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
from aic_task.utils.live_camera_stream import attach_default_camera_stream


# DiffIK action scales must match the controller spec used at env build time.
_POS_SCALE = 0.015
_ROT_SCALE = 0.025


def _resolve_body_index(robot, body_name: str) -> int:
    body_ids, _ = robot.find_bodies(body_name)
    if len(body_ids) == 0:
        raise KeyError(f"Body '{body_name}' not found on robot articulation.")
    return int(body_ids[0])


def _goal_pose(goal_term, mode: str):
    if mode == "entrance":
        return goal_term.entrance_pos_w, goal_term.entrance_quat_w
    return goal_term.seat_pos_w, goal_term.seat_quat_w


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    attach_default_camera_stream(env)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    env.reset()

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    tcp_idx = _resolve_body_index(robot, args_cli.tcp_body)
    eef_idx = _resolve_body_index(robot, args_cli.eef_body)
    goal_term = unwrapped.command_manager.get_term(args_cli.command_name)

    scale = torch.tensor(
        [_POS_SCALE, _POS_SCALE, _POS_SCALE, _ROT_SCALE, _ROT_SCALE, _ROT_SCALE],
        device=unwrapped.device,
    )

    while simulation_app.is_running():
        with torch.inference_mode():
            tcp_pos_w = robot.data.body_pos_w[:, tcp_idx, :]
            tcp_quat_w = robot.data.body_quat_w[:, tcp_idx, :]
            eef_pos_w = robot.data.body_pos_w[:, eef_idx, :]
            eef_quat_w = robot.data.body_quat_w[:, eef_idx, :]

            # Static EEF→TCP offset (rigidly linked, so this is constant).
            tcp_in_eef_pos, tcp_in_eef_quat = subtract_frame_transforms(
                eef_pos_w, eef_quat_w, tcp_pos_w, tcp_quat_w
            )

            # Shift the EEF-frame goal by EEF→TCP to get the desired TCP pose.
            eef_goal_pos_w, eef_goal_quat_w = _goal_pose(goal_term, args_cli.goal)
            tcp_goal_pos_w, tcp_goal_quat_w = combine_frame_transforms(
                eef_goal_pos_w, eef_goal_quat_w, tcp_in_eef_pos, tcp_in_eef_quat
            )

            pos_err, rot_err = compute_pose_error(
                tcp_pos_w, tcp_quat_w, tcp_goal_pos_w, tcp_goal_quat_w, rot_error_type="axis_angle"
            )

            # Convert world-frame error into raw action units; clamp so the env
            # clips each step to one scale increment toward the goal.
            action = torch.cat([pos_err, rot_err], dim=-1) / scale
            action = action.clamp(-1.0, 1.0)

            env.step(action)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
