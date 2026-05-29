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
parser.add_argument(
    "--markers", action="store_true", default=False,
    help="Draw frame markers for the TCP, EEF, and insertion goal poses.",
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
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    subtract_frame_transforms,
)
from isaaclab_tasks.utils import parse_env_cfg

import aic_task.tasks  # noqa: F401
# from aic_task.utils.live_camera_stream import attach_default_camera_stream


# DiffIK action scales must match the controller spec used at env build time.
_POS_SCALE = 0.015
_ROT_SCALE = 0.025

_MARKER_FRAME_SCALE = 0.05


def _make_frame_markers(prim_path: str) -> VisualizationMarkers:
    cfg = FRAME_MARKER_CFG.copy()
    cfg.prim_path = prim_path
    cfg.markers["frame"].scale = (_MARKER_FRAME_SCALE,) * 3
    return VisualizationMarkers(cfg)


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
        use_fabric=True,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    # attach_default_camera_stream(env)

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

    eef_markers = goal_markers = None
    if args_cli.markers:
        eef_markers = _make_frame_markers("/World/Visuals/RandomAgent/EefFrame")
        goal_markers = _make_frame_markers("/World/Visuals/RandomAgent/GoalFrame")

    while simulation_app.is_running():
        with torch.inference_mode():
            tcp_pos_w = robot.data.body_pos_w[:, tcp_idx, :]
            tcp_quat_w = robot.data.body_quat_w[:, tcp_idx, :]
            eef_pos_w = robot.data.body_pos_w[:, eef_idx, :]
            eef_quat_w = robot.data.body_quat_w[:, eef_idx, :]
            root_pos_w = robot.data.root_pos_w
            root_quat_w = robot.data.root_quat_w

            # Static EEF→TCP offset (rigidly linked, so this is constant).
            tcp_in_eef_pos, tcp_in_eef_quat = subtract_frame_transforms(
                eef_pos_w, eef_quat_w, tcp_pos_w, tcp_quat_w
            )

            # Shift the EEF-frame goal by EEF→TCP to get the desired TCP pose.
            eef_goal_pos_w, eef_goal_quat_w = _goal_pose(goal_term, args_cli.goal)
            tcp_goal_pos_w, tcp_goal_quat_w = combine_frame_transforms(
                eef_goal_pos_w, eef_goal_quat_w, tcp_in_eef_pos, tcp_in_eef_quat
            )

            # DiffIK in pose+relative mode expects the delta in the robot root
            # frame (see task_space_actions._compute_frame_pose), so express the
            # current and goal TCP poses in the root frame before differencing.
            tcp_pos_b, tcp_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w, tcp_pos_w, tcp_quat_w
            )
            tcp_goal_pos_b, tcp_goal_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w, tcp_goal_pos_w, tcp_goal_quat_w
            )

            pos_err, rot_err = compute_pose_error(
                tcp_pos_b, tcp_quat_b, tcp_goal_pos_b, tcp_goal_quat_b, rot_error_type="axis_angle"
            )

            # Convert root-frame error into raw action units; clamp so the env
            # clips each step to one scale increment toward the goal.
            action = torch.cat([pos_err, rot_err], dim=-1) / scale
            action = action.clamp(-1.0, 1.0)

            if eef_markers is not None:
                eef_markers.visualize(translations=eef_pos_w, orientations=eef_quat_w)
                goal_markers.visualize(translations=eef_goal_pos_w, orientations=eef_goal_quat_w)

            env.step(action)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
