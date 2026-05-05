# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record AIC port-approach demonstrations with a ground-truth scripted controller.

The controller uses simulator state to compute the target, but it still moves the
robot through the environment's normal relative IK action space. This keeps the
recorded action/state pairs compatible with teleop demonstrations.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Cheat recorder for AIC-Port-Approach-v0.")
parser.add_argument("--task", type=str, default="AIC-Port-Approach-v0", help="Name of the task.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/port_approach_cheat.hdf5",
    help="File path to export recorded demos.",
)
parser.add_argument(
    "--num_demos",
    type=int,
    default=10,
    help="Number of successful demonstrations to record. Set to 0 for infinite.",
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Consecutive success steps required before exporting an episode.",
)
parser.add_argument("--max_episode_steps", type=int, default=900, help="Discard and reset after this many steps.")
parser.add_argument("--pos_gain", type=float, default=0.8, help="Proportional gain for position error.")
parser.add_argument("--rot_gain", type=float, default=0.6, help="Proportional gain for orientation error.")
parser.add_argument(
    "--max_pos_delta",
    type=float,
    default=0.025,
    help="Maximum processed Cartesian delta per env step, in meters.",
)
parser.add_argument(
    "--max_rot_delta",
    type=float,
    default=0.20,
    help="Maximum processed axis-angle delta per env step, in radians.",
)
parser.add_argument(
    "--hold_steps",
    type=int,
    default=20,
    help="Number of low-gain settle steps after reaching the target before counting success.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import math
import os
import time

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import aic_task.tasks  # noqa: F401
from aic_task.tasks.manager_based.port_approach.port_approach_env_cfg import (
    CABLE_TIP_OFFSET_FROM_TCP,
    CABLE_TIP_RPY_FROM_TCP,
    NIC_PORT_APPROACH_OFFSET,
    NIC_PORT_APPROACH_RPY,
    TARGET_NAME,
)


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

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


def compute_cheat_action(
    env: gym.Env, action_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a raw relative IK action toward the configured port-approach pose."""
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

    raw_action = processed_action / torch.clamp(action_scale, min=1.0e-9)
    current_tip_pos_w = tcp_pos_w + math_utils.quat_apply(tcp_quat_w, tcp_tip_offset.expand(num_envs, -1))
    position_error = torch.linalg.norm(target_tip_pos_w - current_tip_pos_w, dim=1)
    orientation_error = math_utils.quat_error_magnitude(
        math_utils.quat_mul(tcp_quat_w, tcp_tip_quat),
        target_tip_quat_w,
    )
    return raw_action, position_error, orientation_error


def main() -> None:
    """Collect compatible demonstrations from a scripted ground-truth controller."""
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.env_name = args_cli.task.split(":")[-1]

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        print("No success termination term found. Cannot mark demos as successful.")

    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    action_dim = env.action_space.shape[-1]
    action_scale = _get_action_scale(env, action_dim)
    rate_limiter = RateLimiter(args_cli.step_hz)

    recorded_demo_count = 0
    success_step_count = 0
    episode_step_count = 0
    settle_step_count = 0

    env.sim.reset()
    env.reset()

    print(f"Recording cheat demonstrations for {args_cli.task}")
    print(f"Saving to: {args_cli.dataset_file}")
    print(f"Action scale: {action_scale[0].detach().cpu().tolist()}")

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running():
            actions, position_error, orientation_error = compute_cheat_action(env, action_scale)

            if settle_step_count < args_cli.hold_steps:
                env.step(actions)
                if float(position_error[0]) < 0.03 and float(orientation_error[0]) < math.radians(12.0):
                    settle_step_count += 1
                else:
                    settle_step_count = 0
            else:
                env.step(actions * 0.25)

            episode_step_count += 1

            if success_term is not None and bool(success_term.func(env, **success_term.params)[0]):
                success_step_count += 1
            else:
                success_step_count = 0

            if success_term is not None and success_step_count >= args_cli.num_success_steps:
                env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                env.recorder_manager.set_success_to_episodes(
                    [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                )
                env.recorder_manager.export_episodes([0])
                recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                print(
                    f"Recorded {recorded_demo_count}/{args_cli.num_demos} demos "
                    f"(pos_err={float(position_error[0]):.4f} m, "
                    f"rot_err={math.degrees(float(orientation_error[0])):.2f} deg)."
                )

                if args_cli.num_demos > 0 and recorded_demo_count >= args_cli.num_demos:
                    break

                env.sim.reset()
                env.recorder_manager.reset()
                env.reset()
                success_step_count = 0
                episode_step_count = 0
                settle_step_count = 0

            if episode_step_count >= args_cli.max_episode_steps:
                print(
                    f"Discarding attempt after {episode_step_count} steps "
                    f"(pos_err={float(position_error[0]):.4f} m, "
                    f"rot_err={math.degrees(float(orientation_error[0])):.2f} deg)."
                )
                env.sim.reset()
                env.recorder_manager.reset()
                env.reset()
                success_step_count = 0
                episode_step_count = 0
                settle_step_count = 0

            if env.sim.is_stopped():
                break

            rate_limiter.sleep(env)

    env.close()
    print(f"Recording session completed with {recorded_demo_count} successful demonstrations.")
    print(f"Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
