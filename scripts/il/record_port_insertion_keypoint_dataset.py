# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record V1 visual-oracle SFP port insertion datasets.

The recorder stores every episode, successful or not.  The scripted oracle uses
ground-truth USD port frames, plug center/tip frames, projected keypoints, and a
contact-sensor force gate during insertion.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record V1 AIC port insertion keypoint datasets.")
parser.add_argument("--task", type=str, default="AIC-Port-Insertion-v0", help="Name of the task.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/visual_port_insertion_keypoints.hdf5",
    help="HDF5 file to write.",
)
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to record. 0 = infinite.")
parser.add_argument("--max_episode_steps", type=int, default=900, help="Maximum control steps per episode.")
parser.add_argument("--step_hz", type=int, default=30, help="Control/render loop rate.")
parser.add_argument("--settle_seconds", type=float, default=1.0, help="Zero-action settling time before recording.")
parser.add_argument("--save_every", type=int, default=1, help="Save every N-th control step.")
parser.add_argument("--env_index", type=int, default=0, help="Environment index to serialize.")
parser.add_argument("--port_name", type=str, default="sfp_port_0", help="Port to target. V1 expects sfp_port_0.")
parser.add_argument("--target_name", type=str, default="nic_card", help="Scene asset containing the port USD frames.")
parser.add_argument("--plug_center_body", type=str, default="sfp_module_link", help="Robot body at plug center.")
parser.add_argument("--plug_tip_body", type=str, default="sfp_tip_link", help="Robot body/regex at plug entry tip.")
parser.add_argument(
    "--plug_x_axis_local",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 1.0),
    help="Plug local axis to align with the port X axis.",
)
parser.add_argument("--tcp_body", type=str, default="gripper_tcp", help="Robot TCP body for relative IK.")
parser.add_argument(
    "--tcp_x_axis_local",
    type=float,
    nargs=3,
    default=(1.0, 0.0, 0.0),
    help="TCP local axis to align with the port Y axis.",
)
parser.add_argument("--contact_sensor_name", type=str, default="plug_contact_forces", help="Contact sensor scene key.")
parser.add_argument(
    "--contact_prim_path",
    type=str,
    default="{ENV_REGEX_NS}/Robot/aic_unified_robot/.*",
    help="ContactSensor prim path. Leaf pattern is matched one level under the parent prim.",
)
parser.set_defaults(disable_contact_sensor=True)
parser.add_argument("--enable_contact_sensor", action="store_false", dest="disable_contact_sensor", help="Enable ContactSensor force labels.")
parser.add_argument("--disable_contact_sensor", action="store_true", help="Run without ContactSensor force labels.")
parser.add_argument(
    "--contact_body_regex",
    type=str,
    default=".*sfp.*|.*plug.*|.*tip.*",
    help="Sensor body regex to include in plug force aggregation.",
)
parser.add_argument(
    "--camera_names",
    nargs="+",
    default=["left_camera", "center_camera", "right_camera"],
    help="Camera sensor names to record.",
)
parser.set_defaults(enable_depth_labels=True)
parser.add_argument("--no_depth_labels", action="store_false", dest="enable_depth_labels", help="Disable depth labels.")
parser.add_argument("--stream", action="store_true", default=False, help="Attach the browser camera stream.")
parser.set_defaults(use_fabric=True)
parser.add_argument("--enable_fabric", action="store_true", dest="use_fabric", help="Use Fabric.")
parser.add_argument("--disable_fabric", action="store_false", dest="use_fabric", help="Disable Fabric for USD-stage debugging.")
parser.set_defaults(disable_randomization=False)
parser.add_argument("--enable_randomization", action="store_false", dest="disable_randomization", help="Keep reset randomization events.")
parser.add_argument("--disable_randomization", action="store_true", help="Disable reset randomization events.")
parser.set_defaults(sim_reset_per_episode=True)
parser.add_argument("--sim_reset_per_episode", action="store_true", help="Call env.sim.reset() at every episode start.")
parser.add_argument("--no_sim_reset_per_episode", action="store_false", dest="sim_reset_per_episode", help="Skip env.sim.reset() at episode start.")
parser.add_argument("--debug_stage_prims", action="store_true", default=False, help="Print matching USD stage prims at startup.")
parser.add_argument("--pos_gain", type=float, default=0.7, help="Oracle position gain.")
parser.add_argument("--rot_gain", type=float, default=0.45, help="Oracle rotation gain.")
parser.add_argument("--max_pos_delta", type=float, default=0.006, help="Max processed position delta outside insert.")
parser.add_argument("--max_rot_delta", type=float, default=0.1, help="Max processed rotation delta outside insert.")
parser.add_argument("--insert_max_pos_delta", type=float, default=0.002, help="Max processed position delta in INSERT.")
parser.add_argument("--insert_max_rot_delta", type=float, default=0.045, help="Max processed rotation delta in INSERT.")
parser.add_argument(
    "--straighten_axis_mode",
    choices=("disabled", "port", "world_down"),
    default="disabled",
    help="Optional rotate-in-place STRAIGHTEN target. Disabled keeps approach continuous.",
)
parser.add_argument(
    "--target_offset_port_frame",
    type=float,
    nargs=3,
    default=(0.005, 0.0, 0.0),
    metavar=("DX", "DY", "DZ"),
    help="Manual target offset in port frame meters: X=port long, Y=opposite/tooth, Z=insertion.",
)
parser.add_argument("--target_roll_offset_deg", type=float, default=-13.0, help="Manual roll target offset about port insertion axis.")
parser.add_argument("--center_enable_distance", type=float, default=0.060, help="Allow CENTER only this close to coarse target.")
parser.add_argument("--rotation_enable_distance", type=float, default=0.060, help="Allow rotation alignment only this close to coarse target.")
parser.add_argument("--align_lift", type=float, default=0.070, help="World-Z lift for ALIGN rotation before descending.")
parser.add_argument("--num_success_steps", type=int, default=12, help="Consecutive SEAT steps to mark success.")
parser.add_argument("--backoff_steps", type=int, default=8, help="Steps to spend in BACKOFF after a force jam.")
parser.add_argument("--jam_steps", type=int, default=3, help="Consecutive jammed INSERT steps before BACKOFF.")
parser.add_argument("--force_contact_threshold", type=float, default=1.0, help="Contact-start threshold in Newtons.")
parser.add_argument("--force_lateral_limit", type=float, default=4.0, help="Lateral jam threshold in Newtons.")
parser.add_argument("--force_axis_limit", type=float, default=10.0, help="Insertion-axis jam threshold in Newtons.")
parser.add_argument("--min_depth", type=float, default=0.01, help="Minimum positive camera depth for labels.")
parser.add_argument("--occlusion_depth_tolerance", type=float, default=0.015, help="Depth mismatch allowed for visibility.")
parser.add_argument("--keypoint_offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
parser.add_argument("--log_every", type=int, default=25, help="Print one recorder debug line every N steps. 0 disables.")
parser.add_argument("--log_projection_details", action="store_true", default=False, help="Print per-camera projection details.")
parser.add_argument("--debug_keypoint", type=str, default="entrance_center", help="Named keypoint for projection logs.")
parser.set_defaults(print_body_names=True)
parser.add_argument("--print_body_names", action="store_true", help="Print robot and contact body names once.")
parser.add_argument("--no_print_body_names", action="store_false", dest="print_body_names", help="Skip body-name debug print.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import math
import time

import gymnasium as gym
import numpy as np
import torch

import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import aic_task.tasks  # noqa: F401
from aic_task.controllers.port_insertion_oracle import (
    InsertionTeacherPhase,
    apply_insertion_phase_gate,
    compute_port_insertion_oracle,
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
    """Record visual-oracle insertion episodes."""
    if args_cli.port_name != "sfp_port_0":
        raise ValueError("V1 recorder only targets sfp_port_0.")
    if args_cli.save_every <= 0:
        raise ValueError(f"--save_every must be positive, got {args_cli.save_every}.")
    if args_cli.max_episode_steps <= 0:
        raise ValueError(f"--max_episode_steps must be positive, got {args_cli.max_episode_steps}.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=max(1, args_cli.env_index + 1),
        use_fabric=args_cli.use_fabric,
    )
    env_cfg.env_name = args_cli.task.split(":")[-1]
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False
    _disable_rewards(env_cfg)
    if args_cli.disable_randomization:
        _disable_randomization(env_cfg)
    if args_cli.disable_contact_sensor and hasattr(env_cfg.scene, args_cli.contact_sensor_name):
        setattr(env_cfg.scene, args_cli.contact_sensor_name, None)
    elif not args_cli.disable_contact_sensor:
        from isaaclab.sensors import ContactSensorCfg

        setattr(
            env_cfg.scene,
            args_cli.contact_sensor_name,
            ContactSensorCfg(
                prim_path=args_cli.contact_prim_path,
                update_period=0.0,
                history_length=5,
                debug_vis=False,
            ),
        )
        env_cfg.scene.robot.spawn.activate_contact_sensors = True
    _configure_camera_data_types(env_cfg, args_cli.camera_names, enable_depth=args_cli.enable_depth_labels)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if args_cli.stream:
        from aic_task.utils.live_camera_stream import attach_default_camera_stream

        attach_default_camera_stream(env)
    if args_cli.env_index < 0 or args_cli.env_index >= env.num_envs:
        raise ValueError(f"--env_index must be in [0, {env.num_envs - 1}], got {args_cli.env_index}.")

    layout = make_default_port_keypoint_layout(
        keypoint_offset=tuple(args_cli.keypoint_offset),
        port_name=args_cli.port_name,
        use_usd_geometry=True,
    )
    action_scale = get_action_scale(env, env.action_space.shape[-1])
    writer = PortKeypointDatasetWriter(
        args_cli.dataset_file,
        task_name=args_cli.task,
        camera_names=args_cli.camera_names,
        keypoint_names=layout.names,
        phase_names={int(phase): phase.name for phase in InsertionTeacherPhase},
        step_hz=args_cli.step_hz,
        env_index=args_cli.env_index,
    )
    rate_limiter = RateLimiter(args_cli.step_hz)

    print(f"[INFO] Recording V1 port insertion dataset for {args_cli.task}")
    print(f"[INFO] Target: {args_cli.target_name}/{args_cli.port_name}")
    print(f"[INFO] Cameras: {', '.join(args_cli.camera_names)}")
    print(f"[INFO] Saving to: {args_cli.dataset_file}")
    print(f"[INFO] Keypoints: {', '.join(layout.names)}")
    if args_cli.print_body_names:
        _print_body_debug(env)

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
                    f"({'success' if success else 'timeout/failure'}). "
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
    if args_cli.sim_reset_per_episode:
        env.sim.reset()
    env.reset()
    warmup_action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=env.device)
    env.step(warmup_action)
    _settle_episode(env, warmup_action, rate_limiter, seconds=args_cli.settle_seconds)

    writer.start_episode()
    success = False
    seat_step_count = 0
    jam_step_count = 0
    backoff_steps_remaining = 0
    printed_runtime_debug = False

    for step in range(args_cli.max_episode_steps):
        labels = compute_port_keypoint_labels(
            env,
            args_cli.camera_names,
            layout,
            target_name=args_cli.target_name,
            min_depth=args_cli.min_depth,
            occlusion_depth_tolerance=args_cli.occlusion_depth_tolerance,
        )

        forced_phase = InsertionTeacherPhase.BACKOFF if backoff_steps_remaining > 0 else None
        oracle = _compute_oracle(env, action_scale, labels, layout, forced_phase=forced_phase)
        if backoff_steps_remaining > 0:
            backoff_steps_remaining -= 1
            jam_step_count = 0
        else:
            if oracle.phase == InsertionTeacherPhase.INSERT and bool(oracle.force.jammed[args_cli.env_index]):
                jam_step_count += 1
            else:
                jam_step_count = 0
            if jam_step_count >= args_cli.jam_steps:
                oracle = _compute_oracle(
                    env,
                    action_scale,
                    labels,
                    layout,
                    forced_phase=InsertionTeacherPhase.BACKOFF,
                )
                backoff_steps_remaining = max(0, args_cli.backoff_steps - 1)
                jam_step_count = 0

        phase = oracle.phase
        action = apply_insertion_phase_gate(oracle.raw_action, phase, env_index=args_cli.env_index)
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

        if not printed_runtime_debug:
            print(_runtime_debug_line(oracle, args_cli.env_index))
            print(_axis_debug_line(oracle, args_cli.env_index))
            print(_port_geometry_debug_line(oracle, args_cli.env_index))
            printed_runtime_debug = True
        if args_cli.log_every > 0 and step % args_cli.log_every == 0:
            print(
                f"[INFO] step={step:04d} phase={phase.name} "
                f"visible={_visible_keypoint_count(labels, args_cli.env_index)} "
                f"in_frame={_keypoint_count(labels, args_cli.env_index, 'in_frame')} "
                f"tip_err={float(oracle.tip_to_target[args_cli.env_index]):.4f} "
                f"dx={float(oracle.tip_delta_port_frame[args_cli.env_index, 0]):+.4f} "
                f"dy={float(oracle.tip_delta_port_frame[args_cli.env_index, 1]):+.4f} "
                f"dz={float(oracle.tip_delta_port_frame[args_cli.env_index, 2]):+.4f} "
                f"axis_err={float(torch.rad2deg(oracle.axis_error[args_cli.env_index])):.1f}deg "
                f"x_err={float(torch.rad2deg(oracle.x_axis_error[args_cli.env_index])):.1f}deg "
                f"drot={float(torch.rad2deg(oracle.roll_delta[args_cli.env_index])):+.1f}deg "
                f"force={float(oracle.force.force_norm[args_cli.env_index]):.2f}N "
                f"lat={float(oracle.force.lateral_force[args_cli.env_index]):.2f}N "
                f"axisF={float(oracle.force.axis_force[args_cli.env_index]):.2f}N "
                f"contact={bool(oracle.force.contacting[args_cli.env_index])} "
                f"jam={bool(oracle.force.jammed[args_cli.env_index])} "
                f"action_norm={float(torch.linalg.norm(action[args_cli.env_index])):.4f}"
            )
            if args_cli.log_projection_details:
                print(_projection_debug_line(labels, args_cli.env_index, args_cli.debug_keypoint))

        tcp_before = _body_pos(env, args_cli.tcp_body, args_cli.env_index)
        plug_before = _body_pos(env, args_cli.plug_center_body, args_cli.env_index)
        env.step(action)
        tcp_delta = torch.linalg.norm(_body_pos(env, args_cli.tcp_body, args_cli.env_index) - tcp_before)
        plug_delta = torch.linalg.norm(_body_pos(env, args_cli.plug_center_body, args_cli.env_index) - plug_before)
        if args_cli.log_every > 0 and step % args_cli.log_every == 0:
            print(
                f"[INFO] motion step={step:04d} "
                f"tcp_delta={float(tcp_delta):.6f} "
                f"plug_delta={float(plug_delta):.6f}"
            )
        if phase == InsertionTeacherPhase.SEAT:
            seat_step_count += 1
        else:
            seat_step_count = 0
        if seat_step_count >= args_cli.num_success_steps:
            success = True
            break
        if env.sim.is_stopped():
            break
        rate_limiter.sleep(env)

    writer.close_episode(success=success)
    return success


def _settle_episode(env: gym.Env, action: torch.Tensor, rate_limiter: RateLimiter, *, seconds: float) -> None:
    """Let passive bodies settle before oracle control starts."""
    settle_steps = max(0, int(round(seconds * args_cli.step_hz)))
    if settle_steps > 0:
        print(f"[INFO] Settling episode for {seconds:.2f}s ({settle_steps} zero-action steps).")
    for _ in range(settle_steps):
        env.step(action)
        if env.sim.is_stopped():
            break
        rate_limiter.sleep(env)


def _compute_oracle(
    env: gym.Env,
    action_scale: torch.Tensor,
    labels: dict,
    layout,
    *,
    forced_phase: InsertionTeacherPhase | None,
):
    return compute_port_insertion_oracle(
        env,
        action_scale,
        labels=labels,
        layout=layout,
        phase=forced_phase,
        plug_center_body=args_cli.plug_center_body,
        plug_tip_body=args_cli.plug_tip_body,
        plug_x_axis_local=tuple(args_cli.plug_x_axis_local),
        tcp_x_axis_local=tuple(args_cli.tcp_x_axis_local),
        tcp_body=args_cli.tcp_body,
        target_name=args_cli.target_name,
        port_name=args_cli.port_name,
        contact_sensor_name=args_cli.contact_sensor_name,
        contact_body_regex=args_cli.contact_body_regex,
        pos_gain=args_cli.pos_gain,
        rot_gain=args_cli.rot_gain,
        max_pos_delta=args_cli.max_pos_delta,
        max_rot_delta=args_cli.max_rot_delta,
        insert_max_pos_delta=args_cli.insert_max_pos_delta,
        insert_max_rot_delta=args_cli.insert_max_rot_delta,
        target_offset_port_frame=tuple(args_cli.target_offset_port_frame),
        target_roll_offset=math.radians(args_cli.target_roll_offset_deg),
        contact_force_threshold=args_cli.force_contact_threshold,
        lateral_force_limit=args_cli.force_lateral_limit,
        axis_force_limit=args_cli.force_axis_limit,
        force_phase_backoff=False,
        straighten_axis_mode=args_cli.straighten_axis_mode,
        center_enable_distance=args_cli.center_enable_distance,
        rotation_enable_distance=args_cli.rotation_enable_distance,
        align_lift=args_cli.align_lift,
        env_index=args_cli.env_index,
    )


def _configure_camera_data_types(env_cfg, camera_names: list[str], *, enable_depth: bool) -> None:
    data_types = ["rgb", "distance_to_image_plane"] if enable_depth else ["rgb"]
    for camera_name in camera_names:
        if hasattr(env_cfg.scene, camera_name):
            getattr(env_cfg.scene, camera_name).data_types = list(data_types)


def _disable_rewards(env_cfg) -> None:
    """Disable rewards while recording scripted demonstrations."""
    for name, value in vars(env_cfg.rewards).items():
        if name.startswith("_") or value is None:
            continue
        setattr(env_cfg.rewards, name, None)


def _disable_randomization(env_cfg) -> None:
    """Disable reset randomization for deterministic smoke tests."""
    for name in ("randomize_light", "randomize_board_and_parts"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)


def _read_camera_frames(env: gym.Env, camera_names: list[str], env_index: int) -> dict[str, np.ndarray]:
    frames = {}
    for camera_name in camera_names:
        camera = env.scene.sensors[camera_name]
        frames[camera_name] = camera.data.output["rgb"][env_index].detach().cpu().numpy()
    return frames


def _read_proprio(env: gym.Env, env_index: int) -> dict[str, torch.Tensor]:
    robot = env.scene["robot"]
    tcp_body_id = _first_body_id(robot, args_cli.tcp_body)
    center_body_id = _first_body_id(robot, args_cli.plug_center_body)
    tip_body_id = _first_body_id(robot, args_cli.plug_tip_body)
    tcp_pos_w = robot.data.body_pos_w[env_index, tcp_body_id]
    tcp_quat_w = robot.data.body_quat_w[env_index, tcp_body_id]
    center_pos_w = robot.data.body_pos_w[env_index, center_body_id]
    center_quat_w = robot.data.body_quat_w[env_index, center_body_id]
    tip_pos_w = robot.data.body_pos_w[env_index, tip_body_id]
    tip_quat_w = robot.data.body_quat_w[env_index, tip_body_id]
    plug_axis = tip_pos_w - center_pos_w
    plug_axis = plug_axis / torch.linalg.norm(plug_axis).clamp_min(1.0e-9)
    return {
        "joint_pos": robot.data.joint_pos[env_index],
        "joint_vel": robot.data.joint_vel[env_index],
        "tcp_pose_w": torch.cat((tcp_pos_w, tcp_quat_w), dim=0),
        "plug_center_pose_w": torch.cat((center_pos_w, center_quat_w), dim=0),
        "plug_tip_pose_w": torch.cat((tip_pos_w, tip_quat_w), dim=0),
        "plug_axis_w": plug_axis,
    }


def _oracle_to_record(oracle, env_index: int) -> dict[str, torch.Tensor]:
    return {
        "raw_action": oracle.raw_action[env_index],
        "processed_action": oracle.processed_action[env_index],
        "desired_tcp_pose_w": torch.cat((oracle.desired_tcp_pos_w[env_index], oracle.desired_tcp_quat_w[env_index])),
        "desired_plug_center_pos_w": oracle.desired_plug_center_pos_w[env_index],
        "desired_plug_axis_w": oracle.desired_plug_axis_w[env_index],
        "desired_plug_x_axis_w": oracle.desired_plug_x_axis_w[env_index],
        "plug_center_pose_w": torch.cat((oracle.plug.center_pos_w[env_index], oracle.plug.center_quat_w[env_index])),
        "plug_tip_pose_w": torch.cat((oracle.plug.tip_pos_w[env_index], oracle.plug.tip_quat_w[env_index])),
        "plug_axis_w": oracle.plug.axis_w[env_index],
        "plug_x_axis_w": oracle.plug.x_axis_w[env_index],
        "plug_length": oracle.plug.length[env_index, 0],
        "entrance_pos_w": oracle.entrance_pos_w[env_index],
        "preinsert_pos_w": oracle.preinsert_pos_w[env_index],
        "seat_pos_w": oracle.seat_pos_w[env_index],
        "opposite_tooth_pos_w": oracle.opposite_tooth_pos_w[env_index],
        "insertion_axis_w": oracle.insertion_axis_w[env_index],
        "port_long_axis_w": oracle.port_long_axis_w[env_index],
        "port_opposite_axis_w": oracle.port_opposite_axis_w[env_index],
        "port_y_half": oracle.port_y_half[env_index, 0],
        "port_insertion_depth": oracle.port_insertion_depth[env_index, 0],
        "tcp_quat_w": oracle.tcp_quat_w[env_index],
        "tcp_x_axis_w": oracle.tcp_x_axis_w[env_index],
        "tip_delta_port_frame": oracle.tip_delta_port_frame[env_index],
        "roll_delta": oracle.roll_delta[env_index],
        "force_w": oracle.force.net_force_w[env_index],
        "force_norm": oracle.force.force_norm[env_index],
        "axis_force": oracle.force.axis_force[env_index],
        "lateral_force": oracle.force.lateral_force[env_index],
        "force_contacting": oracle.force.contacting[env_index].to(dtype=torch.float32),
        "force_jammed": oracle.force.jammed[env_index].to(dtype=torch.float32),
        "tip_to_target": oracle.tip_to_target[env_index],
        "axis_error": oracle.axis_error[env_index],
        "x_axis_error": oracle.x_axis_error[env_index],
    }


def _first_body_id(robot, body_name: str) -> int:
    body_ids = robot.find_bodies(body_name, preserve_order=True)[0]
    if len(body_ids) == 0:
        available = ", ".join(getattr(robot, "body_names", []))
        raise KeyError(f"Robot body '{body_name}' not found. Available robot bodies: {available}")
    return int(body_ids[0])


def _body_pos(env: gym.Env, body_name: str, env_index: int) -> torch.Tensor:
    robot = env.scene["robot"]
    return robot.data.body_pos_w[env_index, _first_body_id(robot, body_name)].clone()


def _visible_keypoint_count(labels: dict, env_index: int) -> int:
    return _keypoint_count(labels, env_index, "visible")


def _keypoint_count(labels: dict, env_index: int, mask_key: str) -> int:
    return int(sum(camera_labels[mask_key][env_index].sum().item() for camera_labels in labels["cameras"].values()))


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


def _runtime_debug_line(oracle, env_index: int) -> str:
    return (
        "[INFO] runtime "
        f"entrance={_vec3(oracle.entrance_pos_w[env_index])} "
        f"preinsert={_vec3(oracle.preinsert_pos_w[env_index])} "
        f"seat={_vec3(oracle.seat_pos_w[env_index])} "
        f"plug_center={_vec3(oracle.plug.center_pos_w[env_index])} "
        f"plug_tip={_vec3(oracle.plug.tip_pos_w[env_index])} "
        f"plug_axis={_vec3(oracle.plug.axis_w[env_index])} "
        f"plug_x={_vec3(oracle.plug.x_axis_w[env_index])} "
        f"tcp_x={_vec3(oracle.tcp_x_axis_w[env_index])} "
        f"insertion_axis={_vec3(oracle.insertion_axis_w[env_index])} "
        f"target_y={_vec3(oracle.desired_plug_x_axis_w[env_index])} "
        f"delta_port={_vec3(oracle.tip_delta_port_frame[env_index])} "
        f"drot={float(torch.rad2deg(oracle.roll_delta[env_index])):+.2f}deg "
        f"force={float(oracle.force.force_norm[env_index]):.3f}N"
    )


def _axis_debug_line(oracle, env_index: int) -> str:
    plug_axes = _plug_local_axes_w(oracle, env_index)
    tcp_axes = _local_axes_w(oracle.tcp_quat_w[env_index])
    port_x = oracle.port_long_axis_w[env_index]
    port_y = oracle.port_opposite_axis_w[env_index]
    port_z = oracle.insertion_axis_w[env_index]
    angle_parts = []
    for name, axis in plug_axes.items():
        angle_parts.append(
            f"plug{name}->port_x={_axis_angle_deg(axis, port_x):.1f}deg "
            f"plug{name}->port_y={_axis_angle_deg(axis, port_y):.1f}deg "
            f"plug{name}->port_z={_axis_angle_deg(axis, port_z):.1f}deg"
        )
    for name, axis in tcp_axes.items():
        angle_parts.append(
            f"tcp{name}->port_x={_axis_angle_deg(axis, port_x):.1f}deg "
            f"tcp{name}->port_y={_axis_angle_deg(axis, port_y):.1f}deg "
            f"tcp{name}->port_z={_axis_angle_deg(axis, port_z):.1f}deg"
        )
    return (
        "[INFO] axes "
        f"plug+X={_vec3(plug_axes['+X'])} "
        f"plug+Y={_vec3(plug_axes['+Y'])} "
        f"plug+Z={_vec3(plug_axes['+Z'])} "
        f"tcp+X={_vec3(tcp_axes['+X'])} "
        f"tcp+Y={_vec3(tcp_axes['+Y'])} "
        f"tcp+Z={_vec3(tcp_axes['+Z'])} "
        f"plug_center_to_tip={_vec3(oracle.plug.axis_w[env_index])} "
        f"port_x_long={_vec3(port_x)} "
        f"port_y_opp={_vec3(port_y)} "
        f"port_z_insert={_vec3(port_z)} "
        + " | ".join(angle_parts)
    )


def _port_geometry_debug_line(oracle, env_index: int) -> str:
    entrance = oracle.entrance_pos_w[env_index]
    seat = oracle.seat_pos_w[env_index]
    opposite = oracle.opposite_tooth_pos_w[env_index]
    x_axis = oracle.port_long_axis_w[env_index]
    y_axis = oracle.port_opposite_axis_w[env_index]
    z_axis = oracle.insertion_axis_w[env_index]
    y_half = oracle.port_y_half[env_index, 0]
    tooth = entrance - y_axis * y_half
    opposite_vec = opposite - entrance
    seat_vec = seat - entrance
    return (
        "[INFO] port-geometry "
        f"entrance={_vec3(entrance)} "
        f"seat={_vec3(seat)} "
        f"opposite_tooth={_vec3(opposite)} "
        f"implied_tooth={_vec3(tooth)} "
        f"preinsert={_vec3(oracle.preinsert_pos_w[env_index])} "
        f"entrance_to_seat={_vec3(seat_vec)} "
        f"entrance_to_opposite={_vec3(opposite_vec)} "
        f"depth={float(oracle.port_insertion_depth[env_index, 0]):.5f} "
        f"y_half={float(y_half):.5f} "
        f"x_dot_y={float(torch.sum(x_axis * y_axis)):.4f} "
        f"x_dot_z={float(torch.sum(x_axis * z_axis)):.4f} "
        f"y_dot_z={float(torch.sum(y_axis * z_axis)):.4f}"
    )


def _plug_local_axes_w(oracle, env_index: int) -> dict[str, torch.Tensor]:
    return _local_axes_w(oracle.plug.center_quat_w[env_index])


def _local_axes_w(quat_w: torch.Tensor) -> dict[str, torch.Tensor]:
    quat = quat_w.unsqueeze(0)
    basis = torch.eye(3, dtype=quat.dtype, device=quat.device)
    axes = math_utils.quat_apply(quat.expand(3, -1), basis)
    return {"+X": axes[0], "+Y": axes[1], "+Z": axes[2]}


def _axis_angle_deg(axis_a: torch.Tensor, axis_b: torch.Tensor) -> float:
    axis_a = torch.nn.functional.normalize(axis_a, dim=0)
    axis_b = torch.nn.functional.normalize(axis_b, dim=0)
    dot = torch.sum(axis_a * axis_b).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(dot)))


def _vec3(value: torch.Tensor) -> tuple[float, float, float]:
    return (round(float(value[0]), 5), round(float(value[1]), 5), round(float(value[2]), 5))


def _print_body_debug(env: gym.Env) -> None:
    robot = env.scene["robot"]
    print("[INFO] Robot bodies:")
    print("  " + ", ".join(getattr(robot, "body_names", [])))
    try:
        target = env.scene[args_cli.target_name]
        prim_paths = list(getattr(getattr(target, "root_physx_view", None), "prim_paths", []))
        print(f"[INFO] {args_cli.target_name} root prim paths:")
        print("  " + ", ".join(str(path) for path in prim_paths))
    except KeyError:
        pass
    print("[INFO] Scene sensors:")
    print("  " + ", ".join(env.scene.sensors.keys()))
    if args_cli.contact_sensor_name in env.scene.sensors:
        sensor = env.scene.sensors[args_cli.contact_sensor_name]
        print(f"[INFO] Contact sensor bodies ({args_cli.contact_sensor_name}):")
        print("  " + ", ".join(getattr(sensor, "body_names", [])))
    if args_cli.debug_stage_prims:
        _print_stage_debug_prims()


def _print_stage_debug_prims() -> None:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    interesting = ("sfp", "plug", "tip", "nic_card", "node_0099100")
    paths = []
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        lower = path.lower()
        if any(token in lower for token in interesting):
            schemas = ",".join(str(schema) for schema in prim.GetAppliedSchemas())
            paths.append(f"{path} [{prim.GetTypeName()}] [{schemas}]")
        if len(paths) >= 80:
            break
    print("[INFO] Stage prims matching sfp/plug/tip/nic_card:")
    print("  " + "\n  ".join(paths))


if __name__ == "__main__":
    main()
    simulation_app.close()
