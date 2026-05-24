"""Test whether the robot can drive a selected body point to a fixed world target.

This utility intentionally removes the NIC/board scene objects and keeps only the
robot plus visual markers.  The action still controls ``gripper_tcp`` through the
configured Differential IK action.  When ``--target_body sfp_tip_link`` is used,
the script computes the gripper TCP pose that would place the live tip link at
the desired world point.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose gripper TCP / SFP tip reachability to one world point.")
parser.add_argument("--task", type=str, default="AIC-Port-Insertion-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--step_hz", type=int, default=30)
parser.add_argument("--max_steps", type=int, default=600)
parser.add_argument(
    "--target_body",
    type=str,
    default="gripper_tcp",
    choices=("gripper_tcp", "sfp_tip_link", "sfp_tip_link_robot", "both", "all"),
    help=(
        "Body or robot-hierarchy point to place at --desired_position. "
        "'both' runs gripper_tcp then sfp_tip_link; 'all' also runs sfp_tip_link_robot."
    ),
)
parser.add_argument(
    "--desired_position",
    type=float,
    nargs=3,
    default=(0.23839998245239258, 0.24242126941680908, 0.251679927110672),
    metavar=("X", "Y", "Z"),
    help="World-space point to reach. Default is the approach target from tcp_error_strict logs.",
)
parser.add_argument(
    "--target_orientation_rpy_deg",
    type=float,
    nargs=3,
    default=(180.0, 0.0, 0.0),
    metavar=("ROLL", "PITCH", "YAW"),
    help="Desired selected-body orientation in world XYZ Euler degrees. Default points target local +Z downward.",
)
parser.add_argument(
    "--target_orientation_offset_rpy_deg",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("ROLL", "PITCH", "YAW"),
    help=(
        "Extra local XYZ Euler offset composed after --target_orientation_rpy_deg. "
        "Use this for small calibration offsets like 0 17 0 without replacing the base downward orientation."
    ),
)
parser.add_argument(
    "--target_quat_wxyz",
    type=float,
    nargs=4,
    default=None,
    metavar=("W", "X", "Y", "Z"),
    help="Desired selected-body world quaternion. Overrides --target_orientation_rpy_deg when provided.",
)
parser.add_argument("--pos_gain", type=float, default=1.2)
parser.add_argument("--rot_gain", type=float, default=0.2)
parser.add_argument("--max_pos_delta", type=float, default=0.020)
parser.add_argument("--max_rot_delta", type=float, default=2.5)
parser.add_argument(
    "--orientation_mode",
    type=str,
    default="target",
    choices=("target", "hold_current", "hold_initial"),
    help=(
        "Selected-body orientation target. target uses target orientation args; "
        "hold_current isolates position; hold_initial holds the selected body's reset orientation."
    ),
)
parser.add_argument("--success_threshold", type=float, default=0.001, help="Point error threshold in meters.")
parser.add_argument("--orientation_success_threshold_deg", type=float, default=1.0)
parser.add_argument("--hold_steps", type=int, default=60, help="Stop after this many consecutive successful steps.")
parser.add_argument("--log_every", type=int, default=10, help="0 disables terminal progress logs.")
parser.add_argument("--log_path", type=str, default=None, help="JSONL output path. Default writes logs/tcp_reachability_*.jsonl.")
parser.add_argument("--frame_scale", type=float, default=0.0005)
parser.add_argument("--point_radius", type=float, default=0.0001)
parser.add_argument(
    "--start_joint_pos",
    type=float,
    nargs=6,
    default=(0.55, -1.3642, -1.6648, -1.6933, 1.5710, 1.4110),
    metavar=("SHOULDER_PAN", "SHOULDER_LIFT", "ELBOW", "WRIST_1", "WRIST_2", "WRIST_3"),
    help="Deterministic UR5e start joints in radians, applied after each reset.",
)
parser.add_argument("--disable_start_joint_reset", action="store_true", default=False)
parser.add_argument("--start_settle_steps", type=int, default=20)
parser.add_argument(
    "--keep_scene_objects",
    action="store_true",
    default=False,
    help="Keep task board/NIC/ports if you want the normal scene. Default removes them.",
)
parser.add_argument("--stream", action="store_true", default=False)
parser.set_defaults(use_fabric=True)
parser.add_argument("--disable_fabric", action="store_false", dest="use_fabric")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.stream:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
from datetime import datetime
import json
from pathlib import Path
import time

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import aic_task.tasks  # noqa: F401
from aic_task.geometry.runtime import _candidate_prim_paths


UR5E_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


class RateLimiter:
    """Enforce a target loop rate by polling and rendering."""

    def __init__(self, hz: int):
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env) -> None:
        next_wakeup = self.last_time + self.sleep_duration
        while time.time() < next_wakeup:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time += self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


class ReachabilityMarkers:
    """Draw the desired point and live robot points."""

    def __init__(self, *, frame_scale: float, point_radius: float, device: str):
        self.frame_scale = frame_scale
        self.device = device

        frame_cfg = FRAME_MARKER_CFG.copy()
        frame_cfg.prim_path = "/World/Visuals/TcpReachabilityTargetFrame"
        self.target_frame_marker = VisualizationMarkers(frame_cfg)

        live_frame_cfg = FRAME_MARKER_CFG.copy()
        live_frame_cfg.prim_path = "/World/Visuals/TcpReachabilityLiveFrames"
        self.live_frame_markers = VisualizationMarkers(live_frame_cfg)

        point_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/TcpReachabilityPoints",
            markers={
                "desired": sim_utils.SphereCfg(
                    radius=point_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.2)),
                ),
                "selected": sim_utils.SphereCfg(
                    radius=point_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
                ),
                "tcp": sim_utils.SphereCfg(
                    radius=point_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.35, 1.0)),
                ),
                "tip": sim_utils.SphereCfg(
                    radius=point_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.65, 0.0)),
                ),
            },
        )
        self.point_markers = VisualizationMarkers(point_cfg)

    def update(
        self,
        desired_pos: torch.Tensor,
        desired_quat: torch.Tensor,
        selected_pos: torch.Tensor,
        selected_quat: torch.Tensor,
        tcp_pos: torch.Tensor,
        tcp_quat: torch.Tensor,
        tip_pos: torch.Tensor,
        tip_quat: torch.Tensor,
    ) -> None:
        scale = torch.full((1, 3), self.frame_scale, dtype=desired_pos.dtype, device=self.device)
        self.target_frame_marker.visualize(
            translations=desired_pos.unsqueeze(0),
            orientations=desired_quat.unsqueeze(0),
            scales=scale,
        )
        live_pos = torch.stack((selected_pos, tcp_pos, tip_pos), dim=0)
        live_quat = torch.stack((selected_quat, tcp_quat, tip_quat), dim=0)
        live_scales = torch.full((3, 3), self.frame_scale, dtype=desired_pos.dtype, device=self.device)
        self.live_frame_markers.visualize(translations=live_pos, orientations=live_quat, scales=live_scales)
        point_pos = torch.stack((desired_pos, selected_pos, tcp_pos, tip_pos), dim=0)
        marker_indices = torch.tensor((0, 1, 2, 3), dtype=torch.int64, device=self.device)
        self.point_markers.visualize(translations=point_pos, marker_indices=marker_indices)


class JsonlLogger:
    """Write metadata and per-step diagnostics."""

    def __init__(self, path: str | None, args) -> None:
        if path is None:
            logs_dir = Path.cwd() / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(logs_dir / f"tcp_reachability_{timestamp}.jsonl")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8")
        self.write(
            {
                "type": "metadata",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "task": args.task,
                "target_body": args.target_body,
                "desired_position": list(args.desired_position),
                "target_orientation_rpy_deg": list(args.target_orientation_rpy_deg),
                "target_orientation_offset_rpy_deg": list(args.target_orientation_offset_rpy_deg),
                "target_quat_wxyz": args.target_quat_wxyz,
                "orientation_mode": args.orientation_mode,
                "pos_gain": args.pos_gain,
                "rot_gain": args.rot_gain,
                "max_pos_delta": args.max_pos_delta,
                "max_rot_delta": args.max_rot_delta,
                "start_joint_pos": list(args.start_joint_pos),
                "scene_objects": "kept" if args.keep_scene_objects else "removed",
            }
        )

    def write(self, payload: dict) -> None:
        self.file.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class SelectedTargetReader:
    """Read a selected target either from articulation body data or from a robot Xform prim."""

    def __init__(self, env: gym.Env, robot, name: str):
        self.env = env
        self.robot = robot
        self.name = name
        self.body_id = _try_body_id(robot, name)
        self.prim_paths: list[str] = []
        if self.body_id is None:
            self.prim_paths = _resolve_robot_xform_prim_paths(env, robot, name)

    @property
    def source(self) -> str:
        return "body" if self.body_id is not None else "xform_prim"

    def pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.body_id is not None:
            return (
                self.robot.data.body_pos_w[:, self.body_id, :],
                self.robot.data.body_quat_w[:, self.body_id, :],
            )
        return _xform_prim_poses(self.prim_paths, dtype=self.robot.data.body_pos_w.dtype, device=self.env.device)


def main() -> None:
    """Run the reachability diagnostic."""
    if args_cli.max_steps <= 0:
        raise ValueError(f"--max_steps must be positive, got {args_cli.max_steps}.")
    if args_cli.step_hz <= 0:
        raise ValueError(f"--step_hz must be positive, got {args_cli.step_hz}.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=args_cli.use_fabric,
    )
    env_cfg.env_name = args_cli.task.split(":")[-1]
    _disable_rewards(env_cfg)
    _disable_terminations(env_cfg)
    if not args_cli.keep_scene_objects:
        _keep_only_robot_scene(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if args_cli.stream:
        from aic_task.utils.live_camera_stream import attach_default_camera_stream

        attach_default_camera_stream(env)

    rate_limiter = RateLimiter(args_cli.step_hz)
    markers = ReachabilityMarkers(
        frame_scale=args_cli.frame_scale,
        point_radius=args_cli.point_radius,
        device=env.device,
    )
    logger = JsonlLogger(args_cli.log_path, args_cli)
    print(f"[INFO] Writing reachability log to {logger.path}")

    try:
        env.sim.reset()
        if args_cli.target_body == "both":
            cases = ("gripper_tcp", "sfp_tip_link")
        elif args_cli.target_body == "all":
            cases = ("gripper_tcp", "sfp_tip_link", "sfp_tip_link_robot")
        else:
            cases = (args_cli.target_body,)
        for case in cases:
            _run_case(env, rate_limiter, markers, logger, case)
    finally:
        logger.close()
        env.close()


def _run_case(
    env: gym.Env,
    rate_limiter: RateLimiter,
    markers: ReachabilityMarkers,
    logger: JsonlLogger,
    target_body: str,
) -> None:
    env.reset()
    if not args_cli.disable_start_joint_reset:
        _apply_start_joint_pose(env, tuple(args_cli.start_joint_pos))
        _settle_start_pose(env, rate_limiter, args_cli.start_settle_steps)

    robot = env.scene["robot"]
    tcp_id = _first_body_id(robot, "gripper_tcp")
    tip_id = _first_body_id(robot, "sfp_tip_link")
    selected_reader = SelectedTargetReader(env, robot, target_body)
    action_scale = _get_action_scale(env, env.action_space.shape[-1])
    desired_pos = torch.tensor(args_cli.desired_position, dtype=robot.data.body_pos_w.dtype, device=env.device)
    target_quat = _target_orientation_quat(dtype=robot.data.body_quat_w.dtype, device=env.device)
    initial_tcp_quat = robot.data.body_quat_w[0, tcp_id, :].clone()
    initial_selected_pos, initial_selected_quat_batch = selected_reader.pose()
    initial_selected_quat = initial_selected_quat_batch[0].clone()
    success_count = 0

    print(f"[INFO] Case {target_body}: desired_position={tuple(args_cli.desired_position)}")
    print(f"[INFO] Case {target_body}: target_quat_wxyz={_tensor_list(target_quat)}")
    print(f"[INFO] Case {target_body}: selected source={selected_reader.source}")
    if selected_reader.prim_paths:
        print(f"[INFO] Case {target_body}: selected prim={selected_reader.prim_paths[0]}")
    print(f"[INFO] Action scale: {action_scale[0].detach().cpu().tolist()}")
    logger.write(
        {
            "type": "case_start",
            "case": target_body,
            "selected_source": selected_reader.source,
            "selected_prim_paths": selected_reader.prim_paths,
            "action_scale": _tensor_list(action_scale[0]),
            "initial_tcp_position": _tensor_list(robot.data.body_pos_w[0, tcp_id, :]),
            "initial_tcp_quaternion": _tensor_list(initial_tcp_quat),
            "initial_tip_position": _tensor_list(robot.data.body_pos_w[0, tip_id, :]),
            "initial_tip_quaternion": _tensor_list(robot.data.body_quat_w[0, tip_id, :]),
            "initial_selected_position": _tensor_list(initial_selected_pos[0]),
            "initial_selected_quaternion": _tensor_list(initial_selected_quat),
            "target_quaternion": _tensor_list(target_quat),
        }
    )

    zero_action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=env.device)
    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        for step in range(args_cli.max_steps):
            tcp_pos = robot.data.body_pos_w[:, tcp_id, :]
            tcp_quat = robot.data.body_quat_w[:, tcp_id, :]
            tip_pos = robot.data.body_pos_w[:, tip_id, :]
            tip_quat = robot.data.body_quat_w[:, tip_id, :]
            selected_pos, selected_quat = selected_reader.pose()

            selected_pos_tcp, selected_quat_tcp = math_utils.subtract_frame_transforms(
                tcp_pos,
                tcp_quat,
                selected_pos,
                selected_quat,
            )
            tcp_pos_selected, tcp_quat_selected = math_utils.subtract_frame_transforms(
                selected_pos,
                selected_quat,
                tcp_pos,
                tcp_quat,
            )
            desired_selected_quat = target_quat.unsqueeze(0).expand_as(selected_quat).clone()
            if args_cli.orientation_mode == "hold_initial":
                desired_selected_quat[:] = initial_selected_quat
            elif args_cli.orientation_mode == "hold_current":
                desired_selected_quat[:] = selected_quat
            desired_tcp_pos, desired_tcp_quat = math_utils.combine_frame_transforms(
                desired_pos.unsqueeze(0),
                desired_selected_quat,
                tcp_pos_selected,
                tcp_quat_selected,
            )
            predicted_selected_pos, predicted_selected_quat = math_utils.combine_frame_transforms(
                desired_tcp_pos,
                desired_tcp_quat,
                selected_pos_tcp,
                selected_quat_tcp,
            )

            processed_action, raw_action, debug = _relative_ik_action(
                robot,
                action_scale,
                tcp_pos,
                tcp_quat,
                desired_tcp_pos,
                desired_tcp_quat,
                pos_gain=args_cli.pos_gain,
                rot_gain=args_cli.rot_gain,
                max_pos_delta=args_cli.max_pos_delta,
                max_rot_delta=args_cli.max_rot_delta,
            )

            selected_error = desired_pos.unsqueeze(0) - selected_pos
            selected_error_norm = torch.linalg.norm(selected_error, dim=1)
            selected_ori_error = math_utils.quat_error_magnitude(selected_quat, desired_selected_quat)
            transform_closure_pos_error = torch.linalg.norm(predicted_selected_pos - desired_pos.unsqueeze(0), dim=1)
            transform_closure_ori_error = math_utils.quat_error_magnitude(
                predicted_selected_quat,
                desired_selected_quat,
            )
            tcp_error = desired_tcp_pos - tcp_pos
            tcp_error_norm = torch.linalg.norm(tcp_error, dim=1)
            tcp_ori_error = math_utils.quat_error_magnitude(tcp_quat, desired_tcp_quat)
            markers.update(
                desired_pos,
                desired_selected_quat[0],
                selected_pos[0],
                selected_quat[0],
                tcp_pos[0],
                tcp_quat[0],
                tip_pos[0],
                tip_quat[0],
            )

            logger.write(
                {
                    "type": "step",
                    "case": target_body,
                    "step": step,
                    "desired_position": _tensor_list(desired_pos),
                    "desired_selected_quaternion": _tensor_list(desired_selected_quat[0]),
                    "selected_body": target_body,
                    "selected_position": _tensor_list(selected_pos[0]),
                    "selected_quaternion": _tensor_list(selected_quat[0]),
                    "selected_error_world": _tensor_list(selected_error[0]),
                    "selected_error_norm": float(selected_error_norm[0]),
                    "selected_orientation_error_deg": float(torch.rad2deg(selected_ori_error)[0]),
                    "tcp_position": _tensor_list(tcp_pos[0]),
                    "tcp_quaternion": _tensor_list(tcp_quat[0]),
                    "tip_position": _tensor_list(tip_pos[0]),
                    "tip_quaternion": _tensor_list(tip_quat[0]),
                    "desired_tcp_position": _tensor_list(desired_tcp_pos[0]),
                    "desired_tcp_quaternion": _tensor_list(desired_tcp_quat[0]),
                    "selected_pos_in_tcp": _tensor_list(selected_pos_tcp[0]),
                    "selected_quat_in_tcp": _tensor_list(selected_quat_tcp[0]),
                    "tcp_pos_in_selected": _tensor_list(tcp_pos_selected[0]),
                    "tcp_quat_in_selected": _tensor_list(tcp_quat_selected[0]),
                    "predicted_selected_from_desired_tcp_position": _tensor_list(predicted_selected_pos[0]),
                    "predicted_selected_from_desired_tcp_quaternion": _tensor_list(predicted_selected_quat[0]),
                    "transform_closure_position_error": float(transform_closure_pos_error[0]),
                    "transform_closure_orientation_error_deg": float(torch.rad2deg(transform_closure_ori_error)[0]),
                    "tcp_position_error_world": _tensor_list(tcp_error[0]),
                    "tcp_position_error_norm": float(tcp_error_norm[0]),
                    "tcp_orientation_error_deg": float(torch.rad2deg(tcp_ori_error)[0]),
                    "base_position_error": _tensor_list(debug["pos_error_b"][0]),
                    "base_rotation_error": _tensor_list(debug["rot_error_b"][0]),
                    "processed_action": _tensor_list(processed_action[0]),
                    "raw_action": _tensor_list(raw_action[0]),
                    "processed_action_norm": float(torch.linalg.norm(processed_action[0])),
                    "raw_action_norm": float(torch.linalg.norm(raw_action[0])),
                    "joint_pos": _tensor_list(_arm_joint_values(robot, "joint_pos")[0]),
                    "joint_vel": _tensor_list(_arm_joint_values(robot, "joint_vel")[0]),
                }
            )

            if args_cli.log_every > 0 and step % args_cli.log_every == 0:
                print(
                    f"[INFO] case={target_body} step={step:04d} "
                    f"selected_err={float(selected_error_norm[0]):.6f} m "
                    f"selected_ori_err={float(torch.rad2deg(selected_ori_error)[0]):.3f} deg "
                    f"tcp_err={float(tcp_error_norm[0]):.6f} m "
                    f"tcp_ori_err={float(torch.rad2deg(tcp_ori_error)[0]):.3f} deg "
                    f"closure=({float(transform_closure_pos_error[0]):.2e} m, "
                    f"{float(torch.rad2deg(transform_closure_ori_error)[0]):.2e} deg) "
                    f"cmd_pos_b={_fmt_vec(processed_action[0, :3])} "
                    f"cmd_rot_b={float(torch.linalg.norm(processed_action[0, 3:6])):.4f} "
                    f"|raw|={float(torch.linalg.norm(raw_action[0])):.3f}"
                )

            position_ok = float(selected_error_norm[0]) <= args_cli.success_threshold
            orientation_ok = float(torch.rad2deg(selected_ori_error)[0]) <= args_cli.orientation_success_threshold_deg
            success_count = success_count + 1 if position_ok and orientation_ok else 0
            env.step(raw_action)
            if success_count >= args_cli.hold_steps:
                print(f"[INFO] Case {target_body}: success held for {args_cli.hold_steps} steps at step {step}.")
                break
            if env.sim.is_stopped():
                break
            rate_limiter.sleep(env)
        else:
            env.step(zero_action)

    logger.write({"type": "case_end", "case": target_body, "success_count": success_count})


def _relative_ik_action(
    robot,
    action_scale: torch.Tensor,
    tcp_pos_w: torch.Tensor,
    tcp_quat_w: torch.Tensor,
    desired_tcp_pos_w: torch.Tensor,
    desired_tcp_quat_w: torch.Tensor,
    *,
    pos_gain: float,
    rot_gain: float,
    max_pos_delta: float,
    max_rot_delta: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    tcp_pos_b, tcp_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        tcp_pos_w,
        tcp_quat_w,
    )
    desired_tcp_pos_b, desired_tcp_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        desired_tcp_pos_w,
        desired_tcp_quat_w,
    )
    pos_error_b, rot_error_b = math_utils.compute_pose_error(
        tcp_pos_b,
        tcp_quat_b,
        desired_tcp_pos_b,
        desired_tcp_quat_b,
        rot_error_type="axis_angle",
    )
    processed_action = torch.zeros(
        (tcp_pos_w.shape[0], action_scale.shape[1]),
        dtype=tcp_pos_w.dtype,
        device=tcp_pos_w.device,
    )
    processed_action[:, 0:3] = _clamp_vector_norm(pos_error_b * pos_gain, max_pos_delta)
    processed_action[:, 3:6] = _clamp_vector_norm(rot_error_b * rot_gain, max_rot_delta)
    raw_action = processed_action / torch.clamp(action_scale, min=1.0e-9)
    return processed_action, raw_action, {"pos_error_b": pos_error_b, "rot_error_b": rot_error_b}


def _target_orientation_quat(*, dtype: torch.dtype, device: str) -> torch.Tensor:
    if args_cli.target_quat_wxyz is not None:
        quat = torch.tensor(args_cli.target_quat_wxyz, dtype=dtype, device=device)
        return torch.nn.functional.normalize(quat, dim=0)
    base_quat = _quat_from_rpy_deg(args_cli.target_orientation_rpy_deg, dtype=dtype, device=device)
    offset_quat = _quat_from_rpy_deg(args_cli.target_orientation_offset_rpy_deg, dtype=dtype, device=device)
    quat = math_utils.quat_mul(base_quat.unsqueeze(0), offset_quat.unsqueeze(0))[0]
    return torch.nn.functional.normalize(quat, dim=0)


def _quat_from_rpy_deg(values: tuple[float, float, float], *, dtype: torch.dtype, device: str) -> torch.Tensor:
    rpy_deg = torch.tensor(values, dtype=dtype, device=device)
    rpy_rad = torch.deg2rad(rpy_deg)
    return math_utils.quat_from_euler_xyz(
        rpy_rad[0].unsqueeze(0),
        rpy_rad[1].unsqueeze(0),
        rpy_rad[2].unsqueeze(0),
    )[0]


def _get_action_scale(env: gym.Env, action_dim: int) -> torch.Tensor:
    action_term = env.action_manager.get_term("arm_action")
    scale = getattr(action_term, "_scale", None)
    if scale is None:
        return torch.ones((env.num_envs, action_dim), device=env.device)
    return scale[:, :action_dim]


def _keep_only_robot_scene(env_cfg) -> None:
    for name in ("aic_scene", "task_board", "sc_port", "sc_port_2", "nic_card"):
        if hasattr(env_cfg.scene, name):
            setattr(env_cfg.scene, name, None)


def _disable_rewards(env_cfg) -> None:
    for name, value in vars(env_cfg.rewards).items():
        if name.startswith("_") or value is None:
            continue
        setattr(env_cfg.rewards, name, None)


def _disable_terminations(env_cfg) -> None:
    for name, value in vars(env_cfg.terminations).items():
        if name.startswith("_") or value is None:
            continue
        setattr(env_cfg.terminations, name, None)


def _apply_start_joint_pose(env: gym.Env, joint_pos: tuple[float, float, float, float, float, float]) -> None:
    robot = env.scene["robot"]
    available = list(getattr(robot, "joint_names", []))
    joint_ids = [available.index(name) for name in UR5E_ARM_JOINT_NAMES]
    positions = torch.tensor(joint_pos, dtype=robot.data.joint_pos.dtype, device=env.device).unsqueeze(0)
    positions = positions.expand(env.num_envs, -1).contiguous()
    velocities = torch.zeros_like(positions)
    robot.write_joint_state_to_sim(positions, velocities, joint_ids=joint_ids)
    robot.set_joint_position_target(positions, joint_ids=joint_ids)


def _settle_start_pose(env: gym.Env, rate_limiter: RateLimiter, steps: int) -> None:
    action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=env.device)
    for _ in range(max(0, steps)):
        env.step(action)
        if env.sim.is_stopped():
            break
        rate_limiter.sleep(env)


def _first_body_id(robot, body_name: str) -> int:
    body_ids = robot.find_bodies(body_name, preserve_order=True)[0]
    if not body_ids:
        available = ", ".join(getattr(robot, "body_names", []))
        raise KeyError(f"Robot body '{body_name}' not found. Available robot bodies: {available}")
    return int(body_ids[0])


def _try_body_id(robot, body_name: str) -> int | None:
    try:
        body_ids = robot.find_bodies(body_name, preserve_order=True)[0]
    except ValueError:
        return None
    if not body_ids:
        return None
    return int(body_ids[0])


def _resolve_robot_xform_prim_paths(env: gym.Env, robot, basename: str) -> list[str]:
    stage = sim_utils.get_current_stage()
    prim_paths = []
    for env_index in range(env.num_envs):
        roots = _asset_search_roots(robot, env_index)
        prim = _find_prim_by_basename(stage, roots, basename)
        if prim is None or not prim.IsValid():
            roots_msg = ", ".join(roots) if roots else "<none>"
            raise KeyError(
                f"Robot target '{basename}' is neither an articulation body nor a USD Xform prim. "
                f"Searched roots: {roots_msg}."
            )
        prim_paths.append(prim.GetPath().pathString)
    return prim_paths


def _asset_search_roots(asset, env_index: int) -> list[str]:
    prim_paths = list(getattr(asset.root_physx_view, "prim_paths", []))
    if not prim_paths:
        return []
    prim_path = str(prim_paths[min(env_index, len(prim_paths) - 1)])
    roots = [prim_path]
    parent = prim_path
    for _ in range(4):
        if "/" not in parent.rstrip("/"):
            break
        parent = parent.rstrip("/").rsplit("/", 1)[0]
        if parent and parent not in roots:
            roots.append(parent)
    return roots


def _find_prim_by_basename(stage, roots: list[str], basename: str):
    matches = []
    for prim in stage.Traverse():
        prim_path = prim.GetPath().pathString
        if not any(prim_path == root or prim_path.startswith(root.rstrip("/") + "/") for root in roots):
            continue
        if prim_path.rsplit("/", 1)[-1] == basename:
            matches.append(prim)
    if matches:
        matches.sort(key=lambda prim: prim.GetPath().pathString.count("/"), reverse=True)
        return matches[0]
    for root in roots:
        for candidate in _candidate_prim_paths(root, basename):
            prim = stage.GetPrimAtPath(candidate)
            if prim.IsValid():
                return prim
    return None


def _xform_prim_poses(prim_paths: list[str], *, dtype: torch.dtype, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    from pxr import Gf, UsdGeom

    stage = sim_utils.get_current_stage()
    cache = UsdGeom.XformCache()
    positions = []
    quats = []
    axes_local = (
        Gf.Vec3d(1.0, 0.0, 0.0),
        Gf.Vec3d(0.0, 1.0, 0.0),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise KeyError(f"Robot target prim disappeared: {prim_path}")
        matrix = cache.GetLocalToWorldTransform(prim)
        trans = matrix.ExtractTranslation()
        positions.append((float(trans[0]), float(trans[1]), float(trans[2])))
        axes = [matrix.TransformDir(axis) for axis in axes_local]
        rot = torch.tensor(
            [
                [float(axes[0][0]), float(axes[1][0]), float(axes[2][0])],
                [float(axes[0][1]), float(axes[1][1]), float(axes[2][1])],
                [float(axes[0][2]), float(axes[1][2]), float(axes[2][2])],
            ],
            dtype=dtype,
            device=device,
        )
        quats.append(math_utils.quat_from_matrix(rot.unsqueeze(0))[0])
    return (
        torch.tensor(positions, dtype=dtype, device=device),
        torch.stack(quats, dim=0),
    )


def _arm_joint_values(robot, attr: str) -> torch.Tensor:
    available = list(getattr(robot, "joint_names", []))
    joint_ids = [available.index(name) for name in UR5E_ARM_JOINT_NAMES]
    return getattr(robot.data, attr)[:, joint_ids]


def _clamp_vector_norm(vector: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.linalg.norm(vector, dim=1, keepdim=True)
    scale = torch.clamp(max_norm / torch.clamp(norm, min=1.0e-9), max=1.0)
    return vector * scale


def _tensor_list(values: torch.Tensor) -> list[float]:
    return [float(value) for value in values.detach().cpu().tolist()]


def _fmt_vec(values: torch.Tensor) -> str:
    values_cpu = values.detach().cpu()
    return "(" + ", ".join(f"{float(value):+.4f}" for value in values_cpu) + ")"


if __name__ == "__main__":
    main()
    simulation_app.close()
