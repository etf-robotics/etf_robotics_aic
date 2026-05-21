"""Visualize live front-point NIC insertion frames in Isaac Sim.

This diagnostic utility draws the geometry used by
``scripts/il/nic_card_insert_oracle.py``:

* NIC port front points.
* SFP tip front points.
* Live plug-front frame, whose +Y axis is the plug insertion direction.
* Target entry/final/current plug-front frames.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize live NIC front-point insertion frames.")
parser.add_argument("--task", type=str, default="AIC-Port-Insertion-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--step_hz", type=int, default=30)
parser.add_argument("--max_steps", type=int, default=100000)
parser.add_argument("--frame_scale", type=float, default=0.025)
parser.add_argument("--point_radius", type=float, default=0.002)
parser.add_argument("--log_every", type=int, default=30, help="0 disables periodic diagnostic logging.")
parser.add_argument("--env_index", type=int, default=0)
parser.add_argument("--drive_oracle", action="store_true", default=False)
parser.add_argument(
    "--approach_offset",
    type=float,
    nargs=3,
    default=(0.0, -0.10, 0.0),
    metavar=("X", "Y", "Z"),
    help="Oracle approach offset in the computed port-front frame.",
)
parser.add_argument(
    "--front_target_xz_offset",
    type=float,
    nargs=2,
    default=(0.0, 0.0),
    metavar=("X", "Z"),
    help="Extra front target offset in the port X/Z plane, meters.",
)
parser.add_argument("--approach_threshold", type=float, default=0.001)
parser.add_argument("--insert_lateral_threshold", type=float, default=0.010)
parser.add_argument("--insert_orientation_threshold_deg", type=float, default=44.0)
parser.add_argument("--insert_lookahead", type=float, default=0.002)
parser.add_argument("--final_threshold", type=float, default=0.003)
parser.add_argument("--insert_speed", type=float, default=0.001)
parser.add_argument("--pos_gain", type=float, default=1.2)
parser.add_argument("--rot_gain", type=float, default=0.2)
parser.add_argument("--max_pos_delta", type=float, default=0.020)
parser.add_argument("--insert_max_pos_delta", type=float, default=0.02)
parser.add_argument("--max_rot_delta", type=float, default=2.5)
parser.add_argument("--stream", action="store_true", default=False)
parser.add_argument(
    "--start_joint_pos",
    type=float,
    nargs=6,
    default=(0.05, -1.3542, -1.6648, -1.33, 1.5710, 1.9110),
    metavar=("SHOULDER_PAN", "SHOULDER_LIFT", "ELBOW", "WRIST_1", "WRIST_2", "WRIST_3"),
    help="Deterministic UR5e start joints in radians, applied after env.reset().",
)
parser.add_argument(
    "--disable_start_joint_reset",
    action="store_true",
    default=False,
    help="Do not force the deterministic start joints after env.reset().",
)
parser.add_argument(
    "--start_settle_steps",
    type=int,
    default=20,
    help="Zero-action settle steps after applying --start_joint_pos.",
)
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
import importlib.util
import math
from pathlib import Path
import sys
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

from aic_task.geometry.runtime import _candidate_prim_paths, _resolve_prim, resolve_asset_root_prim_path


PORT_POINT_PATHS = {
    "port.front_left": "/sfp_port_0_link/sfp_port_0_link_entrance/sfp_port_0_front_left",
    "port.front_right": "/sfp_port_0_link/sfp_port_0_link_entrance/sfp_port_0_front_right",
    "port.seat_link": "/sfp_port_0_link",
    "port.entrance": "/sfp_port_0_link/sfp_port_0_link_entrance",
}

ROBOT_BODY_NAMES = {
    "robot.gripper_tcp": "gripper_tcp",
    "robot.sfp_tip_link": "sfp_tip_link",
}

_ORACLE = None


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


class NicInsertFrameVisualizer:
    """Draw live and target front frames for one environment."""

    def __init__(
        self,
        env: gym.Env,
        env_index: int,
        *,
        frame_scale: float,
        point_radius: float,
        oracle,
        oracle_targets,
        oracle_state,
    ):
        self.env = env
        self.env_index = min(env_index, env.num_envs - 1)
        self.device = env.device
        self.frame_scale = frame_scale
        self.oracle = oracle
        self.oracle_targets = oracle_targets
        self.oracle_state = oracle_state
        self.robot = env.scene["robot"]
        self.nic_card = env.scene["nic_card"]
        self.port_local_poses = self._resolve_port_local_poses()
        self.robot_body_ids = self._resolve_robot_body_ids()

        frame_cfg = FRAME_MARKER_CFG.copy()
        frame_cfg.prim_path = "/World/Visuals/NicInsertFrontFrames"
        self.frame_markers = VisualizationMarkers(frame_cfg)

        point_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/NicInsertFrontPoints",
            markers={
                "port_point": sim_utils.SphereCfg(
                    radius=point_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 1.0)),
                ),
                "module_point": sim_utils.SphereCfg(
                    radius=point_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
                ),
                "target_point": sim_utils.SphereCfg(
                    radius=point_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.25)),
                ),
            },
        )
        self.point_markers = VisualizationMarkers(point_cfg)
        self.latest_frames: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.latest_points: dict[str, torch.Tensor] = {}
        self._print_resolved_items()

    def update(self) -> None:
        frames = self._compute_frames()
        points = self._compute_points(frames)
        self.latest_frames = frames
        self.latest_points = points

        frame_names = list(frames.keys())
        if frame_names:
            frame_pos = torch.stack([frames[name][0] for name in frame_names], dim=0)
            frame_quat = torch.stack([frames[name][1] for name in frame_names], dim=0)
            frame_scales = torch.full((len(frame_names), 3), self.frame_scale, dtype=frame_pos.dtype, device=self.device)
            self.frame_markers.visualize(translations=frame_pos, orientations=frame_quat, scales=frame_scales)

        point_names = list(points.keys())
        if point_names:
            point_pos = torch.stack([points[name] for name in point_names], dim=0)
            marker_indices = torch.tensor([_point_marker_index(name) for name in point_names], dtype=torch.int64, device=self.device)
            self.point_markers.visualize(translations=point_pos, marker_indices=marker_indices)

    def diagnostic_line(self) -> str:
        live = self.latest_frames.get("robot.front_frame")
        target = self.latest_frames.get("target.current_front_frame")
        if live is None or target is None:
            return "missing front-frame data"
        center_err = torch.linalg.norm(live[0] - target[0])
        x_err = _axis_angle_deg(_local_axis(live[1], (1.0, 0.0, 0.0)), _local_axis(target[1], (1.0, 0.0, 0.0)))
        y_err = _axis_angle_deg(_local_axis(live[1], (0.0, 1.0, 0.0)), _local_axis(target[1], (0.0, 1.0, 0.0)))
        left_err = _point_error(self.latest_points, "robot.sfp_tip_front_left", "target.current_front_left")
        right_err = _point_error(self.latest_points, "robot.sfp_tip_front_right", "target.current_front_right")
        return (
            f"front_center_err={float(center_err):.4f} m | "
            f"left_err={left_err:.4f} m | right_err={right_err:.4f} m | "
            f"x_axis_err={x_err:.2f} deg | y_axis_err={y_err:.2f} deg"
        )

    def _resolve_port_local_poses(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        root_path = resolve_asset_root_prim_path(self.nic_card, self.env_index)
        stage = sim_utils.get_current_stage()
        root_prim = stage.GetPrimAtPath(root_path)
        resolved = {}
        for name, relative_path in PORT_POINT_PATHS.items():
            try:
                _, prim = _resolve_prim(stage, root_path, relative_path)
                if not prim.IsValid():
                    raise KeyError(relative_path)
                resolved[name] = _pose_in_root(root_prim, prim, dtype=self.nic_card.data.root_pos_w.dtype, device=self.device)
            except Exception as exc:
                print(f"[WARN] Could not resolve port frame '{name}' ({relative_path}): {exc}")
        return resolved

    def _resolve_robot_body_ids(self) -> dict[str, int]:
        resolved = {}
        for label, body_name in ROBOT_BODY_NAMES.items():
            body_ids = self.robot.find_bodies(body_name, preserve_order=True)[0]
            if len(body_ids) == 0:
                print(f"[WARN] Robot body '{body_name}' not found for frame '{label}'.")
                continue
            resolved[label] = int(body_ids[0])
        return resolved

    def _compute_frames(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        frames: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        card_pos = self.nic_card.data.root_pos_w[self.env_index]
        card_quat = self.nic_card.data.root_quat_w[self.env_index]
        for name, (local_pos, local_quat) in self.port_local_poses.items():
            frames[name] = _compose_pose(card_pos, card_quat, local_pos, local_quat)

        for name, body_id in self.robot_body_ids.items():
            frames[name] = (
                self.robot.data.body_pos_w[self.env_index, body_id].clone(),
                self.robot.data.body_quat_w[self.env_index, body_id].clone(),
            )

        tip_frame = frames.get("robot.sfp_tip_link")
        if tip_frame is not None:
            tip_pos, tip_quat = tip_frame
            front_center, front_quat, _, _ = self.oracle._live_front_geometry(
                tip_pos.unsqueeze(0),
                tip_quat.unsqueeze(0),
                _single_env_state(self.oracle_state, self.env_index),
            )
            frames["robot.front_frame"] = (front_center[0], front_quat[0])

        world_targets = self.oracle.compute_simple_nic_insert_world_targets(self.env, self.oracle_targets)
        env_targets = _single_env_targets(world_targets, self.env_index)
        state = _single_env_state(self.oracle_state, self.env_index)
        _, current_center = self.oracle._current_target_front_center(env_targets, state)
        frames["target.entry_front_frame"] = (world_targets.entry_front_center_w[self.env_index], world_targets.front_quat_w[self.env_index])
        frames["target.final_front_frame"] = (world_targets.final_front_center_w[self.env_index], world_targets.front_quat_w[self.env_index])
        frames["target.current_front_frame"] = (current_center[0], world_targets.front_quat_w[self.env_index])
        return frames

    def _compute_points(self, frames: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
        points = {name: pose[0] for name, pose in frames.items() if name.startswith("port.")}
        tip_frame = frames.get("robot.sfp_tip_link")
        if tip_frame is not None:
            tip_pos, tip_quat = tip_frame
            state = self.oracle_state
            points["robot.sfp_tip_front_left"] = tip_pos + math_utils.quat_apply(
                tip_quat.unsqueeze(0),
                state.front_left_pos_tip[self.env_index].unsqueeze(0),
            )[0]
            points["robot.sfp_tip_front_right"] = tip_pos + math_utils.quat_apply(
                tip_quat.unsqueeze(0),
                state.front_right_pos_tip[self.env_index].unsqueeze(0),
            )[0]

        world_targets = self.oracle.compute_simple_nic_insert_world_targets(self.env, self.oracle_targets)
        env_targets = _single_env_targets(world_targets, self.env_index)
        state = _single_env_state(self.oracle_state, self.env_index)
        _, current_center = self.oracle._current_target_front_center(env_targets, state)
        current_left, current_right = self.oracle._target_front_points(env_targets, state, current_center)
        entry_left, entry_right = self.oracle._target_front_points(
            env_targets,
            state,
            world_targets.entry_front_center_w[self.env_index].unsqueeze(0),
        )
        points["target.current_front_left"] = current_left[0]
        points["target.current_front_right"] = current_right[0]
        points["target.entry_front_left"] = entry_left[0]
        points["target.entry_front_right"] = entry_right[0]
        return points

    def _print_resolved_items(self) -> None:
        print("[INFO] Visualized frames:")
        for name in sorted(list(self.port_local_poses) + list(self.robot_body_ids)):
            print(f"  - {name}")
        print("  - robot.front_frame")
        print("  - target.entry_front_frame")
        print("  - target.final_front_frame")
        print("  - target.current_front_frame")
        print("[INFO] Front target X/Z offset:", tuple(args_cli.front_target_xz_offset))
        print("[INFO] Plug front width:", float(self.oracle_state.tip_width[self.env_index, 0]))
        print("[INFO] Point colors: cyan=port, magenta=module front, green=target front")


def main() -> None:
    """Run the visualizer."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=args_cli.use_fabric,
    )
    env_cfg.env_name = args_cli.task.split(":")[-1]
    _disable_rewards(env_cfg)
    _disable_terminations(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if args_cli.stream:
        from aic_task.utils.live_camera_stream import attach_default_camera_stream

        attach_default_camera_stream(env)

    env.sim.reset()
    env.reset()
    rate_limiter = RateLimiter(args_cli.step_hz)
    if not args_cli.disable_start_joint_reset:
        _apply_start_joint_pose(env, tuple(args_cli.start_joint_pos))
        _settle_start_pose(env, rate_limiter, args_cli.start_settle_steps)

    oracle = _oracle_module()
    oracle_targets = oracle.make_simple_nic_insert_targets(
        env,
        approach_offset_local=tuple(args_cli.approach_offset),
        front_target_xz_offset=tuple(args_cli.front_target_xz_offset),
    )
    oracle_state = oracle.make_simple_nic_insert_state(env)
    action_scale = oracle.get_action_scale(env, env.action_space.shape[-1])
    visualizer = NicInsertFrameVisualizer(
        env,
        args_cli.env_index,
        frame_scale=args_cli.frame_scale,
        point_radius=args_cli.point_radius,
        oracle=oracle,
        oracle_targets=oracle_targets,
        oracle_state=oracle_state,
    )

    zero_action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=env.device)
    if args_cli.drive_oracle:
        print("[INFO] Driving frames with front-point NIC insertion oracle.")
    else:
        print("[INFO] Driving frames with zero action. Use --drive_oracle to move with the oracle.")
    print(f"[INFO] Running front-frame visualizer on {args_cli.task}.")
    if not args_cli.disable_start_joint_reset:
        print(f"[INFO] start_joint_pos: {tuple(args_cli.start_joint_pos)}")
    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        for step in range(args_cli.max_steps):
            visualizer.update()
            if args_cli.log_every > 0 and step % args_cli.log_every == 0:
                print(f"[INFO] step={step:05d} {visualizer.diagnostic_line()}")

            if args_cli.drive_oracle:
                output = oracle.compute_simple_nic_insert_oracle(
                    env,
                    action_scale,
                    oracle_targets,
                    oracle_state,
                    pos_gain=args_cli.pos_gain,
                    rot_gain=args_cli.rot_gain,
                    max_pos_delta=args_cli.max_pos_delta,
                    insert_max_pos_delta=args_cli.insert_max_pos_delta,
                    max_rot_delta=args_cli.max_rot_delta,
                    approach_threshold=args_cli.approach_threshold,
                    insert_lateral_threshold=args_cli.insert_lateral_threshold,
                    insert_orientation_threshold=math.radians(args_cli.insert_orientation_threshold_deg),
                    insert_lookahead=args_cli.insert_lookahead,
                    final_threshold=args_cli.final_threshold,
                    insert_speed=args_cli.insert_speed,
                    step_dt=_env_step_dt(env),
                )
                env.step(output.raw_action)
            else:
                env.step(zero_action)

            if env.sim.is_stopped():
                break
            rate_limiter.sleep(env)

    env.close()


def _oracle_module():
    global _ORACLE
    if _ORACLE is not None:
        return _ORACLE
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "source/aic_task/aic_task/controllers/nic_card_insert_oracle.py"
    spec = importlib.util.spec_from_file_location("_simple_nic_card_insert_oracle", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load oracle module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _ORACLE = module
    return module


def _single_env_targets(world_targets, env_index: int):
    return type(world_targets)(
        entry_front_center_w=world_targets.entry_front_center_w[env_index].unsqueeze(0),
        final_front_center_w=world_targets.final_front_center_w[env_index].unsqueeze(0),
        approach_front_center_w=world_targets.approach_front_center_w[env_index].unsqueeze(0),
        path_w=world_targets.path_w[env_index].unsqueeze(0),
        path_length=world_targets.path_length[env_index].unsqueeze(0),
        port_x_w=world_targets.port_x_w[env_index].unsqueeze(0),
        port_y_w=world_targets.port_y_w[env_index].unsqueeze(0),
        port_z_w=world_targets.port_z_w[env_index].unsqueeze(0),
        front_quat_w=world_targets.front_quat_w[env_index].unsqueeze(0),
    )


def _single_env_state(state, env_index: int):
    return type(state)(
        phase=state.phase[env_index].unsqueeze(0),
        insert_distance=state.insert_distance[env_index].unsqueeze(0),
        hold_steps=state.hold_steps[env_index].unsqueeze(0),
        tip_pos_tcp=state.tip_pos_tcp[env_index].unsqueeze(0),
        tip_quat_tcp=state.tip_quat_tcp[env_index].unsqueeze(0),
        front_center_pos_tip=state.front_center_pos_tip[env_index].unsqueeze(0),
        front_quat_tip=state.front_quat_tip[env_index].unsqueeze(0),
        front_left_pos_tip=state.front_left_pos_tip[env_index].unsqueeze(0),
        front_right_pos_tip=state.front_right_pos_tip[env_index].unsqueeze(0),
        tip_width=state.tip_width[env_index].unsqueeze(0),
    )


def _env_step_dt(env: gym.Env) -> float:
    step_dt = getattr(env, "step_dt", None)
    if step_dt is not None and float(step_dt) > 0.0:
        return float(step_dt)
    return 1.0 / max(1, args_cli.step_hz)


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
    joint_names = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    available = list(getattr(robot, "joint_names", []))
    joint_ids = [available.index(name) for name in joint_names]
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


def _pose_in_root(root_prim, child_prim, *, dtype: torch.dtype, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    from pxr import Gf, UsdGeom

    cache = UsdGeom.XformCache()
    root_matrix = cache.GetLocalToWorldTransform(root_prim)
    child_matrix = cache.GetLocalToWorldTransform(child_prim)
    root_inv = root_matrix.GetInverse()
    child_w = child_matrix.ExtractTranslation()
    child_root = root_inv.Transform(child_w)

    axes = []
    for axis in (Gf.Vec3d(1.0, 0.0, 0.0), Gf.Vec3d(0.0, 1.0, 0.0), Gf.Vec3d(0.0, 0.0, 1.0)):
        axis_root = root_inv.TransformDir(child_matrix.TransformDir(axis))
        axes.append((float(axis_root[0]), float(axis_root[1]), float(axis_root[2])))
    rot = torch.tensor(
        [
            [axes[0][0], axes[1][0], axes[2][0]],
            [axes[0][1], axes[1][1], axes[2][1]],
            [axes[0][2], axes[1][2], axes[2][2]],
        ],
        dtype=dtype,
        device=device,
    )
    quat = math_utils.quat_from_matrix(rot.unsqueeze(0))[0]
    pos = torch.tensor((float(child_root[0]), float(child_root[1]), float(child_root[2])), dtype=dtype, device=device)
    return pos, quat


def _compose_pose(
    parent_pos: torch.Tensor,
    parent_quat: torch.Tensor,
    local_pos: torch.Tensor,
    local_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos = parent_pos + math_utils.quat_apply(parent_quat.unsqueeze(0), local_pos.unsqueeze(0))[0]
    quat = math_utils.quat_mul(parent_quat.unsqueeze(0), local_quat.unsqueeze(0))[0]
    return pos, quat


def _local_axis(quat_w: torch.Tensor, axis_local: tuple[float, float, float]) -> torch.Tensor:
    axis = torch.tensor(axis_local, dtype=quat_w.dtype, device=quat_w.device).unsqueeze(0)
    return math_utils.quat_apply(quat_w.unsqueeze(0), axis)[0]


def _axis_angle_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    a = torch.nn.functional.normalize(a, dim=0)
    b = torch.nn.functional.normalize(b, dim=0)
    dot = torch.clamp(torch.dot(a, b), min=-1.0, max=1.0)
    return math.degrees(float(torch.acos(dot)))


def _point_error(points: dict[str, torch.Tensor], actual_name: str, target_name: str) -> float:
    actual = points.get(actual_name)
    target = points.get(target_name)
    if actual is None or target is None:
        return float("nan")
    return float(torch.linalg.norm(actual - target))


def _point_marker_index(name: str) -> int:
    if name.startswith("port."):
        return 0
    if name.startswith("robot."):
        return 1
    return 2


if __name__ == "__main__":
    main()
    simulation_app.close()
