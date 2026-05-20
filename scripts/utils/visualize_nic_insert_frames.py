"""Visualize live NIC insertion frames and keypoints in Isaac Sim.

This utility is intentionally diagnostic.  It loads ``AIC-Port-Insertion-v0``
and draws the coordinate systems/keypoints we keep discussing:

* NIC port seat, entrance, and four entrance corners.
* Robot gripper TCP, SFP module link, and raw SFP tip link.
* SFP tip side/tooth keypoints, resolved from USD under the live tip body.
* Desired calibrated plug orientation derived from the confirmed port/tip axis mapping.

The script can either hold zero action or drive the current simple insertion
oracle while the markers update.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize live NIC insertion coordinate frames.")
parser.add_argument("--task", type=str, default="AIC-Port-Insertion-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--step_hz", type=int, default=30)
parser.add_argument("--max_steps", type=int, default=100000)
parser.add_argument("--frame_scale", type=float, default=0.06)
parser.add_argument("--point_radius", type=float, default=0.006)
parser.add_argument("--log_every", type=int, default=30, help="0 disables periodic diagnostic logging.")
parser.add_argument("--env_index", type=int, default=0)
parser.add_argument("--drive_oracle", action="store_true", default=False)
parser.add_argument("--disable_orientation_hold", action="store_true", default=False)
parser.add_argument(
    "--approach_offset",
    type=float,
    nargs=3,
    default=(0.0, -0.10, 0.0),
    metavar=("X", "Y", "Z"),
    help="Oracle approach offset in sfp_port_0_link local frame.",
)
parser.add_argument(
    "--target_offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help="Extra oracle target-line offset in the sfp_port_0_link local frame.",
)
parser.add_argument(
    "--plug_frame_offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help="Extra calibrated plug-frame position offset in sfp_tip_link local frame, meters.",
)
parser.add_argument(
    "--plug_frame_rpy_offset_deg",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("ROLL", "PITCH", "YAW"),
    help="Calibrated plug-frame RPY rotation offset in sfp_tip_link local XYZ axes, degrees.",
)
parser.add_argument("--disable_tooth_top_alignment", action="store_true", default=False)
parser.add_argument("--approach_threshold", type=float, default=0.015)
parser.add_argument("--insert_lateral_threshold", type=float, default=0.003)
parser.add_argument("--insert_orientation_threshold_deg", type=float, default=10.0)
parser.add_argument("--insert_misaligned_pos_scale", type=float, default=0.2)
parser.add_argument("--insert_alignment_only", action="store_true", default=False)
parser.add_argument("--insert_lateral_correction_scale", type=float, default=1.0)
parser.add_argument("--insert_rot_scale", type=float, default=0.05)
parser.add_argument("--insert_lookahead", type=float, default=0.002)
parser.add_argument("--insert_recenter_backoff", type=float, default=0.003)
parser.add_argument("--final_threshold", type=float, default=0.003)
parser.add_argument("--insert_speed", type=float, default=0.010)
parser.add_argument("--pos_gain", type=float, default=0.8)
parser.add_argument("--rot_gain", type=float, default=0.5)
parser.add_argument("--max_pos_delta", type=float, default=0.012)
parser.add_argument("--insert_max_pos_delta", type=float, default=0.02)
parser.add_argument("--max_rot_delta", type=float, default=2.5)
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


PORT_FRAME_PATHS = {
    "port.seat_link": "/sfp_port_0_link",
    "port.entrance": "/sfp_port_0_link/sfp_port_0_link_entrance",
    "port.front_left": "/sfp_port_0_link/sfp_port_0_link_entrance/sfp_port_0_front_left",
    "port.back_left": "/sfp_port_0_link/sfp_port_0_link_entrance/sfp_port_0_back_left",
    "port.back_right": "/sfp_port_0_link/sfp_port_0_link_entrance/sfp_port_0_back_right",
    "port.front_right": "/sfp_port_0_link/sfp_port_0_link_entrance/sfp_port_0_front_right",
}

ROBOT_BODY_NAMES = {
    "robot.gripper_tcp": "gripper_tcp",
    "robot.sfp_module_link": "sfp_module_link",
    "robot.sfp_tip_link": "sfp_tip_link",
}

TIP_CHILD_NAMES = {
    "robot.sfp_tip_side_left": "sfp_tip_side_left",
    "robot.sfp_tip_side_right": "sfp_tip_side_right",
    "robot.sfp_tip_tooth_tip": "sfp_tip_tooth_tip",
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
    """Collect and draw live frame/keypoint poses for one environment."""

    def __init__(
        self,
        env: gym.Env,
        env_index: int,
        *,
        frame_scale: float,
        point_radius: float,
        plug_frame_offset: tuple[float, float, float],
        plug_frame_rpy_offset_deg: tuple[float, float, float],
    ):
        self.env = env
        self.env_index = min(env_index, env.num_envs - 1)
        self.device = env.device
        self.frame_scale = frame_scale
        self.robot = env.scene["robot"]
        self.nic_card = env.scene["nic_card"]
        self.plug_frame_offset = torch.tensor(
            plug_frame_offset,
            dtype=self.robot.data.body_pos_w.dtype,
            device=self.device,
        )
        self.plug_frame_quat_tip = _quat_from_rpy_offset_deg(
            plug_frame_rpy_offset_deg,
            dtype=self.robot.data.body_quat_w.dtype,
            device=self.device,
        )
        self.plug_frame_rpy_offset_deg = plug_frame_rpy_offset_deg

        self.port_local_poses = self._resolve_port_local_poses()
        self.robot_body_ids = self._resolve_robot_body_ids()
        self.tip_child_local_poses = self._resolve_tip_child_local_poses()

        frame_cfg = FRAME_MARKER_CFG.copy()
        frame_cfg.prim_path = "/World/Visuals/NicInsertLiveFrames"
        self.frame_markers = VisualizationMarkers(frame_cfg)

        point_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/NicInsertLivePoints",
            markers={
                "port_point": sim_utils.SphereCfg(
                    radius=point_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 1.0)),
                ),
                "module_point": sim_utils.SphereCfg(
                    radius=point_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
                ),
                "derived_point": sim_utils.SphereCfg(
                    radius=point_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.25)),
                ),
            },
        )
        self.point_markers = VisualizationMarkers(point_cfg)

        self.frame_names: list[str] = []
        self.point_names: list[str] = []
        self.latest_frames: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.latest_points: dict[str, torch.Tensor] = {}
        self._print_resolved_items()

    def update(self) -> None:
        frames = self._compute_frames()
        points = self._compute_points(frames)
        self.latest_frames = frames
        self.latest_points = points

        self.frame_names = list(frames.keys())
        if not self.frame_names:
            return
        frame_pos = torch.stack([frames[name][0] for name in self.frame_names], dim=0)
        frame_quat = torch.stack([frames[name][1] for name in self.frame_names], dim=0)
        frame_scales = torch.full((len(self.frame_names), 3), self.frame_scale, dtype=frame_pos.dtype, device=self.device)
        self.frame_markers.visualize(translations=frame_pos, orientations=frame_quat, scales=frame_scales)

        self.point_names = list(points.keys())
        if not self.point_names:
            return
        point_pos = torch.stack([points[name] for name in self.point_names], dim=0)
        marker_indices = torch.tensor([_point_marker_index(name) for name in self.point_names], dtype=torch.int64, device=self.device)
        self.point_markers.visualize(translations=point_pos, marker_indices=marker_indices)

    def diagnostic_line(self) -> str:
        if not self.latest_frames:
            return "no frame data"

        port_seat = self.latest_frames.get("port.seat_link")
        port_entrance = self.latest_frames.get("port.entrance")
        plug = self.latest_frames.get("robot.calibrated_plug_frame")
        if port_seat is None or plug is None:
            return "missing port.seat_link or robot.calibrated_plug_frame"

        port_y = _local_axis(port_seat[1], (0.0, 1.0, 0.0))
        port_z = _local_axis(port_seat[1], (0.0, 0.0, 1.0))
        plug_y = _local_axis(plug[1], (0.0, 1.0, 0.0))
        plug_z = _local_axis(plug[1], (0.0, 0.0, 1.0))
        y_axis_err = _axis_angle_deg(plug_y, -port_z)
        insert_axis_err = _axis_angle_deg(plug_z, -port_y)

        path_axis_err = None
        if port_entrance is not None:
            path_axis = torch.nn.functional.normalize(port_seat[0] - port_entrance[0], dim=0)
            path_axis_err = _axis_angle_deg(path_axis, port_y)

        width_err = self._width_axis_error_deg()
        line_err = None
        if port_entrance is not None:
            line_err = _distance_to_line(plug[0], port_entrance[0], port_y)

        parts = [
            f"plug(+Y) vs port(-Z)={y_axis_err:.2f} deg",
            f"plug(+Z) vs port(-Y)={insert_axis_err:.2f} deg",
            f"plug_to_port_axis={float(line_err):.4f} m" if line_err is not None else "plug_to_port_axis=n/a",
        ]
        if path_axis_err is not None:
            parts.append(f"(seat-entrance) vs port(+Y)={path_axis_err:.2f} deg")
        if width_err is not None:
            parts.append(f"module side axis vs port side axis={width_err:.2f} deg")
        return " | ".join(parts)

    def _resolve_port_local_poses(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        root_path = resolve_asset_root_prim_path(self.nic_card, self.env_index)
        stage = sim_utils.get_current_stage()
        root_prim = stage.GetPrimAtPath(root_path)
        resolved = {}
        for name, relative_path in PORT_FRAME_PATHS.items():
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

    def _resolve_tip_child_local_poses(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        if "robot.sfp_tip_link" not in self.robot_body_ids:
            return {}

        stage = sim_utils.get_current_stage()
        roots = _asset_search_roots(self.robot, self.env_index)
        tip_prim = _find_prim_by_basename(stage, roots, "sfp_tip_link")
        if tip_prim is None:
            print("[WARN] Could not resolve USD prim for sfp_tip_link; tip child keypoints will be hidden.")
            return {}

        resolved = {}
        for label, basename in TIP_CHILD_NAMES.items():
            child_prim = _find_prim_by_basename(stage, roots, basename)
            if child_prim is None:
                print(f"[WARN] Could not resolve USD prim for '{basename}' ({label}).")
                continue
            resolved[label] = _pose_in_root(tip_prim, child_prim, dtype=self.robot.data.body_pos_w.dtype, device=self.device)
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
            for label, (local_pos, local_quat) in self.tip_child_local_poses.items():
                frames[label] = _compose_pose(tip_pos, tip_quat, local_pos, local_quat)
            plug_local_pos = self._plug_frame_pos_tip()
            frames["robot.calibrated_plug_frame"] = _compose_pose(
                tip_pos,
                tip_quat,
                plug_local_pos,
                self.plug_frame_quat_tip,
            )

        desired = self._desired_tip_frame(frames)
        if desired is not None:
            frames["target.desired_plug_from_port"] = desired
        return frames

    def _compute_points(self, frames: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
        points = {
            name: pose[0]
            for name, pose in frames.items()
            if name.startswith(("port.", "robot.sfp_tip_side", "robot.sfp_tip_tooth", "robot.calibrated", "target."))
        }
        side_left = points.get("robot.sfp_tip_side_left")
        side_right = points.get("robot.sfp_tip_side_right")
        if side_left is not None and side_right is not None:
            points["derived.module_side_mid"] = 0.5 * (side_left + side_right)
        port_left = _mean_existing(points, ("port.front_left", "port.back_left"))
        port_right = _mean_existing(points, ("port.front_right", "port.back_right"))
        if port_left is not None:
            points["derived.port_left_mid"] = port_left
        if port_right is not None:
            points["derived.port_right_mid"] = port_right
        if port_left is not None and port_right is not None:
            points["derived.port_side_mid"] = 0.5 * (port_left + port_right)
        return points

    def _desired_tip_frame(self, frames: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor] | None:
        port_seat = frames.get("port.seat_link")
        if port_seat is None:
            port_seat = self._compute_port_seat_frame_only()
        if port_seat is None:
            return None
        desired_quat = _desired_tip_quat_from_port_quat(port_seat[1].unsqueeze(0))[0]
        return port_seat[0].clone(), desired_quat

    def _plug_frame_pos_tip(self) -> torch.Tensor:
        side_left = self.tip_child_local_poses.get("robot.sfp_tip_side_left")
        side_right = self.tip_child_local_poses.get("robot.sfp_tip_side_right")
        if side_left is None or side_right is None:
            base = torch.zeros(3, dtype=self.robot.data.body_pos_w.dtype, device=self.device)
        else:
            base = 0.5 * (side_left[0] + side_right[0])
        return base + self.plug_frame_offset

    def _compute_port_seat_frame_only(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        local = self.port_local_poses.get("port.seat_link")
        if local is None:
            return None
        card_pos = self.nic_card.data.root_pos_w[self.env_index]
        card_quat = self.nic_card.data.root_quat_w[self.env_index]
        return _compose_pose(card_pos, card_quat, local[0], local[1])

    def _width_axis_error_deg(self) -> float | None:
        side_left = self.latest_points.get("robot.sfp_tip_side_left")
        side_right = self.latest_points.get("robot.sfp_tip_side_right")
        port_left = self.latest_points.get("derived.port_left_mid")
        port_right = self.latest_points.get("derived.port_right_mid")
        if side_left is None or side_right is None or port_left is None or port_right is None:
            return None
        module_axis = torch.nn.functional.normalize(side_right - side_left, dim=0)
        port_axis = torch.nn.functional.normalize(port_right - port_left, dim=0)
        return _axis_angle_deg(module_axis, port_axis)

    def _print_resolved_items(self) -> None:
        print("[INFO] Visualized frames:")
        for name in sorted(list(self.port_local_poses) + list(self.robot_body_ids) + list(self.tip_child_local_poses)):
            print(f"  - {name}")
        print("  - robot.calibrated_plug_frame")
        print("  - target.desired_plug_from_port")
        print("[INFO] Port local positions in nic-card root frame:")
        for name, (local_pos, _) in sorted(self.port_local_poses.items()):
            print(f"  - {name}: {_fmt_vec(local_pos)}")
        print("[INFO] Tip-child local positions in sfp_tip_link frame:")
        for name, (local_pos, _) in sorted(self.tip_child_local_poses.items()):
            print(f"  - {name}: {_fmt_vec(local_pos)}")
        print(
            "[INFO] Calibrated plug frame in sfp_tip_link frame: "
            f"pos={_fmt_vec(self._plug_frame_pos_tip())}, "
            f"rpy_offset_deg={self.plug_frame_rpy_offset_deg}"
        )
        print("[INFO] Point colors: cyan=port, magenta=module keypoints, green=derived/target")


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

    visualizer = NicInsertFrameVisualizer(
        env,
        args_cli.env_index,
        frame_scale=args_cli.frame_scale,
        point_radius=args_cli.point_radius,
        plug_frame_offset=tuple(args_cli.plug_frame_offset),
        plug_frame_rpy_offset_deg=tuple(args_cli.plug_frame_rpy_offset_deg),
    )
    oracle_state = None
    oracle_targets = None
    action_scale = None
    oracle = None
    if args_cli.drive_oracle:
        oracle = _oracle_module()
        action_scale = oracle.get_action_scale(env, env.action_space.shape[-1])
        oracle_targets = oracle.make_simple_nic_insert_targets(
            env,
            approach_offset_local=tuple(args_cli.approach_offset),
            use_tooth_top_alignment=not args_cli.disable_tooth_top_alignment,
            plug_frame_offset_tip=tuple(args_cli.plug_frame_offset),
            plug_frame_rpy_offset_deg=tuple(args_cli.plug_frame_rpy_offset_deg),
            target_offset_local=tuple(args_cli.target_offset),
        )
        oracle_state = oracle.make_simple_nic_insert_state(
            env,
            plug_frame_offset_tip=tuple(args_cli.plug_frame_offset),
            plug_frame_rpy_offset_deg=tuple(args_cli.plug_frame_rpy_offset_deg),
        )
        print("[INFO] Driving frames with simple NIC insertion oracle.")
    else:
        print("[INFO] Driving frames with zero action. Use --drive_oracle to move with the simple oracle.")

    zero_action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=env.device)
    rate_limiter = RateLimiter(args_cli.step_hz)

    print(f"[INFO] Running frame visualizer on {args_cli.task}.")
    print("[INFO] orientation mapping: tip +Y -> port -Z, tip +Z -> port -Y, tip +X -> port -X")
    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        for step in range(args_cli.max_steps):
            visualizer.update()
            if args_cli.log_every > 0 and step % args_cli.log_every == 0:
                print(f"[INFO] step={step:05d} {visualizer.diagnostic_line()}")

            if args_cli.drive_oracle:
                assert oracle is not None and action_scale is not None and oracle_targets is not None and oracle_state is not None
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
                    insert_misaligned_pos_scale=args_cli.insert_misaligned_pos_scale,
                    insert_alignment_only=args_cli.insert_alignment_only,
                    insert_rot_scale=args_cli.insert_rot_scale,
                    insert_lookahead=args_cli.insert_lookahead,
                    insert_recenter_backoff=args_cli.insert_recenter_backoff,
                    insert_lateral_correction_scale=args_cli.insert_lateral_correction_scale,
                    final_threshold=args_cli.final_threshold,
                    insert_speed=args_cli.insert_speed,
                    step_dt=_env_step_dt(env),
                    hold_orientation=not args_cli.disable_orientation_hold,
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


def _asset_search_roots(asset, env_index: int) -> list[str]:
    prim_paths = list(getattr(asset.root_physx_view, "prim_paths", []))
    if not prim_paths:
        return []
    prim_path = str(prim_paths[min(env_index, len(prim_paths) - 1)])
    roots = [prim_path]
    parent = prim_path
    for _ in range(2):
        if "/" not in parent.rstrip("/"):
            break
        parent = parent.rstrip("/").rsplit("/", 1)[0]
        if parent and parent not in roots:
            roots.append(parent)
    return roots


def _find_prim_by_basename(stage, roots: list[str], basename: str):
    for root in roots:
        for candidate in _candidate_prim_paths(root, basename):
            prim = stage.GetPrimAtPath(candidate)
            if prim.IsValid():
                return prim
    for prim in stage.Traverse():
        prim_path = prim.GetPath().pathString
        if not any(prim_path == root or prim_path.startswith(root.rstrip("/") + "/") for root in roots):
            continue
        if prim_path.rsplit("/", 1)[-1] == basename:
            return prim
    return None


def _local_axis(quat_w: torch.Tensor, axis_local: tuple[float, float, float]) -> torch.Tensor:
    axis = torch.tensor(axis_local, dtype=quat_w.dtype, device=quat_w.device).unsqueeze(0)
    return math_utils.quat_apply(quat_w.unsqueeze(0), axis)[0]


def _axis_angle_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    a = torch.nn.functional.normalize(a, dim=0)
    b = torch.nn.functional.normalize(b, dim=0)
    dot = torch.clamp(torch.dot(a, b), min=-1.0, max=1.0)
    return math.degrees(float(torch.acos(dot)))


def _fmt_vec(values: torch.Tensor) -> str:
    values_cpu = values.detach().cpu()
    return "(" + ", ".join(f"{float(value):+.6f}" for value in values_cpu) + ")"


def _quat_from_rpy_offset_deg(
    rpy_deg: tuple[float, float, float],
    *,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    roll = torch.tensor([math.radians(float(rpy_deg[0]))], dtype=dtype, device=device)
    pitch = torch.tensor([math.radians(float(rpy_deg[1]))], dtype=dtype, device=device)
    yaw = torch.tensor([math.radians(float(rpy_deg[2]))], dtype=dtype, device=device)
    return math_utils.quat_from_euler_xyz(roll, pitch, yaw)[0]


def _distance_to_line(point: torch.Tensor, line_point: torch.Tensor, line_axis: torch.Tensor) -> torch.Tensor:
    line_axis = torch.nn.functional.normalize(line_axis, dim=0)
    delta = point - line_point
    closest = line_point + torch.dot(delta, line_axis) * line_axis
    return torch.linalg.norm(point - closest)


def _mean_existing(points: dict[str, torch.Tensor], names: tuple[str, str]) -> torch.Tensor | None:
    values = [points.get(name) for name in names]
    if any(value is None for value in values):
        return None
    return 0.5 * (values[0] + values[1])


def _point_marker_index(name: str) -> int:
    if name.startswith("port."):
        return 0
    if name.startswith("robot."):
        return 1
    return 2


def _desired_tip_quat_from_port_quat(port_quat_w: torch.Tensor) -> torch.Tensor:
    port_y = _batched_local_axis(port_quat_w, (0.0, 1.0, 0.0))
    port_z = _batched_local_axis(port_quat_w, (0.0, 0.0, 1.0))

    target_y = -port_z
    target_z = -port_y
    target_x = torch.linalg.cross(target_y, target_z, dim=1)
    target_x = torch.nn.functional.normalize(target_x, dim=1)
    target_y = torch.linalg.cross(target_z, target_x, dim=1)
    target_y = torch.nn.functional.normalize(target_y, dim=1)
    return math_utils.quat_from_matrix(torch.stack((target_x, target_y, target_z), dim=-1))


def _batched_local_axis(quat_w: torch.Tensor, axis_local: tuple[float, float, float]) -> torch.Tensor:
    axis = torch.tensor(axis_local, dtype=quat_w.dtype, device=quat_w.device).unsqueeze(0)
    return math_utils.quat_apply(quat_w, axis.expand(quat_w.shape[0], -1))


if __name__ == "__main__":
    main()
    simulation_app.close()
