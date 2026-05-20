"""Run a simple NIC-card SFP insertion demo oracle.

The demo loads ``AIC-Port-Insertion-v0``, resolves ``sfp_port_0_link`` once,
moves the SFP tip to an approach point in that link frame, then slowly advances
the tip to the link origin.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Simple demo oracle for AIC-Port-Insertion-v0.")
parser.add_argument("--task", type=str, default="AIC-Port-Insertion-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--step_hz", type=int, default=30)
parser.add_argument("--max_episode_steps", type=int, default=1200)
parser.add_argument(
    "--approach_offset",
    type=float,
    nargs=3,
    default=(0.0, -0.10, 0.0),
    metavar=("X", "Y", "Z"),
    help=(
        "Approach offset in the sfp_port_0_link local frame, meters. "
        "Default is 10 cm along -Y because insertion is port-local +Y."
    ),
)
parser.add_argument(
    "--target_offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help="Extra target-line offset in the sfp_port_0_link local frame, meters.",
)
parser.add_argument(
    "--plug_frame_offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help=(
        "Extra calibrated plug-frame position offset in world frame, meters. "
        "Applied after transforming the sfp_tip_side_left/right midpoint to world."
    ),
)
parser.add_argument(
    "--plug_frame_rpy_offset_deg",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("ROLL", "PITCH", "YAW"),
    help="Calibrated plug-frame RPY rotation offset in sfp_tip_link local XYZ axes, degrees.",
)
parser.add_argument(
    "--disable_tooth_top_alignment",
    action="store_true",
    default=False,
    help="Use the entrance-corner centerline directly instead of shifting it from the tooth/top-line relation.",
)
parser.add_argument(
    "--disable_nic_collisions",
    action="store_true",
    default=False,
    help="Debug/demo escape hatch: disable collision APIs under the NIC card before running the oracle.",
)
parser.add_argument("--approach_threshold", type=float, default=0.015)
parser.add_argument(
    "--insert_lateral_threshold",
    type=float,
    default=0.003,
    help="Max perpendicular error to the insertion line before insertion depth is allowed to advance, meters.",
)
parser.add_argument(
    "--insert_orientation_threshold_deg",
    type=float,
    default=10.0,
    help="Max tip orientation error before insertion depth is allowed to advance, degrees.",
)
parser.add_argument(
    "--insert_misaligned_pos_scale",
    type=float,
    default=0.2,
    help="Position command scale in INSERT while orientation is outside the threshold.",
)
parser.add_argument(
    "--insert_alignment_only",
    action="store_true",
    default=False,
    help="In INSERT, command only orientation and do not advance insertion depth.",
)
parser.add_argument(
    "--insert_recenter_backoff",
    type=float,
    default=0.003,
    help="Back off this many meters along the insertion path when INSERT is off-axis or misaligned.",
)
parser.add_argument(
    "--insert_lateral_correction_scale",
    type=float,
    default=1.0,
    help="Scale lateral position correction during INSERT; 0.0 pushes only along the insertion axis.",
)
parser.add_argument(
    "--insert_rot_scale",
    type=float,
    default=0.05,
    help="Scale rotational correction during INSERT after the approach phase has aligned the module.",
)
parser.add_argument(
    "--insert_lookahead",
    type=float,
    default=0.002,
    help="Minimum forward lookahead in meters while INSERT is aligned.",
)
parser.add_argument("--final_threshold", type=float, default=0.003)
parser.add_argument("--insert_speed", type=float, default=0.010, help="Target insertion speed in m/s.")
parser.add_argument("--pos_gain", type=float, default=0.8)
parser.add_argument("--rot_gain", type=float, default=0.5)
parser.add_argument("--max_pos_delta", type=float, default=0.012)
parser.add_argument("--insert_max_pos_delta", type=float, default=0.02)
parser.add_argument("--max_rot_delta", type=float, default=2.5)
parser.add_argument("--hold_steps", type=int, default=60)
parser.add_argument("--log_every", type=int, default=1, help="0 disables periodic logging.")
parser.add_argument(
    "--log_path_xy_error",
    action="store_true",
    default=False,
    help="Log world-XY lateral error to the current insertion path target, ignoring Z.",
)
parser.add_argument("--stream", action="store_true", default=False)
parser.add_argument("--disable_orientation_hold", action="store_true", default=False)
parser.add_argument(
    "--point_log_path",
    type=str,
    default=None,
    help="JSONL path for per-step port/tip/TCP point positions. Default writes to logs/ with a timestamp.",
)
parser.add_argument(
    "--disable_point_log",
    action="store_true",
    default=False,
    help="Disable the per-step JSONL point-position log.",
)
parser.add_argument(
    "--point_log_env_index",
    type=int,
    default=0,
    help="Environment index to write to the per-step point-position log.",
)
parser.add_argument(
    "--start_joint_pos",
    type=float,
    nargs=6,
    default=(0.45, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110),
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
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import aic_task.tasks  # noqa: F401

_ORACLE = None

PORT_POINT_PATHS = {
    "sfp_port_0_link": "/sfp_port_0_link",
    "sfp_port_0_link_entrance": "/sfp_port_0_link/sfp_port_0_link_entrance",
    "sfp_port_0_front_left": "/sfp_port_0_link/sfp_port_0_link_entrance/sfp_port_0_front_left",
    "sfp_port_0_back_left": "/sfp_port_0_link/sfp_port_0_link_entrance/sfp_port_0_back_left",
    "sfp_port_0_back_right": "/sfp_port_0_link/sfp_port_0_link_entrance/sfp_port_0_back_right",
    "sfp_port_0_front_right": "/sfp_port_0_link/sfp_port_0_link_entrance/sfp_port_0_front_right",
}

TIP_POINT_NAMES = {
    "sfp_tip_link": "sfp_tip_link",
    "sfp_tip_side_left": "sfp_tip_side_left",
    "sfp_tip_side_right": "sfp_tip_side_right",
    "sfp_tip_tooth_tip": "sfp_tip_tooth_tip",
}


def _oracle_module():
    """Load the new oracle without importing older controller package exports."""
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


def main() -> None:
    """Run the demo oracle in the normal env action space."""
    if args_cli.max_episode_steps <= 0:
        raise ValueError(f"--max_episode_steps must be positive, got {args_cli.max_episode_steps}.")
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

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if args_cli.stream:
        from aic_task.utils.live_camera_stream import attach_default_camera_stream

        attach_default_camera_stream(env)

    oracle = _oracle_module()
    action_scale = oracle.get_action_scale(env, env.action_space.shape[-1])
    rate_limiter = RateLimiter(args_cli.step_hz)

    env.sim.reset()
    env.reset()
    if not args_cli.disable_start_joint_reset:
        _apply_start_joint_pose(env, tuple(args_cli.start_joint_pos))
        _settle_start_pose(env, rate_limiter, args_cli.start_settle_steps)
    if args_cli.disable_nic_collisions:
        disabled_count = _disable_asset_collisions(env, "nic_card")
        print(f"[INFO] Disabled {disabled_count} collision prims under nic_card.")

    targets = oracle.make_simple_nic_insert_targets(
        env,
        approach_offset_local=tuple(args_cli.approach_offset),
        use_tooth_top_alignment=not args_cli.disable_tooth_top_alignment,
        plug_frame_offset_w=tuple(args_cli.plug_frame_offset),
        plug_frame_rpy_offset_deg=tuple(args_cli.plug_frame_rpy_offset_deg),
        target_offset_local=tuple(args_cli.target_offset),
    )
    world_targets = oracle.compute_simple_nic_insert_world_targets(env, targets)
    state = oracle.make_simple_nic_insert_state(
        env,
        plug_frame_offset_w=tuple(args_cli.plug_frame_offset),
        plug_frame_rpy_offset_deg=tuple(args_cli.plug_frame_rpy_offset_deg),
    )

    print(f"[INFO] Running simple NIC insert oracle on {args_cli.task}")
    print(f"[INFO] Action scale: {action_scale[0].detach().cpu().tolist()}")
    if not args_cli.disable_start_joint_reset:
        print(f"[INFO] start_joint_pos: {tuple(args_cli.start_joint_pos)}")
    print("[INFO] orientation mapping: tip +Y -> port -Z, tip +Z -> port -Y, tip +X -> port -X")
    print(
        "[INFO] insert orientation gate: "
        f"{args_cli.insert_orientation_threshold_deg:.2f} deg, "
        f"misaligned_pos_scale={args_cli.insert_misaligned_pos_scale:.3f}, "
        f"alignment_only={args_cli.insert_alignment_only}, "
        f"insert_lateral_correction_scale={args_cli.insert_lateral_correction_scale:.3f}, "
        f"insert_rot_scale={args_cli.insert_rot_scale:.3f}, "
        f"lookahead={args_cli.insert_lookahead:.4f} m, "
        f"recenter_backoff={args_cli.insert_recenter_backoff:.4f} m"
    )
    print(f"[INFO] approach_offset in sfp_port_0_link frame: {tuple(args_cli.approach_offset)}")
    print(
        f"[INFO] target_offset in sfp_port_0_link frame: {tuple(args_cli.target_offset)}, "
        f"tooth_top_alignment={not args_cli.disable_tooth_top_alignment}"
    )
    print(
        "[INFO] insertion reference in sfp_tip_link frame: "
        f"{state.insertion_ref_pos_tip[0].detach().cpu().tolist()} "
        "(midpoint of sfp_tip_side_left/right)"
    )
    print(
        "[INFO] calibrated plug frame offsets: "
        f"base_pos_tip={state.plug_frame_pos_tip[0].detach().cpu().tolist()}, "
        f"world_offset={state.plug_frame_offset_w[0].detach().cpu().tolist()}, "
        f"rpy_offset_deg={tuple(args_cli.plug_frame_rpy_offset_deg)}"
    )
    print(f"[INFO] cached seat_pos_root[0]: {targets.seat_pos_root[0].detach().cpu().tolist()}")
    print(f"[INFO] live seat_w[0]: {world_targets.seat_w[0].detach().cpu().tolist()}")
    print(f"[INFO] live approach_w[0]: {world_targets.approach_w[0].detach().cpu().tolist()}")

    point_logger = None
    if not args_cli.disable_point_log:
        point_logger = _PointPositionLogger(
            env,
            env_index=args_cli.point_log_env_index,
            output_path=args_cli.point_log_path,
            offsets={
                "approach_offset": tuple(args_cli.approach_offset),
                "target_offset": tuple(args_cli.target_offset),
                "plug_frame_offset_world": tuple(args_cli.plug_frame_offset),
                "plug_frame_rpy_offset_deg": tuple(args_cli.plug_frame_rpy_offset_deg),
                "tooth_top_alignment": not args_cli.disable_tooth_top_alignment,
            },
        )
        print(f"[INFO] Writing point-position log to: {point_logger.path}")

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        try:
            for step in range(args_cli.max_episode_steps):
                output = oracle.compute_simple_nic_insert_oracle(
                    env,
                    action_scale,
                    targets,
                    state,
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
                phase = oracle.SimpleNicInsertPhase(int(output.phase[0])).name
                if point_logger is not None:
                    point_logger.write_step(step, phase, output, state)
                env.step(output.raw_action)

                if args_cli.log_every > 0 and step % args_cli.log_every == 0:
                    path_xy_msg = ""
                    if args_cli.log_path_xy_error:
                        xy_delta = output.tip_pos_w[0, :2] - output.target_tip_pos_w[0, :2]
                        xy_abs = torch.abs(xy_delta)
                        path_xy_msg = (
                            f" path_line_err={float(output.path_lateral_error[0]):.4f} m"
                            f" path_s={float(output.path_distance[0]):.4f} m"
                            f" target_s={float(output.target_path_distance[0]):.4f} m"
                            f" path_xy_err={float(output.lateral_xy_error[0]):.4f} m"
                            f" path_err_port={_fmt_vec(output.path_error_local[0])}"
                            f" abs_xy=({float(xy_abs[0]):.4f}, {float(xy_abs[1]):.4f})"
                            f" ori_err={math.degrees(float(output.orientation_error[0])):.2f} deg"
                            f" pos_scale={float(output.position_scale[0]):.2f}"
                            f" cmd_pos_b={_fmt_vec(output.processed_action[0, :3])}"
                            f" cmd_rot_b={float(torch.linalg.norm(output.processed_action[0, 3:6])):.4f}"
                        )
                    print(
                        f"[INFO] step={step:04d} phase={phase} "
                        f"tip_err={float(output.tip_error[0]):.4f} m "
                        f"insert={float(output.insert_fraction[0]) * 100.0:5.1f}% "
                        f"|a|={float(torch.linalg.norm(output.raw_action[0])):.3f}"
                        f"{path_xy_msg}"
                    )

                hold_done = (
                    int(state.phase[0]) == int(oracle.SimpleNicInsertPhase.HOLD)
                    and int(state.hold_steps[0]) >= args_cli.hold_steps
                )
                if hold_done:
                    print(f"[INFO] Hold complete after {step + 1} steps.")
                    break

                if env.sim.is_stopped():
                    break
                rate_limiter.sleep(env)
        finally:
            if point_logger is not None:
                point_logger.close()

    env.close()


def _env_step_dt(env: gym.Env) -> float:
    step_dt = getattr(env, "step_dt", None)
    if step_dt is not None and float(step_dt) > 0.0:
        return float(step_dt)
    return 1.0 / max(1, args_cli.step_hz)


class _PointPositionLogger:
    """Write live world positions of port/module debug points as JSONL."""

    def __init__(
        self,
        env: gym.Env,
        *,
        env_index: int,
        output_path: str | None,
        offsets: dict,
    ):
        self.env = env
        self.env_index = min(max(0, env_index), env.num_envs - 1)
        self.robot = env.scene["robot"]
        self.nic_card = env.scene["nic_card"]
        self.tcp_id = _first_body_id(self.robot, "gripper_tcp")
        self.tip_id = _first_body_id(self.robot, "sfp_tip_link")
        self.port_local_positions = self._resolve_port_local_positions()
        self.tip_local_positions = self._resolve_tip_local_positions()

        if output_path is None:
            logs_dir = Path("logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(logs_dir / f"nic_card_insert_points_{timestamp}.jsonl")
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self._write_json(
            {
                "type": "metadata",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "task": args_cli.task,
                "env_index": self.env_index,
                "coordinate_frame": "world",
                "offsets": offsets,
                "port_points": list(PORT_POINT_PATHS.keys()),
                "tip_points": list(TIP_POINT_NAMES.keys()),
                "tcp_point": "gripper_tcp",
                "derived_points": ["calibrated_plug_frame", "target_plug_frame"],
            }
        )

    def write_step(self, step: int, phase: str, output, state) -> None:
        self._write_json(
            {
                "type": "step",
                "step": step,
                "phase": phase,
                "port": self._port_world_positions(),
                "tip": self._tip_world_positions(),
                "tcp_gripper": _tensor_list(self.robot.data.body_pos_w[self.env_index, self.tcp_id]),
                "derived": {
                    "calibrated_plug_frame": _tensor_list(output.tip_pos_w[self.env_index]),
                    "target_plug_frame": _tensor_list(output.target_tip_pos_w[self.env_index]),
                },
                "diagnostics": {
                    "tip_error": float(output.tip_error[self.env_index]),
                    "path_lateral_error": float(output.path_lateral_error[self.env_index]),
                    "path_distance": float(output.path_distance[self.env_index]),
                    "target_path_distance": float(output.target_path_distance[self.env_index]),
                    "orientation_error_deg": math.degrees(float(output.orientation_error[self.env_index])),
                    "plug_frame_offset_world": _tensor_list(state.plug_frame_offset_w[self.env_index]),
                },
            }
        )

    def close(self) -> None:
        self._file.close()

    def _write_json(self, payload: dict) -> None:
        self._file.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._file.flush()

    def _port_world_positions(self) -> dict[str, list[float]]:
        card_pos = self.nic_card.data.root_pos_w[self.env_index]
        card_quat = self.nic_card.data.root_quat_w[self.env_index]
        return {
            name: _tensor_list(card_pos + math_utils.quat_apply(card_quat.unsqueeze(0), local_pos.unsqueeze(0))[0])
            for name, local_pos in self.port_local_positions.items()
        }

    def _tip_world_positions(self) -> dict[str, list[float]]:
        tip_pos = self.robot.data.body_pos_w[self.env_index, self.tip_id]
        tip_quat = self.robot.data.body_quat_w[self.env_index, self.tip_id]
        positions = {"sfp_tip_link": _tensor_list(tip_pos)}
        for name, local_pos in self.tip_local_positions.items():
            positions[name] = _tensor_list(tip_pos + math_utils.quat_apply(tip_quat.unsqueeze(0), local_pos.unsqueeze(0))[0])
        return positions

    def _resolve_port_local_positions(self) -> dict[str, torch.Tensor]:
        from aic_task.geometry.runtime import resolve_asset_root_prim_path

        root_path = resolve_asset_root_prim_path(self.nic_card, self.env_index)
        return {
            name: _prim_position_in_root(root_path, prim_path, dtype=self.nic_card.data.root_pos_w.dtype, device=self.env.device)
            for name, prim_path in PORT_POINT_PATHS.items()
        }

    def _resolve_tip_local_positions(self) -> dict[str, torch.Tensor]:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        roots = _asset_search_roots(self.robot, self.env_index)
        tip_prim = _find_prim_by_basename(stage, roots, "sfp_tip_link")
        if tip_prim is None:
            raise KeyError(f"Could not find sfp_tip_link under robot roots: {roots}")
        cache = UsdGeom.XformCache()
        tip_inv = cache.GetLocalToWorldTransform(tip_prim).GetInverse()
        positions = {}
        for name, basename in TIP_POINT_NAMES.items():
            if name == "sfp_tip_link":
                continue
            child_prim = _find_prim_by_basename(stage, roots, basename)
            if child_prim is None:
                raise KeyError(f"Could not find {basename} under robot roots: {roots}")
            child_w = cache.GetLocalToWorldTransform(child_prim).ExtractTranslation()
            child_tip = tip_inv.Transform(child_w)
            positions[name] = torch.tensor(
                (float(child_tip[0]), float(child_tip[1]), float(child_tip[2])),
                dtype=self.robot.data.body_pos_w.dtype,
                device=self.env.device,
            )
        return positions


def _first_body_id(robot, body_name: str) -> int:
    body_ids = robot.find_bodies(body_name, preserve_order=True)[0]
    if len(body_ids) == 0:
        available = ", ".join(getattr(robot, "body_names", []))
        raise KeyError(f"Robot body '{body_name}' not found. Available robot bodies: {available}")
    return int(body_ids[0])


def _prim_position_in_root(root_path: str, prim_path: str, *, dtype: torch.dtype, device: str) -> torch.Tensor:
    import omni.usd
    from pxr import UsdGeom

    from aic_task.geometry.runtime import _candidate_prim_paths, _resolve_prim

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    _, prim = _resolve_prim(stage, root_path, prim_path)
    if not root_prim.IsValid() or not prim.IsValid():
        candidates = ", ".join(_candidate_prim_paths(root_path, prim_path))
        raise KeyError(f"USD prim '{prim_path}' was not found. Tried: {candidates}.")
    cache = UsdGeom.XformCache()
    root_matrix = cache.GetLocalToWorldTransform(root_prim)
    prim_matrix = cache.GetLocalToWorldTransform(prim)
    local = root_matrix.GetInverse().Transform(prim_matrix.ExtractTranslation())
    return torch.tensor((float(local[0]), float(local[1]), float(local[2])), dtype=dtype, device=device)


def _asset_search_roots(asset, env_index: int) -> list[str]:
    prim_paths = list(getattr(asset.root_physx_view, "prim_paths", []))
    if not prim_paths:
        return []
    prim_path = str(prim_paths[min(env_index, len(prim_paths) - 1)])
    roots = [prim_path]
    parent = prim_path
    for _ in range(3):
        if "/" not in parent.rstrip("/"):
            break
        parent = parent.rstrip("/").rsplit("/", 1)[0]
        if parent and parent not in roots:
            roots.append(parent)
    return roots


def _find_prim_by_basename(stage, roots: list[str], basename: str):
    from aic_task.geometry.runtime import _candidate_prim_paths

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


def _tensor_list(values: torch.Tensor) -> list[float]:
    return [float(value) for value in values.detach().cpu().tolist()]


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


def _disable_asset_collisions(env: gym.Env, asset_name: str) -> int:
    import omni.usd
    from pxr import UsdPhysics

    from aic_task.geometry.runtime import resolve_asset_root_prim_path

    stage = omni.usd.get_context().get_stage()
    asset = env.scene[asset_name]
    disabled_count = 0
    root_paths = [resolve_asset_root_prim_path(asset, env_index) for env_index in range(env.num_envs)]
    for root_path in root_paths:
        root_prefix = root_path.rstrip("/") + "/"
        for prim in stage.Traverse():
            prim_path = prim.GetPath().pathString
            if prim_path != root_path and not prim_path.startswith(root_prefix):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            collision_api = UsdPhysics.CollisionAPI(prim)
            collision_api.GetCollisionEnabledAttr().Set(False)
            disabled_count += 1
    return disabled_count


def _fmt_vec(values: torch.Tensor) -> str:
    values_cpu = values.detach().cpu()
    return "(" + ", ".join(f"{float(value):+.4f}" for value in values_cpu) + ")"


if __name__ == "__main__":
    main()
    simulation_app.close()
