"""Run a direct-pose NIC-card SFP insertion demo oracle."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Simple demo oracle for AIC-Port-Insertion-v0.")
parser.add_argument("--task", type=str, default="AIC-Port-Insertion-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--step_hz", type=int, default=30)
parser.add_argument("--max_episode_steps", type=int, default=1200)
parser.add_argument(
    "--assume_port_visible",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Start the approach plan immediately. Later this can be driven by camera keypoint visibility.",
)
parser.add_argument("--approach_nominal_speed", type=float, default=0.08, help="Quintic approach duration speed in m/s.")
parser.add_argument(
    "--approach_end_speed",
    type=float,
    default=0.005,
    help="Desired approach endpoint velocity along the nominal insertion axis in m/s.",
)
parser.add_argument("--approach_min_duration", type=float, default=1.0, help="Minimum quintic approach duration in seconds.")
parser.add_argument("--approach_max_duration", type=float, default=5.0, help="Maximum quintic approach duration in seconds.")
parser.add_argument("--approach_rot_speed_deg", type=float, default=30.0, help="Delayed approach rotation speed in deg/s.")
parser.add_argument("--approach_rot_min_duration", type=float, default=0.5, help="Minimum nonzero approach rotation duration.")
parser.add_argument("--approach_rot_margin", type=float, default=0.25, help="Seconds before approach end for rotation to finish.")
parser.add_argument("--approach_threshold", type=float, default=0.015)
parser.add_argument(
    "--insert_lateral_threshold",
    type=float,
    default=0.010,
    help="Max perpendicular error to the insertion line before insertion depth is allowed to advance, meters.",
)
parser.add_argument(
    "--insert_orientation_threshold_deg",
    type=float,
    default=4.0,
    help="Max tip orientation error before insertion depth is allowed to advance, degrees.",
)
parser.add_argument(
    "--insert_lookahead",
    type=float,
    default=0.002,
    help="Minimum forward lookahead in meters while INSERT is aligned.",
)
parser.add_argument("--final_threshold", type=float, default=0.003)
parser.add_argument("--insert_speed", type=float, default=0.010, help="Target insertion speed in m/s.")
parser.add_argument("--pos_gain", type=float, default=1.2)
parser.add_argument("--rot_gain", type=float, default=0.2)
parser.add_argument("--max_pos_delta", type=float, default=0.020)
parser.add_argument("--insert_max_pos_delta", type=float, default=0.02)
parser.add_argument("--max_rot_delta", type=float, default=2.5)
parser.add_argument("--log_every", type=int, default=5, help="0 disables periodic logging.")
parser.add_argument(
    "--log_path_xy_error",
    action="store_true",
    default=False,
    help="Log world-XY lateral error to the current insertion path target, ignoring Z.",
)
parser.add_argument("--stream", action="store_true", default=False)
parser.add_argument(
    "--point_log_path",
    type=str,
    default=None,
    help="JSONL path for per-step port/tip/TCP point positions. Default writes to logs/ with a timestamp.",
)
parser.add_argument(
    "--disable_point_log",
    action="store_true",
    default=True,
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
    default=(0.55, -1.3642, -1.6648, -1.6933, 1.5710, 1.4110),
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
import json
import math
from pathlib import Path
import time

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import aic_task.controllers.nic_card_insert_oracle as nic_card_insert_oracle
import aic_task.tasks  # noqa: F401

def _port_point_paths(port_index: int) -> dict[str, str]:
    port = f"sfp_port_{port_index}"
    return {
        f"{port}_link": f"/{port}_link",
        f"{port}_link_entrance": f"/{port}_link/{port}_link_entrance",
    }

TIP_POINT_NAMES = {
    "sfp_tip_link": "sfp_tip_link",
    "sfp_tip_front_left": "sfp_tip_front_left",
    "sfp_tip_front_right": "sfp_tip_front_right",
}


def _oracle_module():
    """Return the active NIC insertion oracle module."""
    return nic_card_insert_oracle


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
    if args_cli.approach_nominal_speed <= 0.0:
        raise ValueError(f"--approach_nominal_speed must be positive, got {args_cli.approach_nominal_speed}.")
    if args_cli.approach_min_duration <= 0.0:
        raise ValueError(f"--approach_min_duration must be positive, got {args_cli.approach_min_duration}.")
    if args_cli.approach_max_duration < args_cli.approach_min_duration:
        raise ValueError("--approach_max_duration must be >= --approach_min_duration.")
    if args_cli.approach_rot_speed_deg <= 0.0:
        raise ValueError(f"--approach_rot_speed_deg must be positive, got {args_cli.approach_rot_speed_deg}.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=args_cli.use_fabric,
    )
    env_cfg.env_name = args_cli.task.split(":")[-1]
    _disable_rewards(env_cfg)

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
    goal = oracle.get_insertion_goal(env)
    world_targets = oracle.compute_simple_nic_insert_world_targets(env)
    state = oracle.make_simple_nic_insert_state(env)

    print(f"[INFO] Running simple NIC insert oracle on {args_cli.task}")
    print(f"[INFO] Action scale: {action_scale[0].detach().cpu().tolist()}")
    if not args_cli.disable_start_joint_reset:
        print(f"[INFO] start_joint_pos: {tuple(args_cli.start_joint_pos)}")
    print(f"[INFO] selected port: {goal.cfg.port_name}_link")
    print("[INFO] target mapping: command sfp_tip_link pose -> selected sfp_port_N_link pose")
    print(f"[INFO] assume_port_visible: {args_cli.assume_port_visible}")
    print(
        "[INFO] quintic approach: "
        f"nominal_speed={args_cli.approach_nominal_speed:.4f} m/s, "
        f"end_speed={args_cli.approach_end_speed:.4f} m/s, "
        f"duration=[{args_cli.approach_min_duration:.2f}, {args_cli.approach_max_duration:.2f}] s"
    )
    print(
        "[INFO] delayed approach rotation: "
        f"speed={args_cli.approach_rot_speed_deg:.2f} deg/s, "
        f"min_duration={args_cli.approach_rot_min_duration:.2f} s, "
        f"finish_margin={args_cli.approach_rot_margin:.2f} s"
    )
    print(
        "[INFO] insert orientation gate: "
        f"{args_cli.insert_orientation_threshold_deg:.2f} deg, "
        f"lookahead={args_cli.insert_lookahead:.4f} m"
    )
    print(f"[INFO] approach_offset in selected port frame: {tuple(goal.cfg.approach_offset_local)}")
    print(f"[INFO] approach_pos_noise_local: {tuple(goal.cfg.approach_pos_noise_local)}")
    print(
        "[INFO] approach orientation noise: "
        f"tilt={goal.cfg.approach_tilt_noise_deg:.2f} deg, "
        f"twist={goal.cfg.approach_twist_noise_deg:.2f} deg"
    )
    print(f"[INFO] target_xz_offset in selected port X/Z plane: {tuple(goal.cfg.target_xz_offset)}")
    print(
        f"[INFO] live nominal_approach_tip_target_w[0]: "
        f"{world_targets.nominal_approach_tip_pos_w[0].detach().cpu().tolist()}"
    )
    print(f"[INFO] live approach_tip_target_w[0]: {world_targets.approach_tip_pos_w[0].detach().cpu().tolist()}")
    print(f"[INFO] live approach_tip_quat_w[0]: {world_targets.approach_tip_quat_w[0].detach().cpu().tolist()}")
    print(f"[INFO] live final_tip_target_w[0]: {world_targets.final_tip_pos_w[0].detach().cpu().tolist()}")

    point_logger = None
    if not args_cli.disable_point_log:
        point_logger = _PointPositionLogger(
            env,
            env_index=args_cli.point_log_env_index,
            output_path=args_cli.point_log_path,
            offsets={
                "approach_offset": tuple(goal.cfg.approach_offset_local),
                "approach_pos_noise_local": tuple(goal.cfg.approach_pos_noise_local),
                "approach_tilt_noise_deg": goal.cfg.approach_tilt_noise_deg,
                "approach_twist_noise_deg": goal.cfg.approach_twist_noise_deg,
                "target_xz_offset": tuple(goal.cfg.target_xz_offset),
                "port_index": goal.cfg.port_index,
            },
            port_index=goal.cfg.port_index,
        )
        print(f"[INFO] Writing point-position log to: {point_logger.path}")

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        try:
            for step in range(args_cli.max_episode_steps):
                output = oracle.compute_simple_nic_insert_oracle(
                    env,
                    action_scale,
                    state,
                    pos_gain=args_cli.pos_gain,
                    rot_gain=args_cli.rot_gain,
                    max_pos_delta=args_cli.max_pos_delta,
                    insert_max_pos_delta=args_cli.insert_max_pos_delta,
                    max_rot_delta=args_cli.max_rot_delta,
                    port_visible=args_cli.assume_port_visible,
                    approach_nominal_speed=args_cli.approach_nominal_speed,
                    approach_end_speed=args_cli.approach_end_speed,
                    approach_min_duration=args_cli.approach_min_duration,
                    approach_max_duration=args_cli.approach_max_duration,
                    approach_rot_speed=math.radians(args_cli.approach_rot_speed_deg),
                    approach_rot_min_duration=args_cli.approach_rot_min_duration,
                    approach_rot_margin=args_cli.approach_rot_margin,
                    approach_threshold=args_cli.approach_threshold,
                    insert_lateral_threshold=args_cli.insert_lateral_threshold,
                    insert_orientation_threshold=math.radians(args_cli.insert_orientation_threshold_deg),
                    insert_lookahead=args_cli.insert_lookahead,
                    final_threshold=args_cli.final_threshold,
                    insert_speed=args_cli.insert_speed,
                    step_dt=_env_step_dt(env),
                )
                if point_logger is not None:
                    logged_phase = oracle.SimpleNicInsertPhase(int(output.phase[point_logger.env_index])).name
                    point_logger.write_step(step, logged_phase, output)
                _, _, terminated, time_outs, _ = env.step(output.raw_action)
                reset_mask = terminated | time_outs
                if bool(torch.any(reset_mask)):
                    reset_env_ids = reset_mask.nonzero(as_tuple=False).squeeze(-1)
                    oracle.reset_simple_nic_insert_state(env, state, reset_env_ids)
                    reset_terms = _reset_term_summary(env, reset_env_ids)
                    print(
                        f"[INFO] reset envs={reset_env_ids.detach().cpu().tolist()} "
                        f"terminated={int(torch.count_nonzero(terminated).item())} "
                        f"time_outs={int(torch.count_nonzero(time_outs).item())} "
                        f"terms={reset_terms}"
                    )

                if args_cli.log_every > 0 and step % args_cli.log_every == 0:
                    phase0 = oracle.SimpleNicInsertPhase(int(output.phase[0])).name
                    phase_summary = _phase_summary(oracle, output.phase)
                    path_xy_msg = ""
                    if args_cli.log_path_xy_error:
                        xy_delta = output.tip_pos_w[0, :2] - output.target_tip_pos_w[0, :2]
                        xy_abs = torch.abs(xy_delta)
                        path_xy_msg = (
                            f" path_line_err={float(output.path_lateral_error[0]):.4f} m"
                            f" path_s={float(output.path_distance[0]):.4f} m"
                            f" target_s={float(output.target_path_distance[0]):.4f} m"
                            f" path_err_port={_fmt_vec(output.path_error_local[0])}"
                            f" abs_xy=({float(xy_abs[0]):.4f}, {float(xy_abs[1]):.4f})"
                            f" ori_err={math.degrees(float(output.orientation_error[0])):.2f} deg"
                            f" x_axis_err={math.degrees(float(output.x_axis_error[0])):.2f} deg"
                            f" y_axis_err={math.degrees(float(output.y_axis_error[0])):.2f} deg"
                            f" z_axis_err={math.degrees(float(output.z_axis_error[0])):.2f} deg"
                            f" tcp_pos_err={float(output.tcp_position_error[0]):.4f} m"
                            f" tcp_ori_err={math.degrees(float(output.tcp_orientation_error[0])):.2f} deg"
                            f" cmd_pos_b={_fmt_vec(output.processed_action[0, :3])}"
                            f" cmd_rot_b={float(torch.linalg.norm(output.processed_action[0, 3:6])):.4f}"
                        )
                    print(
                        f"[INFO] step={step:04d} env0_phase={phase0} phases={phase_summary} "
                        f"tip_err={float(output.tip_position_error[0]):.4f} m "
                        f"ori_err={math.degrees(float(output.orientation_error[0])):.2f} deg "
                        f"insert={float(output.insert_fraction[0]) * 100.0:5.1f}% "
                        f"|a|={float(torch.linalg.norm(output.raw_action[0])):.3f}"
                        f"{path_xy_msg}"
                    )

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


def _phase_summary(oracle, phases: torch.Tensor) -> str:
    counts = []
    for phase in oracle.SimpleNicInsertPhase:
        count = int(torch.count_nonzero(phases == int(phase)).item())
        counts.append(f"{phase.name}={count}")
    return "[" + ", ".join(counts) + "]"


def _reset_term_summary(env: gym.Env, reset_env_ids: torch.Tensor) -> str:
    terms = []
    for name in getattr(env.termination_manager, "active_terms", []):
        values = env.termination_manager.get_term(name)[reset_env_ids]
        count = int(torch.count_nonzero(values).item())
        if count > 0:
            terms.append(f"{name}={count}")
    return "[" + ", ".join(terms) + "]"


class _PointPositionLogger:
    """Write live world positions of port/module debug points as JSONL."""

    def __init__(
        self,
        env: gym.Env,
        *,
        env_index: int,
        output_path: str | None,
        offsets: dict,
        port_index: int,
    ):
        self.env = env
        self.env_index = min(max(0, env_index), env.num_envs - 1)
        self.port_index = port_index
        self.robot = env.scene["robot"]
        self.nic_card = env.scene["nic_card"]
        self.goal = env.command_manager.get_term("insertion_goal")
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
                "port_points": list(_port_point_paths(self.port_index).keys()),
                "tip_points": list(TIP_POINT_NAMES.keys()),
                "tcp_point": "gripper_tcp",
                "derived_points": ["target_tip", "nominal_approach_tip", "randomized_approach_tip", "final_tip"],
            }
        )

    def write_step(self, step: int, phase: str, output) -> None:
        self._write_json(
            {
                "type": "step",
                "step": step,
                "phase": phase,
                "port": self._port_world_positions(),
                "tip": self._tip_world_positions(),
                "tcp_gripper": _tensor_list(self.robot.data.body_pos_w[self.env_index, self.tcp_id]),
                "derived": {
                    "target_tip": _tensor_list(output.target_tip_pos_w[self.env_index]),
                    "nominal_approach_tip": _tensor_list(self.goal.nominal_approach_tip_pos_w[self.env_index]),
                    "randomized_approach_tip": _tensor_list(self.goal.approach_tip_pos_w[self.env_index]),
                    "final_tip": _tensor_list(self.goal.final_tip_pos_w[self.env_index]),
                },
                "diagnostics": {
                    "tip_position_error": float(output.tip_position_error[self.env_index]),
                    "path_lateral_error": float(output.path_lateral_error[self.env_index]),
                    "path_distance": float(output.path_distance[self.env_index]),
                    "target_path_distance": float(output.target_path_distance[self.env_index]),
                    "orientation_error_deg": math.degrees(float(output.orientation_error[self.env_index])),
                    "x_axis_error_deg": math.degrees(float(output.x_axis_error[self.env_index])),
                    "y_axis_error_deg": math.degrees(float(output.y_axis_error[self.env_index])),
                    "z_axis_error_deg": math.degrees(float(output.z_axis_error[self.env_index])),
                    "tcp_position_error": float(output.tcp_position_error[self.env_index]),
                    "tcp_orientation_error_deg": math.degrees(float(output.tcp_orientation_error[self.env_index])),
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
        root_path = _resolve_asset_root_prim_path(self.nic_card, self.env_index)
        return {
            name: _prim_position_in_root(root_path, prim_path, dtype=self.nic_card.data.root_pos_w.dtype, device=self.env.device)
            for name, prim_path in _port_point_paths(self.port_index).items()
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


def _resolve_asset_root_prim_path(asset, env_index: int, *, usd_root_child: str = "nic_card_link") -> str:
    """Return the USD instance root path without importing old geometry helpers."""
    prim_paths = list(getattr(asset.root_physx_view, "prim_paths", []))
    if not prim_paths:
        cfg_path = getattr(getattr(asset, "cfg", None), "prim_path", "")
        raise RuntimeError(f"Cannot resolve prim paths for asset with cfg path '{cfg_path}'.")

    index = min(env_index, len(prim_paths) - 1)
    prim_path = str(prim_paths[index])
    suffix = f"/{usd_root_child}"
    if prim_path.endswith(suffix):
        return prim_path[: -len(suffix)]
    return prim_path


def _join_prim_path(asset_root_path: str, relative_path: str) -> str:
    return f"{asset_root_path.rstrip('/')}/{relative_path.lstrip('/')}"


def _resolve_prim(stage, asset_root_path: str, relative_path: str):
    """Resolve a prim by exact candidate paths, then by descendant basename."""
    for prim_path in _candidate_prim_paths(asset_root_path, relative_path):
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            return prim_path, prim

    basename = relative_path.rstrip("/").rsplit("/", 1)[-1]
    root_prefixes = _asset_root_prefixes(asset_root_path)
    for prim in stage.Traverse():
        prim_path = prim.GetPath().pathString
        if not any(prim_path == prefix or prim_path.startswith(prefix + "/") for prefix in root_prefixes):
            continue
        if prim_path.rsplit("/", 1)[-1] == basename:
            return prim_path, prim
    return _candidate_prim_paths(asset_root_path, relative_path)[0], stage.GetPrimAtPath("/__missing__")


def _candidate_prim_paths(asset_root_path: str, relative_path: str) -> list[str]:
    """Return path candidates for assets spawned with or without defaultPrim nesting."""
    relative = relative_path.lstrip("/")
    candidates = [_join_prim_path(asset_root_path, relative)]
    if relative.startswith("nic_card_link/"):
        candidates.append(_join_prim_path(asset_root_path, relative.removeprefix("nic_card_link/")))
    if asset_root_path.endswith("/nic_card_link"):
        parent_root = asset_root_path.removesuffix("/nic_card_link")
        candidates.append(_join_prim_path(parent_root, relative))
        if relative.startswith("nic_card_link/"):
            candidates.append(_join_prim_path(parent_root, relative.removeprefix("nic_card_link/")))
    return list(dict.fromkeys(candidates))


def _asset_root_prefixes(asset_root_path: str) -> tuple[str, ...]:
    prefixes = [asset_root_path.rstrip("/")]
    if prefixes[0].endswith("/nic_card_link"):
        prefixes.append(prefixes[0].removesuffix("/nic_card_link"))
    else:
        prefixes.append(prefixes[0] + "/nic_card_link")
    return tuple(dict.fromkeys(prefixes))


def _disable_rewards(env_cfg) -> None:
    for name, value in vars(env_cfg.rewards).items():
        if name.startswith("_") or value is None:
            continue
        setattr(env_cfg.rewards, name, None)


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


def _fmt_vec(values: torch.Tensor) -> str:
    values_cpu = values.detach().cpu()
    return "(" + ", ".join(f"{float(value):+.4f}" for value in values_cpu) + ")"


if __name__ == "__main__":
    main()
    simulation_app.close()
