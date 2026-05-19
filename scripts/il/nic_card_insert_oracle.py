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
parser.add_argument("--approach_threshold", type=float, default=0.015)
parser.add_argument(
    "--insert_lateral_threshold",
    type=float,
    default=0.003,
    help="Max world-XY lateral error before insertion depth is allowed to advance, meters.",
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
import importlib.util
from pathlib import Path
import sys
import time

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import aic_task.tasks  # noqa: F401

_ORACLE = None


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

    targets = oracle.make_simple_nic_insert_targets(
        env,
        approach_offset_local=tuple(args_cli.approach_offset),
    )
    world_targets = oracle.compute_simple_nic_insert_world_targets(env, targets)
    state = oracle.make_simple_nic_insert_state(env)

    print(f"[INFO] Running simple NIC insert oracle on {args_cli.task}")
    print(f"[INFO] Action scale: {action_scale[0].detach().cpu().tolist()}")
    if not args_cli.disable_start_joint_reset:
        print(f"[INFO] start_joint_pos: {tuple(args_cli.start_joint_pos)}")
    print("[INFO] fixed tip roll correction: 180.00 deg")
    print(f"[INFO] approach_offset in sfp_port_0_link frame: {tuple(args_cli.approach_offset)}")
    print(f"[INFO] cached seat_pos_root[0]: {targets.seat_pos_root[0].detach().cpu().tolist()}")
    print(f"[INFO] live seat_w[0]: {world_targets.seat_w[0].detach().cpu().tolist()}")
    print(f"[INFO] live approach_w[0]: {world_targets.approach_w[0].detach().cpu().tolist()}")

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
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
                final_threshold=args_cli.final_threshold,
                insert_speed=args_cli.insert_speed,
                step_dt=_env_step_dt(env),
                hold_orientation=not args_cli.disable_orientation_hold,
            )
            env.step(output.raw_action)

            if args_cli.log_every > 0 and step % args_cli.log_every == 0:
                phase = oracle.SimpleNicInsertPhase(int(output.phase[0])).name
                path_xy_msg = ""
                if args_cli.log_path_xy_error:
                    xy_delta = output.tip_pos_w[0, :2] - output.target_tip_pos_w[0, :2]
                    xy_abs = torch.abs(xy_delta)
                    path_xy_msg = (
                        f" path_line_err={float(output.path_lateral_error[0]):.4f} m"
                        f" path_xy_err={float(output.lateral_xy_error[0]):.4f} m"
                        f" abs_xy=({float(xy_abs[0]):.4f}, {float(xy_abs[1]):.4f})"
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

    env.close()


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


def _fmt_vec(values: torch.Tensor) -> str:
    values_cpu = values.detach().cpu()
    return "(" + ", ".join(f"{float(value):+.4f}" for value in values_cpu) + ")"


if __name__ == "__main__":
    main()
    simulation_app.close()
