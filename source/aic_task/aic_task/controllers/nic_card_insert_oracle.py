"""Simple demo oracle for inserting the SFP module into NIC port 0.

This controller is intentionally small and geometry-first.  It reads the fixed
``sfp_port_0_link`` pose once, moves the SFP tip to an offset approach point in
that port frame, then advances the target straight to the link origin.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils

from aic_task.geometry.runtime import (
    _candidate_prim_paths,
    _nearby_child_paths,
    _resolve_prim,
    resolve_asset_root_prim_path,
)


class SimpleNicInsertPhase(IntEnum):
    """Phases for the demo-only NIC card insertion oracle."""

    APPROACH = 0
    INSERT = 1
    HOLD = 2


@dataclass
class SimpleNicInsertTargets:
    """Fixed port targets resolved from USD after reset."""

    seat_w: torch.Tensor
    approach_w: torch.Tensor
    path_w: torch.Tensor
    path_length: torch.Tensor
    approach_offset_local: tuple[float, float, float]


@dataclass
class SimpleNicInsertState:
    """Mutable state for the simple insertion sequence."""

    phase: torch.Tensor
    insert_distance: torch.Tensor
    hold_steps: torch.Tensor
    desired_tcp_quat_w: torch.Tensor
    tip_pos_tcp: torch.Tensor


@dataclass
class SimpleNicInsertOracleOutput:
    """Action and diagnostics for one demo control step."""

    raw_action: torch.Tensor
    processed_action: torch.Tensor
    target_tip_pos_w: torch.Tensor
    tip_pos_w: torch.Tensor
    tip_error: torch.Tensor
    phase: torch.Tensor
    insert_fraction: torch.Tensor


def get_action_scale(env: gym.Env, action_dim: int) -> torch.Tensor:
    """Return the Differential IK action scale used to convert processed to raw actions."""
    action_term = env.action_manager.get_term("arm_action")
    scale = getattr(action_term, "_scale", None)
    if scale is None:
        return torch.ones((env.num_envs, action_dim), device=env.device)
    return scale[:, :action_dim]


def make_simple_nic_insert_targets(
    env: gym.Env,
    *,
    target_name: str = "nic_card",
    seat_path: str = "/sfp_port_0_link",
    approach_offset_local: tuple[float, float, float] = (0.0, -0.10, 0.0),
) -> SimpleNicInsertTargets:
    """Resolve the fixed port seat and approach points from live USD geometry.

    ``approach_offset_local`` is expressed in the ``sfp_port_0_link`` frame.
    Since insertion is along port-local ``+Y``, the default backs the plug out
    along ``-Y`` by 10 cm before moving to the seat.
    """
    target = env.scene[target_name]
    device = target.data.root_pos_w.device
    dtype = target.data.root_pos_w.dtype

    seat_points = []
    approach_points = []
    for env_index in range(env.num_envs):
        root_path = resolve_asset_root_prim_path(target, env_index)
        seat_w, approach_w = _seat_and_offset_point_w(
            root_path,
            seat_path,
            approach_offset_local,
        )
        seat_points.append(seat_w)
        approach_points.append(approach_w)

    seat_w = torch.tensor(seat_points, dtype=dtype, device=device)
    approach_w = torch.tensor(approach_points, dtype=dtype, device=device)
    path_w = seat_w - approach_w
    path_length = torch.linalg.norm(path_w, dim=1, keepdim=True).clamp_min(1.0e-9)
    return SimpleNicInsertTargets(
        seat_w=seat_w,
        approach_w=approach_w,
        path_w=path_w,
        path_length=path_length,
        approach_offset_local=approach_offset_local,
    )


def make_simple_nic_insert_state(
    env: gym.Env,
    *,
    tcp_body: str = "gripper_tcp",
    tip_body: str = "sfp_tip_link",
) -> SimpleNicInsertState:
    """Capture the initial TCP orientation and TCP-to-tip offset."""
    robot = env.scene["robot"]
    tcp_id = _first_body_id(robot, tcp_body)
    tip_id = _first_body_id(robot, tip_body)
    tcp_pos_w = robot.data.body_pos_w[:, tcp_id, :].clone()
    tcp_quat_w = robot.data.body_quat_w[:, tcp_id, :].clone()
    tip_pos_w = robot.data.body_pos_w[:, tip_id, :]
    tip_quat_w = robot.data.body_quat_w[:, tip_id, :]
    tip_pos_tcp, _ = math_utils.subtract_frame_transforms(
        tcp_pos_w,
        tcp_quat_w,
        tip_pos_w,
        tip_quat_w,
    )
    return SimpleNicInsertState(
        phase=torch.full(
            (env.num_envs,),
            int(SimpleNicInsertPhase.APPROACH),
            dtype=torch.long,
            device=env.device,
        ),
        insert_distance=torch.zeros((env.num_envs, 1), dtype=tcp_pos_w.dtype, device=env.device),
        hold_steps=torch.zeros((env.num_envs,), dtype=torch.long, device=env.device),
        desired_tcp_quat_w=tcp_quat_w,
        tip_pos_tcp=tip_pos_tcp,
    )


def compute_simple_nic_insert_oracle(
    env: gym.Env,
    action_scale: torch.Tensor,
    targets: SimpleNicInsertTargets,
    state: SimpleNicInsertState,
    *,
    tcp_body: str = "gripper_tcp",
    tip_body: str = "sfp_tip_link",
    pos_gain: float = 0.8,
    rot_gain: float = 0.5,
    max_pos_delta: float = 0.012,
    insert_max_pos_delta: float = 0.002,
    max_rot_delta: float = 0.025,
    approach_threshold: float = 0.006,
    final_threshold: float = 0.003,
    insert_speed: float = 0.010,
    step_dt: float = 1.0 / 30.0,
    hold_orientation: bool = True,
) -> SimpleNicInsertOracleOutput:
    """Compute one relative-IK action for the simple approach-then-insert demo."""
    robot = env.scene["robot"]
    tcp_id = _first_body_id(robot, tcp_body)
    tip_id = _first_body_id(robot, tip_body)
    tcp_pos_w = robot.data.body_pos_w[:, tcp_id, :]
    tcp_quat_w = robot.data.body_quat_w[:, tcp_id, :]
    tip_pos_w = robot.data.body_pos_w[:, tip_id, :]

    insert_mask = state.phase == int(SimpleNicInsertPhase.INSERT)
    state.insert_distance[insert_mask] += insert_speed * step_dt
    state.insert_distance[:] = torch.minimum(state.insert_distance, targets.path_length)
    insert_fraction = (state.insert_distance / targets.path_length).clamp(0.0, 1.0)
    insert_target_w = targets.approach_w + targets.path_w * insert_fraction

    target_tip_pos_w = torch.where(
        (state.phase == int(SimpleNicInsertPhase.APPROACH)).unsqueeze(1),
        targets.approach_w,
        insert_target_w,
    )
    target_tip_pos_w = torch.where(
        (state.phase == int(SimpleNicInsertPhase.HOLD)).unsqueeze(1),
        targets.seat_w,
        target_tip_pos_w,
    )

    tip_error = torch.linalg.norm(target_tip_pos_w - tip_pos_w, dim=1)
    reached_approach = (state.phase == int(SimpleNicInsertPhase.APPROACH)) & (tip_error <= approach_threshold)
    state.phase[reached_approach] = int(SimpleNicInsertPhase.INSERT)

    reached_final = (
        (state.phase == int(SimpleNicInsertPhase.INSERT))
        & (insert_fraction.squeeze(1) >= 1.0)
        & (tip_error <= final_threshold)
    )
    state.phase[reached_final] = int(SimpleNicInsertPhase.HOLD)
    state.hold_steps[state.phase == int(SimpleNicInsertPhase.HOLD)] += 1

    desired_tcp_quat_w = state.desired_tcp_quat_w if hold_orientation else tcp_quat_w
    desired_tcp_pos_w = target_tip_pos_w - math_utils.quat_apply(desired_tcp_quat_w, state.tip_pos_tcp)
    processed_action = _relative_ik_processed_action(
        robot,
        tcp_pos_w,
        tcp_quat_w,
        desired_tcp_pos_w,
        desired_tcp_quat_w,
        action_scale,
        pos_gain=pos_gain,
        rot_gain=rot_gain if hold_orientation else 0.0,
        max_pos_delta=insert_max_pos_delta if bool(insert_mask.any()) else max_pos_delta,
        max_rot_delta=max_rot_delta,
    )
    raw_action = processed_action / torch.clamp(action_scale, min=1.0e-9)
    return SimpleNicInsertOracleOutput(
        raw_action=raw_action,
        processed_action=processed_action,
        target_tip_pos_w=target_tip_pos_w,
        tip_pos_w=tip_pos_w,
        tip_error=tip_error,
        phase=state.phase.clone(),
        insert_fraction=insert_fraction.squeeze(1),
    )


def _seat_and_offset_point_w(
    asset_root_path: str,
    seat_path: str,
    offset_local: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return a prim translation and a point offset in that prim's local frame."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim_path, prim = _resolve_prim(stage, asset_root_path, seat_path)
    if not prim.IsValid():
        candidates = ", ".join(_candidate_prim_paths(asset_root_path, seat_path))
        children = ", ".join(_nearby_child_paths(stage, asset_root_path))
        raise KeyError(
            f"USD prim for '{seat_path}' was not found under '{asset_root_path}'. "
            f"Tried: {candidates}. Nearby children: {children}"
        )
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    seat = matrix.ExtractTranslation()
    offset_w = matrix.TransformDir(Gf.Vec3d(*offset_local))
    approach = seat + offset_w
    return (
        (float(seat[0]), float(seat[1]), float(seat[2])),
        (float(approach[0]), float(approach[1]), float(approach[2])),
    )


def _relative_ik_processed_action(
    robot,
    tcp_pos_w: torch.Tensor,
    tcp_quat_w: torch.Tensor,
    desired_tcp_pos_w: torch.Tensor,
    desired_tcp_quat_w: torch.Tensor,
    action_scale: torch.Tensor,
    *,
    pos_gain: float,
    rot_gain: float,
    max_pos_delta: float,
    max_rot_delta: float,
) -> torch.Tensor:
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
    return processed_action


def _first_body_id(robot, body_name: str) -> int:
    body_ids = robot.find_bodies(body_name, preserve_order=True)[0]
    if len(body_ids) == 0:
        available = ", ".join(getattr(robot, "body_names", []))
        raise KeyError(f"Robot body '{body_name}' not found. Available robot bodies: {available}")
    return int(body_ids[0])


def _clamp_vector_norm(vector: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.linalg.norm(vector, dim=1, keepdim=True)
    scale = torch.clamp(max_norm / torch.clamp(norm, min=1.0e-9), max=1.0)
    return vector * scale
