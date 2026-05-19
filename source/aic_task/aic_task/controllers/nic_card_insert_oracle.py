"""Simple demo oracle for inserting the SFP module into NIC port 0.

This controller is intentionally small and geometry-first.  It reads the fixed
``sfp_port_0_link`` pose once in the NIC-card root frame, then recomputes world
targets from the live NIC-card rigid body pose every control step.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

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
    """Fixed port geometry cached in the live NIC-card root frame."""

    seat_pos_root: torch.Tensor
    seat_quat_root: torch.Tensor
    approach_offset_local: tuple[float, float, float]


@dataclass
class SimpleNicInsertWorldTargets:
    """World-frame port targets computed from the current NIC-card pose."""

    seat_w: torch.Tensor
    approach_w: torch.Tensor
    path_w: torch.Tensor
    path_length: torch.Tensor
    seat_quat_w: torch.Tensor


@dataclass
class SimpleNicInsertState:
    """Mutable state for the simple insertion sequence."""

    phase: torch.Tensor
    insert_distance: torch.Tensor
    hold_steps: torch.Tensor
    desired_tcp_quat_w: torch.Tensor
    tip_pos_tcp: torch.Tensor
    tip_quat_tcp: torch.Tensor


@dataclass
class SimpleNicInsertOracleOutput:
    """Action and diagnostics for one demo control step."""

    raw_action: torch.Tensor
    processed_action: torch.Tensor
    target_tip_pos_w: torch.Tensor
    tip_pos_w: torch.Tensor
    tip_error: torch.Tensor
    lateral_xy_error: torch.Tensor
    path_lateral_error: torch.Tensor
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
    """Resolve the fixed port seat pose from USD in the NIC-card root frame.

    ``approach_offset_local`` is expressed in the ``sfp_port_0_link`` frame.
    Since insertion is along port-local ``+Y``, the default backs the plug out
    along ``-Y`` by 10 cm before moving to the seat.  World-frame targets are
    recomputed during control from ``env.scene[target_name].data.root_*_w``.
    """
    target = env.scene[target_name]
    device = target.data.root_pos_w.device
    dtype = target.data.root_pos_w.dtype

    seat_positions_root = []
    seat_quats_root = []
    for env_index in range(env.num_envs):
        root_path = resolve_asset_root_prim_path(target, env_index)
        seat_pos_root, seat_quat_root = _seat_pose_in_asset_root(root_path, seat_path)
        seat_positions_root.append(seat_pos_root)
        seat_quats_root.append(seat_quat_root)

    return SimpleNicInsertTargets(
        seat_pos_root=torch.tensor(seat_positions_root, dtype=dtype, device=device),
        seat_quat_root=torch.tensor(seat_quats_root, dtype=dtype, device=device),
        approach_offset_local=approach_offset_local,
    )


def compute_simple_nic_insert_world_targets(
    env: gym.Env,
    targets: SimpleNicInsertTargets,
    *,
    target_name: str = "nic_card",
) -> SimpleNicInsertWorldTargets:
    """Compute current world-frame seat and approach targets from live card state."""
    target = env.scene[target_name]
    seat_w = target.data.root_pos_w + math_utils.quat_apply(
        target.data.root_quat_w,
        targets.seat_pos_root,
    )
    seat_quat_w = math_utils.quat_mul(target.data.root_quat_w, targets.seat_quat_root)
    approach_offset = torch.tensor(
        targets.approach_offset_local,
        dtype=seat_w.dtype,
        device=seat_w.device,
    ).unsqueeze(0)
    approach_w = seat_w + math_utils.quat_apply(
        seat_quat_w,
        approach_offset.expand(env.num_envs, -1),
    )
    path_w = seat_w - approach_w
    path_length = torch.linalg.norm(path_w, dim=1, keepdim=True).clamp_min(1.0e-9)
    return SimpleNicInsertWorldTargets(
        seat_w=seat_w,
        approach_w=approach_w,
        path_w=path_w,
        path_length=path_length,
        seat_quat_w=seat_quat_w,
    )


def make_simple_nic_insert_state(
    env: gym.Env,
    *,
    tcp_body: str = "gripper_tcp",
    tip_body: str = "sfp_tip_link",
) -> SimpleNicInsertState:
    """Capture the initial controlled-body orientation and body-to-tip offset."""
    robot = env.scene["robot"]
    tcp_id = _first_body_id(robot, tcp_body)
    tip_id = _first_body_id(robot, tip_body)
    tcp_pos_w = robot.data.body_pos_w[:, tcp_id, :].clone()
    tcp_quat_w = robot.data.body_quat_w[:, tcp_id, :].clone()
    tip_pos_w = robot.data.body_pos_w[:, tip_id, :]
    tip_quat_w = robot.data.body_quat_w[:, tip_id, :]
    tip_pos_tcp, tip_quat_tcp = math_utils.subtract_frame_transforms(
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
        tip_quat_tcp=tip_quat_tcp,
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
    approach_threshold: float = 0.015,
    insert_lateral_threshold: float = 0.003,
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
    world_targets = compute_simple_nic_insert_world_targets(env, targets)

    insert_fraction, target_tip_pos_w = _current_target_tip_pos(world_targets, state)
    tip_error = torch.linalg.norm(target_tip_pos_w - tip_pos_w, dim=1)
    lateral_xy_error = _world_xy_error(tip_pos_w, target_tip_pos_w)
    _, _, path_lateral_error = _project_tip_to_path(world_targets, tip_pos_w)
    reached_approach = (state.phase == int(SimpleNicInsertPhase.APPROACH)) & (tip_error <= approach_threshold)
    state.phase[reached_approach] = int(SimpleNicInsertPhase.INSERT)

    insert_mask = state.phase == int(SimpleNicInsertPhase.INSERT)
    advance_mask = insert_mask & (path_lateral_error <= insert_lateral_threshold)
    state.insert_distance[advance_mask] += insert_speed * step_dt
    state.insert_distance[:] = torch.minimum(state.insert_distance, world_targets.path_length)

    insert_fraction, target_tip_pos_w = _current_target_tip_pos(world_targets, state)
    tip_error = torch.linalg.norm(target_tip_pos_w - tip_pos_w, dim=1)
    lateral_xy_error = _world_xy_error(tip_pos_w, target_tip_pos_w)
    _, _, path_lateral_error = _project_tip_to_path(world_targets, tip_pos_w)

    reached_final = (
        (state.phase == int(SimpleNicInsertPhase.INSERT))
        & (insert_fraction.squeeze(1) >= 1.0)
        & (tip_error <= final_threshold)
    )
    state.phase[reached_final] = int(SimpleNicInsertPhase.HOLD)
    state.hold_steps[state.phase == int(SimpleNicInsertPhase.HOLD)] += 1

    desired_tip_quat_w = (
        _desired_tip_quat_from_port(world_targets.seat_quat_w)
        if hold_orientation
        else math_utils.quat_mul(tcp_quat_w, state.tip_quat_tcp)
    )
    desired_tcp_quat_w = _desired_tcp_quat_from_tip(desired_tip_quat_w, state.tip_quat_tcp)
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
        lateral_xy_error=lateral_xy_error,
        path_lateral_error=path_lateral_error,
        phase=state.phase.clone(),
        insert_fraction=insert_fraction.squeeze(1),
    )


def _current_target_tip_pos(
    world_targets: SimpleNicInsertWorldTargets,
    state: SimpleNicInsertState,
) -> tuple[torch.Tensor, torch.Tensor]:
    insert_fraction = (state.insert_distance / world_targets.path_length).clamp(0.0, 1.0)
    insert_target_w = world_targets.approach_w + world_targets.path_w * insert_fraction
    target_tip_pos_w = torch.where(
        (state.phase == int(SimpleNicInsertPhase.APPROACH)).unsqueeze(1),
        world_targets.approach_w,
        insert_target_w,
    )
    target_tip_pos_w = torch.where(
        (state.phase == int(SimpleNicInsertPhase.HOLD)).unsqueeze(1),
        world_targets.seat_w,
        target_tip_pos_w,
    )
    return insert_fraction, target_tip_pos_w


def _world_xy_error(actual_w: torch.Tensor, target_w: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(actual_w[:, :2] - target_w[:, :2], dim=1)


def _project_tip_to_path(
    world_targets: SimpleNicInsertWorldTargets,
    tip_pos_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    path_axis_w = world_targets.path_w / world_targets.path_length
    tip_from_approach = tip_pos_w - world_targets.approach_w
    path_distance = torch.sum(tip_from_approach * path_axis_w, dim=1, keepdim=True)
    path_distance = torch.clamp(path_distance, min=0.0)
    path_distance = torch.minimum(path_distance, world_targets.path_length)
    closest_w = world_targets.approach_w + path_axis_w * path_distance
    path_lateral_error = torch.linalg.norm(tip_pos_w - closest_w, dim=1)
    return path_distance, closest_w, path_lateral_error


def _desired_tcp_quat_from_tip(desired_tip_quat_w: torch.Tensor, tip_quat_tcp: torch.Tensor) -> torch.Tensor:
    return math_utils.quat_mul(desired_tip_quat_w, math_utils.quat_inv(tip_quat_tcp))


def _seat_pose_in_asset_root(
    asset_root_path: str,
    seat_path: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Return the seat prim pose relative to the asset root prim."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(asset_root_path)
    _, prim = _resolve_prim(stage, asset_root_path, seat_path)
    if not root_prim.IsValid() or not prim.IsValid():
        candidates = ", ".join(_candidate_prim_paths(asset_root_path, seat_path))
        children = ", ".join(_nearby_child_paths(stage, asset_root_path))
        raise KeyError(
            f"USD prim for root '{asset_root_path}' or seat '{seat_path}' was not found. "
            f"Tried: {candidates}. Nearby children: {children}"
        )
    cache = UsdGeom.XformCache()
    root_matrix = cache.GetLocalToWorldTransform(root_prim)
    seat_matrix = cache.GetLocalToWorldTransform(prim)
    root_inv = root_matrix.GetInverse()
    seat_w = seat_matrix.ExtractTranslation()
    seat_root = root_inv.Transform(seat_w)
    local_axes = []
    for axis in (Gf.Vec3d(1.0, 0.0, 0.0), Gf.Vec3d(0.0, 1.0, 0.0), Gf.Vec3d(0.0, 0.0, 1.0)):
        axis_root = root_inv.TransformDir(seat_matrix.TransformDir(axis))
        local_axes.append((float(axis_root[0]), float(axis_root[1]), float(axis_root[2])))
    rotation_matrix = torch.tensor(
        [
            [local_axes[0][0], local_axes[1][0], local_axes[2][0]],
            [local_axes[0][1], local_axes[1][1], local_axes[2][1]],
            [local_axes[0][2], local_axes[1][2], local_axes[2][2]],
        ],
        dtype=torch.float64,
    )
    seat_quat_root = math_utils.quat_from_matrix(rotation_matrix.unsqueeze(0))[0]
    return (
        (float(seat_root[0]), float(seat_root[1]), float(seat_root[2])),
        tuple(float(value) for value in seat_quat_root.tolist()),
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


def _desired_tip_quat_from_port(port_quat_w: torch.Tensor) -> torch.Tensor:
    """Return a tip orientation where tip local -Z follows port local +Y."""
    port_y = _local_axis_w(port_quat_w, (0.0, 1.0, 0.0))
    port_z = _local_axis_w(port_quat_w, (0.0, 0.0, 1.0))

    # Tip local -Z is the observed module insertion direction, so tip +Z must
    # point opposite the port insertion axis.
    target_z = -port_y
    target_y = _project_axis_onto_plane(port_z, target_z)
    target_x = torch.linalg.cross(target_y, target_z, dim=1)
    target_x = torch.nn.functional.normalize(target_x, dim=1)
    target_y = torch.linalg.cross(target_z, target_x, dim=1)
    target_y = torch.nn.functional.normalize(target_y, dim=1)

    # Empirically this tip frame needs a 180 degree roll about the insertion
    # axis after aligning -Z to the port axis.  This is the fixed correction
    # that was previously exposed as --tip_roll_offset_deg 180.
    angles = torch.full((port_quat_w.shape[0],), math.pi, dtype=port_quat_w.dtype, device=port_quat_w.device)
    q_roll = math_utils.quat_from_angle_axis(angles, port_y)
    target_x = math_utils.quat_apply(q_roll, target_x)
    target_y = math_utils.quat_apply(q_roll, target_y)
    target_z = math_utils.quat_apply(q_roll, target_z)

    return _quat_from_frame_axes(target_x, target_y, target_z)


def _local_axis_w(quat_w: torch.Tensor, axis_local: tuple[float, float, float]) -> torch.Tensor:
    axis = torch.tensor(axis_local, dtype=quat_w.dtype, device=quat_w.device).unsqueeze(0)
    return math_utils.quat_apply(quat_w, axis.expand(quat_w.shape[0], -1))


def _project_axis_onto_plane(axis: torch.Tensor, plane_normal: torch.Tensor) -> torch.Tensor:
    plane_normal = torch.nn.functional.normalize(plane_normal, dim=1)
    projected = axis - torch.sum(axis * plane_normal, dim=1, keepdim=True) * plane_normal
    norm = torch.linalg.norm(projected, dim=1, keepdim=True)
    fallback_axis = _orthogonal_axis(plane_normal)
    return torch.where(norm > 1.0e-6, projected / norm.clamp_min(1.0e-9), fallback_axis)


def _orthogonal_axis(vector: torch.Tensor) -> torch.Tensor:
    x_axis = torch.zeros_like(vector)
    x_axis[:, 0] = 1.0
    y_axis = torch.zeros_like(vector)
    y_axis[:, 1] = 1.0
    candidate = torch.where(torch.abs(vector[:, 0:1]) < 0.9, x_axis, y_axis)
    axis = torch.linalg.cross(vector, candidate, dim=1)
    return torch.nn.functional.normalize(axis, dim=1)


def _quat_from_frame_axes(x_axis: torch.Tensor, y_axis: torch.Tensor, z_axis: torch.Tensor) -> torch.Tensor:
    frame = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    return math_utils.quat_from_matrix(frame)


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
