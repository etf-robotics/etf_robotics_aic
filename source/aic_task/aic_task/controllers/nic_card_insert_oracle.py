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
    insertion_ref_pos_tip: torch.Tensor


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
    path_error_local: torch.Tensor
    path_distance: torch.Tensor
    target_path_distance: torch.Tensor
    orientation_error: torch.Tensor
    position_scale: torch.Tensor
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
    use_corner_centerline: bool = True,
    use_tooth_top_alignment: bool = True,
    tip_body: str = "sfp_tip_link",
    insertion_ref_child_names: tuple[str, str] = ("sfp_tip_side_left", "sfp_tip_side_right"),
    tooth_child_name: str = "sfp_tip_tooth_tip",
    target_offset_local: tuple[float, float, float] = (0.0, 0.0, 0.0),
    corner_paths: tuple[str, str, str, str] = (
        "/sfp_port_0_link/sfp_port_0_link_entrance/sfp_port_0_front_left",
        "/sfp_port_0_link/sfp_port_0_link_entrance/sfp_port_0_back_left",
        "/sfp_port_0_link/sfp_port_0_link_entrance/sfp_port_0_back_right",
        "/sfp_port_0_link/sfp_port_0_link_entrance/sfp_port_0_front_right",
    ),
    approach_offset_local: tuple[float, float, float] = (0.0, -0.10, 0.0),
) -> SimpleNicInsertTargets:
    """Resolve the fixed port seat pose from USD in the NIC-card root frame.

    ``approach_offset_local`` is expressed in the ``sfp_port_0_link`` frame.
    Since insertion is along port-local ``+Y``, the default backs the plug out
    along ``-Y`` by 10 cm before moving to the seat.  World-frame targets are
    recomputed during control from ``env.scene[target_name].data.root_*_w``.
    """
    target = env.scene[target_name]
    robot = env.scene["robot"]
    device = target.data.root_pos_w.device
    dtype = target.data.root_pos_w.dtype

    seat_positions_root = []
    seat_quats_root = []
    for env_index in range(env.num_envs):
        root_path = resolve_asset_root_prim_path(target, env_index)
        seat_pos_root, seat_quat_root = _seat_pose_in_asset_root(root_path, seat_path)
        if use_corner_centerline:
            seat_pos_root = _centerline_seat_pos_from_corners(
                root_path,
                seat_pos_root,
                seat_quat_root,
                corner_paths,
            )
        if use_tooth_top_alignment:
            try:
                insertion_ref_pos_tip = _tip_child_midpoint_in_tip(
                    robot,
                    env_index,
                    tip_body,
                    insertion_ref_child_names,
                    dtype=torch.float64,
                    device="cpu",
                )
                tooth_pos_tip = _tip_child_pos_in_tip(
                    robot,
                    env_index,
                    tip_body,
                    tooth_child_name,
                    dtype=torch.float64,
                    device="cpu",
                )
                seat_pos_root = _tooth_top_aligned_seat_pos_from_corners(
                    root_path,
                    seat_pos_root,
                    seat_quat_root,
                    corner_paths,
                    tooth_pos_tip=tuple(float(value) for value in tooth_pos_tip.tolist()),
                    insertion_ref_pos_tip=tuple(float(value) for value in insertion_ref_pos_tip.tolist()),
                )
            except Exception as exc:
                print(f"[WARN] Could not derive tooth/top-aligned target line; using {seat_pos_root}: {exc}")
        if target_offset_local != (0.0, 0.0, 0.0):
            seat_pos_root = _offset_pos_in_local_frame(seat_pos_root, seat_quat_root, target_offset_local)
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
    insertion_ref_child_names: tuple[str, str] = ("sfp_tip_side_left", "sfp_tip_side_right"),
) -> SimpleNicInsertState:
    """Capture the initial controlled-body orientation and module insertion reference."""
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
    insertion_ref_pos_tip = _insertion_ref_pos_in_tip(
        robot,
        env.num_envs,
        tip_body,
        insertion_ref_child_names,
        dtype=tcp_pos_w.dtype,
        device=env.device,
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
        insertion_ref_pos_tip=insertion_ref_pos_tip,
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
    insert_orientation_threshold: float = math.radians(10.0),
    insert_misaligned_pos_scale: float = 0.2,
    insert_alignment_only: bool = False,
    insert_rot_scale: float = 0.05,
    insert_lookahead: float = 0.002,
    insert_recenter_backoff: float = 0.003,
    insert_lateral_correction_scale: float = 1.0,
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
    tip_quat_w = robot.data.body_quat_w[:, tip_id, :]
    insertion_ref_pos_w = tip_pos_w + math_utils.quat_apply(tip_quat_w, state.insertion_ref_pos_tip)
    tip_pos_tcp, tip_quat_tcp = math_utils.subtract_frame_transforms(
        tcp_pos_w,
        tcp_quat_w,
        tip_pos_w,
        tip_quat_w,
    )
    state.tip_pos_tcp[:] = tip_pos_tcp
    state.tip_quat_tcp[:] = tip_quat_tcp
    world_targets = compute_simple_nic_insert_world_targets(env, targets)

    if hold_orientation:
        desired_tip_quat_w = _desired_tip_quat_from_port(world_targets.seat_quat_w)
        desired_tcp_quat_w = _desired_tcp_quat_from_tip(desired_tip_quat_w, tip_quat_tcp)
        orientation_error = math_utils.quat_error_magnitude(tip_quat_w, desired_tip_quat_w)
    else:
        desired_tcp_quat_w = tcp_quat_w
        orientation_error = torch.zeros((env.num_envs,), dtype=tcp_pos_w.dtype, device=tcp_pos_w.device)

    insert_fraction, target_tip_pos_w = _current_target_tip_pos(world_targets, state)
    tip_error = torch.linalg.norm(target_tip_pos_w - insertion_ref_pos_w, dim=1)
    lateral_xy_error = _world_xy_error(insertion_ref_pos_w, target_tip_pos_w)
    path_distance, closest_w, path_lateral_error = _project_tip_to_path(world_targets, insertion_ref_pos_w)
    path_error_local = _path_error_in_port_frame(world_targets, insertion_ref_pos_w, closest_w)
    reached_approach = (state.phase == int(SimpleNicInsertPhase.APPROACH)) & (tip_error <= approach_threshold)
    state.phase[reached_approach] = int(SimpleNicInsertPhase.INSERT)

    insert_mask = state.phase == int(SimpleNicInsertPhase.INSERT)
    advance_mask = insert_mask & (path_lateral_error <= insert_lateral_threshold)
    recenter_mask = insert_mask & (path_lateral_error > insert_lateral_threshold)
    state.insert_distance[insert_mask] = path_distance[insert_mask]
    if insert_recenter_backoff > 0.0:
        state.insert_distance[recenter_mask] = torch.clamp(
            path_distance[recenter_mask] - insert_recenter_backoff,
            min=0.0,
        )
    if insert_alignment_only:
        advance_mask[:] = False
    step_distance = max(insert_speed * step_dt, insert_lookahead)
    state.insert_distance[advance_mask] = path_distance[advance_mask] + step_distance
    state.insert_distance[:] = torch.minimum(state.insert_distance, world_targets.path_length)

    insert_fraction, target_tip_pos_w = _current_target_tip_pos(world_targets, state)
    tip_error = torch.linalg.norm(target_tip_pos_w - insertion_ref_pos_w, dim=1)
    lateral_xy_error = _world_xy_error(insertion_ref_pos_w, target_tip_pos_w)
    path_distance, closest_w, path_lateral_error = _project_tip_to_path(world_targets, insertion_ref_pos_w)
    path_error_local = _path_error_in_port_frame(world_targets, insertion_ref_pos_w, closest_w)

    reached_final = (
        (state.phase == int(SimpleNicInsertPhase.INSERT))
        & (insert_fraction.squeeze(1) >= 1.0)
        & (tip_error <= final_threshold)
    )
    state.phase[reached_final] = int(SimpleNicInsertPhase.HOLD)
    state.hold_steps[state.phase == int(SimpleNicInsertPhase.HOLD)] += 1

    position_error_w = target_tip_pos_w - insertion_ref_pos_w
    if bool(insert_mask.any()) and insert_lateral_correction_scale < 1.0:
        path_axis_w = world_targets.path_w / world_targets.path_length
        path_error_w = torch.sum(position_error_w * path_axis_w, dim=1, keepdim=True) * path_axis_w
        lateral_error_w = position_error_w - path_error_w
        insert_lateral_scale = max(0.0, insert_lateral_correction_scale)
        insert_position_error_w = path_error_w + lateral_error_w * insert_lateral_scale
        position_error_w = torch.where(insert_mask.unsqueeze(1), insert_position_error_w, position_error_w)
    desired_tcp_pos_w = tcp_pos_w + position_error_w
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
    position_scale = torch.ones((env.num_envs,), dtype=processed_action.dtype, device=processed_action.device)
    if hold_orientation:
        insert_action_mask = state.phase == int(SimpleNicInsertPhase.INSERT)
        if insert_alignment_only:
            position_scale[insert_action_mask] = 0.0
        else:
            processed_action[insert_action_mask, 3:6] *= insert_rot_scale
        processed_action[:, 0:3] *= position_scale.unsqueeze(1)
    raw_action = processed_action / torch.clamp(action_scale, min=1.0e-9)
    return SimpleNicInsertOracleOutput(
        raw_action=raw_action,
        processed_action=processed_action,
        target_tip_pos_w=target_tip_pos_w,
        tip_pos_w=insertion_ref_pos_w,
        tip_error=tip_error,
        lateral_xy_error=lateral_xy_error,
        path_lateral_error=path_lateral_error,
        path_error_local=path_error_local,
        path_distance=path_distance.squeeze(1),
        target_path_distance=state.insert_distance.squeeze(1).clone(),
        orientation_error=orientation_error,
        position_scale=position_scale,
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


def _path_error_in_port_frame(
    world_targets: SimpleNicInsertWorldTargets,
    tip_pos_w: torch.Tensor,
    closest_w: torch.Tensor,
) -> torch.Tensor:
    error_w = tip_pos_w - closest_w
    return math_utils.quat_apply(math_utils.quat_inv(world_targets.seat_quat_w), error_w)


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


def _centerline_seat_pos_from_corners(
    asset_root_path: str,
    seat_pos_root: tuple[float, float, float],
    seat_quat_root: tuple[float, float, float, float],
    corner_paths: tuple[str, str, str, str],
) -> tuple[float, float, float]:
    """Move the target line to the entrance corner center, keeping the seat insertion level."""
    try:
        corner_positions = [_prim_position_in_asset_root(asset_root_path, corner_path) for corner_path in corner_paths]
    except Exception as exc:
        print(f"[WARN] Could not derive port centerline from entrance corners; using {seat_pos_root}: {exc}")
        return seat_pos_root

    dtype = torch.float64
    corner_center = torch.tensor(corner_positions, dtype=dtype).mean(dim=0)
    seat_pos = torch.tensor(seat_pos_root, dtype=dtype)
    seat_quat = torch.tensor(seat_quat_root, dtype=dtype).unsqueeze(0)
    port_y = math_utils.quat_apply(seat_quat, torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype))[0]
    port_y = torch.nn.functional.normalize(port_y, dim=0)
    seat_depth = torch.dot(seat_pos - corner_center, port_y)
    centerline_seat = corner_center + seat_depth * port_y
    return tuple(float(value) for value in centerline_seat.tolist())


def _tooth_top_aligned_seat_pos_from_corners(
    asset_root_path: str,
    seat_pos_root: tuple[float, float, float],
    seat_quat_root: tuple[float, float, float, float],
    corner_paths: tuple[str, str, str, str],
    *,
    tooth_pos_tip: tuple[float, float, float],
    insertion_ref_pos_tip: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Shift the insertion reference so the tooth keypoint rides the port top line."""
    corner_positions = [_prim_position_in_asset_root(asset_root_path, corner_path) for corner_path in corner_paths]

    dtype = torch.float64
    corners = torch.tensor(corner_positions, dtype=dtype)
    seat_pos = torch.tensor(seat_pos_root, dtype=dtype)
    seat_quat = torch.tensor(seat_quat_root, dtype=dtype).unsqueeze(0)
    port_y = math_utils.quat_apply(seat_quat, torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype))[0]
    port_z = math_utils.quat_apply(seat_quat, torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype))[0]
    port_y = torch.nn.functional.normalize(port_y, dim=0)
    port_z = torch.nn.functional.normalize(port_z, dim=0)

    # User-confirmed top direction is port-local -Z.  The two corners with the
    # smallest projection on +Z form the top edge of the port opening.
    z_coord = torch.sum(corners * port_z, dim=1)
    top_corner_ids = torch.topk(z_coord, k=2, largest=False).indices
    top_center = corners[top_corner_ids].mean(dim=0)
    seat_depth = torch.dot(seat_pos - top_center, port_y)
    tooth_target_at_seat = top_center + seat_depth * port_y

    desired_tip_quat_root = _desired_tip_quat_from_port(seat_quat)[0].unsqueeze(0)
    tooth_from_ref_tip = torch.tensor(tooth_pos_tip, dtype=dtype) - torch.tensor(insertion_ref_pos_tip, dtype=dtype)
    tooth_from_ref_root = math_utils.quat_apply(desired_tip_quat_root, tooth_from_ref_tip.unsqueeze(0))[0]
    insertion_ref_seat = tooth_target_at_seat - tooth_from_ref_root
    return tuple(float(value) for value in insertion_ref_seat.tolist())


def _offset_pos_in_local_frame(
    pos_root: tuple[float, float, float],
    quat_root: tuple[float, float, float, float],
    offset_local: tuple[float, float, float],
) -> tuple[float, float, float]:
    dtype = torch.float64
    pos = torch.tensor(pos_root, dtype=dtype)
    quat = torch.tensor(quat_root, dtype=dtype).unsqueeze(0)
    offset = torch.tensor(offset_local, dtype=dtype).unsqueeze(0)
    shifted = pos + math_utils.quat_apply(quat, offset)[0]
    return tuple(float(value) for value in shifted.tolist())


def _prim_position_in_asset_root(asset_root_path: str, prim_path: str) -> tuple[float, float, float]:
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(asset_root_path)
    _, prim = _resolve_prim(stage, asset_root_path, prim_path)
    if not root_prim.IsValid() or not prim.IsValid():
        candidates = ", ".join(_candidate_prim_paths(asset_root_path, prim_path))
        raise KeyError(f"USD prim '{prim_path}' was not found. Tried: {candidates}.")
    cache = UsdGeom.XformCache()
    root_matrix = cache.GetLocalToWorldTransform(root_prim)
    prim_matrix = cache.GetLocalToWorldTransform(prim)
    position_root = root_matrix.GetInverse().Transform(prim_matrix.ExtractTranslation())
    return (float(position_root[0]), float(position_root[1]), float(position_root[2]))


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
    """Return the raw SFP tip orientation for insertion into the NIC port.

    Confirmed frame mapping:
    - tip local +Y aligns with port local -Z
    - tip local +Z aligns with port local -Y

    These two constraints define the right-handed tip frame.  They imply tip
    local +X aligns with port local -X.
    """
    port_y = _local_axis_w(port_quat_w, (0.0, 1.0, 0.0))
    port_z = _local_axis_w(port_quat_w, (0.0, 0.0, 1.0))

    target_y = -port_z
    target_z = -port_y
    target_x = torch.linalg.cross(target_y, target_z, dim=1)
    target_x = torch.nn.functional.normalize(target_x, dim=1)
    target_y = torch.linalg.cross(target_z, target_x, dim=1)
    target_y = torch.nn.functional.normalize(target_y, dim=1)
    return _quat_from_frame_axes(target_x, target_y, target_z)


def _local_axis_w(quat_w: torch.Tensor, axis_local: tuple[float, float, float]) -> torch.Tensor:
    axis = torch.tensor(axis_local, dtype=quat_w.dtype, device=quat_w.device).unsqueeze(0)
    return math_utils.quat_apply(quat_w, axis.expand(quat_w.shape[0], -1))


def _quat_from_frame_axes(x_axis: torch.Tensor, y_axis: torch.Tensor, z_axis: torch.Tensor) -> torch.Tensor:
    frame = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    return math_utils.quat_from_matrix(frame)


def _first_body_id(robot, body_name: str) -> int:
    body_ids = robot.find_bodies(body_name, preserve_order=True)[0]
    if len(body_ids) == 0:
        available = ", ".join(getattr(robot, "body_names", []))
        raise KeyError(f"Robot body '{body_name}' not found. Available robot bodies: {available}")
    return int(body_ids[0])


def _insertion_ref_pos_in_tip(
    robot,
    num_envs: int,
    tip_body: str,
    child_names: tuple[str, str],
    *,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """Return the module position reference in the tip frame for every env.

    The tip-link origin is not a reliable insertion reference for this asset.
    The side keypoints are the measured geometry, so we drive their midpoint
    along the port centerline.
    """
    refs = []
    for env_index in range(num_envs):
        try:
            refs.append(_tip_child_midpoint_in_tip(robot, env_index, tip_body, child_names, dtype=dtype, device=device))
        except Exception as exc:
            print(
                f"[WARN] Could not resolve insertion reference from {child_names}; "
                f"falling back to {tip_body} origin: {exc}"
            )
            refs.append(torch.zeros(3, dtype=dtype, device=device))
    return torch.stack(refs, dim=0)


def _tip_child_midpoint_in_tip(
    robot,
    env_index: int,
    tip_body: str,
    child_names: tuple[str, str],
    *,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    roots = _asset_search_roots(robot, env_index)
    tip_prim = _find_prim_by_basename(stage, roots, tip_body)
    if tip_prim is None:
        raise KeyError(f"Could not find tip prim '{tip_body}' under {roots}.")

    cache = UsdGeom.XformCache()
    tip_matrix = cache.GetLocalToWorldTransform(tip_prim)
    tip_inv = tip_matrix.GetInverse()
    child_positions = []
    for child_name in child_names:
        child_positions.append(
            tuple(
                float(value)
                for value in _tip_child_pos_in_tip(
                    robot,
                    env_index,
                    tip_body,
                    child_name,
                    dtype=dtype,
                    device=device,
                    stage=stage,
                    roots=roots,
                    tip_inv=tip_inv,
                    cache=cache,
                ).tolist()
            )
        )
    return torch.tensor(child_positions, dtype=dtype, device=device).mean(dim=0)


def _tip_child_pos_in_tip(
    robot,
    env_index: int,
    tip_body: str,
    child_name: str,
    *,
    dtype: torch.dtype,
    device: str,
    stage=None,
    roots: list[str] | None = None,
    tip_inv=None,
    cache=None,
) -> torch.Tensor:
    import omni.usd
    from pxr import UsdGeom

    if stage is None:
        stage = omni.usd.get_context().get_stage()
    if roots is None:
        roots = _asset_search_roots(robot, env_index)
    if cache is None:
        cache = UsdGeom.XformCache()
    if tip_inv is None:
        tip_prim = _find_prim_by_basename(stage, roots, tip_body)
        if tip_prim is None:
            raise KeyError(f"Could not find tip prim '{tip_body}' under {roots}.")
        tip_inv = cache.GetLocalToWorldTransform(tip_prim).GetInverse()

    child_prim = _find_prim_by_basename(stage, roots, child_name)
    if child_prim is None:
        raise KeyError(f"Could not find child prim '{child_name}' under {roots}.")
    child_w = cache.GetLocalToWorldTransform(child_prim).ExtractTranslation()
    child_tip = tip_inv.Transform(child_w)
    return torch.tensor((float(child_tip[0]), float(child_tip[1]), float(child_tip[2])), dtype=dtype, device=device)


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


def _clamp_vector_norm(vector: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.linalg.norm(vector, dim=1, keepdim=True)
    scale = torch.clamp(max_norm / torch.clamp(norm, min=1.0e-9), max=1.0)
    return vector * scale
