"""Front-keypoint oracle for inserting the SFP module into NIC port 0.

The controlled geometry is the front edge of the plug:

* ``sfp_tip_front_left``
* ``sfp_tip_front_right``

The target geometry is the matching front edge of the port opening.  The
left/right points define the frame position and width axis, while the plug
insertion axis is ``sfp_tip_link`` local ``-Z`` aligned to port ``+Y``.
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
    """Fixed port-front geometry cached in the NIC-card root frame."""

    entry_front_center_root: torch.Tensor
    final_front_center_root: torch.Tensor
    approach_front_center_root: torch.Tensor
    port_x_root: torch.Tensor
    port_y_root: torch.Tensor
    port_z_root: torch.Tensor
    front_target_xz_offset: tuple[float, float]
    approach_offset_local: tuple[float, float, float]


@dataclass
class SimpleNicInsertWorldTargets:
    """World-frame port/front targets computed from the live NIC-card pose."""

    entry_front_center_w: torch.Tensor
    final_front_center_w: torch.Tensor
    approach_front_center_w: torch.Tensor
    path_w: torch.Tensor
    path_length: torch.Tensor
    port_x_w: torch.Tensor
    port_y_w: torch.Tensor
    port_z_w: torch.Tensor
    front_quat_w: torch.Tensor


@dataclass
class SimpleNicInsertState:
    """Mutable state and plug-front calibration."""

    phase: torch.Tensor
    insert_distance: torch.Tensor
    hold_steps: torch.Tensor
    tip_pos_tcp: torch.Tensor
    tip_quat_tcp: torch.Tensor
    front_center_pos_tip: torch.Tensor
    front_quat_tip: torch.Tensor
    front_left_pos_tip: torch.Tensor
    front_right_pos_tip: torch.Tensor
    tip_width: torch.Tensor


@dataclass
class SimpleNicInsertOracleOutput:
    """Action and diagnostics for one control step."""

    raw_action: torch.Tensor
    processed_action: torch.Tensor
    front_center_w: torch.Tensor
    front_left_w: torch.Tensor
    front_right_w: torch.Tensor
    target_front_center_w: torch.Tensor
    target_front_left_w: torch.Tensor
    target_front_right_w: torch.Tensor
    front_center_error: torch.Tensor
    front_left_error: torch.Tensor
    front_right_error: torch.Tensor
    path_lateral_error: torch.Tensor
    path_error_local: torch.Tensor
    path_distance: torch.Tensor
    target_path_distance: torch.Tensor
    orientation_error: torch.Tensor
    x_axis_error: torch.Tensor
    y_axis_error: torch.Tensor
    tcp_position_error: torch.Tensor
    tcp_orientation_error: torch.Tensor
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
    front_left_path: str = "/sfp_port_0_link/sfp_port_0_link_entrance/sfp_port_0_front_left",
    front_right_path: str = "/sfp_port_0_link/sfp_port_0_link_entrance/sfp_port_0_front_right",
    front_target_xz_offset: tuple[float, float] = (0.0, 0.0),
    approach_offset_local: tuple[float, float, float] = (0.0, -0.10, 0.0),
) -> SimpleNicInsertTargets:
    """Resolve the port-front target line in the NIC-card root frame.

    ``front_target_xz_offset`` is expressed in the computed port-front X/Z
    plane.  Y is reserved for insertion depth and is intentionally not exposed
    as a target correction.
    """
    target = env.scene[target_name]
    device = target.data.root_pos_w.device
    dtype = target.data.root_pos_w.dtype

    entry_centers = []
    final_centers = []
    approach_centers = []
    port_x_axes = []
    port_y_axes = []
    port_z_axes = []
    for env_index in range(env.num_envs):
        root_path = resolve_asset_root_prim_path(target, env_index)
        seat_pos_root, seat_quat_root = _seat_pose_in_asset_root(root_path, seat_path)
        front_left_root = _prim_position_in_asset_root(root_path, front_left_path)
        front_right_root = _prim_position_in_asset_root(root_path, front_right_path)

        seat_pos = torch.tensor(seat_pos_root, dtype=torch.float64)
        seat_quat = torch.tensor(seat_quat_root, dtype=torch.float64).unsqueeze(0)
        front_left = torch.tensor(front_left_root, dtype=torch.float64)
        front_right = torch.tensor(front_right_root, dtype=torch.float64)

        port_y = _normalize(
            math_utils.quat_apply(seat_quat, torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64))[0]
        )
        port_x = _normalize(front_right - front_left)
        port_x = _normalize(port_x - torch.dot(port_x, port_y) * port_y)
        port_z = _normalize(torch.linalg.cross(port_x, port_y, dim=0))
        port_y = _normalize(torch.linalg.cross(port_z, port_x, dim=0))

        entry_center = 0.5 * (front_left + front_right)
        entry_center = entry_center + float(front_target_xz_offset[0]) * port_x
        entry_center = entry_center + float(front_target_xz_offset[1]) * port_z
        seat_depth = torch.dot(seat_pos - entry_center, port_y)
        final_center = entry_center + seat_depth * port_y
        approach_offset = (
            float(approach_offset_local[0]) * port_x
            + float(approach_offset_local[1]) * port_y
            + float(approach_offset_local[2]) * port_z
        )
        approach_center = entry_center + approach_offset

        entry_centers.append(entry_center.tolist())
        final_centers.append(final_center.tolist())
        approach_centers.append(approach_center.tolist())
        port_x_axes.append(port_x.tolist())
        port_y_axes.append(port_y.tolist())
        port_z_axes.append(port_z.tolist())

    return SimpleNicInsertTargets(
        entry_front_center_root=torch.tensor(entry_centers, dtype=dtype, device=device),
        final_front_center_root=torch.tensor(final_centers, dtype=dtype, device=device),
        approach_front_center_root=torch.tensor(approach_centers, dtype=dtype, device=device),
        port_x_root=torch.tensor(port_x_axes, dtype=dtype, device=device),
        port_y_root=torch.tensor(port_y_axes, dtype=dtype, device=device),
        port_z_root=torch.tensor(port_z_axes, dtype=dtype, device=device),
        front_target_xz_offset=front_target_xz_offset,
        approach_offset_local=approach_offset_local,
    )


def compute_simple_nic_insert_world_targets(
    env: gym.Env,
    targets: SimpleNicInsertTargets,
    *,
    target_name: str = "nic_card",
) -> SimpleNicInsertWorldTargets:
    """Compute the live world-frame port-front target line."""
    target = env.scene[target_name]
    card_pos = target.data.root_pos_w
    card_quat = target.data.root_quat_w

    entry = _root_point_to_world(card_pos, card_quat, targets.entry_front_center_root)
    final = _root_point_to_world(card_pos, card_quat, targets.final_front_center_root)
    approach = _root_point_to_world(card_pos, card_quat, targets.approach_front_center_root)
    port_x = _normalize_rows(math_utils.quat_apply(card_quat, targets.port_x_root))
    port_y = _normalize_rows(math_utils.quat_apply(card_quat, targets.port_y_root))
    port_z = _normalize_rows(math_utils.quat_apply(card_quat, targets.port_z_root))
    front_quat = _quat_from_frame_axes(port_x, port_y, port_z)
    path = final - approach
    path_length = torch.linalg.norm(path, dim=1, keepdim=True).clamp_min(1.0e-9)
    return SimpleNicInsertWorldTargets(
        entry_front_center_w=entry,
        final_front_center_w=final,
        approach_front_center_w=approach,
        path_w=path,
        path_length=path_length,
        port_x_w=port_x,
        port_y_w=port_y,
        port_z_w=port_z,
        front_quat_w=front_quat,
    )


def make_simple_nic_insert_state(
    env: gym.Env,
    *,
    tcp_body: str = "gripper_tcp",
    tip_body: str = "sfp_tip_link",
    front_child_names: tuple[str, str] = ("sfp_tip_front_left", "sfp_tip_front_right"),
) -> SimpleNicInsertState:
    """Capture the front-frame calibration in ``sfp_tip_link`` coordinates."""
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
    left_pos, _ = _tip_child_pose_in_tip_batch(
        robot,
        env.num_envs,
        tip_body,
        front_child_names[0],
        dtype=tcp_pos_w.dtype,
        device=env.device,
    )
    right_pos, _ = _tip_child_pose_in_tip_batch(
        robot,
        env.num_envs,
        tip_body,
        front_child_names[1],
        dtype=tcp_pos_w.dtype,
        device=env.device,
    )
    front_center = 0.5 * (left_pos + right_pos)
    front_quat = _front_frame_quat_from_points(left_pos, right_pos)
    return SimpleNicInsertState(
        phase=torch.full(
            (env.num_envs,),
            int(SimpleNicInsertPhase.APPROACH),
            dtype=torch.long,
            device=env.device,
        ),
        insert_distance=torch.zeros((env.num_envs, 1), dtype=tcp_pos_w.dtype, device=env.device),
        hold_steps=torch.zeros((env.num_envs,), dtype=torch.long, device=env.device),
        tip_pos_tcp=tip_pos_tcp,
        tip_quat_tcp=tip_quat_tcp,
        front_center_pos_tip=front_center,
        front_quat_tip=front_quat,
        front_left_pos_tip=left_pos,
        front_right_pos_tip=right_pos,
        tip_width=torch.linalg.norm(right_pos - left_pos, dim=1, keepdim=True),
    )


def compute_simple_nic_insert_oracle(
    env: gym.Env,
    action_scale: torch.Tensor,
    targets: SimpleNicInsertTargets,
    state: SimpleNicInsertState,
    *,
    tcp_body: str = "gripper_tcp",
    tip_body: str = "sfp_tip_link",
    pos_gain: float = 1.2,
    rot_gain: float = 0.2,
    max_pos_delta: float = 0.020,
    insert_max_pos_delta: float = 0.002,
    max_rot_delta: float = 0.025,
    approach_threshold: float = 0.015,
    insert_lateral_threshold: float = 0.010,
    insert_orientation_threshold: float = math.radians(4.0),
    insert_lookahead: float = 0.002,
    final_threshold: float = 0.003,
    insert_speed: float = 0.010,
    step_dt: float = 1.0 / 30.0,
) -> SimpleNicInsertOracleOutput:
    """Compute one relative-IK action for the front-point insertion oracle."""
    robot = env.scene["robot"]
    tcp_id = _first_body_id(robot, tcp_body)
    tip_id = _first_body_id(robot, tip_body)
    tcp_pos_w = robot.data.body_pos_w[:, tcp_id, :]
    tcp_quat_w = robot.data.body_quat_w[:, tcp_id, :]
    tip_pos_w = robot.data.body_pos_w[:, tip_id, :]
    tip_quat_w = robot.data.body_quat_w[:, tip_id, :]
    tip_pos_tcp, tip_quat_tcp = math_utils.subtract_frame_transforms(
        tcp_pos_w,
        tcp_quat_w,
        tip_pos_w,
        tip_quat_w,
    )
    state.tip_pos_tcp[:] = tip_pos_tcp
    state.tip_quat_tcp[:] = tip_quat_tcp

    world_targets = compute_simple_nic_insert_world_targets(env, targets)
    front_center_w, front_quat_w, front_left_w, front_right_w = _live_front_geometry(tip_pos_w, tip_quat_w, state)

    insert_fraction, target_front_center_w = _current_target_front_center(world_targets, state)
    target_front_left_w, target_front_right_w = _target_front_points(world_targets, state, target_front_center_w)
    center_error = torch.linalg.norm(target_front_center_w - front_center_w, dim=1)
    path_distance, closest_w, path_lateral_error = _project_front_to_path(world_targets, front_center_w)
    path_error_local = _path_error_in_front_frame(world_targets, front_center_w, closest_w)
    orientation_error = math_utils.quat_error_magnitude(front_quat_w, world_targets.front_quat_w)
    x_axis_error = _axis_angle(
        _local_axis_w(front_quat_w, (1.0, 0.0, 0.0)),
        world_targets.port_x_w,
    )
    y_axis_error = _axis_angle(
        _local_axis_w(front_quat_w, (0.0, 1.0, 0.0)),
        world_targets.port_y_w,
    )

    reached_approach = (
        (state.phase == int(SimpleNicInsertPhase.APPROACH))
        & (center_error <= approach_threshold)
        & (orientation_error <= insert_orientation_threshold)
    )
    state.phase[reached_approach] = int(SimpleNicInsertPhase.INSERT)

    insert_mask = state.phase == int(SimpleNicInsertPhase.INSERT)
    advance_mask = (
        insert_mask
        & (path_lateral_error <= insert_lateral_threshold)
        & (orientation_error <= insert_orientation_threshold)
    )
    state.insert_distance[insert_mask] = path_distance[insert_mask]
    step_distance = max(insert_speed * step_dt, insert_lookahead)
    state.insert_distance[advance_mask] = path_distance[advance_mask] + step_distance
    state.insert_distance[:] = torch.minimum(state.insert_distance, world_targets.path_length)

    insert_fraction, target_front_center_w = _current_target_front_center(world_targets, state)
    target_front_left_w, target_front_right_w = _target_front_points(world_targets, state, target_front_center_w)
    center_error = torch.linalg.norm(target_front_center_w - front_center_w, dim=1)
    left_error = torch.linalg.norm(target_front_left_w - front_left_w, dim=1)
    right_error = torch.linalg.norm(target_front_right_w - front_right_w, dim=1)
    path_distance, closest_w, path_lateral_error = _project_front_to_path(world_targets, front_center_w)
    path_error_local = _path_error_in_front_frame(world_targets, front_center_w, closest_w)

    reached_final = (
        (state.phase == int(SimpleNicInsertPhase.INSERT))
        & (insert_fraction.squeeze(1) >= 1.0)
        & (center_error <= final_threshold)
    )
    state.phase[reached_final] = int(SimpleNicInsertPhase.HOLD)
    state.hold_steps[state.phase == int(SimpleNicInsertPhase.HOLD)] += 1

    desired_tip_quat_w = math_utils.quat_mul(
        world_targets.front_quat_w,
        math_utils.quat_inv(state.front_quat_tip),
    )
    desired_tip_pos_w = target_front_center_w - math_utils.quat_apply(desired_tip_quat_w, state.front_center_pos_tip)
    desired_tcp_quat_w = math_utils.quat_mul(desired_tip_quat_w, math_utils.quat_inv(tip_quat_tcp))
    desired_tcp_pos_w = desired_tip_pos_w - math_utils.quat_apply(desired_tcp_quat_w, tip_pos_tcp)
    tcp_position_error = torch.linalg.norm(desired_tcp_pos_w - tcp_pos_w, dim=1)
    tcp_orientation_error = math_utils.quat_error_magnitude(tcp_quat_w, desired_tcp_quat_w)
    processed_action = _relative_ik_processed_action(
        robot,
        tcp_pos_w,
        tcp_quat_w,
        desired_tcp_pos_w,
        desired_tcp_quat_w,
        action_scale,
        pos_gain=pos_gain,
        rot_gain=rot_gain,
        max_pos_delta=insert_max_pos_delta if bool(insert_mask.any()) else max_pos_delta,
        max_rot_delta=max_rot_delta,
    )
    raw_action = processed_action / torch.clamp(action_scale, min=1.0e-9)
    return SimpleNicInsertOracleOutput(
        raw_action=raw_action,
        processed_action=processed_action,
        front_center_w=front_center_w,
        front_left_w=front_left_w,
        front_right_w=front_right_w,
        target_front_center_w=target_front_center_w,
        target_front_left_w=target_front_left_w,
        target_front_right_w=target_front_right_w,
        front_center_error=center_error,
        front_left_error=left_error,
        front_right_error=right_error,
        path_lateral_error=path_lateral_error,
        path_error_local=path_error_local,
        path_distance=path_distance.squeeze(1),
        target_path_distance=state.insert_distance.squeeze(1).clone(),
        orientation_error=orientation_error,
        x_axis_error=x_axis_error,
        y_axis_error=y_axis_error,
        tcp_position_error=tcp_position_error,
        tcp_orientation_error=tcp_orientation_error,
        phase=state.phase.clone(),
        insert_fraction=insert_fraction.squeeze(1),
    )


def _current_target_front_center(
    world_targets: SimpleNicInsertWorldTargets,
    state: SimpleNicInsertState,
) -> tuple[torch.Tensor, torch.Tensor]:
    insert_fraction = (state.insert_distance / world_targets.path_length).clamp(0.0, 1.0)
    insert_target_w = world_targets.approach_front_center_w + world_targets.path_w * insert_fraction
    target_w = torch.where(
        (state.phase == int(SimpleNicInsertPhase.APPROACH)).unsqueeze(1),
        world_targets.approach_front_center_w,
        insert_target_w,
    )
    target_w = torch.where(
        (state.phase == int(SimpleNicInsertPhase.HOLD)).unsqueeze(1),
        world_targets.final_front_center_w,
        target_w,
    )
    return insert_fraction, target_w


def _target_front_points(
    world_targets: SimpleNicInsertWorldTargets,
    state: SimpleNicInsertState,
    center_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    half_width = 0.5 * state.tip_width
    left = center_w - half_width * world_targets.port_x_w
    right = center_w + half_width * world_targets.port_x_w
    return left, right


def _live_front_geometry(
    tip_pos_w: torch.Tensor,
    tip_quat_w: torch.Tensor,
    state: SimpleNicInsertState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    center = tip_pos_w + math_utils.quat_apply(tip_quat_w, state.front_center_pos_tip)
    quat = math_utils.quat_mul(tip_quat_w, state.front_quat_tip)
    left = tip_pos_w + math_utils.quat_apply(tip_quat_w, state.front_left_pos_tip)
    right = tip_pos_w + math_utils.quat_apply(tip_quat_w, state.front_right_pos_tip)
    return center, quat, left, right


def _project_front_to_path(
    world_targets: SimpleNicInsertWorldTargets,
    front_center_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    path_axis_w = world_targets.path_w / world_targets.path_length
    front_from_approach = front_center_w - world_targets.approach_front_center_w
    path_distance = torch.sum(front_from_approach * path_axis_w, dim=1, keepdim=True)
    path_distance = torch.clamp(path_distance, min=0.0)
    path_distance = torch.minimum(path_distance, world_targets.path_length)
    closest_w = world_targets.approach_front_center_w + path_axis_w * path_distance
    path_lateral_error = torch.linalg.norm(front_center_w - closest_w, dim=1)
    return path_distance, closest_w, path_lateral_error


def _path_error_in_front_frame(
    world_targets: SimpleNicInsertWorldTargets,
    front_center_w: torch.Tensor,
    closest_w: torch.Tensor,
) -> torch.Tensor:
    error_w = front_center_w - closest_w
    return torch.stack(
        (
            torch.sum(error_w * world_targets.port_x_w, dim=1),
            torch.sum(error_w * world_targets.port_y_w, dim=1),
            torch.sum(error_w * world_targets.port_z_w, dim=1),
        ),
        dim=1,
    )


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


def _tip_child_pose_in_tip_batch(
    robot,
    num_envs: int,
    tip_body: str,
    child_name: str,
    *,
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    poses = [
        _tip_child_pose_in_tip(
            robot,
            env_index,
            tip_body,
            child_name,
            dtype=dtype,
            device=device,
        )
        for env_index in range(num_envs)
    ]
    pos = torch.stack([pose[0] for pose in poses], dim=0)
    quat = torch.stack([pose[1] for pose in poses], dim=0)
    return pos, quat


def _tip_child_pose_in_tip(
    robot,
    env_index: int,
    tip_body: str,
    child_name: str,
    *,
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    roots = _asset_search_roots(robot, env_index)
    tip_prim = _find_prim_by_basename(stage, roots, tip_body)
    child_prim = _find_prim_by_basename(stage, roots, child_name)
    if tip_prim is None:
        raise KeyError(f"Could not find tip prim '{tip_body}' under {roots}.")
    if child_prim is None:
        raise KeyError(f"Could not find child prim '{child_name}' under {roots}.")
    return _pose_in_root(tip_prim, child_prim, dtype=dtype, device=device)


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


def _front_frame_quat_from_points(
    left_pos: torch.Tensor,
    right_pos: torch.Tensor,
) -> torch.Tensor:
    """Build the controlled plug-front frame in ``sfp_tip_link`` coordinates.

    The left/right keypoints define front-frame +X.  The plug insertion axis is
    the tip-link local -Z direction, so front-frame +Y is built from local -Z
    after removing the component parallel to +X.
    """
    front_x = _normalize_rows(right_pos - left_pos)
    front_y = torch.tensor((0.0, 0.0, -1.0), dtype=front_x.dtype, device=front_x.device).unsqueeze(0)
    front_y = front_y.expand_as(front_x)
    degenerate_y = torch.linalg.norm(torch.linalg.cross(front_x, front_y, dim=1), dim=1, keepdim=True) < 1.0e-4
    fallback_y = torch.tensor((0.0, 1.0, 0.0), dtype=front_x.dtype, device=front_x.device).unsqueeze(0)
    front_y = torch.where(degenerate_y, fallback_y.expand_as(front_y), front_y)
    front_y = _normalize_rows(front_y - torch.sum(front_y * front_x, dim=1, keepdim=True) * front_x)
    front_z = _normalize_rows(torch.linalg.cross(front_x, front_y, dim=1))
    front_y = _normalize_rows(torch.linalg.cross(front_z, front_x, dim=1))
    return _quat_from_frame_axes(front_x, front_y, front_z)


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


def _root_point_to_world(root_pos_w: torch.Tensor, root_quat_w: torch.Tensor, point_root: torch.Tensor) -> torch.Tensor:
    return root_pos_w + math_utils.quat_apply(root_quat_w, point_root)


def _local_axis_w(quat_w: torch.Tensor, axis_local: tuple[float, float, float]) -> torch.Tensor:
    axis = torch.tensor(axis_local, dtype=quat_w.dtype, device=quat_w.device).unsqueeze(0)
    return math_utils.quat_apply(quat_w, axis.expand(quat_w.shape[0], -1))


def _axis_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = _normalize_rows(a)
    b = _normalize_rows(b)
    dot = torch.clamp(torch.sum(a * b, dim=1), min=-1.0, max=1.0)
    return torch.acos(dot)


def _quat_from_frame_axes(x_axis: torch.Tensor, y_axis: torch.Tensor, z_axis: torch.Tensor) -> torch.Tensor:
    frame = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    return math_utils.quat_from_matrix(frame)


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.clamp(torch.linalg.norm(vector), min=1.0e-9)


def _normalize_rows(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.clamp(torch.linalg.norm(vector, dim=1, keepdim=True), min=1.0e-9)


def _first_body_id(robot, body_name: str) -> int:
    body_ids = robot.find_bodies(body_name, preserve_order=True)[0]
    if len(body_ids) == 0:
        available = ", ".join(getattr(robot, "body_names", []))
        raise KeyError(f"Robot body '{body_name}' not found. Available robot bodies: {available}")
    return int(body_ids[0])


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
