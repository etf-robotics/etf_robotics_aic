"""Simple pose oracle for inserting an SFP module into a NIC port.

This oracle intentionally uses only one controlled plug frame:
``sfp_tip_link``.  The desired pose for that frame comes from the task's
``insertion_goal`` command.  The TCP command is then computed from the live TCP
pose relative to ``sfp_tip_link`` each step.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils


class SimpleNicInsertPhase(IntEnum):
    """Phases for the demo-only NIC card insertion oracle."""

    SEARCH_FOR_PORT = 0
    PLAN_APPROACH = 1
    APPROACH_P = 2
    APPROACH_P_R = 3
    INSERT = 4
    HOLD = 5


@dataclass
class SimpleNicInsertTargets:
    """Fixed selected-port pose cached in the NIC-card root frame."""

    final_tip_pos_root: torch.Tensor
    approach_tip_pos_root: torch.Tensor
    port_quat_root: torch.Tensor
    port_x_root: torch.Tensor
    port_y_root: torch.Tensor
    port_z_root: torch.Tensor
    target_xz_offset: tuple[float, float]
    approach_offset_local: tuple[float, float, float]
    port_index: int
    port_path: str


@dataclass
class SimpleNicInsertWorldTargets:
    """World-frame target pose and insertion path for the selected port."""

    final_tip_pos_w: torch.Tensor
    nominal_approach_tip_pos_w: torch.Tensor
    approach_tip_pos_w: torch.Tensor
    target_tip_quat_w: torch.Tensor
    approach_tip_quat_w: torch.Tensor
    target_x_w: torch.Tensor
    target_y_w: torch.Tensor
    target_z_w: torch.Tensor
    path_w: torch.Tensor
    path_axis_w: torch.Tensor
    path_length: torch.Tensor
    port_x_w: torch.Tensor
    port_y_w: torch.Tensor
    port_z_w: torch.Tensor


@dataclass
class SimpleNicInsertState:
    """Mutable phase state plus live TCP-in-tip calibration."""

    phase: torch.Tensor
    insert_distance: torch.Tensor
    hold_steps: torch.Tensor
    tcp_pos_tip: torch.Tensor
    tcp_quat_tip: torch.Tensor
    prev_tip_pos_w: torch.Tensor
    prev_tip_vel_w: torch.Tensor
    plan_valid: torch.Tensor
    plan_start_pos_w: torch.Tensor
    plan_start_vel_w: torch.Tensor
    plan_start_acc_w: torch.Tensor
    plan_end_pos_w: torch.Tensor
    plan_end_vel_w: torch.Tensor
    plan_end_acc_w: torch.Tensor
    plan_start_quat_w: torch.Tensor
    plan_approach_quat_w: torch.Tensor
    plan_final_quat_w: torch.Tensor
    plan_elapsed: torch.Tensor
    plan_duration: torch.Tensor
    plan_rot_start: torch.Tensor
    plan_rot_duration: torch.Tensor


@dataclass
class SimpleNicInsertOracleOutput:
    """Action and diagnostics for one control step."""

    raw_action: torch.Tensor
    processed_action: torch.Tensor
    tip_pos_w: torch.Tensor
    target_tip_pos_w: torch.Tensor
    tip_position_error: torch.Tensor
    path_lateral_error: torch.Tensor
    path_error_local: torch.Tensor
    path_distance: torch.Tensor
    target_path_distance: torch.Tensor
    orientation_error: torch.Tensor
    x_axis_error: torch.Tensor
    y_axis_error: torch.Tensor
    z_axis_error: torch.Tensor
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


def get_insertion_goal(env: gym.Env, command_name: str = "insertion_goal"):
    """Return the task command that owns the desired SFP-tip insertion goal."""
    return env.command_manager.get_term(command_name)


def make_simple_nic_insert_targets(
    env: gym.Env,
    *,
    target_name: str = "nic_card",
    port_index: int = 0,
    target_xz_offset: tuple[float, float] = (0.0, 0.0),
    approach_offset_local: tuple[float, float, float] = (0.0, -0.05, 0.0),
) -> SimpleNicInsertTargets:
    """Resolve the selected port link target in the NIC-card root frame.

    ``target_xz_offset`` is expressed in the selected port link's local X/Z
    plane.  It shifts the final insertion pose without changing the port-link
    orientation.  ``approach_offset_local`` is also port-local; the default
    starts before the port along local -Y, then INSERT advances toward +Y.
    """
    if port_index not in (0, 1):
        raise ValueError(f"port_index must be 0 or 1, got {port_index}.")

    target = env.scene[target_name]
    device = target.data.root_pos_w.device
    dtype = target.data.root_pos_w.dtype
    port_path = f"/sfp_port_{port_index}_link"

    final_positions = []
    approach_positions = []
    port_quats = []
    port_x_axes = []
    port_y_axes = []
    port_z_axes = []
    for env_index in range(env.num_envs):
        root_path = _resolve_asset_root_prim_path(target, env_index)
        port_pos_root, port_quat_root = _prim_pose_in_asset_root(root_path, port_path)

        port_pos = torch.tensor(port_pos_root, dtype=torch.float64)
        port_quat = torch.tensor(port_quat_root, dtype=torch.float64).unsqueeze(0)
        port_x = _normalize(
            math_utils.quat_apply(port_quat, torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64))[0]
        )
        port_y = _normalize(
            math_utils.quat_apply(port_quat, torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64))[0]
        )
        port_z = _normalize(
            math_utils.quat_apply(port_quat, torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64))[0]
        )

        # Only expose a correction in the port cross-section.  The insertion
        # direction itself remains port-local +Y and is controlled by phase.
        final_pos = port_pos + float(target_xz_offset[0]) * port_x + float(target_xz_offset[1]) * port_z
        approach_pos = (
            final_pos
            + float(approach_offset_local[0]) * port_x
            + float(approach_offset_local[1]) * port_y
            + float(approach_offset_local[2]) * port_z
        )

        final_positions.append(final_pos.tolist())
        approach_positions.append(approach_pos.tolist())
        port_quats.append(tuple(float(value) for value in port_quat_root))
        port_x_axes.append(port_x.tolist())
        port_y_axes.append(port_y.tolist())
        port_z_axes.append(port_z.tolist())

    return SimpleNicInsertTargets(
        final_tip_pos_root=torch.tensor(final_positions, dtype=dtype, device=device),
        approach_tip_pos_root=torch.tensor(approach_positions, dtype=dtype, device=device),
        port_quat_root=torch.tensor(port_quats, dtype=dtype, device=device),
        port_x_root=torch.tensor(port_x_axes, dtype=dtype, device=device),
        port_y_root=torch.tensor(port_y_axes, dtype=dtype, device=device),
        port_z_root=torch.tensor(port_z_axes, dtype=dtype, device=device),
        target_xz_offset=target_xz_offset,
        approach_offset_local=approach_offset_local,
        port_index=port_index,
        port_path=port_path,
    )


def compute_simple_nic_insert_world_targets(
    env: gym.Env,
    targets: SimpleNicInsertTargets | None = None,
    *,
    target_name: str = "nic_card",
    command_name: str = "insertion_goal",
) -> SimpleNicInsertWorldTargets:
    """Compute the live world-frame target pose from the current NIC-card pose."""
    if targets is None:
        return _world_targets_from_command(env, command_name=command_name)

    target = env.scene[target_name]
    card_pos = target.data.root_pos_w
    card_quat = target.data.root_quat_w

    final = _root_point_to_world(card_pos, card_quat, targets.final_tip_pos_root)
    approach = _root_point_to_world(card_pos, card_quat, targets.approach_tip_pos_root)
    port_quat_w = math_utils.quat_mul(card_quat, targets.port_quat_root)
    # The link positions are good, but the desired plug frame needs one fixed
    # local correction: keep port X aligned with tip X, then rotate +90 deg
    # around that X axis.  Translation offsets below still use the original
    # port-local X/Z plane, not this corrected orientation frame.
    target_quat = math_utils.quat_mul(port_quat_w, _target_orientation_offset(port_quat_w))
    port_x = _normalize_rows(math_utils.quat_apply(card_quat, targets.port_x_root))
    port_y = _normalize_rows(math_utils.quat_apply(card_quat, targets.port_y_root))
    port_z = _normalize_rows(math_utils.quat_apply(card_quat, targets.port_z_root))
    target_x = _local_axis_w(target_quat, (1.0, 0.0, 0.0))
    target_y = _local_axis_w(target_quat, (0.0, 1.0, 0.0))
    target_z = _local_axis_w(target_quat, (0.0, 0.0, 1.0))
    path = final - approach
    path_length = torch.linalg.norm(path, dim=1, keepdim=True).clamp_min(1.0e-9)
    path_axis = path / path_length
    return SimpleNicInsertWorldTargets(
        final_tip_pos_w=final,
        nominal_approach_tip_pos_w=approach,
        approach_tip_pos_w=approach,
        target_tip_quat_w=target_quat,
        approach_tip_quat_w=target_quat,
        target_x_w=target_x,
        target_y_w=target_y,
        target_z_w=target_z,
        path_w=path,
        path_axis_w=path_axis,
        path_length=path_length,
        port_x_w=port_x,
        port_y_w=port_y,
        port_z_w=port_z,
    )


def _world_targets_from_command(env: gym.Env, command_name: str = "insertion_goal") -> SimpleNicInsertWorldTargets:
    """Expose the command term as the oracle's world-target dataclass."""
    goal = get_insertion_goal(env, command_name=command_name)
    nominal_approach = getattr(goal, "nominal_approach_tip_pos_w", goal.approach_tip_pos_w)
    approach_quat = getattr(goal, "approach_tip_quat_w", goal.target_tip_quat_w)
    return SimpleNicInsertWorldTargets(
        final_tip_pos_w=goal.final_tip_pos_w,
        nominal_approach_tip_pos_w=nominal_approach,
        approach_tip_pos_w=goal.approach_tip_pos_w,
        target_tip_quat_w=goal.target_tip_quat_w,
        approach_tip_quat_w=approach_quat,
        target_x_w=goal.target_x_w,
        target_y_w=goal.target_y_w,
        target_z_w=goal.target_z_w,
        path_w=goal.path_w,
        path_axis_w=getattr(goal, "path_axis_w", goal.path_w / goal.path_length),
        path_length=goal.path_length,
        port_x_w=goal.port_x_w,
        port_y_w=goal.port_y_w,
        port_z_w=goal.port_z_w,
    )


def make_simple_nic_insert_state(
    env: gym.Env,
    *,
    tcp_body: str = "gripper_tcp",
    tip_body: str = "sfp_tip_link",
) -> SimpleNicInsertState:
    """Initialize phase state and the live TCP pose expressed in ``sfp_tip_link``."""
    robot = env.scene["robot"]
    tcp_id = _first_body_id(robot, tcp_body)
    tip_id = _first_body_id(robot, tip_body)
    tcp_pos_w = robot.data.body_pos_w[:, tcp_id, :].clone()
    tcp_quat_w = robot.data.body_quat_w[:, tcp_id, :].clone()
    tip_pos_w = robot.data.body_pos_w[:, tip_id, :].clone()
    tip_quat_w = robot.data.body_quat_w[:, tip_id, :].clone()
    tcp_pos_tip, tcp_quat_tip = math_utils.subtract_frame_transforms(
        tip_pos_w,
        tip_quat_w,
        tcp_pos_w,
        tcp_quat_w,
    )
    identity_quat = torch.zeros_like(tip_quat_w)
    identity_quat[:, 0] = 1.0
    return SimpleNicInsertState(
        phase=torch.full(
            (env.num_envs,),
            int(SimpleNicInsertPhase.SEARCH_FOR_PORT),
            dtype=torch.long,
            device=env.device,
        ),
        insert_distance=torch.zeros((env.num_envs, 1), dtype=tcp_pos_w.dtype, device=env.device),
        hold_steps=torch.zeros((env.num_envs,), dtype=torch.long, device=env.device),
        tcp_pos_tip=tcp_pos_tip,
        tcp_quat_tip=tcp_quat_tip,
        prev_tip_pos_w=tip_pos_w.clone(),
        prev_tip_vel_w=torch.zeros_like(tip_pos_w),
        plan_valid=torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device),
        plan_start_pos_w=torch.zeros_like(tip_pos_w),
        plan_start_vel_w=torch.zeros_like(tip_pos_w),
        plan_start_acc_w=torch.zeros_like(tip_pos_w),
        plan_end_pos_w=torch.zeros_like(tip_pos_w),
        plan_end_vel_w=torch.zeros_like(tip_pos_w),
        plan_end_acc_w=torch.zeros_like(tip_pos_w),
        plan_start_quat_w=identity_quat.clone(),
        plan_approach_quat_w=identity_quat.clone(),
        plan_final_quat_w=identity_quat.clone(),
        plan_elapsed=torch.zeros((env.num_envs, 1), dtype=tcp_pos_w.dtype, device=env.device),
        plan_duration=torch.ones((env.num_envs, 1), dtype=tcp_pos_w.dtype, device=env.device),
        plan_rot_start=torch.zeros((env.num_envs, 1), dtype=tcp_pos_w.dtype, device=env.device),
        plan_rot_duration=torch.zeros((env.num_envs, 1), dtype=tcp_pos_w.dtype, device=env.device),
    )


def reset_simple_nic_insert_state(
    env: gym.Env,
    state: SimpleNicInsertState,
    env_ids: torch.Tensor | list[int] | None = None,
    *,
    tcp_body: str = "gripper_tcp",
    tip_body: str = "sfp_tip_link",
) -> None:
    """Reset oracle phase state for envs that IsaacLab has just reset."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    elif not torch.is_tensor(env_ids):
        env_ids = torch.tensor(env_ids, dtype=torch.long, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    robot = env.scene["robot"]
    tcp_id = _first_body_id(robot, tcp_body)
    tip_id = _first_body_id(robot, tip_body)
    tcp_pos_w = robot.data.body_pos_w[env_ids, tcp_id, :].clone()
    tcp_quat_w = robot.data.body_quat_w[env_ids, tcp_id, :].clone()
    tip_pos_w = robot.data.body_pos_w[env_ids, tip_id, :].clone()
    tip_quat_w = robot.data.body_quat_w[env_ids, tip_id, :].clone()
    tcp_pos_tip, tcp_quat_tip = math_utils.subtract_frame_transforms(
        tip_pos_w,
        tip_quat_w,
        tcp_pos_w,
        tcp_quat_w,
    )

    state.phase[env_ids] = int(SimpleNicInsertPhase.SEARCH_FOR_PORT)
    state.insert_distance[env_ids] = 0.0
    state.hold_steps[env_ids] = 0
    state.tcp_pos_tip[env_ids] = tcp_pos_tip
    state.tcp_quat_tip[env_ids] = tcp_quat_tip
    state.prev_tip_pos_w[env_ids] = tip_pos_w
    state.prev_tip_vel_w[env_ids] = 0.0
    state.plan_valid[env_ids] = False
    state.plan_start_pos_w[env_ids] = 0.0
    state.plan_start_vel_w[env_ids] = 0.0
    state.plan_start_acc_w[env_ids] = 0.0
    state.plan_end_pos_w[env_ids] = 0.0
    state.plan_end_vel_w[env_ids] = 0.0
    state.plan_end_acc_w[env_ids] = 0.0
    state.plan_start_quat_w[env_ids] = 0.0
    state.plan_start_quat_w[env_ids, 0] = 1.0
    state.plan_approach_quat_w[env_ids] = 0.0
    state.plan_approach_quat_w[env_ids, 0] = 1.0
    state.plan_final_quat_w[env_ids] = 0.0
    state.plan_final_quat_w[env_ids, 0] = 1.0
    state.plan_elapsed[env_ids] = 0.0
    state.plan_duration[env_ids] = 1.0
    state.plan_rot_start[env_ids] = 0.0
    state.plan_rot_duration[env_ids] = 0.0


def compute_simple_nic_insert_oracle(
    env: gym.Env,
    action_scale: torch.Tensor,
    targets_or_state: SimpleNicInsertTargets | SimpleNicInsertState,
    state: SimpleNicInsertState | None = None,
    *,
    command_name: str = "insertion_goal",
    tcp_body: str = "gripper_tcp",
    tip_body: str = "sfp_tip_link",
    pos_gain: float = 1.2,
    rot_gain: float = 0.2,
    max_pos_delta: float = 0.020,
    insert_max_pos_delta: float = 0.002,
    max_rot_delta: float = 0.025,
    port_visible: bool | torch.Tensor = True,
    approach_nominal_speed: float = 0.08,
    approach_end_speed: float = 0.005,
    approach_min_duration: float = 1.0,
    approach_max_duration: float = 5.0,
    approach_rot_speed: float = math.radians(30.0),
    approach_rot_min_duration: float = 0.5,
    approach_rot_margin: float = 0.25,
    approach_threshold: float = 0.015,
    insert_lateral_threshold: float = 0.010,
    insert_orientation_threshold: float = math.radians(4.0),
    insert_lookahead: float = 0.002,
    final_threshold: float = 0.003,
    insert_speed: float = 0.010,
    step_dt: float = 1.0 / 30.0,
) -> SimpleNicInsertOracleOutput:
    """Compute one relative-IK action for direct ``sfp_tip_link`` insertion."""
    if state is None:
        targets = None
        state = targets_or_state
    else:
        targets = targets_or_state

    robot = env.scene["robot"]
    tcp_id = _first_body_id(robot, tcp_body)
    tip_id = _first_body_id(robot, tip_body)
    tcp_pos_w = robot.data.body_pos_w[:, tcp_id, :]
    tcp_quat_w = robot.data.body_quat_w[:, tcp_id, :]
    tip_pos_w = robot.data.body_pos_w[:, tip_id, :]
    tip_quat_w = robot.data.body_quat_w[:, tip_id, :]
    safe_step_dt = max(float(step_dt), 1.0e-6)
    tip_vel_w = _clamp_vector_norm((tip_pos_w - state.prev_tip_pos_w) / safe_step_dt, max_pos_delta / safe_step_dt)

    # Recompute this live.  The cable/tool can flex, so a cached TCP-to-tip
    # transform is exactly the stale-transform trap we hit during debugging.
    tcp_pos_tip, tcp_quat_tip = math_utils.subtract_frame_transforms(
        tip_pos_w,
        tip_quat_w,
        tcp_pos_w,
        tcp_quat_w,
    )
    state.tcp_pos_tip[:] = tcp_pos_tip
    state.tcp_quat_tip[:] = tcp_quat_tip

    world_targets = compute_simple_nic_insert_world_targets(env, targets, command_name=command_name)

    visible_mask = _as_bool_mask(port_visible, env.num_envs, device=env.device)
    search_ready = (state.phase == int(SimpleNicInsertPhase.SEARCH_FOR_PORT)) & visible_mask
    state.phase[search_ready] = int(SimpleNicInsertPhase.PLAN_APPROACH)

    plan_mask = state.phase == int(SimpleNicInsertPhase.PLAN_APPROACH)
    if bool(torch.any(plan_mask)):
        _plan_approach_trajectory(
            state,
            world_targets,
            tip_pos_w,
            tip_vel_w,
            tip_quat_w,
            plan_mask,
            approach_nominal_speed=approach_nominal_speed,
            approach_end_speed=approach_end_speed,
            approach_min_duration=approach_min_duration,
            approach_max_duration=approach_max_duration,
            approach_rot_speed=approach_rot_speed,
            approach_rot_min_duration=approach_rot_min_duration,
            approach_rot_margin=approach_rot_margin,
        )

    approach_motion_mask = (state.phase == int(SimpleNicInsertPhase.APPROACH_P)) | (
        state.phase == int(SimpleNicInsertPhase.APPROACH_P_R)
    )
    start_rotation = approach_motion_mask & (state.plan_elapsed >= state.plan_rot_start).squeeze(1)
    state.phase[start_rotation] = int(SimpleNicInsertPhase.APPROACH_P_R)

    insert_fraction, target_tip_pos_w, target_tip_quat_w = _current_target_tip_pose(
        world_targets,
        state,
        tip_pos_w,
        tip_quat_w,
    )
    tip_position_error = torch.linalg.norm(target_tip_pos_w - tip_pos_w, dim=1)
    orientation_error = math_utils.quat_error_magnitude(tip_quat_w, target_tip_quat_w)
    final_orientation_error = math_utils.quat_error_magnitude(tip_quat_w, world_targets.target_tip_quat_w)
    path_distance, closest_w, path_lateral_error = _project_tip_to_path(world_targets, tip_pos_w)
    path_error_local = _path_error_in_port_frame(world_targets, tip_pos_w, closest_w)

    finished_approach_path = (
        ((state.phase == int(SimpleNicInsertPhase.APPROACH_P)) | (state.phase == int(SimpleNicInsertPhase.APPROACH_P_R)))
        & (state.plan_elapsed.squeeze(1) >= state.plan_duration.squeeze(1))
        & (tip_position_error <= approach_threshold)
        & (orientation_error <= insert_orientation_threshold)
    )
    state.phase[finished_approach_path] = int(SimpleNicInsertPhase.INSERT)

    insert_mask = state.phase == int(SimpleNicInsertPhase.INSERT)
    advance_mask = (
        insert_mask
        & (path_lateral_error <= insert_lateral_threshold)
        & (final_orientation_error <= insert_orientation_threshold)
    )
    state.insert_distance[insert_mask] = path_distance[insert_mask]
    step_distance = max(insert_speed * step_dt, insert_lookahead)
    state.insert_distance[advance_mask] = path_distance[advance_mask] + step_distance
    state.insert_distance[:] = torch.minimum(state.insert_distance, world_targets.path_length)

    insert_fraction, target_tip_pos_w, target_tip_quat_w = _current_target_tip_pose(
        world_targets,
        state,
        tip_pos_w,
        tip_quat_w,
    )
    tip_position_error = torch.linalg.norm(target_tip_pos_w - tip_pos_w, dim=1)
    path_distance, closest_w, path_lateral_error = _project_tip_to_path(world_targets, tip_pos_w)
    path_error_local = _path_error_in_port_frame(world_targets, tip_pos_w, closest_w)
    orientation_error = math_utils.quat_error_magnitude(tip_quat_w, target_tip_quat_w)
    final_orientation_error = math_utils.quat_error_magnitude(tip_quat_w, world_targets.target_tip_quat_w)
    x_axis_error = _axis_angle(_local_axis_w(tip_quat_w, (1.0, 0.0, 0.0)), world_targets.target_x_w)
    y_axis_error = _axis_angle(_local_axis_w(tip_quat_w, (0.0, 1.0, 0.0)), world_targets.target_y_w)
    z_axis_error = _axis_angle(_local_axis_w(tip_quat_w, (0.0, 0.0, 1.0)), world_targets.target_z_w)

    reached_final = (
        (state.phase == int(SimpleNicInsertPhase.INSERT))
        & (insert_fraction.squeeze(1) >= 1.0)
        & (tip_position_error <= final_threshold)
        & (final_orientation_error <= insert_orientation_threshold)
    )
    state.phase[reached_final] = int(SimpleNicInsertPhase.HOLD)
    state.hold_steps[state.phase == int(SimpleNicInsertPhase.HOLD)] += 1

    approach_motion_mask = (state.phase == int(SimpleNicInsertPhase.APPROACH_P)) | (
        state.phase == int(SimpleNicInsertPhase.APPROACH_P_R)
    )
    state.plan_elapsed[approach_motion_mask] = torch.minimum(
        state.plan_elapsed[approach_motion_mask] + safe_step_dt,
        state.plan_duration[approach_motion_mask],
    )

    # Desired TCP pose is just desired tip pose composed with the live
    # TCP-in-tip transform: T_world_tcp_des = T_world_tip_des * T_tip_tcp_live.
    desired_tcp_pos_w, desired_tcp_quat_w = math_utils.combine_frame_transforms(
        target_tip_pos_w,
        target_tip_quat_w,
        tcp_pos_tip,
        tcp_quat_tip,
    )
    tcp_position_error = torch.linalg.norm(desired_tcp_pos_w - tcp_pos_w, dim=1)
    tcp_orientation_error = math_utils.quat_error_magnitude(tcp_quat_w, desired_tcp_quat_w)
    current_insert_mask = state.phase == int(SimpleNicInsertPhase.INSERT)
    per_env_max_pos_delta = torch.where(
        current_insert_mask.unsqueeze(1),
        torch.full_like(path_distance, insert_max_pos_delta),
        torch.full_like(path_distance, max_pos_delta),
    )
    processed_action = _relative_ik_processed_action(
        robot,
        tcp_pos_w,
        tcp_quat_w,
        desired_tcp_pos_w,
        desired_tcp_quat_w,
        action_scale,
        pos_gain=pos_gain,
        rot_gain=rot_gain,
        max_pos_delta=per_env_max_pos_delta,
        max_rot_delta=max_rot_delta,
    )
    raw_action = processed_action / torch.clamp(action_scale, min=1.0e-9)
    state.prev_tip_pos_w[:] = tip_pos_w
    state.prev_tip_vel_w[:] = tip_vel_w
    return SimpleNicInsertOracleOutput(
        raw_action=raw_action,
        processed_action=processed_action,
        tip_pos_w=tip_pos_w,
        target_tip_pos_w=target_tip_pos_w,
        tip_position_error=tip_position_error,
        path_lateral_error=path_lateral_error,
        path_error_local=path_error_local,
        path_distance=path_distance.squeeze(1),
        target_path_distance=state.insert_distance.squeeze(1).clone(),
        orientation_error=orientation_error,
        x_axis_error=x_axis_error,
        y_axis_error=y_axis_error,
        z_axis_error=z_axis_error,
        tcp_position_error=tcp_position_error,
        tcp_orientation_error=tcp_orientation_error,
        phase=state.phase.clone(),
        insert_fraction=insert_fraction.squeeze(1),
    )


def _plan_approach_trajectory(
    state: SimpleNicInsertState,
    world_targets: SimpleNicInsertWorldTargets,
    tip_pos_w: torch.Tensor,
    tip_vel_w: torch.Tensor,
    tip_quat_w: torch.Tensor,
    plan_mask: torch.Tensor,
    *,
    approach_nominal_speed: float,
    approach_end_speed: float,
    approach_min_duration: float,
    approach_max_duration: float,
    approach_rot_speed: float,
    approach_rot_min_duration: float,
    approach_rot_margin: float,
) -> None:
    """Latch a per-env quintic approach plan from the live tip pose."""
    env_ids = plan_mask.nonzero(as_tuple=False).squeeze(-1)
    if env_ids.numel() == 0:
        return

    start_pos = tip_pos_w[env_ids].clone()
    end_pos = world_targets.approach_tip_pos_w[env_ids].clone()
    distance = torch.linalg.norm(end_pos - start_pos, dim=1, keepdim=True)
    nominal_speed = max(float(approach_nominal_speed), 1.0e-6)
    min_duration = max(float(approach_min_duration), 1.0e-3)
    max_duration = max(min_duration, float(approach_max_duration))
    duration = torch.clamp(distance / nominal_speed, min=min_duration, max=max_duration)

    state.plan_start_pos_w[env_ids] = start_pos
    state.plan_start_vel_w[env_ids] = tip_vel_w[env_ids]
    state.plan_start_acc_w[env_ids] = 0.0
    state.plan_end_pos_w[env_ids] = end_pos
    state.plan_end_vel_w[env_ids] = float(approach_end_speed) * world_targets.path_axis_w[env_ids]
    state.plan_end_acc_w[env_ids] = 0.0
    state.plan_start_quat_w[env_ids] = _normalize_rows(tip_quat_w[env_ids])
    state.plan_approach_quat_w[env_ids] = _normalize_rows(world_targets.approach_tip_quat_w[env_ids])
    state.plan_final_quat_w[env_ids] = _normalize_rows(world_targets.target_tip_quat_w[env_ids])
    state.plan_elapsed[env_ids] = 0.0
    state.plan_duration[env_ids] = duration

    rot_angle = math_utils.quat_error_magnitude(
        state.plan_start_quat_w[env_ids],
        state.plan_approach_quat_w[env_ids],
    ).unsqueeze(1)
    rot_speed = max(float(approach_rot_speed), 1.0e-6)
    rot_duration = rot_angle / rot_speed
    has_rotation = rot_angle > 1.0e-5
    rot_duration = torch.where(
        has_rotation,
        torch.clamp(rot_duration, min=max(0.0, float(approach_rot_min_duration))),
        torch.zeros_like(rot_duration),
    )
    rot_duration = torch.minimum(rot_duration, duration)
    rot_start = torch.clamp(duration - rot_duration - max(0.0, float(approach_rot_margin)), min=0.0)
    state.plan_rot_duration[env_ids] = rot_duration
    state.plan_rot_start[env_ids] = rot_start
    state.plan_valid[env_ids] = True
    state.phase[env_ids] = torch.where(
        rot_start.squeeze(1) <= 1.0e-6,
        torch.full_like(env_ids, int(SimpleNicInsertPhase.APPROACH_P_R)),
        torch.full_like(env_ids, int(SimpleNicInsertPhase.APPROACH_P)),
    )


def _current_target_tip_pose(
    world_targets: SimpleNicInsertWorldTargets,
    state: SimpleNicInsertState,
    live_tip_pos_w: torch.Tensor,
    live_tip_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    insert_fraction = (state.insert_distance / world_targets.path_length).clamp(0.0, 1.0)
    insert_target_w = world_targets.nominal_approach_tip_pos_w + world_targets.path_w * insert_fraction
    target_pos_w = insert_target_w
    target_quat_w = world_targets.target_tip_quat_w

    search_mask = (state.phase == int(SimpleNicInsertPhase.SEARCH_FOR_PORT)).unsqueeze(1)
    target_pos_w = torch.where(search_mask, live_tip_pos_w, target_pos_w)
    target_quat_w = torch.where(search_mask, live_tip_quat_w, target_quat_w)

    approach_mask = (
        (state.phase == int(SimpleNicInsertPhase.APPROACH_P))
        | (state.phase == int(SimpleNicInsertPhase.APPROACH_P_R))
    ).unsqueeze(1)
    approach_pos = _quintic_position(
        state.plan_start_pos_w,
        state.plan_start_vel_w,
        state.plan_start_acc_w,
        state.plan_end_pos_w,
        state.plan_end_vel_w,
        state.plan_end_acc_w,
        state.plan_duration,
        state.plan_elapsed,
    )
    target_pos_w = torch.where(approach_mask, approach_pos, target_pos_w)

    no_rotation_mask = (state.phase == int(SimpleNicInsertPhase.APPROACH_P)).unsqueeze(1)
    target_quat_w = torch.where(no_rotation_mask, state.plan_start_quat_w, target_quat_w)

    rotation_mask = (state.phase == int(SimpleNicInsertPhase.APPROACH_P_R)).unsqueeze(1)
    rot_den = torch.clamp(state.plan_rot_duration, min=1.0e-6)
    rot_progress = ((state.plan_elapsed - state.plan_rot_start) / rot_den).clamp(0.0, 1.0)
    rot_progress = torch.where(state.plan_rot_duration <= 1.0e-6, torch.ones_like(rot_progress), rot_progress)
    approach_quat = _quat_slerp(
        state.plan_start_quat_w,
        state.plan_approach_quat_w,
        _smooth5(rot_progress),
    )
    target_quat_w = torch.where(rotation_mask, approach_quat, target_quat_w)

    hold_mask = (state.phase == int(SimpleNicInsertPhase.HOLD)).unsqueeze(1)
    target_pos_w = torch.where(hold_mask, world_targets.final_tip_pos_w, target_pos_w)
    target_quat_w = torch.where(hold_mask, world_targets.target_tip_quat_w, target_quat_w)
    return insert_fraction, target_pos_w, target_quat_w


def _quintic_position(
    p0: torch.Tensor,
    v0: torch.Tensor,
    a0: torch.Tensor,
    p1: torch.Tensor,
    v1: torch.Tensor,
    a1: torch.Tensor,
    duration: torch.Tensor,
    elapsed: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a quintic polynomial with position, velocity, and acceleration constraints."""
    duration = torch.clamp(duration, min=1.0e-6)
    t = torch.minimum(elapsed, duration)
    a = p1 - (p0 + v0 * duration + 0.5 * a0 * duration * duration)
    b = v1 - (v0 + a0 * duration)
    c = a1 - a0
    duration2 = duration * duration
    duration3 = duration2 * duration
    duration4 = duration3 * duration
    duration5 = duration4 * duration
    c3 = (10.0 * a - 4.0 * b * duration + 0.5 * c * duration2) / duration3
    c4 = (-15.0 * a + 7.0 * b * duration - c * duration2) / duration4
    c5 = (6.0 * a - 3.0 * b * duration + 0.5 * c * duration2) / duration5
    return p0 + v0 * t + 0.5 * a0 * t * t + c3 * t**3 + c4 * t**4 + c5 * t**5


def _smooth5(progress: torch.Tensor) -> torch.Tensor:
    progress = progress.clamp(0.0, 1.0)
    return progress**3 * (10.0 - 15.0 * progress + 6.0 * progress * progress)


def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
    """Batch quaternion slerp for Isaac wxyz quaternions."""
    q0 = _normalize_rows(q0)
    q1 = _normalize_rows(q1)
    dot = torch.sum(q0 * q1, dim=1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot).clamp(-1.0, 1.0)
    progress = progress.reshape(-1, 1).to(dtype=q0.dtype, device=q0.device)
    linear = _normalize_rows(q0 + progress * (q1 - q0))
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * progress
    s0 = torch.sin(theta_0 - theta) / torch.clamp(sin_theta_0, min=1.0e-6)
    s1 = torch.sin(theta) / torch.clamp(sin_theta_0, min=1.0e-6)
    spherical = _normalize_rows(s0 * q0 + s1 * q1)
    return torch.where(dot > 0.9995, linear, spherical)


def _project_tip_to_path(
    world_targets: SimpleNicInsertWorldTargets,
    tip_pos_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    path_axis_w = world_targets.path_axis_w
    tip_from_approach = tip_pos_w - world_targets.nominal_approach_tip_pos_w
    path_distance = torch.sum(tip_from_approach * path_axis_w, dim=1, keepdim=True)
    path_distance = torch.clamp(path_distance, min=0.0)
    path_distance = torch.minimum(path_distance, world_targets.path_length)
    closest_w = world_targets.nominal_approach_tip_pos_w + path_axis_w * path_distance
    path_lateral_error = torch.linalg.norm(tip_pos_w - closest_w, dim=1)
    return path_distance, closest_w, path_lateral_error


def _path_error_in_port_frame(
    world_targets: SimpleNicInsertWorldTargets,
    tip_pos_w: torch.Tensor,
    closest_w: torch.Tensor,
) -> torch.Tensor:
    error_w = tip_pos_w - closest_w
    return torch.stack(
        (
            torch.sum(error_w * world_targets.port_x_w, dim=1),
            torch.sum(error_w * world_targets.port_y_w, dim=1),
            torch.sum(error_w * world_targets.port_z_w, dim=1),
        ),
        dim=1,
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
    max_pos_delta: float | torch.Tensor,
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


def _prim_pose_in_asset_root(
    asset_root_path: str,
    prim_path: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Return a prim pose relative to an asset root without geometry helpers."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(asset_root_path)
    _, prim = _resolve_prim(stage, asset_root_path, prim_path)
    if not root_prim.IsValid() or not prim.IsValid():
        candidates = ", ".join(_candidate_prim_paths(asset_root_path, prim_path))
        children = ", ".join(_nearby_child_paths(stage, asset_root_path))
        raise KeyError(
            f"USD prim for root '{asset_root_path}' or port '{prim_path}' was not found. "
            f"Tried: {candidates}. Nearby children: {children}"
        )

    cache = UsdGeom.XformCache()
    root_matrix = cache.GetLocalToWorldTransform(root_prim)
    prim_matrix = cache.GetLocalToWorldTransform(prim)
    root_inv = root_matrix.GetInverse()
    prim_w = prim_matrix.ExtractTranslation()
    prim_root = root_inv.Transform(prim_w)

    axes = []
    for axis in (Gf.Vec3d(1.0, 0.0, 0.0), Gf.Vec3d(0.0, 1.0, 0.0), Gf.Vec3d(0.0, 0.0, 1.0)):
        axis_root = root_inv.TransformDir(prim_matrix.TransformDir(axis))
        axes.append((float(axis_root[0]), float(axis_root[1]), float(axis_root[2])))
    rotation_matrix = torch.tensor(
        [
            [axes[0][0], axes[1][0], axes[2][0]],
            [axes[0][1], axes[1][1], axes[2][1]],
            [axes[0][2], axes[1][2], axes[2][2]],
        ],
        dtype=torch.float64,
    )
    prim_quat_root = math_utils.quat_from_matrix(rotation_matrix.unsqueeze(0))[0]
    return (
        (float(prim_root[0]), float(prim_root[1]), float(prim_root[2])),
        tuple(float(value) for value in prim_quat_root.tolist()),
    )


def _root_point_to_world(root_pos_w: torch.Tensor, root_quat_w: torch.Tensor, point_root: torch.Tensor) -> torch.Tensor:
    return root_pos_w + math_utils.quat_apply(root_quat_w, point_root)


def _target_orientation_offset(reference_quat: torch.Tensor) -> torch.Tensor:
    """Return a batch of local +90 deg X rotations in Isaac's wxyz order."""
    half_angle = math.pi / 4.0
    offset = torch.tensor(
        (math.cos(half_angle), math.sin(half_angle), 0.0, 0.0),
        dtype=reference_quat.dtype,
        device=reference_quat.device,
    )
    return offset.unsqueeze(0).expand(reference_quat.shape[0], -1)


def _local_axis_w(quat_w: torch.Tensor, axis_local: tuple[float, float, float]) -> torch.Tensor:
    axis = torch.tensor(axis_local, dtype=quat_w.dtype, device=quat_w.device).unsqueeze(0)
    return math_utils.quat_apply(quat_w, axis.expand(quat_w.shape[0], -1))


def _axis_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = _normalize_rows(a)
    b = _normalize_rows(b)
    dot = torch.clamp(torch.sum(a * b, dim=1), min=-1.0, max=1.0)
    return torch.acos(dot)


def _as_bool_mask(value: bool | torch.Tensor, num_envs: int, *, device: str) -> torch.Tensor:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return torch.full((num_envs,), bool(value.item()), dtype=torch.bool, device=device)
        return value.to(device=device, dtype=torch.bool).reshape(-1)[:num_envs]
    return torch.full((num_envs,), bool(value), dtype=torch.bool, device=device)


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


def _resolve_asset_root_prim_path(asset, env_index: int, *, usd_root_child: str = "nic_card_link") -> str:
    """Return the USD instance root path for an Isaac Lab asset/env index."""
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


def _nearby_child_paths(stage, asset_root_path: str, *, max_count: int = 24) -> list[str]:
    """Return a short list of descendant paths to make path errors actionable."""
    prefixes = _asset_root_prefixes(asset_root_path)
    paths = []
    for prim in stage.Traverse():
        prim_path = prim.GetPath().pathString
        if any(prim_path == prefix or prim_path.startswith(prefix + "/") for prefix in prefixes):
            paths.append(prim_path)
        if len(paths) >= max_count:
            break
    return paths


def _clamp_vector_norm(vector: torch.Tensor, max_norm: float | torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(max_norm):
        max_norm = torch.tensor(max_norm, dtype=vector.dtype, device=vector.device)
    max_norm = max_norm.to(dtype=vector.dtype, device=vector.device)
    if max_norm.ndim == 0:
        max_norm = max_norm.reshape(1, 1)
    norm = torch.linalg.norm(vector, dim=1, keepdim=True)
    scale = torch.clamp(max_norm / torch.clamp(norm, min=1.0e-9), max=1.0)
    return vector * scale
