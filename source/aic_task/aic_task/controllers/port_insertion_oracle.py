"""Ground-truth port-insertion oracle with force-gated insertion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils

from aic_task.geometry import compute_port_runtime_tensors
from aic_task.vision.port_keypoints import PortKeypointLayout


class InsertionTeacherPhase(IntEnum):
    """Visual/servo phases for V1 insertion demonstrations."""

    SEARCH = 0
    CENTER = 1
    COARSE_APPROACH = 2
    STRAIGHTEN = 3
    ALIGN = 4
    PRE_INSERT = 5
    INSERT = 6
    BACKOFF = 7
    SEAT = 8


@dataclass
class PlugFrameState:
    """Current plug frame state derived from center and tip bodies."""

    center_pos_w: torch.Tensor
    center_quat_w: torch.Tensor
    tip_pos_w: torch.Tensor
    tip_quat_w: torch.Tensor
    axis_w: torch.Tensor
    x_axis_w: torch.Tensor
    length: torch.Tensor


@dataclass
class InsertionForceState:
    """Contact force diagnostics for the insertion controller."""

    net_force_w: torch.Tensor
    force_norm: torch.Tensor
    axis_force: torch.Tensor
    lateral_force: torch.Tensor
    contacting: torch.Tensor
    jammed: torch.Tensor


@dataclass
class PortInsertionOracleOutput:
    """Oracle action and diagnostics for one control step."""

    raw_action: torch.Tensor
    processed_action: torch.Tensor
    phase: InsertionTeacherPhase
    desired_tcp_pos_w: torch.Tensor
    desired_tcp_quat_w: torch.Tensor
    desired_plug_center_pos_w: torch.Tensor
    desired_plug_axis_w: torch.Tensor
    desired_plug_x_axis_w: torch.Tensor
    plug: PlugFrameState
    force: InsertionForceState
    entrance_pos_w: torch.Tensor
    preinsert_pos_w: torch.Tensor
    seat_pos_w: torch.Tensor
    opposite_tooth_pos_w: torch.Tensor
    insertion_axis_w: torch.Tensor
    port_long_axis_w: torch.Tensor
    port_opposite_axis_w: torch.Tensor
    port_y_half: torch.Tensor
    port_insertion_depth: torch.Tensor
    tcp_quat_w: torch.Tensor
    tcp_x_axis_w: torch.Tensor
    tip_delta_port_frame: torch.Tensor
    roll_delta: torch.Tensor
    tip_to_target: torch.Tensor
    axis_error: torch.Tensor
    x_axis_error: torch.Tensor


def get_action_scale(env: gym.Env, action_dim: int) -> torch.Tensor:
    """Return the Differential IK action scale used to convert processed to raw actions."""
    action_term = env.action_manager.get_term("arm_action")
    scale = getattr(action_term, "_scale", None)
    if scale is None:
        return torch.ones((env.num_envs, action_dim), device=env.device)
    return scale[:, :action_dim]


def compute_port_insertion_oracle(
    env: gym.Env,
    action_scale: torch.Tensor,
    *,
    labels: dict,
    layout: PortKeypointLayout,
    phase: InsertionTeacherPhase | None = None,
    plug_center_body: str = "sfp_module_link",
    plug_tip_body: str = "sfp_tip_link",
    tcp_body: str = "gripper_tcp",
    target_name: str = "nic_card",
    port_name: str = "sfp_port_0",
    contact_sensor_name: str = "plug_contact_forces",
    contact_body_regex: str = ".*sfp.*|.*plug.*|.*tip.*",
    plug_x_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
    tcp_x_axis_local: tuple[float, float, float] = (1.0, 0.0, 0.0),
    target_offset_port_frame: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_roll_offset: float = 0.0,
    pos_gain: float = 0.7,
    rot_gain: float = 0.45,
    insert_pos_gain: float = 1.2,
    insert_rot_gain: float = 0.8,
    max_pos_delta: float = 0.012,
    max_rot_delta: float = 0.14,
    insert_max_pos_delta: float = 0.001,
    insert_max_rot_delta: float = 0.045,
    contact_force_threshold: float = 1.0,
    lateral_force_limit: float = 400.0,
    axis_force_limit: float = 1000.0,
    force_phase_backoff: bool = True,
    straighten_axis_mode: Literal["port", "world_down", "disabled"] = "disabled",
    center_enable_distance: float = 0.060,
    rotation_enable_distance: float = 0.6,
    align_lift: float = 0.070,
    env_index: int = 0,
) -> PortInsertionOracleOutput:
    """Compute the force-gated privileged insertion action."""
    robot = env.scene["robot"]
    port = compute_port_runtime_tensors(env, target_name=target_name, port_name=port_name)
    tcp_id = _first_body_id(robot, tcp_body)
    tcp_pos_w = robot.data.body_pos_w[:, tcp_id, :]
    tcp_quat_w = robot.data.body_quat_w[:, tcp_id, :]
    tcp_x_axis_w = _local_axis_w(tcp_quat_w, tcp_x_axis_local)
    plug = read_plug_frame_state(
        env,
        plug_center_body=plug_center_body,
        plug_tip_body=plug_tip_body,
        plug_x_axis_local=plug_x_axis_local,
    )
    straighten_axis_w = _straighten_axis_for_mode(straighten_axis_mode, plug.axis_w, port.insertion_axis_w)
    force = read_insertion_force_state(
        env,
        insertion_axis_w=port.insertion_axis_w,
        sensor_name=contact_sensor_name,
        body_regex=contact_body_regex,
        contact_force_threshold=contact_force_threshold,
        lateral_force_limit=lateral_force_limit,
        axis_force_limit=axis_force_limit,
    )
    phase_target_x_axis_w = _rotate_axis_about_axis(port.opposite_axis_w, port.insertion_axis_w, target_roll_offset)

    if phase is None:
        phase = choose_insertion_phase(
            labels,
            layout,
            plug=plug,
            force=force,
            preinsert_w=port.preinsert_w,
            coarse_approach_w=port.coarse_approach_w,
            seat_w=port.seat_w,
            insertion_axis_w=port.insertion_axis_w,
            target_x_axis_w=phase_target_x_axis_w,
            current_x_axis_w=tcp_x_axis_w,
            straighten_axis_w=straighten_axis_w,
            force_phase_backoff=force_phase_backoff,
            center_enable_distance=center_enable_distance,
            env_index=env_index,
        )

    desired_tip_pos_w, desired_axis_w = _desired_tip_and_axis_for_phase(
        phase,
        plug,
        port,
        env_index,
        straighten_axis_w=straighten_axis_w,
    )
    target_offset_w = _port_frame_vector_to_world(
        target_offset_port_frame,
        port.long_axis_w,
        port.opposite_axis_w,
        port.insertion_axis_w,
    )
    if phase != InsertionTeacherPhase.STRAIGHTEN:
        desired_tip_pos_w = desired_tip_pos_w + target_offset_w
    if phase == InsertionTeacherPhase.ALIGN and align_lift != 0.0:
        desired_tip_pos_w = _apply_world_z_lift(desired_tip_pos_w, align_lift)
    tip_to_coarse = torch.linalg.norm(port.coarse_approach_w - plug.tip_pos_w, dim=1)
    rotation_enabled = float(tip_to_coarse[env_index]) <= rotation_enable_distance or phase in (
        InsertionTeacherPhase.PRE_INSERT,
        InsertionTeacherPhase.INSERT,
        InsertionTeacherPhase.SEAT,
        InsertionTeacherPhase.BACKOFF,
    )
    if not rotation_enabled:
        desired_axis_w = plug.axis_w
    desired_x_axis_w = _project_axis_onto_plane(port.opposite_axis_w, desired_axis_w)
    desired_x_axis_w = _rotate_axis_about_axis(desired_x_axis_w, desired_axis_w, target_roll_offset)
    if not rotation_enabled:
        desired_x_axis_w = _project_axis_onto_plane(tcp_x_axis_w, desired_axis_w)
    desired_center_pos_w = desired_tip_pos_w - desired_axis_w * plug.length
    if phase == InsertionTeacherPhase.STRAIGHTEN:
        desired_center_pos_w = plug.center_pos_w.clone()

    desired_tcp_pos_w, desired_tcp_quat_w = _desired_tcp_pose_from_plug_frame(
        tcp_pos_w,
        tcp_quat_w,
        tcp_x_axis_w,
        plug,
        desired_center_pos_w,
        desired_axis_w,
        desired_x_axis_w,
    )
    is_insert_phase = phase == InsertionTeacherPhase.INSERT
    processed_action = _relative_ik_processed_action(
        robot,
        tcp_pos_w,
        tcp_quat_w,
        desired_tcp_pos_w,
        desired_tcp_quat_w,
        action_scale,
        pos_gain=insert_pos_gain if is_insert_phase else pos_gain,
        rot_gain=insert_rot_gain if is_insert_phase else rot_gain,
        max_pos_delta=insert_max_pos_delta if is_insert_phase else max_pos_delta,
        max_rot_delta=insert_max_rot_delta if is_insert_phase else max_rot_delta,
    )
    raw_action = processed_action / torch.clamp(action_scale, min=1.0e-9)

    tip_target = port.seat_w if phase in (InsertionTeacherPhase.INSERT, InsertionTeacherPhase.SEAT) else desired_tip_pos_w
    if phase in (InsertionTeacherPhase.INSERT, InsertionTeacherPhase.SEAT):
        tip_target = tip_target + target_offset_w
    tip_delta_port_frame = _world_vector_to_port_frame(
        tip_target - plug.tip_pos_w,
        port.long_axis_w,
        port.opposite_axis_w,
        port.insertion_axis_w,
    )
    tip_to_target = torch.linalg.norm(tip_target - plug.tip_pos_w, dim=1)
    axis_error = _axis_error(plug.axis_w, desired_axis_w)
    current_x_axis_w = _project_axis_onto_plane(tcp_x_axis_w, desired_axis_w)
    x_axis_error = _axis_error(current_x_axis_w, desired_x_axis_w)
    roll_delta = _signed_axis_angle(current_x_axis_w, desired_x_axis_w, desired_axis_w)

    return PortInsertionOracleOutput(
        raw_action=raw_action,
        processed_action=processed_action,
        phase=phase,
        desired_tcp_pos_w=desired_tcp_pos_w,
        desired_tcp_quat_w=desired_tcp_quat_w,
        desired_plug_center_pos_w=desired_center_pos_w,
        desired_plug_axis_w=desired_axis_w,
        desired_plug_x_axis_w=desired_x_axis_w,
        plug=plug,
        force=force,
        entrance_pos_w=port.entrance_w,
        preinsert_pos_w=port.preinsert_w,
        seat_pos_w=port.seat_w,
        opposite_tooth_pos_w=port.opposite_tooth_w,
        insertion_axis_w=port.insertion_axis_w,
        port_long_axis_w=port.long_axis_w,
        port_opposite_axis_w=port.opposite_axis_w,
        port_y_half=port.y_half,
        port_insertion_depth=port.insertion_depth,
        tcp_quat_w=tcp_quat_w,
        tcp_x_axis_w=tcp_x_axis_w,
        tip_delta_port_frame=tip_delta_port_frame,
        roll_delta=roll_delta,
        tip_to_target=tip_to_target,
        axis_error=axis_error,
        x_axis_error=x_axis_error,
    )


def choose_insertion_phase(
    labels: dict,
    layout: PortKeypointLayout,
    *,
    plug: PlugFrameState,
    force: InsertionForceState,
    preinsert_w: torch.Tensor,
    coarse_approach_w: torch.Tensor,
    seat_w: torch.Tensor,
    insertion_axis_w: torch.Tensor,
    target_x_axis_w: torch.Tensor | None = None,
    current_x_axis_w: torch.Tensor | None = None,
    straighten_axis_w: torch.Tensor | None = None,
    force_phase_backoff: bool,
    env_index: int,
    center_camera: str = "center_camera",
    vertical_threshold: float = math.radians(8.0),
    align_axis_threshold: float = math.radians(8.0),
    coarse_threshold: float = 0.2,
    center_enable_distance: float = 0.060,
    preinsert_threshold: float = 0.18,
    seat_threshold: float = 0.004,
) -> InsertionTeacherPhase:
    """Choose the visual-compatible insertion phase."""
    cameras = labels["cameras"]
    camera_masks = {name: camera_labels["in_frame"][env_index] for name, camera_labels in cameras.items()}
    any_keypoint_in_frame = any(bool(mask.any()) for mask in camera_masks.values())

    del center_camera
    tip_to_coarse = torch.linalg.norm(coarse_approach_w - plug.tip_pos_w, dim=1)
    tip_to_preinsert = torch.linalg.norm(preinsert_w - plug.tip_pos_w, dim=1)
    tip_to_seat = torch.linalg.norm(seat_w - plug.tip_pos_w, dim=1)

    keypoint_names = tuple(labels.get("keypoint_names", layout.names))
    important_names = ("entrance_center", "opposite_tooth_anchor", "tooth_anchor")
    important_indices = [keypoint_names.index(name) for name in important_names if name in keypoint_names]
    important_in_any_camera = 0
    for index in important_indices:
        if any(bool(mask[index].item()) for mask in camera_masks.values()):
            important_in_any_camera += 1
    close_enough_to_center = float(tip_to_coarse[env_index]) <= center_enable_distance
    if (
        important_indices
        and any_keypoint_in_frame
        and close_enough_to_center
        and important_in_any_camera < min(2, len(important_indices))
    ):
        return InsertionTeacherPhase.CENTER

    # STRAIGHTEN rotates in-place; use the actual port axis unless a caller asks for legacy world-down.
    if straighten_axis_w is not None:
        straighten_error = _axis_error(plug.axis_w, straighten_axis_w)
        if float(straighten_error[env_index]) > vertical_threshold:
            return InsertionTeacherPhase.STRAIGHTEN

    axis_error = _axis_error(plug.axis_w, insertion_axis_w)
    if target_x_axis_w is None:
        x_axis_error = torch.zeros_like(axis_error)
    else:
        source_x_axis_w = plug.x_axis_w if current_x_axis_w is None else current_x_axis_w
        plug_x_axis = _project_axis_onto_plane(source_x_axis_w, insertion_axis_w)
        target_x_axis = _project_axis_onto_plane(target_x_axis_w, insertion_axis_w)
        x_axis_error = _axis_error(plug_x_axis, target_x_axis)
    if force_phase_backoff and bool(force.jammed[env_index]):
        return InsertionTeacherPhase.BACKOFF
    axis_aligned = float(axis_error[env_index]) <= align_axis_threshold
    x_aligned = float(x_axis_error[env_index]) <= align_axis_threshold
    if float(tip_to_seat[env_index]) <= seat_threshold and axis_aligned and x_aligned:
        return InsertionTeacherPhase.SEAT
    if float(tip_to_coarse[env_index]) > coarse_threshold and float(tip_to_preinsert[env_index]) > preinsert_threshold:
        return InsertionTeacherPhase.COARSE_APPROACH
    if not axis_aligned or not x_aligned:
        return InsertionTeacherPhase.ALIGN
    if float(tip_to_preinsert[env_index]) > preinsert_threshold:
        return InsertionTeacherPhase.PRE_INSERT
    return InsertionTeacherPhase.INSERT


def read_plug_frame_state(
    env: gym.Env,
    *,
    plug_center_body: str,
    plug_tip_body: str,
    plug_x_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> PlugFrameState:
    """Read plug center/tip body poses and derive the plug axis."""
    robot = env.scene["robot"]
    center_id = _first_body_id(robot, plug_center_body)
    tip_id = _first_body_id(robot, plug_tip_body)
    center_pos = robot.data.body_pos_w[:, center_id, :]
    center_quat = robot.data.body_quat_w[:, center_id, :]
    tip_pos = robot.data.body_pos_w[:, tip_id, :]
    tip_quat = robot.data.body_quat_w[:, tip_id, :]
    axis_vec = tip_pos - center_pos
    length = torch.linalg.norm(axis_vec, dim=1, keepdim=True).clamp_min(1.0e-9)
    axis = axis_vec / length
    local_x_axis = torch.tensor(plug_x_axis_local, dtype=center_pos.dtype, device=center_pos.device).unsqueeze(0)
    x_axis = _project_axis_onto_plane(
        math_utils.quat_apply(center_quat, local_x_axis.expand(center_pos.shape[0], -1)),
        axis,
    )
    return PlugFrameState(
        center_pos_w=center_pos,
        center_quat_w=center_quat,
        tip_pos_w=tip_pos,
        tip_quat_w=tip_quat,
        axis_w=axis,
        x_axis_w=x_axis,
        length=length,
    )


def read_insertion_force_state(
    env: gym.Env,
    *,
    insertion_axis_w: torch.Tensor,
    sensor_name: str,
    body_regex: str,
    contact_force_threshold: float,
    lateral_force_limit: float,
    axis_force_limit: float,
) -> InsertionForceState:
    """Read plug contact forces and decompose them along/lateral to insertion axis."""
    if sensor_name not in env.scene.sensors:
        zeros = torch.zeros((env.num_envs, 3), dtype=insertion_axis_w.dtype, device=insertion_axis_w.device)
        scalar = torch.zeros(env.num_envs, dtype=insertion_axis_w.dtype, device=insertion_axis_w.device)
        return InsertionForceState(zeros, scalar, scalar, scalar, scalar.bool(), scalar.bool())

    import re

    sensor = env.scene.sensors[sensor_name]
    forces = sensor.data.net_forces_w
    body_ids = [idx for idx, name in enumerate(sensor.body_names) if re.search(body_regex, name)]
    if not body_ids:
        body_ids = list(range(forces.shape[1]))
    net_force = forces[:, body_ids, :].sum(dim=1)
    axis_component = torch.sum(net_force * insertion_axis_w, dim=1)
    axis_force = torch.abs(axis_component)
    lateral_vec = net_force - axis_component[:, None] * insertion_axis_w
    lateral_force = torch.linalg.norm(lateral_vec, dim=1)
    force_norm = torch.linalg.norm(net_force, dim=1)
    contacting = force_norm > contact_force_threshold
    jammed = (lateral_force > lateral_force_limit) | (axis_force > axis_force_limit)
    return InsertionForceState(net_force, force_norm, axis_force, lateral_force, contacting, jammed)


def apply_insertion_phase_gate(raw_action: torch.Tensor, phase: InsertionTeacherPhase, *, env_index: int = 0) -> torch.Tensor:
    """Keep early visual phases conservative."""
    gated = raw_action.clone()
    if phase in (InsertionTeacherPhase.SEARCH, InsertionTeacherPhase.CENTER):
        gated[env_index, 3:6] = 0.0
    if phase == InsertionTeacherPhase.SEAT:
        gated[env_index] *= 0.2
    return gated


apply_phase_gate = apply_insertion_phase_gate


def _desired_tip_and_axis_for_phase(
    phase: InsertionTeacherPhase,
    plug: PlugFrameState,
    port,
    env_index: int,
    *,
    straighten_axis_w: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del env_index
    if phase == InsertionTeacherPhase.STRAIGHTEN:
        desired_axis_w = port.insertion_axis_w if straighten_axis_w is None else straighten_axis_w
        return plug.tip_pos_w, desired_axis_w
    if phase in (InsertionTeacherPhase.SEARCH, InsertionTeacherPhase.CENTER, InsertionTeacherPhase.COARSE_APPROACH):
        return port.coarse_approach_w, port.insertion_axis_w
    if phase == InsertionTeacherPhase.BACKOFF:
        return port.backoff_w, port.insertion_axis_w
    if phase in (InsertionTeacherPhase.INSERT, InsertionTeacherPhase.SEAT):
        return port.seat_w, port.insertion_axis_w
    return port.preinsert_w, port.insertion_axis_w


def _desired_tcp_pose_from_plug_frame(
    tcp_pos_w: torch.Tensor,
    tcp_quat_w: torch.Tensor,
    tcp_x_axis_w: torch.Tensor,
    plug: PlugFrameState,
    desired_center_pos_w: torch.Tensor,
    desired_axis_w: torch.Tensor,
    desired_x_axis_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_delta = _quat_between_frames(tcp_x_axis_w, plug.axis_w, desired_x_axis_w, desired_axis_w)
    # q_delta is built from world-frame axes, so it must be applied as a world-frame delta.
    desired_tcp_quat = math_utils.quat_mul(q_delta, tcp_quat_w)
    center_pos_tcp, _ = math_utils.subtract_frame_transforms(
        tcp_pos_w,
        tcp_quat_w,
        plug.center_pos_w,
        plug.center_quat_w,
    )
    desired_tcp_pos = desired_center_pos_w - math_utils.quat_apply(desired_tcp_quat, center_pos_tcp)
    return desired_tcp_pos, desired_tcp_quat


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
    processed_action = torch.zeros((tcp_pos_w.shape[0], action_scale.shape[1]), dtype=tcp_pos_w.dtype, device=tcp_pos_w.device)
    processed_action[:, 0:3] = _clamp_vector_norm(pos_error_b * pos_gain, max_pos_delta)
    processed_action[:, 3:6] = _clamp_vector_norm(rot_error_b * rot_gain, max_rot_delta)
    return processed_action


def _first_body_id(robot, body_name: str) -> int:
    body_ids = robot.find_bodies(body_name, preserve_order=True)[0]
    if len(body_ids) == 0:
        available = ", ".join(getattr(robot, "body_names", []))
        raise KeyError(f"Robot body '{body_name}' not found. Available robot bodies: {available}")
    return int(body_ids[0])


def _local_axis_w(quat_w: torch.Tensor, axis_local: tuple[float, float, float]) -> torch.Tensor:
    axis = torch.tensor(axis_local, dtype=quat_w.dtype, device=quat_w.device).unsqueeze(0)
    return math_utils.quat_apply(quat_w, axis.expand(quat_w.shape[0], -1))


def _apply_world_z_lift(position_w: torch.Tensor, lift: float) -> torch.Tensor:
    lifted = position_w.clone()
    lifted[:, 2] += lift
    return lifted


def _port_frame_vector_to_world(
    values: tuple[float, float, float],
    x_axis_w: torch.Tensor,
    y_axis_w: torch.Tensor,
    z_axis_w: torch.Tensor,
) -> torch.Tensor:
    components = torch.tensor(values, dtype=x_axis_w.dtype, device=x_axis_w.device)
    return components[0] * x_axis_w + components[1] * y_axis_w + components[2] * z_axis_w


def _world_vector_to_port_frame(
    vector_w: torch.Tensor,
    x_axis_w: torch.Tensor,
    y_axis_w: torch.Tensor,
    z_axis_w: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        (
            torch.sum(vector_w * x_axis_w, dim=1),
            torch.sum(vector_w * y_axis_w, dim=1),
            torch.sum(vector_w * z_axis_w, dim=1),
        ),
        dim=1,
    )


def _rotate_axis_about_axis(axis_w: torch.Tensor, rotation_axis_w: torch.Tensor, angle: float) -> torch.Tensor:
    if abs(angle) <= 1.0e-12:
        return axis_w
    angles = torch.full((axis_w.shape[0],), angle, dtype=axis_w.dtype, device=axis_w.device)
    q_delta = math_utils.quat_from_angle_axis(angles, rotation_axis_w)
    return math_utils.quat_apply(q_delta, axis_w)


def _signed_axis_angle(source_axis_w: torch.Tensor, target_axis_w: torch.Tensor, rotation_axis_w: torch.Tensor) -> torch.Tensor:
    source_axis_w = _project_axis_onto_plane(source_axis_w, rotation_axis_w)
    target_axis_w = _project_axis_onto_plane(target_axis_w, rotation_axis_w)
    cross = torch.linalg.cross(source_axis_w, target_axis_w, dim=1)
    sin_angle = torch.sum(cross * rotation_axis_w, dim=1)
    cos_angle = torch.sum(source_axis_w * target_axis_w, dim=1).clamp(-1.0, 1.0)
    return torch.atan2(sin_angle, cos_angle)


def _straighten_axis_for_mode(
    mode: Literal["port", "world_down", "disabled"],
    plug_axis_w: torch.Tensor,
    insertion_axis_w: torch.Tensor,
) -> torch.Tensor | None:
    if mode == "port":
        return insertion_axis_w
    if mode == "world_down":
        return _world_down_axis(plug_axis_w)
    if mode == "disabled":
        return None
    raise ValueError(f"Unsupported straighten_axis_mode: {mode!r}. Expected 'port', 'world_down', or 'disabled'.")


def _world_down_axis(like: torch.Tensor) -> torch.Tensor:
    axis = torch.zeros_like(like)
    axis[:, 2] = -1.0
    return axis


def _axis_error(current_axis: torch.Tensor, desired_axis: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(current_axis * desired_axis, dim=1).clamp(-1.0, 1.0)
    return torch.acos(dot)


def _quat_between_frames(
    source_x_axis: torch.Tensor,
    source_z_axis: torch.Tensor,
    target_x_axis: torch.Tensor,
    target_z_axis: torch.Tensor,
) -> torch.Tensor:
    source_frame = _frame_matrix_from_xz(source_x_axis, source_z_axis)
    target_frame = _frame_matrix_from_xz(target_x_axis, target_z_axis)
    delta_frame = torch.matmul(target_frame, source_frame.transpose(-1, -2))
    return math_utils.quat_from_matrix(delta_frame)


def _frame_matrix_from_xz(x_axis: torch.Tensor, z_axis: torch.Tensor) -> torch.Tensor:
    z_axis = torch.nn.functional.normalize(z_axis, dim=1)
    x_axis = _project_axis_onto_plane(x_axis, z_axis)
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=1)
    y_axis = torch.nn.functional.normalize(y_axis, dim=1)
    x_axis = torch.linalg.cross(y_axis, z_axis, dim=1)
    x_axis = torch.nn.functional.normalize(x_axis, dim=1)
    return torch.stack((x_axis, y_axis, z_axis), dim=-1)


def _project_axis_onto_plane(axis: torch.Tensor, plane_normal: torch.Tensor) -> torch.Tensor:
    plane_normal = torch.nn.functional.normalize(plane_normal, dim=1)
    projected = axis - torch.sum(axis * plane_normal, dim=1, keepdim=True) * plane_normal
    norm = torch.linalg.norm(projected, dim=1, keepdim=True)
    fallback_axis = _orthogonal_axis(plane_normal)
    return torch.where(norm > 1.0e-6, projected / norm.clamp_min(1.0e-9), fallback_axis)


def _quat_between_vectors(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source = torch.nn.functional.normalize(source, dim=1)
    target = torch.nn.functional.normalize(target, dim=1)
    cross = torch.linalg.cross(source, target, dim=1)
    dot = torch.sum(source * target, dim=1).clamp(-1.0, 1.0)
    angle = torch.acos(dot)
    axis_norm = torch.linalg.norm(cross, dim=1, keepdim=True)
    fallback_axis = _orthogonal_axis(source)
    axis = torch.where(axis_norm > 1.0e-6, cross / axis_norm.clamp_min(1.0e-9), fallback_axis)
    return math_utils.quat_from_angle_axis(angle, axis)


def _orthogonal_axis(vector: torch.Tensor) -> torch.Tensor:
    """Return a stable unit axis orthogonal to each input vector."""
    x_axis = torch.zeros_like(vector)
    x_axis[:, 0] = 1.0
    y_axis = torch.zeros_like(vector)
    y_axis[:, 1] = 1.0
    candidate = torch.where(torch.abs(vector[:, 0:1]) < 0.9, x_axis, y_axis)
    axis = torch.linalg.cross(vector, candidate, dim=1)
    return torch.nn.functional.normalize(axis, dim=1)


def _clamp_vector_norm(vector: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.linalg.norm(vector, dim=1, keepdim=True)
    scale = torch.clamp(max_norm / torch.clamp(norm, min=1.0e-9), max=1.0)
    return vector * scale
