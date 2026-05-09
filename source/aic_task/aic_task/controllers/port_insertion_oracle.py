"""Ground-truth port-insertion oracle with force-gated insertion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

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
    plug: PlugFrameState
    force: InsertionForceState
    entrance_pos_w: torch.Tensor
    preinsert_pos_w: torch.Tensor
    seat_pos_w: torch.Tensor
    insertion_axis_w: torch.Tensor
    tip_to_target: torch.Tensor
    axis_error: torch.Tensor


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
    pos_gain: float = 0.7,
    rot_gain: float = 0.45,
    max_pos_delta: float = 0.012,
    max_rot_delta: float = 0.14,
    insert_max_pos_delta: float = 0.002,
    insert_max_rot_delta: float = 0.045,
    contact_force_threshold: float = 1.0,
    lateral_force_limit: float = 4.0,
    axis_force_limit: float = 10.0,
    force_phase_backoff: bool = True,
    env_index: int = 0,
) -> PortInsertionOracleOutput:
    """Compute the force-gated privileged insertion action."""
    robot = env.scene["robot"]
    port = compute_port_runtime_tensors(env, target_name=target_name, port_name=port_name)
    plug = read_plug_frame_state(env, plug_center_body=plug_center_body, plug_tip_body=plug_tip_body)
    force = read_insertion_force_state(
        env,
        insertion_axis_w=port.insertion_axis_w,
        sensor_name=contact_sensor_name,
        body_regex=contact_body_regex,
        contact_force_threshold=contact_force_threshold,
        lateral_force_limit=lateral_force_limit,
        axis_force_limit=axis_force_limit,
    )

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
            force_phase_backoff=force_phase_backoff,
            env_index=env_index,
        )

    desired_tip_pos_w, desired_axis_w = _desired_tip_and_axis_for_phase(phase, plug, port, env_index)
    desired_center_pos_w = desired_tip_pos_w - desired_axis_w * plug.length
    if phase == InsertionTeacherPhase.STRAIGHTEN:
        desired_center_pos_w = plug.center_pos_w.clone()

    tcp_id = _first_body_id(robot, tcp_body)
    tcp_pos_w = robot.data.body_pos_w[:, tcp_id, :]
    tcp_quat_w = robot.data.body_quat_w[:, tcp_id, :]

    desired_tcp_pos_w, desired_tcp_quat_w = _desired_tcp_pose_from_plug_axis(
        tcp_pos_w,
        tcp_quat_w,
        plug,
        desired_center_pos_w,
        desired_axis_w,
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
        max_pos_delta=insert_max_pos_delta if phase == InsertionTeacherPhase.INSERT else max_pos_delta,
        max_rot_delta=insert_max_rot_delta if phase == InsertionTeacherPhase.INSERT else max_rot_delta,
    )
    raw_action = processed_action / torch.clamp(action_scale, min=1.0e-9)

    tip_target = port.seat_w if phase in (InsertionTeacherPhase.INSERT, InsertionTeacherPhase.SEAT) else desired_tip_pos_w
    tip_to_target = torch.linalg.norm(tip_target - plug.tip_pos_w, dim=1)
    axis_error = _axis_error(plug.axis_w, desired_axis_w)

    return PortInsertionOracleOutput(
        raw_action=raw_action,
        processed_action=processed_action,
        phase=phase,
        desired_tcp_pos_w=desired_tcp_pos_w,
        desired_tcp_quat_w=desired_tcp_quat_w,
        desired_plug_center_pos_w=desired_center_pos_w,
        desired_plug_axis_w=desired_axis_w,
        plug=plug,
        force=force,
        entrance_pos_w=port.entrance_w,
        preinsert_pos_w=port.preinsert_w,
        seat_pos_w=port.seat_w,
        insertion_axis_w=port.insertion_axis_w,
        tip_to_target=tip_to_target,
        axis_error=axis_error,
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
    force_phase_backoff: bool,
    env_index: int,
    center_camera: str = "center_camera",
    vertical_threshold: float = math.radians(8.0),
    align_axis_threshold: float = math.radians(8.0),
    coarse_threshold: float = 0.020,
    preinsert_threshold: float = 0.012,
    seat_threshold: float = 0.004,
) -> InsertionTeacherPhase:
    """Choose the visual-compatible insertion phase."""
    cameras = labels["cameras"]
    camera_masks = {name: camera_labels["in_frame"][env_index] for name, camera_labels in cameras.items()}
    if not any(bool(mask.any()) for mask in camera_masks.values()):
        return InsertionTeacherPhase.SEARCH

    center_mask = camera_masks.get(center_camera, next(iter(camera_masks.values())))
    keypoint_names = tuple(labels.get("keypoint_names", layout.names))
    important_names = ("entrance_center", "opposite_tooth_anchor", "tooth_anchor")
    important_indices = [keypoint_names.index(name) for name in important_names if name in keypoint_names]
    if important_indices and int(center_mask[important_indices].sum().item()) < min(2, len(important_indices)):
        return InsertionTeacherPhase.CENTER

    vertical_axis = _world_down_axis(plug.axis_w)
    vertical_error = _axis_error(plug.axis_w, vertical_axis)
    if float(vertical_error[env_index]) > vertical_threshold:
        return InsertionTeacherPhase.STRAIGHTEN

    axis_error = _axis_error(plug.axis_w, insertion_axis_w)
    tip_to_coarse = torch.linalg.norm(coarse_approach_w - plug.tip_pos_w, dim=1)
    tip_to_preinsert = torch.linalg.norm(preinsert_w - plug.tip_pos_w, dim=1)
    tip_to_seat = torch.linalg.norm(seat_w - plug.tip_pos_w, dim=1)

    if force_phase_backoff and bool(force.jammed[env_index]):
        return InsertionTeacherPhase.BACKOFF
    if float(tip_to_seat[env_index]) <= seat_threshold and float(axis_error[env_index]) <= align_axis_threshold:
        return InsertionTeacherPhase.SEAT
    if float(axis_error[env_index]) > align_axis_threshold:
        return InsertionTeacherPhase.ALIGN
    if float(tip_to_coarse[env_index]) > coarse_threshold and float(tip_to_preinsert[env_index]) > preinsert_threshold:
        return InsertionTeacherPhase.COARSE_APPROACH
    if float(tip_to_preinsert[env_index]) > preinsert_threshold:
        return InsertionTeacherPhase.PRE_INSERT
    return InsertionTeacherPhase.INSERT


def read_plug_frame_state(
    env: gym.Env,
    *,
    plug_center_body: str,
    plug_tip_body: str,
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
    return PlugFrameState(
        center_pos_w=center_pos,
        center_quat_w=center_quat,
        tip_pos_w=tip_pos,
        tip_quat_w=tip_quat,
        axis_w=axis,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    del env_index
    if phase == InsertionTeacherPhase.STRAIGHTEN:
        return plug.tip_pos_w, _world_down_axis(plug.axis_w)
    if phase in (InsertionTeacherPhase.SEARCH, InsertionTeacherPhase.CENTER, InsertionTeacherPhase.COARSE_APPROACH):
        return port.coarse_approach_w, port.insertion_axis_w
    if phase == InsertionTeacherPhase.BACKOFF:
        return port.backoff_w, port.insertion_axis_w
    if phase in (InsertionTeacherPhase.INSERT, InsertionTeacherPhase.SEAT):
        return port.seat_w, port.insertion_axis_w
    return port.preinsert_w, port.insertion_axis_w


def _desired_tcp_pose_from_plug_axis(
    tcp_pos_w: torch.Tensor,
    tcp_quat_w: torch.Tensor,
    plug: PlugFrameState,
    desired_center_pos_w: torch.Tensor,
    desired_axis_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_delta = _quat_between_vectors(plug.axis_w, desired_axis_w)
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


def _world_down_axis(like: torch.Tensor) -> torch.Tensor:
    axis = torch.zeros_like(like)
    axis[:, 2] = -1.0
    return axis


def _axis_error(current_axis: torch.Tensor, desired_axis: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(current_axis * desired_axis, dim=1).clamp(-1.0, 1.0)
    return torch.acos(dot)


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
