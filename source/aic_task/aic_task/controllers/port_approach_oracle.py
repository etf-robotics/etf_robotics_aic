"""Ground-truth port-approach oracle with visual-phase gating."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils

from aic_task.tasks.manager_based.port_approach.port_approach_env_cfg import (
    CABLE_TIP_OFFSET_FROM_TCP,
    CABLE_TIP_RPY_FROM_TCP,
    NIC_PORT_APPROACH_OFFSET,
    NIC_PORT_APPROACH_RPY,
    TARGET_NAME,
)
from aic_task.vision.port_keypoints import PortKeypointLayout


class TeacherPhase(IntEnum):
    """Visual pre-approach phases stored in the dataset."""

    SEARCH = 0
    CENTER = 1
    COARSE_APPROACH = 2
    ALIGN = 3
    HOLD = 4


@dataclass
class PortApproachOracleOutput:
    """Oracle action and privileged diagnostics for one control step."""

    raw_action: torch.Tensor
    processed_action: torch.Tensor
    position_error: torch.Tensor
    orientation_error: torch.Tensor
    desired_tcp_pos_w: torch.Tensor
    desired_tcp_quat_w: torch.Tensor
    target_tip_pos_w: torch.Tensor
    target_tip_quat_w: torch.Tensor
    current_tip_pos_w: torch.Tensor
    current_tip_quat_w: torch.Tensor


def get_action_scale(env: gym.Env, action_dim: int) -> torch.Tensor:
    """Return the Differential IK action scale used to convert processed to raw actions."""
    action_term = env.action_manager.get_term("arm_action")
    scale = getattr(action_term, "_scale", None)
    if scale is None:
        return torch.ones((env.num_envs, action_dim), device=env.device)
    return scale[:, :action_dim]


def compute_port_approach_oracle(
    env: gym.Env,
    action_scale: torch.Tensor,
    *,
    pos_gain: float = 0.8,
    rot_gain: float = 0.6,
    max_pos_delta: float = 0.025,
    max_rot_delta: float = 0.20,
) -> PortApproachOracleOutput:
    """Compute the privileged target-pose oracle action for the port approach task."""
    robot = env.scene["robot"]
    target = env.scene[TARGET_NAME]
    num_envs = env.num_envs

    tcp_body_id = robot.find_bodies("gripper_tcp", preserve_order=True)[0][0]
    tcp_pos_w = robot.data.body_pos_w[:, tcp_body_id, :]
    tcp_quat_w = robot.data.body_quat_w[:, tcp_body_id, :]

    target_offset = _constant_vec(NIC_PORT_APPROACH_OFFSET, target.data.root_pos_w)
    target_tip_pos_w = target.data.root_pos_w + math_utils.quat_apply(
        target.data.root_quat_w,
        target_offset.expand(num_envs, -1),
    )
    target_tip_quat_w = math_utils.quat_mul(
        target.data.root_quat_w,
        _constant_rpy_quat(NIC_PORT_APPROACH_RPY, num_envs, target.data.root_quat_w),
    )

    tcp_tip_quat = _constant_rpy_quat(CABLE_TIP_RPY_FROM_TCP, num_envs, tcp_quat_w)
    desired_tcp_quat_w = math_utils.quat_mul(target_tip_quat_w, math_utils.quat_inv(tcp_tip_quat))
    tcp_tip_offset = _constant_vec(CABLE_TIP_OFFSET_FROM_TCP, tcp_pos_w)
    desired_tcp_pos_w = target_tip_pos_w - math_utils.quat_apply(
        desired_tcp_quat_w,
        tcp_tip_offset.expand(num_envs, -1),
    )

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

    processed_action = torch.zeros((num_envs, action_scale.shape[1]), dtype=tcp_pos_w.dtype, device=env.device)
    processed_action[:, 0:3] = _clamp_vector_norm(pos_error_b * pos_gain, max_pos_delta)
    processed_action[:, 3:6] = _clamp_vector_norm(rot_error_b * rot_gain, max_rot_delta)
    raw_action = processed_action / torch.clamp(action_scale, min=1.0e-9)

    current_tip_pos_w = tcp_pos_w + math_utils.quat_apply(tcp_quat_w, tcp_tip_offset.expand(num_envs, -1))
    current_tip_quat_w = math_utils.quat_mul(tcp_quat_w, tcp_tip_quat)
    position_error = torch.linalg.norm(target_tip_pos_w - current_tip_pos_w, dim=1)
    orientation_error = math_utils.quat_error_magnitude(current_tip_quat_w, target_tip_quat_w)

    return PortApproachOracleOutput(
        raw_action=raw_action,
        processed_action=processed_action,
        position_error=position_error,
        orientation_error=orientation_error,
        desired_tcp_pos_w=desired_tcp_pos_w,
        desired_tcp_quat_w=desired_tcp_quat_w,
        target_tip_pos_w=target_tip_pos_w,
        target_tip_quat_w=target_tip_quat_w,
        current_tip_pos_w=current_tip_pos_w,
        current_tip_quat_w=current_tip_quat_w,
    )


def choose_preapproach_phase(
    labels: dict,
    layout: PortKeypointLayout,
    *,
    env_index: int,
    position_error: float,
    orientation_error: float,
    center_camera: str = "center_camera",
    align_position_threshold: float = 0.04,
    position_hold_threshold: float = 0.025,
    orientation_hold_threshold: float = math.radians(15.0),
) -> TeacherPhase:
    """Choose a visual-compatible teacher phase from visibility and oracle errors."""
    cameras = labels["cameras"]
    entry_idx = layout.index("entry_center")
    approach_idx = layout.index("approach_center")
    mouth_indices = [layout.index(name) for name in layout.names if name.startswith("mouth_")]
    camera_masks = {
        camera_name: camera_labels["in_frame"][env_index]
        for camera_name, camera_labels in cameras.items()
    }

    any_projected = any(bool(mask.any()) for mask in camera_masks.values())
    if not any_projected:
        return TeacherPhase.SEARCH

    center_mask = camera_masks.get(center_camera, next(iter(camera_masks.values())))
    any_entry_or_approach = any(bool(mask[entry_idx] or mask[approach_idx]) for mask in camera_masks.values())
    if not bool(center_mask[entry_idx] or center_mask[approach_idx]) and not any_entry_or_approach:
        return TeacherPhase.CENTER

    if position_error <= position_hold_threshold and orientation_error <= orientation_hold_threshold:
        return TeacherPhase.HOLD

    center_mouth_points = int(center_mask[mouth_indices].sum().item())
    any_mouth_points = max(int(mask[mouth_indices].sum().item()) for mask in camera_masks.values())
    if center_mouth_points >= 2 or (position_error <= align_position_threshold and any_mouth_points >= 2):
        return TeacherPhase.ALIGN

    return TeacherPhase.COARSE_APPROACH


def apply_phase_gate(
    raw_action: torch.Tensor,
    phase: TeacherPhase,
    *,
    env_index: int = 0,
    hold_scale: float = 0.25,
) -> torch.Tensor:
    """Gate oracle actions so early phases do not use hidden orientation decisions."""
    gated_action = raw_action.clone()
    if phase in (TeacherPhase.SEARCH, TeacherPhase.CENTER, TeacherPhase.COARSE_APPROACH):
        gated_action[env_index, 3:6] = 0.0
    elif phase == TeacherPhase.HOLD:
        gated_action[env_index] *= hold_scale
    return gated_action


def _constant_vec(values: tuple[float, float, float], like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(values, dtype=like.dtype, device=like.device).unsqueeze(0)


def _constant_rpy_quat(rpy: tuple[float, float, float], num_envs: int, like: torch.Tensor) -> torch.Tensor:
    rpy_tensor = torch.tensor(rpy, dtype=like.dtype, device=like.device)
    quat = math_utils.quat_from_euler_xyz(rpy_tensor[0], rpy_tensor[1], rpy_tensor[2])
    return quat.unsqueeze(0).expand(num_envs, -1)


def _clamp_vector_norm(vector: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.linalg.norm(vector, dim=1, keepdim=True)
    scale = torch.clamp(max_norm / torch.clamp(norm, min=1.0e-9), max=1.0)
    return vector * scale
