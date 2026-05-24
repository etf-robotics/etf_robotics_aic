# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable event functions for AIC manager-based tasks."""

from __future__ import annotations

import math
import random
import re
from typing import TYPE_CHECKING

import omni.usd
import torch
from pxr import Gf, Sdf, UsdGeom, UsdLux

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_ENV_REGEX_RE = re.compile(r"env_(?:\.\*|\[\^/\]\*)")
_cached_orientations: dict[str, torch.Tensor] = {}


def randomize_dome_light(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float] = (1500.0, 3500.0),
    color_range: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.5, 0.5, 0.5),
        (1.0, 1.0, 1.0),
    ),
) -> None:
    """Randomize the shared dome light intensity and color on reset."""
    stage = omni.usd.get_context().get_stage()
    light_prim = stage.GetPrimAtPath("/World/light")
    if not light_prim.IsValid():
        return

    light = UsdLux.DomeLight(light_prim)
    intensity = torch.empty(1).uniform_(intensity_range[0], intensity_range[1]).item()
    light.GetIntensityAttr().Set(intensity)

    color_min, color_max = color_range
    color = tuple(torch.empty(1).uniform_(color_min[i], color_max[i]).item() for i in range(3))
    light.GetColorAttr().Set(Gf.Vec3f(*color))


def _sample_axis(pose_range: dict, snap_step: dict, axis: str) -> float:
    """Sample an axis offset, snapping to a grid step when configured."""
    lo, hi = pose_range.get(axis, (0.0, 0.0))
    step = snap_step.get(axis, 0.0)
    if step > 0 and (hi - lo) > 0:
        n_lo = math.ceil(lo / step)
        n_hi = math.floor(hi / step)
        return random.randint(n_lo, n_hi) * step
    return torch.empty(1).uniform_(lo, hi).item()


def _yaw_quat(angles: torch.Tensor) -> torch.Tensor:
    """Quaternion (wxyz) representing a rotation about world-Z by ``angles``."""
    half = angles * 0.5
    w = torch.cos(half)
    z = torch.sin(half)
    zeros = torch.zeros_like(angles)
    return torch.stack([w, zeros, zeros, z], dim=-1)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two (..., 4) wxyz quaternions."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def _write_usd_xform_pose(
    stage,
    prim_path_template: str,
    env_ids: torch.Tensor,
    env_origins: torch.Tensor,
    world_pos: torch.Tensor,
    world_rot: torch.Tensor,
) -> None:
    """Mirror a per-env rigid body pose onto its USD Xform."""
    local_pos = (world_pos - env_origins).tolist()
    rotations = world_rot.tolist()

    for i, env_id in enumerate(env_ids.tolist()):
        prim_path = _ENV_REGEX_RE.sub(f"env_{env_id}", prim_path_template)
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue

        xform = UsdGeom.Xformable(prim)
        tx, ty, tz = local_pos[i]
        qw, qx, qy, qz = rotations[i]

        for op in xform.GetOrderedXformOps():
            name = op.GetOpName()
            if "translate" in name:
                value = Gf.Vec3f(tx, ty, tz) if op.GetTypeName() == Sdf.ValueTypeNames.Float3 else Gf.Vec3d(tx, ty, tz)
                op.Set(value)
            elif "orient" in name:
                value = Gf.Quatf(qw, qx, qy, qz) if op.GetTypeName() == Sdf.ValueTypeNames.Quatf else Gf.Quatd(qw, qx, qy, qz)
                op.Set(value)


def randomize_board_and_parts(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    board_scene_name: str = "task_board",
    board_default_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    board_range: dict = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
    parts: list[dict] = (),
    sync_usd_xforms: bool = True,
) -> None:
    """Randomize a task board and rigid parts that are anchored to it.

    The board can be jittered in x/y and rotated about world-Z by ``board_range["yaw"]``.
    Parts are kept rigidly attached to the board: their per-part ``offset`` is treated
    as a board-local vector and rotated into world by the sampled board yaw.
    Per-part ``pose_range["yaw"]`` adds independent yaw jitter on top (default 0).
    """
    device = env.device
    num_resets = len(env_ids)
    env_origins = env.scene.env_origins[env_ids]
    stage = omni.usd.get_context().get_stage() if sync_usd_xforms else None

    all_names = [board_scene_name] + [part["scene_name"] for part in parts]
    for name in all_names:
        if name not in _cached_orientations:
            _cached_orientations[name] = env.scene[name].data.root_state_w[:, 3:7].clone()

    # --- Board pose: x/y jitter + world-Z yaw delta ---
    board_asset = env.scene[board_scene_name]
    board_pos = torch.tensor([board_default_pos], device=device).expand(num_resets, -1).clone()
    board_pos[:, 0] += torch.empty(num_resets, device=device).uniform_(*board_range.get("x", (0.0, 0.0)))
    board_pos[:, 1] += torch.empty(num_resets, device=device).uniform_(*board_range.get("y", (0.0, 0.0)))
    board_world_pos = board_pos + env_origins

    yaw_lo, yaw_hi = board_range.get("yaw", (0.0, 0.0))
    board_yaw_delta = torch.empty(num_resets, device=device).uniform_(yaw_lo, yaw_hi)
    cached_board_rot = _cached_orientations[board_scene_name][env_ids].to(device)
    board_rot = _quat_mul(_yaw_quat(board_yaw_delta), cached_board_rot)

    board_asset.write_root_pose_to_sim(torch.cat([board_world_pos, board_rot], dim=-1), env_ids=env_ids)
    board_asset.write_root_velocity_to_sim(torch.zeros(num_resets, 6, device=device), env_ids=env_ids)
    if sync_usd_xforms:
        _write_usd_xform_pose(stage, board_asset.cfg.prim_path, env_ids, env_origins, board_world_pos, board_rot)

    # --- Parts: rigidly carried by the (now-rotated) board, plus optional per-part yaw jitter ---
    cos_y = torch.cos(board_yaw_delta)
    sin_y = torch.sin(board_yaw_delta)

    for part_cfg in parts:
        part_name = part_cfg["scene_name"]
        part_asset = env.scene[part_name]
        cached_part_rot = _cached_orientations[part_name][env_ids].to(device)

        offset_x, offset_y, offset_z = part_cfg["offset"]
        pose_range = part_cfg.get("pose_range", {})
        snap_step = part_cfg.get("snap_step", {})

        local_x = torch.empty(num_resets, device=device)
        local_y = torch.empty(num_resets, device=device)
        part_yaw_jitter = torch.empty(num_resets, device=device)
        for idx in range(num_resets):
            local_x[idx] = offset_x + _sample_axis(pose_range, snap_step, "x")
            local_y[idx] = offset_y + _sample_axis(pose_range, snap_step, "y")
            part_yaw_jitter[idx] = _sample_axis(pose_range, snap_step, "yaw")

        rotated_x = cos_y * local_x - sin_y * local_y
        rotated_y = sin_y * local_x + cos_y * local_y

        part_pos = torch.empty(num_resets, 3, device=device)
        part_pos[:, 0] = board_world_pos[:, 0] + rotated_x
        part_pos[:, 1] = board_world_pos[:, 1] + rotated_y
        part_pos[:, 2] = board_world_pos[:, 2] + offset_z

        part_rot = _quat_mul(_yaw_quat(board_yaw_delta + part_yaw_jitter), cached_part_rot)

        part_asset.write_root_pose_to_sim(torch.cat([part_pos, part_rot], dim=-1), env_ids=env_ids)
        part_asset.write_root_velocity_to_sim(torch.zeros(num_resets, 6, device=device), env_ids=env_ids)
        if sync_usd_xforms:
            _write_usd_xform_pose(stage, part_asset.cfg.prim_path, env_ids, env_origins, part_pos, part_rot)
