# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms for the port-insertion task."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass


class InsertionGoalCommand(CommandTerm):
    """Fixed SFP insertion goal for one NIC port.

    The command is expressed as a desired pose for ``sfp_tip_link``.  The
    controller can then convert that desired tip pose into a desired
    ``gripper_tcp`` pose using the live TCP-in-tip transform.
    """

    cfg: "InsertionGoalCommandCfg"

    def __init__(self, cfg: "InsertionGoalCommandCfg", env):
        super().__init__(cfg, env)

        self.target = env.scene[cfg.target_name]
        dtype = self.target.data.root_pos_w.dtype
        device = self.target.data.root_pos_w.device

        self.final_tip_pos_root = torch.zeros((self.num_envs, 3), dtype=dtype, device=device)
        self.approach_tip_pos_root = torch.zeros((self.num_envs, 3), dtype=dtype, device=device)
        self.port_quat_root = torch.zeros((self.num_envs, 4), dtype=dtype, device=device)
        self.port_quat_root[:, 0] = 1.0
        self.port_x_root = torch.zeros((self.num_envs, 3), dtype=dtype, device=device)
        self.port_y_root = torch.zeros((self.num_envs, 3), dtype=dtype, device=device)
        self.port_z_root = torch.zeros((self.num_envs, 3), dtype=dtype, device=device)

        self.final_tip_pos_w = torch.zeros_like(self.final_tip_pos_root)
        self.approach_tip_pos_w = torch.zeros_like(self.approach_tip_pos_root)
        self.target_tip_quat_w = torch.zeros_like(self.port_quat_root)
        self.target_tip_quat_w[:, 0] = 1.0
        self.target_x_w = torch.zeros_like(self.port_x_root)
        self.target_y_w = torch.zeros_like(self.port_y_root)
        self.target_z_w = torch.zeros_like(self.port_z_root)
        self.path_w = torch.zeros_like(self.port_x_root)
        self.path_axis_w = torch.zeros_like(self.port_x_root)
        self.path_length = torch.ones((self.num_envs, 1), dtype=dtype, device=device)
        self.port_x_w = torch.zeros_like(self.port_x_root)
        self.port_y_w = torch.zeros_like(self.port_y_root)
        self.port_z_w = torch.zeros_like(self.port_z_root)

        # Tensor schema:
        # final_tip_pos_w(3), target_tip_quat_w(4), approach_tip_pos_w(3),
        # path_axis_w(3), path_length(1), port_x/y/z_w(9), port_index(1).
        self._command = torch.zeros((self.num_envs, 24), dtype=dtype, device=device)

    def __str__(self) -> str:
        return (
            "InsertionGoalCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tPort: {self.cfg.port_name}\n"
            f"\tTarget X/Z offset: {self.cfg.target_xz_offset}\n"
            f"\tApproach offset: {self.cfg.approach_offset_local}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}"
        )

    @property
    def command(self) -> torch.Tensor:
        """Compact world-frame goal tensor for optional observation/debug use."""
        return self._command

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        env_id_list = _as_env_id_list(env_ids, self.num_envs)
        if not env_id_list:
            return

        final_positions = []
        approach_positions = []
        port_quats = []
        port_x_axes = []
        port_y_axes = []
        port_z_axes = []
        port_path = _port_link_path(self.cfg.port_name)

        for env_index in env_id_list:
            root_path = _resolve_asset_root_prim_path(self.target, env_index)
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

            final_pos = (
                port_pos
                + float(self.cfg.target_xz_offset[0]) * port_x
                + float(self.cfg.target_xz_offset[1]) * port_z
            )
            approach_pos = (
                final_pos
                + float(self.cfg.approach_offset_local[0]) * port_x
                + float(self.cfg.approach_offset_local[1]) * port_y
                + float(self.cfg.approach_offset_local[2]) * port_z
            )

            final_positions.append(final_pos.tolist())
            approach_positions.append(approach_pos.tolist())
            port_quats.append(tuple(float(value) for value in port_quat_root))
            port_x_axes.append(port_x.tolist())
            port_y_axes.append(port_y.tolist())
            port_z_axes.append(port_z.tolist())

        env_ids_t = torch.tensor(env_id_list, dtype=torch.long, device=self.final_tip_pos_root.device)
        self.final_tip_pos_root[env_ids_t] = torch.tensor(
            final_positions, dtype=self.final_tip_pos_root.dtype, device=self.final_tip_pos_root.device
        )
        self.approach_tip_pos_root[env_ids_t] = torch.tensor(
            approach_positions, dtype=self.approach_tip_pos_root.dtype, device=self.approach_tip_pos_root.device
        )
        self.port_quat_root[env_ids_t] = torch.tensor(
            port_quats, dtype=self.port_quat_root.dtype, device=self.port_quat_root.device
        )
        self.port_x_root[env_ids_t] = torch.tensor(
            port_x_axes, dtype=self.port_x_root.dtype, device=self.port_x_root.device
        )
        self.port_y_root[env_ids_t] = torch.tensor(
            port_y_axes, dtype=self.port_y_root.dtype, device=self.port_y_root.device
        )
        self.port_z_root[env_ids_t] = torch.tensor(
            port_z_axes, dtype=self.port_z_root.dtype, device=self.port_z_root.device
        )
        self._update_command()

    def _update_command(self) -> None:
        card_pos = self.target.data.root_pos_w
        card_quat = self.target.data.root_quat_w

        self.final_tip_pos_w[:] = _root_point_to_world(card_pos, card_quat, self.final_tip_pos_root)
        self.approach_tip_pos_w[:] = _root_point_to_world(card_pos, card_quat, self.approach_tip_pos_root)
        port_quat_w = math_utils.quat_mul(card_quat, self.port_quat_root)
        self.target_tip_quat_w[:] = math_utils.quat_mul(port_quat_w, _target_orientation_offset(port_quat_w))

        self.port_x_w[:] = _normalize_rows(math_utils.quat_apply(card_quat, self.port_x_root))
        self.port_y_w[:] = _normalize_rows(math_utils.quat_apply(card_quat, self.port_y_root))
        self.port_z_w[:] = _normalize_rows(math_utils.quat_apply(card_quat, self.port_z_root))
        self.target_x_w[:] = _local_axis_w(self.target_tip_quat_w, (1.0, 0.0, 0.0))
        self.target_y_w[:] = _local_axis_w(self.target_tip_quat_w, (0.0, 1.0, 0.0))
        self.target_z_w[:] = _local_axis_w(self.target_tip_quat_w, (0.0, 0.0, 1.0))

        self.path_w[:] = self.final_tip_pos_w - self.approach_tip_pos_w
        self.path_length[:] = torch.linalg.norm(self.path_w, dim=1, keepdim=True).clamp_min(1.0e-9)
        self.path_axis_w[:] = self.path_w / self.path_length

        self._command[:, 0:3] = self.final_tip_pos_w
        self._command[:, 3:7] = self.target_tip_quat_w
        self._command[:, 7:10] = self.approach_tip_pos_w
        self._command[:, 10:13] = self.path_axis_w
        self._command[:, 13:14] = self.path_length
        self._command[:, 14:17] = self.port_x_w
        self._command[:, 17:20] = self.port_y_w
        self._command[:, 20:23] = self.port_z_w
        self._command[:, 23] = float(self.cfg.port_index)


@configclass
class InsertionGoalCommandCfg(CommandTermCfg):
    """Configuration for the fixed port-insertion goal."""

    class_type: type[CommandTerm] = InsertionGoalCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9 + 1.0)
    debug_vis: bool = False
    target_name: str = "nic_card"
    port_name: str = "sfp_port_0"
    port_index: int = 0
    target_xz_offset: tuple[float, float] = (0.0, 0.001)
    approach_offset_local: tuple[float, float, float] = (0.0, -0.09, 0.0)


def _as_env_id_list(env_ids: Sequence[int], num_envs: int) -> list[int]:
    if isinstance(env_ids, slice):
        return list(range(num_envs))[env_ids]
    if torch.is_tensor(env_ids):
        return [int(env_id) for env_id in env_ids.detach().cpu().tolist()]
    return [int(env_id) for env_id in env_ids]


def _port_link_path(port_name: str) -> str:
    port_name = port_name.strip("/")
    if port_name.endswith("_link"):
        return f"/{port_name}"
    return f"/{port_name}_link"


def _prim_pose_in_asset_root(
    asset_root_path: str,
    prim_path: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Return a prim pose relative to an asset root."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(asset_root_path)
    _, prim = _resolve_prim(stage, asset_root_path, prim_path)
    if not root_prim.IsValid() or not prim.IsValid():
        candidates = ", ".join(_candidate_prim_paths(asset_root_path, prim_path))
        raise KeyError(f"USD prim '{prim_path}' was not found. Tried: {candidates}.")

    cache = UsdGeom.XformCache()
    root_matrix = cache.GetLocalToWorldTransform(root_prim)
    prim_matrix = cache.GetLocalToWorldTransform(prim)
    root_inv = root_matrix.GetInverse()
    prim_root = root_inv.Transform(prim_matrix.ExtractTranslation())

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


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.clamp(torch.linalg.norm(vector), min=1.0e-9)


def _normalize_rows(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.clamp(torch.linalg.norm(vector, dim=1, keepdim=True), min=1.0e-9)


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
