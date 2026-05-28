# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms for the port-insertion task."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass


class InsertionGoalCommand(CommandTerm):
    """World-frame EEF poses for the selected port entrance and seat."""

    cfg: "InsertionGoalCommandCfg"

    def __init__(self, cfg: "InsertionGoalCommandCfg", env):
        super().__init__(cfg, env)

        self.target = env.scene[cfg.target_scene_name]
        dtype = self.target.data.root_pos_w.dtype
        device = self.target.data.root_pos_w.device

        self.entrance_pos_root = torch.zeros((self.num_envs, 3), dtype=dtype, device=device)
        self.entrance_quat_root = _identity_quats(self.num_envs, dtype=dtype, device=device)
        self.seat_pos_root = torch.zeros((self.num_envs, 3), dtype=dtype, device=device)
        self.seat_quat_root = _identity_quats(self.num_envs, dtype=dtype, device=device)

        self.entrance_pos_w = torch.zeros_like(self.entrance_pos_root)
        self.entrance_quat_w = _identity_quats(self.num_envs, dtype=dtype, device=device)
        self.seat_pos_w = torch.zeros_like(self.seat_pos_root)
        self.seat_quat_w = _identity_quats(self.num_envs, dtype=dtype, device=device)

        # Tensor schema: entrance pose (pos, quat), seat pose (pos, quat).
        self._command = torch.zeros((self.num_envs, 14), dtype=dtype, device=device)

    def __str__(self) -> str:
        return (
            "InsertionGoalCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tTarget scene name: {self.cfg.target_scene_name}\n"
            f"\tPort: {self.cfg.port_name}\n"
            f"\tPort entrance frame: {self.cfg.port_entrance_frame_path}\n"
            f"\tPort seat frame: {self.cfg.port_seat_frame_path}\n"
            f"\tEEF pose in port frame: pos={self.cfg.eef_pos_in_port_frame}, "
            f"rot={self.cfg.eef_quat_in_port_frame}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}"
        )

    @property
    def command(self) -> torch.Tensor:
        """Compact world-frame goal tensor for planner observations."""

        return self._command

    @property
    def final_tip_pos_w(self) -> torch.Tensor:
        """Compatibility alias until termination terms are renamed."""

        return self.seat_pos_w

    @property
    def target_tip_quat_w(self) -> torch.Tensor:
        """Compatibility alias until termination terms are renamed."""

        return self.seat_quat_w

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        env_id_list = _as_env_id_list(env_ids, self.num_envs)
        if not env_id_list:
            return
        if self.cfg.port_entrance_frame_path is None:
            raise ValueError("InsertionGoalCommand requires a port entrance frame path.")

        eef_pos_port = torch.tensor(self.cfg.eef_pos_in_port_frame, dtype=torch.float64)
        eef_quat_port = torch.tensor(self.cfg.eef_quat_in_port_frame, dtype=torch.float64).unsqueeze(0)
        entrance_positions = []
        entrance_quats = []
        seat_positions = []
        seat_quats = []

        for env_index in env_id_list:
            root_path = _resolve_asset_root_prim_path(
                self.target,
                env_index,
            )
            entrance_pos_root, entrance_quat_root = _prim_pose_in_asset_root(
                root_path,
                self.cfg.port_entrance_frame_path,
                usd_root_child=self.cfg.target_root_prim,
            )
            seat_pos_root, seat_quat_root = _prim_pose_in_asset_root(
                root_path,
                self.cfg.port_seat_frame_path,
                usd_root_child=self.cfg.target_root_prim,
            )

            entrance_pos, entrance_quat = _compose_pose(
                torch.tensor(entrance_pos_root, dtype=torch.float64),
                torch.tensor(entrance_quat_root, dtype=torch.float64).unsqueeze(0),
                eef_pos_port,
                eef_quat_port,
            )
            seat_pos, seat_quat = _compose_pose(
                torch.tensor(seat_pos_root, dtype=torch.float64),
                torch.tensor(seat_quat_root, dtype=torch.float64).unsqueeze(0),
                eef_pos_port,
                eef_quat_port,
            )

            entrance_positions.append(entrance_pos.tolist())
            entrance_quats.append(tuple(float(value) for value in entrance_quat.tolist()))
            seat_positions.append(seat_pos.tolist())
            seat_quats.append(tuple(float(value) for value in seat_quat.tolist()))

        env_ids_t = torch.tensor(env_id_list, dtype=torch.long, device=self.seat_pos_root.device)
        self.entrance_pos_root[env_ids_t] = torch.tensor(
            entrance_positions, dtype=self.entrance_pos_root.dtype, device=self.entrance_pos_root.device
        )
        self.entrance_quat_root[env_ids_t] = torch.tensor(
            entrance_quats, dtype=self.entrance_quat_root.dtype, device=self.entrance_quat_root.device
        )
        self.seat_pos_root[env_ids_t] = torch.tensor(
            seat_positions, dtype=self.seat_pos_root.dtype, device=self.seat_pos_root.device
        )
        self.seat_quat_root[env_ids_t] = torch.tensor(
            seat_quats, dtype=self.seat_quat_root.dtype, device=self.seat_quat_root.device
        )
        self._update_command()

    def _update_command(self) -> None:
        target_pos_w = self.target.data.root_pos_w
        target_quat_w = self.target.data.root_quat_w

        self.entrance_pos_w[:], self.entrance_quat_w[:] = _compose_pose(
            target_pos_w,
            target_quat_w,
            self.entrance_pos_root,
            self.entrance_quat_root,
        )
        self.seat_pos_w[:], self.seat_quat_w[:] = _compose_pose(
            target_pos_w,
            target_quat_w,
            self.seat_pos_root,
            self.seat_quat_root,
        )

        self._command[:, 0:3] = self.entrance_pos_w
        self._command[:, 3:7] = self.entrance_quat_w
        self._command[:, 7:10] = self.seat_pos_w
        self._command[:, 10:14] = self.seat_quat_w


@configclass
class InsertionGoalCommandCfg(CommandTermCfg):
    """Configuration for the selected target/port insertion goal."""

    class_type: type[CommandTerm] = InsertionGoalCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9 + 1.0)
    debug_vis: bool = False

    target_scene_name: str = "target"
    target_root_prim: str | None = "nic_card_link"
    port_name: str = "sfp_port_0"
    port_seat_frame_path: str = "/sfp_port_0_link"
    port_entrance_frame_path: str | None = "/sfp_port_0_link/sfp_port_0_link_entrance"
    eef_pos_in_port_frame: tuple[float, float, float] = (0.0, 0.0, 0.0)
    eef_quat_in_port_frame: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


def _identity_quats(num_envs: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    quats = torch.zeros((num_envs, 4), dtype=dtype, device=device)
    quats[:, 0] = 1.0
    return quats


def _compose_pose(
    parent_pos: torch.Tensor,
    parent_quat: torch.Tensor,
    child_pos: torch.Tensor,
    child_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ``T_parent_child`` onto a parent pose in Isaac's wxyz convention."""

    parent_was_unbatched = parent_pos.ndim == 1
    if parent_pos.ndim == 1:
        parent_pos = parent_pos.unsqueeze(0)
    if child_pos.ndim == 1:
        child_pos = child_pos.unsqueeze(0)
    if parent_quat.ndim == 1:
        parent_quat = parent_quat.unsqueeze(0)
    if child_quat.ndim == 1:
        child_quat = child_quat.unsqueeze(0)

    pos = parent_pos + math_utils.quat_apply(parent_quat, child_pos.expand(parent_pos.shape[0], -1))
    quat = math_utils.quat_mul(parent_quat, child_quat.expand(parent_quat.shape[0], -1))
    if parent_was_unbatched:
        return pos[0], quat[0]
    return pos, quat


def _as_env_id_list(env_ids: Sequence[int], num_envs: int) -> list[int]:
    if isinstance(env_ids, slice):
        return list(range(num_envs))[env_ids]
    if torch.is_tensor(env_ids):
        return [int(env_id) for env_id in env_ids.detach().cpu().tolist()]
    return [int(env_id) for env_id in env_ids]


def _prim_pose_in_asset_root(
    asset_root_path: str,
    prim_path: str,
    *,
    usd_root_child: str | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Return a prim pose relative to an asset root."""

    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(asset_root_path)
    _, prim = _resolve_prim(stage, asset_root_path, prim_path, usd_root_child=usd_root_child)
    if not root_prim.IsValid() or not prim.IsValid():
        candidates = ", ".join(_candidate_prim_paths(asset_root_path, prim_path, usd_root_child=usd_root_child))
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


def _resolve_asset_root_prim_path(asset, env_index: int) -> str:
    """Return the USD prim whose pose is exposed by the Isaac Lab asset data."""

    prim_paths = list(getattr(asset.root_physx_view, "prim_paths", []))
    if not prim_paths:
        cfg_path = getattr(getattr(asset, "cfg", None), "prim_path", "")
        raise RuntimeError(f"Cannot resolve prim paths for asset with cfg path '{cfg_path}'.")

    index = min(env_index, len(prim_paths) - 1)
    return str(prim_paths[index])


def _join_prim_path(asset_root_path: str, relative_path: str) -> str:
    return f"{asset_root_path.rstrip('/')}/{relative_path.lstrip('/')}"


def _resolve_prim(stage, asset_root_path: str, relative_path: str, *, usd_root_child: str | None):
    """Resolve a prim by exact candidate paths, then by descendant basename."""

    for prim_path in _candidate_prim_paths(asset_root_path, relative_path, usd_root_child=usd_root_child):
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            return prim_path, prim

    basename = relative_path.rstrip("/").rsplit("/", 1)[-1]
    root_prefixes = _asset_root_prefixes(asset_root_path, usd_root_child=usd_root_child)
    for prim in stage.Traverse():
        prim_path = prim.GetPath().pathString
        if not any(prim_path == prefix or prim_path.startswith(prefix + "/") for prefix in root_prefixes):
            continue
        if prim_path.rsplit("/", 1)[-1] == basename:
            return prim_path, prim
    return (
        _candidate_prim_paths(asset_root_path, relative_path, usd_root_child=usd_root_child)[0],
        stage.GetPrimAtPath("/__missing__"),
    )


def _candidate_prim_paths(
    asset_root_path: str,
    relative_path: str,
    *,
    usd_root_child: str | None,
) -> list[str]:
    """Return path candidates for assets spawned with or without defaultPrim nesting."""

    relative = relative_path.lstrip("/")
    candidates = [_join_prim_path(asset_root_path, relative)]
    if usd_root_child is not None:
        root_prefix = f"{usd_root_child}/"
        if not relative.startswith(root_prefix):
            candidates.append(_join_prim_path(asset_root_path, f"{usd_root_child}/{relative}"))
        if relative.startswith(root_prefix):
            candidates.append(_join_prim_path(asset_root_path, relative.removeprefix(root_prefix)))
        root_suffix = f"/{usd_root_child}"
        if asset_root_path.endswith(root_suffix):
            parent_root = asset_root_path.removesuffix(root_suffix)
            candidates.append(_join_prim_path(parent_root, relative))
            if relative.startswith(root_prefix):
                candidates.append(_join_prim_path(parent_root, relative.removeprefix(root_prefix)))
    return list(dict.fromkeys(candidates))


def _asset_root_prefixes(asset_root_path: str, *, usd_root_child: str | None) -> tuple[str, ...]:
    prefixes = [asset_root_path.rstrip("/")]
    if usd_root_child is not None:
        suffix = f"/{usd_root_child}"
        if prefixes[0].endswith(suffix):
            prefixes.append(prefixes[0].removesuffix(suffix))
        else:
            prefixes.append(prefixes[0] + suffix)
    return tuple(dict.fromkeys(prefixes))


__all__ = [
    "InsertionGoalCommand",
    "InsertionGoalCommandCfg",
]
