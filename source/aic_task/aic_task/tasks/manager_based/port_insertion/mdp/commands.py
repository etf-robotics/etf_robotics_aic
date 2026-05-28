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

        self.target = env.scene[cfg.target_scene_name]
        dtype = self.target.data.root_pos_w.dtype
        device = self.target.data.root_pos_w.device

        self.final_tip_pos_root = torch.zeros((self.num_envs, 3), dtype=dtype, device=device)
        self.nominal_approach_tip_pos_root = torch.zeros((self.num_envs, 3), dtype=dtype, device=device)
        self.approach_tip_pos_root = torch.zeros((self.num_envs, 3), dtype=dtype, device=device)
        self.port_quat_root = torch.zeros((self.num_envs, 4), dtype=dtype, device=device)
        self.port_quat_root[:, 0] = 1.0
        self.approach_tip_quat_root = torch.zeros((self.num_envs, 4), dtype=dtype, device=device)
        self.approach_tip_quat_root[:, 0] = 1.0
        self.port_x_root = torch.zeros((self.num_envs, 3), dtype=dtype, device=device)
        self.port_y_root = torch.zeros((self.num_envs, 3), dtype=dtype, device=device)
        self.port_z_root = torch.zeros((self.num_envs, 3), dtype=dtype, device=device)

        self.final_tip_pos_w = torch.zeros_like(self.final_tip_pos_root)
        self.nominal_approach_tip_pos_w = torch.zeros_like(self.nominal_approach_tip_pos_root)
        self.approach_tip_pos_w = torch.zeros_like(self.approach_tip_pos_root)
        self.target_tip_quat_w = torch.zeros_like(self.port_quat_root)
        self.target_tip_quat_w[:, 0] = 1.0
        self.approach_tip_quat_w = torch.zeros_like(self.approach_tip_quat_root)
        self.approach_tip_quat_w[:, 0] = 1.0
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
        # final_tip_pos_w(3), target_tip_quat_w(4), randomized approach
        # pos/quat(7), nominal approach pos(3), path_axis_w(3),
        # path_length(1), port_x/y/z_w(9), port_index(1).
        self._command = torch.zeros((self.num_envs, 31), dtype=dtype, device=device)

    def __str__(self) -> str:
        return (
            "InsertionGoalCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tTarget scene name: {self.cfg.target_scene_name}\n"
            f"\tPort: {self.cfg.port_name}\n"
            f"\tPort seat frame: {self.cfg.port_seat_frame_path}\n"
            f"\tTarget X/Z offset: {self.cfg.target_xz_offset}\n"
            f"\tApproach offset: {self.cfg.approach_offset_local}\n"
            f"\tApproach position noise: {self.cfg.approach_pos_noise_local}\n"
            f"\tApproach tilt/twist noise deg: "
            f"{self.cfg.approach_tilt_noise_deg}/{self.cfg.approach_twist_noise_deg}\n"
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
        nominal_approach_positions = []
        approach_positions = []
        port_quats = []
        approach_quats = []
        port_x_axes = []
        port_y_axes = []
        port_z_axes = []

        for env_index in env_id_list:
            root_path = _resolve_asset_root_prim_path(
                self.target,
                env_index,
                usd_root_child=self.cfg.target_root_prim,
            )
            port_pos_root, port_quat_root = _prim_pose_in_asset_root(
                root_path,
                self.cfg.port_seat_frame_path,
                usd_root_child=self.cfg.target_root_prim,
            )

            port_pos = torch.tensor(port_pos_root, dtype=torch.float64)
            port_quat = torch.tensor(port_quat_root, dtype=torch.float64).unsqueeze(0)
            port_x = _axis_from_quat(port_quat, (1.0, 0.0, 0.0))
            port_y = _axis_from_quat(port_quat, self.cfg.insertion_axis_local)
            port_z = _axis_from_quat(port_quat, (0.0, 0.0, 1.0))

            final_pos = (
                port_pos
                + float(self.cfg.target_xz_offset[0]) * port_x
                + float(self.cfg.target_xz_offset[1]) * port_z
            )
            nominal_approach_pos = (
                final_pos
                + float(self.cfg.approach_offset_local[0]) * port_x
                + float(self.cfg.approach_offset_local[1]) * port_y
                + float(self.cfg.approach_offset_local[2]) * port_z
            )
            approach_noise = _sample_local_position_noise(
                self.cfg.approach_pos_noise_local,
                dtype=torch.float64,
            )
            approach_pos = (
                nominal_approach_pos
                + approach_noise[0] * port_x
                + approach_noise[1] * port_y
                + approach_noise[2] * port_z
            )

            target_tip_quat_root = math_utils.quat_mul(port_quat, _target_orientation_offset(port_quat))[0]
            approach_quat_root = math_utils.quat_mul(
                target_tip_quat_root.unsqueeze(0),
                _sample_approach_orientation_noise(
                    self.cfg.approach_tilt_noise_deg,
                    self.cfg.approach_twist_noise_deg,
                    dtype=torch.float64,
                ),
            )[0]

            final_positions.append(final_pos.tolist())
            nominal_approach_positions.append(nominal_approach_pos.tolist())
            approach_positions.append(approach_pos.tolist())
            port_quats.append(tuple(float(value) for value in port_quat_root))
            approach_quats.append(tuple(float(value) for value in approach_quat_root.tolist()))
            port_x_axes.append(port_x.tolist())
            port_y_axes.append(port_y.tolist())
            port_z_axes.append(port_z.tolist())

        env_ids_t = torch.tensor(env_id_list, dtype=torch.long, device=self.final_tip_pos_root.device)
        self.final_tip_pos_root[env_ids_t] = torch.tensor(
            final_positions, dtype=self.final_tip_pos_root.dtype, device=self.final_tip_pos_root.device
        )
        self.nominal_approach_tip_pos_root[env_ids_t] = torch.tensor(
            nominal_approach_positions,
            dtype=self.nominal_approach_tip_pos_root.dtype,
            device=self.nominal_approach_tip_pos_root.device,
        )
        self.approach_tip_pos_root[env_ids_t] = torch.tensor(
            approach_positions, dtype=self.approach_tip_pos_root.dtype, device=self.approach_tip_pos_root.device
        )
        self.port_quat_root[env_ids_t] = torch.tensor(
            port_quats, dtype=self.port_quat_root.dtype, device=self.port_quat_root.device
        )
        self.approach_tip_quat_root[env_ids_t] = torch.tensor(
            approach_quats, dtype=self.approach_tip_quat_root.dtype, device=self.approach_tip_quat_root.device
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
        self.nominal_approach_tip_pos_w[:] = _root_point_to_world(
            card_pos, card_quat, self.nominal_approach_tip_pos_root
        )
        self.approach_tip_pos_w[:] = _root_point_to_world(card_pos, card_quat, self.approach_tip_pos_root)
        port_quat_w = math_utils.quat_mul(card_quat, self.port_quat_root)
        self.target_tip_quat_w[:] = math_utils.quat_mul(port_quat_w, _target_orientation_offset(port_quat_w))
        self.approach_tip_quat_w[:] = math_utils.quat_mul(card_quat, self.approach_tip_quat_root)

        self.port_x_w[:] = _normalize_rows(math_utils.quat_apply(card_quat, self.port_x_root))
        self.port_y_w[:] = _normalize_rows(math_utils.quat_apply(card_quat, self.port_y_root))
        self.port_z_w[:] = _normalize_rows(math_utils.quat_apply(card_quat, self.port_z_root))
        self.target_x_w[:] = _local_axis_w(self.target_tip_quat_w, (1.0, 0.0, 0.0))
        self.target_y_w[:] = _local_axis_w(self.target_tip_quat_w, (0.0, 1.0, 0.0))
        self.target_z_w[:] = _local_axis_w(self.target_tip_quat_w, (0.0, 0.0, 1.0))

        self.path_w[:] = self.final_tip_pos_w - self.nominal_approach_tip_pos_w
        self.path_length[:] = torch.linalg.norm(self.path_w, dim=1, keepdim=True).clamp_min(1.0e-9)
        self.path_axis_w[:] = self.path_w / self.path_length

        self._command[:, 0:3] = self.final_tip_pos_w
        self._command[:, 3:7] = self.target_tip_quat_w
        self._command[:, 7:10] = self.approach_tip_pos_w
        self._command[:, 10:14] = self.approach_tip_quat_w
        self._command[:, 14:17] = self.nominal_approach_tip_pos_w
        self._command[:, 17:20] = self.path_axis_w
        self._command[:, 20:21] = self.path_length
        self._command[:, 21:24] = self.port_x_w
        self._command[:, 24:27] = self.port_y_w
        self._command[:, 27:30] = self.port_z_w
        self._command[:, 30] = float(self.cfg.port_index)


@configclass
class InsertionGoalCommandCfg(CommandTermCfg):
    """Configuration for one explicit target/port insertion goal.

    The builder should populate the target and port fields from
    ``TargetAssetSpec`` and ``TargetPortSpec``. This keeps runtime command code
    independent of a particular USD asset name such as ``nic_card``.
    """

    class_type: type[CommandTerm] = InsertionGoalCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9 + 1.0)
    debug_vis: bool = False

    target_scene_name: str = "target"
    target_root_prim: str | None = "nic_card_link"
    port_name: str = "sfp_port_0"
    port_index: int = 0
    port_link_path: str = "/sfp_port_0_link"
    port_seat_frame_path: str = "/sfp_port_0_link"
    port_entrance_frame_path: str | None = "/sfp_port_0_link/sfp_port_0_link_entrance"
    insertion_axis_local: tuple[float, float, float] = (0.0, 1.0, 0.0)

    target_xz_offset: tuple[float, float] = (0.0, 0.001)
    approach_offset_local: tuple[float, float, float] = (0.0, -0.09, 0.0)
    approach_pos_noise_local: tuple[float, float, float] = (0.01, 0.0, 0.01)
    approach_tilt_noise_deg: float = 2.0
    approach_twist_noise_deg: float = 5.0


def _sample_local_position_noise(max_abs_local: tuple[float, float, float], *, dtype: torch.dtype) -> torch.Tensor:
    """Sample one local position offset whose norm never exceeds the largest configured bound."""
    bounds = torch.tensor(max_abs_local, dtype=dtype)
    noise = (2.0 * torch.rand(3, dtype=dtype) - 1.0) * torch.abs(bounds)
    max_norm = torch.max(torch.abs(bounds))
    norm = torch.linalg.norm(noise)
    if float(max_norm) > 0.0 and float(norm) > float(max_norm):
        noise = noise * (max_norm / norm)
    return noise


def _sample_approach_orientation_noise(
    tilt_noise_deg: float,
    twist_noise_deg: float,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample a small local tip-frame orientation offset in Isaac's wxyz order."""
    tilt_max = math.radians(max(0.0, float(tilt_noise_deg)))
    twist_max = math.radians(max(0.0, float(twist_noise_deg)))
    tilt = (2.0 * torch.rand(2, dtype=dtype) - 1.0) * tilt_max
    tilt_norm = torch.linalg.norm(tilt)
    if tilt_max > 0.0 and float(tilt_norm) > tilt_max:
        tilt = tilt * (tilt_max / tilt_norm)
    twist = (2.0 * torch.rand(1, dtype=dtype) - 1.0) * twist_max
    # Local X/Z are treated as tilt axes.  Local Y is treated as insertion-axis twist.
    quat = math_utils.quat_from_euler_xyz(
        tilt[0:1],
        twist,
        tilt[1:2],
    )
    return quat


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
        candidates = ", ".join(
            _candidate_prim_paths(asset_root_path, prim_path, usd_root_child=usd_root_child)
        )
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


def _axis_from_quat(quat: torch.Tensor, axis_local: tuple[float, float, float]) -> torch.Tensor:
    axis = torch.tensor([axis_local], dtype=quat.dtype, device=quat.device)
    return _normalize(math_utils.quat_apply(quat, axis)[0])


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.clamp(torch.linalg.norm(vector), min=1.0e-9)


def _normalize_rows(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.clamp(torch.linalg.norm(vector, dim=1, keepdim=True), min=1.0e-9)


def _resolve_asset_root_prim_path(asset, env_index: int, *, usd_root_child: str | None) -> str:
    """Return the USD instance root path for an Isaac Lab asset/env index."""
    prim_paths = list(getattr(asset.root_physx_view, "prim_paths", []))
    if not prim_paths:
        cfg_path = getattr(getattr(asset, "cfg", None), "prim_path", "")
        raise RuntimeError(f"Cannot resolve prim paths for asset with cfg path '{cfg_path}'.")

    index = min(env_index, len(prim_paths) - 1)
    prim_path = str(prim_paths[index])
    if usd_root_child is None:
        return prim_path
    suffix = f"/{usd_root_child}"
    if prim_path.endswith(suffix):
        return prim_path[: -len(suffix)]
    return prim_path


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
