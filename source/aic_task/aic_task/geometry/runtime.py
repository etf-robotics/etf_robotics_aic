"""Runtime USD geometry helpers for AIC port insertion.

The static geometry module stores names and fallback dimensions.  This module
resolves composed USD transforms from the live Isaac stage so labels and oracle
targets follow the actual asset instance in each environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry import (
    BACKOFF_DISTANCE,
    COARSE_APPROACH_GAP,
    OPPOSITE_TOOTH_NODE_PATH,
    PORTS_BY_NAME,
    PORT_LONG_HALF,
    PREINSERT_GAP,
    AXIS_KEYPOINT_LENGTH,
)


@dataclass
class PortRuntimeTensors:
    """World-frame port geometry tensors for all envs."""

    entrance_w: torch.Tensor
    seat_w: torch.Tensor
    opposite_tooth_w: torch.Tensor
    insertion_axis_w: torch.Tensor
    opposite_axis_w: torch.Tensor
    long_axis_w: torch.Tensor
    y_half: torch.Tensor
    insertion_depth: torch.Tensor

    @property
    def tooth_axis_w(self) -> torch.Tensor:
        return -self.opposite_axis_w

    @property
    def preinsert_w(self) -> torch.Tensor:
        return self.entrance_w - self.insertion_axis_w * PREINSERT_GAP

    @property
    def coarse_approach_w(self) -> torch.Tensor:
        return self.entrance_w - self.insertion_axis_w * COARSE_APPROACH_GAP

    @property
    def backoff_w(self) -> torch.Tensor:
        return self.entrance_w - self.insertion_axis_w * (PREINSERT_GAP + BACKOFF_DISTANCE)


def resolve_asset_root_prim_path(asset, env_index: int, *, usd_root_child: str = "nic_card_link") -> str:
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


def compute_port_runtime_tensors(env, *, target_name: str = "nic_card", port_name: str = "sfp_port_0") -> PortRuntimeTensors:
    """Resolve port entrance/seat/anchor tensors from the live USD stage."""
    target = env.scene[target_name]
    port = PORTS_BY_NAME[port_name]
    device = target.data.root_pos_w.device
    dtype = target.data.root_pos_w.dtype

    entrance_points = []
    seat_points = []
    opposite_points = []
    for env_index in range(env.num_envs):
        root_path = resolve_asset_root_prim_path(target, env_index)
        entrance = _prim_translation_w(root_path, port.entrance_path)
        seat = _prim_translation_w(root_path, port.seat_path)
        if port.opposite_tooth_node_path is not None:
            opposite = _prim_translation_w(root_path, port.opposite_tooth_node_path)
        elif port.opposite_tooth_source_port == "sfp_port_0":
            source = PORTS_BY_NAME["sfp_port_0"]
            source_entrance = _prim_translation_w(root_path, source.entrance_path)
            source_opposite = _prim_translation_w(root_path, OPPOSITE_TOOTH_NODE_PATH)
            offset = tuple(source_opposite[i] - source_entrance[i] for i in range(3))
            opposite = tuple(entrance[i] + offset[i] for i in range(3))
        else:
            raise ValueError(f"Port '{port_name}' does not define an opposite-tooth anchor.")
        entrance_points.append(entrance)
        seat_points.append(seat)
        opposite_points.append(opposite)

    entrance_w = torch.tensor(entrance_points, dtype=dtype, device=device)
    seat_w = torch.tensor(seat_points, dtype=dtype, device=device)
    opposite_w = torch.tensor(opposite_points, dtype=dtype, device=device)

    insertion_vec = seat_w - entrance_w
    insertion_depth = torch.linalg.norm(insertion_vec, dim=1, keepdim=True).clamp_min(1.0e-9)
    insertion_axis = insertion_vec / insertion_depth

    opposite_vec = opposite_w - entrance_w
    y_half = torch.linalg.norm(opposite_vec, dim=1, keepdim=True).clamp_min(1.0e-9)
    opposite_axis = opposite_vec / y_half

    long_axis = torch.linalg.cross(opposite_axis, insertion_axis, dim=1)
    long_norm = torch.linalg.norm(long_axis, dim=1, keepdim=True).clamp_min(1.0e-9)
    long_axis = long_axis / long_norm

    return PortRuntimeTensors(
        entrance_w=entrance_w,
        seat_w=seat_w,
        opposite_tooth_w=opposite_w,
        insertion_axis_w=insertion_axis,
        opposite_axis_w=opposite_axis,
        long_axis_w=long_axis,
        y_half=y_half,
        insertion_depth=insertion_depth,
    )


def compute_port_keypoints_w_from_runtime(runtime: PortRuntimeTensors) -> tuple[tuple[str, ...], torch.Tensor]:
    """Return named keypoints in world frame from resolved runtime port geometry."""
    names = (
        "entrance_center",
        "preinsert_center",
        "seat_center",
        "entry_center",
        "approach_center",
        "opposite_tooth_anchor",
        "tooth_anchor",
        "corner_opposite_left",
        "corner_opposite_right",
        "corner_tooth_left",
        "corner_tooth_right",
        "mouth_top_left",
        "mouth_top_right",
        "mouth_bottom_left",
        "mouth_bottom_right",
        "axis_insertion_plus",
    )
    x = runtime.long_axis_w
    y = runtime.opposite_axis_w
    z = runtime.insertion_axis_w
    h = runtime.y_half
    points = (
        runtime.entrance_w,
        runtime.preinsert_w,
        runtime.seat_w,
        runtime.entrance_w,
        runtime.preinsert_w,
        runtime.opposite_tooth_w,
        runtime.entrance_w - y * h,
        runtime.entrance_w - x * PORT_LONG_HALF + y * h,
        runtime.entrance_w + x * PORT_LONG_HALF + y * h,
        runtime.entrance_w - x * PORT_LONG_HALF - y * h,
        runtime.entrance_w + x * PORT_LONG_HALF - y * h,
        runtime.entrance_w - x * PORT_LONG_HALF + y * h,
        runtime.entrance_w + x * PORT_LONG_HALF + y * h,
        runtime.entrance_w - x * PORT_LONG_HALF - y * h,
        runtime.entrance_w + x * PORT_LONG_HALF - y * h,
        runtime.entrance_w + z * AXIS_KEYPOINT_LENGTH,
    )
    return names, torch.stack(points, dim=1)


def _prim_translation_w(asset_root_path: str, relative_path: str):
    """Return a USD prim world translation as a pxr Gf.Vec3d."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim_path, prim = _resolve_prim(stage, asset_root_path, relative_path)
    if not prim.IsValid():
        candidates = ", ".join(_candidate_prim_paths(asset_root_path, relative_path))
        children = ", ".join(_nearby_child_paths(stage, asset_root_path))
        raise KeyError(
            f"USD prim for '{relative_path}' was not found under '{asset_root_path}'. "
            f"Tried: {candidates}. Nearby children: {children}"
        )
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    translation = matrix.ExtractTranslation()
    return (float(translation[0]), float(translation[1]), float(translation[2]))


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
