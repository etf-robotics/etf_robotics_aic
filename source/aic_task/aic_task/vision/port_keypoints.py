"""Port keypoint layout used for visual-oracle dataset labels.

The points are expressed in the ``nic_card`` asset frame.  They are deliberately
small and semantic rather than mesh-derived: the first iteration needs stable
visual supervision for the approach frame, not a full CAD annotation pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from aic_task.tasks.manager_based.port_approach.port_approach_env_cfg import (
    NIC_PORT_APPROACH_OFFSET,
    NIC_PORT_APPROACH_RPY,
    NIC_PORT_ENTRY_OFFSET,
)


@dataclass(frozen=True)
class PortKeypointLayout:
    """Named 3D keypoints expressed in the NIC-card frame."""

    names: tuple[str, ...]
    points_nic: tuple[tuple[float, float, float], ...]

    def as_tensor(self, *, device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return keypoints as a ``(K, 3)`` tensor."""
        return torch.tensor(self.points_nic, dtype=dtype, device=device)

    def index(self, name: str) -> int:
        """Return the integer index for a named keypoint."""
        return self.names.index(name)


def make_default_port_keypoint_layout(
    *,
    entry_offset: tuple[float, float, float] | None = None,
    approach_offset: tuple[float, float, float] | None = None,
    keypoint_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mouth_half_width: float = 0.012,
    mouth_half_height: float = 0.006,
    axis_length: float = 0.025,
) -> PortKeypointLayout:
    """Create the default semantic keypoint layout for the NIC port.

    ``NIC_PORT_APPROACH_RPY`` defines the local approach frame: its ``+Z`` axis is
    the insertion direction.  Mouth corners lie in the frame's local XY plane at
    the entry point; axis helper points give the perception model orientation
    supervision even when corners are partially occluded.
    """
    global_offset = _vec(keypoint_offset)
    entry = _vec(NIC_PORT_ENTRY_OFFSET if entry_offset is None else entry_offset) + global_offset
    approach = _vec(NIC_PORT_APPROACH_OFFSET if approach_offset is None else approach_offset) + global_offset
    rot = _rpy_xyz_matrix(*NIC_PORT_APPROACH_RPY)

    def at_entry(local_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
        return _tuple(entry + rot @ _vec(local_xyz))

    names = (
        "entry_center",
        "approach_center",
        "axis_x_plus",
        "axis_y_plus",
        "axis_z_plus",
        "mouth_top_left",
        "mouth_top_right",
        "mouth_bottom_right",
        "mouth_bottom_left",
    )
    points = (
        _tuple(entry),
        _tuple(approach),
        at_entry((axis_length, 0.0, 0.0)),
        at_entry((0.0, axis_length, 0.0)),
        at_entry((0.0, 0.0, axis_length)),
        at_entry((-mouth_half_width, -mouth_half_height, 0.0)),
        at_entry((mouth_half_width, -mouth_half_height, 0.0)),
        at_entry((mouth_half_width, mouth_half_height, 0.0)),
        at_entry((-mouth_half_width, mouth_half_height, 0.0)),
    )
    return PortKeypointLayout(names=names, points_nic=points)


def _vec(values: tuple[float, float, float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def _tuple(values: torch.Tensor) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _rpy_xyz_matrix(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """Return an XYZ Euler rotation matrix matching Isaac Lab's RPY convention."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = torch.tensor(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)), dtype=torch.float64)
    ry = torch.tensor(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)), dtype=torch.float64)
    rz = torch.tensor(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)), dtype=torch.float64)
    return rz @ ry @ rx
