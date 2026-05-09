"""Port keypoint layouts for visual-oracle insertion datasets."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from aic_task.geometry import (
    AXIS_KEYPOINT_LENGTH,
    PORT_KEYPOINTS_LOCAL_FALLBACK,
    PORT_LONG_HALF,
    PORT_Y_HALF_FALLBACK,
)

DEFAULT_MOUTH_HALF_WIDTH = PORT_LONG_HALF
DEFAULT_MOUTH_HALF_HEIGHT = PORT_Y_HALF_FALLBACK
DEFAULT_AXIS_LENGTH = AXIS_KEYPOINT_LENGTH


@dataclass(frozen=True)
class PortKeypointLayout:
    """Named 3D keypoints for a port.

    For the new insertion pipeline, ``use_usd_geometry`` means the points are
    resolved from live USD frames by ``projection.compute_port_keypoints_w``.
    ``points_nic`` remains as a fallback/compatibility representation.
    """

    names: tuple[str, ...]
    points_nic: tuple[tuple[float, float, float], ...]
    port_name: str = "sfp_port_0"
    use_usd_geometry: bool = True

    def as_tensor(self, *, device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return fallback keypoints as a ``(K, 3)`` tensor."""
        return torch.tensor(self.points_nic, dtype=dtype, device=device)

    def index(self, name: str) -> int:
        """Return the integer index for a named keypoint."""
        return self.names.index(name)


def make_default_port_keypoint_layout(
    *,
    entry_offset: tuple[float, float, float] | None = None,
    approach_offset: tuple[float, float, float] | None = None,
    keypoint_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mouth_half_width: float = DEFAULT_MOUTH_HALF_WIDTH,
    mouth_half_height: float = DEFAULT_MOUTH_HALF_HEIGHT,
    axis_length: float = DEFAULT_AXIS_LENGTH,
    port_name: str = "sfp_port_0",
    use_usd_geometry: bool = True,
) -> PortKeypointLayout:
    """Create the default semantic keypoint layout for the NIC SFP port.

    The legacy keyword arguments are accepted so existing scripts keep parsing,
    but the V1 insertion layout is driven by ``aic_task.geometry``.  When
    ``use_usd_geometry`` is true, runtime USD frames override these fallback
    points during projection.
    """
    del entry_offset, approach_offset, mouth_half_width, mouth_half_height, axis_length
    offset = _vec(keypoint_offset)
    names = tuple(PORT_KEYPOINTS_LOCAL_FALLBACK.keys())
    points = tuple(_tuple(_vec(point) + offset) for point in PORT_KEYPOINTS_LOCAL_FALLBACK.values())
    return PortKeypointLayout(
        names=names,
        points_nic=points,
        port_name=port_name,
        use_usd_geometry=use_usd_geometry,
    )


def _vec(values: tuple[float, float, float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def _tuple(values: torch.Tensor) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))
