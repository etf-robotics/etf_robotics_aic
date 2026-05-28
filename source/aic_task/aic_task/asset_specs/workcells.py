"""Concrete workcell and fixture asset specs."""

from __future__ import annotations

from dataclasses import dataclass

from .base import AssetIdentity, AssetPropertySpec, UsdAssetInterface, asset_path


@dataclass(frozen=True)
class StaticAssetSpec(AssetPropertySpec):
    """Asset-level contract for static workcell or fixture USDs."""


AIC_WORKCELL_ASSET = StaticAssetSpec(
    identity=AssetIdentity(name="aic_workcell", role="workcell"),
    usd=UsdAssetInterface(kind="static", root_prim=None),
    usd_path=asset_path("workcells", "aic", "aic.usd"),
)

TASK_BOARD_ASSET = StaticAssetSpec(
    identity=AssetIdentity(name="task_board", role="fixture"),
    usd=UsdAssetInterface(kind="rigid_object", root_prim="base_visual"),
    usd_path=asset_path("workcells", "task_board", "task_board_rigid.usd"),
)

# Backward-compatible names while callers migrate to the asset/scene split.
AIC_WORKCELL = AIC_WORKCELL_ASSET
TASK_BOARD = TASK_BOARD_ASSET
