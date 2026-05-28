"""Concrete rigid-part asset specs.

These are reusable assets, not task targets.  A NIC card becomes an
``insertion_target`` only when a scene spec places it in that role.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import AssetIdentity, AssetPropertySpec, UsdAssetInterface, asset_path


@dataclass(frozen=True)
class RigidPartAssetSpec(AssetPropertySpec):
    """Reusable rigid USD asset that can be assigned a scene role."""


TargetAssetSpec = RigidPartAssetSpec


NIC_CARD_ASSET = RigidPartAssetSpec(
    identity=AssetIdentity(name="nic_card", role="part"),
    usd=UsdAssetInterface(kind="rigid_object", root_prim="nic_card_link"),
    usd_path=asset_path("targets", "nic_card", "nic_card.usd"),
)

SC_PORT_ASSET = RigidPartAssetSpec(
    identity=AssetIdentity(name="sc_port", role="part"),
    usd=UsdAssetInterface(kind="rigid_object", root_prim="sc_port_visual"),
    usd_path=asset_path("targets", "sc_port", "sc_port.usd"),
)

# Backward-compatible names while callers migrate to the asset/scene split.
NIC_CARD_TARGET = NIC_CARD_ASSET
SC_PORT_TARGET = SC_PORT_ASSET
