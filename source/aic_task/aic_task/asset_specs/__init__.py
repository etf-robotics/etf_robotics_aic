"""Asset and scene specifications for AIC tasks.

Concrete assets are exported with ``*_ASSET`` names.  Scene slots are exported
with role names such as ``INSERTION_TARGET`` so environment code can stay stable
when a different concrete asset is assigned to the same role.
"""

from .base import (
    ASSET_DIR,
    AssetIdentity,
    AssetRole,
    AssetSpec,
    BodyRoleSpec,
    JointGroupSpec,
    NamedBodySpec,
    NamedFrameSpec,
    SceneAssetSpec,
    SceneRole,
    SceneSpec,
    UsdAssetInterface,
    UsdAssetKind,
    asset_path,
)
from .robots import (
    ROBOT_EEF,
    ROBOT_PLUG_CENTER,
    ROBOT_TCP,
    UR5E_ARM,
    UR5E_CABLE_ASSET,
    RobotAssetSpec,
)
from .scene import (
    AIC_PORT_INSERTION_SCENE,
    BOARD,
    BOARD_SCENE_ASSET,
    INSERTION_TARGET,
    INSERTION_TARGET_SCENE_ASSET,
    ROBOT,
    ROBOT_SCENE_ASSET,
    SC_PORT_1,
    SC_PORT_1_SCENE_ASSET,
    SC_PORT_2,
    SC_PORT_2_SCENE_ASSET,
    WORKCELL,
    WORKCELL_SCENE_ASSET,
)
from .targets import (
    NIC_CARD_ASSET,
    RigidPartAssetSpec,
    SC_PORT_ASSET,
)
from .workcells import AIC_WORKCELL_ASSET, TASK_BOARD_ASSET, StaticAssetSpec

__all__ = [
    "AIC_PORT_INSERTION_SCENE",
    "AIC_WORKCELL_ASSET",
    "ASSET_DIR",
    "AssetIdentity",
    "AssetRole",
    "AssetSpec",
    "BOARD",
    "BOARD_SCENE_ASSET",
    "BodyRoleSpec",
    "INSERTION_TARGET",
    "INSERTION_TARGET_SCENE_ASSET",
    "JointGroupSpec",
    "NIC_CARD_ASSET",
    "NamedBodySpec",
    "NamedFrameSpec",
    "ROBOT",
    "ROBOT_EEF",
    "ROBOT_PLUG_CENTER",
    "ROBOT_SCENE_ASSET",
    "ROBOT_TCP",
    "RigidPartAssetSpec",
    "RobotAssetSpec",
    "SC_PORT_ASSET",
    "SC_PORT_1",
    "SC_PORT_1_SCENE_ASSET",
    "SC_PORT_2",
    "SC_PORT_2_SCENE_ASSET",
    "SceneAssetSpec",
    "SceneRole",
    "SceneSpec",
    "StaticAssetSpec",
    "TASK_BOARD_ASSET",
    "UR5E_ARM",
    "UR5E_CABLE_ASSET",
    "UsdAssetInterface",
    "UsdAssetKind",
    "WORKCELL",
    "WORKCELL_SCENE_ASSET",
    "asset_path",
]
