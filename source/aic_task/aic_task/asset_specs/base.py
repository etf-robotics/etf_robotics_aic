"""Shared types for asset specs and scene specs.

Asset specs describe reusable USD assets: what file they come from and what
stable interface they expose.  Scene specs describe how one of those assets is
instantiated in a concrete environment slot.  Keeping those two concepts
separate lets task code ask for role names such as ``insertion_target`` while
the concrete USD asset underneath that role can be swapped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


PACKAGE_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = PACKAGE_DIR / "assets"


def asset_path(*parts: str) -> str:
    """Return an absolute path inside ``aic_task/assets``.

    The spec layer keeps paths centralized so environment configs do not need
    to know the on-disk folder layout for every concrete asset.
    """

    return str(ASSET_DIR.joinpath(*parts))


AssetRole = Literal["robot", "part", "workcell", "fixture"]
SceneRole = Literal["robot", "target", "fixture", "workcell", "distractor", "support"]
UsdAssetKind = Literal["articulation", "rigid_object", "static"]
Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


@dataclass(frozen=True)
class AssetIdentity:
    """Stable identity for an asset family, independent of file location."""

    name: str
    role: AssetRole


@dataclass(frozen=True)
class UsdAssetInterface:
    """USD-level interface exposed by an asset file.

    ``root_prim`` is the authored/default root prim of the USD asset, when the
    asset has one that task code may need to reason about.
    """

    kind: UsdAssetKind
    root_prim: str | None = None


@dataclass(frozen=True)
class NamedBodySpec:
    """A named rigid/articulation body that other code may reference."""

    name: str
    purpose: str


@dataclass(frozen=True)
class BodyRoleSpec:
    """Semantic role resolved to a concrete articulation body.

    Use this when task code wants to ask for a role such as ``eef`` or ``tcp``
    without hardcoding the body name that currently fulfills that role.
    """

    name: str
    body_name: str
    purpose: str


@dataclass(frozen=True)
class NamedFrameSpec:
    """A semantic frame or child prim inside the asset USD hierarchy."""

    name: str
    path: str
    purpose: str


@dataclass(frozen=True)
class JointGroupSpec:
    """A named group of joints belonging to an articulation asset."""

    name: str
    joint_names: tuple[str, ...]
    default_positions: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AssetSpec:
    """Reusable asset contract.

    This must stay true no matter where the asset is placed.  Do not put task
    targets, randomization ranges, rewards, insertion offsets, or selected port
    choices here; those belong to scene/task specs.
    """

    identity: AssetIdentity
    usd: UsdAssetInterface
    usd_path: str

    @property
    def name(self) -> str:
        """Concrete asset family name, not the role it plays in a scene."""

        return self.identity.name


@dataclass(frozen=True)
class AssetPropertySpec(AssetSpec):
    """Compatibility name for asset-level specs.

    Older code used ``AssetPropertySpec`` for the same idea.  Keep the name as
    an aliasable base while new code can say ``AssetSpec`` directly.
    """


@dataclass(frozen=True)
class SceneAssetSpec:
    """One named asset instance in an environment scene.

    ``name`` is the stable key that scripts should use with ``env.scene[name]``.
    It should describe the role in this scene, for example ``insertion_target``,
    not the concrete model file, for example ``nic_card``.
    """

    name: str
    role: SceneRole
    asset: AssetSpec
    prim_path: str
    init_pos: Vector3 = (0.0, 0.0, 0.0)
    init_rot: Quaternion = (1.0, 0.0, 0.0, 0.0)
    purpose: str = ""
    kinematic: bool = False


@dataclass(frozen=True)
class SceneSpec:
    """A collection of stable scene slots for one environment layout."""

    name: str
    assets: tuple[SceneAssetSpec, ...]

    def asset(self, name: str) -> SceneAssetSpec:
        """Return the scene slot with the requested stable scene name."""

        for scene_asset in self.assets:
            if scene_asset.name == name:
                return scene_asset
        raise KeyError(f"Unknown scene asset '{name}' for scene '{self.name}'.")
