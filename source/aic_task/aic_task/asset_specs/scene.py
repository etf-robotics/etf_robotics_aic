"""Scene-level asset slots for AIC environments.

Use these names in environment configs and scripts.  The slot name is stable
(``insertion_target``), while the concrete asset assigned to that slot
(``NIC_CARD_ASSET`` today) can change without renaming every caller.
"""

from __future__ import annotations

from .base import SceneAssetSpec, SceneSpec
from .robots import UR5E_CABLE_ASSET
from .targets import NIC_CARD_ASSET, SC_PORT_ASSET
from .workcells import AIC_WORKCELL_ASSET, TASK_BOARD_ASSET


ROBOT = "robot"
WORKCELL = "workcell"
BOARD = "board"
SC_PORT_1 = "sc_port_1"
SC_PORT_2 = "sc_port_2"
INSERTION_TARGET = "insertion_target"


ROBOT_SCENE_ASSET = SceneAssetSpec(
    name=ROBOT,
    role="robot",
    asset=UR5E_CABLE_ASSET,
    prim_path="{ENV_REGEX_NS}/Robot",
    init_pos=(-0.18, -0.122, 0.0),
    init_rot=(0.0, 0.0, 0.0, 1.0),
    purpose="controlled robot; task controllers should address this slot as env.scene['robot']",
)

WORKCELL_SCENE_ASSET = SceneAssetSpec(
    name=WORKCELL,
    role="workcell",
    asset=AIC_WORKCELL_ASSET,
    prim_path="{ENV_REGEX_NS}/workcell",
    init_pos=(0.0, 0.0, -1.15),
    init_rot=(1.0, 0.0, 0.0, 0.0),
    purpose="static surrounding AIC workcell",
)

BOARD_SCENE_ASSET = SceneAssetSpec(
    name=BOARD,
    role="fixture",
    asset=TASK_BOARD_ASSET,
    prim_path="{ENV_REGEX_NS}/board",
    init_pos=(0.2837, 0.229, 0.0),
    init_rot=(1.0, 0.0, 0.0, 0.0),
    purpose="task-board fixture that carries interchangeable parts",
    kinematic=True,
)

SC_PORT_1_SCENE_ASSET = SceneAssetSpec(
    name=SC_PORT_1,
    role="support",
    asset=SC_PORT_ASSET,
    prim_path="{ENV_REGEX_NS}/sc_port_1",
    init_pos=(0.2904, 0.1928, 0.005),
    init_rot=(0.73136, 0.0, 0.0, -0.682),
    purpose="first passive SC port on the board",
    kinematic=True,
)

SC_PORT_2_SCENE_ASSET = SceneAssetSpec(
    name=SC_PORT_2,
    role="support",
    asset=SC_PORT_ASSET,
    prim_path="{ENV_REGEX_NS}/sc_port_2",
    init_pos=(0.2913, 0.1507, 0.005),
    init_rot=(0.73136, 0.0, 0.0, -0.682),
    purpose="second passive SC port on the board",
    kinematic=True,
)

INSERTION_TARGET_SCENE_ASSET = SceneAssetSpec(
    name=INSERTION_TARGET,
    role="target",
    asset=NIC_CARD_ASSET,
    prim_path="{ENV_REGEX_NS}/insertion_target",
    init_pos=(0.25135, 0.25229, 0.0743),
    init_rot=(0.0, 0.0, -0.7068252, 0.7073883),
    purpose="active object for port insertion; scripts should not depend on the concrete NIC-card name",
    kinematic=True,
)

AIC_PORT_INSERTION_SCENE = SceneSpec(
    name="aic_port_insertion",
    assets=(
        ROBOT_SCENE_ASSET,
        WORKCELL_SCENE_ASSET,
        BOARD_SCENE_ASSET,
        SC_PORT_1_SCENE_ASSET,
        SC_PORT_2_SCENE_ASSET,
        INSERTION_TARGET_SCENE_ASSET,
    ),
)
