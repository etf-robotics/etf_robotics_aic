---
scope: reusable asset / scene-slot / layout contracts; the bottom layer of the assembly pattern
audience: AI agents working in this repo
last_verified_commit: 8d9a44e
related:
  - ../../docs/01_package_structure.md
  - ../../docs/04_assembly_pattern.md
  - ../../docs/03_port_insertion_overview.md
---

# Asset Specs

This package keeps reusable asset facts separate from task assembly and IsaacLab
configuration code.

## Layers

`AssetSpec` describes facts that stay true everywhere an asset is used:

```python
UR5E_CABLE_ASSET.usd_path
UR5E_CABLE_ASSET.body_name_for_role(ROBOT_ROLE_TCP)  # "gripper_tcp"
UR5E_CABLE_ASSET.body_name_for_role(ROBOT_ROLE_EEF)  # "sfp_tip_link"

NIC_CARD_ASSET.usd.root_prim                         # "nic_card_link"
NIC_CARD_ASSET.port("sfp_port_0").link_path          # "/sfp_port_0_link"
NIC_CARD_ASSET.port("sfp_port_0").insertion_axis_local
```

`SceneSlotSpec` describes how one asset is instantiated in one layout:

```python
AIC_PORT_INSERTION_LAYOUT.target_slot.name       # "target"
AIC_PORT_INSERTION_LAYOUT.target_slot.asset      # NIC_CARD_ASSET
AIC_PORT_INSERTION_LAYOUT.target_slot.prim_path  # "{ENV_REGEX_NS}/target"
```

`SceneLayoutSpec` groups the robot, target, board, workcell, auxiliary slots,
and layout reset randomization:

```python
for slot in AIC_PORT_INSERTION_LAYOUT.all_slots():
    print(slot.name, slot.asset.usd_path)
```

## Stable Runtime Names

Use scene slot names in runtime code:

```python
from aic_task.asset_specs import AIC_PORT_INSERTION_LAYOUT

robot = env.scene[AIC_PORT_INSERTION_LAYOUT.robot_slot.name]
target = env.scene[AIC_PORT_INSERTION_LAYOUT.target_slot.name]
```

Do not use concrete asset names such as `nic_card` as scene keys. The NIC card is
currently spawned in the `target` scene slot.

Use robot body roles instead of hardcoded body names:

```python
from aic_task.asset_specs import ROBOT_ROLE_EEF, ROBOT_ROLE_TCP, UR5E_CABLE_ASSET

tcp_body = UR5E_CABLE_ASSET.body_name_for_role(ROBOT_ROLE_TCP)
eef_body = UR5E_CABLE_ASSET.body_name_for_role(ROBOT_ROLE_EEF)
```

For the current UR5e cable asset:

```python
tcp_body == "gripper_tcp"
eef_body == "sfp_tip_link"
```

## Current Constants

Asset contracts:

```python
UR5E_CABLE_ASSET
NIC_CARD_ASSET
SC_PORT_ASSET
TASK_BOARD_ASSET
AIC_WORKCELL_ASSET
```

Scene layout:

```python
AIC_PORT_INSERTION_LAYOUT
```

Scene slot names:

```python
SCENE_SLOT_ROBOT      # "robot"
SCENE_SLOT_TARGET     # "target"
SCENE_SLOT_BOARD      # "board"
SCENE_SLOT_WORKCELL   # "workcell"
SCENE_SLOT_SC_PORT_1  # "sc_port_1"
SCENE_SLOT_SC_PORT_2  # "sc_port_2"
```

The current NIC target contract defines only port 0:

```python
NIC_SFP_PORT_0
NIC_CARD_ASSET.default_port == "sfp_port_0"
```

## Ownership

Robot specs own robot facts: USD path, root prim, joint groups, default joint
positions, TCP/EEF body roles, camera frames, actuator defaults, and
robot-specific spawn physics defaults.

Target specs own target facts: USD path, root prim, available ports, port frame
paths, and insertion-axis convention.

Scene layouts own placement facts: scene keys, prim paths, default poses,
kinematic flags, auxiliary assets, and layout randomization.

Task specs should choose an assembly from these pieces. Builders should convert
the specs into IsaacLab config objects.

## Related docs

- [01_package_structure.md](../../docs/01_package_structure.md) — where this
  package sits in the broader directory map.
- [04_assembly_pattern.md](../../docs/04_assembly_pattern.md) — design
  rationale for the asset-spec → spec → builder → env-cfg layering, and the
  rules that keep it intact.
- [03_port_insertion_overview.md](../../docs/03_port_insertion_overview.md) —
  the concrete task that consumes `AIC_PORT_INSERTION_LAYOUT`,
  `UR5E_CABLE_ASSET`, `NIC_CARD_ASSET`, and friends.
