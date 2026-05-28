# Asset And Scene Specs

## Usage

Import stable scene names from `aic_task.asset_specs` and use them as the keys
for `env.scene[...]`.

```python
from aic_task.asset_specs import INSERTION_TARGET, ROBOT, SC_PORT_1, SC_PORT_2

robot = env.scene[ROBOT]
target = env.scene[INSERTION_TARGET]
sc_port_1 = env.scene[SC_PORT_1]
sc_port_2 = env.scene[SC_PORT_2]
```

Use scene names for runtime code.  Do not use concrete asset names such as
`nic_card` when the object is playing a task role.

```python
target_pos_w = env.scene[INSERTION_TARGET].data.root_pos_w
target_quat_w = env.scene[INSERTION_TARGET].data.root_quat_w
```

For one environment:

```python
env_id = 0
target_pos = env.scene[INSERTION_TARGET].data.root_pos_w[env_id]
target_quat = env.scene[INSERTION_TARGET].data.root_quat_w[env_id]
```

Use robot body roles when code wants semantic meaning instead of a hardcoded
body name.

```python
from aic_task.asset_specs import ROBOT_EEF, ROBOT_TCP, UR5E_CABLE_ASSET

eef_body = UR5E_CABLE_ASSET.body_name_for_role(ROBOT_EEF)
tcp_body = UR5E_CABLE_ASSET.body_name_for_role(ROBOT_TCP)

# eef_body == "sfp_tip_link"
# tcp_body == "gripper_tcp"
```

Then use those body names with Isaac Lab runtime APIs.

```python
eef_id = env.scene[ROBOT].find_bodies(eef_body, preserve_order=True)[0][0]
eef_pos_w = env.scene[ROBOT].data.body_pos_w[:, eef_id]
eef_quat_w = env.scene[ROBOT].data.body_quat_w[:, eef_id]
```

To get a configured scene slot before the environment is built:

```python
from aic_task.asset_specs import AIC_PORT_INSERTION_SCENE, INSERTION_TARGET

target_slot = AIC_PORT_INSERTION_SCENE.asset(INSERTION_TARGET)

target_name = target_slot.name
target_prim_path = target_slot.prim_path
target_default_pos = target_slot.init_pos
target_default_quat = target_slot.init_rot
target_usd_path = target_slot.asset.usd_path
target_root_prim = target_slot.asset.usd.root_prim
```

For randomization, use the scene slot name and default pose as the anchor, then
apply task-specific ranges in the environment event config.

```python
from aic_task.asset_specs import AIC_PORT_INSERTION_SCENE, INSERTION_TARGET

target_slot = AIC_PORT_INSERTION_SCENE.asset(INSERTION_TARGET)

randomize_target = {
    "scene_name": target_slot.name,
    "default_pos": target_slot.init_pos,
    "pose_range": {"x": (-0.005, 0.005), "y": (0.0, 0.12)},
}
```

The asset spec tells you what to spawn.  The scene spec tells you where and
under which environment name to spawn it.  The task or event config decides how
to randomize it.

## Intent

The goal is to make environment and script code depend on stable roles instead
of concrete USD asset names.

For example, the current port-insertion scene uses a NIC card as the insertion
target:

```python
INSERTION_TARGET_SCENE_ASSET.asset == NIC_CARD_ASSET
```

Runtime code should still ask for:

```python
env.scene[INSERTION_TARGET]
```

If the target asset changes later, the scene slot can point at another asset and
the script interface stays the same.

## Structure

There are three layers.

Asset specs describe reusable concrete USD assets:

```python
NIC_CARD_ASSET
SC_PORT_ASSET
TASK_BOARD_ASSET
AIC_WORKCELL_ASSET
UR5E_CABLE_ASSET
```

Scene specs describe named instances in one environment:

```python
ROBOT                 # "robot"
WORKCELL              # "workcell"
BOARD                 # "board"
SC_PORT_1             # "sc_port_1"
SC_PORT_2             # "sc_port_2"
INSERTION_TARGET      # "insertion_target"
```

Robot body roles describe semantic robot bodies:

```python
ROBOT_EEF          # "eef" -> "sfp_tip_link"
ROBOT_TCP          # "tcp" -> "gripper_tcp"
ROBOT_PLUG_CENTER  # "plug_center" -> "sfp_module_link"
```

## Asset Specs

`AssetSpec` stores facts that remain true wherever the asset is used.

```python
AssetSpec(
    identity=AssetIdentity(name="nic_card", role="part"),
    usd=UsdAssetInterface(kind="rigid_object", root_prim="nic_card_link"),
    usd_path=asset_path("targets", "nic_card", "nic_card.usd"),
)
```

Parameters:

- `identity.name`: concrete asset family name.
- `identity.role`: broad asset category, such as `robot`, `part`, `fixture`, or
  `workcell`.
- `usd.kind`: how the asset should be spawned by Isaac Lab: `articulation`,
  `rigid_object`, or `static`.
- `usd.root_prim`: authored root prim or root body in the USD, when code needs
  to resolve prim paths relative to the asset root.
- `usd_path`: absolute path to the USD file.

Asset specs should not contain task targets, rewards, randomization ranges,
selected insertion ports, camera labels, or success conditions.

## Scene Specs

`SceneAssetSpec` stores facts about one instance of an asset in one scene.

```python
SceneAssetSpec(
    name=INSERTION_TARGET,
    role="target",
    asset=NIC_CARD_ASSET,
    prim_path="{ENV_REGEX_NS}/insertion_target",
    init_pos=(0.25135, 0.25229, 0.0743),
    init_rot=(0.0, 0.0, -0.7068252, 0.7073883),
    kinematic=True,
)
```

Parameters:

- `name`: stable environment key used by `env.scene[name]`.
- `role`: what the asset instance does in this scene.
- `asset`: the concrete `AssetSpec` to spawn.
- `prim_path`: Isaac Lab prim path for this instance.
- `init_pos`: default world position in the environment.
- `init_rot`: default world quaternion in Isaac Lab order.
- `purpose`: human-readable reason this slot exists.
- `kinematic`: whether rigid-object scene configs should spawn it as kinematic.

## Adding A Different Asset

To use a different concrete target while keeping script code stable:

```python
NEW_CARD_ASSET = RigidPartAssetSpec(
    identity=AssetIdentity(name="new_card", role="part"),
    usd=UsdAssetInterface(kind="rigid_object", root_prim="new_card_link"),
    usd_path=asset_path("targets", "new_card", "new_card.usd"),
)

NEW_INSERTION_TARGET_SCENE_ASSET = SceneAssetSpec(
    name=INSERTION_TARGET,
    role="target",
    asset=NEW_CARD_ASSET,
    prim_path="{ENV_REGEX_NS}/insertion_target",
    init_pos=(0.25, 0.25, 0.07),
    init_rot=(1.0, 0.0, 0.0, 0.0),
    kinematic=True,
)
```

The environment and scripts can still use:

```python
env.scene[INSERTION_TARGET]
```

Only the scene spec changes.

## Limitations

These specs do not spawn assets by themselves.  Environment config code still
needs to turn each `SceneAssetSpec` into an Isaac Lab `ArticulationCfg`,
`RigidObjectCfg`, or `AssetBaseCfg`.

These specs do not read runtime poses.  Runtime position and orientation always
come from Isaac Lab objects such as:

```python
env.scene[INSERTION_TARGET].data.root_pos_w
env.scene[INSERTION_TARGET].data.root_quat_w
env.scene[ROBOT].data.body_pos_w
env.scene[ROBOT].data.body_quat_w
```

These specs do not define insertion geometry.  Port names, port frames,
approach offsets, rewards, terminations, keypoints, and oracle behavior belong
in task-specific specs or task code.

These specs do not validate that the USD file exists when imported.  The path is
centralized here so environment setup can fail clearly when it tries to spawn a
missing asset.
