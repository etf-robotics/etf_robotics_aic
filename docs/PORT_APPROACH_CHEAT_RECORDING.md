# Port Approach Cheat Recorder

This note explains how to use `scripts/il/cheat_record_port_approach.py` and how the target pose is defined for the NIC-card port approach task.

The script records demonstrations for `AIC-Port-Approach-v0` using simulator ground truth. It is called a "cheat" recorder because it reads the current TCP pose and NIC-card pose directly from the simulator, computes the desired TCP pose, and then sends normal relative IK actions through the same action interface used by teleoperation. The exported demonstrations are therefore compatible with teleop-style action/state datasets.

## Running

From the repo root:

```bash
./isaaclab.sh -p scripts/il/cheat_record_port_approach.py \
  --task AIC-Port-Approach-v0 \
  --dataset_file ./datasets/port_approach_cheat.hdf5 \
  --num_demos 10
```

Useful options:

- `--num_demos 0`: run indefinitely until stopped.
- `--step_hz 30`: control/render loop rate.
- `--pos_gain` and `--rot_gain`: proportional gains for the relative IK action.
- `--max_pos_delta` and `--max_rot_delta`: per-step action clamps.
- `--hold_steps`: low-gain settle period before success counting.
- `--num_success_steps`: consecutive successful steps required before export.

The script exports only successful episodes through Isaac Lab's `ActionStateRecorderManagerCfg`.

## Pose Definition

The important constants live in:

```text
source/aic_task/aic_task/tasks/manager_based/port_approach/port_approach_env_cfg.py
```

They define two local frames:

```text
gripper_tcp -> cable_tip_frame
nic_card    -> port_approach_frame
```

The controller solves for the TCP pose that makes these two child frames coincide:

```text
tcp_world * cable_tip_from_tcp == nic_card_world * port_approach_from_nic
```

In code this is:

```python
target_tip_quat_w = target.root_quat_w * NIC_PORT_APPROACH_RPY
desired_tcp_quat_w = target_tip_quat_w * inverse(CABLE_TIP_RPY_FROM_TCP)

target_tip_pos_w = target.root_pos_w + rotate(target.root_quat_w, NIC_PORT_APPROACH_OFFSET)
desired_tcp_pos_w = target_tip_pos_w - rotate(desired_tcp_quat_w, CABLE_TIP_OFFSET_FROM_TCP)
```

### Cable Tip Constants

```python
CABLE_TIP_OFFSET_FROM_TCP = (...)
CABLE_TIP_RPY_FROM_TCP = (...)
```

These describe the plug tip relative to `gripper_tcp`.

- The offset is in the TCP local frame, not world frame.
- The world position of the tip changes whenever TCP rotates.
- The RPY describes the cable-tip frame orientation relative to TCP.

For the current viewport alignment, the plug insertion axis is treated as the cable-tip `+Z` axis. The plug has a small tilt around TCP X, so this value is tuned in `CABLE_TIP_RPY_FROM_TCP`.

### NIC Port Constants

```python
NIC_PORT_ENTRY_OFFSET = (...)
NIC_PORT_APPROACH_OFFSET = (...)
NIC_PORT_APPROACH_RPY = (...)
```

These describe a point near the NIC connector relative to the `nic_card` frame.

- `NIC_PORT_ENTRY_OFFSET` is the port entry point.
- `NIC_PORT_APPROACH_OFFSET` is the pre-insertion approach point.
- The approach point is placed farther away from the connector along local Y.
- Insertion from the approach point into the connector is therefore along `nic_card +Y`.

The current target orientation maps the cable-tip `+Z` insertion axis onto `nic_card +Y`, and applies a 180-degree flip so the plug keying enters the correct way.

## Measuring The TCP-To-Tip Offset

Isaac Sim has a Measure tool extension (`omni.kit.tool.measure`) that can be enabled from `Window > Extensions`. It is useful for checking distances, but for `CABLE_TIP_OFFSET_FROM_TCP` you need the signed XYZ vector in the TCP frame.

The most reliable workflow is:

1. Add a small Xform or sphere at the exact plug tip in the viewport.
2. Read the world transform of `gripper_tcp`.
3. Read the world position of the tip marker.
4. Transform the tip marker into the TCP frame.

Example for the Isaac Python console:

```python
import omni.usd
from pxr import Usd, UsdGeom

stage = omni.usd.get_context().get_stage()

tcp_path = "/World/envs/env_0/Robot/aic_unified_robot/gripper_tcp"
tip_path = "/World/tip_probe"

tcp_m = UsdGeom.Xformable(stage.GetPrimAtPath(tcp_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
tip_m = UsdGeom.Xformable(stage.GetPrimAtPath(tip_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

tip_w = tip_m.ExtractTranslation()
offset_tcp = tcp_m.GetInverse().Transform(tip_w)

print(offset_tcp)
```

Use the printed vector as:

```python
CABLE_TIP_OFFSET_FROM_TCP = (x, y, z)
```

## Debugging Orientation

When the position is good but the plug orientation is wrong, identify which error you see:

- Nose points the wrong way: the insertion axis mapping is wrong.
- Nose points into the port but plug is upside down: add or remove a 180-degree rotation around the insertion axis.
- Nose points into the port but tilt is mirrored: flip the sign of the small TCP X tilt in `CABLE_TIP_RPY_FROM_TCP`.

The useful mental model is: define the cable-tip child frame and the NIC approach child frame, then let the script compute the TCP pose required to make them match.
