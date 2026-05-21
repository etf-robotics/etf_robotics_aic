# NIC Card Insert Oracle Changes

This note tracks the current front-keypoint insertion oracle.

## Current Logic

The old side-midpoint, tooth, and manual plug-frame offset logic has been removed from the control path.

The oracle now uses:

- port front points: `sfp_port_0_front_left`, `sfp_port_0_front_right`
- plug front points: `sfp_tip_front_left`, `sfp_tip_front_right`
- port insertion direction: `sfp_port_0_link` local `+Y`
- plug insertion direction: `sfp_tip_link` local `-Z`

The plug front points define the controlled front edge. Their midpoint is the position reference, and the left-to-right segment is the front-frame `+X` axis. The front-frame `+Y` axis is built from `sfp_tip_link` local `-Z`, projected orthogonal to that front edge.

## Target Frame

The port target frame is built from the port front edge:

- `port_x = normalize(port_front_right - port_front_left)`
- `port_y = normalize(sfp_port_0_link +Y in world)`
- `port_x` is orthogonalized against `port_y`
- `port_z = normalize(cross(port_x, port_y))`

The target front center starts at the midpoint of the two port front points. It can be shifted only in the port `X/Z` plane with:

```bash
--front_target_xz_offset X Z
```

Insertion advances from the approach front center toward the final front center along port `+Y`.

## TCP Conversion

The controller still commands `gripper_tcp`.

Each step:

- reads the live `gripper_tcp -> sfp_tip_link` transform;
- computes the desired `sfp_tip_link` pose from the desired front frame;
- transforms that desired tip pose back into a desired TCP pose;
- sends a relative DiffIK command.

This avoids caching a stale TCP-to-tip transform while still controlling the actual robot TCP.

## Default Demo Tuning

The current defaults are tuned for the front-point frame:

```text
--pos_gain 1.2
--rot_gain 0.2
--max_pos_delta 0.020
--insert_lateral_threshold 0.010
--insert_orientation_threshold_deg 4.0
```

The key bug fixed last was the aligned axis: the oracle was effectively aligning the wrong plug/TCP axis. The front frame now uses `sfp_tip_link -Z` as the insertion axis, not the front child Xform `+Y`.

## Useful Commands

Standard visible run:

```bash
../isaaclab.sh -p scripts/il/nic_card_insert_oracle.py --enable_cameras
```

Offset checks:

```bash
../isaaclab.sh -p scripts/il/nic_card_insert_oracle.py --front_target_xz_offset 0.001 0.0 --enable_cameras
../isaaclab.sh -p scripts/il/nic_card_insert_oracle.py --front_target_xz_offset -0.001 0.0 --enable_cameras
../isaaclab.sh -p scripts/il/nic_card_insert_oracle.py --front_target_xz_offset 0.0 0.001 --enable_cameras
```

Frame visualizer:

```bash
../isaaclab.sh -p scripts/utils/visualize_nic_insert_frames.py --drive_oracle --enable_cameras
```

## Latest Validation

Syntax check passed for:

```bash
python -m py_compile scripts/il/nic_card_insert_oracle.py source/aic_task/aic_task/controllers/nic_card_insert_oracle.py scripts/utils/visualize_nic_insert_frames.py
```

A normal camera-enabled run for 360 steps switched from `APPROACH` to `INSERT` at step 38 and advanced about 10 cm along the insertion path. The logged front-point errors settled around 6-7 mm with the current contact/cable setup.
