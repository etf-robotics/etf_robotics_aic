# NIC Card Insert Oracle Changes

This note documents the difference between the latest committed code and the current working-tree version of the NIC-card SFP insertion demo.

Changed files:

- `source/aic_task/aic_task/controllers/nic_card_insert_oracle.py`
- `scripts/il/nic_card_insert_oracle.py`
- `scripts/utils/visualize_nic_insert_frames.py`

## What Solved The Main Problem

The main problem was not just a wrong insertion offset. The oracle was mixing position and orientation references in a way that could make the command look reasonable while the physical module did not correct the visible error.

The working logic now uses the measured module geometry as the position reference:

- The driven insertion point is the midpoint of `sfp_tip_side_left` and `sfp_tip_side_right`.
- The `sfp_tip_link` origin is no longer treated as the physical insertion reference.
- The TCP-to-tip transform is recomputed every control step instead of relying on the initial cached transform.
- The controller still commands `gripper_tcp`, but the desired TCP motion is computed from the live error of the module insertion reference.

The orientation was also cleaned up to use only the confirmed frame relationship:

```text
tip +Y -> port -Z
tip +Z -> port -Y
tip +X -> port -X
```

This removed the old experimental 180-degree flip and roll-offset logic. The current `_desired_tip_quat_from_port()` builds the target tip frame directly from the port axes.

In practice, the thing that finally made the behavior understandable was the frame visualizer. It showed the real USD/body frames in the running simulation, which let us stop guessing about which local axis meant what.

## Target Line Logic

The committed version mostly targeted `sfp_port_0_link` directly. The current version builds a more physical target line from the port geometry.

The new target creation can:

- read the four entrance corner prims under `sfp_port_0_link_entrance`;
- move the insertion line to the entrance-corner centerline while preserving the original insertion depth from `sfp_port_0_link`;
- optionally shift that line so `sfp_tip_tooth_tip` rides the port top edge;
- apply an extra manual `--target_offset X Y Z` in the `sfp_port_0_link` local frame.

The approach point remains local to the port frame. The default is still:

```text
--approach_offset 0.0 -0.10 0.0
```

That means the approach is 10 cm opposite insertion, because insertion is port-local `+Y`.

## Control Logic Changes

The insert phase now projects the live insertion reference onto the approach-to-seat line.

Each step computes:

- `path_distance`: current progress along the insertion line;
- `target_path_distance`: commanded progress along the insertion line;
- `path_lateral_error`: perpendicular error from the insertion line;
- `path_error_local`: the signed lateral error expressed in the port frame.

Insertion progress is no longer just a blind counter. In `INSERT`:

- if the module is close enough to the path, the target advances by at least `--insert_lookahead`;
- if the module is too far from the path, the target backs off by `--insert_recenter_backoff`;
- lateral correction can be scaled with `--insert_lateral_correction_scale`;
- rotation correction can be damped with `--insert_rot_scale`.

The current defaults intentionally prioritize keeping the aligned pose stable during the sensitive insertion phase:

```text
--insert_orientation_threshold_deg 10.0
--insert_recenter_backoff 0.003
--insert_lateral_correction_scale 1.0
--insert_rot_scale 0.05
--insert_lookahead 0.002
```

`--insert_rot_scale` was especially important because full rotation correction during insertion could fight the position correction and make the module appear stuck even while commands were being sent.

## Script Changes

`scripts/il/nic_card_insert_oracle.py` gained several debug and tuning arguments:

- `--target_offset X Y Z`
- `--disable_tooth_top_alignment`
- `--disable_nic_collisions`
- `--insert_recenter_backoff`
- `--insert_lateral_correction_scale`
- `--insert_rot_scale`
- `--insert_lookahead`

It also prints more useful startup state:

- confirmed orientation mapping;
- approach and target offsets;
- the insertion reference in `sfp_tip_link`;
- cached seat pose in NIC-root frame;
- live seat and approach positions.

When `--log_path_xy_error` is enabled, the log now includes path progress and signed local path error:

```text
path_line_err
path_s
target_s
path_xy_err
path_err_port
ori_err
cmd_pos_b
cmd_rot_b
```

`--disable_nic_collisions` is deliberately a debug/demo flag. It proved that the oracle can complete the path when contact geometry is removed, but it should not be treated as the real physical solution.

## Visualizer Changes

`scripts/utils/visualize_nic_insert_frames.py` was extended to mirror the oracle options so the same target logic can be inspected visually.

It now supports:

- `--target_offset`;
- `--disable_tooth_top_alignment`;
- insert-phase tuning arguments matching the oracle script;
- printing port keypoint local positions;
- printing tip-child local positions;
- visualizing the derived target frame.

This is the tool that made the frame mapping definite.

## Useful Run Commands

Standard visible run:

```bash
../isaaclab.sh -p scripts/il/nic_card_insert_oracle.py \
  --max_episode_steps 700 \
  --log_path_xy_error \
  --log_every 25 \
  --enable_cameras
```

Debug/demo run with NIC collisions disabled:

```bash
../isaaclab.sh -p scripts/il/nic_card_insert_oracle.py \
  --max_episode_steps 500 \
  --log_path_xy_error \
  --log_every 25 \
  --disable_nic_collisions \
  --disable_tooth_top_alignment \
  --insert_lookahead 0.008 \
  --insert_rot_scale 0.2 \
  --insert_lateral_threshold 0.006 \
  --insert_recenter_backoff 0.0 \
  --enable_cameras
```

Frame visualizer:

```bash
../isaaclab.sh -p scripts/utils/visualize_nic_insert_frames.py \
  --drive_oracle \
  --enable_cameras
```

## Current Interpretation

The controller conflict was solved by using the correct physical reference and the confirmed axis mapping, not by adding another arbitrary rotation offset.

The remaining hard part is contact: when collisions are enabled, the module can still jam at the port entrance. Since the no-collision run can complete, the next investigation should focus on contact geometry, clearances, and whether the tooth/top-line target is too aggressive for the current collision meshes.
