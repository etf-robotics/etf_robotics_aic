# Visual Port Keypoint Dataset

This is the first iteration of the perception-first port approach pipeline.
It records RGB observations plus simulator-generated keypoint labels, so the
perception model can learn the NIC port geometry before we ask a policy to solve
the full insertion task.

## Recording

From the repo root:

```bash
./isaaclab.sh -p scripts/il/record_port_keypoint_dataset.py \
  --task AIC-Port-Approach-v0 \
  --dataset_file ./datasets/visual_port_keypoints.hdf5 \
  --num_episodes 20 \
  --max_episode_steps 350
```

Useful options:

- `--stream`: attach the existing browser camera stream while recording.
- `--save_every N`: save every N-th control step.
- `--no_depth_labels`: record only RGB-derived in-frame labels, without depth-based occlusion checks.
- `--env_index`: serialize a specific env if running more than one.
- `--log_every N`: print phase, visible keypoint count, errors, and action norm every N steps.
- `--phase_mask_key {in_frame,visible}`: choose whether phase transitions use loose projected keypoints or
  stricter depth-visible keypoints. The default is `in_frame`.
- `--camera_frame {ros,opengl,world}`: camera orientation convention for projection. Start with `ros`;
  try `opengl` if `in_frame` stays zero while the port is clearly in a camera view.
- `--log_projection_details`: print per-camera UV and depth ranges for projected keypoints.
- `--keypoint_offset X Y Z`: shift all keypoints in the NIC-card frame.
- `--entry_offset X Y Z` and `--approach_offset X Y Z`: override the two main semantic points.
- `--mouth_half_width`, `--mouth_half_height`, `--axis_length`: tune the auxiliary keypoint geometry.
- `--occlusion_depth_tolerance`: relax or tighten the strict `keypoints_visible` depth check.

## What It Records

Each HDF5 episode stores:

```text
data/demo_N/
  obs/<camera>/rgb
  labels/<camera>/keypoints_uv
  labels/<camera>/keypoints_depth
  labels/<camera>/keypoints_visible
  labels/<camera>/keypoints_in_frame
  labels/<camera>/points_camera
  labels/port_keypoints_w
  labels/phase
  labels/oracle/*
  proprio/joint_pos
  proprio/joint_vel
  proprio/tcp_pose_w
  actions/oracle
  camera/<camera>/intrinsic
  camera/<camera>/pos_w
  camera/<camera>/quat_w_ros
```

The student model should use `obs/*` and allowed proprioception.  The keypoint,
oracle, camera pose, and phase entries are labels/debug data, not actor inputs.

## Inspecting Labels

Export a few mosaic frames with keypoints overlaid:

```bash
./isaaclab.sh -p scripts/view_camera_dataset.py \
  --dataset_file ./datasets/visual_port_keypoints.hdf5 \
  --demo demo_0 \
  --draw_keypoints \
  --export_dir ./datasets/visual_port_keypoints_preview \
  --max_frames 50
```

If the recorder stays in `SEARCH`, run a short debug pass:

```bash
./isaaclab.sh -p scripts/il/record_port_keypoint_dataset.py \
  --task AIC-Port-Approach-v0 \
  --num_episodes 1 \
  --max_episode_steps 120 \
  --log_every 10 \
  --log_projection_details
```

Interpretation:

- `front=0`: keypoints are behind that camera for the selected camera frame convention.
- `front>0 in=0`: keypoints are in front of the camera but outside the image; check FOV, camera aim, or offsets.
- `in>0 visible=0`: projection is plausible, but the strict depth occlusion test is rejecting points.
  Tune `--occlusion_depth_tolerance` or use `--phase_mask_key in_frame`.

## Teacher Phases

The recorder uses a visual-compatible teacher phase label:

```text
0 SEARCH
1 CENTER
2 COARSE_APPROACH
3 ALIGN
4 HOLD
```

The oracle still computes the privileged target pose, but early phases gate the
action so hidden orientation is not used before the port geometry is visible.
Phase transitions use `keypoints_in_frame` by default, because semantic points
can be slightly behind the rendered surface and fail the stricter depth
occlusion check even when the port geometry is in the camera view.
In `SEARCH`, `CENTER`, and `COARSE_APPROACH`, the rotational action is zeroed
while translation is kept active so the recorder can collect moving visual
pre-approach data.  In `ALIGN`, the full oracle action is allowed.  In `HOLD`,
the action is damped.

## Keypoint Layout

The default semantic keypoints are:

```text
entry_center
approach_center
axis_x_plus
axis_y_plus
axis_z_plus
mouth_top_left
mouth_top_right
mouth_bottom_right
mouth_bottom_left
```

They are defined in the NIC-card frame in
`source/aic_task/aic_task/vision/port_keypoints.py`.  The mouth dimensions are
first-iteration calibration constants; tune them after inspecting overlays.
